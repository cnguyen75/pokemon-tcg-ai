"""agent2: deck-agnostic PTCG policy framework + per-deck configs.

Design:
- DeckConfig (deck_configs.py, exec'd): card roles, fetch priorities, energy
  routing, bench priorities, setup preferences.
- ArchetypeID: classify opponent from revealed cards.
- AttackPlanner: each MAIN select, enumerate (my pokemon x attacks x targets),
  damage math with weakness/resistance/known blocks, produce an AttackPlan
  that all per-context scorers consult.
- Per-context scoring: MAIN ordering ability > plays > evolve > items >
  supporter > attach > attack > end, plus every sub-select context.
- Lethal verification: when the plan needs Boss's Orders to win the game,
  verify with a short determinized search before burning the supporter.
- Robustness: full try/except, per-move deadline, panic mode on low overage.
"""

from collections import Counter
import math
import os
import random
import sys
import time


# kaggle_environments exec()s this file, so __file__ may not exist
AGENT_DIR = "/kaggle_simulations/agent"
if not os.path.isdir(os.path.join(AGENT_DIR, "cg")):
    try:
        AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        AGENT_DIR = os.getcwd()
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

from cg.api import (  # noqa: E402
    AreaType,
    CardType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    to_observation_class,
)


SEARCH_OK = False
try:
    from cg.api import all_attack, all_card_data, search_begin, search_end, search_step

    SEARCH_OK = True
except Exception:
    from cg.api import all_attack, all_card_data

CARD = {}
ATK = {}
try:
    CARD = {c.cardId: c for c in all_card_data()}
    ATK = {a.attackId: a for a in all_attack()}
except Exception:
    pass

# ------------------------------------------------------------ deck config


def _load_configs():
    ns = {}
    path = os.path.join(AGENT_DIR, "deck_configs.py")
    with open(path, encoding="utf-8") as f:
        exec(compile(f.read(), path, "exec"), ns)
    return ns["DECK_CONFIGS"]


DECK_CONFIGS = _load_configs()
DECK_NAME = os.environ.get("CG_DECK_NAME", "starmie")
if DECK_NAME not in DECK_CONFIGS:
    DECK_NAME = "starmie"
CFG = DECK_CONFIGS[DECK_NAME]
DECK = list(CFG["deck"])
DECK_TOTAL = Counter(DECK)

BOSS = 1182
FILLER_BASIC = 1072  # Snorlax: safe opponent-active filler for search worlds
FILLER_CARD = 1
PIERCE_ATTACKS = {1488}  # Nebula Beam: ignores W/R and defender effects
EX_WALL = {345}  # Crustle: no damage from Pokemon ex / Mega ex
# basics that evolve into a big attacker: KOing them early starves the opponent
GUST_WORTH = {677, 741, 1030, 169, 722, 268, 646, 860, 878, 344, 673}
PANIC_RESERVE = 60.0
LOW_DECK = 6
# go-first preference at the opening coin (archetype unknown at that point)
IS_FIRST_YES = int(os.environ.get("CG_A2_FIRST_YES", "100"))

ENERGY_TYPES = (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)

# archetype signatures: flagship first, then early-game cards
_ARCH_FLAG = [
    (678, "lucario"),
    (648, "grimmsnarl"),
    (743, "alakazam"),
    (1031, "starmie"),
    (190, "archaludon"),
    (345, "crustle_wall"),
    (879, "trevenant"),
    (861, "froslass"),
    (756, "kangaskhan"),
    (723, "abomasnow"),
    (269, "bellibolt"),
]
_ARCH_EARLY = [
    (677, "lucario"),
    (646, "grimmsnarl"),
    (741, "alakazam"),
    (1030, "starmie"),
    (169, "archaludon"),
    (344, "crustle_wall"),
    (878, "trevenant"),
    (722, "abomasnow"),
    (268, "bellibolt"),
    (270, "bellibolt"),
    (271, "bellibolt"),
    # lucario-shell support basics (Makuhita/Lunatone/Solrock):
    # the public 1084 list opens with these long before Riolu shows
    (673, "lucario"),
    (674, "lucario"),
    (675, "lucario"),
    (676, "lucario"),
]


def prize_yield(cid):
    d = CARD.get(cid)
    if d is None:
        return 1
    if d.megaEx:
        return 3
    if d.ex:
        return 2
    return 1


def energy_units(cid, target_basic):
    """Energy units a hand card provides when attached to a Pokemon."""
    d = CARD.get(cid)
    if d is None:
        return [0]
    if d.cardType == CardType.BASIC_ENERGY:
        return [int(d.energyType)]
    if cid == 17:  # Ignition: CCC on an Evolution Pokemon, discarded at EOT
        return [0] if target_basic else [0, 0, 0]
    return [0]


def satisfies(attached, cost):
    """Can `attached` energy units pay `cost` (0 = colorless)?"""
    pool = list(attached)
    n_colorless = 0
    for t in cost:
        if t == 0:
            n_colorless += 1
            continue
        if t in pool:
            pool.remove(t)
        elif 10 in pool:  # rainbow
            pool.remove(10)
        elif t in (5, 7) and 11 in pool:  # team rocket = psychic/dark
            pool.remove(11)
        else:
            return False
    return len(pool) >= n_colorless


def eff_damage(att_id, def_id, attack):
    dmg = attack.damage or 0
    if attack.attackId in PIERCE_ATTACKS:
        return dmg
    a, d = CARD.get(att_id), CARD.get(def_id)
    if a is None or d is None:
        return dmg
    if def_id in EX_WALL and (a.ex or a.megaEx):
        return 0
    if d.weakness is not None and d.weakness == a.energyType:
        dmg *= 2
    elif d.resistance is not None and d.resistance == a.energyType:
        dmg = max(dmg - 30, 0)
    return dmg


class Plan:
    __slots__ = (
        "attacker_bi",
        "target_bi",
        "attack_id",
        "dmg",
        "ko",
        "wins",
        "needs_energy",
        "needs_gust",
        "needs_switch",
        "score",
        "attacker_serial",
        "target_serial",
        "retreat_fuel",
    )

    def __init__(self):
        self.attacker_bi = -1
        self.target_bi = -1
        self.attack_id = -1
        self.dmg = 0
        self.ko = False
        self.wins = False
        self.needs_energy = 0  # card id of energy to attach first, or 0
        self.needs_gust = False
        self.needs_switch = False
        self.score = -1.0
        self.attacker_serial = -1
        self.target_serial = -1
        self.retreat_fuel = 0  # energy card id to attach to ACTIVE so it can retreat


# persistent per-game state
_G = {
    "turn": -99,
    "plan": Plan(),
    "lethal_fail": set(),
    "arch": "unknown",
    "search_t": 0.0,
    "overage": None,
    "move_t0": 0.0,
}
IN_VERIFY = False

# ---------------------------------------------------------- search budget
MOVE_DEADLINE = 2.4  # hard per-move ceiling (s)
SEARCH_BUDGET = float(os.environ.get("CG_A2_SEARCH_BUDGET", "1.9"))  # per-search (s)
SEARCH_MAX_WORLDS = int(os.environ.get("CG_A2_SEARCH_WORLDS", "10"))
SEARCH_MAX_CANDS = int(os.environ.get("CG_A2_SEARCH_CANDS", "5"))
# ------------------------------------------------ within-turn beam search
# When on, det_search optimizes our WHOLE turn's action sequence per candidate
# first-move via a beam (width W, policy top-B expansion per node) instead of a
# single-first-move + policy rollout. The best end-of-turn states then get the
# opponent's reply modeled by policy to `horizon`, and the leaf is value-net
# scored. This is the AlphaZero-lite depth: good eval can now beat the policy's
# tactical mistakes because we actually search our sequence, not just move 1.
BEAM_ON = os.environ.get("CG_A2_BEAM", "1") not in ("", "0")
BEAM_WIDTH = int(os.environ.get("CG_A2_BEAM_WIDTH", "4"))  # live states kept/level
BEAM_EXPAND_B = int(os.environ.get("CG_A2_EXPAND_B", "3"))  # policy top-B per node
BEAM_DEPTH = int(os.environ.get("CG_A2_BEAM_DEPTH", "26"))  # max our-turn selects
BEAM_TERMK = int(os.environ.get("CG_A2_BEAM_TERMK", "3"))  # end-states opp-rolled
SEARCH_GAME_CAP = 110.0  # cumulative search seconds per game
SEARCH_OVERAGE_MIN = 90.0  # skip search below this remaining overage
# only override the policy when the challenger beats it by this mean margin
SEARCH_MARGIN = float(os.environ.get("CG_A2_SEARCH_MARGIN", "150"))
# player-turns to roll out: 2 = end of the opponent's reply turn
SEARCH_HORIZON = int(os.environ.get("CG_A2_SEARCH_HORIZON", "2"))
ROLLOUT_STEPS = 140
DEBUG = bool(os.environ.get("CG_A2_DEBUG"))
# decision classes disabled for search (comma list, e.g. "atk,promote")
SEARCH_OFF = frozenset(x for x in os.environ.get("CG_A2_SEARCH_OFF", "").split(",") if x)

# ---------------------------------------------- learned leaf value net
# CG_A2_VALUE_NET=1 replaces the handcrafted leaf eval in det_search with a
# small MLP win-prob net (value_net.npz), blended with a squashed handcrafted
# score. Leaf scores move to the [0, 1000] win-prob scale, so the override
# margin becomes CG_A2_VALUE_MARGIN (win-prob points per world, x10).
VALUE_NET_ON = os.environ.get("CG_A2_VALUE_NET", "1") not in ("", "0")
VALUE_BLEND = float(os.environ.get("CG_A2_VALUE_BLEND", "0.6"))
VALUE_MARGIN = float(os.environ.get("CG_A2_VALUE_MARGIN", "25"))
VN_OK = False
if VALUE_NET_ON:
    try:
        import value_features as _vf
        import value_net as _vn

        VN_OK = _vn.load(AGENT_DIR)
    except Exception:
        VN_OK = False
    if not VN_OK:
        # loud (once) so local runs don't silently measure the wrong agent:
        # league runs need `uv run --python 3.13 --with numpy ...`
        print(
            "[a2] WARNING value net requested but unavailable "
            "(numpy/value_net.npz); falling back to handcrafted eval",
            file=sys.stderr,
            flush=True,
        )


# search instrumentation (per game; printed on reset when CG_A2_DEBUG)
def _fresh_stats():
    return {
        "searches": 0,
        "t": 0.0,
        "worlds": 0,
        "cands": 0,
        "coins": 0,
        "overrides": 0,
        "fallbacks": 0,
        "moves": 0,
        "max_move": 0.0,
        "nodes": 0,
        "depth_sum": 0,
        "lines": 0,
        "opp_rollouts": 0,
        "by_class": Counter(),
    }


_S = _fresh_stats()


_MERGED_CFG = {}


def _merged_cfg(cfg, arch, ov):
    """cfg with per-matchup overrides applied (dict keys merge, others replace)."""
    key = (id(cfg), arch)
    m = _MERGED_CFG.get(key)
    if m is None:
        m = dict(cfg)
        for k, v in ov.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                mv = dict(cfg[k])
                mv.update(v)
                m[k] = mv
            else:
                m[k] = v
        _MERGED_CFG[key] = m
    return m


def get_card(obs, area, index, player_index):
    try:
        pl = obs.current.players[player_index]
        if area == AreaType.DECK:
            return obs.select.deck[index]
        if area == AreaType.HAND:
            return pl.hand[index]
        if area == AreaType.DISCARD:
            return pl.discard[index]
        if area == AreaType.ACTIVE:
            return pl.active[index]
        if area == AreaType.BENCH:
            return pl.bench[index]
        if area == AreaType.PRIZE:
            return pl.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING:
            return obs.current.looking[index]
    except Exception:
        return None
    return None


class Policy:
    def __init__(self, obs, cfg=None):
        self.obs = obs
        self.state = obs.current
        self.sel = obs.select
        self.ctx = self.sel.context
        self.me = self.state.yourIndex
        self.my = self.state.players[self.me]
        self.op = self.state.players[1 - self.me]
        self.cfg = cfg or CFG

        self.my_board = list(self.my.active or []) + list(self.my.bench or [])
        self.op_board = list(self.op.active or []) + list(self.op.bench or [])
        self.hand_counts = Counter(c.id for c in (self.my.hand or []) if c is not None)
        self.discard_counts = Counter(c.id for c in (self.my.discard or []) if c is not None)
        self.field_counts = Counter(p.id for p in self.my_board if p is not None)
        self.op_prizes = len(self.op.prize or [])
        self.my_prizes = len(self.my.prize or [])
        self._visible_seen = self._visible_own()
        self.arch = self._archetype()
        ov = self.cfg.get("matchup_overrides", {}).get(self.arch)
        if ov:
            self.cfg = _merged_cfg(self.cfg, self.arch, ov)
        self.can_retreat = False
        self.supporter_ok = not self.state.supporterPlayed
        if self.ctx == SelectContext.MAIN:
            for o in self.sel.option:
                if o.type == OptionType.RETREAT:
                    self.can_retreat = True

    # ------------------------------------------------------- bookkeeping

    def _archetype(self):
        seen = set()
        for p in self.op_board:
            if p is None:
                continue
            seen.add(p.id)
            for c in p.preEvolution or []:
                if c is not None:
                    seen.add(c.id)
        for c in self.op.discard or []:
            if c is not None:
                seen.add(c.id)
        for cid, name in _ARCH_FLAG:
            if cid in seen:
                return name
        for cid, name in _ARCH_EARLY:
            if cid in seen:
                return name
        return _G["arch"] if _G["arch"] != "unknown" else "unknown"

    def _visible_own(self):
        seen = Counter()
        for c in self.my.hand or []:
            if c is not None:
                seen[c.id] += 1
        for c in self.my.discard or []:
            if c is not None:
                seen[c.id] += 1
        for c in self.my.prize or []:
            if c is not None:
                seen[c.id] += 1
        for p in self.my_board:
            if p is None:
                continue
            seen[p.id] += 1
            for c in p.energyCards or []:
                seen[c.id] += 1
            for c in p.tools or []:
                seen[c.id] += 1
            for c in p.preEvolution or []:
                if c is not None:
                    seen[c.id] += 1
        for c in self.state.stadium or []:
            if c is not None and c.playerIndex == self.me:
                seen[c.id] += 1
        return seen

    def copies_hidden(self, cid):
        """Copies of cid still in our deck or facedown prizes."""
        return max(DECK_TOTAL.get(cid, 0) - self._visible_seen.get(cid, 0), 0)

    def line_count(self, base_id):
        line = self.cfg["lines"].get(base_id, (base_id,))
        return sum(self.field_counts.get(i, 0) for i in line)

    def line_cap_ok(self, cid):
        """True if we still want another copy of this line in play."""
        for base, line in self.cfg["lines"].items():
            if cid in line:
                return self.line_count(base) < self.cfg["bench_cap"].get(base, 1)
        return sum(self.field_counts.values()) < 4

    def opp_ability_count(self):
        n = 0
        for p in self.op_board:
            if p is None:
                continue
            d = CARD.get(p.id)
            if d is not None and d.skills:
                n += 1
        return n

    def my_damage_total(self):
        return sum((p.maxHp - p.hp) for p in self.my_board if p is not None)

    # ------------------------------------------------------- attack planner

    def plan_attack(self, exclude=()):
        plan = Plan()
        boss_ok = self.hand_counts.get(BOSS, 0) > 0 and self.supporter_ok
        # can we unlock a retreat by attaching one energy to the active?
        retreat_fuel = 0
        active = self.my_board[0] if self.my_board else None
        if (
            not self.can_retreat
            and active is not None
            and not self.state.energyAttached
            and not self.state.retreated
        ):
            ad = CARD.get(active.id)
            if ad is not None and ad.retreatCost - len(active.energies or []) == 1:
                for cid, cnt in self.hand_counts.items():
                    d = CARD.get(cid)
                    if cnt > 0 and d is not None and d.cardType in ENERGY_TYPES:
                        retreat_fuel = cid
                        break
        for bi, p in enumerate(self.my_board):
            if p is None:
                continue
            if bi > 0 and not self.can_retreat and not retreat_fuel:
                continue
            uses_fuel = bi > 0 and not self.can_retreat
            pd = CARD.get(p.id)
            if pd is None:
                continue
            for aid in pd.attacks or []:
                a = ATK.get(aid)
                if a is None:
                    continue
                attached = [int(e) for e in (p.energies or [])]
                need_card = 0
                if not satisfies(attached, a.energies):
                    # can one attach from hand enable this attack?
                    if self.state.energyAttached or uses_fuel:
                        continue  # only one manual attach per turn
                    for cid, cnt in self.hand_counts.items():
                        if cnt <= 0:
                            continue
                        d = CARD.get(cid)
                        if d is None or d.cardType not in ENERGY_TYPES:
                            continue
                        if satisfies(attached + energy_units(cid, pd.basic), a.energies):
                            need_card = cid
                            break
                    if not need_card:
                        continue
                snipe = self.cfg.get("bench_snipe", {}).get(aid, 0)
                bonus = self.cfg.get("attack_bonus", {}).get(aid, 0)
                if aid == 1488:
                    bonus += 5  # Nebula pierces protections: break ties its way
                if aid == 965:  # Turbo Flare: worth a lot while the bench is hungry
                    deficit = 0
                    for pb in self.my_board[1:]:
                        if pb is None:
                            continue
                        for pid, cap in self.cfg["energy_route"]:
                            if pb.id == pid:
                                deficit += max(cap - len(pb.energies or []), 0)
                                break
                    bonus += 70 * min(deficit, 4)
                for ti, q in enumerate(self.op_board):
                    if q is None:
                        continue
                    if ti > 0 and not boss_ok:
                        continue
                    if (p.serial, q.serial, aid) in exclude:
                        continue
                    dmg = eff_damage(p.id, q.id, a)
                    if dmg <= 0 and bonus <= 0:
                        continue
                    ko = dmg >= q.hp and dmg > 0
                    py = prize_yield(q.id)
                    wins = ko and self.op_prizes <= py
                    if ti > 0 and not (
                        ko and (wins or py >= 2 or len(q.energies or []) >= 3 or q.id in GUST_WORTH)
                    ):
                        continue  # Boss is precious: gust only for KOs that matter
                    if wins:
                        score = 10_000_000.0
                    elif ko:
                        score = 100_000.0 * py + dmg
                    else:
                        score = 40_000.0 * min(dmg / max(q.hp, 1), 1.0) + bonus * 60
                    # bench snipe upside (target picked in a sub-select)
                    if snipe:
                        for q2 in self.op_board[1:]:
                            if q2 is not None and 0 < q2.hp <= snipe:
                                sp = prize_yield(q2.id)
                                if ko and self.op_prizes <= py + sp:
                                    score = 10_000_000.0
                                else:
                                    score += 50_000.0 * sp
                                break
                    score += bonus * 20 + dmg * 0.01
                    if ti > 0 and not wins:
                        score -= 30_000.0
                    if bi > 0:
                        score -= 8_000.0
                        if uses_fuel:
                            score -= 4_000.0  # burns the attach + an energy
                    if need_card:
                        score -= 300.0
                    if score > plan.score:
                        plan.score = score
                        plan.attacker_bi, plan.target_bi = bi, ti
                        plan.attack_id, plan.dmg = aid, dmg
                        plan.ko, plan.wins = ko, wins
                        plan.needs_energy = need_card
                        plan.needs_gust = ti > 0
                        plan.needs_switch = bi > 0
                        plan.attacker_serial = p.serial
                        plan.target_serial = q.serial
                        plan.retreat_fuel = retreat_fuel if uses_fuel else 0
        return plan

    # ------------------------------------------------------- need scoring

    def need_score(self, cid):
        """How much we want this card in hand right now."""
        d = CARD.get(cid)
        if d is None:
            return 5
        base = self.cfg["fetch_prio"].get(cid, 25 if d.cardType == CardType.POKEMON else 15)
        s = float(base)
        s -= 30.0 * self.hand_counts.get(cid, 0)
        if d.cardType == CardType.POKEMON:
            if d.basic:
                s += 25 if self.line_cap_ok(cid) else -40
            else:
                pres = self.cfg["evo_top"].get(cid, ())
                if pres:  # a line top (e.g. Grimmsnarl ex, Mega Starmie ex)
                    base_id = pres[0]
                    mid_id = pres[1] if len(pres) > 1 else None
                    if self.hand_counts.get(cid, 0) >= 1:
                        s -= 60  # one copy in hand is enough; fetch the bottleneck
                    elif mid_id is None:
                        if self.field_counts.get(base_id, 0):
                            s += 70
                    else:
                        can_jump = self.hand_counts.get(1079, 0) > 0 and self.field_counts.get(
                            base_id, 0
                        )
                        if self.field_counts.get(mid_id, 0) or can_jump:
                            s += 70
                        elif not self.field_counts.get(base_id, 0):
                            s -= 25
                else:
                    # a middle stage: bottleneck when base in play + top in hand
                    for top, pr in self.cfg["evo_top"].items():
                        if len(pr) > 1 and pr[1] == cid:
                            if (
                                self.field_counts.get(pr[0], 0)
                                and self.hand_counts.get(top, 0)
                                and not self.hand_counts.get(1079, 0)
                            ):
                                s += 65
        elif d.cardType in ENERGY_TYPES:
            plan = _G["plan"]
            if plan.needs_energy == cid:
                s += 50
            n_energy_hand = sum(
                c
                for i, c in self.hand_counts.items()
                if CARD.get(i) is not None and CARD[i].cardType in ENERGY_TYPES
            )
            s -= 12 * n_energy_hand
        elif d.cardType == CardType.SUPPORTER:
            has_sup = any(
                CARD.get(i) is not None and CARD[i].cardType == CardType.SUPPORTER
                for i in self.hand_counts
            )
            if not has_sup:
                s += 20
        if cid == 1079:  # Rare Candy: gold when a stage-2 jump is brewing
            for top, pres in self.cfg["evo_top"].items():
                td = CARD.get(top)
                if td is not None and td.stage2 and pres:
                    if any(p is not None and p.id == pres[0] for p in self.my_board):
                        if self.hand_counts.get(top, 0):
                            s += 65
                        elif self.copies_hidden(top):
                            s += 30
        if cid == 1086 and self.line_count(next(iter(self.cfg["bench_cap"]), 0)) < 2:
            s += 25  # board is thin: basics fetcher gains value
        if cid == BOSS and _G["plan"].needs_gust:
            s += 60
        return s

    # ------------------------------------------------------- MAIN scoring

    def sc_ability(self, o):
        card = get_card(
            self.obs, o.area, o.index, o.playerIndex if o.playerIndex is not None else self.me
        )
        cid = getattr(card, "id", 0)
        if cid == 1259:  # Spikemuth Gym: fetch a Marnie's pokemon
            best = max(
                (self.need_score(i) for i in (646, 647, 648) if self.copies_hidden(i) > 0),
                default=-1,
            )
            return 21000 if best > 0 else -1
        if cid == 112:  # Munkidori Adrena-Brain
            return 20000 if self.my_damage_total() > 0 else -1
        return 19000

    def sc_play(self, o):
        card = get_card(self.obs, AreaType.HAND, o.index, self.me)
        if card is None:
            return 0
        cid = card.id
        d = CARD.get(cid)
        if d is None:
            return 100
        if d.cardType == CardType.POKEMON:
            if not self.line_cap_ok(cid):
                return -1
            if cid in self.cfg.get("ability_tech", ()) and self.opp_ability_count() == 0:
                return 450  # anti-ability tech is a liability vs this opponent
            return 20000 + self.cfg["bench_value"].get(cid, 10)
        if d.cardType == CardType.SUPPORTER:
            return self.sc_supporter(cid)
        if d.cardType == CardType.STADIUM:
            cur = self.state.stadium[0] if self.state.stadium else None
            if cur is not None and cur.id == cid and cur.playerIndex == self.me:
                return -1
            return 8600
        # items
        if cid == 1086:  # Poffin
            if len(self.my.bench or []) >= self.my.benchMax:
                return -1
            want = any(
                CARD.get(i) is not None
                and CARD[i].cardType == CardType.POKEMON
                and CARD[i].basic
                and CARD[i].hp <= 70
                and self.line_cap_ok(i)
                and self.copies_hidden(i) > 0
                for i in DECK_TOTAL
            )
            return 12000 if want else -1
        if cid == 1079:  # Rare Candy (engine only offers when legal)
            return 11000
        if cid == 1080:  # Unfair Stamp
            return 8000
        if cid == 1121:  # Ultra Ball
            best = max(
                (
                    self.need_score(i)
                    for i in DECK_TOTAL
                    if CARD.get(i) is not None
                    and CARD[i].cardType == CardType.POKEMON
                    and self.copies_hidden(i) > 0
                ),
                default=-1,
            )
            return 9000 if best > 40 else 400
        if cid == 1122:  # Pokegear
            has_sup = any(
                CARD.get(i) is not None and CARD[i].cardType == CardType.SUPPORTER
                for i in self.hand_counts
            )
            if not has_sup:
                return 8200
            # matchup knob: bank the matchup supporter (e.g. Wally vs
            # lucario) even when we already hold some other supporter
            bank = self.cfg.get("gear_bank", 0)
            if bank and not self.hand_counts.get(bank, 0) and self.copies_hidden(bank) > 0:
                return 7800
            return 300
        if cid == 1120:  # Crushing Hammer
            if any(p is not None and (p.energyCards or []) for p in self.op_board):
                return 4000
            return -1
        if cid == 1097:  # Night Stretcher
            best = max(
                (
                    self.need_score(c.id)
                    for c in (self.my.discard or [])
                    if c is not None
                    and CARD.get(c.id) is not None
                    and (
                        CARD[c.id].cardType == CardType.POKEMON
                        or CARD[c.id].cardType == CardType.BASIC_ENERGY
                    )
                ),
                default=-1,
            )
            return 500 + best * 20 if best > 30 else -1
        if cid == 1145:  # Mega Signal
            for top in self.cfg["evo_top"]:
                td = CARD.get(top)
                if (
                    td is not None
                    and td.megaEx
                    and self.hand_counts.get(top, 0) == 0
                    and self.copies_hidden(top) > 0
                ):
                    return 10000
            return -1
        if cid == 1152:  # Poke Pad
            best = max(
                (
                    self.need_score(i)
                    for i in DECK_TOTAL
                    if CARD.get(i) is not None
                    and CARD[i].cardType == CardType.POKEMON
                    and not CARD[i].ex
                    and not CARD[i].megaEx
                    and self.copies_hidden(i) > 0
                ),
                default=-1,
            )
            return 9200 if best > 35 else -1
        return 9800  # unknown item: probably fine to play

    def sc_supporter(self, cid):
        plan = _G["plan"]
        if cid == BOSS:
            return 15000 if plan.needs_gust else -1
        if cid == 1189:  # Salvatore: instant evolve (engine offers when legal)
            jit = self.cfg.get("jit_evolve", ())
            if jit:
                act = self.my_board[0] if self.my_board else None
                bases = {pr[0] for top, pr in self.cfg["evo_top"].items() if top in jit}
                if act is not None and act.id in bases:
                    return 14000
                return -1  # don't create a benched 3-prize target
            return 14000
        if cid == 1229:  # Wally's Compassion
            best, fuel = 0, self.hand_counts.get(17, 0) + self.hand_counts.get(3, 0)
            for p in self.my_board:
                if p is not None and CARD.get(p.id) is not None and CARD[p.id].megaEx:
                    dmg = p.maxHp - p.hp
                    if dmg > best:
                        best = dmg
                        # the heal returns this mega's energy to hand: that IS fuel
                        fuel += len(p.energyCards or [])
            if best >= 240 and fuel:
                return 14500  # undo a full Mega Brave: beats even Salvatore
            if best >= 120 and fuel:
                return 12500
            return 9000 if best >= 240 else -1
        if cid == 1231:  # Dawn: fetch basic + stage1 + stage2
            missing = sum(
                1
                for i in (646, 647, 648)
                if self.hand_counts.get(i, 0) == 0 and self.copies_hidden(i) > 0
            )
            return 8500 if missing >= 2 else 7200
        if cid == 1225:  # Hilda: evolution pokemon + energy
            for top in self.cfg["evo_top"]:
                if self.hand_counts.get(top, 0) == 0 and self.copies_hidden(top) > 0:
                    return 8300
            return 5800
        if cid == 1219:  # Petrel: fetch any trainer
            best = max(
                (
                    self.need_score(i)
                    for i in DECK_TOTAL
                    if CARD.get(i) is not None
                    and CARD[i].cardType
                    in (CardType.ITEM, CardType.TOOL, CardType.SUPPORTER, CardType.STADIUM)
                    and self.copies_hidden(i) > 0
                ),
                default=0,
            )
            return 7000 + best * 6
        if cid in (1227, 1223):  # hand refresh
            if self.my.deckCount < 10:
                return -1
            useful = self._hand_useful()
            base = 6800 if cid == 1227 else 2500
            return max(base - 1400 * useful, -1)
        return self.cfg.get("supporter_base", {}).get(cid, 5000)

    def _hand_useful(self):
        n = 0
        for cid, cnt in self.hand_counts.items():
            d = CARD.get(cid)
            if d is None:
                continue
            if d.cardType == CardType.POKEMON:
                if d.basic and self.line_cap_ok(cid):
                    n += cnt
                elif not d.basic:
                    pres = self.cfg["evo_top"].get(cid, ())
                    if any(p is not None and p.id in pres for p in self.my_board):
                        n += cnt
            elif d.cardType in (CardType.ITEM, CardType.TOOL):
                n += cnt
            elif d.cardType in ENERGY_TYPES and not self.state.energyAttached:
                n += 1
        return n

    def sc_attach(self, o):
        card = get_card(self.obs, o.area, o.index, self.me)
        target = get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.me)
        if card is None or not isinstance(target, Pokemon):
            return 0
        cid = card.id
        d = CARD.get(cid)
        bi = o.inPlayIndex if o.inPlayArea == AreaType.ACTIVE else o.inPlayIndex + 1
        plan = _G["plan"]
        if d is not None and d.cardType == CardType.TOOL:
            route = self.cfg.get("tool_route", {}).get(cid, {})
            v = route.get(target.id, -1)
            if v < 0:
                # only slap a tool on an off-plan target if nothing better exists
                better = any(
                    p is not None and p.id in route and not (p.tools or []) for p in self.my_board
                )
                return -1 if better else 900
            # hold the tool if a strictly better holder is available or coming
            for i in (i for i, x in route.items() if x > v):
                if any(p is not None and p.id == i and not (p.tools or []) for p in self.my_board):
                    return -1  # better holder in play with a free slot: use it
                if not self.field_counts.get(i, 0) and (
                    self.hand_counts.get(i, 0) or self.copies_hidden(i)
                ):
                    return -1  # better holder still coming: hold
            return 7000 + v
        # energy
        if cid == 17:  # Ignition: burn only to enable the planned attack
            td = CARD.get(target.id)
            if td is not None and td.basic:
                return -1
            if plan.needs_energy == 17 and bi == plan.attacker_bi:
                return 9500
            return -1
        if plan.needs_energy == cid and bi == plan.attacker_bi:
            return 9500
        if plan.retreat_fuel == cid and bi == 0:
            return 9400  # unlock the active's retreat for the planned switch
        for pos, (pid, cap) in enumerate(self.cfg["energy_route"]):
            if target.id == pid:
                if len(target.energies or []) < cap:
                    return 6500 - pos * 100
                return 150 - pos
        return 100

    def sc_evolve(self, o):
        card = get_card(self.obs, o.area, o.index, self.me)
        target = get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.me)
        cid = getattr(card, "id", 0)
        if cid == 104 and self.opp_ability_count() == 0:
            return 550  # Froslass shroud would only chip our own board
        if cid in self.cfg.get("jit_evolve", ()) and o.inPlayArea == AreaType.BENCH:
            # a benched Mega is a 3-prize gust target with no upside until
            # it attacks: keep the 1-prize basic and evolve when promoted
            return 540
        s = 9000.0
        if cid in self.cfg["evo_top"]:
            s += 500
        if isinstance(target, Pokemon):
            s += len(target.energies or []) * 10
            s -= (target.maxHp - target.hp) * 0.5
        if o.inPlayArea == AreaType.ACTIVE:
            s += 20
        return s

    def sc_attack(self, o):
        plan = _G["plan"]
        if plan.score > 0:
            if plan.needs_energy or plan.needs_gust or plan.needs_switch or plan.retreat_fuel:
                return 40  # do the setup first; replan will re-enable
            if o.attackId == plan.attack_id and plan.target_bi == 0:
                return 900
            a = ATK.get(o.attackId)
            return 300 + (a.damage or 0) * 0.1 if a else 300
        a = ATK.get(o.attackId)
        if a is None:
            return 10
        q = self.op_board[0] if self.op_board else None
        me_active = self.my_board[0] if self.my_board else None
        dmg = (
            eff_damage(me_active.id, q.id, a)
            if (q is not None and me_active is not None)
            else (a.damage or 0)
        )
        bonus = self.cfg.get("attack_bonus", {}).get(o.attackId, 0)
        if dmg <= 0 and bonus <= 0:
            return 5
        return 200 + dmg * 0.5 + bonus

    # ------------------------------------------------------- card contexts

    def sc_card(self, o):
        card = get_card(
            self.obs, o.area, o.index, o.playerIndex if o.playerIndex is not None else self.me
        )
        ctx = self.ctx
        plan = _G["plan"]
        if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
            return self.cfg["setup_active"].get(getattr(card, "id", 0), 1)
        if ctx == SelectContext.SETUP_BENCH_POKEMON:
            return 10 + self.cfg["bench_value"].get(getattr(card, "id", 0), 0)
        if ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            return self.sc_promote(o, card)
        if ctx in (SelectContext.TO_BENCH, SelectContext.TO_FIELD):
            cid = getattr(card, "id", 0)
            if not self.line_cap_ok(cid):
                return -1
            if cid in self.cfg.get("ability_tech", ()) and self.opp_ability_count() == 0:
                return 2
            return 100 + self.cfg["bench_value"].get(cid, 5)
        if ctx == SelectContext.TO_HAND:
            return self.need_score(getattr(card, "id", 0))
        if ctx in (
            SelectContext.DISCARD,
            SelectContext.TO_DECK,
            SelectContext.TO_DECK_BOTTOM,
            SelectContext.TO_PRIZE,
        ):
            return 200 - self.need_score(getattr(card, "id", 0))
        if ctx == SelectContext.NOT_MOVE:
            return self.need_score(getattr(card, "id", 0))
        if ctx in (
            SelectContext.DAMAGE_COUNTER,
            SelectContext.DAMAGE_COUNTER_ANY,
            SelectContext.DAMAGE,
        ):
            return self.sc_damage_target(o, card)
        if ctx in (SelectContext.REMOVE_DAMAGE_COUNTER, SelectContext.HEAL):
            if isinstance(card, Pokemon):
                return (card.maxHp - card.hp) + self.cfg["attacker_value"].get(card.id, 0) * 0.5
            return 0
        if ctx == SelectContext.EVOLVES_FROM:
            if isinstance(card, Pokemon):
                s = len(card.energies or []) * 10 + self.cfg["attacker_value"].get(card.id, 0)
                if o.area == AreaType.ACTIVE:
                    s += 500  # evolve the active, never a benched gust target
                return s
            return 0
        if ctx in (SelectContext.EVOLVES_TO, SelectContext.DEVOLVE):
            return self.need_score(getattr(card, "id", 0))
        if ctx == SelectContext.ATTACH_FROM:
            # choose which of our pokemon receives an energy (Punk Up, Turbo Flare)
            if isinstance(card, Pokemon):
                plan_bonus = 0
                if plan.needs_energy and card.serial == plan.attacker_serial:
                    plan_bonus = 3000
                for pos, (pid, cap) in enumerate(self.cfg["energy_route"]):
                    if card.id == pid:
                        if len(card.energies or []) < cap:
                            return 1000 - pos * 50 + plan_bonus
                        return 20 - pos + plan_bonus
                return 10 + plan_bonus
            return 0
        if ctx == SelectContext.ATTACH_TO:
            return self.need_score(getattr(card, "id", 0))
        if ctx == SelectContext.EFFECT_TARGET:
            if isinstance(card, Pokemon):
                if o.playerIndex == self.me:
                    return self.cfg["attacker_value"].get(card.id, 5)
                return prize_yield(card.id) * 10 + len(card.energies or [])
            return 1
        if ctx == SelectContext.LOOK:
            return 1
        # unknown card context: mildly prefer cards we value
        return self.need_score(getattr(card, "id", 0)) * 0.1 + 1

    def sc_promote(self, o, card):
        """SWITCH / TO_ACTIVE: our promote or opponent gust target."""
        plan = _G["plan"]
        if o.playerIndex is not None and o.playerIndex != self.me:
            # gust: pull the planned target
            if isinstance(card, Pokemon) and card.serial == plan.target_serial:
                return 5000
            if isinstance(card, Pokemon):
                if card.id in EX_WALL:
                    return -50
                s = prize_yield(card.id) * 20 + len(card.energies or []) * 5 - card.hp * 0.02
                # prefer something we can actually kill
                if self.my_board and self.my_board[0] is not None:
                    pd = CARD.get(self.my_board[0].id)
                    for aid in (pd.attacks or []) if pd else []:
                        a = ATK.get(aid)
                        if a and eff_damage(self.my_board[0].id, card.id, a) >= card.hp:
                            s += 500
                return s
            return 0
        if not isinstance(card, Pokemon):
            return 0
        s = float(self.cfg["attacker_value"].get(card.id, 5))
        s += len(card.energies or []) * 8
        d = CARD.get(card.id)
        if d is not None:
            for aid in d.attacks or []:
                a = ATK.get(aid)
                if a is not None and satisfies([int(e) for e in (card.energies or [])], a.energies):
                    s += 60 + (a.damage or 0) * 0.2
                    break
            s -= d.retreatCost * 4
        if card.serial == plan.attacker_serial:
            s += 1000
        s += card.hp * 0.15  # sturdier promotes bleed fewer prizes
        return s

    def sc_damage_target(self, o, card):
        if not isinstance(card, Pokemon):
            return 0
        if o.playerIndex == self.me:
            return -50  # never place counters on ourselves if avoidable
        cap = self.sel.remainDamageCounter or 0
        dmg = cap * 10
        s = prize_yield(card.id) * 30 + len(card.energies or []) * 3
        if dmg and card.hp <= dmg:
            s += 2000 + prize_yield(card.id) * 1000
            if self.op_prizes <= prize_yield(card.id):
                s += 100000
        else:
            s += (card.maxHp - card.hp) * 0.3
            plan = _G["plan"]
            if card.serial == plan.target_serial:
                s += 300
        return s

    # ------------------------------------------------------- other types

    def sc_attached(self, o):
        """TOOL_CARD / ENERGY_CARD options (e.g. Crushing Hammer target)."""
        holder = get_card(self.obs, o.area, o.index, o.playerIndex)
        if o.playerIndex is not None and o.playerIndex != self.me:
            s = 100.0
            if isinstance(holder, Pokemon):
                s += 50 if o.area == AreaType.ACTIVE else 0
                s += len(holder.energies or []) * 8 + prize_yield(holder.id) * 10
            return s
        # our own attachment being discarded: prefer losing Ignition & co
        cid = 0
        if isinstance(holder, Pokemon):
            try:
                if o.energyIndex is not None:
                    cid = holder.energyCards[o.energyIndex].id
                elif o.toolIndex is not None:
                    cid = holder.tools[o.toolIndex].id
            except Exception:
                cid = 0
        if cid == 17:
            return 100
        if CARD.get(cid) is not None and CARD[cid].cardType == CardType.SPECIAL_ENERGY:
            return 60
        return 10

    def sc_energy(self, o):
        """SelectType.ENERGY options (retreat cost, discard effects)."""
        holder = get_card(self.obs, o.area, o.index, o.playerIndex)
        if o.playerIndex is not None and o.playerIndex != self.me:
            return 100
        cid = 0
        if isinstance(holder, Pokemon):
            try:
                if o.energyIndex is not None:
                    cid = holder.energyCards[o.energyIndex].id
            except Exception:
                cid = 0
        if cid == 17:
            return 100
        if CARD.get(cid) is not None and CARD[cid].cardType == CardType.SPECIAL_ENERGY:
            return 60
        return 10

    def sc_yes(self):
        ctx = self.ctx
        if ctx == SelectContext.IS_FIRST:
            return IS_FIRST_YES
        if ctx == SelectContext.MULLIGAN:
            # a "starter" is a true basic OR a setup-playable pokemon like
            # Cinderace (Explosiveness: may start active from hand). Redrawing
            # away double-Cinderace hands is how we ended up in lone-Staryu
            # turn-2 donk losses on the ladder.
            has_starter = any(
                CARD.get(c.id) is not None
                and CARD[c.id].cardType == CardType.POKEMON
                and (CARD[c.id].basic or c.id in self.cfg["setup_active"])
                for c in (self.my.hand or [])
                if c is not None
            )
            return -10 if has_starter else 100
        return 50  # ACTIVATE / FIRST_EFFECT / COIN_HEAD: default yes

    def sc_number(self, o):
        n = o.number or 0
        if self.ctx == SelectContext.DRAW_COUNT:
            if self.my.deckCount - n < 1:
                return n - 1000
            return n
        return n

    # ------------------------------------------------------- dispatch

    def score_option(self, o):
        t = o.type
        if t == OptionType.NUMBER:
            return self.sc_number(o)
        if t == OptionType.YES:
            return self.sc_yes()
        if t == OptionType.NO:
            return 0
        if t == OptionType.CARD:
            return self.sc_card(o)
        if t in (OptionType.TOOL_CARD, OptionType.ENERGY_CARD):
            return self.sc_attached(o)
        if t == OptionType.ENERGY:
            return self.sc_energy(o)
        if t == OptionType.PLAY:
            return self.sc_play(o)
        if t == OptionType.ATTACH:
            return self.sc_attach(o)
        if t == OptionType.EVOLVE:
            return self.sc_evolve(o)
        if t == OptionType.ABILITY:
            return self.sc_ability(o)
        if t == OptionType.RETREAT:
            return 2600 if _G["plan"].needs_switch and _G["plan"].score > 0 else -5
        if t == OptionType.ATTACK:
            return self.sc_attack(o)
        if t == OptionType.DISCARD:
            return -10
        if t == OptionType.END:
            # positive default: never burn Boss/retreat/etc. as a "least bad" pick
            return 1
        if t == OptionType.SKILL:
            return 1
        if t == OptionType.SPECIAL_CONDITION:
            return 1
        return 0

    COST_CONTEXTS = {
        SelectContext.DISCARD,
        SelectContext.TO_DECK,
        SelectContext.TO_DECK_BOTTOM,
        SelectContext.TO_PRIZE,
        SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
        SelectContext.DISCARD_ENERGY,
        SelectContext.DISCARD_ENERGY_CARD,
        SelectContext.DISCARD_TOOL_CARD,
    }

    def choose(self):
        sel = self.sel
        n = len(sel.option)
        if n == 0 or sel.maxCount == 0:
            return []
        if self.ctx == SelectContext.MAIN:
            _G["plan"] = self.plan_attack()
            self.maybe_verify_lethal()
        scores = [self.score_option(o) for o in sel.option]
        lo = max(sel.minCount, 0)
        hi = min(sel.maxCount, n)
        order = sorted(range(n), key=lambda i: -scores[i])
        cost_ctx = self.ctx in self.COST_CONTEXTS and all(
            o.playerIndex in (None, self.me) for o in sel.option
        )
        if cost_ctx:
            chosen = order[:lo]
        else:
            chosen = [i for i in order if scores[i] > 0][:hi]
            if len(chosen) < lo:
                extra = [i for i in order if i not in chosen]
                chosen += extra[: lo - len(chosen)]
        if not chosen and lo > 0:
            chosen = order[:lo]
        # pivotal-decision determinized search: re-pick single selections
        if len(chosen) == 1 and not IN_VERIFY and SEARCH_OK:
            try:
                klass, cands = self._search_candidates(scores, order)
            except Exception:
                klass, cands = None, None
            if cands and len(cands) >= 2 and chosen[0] in cands and klass not in SEARCH_OFF:
                # policy pick first: det_search treats cands[0] as incumbent
                cands.remove(chosen[0])
                cands.insert(0, chosen[0])
                best = run_search(self, cands)
                if best is not None:
                    _S["by_class"][klass + ":s"] += 1
                    if best != chosen[0]:
                        _S["overrides"] += 1
                        _S["by_class"][klass + ":o"] += 1
                        chosen = [best]
        return chosen

    # ---------------------------------------------- pivotal candidate sets

    def _jit_attack_ready(self, o):
        """Would the evolved active be able to attack immediately?"""
        if o.inPlayArea != AreaType.ACTIVE:
            return False
        card = get_card(self.obs, o.area, o.index, self.me)
        cid = getattr(card, "id", 0)
        if cid not in self.cfg.get("jit_evolve", ()):
            return False
        target = get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.me)
        if not isinstance(target, Pokemon):
            return False
        att = [int(e) for e in (target.energies or [])]
        d = CARD.get(cid)
        for aid in (d.attacks or []) if d is not None else []:
            a = ATK.get(aid)
            if a is not None and satisfies(att, a.energies):
                return True
        return False

    def _search_candidates(self, scores, order):
        """Return (class, option indices) worth a determinized search.

        Only pivotal single-pick decision classes are searched. Candidates
        the policy suppresses for gust-safety config reasons (bench
        jit-evolve, ability tech, EX walls: scores < 600 in MAIN, <= 0 in
        card contexts) are deliberately excluded: opponents in search worlds
        hold filler hands and can never Boss, so a rollout would
        systematically override that config wisdom.
        """
        sel = self.sel
        if sel.maxCount != 1 or len(sel.option) < 2:
            return None, None
        ctx = self.ctx
        top = order[0]
        topo = sel.option[top]
        plan = _G["plan"]
        if ctx == SelectContext.MAIN:
            tt = topo.type
            if tt in (OptionType.ATTACK, OptionType.END):
                # attack choice / timing, but only when there is no clean KO
                if plan.wins:
                    return None, None
                if (
                    plan.ko
                    and plan.target_bi == 0
                    and not (plan.needs_energy or plan.needs_switch or plan.retreat_fuel)
                ):
                    return None, None
                cands = [
                    i for i in order if sel.option[i].type == OptionType.ATTACK and scores[i] > 0
                ][: SEARCH_MAX_CANDS - 1]
                endi = next((i for i in order if sel.option[i].type == OptionType.END), None)
                if endi is not None and endi not in cands:
                    cands.append(endi)
                return "atk", (cands if len(cands) >= 2 else None)
            klass = None
            if tt == OptionType.PLAY:
                card = get_card(self.obs, AreaType.HAND, topo.index, self.me)
                if getattr(card, "id", 0) == BOSS:
                    klass = "boss"  # Wally heal-timing tested: -1pt, reverted
            elif tt == OptionType.EVOLVE:
                if self._jit_attack_ready(topo):
                    klass = "jit"
            elif tt == OptionType.RETREAT:
                klass = "retreat"
            if klass is not None:
                # boss timing / retreat / jit-evolve: top vs real alternatives
                cands = [top]
                for i in order[1:]:
                    o = sel.option[i]
                    if o.type == OptionType.END or scores[i] >= 600:
                        cands.append(i)
                    if len(cands) >= SEARCH_MAX_CANDS:
                        break
                return klass, (cands if len(cands) >= 2 else None)
            # whole-turn sequence search: beam our action sequence from this
            # early constructive decision to end-of-turn. This is the depth the
            # policy lacks -- ordering plays/attaches/evolves so the turn's END
            # state is best, not just greedily by per-move score. Skip when a
            # lethal is already lined up (policy executes it) and when the top
            # move isn't a build-up action.
            if not BEAM_ON or plan.wins:
                return None, None
            _CONSTRUCTIVE = (
                OptionType.PLAY,
                OptionType.ATTACH,
                OptionType.EVOLVE,
                OptionType.ABILITY,
                OptionType.RETREAT,
                OptionType.ATTACK,
            )
            if tt not in _CONSTRUCTIVE:
                return None, None
            cands = [i for i in order if sel.option[i].type in _CONSTRUCTIVE and scores[i] > 0][
                :SEARCH_MAX_CANDS
            ]
            endi = next((i for i in order if sel.option[i].type == OptionType.END), None)
            if endi is not None and endi not in cands and len(cands) < SEARCH_MAX_CANDS:
                cands.append(endi)
            return "seq", (cands if len(cands) >= 2 else None)
        if ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            # gust target (opponent options) or our promotion / retreat target
            gust = any(o.playerIndex is not None and o.playerIndex != self.me for o in sel.option)
            cands = [i for i in order if scores[i] > 0][:SEARCH_MAX_CANDS]
            return ("gust" if gust else "promote"), (cands if len(cands) >= 2 else None)
        if ctx == SelectContext.DISCARD_ENERGY_CARD:
            # Crushing Hammer & co: which opponent energy to remove
            cands = [
                i
                for i in order
                if sel.option[i].playerIndex is not None and sel.option[i].playerIndex != self.me
            ][:SEARCH_MAX_CANDS]
            return "hammer", (cands if len(cands) >= 2 else None)
        return None, None

    def beam_children(self, b):
        """Top-b next-action choices from THIS select, for the beam expander.

        Returns a list of `select` lists (each a valid argument to
        search_step). Only branches single-pick (maxCount==1) selects; for
        multi-pick / cost contexts it returns the one policy choice so the
        beam does not explode on sub-select combinatorics.
        """
        sel = self.sel
        n = len(sel.option)
        if n == 0 or sel.maxCount == 0:
            return [[]]
        if self.ctx == SelectContext.MAIN:
            _G["plan"] = self.plan_attack()
        if sel.maxCount != 1 or n < 2:
            try:
                return [self.choose()]
            except Exception:
                return [_fallback_choice(sel)]
        scores = [self.score_option(o) for o in sel.option]
        order = sorted(range(n), key=lambda i: -scores[i])
        cand = [i for i in order if scores[i] > 0][:b]
        if not cand:
            cand = order[:1]
        return [[i] for i in cand]

    # ------------------------------------------------------- lethal verify

    def maybe_verify_lethal(self):
        global IN_VERIFY
        plan = _G["plan"]
        if IN_VERIFY or not SEARCH_OK or not plan.wins or not plan.needs_gust:
            return
        key = (plan.attacker_serial, plan.target_serial, plan.attack_id)
        if key in _G["lethal_fail"]:
            plan2 = self.plan_attack(exclude=_G["lethal_fail"])
            _G["plan"] = plan2
            return
        ok = True
        saved_plan, saved_arch = _G["plan"], _G["arch"]
        try:
            IN_VERIFY = True
            ok = verify_win(self.obs, self.me)
        except Exception:
            ok = True  # trust the plan on verifier failure
        finally:
            IN_VERIFY = False
            _G["plan"], _G["arch"] = saved_plan, saved_arch
        if not ok:
            _G["lethal_fail"].add(key)
            _G["plan"] = self.plan_attack(exclude=_G["lethal_fail"])


# ------------------------------------------------------------ search world


def sample_own_hidden(obs, me):
    ps = obs.current.players[me]
    unseen = Counter(DECK)
    seen = Counter()
    for c in ps.hand or []:
        if c is not None:
            seen[c.id] += 1
    for c in ps.discard or []:
        if c is not None:
            seen[c.id] += 1
    for c in ps.prize or []:
        if c is not None:
            seen[c.id] += 1
    for zone in ((ps.active or []), (ps.bench or [])):
        for p in zone:
            if p is None:
                continue
            seen[p.id] += 1
            for c in p.energyCards or []:
                seen[c.id] += 1
            for c in p.tools or []:
                seen[c.id] += 1
            for c in p.preEvolution or []:
                if c is not None:
                    seen[c.id] += 1
    for c in obs.current.stadium or []:
        if c is not None and c.playerIndex == me:
            seen[c.id] += 1
    unseen.subtract(seen)
    pool = [cid for cid, cnt in unseen.items() for _ in range(max(cnt, 0))]
    random.shuffle(pool)
    n_deck = ps.deckCount
    n_prize = len(ps.prize or [])
    need = n_deck + n_prize
    if len(pool) < need:
        pool = pool + [FILLER_CARD] * (need - len(pool))
    return pool[:n_deck], pool[n_deck : n_deck + n_prize]


def _opp_energy_id(obs, me):
    """Most common basic-energy card the opponent has revealed, if any."""
    ps = obs.current.players[1 - me]
    cnt = Counter()
    for p in list(ps.active or []) + list(ps.bench or []):
        if p is None:
            continue
        for c in p.energyCards or []:
            d = CARD.get(c.id)
            if d is not None and d.cardType == CardType.BASIC_ENERGY:
                cnt[c.id] += 1
    for c in ps.discard or []:
        if c is not None:
            d = CARD.get(c.id)
            if d is not None and d.cardType == CardType.BASIC_ENERGY:
                cnt[c.id] += 1
    return cnt.most_common(1)[0][0] if cnt else None


def begin_world(obs, me, manual_coin=False, opp_energy=False):
    ps_opp = obs.current.players[1 - me]
    your_deck, your_prize = sample_own_hidden(obs, me)
    opp_active = []
    act = ps_opp.active or []
    if len(act) > 0 and act[0] is None:
        opp_active = [FILLER_BASIC]
    n_hand, n_deck = ps_opp.handCount, ps_opp.deckCount
    opp_hand = [FILLER_CARD] * n_hand
    opp_deck = [FILLER_BASIC] * n_deck
    if opp_energy:
        # model the opponent's manual attach each turn: without it their
        # reply turns are toothless and search under-fears every line
        eid = _opp_energy_id(obs, me)
        if eid is not None:
            ne_h = min(2, n_hand)
            opp_hand = [eid] * ne_h + [FILLER_CARD] * (n_hand - ne_h)
            ne_d = n_deck // 3
            opp_deck = [eid] * ne_d + [FILLER_BASIC] * (n_deck - ne_d)
    return search_begin(
        obs,
        your_deck=your_deck,
        your_prize=your_prize,
        opponent_deck=opp_deck,
        opponent_prize=[FILLER_CARD] * len(ps_opp.prize or []),
        opponent_hand=opp_hand,
        opponent_active=opp_active,
        manual_coin=manual_coin,
    )


def _fallback_choice(sel):
    n = len(sel.option)
    lo = max(sel.minCount, 0)
    if n == 0:
        return []
    return list(range(min(max(lo, 1), n)))


def verify_win(obs, me, worlds=3, budget=1.2):
    """Determinized check: does executing our policy this turn win the game?"""
    t0 = time.time()
    wins = tried = 0
    start_turn = obs.current.turn
    try:
        for _ in range(worlds):
            if time.time() - t0 > budget:
                break
            try:
                st = begin_world(obs, me)
            except Exception:
                break
            tried += 1
            ok = False
            for _step in range(60):
                cur = st.observation.current
                if cur.result >= 0:
                    ok = cur.result == me
                    break
                if cur.turn != start_turn:
                    break
                try:
                    choice = Policy(st.observation).choose()
                except Exception:
                    choice = _fallback_choice(st.observation.select)
                try:
                    st = search_step(st.searchId, choice)
                except Exception:
                    break
            wins += 1 if ok else 0
    finally:
        try:
            search_end()
        except Exception:
            pass
    return tried == 0 or wins * 2 >= tried


# ------------------------------------------------------- pivotal search


def eval_world(cur, me, base_my_pr, base_op_pr):
    """Score a rollout endpoint from our perspective."""
    if cur.result >= 0:
        if VN_OK:
            if cur.result == me:
                return 1000.0
            if cur.result == 1 - me:
                return 0.0
            return 500.0
        if cur.result == me:
            return 1_000_000.0
        if cur.result == 1 - me:
            return -1_000_000.0
        return 0.0
    if VN_OK:
        hc = None
        try:
            hc = _vf.handcrafted_eval(cur, me)
            feats = _vf.extract(cur, me, CARD, ATK)
            feats.append(_vf.hc_squash(hc))
            p = _vn.predict(feats)
            if VALUE_BLEND < 1.0:
                q = 1.0 / (1.0 + math.exp(-hc / 15_000.0))
                p = VALUE_BLEND * p + (1.0 - VALUE_BLEND) * q
            return 1000.0 * p
        except Exception:
            # stay on the [0,1000] scale so paired worlds remain comparable
            if hc is not None:
                return 1000.0 / (1.0 + math.exp(-hc / 15_000.0))
            return 500.0
    my = cur.players[me]
    op = cur.players[1 - me]
    my_taken = base_my_pr - len(my.prize or [])
    op_taken = base_op_pr - len(op.prize or [])
    s = 10_000.0 * (my_taken - op_taken)
    # closing in on the last prizes matters more than parity alone
    s -= 1_500.0 * len(my.prize or [])
    my_board = [p for p in list(my.active or []) + list(my.bench or []) if p is not None]
    op_board = [p for p in list(op.active or []) + list(op.bench or []) if p is not None]
    s += sum(p.hp for p in my_board) * 1.0
    s -= sum(p.hp for p in op_board) * 1.0
    s += sum(len(p.energies or []) for p in my_board) * 25.0
    s -= sum(len(p.energies or []) for p in op_board) * 25.0
    s += len(my_board) * 40.0
    return s


def rollout_value(st, me, horizon, base_my_pr, base_op_pr, deadline, coin_budget):
    """Roll a world to `horizon` with the policy on both sides.

    First coin flip is expanded expectimax-style (both outcomes, 0.5 weight);
    later flips are sampled. Returns None if the deadline cut us off.
    """
    for _step in range(ROLLOUT_STEPS):
        cur = st.observation.current
        if cur.result >= 0 or cur.turn >= horizon:
            break
        if time.time() > deadline:
            return None
        sel = st.observation.select
        if sel is None or not sel.option:
            return None
        if sel.context == SelectContext.COIN_HEAD:
            yes = next((i for i, o in enumerate(sel.option) if o.type == OptionType.YES), 0)
            no = next((i for i, o in enumerate(sel.option) if o.type == OptionType.NO), yes)
            if coin_budget > 0 and yes != no:
                _S["coins"] += 1
                v1 = rollout_value(
                    search_step(st.searchId, [yes]),
                    me,
                    horizon,
                    base_my_pr,
                    base_op_pr,
                    deadline,
                    0,
                )
                if v1 is None:
                    return None
                v2 = rollout_value(
                    search_step(st.searchId, [no]), me, horizon, base_my_pr, base_op_pr, deadline, 0
                )
                if v2 is None:
                    return None
                return 0.5 * (v1 + v2)
            choice = [yes if random.random() < 0.5 else no]
        else:
            try:
                choice = Policy(st.observation).choose()
            except Exception:
                choice = _fallback_choice(sel)
        st = search_step(st.searchId, choice)
    return eval_world(st.observation.current, me, base_my_pr, base_op_pr)


def _beam_kind(st, me, start_turn):
    """Classify a beam node: 'live' (our turn continues, expand it),
    'endturn' (turn passed to opp), 'coin' (our coin flip -> hand to the
    expectimax rollout), 'result' (game over), 'dead' (no options)."""
    cur = st.observation.current
    if cur.result >= 0:
        return "result"
    if cur.turn != start_turn:
        return "endturn"
    sel = st.observation.select
    if sel is None or not sel.option or sel.maxCount == 0:
        return "dead"
    if sel.context == SelectContext.COIN_HEAD:
        return "coin"
    return "live"


def beam_turn_value(
    root_id, first_choice, me, start_turn, horizon, base_my_pr, base_op_pr, deadline
):
    """Beam-search OUR remaining turn after `first_choice`, then model the
    opponent's reply with the policy to `horizon`; return our best achievable
    leaf value (max over our action sequences, opponent fixed-policy).

    Beam: keep up to BEAM_WIDTH live states, expand each by the policy's
    BEAM_EXPAND_B best next-actions, rank the survivors by value-net eval of
    the mid-turn state. Lines that reach END/attack/coin are parked as
    end-of-turn terminals; the top BEAM_TERMK get an opponent rollout. Returns
    None only if nothing at all could be evaluated before the deadline.
    """
    try:
        st0 = search_step(root_id, first_choice)
    except Exception:
        return None
    kind = _beam_kind(st0, me, start_turn)
    live = [st0] if kind == "live" else []
    terminals = [] if kind == "live" else ([st0] if kind != "dead" else [])
    depth = 0
    while live and depth < BEAM_DEPTH:
        if time.time() > deadline:
            terminals.extend(live)
            live = []
            break
        children = []
        stop = False
        for st in live:
            if stop:
                break
            try:
                pol = Policy(st.observation)
                choices = pol.beam_children(BEAM_EXPAND_B)
            except Exception:
                choices = [_fallback_choice(st.observation.select)]
            for ch in choices:
                if time.time() > deadline:
                    stop = True
                    break
                try:
                    c = search_step(st.searchId, ch)
                except Exception:
                    continue
                children.append(c)
                _S["nodes"] += 1
        new_live = []
        for c in children:
            k = _beam_kind(c, me, start_turn)
            if k == "live":
                new_live.append(c)
            elif k != "dead":
                terminals.append(c)
        if new_live:
            new_live.sort(
                key=lambda s: -eval_world(s.observation.current, me, base_my_pr, base_op_pr)
            )
            live = new_live[:BEAM_WIDTH]
        else:
            live = []
        depth += 1
    terminals.extend(live)
    if not terminals:
        return None
    _S["depth_sum"] += depth
    _S["lines"] += 1
    scored = [(eval_world(t.observation.current, me, base_my_pr, base_op_pr), t) for t in terminals]
    scored.sort(key=lambda x: -x[0])
    best = None
    for se, t in scored[:BEAM_TERMK]:
        cur = t.observation.current
        if cur.result >= 0 or cur.turn >= horizon or time.time() > deadline:
            v = se
        else:
            v = rollout_value(t, me, horizon, base_my_pr, base_op_pr, deadline, 1)
            if v is None:
                v = se
            else:
                _S["opp_rollouts"] += 1
        if best is None or v > best:
            best = v
    return best


def det_search(pol, cands):
    """Determinized paired-world search over candidate first-selections.

    Per world, each candidate first-move gets a within-turn beam (BEAM_ON) that
    optimizes our whole action sequence to end-of-turn, then the opponent reply
    is modeled to `horizon`; legacy mode (BEAM_ON off) does the single-move +
    policy rollout. Worlds are paired: a candidate that can't be evaluated
    discards the whole world so challenger/incumbent stay comparable.
    """
    obs, me = pol.obs, pol.me
    t0 = time.time()
    deadline = min(t0 + SEARCH_BUDGET, _G["move_t0"] + MOVE_DEADLINE - 0.1)
    if deadline - t0 < 0.05:
        return None
    base_my_pr = len(pol.my.prize or [])
    base_op_pr = len(pol.op.prize or [])
    start_turn = obs.current.turn
    horizon = start_turn + SEARCH_HORIZON
    totals = [0.0] * len(cands)
    n_worlds = 0
    _S["searches"] += 1
    _S["cands"] += len(cands)
    try:
        for _w in range(SEARCH_MAX_WORLDS):
            if time.time() > deadline:
                break
            try:
                root = begin_world(obs, me, manual_coin=True, opp_energy=True)
            except Exception:
                break
            vals = []
            for ci in cands:
                try:
                    if BEAM_ON:
                        v = beam_turn_value(
                            root.searchId,
                            [ci],
                            me,
                            start_turn,
                            horizon,
                            base_my_pr,
                            base_op_pr,
                            deadline,
                        )
                    else:
                        st = search_step(root.searchId, [ci])
                        v = rollout_value(st, me, horizon, base_my_pr, base_op_pr, deadline, 1)
                except Exception:
                    v = None
                if v is None:
                    break
                vals.append(v)
            if len(vals) < len(cands):
                break  # unpaired world: discard partial values
            for i, v in enumerate(vals):
                totals[i] += v
            n_worlds += 1
            _S["worlds"] += 1
    finally:
        try:
            search_end()
        except Exception:
            pass
    dt = time.time() - t0
    _S["t"] += dt
    _G["search_t"] += dt
    if n_worlds == 0:
        _S["fallbacks"] += 1
        return None
    # cands[0] is the policy's own pick: challengers must beat it clearly,
    # otherwise near-tie noise (sampled coins) would override config wisdom
    margin = VALUE_MARGIN if VN_OK else SEARCH_MARGIN
    if VN_OK:
        margin = pol.cfg.get("value_margin", margin)
    best = max(range(len(cands)), key=lambda i: totals[i])
    if totals[best] - totals[0] <= margin * n_worlds:
        return cands[0]
    return cands[best]


def run_search(pol, cands):
    """Gate + wrap det_search: budgets, matchup gates, state save/restore."""
    global IN_VERIFY
    if IN_VERIFY or not SEARCH_OK or pol.obs.search_begin_input is None:
        return None
    gate = pol.cfg.get("search_matchup", {})
    if not gate.get(pol.arch, gate.get("default", True)):
        return None
    ov = _G["overage"]
    if ov is not None and ov < SEARCH_OVERAGE_MIN:
        return None
    if _G["search_t"] >= SEARCH_GAME_CAP or _S["searches"] >= 110:
        return None
    if time.time() - _G["move_t0"] > MOVE_DEADLINE - 0.4:
        return None
    saved_plan, saved_arch = _G["plan"], _G["arch"]
    IN_VERIFY = True
    try:
        return det_search(pol, cands)
    except Exception:
        _S["fallbacks"] += 1
        return None
    finally:
        IN_VERIFY = False
        _G["plan"], _G["arch"] = saved_plan, saved_arch


# ------------------------------------------------------------ entry point


def _game_reset():
    if DEBUG and _S["moves"]:
        n = max(_S["searches"], 1)
        lines = max(_S["lines"], 1)
        print(
            f"[a2] searches={_S['searches']} t={_S['t']:.2f}s "
            f"({_S['t'] / n:.3f}s/ea) worlds={_S['worlds']} "
            f"cands={_S['cands']} nodes={_S['nodes']} "
            f"({_S['nodes'] / n:.0f}/search) lines={_S['lines']} "
            f"mean_depth={_S['depth_sum'] / lines:.1f} "
            f"opp_rollouts={_S['opp_rollouts']} coins={_S['coins']} "
            f"overrides={_S['overrides']} fallbacks={_S['fallbacks']} "
            f"moves={_S['moves']} max_move={_S['max_move']:.2f}s "
            f"{dict(_S['by_class'])}",
            file=sys.stderr,
            flush=True,
        )
    _S.update(_fresh_stats())
    _G["search_t"] = 0.0
    _G["lethal_fail"] = set()
    _G["arch"] = "unknown"
    _G["plan"] = Plan()


def agent(obs_dict: dict) -> list[int]:
    _G["move_t0"] = time.time()
    try:
        return _agent_inner(obs_dict)
    finally:
        dt = time.time() - _G["move_t0"]
        _S["moves"] += 1
        if dt > _S["max_move"]:
            _S["max_move"] = dt


def _agent_inner(obs_dict: dict) -> list[int]:
    try:
        if obs_dict.get("select") is None:
            _G["turn"] = -99
            _game_reset()
            return list(DECK)
        obs: Observation = to_observation_class(obs_dict)
        turn = obs.current.turn
        if turn < _G["turn"]:
            _game_reset()
        elif turn > _G["turn"]:
            _G["lethal_fail"] = set()
        _G["turn"] = turn
        overage = obs_dict.get("remainingOverageTime")
        _G["overage"] = overage if isinstance(overage, (int, float)) else None
        pol = Policy(obs)
        _G["arch"] = pol.arch
        if isinstance(overage, (int, float)) and overage < PANIC_RESERVE:
            # panic: pure policy, no search
            global IN_VERIFY
            IN_VERIFY = True
            try:
                return pol.choose()
            finally:
                IN_VERIFY = False
        return pol.choose()
    except Exception:
        try:
            obs = to_observation_class(obs_dict)
            return _fallback_choice(obs.select)
        except Exception:
            sel = obs_dict.get("select") or {}
            n = len(sel.get("option", []) or [])
            lo = max(sel.get("minCount", 1), 0)
            return list(range(min(max(lo, 1), n)))
