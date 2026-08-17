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
import os
import random
import sys
import time
from collections import Counter

# kaggle_environments exec()s this file, so __file__ may not exist
AGENT_DIR = "/kaggle_simulations/agent"
if not os.path.isdir(os.path.join(AGENT_DIR, "cg")):
    try:
        AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        AGENT_DIR = os.getcwd()
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

from cg.api import (
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

# ablation switches for local tuning: CG_A2_OFF="d1,d3" disables components
A2_OFF = set(filter(None, os.environ.get("CG_A2_OFF", "").split(",")))

CARD = {}
ATK = {}
try:
    CARD = {c.cardId: c for c in all_card_data()}
    ATK = {a.attackId: a for a in all_attack()}
except Exception:
    pass

# attacks that lock themselves (or all attacks) for the user's next turn,
# e.g. Mega Brave: "During your next turn, this Pokemon can't use ..."
SELF_LOCK = set()
try:
    for _a in ATK.values():
        t = (_a.text or "").lower()
        if "during your next turn, this pok" in t and "use" in t:
            SELF_LOCK.add(_a.attackId)
except Exception:
    SELF_LOCK = set()

# card id -> ids it can evolve into (for threat projection: a charging Riolu
# is really a charging Mega Lucario)
EVO_NEXT = {}
try:
    _by_name = {}
    for _c in CARD.values():
        _by_name.setdefault(_c.name, []).append(_c.cardId)
    for _c in CARD.values():
        if _c.evolvesFrom:
            for _b in _by_name.get(_c.evolvesFrom, []):
                EVO_NEXT.setdefault(_b, []).append(_c.cardId)
except Exception:
    EVO_NEXT = {}


def threat_ids(cid, depth=2):
    """cid plus everything it can evolve into within `depth` steps."""
    out = [cid]
    frontier = [cid]
    for _ in range(depth):
        nxt = []
        for c in frontier:
            for e in EVO_NEXT.get(c, ()):
                if e not in out:
                    out.append(e)
                    nxt.append(e)
        frontier = nxt
    return out

# ------------------------------------------------------------ deck config

def _load_configs():
    ns = {}
    path = os.path.join(AGENT_DIR, "deck_configs.py")
    with open(path, "r", encoding="utf-8") as f:
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
FILLER_BASIC = 1072   # Snorlax: safe opponent-active filler for search worlds
FILLER_CARD = 1
PIERCE_ATTACKS = {1488}          # Nebula Beam: ignores W/R and defender effects
EX_WALL = {345}                  # Crustle: no damage from Pokemon ex / Mega ex
# basics that evolve into a big attacker: KOing them early starves the opponent
GUST_WORTH = {677, 741, 1030, 169, 722, 268, 646, 860, 878, 344}
# Boss's Orders copies each known archetype plays (reply/gust model)
# (measured: lucario truly runs 3 but modeling 3 = 36.8% and 1 = 38.0%
# vs the default 2's 39.8% — leave lucario on the default budget)
OP_BOSS_TOTAL = {"starmie": 1, "grimmsnarl": 2, "archaludon": 4}
PANIC_RESERVE = 60.0
LOW_DECK = 6

ENERGY_TYPES = (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)

# archetype signatures: flagship first, then early-game cards
_ARCH_FLAG = [(678, "lucario"), (648, "grimmsnarl"), (743, "alakazam"),
              (1031, "starmie"), (190, "archaludon"), (345, "crustle_wall"),
              (879, "trevenant"), (861, "froslass"), (756, "kangaskhan"),
              (723, "abomasnow"), (269, "bellibolt")]
_ARCH_EARLY = [(677, "lucario"), (646, "grimmsnarl"), (741, "alakazam"),
               (1030, "starmie"), (169, "archaludon"), (344, "crustle_wall"),
               (878, "trevenant"), (722, "abomasnow"), (268, "bellibolt"),
               (270, "bellibolt"), (271, "bellibolt")]


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


def units_missing(attached, cost):
    """Manual-attach distance to paying cost: 0 = ready now, 1 = one attach
    away, else an estimate >= 2. Cheap, pure arithmetic."""
    if satisfies(attached, cost):
        return 0
    for t in (set(cost) or {0}):
        # t == 0: any energy pays a colorless slot; 12 is a nonexistent type
        # that satisfies() can only spend on colorless.
        if satisfies(attached + [t if t else 12], cost):
            return 1
    return max(len(cost) - len(attached), 2)


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
    __slots__ = ("attacker_bi", "target_bi", "attack_id", "dmg", "ko", "wins",
                 "needs_energy", "needs_gust", "needs_switch", "score",
                 "attacker_serial", "target_serial", "retreat_fuel")

    def __init__(self):
        self.attacker_bi = -1
        self.target_bi = -1
        self.attack_id = -1
        self.dmg = 0
        self.ko = False
        self.wins = False
        self.needs_energy = 0      # card id of energy to attach first, or 0
        self.needs_gust = False
        self.needs_switch = False
        self.score = -1.0
        self.attacker_serial = -1
        self.target_serial = -1
        self.retreat_fuel = 0      # energy card id to attach to ACTIVE so it can retreat


# persistent per-game state
_G = {"turn": -99, "plan": Plan(), "lethal_fail": set(), "arch": "unknown",
      "op_lock": {}}
IN_VERIFY = False


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
        mo = self.cfg.get("matchup", {}).get(self.arch)
        if mo:
            self.cfg = {**self.cfg, **mo}
        # opponent reply model: assume a gust is available while their
        # archetype's known Boss's Orders count exceeds what their discard
        # shows (lucario runs 3, starmie 1, grimm 2, archaludon 4)
        boss_total = OP_BOSS_TOTAL.get(self.arch, 2)
        self.op_gust = sum(1 for c in (self.op.discard or [])
                           if c is not None and c.id == BOSS) < boss_total
        self._threats = None          # [(serial, card_id, Attack)]
        self._reply_cache = {}
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
            for c in (p.preEvolution or []):
                if c is not None:
                    seen.add(c.id)
        for c in (self.op.discard or []):
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
        for c in (self.my.hand or []):
            if c is not None:
                seen[c.id] += 1
        for c in (self.my.discard or []):
            if c is not None:
                seen[c.id] += 1
        for c in (self.my.prize or []):
            if c is not None:
                seen[c.id] += 1
        for p in self.my_board:
            if p is None:
                continue
            seen[p.id] += 1
            for c in (p.energyCards or []):
                seen[c.id] += 1
            for c in (p.tools or []):
                seen[c.id] += 1
            for c in (p.preEvolution or []):
                if c is not None:
                    seen[c.id] += 1
        for c in (self.state.stadium or []):
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

    # ------------------------------------------------------- danger model

    def reply_threats(self):
        """Opponent attacks plausibly usable next turn (any of their board
        can be next-turn active; they get one manual attach)."""
        if self._threats is None:
            th = []
            for q in self.op_board:
                if q is None:
                    continue
                attached = [int(e) for e in (q.energies or [])]
                qd = CARD.get(q.id)
                if qd is None:
                    continue
                lock = _G["op_lock"]
                for aid in (qd.attacks or []):
                    a = ATK.get(aid)
                    if a is None or not (a.damage or 0):
                        continue
                    # e.g. Mega Brave just used: locked for their next turn
                    if lock.get(q.serial) == (aid, self.state.turn):
                        continue
                    if units_missing(attached, a.energies) <= 1:
                        th.append((q.serial, q.id, a))
            self._threats = th
        return self._threats

    def reply_dmg_vs(self, def_id, exclude_serial=-1):
        """Best opponent reply damage vs one of our Pokemon with card id
        def_id, optionally pretending the Pokemon with exclude_serial is
        already KO'd (e.g. by our own attack this turn)."""
        key = (def_id, exclude_serial)
        v = self._reply_cache.get(key)
        if v is None:
            v = 0
            for serial, qid, a in self.reply_threats():
                if serial == exclude_serial:
                    continue
                d = eff_damage(qid, def_id, a)
                if d > v:
                    v = d
            self._reply_cache[key] = v
        return v

    def dies_to_reply(self, p, on_bench=False, exclude_serial=-1):
        """Does this Pokemon of ours die to the opponent's best reply?
        Bench Pokemon are only reachable if they have a gust left."""
        if p is None:
            return False
        if on_bench and not self.op_gust:
            return False
        return self.reply_dmg_vs(p.id, exclude_serial) >= p.hp

    # ------------------------------------------------------- attack planner

    def plan_attack(self, exclude=()):
        plan = Plan()
        boss_ok = self.hand_counts.get(BOSS, 0) > 0 and self.supporter_ok
        # can we unlock a retreat by attaching one energy to the active?
        retreat_fuel = 0
        active = self.my_board[0] if self.my_board else None
        if (not self.can_retreat and active is not None
                and not self.state.energyAttached and not self.state.retreated):
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
            for aid in (pd.attacks or []):
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
                    if ti > 0 and not (ko and (wins or py >= 2
                                               or len(q.energies or []) >= 3
                                               or q.id in GUST_WORTH)):
                        continue  # Boss is precious: gust only for KOs that matter
                    if wins:
                        score = 10_000_000.0
                    elif ko:
                        score = 100_000.0 * py + dmg
                    else:
                        score = 40_000.0 * min(dmg / max(q.hp, 1), 1.0) + bonus * 60
                    if not wins:
                        # danger: the attacker ends the turn active, facing
                        # their best reply (minus anything we just KO'd)
                        if "d1" not in A2_OFF:
                            rd = self.reply_dmg_vs(p.id, q.serial if ko else -1)
                            if rd >= p.hp:
                                score -= (500.0 if ko else 900.0) * prize_yield(p.id)
                        if ("d1b" not in A2_OFF and not ko and dmg > 0
                                and q.hp - dmg <= dmg):
                            score += 1500.0  # sets up our own KO next turn
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
        base = self.cfg["fetch_prio"].get(
            cid, 25 if d.cardType == CardType.POKEMON else 15)
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
                        can_jump = (self.hand_counts.get(1079, 0) > 0
                                    and self.field_counts.get(base_id, 0))
                        if self.field_counts.get(mid_id, 0) or can_jump:
                            s += 70
                        elif not self.field_counts.get(base_id, 0):
                            s -= 25
                else:
                    # a middle stage: bottleneck when base in play + top in hand
                    for top, pr in self.cfg["evo_top"].items():
                        if len(pr) > 1 and pr[1] == cid:
                            if (self.field_counts.get(pr[0], 0)
                                    and self.hand_counts.get(top, 0)
                                    and not self.hand_counts.get(1079, 0)):
                                s += 65
        elif d.cardType in ENERGY_TYPES:
            plan = _G["plan"]
            if plan.needs_energy == cid:
                s += 50
            n_energy_hand = sum(c for i, c in self.hand_counts.items()
                                if CARD.get(i) is not None
                                and CARD[i].cardType in ENERGY_TYPES)
            s -= 12 * n_energy_hand
        elif d.cardType == CardType.SUPPORTER:
            has_sup = any(CARD.get(i) is not None
                          and CARD[i].cardType == CardType.SUPPORTER
                          for i in self.hand_counts)
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
        card = get_card(self.obs, o.area, o.index, o.playerIndex
                        if o.playerIndex is not None else self.me)
        cid = getattr(card, "id", 0)
        if cid == 1259:  # Spikemuth Gym: fetch a Marnie's pokemon
            best = max((self.need_score(i) for i in (646, 647, 648)
                        if self.copies_hidden(i) > 0), default=-1)
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
            if ("d2" not in A2_OFF and prize_yield(cid) >= 2 and self.op_gust
                    and self.reply_dmg_vs(cid) >= d.hp):
                return 460  # multi-prize gust bait: bench only if nothing else
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
            want = any(CARD.get(i) is not None and CARD[i].cardType == CardType.POKEMON
                       and CARD[i].basic and CARD[i].hp <= 70
                       and self.line_cap_ok(i) and self.copies_hidden(i) > 0
                       for i in DECK_TOTAL)
            return 12000 if want else -1
        if cid == 1079:  # Rare Candy (engine only offers when legal)
            return 11000
        if cid == 1080:  # Unfair Stamp
            return 8000
        if cid == 1121:  # Ultra Ball
            best = max((self.need_score(i) for i in DECK_TOTAL
                        if CARD.get(i) is not None
                        and CARD[i].cardType == CardType.POKEMON
                        and self.copies_hidden(i) > 0), default=-1)
            return 9000 if best > 40 else 400
        if cid == 1122:  # Pokegear
            has_sup = any(CARD.get(i) is not None
                          and CARD[i].cardType == CardType.SUPPORTER
                          for i in self.hand_counts)
            return 8200 if not has_sup else 300
        if cid == 1120:  # Crushing Hammer
            if any(p is not None and (p.energyCards or []) for p in self.op_board):
                return 4000
            return -1
        if cid == 1097:  # Night Stretcher
            best = max((self.need_score(c.id) for c in (self.my.discard or [])
                        if c is not None and CARD.get(c.id) is not None
                        and (CARD[c.id].cardType == CardType.POKEMON
                             or CARD[c.id].cardType == CardType.BASIC_ENERGY)),
                       default=-1)
            return 500 + best * 20 if best > 30 else -1
        if cid == 1145:  # Mega Signal
            for top in self.cfg["evo_top"]:
                td = CARD.get(top)
                if td is not None and td.megaEx and self.hand_counts.get(top, 0) == 0 \
                        and self.copies_hidden(top) > 0:
                    return 10000
            return -1
        if cid == 1152:  # Poke Pad
            best = max((self.need_score(i) for i in DECK_TOTAL
                        if CARD.get(i) is not None
                        and CARD[i].cardType == CardType.POKEMON
                        and not CARD[i].ex and not CARD[i].megaEx
                        and self.copies_hidden(i) > 0), default=-1)
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
            # (measured: gating this tier on Ignition-in-hand = 35.0%/47.8%
            # vs 39.8%/52.9% — the loose heal is strongly positive; keep it)
            if best >= 120 and fuel:
                return 12500
            return 9000 if best >= 240 else -1
        if cid == 1231:  # Dawn: fetch basic + stage1 + stage2
            missing = sum(1 for i in (646, 647, 648)
                          if self.hand_counts.get(i, 0) == 0 and self.copies_hidden(i) > 0)
            return 8500 if missing >= 2 else 7200
        if cid == 1225:  # Hilda: evolution pokemon + energy
            for top in self.cfg["evo_top"]:
                if self.hand_counts.get(top, 0) == 0 and self.copies_hidden(top) > 0:
                    return 8300
            return 5800
        if cid == 1219:  # Petrel: fetch any trainer
            best = max((self.need_score(i) for i in DECK_TOTAL
                        if CARD.get(i) is not None
                        and CARD[i].cardType in (CardType.ITEM, CardType.TOOL,
                                                 CardType.SUPPORTER, CardType.STADIUM)
                        and self.copies_hidden(i) > 0), default=0)
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
                better = any(p is not None and p.id in route and not (p.tools or [])
                             for p in self.my_board)
                return -1 if better else 900
            # hold the tool if a strictly better holder is available or coming
            for i in (i for i, x in route.items() if x > v):
                if any(p is not None and p.id == i and not (p.tools or [])
                       for p in self.my_board):
                    return -1  # better holder in play with a free slot: use it
                if not self.field_counts.get(i, 0) and (
                        self.hand_counts.get(i, 0) or self.copies_hidden(i)):
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
                    s = 6500 - pos * 100
                    # over-investment: this attach makes >= 2 energy on a body
                    # that dies to their reply for free; prefer another carrier
                    if ("d3" not in A2_OFF and len(target.energies or []) >= 1
                            and self.dies_to_reply(target, on_bench=bi > 0)):
                        for pos2, (pid2, cap2) in enumerate(self.cfg["energy_route"]):
                            if pid2 == pid:
                                continue
                            for bj, alt in enumerate(self.my_board):
                                if (alt is not None and alt.id == pid2
                                        and len(alt.energies or []) < cap2
                                        and not self.dies_to_reply(alt, on_bench=bj > 0)):
                                    return 2800 - pos * 100
                    return s
                return 150 - pos
        return 100

    def sc_evolve(self, o):
        card = get_card(self.obs, o.area, o.index, self.me)
        target = get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.me)
        cid = getattr(card, "id", 0)
        if cid == 104 and self.opp_ability_count() == 0:
            return 550  # Froslass shroud would only chip our own board
        if (cid in self.cfg.get("jit_evolve", ())
                and o.inPlayArea == AreaType.BENCH):
            # a benched Mega is a 3-prize gust target with no upside until
            # it attacks: keep the 1-prize basic and evolve when promoted.
            # Exception: once the base carries enough energy the Mega can
            # fight back (or retreat) when gusted, while the tiny basic is
            # itself prime gust bait ("bench_evolve_energy" per matchup).
            bee = self.cfg.get("bench_evolve_energy")
            tgt = get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.me)
            if (bee is None or not isinstance(tgt, Pokemon)
                    or len(tgt.energies or []) < bee):
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
        dmg = eff_damage(me_active.id, q.id, a) if (q is not None and me_active is not None) else (a.damage or 0)
        bonus = self.cfg.get("attack_bonus", {}).get(o.attackId, 0)
        if dmg <= 0 and bonus <= 0:
            return 5
        return 200 + dmg * 0.5 + bonus

    # ------------------------------------------------------- card contexts

    def sc_card(self, o):
        card = get_card(self.obs, o.area, o.index,
                        o.playerIndex if o.playerIndex is not None else self.me)
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
            s = 100 + self.cfg["bench_value"].get(cid, 5)
            # gust bait: a multi-prize body that dies to their reply on the
            # bench is a liability
            d = CARD.get(cid)
            if ("d2" not in A2_OFF and d is not None and prize_yield(cid) >= 2
                    and self.op_gust and self.reply_dmg_vs(cid) >= d.hp):
                s -= 90
            return s
        if ctx == SelectContext.TO_HAND:
            return self.need_score(getattr(card, "id", 0))
        if ctx in (SelectContext.DISCARD, SelectContext.TO_DECK,
                   SelectContext.TO_DECK_BOTTOM, SelectContext.TO_PRIZE):
            return 200 - self.need_score(getattr(card, "id", 0))
        if ctx == SelectContext.NOT_MOVE:
            return self.need_score(getattr(card, "id", 0))
        if ctx in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY,
                   SelectContext.DAMAGE):
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
            for aid in (d.attacks or []):
                a = ATK.get(aid)
                if a is not None and satisfies([int(e) for e in (card.energies or [])], a.energies):
                    s += 60 + (a.damage or 0) * 0.2
                    break
            s -= d.retreatCost * 4
        if card.serial == plan.attacker_serial:
            s += 1000
        s += card.hp * 0.15  # sturdier promotes bleed fewer prizes
        # danger: don't hand the opponent free prizes. If this promote dies
        # to their best reply, penalize by what it bleeds; when everything
        # dies, this makes us sacrifice the cheapest body instead of a Mega.
        if "d4" not in A2_OFF and self.reply_dmg_vs(card.id) >= card.hp:
            s -= 200.0 * prize_yield(card.id)
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
            # chip what we can finish: an energy-laden small body (their
            # charging Staryu / Riolu) beats sprinkling 50s on a 330hp Mega
            if dmg and card.hp <= 2 * dmg:
                s += 150 + len(card.energies or []) * 25
        return s

    # ------------------------------------------------------- other types

    def sc_attached(self, o):
        """TOOL_CARD / ENERGY_CARD options (e.g. Crushing Hammer target)."""
        holder = get_card(self.obs, o.area, o.index, o.playerIndex)
        if o.playerIndex is not None and o.playerIndex != self.me:
            s = 100.0
            if isinstance(holder, Pokemon):
                if o.energyIndex is None or "d5" in A2_OFF:
                    # tool removal: fall back to generic value
                    s += (50 if o.area == AreaType.ACTIVE else 0)
                    s += len(holder.energies or []) * 8 + prize_yield(holder.id) * 10
                    return s
                # energy removal (Crushing Hammer & co): strip the Pokemon
                # closest to powering its scariest attack, counting what it
                # evolves into (a charging Riolu is a charging Mega Lucario)
                attached = [int(e) for e in (holder.energies or [])]
                after = list(attached)
                try:
                    ec = holder.energyCards[o.energyIndex]
                    ed = CARD.get(ec.id)
                    u = int(ed.energyType) if (ed is not None
                            and ed.cardType == CardType.BASIC_ENERGY) else -1
                    if u in after:
                        after.remove(u)
                    elif after:
                        after.pop()
                except Exception:
                    after = attached[:-1]
                t_before = t_after = 0.0
                for cid2 in threat_ids(holder.id):
                    d2 = CARD.get(cid2)
                    if d2 is None:
                        continue
                    for aid in (d2.attacks or []):
                        a = ATK.get(aid)
                        if a is None or not (a.damage or 0):
                            continue
                        t_before = max(t_before, a.damage / (1.0 + units_missing(attached, a.energies)))
                        t_after = max(t_after, a.damage / (1.0 + units_missing(after, a.energies)))
                s += (t_before - t_after) * 6.0 + t_before * 0.3
                s += (30 if o.area == AreaType.ACTIVE else 0)
                s += len(attached) * 2 + prize_yield(holder.id) * 5
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
            # (measured: choosing second = 47.2%/38.2% vs 49.5%/37.7-39%
            # choosing first — keep first)
            return 100
        if ctx == SelectContext.MULLIGAN:
            has_basic = any(CARD.get(c.id) is not None and CARD[c.id].cardType == CardType.POKEMON
                            and CARD[c.id].basic
                            for c in (self.my.hand or []) if c is not None)
            return -10 if has_basic else 100
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

    COST_CONTEXTS = {SelectContext.DISCARD, SelectContext.TO_DECK,
                     SelectContext.TO_DECK_BOTTOM, SelectContext.TO_PRIZE,
                     SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
                     SelectContext.DISCARD_ENERGY, SelectContext.DISCARD_ENERGY_CARD,
                     SelectContext.DISCARD_TOOL_CARD}

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
            o.playerIndex in (None, self.me) for o in sel.option)
        if cost_ctx:
            chosen = order[:lo]
        else:
            chosen = [i for i in order if scores[i] > 0][:hi]
            if len(chosen) < lo:
                extra = [i for i in order if i not in chosen]
                chosen += extra[:lo - len(chosen)]
        if not chosen and lo > 0:
            chosen = order[:lo]
        return chosen

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
        try:
            IN_VERIFY = True
            ok = verify_win(self.obs, self.me)
        except Exception:
            ok = True  # trust the plan on verifier failure
        finally:
            IN_VERIFY = False
        if not ok:
            _G["lethal_fail"].add(key)
            _G["plan"] = self.plan_attack(exclude=_G["lethal_fail"])


# ------------------------------------------------------------ search world

def sample_own_hidden(obs, me):
    ps = obs.current.players[me]
    unseen = Counter(DECK)
    seen = Counter()
    for c in (ps.hand or []):
        if c is not None:
            seen[c.id] += 1
    for c in (ps.discard or []):
        if c is not None:
            seen[c.id] += 1
    for c in (ps.prize or []):
        if c is not None:
            seen[c.id] += 1
    for zone in ((ps.active or []), (ps.bench or [])):
        for p in zone:
            if p is None:
                continue
            seen[p.id] += 1
            for c in (p.energyCards or []):
                seen[c.id] += 1
            for c in (p.tools or []):
                seen[c.id] += 1
            for c in (p.preEvolution or []):
                if c is not None:
                    seen[c.id] += 1
    for c in (obs.current.stadium or []):
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
    return pool[:n_deck], pool[n_deck:n_deck + n_prize]


def begin_world(obs, me):
    ps_opp = obs.current.players[1 - me]
    your_deck, your_prize = sample_own_hidden(obs, me)
    opp_active = []
    act = ps_opp.active or []
    if len(act) > 0 and act[0] is None:
        opp_active = [FILLER_BASIC]
    return search_begin(
        obs,
        your_deck=your_deck,
        your_prize=your_prize,
        opponent_deck=[FILLER_BASIC] * ps_opp.deckCount,
        opponent_prize=[FILLER_CARD] * len(ps_opp.prize or []),
        opponent_hand=[FILLER_CARD] * ps_opp.handCount,
        opponent_active=opp_active,
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


# ------------------------------------------------------------ entry point

def agent(obs_dict: dict) -> list[int]:
    try:
        if obs_dict.get("select") is None:
            _G["turn"] = -99
            _G["lethal_fail"] = set()
            _G["arch"] = "unknown"
            _G["plan"] = Plan()
            return list(DECK)
        obs: Observation = to_observation_class(obs_dict)
        turn = obs.current.turn
        if turn < _G["turn"]:
            _G["lethal_fail"] = set()
            _G["arch"] = "unknown"
            _G["plan"] = Plan()
            _G["op_lock"] = {}
        elif turn > _G["turn"]:
            _G["lethal_fail"] = set()
        _G["turn"] = turn
        # watch the log stream for opponent attacks that lock themselves
        # out next turn (Mega Brave etc.): their reply is weaker that turn
        try:
            me = obs.current.yourIndex
            for lg in (obs_dict.get("logs") or []):
                if (lg.get("type") == 15 and lg.get("playerIndex") != me
                        and lg.get("attackId") in SELF_LOCK):
                    _G["op_lock"][lg.get("serial")] = (lg.get("attackId"), turn)
        except Exception:
            pass
        overage = obs_dict.get("remainingOverageTime")
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
