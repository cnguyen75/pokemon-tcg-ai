"""Deck-agnostic numeric features for the learned value net.

Shared between training (train/*.py, feeding on replay dicts) and inference
(agent2/main.py leaf eval, feeding on cg.api dataclasses). Everything here is
plain python + a CARD/ATK db injected by the caller, so the module imports
with no dependencies.

extract(cur, me, CARD, ATK) -> list[float]  (len == len(FEATURE_NAMES))
handcrafted_eval(cur, me)   -> float        (static version of eval_world,
                                             used as the AUC baseline)

`cur` may be an Observation.current dict (replay JSON) or a State dataclass.
`me` is the player index the features are computed for; the vector is the
symmetric my-side/opp-side view from that player's perspective.
"""

import math

_RAINBOW = 10
_TEAM_ROCKET = 11  # psychic(5) / darkness(7)

# ---------------------------------------------------------------------------
# Opponent-archetype signatures.  MUST stay in lock-step with agent2/main.py's
# _ARCH_FLAG / _ARCH_EARLY so the net trains and infers on the SAME observable
# runtime detector (flagship card first, then early-game shell cards).  Only
# revealed opponent cards are used (board + preEvolution chains + discard), so
# the signal exists identically at extraction and inference time.
_ARCH_FLAG = [(678, "lucario"), (648, "grimmsnarl"), (743, "alakazam"),
              (1031, "starmie"), (190, "archaludon"), (345, "crustle_wall"),
              (879, "trevenant"), (861, "froslass"), (756, "kangaskhan"),
              (723, "abomasnow"), (269, "bellibolt")]
_ARCH_EARLY = [(677, "lucario"), (646, "grimmsnarl"), (741, "alakazam"),
               (1030, "starmie"), (169, "archaludon"), (344, "crustle_wall"),
               (878, "trevenant"), (722, "abomasnow"), (268, "bellibolt"),
               (270, "bellibolt"), (271, "bellibolt"),
               (673, "lucario"), (674, "lucario"), (675, "lucario"),
               (676, "lucario")]
# one-hot categories = the ordered set of names the runtime detector emits
_ARCH_NAMES = ["lucario", "grimmsnarl", "alakazam", "starmie", "archaludon",
               "crustle_wall", "trevenant", "froslass", "kangaskhan",
               "abomasnow", "bellibolt"]
# observable opponent capability cards (revealed in discard / stadium)
_HEAL_IDS = frozenset({1147})               # Jumbo Ice Cream
_DISRUPT_IDS = frozenset({1182, 1213, 1244})  # Boss's Orders, Judge, Full Metal Lab
_FULL_METAL_LAB = 1244


def _g(o, k, d=None):
    if isinstance(o, dict):
        v = o.get(k, d)
    else:
        v = getattr(o, k, d)
    return d if v is None else v


def _satisfies(attached, cost):
    """Can `attached` energy units pay `cost` (0 = colorless)?"""
    pool = list(attached)
    n_colorless = 0
    for t in cost:
        if t == 0:
            n_colorless += 1
            continue
        if t in pool:
            pool.remove(t)
        elif _RAINBOW in pool:
            pool.remove(_RAINBOW)
        elif t in (5, 7) and _TEAM_ROCKET in pool:
            pool.remove(_TEAM_ROCKET)
        else:
            return False
    return len(pool) >= n_colorless


def _eff_damage(att_d, def_d, attack):
    dmg = _g(attack, "damage", 0) or 0
    if att_d is None or def_d is None:
        return dmg
    if def_d.weakness is not None and def_d.weakness == att_d.energyType:
        dmg *= 2
    elif def_d.resistance is not None and def_d.resistance == att_d.energyType:
        dmg = max(dmg - 30, 0)
    return dmg


def _pyield(d):
    if d is None:
        return 1
    if d.megaEx:
        return 3
    if d.ex:
        return 2
    return 1


def _stage(d):
    if d is None:
        return 0
    if d.stage2:
        return 2
    if d.stage1:
        return 1
    return 0


def _poke_iter(ps):
    """Board pokemon of a player state: (pokemon, is_active)."""
    out = []
    for p in (_g(ps, "active", []) or []):
        if p is not None:
            out.append((p, True))
    for p in (_g(ps, "bench", []) or []):
        if p is not None:
            out.append((p, False))
    return out


def _atk_ready_best(p, def_id, CARD, ATK):
    """(ready, best eff dmg vs def_id, min missing energy) for pokemon p."""
    pid = _g(p, "id", 0)
    pd = CARD.get(pid)
    if pd is None:
        return 0, 0, 3
    att = [int(e) for e in (_g(p, "energies", []) or [])]
    dd = CARD.get(def_id) if def_id else None
    ready, best, missing = 0, 0, 3
    for aid in (pd.attacks or []):
        a = ATK.get(aid)
        if a is None:
            continue
        gap = max(len(a.energies) - len(att), 0)
        if gap < missing:
            missing = gap
        if _satisfies(att, a.energies):
            ready = 1
            missing = 0
            dmg = _eff_damage(pd, dd, a) if dd is not None else (a.damage or 0)
            if dmg > best:
                best = dmg
    return ready, best, missing


_SIDE = [
    "prizes", "deck", "hand", "discard",
    "cond_poison", "cond_burn", "cond_sleep", "cond_para", "cond_conf",
    "act_exists", "act_hp", "act_maxhp", "act_hpfrac", "act_dmg_taken",
    "act_energy", "act_ready", "act_best_dmg", "act_can_ko", "act_yield",
    "act_retreat_deficit", "act_stage", "act_ex",
    "bench_n", "bench_energy", "bench_ready", "bench_hp_total",
    "bench_max_yield", "bench_min_hpfrac",
    "board_best_dmg", "board_can_ko", "board_energy", "board_evolved",
    # v2 additions (transfer-friendly across archetypes)
    "act_missing_energy", "act_dmg_frac", "act_two_turn_ko",
    "board_dmg_taken", "bench_free", "close_to_win",
]

# opponent archetype one-hot + observable capability signals (cycle-2 addition)
ARCH_FEATURE_NAMES = (["arch_" + n for n in _ARCH_NAMES]
                      + ["arch_opp_ex", "arch_opp_mega",
                         "arch_opp_heal", "arch_opp_disrupt"])
N_ARCH_FEATURES = len(ARCH_FEATURE_NAMES)

FEATURE_NAMES = (["my_" + n for n in _SIDE] + ["op_" + n for n in _SIDE]
                 + ["turn", "my_turn", "prize_diff",
                    "me_win_if_ko", "op_win_if_ko",
                    "my_deck_low", "op_deck_low",
                    "stadium_mine", "stadium_theirs"]
                 + ARCH_FEATURE_NAMES)

N_FEATURES = len(FEATURE_NAMES)

# the net input = FEATURE_NAMES + a squashed copy of the handcrafted eval
# (residual learning: the net starts from parity with the old leaf eval)
HC_SCALE = 25_000.0
NET_FEATURE_NAMES = FEATURE_NAMES + ["hc_tanh"]
N_NET_FEATURES = len(NET_FEATURE_NAMES)


def hc_squash(hc):
    return math.tanh(hc / HC_SCALE)


def _side_feats(ps, opp_active, CARD, ATK):
    """32 features for one side; opp_active = enemy active pokemon or None."""
    f = []
    prize = _g(ps, "prize", []) or []
    f.append(len(prize) / 6.0)
    f.append(min(_g(ps, "deckCount", 0), 40) / 40.0)
    f.append(min(_g(ps, "handCount", 0), 12) / 12.0)
    f.append(len(_g(ps, "discard", []) or []) / 60.0)
    for k in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
        f.append(1.0 if _g(ps, k, False) else 0.0)

    def_id = _g(opp_active, "id", 0) if opp_active is not None else 0
    def_hp = _g(opp_active, "hp", 0) if opp_active is not None else 0

    act_list = _g(ps, "active", []) or []
    act = act_list[0] if act_list else None
    if act is None:
        f.extend([0.0] * 13)  # act_exists .. act_ex
        act_ready, act_best, act_missing = 0, 0, 3
    else:
        d = CARD.get(_g(act, "id", 0))
        hp = _g(act, "hp", 0)
        maxhp = max(_g(act, "maxHp", 0), 1)
        n_en = len(_g(act, "energies", []) or [])
        act_ready, act_best, act_missing = _atk_ready_best(act, def_id, CARD, ATK)
        f.append(1.0)
        f.append(min(hp, 400) / 400.0)
        f.append(min(maxhp, 400) / 400.0)
        f.append(hp / maxhp)
        f.append(min(maxhp - hp, 400) / 400.0)
        f.append(min(n_en, 5) / 5.0)
        f.append(float(act_ready))
        f.append(min(act_best, 400) / 400.0)
        f.append(1.0 if (def_hp and act_best >= def_hp) else 0.0)
        f.append(_pyield(d) / 3.0)
        rc = d.retreatCost if d is not None else 1
        f.append(min(max(rc - n_en, 0), 4) / 4.0)
        f.append(_stage(d) / 2.0)
        f.append(1.0 if (d is not None and (d.ex or d.megaEx)) else 0.0)

    bench = [p for p in (_g(ps, "bench", []) or []) if p is not None]
    b_energy = sum(len(_g(p, "energies", []) or []) for p in bench)
    b_ready, b_best = 0, 0
    b_hp, b_maxy, b_minfrac = 0, 0, 1.0
    for p in bench:
        r, bd, _m = _atk_ready_best(p, def_id, CARD, ATK)
        b_ready += r
        if bd > b_best:
            b_best = bd
        hp = _g(p, "hp", 0)
        b_hp += hp
        d = CARD.get(_g(p, "id", 0))
        y = _pyield(d)
        if y > b_maxy:
            b_maxy = y
        frac = hp / max(_g(p, "maxHp", 0), 1)
        if frac < b_minfrac:
            b_minfrac = frac
    f.append(len(bench) / 5.0)
    f.append(min(b_energy, 10) / 10.0)
    f.append(min(b_ready, 3) / 3.0)
    f.append(min(b_hp, 1500) / 1500.0)
    f.append(b_maxy / 3.0)
    f.append(b_minfrac)

    board_best = max(act_best, b_best)
    n_evolved = sum(1 for p, _a in _poke_iter(ps)
                    if _stage(CARD.get(_g(p, "id", 0))) > 0)
    act_energy = len(_g(act, "energies", []) or []) if act is not None else 0
    f.append(min(board_best, 400) / 400.0)
    f.append(1.0 if (def_hp and board_best >= def_hp) else 0.0)
    f.append(min(act_energy + b_energy, 12) / 12.0)
    f.append(n_evolved / 5.0)
    # v2 additions
    f.append(min(act_missing, 3) / 3.0)
    f.append(min(act_best / def_hp, 2.0) / 2.0 if def_hp else 0.0)
    f.append(1.0 if (def_hp and 2 * act_best >= def_hp) else 0.0)
    dmg_taken = sum(max(_g(p, "maxHp", 0) - _g(p, "hp", 0), 0)
                    for p, _a in _poke_iter(ps))
    f.append(min(dmg_taken, 600) / 600.0)
    f.append(max(_g(ps, "benchMax", 5) - len(bench), 0) / 5.0)
    f.append(1.0 if len(prize) <= 2 else 0.0)
    return f


def _opp_seen_ids(op):
    """Revealed opponent card ids: board pokemon + preEvolution + discard."""
    seen = set()
    for p, _a in _poke_iter(op):
        seen.add(_g(p, "id", 0))
        for c in (_g(p, "preEvolution", []) or []):
            if c is not None:
                seen.add(_g(c, "id", 0))
    for c in (_g(op, "discard", []) or []):
        if c is not None:
            seen.add(_g(c, "id", 0))
    return seen


def detect_archetype(op):
    """Runtime opponent-archetype detector (observable cards only).

    Same flagship-then-early-shell logic as agent2/main.py Policy._archetype,
    minus the sticky cross-turn memory (each call sees one state).  Returns an
    _ARCH_NAMES entry, or "unknown" when nothing is revealed yet.
    """
    seen = _opp_seen_ids(op)
    for cid, name in _ARCH_FLAG:
        if cid in seen:
            return name
    for cid, name in _ARCH_EARLY:
        if cid in seen:
            return name
    return "unknown"


def _arch_block(op, cur, me, CARD):
    """Opponent (player 1-me) archetype one-hot + capability signals."""
    seen = _opp_seen_ids(op)
    arch = "unknown"
    for cid, name in _ARCH_FLAG:
        if cid in seen:
            arch = name
            break
    else:
        for cid, name in _ARCH_EARLY:
            if cid in seen:
                arch = name
                break
    f = [1.0 if arch == n else 0.0 for n in _ARCH_NAMES]

    ex_count, mega = 0, 0.0
    for p, _a in _poke_iter(op):
        d = CARD.get(_g(p, "id", 0))
        if d is None:
            continue
        if d.ex:
            ex_count += 1
        if d.megaEx:
            mega = 1.0
    dids = set()
    for c in (_g(op, "discard", []) or []):
        if c is not None:
            dids.add(_g(c, "id", 0))
    heal = 1.0 if (_HEAL_IDS & dids) else 0.0
    disrupt = 1.0 if (_DISRUPT_IDS & dids) else 0.0
    stad = _g(cur, "stadium", []) or []
    if stad and _g(stad[0], "id", 0) == _FULL_METAL_LAB \
            and _g(stad[0], "playerIndex", -1) == (1 - me):
        disrupt = 1.0
    f.append(min(ex_count, 4) / 4.0)
    f.append(mega)
    f.append(heal)
    f.append(disrupt)
    return f


def _active_of(ps):
    a = _g(ps, "active", []) or []
    return a[0] if a else None


def extract(cur, me, CARD, ATK):
    """Feature vector (len N_FEATURES) for player `me` given state `cur`."""
    players = _g(cur, "players")
    my, op = players[me], players[1 - me]
    my_act, op_act = _active_of(my), _active_of(op)
    f = _side_feats(my, op_act, CARD, ATK)
    f += _side_feats(op, my_act, CARD, ATK)
    f.append(min(_g(cur, "turn", 0), 30) / 30.0)
    f.append(1.0 if _g(cur, "yourIndex", 0) == me else 0.0)
    my_pr = len(_g(my, "prize", []) or [])
    op_pr = len(_g(op, "prize", []) or [])
    f.append((op_pr - my_pr) / 6.0)
    op_y = _pyield(CARD.get(_g(op_act, "id", 0))) if op_act is not None else 1
    my_y = _pyield(CARD.get(_g(my_act, "id", 0))) if my_act is not None else 1
    f.append(1.0 if my_pr <= op_y else 0.0)   # KOing their active wins me the game
    f.append(1.0 if op_pr <= my_y else 0.0)   # ... and vice versa
    f.append(1.0 if _g(my, "deckCount", 60) <= 3 else 0.0)
    f.append(1.0 if _g(op, "deckCount", 60) <= 3 else 0.0)
    stad = _g(cur, "stadium", []) or []
    s_owner = _g(stad[0], "playerIndex", -1) if stad else -1
    f.append(1.0 if s_owner == me else 0.0)
    f.append(1.0 if s_owner == 1 - me else 0.0)
    f += _arch_block(op, cur, me, CARD)
    return f


def extract_net(cur, me, CARD, ATK):
    """Full net input: base features + squashed handcrafted eval."""
    f = extract(cur, me, CARD, ATK)
    f.append(hc_squash(handcrafted_eval(cur, me)))
    return f


def handcrafted_eval(cur, me):
    """Static replica of main.py eval_world (base prizes = 6): AUC baseline."""
    result = _g(cur, "result", -1)
    if result >= 0:
        if result == me:
            return 1_000_000.0
        if result == 1 - me:
            return -1_000_000.0
        return 0.0
    players = _g(cur, "players")
    my, op = players[me], players[1 - me]
    my_taken = 6 - len(_g(my, "prize", []) or [])
    op_taken = 6 - len(_g(op, "prize", []) or [])
    s = 10_000.0 * (my_taken - op_taken)
    s -= 1_500.0 * len(_g(my, "prize", []) or [])
    my_board = [p for p, _a in _poke_iter(my)]
    op_board = [p for p, _a in _poke_iter(op)]
    s += sum(_g(p, "hp", 0) for p in my_board) * 1.0
    s -= sum(_g(p, "hp", 0) for p in op_board) * 1.0
    s += sum(len(_g(p, "energies", []) or []) for p in my_board) * 25.0
    s -= sum(len(_g(p, "energies", []) or []) for p in op_board) * 25.0
    s += len(my_board) * 40.0
    return s
