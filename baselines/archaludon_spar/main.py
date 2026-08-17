"""PTCG agent: flat determinized search (PIMC) to end of own turn.

V1 design (see project plan):
- Info tracking: own unseen pool = decklist minus every card we can see;
  sampled consistently into deck order + prizes per determinization.
- Search: for each candidate first selection, greedy-heuristic rollout to
  end of our turn, evaluate with prize-race eval, average over D worlds.
- Never crash, never time out: global try/except + per-move deadline with
  heuristic fallback.
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
    Observation,
    OptionType,
    all_attack,
    search_begin,
    search_end,
    search_step,
    to_observation_class,
)

ATTACK_DAMAGE = {}
try:
    for _a in all_attack():
        ATTACK_DAMAGE[_a.attackId] = getattr(_a, "damage", 0) or 0
except Exception:
    pass

# ---------------------------------------------------------------- deck

def read_deck_csv():
    path = os.path.join(AGENT_DIR, "deck.csv")
    if not os.path.exists(path):
        path = "deck.csv"
    with open(path) as f:
        rows = f.read().split("\n")
    return [int(rows[i]) for i in range(60)]

DECK = read_deck_csv()

FILLER_BASIC = 1072   # Snorlax: safe opponent-active filler (engine needs a settable card)
FILLER_CARD = 1       # basic energy filler for opponent hidden zones

# Time management: overage pool observed at ~600s. Stay far below it.
MOVE_BUDGET = 3.0     # soft seconds per planned move
PANIC_RESERVE = 60.0  # if remainingOverageTime dips below this, play instantly

D_WORLDS = 6          # determinizations per decision
MAX_CANDIDATES = 40   # cap on enumerated first selections
ROLLOUT_STEPS = 60    # safety cap on greedy rollout length


# ---------------------------------------------------------------- helpers

def visible_own_cards(ps):
    """Multiset of our cards whose identity we can currently see."""
    seen = []
    for c in ps.hand or []:
        if c is not None:
            seen.append(c.id)
    for c in ps.discard or []:
        if c is not None:
            seen.append(c.id)
    for zone in (ps.active or []), (ps.bench or []):
        for p in zone:
            if p is None:
                continue
            seen.append(p.id)
            for c in (p.energyCards or []):
                seen.append(c.id)
            for c in (p.tools or []):
                seen.append(c.id)
            for c in (p.preEvolution or []):
                if c is not None:
                    seen.append(c.id)
    return seen


def sample_own_hidden(obs, me):
    """Split our unseen cards consistently into (deck order, prizes)."""
    ps = obs.current.players[me]
    unseen = Counter(DECK)
    for cid in visible_own_cards(ps):
        if unseen[cid] > 0:
            unseen[cid] -= 1
    pool = list(unseen.elements())
    random.shuffle(pool)
    n_deck = ps.deckCount
    n_prize = len(ps.prize or [])
    # pool may mismatch counts if tracking missed a zone; pad/trim defensively
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


# ---------------------------------------------------------------- eval

def eval_state(cur, me):
    if cur.result >= 0:
        if cur.result == 2:
            return 0.0
        return 1000.0 if cur.result == me else -1000.0
    my = cur.players[me]
    op = cur.players[1 - me]
    v = 0.0
    # prize race is the game
    v += 25.0 * ((6 - len(my.prize or [])) - (6 - len(op.prize or [])))
    # board development
    for sign, ps in ((1.0, my), (-1.0, op)):
        for zone in (ps.active or []), (ps.bench or []):
            for p in zone:
                if p is None:
                    continue
                v += sign * (2.0 + 3.0 * (p.hp / max(p.maxHp, 1))
                             + 1.5 * len(p.energyCards or [])
                             + 0.5 * len(p.tools or []))
        v += sign * 0.4 * ps.handCount
        # deck-out pressure: losing at turn start with 0 cards
        if ps.deckCount <= 2:
            v -= sign * (3.0 * (3 - ps.deckCount))
    if not (my.active or []):
        v -= 10.0
    return v


# ---------------------------------------------------------------- action enumeration

_TYPE_PRIO = {
    OptionType.ABILITY: 0,
    OptionType.PLAY: 1,
    OptionType.EVOLVE: 2,
    OptionType.ATTACH: 3,
    OptionType.ATTACK: 4,
    OptionType.RETREAT: 6,
    OptionType.END: 7,
}


def option_priority(o):
    p = _TYPE_PRIO.get(o.type, 5)
    if o.type == OptionType.ATTACK:
        # prefer higher printed damage within attacks
        dmg = ATTACK_DAMAGE.get(getattr(o, "attackId", -1), 0)
        return (p, -dmg)
    return (p, 0)


def enumerate_selections(select):
    """Candidate index-lists honoring min/max count, capped."""
    n = len(select.option)
    lo = max(select.minCount, 0)
    hi = min(select.maxCount, n)
    cands = []
    if hi >= 1:
        order = sorted(range(n), key=lambda i: option_priority(select.option[i]))
        if hi > 1:
            # multi-picks first, maximal selections preferred (optional effects
            # like draws/searches are usually take-as-many-as-you-can)
            base = order[:hi]
            for k in range(hi, max(lo, 2) - 1, -1):
                cands.append(sorted(base[:k]))
            if len(order) > hi:
                cands.append(sorted(order[1:hi + 1]))
        if lo <= 1:
            cands.extend([[i] for i in order])
    if lo == 0:
        cands.append([])  # voluntary pass considered last
    # dedupe, keep order
    seen, out = set(), []
    for c in cands:
        t = tuple(c)
        if t not in seen and lo <= len(c) <= hi:
            seen.add(t)
            out.append(c)
    return out[:MAX_CANDIDATES] or [list(range(min(max(lo, 1), n)))]


def heuristic_choice(select):
    """Instant fallback move honoring constraints."""
    cands = enumerate_selections(select)
    return cands[0] if cands else []


# ---------------------------------------------------------------- search

def greedy_rollout(state, me):
    """Play out our turn AND the opponent's greedy reply, then evaluate.

    Both sides follow the same heuristic policy. The opponent's hidden hand
    is filler in our determinization, so their reply is mostly board attacks
    plus an energy attach — a cheap 1-ply threat model that punishes lines
    that leave us exposed to a return KO.
    """
    start_turn = state.observation.current.turn
    for _ in range(ROLLOUT_STEPS):
        cur = state.observation.current
        if cur.result >= 0:
            return eval_state(cur, me)
        # stop once the opponent's reply turn is over and it's our turn again
        if cur.turn >= start_turn + 2 and cur.yourIndex == me:
            return eval_state(cur, me)
        sel = state.observation.select
        try:
            state = search_step(state.searchId, heuristic_choice(sel))
        except Exception:
            return eval_state(cur, me)
    return eval_state(state.observation.current, me)


DEBUG = os.environ.get("CG_AGENT_DEBUG") == "1"
_stats = {"plans": 0, "worlds_ok": 0, "worlds_fail": 0, "steps_fail": 0}


def plan(obs, me, deadline):
    candidates = enumerate_selections(obs.select)
    if len(candidates) <= 1:
        return candidates[0] if candidates else []
    _stats["plans"] += 1
    scores = {tuple(c): 0.0 for c in candidates}
    counts = {tuple(c): 0 for c in candidates}
    for _ in range(D_WORLDS):
        if time.time() > deadline:
            break
        try:
            root = begin_world(obs, me)
            _stats["worlds_ok"] += 1
        except Exception as e:
            _stats["worlds_fail"] += 1
            if DEBUG:
                import traceback
                print(f"[agent] begin_world fail: {e!r}", file=sys.stderr)
                traceback.print_exc()
            break
        try:
            for c in candidates:
                if time.time() > deadline:
                    break
                try:
                    st = search_step(root.searchId, c)
                except Exception:
                    continue
                cur = st.observation.current
                if cur.result >= 0:
                    v = eval_state(cur, me)
                else:
                    v = greedy_rollout(st, me)
                scores[tuple(c)] += v
                counts[tuple(c)] += 1
        finally:
            try:
                search_end()
            except Exception:
                pass
    evaluated = [(scores[t] / counts[t], list(t)) for t in scores if counts[t] > 0]
    if not evaluated:
        return heuristic_choice(obs.select)
    evaluated.sort(key=lambda x: -x[0])
    return evaluated[0][1]


# ---------------------------------------------------------------- entry point

def agent(obs_dict: dict) -> list[int]:
    try:
        if obs_dict.get("select") is None:
            return DECK
        obs: Observation = to_observation_class(obs_dict)
        me = obs.current.yourIndex
        overage = obs_dict.get("remainingOverageTime")
        budget = MOVE_BUDGET
        if isinstance(overage, (int, float)) and overage < PANIC_RESERVE:
            return heuristic_choice(obs.select)
        deadline = time.time() + budget
        return plan(obs, me, deadline)
    except Exception:
        try:
            obs = to_observation_class(obs_dict)
            return heuristic_choice(obs.select)
        except Exception:
            # last resort: minimal legal reply
            sel = obs_dict.get("select") or {}
            n = len(sel.get("option", []) or [])
            lo = max(sel.get("minCount", 1), 0)
            return list(range(min(max(lo, 1), n)))
