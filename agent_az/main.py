"""agent_az: AlphaZero-style determinized PUCT-MCTS agent (Phase 1).

The network (az_net.py, BC-bootstrapped by train/train_az.py) supplies BOTH the
policy prior over legal MAIN options AND the value (win-prob) leaf evaluation.
Search runs over the engine Search API (search_begin/step/end) with per-simulation
determinization of the hidden information (opponent deck/hand/prizes + our own
deck/prizes), exactly like agent2.begin_world.

Scope of search: MCTS drives only the MAIN single-pick decisions (SelectType.MAIN,
SelectContext.MAIN, maxCount==1, >=2 options) -- the same decision class the net
was trained on. Every other selection (sub-picks inside a card play, coin flips,
setup, multi-pick) is delegated to agent2's scripted Policy, which we import; this
keeps the tree tractable and reuses proven micro-decisions.

Determinized PUCT-MCTS (IS-MCTS style):
  each simulation samples a fresh determinized world from the ROOT observation,
  then descends a shared tree keyed by the action sequence. At a leaf the net gives
  (prior, value). Because MY hand/board are fully observed at the root, my legal
  MAIN option list is stable across worlds, so option-index children share stats.
  Opponent turns inside a simulation are played by the scripted Policy until it is
  our MAIN move again (determinized opponent model). N sims bounded by a hard
  per-move time budget with a scripted-policy panic fallback.

  uv run --python 3.13 --with torch --with numpy python tools/gauntlet.py agent_az
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import time
from typing import Any


AGENT_DIR = "/kaggle_simulations/agent"
if not os.path.isdir(os.path.join(AGENT_DIR, "cg")):
    try:
        AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        AGENT_DIR = os.getcwd()
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)
REPO = os.path.dirname(AGENT_DIR)

# For a SELF-CONTAINED Kaggle submission, agent_az bundles agent2's scripted
# Policy (choose(), begin_world, DECK) + its helpers (deck_configs.py,
# value_features.py, value_net.py/.npz) under agent_az/scripted/. We load that
# bundled copy first. If it is absent (older checkout), fall back to the sibling
# agent2/ directory so local dev / self-play still works.
import importlib.util as _ilu  # noqa: E402

from cg.api import (  # noqa: E402
    SelectContext,
    SelectType,
    all_attack,
    all_card_data,
    to_observation_class,
)


_SCRIPTED_BUNDLE = os.path.join(AGENT_DIR, "scripted")
if os.path.isfile(os.path.join(_SCRIPTED_BUNDLE, "scripted_policy.py")):
    _a2_path = os.path.join(_SCRIPTED_BUNDLE, "scripted_policy.py")
    _a2_home = _SCRIPTED_BUNDLE
else:
    _a2_path = os.path.join(REPO, "agent2", "main.py")
    _a2_home = os.path.join(REPO, "agent2")
if _a2_home not in sys.path:
    sys.path.insert(0, _a2_home)

_spec = _ilu.spec_from_file_location("az_scripted_policy", _a2_path)
_a2 = _ilu.module_from_spec(_spec)
_cwd = os.getcwd()
os.chdir(_a2_home)
try:
    _spec.loader.exec_module(_a2)
finally:
    os.chdir(_cwd)

Policy = _a2.Policy
begin_world = _a2.begin_world
DECK = list(_a2.DECK)
SEARCH_OK = _a2.SEARCH_OK
from cg.api import search_end, search_step  # noqa: E402


CARD = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: a for a in all_attack()}

import features as feats  # noqa: E402
import numpy as np  # noqa: E402


NET = None
NET_OK = False
_NORM = None
try:
    from az_net import AZNet
    import torch

    _ckpt_path = os.path.join(AGENT_DIR, "az_net.pt")
    _ck = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
    NET = AZNet(
        _ck["d_state"], _ck["d_opt"], d_model=_ck["d_model"], n_blocks=int(_ck.get("n_blocks", 0))
    )
    NET.load_state_dict(_ck["state_dict"])
    NET.eval()
    torch.set_num_threads(1)
    _NORM = {k: np.asarray(_ck[k], dtype=np.float32) for k in ("s_mu", "s_sd", "o_mu", "o_sd")}
    NET_OK = True
except Exception as _e:  # pragma: no cover
    NET_OK = False

MOVE_DEADLINE = float(os.environ.get("CG_AZ_DEADLINE", "2.2"))  # hard per-move (s)
SEARCH_BUDGET = float(os.environ.get("CG_AZ_BUDGET", "1.6"))  # search time/move (s)
MAX_SIMS = int(os.environ.get("CG_AZ_SIMS", "240"))
C_PUCT = float(os.environ.get("CG_AZ_CPUCT", "1.5"))
ROLLOUT_DEPTH = int(os.environ.get("CG_AZ_ROLL_DEPTH", "2"))  # opp turns modeled
PANIC_RESERVE = 60.0

_STATS = {"moves": 0, "sims": 0, "search_moves": 0, "t": 0.0, "max_move": 0.0, "fallbacks": 0}


def _is_main_pick(sel: Any) -> bool:
    return (
        sel is not None
        and sel.type == int(SelectType.MAIN)
        and sel.context == int(SelectContext.MAIN)
        and sel.maxCount == 1
        and len(sel.option or []) >= 2
    )


def _fallback_choice(sel: Any) -> list[int]:
    n = len(sel.option or [])
    lo = max(sel.minCount, 0)
    if n == 0:
        return []
    return list(range(min(max(lo, 1), n)))


def _net_eval(cur: Any, me: int, options: list[Any]) -> tuple[np.ndarray, float]:
    """(prior over options np(K,), value in [0,1]) from the shared net."""
    xs = np.asarray(feats.state_features(cur, me, CARD, ATK), dtype=np.float32)
    xs = (xs - _NORM["s_mu"]) / _NORM["s_sd"]
    xo = np.asarray(
        [feats.option_features(o, cur, me, CARD, ATK) for o in options], dtype=np.float32
    )
    xo = (xo - _NORM["o_mu"]) / _NORM["o_sd"]
    probs, v = NET.eval_leaf(torch.from_numpy(xs), torch.from_numpy(xo))
    return probs.numpy(), float(v)


class Node:
    __slots__ = ("P", "N", "W", "n_opts")

    def __init__(self, prior: np.ndarray, n_opts: int) -> None:
        self.P = prior  # np(K,) prior over options
        self.N = np.zeros(n_opts, dtype=np.float64)  # visit count / child
        self.W = np.zeros(n_opts, dtype=np.float64)  # total value / child
        self.n_opts = n_opts

    def select(self) -> int:
        q = np.where(self.N > 0, self.W / np.maximum(self.N, 1), 0.0)
        u = C_PUCT * self.P * math.sqrt(self.N.sum() + 1.0) / (1.0 + self.N)
        return int(np.argmax(q + u))


def _scripted_choice(obs: Any) -> list[int]:
    try:
        return Policy(obs).choose()
    except Exception:
        return _fallback_choice(obs.select)


def _advance_to_our_main(st: Any, me: int, root_turn: int, deadline: float) -> tuple[Any, str]:
    """From search state `st`, keep stepping with the scripted policy until we
    reach OUR MAIN pick, a turn advance past the rollout horizon, or terminal.
    Returns (st, kind) with kind in {'main','end','terminal','dead'}."""
    for _ in range(80):
        cur = st.observation.current
        if cur.result >= 0:
            return st, "terminal"
        sel = st.observation.select
        if sel is None or not sel.option or sel.maxCount == 0:
            return st, "dead"
        mine = cur.yourIndex == me
        if mine and _is_main_pick(sel):
            return st, "main"
        # not our MAIN pick: play scripted (ours sub-picks or opponent turn)
        choice = _scripted_choice(st.observation)
        try:
            st = search_step(st.searchId, choice)
        except Exception:
            return st, "dead"
        if time.time() > deadline:
            return st, "end"
    return st, "end"


def _simulate(
    root_obs: Any,
    me: int,
    root_opts: list[Any],
    root_node: Node,
    tree: dict[tuple[int, ...], Node],
    deadline: float,
) -> None:
    """One determinized MCTS simulation from a freshly sampled world."""
    try:
        st = begin_world(root_obs, me, manual_coin=False, opp_energy=True)
    except Exception:
        return
    root_turn = root_obs.current.turn
    # root MAIN pick is guaranteed by construction; select an option
    a0 = root_node.select()
    path = [(root_node, a0)]
    try:
        st = search_step(st.searchId, [a0])
    except Exception:
        return
    horizon_turns = root_turn + 2 * ROLLOUT_DEPTH
    key = (a0,)
    depth = 0
    while True:
        st, kind = _advance_to_our_main(st, me, root_turn, deadline)
        cur = st.observation.current
        if kind == "terminal":
            v = 1.0 if cur.result == me else (0.5 if cur.result == 2 else 0.0)
            break
        if kind in ("dead", "end") or cur.turn > horizon_turns or depth >= 6:
            # leaf: net value (from our perspective)
            try:
                if kind == "main":
                    opts = st.observation.select.option
                    _, v = _net_eval(cur, me, opts)
                else:
                    # not at a MAIN node: value-only via a 2-option dummy is
                    # unnecessary; use net value on current state options if any
                    opts = st.observation.select.option if st.observation.select else []
                    if opts:
                        _, v = _net_eval(cur, me, opts)
                    else:
                        v = 0.5
            except Exception:
                v = 0.5
            break
        # kind == 'main' and within horizon: expand or descend tree
        node = tree.get(key)
        if node is None:
            try:
                prior, v = _net_eval(cur, me, st.observation.select.option)
            except Exception:
                v = 0.5
                break
            tree[key] = Node(prior, len(st.observation.select.option))
            break  # expanded a new leaf; stop and backprop its value
        a = node.select()
        path.append((node, a))
        try:
            st = search_step(st.searchId, [a])
        except Exception:
            v = 0.5
            break
        key = key + (a,)
        depth += 1

    for nd, a in path:
        nd.N[a] += 1
        nd.W[a] += v


def mcts_run(obs: Any, me: int) -> tuple[Node, np.ndarray, int]:
    """Run determinized PUCT-MCTS at a MAIN node. Returns (root, prior, sims).
    root.N holds the per-option visit counts (the improved policy target)."""
    t0 = time.time()
    deadline = min(t0 + SEARCH_BUDGET, _G["move_t0"] + MOVE_DEADLINE - 0.15)
    opts = obs.select.option
    cur = obs.current
    prior, _ = _net_eval(cur, me, opts)
    root = Node(prior, len(opts))
    tree = {}
    sims = 0
    try:
        while sims < MAX_SIMS and time.time() < deadline:
            _simulate(obs, me, opts, root, tree, deadline)
            sims += 1
    finally:
        with contextlib.suppress(Exception):
            search_end()
    _STATS["sims"] += sims
    _STATS["search_moves"] += 1
    return root, prior, sims


def mcts_choice(obs: Any, me: int) -> list[int]:
    """Return [best MAIN option index] via determinized PUCT-MCTS."""
    try:
        root, prior, _ = mcts_run(obs, me)
    except Exception:
        return _fallback_choice(obs.select)
    if root.N.sum() < 1:
        # no sim completed: fall back to raw policy prior
        return [int(np.argmax(prior))]
    return [int(np.argmax(root.N))]


_G = {"move_t0": 0.0, "turn": -99}


def _game_reset() -> None:
    _G["turn"] = -99


def agent(obs_dict: dict) -> list[int]:
    _G["move_t0"] = time.time()
    try:
        out = _agent_inner(obs_dict)
    finally:
        dt = time.time() - _G["move_t0"]
        _STATS["moves"] += 1
        _STATS["t"] += dt
        if dt > _STATS["max_move"]:
            _STATS["max_move"] = dt
    return out


def _agent_inner(obs_dict: dict) -> list[int]:
    try:
        if obs_dict.get("select") is None:
            _game_reset()
            return list(DECK)
        obs = to_observation_class(obs_dict)
        turn = obs.current.turn
        if turn < _G["turn"]:
            _game_reset()
        _G["turn"] = turn
        me = obs.current.yourIndex
        sel = obs.select
        overage = obs_dict.get("remainingOverageTime")
        low_time = isinstance(overage, (int, float)) and overage < PANIC_RESERVE

        # MCTS only on MAIN single-pick nodes with the net available + time
        if (
            NET_OK
            and SEARCH_OK
            and not low_time
            and obs.search_begin_input is not None
            and _is_main_pick(sel)
        ):
            return mcts_choice(obs, me)
        # everything else: scripted policy (sub-picks, coins, setup, multi-pick)
        _STATS["fallbacks"] += 1
        return _scripted_choice(obs)
    except Exception:
        try:
            obs = to_observation_class(obs_dict)
            return _fallback_choice(obs.select)
        except Exception:
            sel = obs_dict.get("select") or {}
            n = len(sel.get("option", []) or [])
            lo = max(sel.get("minCount", 1), 0)
            return list(range(min(max(lo, 1), n)))
