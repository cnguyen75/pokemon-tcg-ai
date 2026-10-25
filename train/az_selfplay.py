"""Phase-2 self-play generator for the AZ expert-iteration loop.

agent_az plays FULL games with the seeded engine (engine_build/test_cg, CG_SEED).
At every acting MAIN single-pick node (the decision class the net searches over)
we run determinized PUCT-MCTS and RECORD:
  - state features        (D_STATE=101)  from features.state_features
  - option feature block   (K, D_OPT=38) from features.option_features
  - MCTS visit distribution (K,)         normalized root.N  -- the AZ policy target
  - acting side + game id                 (for outcome labeling / holdout)
Every other selection (sub-picks, coin flips, setup, multi-pick, and the
opponent's turns) is delegated to agent2's scripted Policy (agent_az imports it).

During self-play we SAMPLE the played move from the visit distribution with a
temperature (exploration for game diversity); the recorded TARGET is always the
true visit distribution, not the sampled move. Games are labeled with the final
outcome (value target, from the acting side's perspective).

Opponent modes (for diversity):
  --opp self      : agent_az (search ON) vs agent_az (search ON)  [pure self-play]
  --opp OPP_DIR   : agent_az (recorded) vs a baseline pilot dir   [meta-diverse]
We only ever RECORD agent_az's own MAIN decisions.

  uv run --python 3.13 --with torch --with numpy python train/az_selfplay.py \
      --games 200 --sims 80 --procs 8 --seed 0 --opp self --out-tag r1

Writes train/data/az_selfplay_<tag>_<seed>.npz  with ragged option blocks:
  Xs (n_dec, 101)  X_opt (n_opt, 38)  offs (n_dec+1)  pi (n_opt,)  y (n_dec,)
  ep (n_dec,)  side (n_dec,)
Reports games/sec, games/hr, and total (state, visit-policy, value) records.
"""

import argparse
import importlib.util as _ilu
from multiprocessing import Pool
import os
import sys
import time
from types import ModuleType
from typing import Any

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CG_HOME = os.path.join(REPO, "engine_build", "test_cg")
AGENT_AZ = os.path.join(REPO, "agent_az")
AGENT2 = os.path.join(REPO, "agent2")


def _load_az(sims: int, budget: float, deadline: float) -> ModuleType:
    """Import agent_az/main.py in-process with self-play search bounds set via
    env BEFORE import (main.py reads them at module load)."""
    os.environ["CG_AZ_SIMS"] = str(sims)
    os.environ["CG_AZ_BUDGET"] = str(budget)
    os.environ["CG_AZ_DEADLINE"] = str(deadline)
    for m in ("value_features", "features", "az_net", "main"):
        sys.modules.pop(m, None)
    sys.path.insert(0, CG_HOME)
    sys.path.insert(0, AGENT_AZ)
    sys.path.insert(0, AGENT2)
    spec = _ilu.spec_from_file_location("az_main", os.path.join(AGENT_AZ, "main.py"))
    mod = _ilu.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(AGENT_AZ)
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


def _load_opp(path: str) -> ModuleType:
    path = os.path.abspath(path)
    for m in ("value_features", "value_net", "deck_configs", "policy_net", "features"):
        sys.modules.pop(m, None)
    sys.path.insert(0, path)
    spec = _ilu.spec_from_file_location(
        "opp_" + os.path.basename(path), os.path.join(path, "main.py")
    )
    mod = _ilu.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(path)
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
        sys.path.pop(0)
    if not hasattr(mod, "DECK"):
        with open(os.path.join(path, "deck.csv")) as f:
            rows = f.read().split("\n")
        mod.DECK = [int(rows[i]) for i in range(60)]
    return mod


def play_worker(
    args: tuple[int, int, int, float, float, float, str],
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[int],
    np.ndarray,
    np.ndarray,
    int,
    int,
    float,
    int,
]:
    (n_games, seed, sims, budget, deadline, temp, opp_kind) = args
    os.environ["CG_SEED"] = str(seed)
    az = _load_az(sims, budget, deadline)
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    _t_start = time.time()
    np_rng = np.random.RandomState(seed & 0x7FFFFFFF)

    to_obs = to_observation_class
    is_main = az._is_main_pick
    scripted = az._scripted_choice
    mcts_run = az.mcts_run
    state_features = az.feats.state_features
    option_features = az.feats.option_features
    CARD, ATK = az.CARD, az.ATK
    NORM = az._NORM  # noqa (unused; recorded feats are RAW, standardized at train)

    # opponent agent module (self-play => second az instance is same code)
    if opp_kind == "self":
        opp_mod = az
        opp_deck = az.DECK
    else:
        opp_mod = _load_opp(opp_kind)
        opp_deck = opp_mod.DECK
    az_deck = az.DECK

    # per-decision records: (state, opt_block, visit_dist, game_id, side)
    Xs, Xo_blocks, pis, gids, sides, gid_result = [], [], [], [], [], {}
    n_draw = 0
    az_move_wall = 0.0
    az_moves = 0

    def az_agent_recording(obs_dict: dict[str, Any], me: int, gid: int) -> list[int]:
        """agent_az's decision at THIS node, recording MAIN visit dists."""
        nonlocal az_move_wall, az_moves
        if obs_dict.get("select") is None:
            return list(az_deck)
        obs = to_obs(obs_dict)
        sel = obs.select
        # reset per-move timer that mcts_run relies on
        az._G["move_t0"] = time.time()
        if az.NET_OK and az.SEARCH_OK and obs.search_begin_input is not None and is_main(sel):
            t0 = time.time()
            try:
                root, prior, nsim = mcts_run(obs, me)
            except Exception:
                return scripted(obs)
            az_move_wall += time.time() - t0
            az_moves += 1
            N = root.N
            if N.sum() < 1:
                return [int(np.argmax(prior))]
            opts = sel.option
            cur = obs.current
            # record raw feats + normalized visit distribution
            try:
                xs = np.asarray(state_features(cur, me, CARD, ATK), dtype=np.float32)
                xo = np.asarray(
                    [option_features(o, cur, me, CARD, ATK) for o in opts], dtype=np.float32
                )
            except Exception:
                xs = None
            if xs is not None and len(xo) == len(N):
                visit = N / N.sum()
                Xs.append(xs)
                Xo_blocks.append(xo)
                pis.append(visit.astype(np.float32))
                gids.append(gid)
                sides.append(me)
            # play a move sampled from the visit dist (temperature) for diversity
            if temp <= 1e-3:
                a = int(np.argmax(N))
            else:
                pw = np.power(N, 1.0 / temp)
                s = pw.sum()
                a = int(np_rng.choice(len(N), p=pw / s)) if s > 0 else int(np.argmax(N))
            return [a]
        return scripted(obs)

    for g in range(n_games):
        if g % 10 == 0:
            print(
                f"[w seed{seed}] game {g}/{n_games} "
                f"({len(gids)} recs, {time.time() - _t_start:.0f}s)",
                flush=True,
            )
        gid = seed * 1_000_000 + g
        az_index = g % 2
        obs, start = battle_start(
            az_deck if az_index == 0 else opp_deck, az_deck if az_index == 1 else opp_deck
        )
        if start.errorPlayer >= 0:
            raise RuntimeError(f"deck error p{start.errorPlayer} t{start.errorType}")
        rec_lo = len(gids)
        while obs["current"]["result"] < 0:
            idx = obs["current"]["yourIndex"]
            if idx == az_index or opp_kind == "self":
                obs = battle_select(az_agent_recording(obs, idx, gid))
            else:
                obs = battle_select(opp_mod.agent(obs))
        battle_finish()
        r = obs["current"]["result"]
        if r == 2:
            n_draw += 1
            # drop draw records (no value signal)
            del gids[rec_lo:]
            del Xs[rec_lo:]
            del Xo_blocks[rec_lo:]
            del pis[rec_lo:]
            del sides[rec_lo:]
            continue
        gid_result[gid] = r

    # build outcome labels (1 if acting side won)
    ys = np.array(
        [1 if gid_result.get(gid) == side else 0 for gid, side in zip(gids, sides, strict=False)],
        dtype=np.int8,
    )
    return (
        Xs,
        Xo_blocks,
        pis,
        ys,
        np.asarray(gids, dtype=np.int64),
        np.asarray(sides, dtype=np.int8),
        n_games,
        n_draw,
        az_move_wall,
        az_moves,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=80)
    ap.add_argument(
        "--budget",
        type=float,
        default=8.0,
        help="search time budget per move (s); high so sims cap binds",
    )
    ap.add_argument("--deadline", type=float, default=12.0)
    ap.add_argument(
        "--temp", type=float, default=0.8, help="visit-count sampling temperature for played move"
    )
    ap.add_argument("--procs", type=int, default=max(os.cpu_count() - 2, 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--opp", default="self", help="'self' or a baseline agent dir (e.g. baselines/fw_grimm)"
    )
    ap.add_argument("--out-tag", default="r1")
    args = ap.parse_args()

    opp_kind = args.opp
    if opp_kind != "self":
        opp_kind = os.path.abspath(opp_kind)

    per = -(-args.games // args.procs)
    base = args.seed * 100_000
    jobs = [
        (per, base + i, args.sims, args.budget, args.deadline, args.temp, opp_kind)
        for i in range(args.procs)
    ]
    t0 = time.time()
    with Pool(args.procs) as pool:
        outs = pool.map(play_worker, jobs)
    dt = time.time() - t0

    Xs = [x for o in outs for x in o[0]]
    Xo_blocks = [b for o in outs for b in o[1]]
    pis = [p for o in outs for p in o[2]]
    ys = np.concatenate([o[3] for o in outs]) if outs else np.zeros(0, np.int8)
    gids = np.concatenate([o[4] for o in outs]) if outs else np.zeros(0, np.int64)
    sides = np.concatenate([o[5] for o in outs]) if outs else np.zeros(0, np.int8)
    n_games = sum(o[6] for o in outs)
    n_draw = sum(o[7] for o in outs)
    move_wall = sum(o[8] for o in outs)
    n_moves = sum(o[9] for o in outs)

    # pack ragged option blocks into CSR
    n_dec = len(Xs)
    offs = np.zeros(n_dec + 1, dtype=np.int64)
    for i, b in enumerate(Xo_blocks):
        offs[i + 1] = offs[i] + len(b)
    X_opt = (
        np.concatenate(Xo_blocks, axis=0).astype(np.float32)
        if Xo_blocks
        else np.zeros((0, 38), np.float32)
    )
    pi = np.concatenate(pis, axis=0).astype(np.float32) if pis else np.zeros(0, np.float32)
    Xs_arr = np.stack(Xs).astype(np.float32) if Xs else np.zeros((0, 101), np.float32)

    tag = f"{args.out_tag}_{args.opp if args.opp == 'self' else os.path.basename(opp_kind)}"
    out = os.path.join(REPO, "train", "data", f"az_selfplay_{tag}_{args.seed}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, Xs=Xs_arr, X_opt=X_opt, offs=offs, pi=pi, y=ys, ep=gids, side=sides)

    gph = n_games / dt * 3600 if dt else 0
    avg_move = move_wall / max(n_moves, 1)
    print(
        f"[az_selfplay {tag} seed{args.seed}] {n_games} games "
        f"({n_draw} draws) in {dt:.0f}s | {n_games / dt:.3f} g/s "
        f"= {gph:.0f} games/hr ({args.procs} procs, {args.sims} sims)",
        flush=True,
    )
    print(
        f"  records: {n_dec} MAIN decisions, {X_opt.shape[0]} option rows, "
        f"value y-mean {ys.mean():.3f} | {avg_move * 1000:.0f} ms/MCTS-move "
        f"-> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
