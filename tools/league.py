"""Local league: pit two agent modules against each other over N games.

Usage: uv run --python 3.13 python tools/league.py AGENT_A AGENT_B N [--procs P]
Agent spec = path to a directory containing main.py + deck.csv.
Reports win rate with a 95% Wilson interval. Seat is alternated.
"""

import argparse
import importlib.util
import math
from multiprocessing import Pool
import os
import random
import sys
import time


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CG_HOME = os.path.join(REPO, "data", "sample_submission", "sample_submission")


def load_agent(path):
    path = os.path.abspath(path)
    # purge agent-local helper modules so each agent imports its OWN copies
    # (agent2 and baselines/v5 both ship value_net.py; a cached module would
    # silently make both agents share one set of weights)
    for m in ("value_net", "value_features", "deck_configs", "policy_net"):
        sys.modules.pop(m, None)
    sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(
        "agent_" + os.path.basename(path), os.path.join(path, "main.py")
    )
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(path)  # so read_deck_csv finds deck.csv
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


def play_games(args):
    a_path, b_path, n_games, seed = args
    sys.path.insert(0, CG_HOME)
    from cg.game import battle_finish, battle_select, battle_start

    random.seed(seed)
    a = load_agent(a_path)
    b = load_agent(b_path)
    results = []  # +1 a wins, -1 b wins, 0 draw
    t0 = time.time()
    for g in range(n_games):
        a_index = g % 2
        agents = [None, None]
        agents[a_index] = a.agent
        agents[1 - a_index] = b.agent
        obs, start = battle_start(
            a.DECK if a_index == 0 else b.DECK, a.DECK if a_index == 1 else b.DECK
        )
        if start.errorPlayer >= 0:
            raise RuntimeError(f"deck error p{start.errorPlayer} t{start.errorType}")
        # deck-choice step happens inside battle_start; play from here
        while obs["current"]["result"] < 0:
            idx = obs["current"]["yourIndex"]
            obs = battle_select(agents[idx](obs))
        battle_finish()
        r = obs["current"]["result"]
        if r == 2:
            results.append(0)
        elif r == a_index:
            results.append(1)
        else:
            results.append(-1)
    return results, time.time() - t0


def wilson(w, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = w / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, center - rad, center + rad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_a")
    ap.add_argument("agent_b")
    ap.add_argument("n", type=int)
    ap.add_argument("--procs", type=int, default=max(os.cpu_count() - 2, 1))
    ap.add_argument(
        "--seed-base",
        type=int,
        default=1000,
        help="worker seeds = seed_base + i; games are deterministic "
        "given the seed, so repeated chunks MUST vary this",
    )
    args = ap.parse_args()

    per = math.ceil(args.n / args.procs)
    jobs = [(args.agent_a, args.agent_b, per, args.seed_base + i) for i in range(args.procs)]
    t0 = time.time()
    with Pool(args.procs) as pool:
        outs = pool.map(play_games, jobs)
    results = [r for rs, _ in outs for r in rs]
    n = len(results)
    w = sum(1 for r in results if r == 1)
    l = sum(1 for r in results if r == -1)
    d = n - w - l
    p, lo, hi = wilson(w, w + l)
    dt = time.time() - t0
    print(
        f"{os.path.basename(args.agent_a)} vs {os.path.basename(args.agent_b)}: "
        f"{w}W-{l}L-{d}D of {n}  winrate {p * 100:.1f}% "
        f"[{lo * 100:.1f}, {hi * 100:.1f}]  ({dt:.0f}s, {n / dt:.2f} games/s)"
    )


if __name__ == "__main__":
    main()
