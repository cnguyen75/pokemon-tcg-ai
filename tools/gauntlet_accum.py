"""Chunked, seed-varied gauntlet that accumulates W/L per opponent and writes
incremental JSON so a poller can read progress without idling on a monitor.

Runs the SAME 5-opponent meta-weighted panel as tools/gauntlet.py, but in
small chunks (so no single invocation runs long) and appends results to a JSON
file. Reports per-opponent Wilson CIs + meta-weighted WR when done.

Usage:
  uv run --python 3.13 --with numpy python tools/gauntlet_accum.py \\
      AGENT_DIR TAG --target-per 300 --chunk 30 --procs 12 --seed-base 7000

Env passed through to the agent (e.g. CG_A2_POLICY=1 CG_A2_POLICY_TOPK=2).
"""

import argparse
import json
import math
from multiprocessing import Pool
import os
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gauntlet import PANEL  # noqa: E402
from league import play_games, wilson  # noqa: E402


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "runs", "policy_gauntlet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_dir")
    ap.add_argument("tag")
    ap.add_argument("--target-per", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--seed-base", type=int, default=7000)
    args = ap.parse_args()

    target = os.path.abspath(args.agent_dir)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{args.tag}.json")

    state = {
        "tag": args.tag,
        "agent": target,
        "target_per": args.target_per,
        "opps": {opp: {"w": 0, "l": 0, "d": 0, "weight": wt} for opp, (_, wt) in PANEL.items()},
    }
    if os.path.exists(out_path):
        try:
            with open(out_path) as fh:
                prev = json.load(fh)
            if prev.get("agent") == target:
                state = prev
        except Exception:
            pass

    t0 = time.time()
    seed = args.seed_base
    while True:
        # pick the opponent with the fewest completed decisive games
        need = [
            (opp, o)
            for opp, o in state["opps"].items()
            if (o["w"] + o["l"] + o["d"]) < args.target_per
        ]
        if not need:
            break
        # run one chunk for each under-target opponent, round-robin
        for opp, o in need:
            done = o["w"] + o["l"] + o["d"]
            n = min(args.chunk, args.target_per - done)
            opp_dir, _wt = PANEL[opp]
            per = math.ceil(n / args.procs)
            seed += 1
            jobs = [(target, opp_dir, per, seed * 100 + i) for i in range(args.procs)]
            with Pool(args.procs) as pool:
                outs = pool.map(play_games, jobs)
            results = [r for rs, _ in outs for r in rs][:n]
            o["w"] += sum(1 for r in results if r == 1)
            o["l"] += sum(1 for r in results if r == -1)
            o["d"] += sum(1 for r in results if r == 0)
            state["elapsed"] = time.time() - t0
            with open(out_path, "w") as fh:
                json.dump(state, fh, indent=1)
            tot = sum(x["w"] + x["l"] + x["d"] for x in state["opps"].values())
            print(
                f"[{args.tag}] {opp:18s} "
                f"{o['w']}-{o['l']}-{o['d']} "
                f"(total {tot}, {state['elapsed']:.0f}s)",
                flush=True,
            )

    print(f"\n=== {args.tag} FINAL ===", flush=True)
    wsum, wtot = 0.0, 0.0
    for opp, o in state["opps"].items():
        n_dec = o["w"] + o["l"]
        p, lo, hi = wilson(o["w"], n_dec)
        wt = o["weight"]
        wsum += wt * p
        wtot += wt
        print(
            f"{opp:18s} w={wt:.3f} {o['w']}-{o['l']}-{o['d']} "
            f"WR {p * 100:5.1f}% [{lo * 100:.1f},{hi * 100:.1f}]",
            flush=True,
        )
    meta = wsum / wtot if wtot else 0.0
    state["meta_wr"] = meta
    with open(out_path, "w") as fh:
        json.dump(state, fh, indent=1)
    print(f"META-WEIGHTED WR: {meta * 100:.2f}%  ({state.get('elapsed', 0):.0f}s)", flush=True)


if __name__ == "__main__":
    main()
