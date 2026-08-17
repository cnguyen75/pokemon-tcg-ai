"""Sequential full-core gauntlet: one config, all 5 panel opponents, N games
each at a FIXED per-opponent seed base so different configs are paired
(same seeds) per opponent. Prints per-opponent Wilson CIs + meta-weighted WR.

Usage: uv run --python 3.13 --with numpy python tools/gauntlet_seq.py \\
    AGENT_DIR TAG --games-per 300 --procs 13 --seed-base 8800
Agent env (CG_A2_POLICY etc.) passed through the environment.
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from league import play_games, wilson  # noqa: E402
from gauntlet import PANEL  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "runs", "policy_gauntlet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_dir")
    ap.add_argument("tag")
    ap.add_argument("--games-per", type=int, default=300)
    ap.add_argument("--procs", type=int, default=13)
    ap.add_argument("--seed-base", type=int, default=8800)
    a = ap.parse_args()
    target = os.path.abspath(a.agent_dir)
    os.makedirs(OUT, exist_ok=True)
    n = a.games_per
    per = math.ceil(n / a.procs)
    res = {}
    wsum = wtot = 0.0
    t0 = time.time()
    # deterministic per-opponent seed offset so configs pair up
    OFF = {opp: (i + 1) * 1000 for i, opp in enumerate(PANEL)}
    for opp, (opp_dir, wt) in PANEL.items():
        sb = a.seed_base + OFF[opp]
        jobs = [(target, opp_dir, per, sb + i) for i in range(a.procs)]
        with Pool(a.procs) as pool:
            outs = pool.map(play_games, jobs)
        rr = [r for rs, _ in outs for r in rs][:n]
        w = sum(1 for r in rr if r == 1)
        l = sum(1 for r in rr if r == -1)
        d = sum(1 for r in rr if r == 0)
        p, lo, hi = wilson(w, w + l)
        res[opp] = {"w": w, "l": l, "d": d, "wr": p, "lo": lo, "hi": hi,
                    "weight": wt}
        wsum += wt * p
        wtot += wt
        print(f"{opp:18s} w={wt:.3f} {w}-{l}-{d} "
              f"WR {p*100:5.1f}% [{lo*100:.1f},{hi*100:.1f}]  "
              f"({time.time()-t0:.0f}s)", flush=True)
    meta = wsum / wtot
    res["_meta_wr"] = meta
    json.dump(res, open(os.path.join(OUT, a.tag + ".json"), "w"), indent=1)
    print(f"META-WEIGHTED WR: {meta*100:.2f}%   ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
