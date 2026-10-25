"""Lucario matchup-override knob screen (config-level, deck unchanged).

The lucario override's value_margin=30 was tuned 7/6 for the OLD numpy value
net; the AZ head (0.80 vs 0.72 AUC) may deserve more search authority (lower
margin). hprio fetches Crushing Hammer less vs lucario only (hammers feed
Wave Thrust's trash-recycle; deck keeps them for alakazam/grimm).

Control: v11 vs lucario1084 = 42.6% at these seeds (screen_v12.log).

MUST be a real script file: macOS multiprocessing spawn cannot re-import a
stdin heredoc __main__ -> Pool hangs forever (learned 7/21).

Usage: uv run --python 3.13 --with numpy,torch python tools/screen_lucknobs.py VARIANT [OPP]
"""

import math
from multiprocessing import Pool
import os
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from league import play_games, wilson  # noqa: E402


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES = 200
SEED = 7000
PROCS = max((os.cpu_count() or 4) - 2, 1)


def main() -> None:
    variant = sys.argv[1]
    opp = sys.argv[2] if len(sys.argv) > 2 else "lucario1084"
    apath = os.path.join(REPO, "baselines", variant)
    opath = os.path.join(REPO, "baselines", opp)
    per = math.ceil(GAMES / PROCS)
    jobs = [(apath, opath, per, SEED + i) for i in range(PROCS)]
    t0 = time.time()
    with Pool(PROCS) as pool:
        outs = pool.map(play_games, jobs)
    res = [r for rs, _ in outs for r in rs]
    w = sum(r == 1 for r in res)
    losses = sum(r == -1 for r in res)
    _, lo, hi = wilson(w, w + losses)
    print(
        f"{variant} vs {opp}: {w}-{losses} ({100 * w / max(w + losses, 1):.1f}%) "
        f"CI[{100 * lo:.1f},{100 * hi:.1f}] {len(res)}g "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
