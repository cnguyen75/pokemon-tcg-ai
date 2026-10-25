"""Regression check for the v12 Hammer-cut variants vs the REST of the panel.

The leak screen (screen_v12.py, SEED 7000, 204g/cell) showed:
  vs archaludon: v11 30.9 | v12a 31.9 | v12b 28.9  -> no movement
  vs lucario:    v11 42.6 | v12a 48.0 | v12b 49.5  -> +5-7pp, both variants
Hammer cuts could still bleed elsewhere; this measures v11/v12a/v12b vs
alakazam, grimmsnarl and the mirror at the same seeds so the meta-weighted
delta can be computed over the full gauntlet panel.

Usage:
  uv run --python 3.13 --with numpy,torch python tools/screen_v12_rest.py
"""

import math
from multiprocessing import Pool
import os
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from league import play_games, wilson  # noqa: E402


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def B(name: str) -> str:
    return os.path.join(REPO, "baselines", name)


AGENTS = {"v11": B("v11"), "v12a": B("v12a"), "v12b": B("v12b"), "v12c": B("v12c")}
OPPS = {
    "alakazam": B("fw_alakazam"),
    "grimm": B("fw_grimm"),
    "mirror": B("v71_ref"),
    "archaludon": B("fw_archaludon"),
    "lucario": B("lucario1084"),
}
GAMES = 200
SEED = 7000
PROCS = max((os.cpu_count() or 4) - 2, 1)


def main() -> None:
    # Optional argv: AGENT OPP -> run just that cell (background jobs get
    # killed by the external watchdog; run cells in foreground instead).
    agents, opps = AGENTS, OPPS
    if len(sys.argv) == 3:
        agents = {sys.argv[1]: AGENTS[sys.argv[1]]}
        opps = {sys.argv[2]: OPPS[sys.argv[2]]}
    for aname, apath in agents.items():
        for oname, opath in opps.items():
            per = math.ceil(GAMES / PROCS)
            jobs = [(apath, opath, per, SEED + i) for i in range(PROCS)]
            t0 = time.time()
            with Pool(PROCS) as pool:
                outs = pool.map(play_games, jobs)
            res = [r for rs, _ in outs for r in rs]
            w = sum(r == 1 for r in res)
            l = sum(r == -1 for r in res)
            _, lo, hi = wilson(w, w + l)
            print(
                f"{aname:5s} vs {oname:10s}: {w}-{l} "
                f"({100 * w / max(w + l, 1):.1f}%) "
                f"CI[{100 * lo:.1f},{100 * hi:.1f}]  "
                f"{len(res)}g {time.time() - t0:.0f}s",
                flush=True,
            )
    print("SCREEN_DONE", flush=True)


if __name__ == "__main__":
    main()
