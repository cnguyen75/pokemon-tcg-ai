"""Paired deck screen: v11 control vs Hammer-cut variants, vs the live leaks.

2026-07-21 replay analysis (172 real ladder games, v10+v11): Archaludon 24%
and Mega Lucario 43% are the two bleeds (~31% of field combined). Both re-
attach energy FROM THE TRASH (Archaludon ex evolve ability, Wave Thrust), so
Crushing Hammer feeds them; in losses vs both we cast 2-3x more Hammers than
in wins.

  v12a: -4 Crushing Hammer  +1 Cape +1 Ultra Ball +1 Boss's +1 Night Stretcher
  v12b: -2 Crushing Hammer  +1 Cape +1 Ultra Ball

Same seeds across agents -> paired comparison. Control = baselines/v11.

Usage:
  uv run --python 3.13 --with numpy,torch python tools/screen_v12.py
"""

from collections import Counter
import math
from multiprocessing import Pool
import os
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from league import load_agent, play_games, wilson  # noqa: E402


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def B(name: str) -> str:
    return os.path.join(REPO, "baselines", name)


# v11 control already measured at SEED 7000 (logs/screen_v12.log, 2026-07-21):
#   v11 vs archaludon 63-141 (30.9%) CI[24.9,37.5]; vs lucario 87-117 (42.6%)
#   CI[36.1,49.5] -- both within a point or two of the LIVE ladder rates,
#   validating pilot fidelity on the leak matchups.
# Hero's Cape (1159) is the deck's ACE SPEC -> a second copy is ILLEGAL
# (engine deck error t4); variants rebuilt with Mega Signal instead.
AGENTS = {"v12a": B("v12a"), "v12b": B("v12b")}
OPPS = {"archaludon": B("fw_archaludon"), "lucario": B("lucario1084")}
GAMES = 200
SEED = 7000
PROCS = max((os.cpu_count() or 4) - 2, 1)


def main() -> None:
    # Deck verification first (the v11-confound lesson: never trust a build
    # until the loaded agent's actual deck is printed).
    for name, path in AGENTS.items():
        c = Counter(load_agent(path).DECK)
        print(
            f"DECK {name}: hammer={c[1120]} cape={c[1159]} ball={c[1121]} "
            f"boss={c[1182]} stretcher={c[1097]} judge={c[1213]} "
            f"total={sum(c.values())}",
            flush=True,
        )

    for aname, apath in AGENTS.items():
        for oname, opath in OPPS.items():
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
                f"{aname:5s} vs {oname:10s}: {w}-{losses} "
                f"({100 * w / max(w + losses, 1):.1f}%) "
                f"CI[{100 * lo:.1f},{100 * hi:.1f}]  "
                f"{len(res)}g {time.time() - t0:.0f}s",
                flush=True,
            )
    print("SCREEN_DONE", flush=True)


if __name__ == "__main__":
    main()
