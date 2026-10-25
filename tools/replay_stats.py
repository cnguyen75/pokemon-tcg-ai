"""Aggregate win/loss patterns from OUR cached ladder replays vs one archetype.

Contrasts our wins vs losses against real opponents (same idea as
matchstats.py, but over real Kaggle replays instead of local games).

Usage:
  uv run --python 3.13 python tools/replay_stats.py ARCHKEY   # e.g. Alakazam
"""

from collections import Counter
import csv
import glob
import json
import os
import sys
from typing import Any


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MY_TEAM = "Colin19"

NAMES, CAT = {}, {}
with open(os.path.join(ROOT, "data", "EN_Card_Data.csv")) as fh:
    for r in csv.DictReader(fh):
        cid = int(r["Card ID"])
        NAMES.setdefault(cid, r["Card Name"])
        CAT.setdefault(cid, r.get("Stage (Pokémon)/Type (Energy and Trainer)", ""))


def nm(cid: int) -> str:
    return NAMES.get(cid, f"#{cid}")


def game_logs(d: dict[str, Any], my_idx: int) -> list[dict[str, Any]]:
    """My agent's full log stream: per-step DELTAS, concatenated."""
    logs = []
    for step in d["steps"]:
        obs = step[my_idx].get("observation") or {}
        logs.extend(obs.get("logs") or [])
    return logs


def main() -> None:
    key = sys.argv[1]
    agg = {"W": Counter(), "L": Counter()}
    games = {"W": 0, "L": 0}
    for f in glob.glob(os.path.join(ROOT, "episodes", "mine", "episode-*-replay.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
            tn = d["info"]["TeamNames"]
            if MY_TEAM not in tn:
                continue
            my = tn.index(MY_TEAM)
            da = d["steps"][0][0]["visualize"][0]["action"]
            opp_poke = Counter(nm(c) for c in da[1 - my] if "Pok" in CAT.get(c, ""))
            if not any(key.lower() in n.lower() for n in opp_poke):
                continue
            res = "W" if d["rewards"][my] == 1 else "L"
        except Exception:
            continue
        games[res] += 1
        c = agg[res]
        logs = game_logs(d, my)
        turn = 0
        for lg in logs:
            t, p = lg.get("type"), lg.get("playerIndex")
            side = "my" if p == my else "op"
            cid = lg.get("cardId") or 0
            if t == 2:
                turn += 1
            elif t == 10:  # play card
                n = nm(cid)
                if n in (
                    "Judge",
                    "Unfair Stamp",
                    "Xerosic’s Machinations",
                    "Lillie's Determination",
                    "Wally's Compassion",
                    "Boss’s Orders",
                    "Hero’s Cape",
                    "Enhanced Hammer",
                    "Crushing Hammer",
                    "Night Stretcher",
                ):
                    c[f"{side}_{n}"] += 1
            elif t == 15:  # attack
                c[f"{side}_atk_{lg.get('attackId')}"] += 1
            elif t == 8 and lg.get("areaTo") == 3 and lg.get("areaFrom") in (4, 5):
                # pokemon to discard from active/bench = KO
                n = nm(cid)
                if "Pok" in CAT.get(cid, ""):
                    c[f"{side}_lost_{n}"] += 1
        c["turns"] += turn
    for res in ("L", "W"):
        g = games[res]
        print(f"\n===== {res} ({g} games) =====")
        if not g:
            continue
        for k, v in sorted(agg[res].items(), key=lambda kv: -kv[1] / g):
            print(f"  {k}: {v}  ({v / g:.2f}/game)")


if __name__ == "__main__":
    main()
