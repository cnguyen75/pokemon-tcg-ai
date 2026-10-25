"""Per-turn timeline autopsy of our games vs one archetype, from obs.current
board snapshots in cached ladder replays (episodes/mine/).

Prints one line per (our) turn: prize race, both actives (id/hp/energy),
our bench shape. KOs/heals show up as hp/prize transitions between turns.

Usage:
  uv run --python 3.13 python tools/arch_autopsy.py FLAGSHIP_CARD_ID [W|L]
  e.g. Archaludon ex = 190
"""

import csv
import glob
import json
import os
import sys
from typing import Any


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MY_TEAM = "Colin19"

NAMES = {}
with open(os.path.join(ROOT, "data", "EN_Card_Data.csv")) as fh:
    for r in csv.DictReader(fh):
        NAMES.setdefault(int(r["Card ID"]), r["Card Name"])


def nm(cid: int) -> str:
    n = NAMES.get(cid, f"#{cid}")
    return n[:14]


def mon(m: dict[str, Any] | None) -> str:
    if not m:
        return "-"
    e = len(m.get("energies") or [])
    tool = "+T" if m.get("tools") else ""
    return f"{nm(m['id'])} {m['hp']}/{m['maxHp']}hp {e}e{tool}"


def main() -> None:
    flagship = int(sys.argv[1])
    want = sys.argv[2] if len(sys.argv) > 2 else "L"
    for f in sorted(glob.glob(os.path.join(ROOT, "episodes", "mine", "episode-*-replay.json"))):
        try:
            with open(f) as fh:
                d = json.load(fh)
            tn = d["info"]["TeamNames"]
            if MY_TEAM not in tn:
                continue
            my = tn.index(MY_TEAM)
            opp_deck = d["steps"][0][0]["visualize"][0]["action"][1 - my]
            if flagship not in opp_deck:
                continue
            res = "W" if d["rewards"][my] == 1 else "L"
            if res != want:
                continue
        except Exception:
            continue
        print(f"\n===== {res} {os.path.basename(f)} =====")
        last_turn = -1
        for step in d["steps"]:
            obs = step[my].get("observation") or {}
            cur = obs.get("current")
            if not cur:
                continue
            turn = cur.get("turn", 0)
            if turn == last_turn:
                continue
            last_turn = turn
            me = cur["players"][cur["yourIndex"]]
            op = cur["players"][1 - cur["yourIndex"]]
            mya = (me.get("active") or [None])[0]
            opa = (op.get("active") or [None])[0]
            bench = ",".join(
                f"{nm(b['id'])}({len(b.get('energies') or [])}e)" for b in (me.get("bench") or [])
            )
            print(
                f"t{turn:2d} prz {me['prize']}-{op['prize']} | "
                f"me: {mon(mya)} | op: {mon(opa)} | "
                f"bench: {bench} | hand {me['handCount']} "
                f"deck {me['deckCount']}"
            )


if __name__ == "__main__":
    main()
