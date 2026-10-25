"""Print a turn-by-turn story of games where agent A loses to agent B.

Usage: uv run --python 3.13 python tools/tracegame.py A B N_LOSSES [seed] [--wins]
"""

import csv
import os
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CG_HOME = os.path.join(REPO, "data", "sample_submission", "sample_submission")
sys.path.insert(0, CG_HOME)
sys.path.insert(0, os.path.join(REPO, "tools"))
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from league import load_agent  # noqa: E402


NAMES = {}
for r in csv.DictReader(open(os.path.join(REPO, "data", "EN_Card_Data.csv"))):
    NAMES.setdefault(int(r["Card ID"]), r["Card Name"])
ATTACKS = {}
for r in csv.DictReader(open(os.path.join(REPO, "data", "EN_Card_Data.csv"))):
    pass


def nm(cid):
    return NAMES.get(cid, f"#{cid}")


AREA = {1: "deck", 2: "hand", 3: "discard", 4: "ACTIVE", 5: "bench", 6: "prize", 7: "stadium"}


def tell(logs, a_index):
    turn = 0
    prizes = [0, 0]
    for lg in logs:
        t = lg.get("type")
        p = lg.get("playerIndex")
        side = "A" if p == a_index else "B"
        if t == 2:
            turn += 1
            print(f"--- turn {turn} [{side}] ---")
        elif t == 10:
            print(f"  {side} plays {nm(lg.get('cardId'))}")
        elif t == 11:
            print(f"  {side} attach {nm(lg.get('cardId'))} -> {nm(lg.get('cardIdTarget'))}")
        elif t == 12:
            print(f"  {side} evolve {nm(lg.get('cardIdTarget'))} -> {nm(lg.get('cardId'))}")
        elif t == 15:
            print(f"  {side} ATTACK {nm(lg.get('cardId'))} atk={lg.get('attackId')}")
        elif t == 16:
            print(f"    hp {nm(lg.get('cardId'))} {lg.get('value')}")
        elif t == 8:
            print(f"  {side} switch {nm(lg.get('cardIdActive'))} <-> {nm(lg.get('cardIdBench'))}")
        elif t == 6:
            fa, ta = lg.get("fromArea"), lg.get("toArea")
            cid = lg.get("cardId")
            if fa == 6:
                prizes[0 if p == a_index else 1] += 1
                print(f"  {side} takes prize #{prizes[0 if p == a_index else 1]}")
            elif ta == 3 and fa in (4, 5):
                print(f"  {side} -> discard: {nm(cid)} (from {AREA.get(fa)})")
        elif t == 23:
            w = "A" if lg.get("result") == a_index else "B"
            print(f"RESULT winner={w} reason={lg.get('reason')}")


def main():
    a_path, b_path, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
    import random

    random.seed(int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 1000)
    show_wins = "--wins" in sys.argv
    a = load_agent(a_path)
    b = load_agent(b_path)
    shown = 0
    g = -1
    while shown < want and g < 400:
        g += 1
        a_index = g % 2
        agents = [None, None]
        agents[a_index] = a.agent
        agents[1 - a_index] = b.agent
        obs, start = battle_start(
            a.DECK if a_index == 0 else b.DECK, a.DECK if a_index == 1 else b.DECK
        )
        logs = []
        while obs["current"]["result"] < 0:
            idx = obs["current"]["yourIndex"]
            if idx == a_index:
                logs.extend(obs.get("logs") or [])
            obs = battle_select(agents[idx](obs))
        if obs["current"]["yourIndex"] == a_index:
            logs.extend(obs.get("logs") or [])
        battle_finish()
        r = obs["current"]["result"]
        won = r == a_index
        if r != 2 and won == show_wins:
            shown += 1
            print(
                f"\n############ game {g} (A seat {a_index}, A {'WON' if won else 'LOST'}) ############"
            )
            tell(logs, a_index)


if __name__ == "__main__":
    main()
