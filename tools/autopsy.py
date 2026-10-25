"""Turn a replay JSON into a readable game story.

Usage: python tools/autopsy.py REPLAY.json [--us TEAMNAME]
"""

import csv
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
names = {}
for r in csv.DictReader(open(os.path.join(ROOT, "data", "EN_Card_Data.csv"))):
    names.setdefault(int(r["Card ID"]), r["Card Name"])
attacks = {}  # attackId -> name filled lazily from cg if available


def nm(cid):
    return names.get(cid, f"#{cid}")


def main():
    path = sys.argv[1]
    d = json.load(open(path))
    teams = d["info"]["TeamNames"]
    rew = d["rewards"]
    print(f"{teams[0]} (r={rew[0]}) vs {teams[1]} (r={rew[1]})  steps={len(d['steps'])}")

    # logs are incremental deltas: accumulate one agent's stream across steps
    perspective = 0
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        perspective = int(sys.argv[2])
    logs = []
    last_step_id = -1
    for step in d["steps"]:
        obs = step[perspective].get("observation") or {}
        sid = obs.get("step", -1)
        if sid != last_step_id:
            logs.extend(obs.get("logs") or [])
            last_step_id = sid

    turn = 0
    prizes = [0, 0]
    for lg in logs:
        t = lg.get("type")
        p = lg.get("playerIndex")
        if t == 2:
            turn += 1
            print(f"\n-- turn {turn} (player {p}) --")
        elif t == 10:
            print(f"  P{p} plays {nm(lg.get('cardId'))}")
        elif t == 11:
            print(f"  P{p} attaches {nm(lg.get('cardId'))} -> {nm(lg.get('cardIdTarget'))}")
        elif t == 12:
            print(f"  P{p} evolves {nm(lg.get('cardIdTarget'))} -> {nm(lg.get('cardId'))}")
        elif t == 15:
            print(f"  P{p} ATTACKS with {nm(lg.get('cardId'))} (attack {lg.get('attackId')})")
        elif t == 16:
            v = lg.get("value")
            print(f"    HP change {nm(lg.get('cardId'))}: {v}")
        elif t == 8:
            print(f"  P{p} switch {nm(lg.get('cardIdActive'))} <-> {nm(lg.get('cardIdBench'))}")
        elif t == 6:
            fa, ta = lg.get("fromArea"), lg.get("toArea")
            # 5=prize per AreaType? print prize takes only when to hand from prize
            if fa == 6:
                prizes[p] += 1
                print(f"  P{p} takes prize #{prizes[p]} ({nm(lg.get('cardId'))})")
        elif t == 23:
            print(
                f"\nRESULT: winner=P{lg.get('result')} reason={lg.get('reason')} "
                f"(1=prizes,2=deckout,3=no-active,4=effect)"
            )
    print(f"\nprizes taken: P0={prizes[0]} P1={prizes[1]}")


if __name__ == "__main__":
    main()
