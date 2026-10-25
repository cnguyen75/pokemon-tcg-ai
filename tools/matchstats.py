"""Aggregate loss-pattern stats for agent A vs agent B over N local games.

Usage: uv run --python 3.13 python tools/matchstats.py AGENT_A AGENT_B N [seed]
Collects from agent A's engine log stream: what died on each side, gusts,
capes, Wally heals, hammers, attack usage — split by win/loss for A.
"""

from collections import Counter
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

POKE = {
    1031,
    1030,
    666,
    678,
    677,
    674,
    673,
    675,
    676,
    721,
    722,
    723,
    646,
    647,
    648,
    860,
    104,
    112,
    169,
    190,
}


def nm(cid: int | None) -> str:
    return NAMES.get(cid, f"#{cid}")


def main() -> None:
    a_path, b_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    import random

    random.seed(int(sys.argv[4]) if len(sys.argv) > 4 else 1000)
    a = load_agent(a_path)
    b = load_agent(b_path)

    stats = {True: Counter(), False: Counter()}
    turn_counts = {True: [], False: []}
    for g in range(n):
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
        if r == 2:
            continue
        won = r == a_index
        st = stats[won]
        st["games"] += 1
        turn = 0
        prizes = [0, 0]
        my_mega_turn = None
        for lg in logs:
            t = lg.get("type")
            p = lg.get("playerIndex")
            mine = p == a_index
            side = "my" if mine else "op"
            if t == 2:
                turn += 1
            elif t == 10:
                cid = lg.get("cardId")
                if cid == 1182:
                    st[f"{side}_boss"] += 1
                if cid == 1229 and mine:
                    st["my_wally"] += 1
                if cid == 1120 and mine:
                    st["my_hammer"] += 1
                if cid == 1141 and not mine:
                    st["op_powerpro"] += 1
            elif t == 11:
                cid = lg.get("cardId")
                tgt = lg.get("cardIdTarget")
                if cid == 1159:
                    st[f"{side}_cape_on_{nm(tgt)}"] += 1
            elif t == 15:
                if mine:
                    st[f"my_atk_{lg.get('attackId')}"] += 1
                else:
                    st[f"op_atk_{lg.get('attackId')}"] += 1
            elif t == 12:
                if mine and lg.get("cardId") == 1031 and my_mega_turn is None:
                    my_mega_turn = turn
            elif t == 6:
                fa, ta = lg.get("fromArea"), lg.get("toArea")
                cid = lg.get("cardId")
                if fa == 6:
                    prizes[0 if mine else 1] += 1
                if ta == 3 and fa in (4, 5) and cid in POKE:
                    st[f"{side}_lost_{nm(cid)}({'A' if fa == 4 else 'B'})"] += 1
            elif t == 23:
                st[
                    f"reason_{lg.get('reason')}_{'me' if lg.get('result') == a_index else 'op'}"
                ] += 1
        st["my_prizes"] += prizes[0]
        st["op_prizes"] += prizes[1]
        if my_mega_turn is None:
            st["no_mega_ever"] += 1
        else:
            st["mega_turn_sum"] += my_mega_turn
            st["mega_games"] += 1
        turn_counts[won].append(turn)

    for won in (False, True):
        st = stats[won]
        tc = turn_counts[won]
        print(
            f"\n===== A {'WON' if won else 'LOST'} ({st['games']} games, "
            f"avg turns {sum(tc) / max(len(tc), 1):.1f}) ====="
        )
        for k, v in sorted(st.items(), key=lambda kv: -kv[1]):
            if k == "games":
                continue
            print(f"  {k}: {v}  ({v / max(st['games'], 1):.2f}/game)")


if __name__ == "__main__":
    main()
