"""Collect every quotable number for the project page, with where it comes from.

Usage:
    python tools/site_results.py OUT.json \
        --submissions SUBMISSIONS.csv --leaderboard LEADERBOARD.csv \
        --team colin19 --episode-lists DIR

Inputs outside the repo (none are committed):
  SUBMISSIONS.csv  `kaggle competitions submissions pokemon-tcg-ai-battle --csv`
  LEADERBOARD.csv  `kaggle competitions leaderboard pokemon-tcg-ai-battle -d` (unzipped)
  DIR              ListEpisodes responses saved as ep_<submissionId>.json from
                   https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes
                   (public, no sign-in), read together with the cached replays in
                   episodes/ and the card list in data/ (both local only)

Parts:
  local_screens  every completed 204-game cell in the tracked logs, with the
                 95% Wilson interval recomputed from its win-loss count and
                 checked against the interval the log printed
  ladder         each distinct build's Kaggle rating and the team's leaderboard row
  matchups       per-archetype ladder win rates from cached replays
"""
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = ["logs/arch_screen.log", "logs/screen_lucknobs.log", "logs/screen_v12.log",
        "logs/screen_v12b.log", "logs/screen_v12_rest_cells.log"]
CELL = re.compile(r"^(\S+)\s+vs\s+(\S+?)\s*:\s+(\d+)-(\d+) \(([\d.]+)%\) CI\[([\d.]+),([\d.]+)\]\s+(\d+)g")


def wilson(w, n, z=1.96):
    p = w / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - r), 1), round(100 * (c + r), 1)]


def local_screens():
    cells = []
    for path in LOGS:
        for i, line in enumerate(open(os.path.join(REPO, path)), 1):
            m = CELL.match(line.strip())
            if not m:
                continue
            a, b, w, l, pct, lo, hi, n = m.groups()
            w, l, n = int(w), int(l), int(n)
            ci = wilson(w, w + l)
            cells.append({"source": f"{path}:{i}", "agent": a, "opponent": b,
                          "wins": w, "losses": l, "games": n,
                          "win_rate": round(100 * w / (w + l), 1), "ci95": ci,
                          "logged_ci95": [float(lo), float(hi)],
                          "ci_matches_log": ci == [float(lo), float(hi)]})
    repeats = defaultdict(list)
    for c in cells:
        repeats[(c["agent"], c["opponent"])].append(c["win_rate"])
    return {
        "command": "python tools/site_results.py OUT.json ...",
        "cells": cells,
        "repeated_cells": {f"{a} vs {b}": v for (a, b), v in repeats.items() if len(v) > 1},
        "ci95_half_width_at_50pct_204_games": round((wilson(102, 204)[1] - wilson(102, 204)[0]) / 2, 1),
    }


def ladder(sub_csv, lb_csv, team):
    rows = list(csv.DictReader(open(sub_csv)))
    rows.sort(key=lambda r: r["date"])
    builds, copies = [], Counter()
    for r in rows:
        name = r["fileName"].replace("submission_", "").replace(".tar.gz", "")
        copies[name] += 1
        if copies[name] > 1:
            continue
        score = float(r["publicScore"]) if r["publicScore"] else None
        builds.append({"build": name, "submitted": r["date"][:10], "rating": score,
                       "status": r["status"].split(".")[-1].lower(), "submission_id": int(r["ref"])})
    scored = [float(r["publicScore"]) for r in rows if r["publicScore"]]
    with open(lb_csv, encoding="utf-8-sig") as f:
        lb = list(csv.DictReader(f))
    mine = next(r for r in lb if team in r["TeamMemberUserNames"].split(","))
    return {
        "source": "Kaggle submissions list and public leaderboard CSV (see the file's docstring)",
        "submissions": len(rows),
        "copies_per_build": {k: v for k, v in copies.items() if v > 1},
        "builds_first_submission": builds,
        "highest_rating_any_submission": max(scored),
        "leaderboard": {"rank": int(mine["Rank"]), "teams": len(lb),
                        "score": float(mine["Score"]), "file": os.path.basename(lb_csv)},
    }


def matchups(ep_dir, submission_ids):
    names, cat = {}, {}
    for r in csv.DictReader(open(os.path.join(REPO, "data", "EN_Card_Data.csv"))):
        cid = int(r["Card ID"])
        names.setdefault(cid, r["Card Name"])
        cat.setdefault(cid, r.get("Stage (Pokémon)/Type (Energy and Trainer)", ""))

    def archetype(deck):  # same rule as tools/my_episodes.py
        poke = Counter(names.get(c, str(c)) for c in deck if "Pok" in cat.get(c, ""))
        tops = [n for n, k in poke.most_common(6) if k >= 2][:3]
        return " / ".join(sorted(tops)) if tops else "?"

    cached = {}
    for f in glob.glob(os.path.join(REPO, "episodes", "**", "episode-*-replay.json"), recursive=True):
        cached[int(os.path.basename(f).split("-")[1])] = f
    out = {}
    for sid in submission_ids:
        eps = json.load(open(os.path.join(ep_dir, f"ep_{sid}.json")))["episodes"]
        agg, used, total_w = defaultdict(lambda: [0, 0]), 0, 0
        for e in eps:
            us = next((a for a in e["agents"] if a.get("submissionId") == sid), None)
            if us is None:
                continue
            total_w += us.get("reward") == 1
            if e["id"] not in cached:
                continue
            rp = json.load(open(cached[e["id"]]))
            deck = rp["steps"][0][0]["visualize"][0]["action"][1 - us.get("index", 0)]
            a = archetype(deck)
            agg[a][1] += 1
            agg[a][0] += us.get("reward") == 1
            used += 1
        out[str(sid)] = {
            "episodes_listed": len(eps), "ladder_wins_all_listed": total_w,
            "episodes_with_cached_replay": used,
            "by_archetype": {a: {"wins": w, "games": g, "win_rate": round(100 * w / g, 1),
                                 "ci95": wilson(w, g)}
                             for a, (w, g) in sorted(agg.items(), key=lambda kv: -kv[1][1])
                             if g >= 5},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--submissions", required=True)
    ap.add_argument("--leaderboard", required=True)
    ap.add_argument("--team", required=True, help="Kaggle username on the team row")
    ap.add_argument("--episode-lists", required=True)
    ap.add_argument("--matchup-submissions", default="54842929,55018765")
    args = ap.parse_args()
    res = {
        "local_screens": local_screens(),
        "ladder": ladder(args.submissions, args.leaderboard, args.team),
        "matchups": matchups(args.episode_lists,
                             [int(s) for s in args.matchup_submissions.split(",")]),
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
        f.write("\n")
    s = res["local_screens"]
    print(f"{len(s['cells'])} local cells, {sum(c['ci_matches_log'] for c in s['cells'])} "
          f"intervals match the log; leaderboard {res['ladder']['leaderboard']}")


if __name__ == "__main__":
    main()
