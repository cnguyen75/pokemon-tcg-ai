"""Per-archetype W/L for OUR submission from its real ladder episodes.

Lists episodes via Kaggle's EpisodeService, downloads each replay, classifies
the opponent's deck from the step-0 decklist, and aggregates win rates.
This measures matchups against the REAL field — no local pilots needed.

Usage:
  uv run --python 3.13 python tools/my_episodes.py SUBMISSION_ID [--max N]
"""

import argparse
import csv
import json
import os
import time
from typing import Any
import urllib.request


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "episodes", "mine")
API = "https://www.kaggle.com/api/i/competitions.EpisodeService"

NAMES = {}
CATEGORY = {}
for r in csv.DictReader(open(os.path.join(ROOT, "data", "EN_Card_Data.csv"))):
    cid = int(r["Card ID"])
    NAMES.setdefault(cid, r["Card Name"])
    # Pokémon are identified by the Stage column ("Basic Pokémon", "Stage 1
    # Pokémon", ...); the Category column is n/a for them
    CATEGORY.setdefault(cid, r.get("Stage (Pokémon)/Type (Energy and Trainer)", ""))


def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{API}/{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as f:
        return json.load(f)


def archetype(deck: list[int]) -> str:
    from collections import Counter

    poke = Counter()
    for cid in deck:
        if "Pok" in CATEGORY.get(cid, ""):
            poke[NAMES.get(cid, str(cid))] += 1
    # top evolution line names, highest counts first, skip generic support
    tops = [n for n, c in poke.most_common(6) if c >= 2][:3]
    return " / ".join(sorted(tops)) if tops else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_id", type=int)
    ap.add_argument("--max", type=int, default=200)
    args = ap.parse_args()

    eps = _post("ListEpisodes", {"submissionId": args.submission_id})["episodes"]
    eps = [e for e in eps if e.get("state") == "COMPLETED"][: args.max]
    print(f"{len(eps)} completed episodes for submission {args.submission_id}", flush=True)

    os.makedirs(CACHE, exist_ok=True)
    from collections import defaultdict

    agg = defaultdict(lambda: [0, 0])  # archetype -> [wins, games]
    errs = 0
    for i, e in enumerate(eps):
        eid = e["id"]
        # our index & reward
        us = next((a for a in e["agents"] if a.get("submissionId") == args.submission_id), None)
        if us is None:
            continue
        my_idx = us.get("index", 0)
        rew = us.get("reward")
        cache_f = os.path.join(CACHE, f"episode-{eid}-replay.json")
        try:
            if not os.path.exists(cache_f):
                import subprocess

                subprocess.run(
                    [
                        os.path.expanduser("~/.local/bin/kaggle"),
                        "competitions",
                        "replay",
                        str(eid),
                        "-p",
                        CACHE,
                        "-q",
                    ],
                    capture_output=True,
                    timeout=120,
                )
                time.sleep(0.4)
            d = json.load(open(cache_f))
            deck_action = d["steps"][0][0]["visualize"][0]["action"]
            opp_deck = deck_action[1 - my_idx]
        except Exception:
            errs += 1
            continue
        arch = archetype(opp_deck)
        agg[arch][1] += 1
        if rew == 1:
            agg[arch][0] += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(eps)} replays processed", flush=True)

    print(f"\n=== per-archetype W/L (submission {args.submission_id}, {errs} fetch errors) ===")
    tot_w = tot_g = 0
    for arch, (w, g) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        tot_w += w
        tot_g += g
        print(f"  {g:3d}g {100 * w / g:5.1f}%  {arch}")
    if tot_g:
        print(f"  ---\n  {tot_g:3d}g {100 * tot_w / tot_g:5.1f}%  OVERALL")


if __name__ == "__main__":
    main()
