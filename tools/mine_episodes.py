"""Nightly episode miner: download a sample of yesterday's top-rated episodes.

Downloads up to MAX_FILES episode JSONs from the daily dataset
kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD into episodes/daily/YYYY-MM-DD/,
then writes a compact decklist/meta summary next to them.

Run: python tools/mine_episodes.py [YYYY-MM-DD]  (default: yesterday UTC)
"""

from collections import Counter, defaultdict
import csv
from datetime import UTC, datetime, timedelta
import json
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE = os.path.expanduser("~/.local/bin/kaggle")
MAX_FILES = int(os.environ.get("MINE_MAX_FILES", "400"))  # ~1.5GB/day; full day is ~21GB


def sh(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=600)


def main() -> int:
    if len(sys.argv) > 1:
        day = sys.argv[1]
    else:
        day = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
    slug = f"kaggle/pokemon-tcg-ai-battle-episodes-{day}"
    outdir = os.path.join(ROOT, "episodes", "daily", day)
    os.makedirs(outdir, exist_ok=True)

    r = sh([KAGGLE, "datasets", "files", slug, "--page-size", str(MAX_FILES)])
    files = [
        ln.split()[0]
        for ln in r.stdout.splitlines()
        if ln.strip().endswith(".json") or (ln.split() and ln.split()[0].endswith(".json"))
    ]
    files = files[:MAX_FILES]
    if not files:
        print(f"no files found for {slug}; stdout head: {r.stdout[:300]} err: {r.stderr[:300]}")
        return 1
    got = 0
    for f in files:
        dest = os.path.join(outdir, f)
        if os.path.exists(dest):
            continue
        rr = sh([KAGGLE, "datasets", "download", slug, "-f", f, "-p", outdir, "--unzip", "-q"])
        if rr.returncode == 0:
            got += 1
    print(f"{day}: downloaded {got} new episodes into {outdir}")

    # meta summary: deck archetypes by (top-3 non-trainer cards), win rates
    names = {}
    for row in csv.DictReader(open(os.path.join(ROOT, "data", "EN_Card_Data.csv"))):
        names.setdefault(int(row["Card ID"]), row["Card Name"])
    arch = defaultdict(lambda: [0, 0])
    for f in os.listdir(outdir):
        if not f.endswith(".json") or f == "meta_summary.json":
            continue
        try:
            d = json.load(open(os.path.join(outdir, f)))
            decks = d["steps"][0][0]["visualize"][0]["action"]
            rew = d["rewards"]
        except Exception:
            continue
        for i in range(2):
            c = Counter(decks[i])
            key = " / ".join(
                sorted(
                    names.get(cid, "?")
                    for cid, ct in c.most_common(8)
                    if 20 < cid < 1077 and ct >= 3
                )[:3]
            )
            arch[key][1] += 1
            if rew[i] == 1:
                arch[key][0] += 1
    summary = sorted(((w, g, k) for k, (w, g) in arch.items()), key=lambda x: -x[1])
    with open(os.path.join(outdir, "meta_summary.json"), "w") as fh:
        json.dump([{"archetype": k, "wins": w, "games": g} for w, g, k in summary], fh, indent=1)
    print("top archetypes:")
    for w, g, k in summary[:10]:
        print(f"  {g:4d} games {100 * w / max(g, 1):5.1f}% wr  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
