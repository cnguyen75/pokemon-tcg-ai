"""Parse kaggle replay JSONs into (features, label, episode_id) training rows.

For every step where an agent acted (status ACTIVE, fresh observation.current
with matching yourIndex), extract the feature vector from the actor's
perspective AND the mirrored opponent perspective (so the my_turn feature and
both-side symmetry are learnable). Label = did that perspective's player win.

Usage: uv run --python 3.13 --with numpy python train/extract_episodes.py
Writes train/data/episodes.npz  (X float32, y int8, ep int32, hc float32).
"""

import glob
import json
import os
import sys

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agent2"))
sys.path.insert(0, os.path.join(REPO, "data", "sample_submission", "sample_submission"))

from cg.api import all_attack, all_card_data  # noqa: E402
import value_features as vf  # noqa: E402


CARD = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: a for a in all_attack()}

DIRS = glob.glob(os.path.join(REPO, "episodes", "daily", "*")) + [
    os.path.join(REPO, "episodes", d)
    for d in (
        "day30",
        "ours",
        "v2games",
        "v3games",
        "v4games",
        "v5games",
        "v51games",
        "v52games",
        "v5recent",
    )
]


def parse_file(path: str, ep_id: int) -> tuple[list[list[float]], list[int], list[float]]:
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return [], [], []
    if not isinstance(d, dict) or "steps" not in d:
        return [], [], []  # meta_summary.json and friends
    rew = d.get("rewards")
    if (
        not isinstance(rew, list)
        or len(rew) != 2
        or rew[0] is None
        or rew[1] is None
        or rew[0] == rew[1]
    ):
        return [], [], []
    won = [1 if rew[0] > rew[1] else 0, 1 if rew[1] > rew[0] else 0]
    X, y, hc = [], [], []
    for step in d.get("steps", []):
        for j, ag in enumerate(step):
            if ag.get("status") != "ACTIVE":
                continue
            obs = ag.get("observation") or {}
            cur = obs.get("current")
            if not cur or obs.get("select") is None:
                continue
            if cur.get("yourIndex") != j:
                continue
            if cur.get("result", -1) >= 0:
                continue
            for me in (j, 1 - j):
                try:
                    X.append(vf.extract(cur, me, CARD, ATK))
                    y.append(won[me])
                    hc.append(vf.handcrafted_eval(cur, me))
                except Exception:
                    if len(X) > len(y):
                        X.pop()
    return X, y, hc


def main() -> None:
    files = []
    for dpath in DIRS:
        files += sorted(glob.glob(os.path.join(dpath, "*.json")))
    print(f"{len(files)} replay files")
    Xs, ys, eps, hcs = [], [], [], []
    n_used = 0
    by_src = {}  # source dir -> [files_used, rows]
    for i, path in enumerate(files):
        X, y, hc = parse_file(path, i)
        if X:
            n_used += 1
            Xs.extend(X)
            ys.extend(y)
            hcs.extend(hc)
            eps.extend([i] * len(X))
            src = os.path.relpath(os.path.dirname(path), os.path.join(REPO, "episodes"))
            st = by_src.setdefault(src, [0, 0])
            st[0] += 1
            st[1] += len(X)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)} files, {len(ys)} rows", flush=True)
    print("--- rows by source ---")
    for src in sorted(by_src):
        fu, rr = by_src[src]
        print(f"  {src:22s} {fu:5d} games  {rr:7d} rows")
    X = np.asarray(Xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int8)
    ep = np.asarray(eps, dtype=np.int32)
    hc = np.asarray(hcs, dtype=np.float32)
    out = os.path.join(REPO, "train", "data", "episodes.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, X=X, y=y, ep=ep, hc=hc)
    print(
        f"episodes used: {n_used}, rows: {len(y)}, "
        f"winrate {y.mean():.3f}, features {X.shape[1]} -> {out}"
    )


if __name__ == "__main__":
    main()
