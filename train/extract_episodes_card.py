"""EXPERIMENT: re-extract replays with base(100)+hc + card-level features.

Writes train/data/episodes_card.npz:
  Xbase float32 [N, 101]  (100 base features + squashed hc, matches shipped net)
  Xcard float16 [N, C]    (card-level count vectors; float16 to save disk)
  y int8, ep int32, hc float32

Kept split (Xbase vs Xcard) so we can:
 - feed the shipped 100-feat net on Xbase[:, :100] (its exact input is
   base100 + hc; shipped net input dim is 101),
 - train the card net on hstack([Xbase, Xcard]).
"""

import glob
import json
import os
import sys

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agent2"))
sys.path.insert(0, os.path.join(REPO, "train"))

import card_features as cf  # noqa: E402
from cg.api import all_attack, all_card_data  # noqa: E402
import value_features as vf  # noqa: E402


CARD = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: a for a in all_attack()}
VOCAB = cf.load_vocab()
IDX = cf.make_index(VOCAB)

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
        "v6games",
    )
]
DIRS = [d for d in DIRS if os.path.isdir(d)]


def parse_file(
    path: str,
) -> tuple[list[list[float]], list[list[float]], list[int], list[float]]:
    try:
        d = json.load(open(path))
    except Exception:
        return [], [], [], []
    if not isinstance(d, dict) or "steps" not in d:
        return [], [], [], []
    rew = d.get("rewards")
    if (
        not isinstance(rew, list)
        or len(rew) != 2
        or rew[0] is None
        or rew[1] is None
        or rew[0] == rew[1]
    ):
        return [], [], [], []
    won = [1 if rew[0] > rew[1] else 0, 1 if rew[1] > rew[0] else 0]
    Xb, Xc, y, hc = [], [], [], []
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
                    base = vf.extract(cur, me, CARD, ATK)
                    base.append(vf.hc_squash(vf.handcrafted_eval(cur, me)))
                    card = cf.card_only_feats(cur, me, CARD, VOCAB, IDX)
                except Exception:
                    continue
                Xb.append(base)
                Xc.append(card)
                y.append(won[me])
                hc.append(vf.handcrafted_eval(cur, me))
    return Xb, Xc, y, hc


def main() -> None:
    files = []
    for dp in DIRS:
        files += sorted(glob.glob(os.path.join(dp, "*.json")))
    print(f"{len(files)} replay files, vocab {len(VOCAB)}", flush=True)
    Xbs, Xcs, ys, eps, hcs = [], [], [], [], []
    n_used = 0
    for i, path in enumerate(files):
        Xb, Xc, y, hc = parse_file(path)
        if Xb:
            n_used += 1
            Xbs.extend(Xb)
            Xcs.extend(Xc)
            ys.extend(y)
            hcs.extend(hc)
            eps.extend([i] * len(Xb))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)} files, {len(ys)} rows", flush=True)
    Xbase = np.asarray(Xbs, dtype=np.float32)
    Xcard = np.asarray(Xcs, dtype=np.float16)
    y = np.asarray(ys, dtype=np.int8)
    ep = np.asarray(eps, dtype=np.int32)
    hc = np.asarray(hcs, dtype=np.float32)
    out = os.path.join(REPO, "train", "data", "episodes_card.npz")
    np.savez_compressed(out, Xbase=Xbase, Xcard=Xcard, y=y, ep=ep, hc=hc)
    print(
        f"episodes used: {n_used}, rows: {len(y)}, winrate {y.mean():.3f}, "
        f"base {Xbase.shape[1]} card {Xcard.shape[1]} -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
