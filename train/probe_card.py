"""Robustness probes for the card-representation experiment.

1. Logistic-regression probes (well-regularized linear models can't overfit a
   wide sparse input the way a deep MLP can) on:
     - base only (101)
     - card only (1530)
     - base+card (1631)
   If card-only >> chance and base+card >> base, representation adds signal.
2. Leakage check on the shipped net: evaluate all models on the NEWEST
   episodes (daily 2026-07-06/07 + v6games) which postdate the shipped net's
   training data, vs older episodes. If shipped's edge collapses on fresh
   episodes, its holdout lead was leakage, not skill.
"""

import glob
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agent2"))
sys.path.insert(0, os.path.join(REPO, "train"))
SEED = 7


def auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return 0.5 if n1 == 0 or n0 == 0 else (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def split_by_episode(ep: np.ndarray, frac: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    uniq = np.unique(ep)
    rng.shuffle(uniq)
    hold = set(uniq[: max(int(len(uniq) * frac), 1)].tolist())
    m = np.fromiter((e in hold for e in ep), dtype=bool, count=len(ep))
    return ~m, m


def per_ep(yho: np.ndarray, p: np.ndarray, epho: np.ndarray) -> np.ndarray:
    by = {}
    for i, e in enumerate(epho):
        by.setdefault(e, []).append(i)
    vals = []
    for _e, idx in by.items():
        idx = np.asarray(idx)
        yy = yho[idx]
        if yy.min() != yy.max():
            vals.append(auc(yy, p[idx]))
    return np.asarray(vals)


def main() -> None:
    d = np.load(os.path.join(REPO, "train", "data", "episodes_card.npz"))
    Xb = d["Xbase"].astype(np.float32)
    Xc = d["Xcard"].astype(np.float32)
    y = d["y"].astype(np.int64)
    ep = d["ep"]
    tr, ho = split_by_episode(ep)
    yho, epho = y[ho], ep[ho]

    def norm_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mn = X[tr].mean(0)
        sd = X[tr].std(0)
        sd[sd < 1e-6] = 1.0
        return mn, sd

    def lr(X: np.ndarray, C: float) -> np.ndarray:
        mn, sd = norm_fit(X)
        Xt = (X[tr] - mn) / sd
        Xh = (X[ho] - mn) / sd
        m = LogisticRegression(C=C, max_iter=400, solver="lbfgs")
        m.fit(Xt, y[tr])
        return m.predict_proba(Xh)[:, 1]

    print("=== logistic-regression probes (pooled / per-episode AUC) ===")
    Xbc = np.hstack([Xb, Xc])
    for name, X, C in [
        ("base(101)", Xb, 1.0),
        ("card(1530)", Xc, 0.05),
        ("base+card(1631)", Xbc, 0.05),
    ]:
        p = lr(X, C)
        pe = per_ep(yho, p, epho)
        print(f"  {name:16s} pooled {auc(yho, p):.4f}  per-ep {pe.mean():.4f}")

    print("\n=== shipped-net leakage check ===")
    # map episode index -> source dir (re-derive file ordering from extractor)
    DIRS = glob.glob(os.path.join(REPO, "episodes", "daily", "*")) + [
        os.path.join(REPO, "episodes", dd)
        for dd in (
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
    DIRS = [x for x in DIRS if os.path.isdir(x)]
    files = []
    for dp in DIRS:
        files += sorted(glob.glob(os.path.join(dp, "*.json")))
    # fresh = episodes NOT in old episodes.npz training pool.
    # old episodes.npz was built Jul 7 09:51 from 1110 eps; the shipped net
    # (Jul 8) trained on it. v6games + daily 07-06/07 are the freshest sets.
    fresh_tags = ("v6games", "daily/2026-07-06", "daily/2026-07-07")

    def is_fresh(idx: int) -> bool:
        p = files[idx] if idx < len(files) else ""
        return any(t in p for t in fresh_tags)

    fresh_mask = np.fromiter((is_fresh(e) for e in epho), dtype=bool, count=len(epho))
    print(
        f"  holdout rows: {len(yho)}  fresh(v6/07-06/07): {fresh_mask.sum()}  "
        f"older: {(~fresh_mask).sum()}"
    )

    z = np.load(os.path.join(REPO, "agent2", "value_net.npz"))

    def ship(Xraw: np.ndarray) -> np.ndarray:
        x = (Xraw.astype(np.float64) - z["mean"]) / z["std"]
        i = 1
        while f"W{i + 1}" in z:
            x = np.maximum(x @ z[f"W{i}"].T + z[f"b{i}"], 0.0)
            i += 1
        return 1 / (1 + np.exp(-(x @ z[f"W{i}"].T + z[f"b{i}"]).ravel()))

    p_ship = ship(Xb[ho])
    for label, m in [
        ("ALL holdout", np.ones(len(yho), bool)),
        ("FRESH (unseen by shipped)", fresh_mask),
        ("OLDER (likely trained on)", ~fresh_mask),
    ]:
        if m.sum() < 50:
            continue
        pe = per_ep(yho[m], p_ship[m], epho[m])
        print(
            f"  shipped on {label:28s}: pooled {auc(yho[m], p_ship[m]):.4f}  "
            f"per-ep {pe.mean():.4f} (n_ep={len(pe)})"
        )


if __name__ == "__main__":
    main()
