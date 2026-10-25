"""Leak-free comparison of candidate nets vs the shipped v5 net.

The train/train_net.py holdout contains episodes the SHIPPED net trained on
(the episode split changed when new data arrived), which biases the old-vs-new
comparison in the shipped net's favor. Here we restrict to data that is fresh
since the last training cycle -- daily/2026-07-04, v5games, and seed-3
self-play -- intersected with the new nets' holdout, so every row is unseen
by ALL nets being compared.
"""

import glob
import os

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "train", "data")
HOLDOUT = 0.15
SEED = 7

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

# data the SHIPPED v5 net never trained on (cycle-1 stopped at 07-04 + v5games)
FRESH_TOKENS = ("2026-07-05", "v51games", "v52games", "v5recent")


def split_by_episode(ep: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(ep)
    rng.shuffle(uniq)
    n_hold = max(int(len(uniq) * HOLDOUT), 1)
    hold = set(uniq[:n_hold].tolist())
    mask = np.fromiter((e in hold for e in ep), dtype=bool, count=len(ep))
    return ~mask, mask


def auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _match_dim(z: np.lib.npyio.NpzFile, Xraw: np.ndarray) -> np.ndarray:
    """Map the 101-dim cycle-2 feature matrix onto a net's expected input.

    The cycle-2 base features are the original 85 followed by the 15 archetype
    features, then hc last.  A legacy 86-dim net (shipped v5) consumes the
    original 85 + hc, i.e. columns [:85] plus the final hc column.
    """
    want = z["mean"].shape[0]
    if want == Xraw.shape[1]:
        return Xraw
    if want == 86 and Xraw.shape[1] >= 86:
        return np.hstack([Xraw[:, :85], Xraw[:, -1:]])
    raise ValueError(f"cannot map {Xraw.shape[1]}-dim features to {want}-dim net")


def npz_predict(path: str, Xraw: np.ndarray) -> np.ndarray:
    z = np.load(path)
    Xraw = _match_dim(z, Xraw)
    x = (Xraw.astype(np.float64) - z["mean"]) / z["std"]
    i = 1
    while f"W{i + 1}" in z:
        x = np.maximum(x @ z[f"W{i}"].T + z[f"b{i}"], 0.0)
        i += 1
    logits = (x @ z[f"W{i}"].T + z[f"b{i}"]).ravel()
    return 1.0 / (1.0 + np.exp(-logits))


def per_episode_aucs(
    ep: np.ndarray, y: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    by = {}
    for i, e in enumerate(ep):
        by.setdefault(e, []).append(i)
    keys, vals = [], []
    for e, idx in by.items():
        idx = np.asarray(idx)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        keys.append(e)
        vals.append(auc(yy, scores[idx]))
    return np.asarray(keys), np.asarray(vals)


NETS = {
    "shipped_v5": os.path.join(REPO, "train", "value_net_v5.npz"),
    "cand_A_64x32": os.path.join(REPO, "train", "candidate_a.npz"),
    "cand_B_128x64x32": os.path.join(REPO, "train", "candidate_b.npz"),
    "cand_C_wep0.7": os.path.join(REPO, "train", "candidate_c.npz"),
    "cand_D_wep0.3": os.path.join(REPO, "train", "candidate_d.npz"),
}
# shipped_v5 has a different input dim (86) than the cycle-2 candidates (101);
# skip nets whose weight shapes don't match the feature matrix being scored.
CANDS = ("cand_A_64x32", "cand_B_128x64x32", "cand_C_wep0.7", "cand_D_wep0.3")


def report(
    tag: str, ep: np.ndarray, y: np.ndarray, X: np.ndarray, hc: np.ndarray, fresh_mask: np.ndarray
) -> None:
    print(
        f"\n===== {tag}: {len(np.unique(ep[fresh_mask]))} fresh episodes, "
        f"{fresh_mask.sum()} rows ====="
    )
    epf, yf, Xf, hcf = ep[fresh_mask], y[fresh_mask], X[fresh_mask], hc[fresh_mask]
    preds = {name: npz_predict(p, Xf) for name, p in NETS.items()}
    preds["handcrafted"] = hcf
    per_ep = {}
    for name, s in preds.items():
        keys, vals = per_episode_aucs(epf, yf, s)
        per_ep[name] = dict(zip(keys.tolist(), vals.tolist()))
        print(
            f"{name:18s} overall auc {auc(yf, s):.4f}  "
            f"per-episode auc {vals.mean():.4f} (n={len(vals)})"
        )
    # paired deltas vs shipped
    for name in ("cand_A_64x32", "cand_B_128x64x32", "cand_C_wep0.7", "cand_D_wep0.3"):
        common = sorted(set(per_ep[name]) & set(per_ep["shipped_v5"]))
        d = np.asarray([per_ep[name][e] - per_ep["shipped_v5"][e] for e in common])
        t = d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12)
        print(
            f"paired {name} minus shipped: {d.mean():+.4f} "
            f"(t={t:.2f}, wins {(d > 0).mean() * 100:.0f}% of {len(d)} eps)"
        )


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ---- episodes
    d = np.load(os.path.join(DATA, "episodes.npz"))
    Xe = np.hstack([d["X"], np.tanh(d["hc"] / 25_000.0)[:, None]]).astype(np.float32)
    ye, epe, hce = d["y"], d["ep"].astype(np.int64), d["hc"]
    tr_e, ho_e = split_by_episode(epe, rng)

    files = []
    for dpath in DIRS:
        files += sorted(glob.glob(os.path.join(dpath, "*.json")))
    fresh_ids = {i for i, p in enumerate(files) if any(tok in p for tok in FRESH_TOKENS)}
    fresh_row = np.fromiter((e in fresh_ids for e in epe), dtype=bool, count=len(epe))
    print(
        f"episodes: {len(files)} files, fresh files {len(fresh_ids)}, "
        f"holdout eps {len(np.unique(epe[ho_e]))}"
    )

    report(
        "FRESH ladder episodes ∩ holdout (unseen by ALL nets)", epe, ye, Xe, hce, ho_e & fresh_row
    )
    report(
        "ALL fresh ladder episodes (shipped: unseen; candidates: 85% seen)",
        epe,
        ye,
        Xe,
        hce,
        fresh_row,
    )

    # ---- selfplay (reproduce the second rng call in training order)
    sp_files = sorted(glob.glob(os.path.join(DATA, "selfplay_*.npz")))
    Xs_, ys_, eps_, hcs_ = [], [], [], []
    seed3 = set()
    for p in sp_files:
        z = np.load(p)
        Xs_.append(z["X"])
        ys_.append(z["y"])
        e = z["ep"].astype(np.int64) + 10**10
        eps_.append(e)
        hcs_.append(z["hc"])
        if p.endswith("_3.npz"):
            seed3.update(np.unique(e).tolist())
    Xs = np.hstack([np.concatenate(Xs_), np.tanh(np.concatenate(hcs_) / 25_000.0)[:, None]]).astype(
        np.float32
    )
    ys = np.concatenate(ys_)
    eps = np.concatenate(eps_)
    hcs = np.concatenate(hcs_)
    tr_s, ho_s = split_by_episode(eps, rng)
    fresh_s = np.fromiter((e in seed3 for e in eps), dtype=bool, count=len(eps))
    report("FRESH seed-3 selfplay ∩ holdout (unseen by ALL nets)", eps, ys, Xs, hcs, ho_s & fresh_s)


if __name__ == "__main__":
    main()
