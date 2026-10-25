"""EXPERIMENT: does a card-level representation beat the saturated 100-feat net?

Trains on train/data/episodes_card.npz (Xbase 101 = 100 base + hc, Xcard 1530).
Holds out 15% BY EPISODE (seed 7). Compares, on the SAME holdout:
  (a) card net    = MLP on [Xbase | Xcard]  (dim 1631)
  (b) base net    = MLP on  Xbase          (dim 101)  [control: same data/split]
  (c) shipped net = agent2/value_net.npz on Xbase     (the saturated 100-feat net)
  (d) handcrafted eval
Metrics: pooled AUC + per-episode AUC (paired t-stat, win%), permutation
importance of card-feature groups, inference cost.
"""

from collections.abc import Callable
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "train", "data")
SEED = 7
HOLDOUT = 0.15
ARCH = [int(x) for x in os.environ.get("CARD_ARCH", "512,256,128").split(",")]
DROPOUT = float(os.environ.get("CARD_DROPOUT", "0.20"))
EPOCHS = int(os.environ.get("CARD_EPOCHS", "20"))
SHIPPED = os.path.join(REPO, "agent2", "value_net.npz")


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


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def split_by_episode(
    ep: np.ndarray, rng: np.random.Generator, frac: float
) -> tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(ep)
    rng.shuffle(uniq)
    n_hold = max(int(len(uniq) * frac), 1)
    hold = set(uniq[:n_hold].tolist())
    mask = np.fromiter((e in hold for e in ep), dtype=bool, count=len(ep))
    return ~mask, mask


def shipped_predict(path: str, Xraw: np.ndarray) -> np.ndarray:
    z = np.load(path)
    x = (Xraw.astype(np.float64) - z["mean"]) / z["std"]
    i = 1
    while f"W{i + 1}" in z:
        x = np.maximum(x @ z[f"W{i}"].T + z[f"b{i}"], 0.0)
        i += 1
    logits = (x @ z[f"W{i}"].T + z[f"b{i}"]).ravel()
    return 1.0 / (1.0 + np.exp(-logits))


def build_mlp(indim: int, arch: list[int], dropout: float) -> nn.Sequential:
    layers, prev = [], indim
    for h in arch:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train_mlp(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    arch: list[int],
    dropout: float,
    epochs: int,
    tag: str,
    wd: float = 3e-4,
) -> tuple[nn.Sequential, np.ndarray, np.ndarray, Callable[..., np.ndarray]]:
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-6] = 1.0
    Xtr_n = ((Xtr - mean) / std).astype(np.float32)
    Xva_n = ((Xva - mean) / std).astype(np.float32)
    model = build_mlp(Xtr.shape[1], arch, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.4)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(Xtr_n)
    yt = torch.from_numpy(ytr.astype(np.float32))
    n = len(yt)
    bs = 8192

    def predict(Xn: np.ndarray, m: nn.Module | None = None) -> np.ndarray:
        m = m or model
        m.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(Xn), 65536):
                out.append(
                    torch.sigmoid(m(torch.from_numpy(Xn[i : i + 65536])).squeeze(-1)).numpy()
                )
        return np.concatenate(out) if out else np.empty(0)

    best_val, best_state, best_ep = -1.0, None, -1
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            out = model(Xt[idx]).squeeze(-1)
            loss = lossf(out, yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        pv = predict(Xva_n)
        va = auc(yva, pv)
        if va > best_val:
            best_val, best_ep = va, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f"  [{tag}] epoch {epoch}: train {tot / n:.4f} val-auc {va:.4f}", flush=True)
    model.load_state_dict(best_state)
    print(f"  [{tag}] best epoch {best_ep} val-auc {best_val:.4f}", flush=True)
    return model, mean, std, predict


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    d = np.load(os.path.join(DATA, "episodes_card.npz"))
    Xbase = d["Xbase"].astype(np.float32)
    Xcard = d["Xcard"].astype(np.float32)
    y = d["y"].astype(np.int64)
    ep = d["ep"]
    hc = d["hc"]
    Xfull = np.hstack([Xbase, Xcard])
    print(
        f"rows {len(y)}  base {Xbase.shape[1]}  card {Xcard.shape[1]}  "
        f"full {Xfull.shape[1]}  eps {len(np.unique(ep))}  wr {y.mean():.3f}"
    )

    tr, ho = split_by_episode(ep, rng, HOLDOUT)
    # validation carved from training episodes for early stopping
    tr_eps = np.unique(ep[tr])
    rng.shuffle(tr_eps)
    val_set = set(tr_eps[: max(int(len(tr_eps) * 0.15), 1)].tolist())
    va = tr & np.fromiter((e in val_set for e in ep), dtype=bool, count=len(ep))
    tr = tr & ~va
    print(
        f"train rows {tr.sum()}  val rows {va.sum()}  holdout rows {ho.sum()}  "
        f"holdout eps {len(np.unique(ep[ho]))}"
    )

    yho = y[ho]
    hcho = hc[ho]
    epho = ep[ho]

    def per_ep_auc_mean(p: np.ndarray) -> float:
        vals = []
        seen = {}
        for i, e in enumerate(ep[va]):
            seen.setdefault(e, []).append(i)
        for _e, idx in seen.items():
            idx = np.asarray(idx)
            yy = y[va][idx]
            if yy.min() == yy.max():
                continue
            vals.append(auc(yy, p[idx]))
        return float(np.mean(vals))

    # (a) CARD net: sweep regularization, pick best by per-episode val AUC
    print("\n--- training CARD net (base+card): reg sweep ---")
    best = None
    for arch, dp, wd in [
        (ARCH, DROPOUT, 3e-4),
        ([256, 128], 0.35, 1e-3),
        ([256, 128], 0.5, 3e-3),
        ([128], 0.5, 3e-3),
    ]:
        tag = f"card_{'x'.join(map(str, arch))}_dp{dp}_wd{wd}"
        m, mn, sd, pr = train_mlp(Xfull[tr], y[tr], Xfull[va], y[va], arch, dp, EPOCHS, tag, wd=wd)
        pv = pr(((Xfull[va] - mn) / sd).astype(np.float32))
        vscore = per_ep_auc_mean(pv)
        print(f"  >> {tag}: per-ep VAL auc {vscore:.4f}")
        if best is None or vscore > best[0]:
            best = (vscore, m, mn, sd, pr, tag)
    print(f"  SELECTED card config: {best[5]} (per-ep val {best[0]:.4f})")
    cm, cmean, cstd, cpred = best[1], best[2], best[3], best[4]
    p_card = cpred(((Xfull[ho] - cmean) / cstd).astype(np.float32))

    # (b) BASE net (control: same data & split, base features only). Sweep too.
    print("\n--- training BASE net (control, base only): reg sweep ---")
    bbest = None
    for arch, dp, _wd in [
        ([64, 32], 0.15, 3e-4),
        ([128, 64], 0.2, 3e-4),
        ([256, 128, 64], 0.25, 3e-4),
    ]:
        tag = f"base_{'x'.join(map(str, arch))}"
        m, mn, sd, pr = train_mlp(Xbase[tr], y[tr], Xbase[va], y[va], arch, dp, EPOCHS, tag)
        pv = pr(((Xbase[va] - mn) / sd).astype(np.float32))
        vscore = per_ep_auc_mean(pv)
        print(f"  >> {tag}: per-ep VAL auc {vscore:.4f}")
        if bbest is None or vscore > bbest[0]:
            bbest = (vscore, m, mn, sd, pr, tag)
    print(f"  SELECTED base config: {bbest[5]} (per-ep val {bbest[0]:.4f})")
    bmean, bstd, bpred = bbest[2], bbest[3], bbest[4]
    p_base = bpred(((Xbase[ho] - bmean) / bstd).astype(np.float32))

    # (c) shipped net, (d) handcrafted
    p_ship = shipped_predict(SHIPPED, Xbase[ho])
    p_hc = hcho

    print("\n=== POOLED holdout metrics ===")
    for name, p in (
        ("card", p_card),
        ("base(ctrl)", p_base),
        ("shipped", p_ship),
        ("handcraft", p_hc),
    ):
        ll = logloss(yho, p) if name != "handcraft" else float("nan")
        print(f"  {name:11s} auc {auc(yho, p):.4f}  logloss {ll:.4f}")

    by = {}
    for i, e in enumerate(epho):
        by.setdefault(e, []).append(i)
    by = {e: np.asarray(v) for e, v in by.items()}

    def per_ep(p: np.ndarray) -> dict[int, float]:
        out = []
        for e, idx in by.items():
            yy = yho[idx]
            if yy.min() == yy.max():
                continue
            out.append((e, auc(yy, p[idx])))
        return dict(out)

    pe_card = per_ep(p_card)
    pe_base = per_ep(p_base)
    pe_ship = per_ep(p_ship)
    pe_hc = per_ep(p_hc)
    common = sorted(set(pe_card) & set(pe_ship) & set(pe_hc) & set(pe_base))
    a_card = np.array([pe_card[e] for e in common])
    a_base = np.array([pe_base[e] for e in common])
    a_ship = np.array([pe_ship[e] for e in common])
    a_hc = np.array([pe_hc[e] for e in common])
    print(f"\n=== PER-EPISODE AUC (n={len(common)} decisive episodes) ===")
    print(f"  card   {a_card.mean():.4f}")
    print(f"  base   {a_base.mean():.4f}")
    print(f"  ship   {a_ship.mean():.4f}")
    print(f"  hc     {a_hc.mean():.4f}")

    def paired(a: np.ndarray, b: np.ndarray, la: str, lb: str) -> None:
        dd = a - b
        t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)) + 1e-12)
        print(
            f"  {la} - {lb}: delta {dd.mean():+.4f}  t={t:.2f}  "
            f"{la} wins {(dd > 0).mean() * 100:.0f}%"
        )

    print("\n=== DECISIVE paired comparisons (per-episode) ===")
    paired(a_card, a_ship, "card", "ship")
    paired(a_card, a_base, "card", "base")
    paired(a_card, a_hc, "card", "hc")
    paired(a_base, a_ship, "base", "ship")

    import card_features as cf
    from cg.api import all_card_data

    CARD = {c.cardId: c for c in all_card_data()}
    VOCAB = cf.load_vocab()
    W = len(VOCAB) + cf.N_OTHER
    zones = cf._ZONES
    Xho_full = ((Xfull[ho] - cmean) / cstd).astype(np.float32)
    base_auc = auc(yho, cpred(Xho_full))
    print(f"\n=== permutation importance (card net, base auc {base_auc:.4f}) ===")
    rng2 = np.random.default_rng(11)
    off = Xbase.shape[1]  # card block starts here in Xfull
    for label, cols in (
        [("ALL_CARD", list(range(off, Xfull.shape[1])))]
        + [
            (f"zone_{z}", list(range(off + zi * W, off + (zi + 1) * W)))
            for zi, z in enumerate(zones)
        ]
        + [("ALL_BASE", list(range(0, off)))]
    ):
        Xp = Xho_full.copy()
        for c in cols:
            rng2.shuffle(Xp[:, c])
        drop = base_auc - auc(yho, cpred(Xp))
        print(f"  {label:16s} ({len(cols):4d} feats): auc drop {drop:+.4f}")

    print("\n=== inference cost (card_only_feats extractor) ===")
    import glob
    import json

    from cg.api import all_attack

    ATK = {a.attackId: a for a in all_attack()}
    IDX = cf.make_index(VOCAB)
    states = []
    for path in sorted(glob.glob(os.path.join(REPO, "episodes", "v6games", "*.json")))[:5]:
        try:
            with open(path) as fh:
                dd = json.load(fh)
        except Exception:
            continue
        for step in dd.get("steps", []):
            for j, ag in enumerate(step):
                if ag.get("status") != "ACTIVE":
                    continue
                cur = (ag.get("observation") or {}).get("current")
                if cur and cur.get("result", -1) < 0:
                    states.append((cur, j))
            if len(states) > 300:
                break
        if len(states) > 300:
            break
    states = states[:300]
    t0 = time.perf_counter()
    for cur, me in states:
        cf.card_only_feats(cur, me, CARD, VOCAB, IDX)
    dt = (time.perf_counter() - t0) / len(states) * 1000
    print(f"  card_only_feats: {dt:.3f} ms/state (n={len(states)})")
    t0 = time.perf_counter()
    for cur, me in states:
        cf.card_extract(cur, me, CARD, ATK, VOCAB, IDX)
    dt2 = (time.perf_counter() - t0) / len(states) * 1000
    print(f"  full card_extract (base+card): {dt2:.3f} ms/state")
    xb = np.random.randn(1, Xfull.shape[1]).astype(np.float32)
    cm.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(2000):
            cm(torch.from_numpy(xb))
        dt3 = (time.perf_counter() - t0) / 2000 * 1000
    print(f"  card-net forward (torch, 1 row): {dt3:.3f} ms")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "agent2"))
    sys.path.insert(0, os.path.join(REPO, "train"))
    main()
