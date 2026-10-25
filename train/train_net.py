"""Train the leaf value net and export plain-numpy weights.

- Loads train/data/episodes.npz + train/data/selfplay_*.npz
- Splits 15% held-out BY EPISODE (per source), never by position
- Weights episode data and self-play data to equal total weight
- MLP 73 -> 64 -> 32 -> 1 (sigmoid), BCE, Adam
- Reports held-out AUC/logloss vs (i) always-0.5 (ii) handcrafted eval score
- Exports agent2/value_net.npz (mean, std, W1, b1, W2, b2, W3, b3)
- Permutation importance over feature groups on held-out data

Usage: uv run --python 3.13 --with torch --with numpy python train/train_net.py
"""

import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agent2"))
from value_features import NET_FEATURE_NAMES as FEATURE_NAMES  # noqa: E402


DATA = os.path.join(REPO, "train", "data")
OUT = os.environ.get("TRAIN_OUT", os.path.join(REPO, "agent2", "value_net.npz"))
ARCH = [int(x) for x in os.environ.get("TRAIN_ARCH", "64,32").split(",")]
DROPOUT = float(os.environ.get("TRAIN_DROPOUT", "0.15"))
OLD_NET = os.environ.get("TRAIN_OLD_NET", os.path.join(REPO, "train", "value_net_v5.npz"))
HOLDOUT = 0.15
SEED = 7


def load_source(
    paths: list[str], tag_offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xs, ys, eps, hcs = [], [], [], []
    for p in paths:
        d = np.load(p)
        Xs.append(d["X"])
        ys.append(d["y"])
        eps.append(d["ep"].astype(np.int64) + tag_offset)
        hcs.append(d["hc"])
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(eps), np.concatenate(hcs))


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
    # midranks for ties
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


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    Xe, ye, epe, hce = load_source([os.path.join(DATA, "episodes.npz")], 0)
    sp_files = sorted(glob.glob(os.path.join(DATA, "selfplay_*.npz")))
    if sp_files:
        Xs, ys_, eps, hcs = load_source(sp_files, 10**10)
    else:
        # bootstrap pass: episodes only (no self-play yet)
        Xs = np.empty((0, Xe.shape[1]), dtype=np.float32)
        ys_ = np.empty((0,), dtype=Xe.dtype)
        eps = np.empty((0,), dtype=np.int64)
        hcs = np.empty((0,), dtype=np.float32)
    # append the squashed handcrafted eval as the final net feature
    Xe = np.hstack([Xe, np.tanh(hce / 25_000.0)[:, None]]).astype(np.float32)
    Xs = np.hstack([Xs, np.tanh(hcs / 25_000.0)[:, None]]).astype(np.float32)
    print(
        f"episodes: {len(ye)} rows / {len(np.unique(epe))} eps; "
        f"selfplay: {len(ys_)} rows / {len(np.unique(eps))} games"
    )

    tr_e, ho_e = split_by_episode(epe, rng)
    tr_s, ho_s = split_by_episode(eps, rng)

    # carve a validation set (for early stopping) out of the TRAINING episodes
    val_uniq = np.unique(epe[tr_e])
    rng.shuffle(val_uniq)
    val_set = set(val_uniq[: max(int(len(val_uniq) * 0.15), 1)].tolist())
    val_mask = np.fromiter((e in val_set for e in epe), dtype=bool, count=len(epe))
    va_e = tr_e & val_mask
    tr_e = tr_e & ~val_mask

    # source weights: episodes (real top-ladder decks) slightly favored
    W_EP = float(os.environ.get("TRAIN_W_EP", "0.5"))
    w_e = np.full(len(ye), W_EP / tr_e.sum(), dtype=np.float32)
    w_s = np.full(len(ys_), (1 - W_EP) / tr_s.sum(), dtype=np.float32)

    Xtr = np.concatenate([Xe[tr_e], Xs[tr_s]])
    ytr = np.concatenate([ye[tr_e], ys_[tr_s]]).astype(np.float32)
    wtr = np.concatenate([w_e[tr_e], w_s[tr_s]])
    wtr *= len(wtr) / wtr.sum()

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-6] = 1.0

    def norm(X: np.ndarray) -> np.ndarray:
        return (X - mean) / std

    Xtr_n = norm(Xtr)

    dev = "cpu"
    layers, prev = [], Xtr.shape[1]
    for h in ARCH:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(DROPOUT)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    model = nn.Sequential(*layers).to(dev)
    print(
        f"arch {Xtr.shape[1]} -> {' -> '.join(map(str, ARCH))} -> 1, dropout {DROPOUT}, out {OUT}"
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.4)
    lossf = nn.BCEWithLogitsLoss(reduction="none")

    Xt = torch.from_numpy(Xtr_n).float()
    yt = torch.from_numpy(ytr).float()
    wt = torch.from_numpy(wtr).float()
    n = len(yt)
    bs = 8192

    Xho_e, yho_e, hho_e = norm(Xe[ho_e]), ye[ho_e], hce[ho_e]
    Xho_s, yho_s, hho_s = norm(Xs[ho_s]), ys_[ho_s], hcs[ho_s]
    Xva_e, yva_e = norm(Xe[va_e]), ye[va_e]

    def predict(Xn: np.ndarray) -> np.ndarray:
        if len(Xn) == 0:
            return np.empty((0,), dtype=np.float32)
        with torch.no_grad():
            out = []
            for i in range(0, len(Xn), 65536):
                out.append(
                    torch.sigmoid(model(torch.from_numpy(Xn[i : i + 65536]).float())).numpy()
                )
        return np.concatenate(out).ravel()

    import copy

    best_val, best_state, best_ep = -1.0, None, -1
    for epoch in range(15):
        perm = torch.randperm(n)
        tot = 0.0
        model.train()
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            out = model(Xt[idx]).squeeze(-1)
            loss = (lossf(out, yt[idx]) * wt[idx]).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        pv = predict(Xva_e)
        va = auc(yva_e, pv)
        if va > best_val:
            best_val, best_ep = va, epoch
            best_state = copy.deepcopy(model.state_dict())
        pe = predict(Xho_e)
        ps = predict(Xho_s)
        print(
            f"epoch {epoch}: train {tot / n:.4f} val-ep auc {va:.4f} | "
            f"ho-episodes auc {auc(yho_e, pe):.4f} ll {logloss(yho_e, pe):.4f} | "
            f"ho-selfplay auc {auc(yho_s, ps):.4f} ll {logloss(yho_s, ps):.4f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    model.eval()
    print(f"selected epoch {best_ep} (val auc {best_val:.4f})")

    pe, ps = predict(Xho_e), predict(Xho_s)
    print("\n=== held-out results ===")
    for name, yh, ph, hh in (("episodes", yho_e, pe, hho_e), ("selfplay", yho_s, ps, hho_s)):
        print(
            f"{name}: net auc {auc(yh, ph):.4f} ll {logloss(yh, ph):.4f} | "
            f"handcrafted auc {auc(yh, hh):.4f} | "
            f"always-0.5 ll {logloss(yh, np.full(len(yh), 0.5)):.4f}"
        )

    # single-subset AUCs swing +-0.04 by episode draw, so gate on the
    # paired delta, not the raw numbers.
    ho_eps = epe[ho_e]
    by_ep = {}
    for i, e in enumerate(ho_eps):
        by_ep.setdefault(e, []).append(i)
    by_ep = {e: np.asarray(v) for e, v in by_ep.items()}
    uniq_ho = np.asarray(list(by_ep))
    rngb = np.random.default_rng(3)
    deltas = []
    for _b in range(200):
        samp = rngb.choice(uniq_ho, size=len(uniq_ho), replace=True)
        idx = np.concatenate([by_ep[e] for e in samp])
        deltas.append(auc(yho_e[idx], pe[idx]) - auc(yho_e[idx], hho_e[idx]))
    deltas = np.asarray(deltas)
    print(
        f"paired bootstrap (ho-episodes) net-minus-handcrafted AUC: "
        f"mean {deltas.mean():+.4f} 95% [{np.percentile(deltas, 2.5):+.4f}, "
        f"{np.percentile(deltas, 97.5):+.4f}]  P(net>hc) {(deltas > 0).mean():.2f}"
    )

    # game, so within-game discrimination is the metric that matters most.
    def per_episode(scores: np.ndarray) -> np.ndarray:
        vals = []
        for e, idx in by_ep.items():
            yy = yho_e[idx]
            if yy.min() == yy.max():
                continue
            vals.append(auc(yy, scores[idx]))
        return np.asarray(vals)

    pa_net = per_episode(pe)
    pa_hc = per_episode(hho_e)
    dd = pa_net - pa_hc
    print(
        f"per-episode AUC (ho-episodes, n={len(pa_net)}): "
        f"net {pa_net.mean():.4f} vs handcrafted {pa_hc.mean():.4f} "
        f"delta {dd.mean():+.4f} (t={dd.mean() / (dd.std() / np.sqrt(len(dd)) + 1e-12):.1f}, "
        f"net wins {(dd > 0).mean() * 100:.0f}% of episodes)"
    )

    def per_episode_s(scores: np.ndarray, ho_eps_s: np.ndarray, yho: np.ndarray) -> np.ndarray:
        by = {}
        for i, e in enumerate(ho_eps_s):
            by.setdefault(e, []).append(i)
        vals = []
        for e, idx in by.items():
            idx = np.asarray(idx)
            yy = yho[idx]
            if yy.min() == yy.max():
                continue
            vals.append(auc(yy, scores[idx]))
        return np.asarray(vals)

    pa_net_s = per_episode_s(ps, eps[ho_s], yho_s)
    pa_hc_s = per_episode_s(hho_s, eps[ho_s], yho_s)
    print(
        f"per-game AUC (ho-selfplay, n={len(pa_net_s)}): "
        f"net {pa_net_s.mean():.4f} vs handcrafted {pa_hc_s.mean():.4f}"
    )

    def npz_predict(path: str, Xraw: np.ndarray) -> np.ndarray:
        z = np.load(path)
        want = z["mean"].shape[0]
        if want != Xraw.shape[1] and want == 86 and Xraw.shape[1] >= 86:
            # legacy 86-dim net: original 85 base features + hc (last col)
            Xraw = np.hstack([Xraw[:, :85], Xraw[:, -1:]])
        x = (Xraw.astype(np.float64) - z["mean"]) / z["std"]
        i = 1
        while f"W{i + 1}" in z:
            x = np.maximum(x @ z[f"W{i}"].T + z[f"b{i}"], 0.0)
            i += 1
        logits = (x @ z[f"W{i}"].T + z[f"b{i}"]).ravel()
        return 1.0 / (1.0 + np.exp(-logits))

    if os.path.exists(OLD_NET):
        po_e = npz_predict(OLD_NET, Xe[ho_e])
        po_s = npz_predict(OLD_NET, Xs[ho_s])
        pa_old = per_episode(po_e)
        pa_old_s = per_episode_s(po_s, eps[ho_s], yho_s)
        print(f"\n=== shipped net ({os.path.basename(OLD_NET)}) on SAME holdout ===")
        print(
            f"ho-episodes: auc {auc(yho_e, po_e):.4f} "
            f"ll {logloss(yho_e, po_e):.4f} | per-episode auc {pa_old.mean():.4f}"
        )
        print(f"ho-selfplay: auc {auc(yho_s, po_s):.4f} | per-game auc {pa_old_s.mean():.4f}")
        dd_no = pa_net - pa_old
        print(
            f"per-episode AUC delta new-minus-shipped: {dd_no.mean():+.4f} "
            f"(t={dd_no.mean() / (dd_no.std() / np.sqrt(len(dd_no)) + 1e-12):.1f}, "
            f"new wins {(dd_no > 0).mean() * 100:.0f}% of episodes)"
        )

    lin = [m for m in model if isinstance(m, nn.Linear)]
    out_d = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    for i, m in enumerate(lin, 1):
        out_d[f"W{i}"] = m.weight.detach().numpy()
        out_d[f"b{i}"] = m.bias.detach().numpy()
    np.savez(OUT, **out_d)
    print(f"exported {OUT} ({len(lin)} linear layers)")

    groups = {
        "prizes": [i for i, f in enumerate(FEATURE_NAMES) if "prize" in f or "win_if_ko" in f],
        "threat_dmg": [
            i for i, f in enumerate(FEATURE_NAMES) if "dmg" in f or "can_ko" in f or "ready" in f
        ],
        "energy": [i for i, f in enumerate(FEATURE_NAMES) if "energy" in f],
        "hp_board": [
            i
            for i, f in enumerate(FEATURE_NAMES)
            if "hp" in f
            or "bench_n" in f
            or "act_exists" in f
            or "evolved" in f
            or "stage" in f
            or "_ex" in f
        ],
        "hand_deck": [
            i for i, f in enumerate(FEATURE_NAMES) if "hand" in f or "deck" in f or "discard" in f
        ],
        "tempo_misc": [
            i
            for i, f in enumerate(FEATURE_NAMES)
            if f in ("turn", "my_turn", "stadium_mine", "stadium_theirs")
            or "cond_" in f
            or "retreat" in f
            or "yield" in f
        ],
        "hc_eval": [i for i, f in enumerate(FEATURE_NAMES) if f == "hc_tanh"],
        # cycle-2 archetype block (the whole point of this retrain)
        "archetype_all": [i for i, f in enumerate(FEATURE_NAMES) if f.startswith("arch_")],
        "arch_onehot": [
            i
            for i, f in enumerate(FEATURE_NAMES)
            if f.startswith("arch_") and not f.startswith("arch_opp_")
        ],
        "arch_capability": [i for i, f in enumerate(FEATURE_NAMES) if f.startswith("arch_opp_")],
    }
    base_auc = auc(yho_e, pe)
    print(f"\n=== permutation importance (ho-episodes, base auc {base_auc:.4f}) ===")
    rng2 = np.random.default_rng(11)
    for gname, idxs in groups.items():
        Xp = Xho_e.copy()
        for i in idxs:
            rng2.shuffle(Xp[:, i])
        pp = predict(Xp)
        print(f"{gname:12s} ({len(idxs):2d} feats): auc drop {base_auc - auc(yho_e, pp):+.4f}")


if __name__ == "__main__":
    main()
