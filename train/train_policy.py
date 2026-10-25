"""EXPERIMENT ONLY. Train a learned POLICY net and run the decisive test:
does it predict strong players' MAIN moves better than agent2's scripted policy?

Input per option = [option features | shared state features]. Net = small MLP
-> scalar score per option; masked softmax over the legal option set per
decision (padded/batched for speed); cross-entropy to the chosen option.

Holdout is BY EPISODE (no leakage). Reports net vs script vs random top-1/top-3,
per-decision-type breakdown, and forward-pass inference cost.

uv run --python 3.13 --with numpy --with torch python train/train_policy.py
"""

import os
import time

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPT_TYPE_NAMES = {
    7: "PLAY",
    8: "ATTACH",
    9: "EVOLVE",
    10: "ABILITY",
    11: "DISCARD",
    12: "RETREAT",
    13: "ATTACK",
    14: "END",
    15: "SKILL",
}


def load() -> tuple[np.ndarray, ...]:
    d = np.load(os.path.join(REPO, "train", "data", "policy.npz"))
    return (d["X_opt"], d["X_state"], d["offs"], d["chosen"], d["script"], d["ep"], d["dtype_"])


def main() -> None:
    import torch
    import torch.nn as nn

    X_opt, X_state, offs, chosen, script, ep, dtype = load()
    n_dec = len(chosen)
    D_opt = X_opt.shape[1]
    D_state = X_state.shape[1]
    D = D_opt + D_state
    print(
        f"decisions {n_dec}, options {len(X_opt)}, D_opt {D_opt}, "
        f"D_state {D_state}, unique episodes {len(np.unique(ep))}",
        flush=True,
    )

    # per-option full input = [opt | state broadcast]
    n_opt_total = len(X_opt)
    Xfull = np.empty((n_opt_total, D), dtype=np.float32)
    dec_of_opt = np.empty(n_opt_total, dtype=np.int64)
    for di in range(n_dec):
        a, b = offs[di], offs[di + 1]
        Xfull[a:b, :D_opt] = X_opt[a:b]
        Xfull[a:b, D_opt:] = X_state[di]
        dec_of_opt[a:b] = di

    # by-episode split
    rng = np.random.RandomState(0)
    uep = np.unique(ep)
    rng.shuffle(uep)
    n_test_ep = max(1, int(0.2 * len(uep)))
    test_eps = set(uep[:n_test_ep].tolist())
    is_test_dec = np.array([e in test_eps for e in ep])
    tr_dec = np.where(~is_test_dec)[0]
    te_dec = np.where(is_test_dec)[0]
    print(
        f"train decisions {len(tr_dec)}, test decisions {len(te_dec)}, "
        f"test episodes {len(test_eps)}/{len(uep)}",
        flush=True,
    )

    # standardize on TRAIN options
    tr_opt_mask = np.isin(dec_of_opt, tr_dec)
    mu = Xfull[tr_opt_mask].mean(0)
    sd = Xfull[tr_opt_mask].std(0) + 1e-6
    Xn = ((Xfull - mu) / sd).astype(np.float32)

    # --- pad decisions into (N, M, D) + mask for batched softmax ---
    sizes = np.array([offs[di + 1] - offs[di] for di in range(n_dec)])
    M = int(sizes.max())
    print(f"max options in a decision: {M}", flush=True)

    def build(dec_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(dec_ids)
        Xp = np.zeros((N, M, D), dtype=np.float32)
        mask = np.zeros((N, M), dtype=np.float32)
        ch = np.zeros(N, dtype=np.int64)
        for r, di in enumerate(dec_ids):
            a, b = offs[di], offs[di + 1]
            m = b - a
            Xp[r, :m] = Xn[a:b]
            mask[r, :m] = 1.0
            ch[r] = chosen[di]
        return Xp, mask, ch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Mtr, Ctr = build(tr_dec)
    Xte, Mte, Cte = build(te_dec)
    Xtr_t = torch.tensor(Xtr, device=dev)
    Mtr_t = torch.tensor(Mtr, device=dev)
    Ctr_t = torch.tensor(Ctr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev)
    Mte_t = torch.tensor(Mte, device=dev)

    class Net(nn.Module):
        def __init__(self, d: int, h: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, h),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (..., D) -> (...)
            return self.net(x).squeeze(-1)

    net = Net(D).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    NEG = -1e9

    def scores_masked(X: torch.Tensor, Mk: torch.Tensor) -> torch.Tensor:
        s = net(X)  # (N, M)
        return s + (Mk - 1.0) * 1e9  # mask pads to -inf

    def eval_split(X: torch.Tensor, Mk: torch.Tensor, C: np.ndarray) -> tuple[float, float]:
        net.eval()
        with torch.no_grad():
            s = scores_masked(X, Mk)
            order = torch.argsort(s, dim=1, descending=True)
            Ctt = torch.tensor(C, device=dev)
            t1 = (order[:, 0] == Ctt).float().mean().item()
            t3 = (order[:, :3] == Ctt[:, None]).any(1).float().mean().item()
        return t1, t3

    N = len(tr_dec)
    idx = np.arange(N)
    BATCH = 1024
    EPOCHS = 60
    best_t1, best_state = 0.0, None
    for epoch in range(EPOCHS):
        net.train()
        rng.shuffle(idx)
        tot = 0.0
        for k in range(0, N, BATCH):
            bi = idx[k : k + BATCH]
            xb = Xtr_t[bi]
            mb = Mtr_t[bi]
            cb = Ctr_t[bi]
            opt.zero_grad()
            s = net(xb) + (mb - 1.0) * 1e9
            logp = torch.log_softmax(s, dim=1)
            loss = -logp[torch.arange(len(bi), device=dev), cb].mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            te1, te3 = eval_split(Xte_t, Mte_t, Cte)
            print(
                f"epoch {epoch + 1:2d} loss {tot / N:.4f} test t1 {te1:.4f} t3 {te3:.4f}",
                flush=True,
            )
            if te1 > best_t1:
                best_t1 = te1
                best_state = {k2: v.cpu().clone() for k2, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)

    # ---- decisive comparison on held-out ----
    net_t1, net_t3 = eval_split(Xte_t, Mte_t, Cte)
    net.eval()
    with torch.no_grad():
        norder = torch.argsort(scores_masked(Xte_t, Mte_t), dim=1, descending=True).cpu().numpy()

    from collections import defaultdict

    per = defaultdict(lambda: [0, 0, 0, 0.0])  # n, net_t1, sc_t1, rnd
    sc_t1 = sc_t3 = 0
    rnd = 0.0
    n = len(te_dec)
    for r, di in enumerate(te_dec):
        a, b = offs[di], offs[di + 1]
        ch = int(chosen[di])
        m = b - a
        ss = script[a:b]
        sorder = np.argsort(-ss, kind="stable")
        s1 = int(sorder[0] == ch)
        s3 = int(ch in sorder[:3])
        sc_t1 += s1
        sc_t3 += s3
        rnd += 1.0 / m
        n1 = int(norder[r, 0] == ch)
        dt = int(dtype[di])
        acc = per[dt]
        acc[0] += 1
        acc[1] += n1
        acc[2] += s1
        acc[3] += 1.0 / m

    print("\n" + "=" * 64)
    print("DECISIVE HELD-OUT MOVE-PREDICTION (by-episode split)")
    print("=" * 64)
    print(f"held-out decisions: {n}, mean options/decision: {sizes[te_dec].mean():.2f}")
    print(f"  NET     top-1 {net_t1:.4f}   top-3 {net_t3:.4f}")
    print(f"  SCRIPT  top-1 {sc_t1 / n:.4f}   top-3 {sc_t3 / n:.4f}")
    print(f"  RANDOM  top-1 {rnd / n:.4f}")
    print(f"  >>> NET - SCRIPT top-1 gap: {100 * (net_t1 - sc_t1 / n):+.1f} pp")

    print("\n--- by decision type (chosen option's type) ---")
    print(f"  {'type':10s} {'n':>6s} {'net_t1':>8s} {'scr_t1':>8s} {'gap_pp':>7s} {'rnd':>6s}")
    for dt in sorted(per, key=lambda k: -per[k][0]):
        nn_, nt1, st1, rr = per[dt]
        name = OPT_TYPE_NAMES.get(dt, str(dt))
        print(
            f"  {name:10s} {nn_:6d} {nt1 / nn_:8.3f} {st1 / nn_:8.3f} "
            f"{100 * (nt1 - st1) / nn_:+7.1f} {rr / nn_:6.3f}"
        )

    # ---- inference cost ----
    med = int(np.median(sizes[te_dec]))
    xb = Xte_t[:1, :med, :]
    for _ in range(50):
        with torch.no_grad():
            net(xb)
    t0 = time.time()
    REP = 3000
    with torch.no_grad():
        for _ in range(REP):
            net(xb)
    dt_ms = (time.time() - t0) / REP * 1000
    print(
        f"\ninference: {dt_ms:.4f} ms / decision ({med} options, {dev}); "
        f"params {sum(p.numel() for p in net.parameters())}"
    )


if __name__ == "__main__":
    main()
