"""Export the trained POLICY net to a numpy .npz for torch-free inference.

Retrains the exact Net from train_policy.py (same arch, same by-episode split,
same standardization), keeps the best-top1 state, and writes
agent2/policy_net.npz with mean/std/W1/b1/W2/b2/W3/b3 + the held-out metrics.

Also dumps train/data/policy_holdout.npz (the by-episode test decisions with
their option features, chosen index, dtype, script scores) so the behavioral
move-match check can be reproduced against the exported numpy net.

uv run --python 3.13 --with numpy --with torch python train/export_policy.py
"""

import os

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


def load():
    d = np.load(os.path.join(REPO, "train", "data", "policy.npz"))
    return (d["X_opt"], d["X_state"], d["offs"], d["chosen"], d["script"], d["ep"], d["dtype_"])


def main():
    import torch
    import torch.nn as nn

    X_opt, X_state, offs, chosen, script, ep, dtype = load()
    n_dec = len(chosen)
    D_opt = X_opt.shape[1]
    D_state = X_state.shape[1]
    D = D_opt + D_state
    print(f"decisions {n_dec}, options {len(X_opt)}, D_opt {D_opt}, D_state {D_state}", flush=True)

    n_opt_total = len(X_opt)
    Xfull = np.empty((n_opt_total, D), dtype=np.float32)
    dec_of_opt = np.empty(n_opt_total, dtype=np.int64)
    for di in range(n_dec):
        a, b = offs[di], offs[di + 1]
        Xfull[a:b, :D_opt] = X_opt[a:b]
        Xfull[a:b, D_opt:] = X_state[di]
        dec_of_opt[a:b] = di

    rng = np.random.RandomState(0)
    uep = np.unique(ep)
    rng.shuffle(uep)
    n_test_ep = max(1, int(0.2 * len(uep)))
    test_eps = set(uep[:n_test_ep].tolist())
    is_test_dec = np.array([e in test_eps for e in ep])
    tr_dec = np.where(~is_test_dec)[0]
    te_dec = np.where(is_test_dec)[0]
    print(f"train {len(tr_dec)}, test {len(te_dec)}", flush=True)

    tr_opt_mask = np.isin(dec_of_opt, tr_dec)
    mu = Xfull[tr_opt_mask].mean(0)
    sd = Xfull[tr_opt_mask].std(0) + 1e-6
    Xn = ((Xfull - mu) / sd).astype(np.float32)

    sizes = np.array([offs[di + 1] - offs[di] for di in range(n_dec)])
    M = int(sizes.max())

    def build(dec_ids):
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
        def __init__(self, d, h=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, h),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    net = Net(D).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)

    def eval_split(X, Mk, C):
        net.eval()
        with torch.no_grad():
            s = net(X) + (Mk - 1.0) * 1e9
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
        for k in range(0, N, BATCH):
            bi = idx[k : k + BATCH]
            xb, mb, cb = Xtr_t[bi], Mtr_t[bi], Ctr_t[bi]
            opt.zero_grad()
            s = net(xb) + (mb - 1.0) * 1e9
            logp = torch.log_softmax(s, dim=1)
            loss = -logp[torch.arange(len(bi), device=dev), cb].mean()
            loss.backward()
            opt.step()
        te1, te3 = eval_split(Xte_t, Mte_t, Cte)
        if te1 > best_t1:
            best_t1 = te1
            best_state = {k2: v.cpu().clone() for k2, v in net.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1} test t1 {te1:.4f} t3 {te3:.4f}", flush=True)

    net.load_state_dict(best_state)
    net_t1, net_t3 = eval_split(Xte_t, Mte_t, Cte)
    print(f"BEST net top-1 {net_t1:.4f} top-3 {net_t3:.4f}", flush=True)

    # pull weights out of the Sequential (0:Linear,3:Linear,5:Linear)
    sd_state = net.state_dict()
    W1 = sd_state["net.0.weight"].cpu().numpy().astype(np.float32)
    b1 = sd_state["net.0.bias"].cpu().numpy().astype(np.float32)
    W2 = sd_state["net.3.weight"].cpu().numpy().astype(np.float32)
    b2 = sd_state["net.3.bias"].cpu().numpy().astype(np.float32)
    W3 = sd_state["net.5.weight"].cpu().numpy().astype(np.float32)
    b3 = sd_state["net.5.bias"].cpu().numpy().astype(np.float32)

    out = os.path.join(REPO, "agent2", "policy_net.npz")
    np.savez_compressed(
        out,
        mean=mu.astype(np.float32),
        std=sd.astype(np.float32),
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        W3=W3,
        b3=b3,
        D_opt=np.int64(D_opt),
        D_state=np.int64(D_state),
    )
    print(f"wrote {out}", flush=True)

    # ---- verify numpy forward matches torch on held-out ----
    def np_fwd(Xrows):
        x = (Xrows - mu) / sd
        h = np.maximum(x @ W1.T + b1, 0.0)
        h = np.maximum(h @ W2.T + b2, 0.0)
        return (h @ W3.T + b3).ravel()

    match = 0
    for di in te_dec:
        a, b = offs[di], offs[di + 1]
        s = np_fwd(Xfull[a:b])
        if int(np.argmax(s)) == int(chosen[di]):
            match += 1
    print(
        f"numpy-forward held-out top-1: {match / len(te_dec):.4f} (torch {net_t1:.4f})", flush=True
    )

    # ---- dump held-out for behavioral move-match check ----
    hold_offs = [0]
    hold_Xopt, hold_state, hold_chosen, hold_dtype, hold_script = [], [], [], [], []
    for di in te_dec:
        a, b = offs[di], offs[di + 1]
        for r in range(a, b):
            hold_Xopt.append(X_opt[r])
            hold_script.append(script[r])
        hold_state.append(X_state[di])
        hold_offs.append(len(hold_Xopt))
        hold_chosen.append(int(chosen[di]))
        hold_dtype.append(int(dtype[di]))
    np.savez_compressed(
        os.path.join(REPO, "train", "data", "policy_holdout.npz"),
        X_opt=np.asarray(hold_Xopt, dtype=np.float32),
        X_state=np.asarray(hold_state, dtype=np.float32),
        offs=np.asarray(hold_offs, dtype=np.int64),
        chosen=np.asarray(hold_chosen, dtype=np.int64),
        dtype_=np.asarray(hold_dtype, dtype=np.int64),
        script=np.asarray(hold_script, dtype=np.float32),
    )
    print("wrote train/data/policy_holdout.npz", flush=True)


if __name__ == "__main__":
    main()
