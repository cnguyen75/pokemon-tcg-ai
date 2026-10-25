"""Phase-1 BC bootstrap: train the AZ joint policy+value net on mined episodes.

Policy head  = cross-entropy to the winner's chosen MAIN option (policy.npz,
               reuses train/extract_policy featurization).
Value head   = BCE to game outcome (episodes.npz, per-state win/loss label).
Torso        = SHARED state encoder (that is the joint-training point).

Held out BY EPISODE independently in each dataset (no leakage). Reports policy
top-1/top-3 (target: ~match the earlier bolt-on policy net's 0.336/0.659) and
value AUC, plus param count and inference ms/forward.

  uv run --python 3.13 --with torch --with numpy --with scikit-learn \
      python train/train_az.py [--epochs N] [--out agent_az/az_net.pt]

Chunk-friendly: prints every epoch, writes best-by-policy-top1 checkpoint.
"""

import argparse
import os
import sys
import time

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agent_az"))
sys.path.insert(0, os.path.join(REPO, "agent2"))

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


def by_episode_split(
    ep: np.ndarray, frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    uep = np.unique(ep)
    rng.shuffle(uep)
    n_te = max(1, int(frac * len(uep)))
    te = set(uep[:n_te].tolist())
    is_te = np.array([e in te for e in ep])
    return np.where(~is_te)[0], np.where(is_te)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dmodel", type=int, default=256)
    ap.add_argument("--nblocks", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", default=os.path.join(REPO, "agent_az", "az_net.pt"))
    args = ap.parse_args()

    from az_net import D_OPT, D_STATE, AZNet, param_count
    from sklearn.metrics import roc_auc_score
    import torch
    import torch.nn.functional as F

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # ---------------- POLICY data (policy.npz) ----------------
    dp = np.load(os.path.join(REPO, "train", "data", "policy.npz"))
    X_opt, X_state, offs, chosen, ep_p, dtype = (
        dp["X_opt"],
        dp["X_state"],
        dp["offs"],
        dp["chosen"],
        dp["ep"],
        dp["dtype_"],
    )
    n_dec = len(chosen)
    Dopt = X_opt.shape[1]
    Dstate = X_state.shape[1]
    assert Dopt == D_OPT and Dstate == D_STATE, (Dopt, Dstate)

    sizes = np.array([offs[i + 1] - offs[i] for i in range(n_dec)])
    M = int(sizes.max())
    tr_dec, te_dec = by_episode_split(ep_p, seed=0)
    print(
        f"[policy] decisions {n_dec} (tr {len(tr_dec)} te {len(te_dec)}), "
        f"D_opt {Dopt} D_state {Dstate}, max opts {M}",
        flush=True,
    )

    # standardize state + option feats on TRAIN
    tr_opt_mask = np.zeros(len(X_opt), dtype=bool)
    for di in tr_dec:
        tr_opt_mask[offs[di] : offs[di + 1]] = True
    o_mu, o_sd = X_opt[tr_opt_mask].mean(0), X_opt[tr_opt_mask].std(0) + 1e-6
    Xopt_n = ((X_opt - o_mu) / o_sd).astype(np.float32)
    # state normalization is computed on the VALUE dataset below (larger, more
    # representative of the states the shared torso must judge); both heads then
    # see one input distribution.

    def build_pol(dec_ids: np.ndarray) -> tuple[torch.Tensor, ...]:
        N = len(dec_ids)
        Xo = np.zeros((N, M, Dopt), dtype=np.float32)
        Xs = np.zeros((N, Dstate), dtype=np.float32)
        mask = np.zeros((N, M), dtype=np.float32)
        ch = np.zeros(N, dtype=np.int64)
        for r, di in enumerate(dec_ids):
            a, b = offs[di], offs[di + 1]
            m = b - a
            Xo[r, :m] = Xopt_n[a:b]
            Xs[r] = Xstate_n[di]
            mask[r, :m] = 1.0
            ch[r] = chosen[di]
        return (torch.tensor(Xs), torch.tensor(Xo), torch.tensor(mask), torch.tensor(ch))

    # ---------------- VALUE data (episodes.npz) ----------------
    dv = np.load(os.path.join(REPO, "train", "data", "episodes.npz"))
    Xv, yv, ep_v, hcv = dv["X"], dv["y"], dv["ep"], dv["hc"]
    # value net input = 100 base feats + hc_tanh (matches D_STATE=101)
    hc_tanh = np.tanh(hcv / 25_000.0).astype(np.float32)  # vf.HC_SCALE
    Xv_full = np.concatenate([Xv, hc_tanh[:, None]], axis=1).astype(np.float32)
    assert Xv_full.shape[1] == D_STATE, Xv_full.shape
    trv, tev = by_episode_split(ep_v, seed=0)
    # State normalization from the VALUE train set (larger, broader coverage);
    # the shared torso and both heads use these same stats.
    s_mu = Xv_full[trv].mean(0)
    s_sd = Xv_full[trv].std(0) + 1e-6
    Xstate_n = ((X_state - s_mu) / s_sd).astype(np.float32)

    Xs_tr, Xo_tr, Mk_tr, Ch_tr = build_pol(tr_dec)
    Xs_te, Xo_te, Mk_te, Ch_te = build_pol(te_dec)
    for t in (Xs_tr, Xo_tr, Mk_tr, Ch_tr):
        t.data = t.data.to(dev)
    Xs_te, Xo_te, Mk_te = Xs_te.to(dev), Xo_te.to(dev), Mk_te.to(dev)

    Xv_n = ((Xv_full - s_mu) / s_sd).astype(np.float32)
    Xv_tr = torch.tensor(Xv_n[trv], device=dev)
    yv_tr = torch.tensor(yv[trv].astype(np.float32), device=dev)
    Xv_te = torch.tensor(Xv_n[tev], device=dev)
    yv_te = yv[tev].astype(np.float32)
    print(
        f"[value]  states {len(yv)} (tr {len(trv)} te {len(tev)}), base y mean {yv.mean():.3f}",
        flush=True,
    )

    # ---------------- model ----------------
    net = AZNet(D_STATE, D_OPT, d_model=args.dmodel, n_blocks=args.nblocks).to(dev)
    print(
        f"[net] d_model {args.dmodel} n_blocks {args.nblocks}, params {param_count(net):,}",
        flush=True,
    )
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)

    def eval_policy() -> tuple[float, float]:
        net.eval()
        with torch.no_grad():
            logits = net.policy_logits_from_h(net.encode(Xs_te), Xo_te, Mk_te)
            order = torch.argsort(logits, dim=1, descending=True)
            C = Ch_te.to(dev)
            t1 = (order[:, 0] == C).float().mean().item()
            t3 = (order[:, :3] == C[:, None]).any(1).float().mean().item()
        return t1, t3

    def eval_value() -> tuple[float, float]:
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net.value_logit(Xv_te)).cpu().numpy()
        try:
            auc = roc_auc_score(yv_te, p)
        except Exception:
            auc = float("nan")
        acc = ((p > 0.5).astype(np.float32) == yv_te).mean()
        return auc, acc

    Np = len(tr_dec)
    Nv = len(trv)
    idx_p = np.arange(Np)
    idx_v = np.arange(Nv)
    B = args.batch
    rng = np.random.RandomState(0)
    best_t1, best_state = 0.0, None
    LAMBDA_V = 1.0  # value-loss weight

    for epoch in range(args.epochs):
        net.train()
        rng.shuffle(idx_p)
        rng.shuffle(idx_v)
        # iterate over policy batches; draw a value minibatch alongside each
        pol_loss = val_loss = 0.0
        nb = 0
        n_vbatch = max(1, Nv // B)
        for bi, k in enumerate(range(0, Np, B)):
            bp = idx_p[k : k + B]
            # value minibatch (cycle through value data, always full-size)
            vk = bi % n_vbatch
            bv = idx_v[vk * B : vk * B + B]
            opt.zero_grad()
            # policy loss (shared torso)
            h = net.encode(Xs_tr[bp])
            logits = net.policy_logits_from_h(h, Xo_tr[bp], Mk_tr[bp])
            logp = torch.log_softmax(logits, dim=1)
            lp = -logp[torch.arange(len(bp), device=dev), Ch_tr[bp]].mean()
            # value loss (shared torso, separate states)
            vlogit = net.value_logit(Xv_tr[bv])
            lv = F.binary_cross_entropy_with_logits(vlogit, yv_tr[bv])
            loss = lp + LAMBDA_V * lv
            loss.backward()
            opt.step()
            pol_loss += lp.item()
            val_loss += lv.item()
            nb += 1
        t1, t3 = eval_policy()
        auc, vacc = eval_value()
        print(
            f"ep {epoch + 1:2d} pol_loss {pol_loss / nb:.4f} val_loss "
            f"{val_loss / nb:.4f} | test top1 {t1:.4f} top3 {t3:.4f} | "
            f"val AUC {auc:.4f} acc {vacc:.3f}",
            flush=True,
        )
        if t1 > best_t1:
            best_t1 = t1
            best_state = {k2: v.detach().cpu().clone() for k2, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    net.to(dev)

    t1, t3 = eval_policy()
    auc, vacc = eval_value()

    # per-decision-type policy breakdown on held-out
    net.eval()
    with torch.no_grad():
        norder = (
            torch.argsort(
                net.policy_logits_from_h(net.encode(Xs_te), Xo_te, Mk_te), dim=1, descending=True
            )
            .cpu()
            .numpy()
        )
    from collections import defaultdict

    per = defaultdict(lambda: [0, 0])
    for r, di in enumerate(te_dec):
        ch = int(chosen[di])
        dt = int(dtype[di])
        per[dt][0] += 1
        per[dt][1] += int(norder[r, 0] == ch)

    # inference cost: single-leaf forward (what MCTS pays per node)
    med = int(np.median(sizes[te_dec]))
    xs1 = Xs_te[0]
    xo1 = Xo_te[0, :med]
    for _ in range(50):
        net.eval_leaf(xs1, xo1)
    t0 = time.time()
    REP = 2000
    for _ in range(REP):
        net.eval_leaf(xs1, xo1)
    ms = (time.time() - t0) / REP * 1000

    print("\n" + "=" * 66)
    print("AZ BC BOOTSTRAP  (by-episode holdout)")
    print("=" * 66)
    print(f"  params            {param_count(net):,}  (d_model {args.dmodel})")
    print(f"  POLICY  top-1 {t1:.4f}  top-3 {t3:.4f}   (bolt-on ref 0.336/0.659)")
    print(f"  VALUE   AUC   {auc:.4f}  acc {vacc:.3f}")
    print(f"  leaf inference    {ms:.4f} ms/forward ({med} opts, {dev})")
    print("\n  --- policy top-1 by chosen-option type ---")
    for dt in sorted(per, key=lambda k: -per[k][0]):
        n_, c_ = per[dt]
        name = OPT_TYPE_NAMES.get(dt, str(dt))
        print(f"    {name:10s} {n_:6d}  top1 {c_ / max(n_, 1):.3f}")

    # save checkpoint + normalization stats (inference needs them)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "d_model": args.dmodel,
            "n_blocks": args.nblocks,
            "d_state": D_STATE,
            "d_opt": D_OPT,
            "s_mu": s_mu.astype(np.float32),
            "s_sd": s_sd.astype(np.float32),
            "o_mu": o_mu.astype(np.float32),
            "o_sd": o_sd.astype(np.float32),
            "policy_top1": t1,
            "policy_top3": t3,
            "value_auc": auc,
        },
        args.out,
    )
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
