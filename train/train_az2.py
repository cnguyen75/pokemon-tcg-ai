"""Phase-2 expert-iteration retrain: BC episodes + AZ self-play records.

Extends the BC bootstrap (train_az.py). Trains the SAME AZNet jointly on:

  POLICY head:
    - BC decisions   (policy.npz):   cross-entropy to the winner's chosen MAIN
                                     option (one-hot target).
    - self-play      (az_selfplay_*):cross-entropy to the MCTS VISIT DISTRIBUTION
                                     (soft target -- the improved policy). This is
                                     the AZ signal: search-improved move probs, not
                                     the raw BC argmax.
  VALUE head:
    - BC states      (episodes.npz): BCE to replay outcome.
    - self-play      (az_selfplay_*):BCE to the self-play game outcome at the
                                     searched state (search-consistent value).

Self-play policy + value share the SAME rows (each MAIN decision carries both a
visit dist and an outcome), so one self-play minibatch trains both heads on the
distribution the net actually plays.

Held out by game (ep) within the self-play set for a sanity check; BC holdout
top-1/3 + value AUC reported as before. Exports agent_az/az_net.pt.

  uv run --python 3.13 --with torch --with numpy --with scikit-learn \
      python train/train_az2.py --sp-glob 'az_selfplay_r1_*' --epochs 40 \
      --out agent_az/az_net.pt

State/option normalization stats are taken from the SAME sources as train_az.py
(value train set for state, BC train options for option) so the exported net is
inference-compatible with the existing agent_az featurization.
"""

import argparse
import glob
import os
import sys

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
    ep: np.ndarray, frac: float = 0.15, seed: int = 0
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
    ap.add_argument(
        "--warm-start", default="", help="checkpoint to warm-start weights from (same arch)"
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument(
        "--sp-glob", default="az_selfplay_r1_*", help="glob (in train/data) for self-play npz files"
    )
    ap.add_argument(
        "--sp-weight", type=float, default=1.5, help="loss weight on self-play policy+value vs BC"
    )
    ap.add_argument("--out", default=os.path.join(REPO, "agent_az", "az_net.pt"))
    args = ap.parse_args()

    from az_net import D_OPT, D_STATE, AZNet, param_count
    from sklearn.metrics import roc_auc_score
    import torch
    import torch.nn.functional as F

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    DATA = os.path.join(REPO, "train", "data")

    dp = np.load(os.path.join(DATA, "policy.npz"))
    X_opt, X_state, offs, chosen, ep_p, dtype = (
        dp["X_opt"],
        dp["X_state"],
        dp["offs"],
        dp["chosen"],
        dp["ep"],
        dp["dtype_"],
    )
    n_dec = len(chosen)
    Dopt, Dstate = X_opt.shape[1], X_state.shape[1]
    assert Dopt == D_OPT and Dstate == D_STATE, (Dopt, Dstate)
    sizes = np.array([offs[i + 1] - offs[i] for i in range(n_dec)])
    tr_dec, te_dec = by_episode_split(ep_p, frac=0.2, seed=0)

    tr_opt_mask = np.zeros(len(X_opt), dtype=bool)
    for di in tr_dec:
        tr_opt_mask[offs[di] : offs[di + 1]] = True
    o_mu, o_sd = X_opt[tr_opt_mask].mean(0), X_opt[tr_opt_mask].std(0) + 1e-6
    Xopt_n = ((X_opt - o_mu) / o_sd).astype(np.float32)

    dv = np.load(os.path.join(DATA, "episodes.npz"))
    Xv, yv, ep_v, hcv = dv["X"], dv["y"], dv["ep"], dv["hc"]
    # value augmentation: hybrid self-play outcome labels (selfplay_azval_*),
    # same (X, y, ep, hc) format — value data on the strongest agent's own
    _vg = sorted(glob.glob(os.path.join(DATA, "selfplay_azval_*.npz")))
    if _vg:
        _X, _y, _e, _h = [Xv], [yv], [ep_v], [hcv]
        for f in _vg:
            d2 = np.load(f)
            _X.append(d2["X"])
            _y.append(d2["y"])
            _e.append(d2["ep"])
            _h.append(d2["hc"])
        Xv = np.concatenate(_X)
        yv = np.concatenate(_y)
        ep_v = np.concatenate(_e)
        hcv = np.concatenate(_h)
        print(f"[value-aug] {len(_vg)} azval files -> {len(yv)} value rows", flush=True)
    hc_tanh = np.tanh(hcv / 25_000.0).astype(np.float32)
    Xv_full = np.concatenate([Xv, hc_tanh[:, None]], axis=1).astype(np.float32)
    assert Xv_full.shape[1] == D_STATE
    trv, tev = by_episode_split(ep_v, frac=0.2, seed=0)
    s_mu = Xv_full[trv].mean(0)
    s_sd = Xv_full[trv].std(0) + 1e-6
    Xstate_n = ((X_state - s_mu) / s_sd).astype(np.float32)

    sp_files = sorted(glob.glob(os.path.join(DATA, args.sp_glob + ".npz")))
    if not sp_files:
        sp_files = sorted(glob.glob(os.path.join(DATA, args.sp_glob)))
    print(
        f"[selfplay] {len(sp_files)} files: {[os.path.basename(f) for f in sp_files]}", flush=True
    )
    sp_Xs, sp_Xo, sp_offs, sp_pi, sp_y, sp_ep = [], [], [0], [], [], []
    for f in sp_files:
        d = np.load(f)
        if d["Xs"].shape[0] == 0:
            continue
        base = sp_offs[-1]
        sp_Xs.append(d["Xs"])
        sp_Xo.append(d["X_opt"])
        for k in range(len(d["offs"]) - 1):
            sp_offs.append(base + int(d["offs"][k + 1]))
        sp_pi.append(d["pi"])
        sp_y.append(d["y"])
        sp_ep.append(d["ep"])
    sp_Xs = np.concatenate(sp_Xs, 0) if sp_Xs else np.zeros((0, D_STATE), np.float32)
    sp_Xo = np.concatenate(sp_Xo, 0) if sp_Xo else np.zeros((0, D_OPT), np.float32)
    sp_offs = np.asarray(sp_offs, dtype=np.int64)
    sp_pi = np.concatenate(sp_pi, 0) if sp_pi else np.zeros(0, np.float32)
    sp_y = np.concatenate(sp_y, 0).astype(np.float32) if sp_y else np.zeros(0, np.float32)
    sp_ep = np.concatenate(sp_ep, 0) if sp_ep else np.zeros(0, np.int64)
    n_sp = len(sp_ep)
    print(
        f"[selfplay] {n_sp} MAIN decisions, {sp_Xo.shape[0]} option rows, "
        f"value y-mean {sp_y.mean() if n_sp else 0:.3f}, "
        f"{len(np.unique(sp_ep))} games",
        flush=True,
    )

    # standardize self-play feats with the SAME stats as inference/BC
    sp_Xs_n = ((sp_Xs - s_mu) / s_sd).astype(np.float32)
    sp_Xo_n = ((sp_Xo - o_mu) / o_sd).astype(np.float32)
    sp_sizes = np.diff(sp_offs)
    M = int(max(sizes.max(), sp_sizes.max() if n_sp else 0))

    sp_tr, sp_te = (
        by_episode_split(sp_ep, frac=0.12, seed=0) if n_sp else (np.zeros(0, int), np.zeros(0, int))
    )

    def build_bc(dec_ids: np.ndarray) -> tuple[torch.Tensor, ...]:
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

    def build_sp(dec_ids: np.ndarray) -> tuple[torch.Tensor, ...]:
        N = len(dec_ids)
        Xo = np.zeros((N, M, Dopt), dtype=np.float32)
        Xs = np.zeros((N, Dstate), dtype=np.float32)
        mask = np.zeros((N, M), dtype=np.float32)
        pit = np.zeros((N, M), dtype=np.float32)
        yv_ = np.zeros(N, dtype=np.float32)
        for r, di in enumerate(dec_ids):
            a, b = sp_offs[di], sp_offs[di + 1]
            m = b - a
            Xo[r, :m] = sp_Xo_n[a:b]
            Xs[r] = sp_Xs_n[di]
            mask[r, :m] = 1.0
            pit[r, :m] = sp_pi[a:b]
            yv_[r] = sp_y[di]
        return (
            torch.tensor(Xs),
            torch.tensor(Xo),
            torch.tensor(mask),
            torch.tensor(pit),
            torch.tensor(yv_),
        )

    Xs_tr, Xo_tr, Mk_tr, Ch_tr = build_bc(tr_dec)
    Xs_te, Xo_te, Mk_te, Ch_te = build_bc(te_dec)
    Xv_n = ((Xv_full - s_mu) / s_sd).astype(np.float32)
    Xv_tr = torch.tensor(Xv_n[trv])
    yv_tr = torch.tensor(yv[trv].astype(np.float32))
    Xv_te = torch.tensor(Xv_n[tev])
    yv_te = yv[tev].astype(np.float32)

    if n_sp:
        sXs_tr, sXo_tr, sMk_tr, sPi_tr, sY_tr = build_sp(sp_tr)
        sXs_te, sXo_te, sMk_te, sPi_te, sY_te = build_sp(sp_te)

    for t in (Xs_tr, Xo_tr, Mk_tr, Ch_tr, Xv_tr, yv_tr):
        t.data = t.data.to(dev)
    Xs_te, Xo_te, Mk_te = Xs_te.to(dev), Xo_te.to(dev), Mk_te.to(dev)
    Xv_te = Xv_te.to(dev)
    if n_sp:
        for t in (sXs_tr, sXo_tr, sMk_tr, sPi_tr, sY_tr):
            t.data = t.data.to(dev)
        sXs_te, sXo_te, sMk_te = sXs_te.to(dev), sXo_te.to(dev), sMk_te.to(dev)
        sPi_te = sPi_te.to(dev)
        sY_te_np = sY_te.cpu().numpy()

    print(
        f"[policy-bc] {len(tr_dec)} tr {len(te_dec)} te | "
        f"[value-bc] {len(trv)} tr {len(tev)} te | "
        f"[selfplay] {len(sp_tr)} tr {len(sp_te)} te | max opts {M}",
        flush=True,
    )

    net = AZNet(D_STATE, D_OPT, d_model=args.dmodel, n_blocks=args.nblocks).to(dev)
    print(
        f"[net] d_model {args.dmodel} n_blocks {args.nblocks}, params {param_count(net):,}",
        flush=True,
    )
    if args.warm_start and os.path.exists(args.warm_start):
        wck = torch.load(args.warm_start, map_location=dev, weights_only=False)
        net.load_state_dict(wck["state_dict"])
        print(f"[warm-start] loaded weights from {args.warm_start}", flush=True)
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

    def eval_sp() -> tuple[float, float, float]:
        if not n_sp:
            return float("nan"), float("nan"), float("nan")
        net.eval()
        with torch.no_grad():
            lg = net.policy_logits_from_h(net.encode(sXs_te), sXo_te, sMk_te)
            logp = torch.log_softmax(lg, dim=1)
            # KL(target_visit || net) on held-out self-play
            tgt = sPi_te
            kl = (tgt * (torch.log(tgt + 1e-9) - logp)).sum(1)
            kl = kl[torch.isfinite(kl)].mean().item()
            # top-1 agreement: net argmax == visit argmax
            agree = (lg.argmax(1) == tgt.argmax(1)).float().mean().item()
            pv = torch.sigmoid(net.value_logit(sXs_te)).cpu().numpy()
        try:
            vauc = roc_auc_score(sY_te_np, pv)
        except Exception:
            vauc = float("nan")
        return kl, agree, vauc

    Np, Nv = len(tr_dec), len(trv)
    idx_p, idx_v = np.arange(Np), np.arange(Nv)
    Nsp = len(sp_tr)
    idx_sp = np.arange(Nsp)
    B = args.batch
    rng = np.random.RandomState(0)
    best_score, best_state = -1.0, None
    LAMBDA_V = 1.0
    SPW = args.sp_weight

    for epoch in range(args.epochs):
        net.train()
        rng.shuffle(idx_p)
        rng.shuffle(idx_v)
        if Nsp:
            rng.shuffle(idx_sp)
        pol_loss = val_loss = sp_loss = 0.0
        nb = 0
        n_vbatch = max(1, Nv // B)
        n_spbatch = max(1, Nsp // B) if Nsp else 1
        for bi, k in enumerate(range(0, Np, B)):
            bp = idx_p[k : k + B]
            vk = bi % n_vbatch
            bv = idx_v[vk * B : vk * B + B]
            opt.zero_grad()
            h = net.encode(Xs_tr[bp])
            logits = net.policy_logits_from_h(h, Xo_tr[bp], Mk_tr[bp])
            logp = torch.log_softmax(logits, dim=1)
            lp = -logp[torch.arange(len(bp), device=dev), Ch_tr[bp]].mean()
            vlogit = net.value_logit(Xv_tr[bv])
            lv = F.binary_cross_entropy_with_logits(vlogit, yv_tr[bv])
            loss = lp + LAMBDA_V * lv
            # self-play soft-CE policy + value (shares rows)
            lsp = torch.tensor(0.0, device=dev)
            if Nsp:
                sk = bi % n_spbatch
                bs = idx_sp[sk * B : sk * B + B]
                if len(bs) == 0:
                    bs = idx_sp[:B]
                hs = net.encode(sXs_tr[bs])
                slog = net.policy_logits_from_h(hs, sXo_tr[bs], sMk_tr[bs])
                slogp = torch.log_softmax(slog, dim=1)
                tgt = sPi_tr[bs]
                sp_pol = -(tgt * slogp).sum(1).mean()  # soft CE
                sv = net.value_from_h(hs)
                sp_val = F.binary_cross_entropy_with_logits(sv, sY_tr[bs])
                lsp = sp_pol + LAMBDA_V * sp_val
                loss = loss + SPW * lsp
            loss.backward()
            opt.step()
            pol_loss += lp.item()
            val_loss += lv.item()
            sp_loss += float(lsp)
            nb += 1
        t1, t3 = eval_policy()
        auc, vacc = eval_value()
        kl, agree, spauc = eval_sp()
        # model-selection score: prioritize self-play fit if present, else BC top1
        score = (agree if n_sp else t1) + 0.5 * (spauc if n_sp else auc)
        print(
            f"ep {epoch + 1:2d} bc_pol {pol_loss / nb:.3f} bc_val {val_loss / nb:.3f} "
            f"sp {sp_loss / nb:.3f} | BC t1 {t1:.3f} t3 {t3:.3f} vAUC {auc:.3f} "
            f"| SP KL {kl:.3f} agree {agree:.3f} vAUC {spauc:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {k2: v.detach().cpu().clone() for k2, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    net.to(dev)
    t1, t3 = eval_policy()
    auc, vacc = eval_value()
    kl, agree, spauc = eval_sp()

    print("\n" + "=" * 66)
    print("AZ EXPERT-ITERATION RETRAIN  (BC + self-play)")
    print("=" * 66)
    print(f"  params            {param_count(net):,}  (d_model {args.dmodel})")
    print(f"  BC   POLICY top-1 {t1:.4f}  top-3 {t3:.4f}   (BC-boot ref 0.340/0.663)")
    print(f"  BC   VALUE  AUC   {auc:.4f}  acc {vacc:.3f}")
    print(f"  SP   policy agree {agree:.4f}  KL {kl:.4f}  value AUC {spauc:.4f}")

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
            "sp_agree": agree,
            "sp_value_auc": spauc,
        },
        args.out,
    )
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
