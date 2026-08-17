"""Behavioral move-match check: does the policy-ON candidate generation match
top-player MAIN moves better than scripts alone, on the by-episode HELD-OUT set?

Uses train/data/policy_holdout.npz (held-out decisions dumped by
export_policy.py: per-option training features, chosen index, dtype, scripted
score) and the exported numpy policy net (agent2/policy_net.npz) via the same
numpy forward the agent uses at inference.

Reports, overall and per decision type:
  SCRIPT  top-1   = argmax scripted score == winner's move
  POLICY  top-1   = argmax policy logit  == winner's move
  UNION coverage  = winner's move is IN the {script top-k} U {policy top-k}
                    candidate set the beam searches (this is what unblocks the
                    beam: it can only pick moves the candidate gen proposes)
for SCRIPT-only cand set vs SCRIPT+POLICY union.
"""
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agent2"))

OPT_TYPE_NAMES = {7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY",
                  11: "DISCARD", 12: "RETREAT", 13: "ATTACK", 14: "END",
                  15: "SKILL"}
# classes the net is allowed to inject (agent default CG_A2_POLICY_CLASSES)
INJECT_TYPES = {7, 8, 12}  # PLAY, ATTACH, RETREAT
SCRIPT_MAX_CANDS = 5       # SEARCH_MAX_CANDS
POLICY_TOPK = int(os.environ.get("TOPK", "2"))

d = np.load(os.path.join(REPO, "train", "data", "policy_holdout.npz"))
X_opt, X_state, offs, chosen, dtype, script = (
    d["X_opt"], d["X_state"], d["offs"], d["chosen"], d["dtype_"], d["script"])

W = np.load(os.path.join(REPO, "agent2", "policy_net.npz"))
mean, std = W["mean"], W["std"]
W1, b1, W2, b2, W3, b3 = W["W1"], W["b1"], W["W2"], W["b2"], W["W3"], W["b3"]
D_opt = int(W["D_opt"])


def policy_logits(a, b, di):
    X = np.concatenate([X_opt[a:b],
                        np.tile(X_state[di], (b - a, 1))], axis=1).astype(np.float32)
    x = (X - mean) / std
    h = np.maximum(x @ W1.T + b1, 0.0)
    h = np.maximum(h @ W2.T + b2, 0.0)
    return (h @ W3.T + b3).ravel()


# per-option type from the one-hot in X_opt[:, :17]
def opt_type(i):
    return int(np.argmax(X_opt[i, :17]))


n = len(chosen)
# overall counters
sc_t1 = pol_t1 = 0
cov_script = cov_union = 0
per = defaultdict(lambda: [0, 0, 0, 0, 0])  # n, sc_t1, pol_t1, cov_s, cov_u

for di in range(n):
    a, b = int(offs[di]), int(offs[di + 1])
    m = b - a
    ch = int(chosen[di])  # local index within block
    ss = script[a:b]
    zz = policy_logits(a, b, di)

    sc_top = int(np.argmax(ss))
    pol_top = int(np.argmax(zz))
    sc_t1 += (sc_top == ch)
    pol_t1 += (pol_top == ch)

    # script candidate set: constructive options with score>0, top-5 by score
    sorder = np.argsort(-ss, kind="stable")
    script_cands = [i for i in sorder if ss[i] > 0][:SCRIPT_MAX_CANDS]
    if not script_cands:
        script_cands = [sc_top]
    # union: add net's top-k PLAY/ATTACH/RETREAT not already in script set
    porder = np.argsort(-zz, kind="stable")
    elig = [i for i in porder
            if opt_type(a + i) in INJECT_TYPES and i not in script_cands]
    union = list(script_cands) + elig[:POLICY_TOPK]

    cs = ch in script_cands
    cu = ch in union
    cov_script += cs
    cov_union += cu

    dt = int(dtype[di])
    acc = per[dt]
    acc[0] += 1
    acc[1] += (sc_top == ch)
    acc[2] += (pol_top == ch)
    acc[3] += cs
    acc[4] += cu

print(f"held-out decisions: {n}   (POLICY_TOPK={POLICY_TOPK})")
print(f"SCRIPT top-1 : {sc_t1/n:.4f}")
print(f"POLICY top-1 : {pol_t1/n:.4f}   ({100*(pol_t1-sc_t1)/n:+.1f}pp)")
print(f"winner-move coverage in candidate set:")
print(f"  SCRIPT-only cands : {cov_script/n:.4f}")
print(f"  SCRIPT+POLICY union: {cov_union/n:.4f}   "
      f"({100*(cov_union-cov_script)/n:+.1f}pp)")

print("\n--- by decision type ---")
print(f"  {'type':9s} {'n':>6s} {'sc_t1':>7s} {'pol_t1':>7s} {'gap':>6s} "
      f"{'cov_s':>7s} {'cov_u':>7s} {'d_cov':>6s}")
for dt in sorted(per, key=lambda k: -per[k][0]):
    nn_, s1, p1, cs, cu = per[dt]
    name = OPT_TYPE_NAMES.get(dt, str(dt))
    print(f"  {name:9s} {nn_:6d} {s1/nn_:7.3f} {p1/nn_:7.3f} "
          f"{100*(p1-s1)/nn_:+6.1f} {cs/nn_:7.3f} {cu/nn_:7.3f} "
          f"{100*(cu-cs)/nn_:+6.1f}")
