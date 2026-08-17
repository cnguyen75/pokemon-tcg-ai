"""Parity check: agent2/policy_net inference featurizer == training featurizer.

For MAIN single-pick decisions in replays, compare
train/extract_policy.option_features (dict) vs
agent2/policy_net._option_features (dataclass via to_observation_class), and the
101-dim state features, and the final per-option scores/argmax.
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agent2"))
sys.path.insert(0, os.path.join(REPO, "train"))

import value_features as vf
from cg.api import (all_card_data, all_attack, to_observation_class,
                    SelectType, SelectContext)
import extract_policy as ep
import policy_net as pn

CARD = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: a for a in all_attack()}
pn.load(os.path.join(REPO, "agent2"), CARD, ATK)
assert pn.OK, "policy_net failed to load"

files = sorted(glob.glob(os.path.join(REPO, "episodes", "day30", "*.json")))[:60]
opt_maxdiff = 0.0
state_maxdiff = 0.0
score_argmatch = 0
n_dec = 0
n_opt = 0
for path in files:
    try:
        d = json.load(open(path))
    except Exception:
        continue
    if not isinstance(d, dict) or "steps" not in d:
        continue
    for step in d.get("steps", []):
        for j, ag in enumerate(step):
            if ag.get("status") != "ACTIVE":
                continue
            obs = ag.get("observation") or {}
            sel = obs.get("select")
            cur = obs.get("current")
            if not sel or not cur:
                continue
            if sel.get("type") != int(SelectType.MAIN):
                continue
            if sel.get("context") != int(SelectContext.MAIN):
                continue
            if sel.get("maxCount") != 1:
                continue
            opt = sel.get("option") or []
            if len(opt) < 2:
                continue
            if cur.get("yourIndex") != j:
                continue
            me = j
            # training-side features (dict)
            try:
                tr_state = vf.extract(cur, me, CARD, ATK)
                tr_state.append(vf.hc_squash(vf.handcrafted_eval(cur, me)))
                tr_opts = [ep.option_features(o, cur, me) for o in opt]
            except Exception:
                continue
            # inference-side features (dataclass)
            try:
                oc = to_observation_class(obs)
                live_cur = oc.current
                live_opts = oc.select.option
                inf_state = vf.extract(live_cur, me, CARD, ATK)
                inf_state.append(vf.hc_squash(vf.handcrafted_eval(live_cur, me)))
                inf_opts = [pn._option_features(o, live_cur, me) for o in live_opts]
            except Exception:
                continue
            if len(inf_opts) != len(tr_opts):
                continue
            state_maxdiff = max(state_maxdiff,
                                float(np.max(np.abs(np.array(tr_state) - np.array(inf_state)))))
            for a, b in zip(tr_opts, inf_opts):
                opt_maxdiff = max(opt_maxdiff,
                                  float(np.max(np.abs(np.array(a) - np.array(b)))))
                n_opt += 1
            # end-to-end scores via inference path
            zs = pn.score_options(live_cur, me, live_opts)
            # reference: build same rows from training feats and run numpy fwd
            Xt = np.array([list(tf) + list(tr_state) for tf in tr_opts],
                          dtype=np.float32)
            xt = (Xt - pn._W["mean"]) / pn._W["std"]
            h = np.maximum(xt @ pn._W["W1t"] + pn._W["b1"], 0.0)
            h = np.maximum(h @ pn._W["W2t"] + pn._W["b2"], 0.0)
            zt = (h @ pn._W["W3t"] + pn._W["b3"]).ravel()
            if int(np.argmax(zs)) == int(np.argmax(zt)):
                score_argmatch += 1
            n_dec += 1

print(f"decisions {n_dec}, options {n_opt}")
print(f"option feature max abs diff : {opt_maxdiff:.3e}")
print(f"state  feature max abs diff : {state_maxdiff:.3e}")
print(f"score argmax agreement      : {score_argmatch}/{n_dec} "
      f"({100*score_argmatch/max(n_dec,1):.1f}%)")
