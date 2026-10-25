"""EXPERIMENT ONLY (does NOT touch agent2). Policy imitation-learning data.

Extract (observation, legal options, chosen option index) tuples for MAIN-phase
single-pick decisions from top-player replays, featurize each OPTION, and record
the SCRIPTED policy's score for each option (so net vs script can be compared on
the exact same held-out nodes).

Focus: SelectType.MAIN (0) + SelectContext.MAIN (0), maxCount==1, >=2 options,
single chosen index. These are the "which move to make" decisions.

Writes train/data/policy.npz:
  X_opt   (n_opt, D_opt)   per-option feature rows
  X_state (n_dec, D_state) per-decision shared state features
  offs    (n_dec+1,)       CSR-style offsets into X_opt for each decision
  chosen  (n_dec,)         index (within the decision's option block) chosen
  script  (n_opt,)         scripted-policy score for each option
  ep      (n_dec,)         episode id (for by-episode holdout)
  dtype   (n_dec,)         decision-type code (option type of the CHOSEN option)
"""

import glob
import json
import os
import sys
from typing import Any

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agent2"))

from cg.api import (  # noqa: E402
    AreaType,
    OptionType,
    SelectContext,
    SelectType,
    all_attack,
    all_card_data,
    to_observation_class,
)
import main as agent2_main  # the scripted Policy  # noqa: E402
import value_features as vf  # noqa: E402


CARD = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: a for a in all_attack()}

# Top-player replay sources (mirror extract_episodes.py plus daily).
DIRS = glob.glob(os.path.join(REPO, "episodes", "daily", "*")) + [
    os.path.join(REPO, "episodes", d)
    for d in ("day30", "v5games", "v51games", "v52games", "v5recent", "v6games", "ours")
]

N_OPT_TYPES = 17  # OptionType enum size


def _g(o: Any, k: str, d: Any = None) -> Any:
    if isinstance(o, dict):
        v = o.get(k, d)
    else:
        v = getattr(o, k, d)
    return d if v is None else v


def _card_at(
    cur_players: list[dict[str, Any]], me: int, area: int, index: int, playerIndex: int | None
) -> dict[str, Any] | None:
    """Resolve the Card/Pokemon dict an option refers to (replay JSON dict)."""
    pi = playerIndex if playerIndex is not None else me
    try:
        pl = cur_players[pi]
        if area == AreaType.HAND:
            h = pl.get("hand")
            return h[index] if h else None
        if area == AreaType.ACTIVE:
            return (pl.get("active") or [None])[index]
        if area == AreaType.BENCH:
            return (pl.get("bench") or [])[index]
        if area == AreaType.DISCARD:
            return (pl.get("discard") or [])[index]
    except Exception:
        return None
    return None


def _card_static(cid: int) -> list[float]:
    """(is_pokemon, is_item, is_supp, is_stadium, is_energy, is_tool,
    is_basic, is_stage1, is_stage2, is_ex, is_mega, hp_norm, retreat_norm)"""
    d = CARD.get(cid)
    if d is None:
        return [0.0] * 13
    ct = int(d.cardType)
    return [
        1.0 if ct == 0 else 0.0,
        1.0 if ct == 1 else 0.0,
        1.0 if ct == 3 else 0.0,
        1.0 if ct == 4 else 0.0,
        1.0 if ct in (5, 6) else 0.0,
        1.0 if ct == 2 else 0.0,
        1.0 if d.basic else 0.0,
        1.0 if d.stage1 else 0.0,
        1.0 if d.stage2 else 0.0,
        1.0 if d.ex else 0.0,
        1.0 if d.megaEx else 0.0,
        min(d.hp, 400) / 400.0,
        min(d.retreatCost, 4) / 4.0,
    ]


def option_features(o: dict[str, Any], cur: dict[str, Any], me: int) -> list[float]:
    """Feature vector for a single legal option (from replay JSON dicts)."""
    t = int(_g(o, "type", -1))
    f = [1.0 if t == i else 0.0 for i in range(N_OPT_TYPES)]  # 17 type one-hot

    players = _g(cur, "players")
    my = players[me]
    op = players[1 - me]
    op_act = (op.get("active") or [None])[0] if op else None
    op_hp = _g(op_act, "hp", 0) if op_act is not None else 0
    op_id = _g(op_act, "id", 0) if op_act is not None else 0

    cid = 0
    area = _g(o, "area")
    index = _g(o, "index")
    pidx = _g(o, "playerIndex")
    if t in (
        OptionType.PLAY,
        OptionType.ATTACH,
        OptionType.EVOLVE,
        OptionType.ABILITY,
        OptionType.DISCARD,
        OptionType.CARD,
        OptionType.TOOL_CARD,
        OptionType.ENERGY_CARD,
    ):
        card = _card_at(players, me, area, index, pidx)
        cid = _g(card, "id", 0)
    f += _card_static(cid)  # 13

    # attack-specific: damage, energy cost, KO vs opp active, wins-game flag
    atk_feats = [0.0, 0.0, 0.0, 0.0]
    if t == OptionType.ATTACK:
        aid = _g(o, "attackId", 0)
        # find the attacker pokemon = my active (MAIN attacks come from active)
        my_act = (my.get("active") or [None])[0] if my else None
        att_id = _g(my_act, "id", 0) if my_act is not None else 0
        a = ATK.get(aid)
        if a is not None:
            dmg = vf._eff_damage(CARD.get(att_id), CARD.get(op_id), a)
            atk_feats[0] = min(dmg, 400) / 400.0
            atk_feats[1] = min(len(a.energies), 5) / 5.0
            atk_feats[2] = 1.0 if (op_hp and dmg >= op_hp) else 0.0
            n_prz = len(op.get("prize") or [])
            atk_feats[3] = (
                1.0 if (op_hp and dmg >= op_hp and n_prz <= vf._pyield(CARD.get(op_id))) else 0.0
            )
    f += atk_feats  # 4

    # target-side context: does this option target ACTIVE vs BENCH, my vs opp
    ipa = _g(o, "inPlayArea")
    f.append(1.0 if ipa == AreaType.ACTIVE else 0.0)
    f.append(1.0 if ipa == AreaType.BENCH else 0.0)
    f.append(1.0 if pidx == (1 - me) else 0.0)  # targets opponent
    # option-count context handled at state level; small per-option bias
    f.append(1.0)  # bias term
    return f  # 17 + 13 + 4 + 4 = 38


D_OPT = 38


def parse_file(
    path: str, ep_id: int
) -> list[tuple[list[float], list[list[float]], int, list[float], int]] | None:
    try:
        d = json.load(open(path))
    except Exception:
        return None
    if not isinstance(d, dict) or "steps" not in d:
        return None
    rew = d.get("rewards")
    if (
        not isinstance(rew, list)
        or len(rew) != 2
        or rew[0] is None
        or rew[1] is None
        or rew[0] == rew[1]
    ):
        return None
    winner = 0 if rew[0] > rew[1] else 1

    decisions = []  # each: (state_feats, [opt_feats...], chosen_local, [scripts], dtype)
    for step in d.get("steps", []):
        for j, ag in enumerate(step):
            if ag.get("status") != "ACTIVE":
                continue
            # only keep decisions made by the WINNER (learn from strong moves)
            if j != winner:
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
            action = ag.get("action")
            if not isinstance(action, list) or len(action) != 1:
                continue
            chosen = action[0]
            if not (0 <= chosen < len(opt)):
                continue
            if cur.get("yourIndex") != j:
                continue
            me = j
            try:
                state_feats = vf.extract(cur, me, CARD, ATK)
                state_feats.append(vf.hc_squash(vf.handcrafted_eval(cur, me)))
                opt_feats = [option_features(o, cur, me) for o in opt]
            except Exception:
                continue
            # scripted policy scores on this exact node
            try:
                pol = agent2_main.Policy(to_observation_class(obs))
                agent2_main._G["plan"] = pol.plan_attack()
                scripts = [float(pol.score_option(o)) for o in pol.sel.option]
            except Exception:
                scripts = [0.0] * len(opt)
            dtype = int(opt[chosen].get("type", -1))
            decisions.append((state_feats, opt_feats, chosen, scripts, dtype))
    return decisions


def main() -> None:
    files = []
    for dpath in DIRS:
        files += sorted(glob.glob(os.path.join(dpath, "*.json")))
    print(f"{len(files)} replay files", flush=True)

    X_opt, X_state, offs, chosen_l, script_l, ep_l, dtype_l = [], [], [0], [], [], [], []
    n_games = 0
    by_src = {}
    for i, path in enumerate(files):
        res = parse_file(path, i)
        if res:
            n_games += 1
            src = os.path.relpath(os.path.dirname(path), os.path.join(REPO, "episodes"))
            st = by_src.setdefault(src, [0, 0])
            st[0] += 1
            for state_feats, opt_feats, chosen, scripts, dtype in res:
                X_state.append(state_feats)
                for of in opt_feats:
                    X_opt.append(of)
                for sc in scripts:
                    script_l.append(sc)
                offs.append(len(X_opt))
                chosen_l.append(chosen)
                ep_l.append(i)
                dtype_l.append(dtype)
                st[1] += 1
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)} files, {len(chosen_l)} decisions", flush=True)

    print("--- decisions by source ---")
    for src in sorted(by_src):
        g, dd = by_src[src]
        print(f"  {src:20s} {g:5d} games  {dd:7d} decisions")

    out = os.path.join(REPO, "train", "data", "policy.npz")
    np.savez_compressed(
        out,
        X_opt=np.asarray(X_opt, dtype=np.float32),
        X_state=np.asarray(X_state, dtype=np.float32),
        offs=np.asarray(offs, dtype=np.int64),
        chosen=np.asarray(chosen_l, dtype=np.int64),
        script=np.asarray(script_l, dtype=np.float32),
        ep=np.asarray(ep_l, dtype=np.int64),
        dtype_=np.asarray(dtype_l, dtype=np.int64),
    )
    n_opt = len(X_opt)
    n_dec = len(chosen_l)
    print(
        f"games used {n_games}, decisions {n_dec}, options {n_opt}, "
        f"mean_opts {n_opt / max(n_dec, 1):.2f}, D_opt {D_OPT}, "
        f"D_state {len(X_state[0]) if X_state else 0} -> {out}"
    )


if __name__ == "__main__":
    main()
