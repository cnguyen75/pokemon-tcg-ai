"""Generate self-play training data with the seeded local engine.

Plays games between agent dirs (agent2 vs agent2 / baselines) using
engine_build/test_cg (CG_SEED-honoring build), extracting the value-net
feature vector at every acting select from both perspectives, labeled with
the final result.

Usage:
  uv run --python 3.13 --with numpy python train/selfplay.py \
      --matchup starmie_grimm --games 300 --seed 0 --procs 8

Writes train/data/selfplay_<matchup>_<seed>.npz (X, y, ep, hc).
Game ids (ep) are globally unique per (matchup, seed) via hashing.
"""

import argparse
import importlib.util
from multiprocessing import Pool
import os
import random
import sys
import time


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CG_HOME = os.path.join(REPO, "engine_build", "test_cg")
AGENT2 = os.path.join(REPO, "agent2")

# matchup -> (path_a, deck_a, path_b, deck_b);  deck None = agent's own default
MATCHUPS = {
    "starmie_grimm": (AGENT2, "starmie", AGENT2, "grimmsnarl"),
    "starmie_starmie": (AGENT2, "starmie", AGENT2, "starmie"),
    "grimm_grimm": (AGENT2, "grimmsnarl", AGENT2, "grimmsnarl"),
    "starmie_multiply": (AGENT2, "starmie", os.path.join(REPO, "baselines", "multiply"), None),
    "grimm_multiply": (AGENT2, "grimmsnarl", os.path.join(REPO, "baselines", "multiply"), None),
    "starmie_fwgrimm": (AGENT2, "starmie", os.path.join(REPO, "baselines", "fw_grimm"), None),
    "grimm_fwgrimm": (AGENT2, "grimmsnarl", os.path.join(REPO, "baselines", "fw_grimm"), None),
    # cycle-2 hard-matchup coverage (the archetypes the net can't tell apart)
    "starmie_lucario": (AGENT2, "starmie", os.path.join(REPO, "baselines", "lucario1084"), None),
    "grimm_lucario": (AGENT2, "grimmsnarl", os.path.join(REPO, "baselines", "lucario1084"), None),
    "starmie_archaludon": (
        AGENT2,
        "starmie",
        os.path.join(REPO, "baselines", "archaludon_spar"),
        None,
    ),
    "grimm_archaludon": (
        AGENT2,
        "grimmsnarl",
        os.path.join(REPO, "baselines", "archaludon_spar"),
        None,
    ),
    # current-meta gauntlet decks (search ON vs these → deep-search-quality data)
    "starmie_alakazam": (AGENT2, "starmie", os.path.join(REPO, "baselines", "fw_alakazam"), None),
    "starmie_garchomp": (AGENT2, "starmie", os.path.join(REPO, "baselines", "fw_garchomp"), None),
    # hybrid (AZ-value) self-play: value labels on the strongest agent's own
    # play distribution — the data channel the 0.80-AUC head is starved of
    "azval_azval": (
        os.path.join(REPO, "baselines", "a2_azval"),
        None,
        os.path.join(REPO, "baselines", "a2_azval"),
        None,
    ),
    "azval_alakazam": (
        os.path.join(REPO, "baselines", "a2_azval"),
        None,
        os.path.join(REPO, "baselines", "fw_alakazam"),
        None,
    ),
    "azval_kanga": (
        os.path.join(REPO, "baselines", "a2_azval"),
        None,
        os.path.join(REPO, "baselines", "fw_kanga"),
        None,
    ),
    "azval_grimm": (
        os.path.join(REPO, "baselines", "a2_azval"),
        None,
        os.path.join(REPO, "baselines", "fw_grimm"),
        None,
    ),
}


def load_agent(path, deck_name, tag):
    path = os.path.abspath(path)
    if deck_name:
        os.environ["CG_DECK_NAME"] = deck_name
    sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("agent_" + tag, os.path.join(path, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(path)
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
        sys.path.pop(0)
    if not hasattr(mod, "DECK"):
        with open(os.path.join(path, "deck.csv")) as f:
            rows = f.read().split("\n")
        mod.DECK = [int(rows[i]) for i in range(60)]
    return mod


def play_worker(args):
    (a_path, a_deck, b_path, b_deck, n_games, seed) = args
    os.environ["CG_SEED"] = str(seed)
    sys.path.insert(0, CG_HOME)
    sys.path.insert(0, AGENT2)
    from cg.api import all_attack, all_card_data
    from cg.game import battle_finish, battle_select, battle_start
    import value_features as vf

    CARD = {c.cardId: c for c in all_card_data()}
    ATK = {a.attackId: a for a in all_attack()}
    random.seed(seed)
    a = load_agent(a_path, a_deck, "a")
    b = load_agent(b_path, b_deck, "b")

    Xs, ys, eps, hcs = [], [], [], []
    n_draw = 0
    for g in range(n_games):
        a_index = g % 2
        agents = [None, None]
        agents[a_index] = a.agent
        agents[1 - a_index] = b.agent
        obs, start = battle_start(
            a.DECK if a_index == 0 else b.DECK, a.DECK if a_index == 1 else b.DECK
        )
        if start.errorPlayer >= 0:
            raise RuntimeError(f"deck error p{start.errorPlayer} t{start.errorType}")
        gX, ghc, gme = [], [], []
        n_steps = 0
        while obs["current"]["result"] < 0:
            idx = obs["current"]["yourIndex"]
            cur = obs["current"]
            if obs.get("select") is not None:
                for me in (idx, 1 - idx):
                    try:
                        x = vf.extract(cur, me, CARD, ATK)
                        h = vf.handcrafted_eval(cur, me)
                    except Exception:
                        continue
                    gX.append(x)
                    ghc.append(h)
                    gme.append(me)
            n_steps += 1
            obs = battle_select(agents[idx](obs))
        battle_finish()
        r = obs["current"]["result"]
        if r == 2:
            n_draw += 1
            continue
        gid = seed * 100_000 + g
        for x, h, me in zip(gX, ghc, gme):
            Xs.append(x)
            ys.append(1 if me == r else 0)
            eps.append(gid)
            hcs.append(h)
    return Xs, ys, eps, hcs, n_games, n_draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matchup", required=True, choices=sorted(MATCHUPS))
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()

    import numpy as np

    a_path, a_deck, b_path, b_deck = MATCHUPS[args.matchup]
    per = -(-args.games // args.procs)
    base = abs(hash(args.matchup)) % 1000 + args.seed * 10_000
    jobs = [(a_path, a_deck, b_path, b_deck, per, base * 100 + i) for i in range(args.procs)]
    t0 = time.time()
    with Pool(args.procs) as pool:
        outs = pool.map(play_worker, jobs)
    X = np.asarray([x for o in outs for x in o[0]], dtype=np.float32)
    y = np.asarray([v for o in outs for v in o[1]], dtype=np.int8)
    ep = np.asarray([v for o in outs for v in o[2]], dtype=np.int64)
    hc = np.asarray([v for o in outs for v in o[3]], dtype=np.float32)
    n_games = sum(o[4] for o in outs)
    n_draw = sum(o[5] for o in outs)
    out = os.path.join(REPO, "train", "data", f"selfplay_{args.matchup}_{args.seed}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, X=X, y=y, ep=ep, hc=hc)
    dt = time.time() - t0
    print(
        f"{args.matchup} seed{args.seed}: {n_games} games ({n_draw} draws) "
        f"{len(y)} rows in {dt:.0f}s ({n_games / dt:.2f} g/s) -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
