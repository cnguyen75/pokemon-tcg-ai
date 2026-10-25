"""Meta-gauntlet: run a target agent against a weighted panel of STRONG,
ladder-representative opponents and report one meta-weighted win rate plus
per-opponent Wilson 95% CIs.

The panel and weights are the local proxy for the live Kaggle ladder. Local
self-play alone OVER-predicts the ladder (v6 beat a frozen incumbent 59.5%
self-play but only tied ~53% on ladder); a panel of independent STRONG pilots,
weighted by the actual meta, is meant to track the ladder more honestly.

Usage:
  uv run --python 3.13 --with numpy python tools/gauntlet.py AGENT_DIR \\
      [--games-per N] [--procs P] [--seed-base S]

  AGENT_DIR   directory with main.py (+ deck.csv) for the agent under test.
  --games-per games PER opponent (default 150; deep search ~1-10 g/s).
  --procs     worker processes (default cpu-2).
  --seed-base base RNG seed; vary across repeated runs for fresh samples.

Reuses league.play_games / wilson from tools/league.py.

------------------------------------------------------------------------------
META WEIGHTS  (computed 2026-07-08 from episodes/daily/2026-07-0{5,6,7},
              878 deck appearances / 439 games, classified by flagship card)

Defined-field shares (excl. long-tail 'other'): grimmsnarl 30.9%, alakazam
20.5%, kangaskhan 10.3%, cynthia_garchomp 8.9%, okidogi 7.8%, archaludon 4.8%
(COLLAPSED, 32% WR - excluded), starmie 4.8%, comfey_lostbox 4.2%,
lucario 4.1%, crustle_wall 3.5%.

The gauntlet weights each STRONG pilot we have by its meta field-share, plus a
modest mirror slot (v6) for the genuine near-mirror fraction of the ladder,
renormalized over what we can actually measure. Uncovered strong archetypes
are FLAGGED AS GAPS below (not silently folded into self-play, which would
re-introduce the over-prediction we are trying to escape).

GAPS (no strong pilot yet -> NOT in panel; gauntlet under-weights these):
  - kangaskhan (~10%): multi-type Mega Kangaskhan toolbox, too complex for the
    generic framework; only ~46% ladder WR.
  - okidogi (~8%), comfey_lostbox (~4%), starmie (~5%), crustle_wall (~4%):
    no dedicated strong pilot; starmie has an agent2 config but no frozen
    baseline pilot here.
------------------------------------------------------------------------------
"""

import argparse
import math
from multiprocessing import Pool
import os
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from league import play_games, wilson  # noqa: E402


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def B(name: str) -> str:
    return os.path.join(REPO, "baselines", name)


# opponent -> (agent dir, meta weight).  Weights sum to 1.0.
# 2026-07-12/13: panel re-weighted to OUR OWN band's meta (775-810), measured
# from episodes/v71games (106g) + v7games (62g) = the 168 ladder games v7/v7.1
# actually played, classified by flagship card. Band shares | our ladder WR:
#   lucario 17.3%|48%, alakazam 16.1%|37% (v71-only: 20.8%|32%),
#   archaludon 15.5%|19% (v71-only 25%), crustle 13.7%|100%,
#   dragapult 11.9%|90%, grimmsnarl 7.1%|67%, starmie-mirror 6.5%|45%,
#   cynthia_garchomp 3.0%|20%, trevenant/bellibolt/abomasnow/kangas <2.5% ea.
# Weights below = band shares renormalized over the archetypes we can
# faithfully measure (0.625 of the field).
# FIDELITY AUDIT 2026-07-12 (200g agent2(v7.1) vs pilot | our ladder WR):
#   lucario1084    54.4% | 48.3%  VERIFIED (+6pp)
#   fw_grimm       60.8% | 66.7%  VERIFIED (-6pp)
#   fw_archaludon  44.1% | 25.0%  BORDERLINE (+19pp, ladder n=12 -> wide CI)
#   fw_alakazam    71.1% | 31.8%  FICTION -> REBUILT (Powerful Hand damage
#                  scaling via HAND_ATTACKS, see fw_alakazam/main.py); the
#                  rebuilt pilot holds v7.1 to ~59-62% (best 40.8% pilot WR)
#                  -- far more faithful than the fiction but still ~20pp
#                  short of the ladder's 68%. Iterate further before trusting
#                  agent2-vs-alakazam gains at face value.
#   fw_garchomp    76.5% | 20.0%  FICTION -> DROPPED from panel (no rebuild
#                  yet; band share only 3.0%, below the >=5% rebuild bar).
# GAPS (no pilot; deliberately absent): crustle 13.7% (we win 100% on
# ladder), dragapult 11.9% (90%), trevenant/kangaskhan/bellibolt (<2.5%).
# Mirror slot = v71_ref: ladder "starmie" opponents are literally our own
# older submissions + same-list users (45.5% ladder WR ~ a true mirror).
PANEL = {
    "lucario": (B("lucario1084"), 0.277),
    "alakazam": (B("fw_alakazam"), 0.258),
    "archaludon": (B("fw_archaludon"), 0.248),
    "grimmsnarl": (B("fw_grimm"), 0.114),
    "mirror(v71)": (B("v71_ref"), 0.104),
}


def run_matchup(
    target: str, opp_dir: str, n: int, procs: int, seed_base: int
) -> tuple[int, int, int]:
    per = math.ceil(n / procs)
    jobs = [(target, opp_dir, per, seed_base + i) for i in range(procs)]
    with Pool(procs) as pool:
        outs = pool.map(play_games, jobs)
    results = [r for rs, _ in outs for r in rs]
    w = sum(1 for r in results if r == 1)
    losses = sum(1 for r in results if r == -1)
    d = len(results) - w - losses
    return w, losses, d


def main() -> None:
    ap = argparse.ArgumentParser(description="Meta-weighted gauntlet.")
    ap.add_argument("agent_dir")
    ap.add_argument("--games-per", type=int, default=150)
    ap.add_argument("--procs", type=int, default=max(os.cpu_count() - 2, 1))
    ap.add_argument("--seed-base", type=int, default=1000)
    args = ap.parse_args()

    target = os.path.abspath(args.agent_dir)
    name = os.path.basename(target.rstrip("/"))
    print(f"GAUNTLET: {name}  ({args.games_per} games/opponent, {args.procs} procs)\n")
    print(f"{'opponent':18s} {'weight':>6s} {'W-L-D':>12s} {'WR%':>6s} {'95% CI':>16s}")

    t0 = time.time()
    weighted_sum = 0.0
    total_w = 0.0
    rows = []
    for opp, (opp_dir, weight) in PANEL.items():
        # per-opponent seed offset keeps matchups independent
        sb = args.seed_base + hash(opp) % 1000
        w, losses, d = run_matchup(target, opp_dir, args.games_per, args.procs, sb)
        n_dec = w + losses
        p, lo, hi = wilson(w, n_dec)
        rows.append((opp, weight, w, losses, d, p, lo, hi))
        weighted_sum += weight * p
        total_w += weight
        print(
            f"{opp:18s} {weight:6.3f} {f'{w}-{losses}-{d}':>12s} "
            f"{p * 100:6.1f} {f'[{lo * 100:.1f},{hi * 100:.1f}]':>16s}",
            flush=True,
        )

    meta_wr = weighted_sum / total_w if total_w else 0.0
    dt = time.time() - t0
    print(
        f"\nMETA-WEIGHTED WIN RATE: {meta_wr * 100:.1f}%   (panel weight {total_w:.3f}, {dt:.0f}s)"
    )
    print(
        "panel = pilots weighted by OUR 775-810 band shares "
        "(episodes/v71games+v7games, 2026-07-12); gaps not represented: "
        "crustle 14% (100% WR), dragapult 12% (90%), garchomp 3% "
        "(fiction pilot dropped), trevenant/kangaskhan/bellibolt <2.5%."
    )


if __name__ == "__main__":
    main()
