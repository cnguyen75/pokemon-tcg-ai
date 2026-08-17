# Pokémon TCG AI Battle Challenge — competition agent

An agent for the Kaggle [PTCG AI Battle Challenge Simulation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
(July–August 2026): a two-player Pokémon Trading Card Game simulator with hidden
information, where entries play a continuously-running TrueSkill ladder against
other competitors' agents.

**Final result:** rank 2531 / 6892 teams (top 37%), rating 668.
Best converged rating during the run: ~820.

---

## Approach

The agent is a **scripted policy wrapped in a determinized beam search with a
learned value function at the leaves**.

```
observation ──> archetype detection ──> per-matchup config
                        │
                        ▼
        candidate move generation (hand-written scorers,
        ~30 decision contexts: play / attach / evolve /
        retreat / attack / promote / …)
                        │
                        ▼
        within-turn beam search  (width 4, top-3 expansion)
        over whole action *sequences*, not single moves
                        │
                        ▼
        determinized opponent reply  (N sampled worlds for
        hidden hand/deck state)
                        │
                        ▼
        leaf evaluation:  0.6 × handcrafted eval
                        + 0.4 × AlphaZero value head
```

Three components, built in that order:

1. **Scripted policy** — per-context card scorers plus an attack planner with
   weakness/resistance maths, KO and prize-race logic, and per-archetype
   overrides. On its own this was worth ~790 rating.
2. **Determinized beam search** — optimises the entire turn's action sequence
   rather than one move at a time. This was the single largest gain of the
   project (+7pp head-to-head). Shallow search had been *informationally
   sterile*: rolling out both sides with the same policy and scoring with the
   same eval just re-derived the policy's own preferences.
3. **AlphaZero value head** — a small policy+value network trained by self-play
   expert iteration. The *policy* head plateaued at parity with its scripted
   teacher (it distils toward the teacher and can't exceed it), but the **value
   head kept improving** (0.72 → 0.80 outcome AUC) and, grafted into the search
   leaves in place of the numpy value net, beat the previous best agent 58%
   head-to-head over 400 games.

## What turned out to matter

Most of the measured wins came from **finding bugs in the world model**, not
from tuning heuristics.

- **`deck.csv` was never read.** The agent submitted its deck from a config
  constant, so *every* deck experiment for six weeks had silently compared
  identical decks. Found by pulling real ladder replays and noticing a card the
  agent should have been playing appeared zero times in 23 games.
- **An opponent's main attack was modelled as 0 damage.** Alakazam's *Powerful
  Hand* places 2 damage counters per card in hand (20 × hand size ≈ 160–200),
  but the card data lists its damage as `n/a`. The threat model therefore never
  rotated a damaged attacker out of a lethal it couldn't see. Fixing it moved
  that matchup from 43% → 54% on the live ladder, and dragged a second matchup
  from ~35% to ~58%.
- **Local evaluation lies.** Framework-piloted "opponents" were unfaithful
  (one lost to us 71% while the real deck beat us 63%). Ground truth came from
  downloading our own ladder replays and computing per-archetype win rates
  against the actual field.

## What didn't work (kept because negative results are results)

- **Policy micro-surgery** — four separate hand-reasoned "obvious" fixes
  (energy-attachment guards, supporter priority, threat-model rewrites) each
  measured **negative** in paired testing. The beam+value optimiser was
  already making better decisions than local reasoning about individual cards.
- **Scaling search width** — doubling the beam changed runtime by only ~25%:
  the per-turn branching factor is too small to fill a wider beam, so compute
  headroom couldn't be converted into strength.
- **Imitation learning** — a policy net predicted top players' moves +10pp
  better than the scripts, yet *played* worse when used as a candidate
  generator. Move-prediction accuracy ≠ playing strength.
- **More self-play rounds** — the standalone AlphaZero agent plateaued at
  parity with its scripted teacher across four expert-iteration rounds.

## Methodology notes

Two measurement lessons that cost real time:

- **Pool your controls.** The agent is non-deterministic, so a 204-game
  evaluation cell carries ±7pp. A single control cell made three separate
  "improvements" look real; re-running the *control* three times showed the
  baseline itself spanned 42–55%. Nothing below ~10pp is resolvable at that
  sample size.
- **Verify the scoreboard, don't model it.** I spent two weeks optimising
  against an assumed scoring rule (that deactivated submissions still counted
  toward team score). They don't — only the active submissions count. Reading
  our own leaderboard row once would have caught it immediately.

## Layout

| Path | Contents |
|---|---|
| `agent2/` | The main agent: policy, beam search, value-net integration, per-deck configs |
| `agent_az/` | AlphaZero implementation — network, PUCT MCTS, self-play |
| `train/` | Training scripts for the value/policy nets (data not included) |
| `tools/` | Analysis: ladder-replay mining, per-archetype win rates, paired-seed leagues, matchup autopsies |
| `baselines/` | Frozen agent variants used as controls and sparring partners |
| `exp/` | Archived experiments (incl. no-ship branches) |

## Running it

**The competition engine is not included in this repository.** The simulator
(`ptcgProgram`), the card data, and the match replays are covered by
`LicenseRef-PTCG-ABC-Competition-Use-Only`, which explicitly prohibits
redistribution. Reproducing this work requires downloading those from Kaggle
under your own competition entry; the code expects a `cg/` package and
`data/EN_Card_Data.csv` alongside each agent directory.

```bash
# local head-to-head between two agent variants (needs the engine present)
uv run --python 3.13 --with numpy,torch python tools/league.py agent2 baselines/v6

# per-archetype win rates from your own ladder replays
uv run --python 3.13 python tools/my_episodes.py <SUBMISSION_ID>
```
