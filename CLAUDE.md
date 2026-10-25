# Working in this repository

Follow `CODING-RULES.md` in this folder for all code, comments, and documentation. Run
`python3 rules_check.py .` and the formatters named in `rules_check.py` before every commit;
a commit with findings is not ready.

Commit as Colin Nguyen <26684223+cnguyen75@users.noreply.github.com>. Never use any other
identity. Do not push unless asked.

Nothing here may name an employer's internal systems, customers, colleagues, vendors, or
metrics. No credentials, tokens, or private paths in any file, ever.

<!-- Repo-specific notes below: what is private, how to build, what the tests cover. -->

## Repo notes

- The competition engine, card data, and match replays were licensed for competition use only
  and have been deleted. They are gitignored (`data/`, `engine_build/`, `cg/`, `episodes/`,
  `*.dylib`, `*.dll`, `runs/*.tar.gz`) and must never be committed or redistributed. Nothing that
  plays a game runs without the Kaggle package under `data/`.
- Run (needs the package): `uv run --python 3.13 --with-requirements requirements.txt python
  tools/league.py baselines/v12_pwr baselines/v6 8`. Without it: `python3 tools/make_figures.py`
  redraws `figures/` from `results/results.json`.
- Tests: none. Before a commit, run `python3.13 -m compileall -q agent agent2 agent_az tools
  train` as the floor, plus `ruff check . && ruff format --check .` and `python3 rules_check.py .`.
- Frozen, not edited for style: `baselines/` (every submitted build and sparring control, as
  submitted), `exp/` (archived experiments), `docs/replay/` (recorded game and viewer), `results/`,
  `figures/`, `logs/`, `runs/`. ruff and rules_check skip them.
- Data: `train/data/`, checkpoints (`*.pt`, `*.pth`), and `scratch_logs/` are gitignored.
