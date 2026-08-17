"""Exec-style robustness check: load agent2/main.py via exec() WITHOUT
__file__ defined (mimics kaggle_environments), then play 2 self-play games
through cg.game. Exits nonzero on any crash.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT = os.path.join(REPO, "agent2")
CG_HOME = os.path.join(REPO, "data", "sample_submission", "sample_submission")
sys.path.insert(0, CG_HOME)

# exec the agent source with no __file__, from the agent dir (kaggle-style)
os.chdir(AGENT)
src = open(os.path.join(AGENT, "main.py"), "r", encoding="utf-8").read()
ns = {"__name__": "__main__"}  # deliberately no __file__
exec(compile(src, "<agent-exec>", "exec"), ns)
agent_fn = ns["agent"]
deck = ns["DECK"]
assert len(deck) == 60, "DECK must have 60 cards"

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

for g in range(2):
    obs, start = battle_start(deck, deck)
    assert start.errorPlayer < 0, f"deck error {start.errorType}"
    steps = 0
    while obs["current"]["result"] < 0:
        choice = agent_fn(obs)
        assert isinstance(choice, list), "agent must return list[int]"
        obs = battle_select(choice)
        steps += 1
        if steps > 4000:
            raise RuntimeError("game did not terminate")
    battle_finish()
    print(f"self-play game {g}: result={obs['current']['result']} steps={steps} OK")

# also exercise the deck-selection call path (select is None)
r = agent_fn({"select": None})
assert r == deck, "deck-choice call must return DECK"
print("exec-robustness: PASS")
