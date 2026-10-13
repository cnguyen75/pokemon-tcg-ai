"""Turn a recording from tools/record_game.py into the site demo file.

Usage:
    python tools/export_demo.py RECORDING.json DEMO.json

Groups the agent's decisions by turn, keeps card/attack names and numbers
only, and folds the extra "knocked out" event the engine logs for each
pre-evolution card under a knocked-out evolved Pokemon (the stage-1/stage-2
card and the cards beneath it all move to the discard pile). Needs no engine.
"""
import json
import sys
from datetime import date

SCHEMA = "ptcg-replay/1"


def fold_knockouts(events, evolved_from):
    """Drop knocked_out events for the pre-evolution cards of the KO just before.

    evolved_from maps a card name to the names it evolved from in this game
    (taken from the recording's own evolve events)."""
    out = []
    for e in events:
        prev = out[-1] if out else None
        if (e["e"] == "knocked_out" and prev is not None and prev["e"] == "knocked_out"
                and prev["who"] == e["who"] and e["card"] in prev["_chain"]):
            prev["_chain"] |= evolved_from.get(e["card"], set())
            continue
        e = dict(e)
        if e["e"] == "knocked_out":
            e["_chain"] = set(evolved_from.get(e["card"], set()))
        out.append(e)
    for e in out:
        e.pop("_chain", None)
    return out


def evolution_map(raw):
    pairs = {}
    events = [e for s in raw["steps"] for e in s["events"]] + raw["final_events"]
    for e in events:
        if e["e"] == "evolve":
            pairs.setdefault(e["into"], set()).add(e["from"])
    return pairs


def main():
    raw = json.load(open(sys.argv[1]))
    evo = evolution_map(raw)
    turns = {}
    for s in raw["steps"]:
        t = turns.setdefault(s["turn"], {"turn": s["turn"], "decisions": []})
        search = s["search"]
        if search is not None:
            search = dict(search)
            search["overrode_policy"] = (search["picked"] is not None
                                         and search["picked"] != search["candidates"][0])
        t["decisions"].append({
            "context": s["context"],
            "events_before": fold_knockouts(s["events"], evo),
            "state": s["state"],
            "options": s["options"],
            "chosen": s["chosen"],
            "search": search,
            "position_value": s["position_value"],
        })
    demo = {
        "schema": SCHEMA,
        "about": {
            "what": ("One complete local game between the submitted agent and a local "
                     "stand-in opponent, recorded decision by decision."),
            "agent": raw["agent"],
            "opponent": raw["opponent"],
            "seed": raw["seed"],
            "agent_seat": raw["agent_seat"],
            "recorded": date.today().isoformat(),
            "commands": [
                f"uv run --python 3.13 --with-requirements requirements.txt python "
                f"tools/record_game.py baselines/{raw['agent']} baselines/{raw['opponent']} "
                f"RECORDING.json --seed {raw['seed']}",
                "python tools/export_demo.py RECORDING.json demo.json",
            ],
            "value_scale": ("search mean_value and position_value are the agent's leaf "
                            "evaluation on a 0-1000 scale: 0.6 x the AlphaZero value head's "
                            "win probability + 0.4 x a squashed handcrafted score, times 1000. "
                            "Terminal positions score 1000 (win) or 0 (loss)."),
            "scores": ("options[].score is the scripted policy's score; higher is preferred, "
                       "and options scoring 0 or less are not picked unless the rules force a pick."),
            "search_rule": ("the search compares candidates[0] (the policy's pick) with up to "
                            "four challengers over the sampled worlds; a challenger replaces it "
                            "only if its summed value beats the pick's by more than "
                            "margin_per_world x worlds."),
            "content": "card and attack names, HP, damage and counts only; no card text or art",
        },
        "result": {"winner": raw["winner"], "first_player": raw["first_player"],
                   "final_events": fold_knockouts(raw["final_events"], evo),
                   "final_state": raw["final_state"]},
        "turns": [turns[k] for k in sorted(turns)],
    }
    with open(sys.argv[2], "w") as f:
        json.dump(demo, f, separators=(",", ":"), ensure_ascii=False)
    n = sum(len(t["decisions"]) for t in demo["turns"])
    print(f"{len(demo['turns'])} turns, {n} decisions -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
