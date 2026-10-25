"""Record one local game, turn by turn, for the replay demo.

Usage (needs the competition engine present locally; see README):
    uv run --python 3.13 --with-requirements requirements.txt \
        python tools/record_game.py baselines/v12_pwr baselines/fw_alakazam OUT.json --seed 1

Plays AGENT (the recorded side) against OPPONENT once and writes a compact
JSON log. The agent's own code is not modified: its option scorer and its
determinized search are wrapped from outside, so the recording shows what it
already computes:

  - every option the agent was offered at each of its decisions, with the
    scripted policy's score for it;
  - when the pivotal-decision search ran, the candidates it compared and the
    mean leaf value per candidate over the sampled worlds (0-1000 scale: the
    value head's estimated win probability x 1000), the number of worlds, the
    margin a challenger had to clear, and whether the search overrode the
    policy's pick;
  - the agent's leaf evaluation of the current position at each decision;
  - public game events (plays, attaches, evolutions, attacks, HP changes,
    switches, coin flips) from the engine logs.

The output holds card and attack names, HP and damage numbers only: no card
text or art. The agent is time-budgeted, so a second run with the same seed
is not guaranteed to play the same game.
"""

import argparse
import json
import os
import random
import sys
import time
from types import ModuleType
from typing import Any


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import league  # noqa: E402


sys.path.insert(0, league.CG_HOME)
from cg.api import AreaType, OptionType, SelectContext, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


COND = ("poisoned", "burned", "asleep", "paralyzed", "confused")
AREA = {
    AreaType.DECK: "deck",
    AreaType.HAND: "hand",
    AreaType.DISCARD: "discard",
    AreaType.ACTIVE: "active",
    AreaType.BENCH: "bench",
    AreaType.PRIZE: "prize",
    AreaType.LOOKING: "looking",
}


class Recorder:
    def __init__(self, mod: ModuleType) -> None:
        self.mod = mod
        self.card = mod.CARD
        self.atk = mod.ATK
        self.top = None  # the Policy instance of the decision being recorded
        self.scores = []
        self.search = None
        self._wrap()

    def _wrap(self) -> None:
        mod, rec = self.mod, self
        Policy = mod.Policy
        choose, score_option = Policy.choose, Policy.score_option

        def choose_w(pol: Any) -> Any:
            if mod.IN_VERIFY or rec.top is not None:
                return choose(pol)
            rec.top, rec.scores, rec.search = pol, [], None
            try:
                return choose(pol)
            finally:
                rec.top = None

        def score_w(pol: Any, o: Any) -> float:
            s = score_option(pol, o)
            if pol is rec.top and len(rec.scores) < len(pol.sel.option):
                rec.scores.append(s)
            return s

        Policy.choose, Policy.score_option = choose_w, score_w

        det_search, begin_world, beam = mod.det_search, mod.begin_world, mod.beam_turn_value

        def det_w(pol: Any, cands: list[Any]) -> Any:
            rec._worlds = []
            out = det_search(pol, cands)
            full = [w for w in rec._worlds if len(w) == len(cands)]
            margin = mod.VALUE_MARGIN if mod.VN_OK else mod.SEARCH_MARGIN
            if mod.VN_OK:
                margin = pol.cfg.get("value_margin", margin)
            rec.search = {
                "candidates": list(cands),
                "worlds": len(full),
                "mean_value": (
                    [round(sum(w[i] for w in full) / len(full), 1) for i in range(len(cands))]
                    if full
                    else None
                ),
                "margin_per_world": margin,
                "picked": out,
            }
            return out

        def begin_w(*a: Any, **k: Any) -> Any:
            rec._worlds.append([])
            return begin_world(*a, **k)

        def beam_w(root_id: Any, first_choice: Any, *a: Any, **k: Any) -> float | None:
            v = beam(root_id, first_choice, *a, **k)
            if v is not None and rec._worlds:
                rec._worlds[-1].append(v)
            return v

        mod.det_search, mod.begin_world, mod.beam_turn_value = det_w, begin_w, beam_w

    def name(self, cid: int) -> str:
        d = self.card.get(cid)
        return d.name if d is not None else f"card {cid}"

    def poke(self, p: Any) -> dict[str, Any] | None:
        if p is None:
            return None
        return {
            "name": self.name(p.id),
            "hp": p.hp,
            "maxHp": p.maxHp,
            "energy": len(p.energies or []),
            "tools": [self.name(t.id) for t in (p.tools or [])],
        }

    def card_at(self, obs: Any, area: AreaType, index: int, player: int) -> str:
        c = self.mod.get_card(obs, area, index, player)
        return self.name(c.id) if c is not None and getattr(c, "id", None) else "a card"

    def label(self, obs: Any, o: Any, me: int) -> str:
        t = o.type
        pl = o.playerIndex if o.playerIndex is not None else me
        try:
            if t == OptionType.PLAY:
                return f"Play {self.card_at(obs, AreaType.HAND, o.index, me)}"
            if t == OptionType.ATTACH:
                return (
                    f"Attach {self.card_at(obs, o.area, o.index, me)} to "
                    f"{self.card_at(obs, o.inPlayArea, o.inPlayIndex, me)}"
                )
            if t == OptionType.EVOLVE:
                return (
                    f"Evolve {self.card_at(obs, o.inPlayArea, o.inPlayIndex, me)} into "
                    f"{self.card_at(obs, o.area, o.index, me)}"
                )
            if t == OptionType.ABILITY:
                return f"Use ability of {self.card_at(obs, o.area, o.index, me)}"
            if t == OptionType.RETREAT:
                return "Retreat"
            if t == OptionType.ATTACK:
                a = self.atk.get(o.attackId)
                return f"Attack: {a.name}" if a is not None else "Attack"
            if t == OptionType.END:
                return "End turn"
            if t == OptionType.DISCARD:
                return f"Discard {self.card_at(obs, o.area, o.index, me)}"
            if t in (OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD):
                where = AREA.get(o.area, "")
                side = "" if pl == me else "opponent's "
                if o.area == AreaType.PRIZE:
                    return "Prize card"
                return f"{self.card_at(obs, o.area, o.index, pl)} ({side}{where})".replace(
                    " ()", ""
                )
            if t == OptionType.NUMBER:
                return f"Choose {o.number}"
            if t == OptionType.YES:
                return "Yes"
            if t == OptionType.NO:
                return "No"
            if t == OptionType.ENERGY:
                return f"Energy {o.energyIndex}"
            if t == OptionType.SPECIAL_CONDITION:
                return f"Condition {o.specialConditionType}"
        except Exception:
            pass
        return OptionType(t).name.replace("_", " ").title()

    def snapshot(self, cur: Any, me: int) -> list[dict[str, Any]]:
        out = []
        for i, p in enumerate(cur.players):
            s = {
                "active": self.poke(p.active[0]) if p.active else None,
                "bench": [self.poke(b) for b in (p.bench or [])],
                "prizes": len(p.prize or []),
                "hand": p.handCount,
                "deck": p.deckCount,
                "discard": len(p.discard or []),
                "conditions": [c for c in COND if getattr(p, c, False)],
            }
            if i == me and p.hand is not None:
                s["hand_cards"] = [self.name(c.id) for c in p.hand]
            out.append(s)
        return out

    def events(self, logs: list[dict[str, Any]] | None, me: int) -> list[dict[str, Any]]:
        ev = []
        who = lambda i: "agent" if i == me else "opponent"  # noqa: E731
        for lg in logs or []:
            t, p = lg.get("type"), lg.get("playerIndex")
            n = self.name
            if t == 2:
                ev.append({"who": who(p), "e": "turn_start"})
            elif t == 10:
                ev.append({"who": who(p), "e": "play", "card": n(lg["cardId"])})
            elif t == 11:
                ev.append(
                    {
                        "who": who(p),
                        "e": "attach",
                        "card": n(lg["cardId"]),
                        "to": n(lg["cardIdTarget"]),
                    }
                )
            elif t == 12:
                ev.append(
                    {
                        "who": who(p),
                        "e": "evolve",
                        "into": n(lg["cardId"]),
                        "from": n(lg["cardIdTarget"]),
                    }
                )
            elif t == 8:
                ev.append(
                    {
                        "who": who(p),
                        "e": "switch",
                        "active": n(lg["cardIdActive"]),
                        "bench": n(lg["cardIdBench"]),
                    }
                )
            elif t == 15:
                a = self.atk.get(lg.get("attackId"))
                ev.append(
                    {
                        "who": who(p),
                        "e": "attack",
                        "by": n(lg["cardId"]),
                        "attack": a.name if a is not None else "attack",
                    }
                )
            elif t == 16:
                ev.append(
                    {"who": who(p), "e": "hp", "card": n(lg["cardId"]), "change": lg.get("value")}
                )
            elif t == 22:
                ev.append({"who": who(p), "e": "coin", "heads": bool(lg.get("head"))})
            elif (
                t in (6, 7)
                and lg.get("fromArea") in (AreaType.ACTIVE, AreaType.BENCH)
                and lg.get("toArea") == AreaType.DISCARD
                and lg.get("cardId")
                and getattr(self.card.get(lg["cardId"]), "hp", 0)
            ):
                ev.append({"who": who(p), "e": "knocked_out", "card": n(lg["cardId"])})
            elif t in (6, 7) and lg.get("fromArea") == AreaType.PRIZE:
                ev.append({"who": who(p), "e": "prize"})
            elif t in (17, 18, 19, 20, 21):
                ev.append({"who": who(p), "e": COND[t - 17]})
        return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("opponent")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seat", type=int, default=0, help="the agent's seat (0 or 1)")
    args = ap.parse_args()

    random.seed(args.seed)
    a = league.load_agent(os.path.join(REPO, args.agent))
    rec = Recorder(a)
    b = league.load_agent(os.path.join(REPO, args.opponent))
    me = args.seat
    agents = [None, None]
    agents[me], agents[1 - me] = a.agent, b.agent
    decks = [None, None]
    decks[me], decks[1 - me] = a.DECK, b.DECK

    steps = []
    t0 = time.time()
    obs, start = battle_start(*decks)
    if start.errorPlayer >= 0:
        sys.exit(f"deck error p{start.errorPlayer} t{start.errorType}")
    while obs["current"]["result"] < 0:
        idx = obs["current"]["yourIndex"]
        if idx != me:
            obs = battle_select(agents[idx](obs))
            continue
        o = to_observation_class(obs)
        choice = a.agent(obs)
        sel = o.select
        step = {
            "turn": o.current.turn,
            "context": SelectContext(sel.context).name.lower(),
            "events": rec.events(obs.get("logs"), me),
            "state": rec.snapshot(o.current, me),
            "options": [
                {
                    "label": rec.label(o, opt, me),
                    "score": (round(rec.scores[i], 1) if i < len(rec.scores) else None),
                }
                for i, opt in enumerate(sel.option)
            ],
            "chosen": list(choice),
            "search": rec.search,
        }
        try:
            step["position_value"] = round(a.eval_world(o.current, me, 0, 0), 1)
        except Exception:
            step["position_value"] = None
        steps.append(step)
        obs = battle_select(choice)
    battle_finish()
    tail = rec.events(obs.get("logs"), me)
    result = obs["current"]["result"]
    out = {
        "agent": os.path.basename(args.agent.rstrip("/")),
        "opponent": os.path.basename(args.opponent.rstrip("/")),
        "agent_seat": me,
        "seed": args.seed,
        "first_player": "agent" if obs["current"]["firstPlayer"] == me else "opponent",
        "winner": "agent" if result == me else ("opponent" if result == 1 - me else "draw"),
        "final_events": tail,
        "final_state": rec.snapshot(to_observation_class(obs).current, me),
        "seconds": round(time.time() - t0, 1),
        "steps": steps,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(
        f"{out['agent']} vs {out['opponent']}: winner {out['winner']}, "
        f"{len(steps)} agent decisions, {out['seconds']}s -> {args.out}"
    )


if __name__ == "__main__":
    main()
