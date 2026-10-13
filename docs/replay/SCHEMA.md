# `demo.json` schema (`ptcg-replay/1`)

One complete local game, recorded with `tools/record_game.py` and grouped by
turn with `tools/export_demo.py`. It holds card and attack names, HP, damage and
counts only: no card text or art.

```
{
  "schema": "ptcg-replay/1",
  "about": {
    "what", "agent", "opponent",       // build folder names under baselines/
    "seed", "agent_seat",              // 0 = the agent was player 0
    "recorded",                        // YYYY-MM-DD
    "commands": [str],                 // record, then export
    "value_scale", "scores", "search_rule", "content"   // plain-language notes
  },
  "result": {
    "winner": "agent" | "opponent" | "draw",
    "first_player": "agent" | "opponent",
    "final_events": [Event],           // after the agent's last decision
    "final_state": [Side, Side]        // indexed by seat
  },
  "turns": [{
    "turn": int,                       // 0 = setup; odd/even alternate players
    "decisions": [Decision]            // the agent's decisions in this turn, in order
  }]
}
```

**Decision**

| Field | Meaning |
|---|---|
| `context` | What is being chosen, e.g. `main`, `to_hand`, `damage`, `to_active`, `mulligan` (engine's select context, lower case). |
| `events_before` | Events since the agent's previous decision (both players). |
| `state` | `[Side, Side]` by seat, as the agent sees it before deciding. |
| `options` | `[{label, score}]`: every legal option, and the scripted policy's score (higher preferred; `0` or less only if forced; `null` if not scored). |
| `chosen` | Indices into `options` that were played. |
| `search` | `null` if the search did not run; otherwise `{candidates, worlds, mean_value, margin_per_world, picked, overrode_policy}`. `candidates` are indices into `options`, the first being the policy's pick. `mean_value[k]` is the mean best leaf value for `candidates[k]` over `worlds` sampled worlds, 0–1000. `picked` is the index played. |
| `position_value` | The agent's leaf evaluation of the current position, 0–1000 (see `about.value_scale`). |

**Side**: `{active: Mon|null, bench: [Mon], prizes, hand, deck, discard, conditions: [str], hand_cards?: [str]}`. `hand_cards` appears only for the agent's own side.

**Mon**: `{name, hp, maxHp, energy, tools: [str]}`. `energy` is the count of attached energy.

**Event**: `{who: "agent"|"opponent", e, ...}` where `e` is one of `turn_start`, `play {card}`, `attach {card, to}`, `evolve {from, into}`, `switch {active, bench}`, `attack {by, attack}`, `hp {card, change}`, `knocked_out {card}`, `prize`, `coin {heads}`, or a condition (`poisoned`, `burned`, `asleep`, `paralyzed`, `confused`). For `hp`, `knocked_out` and conditions, `who` is the Pokémon's owner.

Energy names keep the card list's symbols (`Basic {W} Energy`); the viewer prints them as words.
