"""Per-deck configuration for the agent2 policy framework.

Loaded by main.py via exec() (no import, so no module-cache collisions when
two agent copies run in one process). Everything here is plain data; the
generic policy in main.py interprets it.

Key card ids
------------
Grimmsnarl deck: 7 dark energy, 104 Froslass, 112 Munkidori, 646 Impidimp,
647 Morgrem, 648 Grimmsnarl ex, 860 Snorunt, 1079 Rare Candy, 1080 Unfair
Stamp, 1086 Poffin, 1097 Night Stretcher, 1152 Poke Pad, 1161 Handheld Fan,
1182 Boss's Orders, 1219 Petrel, 1227 Lillie's Determination, 1231 Dawn,
1259 Spikemuth Gym.

Starmie deck: 3 water energy, 17 Ignition Energy, 666 Cinderace, 1030 Staryu,
1031 Mega Starmie ex, 1086 Poffin, 1097 Night Stretcher, 1120 Crushing
Hammer, 1121 Ultra Ball, 1122 Pokegear, 1145 Mega Signal, 1159 Hero's Cape,
1182 Boss's Orders, 1189 Salvatore, 1223 Harlequin, 1225 Hilda, 1227 Lillie's
Determination, 1229 Wally's Compassion.
"""

GRIMM_DECK = (
    [7] * 10 + [104] * 2 + [112] * 4 + [646] * 4 + [647] * 3 + [648] * 3
    + [860] * 2 + [1079] * 3 + [1080] + [1086] * 4 + [1097] * 3 + [1152] * 4
    + [1161] * 2 + [1182] * 2 + [1219] * 4 + [1227] * 4 + [1231] + [1259] * 4
)

STARMIE_DECK = (
    [3] * 9 + [17] * 4 + [666] * 4 + [1030] * 3 + [1031] * 3 + [1086] * 4
    + [1097] * 2 + [1120] * 4 + [1121] + [1122] * 4 + [1145] * 4 + [1159]
    + [1182] + [1189] * 4 + [1223] * 2 + [1225] * 2 + [1227] * 4 + [1229] * 4
)

DECK_CONFIGS = {
    "grimmsnarl": {
        "deck": GRIMM_DECK,
        # Prefer a body that evolves into the 320hp wall up front.
        "setup_active": {646: 100, 112: 60, 860: 40},
        # Desire to have these in play; counted per evolution line.
        "bench_value": {646: 100, 112: 80, 860: 50},
        "bench_cap": {646: 3, 112: 2, 860: 2},
        # base id -> line members (any counts toward the cap of the base id)
        "lines": {646: (646, 647, 648), 860: (860, 104), 112: (112,)},
        "evo_top": {648: (646, 647), 104: (860,)},
        # Energy routing: ordered (pokemon id, cap of attached energy units).
        # Munkidori first: Punk Up can only feed Marnie's Pokemon, so the
        # manual attach is the only way Adrena-Brain ever turns on.
        "energy_route": [(112, 1), (648, 3), (647, 2), (646, 2), (104, 1)],
        "tool_route": {1161: {648: 100, 112: 40, 104: 30, 647: 20}},
        # who we want active / promoted (higher = better attacker/tank)
        # Impidimp promotes ahead of support mons: it evolves into the wall
        "attacker_value": {648: 100, 646: 45, 647: 42, 112: 22, 104: 25, 860: 8},
        # base desirability of drawing/fetching each card into hand
        "fetch_prio": {
            648: 90, 1079: 85, 646: 80, 647: 55, 112: 60, 7: 40,
            1086: 70, 1259: 65, 1219: 55, 1182: 45, 1227: 40, 1231: 60,
            1097: 30, 1152: 35, 1161: 25, 104: 30, 860: 25, 1080: 30,
        },
        "attack_bonus": {937: 60, 934: 5},   # Shadow Bullet bench snipe
        "bench_snipe": {937: 30},
        "ability_tech": {860},               # Froslass line only vs ability decks
        "supporter_base": {1219: 7000, 1231: 7500, 1227: 6000, 1182: -1},
        # per-archetype determinized-search gate (set empirically)
        "search_matchup": {"default": True},
    },
    "starmie": {
        "deck": STARMIE_DECK,
        # Cinderace's Explosiveness lets it start active: 160hp, free retreat.
        "setup_active": {666: 100, 1030: 10},
        "bench_value": {1030: 100, 666: 40},
        "bench_cap": {1030: 3, 666: 1},
        "lines": {1030: (1030, 1031), 666: (666,)},
        "evo_top": {1031: (1030,)},
        # Nebula Beam is CCC, Jetting Blow W. Charge bench Staryu FIRST:
        # evolution keeps energy, so the next Mega arrives Nebula-ready.
        # (plan.needs_energy still redirects an attach to the active Mega
        # whenever that single attach unlocks a better attack right now.)
        "energy_route": [(1030, 3), (1031, 3), (666, 1)],
        "tool_route": {1159: {1031: 100, 1030: 30, 666: 20}},
        "attacker_value": {1031: 100, 666: 50, 1030: 15},
        "fetch_prio": {
            1031: 90, 1030: 75, 1145: 70, 1189: 80, 17: 65, 3: 45,
            1086: 65, 1225: 60, 1227: 40, 1229: 35, 1122: 35, 1121: 40,
            1120: 30, 1097: 30, 1182: 45, 1159: 45, 1223: 15, 666: 20,
        },
        "attack_bonus": {965: 80, 1487: 40, 1488: 20},  # Turbo Flare accel
        "bench_snipe": {1487: 50},
        "jit_evolve": {1031},  # evolve Mega Starmie just-in-time, active only
        "supporter_base": {1225: 7000, 1227: 6000, 1223: 3000, 1182: -1},
        # per-archetype determinized-search gate (set empirically, 800g):
        # lucario +4.7pts with search ON; grimmsnarl -2.9 with ON -> OFF;
        # starmie mirror a wash (+0.6) -> keep ON.
        "search_matchup": {"default": True, "grimmsnarl": False},
    },
}
