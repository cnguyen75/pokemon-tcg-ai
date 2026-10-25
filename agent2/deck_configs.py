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
    [7] * 10
    + [104] * 2
    + [112] * 4
    + [646] * 4
    + [647] * 3
    + [648] * 3
    + [860] * 2
    + [1079] * 3
    + [1080]
    + [1086] * 4
    + [1097] * 3
    + [1152] * 4
    + [1161] * 2
    + [1182] * 2
    + [1219] * 4
    + [1227] * 4
    + [1231]
    + [1259] * 4
)

STARMIE_DECK = (
    [3] * 9
    + [17] * 4
    + [666] * 4
    + [1030] * 4
    + [1031] * 4
    + [1086] * 4
    + [1097] * 2
    + [1120] * 4
    + [1121]
    + [1122] * 4
    + [1145] * 3
    + [1159]
    + [1182] * 2
    + [1189] * 4
    + [1225] * 2
    + [1227] * 4
    + [1229] * 4
)

# Archaludon ex (metal tempo), 2026-07-09: EXACT dominant ladder list from the
# 750-800 band (5 of 12 v7 Archaludon losses used this precise 60; the rest
# were 1-3 card variants). 8 metal energy, 57 Relicanth (Memory Dive ->
# Raging Hammer OHKO tech), 169 Duraludon, 190 Archaludon ex, 666 Cinderace
# (Turbo Flare engine), 1147 Jumbo Ice Cream, 1244 Full Metal Lab.
ARCHALUDON_DECK = (
    [8] * 11
    + [57]
    + [169] * 4
    + [190] * 4
    + [666] * 4
    + [1097] * 3
    + [1121] * 4
    + [1122] * 4
    + [1147] * 4
    + [1152] * 4
    + [1159]
    + [1182] * 4
    + [1185] * 4
    + [1227] * 4
    + [1244] * 4
)

# Crustle/Mega Kangaskhan wall, 2026-07-16: EXACT dominant list from the daily
# top-episode dump (54 games, 4 teams). Crustle's Mysterious Rock Inn blanks
# all damage from ex attackers (Nebula Beam pierces it — noTargetEffect);
# Mega Kangaskhan draws 2/turn active, Rapid-Fire Combo 200+coins. Sustain
# shell: 4x Jumbo Ice Cream, Cape, Mist/Spiky Energy, 4x Xerosic hand-cut.
KANGA_DECK = (
    [1]
    + [11] * 4
    + [14] * 4
    + [18] * 4
    + [343]
    + [344] * 4
    + [345] * 4
    + [756] * 4
    + [1086] * 4
    + [1087]
    + [1122] * 4
    + [1123] * 4
    + [1147] * 4
    + [1159]
    + [1182] * 2
    + [1197] * 4
    + [1225] * 4
    + [1227] * 4
    + [1264] * 2
)

# Dragapult ex, 2026-07-19: LumenLiquidity's rising top-pool list (18g 56%;
# archetype 66.7% and growing — dominant in paper TCG per Colin, porting in).
# Phantom Dive (aid 154): 200 + 6 counters spread on our BENCH — kills the
# benched Staryu rebuild line. 2-prize attacker vs our 3-prize Mega.
DRAGAPULT_DECK = (
    [2] * 4
    + [5] * 3
    + [7]
    + [112]
    + [119] * 4
    + [120] * 4
    + [121] * 3
    + [131] * 2
    + [132] * 2
    + [133]
    + [140]
    + [235]
    + [791]
    + [1071]
    + [1080]
    + [1086] * 4
    + [1097] * 2
    + [1120] * 3
    + [1121] * 4
    + [1152] * 4
    + [1182] * 3
    + [1198] * 3
    + [1227] * 4
    + [1231]
    + [1256] * 2
)

DECK_CONFIGS = {
    "dragapult": {
        "deck": DRAGAPULT_DECK,
        "setup_active": {119: 100, 235: 40},
        "bench_value": {119: 100, 131: 40, 112: 20},
        "bench_cap": {119: 3, 131: 2, 112: 1},
        "lines": {119: (119, 120, 121), 131: (131, 132, 133), 112: (112,)},
        "evo_top": {120: (119,), 121: (120,), 132: (131,), 133: (132,)},
        # Phantom Dive is R+P; Crispin (1198) accelerates
        "energy_route": [(121, 2), (120, 2), (133, 3)],
        "attacker_value": {121: 100, 120: 40, 133: 35, 119: 10},
        "fetch_prio": {
            121: 90,
            120: 80,
            119: 70,
            1086: 70,
            1121: 55,
            1198: 60,
            2: 45,
            5: 45,
            1227: 40,
            1182: 45,
            1152: 35,
            1097: 30,
        },
        "attack_bonus": {154: 80},  # Phantom Dive: spread wins races
        "bench_snipe": {154: 60},
        "supporter_base": {1198: 6500, 1227: 6000, 1231: 5500, 1182: -1},
        "search_matchup": {"default": True},
    },
    "kanga": {
        "deck": KANGA_DECK,
        "setup_active": {344: 100, 343: 20},
        "bench_value": {344: 100, 756: 70, 343: 10},
        "bench_cap": {344: 3, 756: 2, 343: 1},
        "lines": {344: (344, 345), 756: (756,), 343: (343,)},
        "evo_top": {345: (344,)},
        # Superb Scissors is G+CC (only Grow Grass/Basic G pay the G);
        # Rapid-Fire Combo is CCC so Mist/Spiky route to Kangaskhan.
        "energy_route": [(345, 3), (756, 3), (344, 2)],
        "tool_route": {1159: {756: 100, 345: 80}},
        "attacker_value": {345: 100, 756: 90, 344: 15, 343: 10},
        "fetch_prio": {
            345: 90,
            756: 80,
            344: 75,
            1086: 70,
            1225: 60,
            1147: 55,
            1197: 50,
            18: 45,
            1182: 45,
            1159: 45,
            11: 40,
            14: 40,
            1227: 40,
            1123: 35,
            1122: 35,
        },
        "supporter_base": {1225: 6500, 1227: 6000, 1197: 6800, 1182: -1},
        "search_matchup": {"default": True},
    },
    # framework pilot of the ladder Archaludon list (baselines/fw_archaludon,
    # 2026-07-09): plan = open Cinderace, Turbo Flare energy onto benched
    # Duraludon, evolve into Archaludon ex (Assemble Alloy re-attaches 2 M
    # from the discard, so pitching energy to Ultra Ball is fine), wall
    # behind Full Metal Lab + Jumbo Ice Cream, Metal Defender 2HKOs.
    "archaludon": {
        "deck": ARCHALUDON_DECK,
        "setup_active": {666: 100, 169: 50, 57: 10},
        "bench_value": {169: 100, 57: 60, 666: 30},
        "bench_cap": {169: 3, 57: 1, 666: 1},
        "lines": {169: (169, 190), 57: (57,), 666: (666,)},
        "evo_top": {190: (169,)},
        # Cinderace first (1 energy = Turbo Flare online), then Duraludons to
        # 3 so each evolves into a Metal Defender-ready Archaludon ex.
        "energy_route": [(666, 1), (190, 3), (169, 3), (57, 1)],
        # don't evolve into Archaludon ex until it can fight: attached energy
        # + up to 2 Metal recovered from the discard by Assemble Alloy >= 3
        # (the ladder pilots charge Duraludon/discard FIRST, then evolve into
        # an immediate Metal Defender; evolving at 0 energy wastes the Alloy)
        "evolve_wait": {190: 3},
        "alloy_tops": {190},
        "alloy_energy": 8,
        "tool_route": {1159: {190: 100, 169: 40}},
        "attacker_value": {190: 100, 169: 45, 666: 35, 57: 5},
        "fetch_prio": {
            190: 90,
            169: 80,
            8: 50,
            666: 45,
            57: 50,
            1147: 55,
            1244: 60,
            1185: 55,
            1121: 45,
            1122: 35,
            1152: 45,
            1097: 35,
            1182: 45,
            1227: 40,
            1159: 40,
        },
        # 965 Turbo Flare gets the generic bench-deficit bonus; 253 Metal
        # Defender is the bread-and-butter; 224 Raging Hammer damage scaling
        # is computed in eff_damage (RAGE_ATTACKS).
        "attack_bonus": {965: 80, 253: 10},
        "supporter_base": {1185: 7200, 1227: 6000, 1182: -1},
        "discard_energy_ok": True,  # Assemble Alloy feeds from the discard
        "search_matchup": {"default": True},
    },
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
            648: 90,
            1079: 85,
            646: 80,
            647: 55,
            112: 60,
            7: 40,
            1086: 70,
            1259: 65,
            1219: 55,
            1182: 45,
            1227: 40,
            1231: 60,
            1097: 30,
            1152: 35,
            1161: 25,
            104: 30,
            860: 25,
            1080: 30,
        },
        "attack_bonus": {937: 60, 934: 5},  # Shadow Bullet bench snipe
        "bench_snipe": {937: 30},
        "ability_tech": {860},  # Froslass line only vs ability decks
        "supporter_base": {1219: 7000, 1231: 7500, 1227: 6000, 1182: -1},
        # per-archetype determinized-search gate (set empirically)
        "search_matchup": {"default": True},
    },
    "starmie": {
        "deck": STARMIE_DECK,
        # Cinderace's Explosiveness lets it start active: 160hp, free retreat.
        "setup_active": {666: 100, 1030: 10},
        "bench_value": {1030: 100, 666: 40},
        # cap 2: the #1 pilot never runs 3 Staryu lines; the 3rd bench slot
        # is worth more as gust-safety / Cinderace room (autopsy fix 3)
        "bench_cap": {1030: 2, 666: 1},
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
            1031: 90,
            1030: 75,
            1145: 70,
            1189: 55,
            17: 65,
            3: 45,
            1086: 65,
            1225: 60,
            1227: 40,
            1229: 35,
            1122: 35,
            1121: 40,
            1120: 30,
            1097: 30,
            1182: 45,
            1159: 45,
            666: 20,
        },
        "attack_bonus": {965: 80, 1487: 40, 1488: 20},  # Turbo Flare accel
        "bench_snipe": {1487: 50},
        # jit_evolve now = "prefer active-evolve first" (active 9520 > bench
        # 8800), NOT bench suppression: the old 540 bench score made planned
        # ATTACK (900) outrank bench-evolve and permanently forfeited benched
        # Staryu -> Mega lines (autopsy fix 1: all 53 EVOLVE divergences).
        "jit_evolve": {1031},
        "supporter_base": {1225: 7000, 1227: 6000, 1223: 3000, 1182: -1},
        # per-archetype determinized-search gate (set empirically, 800g):
        # lucario +4.7pts with search ON; grimmsnarl -2.9 with ON -> OFF;
        # starmie mirror a wash (+0.6) -> keep ON.
        # vs lucario1084 baseline (400g): worlds 20 = 41.9%, 60 = 38.9%,
        # OFF = 37.5% -> keep ON at default worlds.
        "search_matchup": {"default": True, "grimmsnarl": False},
        # per-archetype config deltas (dict values merge into the base key).
        # lucario (1084-style Mega Lucario ex): they win by two-hitting our
        # 3-prize Mega Starmie (Mega Brave 270 + Aura Jab 130, Premium Power
        # Pro pushes 270 -> 300/330 = OHKO). The matchup cards are Hero's
        # Cape (330 -> 430hp breaks their 2-hit math), Wally's Compassion
        # (undoes a Mega Brave), and Crushing Hammer (Mega Brave needs FF).
        "matchup_overrides": {
            "lucario": {
                "fetch_prio": {1229: 70, 1159: 75, 1120: 55},
                # Wally's Compassion undoes a Mega Brave: keep one banked
                "gear_bank": 1229,
                # search-override margin (win-prob pts/world, x10). The race
                # vs lucario is decided by config discipline (Cape/Wally/
                # promote order): conservative overrides measure better
                # (816g, net on: margin 15 = 42.6%, margin 30 = 44.4%).
                "value_margin": 30,
            },
            # Archaludon ex (metal tempo): 300hp Metal wall behind Full Metal
            # Lab (1244, -30 to our non-piercing attacks) and Jumbo Ice Cream
            # (1147, heal 80). 2026-07-09 autopsy of the 12 fresh 750-800-band
            # ladder losses (episodes/v7games): (1) 27 Jetting Blows vs 18
            # Nebulas -- the 90-after-Lab chip is erased by the heal AND feeds
            # (2) Raging Hammer, which their Archaludon ex borrows from
            # Duraludon via Relicanth's Memory Dive (9 RH hits in the losses,
            # up to -460: every 90 chip we land adds +90 to the OHKO answer);
            # (3) 24 of 70 of our turns had NO attack at all (energy drought:
            # active Mega died with 1-2 energy, Wally returned energy we could
            # only re-attach 1/turn). Nebula Beam (1488, CCC) does 210, pierces
            # the Lab, 2HKOs through one heal (420 > 300+80), and adds only
            # safe RH fuel (80+210 = 290 < 330). Fix: assemble/keep CCC (chip
            # never steals the route attach; basic-before-Ignition ordering in
            # main.py banks Ignition for Wally turns), and gust-KO their
            # 1-prize support (Relicanth kills the RH tech, Cinderace kills
            # the Turbo Flare engine -- GUST_WORTH now includes both).
            # Archaludon ex (metal tempo). 2026-07-10 autopsy of the 12
            # fresh 750-800-band ladder losses (episodes/v7games, 12L-2W):
            # (1) 27 Jetting Blows vs 18 Nebulas -- the 90-after-Full-Metal-
            # Lab chip is erased by Jumbo Ice Cream (80/turn) and feeds
            # (2) Raging Hammer, which their Archaludon ex borrows from
            # Duraludon via Relicanth's Memory Dive (9 RH hits up to -460 =
            # OHKO on our 330 Mega; every counter we place adds +10);
            # (3) 24/70 of our turns had NO attack (energy drought: each dead
            # Mega takes 3 basics with it). Wins deny their Metal Defender
            # (their attacks in wins are mostly 1-energy Hammer In) and
            # sustain Nebula (2.8/game in wins vs 1.4 in losses).
            # Framework support (main.py): RAGE_ATTACKS damage scaling +
            # Memory Dive in plan/threat, GUST_WORTH += Relicanth/Cinderace,
            # basic-before-Ignition attach, Hilda energy-drought fetch.
            # Measured NEGATIVE and reverted (200g each vs fw_archaludon):
            # no_chip {190} (33%), hammer_any (34.5%), Cinderace sac-tank
            # route (37%), route_bench_bonus (38%), attach_pref_special,
            # chip_attach_min 999. Search ON == OFF here; ON kept (default).
            "archaludon": {
                "attack_bonus": {965: 80, 1487: 0, 1488: 55},
                "bench_snipe": {1487: 0},
                "fetch_prio": {3: 58, 17: 70, 1031: 92, 1030: 80, 1120: 55, 1182: 52, 1229: 72},
                # Pokegear digs/banks Hilda: the direct energy fetch is the
                # drought-breaker in this matchup (Nebula cadence is the game)
                "gear_bank": 1229,  # was 1225 (Hilda) -- NOT IN JUDGE DECK, made
                # Pokegear dead here since v11 (7/22)
                # Boss-stall: a 2-turn Jetting kill on a gusted Duraludon /
                # Relicanth / Cinderace beats chipping the healing wall
                "gust_2turn": True,
                # retreat-rotation pays 2 of the Mega's 3 energies to dodge a
                # 2HKO that Wally undoes for free -- keep the energy instead
                "no_2hko_rotate": True,
                "value_margin": 30,
            },
        },
    },
}
