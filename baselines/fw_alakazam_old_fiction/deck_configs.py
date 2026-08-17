"""Deck config for the Alakazam 'Powerful Hand' draw-engine deck.

Mined as the dominant ladder Alakazam list (2026-07-05..07): 47/70-win
exact list. Win condition: evolve Abra(741)->Kadabra(742)->Alakazam(743)
(Rare Candy 1079 skips Kadabra), stack a huge hand via the Psychic Draw
abilities + Dudunsparce Run Away Draw + Poke Pad / Dawn / Hilda, then swing
Alakazam's Powerful Hand (1 P energy, 20 damage per card in hand) for a
one-shot. Telepath Psychic Energy (19) attaches to a P Pokemon and searches
2 basics to bench. Enhanced Hammer (1081) strips opponent special energy
(hurts Grimmsnarl's dark, Starmie's Ignition, etc.).

This config is interpreted by the generic policy in main.py (identical to
baselines/fw_grimm); only the Grimmsnarl and Starmie configs shipped there,
so the Alakazam entry is authored here. Card ids:
  5 Basic P Energy, 13 Enriching Energy, 19 Telepath Psychic Energy,
  66 Dudunsparce, 305 Dunsparce, 741 Abra, 742 Kadabra, 743 Alakazam,
  140 Fezandipiti ex (n/a here), 1079 Rare Candy, 1081 Enhanced Hammer,
  1086 Buddy-Buddy Poffin, 1097 Night Stretcher, 1129 Sacred Ash,
  1152 Poke Pad, 1182 Boss's Orders, 1184 Lana's Aid, 1225 Hilda,
  1231 Dawn, 1264 Battle Cage.
"""

ALAKAZAM_DECK = (
    [19] * 4 + [305] * 4 + [741] * 4 + [742] * 4 + [743] * 4 + [1079] * 4
    + [1081] * 4 + [1086] * 4 + [1152] * 4 + [1225] * 4 + [1231] * 4
    + [5] * 3 + [66] * 3 + [1097] * 3 + [1182] * 3
    + [13] + [1129] + [1184] + [1264]
)
assert len(ALAKAZAM_DECK) == 60

DECK_CONFIGS = {
    "alakazam": {
        "deck": ALAKAZAM_DECK,
        # Lead with Abra (evolves into the attacker); Dunsparce is a fine
        # turn-1 body (Trading Places to cycle) if no Abra.
        "setup_active": {741: 100, 305: 40},
        # Want the Abra line + one Dunsparce draw body benched.
        "bench_value": {741: 100, 743: 90, 305: 60},
        "bench_cap": {741: 3, 305: 2},
        "lines": {741: (741, 742, 743), 305: (305, 66)},
        # Rare Candy / natural evo top -> the pre-evolutions it can sit on.
        "evo_top": {743: (741, 742), 66: (305,)},
        # Single P energy on Alakazam turns on Powerful Hand; feed the active
        # attacker line first, then Kadabra as a backup Super Psy Bolt body.
        "energy_route": [(743, 1), (742, 1), (741, 1)],
        # No relevant Pokemon-tools in this list.
        "tool_route": {},
        # Alakazam is the whole win condition; Kadabra a chip backup.
        "attacker_value": {743: 100, 742: 30, 741: 8, 66: 5, 305: 5},
        "fetch_prio": {
            743: 92, 1079: 90, 741: 85, 742: 55, 19: 70, 5: 45,
            1086: 72, 305: 55, 66: 40, 1225: 60, 1231: 60, 1152: 45,
            1182: 48, 1081: 50, 1097: 30, 1184: 35, 1129: 20, 1264: 15,
            13: 55,
        },
        # Kadabra's Super Psy Bolt (chip 30) and Alakazam's Powerful Hand.
        "attack_bonus": {},
        "bench_snipe": {},
        # Draw-engine deck: keep hand fat before swinging. Hilda/Dawn draw,
        # Boss's Orders as the closer. -1 keeps Boss out of the auto-play line.
        "supporter_base": {1225: 7500, 1231: 7000, 1184: 4000, 1182: -1},
        # Determinized search helps this combo deck line up evo + energy.
        "search_matchup": {"default": True},
    },
}
