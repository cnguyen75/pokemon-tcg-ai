"""Deck config for the Cynthia's Garchomp ex deck (rising, ~58% ladder WR).

Mined as the dominant ladder Garchomp list (2026-07-05..07): 35/65-win exact
list. Win condition: evolve Cynthia's Gible(379)->Gabite(380)->Garchomp
ex(381, 330hp) and hit Draconic Buster (FF, 260, discards energy) or the
cheaper Corkscrew Dive (F, 100 + draw). Cynthia's Roserade(342) gives all
Cynthia's attacks +30 (Cheer On to Glory). Gabite's Champion's Call searches
any Cynthia's Pokemon; Fighting Gong(1142) fetches Basic F energy or F basic;
Power Weight(1173) is a +70hp tool; Rock Fighting Energy(20) is a protective
special F energy.

Interpreted by the generic policy in main.py (identical to fw_grimm). Card ids:
  6 Basic F Energy, 20 Rock Fighting Energy, 341 Cynthia's Roselia,
  342 Cynthia's Roserade, 379 Cynthia's Gible, 380 Cynthia's Gabite,
  381 Cynthia's Garchomp ex, 387 Cynthia's Spiritomb, 1080 Unfair Stamp,
  1086 Buddy-Buddy Poffin, 1097 Night Stretcher, 1142 Fighting Gong,
  1152 Poke Pad, 1173 Cynthia's Power Weight, 1182 Boss's Orders,
  1197 Xerosic's Machinations, 1203 Surfer, 1225 Hilda,
  1227 Lillie's Determination, 1261 Forest of Vitality.
"""

GARCHOMP_DECK = (
    [6] * 5 + [379] * 4 + [341] * 4 + [1227] * 4 + [1152] * 4 + [1086] * 4
    + [1142] * 4 + [20] * 4 + [380] * 4 + [381] * 3 + [342] * 3 + [1225] * 3
    + [1173] * 3 + [387] * 2 + [1182] * 2 + [1097] * 2 + [1261] * 2
    + [1203] + [1197] + [1080]
)
assert len(GARCHOMP_DECK) == 60

DECK_CONFIGS = {
    "cynthia_garchomp": {
        "deck": GARCHOMP_DECK,
        # Lead with Gible (evolves into the 330hp attacker).
        "setup_active": {379: 100, 341: 30},
        # Want the Gible line + a Roselia (for the +30 Roserade buff) benched.
        "bench_value": {379: 100, 381: 90, 341: 70, 342: 60},
        "bench_cap": {379: 3, 341: 2},
        "lines": {379: (379, 380, 381), 341: (341, 342)},
        "evo_top": {381: (379, 380), 342: (341,)},
        # Garchomp wants FF (Draconic Buster); Gabite/Gible fine on 1 F.
        "energy_route": [(381, 2), (380, 1), (379, 1), (342, 2)],
        "tool_route": {1173: {381: 100, 380: 40, 342: 30}},
        "attacker_value": {381: 100, 380: 35, 342: 30, 379: 8, 341: 5},
        "fetch_prio": {
            381: 92, 380: 80, 379: 82, 6: 45, 20: 60, 1142: 55,
            1086: 72, 342: 65, 341: 60, 1225: 60, 1227: 62, 1152: 45,
            1173: 50, 1182: 48, 1097: 30, 387: 25, 1261: 30, 1080: 40,
            1203: 30, 1197: 30,
        },
        "attack_bonus": {},
        "bench_snipe": {},
        # Lillie's Determination (big draw) and Hilda dig; Boss is the closer.
        "supporter_base": {1225: 7000, 1227: 7500, 1203: 3000, 1182: -1},
        "search_matchup": {"default": True},
    },
}
