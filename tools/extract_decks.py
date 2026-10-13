"""Extract full decklists from episode replay JSONs, dedupe, report by team/result."""
import json, glob, csv, sys
from collections import Counter, defaultdict

import os
CARD_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'EN_Card_Data.csv')
names = {}
for r in csv.DictReader(open(CARD_CSV)):
    names.setdefault(int(r['Card ID']), r['Card Name'])

def deck_key(deck): return tuple(sorted(deck))

decks = {}                      # key -> deck list
stats = defaultdict(lambda: [0,0])  # key -> [wins, games]
teams_by_deck = defaultdict(set)

for f in glob.glob(sys.argv[1] + '/*.json'):
    d = json.load(open(f))
    team = d['info']['TeamNames']
    rew = d['rewards']
    deck_action = d['steps'][0][0]['visualize'][0]['action']
    for i in range(2):
        k = deck_key(deck_action[i])
        decks[k] = list(deck_action[i])
        stats[k][1] += 1
        if rew[i] == 1: stats[k][0] += 1
        teams_by_deck[k].add(team[i])

for k, (w, g) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
    c = Counter(decks[k])
    poke_line = ', '.join(f"{n}x {names.get(i,'?')}" for i,n in sorted(c.items()))
    print(f"\n=== {w}/{g} wins | teams: {sorted(teams_by_deck[k])[:4]}")
    print(f"  {poke_line}")
    print(f"  ids: {sorted(decks[k])}")
