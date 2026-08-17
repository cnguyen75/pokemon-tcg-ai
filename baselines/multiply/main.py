"""Port of the public 945-LB "MultiPly Beam Search Agent" notebook
(kaggle.com/code/aristophanivan/multiply-agent-best-940-lb, Apache 2.0).

Used as a local sparring baseline. The notebook's SEARCH_ALGO calls
search_begin(sbi) with a signature our cg build doesn't accept; it already
wraps everything in try/except and falls back to AdvancedPolicy, which is
where the strength lives.
"""
from __future__ import annotations
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

AGENT_DIR = "/kaggle_simulations/agent"
if not os.path.isdir(os.path.join(AGENT_DIR, "cg")):
    try:
        AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        AGENT_DIR = os.getcwd()
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

from cg.api import (
    AreaType, Card, CardType, EnergyType, Observation, OptionType,
    Pokemon, SelectContext, all_card_data, to_observation_class,
)

_SEARCH_OK = False
try:
    from cg.api import search_begin, search_step, search_end
    _SEARCH_OK = True
except Exception:
    pass

USE_SEARCH = True
SEARCH_TIME_BUDGET = 1.5
SEARCH_MAX_CANDIDATES = 6
BEAM_WIDTH = 3

DECK_PATH = os.path.join(AGENT_DIR, "deck.csv")
if not os.path.exists(DECK_PATH):
    DECK_PATH = "deck.csv"
with open(DECK_PATH, "r", encoding="utf-8") as f:
    my_deck = [int(line) for line in f.read().splitlines() if line.strip()]
DECK = my_deck


class C:
    KYOGRE, SNOVER, MEGA_ABOMASNOW_EX = 721, 722, 723
    MAKUHITA, HARIYAMA = 673, 674
    LUNATONE, SOLROCK = 675, 676
    RIOLU, MEGA_LUCARIO_EX = 677, 678
    BASIC_FIGHTING_ENERGY = 6
    DUSK_BALL, SWITCH, PREMIUM_POWER_PRO, FIGHTING_GONG = 1102, 1123, 1141, 1142
    POKE_PAD, HERO_CAPE, BOSS_ORDERS = 1152, 1159, 1182
    CARMINE, LILLIE_DETERMINATION, GRAVITY_MOUNTAIN = 1192, 1227, 1252
    LUMIOSE_CITY, LILLIES_PEARL, LEGACY_ENERGY = 1267, 1172, 12


MEGA_BRAVE = 983
LOW_DECK_COUNT = 10

all_card = all_card_data()
card_table = {card.cardId: card for card in all_card}


class AttackPlan:
    def __init__(self, attacker=-1, target=-1, attack_index=-1, remain_hp=-1, needs_energy=False):
        self.attacker, self.target = attacker, target
        self.attack_index, self.remain_hp = attack_index, remain_hp
        self.needs_energy = needs_energy


plan = AttackPlan()
pre_turn = -1
ability_used = False


def get_card(obs: Observation, area: AreaType, index: int, player_index: int):
    player = obs.current.players[player_index]
    if area == AreaType.DECK: return obs.select.deck[index]
    if area == AreaType.HAND: return player.hand[index]
    if area == AreaType.DISCARD: return player.discard[index]
    if area == AreaType.ACTIVE: return player.active[index]
    if area == AreaType.BENCH: return player.bench[index]
    if area == AreaType.PRIZE: return player.prize[index]
    if area == AreaType.STADIUM: return obs.current.stadium[index]
    if area == AreaType.LOOKING: return obs.current.looking[index]
    return None


def prize_count(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == C.LEGACY_ENERGY: count -= 1
    for card in pokemon.tools:
        if card.id == C.LILLIES_PEARL and "Lillie" in data.name: count -= 1
    return max(0, count)


def target_score(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 2000 + len(pokemon.energies) * 300 + len(pokemon.tools) * 200
    if data.stage2: score += 500
    elif data.stage1: score += 250
    if pokemon.id in {144, 322, 323, 337}: score -= 200
    if pokemon.id == C.SNOVER: score += 950
    elif pokemon.id == C.MEGA_ABOMASNOW_EX: score += 250
    if pokemon.id == C.RIOLU: score += 800
    elif pokemon.id == C.MEGA_LUCARIO_EX: score += 100
    return score + pokemon.hp


class AdvancedPolicy:
    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.has_ready_lucario_line = False
        self.has_ready_hariyama_line = False
        self.can_switch, self.can_gust, self.can_attack, self.can_use_mega_brave = False, False, False, False
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0

        self._count_cards()
        self._scan_main_options()

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0: return []
        if self.context == SelectContext.MAIN: self._plan_attack()
        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        self._remember_lunatone_ability(ranked)
        return ranked[: self.select.maxCount]

    def _count_cards(self) -> None:
        for pokemon in self.me.active + self.me.bench:
            if pokemon is None: continue
            self.field_counts[pokemon.id] += 1
            if pokemon.id in {C.MAKUHITA, C.HARIYAMA} and len(pokemon.energies) >= 3: self.has_ready_hariyama_line = True
            if pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX} and len(pokemon.energies) >= 2: self.has_ready_lucario_line = True
        for card in self.me.hand: self.hand_counts[card.id] += 1
        for card in self.me.discard: self.discard_counts[card.id] += 1

    def _scan_main_options(self) -> None:
        if self.context != SelectContext.MAIN: return
        for option in self.select.option:
            if option.type == OptionType.PLAY:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.SWITCH: self.can_switch = True
                elif card.id == C.BOSS_ORDERS: self.can_gust = True
            elif option.type == OptionType.EVOLVE:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.HARIYAMA: self.can_gust = True
            elif option.type == OptionType.RETREAT: self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True
                if option.attackId == MEGA_BRAVE: self.can_use_mega_brave = True

    def _my_board(self): return self.me.active + self.me.bench
    def _opponent_board(self): return self.opponent.active + self.opponent.bench
    def _opponent_has(self, ids): return any(p is not None and p.id in ids for p in self._opponent_board())
    def _opponent_is_water_deck(self): return self._opponent_has({C.KYOGRE, C.SNOVER, C.MEGA_ABOMASNOW_EX})
    def _opponent_is_crustle_wall(self): return self._opponent_has({344, 345})

    def _can_evolve_board_index(self, board_index: int) -> bool:
        for option in self.select.option:
            if option.type != OptionType.EVOLVE: continue
            target_index = option.inPlayIndex + (1 if option.inPlayArea == AreaType.BENCH else 0)
            if target_index == board_index: return True
        return False

    def _base_attack(self, pokemon: Pokemon, attack_index: int):
        energy_required, base_damage, base_score = 0, 0, 0
        if pokemon.id == C.MEGA_LUCARIO_EX:
            if attack_index == 0:
                energy_required, base_damage = 1, 130
                base_score += 60 * min(3, self.discard_counts[C.BASIC_FIGHTING_ENERGY])
            else:
                energy_required, base_damage = 2, 270
            if self._opponent_is_water_deck() and len(self.opponent.prize) <= 3: base_score -= 500
        elif attack_index == 1: return None
        elif pokemon.id == C.HARIYAMA: energy_required, base_damage = 3, 210
        elif pokemon.id == C.MAKUHITA: return None
        elif pokemon.id == C.SOLROCK and self.field_counts[C.LUNATONE] >= 1: energy_required, base_damage = 1, 70
        if base_damage <= 0: return None
        return energy_required, base_damage, base_score

    def _base_attack_after_evolution(self, pokemon: Pokemon, board_index: int, attack_index: int):
        if pokemon.id == C.MAKUHITA and attack_index == 0 and self._can_evolve_board_index(board_index): return 3, 210, -100
        return self._base_attack(pokemon, attack_index)

    def _plan_attack(self) -> None:
        global plan
        best_score = -1
        plan = AttackPlan()
        if self.state.turn < 2: return

        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None: continue
            if attacker_index != 0 and not self.can_switch: break

            for attack_index in range(2):
                attack = self._base_attack_after_evolution(my_pokemon, attacker_index, attack_index)
                if attack is None: continue
                energy_required, base_damage, base_score = attack
                energy_count = len(my_pokemon.energies)
                if attack_index == 1 and attacker_index == 0 and energy_count >= 2 and not self.can_use_mega_brave: break
                needs_energy = False
                if energy_count < energy_required:
                    if self.hand_counts[C.BASIC_FIGHTING_ENERGY] >= 1 and not self.state.energyAttached:
                        energy_count += 1
                        needs_energy = energy_count >= energy_required
                    if not needs_energy: continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None: continue
                    if target_index != 0 and not self.can_gust: break
                    if self._opponent_is_crustle_wall() and my_pokemon.id == C.MEGA_LUCARIO_EX and op_pokemon.id == 345: continue

                    damage = base_damage
                    op_data = card_table[op_pokemon.id]
                    if op_data.weakness == EnergyType.FIGHTING: damage *= 2
                    elif op_data.resistance == EnergyType.FIGHTING: damage -= 30

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0: score *= damage / op_pokemon.hp
                    if len(self.opponent.prize) <= prize: score = 500000

                    score += base_score + (220 if attacker_index == 0 else 0) + (300 if target_index == 0 else 0) + energy_count
                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(attacker_index, target_index, attack_index, op_pokemon.hp - damage, needs_energy)

    def _energy_target_score(self, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)
        if pokemon.id in {C.MAKUHITA, C.HARIYAMA}:
            if pokemon.id == C.HARIYAMA: score += 1
            if self._opponent_is_crustle_wall(): score += 260 if energy_count < 3 else 30
            else:
                score += 100 if energy_count < 3 else 0
                score -= 50 if self.has_ready_hariyama_line else 0
        elif pokemon.id == C.LUNATONE: score -= 100
        elif pokemon.id == C.SOLROCK: score += 20 if energy_count < 1 else -100
        elif pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX}:
            if pokemon.id == C.MEGA_LUCARIO_EX: score += 1
            score += 100 if energy_count < 2 else 0
            score -= 50 if self.has_ready_lucario_line else 0
        return score

    def _score_option(self, option) -> float:
        if option.type == OptionType.NUMBER: return option.number
        if option.type == OptionType.YES: return 100 if self.context == SelectContext.IS_FIRST else 1
        if option.type == OptionType.NO: return 0
        if option.type == OptionType.CARD: return self._score_card_choice(option)
        if option.type == OptionType.PLAY: return self._score_play(option)
        if option.type == OptionType.ATTACH: return self._score_attach(option)
        if option.type == OptionType.EVOLVE: return self._score_evolve(option)
        if option.type == OptionType.ABILITY: return self._score_ability(option)
        if option.type == OptionType.RETREAT: return 2000 if plan.attacker >= 1 else -1
        if option.type == OptionType.ATTACK:
            if (self._opponent_is_crustle_wall() and self.me.active and self.opponent.active
                    and self.me.active[0].id == C.MEGA_LUCARIO_EX and self.opponent.active[0].id == 345
                    and plan.target < 0):
                return -1
            return 1100 if (option.attackId == MEGA_BRAVE) == (plan.attack_index == 1) else 1000
        return 0

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None: return 0
        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}: return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return 2 if card.id == C.SOLROCK and self.state.firstPlayer == self.my_index else 4 if card.id == C.SOLROCK else 3 if card.id == C.RIOLU else 1 if card.id == C.MAKUHITA else 0
        if self.context == SelectContext.TO_HAND:
            score = 200 - self.hand_counts[card.id] * 100
            if card.id == C.MAKUHITA:
                score += (80 if self.field_counts[card.id] < 2 else -20) if self._opponent_is_crustle_wall() else (-10 if self.field_counts[card.id] >= 1 else 10)
            elif card.id == C.HARIYAMA:
                score += (120 if self.field_counts[C.MAKUHITA] >= 1 else -5) if self._opponent_is_crustle_wall() else (20 if self.field_counts[C.MAKUHITA] >= 1 else -20)
            elif card.id == C.LUNATONE: score += -250 if self.field_counts[card.id] >= 1 else 60
            elif card.id == C.SOLROCK: score += -250 if self.field_counts[card.id] >= 1 else 50
            elif card.id == C.RIOLU:
                score += -150 if (self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2) else -3 if (self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 1) else 40
            elif card.id == C.MEGA_LUCARIO_EX: score += 40 if self.field_counts[C.RIOLU] >= 1 else -15
            elif card.id == C.BASIC_FIGHTING_ENERGY: score += 30 if not ability_used or not self.state.energyAttached else -1
            return score
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        return 0

    def _score_active_choice(self, option, card) -> float:
        if not isinstance(card, Pokemon): return 0
        if option.playerIndex != self.my_index: return 100 if option.index == plan.target - 1 else 0
        score = len(card.energies) * 2
        if option.index == plan.attacker - 1: score += 100
        if card.id == C.MEGA_LUCARIO_EX: score += 8 if self._opponent_is_water_deck() and len(self.opponent.prize) <= 3 else 20
        elif card.id == C.HARIYAMA and len(card.energies) >= 2: score += 45 if self._opponent_is_crustle_wall() else 15
        elif card.id == C.MAKUHITA and len(card.energies) >= 2: score += 35 if self._opponent_is_crustle_wall() else 10
        elif card.id == C.SOLROCK: score += 5
        elif card.id == C.RIOLU: score += 4
        return score

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        data = card_table[card.id]
        if data.cardType == CardType.POKEMON:
            if card.id in {C.LUNATONE, C.SOLROCK} and self.field_counts[card.id] >= 1: return -1
            if card.id == C.RIOLU and self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2: return -1
            return 20000
        if card.id == C.SWITCH: return 6000 if plan.attacker > 0 else -1
        if card.id == C.PREMIUM_POWER_PRO:
            if self.state.supporterPlayed and plan.remain_hp <= 0: return -1
            if not self.can_attack:
                return 3050 if (not self.state.supporterPlayed and self.hand_counts[C.CARMINE] > 0 and self.hand_counts[C.LILLIE_DETERMINATION] == 0 and not self.me.deckCount <= LOW_DECK_COUNT) else -1
            return 5000
        if card.id == C.BOSS_ORDERS: return 3200 if plan.target >= 1 else -1
        if card.id == C.CARMINE: return -1 if self.me.deckCount <= LOW_DECK_COUNT else 3000
        if card.id == C.LILLIE_DETERMINATION: return -1 if self.me.deckCount <= LOW_DECK_COUNT else 3100
        if card.id == C.GRAVITY_MOUNTAIN:
            return 3500 if any(p is not None and card_table[p.id].stage2 for p in self._opponent_board()) else (1200 if self.stadium_id else -1)
        return 10000

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon): return 0
        if card.id == C.HERO_CAPE:
            score = 7000
            if self._opponent_is_water_deck():
                return 12200 if pokemon.id == C.RIOLU else 12800 if pokemon.id == C.MEGA_LUCARIO_EX else score
            if pokemon.id == C.RIOLU: score += 100
            elif pokemon.id == C.MEGA_LUCARIO_EX: score += 200
            return score
        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if board_index == plan.attacker and plan.needs_energy: score += 200
        return score

    def _score_evolve(self, option) -> float:
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon): return 0
        if pokemon.id == C.MAKUHITA and plan.target == 0 and not self._opponent_is_crustle_wall(): return -1
        return 9000 + len(pokemon.energies)

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card.id == C.LUMIOSE_CITY: return 1
        if card.id == C.LUNATONE and self.me.deckCount <= LOW_DECK_COUNT: return -1
        return 30000

    def _remember_lunatone_ability(self, ranked) -> None:
        global ability_used
        if self.context != SelectContext.MAIN or not ranked: return
        option = self.select.option[ranked[0]]
        if option.type != OptionType.ABILITY: return
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is not None and card.id == C.LUNATONE: ability_used = True


def agent(obs_dict: dict) -> list[int]:
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return my_deck if obs_dict.get("select") is None else [0]
    if obs.select is None: return my_deck

    global pre_turn, ability_used, plan
    if pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        ability_used = False
        plan = AttackPlan()

    try:
        ordered = AdvancedPolicy(obs).choose()
        n = len(obs.select.option)
        ordered = [i for i in ordered if 0 <= i < n]
        if not ordered:
            return list(range(min(max(1, obs.select.minCount), n)))
        k = max(min(obs.select.maxCount, n), min(max(1, obs.select.minCount), n))
        return ordered[:k]
    except Exception:
        n = len(obs.select.option)
        return list(range(min(max(1, obs.select.minCount), n)))
