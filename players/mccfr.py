"""MCCFR player: plays according to a pre-trained strategy over abstract actions.

The strategy maps (bucketed_state, abstract_action) -> probability. Abstract
actions like "play tight continuation" or "discard safe" are mapped to concrete
cards at play time.

Set MCCFR_CHECKPOINT env var to the strategy.pkl path, or it defaults to
mccfr-checkpoints/strategy.pkl.
"""

import os
import pickle
import random
from collections import defaultdict

from classes import Player, SUITS
from utils import is_playable, playable_draws, discard_intelligently


_DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'mccfr-checkpoints', 'strategy.pkl')

# Abstract actions (must match train_mccfr.py)
A_PLAY_TIGHT = 'PT'
A_PLAY_LOOSE = 'PL'
A_PLAY_NEW = 'PN'
A_PLAY_CONTRACT = 'PC'
A_DISC_USELESS = 'DU'
A_DISC_SAFE = 'DS'
A_DISC_RISKY = 'DR'
A_DRAW_DECK = 'DD'
A_DRAW_PILE = 'DP'

_PLAY_PRIORITY = [A_PLAY_TIGHT, A_PLAY_LOOSE, A_PLAY_CONTRACT, A_PLAY_NEW,
                  A_DISC_USELESS, A_DISC_SAFE, A_DISC_RISKY]


def _load_strategy(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data.get('strategy', data)


def _bucketed_key(hand, flags, deck_size, me, phase):
    """Compute bucketed info set key (matches train_mccfr.py)."""
    dp = 3 if deck_size > 32 else (2 if deck_size > 20 else (1 if deck_size > 10 else 0))
    opp = 1 - me
    suit_codes = []
    for s in SUITS:
        f = flags[s]
        my_played = f.played[me]
        my_n = len(my_played)
        my_bucket = 0 if my_n == 0 else (1 if my_n <= 3 else 2)
        opp_bucket = 0 if len(f.played[opp]) == 0 else 1
        hand_in_suit = [c for c in hand if c[0] == s]
        hand_bucket = min(len(hand_in_suit), 2)
        has_playable = 0
        for c in hand_in_suit:
            if is_playable(c, my_played):
                has_playable = 1
                break
        code = my_bucket * 12 + opp_bucket * 6 + hand_bucket * 2 + has_playable
        suit_codes.append(code)
    suit_codes.sort()
    return f"{dp}{phase}{''.join(chr(65 + c) for c in suit_codes)}"


def _classify_play(hand, flags, me):
    """Classify hand cards into abstract play/discard actions."""
    cats = defaultdict(list)
    seen = set()
    for card in hand:
        if card in seen:
            continue
        seen.add(card)
        suit = card[0]
        my_played = flags[suit].played[me]
        opp_played = flags[suit].played[1 - me]
        playable = is_playable(card, my_played)

        if playable:
            if card[1] == '0':
                cats[A_PLAY_CONTRACT].append((card, False))
            elif my_played:
                baseline = int(my_played[-1][1])
                gap = int(card[1]) - baseline - 1
                if gap <= 1:
                    cats[A_PLAY_TIGHT].append((card, False))
                else:
                    cats[A_PLAY_LOOSE].append((card, False))
            else:
                cats[A_PLAY_NEW].append((card, False))

        if not playable and not is_playable(card, opp_played):
            cats[A_DISC_USELESS].append((card, True))
        elif not is_playable(card, opp_played):
            cats[A_DISC_SAFE].append((card, True))
        else:
            cats[A_DISC_RISKY].append((card, True))
    return cats


def _classify_draw(flags, disc_suit, deck_size):
    """Classify available draw actions."""
    cats = {}
    if deck_size > 0:
        cats[A_DRAW_DECK] = ['deck']
    pile_options = []
    for s in SUITS:
        if flags[s].discards and s != disc_suit:
            pile_options.append(flags[s].discards[-1])
    if pile_options:
        cats[A_DRAW_PILE] = pile_options
    return cats


def _pick_play(abstract, candidates):
    """Select concrete play from candidates (prefer lowest value)."""
    if len(candidates) == 1:
        return candidates[0]
    return min(candidates, key=lambda x: x[0][1])


def _pick_draw(abstract, candidates):
    if abstract == A_DRAW_DECK:
        return 'deck'
    return max(candidates, key=lambda x: x[1] if x != 'deck' else '0')


class MCCFRPlayer(Player):
    @classmethod
    def get_name(cls):
        return 'mccfr'

    def __init__(self, p):
        super().__init__(p)
        path = os.environ.get('MCCFR_CHECKPOINT', _DEFAULT_CHECKPOINT)
        try:
            self.strategy = _load_strategy(path)
            self.has_strategy = bool(self.strategy)
        except FileNotFoundError:
            print(f'[mccfr] No checkpoint at {path}, using fallback only')
            self.strategy = {}
            self.has_strategy = False

    def play(self, view):
        me = view.whose_turn
        hand = view.hand.cards
        flags = view.flags

        # Phase 1: play/discard via abstract actions
        cats = _classify_play(hand, flags, me)
        abstract_actions = list(cats.keys())

        key1 = _bucketed_key(hand, flags, view.deck_size, me, 'P')
        strat = self.strategy.get(key1, {}) if self.has_strategy else {}

        legal_strat = {a: strat.get(a, 0.0) for a in abstract_actions}
        total = sum(legal_strat.values())

        if total > 0:
            chosen = _sample({a: v / total for a, v in legal_strat.items()})
        else:
            chosen = next((a for a in _PLAY_PRIORITY if a in cats),
                          abstract_actions[0])

        card, is_discard = _pick_play(chosen, cats[chosen])

        # Phase 2: draw
        disc_suit = card[0] if is_discard else None
        draw_cats = _classify_draw(flags, disc_suit, view.deck_size)
        draw_abstracts = list(draw_cats.keys())

        if len(draw_abstracts) <= 1:
            draw = _pick_draw(draw_abstracts[0], draw_cats[draw_abstracts[0]]) \
                if draw_abstracts else 'deck'
        else:
            key2 = _bucketed_key(hand, flags, view.deck_size, me, 'D')
            strat2 = self.strategy.get(key2, {}) if self.has_strategy else {}
            legal2 = {a: strat2.get(a, 0.0) for a in draw_abstracts}
            total2 = sum(legal2.values())
            if total2 > 0:
                chosen_draw = _sample({a: v / total2 for a, v in legal2.items()})
            else:
                chosen_draw = A_DRAW_DECK if A_DRAW_DECK in draw_cats else draw_abstracts[0]
            draw = _pick_draw(chosen_draw, draw_cats[chosen_draw])

        return card, is_discard, draw


def _sample(strategy):
    actions = list(strategy.keys())
    probs = [strategy[a] for a in actions]
    return random.choices(actions, weights=probs, k=1)[0]
