"""Challenger player for autoresearch-style iterative improvement.

This is the ONLY file the AI agent should modify. Started as a copy of
Committer.  The goal is to iteratively improve win rate vs Committer.
"""

from classes import *
from utils import *

class Challenger(Player):
    @classmethod
    def get_name(cls):
        return 'challenger'

    def play(self, r):
        me = r.whose_turn

        draw = 'deck'
        possible_draws = playable_draws(r.flags, me)
        best_draw, draw_gap, _ = minimize_gap(possible_draws, r.flags, me)

        cards = r.h[me].cards
        playable_cards = [c for c in cards
                             if is_playable(c, r.flags[c[0]].played[me])]

        if playable_cards:
            # Late game: avoid opening new expeditions — not enough turns
            # left to reach breakeven (20 points).
            if len(r.deck) < 16:
                continuing = [c for c in playable_cards
                              if r.flags[c[0]].played[me]]
                if continuing:
                    playable_cards = continuing

            play = best_play(playable_cards, r.flags, me)
            _, _, second_best_gap = minimize_gap(playable_cards, r.flags, me)
            if draw_gap < second_best_gap:  # This draw improves your hand.
                draw = best_draw
            return play, False, draw
        else:
            discard = discard_intelligently(cards, r.flags, me)
            if draw[0] == discard[0]:  # Don't discard to the pile you'll draw
                draw = 'deck'          # from; instead, draw from the deck.
            return discard, True, draw


def best_play(cards, flags, me):
    """Pick the best card to play: minimize gap, break ties by suit potential."""
    scored = []
    for c in cards:
        played = flags[c[0]].played[me]
        # Compute gap (same as minimize_gap)
        baseline = -1
        if played:
            baseline = int(played[-1][1])
        values_left = [x for x in CARDS if int(x) >= baseline]
        if baseline == 0:
            values_left = values_left[1:]
        for other_c in flags[c[0]].played[1-me] + flags[c[0]].discards[:-1]:
            v = other_c[1]
            if v in values_left:
                values_left.remove(v)
        gap = values_left.index(c[1])

        # Count remaining playable values in this suit (potential)
        my_top = int(c[1])
        remaining = sum(1 for v in values_left if int(v) > my_top)

        # Score: gap is primary (lower is better), remaining is tiebreaker (higher better)
        scored.append((-gap, remaining, c))
    scored.sort(reverse=True)
    return scored[0][2]


def minimize_gap(cards, flags, me):
    """Return the play that skips the fewest cards."""
    best_card = ''
    smallest_gap = len(CARDS) + 1
    second_smallest_gap = len(CARDS) + 1
    for c in cards:
        baseline = -1
        played = flags[c[0]].played[me]
        if played:
            baseline = int(played[-1][1])

        values_left_string = CARDS
        values_left = [x for x in values_left_string if int(x) >= baseline]
        if baseline == 0:  # Track contract duplicates correctly.
            values_left = values_left[1:]

        # Discards and opponent's plays can reduce the opportunity cost.
        opponent_played = flags[c[0]].played[1-me]
        # Ignore the top card when deciding, since you could draw it this turn.
        discards = flags[c[0]].discards[:-1]

        for other_c in opponent_played + discards:
            v = other_c[1]
            if v in values_left:
                values_left.remove(v)

        gap = values_left.index(c[1])
        if gap < smallest_gap:
            second_smallest_gap = smallest_gap
            smallest_gap = gap
            best_card = c
    return best_card, smallest_gap, second_smallest_gap
