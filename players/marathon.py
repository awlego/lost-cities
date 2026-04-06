"""Go big or go home: chase 8+ card expeditions with contracts for massive scores.

Key insight: long suits come from TIGHT gap play (filling suits densely)
combined with high tempo (always play, never waste turns). You can't force
long suits by refusing to play other suits — the bottleneck is card
distribution, not hand management.

Strategy:
- Always play if possible (committer-level tempo)
- Identify 1 primary target suit and tighten gap costs there (3x multiplier)
  so the algorithm naturally fills that suit densely with low-gap plays
- Play contracts in the primary suit regardless of gap (they multiply AND
  count toward the 8-card bonus)
- Use committer's sqrt-compressed gap algorithm for non-primary suits
- Aggressively draw primary-suit cards from discard piles
- Strong tiebreaker bonus toward primary suit plays

Win rate vs Committer: ~50% (10k games)
8+ card bonus rate: ~7.5% of games (2.3x committer's ~3.2%)
"""

from classes import *
from utils import *


class Marathon(Player):
    @classmethod
    def get_name(cls):
        return 'marathon'

    def play(self, r):
        me = r.whose_turn
        cards = r.hand.cards
        flags = r.flags

        primary = get_primary(cards, flags, me)
        playable = [c for c in cards if is_playable(c, flags[c[0]].played[me])]

        if playable:
            play_card = pick_play(playable, primary, flags, me, cards)
            draw = pick_draw(flags, me, primary, play_card)
            return play_card, False, draw
        else:
            discard = discard_intelligently(cards, flags, me)
            draw = pick_draw(flags, me, primary, None, discard_suit=discard[0])
            return discard, True, draw


# ---------------------------------------------------------------------------
# Target selection — pick the single best marathon suit
# ---------------------------------------------------------------------------

def get_primary(hand, flags, me):
    """Pick the single best suit for marathon targeting."""
    scores = {}
    for s in SUITS:
        my_played = flags[s].played[me]
        hand_in = [c for c in hand if c[0] == s
                   and is_playable(c, my_played)]
        total = len(my_played) + len(hand_in)
        contracts = sum(1 for c in my_played + hand_in if c[1] == '0')

        # Count unseen cards (potential future draws)
        seen = set()
        for c in my_played + flags[s].played[1 - me] + flags[s].discards:
            seen.add(c)
        for c in hand:
            if c[0] == s:
                seen.add(c)
        unseen = sum(1 for v in CARDS if s + v not in seen)

        scores[s] = total * 3 + contracts * 5 + unseen * 0.5

    ranked = sorted(SUITS, key=lambda s: scores[s], reverse=True)
    primary = set()
    if scores[ranked[0]] >= 3:
        primary.add(ranked[0])
    return primary


# ---------------------------------------------------------------------------
# Play selection — committer core with marathon bias
# ---------------------------------------------------------------------------

# Gap tightening multiplier for primary suit.
# Higher = more conservative gap play = denser suit filling.
# 3.0 gives ~2.3x the 8+ bonus rate of committer at ~50% win rate.
PRIMARY_GAP_MULT = 3.0

# Tiebreaker bonus added to 'remaining' for primary suit plays.
# Makes primary suit win ties against non-primary plays.
PRIMARY_BONUS = 25


def pick_play(playable, primary, flags, me, hand):
    """Pick the best card to play. Uses committer's sqrt-compressed gap
    algorithm with a marathon bias for the primary suit."""
    scored = []
    for c in playable:
        suit = c[0]
        played = flags[suit].played[me]
        is_primary = suit in primary

        # Compute gap cost (committer/challenger algorithm)
        baseline = -1
        if played:
            baseline = int(played[-1][1])
        values_left = [x for x in CARDS if int(x) >= baseline]
        if baseline == 0:
            values_left = values_left[1:]

        # Remove known-gone cards
        for other_c in flags[suit].played[1 - me] + flags[suit].discards[:-1]:
            v = other_c[1]
            if v in values_left:
                values_left.remove(v)

        idx = values_left.index(c[1])

        # Sqrt-compressed gap cost
        gap_cost = sum((int(v) + 1) ** 0.5 for v in values_left[:idx])

        # Multiplier-weighted (contracts make gaps more expensive)
        mult = 1 + sum(1 for p in played if p[1] == '0')
        if c[1] == '0':
            mult += 1
        gap_cost *= mult ** 0.7

        # Remaining playable values (tiebreaker: prefer suits with more room)
        remaining = sum(1 for v in values_left if int(v) > int(c[1]))

        # --- Marathon modifications for primary suit ---
        if is_primary:
            # Tighten gap cost: makes the algorithm fill primary suit more densely
            gap_cost *= PRIMARY_GAP_MULT

            # Override for contracts: always play them in primary suit
            if c[1] == '0':
                gap_cost = -100

            # Strong tiebreaker toward primary suit
            remaining += PRIMARY_BONUS

            # Bonus for approaching 8-card threshold
            count_after = len(played) + 1
            if count_after >= 8:
                remaining += 30
            elif count_after >= 6:
                remaining += 10

        scored.append((-gap_cost, remaining, c))

    scored.sort(reverse=True)
    return scored[0][2]


# ---------------------------------------------------------------------------
# Draw selection — aggressively recycle primary-suit discards
# ---------------------------------------------------------------------------

def pick_draw(flags, me, primary, play_card, discard_suit=None):
    """Draw from discard piles for primary suit; otherwise draw from deck."""
    best = 'deck'
    best_value = -1

    for suit in SUITS:
        discs = flags[suit].discards
        if not discs:
            continue
        top = discs[-1]

        # Can't draw from suit we just played/discarded into
        if play_card and top[0] == play_card[0]:
            continue
        if discard_suit and top[0] == discard_suit:
            continue

        if not is_playable(top, flags[suit].played[me]):
            continue

        if suit in primary:
            val = 50 + int(top[1])
            if top[1] == '0':
                val += 100  # Primary contract from discard = must grab
        else:
            continue  # Only draw for primary suit

        if val > best_value:
            best_value = val
            best = top

    return best
