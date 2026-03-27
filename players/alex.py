"""Alex player for Lost Cities.

Builds on Committer's gap-minimization core with layered heuristics:
1. Suit EV for tiebreaking between equal-gap plays
2. Stall awareness in draw decisions
3. Smart discarding that avoids feeding the opponent
4. Safe discard detection (opponent committed higher)
5. Late-game filter to avoid opening doomed expeditions
"""

from classes import *
from utils import *

# Tunable constants
LATE_GAME_CUTOFF = 14    # Deck size below which opening new suits is penalized
STALL_RATIO = 0.3        # Stall if desired_plays > deck_size * ratio


def minimize_gap(cards, flags, me):
    """Return the play that skips the fewest cards. Borrowed from Committer."""
    best_card = ''
    smallest_gap = len(CARDS) + 1
    second_smallest_gap = len(CARDS) + 1
    for c in cards:
        baseline = -1
        played = flags[c[0]].played[me]
        if played:
            baseline = int(played[-1][1])

        values_left = [x for x in CARDS if int(x) >= baseline]
        if baseline == 0:
            values_left = values_left[1:]

        opponent_played = flags[c[0]].played[1-me]
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


def card_gap(card, flags, me):
    """Compute the gap for a single card."""
    _, gap, _ = minimize_gap([card], flags, me)
    return gap


def suit_ev(suit, hand, flags, me, deck_size):
    """Estimate expected value of committing to a suit."""
    my_played = flags[suit].played[me]
    opp_played = flags[suit].played[1 - me]
    discards = flags[suit].discards

    hand_in_suit = [c for c in hand if c[0] == suit
                    and is_playable(c, my_played)]

    committed = my_played + hand_in_suit
    if not committed:
        return 0.0

    n_contracts = sum(1 for c in committed if c[1] == '0')
    multiplier = 1 + n_contracts
    known_sum = sum_cards(committed)

    # Track seen cards to estimate future draws
    seen = set()
    for c in my_played + opp_played + discards:
        seen.add(c)
    for c in hand:
        if c[0] == suit:
            seen.add(c)

    top_value = 0
    non_contract = [c for c in committed if c[1] != '0']
    if non_contract:
        top_value = int(non_contract[-1][1])

    # Unseen playable cards in this suit
    unseen_playable = []
    for v in CARDS:
        if int(v) >= top_value and v != '0':
            if suit + v not in seen:
                unseen_playable.append(int(v))

    if unseen_playable and deck_size > 0:
        avg_value = sum(v + 1 for v in unseen_playable) / len(unseen_playable)
        remaining_turns = deck_size / 2
        draw_prob = len(unseen_playable) / max(deck_size, 1)
        expected_draws = min(draw_prob * remaining_turns, len(unseen_playable))
        expected_draw_value = expected_draws * avg_value
    else:
        expected_draw_value = 0.0
        expected_draws = 0.0

    ev = multiplier * (known_sum + expected_draw_value - BREAKEVEN)

    if len(committed) + expected_draws >= BONUS_THRESHOLD:
        ev += BONUS_POINTS

    return ev


def count_desired_plays(hand, flags, me, deck_size):
    """Count cards in hand we actively want to play. Returns (count, should_stall)."""
    n_desired = 0
    for c in hand:
        suit = c[0]
        if not is_playable(c, flags[suit].played[me]):
            continue
        if flags[suit].played[me]:
            n_desired += 1
        elif suit_ev(suit, hand, flags, me, deck_size) > 0:
            n_desired += 1

    should_stall = deck_size > 0 and n_desired > deck_size * STALL_RATIO
    return n_desired, should_stall


def score_discard(card, flags, me, hand, deck_size):
    """Score how safe a card is to discard. Higher = safer to discard."""
    suit = card[0]
    score = 0.0

    opp_playable = is_playable(card, flags[suit].played[1 - me])
    my_playable = is_playable(card, flags[suit].played[me])

    # Useless: neither player can use it
    if not opp_playable and not my_playable:
        score += 100
    elif not opp_playable:
        score += 50

    # Opponent committed higher — they definitely can't use this
    opp_played = flags[suit].played[1 - me]
    if opp_played and int(card[1]) < int(opp_played[-1][1]):
        score += 60

    # Penalize feeding the opponent
    score -= points_for_opponent(card, flags, me)

    # Prefer discarding low-value cards
    score -= int(card[1]) * 0.5

    # Penalize discarding cards in suits we're invested in
    my_played = flags[suit].played[me]
    if my_played and my_playable:
        score -= 30

    # Penalize discarding cards in suits with positive EV
    ev = suit_ev(suit, hand, flags, me, deck_size)
    if ev > 0 and my_playable:
        score -= ev * 0.5

    return score


def best_play(playable_cards, flags, me, hand, deck_size):
    """Choose the best card to play.

    Primary sort: gap (lower is better, like Committer).
    Tiebreaker: suit EV (prefer suits we're invested in or that have good potential).
    Late-game filter: avoid opening new suits when few cards remain.
    """
    # Late-game: filter to only continuing expeditions if possible
    if deck_size < LATE_GAME_CUTOFF:
        continuing = [c for c in playable_cards if flags[c[0]].played[me]]
        if continuing:
            playable_cards = continuing

    # Score each card: primary = -gap, secondary = ev + continuation bonus
    scored = []
    for c in playable_cards:
        gap = card_gap(c, flags, me)
        is_continuing = bool(flags[c[0]].played[me])
        ev = suit_ev(c[0], hand, flags, me, deck_size)

        # Tiebreaker score (only matters when gaps are equal)
        tiebreak = 0.0
        if is_continuing:
            tiebreak += 5.0
        tiebreak += ev * 0.1

        scored.append((gap, -tiebreak, c))

    scored.sort()  # Lowest gap first, then highest tiebreak (negated)
    return scored[0][2]


class Alex(Player):
    @classmethod
    def get_name(cls):
        return 'alex'

    '''
    heuristics implemented:
    -[x] open suits that you have length, weighted by the sum of the value of your cards (suit_ev)
    -[x] number of cards in hand that you need/want to play for stalling (count_desired_plays)
    -[x] don't discard obvious plays for your opponents (score_discard)
    -[x] safe discards for cards opponent committed higher to (score_discard)
    -[x] prefer to not open new suits if there is another good play (score_play)
    '''
    def play(self, r):
        me = r.whose_turn
        cards = r.h[me].cards
        deck_size = len(r.deck)

        playable_cards = [c for c in cards
                         if is_playable(c, r.flags[c[0]].played[me])]
        possible_draws = playable_draws(r.flags, me)

        # Draw decision setup (same as Committer)
        draw = 'deck'
        if possible_draws:
            best_draw_card, draw_gap, _ = minimize_gap(possible_draws, r.flags, me)
        else:
            best_draw_card, draw_gap = '', len(CARDS) + 1

        # Stalling check
        _, should_stall = count_desired_plays(cards, r.flags, me, deck_size)

        if playable_cards:
            play_card = best_play(playable_cards, r.flags, me, cards, deck_size)

            # Draw decision: same logic as Committer but with stall awareness
            if not should_stall and possible_draws:
                _, _, second_best_gap = minimize_gap(playable_cards, r.flags, me)
                if draw_gap < second_best_gap:
                    draw = best_draw_card

            return play_card, False, draw
        else:
            # Must discard — use smart discard scoring
            discard_scored = [(score_discard(c, r.flags, me, cards, deck_size), c)
                              for c in cards]
            discard_scored.sort(reverse=True)
            card = discard_scored[0][1]

            # Draw: avoid drawing from pile we just discarded to
            if possible_draws and not should_stall:
                filtered = [d for d in possible_draws if d[0] != card[0]]
                if filtered:
                    best_d, d_gap, _ = minimize_gap(filtered, r.flags, me)
                    if d_gap <= 2:
                        draw = best_d

            return card, True, draw
