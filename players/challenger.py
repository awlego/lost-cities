"""Challenger player for autoresearch-style iterative improvement.

This is the ONLY file the AI agent should modify. Started as a copy of
Committer.  The goal is to iteratively improve win rate vs Committer.
"""

from classes import *
from utils import *


# ===========================================================================
# Hard facts — deterministic from visible game state
# ===========================================================================

# --- Card Counting ---

def unseen_cards(suit, flags, hand):
    """Cards in this suit not in any played pile, discard pile, or your hand.
    These are in the opponent's hand or the deck."""
    seen = set()
    for c in flags[suit].played[0] + flags[suit].played[1] + flags[suit].discards:
        seen.add(c)
    for c in hand:
        if c[0] == suit:
            seen.add(c)
    return [suit + v for v in CARDS if suit + v not in seen]


def suit_remaining(suit, flags, hand):
    """How many cards in this suit are still unseen."""
    return len(unseen_cards(suit, flags, hand))


# --- Suit Facts ---

def suit_commitment(suit, flags, player):
    """How locked-in a player is to a suit: count of cards played including contracts."""
    return len(flags[suit].played[player])


def suit_breakeven_distance(suit, flags, me, hand):
    """How many more points needed to reach breakeven (20).
    Negative means already profitable."""
    my_played = flags[suit].played[me]
    hand_in_suit = [c for c in hand if c[0] == suit
                    and is_playable(c, my_played)]
    committed = my_played + hand_in_suit
    if not committed:
        return BREAKEVEN
    return BREAKEVEN - sum_cards(committed)


def suit_multiplier(suit, flags, player):
    """Current multiplier for a player's expedition (1 + contracts played)."""
    return 1 + sum(1 for c in flags[suit].played[player] if c[1] == '0')


def opponent_wants_suit(suit, flags, me):
    """Does the opponent appear to be collecting this suit?
    True if they've played any cards (especially contracts) in it."""
    return bool(flags[suit].played[1 - me])


# --- Hand Facts ---

def hand_flexibility(hand, flags, me):
    """Number of distinct suits you can play into."""
    suits = set()
    for c in hand:
        if is_playable(c, flags[c[0]].played[me]):
            suits.add(c[0])
    return len(suits)


def hand_deadwood(hand, flags, me):
    """Cards that are unplayable and unsafe to discard. These clog your hand."""
    dead = []
    for c in hand:
        suit = c[0]
        if not is_playable(c, flags[suit].played[me]):
            # It's unplayable for us — but is it safe to discard?
            if is_playable(c, flags[suit].played[1 - me]):
                # Opponent could use it — unsafe to discard
                dead.append(c)
    return dead


def playable_card_count(hand, flags, me):
    """Simple count of cards you can legally play."""
    return sum(1 for c in hand if is_playable(c, flags[c[0]].played[me]))


def turns_of_plays(hand, flags, me):
    """How many consecutive turns you could play (not discard) if you drew blanks.
    Measures hand 'fuel'."""
    return sum(1 for c in hand if is_playable(c, flags[c[0]].played[me]))


# --- Tempo / Endgame Facts ---

def deck_pressure(deck_size):
    """How close to game end, normalized 0-1. 1.0 = game almost over."""
    max_deck = 12 * len(SUITS) - HAND_SIZE * N_PLAYERS
    return 1.0 - (deck_size / max(max_deck, 1))


def cards_until_empty(deck_size):
    """Turns remaining for each player."""
    return deck_size / N_PLAYERS


# --- Scoring Facts ---

def score_expedition(cards):
    """Score a single expedition. Mirrors the logic in Round.get_winner()."""
    if not cards:
        return 0
    mult = 1
    total = 0
    for c in cards:
        if c[1] == '0':
            mult += 1
        else:
            total += int(c[1]) + 1
    score = mult * (total - BREAKEVEN)
    if len(cards) >= BONUS_THRESHOLD:
        score += BONUS_POINTS
    return score


def current_score(flags, player):
    """What a player would score right now if the game ended."""
    total = 0
    for s in SUITS:
        total += score_expedition(flags[s].played[player])
    return total


def score_differential(flags, me):
    """Your score minus opponent's score. Positive = you're winning."""
    return current_score(flags, me) - current_score(flags, 1 - me)


# ===========================================================================
# Heuristic estimates — involve probability, opponent modeling, or judgment
# ===========================================================================

# --- Opponent Estimates ---

def opponent_likely_holds(suit, flags, hand):
    """Estimate how many unseen cards in this suit the opponent holds.
    Based on ratio of opponent's hand size (8) to total unseen cards."""
    unseen = unseen_cards(suit, flags, hand)
    total_unseen = sum(suit_remaining(s, flags, hand) for s in SUITS)
    if total_unseen == 0:
        return 0.0
    return len(unseen) * (HAND_SIZE / total_unseen)


def opponent_multiplier_exposure(suit, flags, me):
    """Opponent's multiplier in a suit where they haven't reached breakeven.
    High value = they're vulnerable and holding their cards hurts them."""
    opp_played = flags[suit].played[1 - me]
    if not opp_played:
        return 0.0
    mult = suit_multiplier(suit, flags, 1 - me)
    opp_sum = sum_cards(opp_played)
    if opp_sum >= BREAKEVEN:
        return 0.0  # They're already profitable, not exposed
    return mult * (BREAKEVEN - opp_sum) / BREAKEVEN  # Normalized exposure


def denial_value(card, flags, me):
    """Value of NOT discarding this card, considering opponent's need.
    Higher = more valuable to hold. Considers whether the card unlocks
    a sequence for the opponent (cascading effect)."""
    suit = card[0]
    opp_played = flags[suit].played[1 - me]
    if not opp_played:
        # Opponent hasn't started this suit — denial value is low
        return 0.1 * (int(card[1]) + 1)

    # Base value: what opponent scores if they get this card
    base = points_for_opponent(card, flags, me)

    # Cascading bonus: if this card is the next playable for opponent,
    # it unlocks higher cards too
    opp_top = int(opp_played[-1][1])
    if int(card[1]) == opp_top or int(card[1]) == opp_top + 1:
        # This is the immediate next card — high cascading potential
        # Count how many higher cards are unseen (could follow)
        unseen = unseen_cards(suit, flags, [])  # Empty hand = don't filter
        following = sum(1 for c in unseen if int(c[1]) > int(card[1]))
        base += following * 1.5

    return base


# --- Suit Estimates ---

def max_possible_score(suit, flags, me, hand):
    """Ceiling score if you played every remaining playable card in this suit.
    Upper bound on suit potential — assumes you draw everything."""
    my_played = flags[suit].played[me]
    hand_in_suit = [c for c in hand if c[0] == suit
                    and is_playable(c, my_played)]
    committed = my_played + hand_in_suit

    if not committed:
        return 0

    # Add all unseen playable cards as hypothetical
    unseen = unseen_cards(suit, flags, hand)
    top = committed[-1][1] if committed else '0'
    playable_unseen = [c for c in unseen if c[1] >= top and c[1] != '0']

    all_cards = committed + playable_unseen
    n_contracts = sum(1 for c in all_cards if c[1] == '0')
    multiplier = 1 + n_contracts
    total = sum_cards(all_cards)
    score = multiplier * (total - BREAKEVEN)
    if len(all_cards) >= BONUS_THRESHOLD:
        score += BONUS_POINTS
    return score


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


def projected_score(flags, me, hand, deck_size):
    """Estimated total score across all suits, including expected future draws."""
    total = 0.0
    for s in SUITS:
        if flags[s].played[me] or any(c[0] == s for c in hand):
            total += suit_ev(s, hand, flags, me, deck_size)
    return total


# --- Tempo Estimates ---

def tempo_advantage(hand, flags, me):
    """Positive = you have more queued plays than opponent has visible momentum.
    Suggests you want the game to go longer."""
    my_playable = playable_card_count(hand, flags, me)
    opp_total_played = sum(len(flags[s].played[1 - me]) for s in SUITS)
    my_total_played = sum(len(flags[s].played[me]) for s in SUITS)
    return my_playable + my_total_played - opp_total_played

def denial_discard(cards, flags, me):
    """Discard to minimize opponent benefit."""
    useless = useless_discards(cards, flags, me)
    if useless:
        return min(useless, key=lambda c: int(c[1]))
    safe = safe_discards(cards, flags, me)
    if safe:
        return min(safe, key=lambda c: int(c[1]))
    # All cards playable by opponent — minimize their benefit
    return min(cards, key=lambda c: points_for_opponent(c, flags, me))


class Challenger(Player):
    @classmethod
    def get_name(cls):
        return 'challenger'

    def play(self, r):
        me = r.whose_turn
        cards = r.hand.cards
        playable_cards = [c for c in cards
                             if is_playable(c, r.flags[c[0]].played[me])]

        if playable_cards:
            # Late game: avoid opening new expeditions — not enough turns
            # left to reach breakeven (20 points).
            if r.deck_size < 12:
                continuing = [c for c in playable_cards
                              if r.flags[c[0]].played[me]]
                if continuing:
                    playable_cards = continuing

            if r.deck_size <= 2:
                play = max(playable_cards, key=lambda c: (
                    score_expedition(r.flags[c[0]].played[me] + [c]) -
                    score_expedition(r.flags[c[0]].played[me])
                ))
            else:
                play = best_play(playable_cards, r.flags, me, r.deck_size)
            return play, False, 'deck'
        else:
            discard = denial_discard(cards, r.flags, me)
            return discard, True, 'deck'


def best_play(cards, flags, me, deck_size=44):
    """Pick the best card to play: gap-based with marginal score delta influence."""
    scored = []
    for c in cards:
        played = flags[c[0]].played[me]
        baseline = -1
        if played:
            baseline = int(played[-1][1])
        values_left = [x for x in CARDS if int(x) >= baseline]
        if baseline == 0:
            values_left = values_left[1:]
        # Remove ALL discards (we never draw from pile)
        for other_c in flags[c[0]].played[1-me] + flags[c[0]].discards:
            v = other_c[1]
            if v in values_left:
                values_left.remove(v)
        idx = values_left.index(c[1])
        gap_cost = sum((int(v) + 1) ** 0.5 for v in values_left[:idx])
        mult = 1 + sum(1 for p in played if p[1] == '0')
        if c[1] == '0':
            mult += 1
        gap_cost *= mult ** 0.7

        my_top = int(c[1])
        remaining = sum(int(v) + 1 for v in values_left if int(v) > my_top)

        # Score delta: marginal influence at all deck sizes
        new_played = played + [c]
        score_delta = score_expedition(new_played) - score_expedition(played)

        if deck_size <= 8:
            # Endgame: stronger score delta influence
            combined = -gap_cost + 0.05 * score_delta
            scored.append((combined + 0.3 * score_delta, remaining, c))
        else:
            # Non-endgame: tiny score delta influence
            combined = -gap_cost + 0.05 * score_delta
            scored.append((combined, remaining, c))
    scored.sort(reverse=True)
    return scored[0][-1]


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

        idx = values_left.index(c[1])
        gap = sum(int(v) + 1 for v in values_left[:idx])
        if gap < smallest_gap:
            second_smallest_gap = smallest_gap
            smallest_gap = gap
            best_card = c
    return best_card, smallest_gap, second_smallest_gap
