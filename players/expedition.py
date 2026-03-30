"""Strategic bot building on Committer's aggressive play style with smarter
discard management (opponent-aware) and clock-controlled draw decisions.

Win rate vs. Kenny: TBD
Win rate vs. Committer: TBD
"""

from classes import *
from utils import *


# Phase boundaries
EARLY_THRESHOLD = 37
LATE_THRESHOLD = 18

# Draw thresholds
MIN_DRAW_VALUE_AHEAD = 7
MIN_DRAW_VALUE_BEHIND = 3


def _game_phase(deck_size):
    if deck_size > EARLY_THRESHOLD:
        return 'early'
    elif deck_size > LATE_THRESHOLD:
        return 'mid'
    return 'late'


def _suit_info(r):
    """Build per-suit information from visible game state."""
    me = r.whose_turn
    hand = r.hand.cards
    infos = {}
    for s in SUITS:
        my_played = r.flags[s].played[me]
        opp_played = r.flags[s].played[1 - me]
        hand_cards = [c for c in hand if c[0] == s]
        infos[s] = {
            'my_played': my_played,
            'opp_played': opp_played,
            'hand_cards': hand_cards,
            'is_started': len(my_played) > 0,
            'my_contracts': sum(1 for c in my_played if c[1] == '0'),
            'opp_contracts': sum(1 for c in opp_played if c[1] == '0'),
            'hand_face': sum_cards(hand_cards),
            'has_number_played': any(c[1] != '0' for c in my_played),
            'discard_count': len(r.flags[s].discards),
        }
    return infos


def _card_counting(r):
    """Compute unseen cards per suit from all visible information."""
    me = r.whose_turn
    counts = {}
    for s in SUITS:
        # Full deck for this suit: three '0' contracts + one each of '1'-'9'
        remaining = {}
        for c in CARDS:
            remaining[c] = remaining.get(c, 0) + 1

        # Subtract all visible cards
        for card in r.hand.cards:
            if card[0] == s:
                remaining[card[1]] -= 1
        for card in r.flags[s].played[me]:
            remaining[card[1]] -= 1
        for card in r.flags[s].played[1 - me]:
            remaining[card[1]] -= 1
        for card in r.flags[s].discards:
            remaining[card[1]] -= 1

        unseen = {v: c for v, c in remaining.items() if c > 0}
        unseen_numbers = {v: c for v, c in unseen.items() if v != '0'}
        counts[s] = {
            'unseen': unseen,
            'unseen_total': sum(unseen.values()),
            'unseen_numbers': unseen_numbers,
        }

    total_unseen = sum(counts[s]['unseen_total'] for s in SUITS)
    return counts, total_unseen


def _prob_in_deck(total_unseen, deck_size, count):
    """Probability that at least one of `count` specific unseen cards is in the deck."""
    if total_unseen <= 0 or deck_size <= 0 or count <= 0:
        return 0.0
    opp_hand = total_unseen - deck_size
    if opp_hand <= 0 or count > opp_hand:
        return 1.0
    # P(all target cards in opponent's hand) via hypergeometric
    p_none = 1.0
    for i in range(count):
        p_none *= (opp_hand - i) / (total_unseen - i)
    return 1.0 - p_none


def _one_card_rule(suit, infos, counts, total_unseen, deck_size):
    """Will playing hand cards + drawing one more reach breakeven (face sum >= 20)?"""
    hand_face = infos[suit]['hand_face']

    if hand_face >= BREAKEVEN:
        return True, 1.0

    needed = BREAKEVEN - hand_face
    # Count unseen number cards whose face value alone fills the gap
    qualifying = 0
    for v, cnt in counts[suit]['unseen_numbers'].items():
        if int(v) + 1 >= needed:
            qualifying += cnt

    if qualifying == 0:
        return False, 0.0

    prob = _prob_in_deck(total_unseen, deck_size, qualifying)
    return prob >= 0.5, prob


def _bonus_chase(suit, infos, counts, phase):
    """Check if the 8-card bonus is worth chasing for this suit.
    Returns (viable, numbers_still_needed) where viable means we should
    invest extra in this suit for the bonus."""
    info = infos[suit]
    if not info['is_started'] or info['my_contracts'] < 1:
        return False, 99

    numbers_played = len(info['my_played']) - info['my_contracts']
    hand_numbers = sum(1 for hc in info['hand_cards'] if hc[1] != '0')
    numbers_still_needed = (BONUS_THRESHOLD - info['my_contracts']) - (numbers_played + hand_numbers)

    if numbers_still_needed <= 0:
        return False, 0  # Already there, no extra chase needed

    if phase == 'late' and numbers_still_needed > 2:
        return False, numbers_still_needed

    unseen_numbers = sum(counts[suit]['unseen_numbers'].values())
    if unseen_numbers < numbers_still_needed:
        return False, numbers_still_needed

    # Viable: opponent discarding this suit is a good signal (cards available)
    viable = numbers_still_needed <= 3 + info['discard_count']
    return viable, numbers_still_needed


def _gap(card, flags, me):
    """Count skipped cards (opportunity cost) for playing this card.
    Same logic as Committer's minimize_gap."""
    s = card[0]
    played = flags[s].played[me]

    baseline = -1
    if played:
        baseline = int(played[-1][1])

    values_left = [x for x in CARDS if int(x) >= baseline]
    if baseline == 0:
        values_left = values_left[1:]

    # Remove cards visible elsewhere (reduces opportunity cost)
    for other_c in flags[s].played[1 - me] + flags[s].discards[:-1]:
        v = other_c[1]
        if v in values_left:
            values_left.remove(v)

    if card[1] in values_left:
        return values_left.index(card[1])
    return len(values_left)


def _choose_play(r, infos, phase, counts, total_unseen):
    """Play aggressively (like Committer) but with strategic opening decisions."""
    me = r.whose_turn
    hand = r.hand.cards

    playable = [c for c in hand if is_playable(c, r.flags[c[0]].played[me])]

    if not playable:
        return _choose_discard(r, infos, me, counts, phase), True

    # Score each playable card
    scored = []
    for c in playable:
        s = c[0]
        info = infos[s]
        g = _gap(c, r.flags, me)
        face_val = int(c[1]) + 1 if c[1] != '0' else 0

        score = -g * 3  # Gap minimization (primary factor)

        if info['is_started']:
            score += 25  # Strong preference for continuing expeditions
        else:
            # Opening a new expedition: use one-card rule
            should_start, prob = _one_card_rule(s, infos, counts, total_unseen, r.deck_size)

            if phase == 'late' and not should_start:
                score -= 50  # Don't open hopeless expeditions late
            elif phase == 'mid' and not should_start:
                score -= 30  # Be cautious mid-game with weak suits
            elif phase == 'early':
                # Early game: be speculative, only penalize truly hopeless
                n_hand = len(info['hand_cards'])
                if prob < 0.1 and n_hand < 2:
                    score -= 15
            if should_start:
                score += prob * 5  # Mild confidence bonus

        # Contracts: play early, never after numbers
        if c[1] == '0':
            if info['has_number_played']:
                score -= 200  # Illegal
            elif phase == 'late':
                score -= 40  # Too late for contracts
            else:
                score += 10  # Good to play contracts early

        # Late game: prefer high-value cards
        if phase == 'late':
            score += face_val * 2

        # Prefer suits with more hand cards (better investment)
        score += len(info['hand_cards']) * 1.5

        # 8-card bonus pursuit: incentivize building toward 8 cards,
        # especially via the wager route (3 wagers = only 5 numbers needed)
        viable, needed = _bonus_chase(s, infos, counts, phase)
        if viable:
            score += 5

        scored.append((score, c))

    scored.sort(reverse=True)
    best_score, best_card = scored[0]

    # If best play is significantly penalized, consider discarding instead
    if best_score < -20 and not infos[best_card[0]]['is_started']:
        return _choose_discard(r, infos, me, counts, phase), True

    return best_card, False


def _choose_discard(r, infos, me, counts, phase):
    """Pick the least damaging card to discard."""
    hand = r.hand.cards

    # Protect started suits, strong hand suits, and bonus-viable suits
    protected = set()
    for s in SUITS:
        info = infos[s]
        if info['is_started']:
            protected.add(s)
        elif info['hand_face'] >= 10 and len(info['hand_cards']) >= 2:
            protected.add(s)
        # Protect suits on the 8-card bonus path
        viable, _ = _bonus_chase(s, infos, counts, phase)
        if viable:
            protected.add(s)

    unprotected = [c for c in hand if c[0] not in protected]
    pool = unprotected if unprotected else list(hand)

    # Useless > Safe > Least opponent value
    useless = useless_discards(pool, r.flags, me)
    if useless:
        useless.sort(key=lambda c: points_for_opponent(c, r.flags, me))
        return useless[0]

    safe = safe_discards(pool, r.flags, me)
    if safe:
        safe.sort(key=lambda c: points_for_opponent(c, r.flags, me))
        return safe[0]

    pool.sort(key=lambda c: points_for_opponent(c, r.flags, me))
    return pool[0]


def _estimate_scores(infos):
    """Estimate visible scores for both players."""
    my_score, opp_score = 0, 0
    for info in infos.values():
        for played, accum in [(info['my_played'], 'me'), (info['opp_played'], 'opp')]:
            if not played:
                continue
            mult = 1 + sum(1 for c in played if c[1] == '0')
            face = sum_cards(played)
            sc = mult * (face - BREAKEVEN)
            if len(played) >= BONUS_THRESHOLD:
                sc += BONUS_POINTS
            if accum == 'me':
                my_score += sc
            else:
                opp_score += sc
    return my_score, opp_score


def _choose_draw(r, play_card, is_discard, infos, counts, total_unseen, phase):
    """Clock-aware draw: accelerate when ahead, extend when behind.
    Also incorporates Committer's idea: draw from discard if it improves hand."""
    me = r.whose_turn
    my_score, opp_score = _estimate_scores(infos)
    ahead = my_score > opp_score

    possible = playable_draws(r.flags, me)

    # Don't draw from suit we just discarded to
    if is_discard:
        possible = [c for c in possible if c[0] != play_card[0]]

    if not possible:
        return 'deck'

    best_draw = 'deck'
    best_score = 0.0
    min_value = MIN_DRAW_VALUE_AHEAD if ahead else MIN_DRAW_VALUE_BEHIND

    for c in possible:
        s = c[0]
        face_val = int(c[1]) + 1
        info = infos[s]

        if info['is_started']:
            # Active expedition: high value, amplified by contracts
            draw_score = face_val * (1 + info['my_contracts']) + 3
        elif info['hand_face'] >= 8:
            # Strong uncommitted suit
            draw_score = face_val * 0.7
        else:
            # Weak suit: usually not worth it
            draw_score = face_val * 0.1

        # Card scarcity: known good card is more valuable when few unseen remain
        if counts[s]['unseen_total'] <= 2:
            draw_score += 3

        # 8-card bonus pursuit: prefer drawing cards that feed bonus-viable suits
        viable, _ = _bonus_chase(s, infos, counts, phase)
        if viable:
            draw_score += 4

        # Clock control
        if ahead:
            draw_score -= 5  # Strong preference for deck (end game fast)
        else:
            draw_score += 3  # Prefer discard draw (extend game)

        if draw_score > best_score and face_val >= min_value:
            best_score = draw_score
            best_draw = c

    return best_draw


class Expedition(Player):
    @classmethod
    def get_name(cls):
        return 'expedition'

    def play(self, r):
        phase = _game_phase(r.deck_size)
        infos = _suit_info(r)
        counts, total_unseen = _card_counting(r)
        card, is_discard = _choose_play(r, infos, phase, counts, total_unseen)
        draw = _choose_draw(r, card, is_discard, infos, counts, total_unseen, phase)
        return card, is_discard, draw
