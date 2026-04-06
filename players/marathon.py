"""Go big or go home: chase 1-2 massive expeditions with 8+ cards and contracts.

Strategy:
- Always play if you can (discarding wastes tempo)
- Channel most plays into 1-2 "marathon" target suits aiming for 8+ cards
- Play contracts eagerly in target suits (multiply AND count toward bonus)
- In non-target suits: play low-gap cards to score some points, don't waste turns
- Aggressively draw from discard piles for target suits
- Discard non-target cards when forced, denying opponent
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

        targets = pick_targets(cards, flags, me, r.deck_size)

        playable = [c for c in cards if is_playable(c, flags[c[0]].played[me])]

        if playable:
            play_card = pick_play(playable, targets, flags, me, r.deck_size, cards)
            draw = pick_draw(flags, me, cards, targets, play_card)
            return play_card, False, draw
        else:
            discard = pick_discard(cards, flags, me, targets)
            draw = pick_draw(flags, me, cards, targets, None)
            if draw != 'deck' and draw[0] == discard[0]:
                draw = 'deck'
            return discard, True, draw


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def suit_marathon_score(suit, hand, flags, me, deck_size):
    """How good is this suit as a marathon (8+ card) target?"""
    my_played = flags[suit].played[me]
    hand_in_suit = [c for c in hand if c[0] == suit
                    and is_playable(c, my_played)]

    already = len(my_played)
    holdable = len(hand_in_suit)
    total = already + holdable

    if total == 0:
        return -100.0

    # Count unseen cards
    seen = set()
    for c in my_played + flags[suit].played[1 - me] + flags[suit].discards:
        seen.add(c)
    for c in hand:
        if c[0] == suit:
            seen.add(c)
    unseen = [suit + v for v in CARDS if suit + v not in seen]

    # Top card value after playing hand cards
    all_cards = my_played + hand_in_suit
    top = max(int(c[1]) for c in all_cards)
    future = [c for c in unseen if int(c[1]) >= top]
    max_possible = total + len(future)

    n_contracts = sum(1 for c in all_cards if c[1] == '0')

    score = 0.0

    # Core: can we reach 8?
    if max_possible >= 8:
        score += 50.0
        # Even better if we're already close
        if total >= 5:
            score += 20.0
        elif total >= 4:
            score += 10.0
    elif max_possible >= 6:
        score += 5.0
    else:
        return -50.0

    # Cards in hand (certain) >> unseen cards (probabilistic)
    score += holdable * 7.0
    score += already * 5.0

    # Contracts are huge
    score += n_contracts * 12.0

    # Draw potential
    if deck_size > 0:
        draw_rate = len(future) / max(deck_size + 8, 1)
        expected = min(draw_rate * deck_size / 2, len(future))
        score += expected * 3.0

    # Opponent contesting hurts
    score -= len(flags[suit].played[1 - me]) * 3.0

    # Low start = more room
    non_c = [c for c in all_cards if c[1] != '0']
    if non_c:
        lowest = min(int(c[1]) for c in non_c)
        score += (9 - lowest) * 1.5

    return score


def pick_targets(hand, flags, me, deck_size):
    """Pick 1-2 marathon target suits."""
    scores = {s: suit_marathon_score(s, hand, flags, me, deck_size) for s in SUITS}

    # Always include already-opened suits
    targets = set()
    for s in SUITS:
        if flags[s].played[me]:
            targets.add(s)

    ranked = sorted(SUITS, key=lambda s: scores[s], reverse=True)

    # Best suit is always a target if it has any potential
    if scores[ranked[0]] > 0:
        targets.add(ranked[0])

    # Second suit if it also looks strong
    if scores[ranked[1]] > 30.0:
        targets.add(ranked[1])

    return targets


# ---------------------------------------------------------------------------
# Play selection
# ---------------------------------------------------------------------------

def pick_play(playable, targets, flags, me, deck_size, hand):
    """Pick the best card to play. Always plays (never returns None)."""
    scored = []
    for c in playable:
        suit = c[0]
        played = flags[suit].played[me]
        is_target = suit in targets
        is_new_expedition = not played
        value = 0.0

        gc = gap_cost(c, flags, me)

        if c[1] == '0':
            # Contracts: amazing in target suits, risky elsewhere
            if is_target:
                value += 80.0
                # Better if we have cards to follow up
                followups = sum(1 for x in hand if x[0] == suit and x[1] != '0'
                                and is_playable(x, played + [c]))
                value += followups * 5.0
            elif is_new_expedition:
                # Don't open new expeditions with contracts unless we have
                # a lot of cards in that suit
                suit_hand = [x for x in hand if x[0] == suit]
                if len(suit_hand) >= 4:
                    value += 30.0
                else:
                    value -= 20.0  # Risky — might not fill the expedition
            else:
                # Contract in already-opened non-target: okay
                value += 25.0
        else:
            card_val = int(c[1]) + 1
            mult = 1 + sum(1 for p in played if p[1] == '0')

            # Points scored
            value += card_val * mult * 0.3

            # Gap penalty — bigger for target suits where we want every card
            if is_target:
                value -= gc * 5.0
            else:
                value -= gc * 3.0

            # Target suit bonus: prefer low cards (more growth room)
            if is_target:
                value += (10 - int(c[1])) * 3.0
                value += 15.0  # General target preference

            # Don't open new expeditions unless we have enough cards
            if is_new_expedition:
                suit_hand = [x for x in hand if x[0] == suit
                             and is_playable(x, played)]
                if is_target:
                    value += 10.0  # Encourage starting target suits
                elif len(suit_hand) >= 3 and gc == 0:
                    value += 0.0  # Neutral: okay to start
                elif deck_size < 20:
                    value -= 30.0  # Late game: don't open new ones
                else:
                    value -= 10.0  # Early but not enough cards

        # Approaching 8-card bonus in target suits
        if is_target and played:
            count_after = len(played) + 1
            if count_after >= 8:
                value += 35.0  # We just hit the bonus!
            elif count_after >= 6:
                value += 15.0
            elif count_after >= 4:
                value += 8.0

        scored.append((value, c))

    scored.sort(reverse=True)
    return scored[0][1]


def gap_cost(card, flags, me):
    """How many playable card slots we're skipping over."""
    played = flags[card[0]].played[me]
    baseline = -1
    if played:
        baseline = int(played[-1][1])

    values_left = [x for x in CARDS if int(x) >= baseline]
    if baseline == 0:
        values_left = values_left[1:]

    for other_c in flags[card[0]].played[1 - me] + flags[card[0]].discards:
        v = other_c[1]
        if v in values_left:
            values_left.remove(v)

    if card[1] not in values_left:
        return 0
    return values_left.index(card[1])


# ---------------------------------------------------------------------------
# Discard selection
# ---------------------------------------------------------------------------

def pick_discard(cards, flags, me, targets):
    """Discard a card, strongly preferring non-target suits."""
    non_target = [c for c in cards if c[0] not in targets]
    target_cards = [c for c in cards if c[0] in targets]

    if non_target:
        return best_discard(non_target, flags, me)

    unplayable = [c for c in target_cards
                  if not is_playable(c, flags[c[0]].played[me])]
    if unplayable:
        return best_discard(unplayable, flags, me)

    return best_discard(target_cards, flags, me)


def best_discard(candidates, flags, me):
    """Pick the safest card to discard."""
    useless = [c for c in candidates
               if not is_playable(c, flags[c[0]].played[1 - me])
               and not is_playable(c, flags[c[0]].played[me])]
    if useless:
        return min(useless, key=lambda c: int(c[1]))

    safe = [c for c in candidates
            if not is_playable(c, flags[c[0]].played[1 - me])]
    if safe:
        return min(safe, key=lambda c: int(c[1]))

    return min(candidates, key=lambda c: points_for_opponent(c, flags, me))


# ---------------------------------------------------------------------------
# Draw selection
# ---------------------------------------------------------------------------

def pick_draw(flags, me, hand, targets, play_card):
    """Draw from discard piles for target suits; otherwise deck."""
    best = 'deck'
    best_value = -1

    for suit in SUITS:
        discs = flags[suit].discards
        if not discs:
            continue
        top = discs[-1]

        if play_card and top[0] == play_card[0]:
            continue

        if not is_playable(top, flags[suit].played[me]):
            continue

        if suit in targets:
            val = 30 + int(top[1])
            if top[1] == '0':
                val += 40  # Target suit contracts from discard = jackpot
        elif flags[suit].played[me]:
            # Already opened non-target: grab high playable cards
            if int(top[1]) >= 5:
                val = int(top[1])
            else:
                continue
        else:
            continue

        if val > best_value:
            best_value = val
            best = top

    return best
