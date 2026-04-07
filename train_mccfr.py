"""Outcome Sampling MCCFR with Linear Weighting for Lost Cities.

Uses both state abstraction (bucketed info set keys with suit symmetry) and
action abstraction (role-based abstract actions) so that strategies generalize
across different concrete hands.

Usage:
    uv run python train_mccfr.py --iterations 1000000 --eval-every 50000
"""

import argparse
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict

sys.setrecursionlimit(5000)

from classes import Round, SUITS, CARDS, HAND_SIZE, N_PLAYERS, BREAKEVEN, \
    BONUS_THRESHOLD, BONUS_POINTS
from utils import is_playable


# ---------------------------------------------------------------------------
# Abstract actions
# ---------------------------------------------------------------------------
# Play phase: role-based categories
A_PLAY_TIGHT = 'PT'     # Play to started expedition, gap <= 1
A_PLAY_LOOSE = 'PL'     # Play to started expedition, gap > 1
A_PLAY_NEW = 'PN'       # Start a new expedition (non-contract)
A_PLAY_CONTRACT = 'PC'  # Play a contract card
A_DISC_USELESS = 'DU'   # Discard: neither player can use
A_DISC_SAFE = 'DS'      # Discard: opponent can't use
A_DISC_RISKY = 'DR'     # Discard: opponent could use

# Draw phase
A_DRAW_DECK = 'DD'      # Draw from deck
A_DRAW_PILE = 'DP'      # Draw from a discard pile


def classify_play_actions(r):
    """Classify concrete actions into abstract categories.

    Returns: dict of abstract_action -> list of concrete (card, is_discard).
    Also returns a flat list of available abstract actions (for CFR).
    """
    me = r.whose_turn
    hand = r.h[me].cards
    flags = r.flags

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

        # Discard classification
        if not playable and not is_playable(card, opp_played):
            cats[A_DISC_USELESS].append((card, True))
        elif not is_playable(card, opp_played):
            cats[A_DISC_SAFE].append((card, True))
        else:
            cats[A_DISC_RISKY].append((card, True))

    return cats


def classify_draw_actions(r, disc_suit):
    """Classify draw actions into abstract categories."""
    cats = {}
    if r.deck:
        cats[A_DRAW_DECK] = ['deck']
    pile_options = []
    for s in SUITS:
        if r.flags[s].discards and s != disc_suit:
            pile_options.append(r.flags[s].discards[-1])
    if pile_options:
        cats[A_DRAW_PILE] = pile_options
    return cats


def pick_concrete_play(abstract, candidates):
    """Select a concrete action from candidates for an abstract play action.

    Heuristic: prefer lower gap for plays, lower value for discards.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # For plays: prefer lowest card value (minimizes gap)
    # For discards: prefer lowest card value (least info to opponent)
    return min(candidates, key=lambda x: x[0][1])


def pick_concrete_draw(abstract, candidates):
    """Select a concrete draw from candidates."""
    if not candidates:
        return None
    if abstract == A_DRAW_DECK:
        return 'deck'
    # For pile draws: prefer highest value card
    return max(candidates, key=lambda x: x[1] if x != 'deck' else '0')


# ---------------------------------------------------------------------------
# Fast state cloning
# ---------------------------------------------------------------------------

def clone_round(r):
    """Deep-copy a Round without copy.deepcopy."""
    r2 = object.__new__(Round)
    r2.verbose = False
    r2.whose_turn = r.whose_turn
    r2.flags = {}
    for s in SUITS:
        f = object.__new__(Round.Flag)
        f.played = [list(r.flags[s].played[0]), list(r.flags[s].played[1])]
        f.discards = list(r.flags[s].discards)
        r2.flags[s] = f
    r2.h = [None, None]
    for i in range(N_PLAYERS):
        h = object.__new__(Round.Hand)
        h.cards = list(r.h[i].cards)
        h.seat = r.h[i].seat
        h.name = r.h[i].name
        r2.h[i] = h
    r2.deck = list(r.deck)
    return r2


# ---------------------------------------------------------------------------
# Bucketed information set key (with suit symmetry)
# ---------------------------------------------------------------------------

def infoset_key(r, player_id, phase):
    """Coarse info set key: per-suit features, sorted for suit symmetry."""
    deck_size = len(r.deck)
    dp = 3 if deck_size > 32 else (2 if deck_size > 20 else (1 if deck_size > 10 else 0))

    hand = r.h[player_id].cards
    opp = 1 - player_id

    suit_codes = []
    for s in SUITS:
        f = r.flags[s]
        my_played = f.played[player_id]
        opp_played = f.played[opp]

        my_n = len(my_played)
        my_bucket = 0 if my_n == 0 else (1 if my_n <= 3 else 2)

        opp_bucket = 0 if len(opp_played) == 0 else 1

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
    suit_str = ''.join(chr(65 + c) for c in suit_codes)
    return f"{dp}{phase}{suit_str}"


# ---------------------------------------------------------------------------
# Regret matching & sampling
# ---------------------------------------------------------------------------

def regret_matching(regret_dict, actions):
    """Compute strategy proportional to positive cumulative regrets."""
    positives = [max(regret_dict.get(a, 0.0), 0.0) for a in actions]
    total = sum(positives)
    if total > 0:
        return {a: p / total for a, p in zip(actions, positives)}
    n = len(actions)
    return {a: 1.0 / n for a in actions}


def sample_action(strategy):
    """Sample one action from a strategy distribution."""
    actions = list(strategy.keys())
    probs = [strategy[a] for a in actions]
    return random.choices(actions, weights=probs, k=1)[0]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def get_scores(r):
    scores = [0, 0]
    for p in range(N_PLAYERS):
        for f in r.flags.values():
            cards = f.played[p]
            if not cards:
                continue
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
            scores[p] += score
    return scores


def sigmoid_reward(r, me, k=20.0):
    scores = get_scores(r)
    diff = scores[me] - scores[1 - me]
    return 1.0 / (1.0 + math.exp(-diff / k))


# ---------------------------------------------------------------------------
# Depth-Limited External Sampling CFR with abstract actions
# ---------------------------------------------------------------------------
#
# Within the depth limit: enumerate ALL traverser actions (proper CFR updates),
# sample ONE opponent action. Beyond the limit: rollout to terminal using
# current strategy.
#
# With ~5-7 abstract play actions and ~2 draw options, each traverser turn
# branches ~10-14 ways. At depth 6 (3 traverser turns): ~10^3 = 1000 branches.

SEARCH_DEPTH = 4   # Total turns to search (2 traverser + 2 opponent)
MAX_ROLLOUT = 200   # Safety limit for rollout turns


def rollout_eval(r, traverser, regrets):
    """Play out the game using current regret-matched policy. Returns utility."""
    turns = 0
    while r.deck and turns < MAX_ROLLOUT:
        me = r.whose_turn
        cats = classify_play_actions(r)
        if not cats:
            break

        # Use learned strategy if available, else heuristic priority
        key = infoset_key(r, me, 'P')
        abstract_actions = list(cats.keys())
        r_entry = regrets.get(key)
        if r_entry:
            strategy = regret_matching(r_entry, abstract_actions)
        else:
            strategy = {a: 1.0 / len(abstract_actions) for a in abstract_actions}
        chosen = sample_action(strategy)
        card, is_discard = pick_concrete_play(chosen, cats[chosen])
        disc_suit = apply_play_action(r, card, is_discard)

        # Draw phase
        draw_cats = classify_draw_actions(r, disc_suit)
        draw_actions = list(draw_cats.keys())
        if not draw_actions:
            break
        if len(draw_actions) == 1:
            draw_chosen = draw_actions[0]
        else:
            dkey = infoset_key(r, me, 'D')
            d_entry = regrets.get(dkey)
            if d_entry:
                dstrat = regret_matching(d_entry, draw_actions)
            else:
                dstrat = {a: 1.0 / len(draw_actions) for a in draw_actions}
            draw_chosen = sample_action(dstrat)
        apply_draw_action(r, pick_concrete_draw(draw_chosen, draw_cats[draw_chosen]))
        turns += 1

    return sigmoid_reward(r, traverser)


def ext_cfr_play(r, traverser, regrets, strategies, t, depth):
    """External sampling CFR at play/discard node.

    depth: remaining turns before switching to rollout.
    At traverser nodes: enumerate all abstract actions (clone state).
    At opponent nodes: sample one action (modify in place).
    At depth 0: rollout to terminal.
    """
    if not r.deck:
        return sigmoid_reward(r, traverser)
    if depth <= 0:
        return rollout_eval(r, traverser, regrets)

    me = r.whose_turn
    key = infoset_key(r, me, 'P')
    cats = classify_play_actions(r)
    abstract_actions = list(cats.keys())

    if not abstract_actions:
        return sigmoid_reward(r, traverser)

    r_dict = regrets[key]
    strategy = regret_matching(r_dict, abstract_actions)

    if me == traverser:
        # ENUMERATE all abstract actions
        values = {}
        for a in abstract_actions:
            concrete = pick_concrete_play(a, cats[a])
            r2 = clone_round(r)
            disc_suit = apply_play_action(r2, concrete[0], concrete[1])
            values[a] = ext_cfr_draw(r2, traverser, regrets, strategies,
                                     disc_suit, t, depth)

        ev = sum(strategy[a] * values[a] for a in abstract_actions)
        # Standard CFR regret update with linear weighting
        for a in abstract_actions:
            r_dict[a] = r_dict.get(a, 0.0) + t * (values[a] - ev)
        return ev

    else:
        # SAMPLE one opponent action
        sampled = sample_action(strategy)
        s_dict = strategies[key]
        for a in abstract_actions:
            s_dict[a] = s_dict.get(a, 0.0) + t * strategy[a]
        concrete = pick_concrete_play(sampled, cats[sampled])
        disc_suit = apply_play_action(r, concrete[0], concrete[1])
        return ext_cfr_draw(r, traverser, regrets, strategies,
                            disc_suit, t, depth)


def ext_cfr_draw(r, traverser, regrets, strategies, disc_suit, t, depth):
    """External sampling CFR at draw node."""
    me = r.whose_turn
    cats = classify_draw_actions(r, disc_suit)
    abstract_actions = list(cats.keys())

    if not abstract_actions:
        return sigmoid_reward(r, traverser)

    # Single option: no decision, just apply
    if len(abstract_actions) == 1:
        apply_draw_action(r, pick_concrete_draw(abstract_actions[0],
                                                 cats[abstract_actions[0]]))
        return ext_cfr_play(r, traverser, regrets, strategies, t, depth - 1)

    key = infoset_key(r, me, 'D')
    r_dict = regrets[key]
    strategy = regret_matching(r_dict, abstract_actions)

    if me == traverser:
        values = {}
        for a in abstract_actions:
            concrete = pick_concrete_draw(a, cats[a])
            r2 = clone_round(r)
            apply_draw_action(r2, concrete)
            # Depth decrements after a full turn completes (after draw)
            values[a] = ext_cfr_play(r2, traverser, regrets, strategies,
                                     t, depth - 1)

        ev = sum(strategy[a] * values[a] for a in abstract_actions)
        for a in abstract_actions:
            r_dict[a] = r_dict.get(a, 0.0) + t * (values[a] - ev)
        return ev

    else:
        sampled = sample_action(strategy)
        s_dict = strategies[key]
        for a in abstract_actions:
            s_dict[a] = s_dict.get(a, 0.0) + t * strategy[a]
        apply_draw_action(r, pick_concrete_draw(sampled, cats[sampled]))
        return ext_cfr_play(r, traverser, regrets, strategies, t, depth - 1)


def apply_play_action(r, card, is_discard):
    me = r.whose_turn
    r.h[me].drop(card)
    suit = card[0]
    if is_discard:
        r.flags[suit].discards.append(card)
        return suit
    else:
        r.flags[suit].played[me].append(card)
        return None


def apply_draw_action(r, draw_target):
    me = r.whose_turn
    drawn_card = r.draw(draw_target)
    r.h[me].add(drawn_card)
    r.whose_turn = 1 - me


# ---------------------------------------------------------------------------
# Training round creation
# ---------------------------------------------------------------------------

def new_training_round():
    r = object.__new__(Round)
    r.verbose = False
    r.whose_turn = 0
    r.flags = {s: Round.Flag() for s in SUITS}
    r.h = [Round.Hand(0, 'P0'), Round.Hand(1, 'P1')]
    r.deck = [s + c for s in SUITS for c in CARDS]
    random.shuffle(r.deck)
    for h in r.h:
        h.cards.extend(r.deck[-HAND_SIZE:])
        del r.deck[-HAND_SIZE:]
    return r


# ---------------------------------------------------------------------------
# Evaluation: play MCCFR strategy vs committer
# ---------------------------------------------------------------------------

def _view_key(view, me, phase):
    """Compute bucketed key from a PlayerView."""
    deck_size = view.deck_size
    dp = 3 if deck_size > 32 else (2 if deck_size > 20 else (1 if deck_size > 10 else 0))
    hand = view.hand.cards
    opp = 1 - me
    suit_codes = []
    for s in SUITS:
        f = view.flags[s]
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


def make_mccfr_play_fn(avg_strategy):
    """Return a play function using the average strategy with abstract actions."""

    def play_fn(view):
        me = view.whose_turn
        hand = view.hand.cards
        flags = view.flags

        # Phase 1: classify and choose abstract play action
        # Build a minimal Round-like object for classification
        r_proxy = type('R', (), {
            'whose_turn': me,
            'h': {me: view.hand},
            'flags': flags,
        })()
        # Use the classify function logic directly
        cats = _classify_play_from_view(hand, flags, me)
        abstract_actions = list(cats.keys())

        key1 = _view_key(view, me, 'P')
        strat = avg_strategy.get(key1, {})

        # Pick abstract action from strategy
        legal_strat = {a: strat.get(a, 0.0) for a in abstract_actions}
        total = sum(legal_strat.values())
        if total > 0:
            chosen = sample_action({a: v / total for a, v in legal_strat.items()})
        else:
            # Fallback priority: play tight > play loose > play new > disc useless > disc safe > disc risky
            priority = [A_PLAY_TIGHT, A_PLAY_LOOSE, A_PLAY_CONTRACT, A_PLAY_NEW,
                        A_DISC_USELESS, A_DISC_SAFE, A_DISC_RISKY]
            chosen = next((a for a in priority if a in cats), abstract_actions[0])

        card, is_discard = pick_concrete_play(chosen, cats[chosen])

        # Phase 2: draw
        disc_suit = card[0] if is_discard else None
        draw_cats = _classify_draw_from_view(flags, disc_suit, view.deck_size)
        draw_abstracts = list(draw_cats.keys())

        if len(draw_abstracts) <= 1:
            draw = pick_concrete_draw(draw_abstracts[0], draw_cats[draw_abstracts[0]]) if draw_abstracts else 'deck'
        else:
            # For draw phase key, adjust state for the played/discarded card
            key2 = _view_key(view, me, 'D')  # Approximate (doesn't adjust for played card)
            strat2 = avg_strategy.get(key2, {})
            legal_strat2 = {a: strat2.get(a, 0.0) for a in draw_abstracts}
            total2 = sum(legal_strat2.values())
            if total2 > 0:
                chosen_draw = sample_action({a: v / total2 for a, v in legal_strat2.items()})
            else:
                chosen_draw = A_DRAW_DECK if A_DRAW_DECK in draw_cats else draw_abstracts[0]
            draw = pick_concrete_draw(chosen_draw, draw_cats[chosen_draw])

        return card, is_discard, draw

    return play_fn


def _classify_play_from_view(hand, flags, me):
    """Classify play actions from view data (no Round object needed)."""
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


def _classify_draw_from_view(flags, disc_suit, deck_size):
    """Classify draw actions from view data."""
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


def evaluate_vs_committer(avg_strategy, num_games=1000):
    from players.committer import Committer
    from classes import PlayerView

    play_fn = make_mccfr_play_fn(avg_strategy)
    committer = Committer(1)
    wins = 0

    for game_idx in range(num_games):
        r = new_training_round()
        if game_idx % 2 == 1:
            r.whose_turn = 1

        mccfr_seat = 0
        while r.deck:
            me = r.whose_turn
            if me == mccfr_seat:
                view = type('V', (), {
                    'whose_turn': me,
                    'flags': r.flags,
                    'hand': r.h[me],
                    'deck_size': len(r.deck),
                })()
                card, is_discard, draw = play_fn(view)
            else:
                pv = PlayerView(r, me)
                card, is_discard, draw = committer.play(pv)

            r.h[me].drop(card)
            suit = card[0]
            if is_discard:
                r.flags[suit].discards.append(card)
            else:
                r.flags[suit].played[me].append(card)
            drawn_card = r.draw(draw)
            r.h[me].add(drawn_card)
            r.whose_turn = 1 - me

        scores = get_scores(r)
        if scores[mccfr_seat] > scores[1 - mccfr_seat]:
            wins += 1

    return wins / num_games


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def get_average_strategy(strategies):
    avg = {}
    for key, action_sums in strategies.items():
        total = sum(action_sums.values())
        if total > 0:
            avg[key] = {a: v / total for a, v in action_sums.items()}
    return avg


def save_checkpoint(regrets, strategies, iteration, path):
    os.makedirs(path, exist_ok=True)
    data = {
        'regrets': dict(regrets),
        'strategies': dict(strategies),
        'iteration': iteration,
    }
    filepath = os.path.join(path, f'checkpoint_{iteration}.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    latest = os.path.join(path, 'latest.pkl')
    with open(latest, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return filepath


def save_strategy_only(strategies, iteration, path):
    os.makedirs(path, exist_ok=True)
    avg = get_average_strategy(strategies)
    filepath = os.path.join(path, 'strategy.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump({'strategy': avg, 'iteration': iteration},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    return filepath


def load_checkpoint(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    regrets = defaultdict(dict, data['regrets'])
    strategies = defaultdict(dict, data['strategies'])
    return regrets, strategies, data['iteration']


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(num_iterations, eval_every=50000, checkpoint_every=100000,
          checkpoint_dir='mccfr-checkpoints'):
    regrets = defaultdict(dict)
    strategies = defaultdict(dict)

    start_time = time.time()

    for t in range(1, num_iterations + 1):
        traverser = t % 2
        r = new_training_round()
        ext_cfr_play(r, traverser, regrets, strategies, t, SEARCH_DEPTH)

        if t % 5000 == 0:
            elapsed = time.time() - start_time
            rate = t / elapsed
            print(f'[iter {t:>8d}] {rate:.0f} iter/s | '
                  f'{len(regrets):,} info sets | '
                  f'{elapsed:.1f}s elapsed')

        if t % eval_every == 0:
            avg = get_average_strategy(strategies)
            wr = evaluate_vs_committer(avg, num_games=500)
            print(f'  >> eval @ {t}: {wr:.1%} vs committer (500 games)')

        if t % checkpoint_every == 0:
            fp = save_checkpoint(regrets, strategies, t, checkpoint_dir)
            print(f'  >> checkpoint saved: {fp}')

    save_checkpoint(regrets, strategies, num_iterations, checkpoint_dir)
    save_strategy_only(strategies, num_iterations, checkpoint_dir)
    avg = get_average_strategy(strategies)
    wr = evaluate_vs_committer(avg, num_games=1000)
    print(f'\nTraining complete: {num_iterations} iterations')
    print(f'Final win rate vs committer: {wr:.1%} (1000 games)')
    print(f'Info sets: {len(regrets):,}')
    print(f'Strategy saved to {checkpoint_dir}/strategy.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MCCFR for Lost Cities')
    parser.add_argument('--iterations', '-n', type=int, default=100000,
                        help='Number of training iterations')
    parser.add_argument('--eval-every', type=int, default=50000,
                        help='Evaluate every N iterations')
    parser.add_argument('--checkpoint-every', type=int, default=100000,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--checkpoint-dir', type=str, default='mccfr-checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    if args.resume:
        print(f'Resuming from {args.resume}')
        regrets, strategies, start_iter = load_checkpoint(args.resume)
        print(f'Loaded {len(regrets):,} info sets from iteration {start_iter}')

    train(args.iterations, args.eval_every, args.checkpoint_every,
          args.checkpoint_dir)
