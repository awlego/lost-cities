"""NashPG neural network adapter for Lost Cities.

Loads a trained NashPG checkpoint (from the open_spiel repo) and plays it as
a standard Player in this framework.  Only requires torch at runtime — no
pyspiel dependency.

Usage:
    python wrapper.py nashpg committer -n 1000
    NASHPG_CHECKPOINT=/path/to/checkpoint python wrapper.py nashpg committer -n 1000
"""

import json
import os
import pathlib

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from classes import Player, SUITS, CARDS, BREAKEVEN, BONUS_THRESHOLD, BONUS_POINTS

# ---------------------------------------------------------------------------
# Copied from open_spiel (small, stable PyTorch utilities)
# ---------------------------------------------------------------------------

def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class _CategoricalMasked(Categorical):
    def __init__(self, logits=None, masks=None, mask_value=None):
        logits = torch.where(masks.bool(), logits, mask_value)
        super().__init__(logits=logits)


class _NashPGNetwork(nn.Module):
    def __init__(self, info_state_size, num_actions,
                 actor_hidden_layers_sizes=(128, 128),
                 critic_hidden_layers_sizes=(128, 128)):
        super().__init__()
        self.num_actions = num_actions
        actor_layers = []
        in_size = info_state_size
        for h in actor_hidden_layers_sizes:
            actor_layers.append(_layer_init(nn.Linear(in_size, h)))
            actor_layers.append(nn.ReLU())
            in_size = h
        actor_layers.append(_layer_init(nn.Linear(in_size, num_actions), std=0.01))
        self.actor = nn.Sequential(*actor_layers)
        self.register_buffer("mask_value", torch.tensor(-1e6))

    def forward(self, x, legal_actions_mask):
        logits = self.actor(x)
        dist = _CategoricalMasked(
            logits=logits, masks=legal_actions_mask, mask_value=self.mask_value)
        return dist.sample()

# ---------------------------------------------------------------------------
# Constants matching OpenSpiel's lost_cities.h
# ---------------------------------------------------------------------------

_NUM_SUITS = 6
_CARDS_PER_SUIT = 12
_NUM_CONTRACTS = 3
_TOTAL_CARDS = 72
_NUM_ACTIONS = 151
_INFO_STATE_SIZE = 517
_DRAW_ACTION_OFFSET = 144  # kDrawDeckAction
_NUM_CARD_LOCATIONS = 5    # my_hand, my_exp, opp_exp, discard, unknown

_SUIT_INDEX = {'b': 0, 'g': 1, 'p': 2, 'r': 3, 'w': 4, 'y': 5}
_SUIT_CHARS = 'bgprwy'

# Phase enum values matching OpenSpiel.
_PHASE_DEAL = 0
_PHASE_PLAY_DISCARD = 1
_PHASE_DRAW = 2

# ---------------------------------------------------------------------------
# Card encoding helpers
# ---------------------------------------------------------------------------

def _lc_char_to_within_suit(ch):
    """Convert a lost-cities card char to OpenSpiel within_suit index.

    '0' → 0 (contract — caller must handle the 0/1/2 disambiguation),
    '1' → 3, '2' → 4, ..., '9' → 11.
    """
    if ch == '0':
        return 0  # placeholder; contracts need per-instance tracking
    return int(ch) + 2


def _face_value_from_lc_char(ch):
    """Face value matching OpenSpiel's FaceValue(): '0'→0, '1'→2, ..., '9'→10."""
    if ch == '0':
        return 0
    return int(ch) + 1


def _card_id_to_lc_string(card_id):
    """OpenSpiel card_id → lost-cities card string (contracts all become 'X0')."""
    suit = card_id // _CARDS_PER_SUIT
    ws = card_id % _CARDS_PER_SUIT
    suit_char = _SUIT_CHARS[suit]
    if ws < _NUM_CONTRACTS:
        return suit_char + '0'
    return suit_char + str(ws - 2)  # ws=3 → '1', ws=11 → '9'


def _score_expedition(cards_lc):
    """Score an expedition from a list of lost-cities card strings.

    Matches OpenSpiel's ScoreExpedition exactly.
    """
    if not cards_lc:
        return 0.0
    num_contracts = 0
    face_sum = 0
    for c in cards_lc:
        if c[1] == '0':
            num_contracts += 1
        else:
            face_sum += int(c[1]) + 1  # face value
    score = (1 + num_contracts) * (face_sum - BREAKEVEN)
    if len(cards_lc) >= BONUS_THRESHOLD:
        score += BONUS_POINTS
    return float(score)

# ---------------------------------------------------------------------------
# NashPG Player
# ---------------------------------------------------------------------------

class NashPG(Player):
    @classmethod
    def get_name(cls):
        return 'nashpg'

    def __init__(self, p):
        super().__init__(p)
        self._player_id = p

        checkpoint_dir = os.environ.get('NASHPG_CHECKPOINT')
        if not checkpoint_dir:
            # Default to the bundled checkpoint.
            default = pathlib.Path(__file__).parent / 'nashpg-checkpoints' / 'v2_1024x2_mc0.2'
            if default.exists():
                checkpoint_dir = str(default)
            else:
                raise ValueError(
                    'Set NASHPG_CHECKPOINT=/path/to/checkpoint to use the NashPG player')

        checkpoint_dir = pathlib.Path(checkpoint_dir)

        # Load architecture from config.
        with open(checkpoint_dir / 'config.json') as f:
            config = json.load(f)

        hidden = tuple(int(s) for s in config['hidden_layers_sizes'])
        actor_sizes = (tuple(int(s) for s in config['actor_hidden_layers_sizes'])
                       if config.get('actor_hidden_layers_sizes') else hidden)
        # Critic not needed for inference, but we only load actor anyway.

        self._network = _NashPGNetwork(
            _INFO_STATE_SIZE, _NUM_ACTIONS, actor_sizes)
        data = torch.load(checkpoint_dir / 'nash_pg.pt', weights_only=True)
        # Load only the actor weights (ignore critic / optimizer / etc).
        actor_state = {k: v for k, v in data['network'].items()
                       if k.startswith('actor.')}
        self._network.load_state_dict(actor_state, strict=False)
        self._network.eval()

    def play(self, r):
        me = r.whose_turn

        # --- Assign card_ids to all visible cards ---
        hand_card_ids = self._assign_card_ids(r, me)

        # --- Phase 1: PLAY / DISCARD ---
        tensor = self._build_tensor(r, me, hand_card_ids, _PHASE_PLAY_DISCARD)
        mask = self._build_play_discard_mask(r, me, hand_card_ids)

        obs_t = torch.as_tensor(tensor).unsqueeze(0)
        mask_t = torch.as_tensor(mask).unsqueeze(0)
        with torch.no_grad():
            action = self._network(obs_t, mask_t).item()

        # Decode play/discard action.
        chosen_card_id = action // 2
        is_discard = (action % 2 == 1)

        # Find the lost-cities card string in hand for this card_id.
        card_str = None
        for lc_str, cid in hand_card_ids:
            if cid == chosen_card_id:
                card_str = lc_str
                break
        assert card_str is not None, f'Action {action} card_id {chosen_card_id} not in hand'

        chosen_suit_idx = chosen_card_id // _CARDS_PER_SUIT
        last_discard_suit = chosen_suit_idx if is_discard else -1

        # --- Phase 2: DRAW ---
        # Update tensor for the draw phase: move card, flip phase.
        draw_tensor = self._update_tensor_for_draw(
            tensor, r, me, hand_card_ids, chosen_card_id, is_discard)
        draw_mask = self._build_draw_mask(r, last_discard_suit)

        draw_obs_t = torch.as_tensor(draw_tensor).unsqueeze(0)
        draw_mask_t = torch.as_tensor(draw_mask).unsqueeze(0)
        with torch.no_grad():
            draw_action = self._network(draw_obs_t, draw_mask_t).item()

        # Decode draw action.
        if draw_action == _DRAW_ACTION_OFFSET:
            draw_str = 'deck'
        else:
            draw_suit_idx = draw_action - _DRAW_ACTION_OFFSET - 1
            draw_suit_char = _SUIT_CHARS[draw_suit_idx]
            draw_str = r.flags[draw_suit_char].discards[-1]

        return card_str, is_discard, draw_str

    # -------------------------------------------------------------------
    # Card ID assignment
    # -------------------------------------------------------------------

    def _assign_card_ids(self, r, me):
        """Assign OpenSpiel card_ids to all visible cards.

        Returns hand_card_ids: list of (lc_string, card_id) for the hand.
        Also sets self._card_locations: dict card_id → location (0-4).
        """
        # Track next contract index per suit.
        contract_next = [0] * _NUM_SUITS

        # card_id → location mapping (default: unknown=4).
        locations = {}

        def alloc_id(suit_char, value_char):
            s = _SUIT_INDEX[suit_char]
            if value_char == '0':
                ws = contract_next[s]
                contract_next[s] += 1
                return s * _CARDS_PER_SUIT + ws
            else:
                ws = int(value_char) + 2
                return s * _CARDS_PER_SUIT + ws

        # Allocate in stable order: my expeditions, opp expeditions, discards.
        opp = 1 - me
        for suit_char in SUITS:
            for card in r.flags[suit_char].played[me]:
                cid = alloc_id(card[0], card[1])
                locations[cid] = 1  # my_expedition
            for card in r.flags[suit_char].played[opp]:
                cid = alloc_id(card[0], card[1])
                locations[cid] = 2  # opp_expedition
            for card in r.flags[suit_char].discards:
                cid = alloc_id(card[0], card[1])
                locations[cid] = 3  # discard

        # Now allocate hand cards.
        hand_card_ids = []
        for card in r.hand.cards:
            cid = alloc_id(card[0], card[1])
            locations[cid] = 0  # my_hand
            hand_card_ids.append((card, cid))

        self._card_locations = locations
        return hand_card_ids

    # -------------------------------------------------------------------
    # Tensor construction
    # -------------------------------------------------------------------

    def _build_tensor(self, r, me, hand_card_ids, phase):
        """Build the 517-dim enriched observation tensor."""
        tensor = np.zeros(_INFO_STATE_SIZE, dtype=np.float32)
        opp = 1 - me

        # player one-hot (offset 0, size 2)
        tensor[me] = 1.0

        # card_locations (offset 2, shape [6, 12, 5])
        off = 2
        # Default all to unknown (location 4).
        for c in range(_TOTAL_CARDS):
            s = c // _CARDS_PER_SUIT
            ws = c % _CARDS_PER_SUIT
            tensor[off + (s * _CARDS_PER_SUIT + ws) * _NUM_CARD_LOCATIONS + 4] = 1.0

        # Overwrite known locations.
        for cid, loc in self._card_locations.items():
            s = cid // _CARDS_PER_SUIT
            ws = cid % _CARDS_PER_SUIT
            idx = off + (s * _CARDS_PER_SUIT + ws) * _NUM_CARD_LOCATIONS
            tensor[idx + 4] = 0.0  # clear unknown
            tensor[idx + loc] = 1.0

        # discard_order (offset 362, shape [6, 12])
        off = 362
        for suit_char in SUITS:
            s = _SUIT_INDEX[suit_char]
            for i, card in enumerate(r.flags[suit_char].discards):
                fv = _face_value_from_lc_char(card[1])
                tensor[off + s * _CARDS_PER_SUIT + i] = fv / 10.0

        # deck_size (offset 434, size 1)
        tensor[434] = r.deck_size / _TOTAL_CARDS

        # phase (offset 435, size 4)
        tensor[435 + phase] = 1.0

        # Derived per-player per-suit features.
        # Player order: [me, opp] — indices pi=0 is me, pi=1 is opponent.
        players = [me, opp]

        # wager_count (offset 439, shape [2, 6])
        off = 439
        for pi, p in enumerate(players):
            for suit_char in SUITS:
                s = _SUIT_INDEX[suit_char]
                wagers = sum(1 for c in r.flags[suit_char].played[p] if c[1] == '0')
                tensor[off + pi * _NUM_SUITS + s] = wagers / 3.0

        # face_sum (offset 451, shape [2, 6])
        off = 451
        for pi, p in enumerate(players):
            for suit_char in SUITS:
                s = _SUIT_INDEX[suit_char]
                fsum = sum(_face_value_from_lc_char(c[1])
                           for c in r.flags[suit_char].played[p])
                tensor[off + pi * _NUM_SUITS + s] = fsum / 54.0

        # expedition_score (offset 463, shape [2, 6])
        off = 463
        for pi, p in enumerate(players):
            for suit_char in SUITS:
                s = _SUIT_INDEX[suit_char]
                score = _score_expedition(r.flags[suit_char].played[p])
                tensor[off + pi * _NUM_SUITS + s] = (score + 80.0) / 236.0

        # expedition_started (offset 475, shape [2, 6])
        off = 475
        for pi, p in enumerate(players):
            for suit_char in SUITS:
                s = _SUIT_INDEX[suit_char]
                tensor[off + pi * _NUM_SUITS + s] = (
                    1.0 if r.flags[suit_char].played[p] else 0.0)

        # min_playable_number (offset 487, shape [2, 6])
        off = 487
        for pi, p in enumerate(players):
            for suit_char in SUITS:
                s = _SUIT_INDEX[suit_char]
                played = r.flags[suit_char].played[p]
                if not played:
                    tensor[off + pi * _NUM_SUITS + s] = 0.0
                else:
                    top_fv = _face_value_from_lc_char(played[-1][1])
                    min_val = 2 if top_fv == 0 else top_fv
                    tensor[off + pi * _NUM_SUITS + s] = min_val / 10.0

        # cards_per_expedition (offset 499, shape [2, 6])
        off = 499
        for pi, p in enumerate(players):
            for suit_char in SUITS:
                s = _SUIT_INDEX[suit_char]
                tensor[off + pi * _NUM_SUITS + s] = (
                    len(r.flags[suit_char].played[p]) / 12.0)

        # unknown_per_suit (offset 511, shape [6])
        off = 511
        for suit_char in SUITS:
            s = _SUIT_INDEX[suit_char]
            hand_in_suit = sum(1 for c in r.hand.cards if c[0] == suit_char)
            exp_me = len(r.flags[suit_char].played[me])
            exp_opp = len(r.flags[suit_char].played[opp])
            disc = len(r.flags[suit_char].discards)
            unknown = _CARDS_PER_SUIT - hand_in_suit - exp_me - exp_opp - disc
            tensor[off + s] = unknown / 12.0

        return tensor

    # -------------------------------------------------------------------
    # Draw-phase tensor update
    # -------------------------------------------------------------------

    def _update_tensor_for_draw(self, base_tensor, r, me, hand_card_ids,
                                played_card_id, is_discard):
        """Return a copy of base_tensor updated for the DRAW phase."""
        tensor = base_tensor.copy()

        s = played_card_id // _CARDS_PER_SUIT
        ws = played_card_id % _CARDS_PER_SUIT

        # Move card from my_hand (0) to my_expedition (1) or discard (3).
        loc_off = 2 + (s * _CARDS_PER_SUIT + ws) * _NUM_CARD_LOCATIONS
        tensor[loc_off + 0] = 0.0  # clear my_hand
        new_loc = 3 if is_discard else 1
        tensor[loc_off + new_loc] = 1.0

        # Flip phase: clear PLAY_DISCARD (1), set DRAW (2).
        tensor[435 + _PHASE_PLAY_DISCARD] = 0.0
        tensor[435 + _PHASE_DRAW] = 1.0

        # Update derived features for the affected suit.
        suit_char = _SUIT_CHARS[s]
        card_lc_char = _card_id_to_lc_string(played_card_id)[1]
        # For contracts, card_lc_char is '0'; for numbers it's the digit.
        card_fv = _face_value_from_lc_char(card_lc_char)

        if not is_discard:
            # Card was played to my expedition — update pi=0 (me) features.
            played_list = r.flags[suit_char].played[me]
            new_cards = played_list + [suit_char + card_lc_char]

            # wager_count
            wagers = sum(1 for c in new_cards if c[1] == '0')
            tensor[439 + 0 * _NUM_SUITS + s] = wagers / 3.0

            # face_sum
            fsum = sum(_face_value_from_lc_char(c[1]) for c in new_cards)
            tensor[451 + 0 * _NUM_SUITS + s] = fsum / 54.0

            # expedition_score
            score = _score_expedition(new_cards)
            tensor[463 + 0 * _NUM_SUITS + s] = (score + 80.0) / 236.0

            # expedition_started
            tensor[475 + 0 * _NUM_SUITS + s] = 1.0

            # min_playable_number
            top_fv = _face_value_from_lc_char(new_cards[-1][1])
            min_val = 2 if top_fv == 0 else top_fv
            tensor[487 + 0 * _NUM_SUITS + s] = min_val / 10.0

            # cards_per_expedition
            tensor[499 + 0 * _NUM_SUITS + s] = len(new_cards) / 12.0

        else:
            # Card was discarded — update discard_order for this suit.
            disc_len = len(r.flags[suit_char].discards)
            tensor[362 + s * _CARDS_PER_SUIT + disc_len] = card_fv / 10.0

        # unknown_per_suit: hand lost one card in this suit.
        # (The card moved from hand to expedition/discard, both visible.)
        # No change to unknown count — the card was already known (in hand).

        return tensor

    # -------------------------------------------------------------------
    # Legal action masks
    # -------------------------------------------------------------------

    def _build_play_discard_mask(self, r, me, hand_card_ids):
        """Legal action mask for the PLAY_DISCARD phase."""
        mask = np.zeros(_NUM_ACTIONS, dtype=np.bool_)
        seen = set()

        for lc_str, card_id in hand_card_ids:
            suit_char = lc_str[0]
            played = r.flags[suit_char].played[me]

            # Play action: legal if playable (ascending face value).
            play_action = card_id * 2
            if play_action not in seen:
                if not played or _face_value_from_lc_char(lc_str[1]) >= _face_value_from_lc_char(played[-1][1]):
                    mask[play_action] = True
                seen.add(play_action)

            # Discard action: always legal.
            disc_action = card_id * 2 + 1
            if disc_action not in seen:
                mask[disc_action] = True
                seen.add(disc_action)

        return mask

    def _build_draw_mask(self, r, last_discard_suit):
        """Legal action mask for the DRAW phase."""
        mask = np.zeros(_NUM_ACTIONS, dtype=np.bool_)

        # Draw from deck (deck_size counts before our play, but we removed
        # one card from hand and added one to expedition/discard — deck unchanged).
        if r.deck_size > 0:
            mask[_DRAW_ACTION_OFFSET] = True

        # Draw from discard piles.
        for suit_char in SUITS:
            s = _SUIT_INDEX[suit_char]
            if s == last_discard_suit:
                continue  # can't draw from the pile we just discarded to
            if r.flags[suit_char].discards:
                mask[_DRAW_ACTION_OFFSET + 1 + s] = True

        return mask
