# Lost Cities: AI Player Improvement Program

## Goal

Maximize the win rate of `players/challenger.py` against `committer` in the Lost Cities card game.

## Rules

1. **Only edit `players/challenger.py`** — this is the ONLY file you may modify
2. The class must remain named `Challenger` with `get_name` returning `'challenger'`
3. Only stdlib imports (no pip packages)
4. You may use anything from `utils.py` and `classes.py`
5. Make ONE focused change per iteration

## Off-Limits Files — DO NOT MODIFY

The following files are the game engine, harness, and other players. Modifying them
would invalidate all benchmarks and break the experiment loop.

- `classes.py` — game engine (Round, Player, Flag, Hand)
- `play.py` — game loop
- `wrapper.py` — CLI runner
- `utils.py` — shared utilities (you may call these, but not edit them)
- `benchmark.py` — benchmark harness
- `program.md` — this file
- `players/kenny.py` — baseline player
- `players/discarder.py` — existing player
- `players/committer.py` — existing player (your opponent)
- `players/granny.py` — existing player
- `players/__init__.py` — player loader
- Any file in `experiments/` — managed by the harness

If you need a new helper function, define it inside `players/challenger.py`.

## Workflow

1. Read `experiments/log.jsonl` to see what has been tried (if it exists)
2. Read `players/challenger.py` to see the current implementation
3. Make one change to `players/challenger.py`
4. Run the benchmark with description and hypothesis flags:
   ```
   uv run python benchmark.py --description 'Brief summary of the code change' --hypothesis 'Why you expect this to improve win rate'
   ```
5. Read the JSON output:
   - `"kept": true` → your change was accepted, it improved win rate
   - `"kept": false` → your change was reverted, try something different

## Game Rules

Lost Cities is a two-player card game with 6 suits (`b g p r w y`) and 12 cards per suit (3 contracts `0` + values `2-10`). Each player builds "expeditions" by playing cards in ascending order per suit. The opponent's hand is not public and you may not look at it.

**Turn:** Play or discard one card, then draw one card (from deck or a discard pile).

**Scoring per suit:**
- Sum card face values (each card's digit + 1), subtract 20 (the expedition cost)
- Multiply by (1 + number of contracts played in that suit)
- If 8+ cards played in a suit, add 20 bonus points
- Unopened expeditions score 0 (not negative)

**Card encoding:** Two chars `[suit][value]`, e.g. `'b3'` = blue 4-point card, `'r0'` = red contract.

## Available State in `play(self, r)`

```python
me = r.whose_turn                    # Your seat (0 or 1)
r.h[me].cards                        # Your hand (list of card strings)
r.flags[suit].played[me]             # Cards you've played in this suit
r.flags[suit].played[1-me]           # Cards opponent played in this suit
r.flags[suit].discards               # Face-up discard pile for this suit
len(r.deck)                          # Cards remaining in draw pile
```

Opponent's hand is hidden. Deck contents are hidden.

## Utility Functions (from utils.py)

- `is_playable(card, played)` — can this card be played on this expedition?
- `playable_draws(flags, me)` — cards available from discard piles that you can play
- `minimize_gap(cards, flags, me)` — find the card that skips fewest values
- `discard_intelligently(cards, flags, me)` — pick the least-damaging discard
- `safe_discards(cards, flags, me)` — cards opponent can't use
- `useless_discards(cards, flags, me)` — cards neither player can use
- `sum_cards(cards)` — total face value of a card list

## Constants (from classes.py)

```python
SUITS = 'bgprwy'      # 6 suits
CARDS = '000123456789' # 12 cards per suit (3 zeros = contracts)
HAND_SIZE = 8
BREAKEVEN = 20         # Cost to open an expedition
BONUS_THRESHOLD = 8    # Cards needed for +20 bonus
BONUS_POINTS = 20
```

## Strategy Ideas

- **Card counting**: Track which cards are still unseen to estimate probabilities
- **Opponent modeling**: Infer what suits they're collecting from their plays/draws
- **Expected value**: Calculate EV before opening an expedition (consider cards remaining)
- **Endgame awareness**: Adjust strategy based on `len(r.deck)` — be more aggressive early, conservative late
- **Denial play**: Avoid discarding cards the opponent needs; draw from their discard piles defensively
- **Contract optimization**: Contracts multiply both gains AND losses — only play when the expedition will clearly profit
- **Suit selection**: Focus on fewer suits rather than spreading thin
- **Draw optimization**: Drawing from discard piles gives information; drawing from deck is random but doesn't tip your hand

## Constraints

- Must be fast: 100k games must complete in under 60 seconds
- No external dependencies
- No file I/O or network calls from within the player
