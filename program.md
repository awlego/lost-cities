# Lost Cities: AI Player Improvement Program

## Goal

Maximize the win rate of `players/challenger.py` against `committer` in the Lost Cities card game.

## Rules

1. **Only edit `players/challenger.py`** — this is the ONLY file you may modify
2. The class must remain named `Challenger` with `get_name` returning `'challenger'`
3. Only stdlib imports (no pip packages)
4. You may use anything from `utils.py` and `classes.py`
5. Must be fast: 100k games must complete in under 60 seconds

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

## Iteration Strategy

Each iteration operates in one of two modes. Choose the right one based on the
experiment log before writing any code.

### Exploitation (small optimization)

- Tune thresholds, reorder logic, refine heuristics, add edge cases
- Make ONE focused change per iteration
- Use when the last 2-3 iterations showed steady gains and the current
  architecture still has obvious room for improvement

### Exploration (architectural rewrite)

- Replace the entire decision-making approach with a fundamentally different one
  (see Architecture Archetypes below)
- A rewrite should still be a single coherent commit, but it can
  touch every line in the file
- Use when:
  - Win rate has plateaued (≤0.5% change over 10+ kept iterations)
  - You have a hypothesis for a fundamentally different approach
  - The current architecture makes a category of improvement impossible
    (e.g., no amount of threshold tuning will add card counting)
  - 3+ consecutive iterations have been reverted
  - You simply have a fun or weird idea you want to try

When exploring, preserve a summary of the previous architecture's best ideas
in a comment block at the top of `challenger.py` so good discoveries survive
rewrites. For example:

```python
# === Lessons from previous architectures ===
# - Contract threshold of 15+ expected points worked well
# - Drawing from opponent discard piles defensively improved win rate ~2%
# - Endgame cutoff at deck < 8 was better than deck < 10
# ============================================
```

## Architecture Archetypes

Don't get stuck in one paradigm. Here are fundamentally different approaches to
consider. Each one can be combined with ideas from the others. There are many more than these.

- **Heuristic priority chain**: Ordered if/else rules — simple, fast, easy to tune,
  but rigid and hard to make holistic decisions with
- **Scoring function**: Assign a numeric score to every legal (play, draw) pair and
  pick the max — flexible, easy to add factors, naturally handles tradeoffs
- **Monte Carlo rollouts**: Simulate random game completions to estimate move EV —
  powerful but must stay within speed budget (100k games < 60s)
- **Phased strategy**: Distinct logic for early game (deck > 30), mid game, and
  endgame (deck < 10) — captures how optimal play shifts over a game
- **Opponent modeling + denial**: Track opponent plays/draws to infer their hand
  and priorities, then optimize against that model
- **Card counting + EV**: Track all unseen cards, compute expected future value
  per expedition before committing resources
- **Hybrid approaches**: e.g., scoring function with phase-dependent weights and
  card counting inputs — often the strongest but most complex

When an exploration rewrite is triggered, pick an archetype you haven't tried yet
(or a novel hybrid) rather than reskinning the same logic.

## Workflow

1. Read `experiments/log.jsonl` to see what has been tried (if it exists)
2. Assess the trend and choose Exploitation or Exploration mode (see above)
3. Read `players/challenger.py` to see the current implementation
4. Make your change to `players/challenger.py`
5. Git commit your change: `git add players/challenger.py && git commit -m 'description of change'`
6. Run the benchmark with description and hypothesis flags:
   ```
   uv run python benchmark.py --description 'Brief summary of the code change' --hypothesis 'Why you expect this to improve win rate'
   ```
7. Read the JSON output:
   - `"kept": true` → your commit is kept, it improved win rate
   - `"kept": false` → the benchmark reverted your commit via `git reset --hard HEAD~1`

## Reading the Log

Before each iteration, read `experiments/log.jsonl` and assess:

- **Current win rate** — your baseline to beat
- **Trend** — are the last 3-5 kept iterations showing gains, or has it flattened?
- **Consecutive reverts** — how many attempts in a row have failed?
- **Architecture history** — what approaches have been tried and how did they perform?

Use this analysis to pick your mode and to avoid repeating failed ideas.

