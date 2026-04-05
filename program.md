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

## Execution Model

You are a single long-lived Claude session running an open-ended research loop.
Execute experiments indefinitely until the user manually interrupts you (Ctrl+C).

### Your Loop

Repeat the following cycle without pausing or asking for permission:

1. **Analyze** — Review recent results (from conversation context) and decide
   whether to exploit or explore (see Iteration Strategy below)
2. **Implement** — Make your change to `players/challenger.py`
3. **Commit** — `git add players/challenger.py && git commit -m 'description'`
4. **Benchmark** — Run `uv run python benchmark.py --description '...' --hypothesis '...'`
5. **Interpret** — Read the JSON output. If kept, note the new baseline. If reverted
   or errored, diagnose why and factor it into your next decision.
6. **Go to 1** — Do not pause. Do not ask "should I continue?" Do not summarize
   and wait for input. Just start the next experiment.

### Autonomy Directive

- Never pause to ask the user for permission or confirmation
- Never say "shall I continue?" or "would you like me to try another approach?"
- Never output a summary and then stop — always proceed to the next experiment
- The user will interrupt you with Ctrl+C when they want you to stop
- You are expected to run dozens or hundreds of experiments autonomously

### Timeout and Crash Handling

- `benchmark.py` enforces a 300-second timeout internally. If your code is too
  slow, the benchmark will time out, revert your commit, and return an error.
  Treat this as a failed experiment and move on.
- If the benchmark crashes (exit code 1 with an error message), read the error:
  - **Syntax error or obvious bug**: Fix it immediately, re-commit, and re-run
  - **Fundamental design flaw** (e.g., impossible import, wrong class structure):
    Abandon the approach and try something different
  - **Intermittent/unclear error**: Try one more time; if it fails again, move on
- If you encounter a git conflict or dirty working tree, run `git status` to
  diagnose, then `git checkout players/challenger.py` to reset and continue

### Context Management

- You remember recent experiments from conversation context — you do not need to
  re-read `experiments/log.jsonl` every iteration
- Re-read `experiments/log.jsonl` in these situations:
  - At the very start of your session (iteration 1)
  - Every 10 experiments, as a checkpoint against context drift
  - After any crash or error, to confirm the log state
  - When you want to review long-term trends across more experiments than you
    can remember from context

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

## Workflow Per Iteration

1. Decide Exploitation or Exploration mode based on recent results
2. Read `players/challenger.py` to see the current implementation
3. Make your change to `players/challenger.py`
4. Git commit: `git add players/challenger.py && git commit -m 'description of change'`
5. Run the benchmark:
   ```
   uv run python benchmark.py --description 'Brief summary of the code change' --hypothesis 'Why you expect this to improve win rate'
   ```
6. Read the JSON output:
   - `"kept": true` → your commit is kept, note the new baseline win rate
   - `"kept": false` → benchmark reverted your commit via `git reset --hard HEAD~1`
   - Error/crash → see Timeout and Crash Handling above
7. Immediately begin the next iteration (go to step 1)

## Reading the Log

When you read `experiments/log.jsonl` (see Context Management for when), assess:

- **Current win rate** — your baseline to beat
- **Trend** — are the last 3-5 kept iterations showing gains, or has it flattened?
- **Consecutive reverts** — how many attempts in a row have failed?
- **Architecture history** — what approaches have been tried and how did they perform?

Use this analysis to pick your mode and to avoid repeating failed ideas.

## Session Startup

When you first begin:

1. Read `experiments/log.jsonl` (if it exists) to understand the full history
2. Read `players/challenger.py` to see the current implementation
3. Assess the current win rate, trend, and what has been tried
4. Begin your first experiment — do not output a lengthy analysis first,
   just note your plan in 1-2 sentences and start coding

If there is no `log.jsonl` yet, the baseline has already been established.
Start with a simple improvement to the default challenger.

