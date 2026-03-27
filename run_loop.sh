#!/bin/bash
# Autoresearch-style loop for iteratively improving the Lost Cities challenger.
# Usage: bash run_loop.sh [max_experiments]

MAX_EXPERIMENTS=${1:-50}

mkdir -p experiments

# Establish baseline on first run if needed.
if [ ! -f experiments/baseline.json ]; then
    echo "=== Establishing baseline ==="
    uv run python benchmark.py
fi

for i in $(seq 1 $MAX_EXPERIMENTS); do
    echo ""
    echo "=== Experiment $i / $MAX_EXPERIMENTS ==="
    exp_start=$SECONDS
    claude --print --dangerously-skip-permissions \
        "Read program.md for your instructions. Read experiments/log.jsonl to see past experiment results (if it exists). Read players/challenger.py to see the current code. Make ONE focused change to players/challenger.py to improve its win rate, then git commit your change. Then run the benchmark with --description and --hypothesis flags, e.g.: uv run python benchmark.py --description 'Added card counting for unseen cards' --hypothesis 'Tracking unseen cards will improve draw decisions and increase win rate by ~1%'. The description should briefly state WHAT you changed in the code. The hypothesis should state WHY you expect it to help. The benchmark will automatically revert the commit if it didn't improve. Report whether the change was kept or reverted and why you think it did or didn't work." \
        2>&1 | tee "experiments/experiment_${i}.log"
    exp_elapsed=$((SECONDS - exp_start))
    echo "=== Experiment $i took ${exp_elapsed}s ==="
done

echo ""
echo "=== Done: $MAX_EXPERIMENTS experiments completed ==="
echo "Results in experiments/log.jsonl"
