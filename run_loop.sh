#!/bin/bash
# Launch a single long-lived Claude session for Lost Cities autoresearch.
# The session loops internally — interrupt with Ctrl+C when done.
# Usage: bash run_loop.sh

mkdir -p experiments

# Establish baseline on first run if needed.
if [ ! -f experiments/baseline.json ]; then
    echo "=== Establishing baseline ==="
    uv run python benchmark.py
fi

echo "=== Starting autoresearch session (Ctrl+C to stop) ==="
claude --dangerously-skip-permissions \
    "Read program.md for your instructions and begin the research loop."
