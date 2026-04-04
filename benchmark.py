#!/usr/bin/env python
"""Benchmark runner with keep/revert gating for autoresearch-style iteration.

Runs challenger vs an opponent, compares against baseline, and keeps or
reverts the challenger file based on whether win rate improved.

Usage:
    uv run python benchmark.py [--opponent committer] [--n 10000]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

EXPERIMENTS_DIR = 'experiments'
BASELINE_FILE = os.path.join(EXPERIMENTS_DIR, 'baseline.json')
LOG_FILE = os.path.join(EXPERIMENTS_DIR, 'log.jsonl')
CHALLENGER_FILE = 'players/challenger.py'


def run_benchmark(opponent, n_games):
    """Run wrapper.py and return (win_rate, stderr, error, elapsed_seconds)."""
    start = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, 'wrapper.py', 'challenger', opponent, '-n', str(n_games)],
            capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        elapsed = round(time.monotonic() - start, 1)
        return None, None, "Benchmark timed out after 300s", elapsed

    elapsed = round(time.monotonic() - start, 1)

    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip()
        return None, None, f"Benchmark crashed: {error}", elapsed

    output = result.stdout.strip()
    # wrapper.py prints: "Name wins X.XXXX +/- Y.YYYY"
    match = re.search(r'(\w+)\s+wins\s+([\d.]+)\s+\+/-\s+([\d.]+)', output)
    if not match:
        return None, None, f"Could not parse output: {output}", elapsed

    winner_name = match.group(1).lower()
    win_rate = float(match.group(2))
    stderr = float(match.group(3))

    # If the opponent is the winner, flip the rate for challenger's perspective.
    if winner_name != 'challenger':
        win_rate = 1.0 - win_rate

    return win_rate, stderr, None, elapsed


def load_baseline():
    """Load the baseline from file, or return None if it doesn't exist."""
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE) as f:
            return json.load(f)
    return None


def save_baseline(win_rate, stderr, experiment_id):
    with open(BASELINE_FILE, 'w') as f:
        json.dump({
            'win_rate': win_rate,
            'stderr': stderr,
            'experiment_id': experiment_id
        }, f, indent=2)


def git_revert_experiment():
    """Revert the last commit, but only if it touched challenger.py."""
    result = subprocess.run(
        ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD'],
        capture_output=True, text=True
    )
    changed_files = result.stdout.strip().split('\n')
    if CHALLENGER_FILE in changed_files:
        subprocess.run(['git', 'reset', '--hard', 'HEAD~1'], check=True)
    else:
        print(f"WARNING: Last commit did not touch {CHALLENGER_FILE}, skipping git reset", file=sys.stderr)


def get_next_experiment_id():
    if not os.path.exists(LOG_FILE):
        return 1
    count = 0
    with open(LOG_FILE) as f:
        for _ in f:
            count += 1
    return count + 1


def main():
    parser = argparse.ArgumentParser(description='Benchmark challenger player.')
    parser.add_argument('--opponent', default='committer', help='Opponent player name')
    parser.add_argument('--n', type=int, default=10000, help='Number of games')
    parser.add_argument('--hypothesis', default=None, help='What you expect this change to do and why')
    parser.add_argument('--description', default=None, help='Brief description of the code change made')
    args = parser.parse_args()

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    experiment_id = get_next_experiment_id()
    baseline = load_baseline()

    # Run the benchmark.
    win_rate, stderr, error, benchmark_seconds = run_benchmark(args.opponent, args.n)

    if error:
        # Failed experiment: revert the commit and log.
        git_revert_experiment()

        record = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'description': args.description,
            'hypothesis': args.hypothesis,
            'error': error,
            'opponent': args.opponent,
            'n_games': args.n,
            'baseline_win_rate': baseline['win_rate'] if baseline else None,
            'improved': False,
            'kept': False,
            'benchmark_seconds': benchmark_seconds
        }
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record, indent=2))
        sys.exit(1)

    # Determine if this is an improvement.
    if baseline is None:
        # First run: this IS the baseline.
        improved = False
        kept = True
        save_baseline(win_rate, stderr, experiment_id)
    else:
        # Conservative gating: new lower bound > old upper bound.
        new_lower = win_rate - stderr
        old_upper = baseline['win_rate'] + baseline['stderr']
        improved = new_lower > old_upper
        kept = improved

        if kept:
            save_baseline(win_rate, stderr, experiment_id)
        else:
            git_revert_experiment()

    record = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'description': args.description,
        'hypothesis': args.hypothesis,
        'win_rate': win_rate,
        'stderr': stderr,
        'n_games': args.n,
        'opponent': args.opponent,
        'baseline_win_rate': baseline['win_rate'] if baseline else win_rate,
        'improved': improved,
        'kept': kept,
        'benchmark_seconds': benchmark_seconds
    }
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(record) + '\n')

    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
