#!/usr/bin/env python
"""Benchmark runner with keep/revert gating for autoresearch-style iteration.

Runs challenger vs an opponent, compares against baseline, and keeps or
reverts the challenger file based on whether win rate improved.

Usage:
    uv run python benchmark.py [--opponent committer] [--n 100000]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

EXPERIMENTS_DIR = 'experiments'
BASELINE_FILE = os.path.join(EXPERIMENTS_DIR, 'baseline.json')
LOG_FILE = os.path.join(EXPERIMENTS_DIR, 'log.jsonl')
CHALLENGER_FILE = 'players/challenger.py'
CHALLENGER_BASELINE = os.path.join(EXPERIMENTS_DIR, 'challenger.py.baseline')


def run_benchmark(opponent, n_games):
    """Run wrapper.py and return (win_rate, stderr) for the challenger."""
    try:
        result = subprocess.run(
            [sys.executable, 'wrapper.py', 'challenger', opponent, '-n', str(n_games)],
            capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        return None, None, "Benchmark timed out after 300s"

    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip()
        return None, None, f"Benchmark crashed: {error}"

    output = result.stdout.strip()
    # wrapper.py prints: "Name wins X.XXXX +/- Y.YYYY"
    match = re.search(r'(\w+)\s+wins\s+([\d.]+)\s+\+/-\s+([\d.]+)', output)
    if not match:
        return None, None, f"Could not parse output: {output}"

    winner_name = match.group(1).lower()
    win_rate = float(match.group(2))
    stderr = float(match.group(3))

    # If the opponent is the winner, flip the rate for challenger's perspective.
    if winner_name != 'challenger':
        win_rate = 1.0 - win_rate

    return win_rate, stderr, None


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
    parser.add_argument('--n', type=int, default=100000, help='Number of games')
    args = parser.parse_args()

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    experiment_id = get_next_experiment_id()
    baseline = load_baseline()

    # First run: establish baseline and save the initial challenger file.
    if baseline is None:
        # Save the initial challenger as the baseline copy.
        shutil.copy2(CHALLENGER_FILE, CHALLENGER_BASELINE)

    # Run the benchmark.
    win_rate, stderr, error = run_benchmark(args.opponent, args.n)

    if error:
        # Failed experiment: revert and log.
        if os.path.exists(CHALLENGER_BASELINE):
            shutil.copy2(CHALLENGER_BASELINE, CHALLENGER_FILE)

        record = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'error': error,
            'opponent': args.opponent,
            'n_games': args.n,
            'baseline_win_rate': baseline['win_rate'] if baseline else None,
            'improved': False,
            'kept': False
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
        shutil.copy2(CHALLENGER_FILE, CHALLENGER_BASELINE)
    else:
        # Conservative gating: new lower bound > old upper bound.
        new_lower = win_rate - stderr
        old_upper = baseline['win_rate'] + baseline['stderr']
        improved = new_lower > old_upper
        kept = improved

        if kept:
            save_baseline(win_rate, stderr, experiment_id)
            shutil.copy2(CHALLENGER_FILE, CHALLENGER_BASELINE)
        else:
            shutil.copy2(CHALLENGER_BASELINE, CHALLENGER_FILE)

    record = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'win_rate': win_rate,
        'stderr': stderr,
        'n_games': args.n,
        'opponent': args.opponent,
        'baseline_win_rate': baseline['win_rate'] if baseline else win_rate,
        'improved': improved,
        'kept': kept
    }
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(record) + '\n')

    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
