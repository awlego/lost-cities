#!/usr/bin/env python
"""Round-robin tournament: every player pair, 10k games each."""

import itertools
import subprocess
import sys
import re
import concurrent.futures

PLAYERS = ['alex', 'challenger', 'committer', 'discarder', 'expedition', 'granny', 'kenny']
N = 10000


def run_match(pair):
    p1, p2 = pair
    result = subprocess.run(
        [sys.executable, 'wrapper.py', p1, p2, '-n', str(N)],
        capture_output=True, text=True, timeout=600
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    return (p1, p2, stdout, stderr)


if __name__ == '__main__':
    pairs = list(itertools.combinations(PLAYERS, 2))
    print('Running %d matchups, %d games each...\n' % (len(pairs), N))

    results = []
    with concurrent.futures.ProcessPoolExecutor() as pool:
        futures = {pool.submit(run_match, pair): pair for pair in pairs}
        for future in concurrent.futures.as_completed(futures):
            pair = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append((pair[0], pair[1], '', str(e)))

    # Sort by matchup order
    pair_order = {pair: i for i, pair in enumerate(pairs)}
    results.sort(key=lambda r: pair_order.get((r[0], r[1]), 999))

    print('%-35s %s' % ('Matchup', 'Result'))
    print('-' * 70)
    for p1, p2, stdout, stderr in results:
        label = '%s vs %s' % (p1, p2)
        if stdout:
            print('%-35s %s' % (label, stdout))
        else:
            print('%-35s ERROR: %s' % (label, stderr[:100]))
