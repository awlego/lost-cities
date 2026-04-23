#!/usr/bin/env python

import sys, argparse, logging, random, math, os
from multiprocessing import Pool
from play import play_one_round
from classes import Player
from players import *


def _all_player_subclasses():
    """Walk the Player subclass tree recursively (direct subclasses only
    ignores nested variants like NashPGv2)."""
    seen = []
    stack = list(Player.__subclasses__())
    while stack:
        cls = stack.pop()
        seen.append(cls)
        stack.extend(cls.__subclasses__())
    return seen


def run_batch(args_tuple):
    """Run a batch of rounds in a worker process."""
    (player_name_1, player_name_2, name_1, name_2,
     batch_size, same_starter, start_index, seed) = args_tuple

    import random
    import importlib
    from play import play_one_round
    from classes import Player

    # Trigger player discovery (equivalent to `from players import *`)
    importlib.import_module('players')

    random.seed(seed)

    available = {cls.get_name(): cls for cls in _all_player_subclasses()}
    players = [available[player_name_1](0), available[player_name_2](1)]
    names_orig = [name_1, name_2]
    names_rev = [name_2, name_1]
    players_rev = [players[1], players[0]]

    wins = {}
    for i in range(batch_size):
        round_index = start_index + i
        # Original code reverses before each round, so round 0 is reversed,
        # round 1 is original order, etc.
        if not same_starter and round_index % 2 == 0:
            cur_players, cur_names = players_rev, names_rev
        else:
            cur_players, cur_names = players, names_orig

        winner = play_one_round(cur_players, cur_names, verbose=False)
        wins[winner] = wins.get(winner, 0) + 1

    return wins


availablePlayers = {}
for playerSubClass in _all_player_subclasses():
    availablePlayers[playerSubClass.get_name()] = playerSubClass

if __name__ == '__main__':
    # Parse command-line args.
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('declaredPlayers', metavar='player', type=str, nargs=2,
        help=', '.join(availablePlayers.keys()))
    parser.add_argument('-n', '--n_rounds', default=1, metavar='n_rounds',
        type=int, help='positive int')
    parser.add_argument('-s', '--same_starter', action='store_true')
    parser.add_argument('-j', '--jobs', default=0, metavar='jobs',
        type=int, help='number of parallel workers (0 = auto-detect CPU count)')

    args = parser.parse_args()

    assert args.n_rounds > 0
    verbose = True
    if args.n_rounds > 1:
        verbose = False

    # Load players.
    players = []
    rawNames = args.declaredPlayers
    for i in range(len(rawNames)):
        assert rawNames[i] in availablePlayers
        players.append(availablePlayers[rawNames[i]](i))
        rawNames[i] = rawNames[i].capitalize()

    # Resolve duplicate names by appending '1', '2', etc. as needed.
    names = []
    counters = {name : 0 for name in rawNames}
    for name in rawNames:
        if rawNames.count(name) > 1:
            counters[name] += 1
            names.append(name + str(counters[name]))
        else:
            names.append(name)

    # Pad names for better verbose display.
    longestName = ''
    for name in names:
        if len(name) > len(longestName):
            longestName = name
    for i in range(len(names)):
        while len(names[i]) < len(longestName):
            names[i] += ' '

    # Resolve job count.
    jobs = args.jobs
    if jobs == 0:
        jobs = os.cpu_count() or 1
    if verbose:
        jobs = 1
    jobs = min(jobs, args.n_rounds)

    # Play rounds.
    if jobs == 1:
        # Original sequential path.
        winners = []
        for i in range(args.n_rounds):
            if verbose:
                print('\nROUND {}:'.format(i))
            if not args.same_starter:
                players.reverse()
                names.reverse()
            winners.append(play_one_round(players, names, verbose))

        if len(winners) > 1:
            nDraws = args.n_rounds - sum([winners.count(n) for n in names])
            for name in names:
                ratio = (winners.count(name) + 0.5 * nDraws) / args.n_rounds
                stdErr = math.sqrt(ratio * (1 - ratio) / args.n_rounds)
                if ratio >= 0.5:
                    print('{0} wins {1:.4f} +/- {2:.4f}'.format(name, ratio, stdErr))
                    break
        elif verbose:
            print('Winner: {}'.format(winners[0]))
    else:
        # Parallel path.
        player_name_1 = args.declaredPlayers[0].lower()
        player_name_2 = args.declaredPlayers[1].lower()

        # Distribute rounds across workers.
        batch_base = args.n_rounds // jobs
        remainder = args.n_rounds % jobs
        seeds = [random.randrange(2**32) for _ in range(jobs)]

        batches = []
        start = 0
        for w in range(jobs):
            size = batch_base + (1 if w < remainder else 0)
            batches.append((player_name_1, player_name_2,
                            names[0], names[1],
                            size, args.same_starter, start, seeds[w]))
            start += size

        with Pool(processes=jobs) as pool:
            results = pool.map(run_batch, batches)

        # Aggregate results.
        total_wins = {}
        for result in results:
            for name, count in result.items():
                total_wins[name] = total_wins.get(name, 0) + count

        nDraws = args.n_rounds - sum(total_wins.get(n, 0) for n in names)
        for name in names:
            ratio = (total_wins.get(name, 0) + 0.5 * nDraws) / args.n_rounds
            stdErr = math.sqrt(ratio * (1 - ratio) / args.n_rounds)
            if ratio >= 0.5:
                print('{0} wins {1:.4f} +/- {2:.4f}'.format(name, ratio, stdErr))
                break
