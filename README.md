# lost-cities
Computer players for a two-player card game similar to Battle Line

## Usage

```bash
python wrapper.py <player1> <player2> [options]
```

### Options

| Flag | Description | Default |
|---|---|---|
| `-n N` | Number of rounds to play | 1 |
| `-s` | Same player starts every round (no alternating) | off |
| `-j J` | Number of parallel workers (0 = auto-detect CPU count) | 0 (auto) |

### Examples

```bash
# Single verbose game
python wrapper.py kenny committer

# 10,000 rounds, sequential
python wrapper.py kenny committer -n 10000 -j 1

# 100,000 rounds, parallel across all CPU cores (default)
python wrapper.py kenny committer -n 100000

# 100,000 rounds, 4 workers
python wrapper.py kenny committer -n 100000 -j 4
```

### Available players

kenny, discarder, granny, committer, expedition, challenger, marathon, ismcts, nashpg

**Note:** `ismcts` (Information Set Monte Carlo Tree Search) is extremely slow.
A 10k-game benchmark can take hours. Only use it if you're prepared to wait.

## Round Robin Results

10,000 games per matchup (ismcts excluded due to speed). Date: 2026-04-05.

| Matchup | Winner | Win Rate |
|---|---|---|
| challenger vs committer | Challenger | 0.5383 +/- 0.0050 |
| challenger vs discarder | Challenger | 0.8748 +/- 0.0033 |
| challenger vs expedition | Challenger | 0.6624 +/- 0.0047 |
| challenger vs granny | Challenger | 0.7303 +/- 0.0044 |
| challenger vs kenny | Challenger | 0.8755 +/- 0.0033 |
| challenger vs marathon | Challenger | 0.5356 +/- 0.0050 |
| challenger vs nashpg | Challenger | 0.6803 +/- 0.0047 |
| committer vs discarder | Committer | 0.8179 +/- 0.0039 |
| committer vs expedition | Committer | 0.6473 +/- 0.0048 |
| committer vs granny | Committer | 0.9198 +/- 0.0027 |
| committer vs kenny | Committer | 0.9154 +/- 0.0028 |
| committer vs marathon | Committer | 0.5060 +/- 0.0050 |
| committer vs nashpg | Committer | 0.6207 +/- 0.0049 |
| discarder vs expedition | Expedition | 0.7402 +/- 0.0044 |
| discarder vs granny | Granny | 0.8496 +/- 0.0036 |
| discarder vs kenny | Kenny | 0.5020 +/- 0.0050 |
| discarder vs marathon | Marathon | 0.8539 +/- 0.0035 |
| discarder vs nashpg | Nashpg | 0.8180 +/- 0.0039 |
| expedition vs granny | Expedition | 0.8650 +/- 0.0034 |
| expedition vs kenny | Expedition | 0.8386 +/- 0.0037 |
| expedition vs marathon | Marathon | 0.6386 +/- 0.0048 |
| expedition vs nashpg | Expedition | 0.5065 +/- 0.0050 |
| granny vs kenny | Granny | 0.8553 +/- 0.0035 |
| granny vs marathon | Marathon | 0.8031 +/- 0.0040 |
| granny vs nashpg | Nashpg | 0.6003 +/- 0.0049 |
| kenny vs marathon | Marathon | 0.8864 +/- 0.0032 |
| kenny vs nashpg | Nashpg | 0.8359 +/- 0.0037 |
| marathon vs nashpg | Marathon | 0.6529 +/- 0.0048 |

### Rankings

| Rank | Player | W-L | Notes |
|---|---|---|---|
| 1 | Challenger | 7-0 | Undefeated; narrow wins over committer and marathon |
| 2 | Committer | 5-2 | Loses narrowly to challenger; tied with marathon |
| 3 | Marathon | 5-2 | Chases 8+ card bonus (3x committer rate); tied with committer |
| 4 | NashPG | 4-3 | Neural net player; tight matchup with expedition |
| 5 | Expedition | 3-4 | Beats bottom 3 convincingly |
| 6 | Granny | 2-5 | Dominates kenny and discarder |
| 7 | Kenny | 1-6 | Barely edges discarder |
| 8 | Discarder | 0-7 | Last place |

## NashPG Player

The `nashpg` player runs a trained neural network (NashPG policy gradient)
from the [open_spiel](https://github.com/awlego/open_spiel) repo. It requires
`torch` and a checkpoint directory.

### Setup

The player needs the Python environment that has PyTorch installed. If you're
using the open_spiel venv:

```bash
NASHPG_CHECKPOINT=~/Repositories/open_spiel/checkpoints/v2_512x2_mc0.2 \
    ~/Repositories/open_spiel/env3.12/bin/python wrapper.py nashpg committer -n 1000
```

### Environment variables

- `NASHPG_CHECKPOINT` (required): path to a checkpoint directory containing
  `config.json` and `nash_pg.pt`. Only 517-dim enriched observation checkpoints
  are supported (the `v2_*` and `milestone_v2_*` checkpoints).

### How it works

The adapter constructs a 517-dim observation tensor directly from the game
state (no pyspiel dependency at runtime) and queries the actor network twice
per turn — once for the play/discard decision and once for the draw decision.
This matches OpenSpiel's two-phase action structure.
