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

kenny, discarder, granny, committer, expedition, alex, challenger, ismcts, nashpg

**Note:** `ismcts` (Information Set Monte Carlo Tree Search) is extremely slow.
A 10k-game benchmark can take hours. Only use it if you're prepared to wait.

## Round Robin Results

10,000 games per matchup (ismcts excluded due to speed). Date: 2026-04-03.

| Matchup | Winner | Win Rate |
|---|---|---|
| alex vs challenger | Challenger | 0.5117 +/- 0.0050 |
| alex vs committer | Committer | 0.5066 +/- 0.0050 |
| alex vs discarder | Alex | 0.8192 +/- 0.0038 |
| alex vs expedition | Alex | 0.6405 +/- 0.0048 |
| alex vs granny | Alex | 0.8958 +/- 0.0031 |
| alex vs kenny | Alex | 0.9017 +/- 0.0030 |
| challenger vs committer | Challenger | 0.5098 +/- 0.0050 |
| challenger vs discarder | Challenger | 0.8569 +/- 0.0035 |
| challenger vs expedition | Challenger | 0.6772 +/- 0.0047 |
| challenger vs granny | Challenger | 0.8924 +/- 0.0031 |
| challenger vs kenny | Challenger | 0.9105 +/- 0.0029 |
| committer vs discarder | Committer | 0.8254 +/- 0.0038 |
| committer vs expedition | Committer | 0.6526 +/- 0.0048 |
| committer vs granny | Committer | 0.9149 +/- 0.0028 |
| committer vs kenny | Committer | 0.9131 +/- 0.0028 |
| discarder vs expedition | Expedition | 0.7451 +/- 0.0044 |
| discarder vs granny | Granny | 0.8533 +/- 0.0035 |
| discarder vs kenny | Discarder | 0.5039 +/- 0.0050 |
| expedition vs granny | Expedition | 0.8574 +/- 0.0035 |
| expedition vs kenny | Expedition | 0.8346 +/- 0.0037 |
| granny vs kenny | Granny | 0.8529 +/- 0.0035 |

### Rankings

| Rank | Player | W-L | Notes |
|---|---|---|---|
| 1 | Challenger | 5-1 | Top 3 are tightly clustered (~1% margins) |
| 2 | Committer | 4-2 | Narrow losses to challenger and alex |
| 3 | Alex | 4-2 | Narrow losses to challenger and committer |
| 4 | Expedition | 3-3 | Beats bottom 3 convincingly |
| 5 | Granny | 2-4 | Dominates kenny and discarder |
| 6 | Discarder | 1-5 | Barely beats kenny |
| 7 | Kenny | 0-6 | Last place |

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
