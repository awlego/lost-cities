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

kenny, discarder, granny, committer, nashpg

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
