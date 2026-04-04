# lost-cities
Computer players for a two-player card game similar to Battle Line

## Usage

```bash
python3 wrapper.py <player1> <player2> [options]
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
python3 wrapper.py kenny committer

# 10,000 rounds, sequential
python3 wrapper.py kenny committer -n 10000 -j 1

# 100,000 rounds, parallel across all CPU cores (default)
python3 wrapper.py kenny committer -n 100000

# 100,000 rounds, 4 workers
python3 wrapper.py kenny committer -n 100000 -j 4
```

### Available players

kenny, discarder, granny, committer
