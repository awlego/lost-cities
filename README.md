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

kenny, discarder, granny, committer, expedition, challenger, marathon, ismcts,
mccfr, nashpg (alias for nashpg_v5), nashpg_v2, nashpg_v5

**Note:** `ismcts` (Information Set Monte Carlo Tree Search) is extremely slow.
A 10k-game benchmark can take hours. Only use it if you're prepared to wait.

## Round Robin Results

10,000 games per matchup (ismcts excluded due to speed). Date: 2026-04-22.

### Rankings (avg win rate across 80k games per bot)

| Rank | Player | Avg WR | W-L | Notes |
|---|---|---|---|---|
| 1 | **Nashpg_v5** | **0.797** | 8-0 | Neural net, 512×2 + LayerNorm, trained to 1M updates; undefeated |
| 2 | Challenger | 0.672 | 7-1 | Only loss to nashpg_v5; narrow wins over committer and marathon |
| 3 | Committer | 0.656 | 5-3 | Loses to challenger, marathon, and nashpg_v5 |
| 4 | Marathon | 0.648 | 6-2 | Chases 8+ card bonus; loses to challenger and nashpg_v5 |
| 5 | Expedition | 0.522 | 3-5 | Beats bottom 3, but loses to nashpg_v2 |
| 6 | Nashpg_v2 | 0.493 | 4-4 | Old 1024×2 checkpoint; crushed by v5 at 85-15 head-to-head |
| 7 | Granny | 0.347 | 2-6 | Only beats kenny and discarder |
| 8 | Discarder | 0.199 | 0-8 | Last place; loses every matchup (but vs kenny is 50.0/50.0) |
| 9 | Kenny | 0.165 | 1-7 | Only "beats" discarder (0.5002 — coin flip) |

### All matchups

| Matchup | Winner | Win Rate |
|---|---|---|
| challenger vs committer | Challenger | 0.5413 +/- 0.0050 |
| challenger vs discarder | Challenger | 0.8800 +/- 0.0032 |
| challenger vs expedition | Challenger | 0.6748 +/- 0.0047 |
| challenger vs granny | Challenger | 0.7524 +/- 0.0043 |
| challenger vs kenny | Challenger | 0.8784 +/- 0.0033 |
| challenger vs marathon | Challenger | 0.5435 +/- 0.0050 |
| challenger vs nashpg_v2 | Challenger | 0.6864 +/- 0.0046 |
| challenger vs nashpg_v5 | Nashpg_v5 | 0.5789 +/- 0.0049 |
| committer vs discarder | Committer | 0.8189 +/- 0.0039 |
| committer vs expedition | Committer | 0.6423 +/- 0.0048 |
| committer vs granny | Committer | 0.9175 +/- 0.0028 |
| committer vs kenny | Committer | 0.9112 +/- 0.0028 |
| committer vs marathon | Marathon | 0.5003 +/- 0.0050 |
| committer vs nashpg_v2 | Committer | 0.6315 +/- 0.0048 |
| committer vs nashpg_v5 | Nashpg_v5 | 0.6323 +/- 0.0048 |
| discarder vs expedition | Expedition | 0.7392 +/- 0.0044 |
| discarder vs granny | Granny | 0.8551 +/- 0.0035 |
| discarder vs kenny | Kenny | 0.5002 +/- 0.0050 |
| discarder vs marathon | Marathon | 0.8478 +/- 0.0036 |
| discarder vs nashpg_v2 | Nashpg_v2 | 0.8230 +/- 0.0038 |
| discarder vs nashpg_v5 | Nashpg_v5 | 0.9459 +/- 0.0023 |
| expedition vs granny | Expedition | 0.8588 +/- 0.0035 |
| expedition vs kenny | Expedition | 0.8363 +/- 0.0037 |
| expedition vs marathon | Marathon | 0.6432 +/- 0.0048 |
| expedition vs nashpg_v2 | Nashpg_v2 | 0.5138 +/- 0.0050 |
| expedition vs nashpg_v5 | Nashpg_v5 | 0.7837 +/- 0.0041 |
| granny vs kenny | Granny | 0.8597 +/- 0.0035 |
| granny vs marathon | Marathon | 0.8070 +/- 0.0039 |
| granny vs nashpg_v2 | Nashpg_v2 | 0.6127 +/- 0.0049 |
| granny vs nashpg_v5 | Nashpg_v5 | 0.9909 +/- 0.0009 |
| kenny vs marathon | Marathon | 0.8814 +/- 0.0032 |
| kenny vs nashpg_v2 | Nashpg_v2 | 0.8299 +/- 0.0038 |
| kenny vs nashpg_v5 | Nashpg_v5 | 0.9802 +/- 0.0014 |
| marathon vs nashpg_v2 | Marathon | 0.6621 +/- 0.0047 |
| marathon vs nashpg_v5 | Nashpg_v5 | 0.6141 +/- 0.0049 |
| nashpg_v2 vs nashpg_v5 | Nashpg_v5 | 0.8520 +/- 0.0036 |

## NashPG Player

The `nashpg_v5` and `nashpg_v2` players run trained neural networks (NashPG
policy gradient) from the
[open_spiel](https://github.com/awlego/open_spiel) repo. They require `torch`.
Two bundled checkpoints are available:

| Name | Architecture | Training | vs committer (10k) |
|---|---|---|---|
| `nashpg_v5` | 512×2 + LayerNorm | 1M updates (Apr 2026) | 0.646 |
| `nashpg_v2` | 1024×2, no LayerNorm | ~100k updates (Apr 3) | 0.369 |

`nashpg` is an alias for `nashpg_v5`. Set `NASHPG_CHECKPOINT=/path/to/ckpt`
to point `nashpg` at an external checkpoint instead of the bundled default.

### Setup

Both checkpoints are bundled in `players/nashpg-checkpoints/`, so no config
is needed. Just use a venv with `torch` installed:

```bash
python wrapper.py nashpg_v5 committer -n 10000
python wrapper.py nashpg_v5 nashpg_v2 -n 10000      # compare models
NASHPG_CHECKPOINT=/other/ckpt python wrapper.py nashpg committer -n 1000
```

### Environment variables

- `NASHPG_CHECKPOINT` (optional): path to a checkpoint directory containing
  `config.json` and `nash_pg.pt`. Only affects the `nashpg` alias, not
  `nashpg_v2` or `nashpg_v5` which are pinned to their bundled checkpoints.
  Only 517-dim enriched-observation checkpoints are supported.

### How it works

The adapter constructs a 517-dim observation tensor directly from the game
state (no pyspiel dependency at runtime) and queries the actor network twice
per turn — once for the play/discard decision and once for the draw decision.
This matches OpenSpiel's two-phase action structure. Architecture (hidden
sizes, LayerNorm) is read from each checkpoint's `config.json` at load time,
so the same code handles both v2 (1024×2, plain MLP) and v5 (512×2 with
LayerNorm).
