# Lost Cities

A two-player card game engine with AI player agents.

## Autoresearch Harness

When running as part of the autoresearch loop (`run_loop.sh`), the AI agent
improves `players/challenger.py` iteratively against a benchmark.

### Editable file

- `players/challenger.py` — the ONLY file the agent may edit

### Do NOT modify

These files are the game engine, benchmark harness, and reference players.
Changing them invalidates all experiment results.

- `classes.py`, `play.py`, `wrapper.py`, `utils.py`
- `benchmark.py`, `program.md`, `run_loop.sh`
- `players/kenny.py`, `players/discarder.py`, `players/committer.py`, `players/granny.py`, `players/__init__.py`
- `players/nashpg.py`, `players/nashpg-checkpoints/`

### Dependencies

Managed via `pyproject.toml`. The NashPG player requires `torch` and `numpy` (installed automatically by `uv run`).

### Running benchmarks

```bash
uv run python benchmark.py              # 10k games vs committer
uv run python benchmark.py --n 10000    # quick test with specific number of games
uv run python wrapper.py challenger committer -n 1  # single verbose game
uv run python wrapper.py nashpg committer -n 1000   # nashpg vs committer
```
