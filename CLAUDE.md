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

### Running benchmarks

```bash
uv run python benchmark.py              # 100k games vs committer
uv run python benchmark.py --n 10000    # quick test
uv run python wrapper.py challenger committer -n 1  # single verbose game
```
