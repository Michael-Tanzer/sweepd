# sweepd

File-based sweep manager for distributed execution across multiple machines/GPUs. Manages hyperparameter sweeps with a simple file-based backend and optional web UI.

## How it works

Sweeps are stored in `~/.sweepd/`. Multiple machines can run `sweep run <id>` simultaneously — each atomically claims a different run via file locking, so no two machines execute the same run. No database or coordinator needed.

## Installation

```bash
# Install from GitHub
uv tool install git+https://github.com/your-username/sweepd

# Or clone and install for development
git clone https://github.com/your-username/sweepd
cd sweepd
uv sync
```

## Quick Start

```bash
# Create a sweep with grid search
sweep create my_sweep \
  -c python -c example_script.py \
  -g "training.lr=0.01,0.001" \
  -g "training.batch_size=8,16"

# Run the sweep (safe to run on multiple machines simultaneously)
sweep run my_sweep

# Or use the web UI
uv run python sweep_web.py
# Then open http://localhost:8765
```

> **Development (without installing):** Replace `sweep` with `uv run python sweep_cli.py` in all commands above.

## CLI Commands

### `sweep list`
List all sweep IDs.

### `sweep show <id>`
Show sweep details: command, total runs, completed runs, and a list of all runs with their index, hash, status (ran/pending), and parameter line.

### `sweep create <id>`
Create a new sweep. Requires either `--runs` or `--grid` to specify runs.

**Flags:**
- `--command` / `-c` (multiple): Command prefix to run for each sweep run. Repeat for multiple arguments (e.g., `-c python -c example_script.py`). If omitted, uses default command from `~/.sweepd/sweep_config.toml` or `config/sweep_defaults.toml`.
- `--runs` / `-r` (multiple): Explicit parameter line(s) to add. Each line is a comma-separated list of `key=value` pairs (e.g., `-r "lr=0.01,bs=8" -r "lr=0.001,bs=16"`).
- `--grid` / `-g` (multiple): Grid specification for generating combinations. Format: `key=val1,val2,val3`. Multiple `--grid` flags create a cartesian product (e.g., `-g "lr=0.01,0.001" -g "bs=8,16"` generates 4 combinations).
- `--base` / `-b`: Base parameter line applied to all grid combinations (e.g., `-b "gpu=0"`). Only used with `--grid`.

**Examples:**
```bash
# Create with explicit runs
sweep create my_sweep -c python -c train.py -r "lr=0.01,bs=8" -r "lr=0.001,bs=16"

# Create with grid search
sweep create my_sweep -c python -c train.py -b "gpu=0" -g "lr=0.01,0.001" -g "bs=8,16"
```

### `sweep add-runs <id>`
Add runs to an existing sweep. Duplicate runs (by hash) are automatically skipped.

**Flags:**
- `--runs` / `-r` (multiple): Explicit parameter line(s) to add.
- `--grid` / `-g` (multiple): Grid specification for generating combinations.
- `--base` / `-b`: Base parameter line applied to all grid combinations.

### `sweep mark-rerun <id>`
Remove run(s) from the completed runs file so they will be executed again. Requires either `--hash` or `--index`.

**Flags:**
- `--hash` (multiple): 6-character run hash to mark for rerun. Can specify multiple hashes.
- `--index` (multiple): Run index (0-based) to mark for rerun. Can specify multiple indices.

**Examples:**
```bash
# Mark by hash
sweep mark-rerun my_sweep --hash a1b2c3

# Mark by index
sweep mark-rerun my_sweep --index 0 --index 5
```

### `sweep mark-ran <id>`
Add run(s) to the completed runs file to mark them as already executed. Requires either `--hash` or `--index`.

**Flags:**
- `--hash` (multiple): 6-character run hash to mark as ran.
- `--index` (multiple): Run index (0-based) to mark as ran.

### `sweep run <id>`
Claim and execute runs from the sweep until all are completed. Shows progress as "Run i/N [hash] param_line". Uses file locking for safe distributed execution across multiple machines.

### `sweep delete <id>`
Delete a sweep and its associated files (config, runs, and ran status). Prompts for confirmation.

### `sweep export-runs <id>`
Print all run parameter lines (one per line) to stdout. Useful for piping or scripting.

### `sweep migrate <id>`
Migrate a legacy sweep from the old `.txt` format to the new layout (`meta.toml` + `runs.txt` + `ran/` directory). Idempotent if already migrated.

## Default Command

If you always use the same command, set it once instead of repeating `-c` on every `sweep create`:

```toml
# ~/.sweepd/sweep_config.toml
command = ["python", "train.py"]
```

Or use `config/sweep_defaults.toml` in your project directory as a fallback.

## File Structure

Sweeps are stored in `~/.sweepd/`:
- `configs/<sweep_id>.meta.toml` - Command to run
- `configs/<sweep_id>.runs.txt` - One param line per run
- `ran/<sweep_id>.txt` - Completed runs (hash + param line)

Each run gets a 6-char hash (order-independent) for easy identification and rerun management.
