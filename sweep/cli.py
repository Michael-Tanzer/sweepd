#!/usr/bin/env python3
"""
Click CLI for the sweep manager: list, show, run, create, add-runs, mark-rerun, mark-ran, delete, export-runs, daemon, web.
"""
import logging
import os
import click

from sweep import (
    add_ran_lines_by_hashes,
    append_runs,
    clone_sweep,
    get_completed_hashes,
    get_default_command,
    get_runs,
    get_sweep_config,
    list_sweep_ids,
    remove_ran_lines_by_hashes,
    run_hash,
    save_meta,
    save_runs,
    split_param_line,
    sweep_daemon,
    sweep_run,
)
from sweep.core import _meta_path, _ran_path, _runs_path


def _expand_grid(base_line: str, grid_specs: list[str]) -> list[str]:
    """
    Expands grid specs into param lines (cartesian product), each combined with base.
    base_line: optional single param line (e.g. "gpu=0").
    grid_specs: list of "key=val1,val2,..." strings.
    Returns list of param lines.
    """
    base_parts = []
    if base_line and base_line.strip():
        base_parts = [p.strip() for p in split_param_line(base_line) if p.strip()]

    if not grid_specs:
        return [",".join(base_parts)] if base_parts else []

    axes = []
    for spec in grid_specs:
        spec = spec.strip()
        if "=" not in spec:
            continue
        key, _, vals = spec.partition("=")
        key = key.strip()
        values = [v.strip() for v in split_param_line(vals) if v.strip()]
        if values:
            axes.append((key, values))

    if not axes:
        return [",".join(base_parts)] if base_parts else []

    def product(idx, so_far):
        if idx >= len(axes):
            line = ",".join(base_parts + so_far) if base_parts else ",".join(so_far)
            return [line]
        out = []
        key, values = axes[idx]
        for v in values:
            out.extend(product(idx + 1, so_far + [f"{key}={v}"]))
        return out

    return product(0, [])


def _dedup_against_existing(existing_param_lines: list[str], new_param_lines: list[str]) -> tuple[list[str], int]:
    """
    Returns (to_add, skipped_count) where to_add are new_param_lines not already in existing (by hash).
    """
    existing_hashes = {run_hash(p) for p in existing_param_lines}
    to_add = []
    skipped = 0
    seen = set()
    for p in new_param_lines:
        h = run_hash(p)
        if h in existing_hashes or h in seen:
            skipped += 1
            continue
        seen.add(h)
        to_add.append(p)
    return to_add, skipped


@click.group()
def cli():
    """File-based sweep manager for distributed execution."""


@cli.command("list")
def cmd_list():
    """List sweep IDs."""
    ids = list_sweep_ids()
    for sid in ids:
        click.echo(sid)


@cli.command("show")
@click.argument("sweep_id")
def cmd_show(sweep_id):
    """Show sweep meta, run count, and runs with index, hash, status."""
    try:
        command, param_lines = get_sweep_config(sweep_id)
    except FileNotFoundError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1)
    completed = get_completed_hashes(sweep_id)
    click.echo(f"Sweep: {sweep_id}")
    click.echo(f"Command: {' '.join(command)}")
    click.echo(f"Total runs: {len(param_lines)}")
    click.echo(f"Completed: {len(completed)}")
    click.echo("")
    for i, line in enumerate(param_lines):
        h = run_hash(line)
        status = "ran" if h in completed else "pending"
        click.echo(f"  {i}  {h}  {status}  {line}")


@cli.command("run")
@click.argument("sweep_id")
def cmd_run(sweep_id):
    """Claim and execute runs until all done (Run i/N progress)."""
    sweep_run(sweep_id)


@cli.command("create")
@click.argument("sweep_id")
@click.option("--command", "-c", multiple=True, help="Command prefix (e.g. python test_sweep.py). Repeat for multiple args.")
@click.option("--runs", "-r", "runs_list", multiple=True, help="Explicit param line(s).")
@click.option("--base", "-b", default="", help="Base param line for grid (e.g. gpu=0).")
@click.option("--grid", "-g", "grid_specs", multiple=True, help="Grid spec: key=val1,val2,...")
def cmd_create(sweep_id, command, runs_list, base, grid_specs):
    """Create a new sweep. Use --runs and/or --grid (with optional --base). Duplicate runs are skipped."""
    if not command:
        cmd = get_default_command()
        if not cmd:
            click.echo("No default command. Use --command or create ~/.sweeps/sweep_config.toml with 'command = [...]'", err=True)
            raise SystemExit(1)
    else:
        cmd = list(command)
    if runs_list and grid_specs:
        click.echo("Use either --runs or --grid, not both.", err=True)
        raise SystemExit(1)
    if runs_list:
        param_lines = list(runs_list)
    elif grid_specs:
        param_lines = _expand_grid(base, list(grid_specs))
    else:
        click.echo("Provide --runs and/or --grid.", err=True)
        raise SystemExit(1)
    if not param_lines:
        click.echo("No runs generated.", err=True)
        raise SystemExit(1)
    try:
        existing = get_runs(sweep_id)
    except FileNotFoundError:
        existing = []
    to_add, skipped = _dedup_against_existing(existing, param_lines)
    save_meta(sweep_id, cmd)
    save_runs(sweep_id, existing + to_add)
    click.echo(f"Created sweep {sweep_id}: {len(to_add)} runs added, {skipped} duplicates skipped.")


@cli.command("add-runs")
@click.argument("sweep_id")
@click.option("--runs", "-r", "runs_list", multiple=True, help="Explicit param line(s).")
@click.option("--base", "-b", default="", help="Base param line for grid.")
@click.option("--grid", "-g", "grid_specs", multiple=True, help="Grid spec: key=val1,val2,...")
def cmd_add_runs(sweep_id, runs_list, base, grid_specs):
    """Add runs to an existing sweep. Duplicate runs are skipped."""
    if not os.path.exists(_meta_path(sweep_id)):
        click.echo(f"Sweep not found: {sweep_id}", err=True)
        raise SystemExit(1)
    if runs_list and grid_specs:
        click.echo("Use either --runs or --grid, not both.", err=True)
        raise SystemExit(1)
    if runs_list:
        new_lines = list(runs_list)
    elif grid_specs:
        new_lines = _expand_grid(base, list(grid_specs))
    else:
        click.echo("Provide --runs and/or --grid.", err=True)
        raise SystemExit(1)
    existing = get_runs(sweep_id)
    to_add, skipped = _dedup_against_existing(existing, new_lines)
    if not to_add:
        click.echo(f"No new runs to add ({skipped} duplicates).")
        return
    append_runs(sweep_id, to_add)
    click.echo(f"Added {len(to_add)} runs, {skipped} duplicates skipped.")


@cli.command("mark-rerun")
@click.argument("sweep_id")
@click.option("--hash", "hashes", multiple=True, help="Run hash (6-char) to mark for rerun.")
@click.option("--index", "indices", multiple=True, type=int, help="Run index to mark for rerun.")
def cmd_mark_rerun(sweep_id, hashes, indices):
    """Remove run(s) from ran file so they will be re-run."""
    if not os.path.exists(_ran_path(sweep_id)):
        click.echo("No completed runs for this sweep.")
        return
    to_remove = set(hashes)
    if indices:
        try:
            _, param_lines = get_sweep_config(sweep_id)
        except FileNotFoundError:
            param_lines = []
        for i in indices:
            if 0 <= i < len(param_lines):
                to_remove.add(run_hash(param_lines[i]))
    if not to_remove:
        click.echo("Specify --hash and/or --index.")
        raise SystemExit(1)
    remove_ran_lines_by_hashes(sweep_id, to_remove)
    click.echo(f"Marked {len(to_remove)} run(s) for rerun.")


@cli.command("mark-ran")
@click.argument("sweep_id")
@click.option("--hash", "hashes", multiple=True, help="Run hash (6-char) to mark as ran.")
@click.option("--index", "indices", multiple=True, type=int, help="Run index to mark as ran.")
def cmd_mark_ran(sweep_id, hashes, indices):
    """Add run(s) to ran file to mark them as completed."""
    to_add = set(hashes)
    if indices:
        try:
            _, param_lines = get_sweep_config(sweep_id)
        except FileNotFoundError:
            param_lines = []
        for i in indices:
            if 0 <= i < len(param_lines):
                to_add.add(run_hash(param_lines[i]))
    if not to_add:
        click.echo("Specify --hash and/or --index.")
        raise SystemExit(1)
    add_ran_lines_by_hashes(sweep_id, to_add)
    click.echo(f"Marked {len(to_add)} run(s) as ran.")


@cli.command("delete")
@click.argument("sweep_id")
@click.confirmation_option(prompt="Delete this sweep and its ran file?")
def cmd_delete(sweep_id):
    """Remove sweep config and ran file."""
    removed = []
    for path in [_meta_path(sweep_id), _runs_path(sweep_id), _ran_path(sweep_id)]:
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    if removed:
        click.echo(f"Deleted: {', '.join(removed)}")
    else:
        click.echo("Sweep not found or already deleted.")


@cli.command("export-runs")
@click.argument("sweep_id")
def cmd_export_runs(sweep_id):
    """Print run param lines (one per line) for piping."""
    try:
        _, param_lines = get_sweep_config(sweep_id)
    except FileNotFoundError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1)
    for line in param_lines:
        click.echo(line)


@cli.command("clone")
@click.argument("source_id")
@click.argument("new_id")
def cmd_clone(source_id, new_id):
    """Clone a sweep (meta + runs) to a new ID. No ran/review history is copied."""
    try:
        clone_sweep(source_id, new_id)
    except FileNotFoundError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1)
    click.echo(f"Cloned {source_id} -> {new_id}")


@cli.command("daemon")
@click.argument("sweep_id", required=False, default=None)
@click.option("--interval", default=30, show_default=True, help="Poll interval in seconds when idle.")
@click.option("--gpu", "gpu_id", default=0, show_default=True, help="GPU index to check VRAM usage on.")
@click.option("--cpu", "cpu_mode", is_flag=True, default=False, help="Skip GPU check (CPU-only mode).")
@click.option("--vram-threshold", default=1000, show_default=True, help="VRAM usage threshold in MB — above this the GPU is considered busy.")
def cmd_daemon(sweep_id, interval, gpu_id, cpu_mode, vram_threshold):
    """Run pending work and keep polling for new runs.

    If SWEEP_ID is given, watches only that sweep. Otherwise watches all sweeps.
    """
    sweep_ids = [sweep_id] if sweep_id else []
    sweep_daemon(sweep_ids, interval=interval, gpu_id=gpu_id, cpu_mode=cpu_mode, vram_threshold_mb=vram_threshold)


@cli.command("web")
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind to.")
@click.option("--port", default=8765, show_default=True, help="Port to listen on.")
def cmd_web(host, port):
    """Start the web UI server."""
    import uvicorn
    from sweep.web_app import app
    click.echo(f"Starting Sweep Manager web UI at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    cli()


if __name__ == "__main__":
    main()
