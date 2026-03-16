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
    get_completed_exit_codes,
    get_completed_hashes,
    get_default_command,
    get_review_hashes,
    get_run_timings,
    get_running_pid,
    get_runs,
    get_sweep_config,
    kill_run,
    list_sweep_ids,
    remove_ran_lines_by_hashes,
    run_hash,
    save_meta,
    save_runs,
    split_param_line,
    sweep_daemon,
    sweep_run,
)
from sweep.core import _log_path, _meta_path, _ran_path, _runs_path


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
    exit_codes = get_completed_exit_codes(sweep_id)
    review_hashes = get_review_hashes(sweep_id)
    timings = get_run_timings(sweep_id)

    # Compute summary counts
    n_completed = 0
    n_failed = 0
    n_running = 0
    n_review = 0
    for line in param_lines:
        h = run_hash(line)
        if h in completed:
            ec = exit_codes.get(h)
            if ec is not None and ec != 0:
                n_failed += 1
            else:
                n_completed += 1
        elif h in review_hashes:
            n_review += 1
        else:
            timing = timings.get(h, {})
            if timing.get("start") is not None and timing.get("end") is None:
                n_running += 1

    click.echo(f"Sweep: {sweep_id}")
    click.echo(f"Command: {' '.join(command)}")
    click.echo(f"Total runs: {len(param_lines)}")
    parts = [f"{n_completed} completed"]
    if n_failed:
        parts.append(f"{n_failed} failed")
    if n_running:
        parts.append(f"{n_running} running")
    if n_review:
        parts.append(f"{n_review} review")
    click.echo(f"Status: {', '.join(parts)}")
    click.echo("")
    for i, line in enumerate(param_lines):
        h = run_hash(line)
        if h in completed:
            ec = exit_codes.get(h)
            if ec is not None and ec != 0:
                status = f"failed (exit {ec})"
            elif ec is not None:
                status = f"completed (exit {ec})"
            else:
                status = "completed"
        elif h in review_hashes:
            status = "review"
        else:
            timing = timings.get(h, {})
            if timing.get("start") is not None and timing.get("end") is None:
                status = "running"
            else:
                status = "pending"
        click.echo(f"  {i}  {h}  {status}  {line}")


@cli.command("info")
@click.argument("run_hashes", nargs=-1, required=True)
@click.option("--sweep-id", default=None, help="Sweep ID to search in. If omitted, searches all sweeps.")
def cmd_info(run_hashes, sweep_id):
    """Show detailed info for one or more runs by hash."""
    from datetime import datetime

    def _find_run(sid, target_hash):
        """Returns (param_line, index) if hash found in sweep, else None."""
        try:
            _, param_lines = get_sweep_config(sid)
        except FileNotFoundError:
            return None
        for i, line in enumerate(param_lines):
            if run_hash(line) == target_hash:
                return line, i
        return None

    any_missing = False
    for idx, run_hash_str in enumerate(run_hashes):
        if idx > 0:
            click.echo("")

        # Find the run
        if sweep_id:
            result = _find_run(sweep_id, run_hash_str)
            if not result:
                click.echo(f"Run {run_hash_str} not found in sweep {sweep_id}.", err=True)
                any_missing = True
                continue
            found_sweep = sweep_id
        else:
            found_sweep = None
            result = None
            for sid in list_sweep_ids():
                result = _find_run(sid, run_hash_str)
                if result:
                    found_sweep = sid
                    break
            if not result:
                click.echo(f"Run {run_hash_str} not found in any sweep.", err=True)
                any_missing = True
                continue

        param_line, index = result

        # Gather status info
        completed = get_completed_hashes(found_sweep)
        review = get_review_hashes(found_sweep)
        exit_codes = get_completed_exit_codes(found_sweep)
        timings = get_run_timings(found_sweep)

        h = run_hash_str
        if h in completed:
            ec = exit_codes.get(h)
            status = "failed" if ec and ec != 0 else "ran"
        elif h in review:
            status = "review"
        else:
            status = "pending"

        timing = timings.get(h, {})
        has_start = timing.get("start") is not None
        has_end = timing.get("end") is not None

        # If pending but has a start time with no end, it's potentially running
        if status == "pending" and has_start and not has_end:
            status_display = "running"
        else:
            status_display = status

        click.echo(f"Sweep:      {found_sweep}")
        click.echo(f"Hash:       {h}")
        click.echo(f"Status:     {status_display}")
        click.echo(f"Params:     {param_line}")

        if has_start:
            start_str = datetime.fromtimestamp(timing["start"]).strftime("%Y-%m-%d %H:%M:%S")
            click.echo(f"Start:      {start_str}")

            if has_end:
                end_str = datetime.fromtimestamp(timing["end"]).strftime("%Y-%m-%d %H:%M:%S")
                click.echo(f"End:        {end_str}")
            else:
                click.echo(f"End:        —")

            duration = timing.get("duration")
            if duration is not None:
                mins, secs = divmod(int(duration), 60)
                hours, mins = divmod(mins, 60)
                if hours:
                    dur_str = f"{hours}h {mins}m {secs}s"
                elif mins:
                    dur_str = f"{mins}m {secs}s"
                else:
                    dur_str = f"{secs}s"
                click.echo(f"Duration:   {dur_str}")
            else:
                click.echo(f"Duration:   —")

            ec = exit_codes.get(h)
            if ec is not None:
                click.echo(f"Exit code:  {ec}")
            else:
                click.echo(f"Exit code:  —")

    if any_missing:
        raise SystemExit(1)


@cli.command("kill")
@click.argument("run_hash_str")
@click.option("--sweep-id", default=None, help="Sweep ID to search in. If omitted, searches all sweeps.")
@click.option("--signal", "sig", default="TERM", help="Signal to send: TERM (default) or KILL.")
def cmd_kill(run_hash_str, sweep_id, sig):
    """Kill a running experiment by its hash."""
    import signal as signal_mod
    sig_map = {"TERM": signal_mod.SIGTERM, "KILL": signal_mod.SIGKILL}
    sig_num = sig_map.get(sig.upper(), signal_mod.SIGTERM)

    sweep_ids_to_check = [sweep_id] if sweep_id else list_sweep_ids()
    for sid in sweep_ids_to_check:
        pid = get_running_pid(sid, run_hash_str)
        if pid is not None:
            success = kill_run(sid, run_hash_str, sig_num)
            if success:
                click.echo(f"Sent {sig.upper()} to PID {pid} (hash {run_hash_str}, sweep {sid}).")
            else:
                click.echo(f"Process PID {pid} not found (may have already exited).", err=True)
                raise SystemExit(1)
            return

    click.echo(f"No running process found for hash {run_hash_str}.", err=True)
    raise SystemExit(1)


@cli.command("logs")
@click.argument("run_hash_str")
@click.option("--sweep-id", default=None, help="Sweep ID to search in. If omitted, searches all sweeps.")
@click.option("-n", "--lines", "num_lines", default=50, show_default=True, help="Number of lines to show.")
@click.option("-f", "--follow", is_flag=True, default=False, help="Follow log output (like tail -f).")
def cmd_logs(run_hash_str, sweep_id, num_lines, follow):
    """Show last N lines of a run's log file."""
    import subprocess as sp

    sweep_ids_to_check = [sweep_id] if sweep_id else list_sweep_ids()
    for sid in sweep_ids_to_check:
        try:
            _, param_lines = get_sweep_config(sid)
        except FileNotFoundError:
            continue
        for line in param_lines:
            if run_hash(line) == run_hash_str:
                log_file = _log_path(sid, run_hash_str)
                if not os.path.exists(log_file):
                    click.echo(f"No log file found at {log_file}", err=True)
                    raise SystemExit(1)
                if follow:
                    try:
                        sp.run(["tail", "-f", log_file])
                    except KeyboardInterrupt:
                        pass
                else:
                    with open(log_file, "r") as f:
                        all_lines = f.readlines()
                    for l in all_lines[-num_lines:]:
                        click.echo(l, nl=False)
                return

    click.echo(f"Run {run_hash_str} not found.", err=True)
    raise SystemExit(1)


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
