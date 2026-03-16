#!/usr/bin/env python3
"""
Tests for sweep CLI (Click): list, show, create, add-runs, mark-rerun, mark-ran, delete, export-runs.
Uses temporary directory for sweep data and Click's CliRunner.
"""
import os
import pytest
from click.testing import CliRunner

from sweep.cli import cli


@pytest.fixture
def default_command(sweep_dir):
    """Create config/sweep_defaults.toml in cwd so create can use default command."""
    config_dir = sweep_dir / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "sweep_defaults.toml").write_text('command = ["python", "test_sweep.py"]\n')
    return sweep_dir


def test_cli_list_empty(sweep_dir):
    """sweep list with no sweeps prints nothing (exit 0)."""
    runner = CliRunner()
    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    assert result.output.strip() == ""


def test_cli_list_shows_sweeps(sweep_dir):
    """sweep list prints one sweep ID per line."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("abc", ["python", "x.py"])
    mod.save_runs("abc", [])
    runner = CliRunner()
    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    assert "abc" in result.output


def test_cli_show_not_found(sweep_dir):
    """sweep show <missing> exits 1 and prints error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "missing"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "Sweep" in result.output


def test_cli_show_success(sweep_dir):
    """sweep show prints meta, counts, and run lines."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("s1", ["uv", "run", "python", "m.py"])
    mod.save_runs("s1", ["a=1", "a=2"])
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "s1"])
    assert result.exit_code == 0
    assert "Sweep: s1" in result.output
    assert "uv run python m.py" in result.output
    assert "Total runs: 2" in result.output
    assert "a=1" in result.output and "a=2" in result.output


def test_cli_create_with_runs(sweep_dir, default_command, monkeypatch):
    """sweep create with --runs creates meta and runs; reports added count."""
    monkeypatch.chdir(sweep_dir)
    runner = CliRunner()
    result = runner.invoke(cli, ["create", "c1", "-r", "x=1", "-r", "x=2"])
    assert result.exit_code == 0
    assert "2 runs added" in result.output
    import sweep as mod
    command, lines = mod.get_sweep_config("c1")
    assert command == ["python", "test_sweep.py"]
    assert lines == ["x=1", "x=2"]


def test_cli_create_with_command_and_grid(sweep_dir, default_command, monkeypatch):
    """sweep create with -c and -g creates sweep with grid-expanded runs."""
    monkeypatch.chdir(sweep_dir)
    runner = CliRunner()
    result = runner.invoke(cli, [
        "create", "g1",
        "-c", "python", "-c", "train.py",
        "-g", "lr=0.01,0.001", "-g", "bs=8,16"
    ])
    assert result.exit_code == 0
    import sweep as mod
    _, lines = mod.get_sweep_config("g1")
    assert len(lines) == 4
    assert "lr=0.01" in lines[0] and "bs=8" in lines[0]


def test_cli_create_no_runs_no_grid_fails(sweep_dir, default_command, monkeypatch):
    """sweep create without --runs or --grid exits 1."""
    monkeypatch.chdir(sweep_dir)
    runner = CliRunner()
    result = runner.invoke(cli, ["create", "nope"])
    assert result.exit_code == 1
    assert "Provide" in result.output or "runs" in result.output.lower()


def test_cli_add_runs(sweep_dir, default_command, monkeypatch):
    """sweep add-runs appends new runs and reports added/skipped."""
    monkeypatch.chdir(sweep_dir)
    runner = CliRunner()
    runner.invoke(cli, ["create", "a1", "-r", "k=1"])
    result = runner.invoke(cli, ["add-runs", "a1", "-r", "k=2", "-r", "k=1"])
    assert result.exit_code == 0
    assert "1" in result.output and "1" in result.output
    import sweep as mod
    lines = mod.get_runs("a1")
    assert "k=1" in lines and "k=2" in lines
    assert lines.count("k=1") == 1


def test_cli_add_runs_sweep_not_found(sweep_dir):
    """sweep add-runs for missing sweep exits 1."""
    runner = CliRunner()
    result = runner.invoke(cli, ["add-runs", "missing", "-r", "a=1"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_cli_mark_rerun(sweep_dir, default_command, monkeypatch):
    """sweep mark-rerun removes hashes from ran file."""
    monkeypatch.chdir(sweep_dir)
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("mr", ["python", "x.py"])
    mod.save_runs("mr", ["a=1", "a=2"])
    (sweep_dir / "ran" / "mr.txt").write_text(
        mod.run_hash("a=1") + "\ta=1\n" + mod.run_hash("a=2") + "\ta=2\n"
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["mark-rerun", "mr", "--hash", mod.run_hash("a=1")])
    assert result.exit_code == 0
    remaining = mod.get_completed_hashes("mr")
    assert mod.run_hash("a=1") not in remaining
    assert mod.run_hash("a=2") in remaining


def test_cli_mark_rerun_by_index(sweep_dir, default_command, monkeypatch):
    """sweep mark-rerun --index removes corresponding run from ran file."""
    monkeypatch.chdir(sweep_dir)
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("mi", ["python", "x.py"])
    mod.save_runs("mi", ["p=1", "p=2"])
    (sweep_dir / "ran" / "mi.txt").write_text(
        mod.run_hash("p=1") + "\tp=1\n" + mod.run_hash("p=2") + "\tp=2\n"
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["mark-rerun", "mi", "--index", "0"])
    assert result.exit_code == 0
    remaining = mod.get_completed_hashes("mi")
    assert mod.run_hash("p=1") not in remaining


def test_cli_mark_ran(sweep_dir, default_command, monkeypatch):
    """sweep mark-ran adds hashes to ran file."""
    monkeypatch.chdir(sweep_dir)
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("mran", ["python", "x.py"])
    mod.save_runs("mran", ["x=1", "x=2"])
    h1 = mod.run_hash("x=1")
    h2 = mod.run_hash("x=2")
    (sweep_dir / "ran" / "mran.txt").write_text(h1 + "\tx=1\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["mark-ran", "mran", "--hash", h2])
    assert result.exit_code == 0
    completed = mod.get_completed_hashes("mran")
    assert h1 in completed
    assert h2 in completed


def test_cli_mark_ran_by_index(sweep_dir, default_command, monkeypatch):
    """sweep mark-ran --index adds corresponding run to ran file."""
    monkeypatch.chdir(sweep_dir)
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("mrani", ["python", "x.py"])
    mod.save_runs("mrani", ["y=1", "y=2"])
    h1 = mod.run_hash("y=1")
    h2 = mod.run_hash("y=2")
    runner = CliRunner()
    result = runner.invoke(cli, ["mark-ran", "mrani", "--index", "0", "--index", "1"])
    assert result.exit_code == 0
    completed = mod.get_completed_hashes("mrani")
    assert h1 in completed
    assert h2 in completed


def test_cli_delete(sweep_dir, default_command, monkeypatch):
    """sweep delete removes meta, runs, and ran files (with --yes)."""
    monkeypatch.chdir(sweep_dir)
    runner = CliRunner()
    runner.invoke(cli, ["create", "d1", "-r", "x=1"])
    result = runner.invoke(cli, ["delete", "d1"], input="y\n")
    assert result.exit_code == 0
    assert not (sweep_dir / "configs" / "d1.meta.toml").exists()
    assert not (sweep_dir / "configs" / "d1.runs.txt").exists()


def test_cli_export_runs(sweep_dir, default_command, monkeypatch):
    """sweep export-runs prints one param line per line."""
    monkeypatch.chdir(sweep_dir)
    runner = CliRunner()
    runner.invoke(cli, ["create", "e1", "-r", "a=1", "-r", "a=2"])
    result = runner.invoke(cli, ["export-runs", "e1"])
    assert result.exit_code == 0
    lines = result.output.strip().split("\n")
    assert lines == ["a=1", "a=2"]


def test_cli_export_runs_not_found(sweep_dir):
    """sweep export-runs for missing sweep exits 1."""
    runner = CliRunner()
    result = runner.invoke(cli, ["export-runs", "nope"])
    assert result.exit_code == 1


def test_cli_clone(sweep_dir, default_command, monkeypatch):
    """sweep clone copies meta+runs without ran history."""
    monkeypatch.chdir(sweep_dir)
    runner = CliRunner()
    runner.invoke(cli, ["create", "orig", "-r", "a=1", "-r", "a=2"])
    result = runner.invoke(cli, ["clone", "orig", "copy1"])
    assert result.exit_code == 0
    assert "Cloned" in result.output
    import sweep as mod
    cmd, lines = mod.get_sweep_config("copy1")
    assert lines == ["a=1", "a=2"]


def test_cli_info_not_found(sweep_dir):
    """sweep info with unknown hash exits 1."""
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "zzzzzz"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_cli_info_pending_run(sweep_dir):
    """sweep info shows pending status for a run with no timing data."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("inf1", ["python", "x.py"])
    mod.save_runs("inf1", ["a=1"])
    h = mod.run_hash("a=1")
    runner = CliRunner()
    result = runner.invoke(cli, ["info", h])
    assert result.exit_code == 0
    assert "pending" in result.output.lower()
    assert "Sweep:      inf1" in result.output
    assert "a=1" in result.output


def test_cli_info_completed_run(sweep_dir):
    """sweep info shows status, times, and exit code for a completed run."""
    import json
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    os.makedirs(sweep_dir / "timing", exist_ok=True)
    mod.save_meta("inf2", ["python", "x.py"])
    mod.save_runs("inf2", ["a=1"])
    h = mod.run_hash("a=1")
    (sweep_dir / "ran" / "inf2.txt").write_text(f"{h}\ta=1\t0\n")
    (sweep_dir / "timing" / "inf2.jsonl").write_text(
        json.dumps({"hash": h, "event": "start", "time": 1710000000.0}) + "\n"
        + json.dumps({"hash": h, "event": "end", "time": 1710000067.0, "exit_code": 0}) + "\n"
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["info", h])
    assert result.exit_code == 0
    assert "ran" in result.output
    assert "Start:" in result.output
    assert "End:" in result.output
    assert "1m 7s" in result.output
    assert "Exit code:  0" in result.output


def test_cli_info_failed_run(sweep_dir):
    """sweep info shows failed status for non-zero exit code."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("inf3", ["python", "x.py"])
    mod.save_runs("inf3", ["a=1"])
    h = mod.run_hash("a=1")
    (sweep_dir / "ran" / "inf3.txt").write_text(f"{h}\ta=1\t1\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["info", h])
    assert result.exit_code == 0
    assert "failed" in result.output.lower()


def test_cli_info_potentially_running(sweep_dir):
    """sweep info shows 'running' for a run with start but no end."""
    import json
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "timing", exist_ok=True)
    mod.save_meta("inf4", ["python", "x.py"])
    mod.save_runs("inf4", ["a=1"])
    h = mod.run_hash("a=1")
    (sweep_dir / "timing" / "inf4.jsonl").write_text(
        json.dumps({"hash": h, "event": "start", "time": 1710000000.0}) + "\n"
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["info", h])
    assert result.exit_code == 0
    assert "running" in result.output.lower()
    assert "—" in result.output  # em dash for missing end/duration


def test_cli_info_with_sweep_id(sweep_dir):
    """sweep info --sweep-id restricts search to specified sweep."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("s1", ["python", "x.py"])
    mod.save_runs("s1", ["a=1"])
    mod.save_meta("s2", ["python", "x.py"])
    mod.save_runs("s2", ["b=1"])
    h = mod.run_hash("a=1")
    runner = CliRunner()
    # Found in s1
    result = runner.invoke(cli, ["info", h, "--sweep-id", "s1"])
    assert result.exit_code == 0
    assert "Sweep:      s1" in result.output
    # Not found in s2
    result = runner.invoke(cli, ["info", h, "--sweep-id", "s2"])
    assert result.exit_code == 1


def test_cli_info_review_status(sweep_dir):
    """sweep info shows review status for runs in review."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "review", exist_ok=True)
    mod.save_meta("inf5", ["python", "x.py"])
    mod.save_runs("inf5", ["a=1"])
    h = mod.run_hash("a=1")
    mod.add_review_lines_by_hashes("inf5", [h])
    runner = CliRunner()
    result = runner.invoke(cli, ["info", h])
    assert result.exit_code == 0
    assert "review" in result.output.lower()


def test_cli_clone_not_found(sweep_dir):
    """sweep clone for missing source exits 1."""
    runner = CliRunner()
    result = runner.invoke(cli, ["clone", "missing", "dst"])
    assert result.exit_code == 1


def test_cli_show_running_status(sweep_dir):
    """sweep show displays 'running' for a run with start but no end in timing."""
    import json
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "timing", exist_ok=True)
    mod.save_meta("sr1", ["python", "x.py"])
    mod.save_runs("sr1", ["a=1", "a=2"])
    h = mod.run_hash("a=1")
    (sweep_dir / "timing" / "sr1.jsonl").write_text(
        json.dumps({"hash": h, "event": "start", "time": 1710000000.0}) + "\n"
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "sr1"])
    assert result.exit_code == 0
    assert "running" in result.output
    assert "pending" in result.output  # a=2 should be pending
    assert "1 running" in result.output


def test_cli_show_completed_with_exit_code(sweep_dir):
    """sweep show displays 'completed (exit 0)' for runs with exit code 0."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("sc1", ["python", "x.py"])
    mod.save_runs("sc1", ["a=1"])
    h = mod.run_hash("a=1")
    (sweep_dir / "ran" / "sc1.txt").write_text(f"{h}\ta=1\t0\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "sc1"])
    assert result.exit_code == 0
    assert "completed (exit 0)" in result.output


def test_cli_show_failed_with_exit_code(sweep_dir):
    """sweep show displays 'failed (exit 1)' for runs with non-zero exit code."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("sf1", ["python", "x.py"])
    mod.save_runs("sf1", ["a=1"])
    h = mod.run_hash("a=1")
    (sweep_dir / "ran" / "sf1.txt").write_text(f"{h}\ta=1\t1\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "sf1"])
    assert result.exit_code == 0
    assert "failed (exit 1)" in result.output
    assert "1 failed" in result.output


def test_cli_show_review_status(sweep_dir):
    """sweep show displays 'review' for runs in the review file."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "review", exist_ok=True)
    mod.save_meta("svr", ["python", "x.py"])
    mod.save_runs("svr", ["a=1"])
    h = mod.run_hash("a=1")
    mod.add_review_lines_by_hashes("svr", [h])
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "svr"])
    assert result.exit_code == 0
    assert "review" in result.output


# --- Feature 3: info accepts multiple hashes ---

def test_cli_info_multiple_hashes(sweep_dir):
    """sweep info with two valid hashes shows info for both."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("mi1", ["python", "x.py"])
    mod.save_runs("mi1", ["a=1", "a=2"])
    h1 = mod.run_hash("a=1")
    h2 = mod.run_hash("a=2")
    runner = CliRunner()
    result = runner.invoke(cli, ["info", h1, h2])
    assert result.exit_code == 0
    assert h1 in result.output
    assert h2 in result.output
    assert "a=1" in result.output
    assert "a=2" in result.output


def test_cli_info_partial_missing(sweep_dir):
    """sweep info with one valid and one invalid hash shows valid, exits 1."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("mi2", ["python", "x.py"])
    mod.save_runs("mi2", ["a=1"])
    h = mod.run_hash("a=1")
    runner = CliRunner()
    result = runner.invoke(cli, ["info", h, "zzzzzz"])
    assert result.exit_code == 1
    assert h in result.output
    assert "a=1" in result.output
    assert "not found" in result.output.lower()


# --- Feature 2: kill command ---

def test_cli_kill_not_running(sweep_dir):
    """sweep kill for a hash with no running process exits 1."""
    runner = CliRunner()
    result = runner.invoke(cli, ["kill", "zzzzzz"])
    assert result.exit_code == 1
    assert "no running process" in result.output.lower()


def test_cli_kill_sends_signal(sweep_dir, monkeypatch):
    """sweep kill sends signal to running process."""
    import json
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "timing", exist_ok=True)
    mod.save_meta("k1", ["python", "x.py"])
    mod.save_runs("k1", ["a=1"])
    h = mod.run_hash("a=1")
    (sweep_dir / "timing" / "k1.jsonl").write_text(
        json.dumps({"hash": h, "event": "start", "time": 1710000000.0, "pid": 12345}) + "\n"
    )
    import sweep.core as core
    killed = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    runner = CliRunner()
    result = runner.invoke(cli, ["kill", h])
    assert result.exit_code == 0
    assert "Sent TERM to PID 12345" in result.output
    assert len(killed) == 1
    assert killed[0][0] == 12345


# --- Feature 5: logs command ---

def test_cli_logs_shows_output(sweep_dir):
    """sweep logs displays last N lines of log file."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("lg1", ["python", "x.py"])
    mod.save_runs("lg1", ["a=1"])
    h = mod.run_hash("a=1")
    log_dir = sweep_dir / "logs" / "lg1"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{h}.log").write_text("line1\nline2\nline3\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["logs", h])
    assert result.exit_code == 0
    assert "line1" in result.output
    assert "line3" in result.output


def test_cli_logs_lines_option(sweep_dir):
    """sweep logs -n limits output to last N lines."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("lg2", ["python", "x.py"])
    mod.save_runs("lg2", ["a=1"])
    h = mod.run_hash("a=1")
    log_dir = sweep_dir / "logs" / "lg2"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{h}.log").write_text("line1\nline2\nline3\nline4\nline5\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["logs", h, "-n", "2"])
    assert result.exit_code == 0
    assert "line4" in result.output
    assert "line5" in result.output
    assert "line1" not in result.output


def test_cli_logs_not_found(sweep_dir):
    """sweep logs for a hash with no log file exits 1."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("lg3", ["python", "x.py"])
    mod.save_runs("lg3", ["a=1"])
    h = mod.run_hash("a=1")
    runner = CliRunner()
    result = runner.invoke(cli, ["logs", h])
    assert result.exit_code == 1
    assert "no log file" in result.output.lower()


def test_cli_logs_hash_not_found(sweep_dir):
    """sweep logs for unknown hash exits 1."""
    runner = CliRunner()
    result = runner.invoke(cli, ["logs", "zzzzzz"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


# --- Additional coverage for edge cases ---

def test_cli_show_completed_no_exit_code(sweep_dir):
    """sweep show displays 'completed' (no exit code) for old-format ran entries."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("snc", ["python", "x.py"])
    mod.save_runs("snc", ["a=1"])
    h = mod.run_hash("a=1")
    # Old format: hash\tparam_line (no exit code field)
    (sweep_dir / "ran" / "snc.txt").write_text(f"{h}\ta=1\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "snc"])
    assert result.exit_code == 0
    assert "completed" in result.output
    assert "(exit" not in result.output


def test_cli_show_mixed_statuses(sweep_dir):
    """sweep show summary line with multiple status types."""
    import json
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    os.makedirs(sweep_dir / "timing", exist_ok=True)
    os.makedirs(sweep_dir / "review", exist_ok=True)
    mod.save_meta("mix", ["python", "x.py"])
    mod.save_runs("mix", ["a=1", "a=2", "a=3", "a=4"])
    h1 = mod.run_hash("a=1")
    h2 = mod.run_hash("a=2")
    h3 = mod.run_hash("a=3")
    # a=1: completed (exit 0)
    (sweep_dir / "ran" / "mix.txt").write_text(f"{h1}\ta=1\t0\n{h2}\ta=2\t1\n")
    # a=3: running (start, no end)
    (sweep_dir / "timing" / "mix.jsonl").write_text(
        json.dumps({"hash": h3, "event": "start", "time": 1710000000.0}) + "\n"
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "mix"])
    assert result.exit_code == 0
    assert "1 completed" in result.output
    assert "1 failed" in result.output
    assert "1 running" in result.output
    assert "pending" in result.output  # a=4 should be pending in the run list


def test_cli_kill_with_sweep_id(sweep_dir, monkeypatch):
    """sweep kill --sweep-id restricts search to specified sweep."""
    import json
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "timing", exist_ok=True)
    mod.save_meta("ks1", ["python", "x.py"])
    mod.save_runs("ks1", ["a=1"])
    mod.save_meta("ks2", ["python", "x.py"])
    mod.save_runs("ks2", ["b=1"])
    h = mod.run_hash("a=1")
    (sweep_dir / "timing" / "ks1.jsonl").write_text(
        json.dumps({"hash": h, "event": "start", "time": 1710000000.0, "pid": 555}) + "\n"
    )
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    runner = CliRunner()
    # Found in ks1
    result = runner.invoke(cli, ["kill", h, "--sweep-id", "ks1"])
    assert result.exit_code == 0
    assert "ks1" in result.output
    # Not found in ks2
    result = runner.invoke(cli, ["kill", h, "--sweep-id", "ks2"])
    assert result.exit_code == 1


def test_cli_kill_signal_kill(sweep_dir, monkeypatch):
    """sweep kill --signal KILL sends SIGKILL."""
    import json
    import signal as signal_mod
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "timing", exist_ok=True)
    mod.save_meta("ksk", ["python", "x.py"])
    mod.save_runs("ksk", ["a=1"])
    h = mod.run_hash("a=1")
    (sweep_dir / "timing" / "ksk.jsonl").write_text(
        json.dumps({"hash": h, "event": "start", "time": 1710000000.0, "pid": 888}) + "\n"
    )
    killed = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    runner = CliRunner()
    result = runner.invoke(cli, ["kill", h, "--signal", "KILL"])
    assert result.exit_code == 0
    assert "Sent KILL" in result.output
    assert killed[0][1] == signal_mod.SIGKILL


def test_cli_logs_with_sweep_id(sweep_dir):
    """sweep logs --sweep-id restricts search to specified sweep."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("ls1", ["python", "x.py"])
    mod.save_runs("ls1", ["a=1"])
    mod.save_meta("ls2", ["python", "x.py"])
    mod.save_runs("ls2", ["b=1"])
    h = mod.run_hash("a=1")
    log_dir = sweep_dir / "logs" / "ls1"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{h}.log").write_text("found it\n")
    runner = CliRunner()
    # Found in ls1
    result = runner.invoke(cli, ["logs", h, "--sweep-id", "ls1"])
    assert result.exit_code == 0
    assert "found it" in result.output
    # Not found in ls2 (hash doesn't exist there)
    result = runner.invoke(cli, ["logs", h, "--sweep-id", "ls2"])
    assert result.exit_code == 1


def test_cli_info_blank_line_separator(sweep_dir):
    """sweep info separates multiple results with blank lines."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("bls", ["python", "x.py"])
    mod.save_runs("bls", ["a=1", "a=2"])
    h1 = mod.run_hash("a=1")
    h2 = mod.run_hash("a=2")
    runner = CliRunner()
    result = runner.invoke(cli, ["info", h1, h2])
    assert result.exit_code == 0
    # Should have a blank line separating the two entries
    assert "\n\n" in result.output
