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
def sweep_dir(tmp_path):
    """Patch sweep module to use tmp_path for all sweep data."""
    import sweep as mod
    import sweep.core as core
    base = str(tmp_path)
    attrs = [
        "get_sweep_dir", "get_configs_dir", "get_ran_dir", "get_review_dir",
        "_meta_path", "_runs_path", "_ran_path", "_review_path",
        "_legacy_sweep_path", "_legacy_ran_path",
    ]
    orig_mod = {k: getattr(mod, k) for k in attrs}
    orig_core = {k: getattr(core, k) for k in attrs}
    patches = {
        "get_sweep_dir": lambda: base,
        "get_configs_dir": lambda: os.path.join(base, "configs"),
        "get_ran_dir": lambda: os.path.join(base, "ran"),
        "get_review_dir": lambda: os.path.join(base, "review"),
        "_meta_path": lambda sid: os.path.join(base, "configs", f"{sid}.meta.toml"),
        "_runs_path": lambda sid: os.path.join(base, "configs", f"{sid}.runs.txt"),
        "_ran_path": lambda sid: os.path.join(base, "ran", f"{sid}.txt"),
        "_review_path": lambda sid: os.path.join(base, "review", f"{sid}.txt"),
        "_legacy_sweep_path": lambda sid: os.path.join(base, f"{sid}.txt"),
        "_legacy_ran_path": lambda sid: os.path.join(base, f"{sid}_ran.txt"),
    }
    for k, v in patches.items():
        setattr(mod, k, v)
        setattr(core, k, v)
    try:
        yield tmp_path
    finally:
        for k, v in orig_mod.items():
            setattr(mod, k, v)
        for k, v in orig_core.items():
            setattr(core, k, v)


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
