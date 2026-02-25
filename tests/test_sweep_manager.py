#!/usr/bin/env python3
"""
Tests for sweep manager: run hash, meta+runs round-trip, grid expansion, claim_next_run, CLI.
Uses a temporary directory for sweep data.
"""
import os
import tempfile
import pytest


@pytest.fixture
def sweep_dir(tmp_path):
    """Override path helpers to use tmp_path for the duration of the test."""
    import sweep as mod
    base = str(tmp_path)
    orig = {
        "get_sweep_dir": mod.get_sweep_dir,
        "get_configs_dir": mod.get_configs_dir,
        "get_ran_dir": mod.get_ran_dir,
        "_meta_path": mod._meta_path,
        "_runs_path": mod._runs_path,
        "_ran_path": mod._ran_path,
        "_legacy_sweep_path": mod._legacy_sweep_path,
        "_legacy_ran_path": mod._legacy_ran_path,
    }
    mod.get_sweep_dir = lambda: base
    mod.get_configs_dir = lambda: os.path.join(base, "configs")
    mod.get_ran_dir = lambda: os.path.join(base, "ran")
    mod._meta_path = lambda sid: os.path.join(base, "configs", f"{sid}.meta.toml")
    mod._runs_path = lambda sid: os.path.join(base, "configs", f"{sid}.runs.txt")
    mod._ran_path = lambda sid: os.path.join(base, "ran", f"{sid}.txt")
    mod._legacy_sweep_path = lambda sid: os.path.join(base, f"{sid}.txt")
    mod._legacy_ran_path = lambda sid: os.path.join(base, f"{sid}_ran.txt")
    try:
        yield tmp_path
    finally:
        for k, v in orig.items():
            setattr(mod, k, v)


def test_run_hash_order_independent():
    """Order-independent hash: a=1,b=2 and b=2,a=1 yield same hash."""
    from sweep import run_hash
    h1 = run_hash("a=1,b=2")
    h2 = run_hash("b=2,a=1")
    assert h1 == h2
    assert len(h1) == 6
    assert all(c in "0123456789abcdef" for c in h1)


def test_run_hash_different_runs():
    """Different param lines yield different hashes."""
    from sweep import run_hash
    assert run_hash("a=1") != run_hash("a=2")
    assert run_hash("a=1,b=2") != run_hash("a=1,b=3")


def test_param_line_to_dict():
    """Parse param line into dict for table UI."""
    from sweep import param_line_to_dict
    d = param_line_to_dict("training.lr=0.01,training.batch_size=8")
    assert d["training.lr"] == "0.01"
    assert d["training.batch_size"] == "8"


def test_meta_runs_roundtrip(sweep_dir):
    """Save meta + runs, load back via get_sweep_config."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("foo", ["python", "script.py"])
    mod.save_runs("foo", ["a=1,b=2", "a=2,b=3"])
    command, param_lines = mod.get_sweep_config("foo")
    assert command == ["python", "script.py"]
    assert param_lines == ["a=1,b=2", "a=2,b=3"]


def test_grid_expansion():
    """Grid expansion produces cartesian product."""
    from sweep_cli import _expand_grid
    lines = _expand_grid("gpu=0", ["training.lr=0.01,0.001", "training.bs=8,16"])
    assert len(lines) == 4
    assert "gpu=0" in lines[0]
    assert "training.lr=0.01" in lines[0]
    assert "training.bs=8" in lines[0]
    assert "training.lr=0.001" in lines[2]
    assert "training.bs=16" in lines[1]


def test_dedup_against_existing():
    """Dedup skips runs that already exist (by hash)."""
    from sweep_cli import _dedup_against_existing
    existing = ["a=1,b=2"]
    new = ["a=1,b=2", "a=2,b=3", "b=3,a=2"]
    to_add, skipped = _dedup_against_existing(existing, new)
    assert skipped >= 1
    assert "a=2,b=3" in to_add or "b=3,a=2" in to_add
    assert len(to_add) + skipped == 3


def test_claim_next_run_and_ran_file(sweep_dir):
    """Claim writes hash\tparam_line to ran file; get_completed_hashes returns them."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("bar", ["python", "x.py"])
    mod.save_runs("bar", ["k=1", "k=2", "k=3"])
    completed = mod.get_completed_hashes("bar")
    assert len(completed) == 0
    p1 = mod.claim_next_run("bar")
    assert p1 == "k=1"
    completed = mod.get_completed_hashes("bar")
    assert len(completed) == 1
    assert mod.run_hash("k=1") in completed
    p2 = mod.claim_next_run("bar")
    assert p2 == "k=2"
    p3 = mod.claim_next_run("bar")
    assert p3 == "k=3"
    p4 = mod.claim_next_run("bar")
    assert p4 is None
    ran_path = sweep_dir / "ran" / "bar.txt"
    content = ran_path.read_text()
    lines = [l for l in content.strip().split("\n") if l]
    assert len(lines) == 3
    assert all("\t" in l for l in lines)


def test_list_sweep_ids(sweep_dir):
    """list_sweep_ids returns sweep IDs from configs/*.meta.toml."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("s1", ["python", "a.py"])
    mod.save_runs("s1", [])
    mod.save_meta("s2", ["uv", "run", "python", "b.py"])
    mod.save_runs("s2", ["x=1"])
    ids = mod.list_sweep_ids()
    assert set(ids) == {"s1", "s2"}


def test_remove_ran_lines_by_hashes(sweep_dir):
    """remove_ran_lines_by_hashes removes lines with given hashes."""
    import sweep as mod
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    ran_path = sweep_dir / "ran" / "baz.txt"
    h1 = mod.run_hash("a=1")
    h2 = mod.run_hash("a=2")
    ran_path.write_text(f"{h1}\ta=1\n{h2}\ta=2\n")
    mod.remove_ran_lines_by_hashes("baz", {h1})
    remaining = ran_path.read_text().strip().split("\n")
    assert len(remaining) == 1
    assert remaining[0].startswith(h2)


def test_add_ran_lines_by_hashes(sweep_dir):
    """add_ran_lines_by_hashes adds lines to ran file for given hashes."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("test_add", ["python", "test.py"])
    mod.save_runs("test_add", ["a=1", "a=2", "a=3"])
    h1 = mod.run_hash("a=1")
    h2 = mod.run_hash("a=2")
    h3 = mod.run_hash("a=3")
    ran_path = sweep_dir / "ran" / "test_add.txt"
    ran_path.write_text(f"{h1}\ta=1\n")
    mod.add_ran_lines_by_hashes("test_add", {h2, h3})
    lines = ran_path.read_text().strip().split("\n")
    assert len(lines) == 3
    hashes_in_file = {line.split("\t")[0] for line in lines}
    assert h1 in hashes_in_file
    assert h2 in hashes_in_file
    assert h3 in hashes_in_file
    mod.add_ran_lines_by_hashes("test_add", {h2})
    lines_after = ran_path.read_text().strip().split("\n")
    assert len(lines_after) == 3


def test_append_runs(sweep_dir):
    """append_runs adds lines to runs.txt."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("qux", ["python", "x.py"])
    mod.save_runs("qux", ["first"])
    mod.append_runs("qux", ["second", "third"])
    lines = mod.get_runs("qux")
    assert lines == ["first", "second", "third"]


def test_get_sweep_config_not_found(sweep_dir):
    """get_sweep_config raises FileNotFoundError when sweep does not exist."""
    import sweep as mod
    with pytest.raises(FileNotFoundError, match="Sweep not found"):
        mod.get_sweep_config("nonexistent")


def test_get_sweep_config_legacy(sweep_dir):
    """get_sweep_config reads legacy ~/.sweeps/<id>.txt when new layout absent."""
    import sweep as mod
    legacy_path = sweep_dir / "legacy_sweep.txt"
    legacy_path.write_text("python script.py\na=1,b=2\nx=3\n")
    command, param_lines = mod.get_sweep_config("legacy_sweep")
    assert command == ["python", "script.py"]
    assert param_lines == ["a=1,b=2", "x=3"]


def test_get_completed_hashes_legacy_ran(sweep_dir):
    """get_completed_hashes reads legacy _ran.txt (param lines only) and hashes them."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("leg", ["python", "x.py"])
    mod.save_runs("leg", ["k=1", "k=2"])
    legacy_ran = sweep_dir / "leg_ran.txt"
    legacy_ran.write_text("k=1\n")
    completed = mod.get_completed_hashes("leg")
    assert mod.run_hash("k=1") in completed
    assert len(completed) == 1


def test_get_completed_hashes_new_format_with_legacy_lines(sweep_dir):
    """get_completed_hashes handles ran file with mixed hash\tline and legacy-only lines."""
    import sweep as mod
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    h = mod.run_hash("a=1")
    (sweep_dir / "ran" / "m.txt").write_text(f"{h}\ta=1\nb=2\n")
    completed = mod.get_completed_hashes("m")
    assert h in completed
    assert mod.run_hash("b=2") in completed
    assert len(completed) == 2


def test_get_runs_not_found_returns_empty(sweep_dir):
    """get_runs returns [] when sweep does not exist (no meta, no legacy)."""
    import sweep as mod
    assert mod.get_runs("missing") == []


def test_migrate_sweep_success(sweep_dir):
    """migrate_sweep creates meta.toml, runs.txt, and ran from legacy files."""
    import sweep as mod
    (sweep_dir / "old.txt").write_text("uv run python main.py\np=1\np=2\n")
    (sweep_dir / "old_ran.txt").write_text("p=1\n")
    result = mod.migrate_sweep("old")
    assert result is True
    assert (sweep_dir / "configs" / "old.meta.toml").exists()
    assert (sweep_dir / "configs" / "old.runs.txt").read_text().strip().split("\n") == ["p=1", "p=2"]
    ran_content = (sweep_dir / "ran" / "old.txt").read_text().strip().split("\n")
    assert len(ran_content) == 1
    assert ran_content[0].startswith(mod.run_hash("p=1"))
    assert "\t" in ran_content[0] and "p=1" in ran_content[0]


def test_migrate_sweep_idempotent(sweep_dir):
    """migrate_sweep returns False and does not overwrite when new layout already exists."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("already", ["python", "a.py"])
    mod.save_runs("already", ["x=1"])
    (sweep_dir / "already.txt").write_text("python b.py\nx=2\n")
    result = mod.migrate_sweep("already")
    assert result is False
    command, param_lines = mod.get_sweep_config("already")
    assert param_lines == ["x=1"]


def test_migrate_sweep_no_legacy_raises(sweep_dir):
    """migrate_sweep raises FileNotFoundError when legacy file does not exist."""
    import sweep as mod
    with pytest.raises(FileNotFoundError, match="Legacy sweep file not found"):
        mod.migrate_sweep("nope")


def test_get_default_command_from_cwd_config(sweep_dir, monkeypatch):
    """get_default_command returns command from config/sweep_defaults.toml when cwd has it."""
    import sweep as mod
    monkeypatch.chdir(sweep_dir)
    config_dir = sweep_dir / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "sweep_defaults.toml").write_text('command = ["uv", "run", "python", "app.py"]\n')
    cmd = mod.get_default_command()
    assert cmd == ["uv", "run", "python", "app.py"]


def test_get_default_command_returns_none_when_no_config(sweep_dir, monkeypatch):
    """get_default_command returns None when no config file exists."""
    import sweep as mod
    monkeypatch.chdir(sweep_dir)
    assert mod.get_default_command() is None


def test_param_line_to_dict_empty():
    """param_line_to_dict handles empty string and single key without value."""
    from sweep import param_line_to_dict
    assert param_line_to_dict("") == {}
    d = param_line_to_dict("foo")
    assert d == {"foo": ""}


def test_list_sweep_ids_includes_legacy(sweep_dir):
    """list_sweep_ids includes sweep IDs from legacy .txt files (excluding _ran.txt)."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("new_one", ["python", "x.py"])
    mod.save_runs("new_one", [])
    (sweep_dir / "legacy_only.txt").write_text("python y.py\nz=1\n")
    (sweep_dir / "legacy_ran.txt").write_text("done\n")
    ids = mod.list_sweep_ids()
    assert "new_one" in ids
    assert "legacy_only" in ids
    assert "legacy_ran" not in ids


def test_save_meta_creates_valid_toml(sweep_dir):
    """save_meta writes TOML that can be read back by _load_meta."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("t", ["uv", "run", "python", "t.py"])
    cmd = mod._load_meta("t")
    assert cmd == ["uv", "run", "python", "t.py"]


def test_load_legacy_sweep_invalid_first_line(sweep_dir):
    """_load_legacy_sweep raises ValueError when first line has fewer than 2 parts."""
    import sweep as mod
    (sweep_dir / "bad.txt").write_text("only_one_token\na=1\n")
    with pytest.raises(ValueError, match="First line must be"):
        mod._load_legacy_sweep("bad")


def test_load_legacy_sweep_too_few_lines(sweep_dir):
    """_load_legacy_sweep raises ValueError when file has fewer than 2 lines."""
    import sweep as mod
    (sweep_dir / "short.txt").write_text("python x.py\n")
    with pytest.raises(ValueError, match="at least 2 lines"):
        mod._load_legacy_sweep("short")


def test_execute_run_builds_correct_command(monkeypatch):
    """execute_run builds command as command + params from param_line and calls subprocess."""
    from sweep import execute_run
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()
    monkeypatch.setattr("sweep.subprocess.run", fake_run)
    execute_run(["python", "train.py"], "lr=0.01,batch_size=8")
    assert len(calls) == 1
    assert calls[0] == ["python", "train.py", "lr=0.01", "batch_size=8"]


def test_sweep_run_loop_until_done(sweep_dir, monkeypatch):
    """sweep_run claims runs and executes until claim_next_run returns None."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("loop", ["python", "x.py"])
    mod.save_runs("loop", ["a=1", "a=2"])
    claimed = []
    executed = []
    original_claim = mod.claim_next_run
    def track_claim(sid):
        p = original_claim(sid)
        if p:
            claimed.append(p)
        return p
    def track_execute(cmd, line):
        executed.append((cmd, line))
        return 0
    monkeypatch.setattr(mod, "claim_next_run", track_claim)
    monkeypatch.setattr(mod, "execute_run", track_execute)
    mod.sweep_run("loop")
    assert len(claimed) == 2
    assert claimed == ["a=1", "a=2"]
    assert len(executed) == 2
    assert executed[0][1] == "a=1"
    assert executed[1][1] == "a=2"
