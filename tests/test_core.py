#!/usr/bin/env python3
"""
Tests for sweep manager: run hash, meta+runs round-trip, grid expansion, claim_next_run, CLI.
Uses a temporary directory for sweep data.
"""
import os
import tempfile
import pytest


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


def test_run_hash_order_independent_with_quoted_commas():
    """Order-independent hash still holds when values contain commas inside quotes."""
    from sweep import run_hash
    h1 = run_hash('a="x, y",b=2')
    h2 = run_hash('b=2,a="x, y"')
    assert h1 == h2


def test_param_line_to_dict():
    """Parse param line into dict for table UI."""
    from sweep import param_line_to_dict
    d = param_line_to_dict("training.lr=0.01,training.batch_size=8")
    assert d["training.lr"] == "0.01"
    assert d["training.batch_size"] == "8"


def test_param_line_to_dict_with_quoted_commas():
    """param_line_to_dict treats commas inside quotes as part of the value."""
    from sweep import param_line_to_dict
    line = 'a="x, y",b=2'
    d = param_line_to_dict(line)
    assert d["a"] == '"x, y"'
    assert d["b"] == "2"


def test_split_param_line_hydra_plus_single_quoted():
    """split_param_line keeps Hydra + override with nested quotes as one segment."""
    from sweep import split_param_line
    line = 'experiment=malaria_patch_baseline_best,trainer.max_epochs=20,+\'logger.aim.run_name="model=${model.net.model.model_name}"\''
    parts = split_param_line(line)
    assert len(parts) == 3
    assert parts[0] == "experiment=malaria_patch_baseline_best"
    assert parts[1] == "trainer.max_epochs=20"
    assert parts[2] == """+'logger.aim.run_name="model=${model.net.model.model_name}"'"""


def test_execute_run_hydra_plus_override(monkeypatch):
    """execute_run passes Hydra + override with nested quotes as a single argument."""
    from sweep import execute_run
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr("sweep.core.subprocess.run", fake_run)
    line = 'experiment=malaria_patch_baseline_best,trainer.max_epochs=20,+\'logger.aim.run_name="model=${model.net.model.model_name}"\''
    execute_run(["uv", "run", "src/train.py"], line)
    assert len(calls) == 1
    assert calls[0] == [
        "uv", "run", "src/train.py",
        "experiment=malaria_patch_baseline_best",
        "trainer.max_epochs=20",
        """+'logger.aim.run_name="model=${model.net.model.model_name}"'""",
    ]


def test_expand_grid_base_with_quoted_comma_in_value():
    """_expand_grid must not split base_line on commas inside quoted values."""
    from sweep.cli import _expand_grid
    # base_line has a quoted value containing a comma — plain split(",") breaks this
    base = """experiment=foo,+'logger.aim.run_name="model=swin, epoch=best"'"""
    lines = _expand_grid(base, ["model/arch=swin_base,swin_small"])
    assert len(lines) == 2
    for line in lines:
        assert """'logger.aim.run_name="model=swin, epoch=best"'""" in line


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
    from sweep.cli import _expand_grid
    lines = _expand_grid("gpu=0", ["training.lr=0.01,0.001", "training.bs=8,16"])
    assert len(lines) == 4
    assert "gpu=0" in lines[0]
    assert "training.lr=0.01" in lines[0]
    assert "training.bs=8" in lines[0]
    assert "training.lr=0.001" in lines[2]
    assert "training.bs=16" in lines[1]


def test_dedup_against_existing():
    """Dedup skips runs that already exist (by hash)."""
    from sweep.cli import _dedup_against_existing
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
    from sweep.core import _load_meta
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("t", ["uv", "run", "python", "t.py"])
    cmd = _load_meta("t")
    assert cmd == ["uv", "run", "python", "t.py"]


def test_load_legacy_sweep_invalid_first_line(sweep_dir):
    """_load_legacy_sweep raises ValueError when first line has fewer than 2 parts."""
    from sweep.core import _load_legacy_sweep
    (sweep_dir / "bad.txt").write_text("only_one_token\na=1\n")
    with pytest.raises(ValueError, match="First line must be"):
        _load_legacy_sweep("bad")


def test_load_legacy_sweep_too_few_lines(sweep_dir):
    """_load_legacy_sweep raises ValueError when file has fewer than 2 lines."""
    from sweep.core import _load_legacy_sweep
    (sweep_dir / "short.txt").write_text("python x.py\n")
    with pytest.raises(ValueError, match="at least 2 lines"):
        _load_legacy_sweep("short")


def test_execute_run_logs_command(monkeypatch, caplog):
    """execute_run logs the command being executed at INFO level."""
    import logging
    from sweep import execute_run
    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 0})()
    monkeypatch.setattr("sweep.core.subprocess.run", fake_run)
    with caplog.at_level(logging.INFO, logger="sweep.core"):
        execute_run(["python", "train.py"], "lr=0.01,batch_size=8")
    assert any("Executing" in r.message for r in caplog.records)


def test_sweep_run_logs_progress(sweep_dir, monkeypatch, caplog):
    """sweep_run logs sweep info and progress at INFO level."""
    import logging
    import sweep as mod
    import sweep.core as core
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("logtest", ["python", "x.py"])
    mod.save_runs("logtest", ["a=1"])
    def fake_execute(cmd, line):
        return 0
    monkeypatch.setattr(mod, "execute_run", fake_execute)
    monkeypatch.setattr(core, "execute_run", fake_execute)
    with caplog.at_level(logging.INFO, logger="sweep.core"):
        mod.sweep_run("logtest")
    messages = " ".join(r.message for r in caplog.records)
    assert "logtest" in messages
    assert "All runs completed" in messages


def test_execute_run_builds_correct_command(monkeypatch):
    """execute_run builds command as command + params from param_line and calls subprocess."""
    from sweep import execute_run
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()
    monkeypatch.setattr("sweep.core.subprocess.run", fake_run)
    execute_run(["python", "train.py"], "lr=0.01,batch_size=8")
    assert len(calls) == 1
    assert calls[0] == ["python", "train.py", "lr=0.01", "batch_size=8"]


def test_execute_run_with_quoted_commas(monkeypatch):
    """execute_run splits param_line on commas that are not inside quotes."""
    from sweep import execute_run
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()
    monkeypatch.setattr("sweep.core.subprocess.run", fake_run)
    line = 'run.name="model=BERT, lr=0.01",run.name="model=RoBERTa, lr=0.01"'
    execute_run(["python", "train.py"], line)
    assert len(calls) == 1
    assert calls[0] == [
        "python",
        "train.py",
        'run.name="model=BERT, lr=0.01"',
        'run.name="model=RoBERTa, lr=0.01"',
    ]


def test_sweep_run_loop_until_done(sweep_dir, monkeypatch):
    """sweep_run claims runs and executes until claim_next_run returns None."""
    import sweep as mod
    import sweep.core as core
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("loop", ["python", "x.py"])
    mod.save_runs("loop", ["a=1", "a=2"])
    claimed = []
    executed = []
    original_claim = core.claim_next_run
    def track_claim(sid):
        p = original_claim(sid)
        if p:
            claimed.append(p)
        return p
    def track_execute(cmd, line):
        executed.append((cmd, line))
        return 0
    monkeypatch.setattr(mod, "claim_next_run", track_claim)
    monkeypatch.setattr(core, "claim_next_run", track_claim)
    monkeypatch.setattr(mod, "execute_run", track_execute)
    monkeypatch.setattr(core, "execute_run", track_execute)
    mod.sweep_run("loop")
    assert len(claimed) == 2
    assert claimed == ["a=1", "a=2"]
    assert len(executed) == 2
    assert executed[0][1] == "a=1"
    assert executed[1][1] == "a=2"


# --- Review system tests ---

def test_get_review_hashes_empty(sweep_dir):
    """get_review_hashes returns empty set when no review file exists."""
    import sweep as mod
    result = mod.get_review_hashes("no_such_sweep")
    assert result == set()


def test_add_review_lines_by_hashes(sweep_dir):
    """Staging runs in review makes them invisible to claim_next_run."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    os.makedirs(sweep_dir / "review", exist_ok=True)
    mod.save_meta("rv1", ["python", "x.py"])
    mod.save_runs("rv1", ["p=1", "p=2"])
    h1 = mod.run_hash("p=1")
    h2 = mod.run_hash("p=2")
    mod.add_review_lines_by_hashes("rv1", [h1, h2])
    review = mod.get_review_hashes("rv1")
    assert h1 in review
    assert h2 in review
    # claim_next_run should skip review runs and return None
    result = mod.claim_next_run("rv1")
    assert result is None


def test_add_review_lines_idempotent(sweep_dir):
    """Adding the same hash twice does not duplicate it in the review file."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "review", exist_ok=True)
    mod.save_meta("rv2", ["python", "x.py"])
    mod.save_runs("rv2", ["q=1"])
    h = mod.run_hash("q=1")
    mod.add_review_lines_by_hashes("rv2", [h])
    mod.add_review_lines_by_hashes("rv2", [h])
    review = mod.get_review_hashes("rv2")
    assert list(review).count(h) == 1


def test_promote_from_review(sweep_dir):
    """Promoting a run removes it from review and makes it claimable."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    os.makedirs(sweep_dir / "review", exist_ok=True)
    mod.save_meta("rv3", ["python", "x.py"])
    mod.save_runs("rv3", ["r=1"])
    h = mod.run_hash("r=1")
    mod.add_review_lines_by_hashes("rv3", [h])
    assert h in mod.get_review_hashes("rv3")
    mod.promote_from_review("rv3", {h})
    assert h not in mod.get_review_hashes("rv3")
    # Now claimable
    result = mod.claim_next_run("rv3")
    assert result == "r=1"


def test_promote_nonexistent_review_file(sweep_dir):
    """promote_from_review does not raise when review file is absent."""
    import sweep as mod
    # Should not raise even if no review file exists
    mod.promote_from_review("no_review_sweep", {"abc123"})


def test_append_runs_as_review(sweep_dir):
    """append_runs_as_review adds runs to runs.txt and stages them in review."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    os.makedirs(sweep_dir / "review", exist_ok=True)
    mod.save_meta("rv4", ["python", "x.py"])
    mod.save_runs("rv4", [])
    mod.append_runs_as_review("rv4", ["s=1", "s=2"])
    lines = mod.get_runs("rv4")
    assert "s=1" in lines and "s=2" in lines
    review = mod.get_review_hashes("rv4")
    assert mod.run_hash("s=1") in review
    assert mod.run_hash("s=2") in review
    # claim_next_run should return None (both runs are in review)
    result = mod.claim_next_run("rv4")
    assert result is None


# --- Exit code tracking tests ---

def test_get_completed_exit_codes(sweep_dir):
    """get_completed_exit_codes reads hash -> exit_code from ran file."""
    from sweep.core import get_completed_exit_codes, run_hash
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    h1 = run_hash("a=1")
    h2 = run_hash("a=2")
    h3 = run_hash("a=3")
    (sweep_dir / "ran" / "ec.txt").write_text(
        f"{h1}\ta=1\t0\n{h2}\ta=2\t1\n{h3}\ta=3\n"
    )
    codes = get_completed_exit_codes("ec")
    assert codes[h1] == 0
    assert codes[h2] == 1
    assert h3 not in codes  # no exit code column = unknown


def test_sweep_run_records_exit_code(sweep_dir, monkeypatch):
    """sweep_run stores exit code in the ran file."""
    import sweep as mod
    import sweep.core as core
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("ecrun", ["python", "x.py"])
    mod.save_runs("ecrun", ["a=1", "a=2"])
    call_count = [0]
    def fake_execute(cmd, line):
        call_count[0] += 1
        return 0 if call_count[0] == 1 else 1
    monkeypatch.setattr(mod, "execute_run", fake_execute)
    monkeypatch.setattr(core, "execute_run", fake_execute)
    mod.sweep_run("ecrun")
    from sweep.core import get_completed_exit_codes
    codes = get_completed_exit_codes("ecrun")
    h1 = mod.run_hash("a=1")
    h2 = mod.run_hash("a=2")
    assert codes[h1] == 0
    assert codes[h2] == 1


# --- Run timing tests ---

def test_record_and_get_run_timings(sweep_dir):
    """record_run_start + record_run_end creates timing entry; get_run_timings reads it."""
    from sweep.core import record_run_start, record_run_end, get_run_timings, run_hash
    os.makedirs(sweep_dir / "timing", exist_ok=True)
    h = run_hash("a=1")
    record_run_start("t1", h)
    record_run_end("t1", h, exit_code=0)
    timings = get_run_timings("t1")
    assert h in timings
    assert timings[h]["exit_code"] == 0
    assert timings[h]["start"] is not None
    assert timings[h]["end"] is not None
    assert timings[h]["end"] >= timings[h]["start"]


def test_get_run_timings_empty(sweep_dir):
    """get_run_timings returns empty dict when no timing file exists."""
    from sweep.core import get_run_timings
    os.makedirs(sweep_dir / "timing", exist_ok=True)
    assert get_run_timings("nonexistent") == {}


def test_run_timings_rerun_clears_stale(sweep_dir):
    """A rerun (new start without end yet) clears stale timing from the first execution."""
    from sweep.core import record_run_start, record_run_end, get_run_timings, run_hash
    os.makedirs(sweep_dir / "timing", exist_ok=True)
    h = run_hash("a=1")
    # First execution: start + end
    record_run_start("tr", h)
    record_run_end("tr", h, exit_code=1)
    # Rerun: only start so far (still in progress)
    record_run_start("tr", h)
    timings = get_run_timings("tr")
    # Should NOT return stale first-execution result
    assert h not in timings


def test_run_timings_rerun_returns_latest(sweep_dir):
    """After a rerun completes, get_run_timings returns the latest execution's data."""
    from sweep.core import record_run_start, record_run_end, get_run_timings, run_hash
    os.makedirs(sweep_dir / "timing", exist_ok=True)
    h = run_hash("a=1")
    # First execution
    record_run_start("tr2", h)
    record_run_end("tr2", h, exit_code=1)
    # Second execution
    record_run_start("tr2", h)
    record_run_end("tr2", h, exit_code=0)
    timings = get_run_timings("tr2")
    assert timings[h]["exit_code"] == 0


def test_run_timings_multiple_runs(sweep_dir):
    """Multiple runs each get their own timing entries."""
    from sweep.core import record_run_start, record_run_end, get_run_timings, run_hash
    os.makedirs(sweep_dir / "timing", exist_ok=True)
    h1 = run_hash("x=1")
    h2 = run_hash("x=2")
    record_run_start("t2", h1)
    record_run_end("t2", h1, exit_code=0)
    record_run_start("t2", h2)
    record_run_end("t2", h2, exit_code=1)
    timings = get_run_timings("t2")
    assert len(timings) == 2
    assert timings[h1]["exit_code"] == 0
    assert timings[h2]["exit_code"] == 1


def test_sweep_run_records_timing(sweep_dir, monkeypatch):
    """sweep_run records start/end timing for each run."""
    import sweep as mod
    import sweep.core as core
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "timing", exist_ok=True)
    mod.save_meta("trun", ["python", "x.py"])
    mod.save_runs("trun", ["a=1"])
    def fake_execute(cmd, line):
        return 0
    monkeypatch.setattr(mod, "execute_run", fake_execute)
    monkeypatch.setattr(core, "execute_run", fake_execute)
    mod.sweep_run("trun")
    timings = core.get_run_timings("trun")
    h = mod.run_hash("a=1")
    assert h in timings
    assert timings[h]["exit_code"] == 0
    assert timings[h]["end"] >= timings[h]["start"]


# --- Sweep cloning tests ---

def test_clone_sweep(sweep_dir):
    """clone_sweep copies meta and runs but not ran/review/timing."""
    import sweep as mod
    from sweep.core import clone_sweep
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("src", ["python", "train.py"])
    mod.save_runs("src", ["lr=0.01", "lr=0.001"])
    # Mark one as ran
    h = mod.run_hash("lr=0.01")
    (sweep_dir / "ran" / "src.txt").write_text(f"{h}\tlr=0.01\n")
    clone_sweep("src", "dst")
    # New sweep has same command and runs
    cmd, lines = mod.get_sweep_config("dst")
    assert cmd == ["python", "train.py"]
    assert lines == ["lr=0.01", "lr=0.001"]
    # No ran history
    assert mod.get_completed_hashes("dst") == set()


def test_clone_sweep_source_not_found(sweep_dir):
    """clone_sweep raises FileNotFoundError for missing source."""
    from sweep.core import clone_sweep
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    with pytest.raises(FileNotFoundError):
        clone_sweep("missing", "dst")
