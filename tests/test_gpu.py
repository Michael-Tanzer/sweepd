#!/usr/bin/env python3
"""
Tests for sweep GPU utilities: is_gpu_free and sweep_daemon.
"""
import os
import subprocess
import pytest


# ---------------------------------------------------------------------------
# is_gpu_free tests (no sweep_dir needed — GPU checks are path-independent)
# ---------------------------------------------------------------------------

def test_is_gpu_free_returns_true(monkeypatch):
    """is_gpu_free returns True when VRAM usage is below the threshold."""
    from sweep.gpu import is_gpu_free

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": "500\n"})()

    monkeypatch.setattr("sweep.gpu.subprocess.run", fake_run)
    assert is_gpu_free(vram_threshold_mb=1000) is True


def test_is_gpu_free_returns_false(monkeypatch):
    """is_gpu_free returns False when VRAM usage meets or exceeds the threshold."""
    from sweep.gpu import is_gpu_free

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": "1500\n"})()

    monkeypatch.setattr("sweep.gpu.subprocess.run", fake_run)
    assert is_gpu_free(vram_threshold_mb=1000) is False


def test_is_gpu_free_nvidia_smi_not_found(monkeypatch):
    """is_gpu_free returns None when nvidia-smi is not installed."""
    from sweep.gpu import is_gpu_free

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr("sweep.gpu.subprocess.run", fake_run)
    assert is_gpu_free() is None


def test_is_gpu_free_nonzero_returncode(monkeypatch):
    """is_gpu_free returns None when nvidia-smi exits with non-zero status."""
    from sweep.gpu import is_gpu_free

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 1, "stdout": ""})()

    monkeypatch.setattr("sweep.gpu.subprocess.run", fake_run)
    assert is_gpu_free() is None


def test_is_gpu_free_timeout(monkeypatch):
    """is_gpu_free returns None when nvidia-smi times out."""
    from sweep.gpu import is_gpu_free

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 10)

    monkeypatch.setattr("sweep.gpu.subprocess.run", fake_run)
    assert is_gpu_free() is None


# ---------------------------------------------------------------------------
# sweep_daemon tests — need both sweep.core and sweep.gpu patched
# ---------------------------------------------------------------------------

@pytest.fixture
def sweep_dir_gpu(tmp_path):
    """
    Patch sweep, sweep.core, and sweep.gpu module namespaces so all path-dependent
    calls use tmp_path. Also patches sweep.gpu's imported references to core functions.
    """
    import sweep as mod
    import sweep.core as core
    import sweep.gpu as gpu

    base = str(tmp_path)
    attrs = [
        "get_sweep_dir", "get_configs_dir", "get_ran_dir", "get_review_dir",
        "_meta_path", "_runs_path", "_ran_path", "_review_path",
        "_legacy_sweep_path", "_legacy_ran_path",
    ]
    orig_mod = {k: getattr(mod, k) for k in attrs}
    orig_core = {k: getattr(core, k) for k in attrs}
    orig_gpu = {k: getattr(gpu, k) for k in attrs if hasattr(gpu, k)}

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
        if hasattr(gpu, k):
            setattr(gpu, k, v)

    # Also patch gpu module's imported references to core functions
    gpu_core_attrs = ["claim_next_run", "execute_run", "get_sweep_config", "list_sweep_ids", "run_hash"]
    orig_gpu_core = {k: getattr(gpu, k) for k in gpu_core_attrs}

    try:
        yield tmp_path
    finally:
        for k, v in orig_mod.items():
            setattr(mod, k, v)
        for k, v in orig_core.items():
            setattr(core, k, v)
        for k, v in orig_gpu.items():
            setattr(gpu, k, v)
        for k, v in orig_gpu_core.items():
            setattr(gpu, k, v)


def test_sweep_daemon_cpu_mode_runs_all_claims_then_exits(sweep_dir_gpu, monkeypatch):
    """
    sweep_daemon in cpu_mode claims and executes all runs, then sleeps.
    We break the infinite loop by making time.sleep raise StopIteration on the
    first "no work" sleep (i.e., after all runs are exhausted).
    """
    import sweep as mod
    import sweep.gpu as gpu

    os.makedirs(sweep_dir_gpu / "configs", exist_ok=True)
    os.makedirs(sweep_dir_gpu / "ran", exist_ok=True)
    mod.save_meta("d1", ["python", "x.py"])
    mod.save_runs("d1", ["a=1", "a=2"])

    executed = []

    def fake_execute(cmd, param_line):
        executed.append(param_line)
        return 0

    # Patch execute_run on the gpu module's namespace (where it was imported)
    monkeypatch.setattr(gpu, "execute_run", fake_execute)

    # Break loop after first "no work" sleep
    sleep_calls = []

    def fake_sleep(secs):
        sleep_calls.append(secs)
        raise StopIteration("done")

    monkeypatch.setattr(gpu, "time", type("T", (), {"sleep": staticmethod(fake_sleep)})())

    with pytest.raises(StopIteration):
        gpu.sweep_daemon(["d1"], interval=1, cpu_mode=True)

    assert executed == ["a=1", "a=2"]
    assert len(sleep_calls) == 1  # exactly one "no work" sleep before we stopped


def test_sweep_daemon_gpu_unavailable_falls_back_to_cpu(sweep_dir_gpu, monkeypatch):
    """
    When is_gpu_free returns None (nvidia-smi unavailable), the daemon sets
    cpu_mode=True and proceeds without GPU checks.
    """
    import sweep as mod
    import sweep.gpu as gpu

    os.makedirs(sweep_dir_gpu / "configs", exist_ok=True)
    os.makedirs(sweep_dir_gpu / "ran", exist_ok=True)
    mod.save_meta("d2", ["python", "x.py"])
    mod.save_runs("d2", ["b=1"])

    executed = []

    def fake_execute(cmd, param_line):
        executed.append(param_line)
        return 0

    monkeypatch.setattr(gpu, "execute_run", fake_execute)

    # nvidia-smi unavailable
    monkeypatch.setattr(gpu, "is_gpu_free", lambda *a, **kw: None)

    sleep_calls = []

    def fake_sleep(secs):
        sleep_calls.append(secs)
        raise StopIteration("done")

    monkeypatch.setattr(gpu, "time", type("T", (), {"sleep": staticmethod(fake_sleep)})())

    with pytest.raises(StopIteration):
        gpu.sweep_daemon(["d2"], interval=1, cpu_mode=False)

    # GPU unavailable → fell back to cpu_mode; still executed the run
    assert executed == ["b=1"]
