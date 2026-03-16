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
def sweep_dir_gpu(sweep_dir):
    """Alias for sweep_dir that also works for GPU daemon tests.

    The gpu module imports core functions directly, so patching core
    (done by sweep_dir) is sufficient since gpu.claim_next_run etc.
    are references to the same core function objects.
    """
    return sweep_dir


def _make_fake_start_process(exit_code=0):
    """Create a fake start_process for daemon tests."""
    executed = []
    def fake_start(cmd, line, log_path=None):
        executed.append(line)
        return type("FakeProc", (), {"pid": 99999, "wait": lambda self: exit_code})()
    return fake_start, executed


def test_sweep_daemon_logs_startup(sweep_dir_gpu, monkeypatch, caplog):
    """sweep_daemon logs startup messages at INFO level."""
    import logging
    import sweep as mod
    import sweep.gpu as gpu

    os.makedirs(sweep_dir_gpu / "configs", exist_ok=True)
    os.makedirs(sweep_dir_gpu / "ran", exist_ok=True)
    mod.save_meta("dlog", ["python", "x.py"])
    mod.save_runs("dlog", [])

    fake_start, _ = _make_fake_start_process()
    monkeypatch.setattr(gpu, "start_process", fake_start)

    sleep_calls = []
    def fake_sleep(secs):
        sleep_calls.append(secs)
        raise StopIteration("done")
    monkeypatch.setattr(gpu, "time", type("T", (), {"sleep": staticmethod(fake_sleep)})())

    with caplog.at_level(logging.INFO, logger="sweep.gpu"):
        with pytest.raises(StopIteration):
            gpu.sweep_daemon(["dlog"], interval=1, cpu_mode=True)
    messages = " ".join(r.message for r in caplog.records)
    assert "Daemon started" in messages


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

    fake_start, executed = _make_fake_start_process()
    monkeypatch.setattr(gpu, "start_process", fake_start)

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

    fake_start, executed = _make_fake_start_process()
    monkeypatch.setattr(gpu, "start_process", fake_start)

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
