#!/usr/bin/env python3
"""
Tests for sweep web API: list sweeps, get sweep, create, add runs, mark rerun, delete, remove runs.
Uses FastAPI TestClient and temporary directory for sweep data.
"""
import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sweep_dir(tmp_path):
    """Patch sweep module to use tmp_path so API reads/writes there."""
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


@pytest.fixture
def client(sweep_dir):
    """TestClient for the FastAPI app. Reload sweep_web after patching so handlers use patched paths."""
    import importlib
    import sweep_web as sw
    importlib.reload(sw)
    return TestClient(sw.app)


@pytest.fixture
def default_command(sweep_dir):
    """Create config/sweep_defaults.toml so API create can use default command."""
    config_dir = sweep_dir / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "sweep_defaults.toml").write_text('command = ["python", "main.py"]\n')
    return sweep_dir


def test_api_list_sweeps_empty(client):
    """GET /api/sweeps returns empty list when no sweeps."""
    r = client.get("/api/sweeps")
    assert r.status_code == 200
    assert r.json() == {"sweep_ids": []}


def test_api_list_sweeps(client, sweep_dir):
    """GET /api/sweeps returns sweep IDs."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("w1", ["python", "x.py"])
    mod.save_runs("w1", [])
    r = client.get("/api/sweeps")
    assert r.status_code == 200
    assert "w1" in r.json()["sweep_ids"]


def test_api_default_command_404(client, sweep_dir, monkeypatch):
    """GET /api/default-command returns 404 when no config."""
    monkeypatch.chdir(sweep_dir)
    r = client.get("/api/default-command")
    assert r.status_code == 404


def test_api_default_command(client, sweep_dir, default_command, monkeypatch):
    """GET /api/default-command returns command when config exists."""
    monkeypatch.chdir(sweep_dir)
    r = client.get("/api/default-command")
    assert r.status_code == 200
    assert r.json()["command"] == ["python", "main.py"]


def test_api_get_sweep_404(client):
    """GET /api/sweeps/{id} returns 404 for missing sweep."""
    r = client.get("/api/sweeps/nonexistent")
    assert r.status_code == 404


def test_api_get_sweep(client, sweep_dir):
    """GET /api/sweeps/{id} returns meta, rows (one key per column), columns."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("g1", ["uv", "run", "python", "train.py"])
    mod.save_runs("g1", ["training.lr=0.01", "training.lr=0.001"])
    r = client.get("/api/sweeps/g1")
    assert r.status_code == 200
    data = r.json()
    assert data["sweep_id"] == "g1"
    assert data["command"] == ["uv", "run", "python", "train.py"]
    assert data["total_count"] == 2
    assert len(data["rows"]) == 2
    assert "training.lr" in data["columns"]
    assert data["rows"][0]["hash"] is not None
    assert data["rows"][0]["status"] in ("ran", "pending")


def test_api_create_sweep(client, sweep_dir, default_command, monkeypatch):
    """POST /api/sweeps creates sweep and returns added/skipped."""
    monkeypatch.chdir(sweep_dir)
    r = client.post("/api/sweeps", json={
        "sweep_id": "new1",
        "runs": ["a=1", "a=2"],
    })
    assert r.status_code == 200
    j = r.json()
    assert j["added"] == 2
    assert j["skipped"] == 0
    r2 = client.get("/api/sweeps/new1")
    assert r2.status_code == 200
    assert r2.json()["total_count"] == 2


def test_api_create_sweep_with_command(client, sweep_dir):
    """POST /api/sweeps with command creates sweep without default config."""
    r = client.post("/api/sweeps", json={
        "sweep_id": "cmd1",
        "command": ["python", "script.py"],
        "runs": ["x=1"],
    })
    assert r.status_code == 200
    r2 = client.get("/api/sweeps/cmd1")
    assert r2.json()["command"] == ["python", "script.py"]


def test_api_create_sweep_no_runs_no_grid_400(client, sweep_dir, default_command, monkeypatch):
    """POST /api/sweeps without runs or grid returns 400."""
    monkeypatch.chdir(sweep_dir)
    r = client.post("/api/sweeps", json={"sweep_id": "bad"})
    assert r.status_code == 400


def test_api_add_runs(client, sweep_dir, default_command, monkeypatch):
    """POST /api/sweeps/{id}/runs adds runs with dedup."""
    import sweep as mod
    monkeypatch.chdir(sweep_dir)
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("add1", ["python", "x.py"])
    mod.save_runs("add1", ["k=1"])
    r = client.post("/api/sweeps/add1/runs", json={
        "runs": ["k=2", "k=1"],
    })
    assert r.status_code == 200
    assert r.json()["added"] == 1
    assert r.json()["skipped"] == 1
    lines = mod.get_runs("add1")
    assert "k=1" in lines and "k=2" in lines


def test_api_add_runs_sweep_not_found_404(client):
    """POST /api/sweeps/{id}/runs for missing sweep returns 404."""
    r = client.post("/api/sweeps/missing/runs", json={"runs": ["a=1"]})
    assert r.status_code == 404


def test_api_mark_rerun(client, sweep_dir):
    """POST /api/sweeps/{id}/runs/mark-rerun removes hashes from ran file."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("mr", ["python", "x.py"])
    mod.save_runs("mr", ["p=1", "p=2"])
    (sweep_dir / "ran" / "mr.txt").write_text(
        mod.run_hash("p=1") + "\tp=1\n" + mod.run_hash("p=2") + "\tp=2\n"
    )
    r = client.post("/api/sweeps/mr/runs/mark-rerun", json={
        "hashes": [mod.run_hash("p=1")],
    })
    assert r.status_code == 200
    assert r.json()["marked"] == 1
    remaining = mod.get_completed_hashes("mr")
    assert mod.run_hash("p=1") not in remaining
    assert mod.run_hash("p=2") in remaining


def test_api_mark_rerun_no_hashes_no_indices_400(client, sweep_dir):
    """POST mark-rerun without hashes or indices returns 400."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("mr2", ["python", "x.py"])
    mod.save_runs("mr2", [])
    r = client.post("/api/sweeps/mr2/runs/mark-rerun", json={})
    assert r.status_code == 400


def test_api_mark_ran(client, sweep_dir):
    """POST /api/sweeps/{id}/runs/mark-ran adds hashes to ran file."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("mran", ["python", "x.py"])
    mod.save_runs("mran", ["q=1", "q=2"])
    h1 = mod.run_hash("q=1")
    h2 = mod.run_hash("q=2")
    (sweep_dir / "ran" / "mran.txt").write_text(h1 + "\tq=1\n")
    r = client.post("/api/sweeps/mran/runs/mark-ran", json={
        "hashes": [h2],
    })
    assert r.status_code == 200
    assert r.json()["marked"] == 1
    completed = mod.get_completed_hashes("mran")
    assert h1 in completed
    assert h2 in completed


def test_api_mark_ran_by_index(client, sweep_dir):
    """POST mark-ran with indices adds corresponding runs to ran file."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("mrani", ["python", "x.py"])
    mod.save_runs("mrani", ["r=1", "r=2"])
    h1 = mod.run_hash("r=1")
    h2 = mod.run_hash("r=2")
    r = client.post("/api/sweeps/mrani/runs/mark-ran", json={
        "indices": [0, 1],
    })
    assert r.status_code == 200
    assert r.json()["marked"] == 2
    completed = mod.get_completed_hashes("mrani")
    assert h1 in completed
    assert h2 in completed


def test_api_mark_ran_no_hashes_no_indices_400(client, sweep_dir):
    """POST mark-ran without hashes or indices returns 400."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("mran2", ["python", "x.py"])
    mod.save_runs("mran2", [])
    r = client.post("/api/sweeps/mran2/runs/mark-ran", json={})
    assert r.status_code == 400


def test_api_delete_sweep(client, sweep_dir, default_command, monkeypatch):
    """DELETE /api/sweeps/{id} removes meta, runs, ran files."""
    import sweep as mod
    monkeypatch.chdir(sweep_dir)
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("del1", ["python", "x.py"])
    mod.save_runs("del1", [])
    r = client.delete("/api/sweeps/del1")
    assert r.status_code == 200
    assert len(r.json()["removed"]) >= 1
    r2 = client.get("/api/sweeps/del1")
    assert r2.status_code == 404


def test_api_remove_runs(client, sweep_dir):
    """POST /api/sweeps/{id}/runs/remove removes runs at indices."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("rm1", ["python", "x.py"])
    mod.save_runs("rm1", ["a=1", "a=2", "a=3"])
    r = client.post("/api/sweeps/rm1/runs/remove", json={"indices": [0, 2]})
    assert r.status_code == 200
    assert r.json()["removed"] == 2
    lines = mod.get_runs("rm1")
    assert lines == ["a=2"]


def test_api_remove_runs_sweep_not_found_404(client):
    """POST runs/remove for missing sweep returns 404."""
    r = client.post("/api/sweeps/missing/runs/remove", json={"indices": [0]})
    assert r.status_code == 404


def test_api_index_serves_html_or_message(client):
    """GET / returns 200; index.html (text/html) or JSON message."""
    r = client.get("/")
    assert r.status_code == 200
    ct = r.headers.get("content-type", "")
    if "application/json" in ct:
        data = r.json()
        assert "message" in data
    else:
        assert "html" in ct.lower() or len(r.content) > 0
