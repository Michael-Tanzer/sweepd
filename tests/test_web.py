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
def client(sweep_dir):
    """TestClient for the FastAPI app. Reload sweep.web_app after patching so handlers use patched paths."""
    import importlib
    import sweep.web_app as sw
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
    mod.save_meta("s1", ["python", "x.py"])
    mod.save_runs("s1", [])
    r = client.get("/api/sweeps")
    assert r.status_code == 200
    assert "s1" in r.json()["sweep_ids"]


def test_api_get_sweep_not_found(client):
    """GET /api/sweeps/<missing> returns 404."""
    r = client.get("/api/sweeps/missing")
    assert r.status_code == 404


def test_api_get_sweep(client, sweep_dir):
    """GET /api/sweeps/<id> returns sweep details."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("s1", ["python", "x.py"])
    mod.save_runs("s1", ["a=1", "a=2"])
    r = client.get("/api/sweeps/s1")
    assert r.status_code == 200
    d = r.json()
    assert d["sweep_id"] == "s1"
    assert d["total_count"] == 2
    assert d["completed_count"] == 0
    assert len(d["rows"]) == 2


def test_api_create_sweep(client, sweep_dir, default_command, monkeypatch):
    """POST /api/sweeps creates a new sweep."""
    monkeypatch.chdir(sweep_dir)
    import importlib, sweep.web_app as sw
    importlib.reload(sw)
    client2 = TestClient(sw.app)
    r = client2.post("/api/sweeps", json={"sweep_id": "new1", "runs": ["x=1", "x=2"]})
    assert r.status_code == 200
    assert r.json()["added"] == 2


def test_api_create_sweep_no_command_no_default(client, sweep_dir, monkeypatch):
    """POST /api/sweeps without command and no default config returns 400."""
    import sweep.web_app as sw
    monkeypatch.setattr(sw, "get_default_command", lambda: None)
    r = client.post("/api/sweeps", json={"sweep_id": "noop", "runs": ["x=1"]})
    assert r.status_code == 400


def test_api_add_runs(client, sweep_dir):
    """POST /api/sweeps/<id>/runs adds runs."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("s2", ["python", "x.py"])
    mod.save_runs("s2", ["a=1"])
    r = client.post("/api/sweeps/s2/runs", json={"runs": ["a=2", "a=3"]})
    assert r.status_code == 200
    assert r.json()["added"] == 2


def test_api_add_runs_dedup(client, sweep_dir):
    """POST /api/sweeps/<id>/runs deduplicates by hash."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("s3", ["python", "x.py"])
    mod.save_runs("s3", ["a=1"])
    r = client.post("/api/sweeps/s3/runs", json={"runs": ["a=1", "a=2"]})
    assert r.status_code == 200
    d = r.json()
    assert d["added"] == 1
    assert d["skipped"] == 1


def test_api_mark_rerun(client, sweep_dir):
    """POST /api/sweeps/<id>/runs/mark-rerun removes hashes from ran file."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("mr1", ["python", "x.py"])
    mod.save_runs("mr1", ["p=1", "p=2"])
    h1 = mod.run_hash("p=1")
    h2 = mod.run_hash("p=2")
    (sweep_dir / "ran" / "mr1.txt").write_text(f"{h1}\tp=1\n{h2}\tp=2\n")
    r = client.post("/api/sweeps/mr1/runs/mark-rerun", json={"hashes": [h1]})
    assert r.status_code == 200
    remaining = mod.get_completed_hashes("mr1")
    assert h1 not in remaining
    assert h2 in remaining


def test_api_mark_ran(client, sweep_dir):
    """POST /api/sweeps/<id>/runs/mark-ran adds hashes to ran file."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "ran", exist_ok=True)
    mod.save_meta("mran1", ["python", "x.py"])
    mod.save_runs("mran1", ["q=1", "q=2"])
    h1 = mod.run_hash("q=1")
    h2 = mod.run_hash("q=2")
    r = client.post("/api/sweeps/mran1/runs/mark-ran", json={"hashes": [h1, h2]})
    assert r.status_code == 200
    completed = mod.get_completed_hashes("mran1")
    assert h1 in completed
    assert h2 in completed


def test_api_delete_sweep(client, sweep_dir):
    """DELETE /api/sweeps/<id> removes sweep files."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("del1", ["python", "x.py"])
    mod.save_runs("del1", ["z=1"])
    r = client.delete("/api/sweeps/del1")
    assert r.status_code == 200
    assert not (sweep_dir / "configs" / "del1.meta.toml").exists()


def test_api_remove_runs(client, sweep_dir):
    """POST /api/sweeps/<id>/runs/remove removes runs at given indices."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    mod.save_meta("rm1", ["python", "x.py"])
    mod.save_runs("rm1", ["a=1", "a=2", "a=3"])
    r = client.post("/api/sweeps/rm1/runs/remove", json={"indices": [1]})
    assert r.status_code == 200
    lines = mod.get_runs("rm1")
    assert lines == ["a=1", "a=3"]


def test_api_remove_runs_not_found(client):
    """POST /api/sweeps/<missing>/runs/remove returns 404."""
    r = client.post("/api/sweeps/missing/runs/remove", json={"indices": [0]})
    assert r.status_code == 404


def test_api_grid_create(client, sweep_dir, monkeypatch):
    """POST /api/sweeps with grid creates cartesian product of runs."""
    import sweep as mod
    r = client.post("/api/sweeps", json={
        "sweep_id": "grid1",
        "command": ["python", "train.py"],
        "grid": ["lr=0.01,0.001", "bs=8,16"]
    })
    assert r.status_code == 200
    assert r.json()["added"] == 4


def test_api_default_command_found(client, sweep_dir):
    """GET /api/default-command returns command when configured."""
    import sweep.web_app as sw
    sw.get_default_command = lambda: ["python", "train.py"]
    r = client.get("/api/default-command")
    assert r.status_code == 200
    assert r.json()["command"] == ["python", "train.py"]


def test_api_default_command_not_found(client):
    """GET /api/default-command returns 404 when no default command is configured."""
    import sweep.web_app as sw
    sw.get_default_command = lambda: None
    r = client.get("/api/default-command")
    assert r.status_code == 404


def test_api_grid_preview(client, sweep_dir):
    """POST /api/grid-preview returns cartesian product of runs."""
    r = client.post("/api/grid-preview", json={
        "base": "",
        "grid": ["lr=0.01,0.001", "bs=8,16"],
    })
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 4
    assert len(data["runs"]) == 4
    assert any("lr=0.01" in line and "bs=8" in line for line in data["runs"])


def test_api_grid_preview_with_quoted_commas(client, sweep_dir):
    """POST /api/grid-preview preserves commas inside quotes."""
    line = 'run.name="model=BERT, lr=0.01","model=RoBERTa, lr=0.01"'
    r = client.post("/api/grid-preview", json={"base": "", "grid": [line]})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert 'run.name="model=BERT, lr=0.01"' in data["runs"]
    assert 'run.name="model=RoBERTa, lr=0.01"' in data["runs"]


def test_api_mark_review(client, sweep_dir):
    """POST /api/sweeps/<id>/runs/mark-review stages runs; GET shows status 'review'."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "review", exist_ok=True)
    mod.save_meta("rev1", ["python", "x.py"])
    mod.save_runs("rev1", ["m=1", "m=2"])
    h1 = mod.run_hash("m=1")
    r = client.post("/api/sweeps/rev1/runs/mark-review", json={"hashes": [h1]})
    assert r.status_code == 200
    assert r.json()["marked"] == 1
    detail = client.get("/api/sweeps/rev1").json()
    rows_by_hash = {row["hash"]: row for row in detail["rows"]}
    assert rows_by_hash[h1]["status"] == "review"
    h2 = mod.run_hash("m=2")
    assert rows_by_hash[h2]["status"] == "pending"


def test_api_promote(client, sweep_dir):
    """POST /api/sweeps/<id>/runs/promote moves runs from review back to pending."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "review", exist_ok=True)
    mod.save_meta("rev2", ["python", "x.py"])
    mod.save_runs("rev2", ["n=1"])
    h = mod.run_hash("n=1")
    mod.add_review_lines_by_hashes("rev2", [h])
    r = client.post("/api/sweeps/rev2/runs/promote", json={"hashes": [h]})
    assert r.status_code == 200
    assert r.json()["promoted"] == 1
    detail = client.get("/api/sweeps/rev2").json()
    assert detail["rows"][0]["status"] == "pending"


def test_api_create_sweep_add_as_review(client, sweep_dir):
    """POST /api/sweeps with add_as_review=true; runs appear with status 'review'."""
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "review", exist_ok=True)
    r = client.post("/api/sweeps", json={
        "sweep_id": "rev3",
        "command": ["python", "x.py"],
        "runs": ["v=1", "v=2"],
        "add_as_review": True,
    })
    assert r.status_code == 200
    assert r.json()["added"] == 2
    detail = client.get("/api/sweeps/rev3").json()
    for row in detail["rows"]:
        assert row["status"] == "review"


def test_api_add_runs_add_as_review(client, sweep_dir):
    """POST /api/sweeps/<id>/runs with add_as_review=true; new runs show as 'review'."""
    import sweep as mod
    os.makedirs(sweep_dir / "configs", exist_ok=True)
    os.makedirs(sweep_dir / "review", exist_ok=True)
    mod.save_meta("rev4", ["python", "x.py"])
    mod.save_runs("rev4", ["w=1"])
    r = client.post("/api/sweeps/rev4/runs", json={"runs": ["w=2"], "add_as_review": True})
    assert r.status_code == 200
    assert r.json()["added"] == 1
    detail = client.get("/api/sweeps/rev4").json()
    rows_by_param = {row["param_line"]: row for row in detail["rows"]}
    assert rows_by_param["w=2"]["status"] == "review"
    assert rows_by_param["w=1"]["status"] == "pending"
