#!/usr/bin/env python3
"""
FastAPI server for sweep manager: API + static web UI.
Run with: uv run python sweep_web.py  or  uvicorn sweep_web:app --host 0.0.0.0 --port 8765
"""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sweep import (
    add_ran_lines_by_hashes,
    add_review_lines_by_hashes,
    append_runs,
    append_runs_as_review,
    get_completed_hashes,
    get_default_command,
    get_review_hashes,
    get_runs,
    get_sweep_config,
    list_sweep_ids,
    param_line_to_dict,
    promote_from_review,
    remove_ran_lines_by_hashes,
    run_hash,
    save_meta,
    save_runs,
    _meta_path,
    _ran_path,
    _review_path,
    _runs_path,
)
from sweep_cli import _expand_grid as expand_grid, _dedup_against_existing as dedup_against_existing

app = FastAPI(title="Sweep Manager")

WEB_DIR = Path(__file__).resolve().parent / "web"
if not WEB_DIR.is_dir():
    WEB_DIR.mkdir(parents=True, exist_ok=True)


class CreateSweepBody(BaseModel):
    sweep_id: str
    command: list[str] | None = None
    runs: list[str] | None = None
    base: str = ""
    grid: list[str] | None = None
    add_as_review: bool = False


class AddRunsBody(BaseModel):
    runs: list[str] | None = None
    base: str = ""
    grid: list[str] | None = None
    add_as_review: bool = False


class MarkRerunBody(BaseModel):
    hashes: list[str] | None = None
    indices: list[int] | None = None


class MarkReviewBody(BaseModel):
    hashes: list[str] | None = None
    indices: list[int] | None = None


class RemoveRunsBody(BaseModel):
    indices: list[int]


def _runs_to_table_rows(param_lines, completed_hashes, review_hashes=None):
    """Returns list of dicts: { index, hash, status, ...key: value } and set of all keys.

    status is one of: 'ran', 'review', 'pending'.
    """
    if review_hashes is None:
        review_hashes = set()
    all_keys = set()
    rows = []
    for i, line in enumerate(param_lines):
        h = run_hash(line)
        d = param_line_to_dict(line)
        all_keys.update(d.keys())
        if h in completed_hashes:
            status = "ran"
        elif h in review_hashes:
            status = "review"
        else:
            status = "pending"
        rows.append({
            "index": i,
            "hash": h,
            "status": status,
            "param_line": line,
            **d,
        })
    return rows, sorted(all_keys)


@app.get("/api/sweeps")
def api_list_sweeps():
    """List sweep IDs."""
    return {"sweep_ids": list_sweep_ids()}


@app.get("/api/default-command")
def api_default_command():
    """Return default command for new sweeps, if configured."""
    cmd = get_default_command()
    if not cmd:
        raise HTTPException(status_code=404, detail="No default command configured")
    return {"command": cmd}


@app.get("/api/sweeps/{sweep_id}")
def api_get_sweep(sweep_id: str):
    """Get sweep meta, runs as table rows (one key per column), and column keys."""
    try:
        command, param_lines = get_sweep_config(sweep_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Sweep not found: {sweep_id}")
    completed = get_completed_hashes(sweep_id)
    review = get_review_hashes(sweep_id)
    rows, columns = _runs_to_table_rows(param_lines, completed, review)
    return {
        "sweep_id": sweep_id,
        "command": command,
        "runs": param_lines,
        "rows": rows,
        "columns": columns,
        "completed_count": len(completed),
        "review_count": len(review),
        "total_count": len(param_lines),
    }


@app.post("/api/sweeps")
def api_create_sweep(body: CreateSweepBody):
    """Create a new sweep. Uses grid or runs; deduplicates."""
    cmd = body.command or get_default_command()
    if not cmd:
        raise HTTPException(status_code=400, detail="No command provided and no default config.")
    if body.runs and body.grid:
        raise HTTPException(status_code=400, detail="Use either runs or grid, not both.")
    if body.runs:
        new_lines = body.runs
    elif body.grid:
        new_lines = expand_grid(body.base or "", body.grid)
    else:
        raise HTTPException(status_code=400, detail="Provide runs or grid.")
    if not new_lines:
        raise HTTPException(status_code=400, detail="No runs generated.")
    try:
        existing = get_runs(body.sweep_id)
    except FileNotFoundError:
        existing = []
    to_add, skipped = dedup_against_existing(existing, new_lines)
    save_meta(body.sweep_id, cmd)
    save_runs(body.sweep_id, existing + to_add)
    if body.add_as_review and to_add:
        add_review_lines_by_hashes(body.sweep_id, [run_hash(p) for p in to_add])
    return {"added": len(to_add), "skipped": skipped}


@app.post("/api/sweeps/{sweep_id}/runs")
def api_add_runs(sweep_id: str, body: AddRunsBody):
    """Add runs to an existing sweep. Deduplicates."""
    if not os.path.exists(_meta_path(sweep_id)):
        raise HTTPException(status_code=404, detail=f"Sweep not found: {sweep_id}")
    if body.runs and body.grid:
        raise HTTPException(status_code=400, detail="Use either runs or grid, not both.")
    if body.runs:
        new_lines = body.runs
    elif body.grid:
        new_lines = expand_grid(body.base or "", body.grid)
    else:
        raise HTTPException(status_code=400, detail="Provide runs or grid.")
    existing = get_runs(sweep_id)
    to_add, skipped = dedup_against_existing(existing, new_lines)
    if not to_add:
        return {"added": 0, "skipped": skipped}
    if body.add_as_review:
        append_runs_as_review(sweep_id, to_add)
    else:
        append_runs(sweep_id, to_add)
    return {"added": len(to_add), "skipped": skipped}


@app.post("/api/sweeps/{sweep_id}/runs/mark-rerun")
def api_mark_rerun(sweep_id: str, body: MarkRerunBody):
    """Remove run(s) from ran file so they will be re-run."""
    to_remove = set(body.hashes or [])
    if body.indices:
        try:
            _, param_lines = get_sweep_config(sweep_id)
        except FileNotFoundError:
            param_lines = []
        for i in body.indices:
            if 0 <= i < len(param_lines):
                to_remove.add(run_hash(param_lines[i]))
    if not to_remove:
        raise HTTPException(status_code=400, detail="Provide hashes or indices.")
    remove_ran_lines_by_hashes(sweep_id, to_remove)
    return {"marked": len(to_remove)}


@app.post("/api/sweeps/{sweep_id}/runs/mark-ran")
def api_mark_ran(sweep_id: str, body: MarkRerunBody):
    """Add run(s) to ran file to mark them as completed."""
    to_add = set(body.hashes or [])
    if body.indices:
        try:
            _, param_lines = get_sweep_config(sweep_id)
        except FileNotFoundError:
            param_lines = []
        for i in body.indices:
            if 0 <= i < len(param_lines):
                to_add.add(run_hash(param_lines[i]))
    if not to_add:
        raise HTTPException(status_code=400, detail="Provide hashes or indices.")
    add_ran_lines_by_hashes(sweep_id, to_add)
    return {"marked": len(to_add)}


@app.post("/api/sweeps/{sweep_id}/runs/mark-review")
def api_mark_review(sweep_id: str, body: MarkReviewBody):
    """Stage run(s) as 'in review' so they are not claimed by runners."""
    to_review = set(body.hashes or [])
    if body.indices:
        try:
            _, param_lines = get_sweep_config(sweep_id)
        except FileNotFoundError:
            param_lines = []
        for i in body.indices:
            if 0 <= i < len(param_lines):
                to_review.add(run_hash(param_lines[i]))
    if not to_review:
        raise HTTPException(status_code=400, detail="Provide hashes or indices.")
    add_review_lines_by_hashes(sweep_id, to_review)
    return {"marked": len(to_review)}


@app.post("/api/sweeps/{sweep_id}/runs/promote")
def api_promote_from_review(sweep_id: str, body: MarkReviewBody):
    """Promote run(s) from 'in review' to 'pending' so runners can claim them."""
    to_promote = set(body.hashes or [])
    if body.indices:
        try:
            _, param_lines = get_sweep_config(sweep_id)
        except FileNotFoundError:
            param_lines = []
        for i in body.indices:
            if 0 <= i < len(param_lines):
                to_promote.add(run_hash(param_lines[i]))
    if not to_promote:
        raise HTTPException(status_code=400, detail="Provide hashes or indices.")
    promote_from_review(sweep_id, to_promote)
    return {"promoted": len(to_promote)}


@app.delete("/api/sweeps/{sweep_id}")
def api_delete_sweep(sweep_id: str):
    """Delete sweep config, ran file, and review file."""
    removed = []
    for path in [_meta_path(sweep_id), _runs_path(sweep_id), _ran_path(sweep_id), _review_path(sweep_id)]:
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    return {"removed": removed}


@app.post("/api/sweeps/{sweep_id}/runs/remove")
def api_remove_runs(sweep_id: str, body: RemoveRunsBody):
    """Remove runs at given indices from runs.txt and from ran file."""
    if not os.path.exists(_meta_path(sweep_id)):
        raise HTTPException(status_code=404, detail=f"Sweep not found: {sweep_id}")
    param_lines = get_runs(sweep_id)
    indices_set = set(body.indices)
    kept = [p for i, p in enumerate(param_lines) if i not in indices_set]
    hashes_to_remove = {run_hash(param_lines[i]) for i in indices_set if 0 <= i < len(param_lines)}
    save_runs(sweep_id, kept)
    remove_ran_lines_by_hashes(sweep_id, hashes_to_remove)
    return {"removed": len(indices_set)}


# Static and index
@app.get("/")
def index():
    """Serve the single-page web UI."""
    index_path = WEB_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return {"message": "Web UI not found. Create web/index.html."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
