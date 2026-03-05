#!/usr/bin/env python3
"""
Simple file-based sweep manager for distributed execution across machines/GPUs.

Uses ~/.sweepd/configs/<sweep_id>.meta.toml, configs/<sweep_id>.runs.txt, and
~/.sweepd/ran/<sweep_id>.txt (hash+run per line). Supports legacy flat .txt during transition.
"""
import hashlib
import logging
import os
import sys
import subprocess
import time
import tomllib
from typing import IO

logger = logging.getLogger(__name__)

try:
    import tomli_w
except ImportError:
    tomli_w = None

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

RUN_HASH_LEN = 6
RAN_SEP = "\t"


def get_sweep_dir() -> str:
    """
    Returns expanded ~/.sweeps directory path (cross-platform).
    """
    return os.path.expanduser("~/.sweepd")


def get_configs_dir() -> str:
    """
    Returns ~/.sweeps/configs directory path.
    """
    return os.path.join(get_sweep_dir(), "configs")


def get_ran_dir() -> str:
    """
    Returns ~/.sweeps/ran directory path.
    """
    return os.path.join(get_sweep_dir(), "ran")


def get_review_dir() -> str:
    """
    Returns ~/.sweepd/review directory path (for runs staged as 'in review').
    """
    return os.path.join(get_sweep_dir(), "review")


def get_timing_dir() -> str:
    """
    Returns ~/.sweepd/timing directory path (for run start/end timestamps).
    """
    return os.path.join(get_sweep_dir(), "timing")


def split_param_line(param_line: str) -> list[str]:
    """
    Splits a param line on commas that are not inside single or double quotes.
    Returns a list of segments (not stripped).
    """
    segments = []
    current = []
    in_single = False
    in_double = False
    for ch in param_line:
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
        elif ch == "," and not in_single and not in_double:
            segments.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        segments.append("".join(current))
    return segments


def param_line_to_dict(param_line: str) -> dict[str, str]:
    """
    Parses a param line (key=val,key2=val2) into a dict. Used for table UI (one key per column).
    """
    out = {}
    for p in split_param_line(param_line):
        p = p.strip()
        if not p:
            continue
        if "=" in p:
            k, _, v = p.partition("=")
            out[k.strip()] = v.strip()
        else:
            out[p] = ""
    return out


def _canonical_param_line(param_line: str) -> str:
    """
    Normalize param line to key-sorted form for order-independent hashing.
    Parses key=value pairs (comma-separated), sorts by key, rejoins.
    """
    parts = [p.strip() for p in split_param_line(param_line) if p.strip()]
    pairs = []
    for p in parts:
        if "=" in p:
            k, _, v = p.partition("=")
            pairs.append((k.strip(), v.strip()))
        else:
            pairs.append((p, ""))
    pairs.sort(key=lambda x: x[0])
    return ",".join(f"{k}={v}" if v else k for k, v in pairs)


def run_hash(param_line: str) -> str:
    """
    Returns 6-char hex hash of param line (order-independent: canonicalizes before hashing).
    """
    canonical = _canonical_param_line(param_line)
    return hashlib.sha256(canonical.encode()).hexdigest()[:RUN_HASH_LEN]


def _meta_path(sweep_id: str) -> str:
    """Path to configs/<sweep_id>.meta.toml."""
    return os.path.join(get_configs_dir(), f"{sweep_id}.meta.toml")


def _runs_path(sweep_id: str) -> str:
    """Path to configs/<sweep_id>.runs.txt."""
    return os.path.join(get_configs_dir(), f"{sweep_id}.runs.txt")


def _ran_path(sweep_id: str) -> str:
    """Path to ran/<sweep_id>.txt."""
    return os.path.join(get_ran_dir(), f"{sweep_id}.txt")


def _review_path(sweep_id: str) -> str:
    """Path to review/<sweep_id>.txt."""
    return os.path.join(get_review_dir(), f"{sweep_id}.txt")


def _timing_path(sweep_id: str) -> str:
    """Path to timing/<sweep_id>.jsonl."""
    return os.path.join(get_timing_dir(), f"{sweep_id}.jsonl")


def _legacy_sweep_path(sweep_id: str) -> str:
    """Path to legacy ~/.sweeps/<sweep_id>.txt."""
    return os.path.join(get_sweep_dir(), f"{sweep_id}.txt")


def _legacy_ran_path(sweep_id: str) -> str:
    """Path to legacy ~/.sweeps/<sweep_id>_ran.txt."""
    return os.path.join(get_sweep_dir(), f"{sweep_id}_ran.txt")


def _load_meta(sweep_id: str) -> list[str]:
    """Load meta.toml; returns list 'command'. Raises if missing or invalid."""
    path = _meta_path(sweep_id)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    cmd = data.get("command")
    if not cmd or not isinstance(cmd, list):
        raise ValueError(f"meta.toml must contain 'command' (list of strings): {path}")
    return [str(x) for x in cmd]


def _load_runs(sweep_id: str) -> list[str]:
    """Load runs.txt; returns list of param lines (non-empty)."""
    path = _runs_path(sweep_id)
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _load_legacy_sweep(sweep_id: str) -> tuple[list[str], list[str]]:
    """Load legacy sweep file; returns (command, param_lines). command = [python_exec, script_path]."""
    path = _legacy_sweep_path(sweep_id)
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Legacy sweep file must have at least 2 lines: {path}")
    first = lines[0].split()
    if len(first) < 2:
        raise ValueError(f"First line must be 'python_exec script_path': {lines[0]}")
    command = [first[0], first[1]]
    param_lines = lines[1:]
    return command, param_lines


def get_sweep_config(sweep_id: str) -> tuple[list[str], list[str]]:
    """
    Reads sweep config (new layout or legacy), returns (command, param_lines).

    command is a list of strings (e.g. ["python", "test_sweep.py"] or ["uv", "run", "python", "test_sweep.py"]).
    param_lines is a list of comma-separated override strings.

    Tries new layout (configs/<id>.meta.toml + configs/<id>.runs.txt) first, then legacy ~/.sweeps/<id>.txt.
    """
    meta_path = _meta_path(sweep_id)
    if os.path.exists(meta_path):
        command = _load_meta(sweep_id)
        param_lines = _load_runs(sweep_id)
        return command, param_lines
    if os.path.exists(_legacy_sweep_path(sweep_id)):
        return _load_legacy_sweep(sweep_id)
    raise FileNotFoundError(f"Sweep not found: {sweep_id} (no {meta_path} or legacy .txt)")


def get_completed_hashes(sweep_id: str) -> set[str]:
    """
    Returns set of completed run hashes (6-char) for the sweep.

    Reads ran/<sweep_id>.txt (lines: hash\\tparam_line) or legacy _ran.txt (param lines, hashed on read).
    """
    ran_path = _ran_path(sweep_id)
    legacy_ran = _legacy_ran_path(sweep_id)
    if os.path.exists(ran_path):
        with open(ran_path, "r") as f:
            out = set()
            for line in f:
                line = line.rstrip("\n\r")
                if not line:
                    continue
                if RAN_SEP in line:
                    h, _ = line.split(RAN_SEP, 1)
                    out.add(h.strip()[:RUN_HASH_LEN])
                else:
                    out.add(run_hash(line))
            return out
    if os.path.exists(legacy_ran):
        with open(legacy_ran, "r") as f:
            return set(run_hash(line.strip()) for line in f if line.strip())
    return set()


def get_completed_exit_codes(sweep_id: str) -> dict[str, int]:
    """
    Returns dict mapping run hash -> exit code for completed runs that have exit codes recorded.

    Reads ran/<sweep_id>.txt (lines: hash\\tparam_line\\texit_code). Lines without
    a third field are omitted (backwards-compatible with old format).
    """
    ran_path = _ran_path(sweep_id)
    if not os.path.exists(ran_path):
        return {}
    codes: dict[str, int] = {}
    with open(ran_path, "r") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split(RAN_SEP)
            if len(parts) >= 3:
                h = parts[0].strip()[:RUN_HASH_LEN]
                try:
                    codes[h] = int(parts[2].strip())
                except ValueError:
                    pass
    return codes


def record_exit_code(sweep_id: str, run_hash_str: str, exit_code: int) -> None:
    """
    Updates the ran file to record exit code for a completed run.

    Rewrites the matching line from 'hash\\tparam_line' to 'hash\\tparam_line\\texit_code'.
    """
    ran_path = _ran_path(sweep_id)
    if not os.path.exists(ran_path):
        return
    with open(ran_path, "r") as f:
        lines = f.readlines()
    updated = []
    for line in lines:
        stripped = line.rstrip("\n\r")
        if not stripped:
            continue
        parts = stripped.split(RAN_SEP)
        h = parts[0].strip()[:RUN_HASH_LEN]
        if h == run_hash_str and len(parts) == 2:
            updated.append(f"{stripped}{RAN_SEP}{exit_code}")
        else:
            updated.append(stripped)
    with open(ran_path, "w") as f:
        for line in updated:
            f.write(line + "\n")


def record_run_start(sweep_id: str, run_hash_str: str) -> None:
    """Append a timing start entry (JSONL) to timing/<sweep_id>.jsonl."""
    import json
    path = _timing_path(sweep_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry = {"hash": run_hash_str, "event": "start", "time": time.time()}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def record_run_end(sweep_id: str, run_hash_str: str, exit_code: int) -> None:
    """Append a timing end entry (JSONL) to timing/<sweep_id>.jsonl."""
    import json
    path = _timing_path(sweep_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry = {"hash": run_hash_str, "event": "end", "time": time.time(), "exit_code": exit_code}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_run_timings(sweep_id: str) -> dict[str, dict]:
    """Read timing JSONL and return dict mapping hash -> {start, end, exit_code, duration}.

    Uses the last start/end pair for each hash (in case of reruns).
    """
    import json
    path = _timing_path(sweep_id)
    if not os.path.exists(path):
        return {}
    starts: dict[str, float] = {}
    results: dict[str, dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = entry.get("hash", "")
            if entry.get("event") == "start":
                starts[h] = entry["time"]
                results.pop(h, None)
            elif entry.get("event") == "end":
                start_time = starts.get(h)
                end_time = entry["time"]
                duration = (end_time - start_time) if start_time is not None else None
                results[h] = {
                    "start": start_time,
                    "end": end_time,
                    "duration": duration,
                    "exit_code": entry.get("exit_code"),
                }
    return results


def get_review_hashes(sweep_id: str) -> set[str]:
    """
    Returns set of run hashes currently staged as 'in review' for the sweep.

    Reads review/<sweep_id>.txt (lines: hash\\tparam_line).
    """
    review_path = _review_path(sweep_id)
    if not os.path.exists(review_path):
        return set()
    with open(review_path, "r") as f:
        out = set()
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if RAN_SEP in line:
                h, _ = line.split(RAN_SEP, 1)
                out.add(h.strip()[:RUN_HASH_LEN])
            else:
                out.add(run_hash(line))
        return out


def lock_file(file_handle: IO) -> None:
    """
    Lock file handle for exclusive access (cross-platform).
    """
    if HAS_FCNTL:
        fcntl.flock(file_handle, fcntl.LOCK_EX)
    elif HAS_MSVCRT:
        msvcrt.locking(file_handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        pass


def unlock_file(file_handle: IO) -> None:
    """
    Unlock file handle (cross-platform).
    """
    if HAS_FCNTL:
        fcntl.flock(file_handle, fcntl.LOCK_UN)
    elif HAS_MSVCRT:
        try:
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
        except IOError:
            pass
    else:
        pass


def claim_next_run(sweep_id: str) -> str | None:
    """
    Atomically finds next unran run (by hash), appends hash\\tparam_line to ran file, returns param_line or None.
    Skips runs that are currently 'in review'.
    """
    os.makedirs(get_ran_dir(), exist_ok=True)
    os.makedirs(get_configs_dir(), exist_ok=True)

    command, param_lines = get_sweep_config(sweep_id)
    review_hashes = get_review_hashes(sweep_id)
    ran_path = _ran_path(sweep_id)
    mode = "a+b" if HAS_MSVCRT else "a+"
    with open(ran_path, mode) as f:
        lock_file(f)
        try:
            f.seek(0)
            if HAS_MSVCRT:
                content = f.read().decode("utf-8")
                completed = set()
                for line in content.splitlines():
                    if not line:
                        continue
                    if RAN_SEP in line:
                        h, _ = line.split(RAN_SEP, 1)
                        completed.add(h.strip()[:RUN_HASH_LEN])
                    else:
                        completed.add(run_hash(line))
            else:
                completed = set()
                for line in f.readlines():
                    line = line.rstrip("\n\r")
                    if not line:
                        continue
                    if RAN_SEP in line:
                        h, _ = line.split(RAN_SEP, 1)
                        completed.add(h.strip()[:RUN_HASH_LEN])
                    else:
                        completed.add(run_hash(line))

            for param_line in param_lines:
                h = run_hash(param_line)
                if h not in completed and h not in review_hashes:
                    line_to_append = f"{h}{RAN_SEP}{param_line}\n"
                    f.seek(0, 2)
                    if HAS_MSVCRT:
                        f.write(line_to_append.encode("utf-8"))
                    else:
                        f.write(line_to_append)
                    f.flush()
                    return param_line
        finally:
            unlock_file(f)
    return None


def execute_run(command: list[str], param_line: str) -> int:
    """
    Runs command + Hydra overrides. command is list of args; param_line is comma-separated overrides.
    Returns exit code of the subprocess.
    """
    params = [p.strip() for p in split_param_line(param_line) if p.strip()]
    cmd = command + params
    logger.info("Executing: %s", " ".join(cmd))
    result = subprocess.run(cmd, stdout=None, stderr=None)
    return result.returncode


def sweep_run(sweep_id: str) -> None:
    """
    Main sweep loop: claim and execute runs until all done. Logs Run i/N progress.
    """
    try:
        command, param_lines = get_sweep_config(sweep_id)
    except Exception as e:
        logger.error("Error reading sweep config: %s", e)
        sys.exit(1)

    completed_hashes = get_completed_hashes(sweep_id)
    total = len(param_lines)
    completed_count = len(completed_hashes)

    logger.info("Sweep ID: %s", sweep_id)
    logger.info("Command: %s", " ".join(command))
    logger.info("Total runs: %d", total)
    logger.info("Already completed: %d", completed_count)

    run_index = 0
    while True:
        param_line = claim_next_run(sweep_id)
        if param_line is None:
            logger.info("All runs completed!")
            break
        run_index += 1
        h = run_hash(param_line)
        logger.info("Run %d/%d [%s] %s", run_index, total, h, param_line)

        record_run_start(sweep_id, h)
        exit_code = execute_run(command, param_line)
        record_run_end(sweep_id, h, exit_code)
        record_exit_code(sweep_id, h, exit_code)
        if exit_code == 0:
            logger.info("Run completed successfully")
        else:
            logger.warning("Run failed with exit code %d, continuing to next run", exit_code)


def get_default_command() -> list[str] | None:
    """
    Returns default command list from ~/.sweeps/sweep_config.toml or config/sweep_defaults.toml if present.
    Otherwise returns None (caller must provide command).
    """
    for path in [
        os.path.join(get_sweep_dir(), "sweep_config.toml"),
        os.path.join(os.getcwd(), "config", "sweep_defaults.toml"),
    ]:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                data = tomllib.load(f)
            cmd = data.get("command")
            if cmd and isinstance(cmd, list):
                return [str(x) for x in cmd]
    return None


def save_meta(sweep_id: str, command: list[str]) -> None:
    """Writes configs/<sweep_id>.meta.toml with the given command list."""
    if tomli_w is None:
        raise RuntimeError("tomli-w is required; install with: uv add tomli-w")
    os.makedirs(get_configs_dir(), exist_ok=True)
    with open(_meta_path(sweep_id), "wb") as f:
        tomli_w.dump({"command": command}, f)


def get_runs(sweep_id: str) -> list[str]:
    """Returns list of param lines for the sweep (from runs.txt or legacy)."""
    if os.path.exists(_meta_path(sweep_id)):
        return _load_runs(sweep_id)
    if os.path.exists(_legacy_sweep_path(sweep_id)):
        return _load_legacy_sweep(sweep_id)[1]
    return []


def save_runs(sweep_id: str, param_lines: list[str]) -> None:
    """Overwrites configs/<sweep_id>.runs.txt with the given param lines."""
    os.makedirs(get_configs_dir(), exist_ok=True)
    with open(_runs_path(sweep_id), "w") as f:
        for line in param_lines:
            f.write(line + "\n")


def append_runs(sweep_id: str, param_lines: list[str]) -> None:
    """Appends param lines to configs/<sweep_id>.runs.txt."""
    os.makedirs(get_configs_dir(), exist_ok=True)
    with open(_runs_path(sweep_id), "a") as f:
        for line in param_lines:
            f.write(line + "\n")


def remove_ran_lines_by_hashes(sweep_id: str, hashes_to_remove: set[str]) -> None:
    """
    Removes from ran/<sweep_id>.txt any line whose hash (first field) is in hashes_to_remove.
    """
    ran_path = _ran_path(sweep_id)
    if not os.path.exists(ran_path):
        return
    with open(ran_path, "r") as f:
        lines = f.readlines()
    kept = []
    for line in lines:
        line = line.rstrip("\n\r")
        if not line:
            continue
        if RAN_SEP in line:
            h = line.split(RAN_SEP, 1)[0].strip()[:RUN_HASH_LEN]
        else:
            h = run_hash(line)
        if h not in hashes_to_remove:
            kept.append(line)
    with open(ran_path, "w") as f:
        for line in kept:
            f.write(line + "\n")


def add_ran_lines_by_hashes(sweep_id: str, hashes_to_add: set[str]) -> None:
    """
    Adds runs to ran/<sweep_id>.txt for the given hashes. Looks up param_line from runs.txt.
    Only adds if hash is not already in ran file. Uses file locking for atomicity.
    """
    if not hashes_to_add:
        return
    try:
        _, param_lines = get_sweep_config(sweep_id)
    except FileNotFoundError:
        return
    hash_to_param = {run_hash(p): p for p in param_lines}
    os.makedirs(get_ran_dir(), exist_ok=True)
    ran_path = _ran_path(sweep_id)
    existing_hashes = get_completed_hashes(sweep_id)
    to_add = []
    for h in hashes_to_add:
        h = h.strip()[:RUN_HASH_LEN]
        if h in existing_hashes:
            continue
        if h in hash_to_param:
            to_add.append(f"{h}{RAN_SEP}{hash_to_param[h]}\n")
    if not to_add:
        return
    mode = "a+b" if HAS_MSVCRT else "a+"
    with open(ran_path, mode) as f:
        lock_file(f)
        try:
            f.seek(0, 2)
            for line in to_add:
                if HAS_MSVCRT:
                    f.write(line.encode("utf-8"))
                else:
                    f.write(line)
            f.flush()
        finally:
            unlock_file(f)


def add_review_lines_by_hashes(sweep_id: str, hashes_to_review: list[str] | set[str]) -> None:
    """
    Moves runs to the review file for the given hashes.
    These runs will be invisible to claim_next_run until promoted.
    Only adds hashes not already in the review file. Uses file locking.
    """
    if not hashes_to_review:
        return
    try:
        _, param_lines = get_sweep_config(sweep_id)
    except FileNotFoundError:
        return
    hash_to_param = {run_hash(p): p for p in param_lines}
    os.makedirs(get_review_dir(), exist_ok=True)
    review_path = _review_path(sweep_id)
    existing_review = get_review_hashes(sweep_id)
    to_add = []
    for h in hashes_to_review:
        h = h.strip()[:RUN_HASH_LEN]
        if h in existing_review:
            continue
        if h in hash_to_param:
            to_add.append(f"{h}{RAN_SEP}{hash_to_param[h]}\n")
    if not to_add:
        return
    mode = "a+b" if HAS_MSVCRT else "a+"
    with open(review_path, mode) as f:
        lock_file(f)
        try:
            f.seek(0, 2)
            for line in to_add:
                if HAS_MSVCRT:
                    f.write(line.encode("utf-8"))
                else:
                    f.write(line)
            f.flush()
        finally:
            unlock_file(f)


def promote_from_review(sweep_id: str, hashes_to_promote: set[str]) -> None:
    """
    Promotes runs from 'in review' to 'pending' by removing them from the review file.
    After this, claim_next_run will pick them up.
    """
    review_path = _review_path(sweep_id)
    if not os.path.exists(review_path):
        return
    hashes_to_remove = {h.strip()[:RUN_HASH_LEN] for h in hashes_to_promote}
    with open(review_path, "r") as f:
        lines = f.readlines()
    kept = []
    for line in lines:
        stripped = line.rstrip("\n\r")
        if not stripped:
            continue
        if RAN_SEP in stripped:
            h = stripped.split(RAN_SEP, 1)[0].strip()[:RUN_HASH_LEN]
        else:
            h = run_hash(stripped)
        if h not in hashes_to_remove:
            kept.append(stripped)
    with open(review_path, "w") as f:
        for line in kept:
            f.write(line + "\n")


def append_runs_as_review(sweep_id: str, param_lines: list[str]) -> None:
    """
    Appends param_lines to runs.txt AND immediately stages them in the review file.
    These runs will not be claimed until promoted via promote_from_review().
    """
    append_runs(sweep_id, param_lines)
    hashes = [run_hash(p) for p in param_lines]
    add_review_lines_by_hashes(sweep_id, hashes)


def clone_sweep(source_id: str, new_id: str) -> None:
    """Clone a sweep: copy meta + runs to a new sweep ID (no ran/review/timing history)."""
    command, param_lines = get_sweep_config(source_id)
    save_meta(new_id, command)
    save_runs(new_id, param_lines)


def list_sweep_ids() -> list[str]:
    """
    Returns list of sweep IDs from configs/*.meta.toml and legacy ~/.sweeps/*.txt (excluding _ran.txt).
    """
    configs_dir = get_configs_dir()
    sweep_dir = get_sweep_dir()
    ids = set()
    if os.path.isdir(configs_dir):
        for name in os.listdir(configs_dir):
            if name.endswith(".meta.toml"):
                ids.add(name[:- len(".meta.toml")])
    if os.path.isdir(sweep_dir):
        for name in os.listdir(sweep_dir):
            if name.endswith("_ran.txt"):
                continue
            if name.endswith(".txt"):
                ids.add(name[:- len(".txt")])
    return sorted(ids)
