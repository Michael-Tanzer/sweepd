"""Shared test fixtures for sweep tests."""
import os
import pytest


@pytest.fixture
def sweep_dir(tmp_path):
    """Override path helpers to use tmp_path for the duration of the test."""
    import sweep as mod
    import sweep.core as core
    base = str(tmp_path)

    # Public attrs: patched on both sweep package and sweep.core
    public_attrs = [
        "get_sweep_dir", "get_configs_dir", "get_ran_dir", "get_review_dir",
        "get_timing_dir",
    ]
    # Private attrs: patched only on sweep.core
    private_attrs = [
        "_meta_path", "_runs_path", "_ran_path", "_review_path",
        "_timing_path", "_legacy_sweep_path", "_legacy_ran_path",
    ]

    orig_mod = {k: getattr(mod, k) for k in public_attrs}
    orig_core = {k: getattr(core, k) for k in public_attrs + private_attrs}

    patches = {
        "get_sweep_dir": lambda: base,
        "get_configs_dir": lambda: os.path.join(base, "configs"),
        "get_ran_dir": lambda: os.path.join(base, "ran"),
        "get_review_dir": lambda: os.path.join(base, "review"),
        "get_timing_dir": lambda: os.path.join(base, "timing"),
        "_meta_path": lambda sid: os.path.join(base, "configs", f"{sid}.meta.toml"),
        "_runs_path": lambda sid: os.path.join(base, "configs", f"{sid}.runs.txt"),
        "_ran_path": lambda sid: os.path.join(base, "ran", f"{sid}.txt"),
        "_review_path": lambda sid: os.path.join(base, "review", f"{sid}.txt"),
        "_timing_path": lambda sid: os.path.join(base, "timing", f"{sid}.jsonl"),
        "_legacy_sweep_path": lambda sid: os.path.join(base, f"{sid}.txt"),
        "_legacy_ran_path": lambda sid: os.path.join(base, f"{sid}_ran.txt"),
    }
    for k, v in patches.items():
        setattr(core, k, v)
        if k in public_attrs:
            setattr(mod, k, v)
    try:
        yield tmp_path
    finally:
        for k, v in orig_mod.items():
            setattr(mod, k, v)
        for k, v in orig_core.items():
            setattr(core, k, v)
