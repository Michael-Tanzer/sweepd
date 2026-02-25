#!/usr/bin/env python3
"""
GPU checking and daemon loop for sweep manager.
"""
import subprocess
import sys
import time

from sweep.core import (
    claim_next_run,
    execute_run,
    get_sweep_config,
    list_sweep_ids,
    run_hash,
)


def is_gpu_free(gpu_id=0, vram_threshold_mb=1000):
    """
    Check whether a GPU has less than vram_threshold_mb MB of VRAM in use.

    Returns True if the GPU appears free, False if busy, None if nvidia-smi is unavailable
    or the check cannot be performed (non-GPU machine or driver not installed).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", f"-i={gpu_id}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        used_mb = int(result.stdout.strip())
        return used_mb < vram_threshold_mb
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        return None


def sweep_daemon(sweep_ids, interval=30, gpu_id=0, cpu_mode=False, vram_threshold_mb=1000):
    """
    Daemon loop: run all pending work across given sweeps (or all sweeps if empty),
    then poll every `interval` seconds for new work.

    gpu_id: which GPU index to check VRAM usage on. Ignored if cpu_mode=True.
    cpu_mode: if True, skip GPU check entirely.
    vram_threshold_mb: VRAM usage below this (MB) means GPU is considered free.
    """
    specific_ids = list(sweep_ids) if sweep_ids else None

    if not cpu_mode:
        gpu_status = is_gpu_free(gpu_id, vram_threshold_mb)
        if gpu_status is None:
            print("nvidia-smi not available or GPU not detected; running without GPU check.")
            cpu_mode = True
        else:
            print(f"GPU {gpu_id} check enabled (threshold: {vram_threshold_mb} MB VRAM).")

    print(f"Daemon started. Poll interval: {interval}s.")
    if specific_ids:
        print(f"Watching sweep(s): {', '.join(specific_ids)}")
    else:
        print("Watching all sweeps.")
    sys.stdout.flush()

    while True:
        ids = specific_ids if specific_ids is not None else list_sweep_ids()
        found_work = False

        for sweep_id in ids:
            try:
                command, _ = get_sweep_config(sweep_id)
            except FileNotFoundError:
                continue

            while True:
                if not cpu_mode:
                    free = is_gpu_free(gpu_id, vram_threshold_mb)
                    if free is False:
                        print(f"GPU {gpu_id} busy (>= {vram_threshold_mb} MB VRAM used). Waiting {interval}s...")
                        sys.stdout.flush()
                        time.sleep(interval)
                        continue
                    # free is None means check failed; proceed anyway

                param_line = claim_next_run(sweep_id)
                if param_line is None:
                    break  # no more pending work in this sweep

                found_work = True
                h = run_hash(param_line)
                print(f"\n[daemon] Sweep '{sweep_id}' [{h}] {param_line}")
                sys.stdout.flush()

                exit_code = execute_run(command, param_line)
                if exit_code == 0:
                    print("[daemon] Run completed successfully.")
                else:
                    print(f"[daemon] Run failed with exit code {exit_code}.")
                sys.stdout.flush()

        if not found_work:
            print(f"No pending work. Sleeping {interval}s...")
            sys.stdout.flush()
            time.sleep(interval)
