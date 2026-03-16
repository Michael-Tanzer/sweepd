#!/usr/bin/env python3
"""
GPU checking and daemon loop for sweep manager.
"""
import logging
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

from sweep.core import (
    _log_path,
    claim_next_run,
    execute_run,
    get_sweep_config,
    list_sweep_ids,
    record_exit_code,
    record_run_end,
    record_run_start,
    run_hash,
    start_process,
)


def is_gpu_free(gpu_id: int = 0, vram_threshold_mb: int = 1000) -> bool | None:
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


def sweep_daemon(sweep_ids: list[str], interval: int = 30, gpu_id: int = 0, cpu_mode: bool = False, vram_threshold_mb: int = 1000) -> None:
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
            logger.info("nvidia-smi not available or GPU not detected; running without GPU check.")
            cpu_mode = True
        else:
            logger.info("GPU %d check enabled (threshold: %d MB VRAM).", gpu_id, vram_threshold_mb)

    logger.info("Daemon started. Poll interval: %ds.", interval)
    if specific_ids:
        logger.info("Watching sweep(s): %s", ", ".join(specific_ids))
    else:
        logger.info("Watching all sweeps.")

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
                        logger.info("GPU %d busy (>= %d MB VRAM used). Waiting %ds...", gpu_id, vram_threshold_mb, interval)
                        time.sleep(interval)
                        continue

                param_line = claim_next_run(sweep_id)
                if param_line is None:
                    break

                found_work = True
                h = run_hash(param_line)
                logger.info("[daemon] Sweep '%s' [%s] %s", sweep_id, h, param_line)

                log_path = _log_path(sweep_id, h)
                proc = start_process(command, param_line, log_path=log_path)
                record_run_start(sweep_id, h, pid=proc.pid)
                exit_code = proc.wait()
                if hasattr(proc, "_tee_thread"):
                    proc._tee_thread.join(timeout=5)
                record_run_end(sweep_id, h, exit_code)
                record_exit_code(sweep_id, h, exit_code)
                if exit_code == 0:
                    logger.info("[daemon] Run completed successfully.")
                else:
                    logger.warning("[daemon] Run failed with exit code %d.", exit_code)

        if not found_work:
            logger.debug("No pending work. Sleeping %ds...", interval)
            time.sleep(interval)
