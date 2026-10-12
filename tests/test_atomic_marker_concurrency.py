"""tests/test_atomic_marker_concurrency.py — Concurrent race tests for atomic_marker."""

import concurrent.futures
import multiprocessing
import os
import shutil
import tempfile
import time

from orchestrator.decorators import atomic_marker


def _worker_process(run_dir: str, marker_id: str, counter_file: str) -> bool:
    """Worker function executed in separate process."""
    executed = False
    with atomic_marker(marker_id, run_dir=run_dir) as should_run:
        if should_run:
            executed = True
            # Simulate work
            time.sleep(0.05)
            # Increment counter safely
            with open(counter_file, "a", encoding="utf-8") as f:
                f.write(f"{os.getpid()}\n")
    return executed


def test_atomic_marker_multiprocess_mutual_exclusion():
    """Verify that across N concurrent processes, exactly 1 executes the side effect."""
    temp_dir = tempfile.mkdtemp(prefix="lil_marker_mp_")
    counter_file = os.path.join(temp_dir, "executed.txt")
    marker_id = "test_concurrent_payment"

    try:
        num_workers = 8
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = [
                executor.submit(_worker_process, temp_dir, marker_id, counter_file)
                for _ in range(num_workers)
            ]
            results = [f.result(timeout=15.0) for f in futures]

        # Exactly 1 process must have returned True
        assert sum(results) == 1, f"Expected exactly 1 execution, got {sum(results)}: {results}"

        # Counter file must have exactly 1 record
        assert os.path.exists(counter_file)
        with open(counter_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 1, f"Expected 1 record in counter file, got {len(lines)}"

        # Subsequent execution in main process must yield False
        with atomic_marker(marker_id, run_dir=temp_dir) as should_run:
            assert not should_run

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_atomic_marker_multithread_mutual_exclusion():
    """Verify that across N concurrent threads in the same process, exactly 1 executes."""
    temp_dir = tempfile.mkdtemp(prefix="lil_marker_mt_")
    marker_id = "test_concurrent_thread_marker"
    execution_count = 0
    lock = multiprocessing.Lock()

    def _thread_worker() -> bool:
        nonlocal execution_count
        with atomic_marker(marker_id, run_dir=temp_dir) as should_run:
            if should_run:
                time.sleep(0.02)
                with lock:
                    execution_count += 1
                return True
            return False

    try:
        num_threads = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(_thread_worker) for _ in range(num_threads)]
            results = [f.result(timeout=10.0) for f in futures]

        assert sum(results) == 1, f"Expected 1 True from threads, got {sum(results)}"
        assert execution_count == 1, f"Expected execution_count == 1, got {execution_count}"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
