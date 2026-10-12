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
            futures = [executor.submit(_worker_process, temp_dir, marker_id, counter_file) for _ in range(num_workers)]
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


def test_atomic_marker_path_traversal_jailed():
    """Verify that path traversal attempts in marker_id are strictly jailed within markers_dir."""
    temp_dir = tempfile.mkdtemp(prefix="lil_marker_sec_")
    markers_dir = os.path.join(temp_dir, "markers")
    try:
        # Malicious traversal attempt
        with atomic_marker("../../escape_attempt", run_dir=temp_dir) as should_run:
            assert should_run

        # Outside directory must NOT contain escape_attempt
        outside_file = os.path.join(temp_dir, "escape_attempt.committed")
        assert not os.path.exists(outside_file)

        # All files must remain strictly jailed inside markers_dir
        assert os.path.exists(markers_dir)
        files = os.listdir(markers_dir)
        assert len(files) > 0
        for f in files:
            assert not f.startswith("..")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_atomic_marker_distinct_identifiers_no_collision():
    """Verify distinct identifiers with punctuation do not collide (SEC-PROTO-02)."""
    temp_dir = tempfile.mkdtemp(prefix="lil_marker_col_")
    try:
        # stripe/charge vs stripe_charge vs stripe:charge
        with atomic_marker("stripe/charge", run_dir=temp_dir) as run1:
            assert run1

        # Different operations must NOT be skipped due to regex collision
        with atomic_marker("stripe_charge", run_dir=temp_dir) as run2:
            assert run2

        with atomic_marker("stripe:charge", run_dir=temp_dir) as run3:
            assert run3

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_atomic_marker_windows_long_path():
    """Verify long marker_id strings (>300 chars) do not crash with Windows MAX_PATH."""
    temp_dir = tempfile.mkdtemp(prefix="lil_marker_long_")
    long_id = "a" * 350
    try:
        with atomic_marker(long_id, run_dir=temp_dir) as should_run:
            assert should_run

        with atomic_marker(long_id, run_dir=temp_dir) as should_run_again:
            assert not should_run_again

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
