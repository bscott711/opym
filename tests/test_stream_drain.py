# Ruff style: Compliant
"""Tests for the write-behind RAM-disk-to-GPFS drain pool
(`opym.stream.drain`), isolated from the receiver/ZMQ so failure-path and
retention-policy behavior can be exercised directly and deterministically.
"""

from __future__ import annotations

import time

from opym.stream.drain import DrainJob, DrainPool


def _wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _make_tree(root, files):
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


def test_drain_copies_verifies_and_makes_destination_visible(tmp_path):
    stage_dir = tmp_path / "stage" / "sample_C0.ome.zarr"
    dest_dir = tmp_path / "raw" / "sample_C0.ome.zarr"
    _make_tree(stage_dir, {"a.bin": b"hello", "sub/b.bin": b"world!!"})

    pool = DrainPool(num_workers=1, retention_s=9999, high_water_bytes=10**12)
    pool.start()
    try:
        pool.enqueue(DrainJob("sess-1", [(stage_dir, dest_dir)]))
        assert _wait_until(lambda: dest_dir.exists())
        assert (dest_dir / "a.bin").read_bytes() == b"hello"
        assert (dest_dir / "sub" / "b.bin").read_bytes() == b"world!!"
        # Staging copy must survive drain (retention_s huge) -- the whole
        # point is the GPU pipeline can keep reading it from RAM.
        assert stage_dir.exists()
        assert (stage_dir / "a.bin").read_bytes() == b"hello"
    finally:
        pool.stop()


def test_drain_covers_multiple_items_and_is_all_or_nothing_visible(tmp_path):
    """A session's channel stores + decon_stage are separate top-level
    items (see `DrainJob`'s docstring) -- both must land, and neither
    becomes visible at its destination until every item in the job has been
    copied and verified."""
    stage_c0 = tmp_path / "stage" / "sample_C0.ome.zarr"
    stage_c1 = tmp_path / "stage" / "sample_C1.ome.zarr"
    dest_c0 = tmp_path / "raw" / "sample_C0.ome.zarr"
    dest_c1 = tmp_path / "raw" / "sample_C1.ome.zarr"
    _make_tree(stage_c0, {"a.bin": b"c0-data"})
    _make_tree(stage_c1, {"a.bin": b"c1-data"})

    pool = DrainPool(num_workers=1, retention_s=9999, high_water_bytes=10**12)
    pool.start()
    try:
        pool.enqueue(
            DrainJob("sess-multi", [(stage_c0, dest_c0), (stage_c1, dest_c1)])
        )
        assert _wait_until(lambda: dest_c0.exists() and dest_c1.exists())
        assert (dest_c0 / "a.bin").read_bytes() == b"c0-data"
        assert (dest_c1 / "a.bin").read_bytes() == b"c1-data"
    finally:
        pool.stop()


def test_drain_failure_leaves_staging_copies_untouched(tmp_path, monkeypatch):
    stage_dir = tmp_path / "stage" / "sample_C0.ome.zarr"
    dest_dir = tmp_path / "raw" / "sample_C0.ome.zarr"
    _make_tree(stage_dir, {"a.bin": b"hello"})

    import opym.stream.drain as drain_mod

    def _boom(*args, **kwargs):
        raise OSError("simulated GPFS write failure")

    monkeypatch.setattr(drain_mod.shutil, "copytree", _boom)

    pool = DrainPool(num_workers=1, retention_s=9999, high_water_bytes=10**12)
    pool.start()
    try:
        pool.enqueue(DrainJob("sess-fail", [(stage_dir, dest_dir)]))
        # Give the worker a moment to process (and fail) the job.
        _wait_until(lambda: pool.pending_count == 0)
        time.sleep(0.1)
        assert not dest_dir.exists()
        assert (stage_dir / "a.bin").read_bytes() == b"hello"
    finally:
        pool.stop()


def test_high_water_mark_evicts_oldest_drained_session_first(tmp_path):
    stage_root = tmp_path / "stage"
    raw_root = tmp_path / "raw"
    payload = b"x" * 1000

    stage_a = stage_root / "a"
    stage_b = stage_root / "b"
    _make_tree(stage_a, {"f.bin": payload})
    _make_tree(stage_b, {"f.bin": payload})

    # High-water mark smaller than both sessions combined (2000 bytes) but
    # large enough for one (1000) -- forces eviction of whichever drained
    # first once the second lands.
    pool = DrainPool(num_workers=1, retention_s=9999, high_water_bytes=1500)
    pool.start()
    try:
        pool.enqueue(DrainJob("sess-a", [(stage_a, raw_root / "a")]))
        assert _wait_until(lambda: (raw_root / "a").exists())

        pool.enqueue(DrainJob("sess-b", [(stage_b, raw_root / "b")]))
        assert _wait_until(lambda: (raw_root / "b").exists())

        # sess-a's staging copy should be evicted to stay under the
        # high-water mark; its GPFS copy (already drained) is untouched.
        assert _wait_until(lambda: not stage_a.exists())
        assert (raw_root / "a" / "f.bin").read_bytes() == payload
        assert (raw_root / "b" / "f.bin").read_bytes() == payload
    finally:
        pool.stop()


def test_drained_session_evicted_after_retention_expires(tmp_path):
    stage_dir = tmp_path / "stage" / "sample"
    dest_dir = tmp_path / "raw" / "sample"
    _make_tree(stage_dir, {"a.bin": b"hello"})

    pool = DrainPool(num_workers=1, retention_s=0.05, high_water_bytes=10**12)
    pool.start()
    try:
        pool.enqueue(DrainJob("sess-1", [(stage_dir, dest_dir)]))
        assert _wait_until(lambda: dest_dir.exists())
        assert _wait_until(lambda: not stage_dir.exists(), timeout=3.0)
        # GPFS copy must be unaffected by staging-side eviction.
        assert (dest_dir / "a.bin").read_bytes() == b"hello"
    finally:
        pool.stop()


def test_concurrent_sweeps_evict_each_session_exactly_once(
    tmp_path, monkeypatch, caplog
):
    """Every idle worker sweeps retention about once a second. Eviction used to
    run after the lock was released, so several workers rmtree'd the same
    session at once and all but one logged FileNotFoundError (seen in
    production on 2026-09-23). A slowed-down rmtree makes those sweeps overlap
    deterministically."""
    import logging
    import shutil

    real_rmtree = shutil.rmtree
    evicted = []

    def slow_rmtree(path, *args, **kwargs):
        evicted.append(str(path))
        time.sleep(0.3)
        real_rmtree(path, *args, **kwargs)

    stage_dir = tmp_path / "stage" / "sample"
    dest_dir = tmp_path / "raw" / "sample"
    _make_tree(stage_dir, {"a.bin": b"hello"})

    pool = DrainPool(num_workers=4, retention_s=0.05, high_water_bytes=10**12)
    pool.start()
    try:
        pool.enqueue(DrainJob("sess-1", [(stage_dir, dest_dir)]))
        assert _wait_until(lambda: dest_dir.exists())
        monkeypatch.setattr(shutil, "rmtree", slow_rmtree)
        with caplog.at_level(logging.ERROR, logger="opym.stream.drain"):
            assert _wait_until(lambda: not stage_dir.exists(), timeout=5.0)
            time.sleep(2.5)  # let every other worker complete a sweep
    finally:
        pool.stop()

    assert evicted.count(str(stage_dir)) == 1
    assert not [r for r in caplog.records if "Failed to evict" in r.getMessage()]



def test_retention_survives_a_restart(tmp_path):
    """A drained session's RAM copy must still be evicted after the process
    that drained it restarts (2026-09-24: one restart orphaned 110 GB)."""
    stage_dir = tmp_path / "stage" / "sample"
    dest_dir = tmp_path / "raw" / "sample"
    manifests = tmp_path / "stage" / ".drain_manifests"
    _make_tree(stage_dir, {"a.bin": b"hello"})

    first = DrainPool(num_workers=1, retention_s=9999, high_water_bytes=10**12,
                      manifest_dir=manifests)
    first.start()
    try:
        first.enqueue(DrainJob("sess-1", [(stage_dir, dest_dir)]))
        assert _wait_until(lambda: (manifests / "sess-1.json").exists())
    finally:
        first.stop()
    assert stage_dir.exists()  # retention not yet expired when it "died"

    restarted = DrainPool(num_workers=1, retention_s=0.05, high_water_bytes=10**12,
                          manifest_dir=manifests)
    restarted.start()
    try:
        assert _wait_until(lambda: not stage_dir.exists(), timeout=5.0)
        assert _wait_until(lambda: not (manifests / "sess-1.json").exists())
    finally:
        restarted.stop()
    assert (dest_dir / "a.bin").read_bytes() == b"hello"
