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
        pool.enqueue(DrainJob("sess-multi", [(stage_c0, dest_c0), (stage_c1, dest_c1)]))
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

    first = DrainPool(
        num_workers=1, retention_s=9999, high_water_bytes=10**12, manifest_dir=manifests
    )
    first.start()
    try:
        first.enqueue(DrainJob("sess-1", [(stage_dir, dest_dir)]))
        assert _wait_until(lambda: (manifests / "sess-1.json").exists())
    finally:
        first.stop()
    assert stage_dir.exists()  # retention not yet expired when it "died"

    restarted = DrainPool(
        num_workers=1, retention_s=0.05, high_water_bytes=10**12, manifest_dir=manifests
    )
    restarted.start()
    try:
        assert _wait_until(lambda: not stage_dir.exists(), timeout=5.0)
        assert _wait_until(lambda: not (manifests / "sess-1.json").exists())
    finally:
        restarted.stop()
    assert (dest_dir / "a.bin").read_bytes() == b"hello"


def test_free_up_evicts_drained_sessions_oldest_first_until_enough(tmp_path):
    """R5: the RAM disk runs short before retention expires."""
    stage = {k: tmp_path / "stage" / k for k in "abc"}
    for p in stage.values():
        _make_tree(p, {"f.bin": b"x" * 1000})
    pool = DrainPool(num_workers=1, retention_s=9999, high_water_bytes=10**12)
    pool.start()
    try:
        for k, p in stage.items():
            pool.enqueue(DrainJob(f"sess-{k}", [(p, tmp_path / "raw" / k)]))
            assert _wait_until(lambda k=k: (tmp_path / "raw" / k).exists())
            time.sleep(0.01)  # distinct drained_at order
        assert pool.free_up(1500) == 2000  # a, then b: 1000 is not enough
        assert not stage["a"].exists() and not stage["b"].exists()
        assert stage["c"].exists()
        assert (tmp_path / "raw" / "a" / "f.bin").exists()  # GPFS copy kept
        assert pool.free_up(10**9) == 1000 and not stage["c"].exists()
        assert pool.free_up(1) == 0  # nothing left to evict
    finally:
        pool.stop()


# --- ING-F4: a failed drain is retried, and a restart does not forget it -------


def _fail_copytree(monkeypatch, times):
    """Make the first `times` copytree calls fail like a full or flaky GPFS."""
    import opym.stream.drain as drain_mod

    real = drain_mod.shutil.copytree
    calls = {"n": 0}

    def _flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= times:
            raise OSError(28, "No space left on device")
        return real(*args, **kwargs)

    monkeypatch.setattr(drain_mod.shutil, "copytree", _flaky)
    return calls


def test_a_failed_drain_is_retried_until_it_lands(tmp_path, monkeypatch):
    stage_dir = tmp_path / "stage" / "sample_C0.ome.zarr"
    dest_dir = tmp_path / "raw" / "sample_C0.ome.zarr"
    manifests = tmp_path / "stage" / ".drain_manifests"
    _make_tree(stage_dir, {"a.bin": b"hello"})
    calls = _fail_copytree(monkeypatch, times=2)
    events = []

    pool = DrainPool(
        num_workers=1,
        retention_s=9999,
        high_water_bytes=10**12,
        manifest_dir=manifests,
        retry_initial_s=0.05,
        retry_max_s=0.1,
        on_state=lambda job, state, detail: events.append((state, detail)),
    )
    pool.start()
    try:
        pool.enqueue(DrainJob("sess-retry", [(stage_dir, dest_dir)]))
        assert _wait_until(lambda: (dest_dir / "a.bin").exists())
        assert (dest_dir / "a.bin").read_bytes() == b"hello"
        assert calls["n"] == 3
        assert _wait_until(lambda: pool.status()["failed"] == {})
        assert [e[0] for e in events] == ["failed", "failed", "drained"]
        assert "No space left" in events[0][1]
        # landed: its pending record is gone, its retention record is there
        assert _wait_until(
            lambda: not (manifests / "pending" / "sess-retry.json").exists()
        )
        assert (manifests / "sess-retry.json").exists()
    finally:
        pool.stop()
    assert stage_dir.exists()  # staging copy untouched throughout


def test_a_drain_that_keeps_failing_says_so(tmp_path, monkeypatch):
    stage_dir = tmp_path / "stage" / "sample_C0.ome.zarr"
    dest_dir = tmp_path / "raw" / "sample_C0.ome.zarr"
    _make_tree(stage_dir, {"a.bin": b"hello"})
    _fail_copytree(monkeypatch, times=10**6)

    pool = DrainPool(
        num_workers=1,
        retention_s=9999,
        high_water_bytes=10**12,
        retry_initial_s=0.05,
        retry_max_s=0.05,
    )
    pool.start()
    try:
        pool.enqueue(DrainJob("sess-bad", [(stage_dir, dest_dir)]))
        assert _wait_until(
            lambda: pool.status()["failed"].get("sess-bad", {}).get("attempts", 0) >= 3
        )
        info = pool.status()["failed"]["sess-bad"]
        assert "No space left" in info["error"]
        assert not dest_dir.exists()
        assert (stage_dir / "a.bin").read_bytes() == b"hello"
    finally:
        pool.stop()


def test_a_queued_drain_survives_a_restart(tmp_path):
    """The receiver told the client `ended` and then restarted before the drain
    ran: the only copy is on the RAM disk, and the new process must drain it."""
    stage_dir = tmp_path / "stage" / "sample_C0.ome.zarr"
    dest_dir = tmp_path / "raw" / "sample_C0.ome.zarr"
    manifests = tmp_path / "stage" / ".drain_manifests"
    _make_tree(stage_dir, {"a.bin": b"hello"})

    first = DrainPool(manifest_dir=manifests)  # never started: nothing drains
    first.enqueue(
        DrainJob("sess-q", [(stage_dir, dest_dir)], status_leaf=tmp_path / "raw")
    )
    assert (manifests / "pending" / "sess-q.json").exists()
    assert not dest_dir.exists()

    events = []
    second = DrainPool(
        num_workers=1,
        retention_s=9999,
        high_water_bytes=10**12,
        manifest_dir=manifests,
        on_state=lambda job, state, detail: events.append((job.status_leaf, state)),
    )
    second.start()
    try:
        assert _wait_until(lambda: (dest_dir / "a.bin").exists())
        assert _wait_until(lambda: not (manifests / "pending" / "sess-q.json").exists())
        assert events == [(tmp_path / "raw", "drained")]
    finally:
        second.stop()


def test_a_pending_drain_whose_staging_copy_is_gone_is_reported_not_retried(tmp_path):
    stage_dir = tmp_path / "stage" / "sample_C0.ome.zarr"
    dest_dir = tmp_path / "raw" / "sample_C0.ome.zarr"
    manifests = tmp_path / "stage" / ".drain_manifests"
    _make_tree(stage_dir, {"a.bin": b"hello"})
    first = DrainPool(manifest_dir=manifests)
    first.enqueue(DrainJob("sess-gone", [(stage_dir, dest_dir)]))
    import shutil

    shutil.rmtree(stage_dir)  # the reboot wiped the RAM disk

    events = []
    second = DrainPool(
        num_workers=1,
        manifest_dir=manifests,
        on_state=lambda job, state, detail: events.append((state, detail)),
    )
    second.start()
    try:
        assert [e[0] for e in events] == ["failed"]
        assert "staging copy gone" in events[0][1]
        assert "sess-gone" in second.status()["failed"]
        assert not (manifests / "pending" / "sess-gone.json").exists()
        assert second.pending_count == 0 and not dest_dir.exists()
    finally:
        second.stop()


def test_a_newer_job_for_a_session_replaces_its_waiting_retry(tmp_path, monkeypatch):
    stage_dir = tmp_path / "stage" / "s"
    dest_dir = tmp_path / "raw" / "s"
    _make_tree(stage_dir, {"a.bin": b"x"})
    _fail_copytree(monkeypatch, times=1)
    pool = DrainPool(num_workers=1, retry_initial_s=3600, retry_max_s=3600)
    pool.start()
    try:
        pool.enqueue(DrainJob("sess-r", [(stage_dir, dest_dir)]))
        assert _wait_until(lambda: pool.status()["retrying"] == 1)
        pool.enqueue(DrainJob("sess-r", [(stage_dir, dest_dir)]))
        assert pool.status()["retrying"] == 0
        assert _wait_until(lambda: (dest_dir / "a.bin").exists())
    finally:
        pool.stop()


def test_an_unsafe_session_id_never_names_a_manifest(tmp_path):
    stage_dir = tmp_path / "stage" / "s"
    manifests = tmp_path / "stage" / ".drain_manifests"
    _make_tree(stage_dir, {"a.bin": b"x"})
    pool = DrainPool(manifest_dir=manifests)
    pool.enqueue(DrainJob("../../escape", [(stage_dir, tmp_path / "raw" / "s")]))
    assert not list(tmp_path.rglob("escape*"))
