"""Tests for opym.lanes: the live lease and backfill admission counting."""

from __future__ import annotations

import os
import time

from opym import lanes


def test_jobs_dir_honors_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PETAKIT_JOBS_DIR", str(tmp_path / "jobs"))
    assert lanes.jobs_dir() == tmp_path / "jobs"
    assert lanes.live_queue_dir() == tmp_path / "jobs" / "queue_live"
    assert lanes.backfill_queue_dir() == tmp_path / "jobs" / "queue"


def test_lease_is_active_only_while_fresh(tmp_path):
    assert not lanes.live_lease_active(tmp_path)
    lanes.write_lease(["sess-1"], tmp_path)
    assert lanes.live_lease_active(tmp_path)

    # A receiver that stops refreshing (crash) releases the GPUs by itself.
    stale = time.time() - lanes.LEASE_MAX_AGE_S - 1
    os.utime(lanes.lease_path(tmp_path), (stale, stale))
    assert not lanes.live_lease_active(tmp_path)

    lanes.write_lease(["sess-1"], tmp_path)
    lanes.release_lease(tmp_path)
    assert not lanes.live_lease_active(tmp_path)


def test_lease_keeper_refreshes_only_every_interval_and_releases(tmp_path):
    keeper = lanes.LeaseKeeper(tmp_path, refresh_s=10)
    path = lanes.lease_path(tmp_path)

    keeper.update([], now=0.0)
    assert not path.exists() and not keeper.held

    keeper.update(["a"], now=0.0)
    assert keeper.held and path.exists()
    os.utime(path, (1, 1))  # mark so a rewrite is visible

    keeper.update(["a"], now=5.0)  # within the refresh interval: untouched
    assert path.stat().st_mtime == 1
    keeper.update(["a"], now=10.0)
    assert path.stat().st_mtime > 1

    keeper.update([], now=11.0)
    assert not keeper.held and not path.exists()


def test_backfill_inflight_counts_queued_claims_and_requeues(tmp_path):
    queue = lanes.backfill_queue_dir(tmp_path)
    queue.mkdir(parents=True)
    assert lanes.backfill_inflight(tmp_path) == 0
    for name in ("a.json", ".active_b.json", ".requeue_c.json"):
        (queue / name).write_text("{}")
    live = lanes.live_queue_dir(tmp_path)
    live.mkdir()
    (live / "live.json").write_text("{}")  # the live lane is not counted
    assert lanes.backfill_inflight(tmp_path) == 3


def test_admission_closes_at_the_cap_and_during_a_live_lease(tmp_path):
    queue = lanes.backfill_queue_dir(tmp_path)
    queue.mkdir(parents=True)
    assert lanes.backfill_admission_open(2, tmp_path)
    (queue / "a.json").write_text("{}")
    (queue / ".active_b.json").write_text("{}")
    assert not lanes.backfill_admission_open(2, tmp_path)
    assert lanes.backfill_admission_open(None, tmp_path)  # uncapped
    assert lanes.backfill_admission_open(0, tmp_path)

    lanes.write_lease(["s"], tmp_path)
    assert not lanes.backfill_admission_open(None, tmp_path)


def test_parallel_admissions_never_overshoot_the_cap(tmp_path):
    """8 processes race to queue a ticket each against a cap of 3."""
    import multiprocessing as mp

    queue = lanes.backfill_queue_dir(tmp_path)
    queue.mkdir(parents=True)
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_admit_one, args=(tmp_path, i)) for i in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    assert len(list(queue.glob("*.json"))) == 3


def _admit_one(jobs, i):
    with lanes.backfill_admission(3, jobs) as admitted:
        if admitted:
            time.sleep(0.05)  # widen the race window
            (lanes.backfill_queue_dir(jobs) / f"t{i}.json").write_text("{}")
