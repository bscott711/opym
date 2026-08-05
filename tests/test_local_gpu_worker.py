"""Tests for local_gpu_worker.py's crash-loop backoff.

Guards against the watchdog hot-looping when Matlab dies at launch (e.g. a
broken license checkout kills Matlab before it ever reaches
run_petakit_server.m, so the queued ticket is never claimed and the same
launch gets retried forever). See _next_backoff_sec's docstring in
local_gpu_worker.py for the reasoning.
"""

from __future__ import annotations

import time

import pytest

from opym import local_gpu_worker
from opym.local_gpu_worker import _next_backoff_sec


class TestNextBackoffSec:
    def test_first_failure_returns_base(self):
        assert _next_backoff_sec(1, base=15, cap=300) == 15

    def test_doubles_each_consecutive_failure(self):
        assert [_next_backoff_sec(n, base=15, cap=300) for n in range(1, 5)] == [
            15,
            30,
            60,
            120,
        ]

    def test_clamps_at_cap(self):
        assert _next_backoff_sec(6, base=15, cap=300) == 300
        assert _next_backoff_sec(20, base=15, cap=300) == 300


class _FakeProcess:
    def __init__(self, returncode: int):
        self.returncode = returncode

    def wait(self):
        pass


def test_process_queue_backs_off_on_failure_and_resets_on_success(monkeypatch, tmp_path):
    """Two consecutive Matlab launch failures should back off with growing
    delays (never the plain poll_interval) and leave the ticket in the
    queue; a subsequent clean run should reset the failure streak and go
    back to the normal poll_interval."""
    queue_dir = tmp_path / "petakit_jobs" / "queue"
    queue_dir.mkdir(parents=True)
    (queue_dir / "job1.json").write_text("{}")

    monkeypatch.setattr(local_gpu_worker, "BASE_DIR", tmp_path / "petakit_jobs")
    monkeypatch.setattr(local_gpu_worker, "QUEUE_DIR", queue_dir)

    # Cycle 1 and 2 fail (Matlab exits nonzero, e.g. license checkout);
    # cycle 3 succeeds (clean idle-timeout shutdown, exit 0).
    cycle_returncodes = [1, 1, 0]
    popen_calls = {"count": 0}

    def fake_popen(cmd, env=None):
        cycle = popen_calls["count"] // 2
        popen_calls["count"] += 1
        return _FakeProcess(cycle_returncodes[cycle])

    monkeypatch.setattr(local_gpu_worker.subprocess, "Popen", fake_popen)

    consolidations = {"count": 0}

    def fake_consolidations(base_dir):
        consolidations["count"] += 1
        return 0

    monkeypatch.setattr(local_gpu_worker, "run_pending_consolidations", fake_consolidations)

    sleep_calls = []

    def fake_sleep(secs):
        sleep_calls.append(secs)
        if len(sleep_calls) >= len(cycle_returncodes):
            raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", fake_sleep)

    local_gpu_worker.process_queue(idle_timeout_sec=60, poll_interval=2)

    # cycle 1 -> backoff(1)=15, cycle 2 -> backoff(2)=30, cycle 3 (success)
    # -> falls through to the normal poll_interval sleep, not a backoff one.
    assert sleep_calls == [15, 30, 2]
    # Matlab was relaunched (not left dead) on every cycle, ticket untouched
    # by our fake process the whole time.
    assert popen_calls["count"] == 6
    # Consolidation only runs after a successful (exit 0) cycle.
    assert consolidations["count"] == 1
