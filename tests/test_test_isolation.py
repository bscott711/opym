"""The suite's own isolation (tests/conftest.py): the production-path
variables point under tmp, and the guard that refuses production paths trips
on them (LIVE-07, TEST-01)."""

from __future__ import annotations

import os
from pathlib import Path

import conftest
import pytest

from opym import lanes
from opym.stream import live_zarr

PRODUCTION_PATHS = {
    "PETAKIT_JOBS_DIR": "/dev/shm/petakit_jobs",
    "OPYM_LIVE_VIEW_ROOT": "/dev/shm/opym_live_view",
    "OPYM_STREAM_STAGE_ROOT": "/dev/shm/opym_stream_stage",
    "OPYM_BACKFILL_REGISTRY_PATH": str(conftest.PRODUCTION_REGISTRY),
}


def _under(child: Path, parent: Path) -> bool:
    return Path(child).resolve().is_relative_to(parent.resolve())


def test_session_env_points_under_the_session_tmp_dir():
    """The session-level redirect, set when conftest was imported, before any
    opym module: what import-time constants (and subprocesses) inherit."""
    assert set(conftest.SESSION_ENV) == set(PRODUCTION_PATHS)
    for name, val in conftest.SESSION_ENV.items():
        assert _under(Path(val), conftest.SESSION_DIR), name
    assert conftest.production_path_violations(conftest.SESSION_ENV) == []
    # And what the running test sees is safe too.
    assert conftest.production_path_violations() == []


def test_each_test_gets_its_own_paths_and_no_staging(tmp_path):
    assert _under(lanes.jobs_dir(), tmp_path)
    assert _under(live_zarr.view_root(), tmp_path)
    assert _under(Path(os.environ["OPYM_BACKFILL_REGISTRY_PATH"]), tmp_path)
    # Staging is opt-in; leaving it set sends raw stores away from the dataset
    # directory and fails 26 stream tests (TEST-01).
    assert "OPYM_STREAM_STAGE_ROOT" not in os.environ
    # Even the fallbacks used when a variable is unset stay in tmp.
    assert _under(Path(lanes.DEFAULT_JOBS_DIR), tmp_path)
    assert _under(live_zarr.DEFAULT_VIEW_ROOT, tmp_path)


@pytest.mark.parametrize("name", sorted(PRODUCTION_PATHS))
def test_guard_trips_on_a_production_path(name):
    """Negative control: the guard is not vacuous."""
    bad = conftest.production_path_violations({name: PRODUCTION_PATHS[name]})
    assert len(bad) == 1 and name in bad[0]
    # Below a production path, and via a symlink-free `..` detour, too.
    sub = PRODUCTION_PATHS[name] + "/sub"
    assert conftest.production_path_violations({name: sub})
    assert conftest.production_path_violations({name: "/tmp/x/../../mmfs2/scratch/y"})


def test_guard_trips_on_the_data_disks():
    for p in ("/mmfs1/data/x", "/mmfs2/scratch/SDSMT.LOCAL/bscott"):
        assert conftest.production_path_violations({"PETAKIT_JOBS_DIR": p})


def test_guard_passes_tmp_and_unset(tmp_path):
    env = {"PETAKIT_JOBS_DIR": str(tmp_path), "OPYM_STREAM_STAGE_ROOT": ""}
    assert conftest.production_path_violations(env) == []


def test_session_start_guard_raises(monkeypatch):
    monkeypatch.setenv("OPYM_LIVE_VIEW_ROOT", "/dev/shm/opym_live_view")
    with pytest.raises(pytest.UsageError):
        conftest._assert_no_production_paths()
