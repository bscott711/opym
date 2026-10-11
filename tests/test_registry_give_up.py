"""Give-up bookkeeping in StatusRegistry: a stage that keeps failing the
same way stops being retried by the --watch backfill (2026-10-10: 11 such
failures were retried every 2 minutes for days)."""

from __future__ import annotations

import sqlite3

from opym.registry import MAX_STAGE_FAILURES, StatusRegistry


def _reg(tmp_path):
    reg = StatusRegistry(tmp_path / "registry.sqlite3")
    reg.register_dataset("ds", root="/r", leaf_dir="/r/ds", master_file="/r/ds/m.tif")
    return reg


def _fail(reg, n, stage="mip_encode"):
    for _ in range(n):
        reg.start_stage("ds", stage)
        reg.finish_stage("ds", stage, status="failed", error="boom")


def test_gives_up_after_max_consecutive_failures(tmp_path):
    reg = _reg(tmp_path)
    _fail(reg, MAX_STAGE_FAILURES - 1)
    assert reg.gave_up_datasets() == {}
    _fail(reg, 1)
    assert reg.gave_up_datasets() == {"ds": ["mip_encode"]}
    assert reg.get_stage("ds", "mip_encode")["status"] == "failed"


def test_success_and_reset_zero_the_streak(tmp_path):
    reg = _reg(tmp_path)
    _fail(reg, MAX_STAGE_FAILURES - 1)
    reg.finish_stage("ds", "mip_encode", status="done")
    _fail(reg, MAX_STAGE_FAILURES - 1)
    assert reg.gave_up_datasets() == {}

    _fail(reg, 1)
    assert reg.gave_up_datasets() == {"ds": ["mip_encode"]}
    reg.reset_stage("ds", "mip_encode")  # e.g. a new deskew superseded it
    assert reg.get_stage("ds", "mip_encode")["consecutive_failures"] == 0
    assert reg.gave_up_datasets() == {}


def test_clear_gave_up_allows_a_retry(tmp_path):
    reg = _reg(tmp_path)
    _fail(reg, MAX_STAGE_FAILURES)
    assert reg.clear_gave_up() == 1
    assert reg.gave_up_datasets() == {}
    assert reg.get_stage("ds", "mip_encode")["status"] == "failed"


def test_old_registry_is_migrated(tmp_path):
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """CREATE TABLE datasets (dataset_key TEXT PRIMARY KEY, root TEXT NOT NULL,
               leaf_dir TEXT NOT NULL, master_file TEXT NOT NULL,
               has_legacy_decon INTEGER NOT NULL DEFAULT 0, discovered_at TEXT NOT NULL);
           CREATE TABLE stage_status (dataset_key TEXT NOT NULL, stage TEXT NOT NULL,
               status TEXT NOT NULL, started_at TEXT, finished_at TEXT, output_path TEXT,
               ticket_path TEXT, error_message TEXT, PRIMARY KEY (dataset_key, stage));
           INSERT INTO datasets VALUES ('ds','/r','/r/ds','/r/ds/m.tif',0,'2026-01-01');
           INSERT INTO stage_status (dataset_key, stage, status)
               VALUES ('ds','mip_encode','failed');"""
    )
    conn.commit()
    conn.close()
    reg = StatusRegistry(path)
    assert reg.get_stage("ds", "mip_encode")["consecutive_failures"] == 0
    _fail(reg, MAX_STAGE_FAILURES)
    assert reg.gave_up_datasets() == {"ds": ["mip_encode"]}
    StatusRegistry(path).close()  # a second open must not re-add the column
