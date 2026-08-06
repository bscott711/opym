# Ruff style: Compliant
"""
Lightweight SQLite status registry for the bulk no-decon backfill
(crop -> zarr -> DSR -> MIP). No aggregate status log existed before this --
only per-dataset `_processing_log.json` files, too slow to re-scan across
GPFS for a live dashboard across ~150-300 datasets.

This module is the *writer* side, used by the bioimaging-side bulk driver.
The standalone `opym-dashboard` repo is the *reader* side and deliberately
does not import this module (or anything else under `opym.*` -- see that
repo's README) to avoid dragging napari/ipywidgets/torch into a small
always-on web service. The SQLite schema below is the contract between the
two; keep both sides in sync if it ever changes.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_key         TEXT PRIMARY KEY,
    root                TEXT NOT NULL,
    leaf_dir            TEXT NOT NULL,
    master_file         TEXT NOT NULL,
    has_legacy_decon    INTEGER NOT NULL DEFAULT 0,
    discovered_at       TEXT NOT NULL,
    signal_flag         TEXT,
    expected_timepoints INTEGER,
    actual_timepoints   INTEGER
);

CREATE TABLE IF NOT EXISTS stage_status (
    dataset_key   TEXT NOT NULL REFERENCES datasets(dataset_key),
    stage         TEXT NOT NULL,
    status        TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    output_path   TEXT,
    ticket_path   TEXT,
    error_message TEXT,
    PRIMARY KEY (dataset_key, stage)
);
"""

# Valid `stage` values, in pipeline order. Not enforced by the schema (kept
# as plain TEXT for forward-compatibility) but every writer in this repo
# should only ever use these.
STAGES = ("roi_detect", "crop_zarr", "crop_tiff", "deskew", "mip_encode")

# Valid `signal_flag` values (also plain TEXT, no CHECK constraint, for the
# same forward-compatibility reason). 'corrupt' marks a raw file that is
# fundamentally unreadable (truncated/corrupted acquisition, or corrupted
# TIFF metadata inside a third-party library's own parser) -- distinct from
# 'dud' (readable, just no detectable signal) so the dashboard can tell
# "nothing will ever come of this" from "weak signal, still processed".
SIGNAL_FLAGS = ("ok", "dud", "unknown", "corrupt")


# Columns added after the original schema shipped -- `CREATE TABLE IF NOT
# EXISTS` is a no-op against an already-existing table, so a registry
# created before this change needs an explicit migration to pick them up.
_NEW_DATASET_COLUMNS = {
    "signal_flag": "TEXT",
    "expected_timepoints": "INTEGER",
    "actual_timepoints": "INTEGER",
}


def _migrate_schema(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(datasets)")}
    for col, col_type in _NEW_DATASET_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE datasets ADD COLUMN {col} {col_type}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StatusRegistry:
    """One instance per process/worker -- sqlite3 connections aren't
    picklable, so `ProcessPoolExecutor` workers must each construct their
    own `StatusRegistry(db_path)` rather than sharing one across a pool.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        _migrate_schema(self._conn)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> StatusRegistry:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @contextmanager
    def _cursor(self):
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        finally:
            cur.close()

    def register_dataset(
        self,
        dataset_key: str,
        *,
        root: str,
        leaf_dir: str,
        master_file: str,
        has_legacy_decon: bool = False,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO datasets
                       (dataset_key, root, leaf_dir, master_file, has_legacy_decon, discovered_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(dataset_key) DO UPDATE SET
                       has_legacy_decon = excluded.has_legacy_decon""",
                (dataset_key, root, leaf_dir, master_file, int(has_legacy_decon), _now()),
            )

    def set_triage(
        self,
        dataset_key: str,
        *,
        signal_flag: str,
        expected_timepoints: int | None = None,
        actual_timepoints: int | None = None,
    ) -> None:
        """Records the cheap upfront signal-presence check + frame-count
        info gathered during `roi_detect` -- lets the bulk orchestrator
        deprioritize likely-empty/aborted datasets (see `backfill/cli.py`'s
        triage phase) and lets the dashboard show a frame-count sanity badge
        (e.g. "1/100" for an acquisition that aborted after one timepoint).
        """
        with self._cursor() as cur:
            cur.execute(
                """UPDATE datasets
                       SET signal_flag=?, expected_timepoints=?, actual_timepoints=?
                     WHERE dataset_key=?""",
                (signal_flag, expected_timepoints, actual_timepoints, dataset_key),
            )

    def start_stage(self, dataset_key: str, stage: str, *, ticket_path: str | None = None) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO stage_status (dataset_key, stage, status, started_at, ticket_path)
                   VALUES (?, ?, 'running', ?, ?)
                   ON CONFLICT(dataset_key, stage) DO UPDATE SET
                       status='running', started_at=excluded.started_at,
                       ticket_path=COALESCE(excluded.ticket_path, stage_status.ticket_path),
                       error_message=NULL""",
                (dataset_key, stage, _now(), ticket_path),
            )

    def finish_stage(
        self,
        dataset_key: str,
        stage: str,
        *,
        status: str,
        output_path: str | None = None,
        ticket_path: str | None = None,
        error: str | None = None,
    ) -> None:
        if status not in ("done", "failed"):
            raise ValueError(f"finish_stage status must be 'done' or 'failed', got {status!r}")
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO stage_status
                       (dataset_key, stage, status, started_at, finished_at,
                        output_path, ticket_path, error_message)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(dataset_key, stage) DO UPDATE SET
                       status=excluded.status, finished_at=excluded.finished_at,
                       output_path=COALESCE(excluded.output_path, stage_status.output_path),
                       ticket_path=COALESCE(excluded.ticket_path, stage_status.ticket_path),
                       error_message=excluded.error_message""",
                (dataset_key, stage, status, _now(), _now(), output_path, ticket_path, error),
            )

    def get_stage(self, dataset_key: str, stage: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM stage_status WHERE dataset_key=? AND stage=?",
                (dataset_key, stage),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))

    def is_stage_done(self, dataset_key: str, stage: str) -> bool:
        row = self.get_stage(dataset_key, stage)
        return bool(row and row["status"] == "done")

    def pending_deskew_datasets(self) -> list[dict]:
        """Datasets whose `deskew` ticket has been submitted (status
        'running', so `ticket_path` is set) but hasn't yet resolved."""
        with self._cursor() as cur:
            cur.execute(
                """SELECT d.*, s.ticket_path FROM stage_status s
                   JOIN datasets d ON d.dataset_key = s.dataset_key
                   WHERE s.stage='deskew' AND s.status='running'"""
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def all_datasets(self) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM datasets")
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
