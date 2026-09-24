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
    actual_timepoints   INTEGER,
    decon_psf           TEXT
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
# 'partial' marks a Micro-Manager multi-file series where one or more
# sibling files were corrupt but a valid LEADING PREFIX was recovered and
# processed -- readable and real, just fewer timepoints than the
# acquisition was configured for (see `expected_timepoints` /
# `actual_timepoints`). Deliberately a prefix, never a subset with gaps --
# see `resolve_readable_master` in bioimaging/backfill/pipeline.py. 'dead'
# marks a raw file with NO valid data at all (the master file itself is
# corrupt) -- distinct from 'corrupt' only so a human scanning the
# dashboard can tell "not even worth a partial-recovery attempt" from
# "recovery was attempted and something is still wrong". 'blocked' marks a
# dataset deferred on a deliberate guard rather than a bug -- e.g. no
# z-step derivable yet (see `resolve_zarr_z_step` and its
# `OPYM_ZARR_ALLOW_DEFAULT_Z_STEP` override) -- so the dashboard's failure
# count reflects genuine problems, not a working-as-intended wait for data
# that hasn't finished landing. Unlike 'corrupt'/'dead', still cheap to
# recheck every pass (no multi-GB re-read) and self-heals automatically
# once the missing metadata arrives -- not made sticky the way those are.
SIGNAL_FLAGS = ("ok", "dud", "unknown", "corrupt", "partial", "dead", "blocked")


# Columns added after the original schema shipped -- `CREATE TABLE IF NOT
# EXISTS` is a no-op against an already-existing table, so a registry
# created before this change needs an explicit migration to pick them up.
_NEW_DATASET_COLUMNS = {
    "signal_flag": "TEXT",
    "expected_timepoints": "INTEGER",
    "actual_timepoints": "INTEGER",
    # Which PSF this dataset's DSR output was deconvolved with; NULL means
    # deskew-only. Deliberately a dataset column rather than a new `decon`
    # entry in STAGES: deconvolution is step A *inside* the same MATLAB
    # deskew ticket, so it has no ticket of its own and cannot succeed or
    # fail independently of `deskew`. A stage row would also be dropped
    # silently by the dashboard, which keeps its own duplicate STAGES tuple
    # (opym-dashboard/app/registry_reader.py) and filters to it -- failures
    # included. What is actually wanted here is provenance: which PSF
    # produced the data on disk.
    "decon_psf": "TEXT",
    # A short fingerprint of the OMW knobs (wienerAlpha/OTFCumThresh/
    # hannWinBounds/dampFactor) this dataset's DSR output was actually
    # produced with. `decon_psf` alone answers "which PSF" but not "which
    # settings" -- two runs with the identical PSF file and different
    # alpha/hann/damp look byte-identical to that column, so a parameter
    # retune (this one: the 22- then 20-variant sweep that picked `super4`)
    # would otherwise be silently skipped as already-done. NULL for
    # deskew-only or for output predating this column.
    "decon_params": "TEXT",
    # (size, mtime) signature of the raw master file at the moment it was
    # marked 'corrupt'/'dead' -- lets a later pass tell "same broken file
    # we already gave up on" (skip re-attempting a guaranteed-identical
    # failure every --watch pass) from "the file changed since then" (a
    # re-upload/repair, worth retrying). See `master_file_fingerprint()`.
    "master_file_fingerprint": "TEXT",
}


def _migrate_schema(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(datasets)")}
    for col, col_type in _NEW_DATASET_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE datasets ADD COLUMN {col} {col_type}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def master_file_fingerprint(path: Path) -> str | None:
    """A cheap (size, mtime) signature for a raw master file -- not a
    content hash (too slow to compute over a multi-GB acquisition on every
    --watch pass), just enough to distinguish "the same broken file we
    already gave up on" from "someone re-uploaded or repaired it". Returns
    None if the file can't be stat'd (e.g. already gone), which never
    matches a previously-recorded fingerprint, so a vanished file is always
    treated as "changed" (worth a fresh attempt) rather than silently
    staying stuck on stale triage.
    """
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return f"{st.st_size}:{st.st_mtime_ns}"


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
        """Every backfill run re-discovers and re-registers every dataset
        (see `run_backfill`), including ones already in the registry -- the
        ON CONFLICT branch must refresh `master_file` or a dataset whose
        raw file was still mid-upload at first discovery (so
        `select_master_ome_tif` picked a numbered continuation file instead
        of the not-yet-written bare-name master) stays wrong forever, even
        after the real master file lands and every later re-discovery pass
        recomputes the correct one. Confirmed via a real stuck dataset:
        `discovered_at` predated the bare master file's mtime by hours, and
        `master_file` never budged across dozens of subsequent runs.
        `discovered_at` itself is intentionally NOT refreshed -- it's "first
        seen," not "last seen."
        """
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO datasets
                       (dataset_key, root, leaf_dir, master_file, has_legacy_decon, discovered_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(dataset_key) DO UPDATE SET
                       master_file = excluded.master_file,
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

    def get_dataset(self, dataset_key: str) -> dict | None:
        """The `datasets` row for one dataset (signal_flag, timepoint
        counts, fingerprint, etc.), or None if it hasn't been registered
        yet. Used to check triage state before redoing expensive work --
        e.g. `detect_rois` skipping a dataset already known 'dead'/'corrupt'
        with an unchanged raw file, rather than re-reading it every pass.
        """
        with self._cursor() as cur:
            cur.execute("SELECT * FROM datasets WHERE dataset_key=?", (dataset_key,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))

    def set_signal_flag(self, dataset_key: str, signal_flag: str) -> None:
        """Updates ONLY `signal_flag`, unlike `set_triage` -- which always
        overwrites `expected_timepoints`/`actual_timepoints` too, defaulting
        them to NULL when not given. Callers that want to reclassify a
        dataset (e.g. 'blocked' on a deliberate guard, or 'dead'/'corrupt'
        on an unreadable raw file) without a freshly-computed frame count
        must use this instead, or they'd silently erase real timepoint
        counts `set_triage` recorded earlier in the SAME dataset's
        lifecycle (confirmed real risk: a zarr-precropped dataset already
        past `roi_detect` before hitting the z-step guard on every deskew
        retry would otherwise lose its frame-count badge on every pass).
        """
        with self._cursor() as cur:
            cur.execute(
                "UPDATE datasets SET signal_flag=? WHERE dataset_key=?",
                (signal_flag, dataset_key),
            )

    def set_master_file_fingerprint(self, dataset_key: str, fingerprint: str | None) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE datasets SET master_file_fingerprint=? WHERE dataset_key=?",
                (fingerprint, dataset_key),
            )

    def set_decon_psf(self, dataset_key: str, psf_path: str | None) -> None:
        """Records which PSF this dataset was deconvolved with (None =
        deskew-only), so the provenance of the data on disk is recoverable
        without re-deriving it from directory names. See
        `_NEW_DATASET_COLUMNS` for why this is a column, not a stage.
        """
        with self._cursor() as cur:
            cur.execute(
                "UPDATE datasets SET decon_psf=? WHERE dataset_key=?",
                (psf_path, dataset_key),
            )

    def get_decon_psf(self, dataset_key: str) -> str | None:
        """The PSF this dataset's on-disk output was produced with, or None
        for deskew-only. Callers compare this against the PSF they are about
        to use: a dataset whose `deskew` stage says `done` was only done for
        the PSF it was done WITH, and re-running with a different one (or
        with decon newly switched on) must not be skipped as already-complete.
        """
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT decon_psf FROM datasets WHERE dataset_key=?", (dataset_key,)
            ).fetchone()
        return row[0] if row and row[0] else None

    def set_decon_params(self, dataset_key: str, params_fingerprint: str | None) -> None:
        """Records a fingerprint of the OMW knobs (see
        `backfill.pipeline.decon_params_fingerprint`) this dataset's on-disk
        output was actually produced with -- the parameter-retune analog of
        `set_decon_psf`. Set alongside it, not instead of it: `decon_psf`
        stays the human-readable "which PSF" answer.
        """
        with self._cursor() as cur:
            cur.execute(
                "UPDATE datasets SET decon_params=? WHERE dataset_key=?",
                (params_fingerprint, dataset_key),
            )

    def get_decon_params(self, dataset_key: str) -> str | None:
        """The OMW parameter fingerprint this dataset's on-disk output was
        produced with, or None for deskew-only or for output written before
        this column existed. Callers compare this against the fingerprint of
        the settings they are about to use, alongside `get_decon_psf`: same
        PSF file but different alpha/hann/damp is a real, on-disk-visible
        difference, not a no-op.
        """
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT decon_params FROM datasets WHERE dataset_key=?", (dataset_key,)
            ).fetchone()
        return row[0] if row and row[0] else None

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

    def reset_stage(self, dataset_key: str, stage: str) -> None:
        """Marks a stage as needing to run again ('pending', no finish time)
        because something upstream of it is being redone -- e.g. a new
        deskew ticket makes the existing MIPs out of date.

        Keeps `output_path`, deliberately: the old output is still on disk
        until the stage re-runs, and opym-dashboard uses a pending row with
        an output_path to label that dataset "reprocessing" rather than
        either "done" (wrong: the output is stale) or "never processed"
        (wrong: there is something to look at). No-op when the stage has no
        row yet.
        """
        with self._cursor() as cur:
            cur.execute(
                """UPDATE stage_status
                   SET status='pending', finished_at=NULL, error_message=NULL
                   WHERE dataset_key=? AND stage=?""",
                (dataset_key, stage),
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
