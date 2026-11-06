# Ruff style: Compliant
"""
Lightweight SQLite status registry for the bulk no-decon backfill
(crop -> zarr -> DSR -> MIP). No aggregate status log existed before this --
only per-dataset `_processing_log.json` files, too slow to re-scan across
GPFS for a live dashboard across ~150-300 datasets.

`StatusRegistry` is the *writer* side, used by the bioimaging-side bulk
driver. `RegistryReader` (below) is the one *read-only* side: opym-dashboard,
opym-viewer and CORE import it instead of each carrying its own copy of the
schema and queries. This module needs only the standard library, so those
small always-on services can import it without the heavy `opym` extras.
The SQLite schema below is the contract; the reader opens the file
`mode=ro` and never creates or migrates anything.
"""

from __future__ import annotations

import base64
import os
import sqlite3
from collections.abc import Iterator
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
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
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

# `stage_status.status` values a writer produces ('pending' only through
# `reset_stage`: a stage to run again, its `output_path` kept).
STAGE_STATUSES = ("pending", "running", "done", "failed")

# What a dataset reads as overall (`overall_status`). 'acquiring' is not in
# the registry: the reader's caller passes the live lane's `acquiring()` map.
OVERALL_STATUSES = (
    "acquiring",
    "blocked",
    "failed",
    "done",
    "reprocessing",
    "running",
    "in_progress",
    "pending",
)

REGISTRY_PATH_ENV = "OPYM_BACKFILL_REGISTRY_PATH"
DEFAULT_REGISTRY_PATH = (
    "/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/registry.sqlite3"
)

# Every column a registry row carries once migrated (original schema first,
# then `_NEW_*_COLUMNS` order). A registry created before a migration has fewer;
# the reader returns whatever the file has.
DATASET_COLUMNS = (
    "dataset_key",
    "root",
    "leaf_dir",
    "master_file",
    "has_legacy_decon",
    "discovered_at",
    "signal_flag",
    "expected_timepoints",
    "actual_timepoints",
    "decon_psf",
    "decon_params",
    "master_file_fingerprint",
)
STAGE_COLUMNS = (
    "dataset_key",
    "stage",
    "status",
    "started_at",
    "finished_at",
    "output_path",
    "ticket_path",
    "error_message",
    "consecutive_failures",
)


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


# Same, for stage_status. `consecutive_failures` counts back-to-back
# `failed` finishes of one stage; any `done` or `reset_stage` zeroes it.
# --watch retries every failed stage on every pass, so a stage that fails
# the same way forever (an unsupported axis order, a test acquisition with
# no MIPs) used to be retried every 2 minutes for weeks, flooding the log.
# After MAX_STAGE_FAILURES in a row the backfill gives up on that dataset
# (see `gave_up_datasets`) until someone clears it (`clear_gave_up`).
_NEW_STAGE_COLUMNS = {
    "consecutive_failures": "INTEGER NOT NULL DEFAULT 0",
}

MAX_STAGE_FAILURES = 3


def _add_column(conn: sqlite3.Connection, table: str, col: str, col_type: str) -> None:
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
    except sqlite3.OperationalError as e:
        # Another process (the backfill's own pool workers open the
        # registry too) added it between our PRAGMA and this ALTER.
        if "duplicate column" not in str(e):
            raise


def _migrate_schema(conn: sqlite3.Connection) -> None:
    for table, columns in (
        ("datasets", _NEW_DATASET_COLUMNS),
        ("stage_status", _NEW_STAGE_COLUMNS),
    ):
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for col, col_type in columns.items():
            if col not in existing:
                _add_column(conn, table, col, col_type)


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
                        output_path, ticket_path, error_message, consecutive_failures)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(dataset_key, stage) DO UPDATE SET
                       status=excluded.status, finished_at=excluded.finished_at,
                       output_path=COALESCE(excluded.output_path, stage_status.output_path),
                       ticket_path=COALESCE(excluded.ticket_path, stage_status.ticket_path),
                       error_message=excluded.error_message,
                       consecutive_failures=CASE excluded.status
                           WHEN 'failed' THEN stage_status.consecutive_failures + 1
                           ELSE 0 END""",
                (
                    dataset_key, stage, status, _now(), _now(), output_path,
                    ticket_path, error, 1 if status == "failed" else 0,
                ),
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
                   SET status='pending', finished_at=NULL, error_message=NULL,
                       consecutive_failures=0
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

    def gave_up_datasets(self, max_failures: int = MAX_STAGE_FAILURES) -> dict[str, list[str]]:
        """{dataset_key: [stage, ...]} for every stage that is `failed` and
        has failed `max_failures` times in a row: the backfill stops
        retrying those datasets. The rows stay `failed` (the dashboard
        still lists them as failures); nothing is deleted."""
        with self._cursor() as cur:
            cur.execute(
                """SELECT dataset_key, stage FROM stage_status
                   WHERE status='failed' AND consecutive_failures >= ?
                   ORDER BY dataset_key, stage""",
                (max_failures,),
            )
            out: dict[str, list[str]] = {}
            for key, stage in cur.fetchall():
                out.setdefault(key, []).append(stage)
            return out

    def clear_gave_up(self, max_failures: int = MAX_STAGE_FAILURES) -> int:
        """Lets the backfill try every given-up stage again (it stays
        `failed` until the retry finishes). Returns how many were cleared."""
        with self._cursor() as cur:
            cur.execute(
                """UPDATE stage_status SET consecutive_failures=0
                   WHERE status='failed' AND consecutive_failures >= ?""",
                (max_failures,),
            )
            return cur.rowcount

    def all_datasets(self) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM datasets")
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Read-only side: the one reader of `registry.sqlite3` for every consumer
# (opym-dashboard, opym-viewer, CORE). Nothing here writes, creates a schema,
# or migrates; a missing file reads as an empty registry.
# ---------------------------------------------------------------------------


def registry_path() -> Path:
    """The registry file: `OPYM_BACKFILL_REGISTRY_PATH`, else the production default."""
    return Path(os.environ.get(REGISTRY_PATH_ENV) or DEFAULT_REGISTRY_PATH)


def encode_key(dataset_key: str) -> str:
    """A dataset_key (an absolute filesystem path, possibly with spaces or other
    special characters) as one URL path segment. Deliberately base64 (not
    percent-encoding): uvicorn/Starlette decode %2F to '/' before route matching,
    which splits a percent-encoded path into several segments and 404s a
    single-segment route parameter. Base64url's alphabet never produces '/'."""
    return base64.urlsafe_b64encode(dataset_key.encode()).decode().rstrip("=")


def decode_key(url_key: str) -> str:
    """Inverse of `encode_key`."""
    padded = url_key + "=" * (-len(url_key) % 4)
    return base64.urlsafe_b64decode(padded.encode()).decode()


@contextmanager
def connect_ro(
    path: Path | str | None = None, *, timeout: float = 10.0
) -> Iterator[sqlite3.Connection | None]:
    """A read-only sqlite3 connection (rows are `sqlite3.Row`), or None when the
    file does not exist yet (e.g. before the first discovery pass) -- callers
    render an empty state rather than crash. Works on any of the backfill's
    sqlite files (`registry`, `triage`, `auto_triage`); it never creates one."""
    path = Path(path) if path is not None else registry_path()
    if not path.exists():
        yield None
        return
    # as_uri() percent-encodes '?', '#', '%' and spaces, which a bare
    # f"file:{path}" would corrupt.
    uri = f"{path.absolute().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def path_context(leaf_dir: str, root: str, max_segments: int = 4) -> str:
    """The path segment(s) between `root` and the leaf directory, capped to the
    last `max_segments`. `root` is identical across every dataset of one upload
    and the leaf name is shown elsewhere, so the segments in between are what
    tell datasets apart."""
    leaf = Path(leaf_dir)
    try:
        rel_parts = leaf.relative_to(root).parts[:-1]
    except ValueError:
        rel_parts = leaf.parts[:-1]
    if not rel_parts:
        return ""
    truncated = rel_parts[-max_segments:]
    prefix = ".../" if len(truncated) < len(rel_parts) else ""
    return prefix + "/".join(truncated)


def mip_encode_is_current(stages: dict[str, dict | None]) -> bool:
    """True when `mip_encode` is done AND was made from the current upstream
    output. A `done` row alone is not enough: when the backfill re-submits a
    dataset (e.g. a decon parameter change) the upstream stages go back to
    `running` while the previous pass's `mip_encode: done` row -- and its movies
    on disk -- are still there. Every other stage must be settled and must not
    have (re)started after the MIPs were encoded. Timestamps are all
    `datetime.isoformat()` in UTC, so they compare correctly as strings."""
    mip = stages.get("mip_encode")
    if not mip or mip["status"] != "done" or not mip.get("finished_at"):
        return False
    for name, s in stages.items():
        if name == "mip_encode" or s is None:
            continue
        if s["status"] in ("running", "pending"):
            return False
        if (s.get("started_at") or "") > mip["finished_at"]:
            return False
    return True


def overall_status(
    stages: dict[str, dict | None], signal_flag: str | None = None
) -> str:
    """One word for a dataset from its stage rows (`{stage: row or None}`), one
    of `OVERALL_STATUSES` (never 'acquiring': that comes from the live lane).

    * 'blocked' beats 'failed': it marks a deliberate guard (no z-step derivable
      yet) working as intended, not a bug -- see `SIGNAL_FLAGS`.
    * 'done' only when `mip_encode` is current (`mip_encode_is_current`).
    * 'reprocessing': earlier output exists but is being redone -- a stale
      `done`, or a `pending` row the backfill reset (it keeps `output_path` for
      exactly this). The MIP browser still shows the old movies for these.
    * 'pending' is a dataset with NO stage rows at all; a stage row that is
      'pending' without an `output_path` reads as 'in_progress'.
    """
    statuses = [s["status"] for s in stages.values() if s is not None]
    if signal_flag == "blocked" and any(s == "failed" for s in statuses):
        return "blocked"
    if any(s == "failed" for s in statuses):
        return "failed"
    if mip_encode_is_current(stages):
        return "done"
    mip = stages.get("mip_encode")
    if mip and (mip["status"] == "done" or mip.get("output_path")):
        return "reprocessing"
    if any(s == "running" for s in statuses):
        return "running"
    if statuses:
        return "in_progress"
    return "pending"


# SQLite's default limit on `?` parameters is 999 (older builds)
_SQL_VARIABLE_CHUNK = 500


class RegistryReader:
    """Read-only view of `registry.sqlite3`. Cheap to construct (nothing is
    opened until the first query); close it, or use it as a context manager.

    A missing file is an empty registry (every query returns `[]`/`{}`/None), and
    a file that appears later is picked up on the next query. Rows come back as
    plain dicts with every column the file has.
    """

    def __init__(self, path: Path | str | None = None, *, timeout: float = 10.0):
        self.path = Path(path) if path is not None else registry_path()
        self._timeout = timeout
        self._conn: sqlite3.Connection | None = None

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> RegistryReader:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _connection(self) -> sqlite3.Connection | None:
        if self._conn is None:
            if not self.path.exists():
                return None
            uri = f"{self.path.absolute().as_uri()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, timeout=self._timeout)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _query(self, sql: str, params: tuple | list = ()) -> list[dict]:
        conn = self._connection()
        if conn is None:
            return []
        return [dict(r) for r in conn.execute(sql, params)]

    def _select_in(self, table: str, keys: list[str]) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(keys), _SQL_VARIABLE_CHUNK):
            part = keys[i : i + _SQL_VARIABLE_CHUNK]
            marks = ",".join("?" * len(part))
            sql = f"SELECT * FROM {table} WHERE dataset_key IN ({marks})"
            out.extend(self._query(sql, part))
        return out

    # -- plain rows -------------------------------------------------------

    def datasets(self, keys: list[str] | None = None) -> list[dict]:
        """The `datasets` rows (all, or only `keys`), every column the file has
        -- including the triage/signal columns `signal_flag`,
        `expected_timepoints`, `actual_timepoints`, `decon_psf`, `decon_params`
        and `master_file_fingerprint`."""
        if keys is None:
            return self._query("SELECT * FROM datasets")
        return self._select_in("datasets", list(dict.fromkeys(keys)))

    def stage_rows(self, keys: list[str] | None = None) -> list[dict]:
        """The `stage_status` rows (all, or only those of datasets `keys`)."""
        if keys is None:
            return self._query("SELECT * FROM stage_status")
        return self._select_in("stage_status", list(dict.fromkeys(keys)))

    def stages_by_key(
        self, keys: list[str] | None = None
    ) -> dict[str, dict[str, dict]]:
        """`{dataset_key: {stage: row}}`."""
        out: dict[str, dict[str, dict]] = {}
        for r in self.stage_rows(keys):
            out.setdefault(r["dataset_key"], {})[r["stage"]] = r
        return out

    def get_dataset(self, dataset_key: str) -> dict | None:
        rows = self._query(
            "SELECT * FROM datasets WHERE dataset_key = ?", (dataset_key,)
        )
        return rows[0] if rows else None

    def get_stage(self, dataset_key: str, stage: str) -> dict | None:
        rows = self._query(
            "SELECT * FROM stage_status WHERE dataset_key = ? AND stage = ?",
            (dataset_key, stage),
        )
        return rows[0] if rows else None

    def leaf_dir(self, dataset_key: str) -> str | None:
        """Just a dataset's `leaf_dir`: one indexed lookup, where `dashboard_rows`
        builds every row. For hot paths (one request per image)."""
        rows = self._query(
            "SELECT leaf_dir FROM datasets WHERE dataset_key = ?", (dataset_key,)
        )
        return rows[0]["leaf_dir"] if rows else None

    def done_stage_output(self, dataset_key: str, stage: str) -> str | None:
        """`output_path` of a stage that is `done` (None if not done or no path)."""
        rows = self._query(
            "SELECT output_path FROM stage_status "
            "WHERE dataset_key = ? AND stage = ? AND status = 'done'",
            (dataset_key, stage),
        )
        return (rows[0]["output_path"] or None) if rows else None

    # -- the dashboard's view ---------------------------------------------

    def dashboard_rows(
        self, live: dict[str, dict] | None = None, keys: list[str] | None = None
    ) -> list[dict]:
        """One row per dataset (newest `discovered_at` first) with a nested
        `stages` dict keyed by every name in `STAGES` (None where there is no
        row), a derived `overall_status`, `path_context`, and `live`. `live` is the
        live lane's `{leaf_dir: {...}}` map: those rows read "acquiring" and carry
        it as `live`. `keys` restricts it to those datasets without reading the
        rest: a request for one card must not cost as much as the whole page.
        """
        live = live or {}
        stages_by_key = self.stages_by_key(keys)
        rows = []
        for ds in self.datasets(keys):
            have = stages_by_key.get(ds["dataset_key"], {})
            stages = {s: have.get(s) for s in STAGES}
            now_live = live.get(ds["leaf_dir"])
            rows.append(
                {
                    **ds,
                    "stages": stages,
                    "overall_status": "acquiring"
                    if now_live
                    else overall_status(stages, ds.get("signal_flag")),
                    "path_context": path_context(ds["leaf_dir"], ds["root"]),
                    "live": now_live,
                }
            )
        rows.sort(key=lambda r: r["discovered_at"], reverse=True)
        return rows

    def find_dashboard_row(
        self, dataset_key: str, live: dict[str, dict] | None = None
    ) -> dict | None:
        rows = self.dashboard_rows(live, keys=[dataset_key])
        return rows[0] if rows else None

    def change_version(self) -> str:
        """The newest timestamp anywhere in the registry (UTC `isoformat()`, so
        strings compare). Every write that changes what a page shows stamps one
        (a new dataset, a stage starting or finishing); pages poll with the
        version they were rendered at."""
        rows = self._query(
            """SELECT max(v) AS v FROM (
                   SELECT max(discovered_at) AS v FROM datasets
                   UNION ALL SELECT max(started_at) FROM stage_status
                   UNION ALL SELECT max(finished_at) FROM stage_status)"""
        )
        return (rows[0]["v"] if rows else None) or ""

    def changed_keys(self, since: str) -> set[str]:
        """Datasets discovered, or with a stage started or finished, after `since`."""
        return {
            r["dataset_key"]
            for r in self._query(
                """SELECT dataset_key FROM datasets WHERE discovered_at > ?
                   UNION SELECT dataset_key FROM stage_status
                   WHERE started_at > ? OR finished_at > ?""",
                (since, since, since),
            )
        }

    # -- the downstream readers' view (CORE: locate processed volumes) ------

    def deskewed_datasets(self) -> list[dict]:
        """`[{dataset_key, dsr_dir}]` for datasets whose `deskew` stage is `done`
        and recorded an `output_path`. The recorded path is the authority on where
        the output went (in place, or in the mirror when the source was read-only)."""
        rows = self._query(
            """SELECT d.dataset_key, deskew.output_path AS dsr_dir
               FROM datasets d
               JOIN stage_status deskew
                   ON deskew.dataset_key = d.dataset_key
                   AND deskew.stage = 'deskew' AND deskew.status = 'done'"""
        )
        return [r for r in rows if r["dsr_dir"]]

    def mip_done_datasets(self) -> list[dict]:
        """`[{dataset_key, leaf_dir, dsr_dir}]` for datasets whose `mip_encode`
        stage is `done`; `dsr_dir` is the `deskew` stage's recorded `output_path`
        (the folder containing `MIPs/`), None if it has no deskew row."""
        return self._query(
            """SELECT d.dataset_key, d.leaf_dir, deskew.output_path AS dsr_dir
               FROM datasets d
               JOIN stage_status mip
                   ON mip.dataset_key = d.dataset_key
                   AND mip.stage = 'mip_encode' AND mip.status = 'done'
               LEFT JOIN stage_status deskew
                   ON deskew.dataset_key = d.dataset_key AND deskew.stage = 'deskew'"""
        )
