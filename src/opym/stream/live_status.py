# Ruff style: Compliant
"""The live lane's hand-off files and names, as consumers read them.

What the live lane (`opym.stream.live`, `opym.stream.live_zarr`) writes for
viewers, the live QC service and the backfill, without the lane's own imports
(light: stdlib only). Consumers import these instead of re-typing the strings.

- `live_latest.json` in the jobs dir points viewers and CORE's live QC at the
  newest session (`LIVE_LATEST_KEYS`, `read_live_latest`).
- `.live_status.json` in a session's DSR directory tells the backfill whether
  the lane is still running (`read_live_status`, `live_status_is_fresh`).
- `T<tttt>_C<c>.npy` view buffers in the RAM-disk store (`buffer_name`).
- QC: `<leaf>/qc/qc_latest.json` (receiver pointer), `live_qc.jsonl` (CORE's
  verdict log) and `rawproj/` (raw-projection sidecars, `opym.stream.qcproj`:
  `RAWPROJ_DIR`, `sidecar_name`, `FORMAT_VERSION`; the fields CORE's
  `RawProjections` reads are listed in that module's docstring).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from opym.stream.qcproj import FORMAT_VERSION as QCPROJ_FORMAT_VERSION
from opym.stream.qcproj import RAWPROJ_DIR, sidecar_name

__all__ = [
    "HEARTBEAT_S",
    "LIVE_LATEST_KEYS",
    "LIVE_LATEST_NAME",
    "LIVE_LATEST_ZARR_KEYS",
    "LIVE_STATUS_NAME",
    "LIVE_STATUS_STALE_S",
    "QCPROJ_FORMAT_VERSION",
    "QC_LATEST_NAME",
    "QC_LOG_NAME",
    "RAWPROJ_DIR",
    "buffer_name",
    "live_status_is_fresh",
    "parse_buffer_name",
    "read_live_latest",
    "read_live_status",
    "sidecar_name",
]

LIVE_STATUS_NAME = ".live_status.json"
LIVE_LATEST_NAME = "live_latest.json"
QC_LATEST_NAME = "qc_latest.json"
QC_LOG_NAME = "live_qc.jsonl"
# A "running" status older than this is treated as abandoned (receiver
# crashed): the backfill then reprocesses the dataset in batch.
LIVE_STATUS_STALE_S = 300.0
HEARTBEAT_S = 30.0

# The keys every `live_latest.json` carries (TIFF lane and zarr lane).
LIVE_LATEST_KEYS = (
    "session_id",
    "base_name",
    "store",
    "dsr_dir",
    "started_at",
    "qc_dir",
    "num_timepoints",
    "n_channels",
    "channel_labels",
    "z_step_um",
)
# The zarr lane adds the RAM-disk store and its buffers ("store" is then the
# archive store).
LIVE_LATEST_ZARR_KEYS = (*LIVE_LATEST_KEYS, "view_store", "buffers_dir")


def read_live_latest(path: Path | str) -> dict | None:
    """The pointer record at `path` (the file, or the jobs dir holding
    `live_latest.json`), or None if missing or unreadable."""
    path = Path(path)
    if path.is_dir():
        path = path / LIVE_LATEST_NAME
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def read_live_status(dsr_dir: Path | str) -> dict | None:
    try:
        return json.loads((Path(dsr_dir) / LIVE_STATUS_NAME).read_text())
    except (OSError, ValueError):
        return None


def live_status_is_fresh(status: dict, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return now - float(status.get("updated_at", 0)) <= LIVE_STATUS_STALE_S


def buffer_name(t: int, c: int) -> str:
    """A (t, c) view buffer's file name (naparym-live reads the same)."""
    return f"T{t:04d}_C{c}.npy"


def parse_buffer_name(name: str) -> tuple[int, int] | None:
    if not (name.startswith("T") and name.endswith(".npy") and "_C" in name):
        return None
    try:
        t, c = name[1:-4].split("_C")
        return int(t), int(c)
    except ValueError:
        return None
