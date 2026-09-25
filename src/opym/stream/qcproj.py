# Ruff style: Compliant
"""Raw-projection sidecars for live QC.

For every staged (t, c) the live lane writes one small `.npz` of projections
of the RAW volume (as streamed, before decon/DSR) into `<leaf>/qc/rawproj/`.
The live QC service (CORE's `celldet-live-qc`, a separate process in its own
venv) reads these instead of the volume itself: the staged frames leave the
RAM disk as soon as their DSR lands, and the projections are ~1/40 of the
volume, ready about a second after the frame arrives rather than after decon.

Raw axes are the receiver's (Z, Y, X) as streamed, which for this OPM are
(scan, tilted, cover): the tilted camera axis is the one carrying depth (see
CORE's `cellcore.opm_geometry`).

Arrays (FORMAT_VERSION 1):

    mip_scan      (tilted, cover)        raw.max(axis=0)
    mip_tilted    (scan, cover)          raw.max(axis=1)
    mip_cover     (scan, tilted)         raw.max(axis=2)
    sum_scan_b2   (tilted//2, cover//2)  raw.sum(axis=0), 2x2-binned, float32
    profile_scan  (scan,)                raw.sum(axis=(1, 2)), float64
    focus_plane   (tilted, cover)        raw[focus_idx]: the scan plane
                                         holding the most signal
Scalars: format_version, t, c, focus_idx, z_step_um, staged_at, shape (3,).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
RAWPROJ_DIR = "rawproj"


def live_qc_enabled() -> bool:
    return os.environ.get("OPYM_LIVE_QC", "") == "1"


def sidecar_name(base_name: str, cidx: int, t: int) -> str:
    return f"{base_name}_C{cidx}_T{t:03d}.npz"


def raw_projections(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Projections of one raw (scan, tilted, cover) volume."""
    raw = np.asarray(raw)
    if raw.ndim != 3:
        raise ValueError(f"raw must be (scan, tilted, cover); got {raw.shape}")
    sum_scan = raw.sum(axis=0, dtype=np.float32)
    h, w = (sum_scan.shape[0] // 2) * 2, (sum_scan.shape[1] // 2) * 2
    b2 = sum_scan[:h, :w].reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))
    profile = raw.sum(axis=(1, 2), dtype=np.float64)
    focus_idx = int(np.argmax(profile))
    return {
        "mip_scan": raw.max(axis=0),
        "mip_tilted": raw.max(axis=1),
        "mip_cover": raw.max(axis=2),
        "sum_scan_b2": b2,
        "profile_scan": profile,
        "focus_plane": np.array(raw[focus_idx]),
        "focus_idx": np.int64(focus_idx),
        "shape": np.array(raw.shape, dtype=np.int64),
    }


def write_sidecar(
    raw: np.ndarray, dst: Path, *, t: int, c: int, z_step_um: float
) -> Path:
    """Write `raw`'s projections to `dst` under a temp name, then rename it
    into place, so a reader never sees a partial file."""
    arrays = raw_projections(raw)
    arrays.update(
        format_version=np.int64(FORMAT_VERSION),
        t=np.int64(t),
        c=np.int64(c),
        z_step_um=np.float64(z_step_um),
        staged_at=np.float64(time.time()),
    )
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.writing")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, dst)
    return dst


def read_sidecar(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
