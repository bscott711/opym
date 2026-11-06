"""Golden vectors and fixtures for opym's read-side contracts (light: stdlib only).

Consumers that used to carry their own copy (opym-dashboard, opym-viewer, CORE,
cell-cinema, bioimaging) keep a regression test on these, so a change to the shared
code that breaks one of them fails here first.

* registry: `build_registry` makes a tiny registry with `StatusRegistry` whose
  datasets cover every overall status, and returns what each must read as.
* timeline: the raw ``t`` (ms) and ``t_epoch_s`` vectors a rig / the receiver write,
  the times and ``opym_time`` record they must give, and `write_raw_store`, which
  writes such a raw store as plain JSON plus raw float64 chunks (no zarr needed).
* output base: `output_base_cases` builds the directory layouts of
  ``opym.utils.resolve_output_base`` and what each must resolve to.
"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# A key with a space, '#', '?' and '%' in it: the URL key and the sqlite URI both
# have to survive it.
AWKWARD_DATASET_KEY = "/data/lab mate/Cell #1? 100%/Cell_1"


def build_registry(path: Path | str) -> dict[str, str]:
    """Write a small registry at ``path`` (via `StatusRegistry`, the writer) and
    return ``{dataset_key: expected overall_status}``.

    One dataset per overall status the reader can derive (everything except
    'acquiring', which comes from the live lane): done, failed, blocked,
    reprocessing (a stale `done`, and a reset `pending` that kept its
    output_path), running, in_progress (a `pending` stage with no output yet),
    and pending (no stage rows). The last two datasets carry the awkward key and a
    flag with timepoint counts.
    """
    from opym.registry import StatusRegistry

    expected: dict[str, str] = {}

    def add(reg, key, status, root="/data/raw", flag=None, counts=(None, None)):
        reg.register_dataset(
            key, root=root, leaf_dir=key, master_file=f"{key}/{Path(key).name}.ome.tif"
        )
        if flag is not None:
            reg.set_triage(
                key,
                signal_flag=flag,
                expected_timepoints=counts[0],
                actual_timepoints=counts[1],
            )
        expected[key] = status

    with StatusRegistry(path) as reg:
        key = "/data/raw/day1/done"
        add(reg, key, "done", flag="ok", counts=(10, 10))
        for stage in ("roi_detect", "crop_zarr", "deskew", "mip_encode"):
            reg.start_stage(key, stage)
            reg.finish_stage(
                key, stage, status="done", output_path=f"{key}/out/{stage}"
            )

        key = "/data/raw/day1/failed"
        add(reg, key, "failed", flag="unknown")
        reg.start_stage(key, "roi_detect")
        reg.finish_stage(key, "roi_detect", status="failed", error="corrupt TIFF")

        key = "/data/raw/day1/blocked"
        add(reg, key, "blocked", flag="blocked", counts=(5, 5))
        reg.start_stage(key, "deskew")
        reg.finish_stage(
            key, "deskew", status="failed", error="cannot determine z step"
        )

        key = "/data/raw/day1/stale"
        add(reg, key, "reprocessing")
        for stage in ("deskew", "mip_encode"):
            reg.start_stage(key, stage)
            reg.finish_stage(
                key, stage, status="done", output_path=f"{key}/out/{stage}"
            )
        reg.start_stage(key, "deskew")  # re-submitted after the MIPs were encoded

        key = "/data/raw/day1/reset"
        add(reg, key, "reprocessing")
        reg.start_stage(key, "mip_encode")
        reg.finish_stage(key, "mip_encode", status="done", output_path=f"{key}/out/mip")
        reg.reset_stage(key, "mip_encode")

        key = "/data/raw/day1/running"
        add(reg, key, "running")
        reg.start_stage(key, "crop_zarr")

        key = "/data/raw/day1/queued"
        add(reg, key, "in_progress")
        reg.start_stage(key, "deskew")
        reg.finish_stage(key, "deskew", status="done", output_path=f"{key}/out/deskew")
        reg.start_stage(key, "mip_encode")
        reg.reset_stage(key, "mip_encode")  # pending, never produced output

        add(reg, "/data/raw/day1/untouched", "pending")
        add(
            reg,
            AWKWARD_DATASET_KEY,
            "pending",
            root="/data/lab mate",
            flag="partial",
            counts=(100, 40),
        )
    return expected


# ---------------------------------------------------------------------------
# Timeline: raw stores
# ---------------------------------------------------------------------------

RAW_STORE_NAME = "Cell_X_GFP_488.ome.zarr"

# The rig's own writer: float64 ms since the acquisition began; unwritten slots read 0.0
# (the array fill value). Volume 5 is not written yet.
RAW_MS = [740.8436, 8000.0, 15500.0, 23000.0, 0.0]
RAW_MS_SECONDS = [0.7408, 8.0, 15.5, 23.0, None]
RAW_MS_RELATIVE = [0.0, 7.2592, 14.7592, 22.2592, None]
RAW_MS_INTERVAL_S = 7.5  # median of the per-volume steps 7.2592, 7.5, 7.5

# The receiver's mirror: unix seconds of each volume's FRAME timestamp, NaN until it
# arrives.
RAW_EPOCH = [1790000000.25, 1790000007.75, 1790000015.25, math.nan, math.nan]
RAW_EPOCH_SECONDS = [1790000000.25, 1790000007.75, 1790000015.25, None, None]
RAW_EPOCH_RELATIVE = [0.0, 7.5, 15.0, None, None]
RAW_EPOCH_INTERVAL_S = 7.5

# A client that never sets the FRAME timestamp sends 0.0: "not recorded", not 1970.
RAW_EPOCH_UNSET = [0.0, 0.0, 0.0]


def expected_opym_time(relative: list, interval_s: float, source: str) -> dict:
    """The ``opym_time`` record `write_times` must store for ``relative`` times."""
    return {
        "times_s": relative,
        "unit": "s",
        "origin": "first volume",
        "source": source,
        "complete": all(v is not None for v in relative),
        "interval_s": interval_s,
    }


EXPECTED_OPYM_TIME_MS = expected_opym_time(
    RAW_MS_RELATIVE, RAW_MS_INTERVAL_S, f"raw store {RAW_STORE_NAME}"
)
EXPECTED_OPYM_TIME_EPOCH = expected_opym_time(
    RAW_EPOCH_RELATIVE, RAW_EPOCH_INTERVAL_S, f"raw store {RAW_STORE_NAME}"
)


def write_f8_array(
    array_dir: Path | str,
    values: list,
    *,
    chunks: int | None = None,
    fill_value: float = 0.0,
    attrs: dict | None = None,
) -> None:
    """A 1-D uncompressed little-endian float64 zarr v2 array, written by hand.

    A chunk whose values are all the fill value is not written (as zarr does), so a
    reader's missing-chunk path is exercised too. Chunk files are padded to ``chunks``.
    """
    array_dir = Path(array_dir)
    array_dir.mkdir(parents=True, exist_ok=True)
    n = len(values)
    chunk = chunks or max(n, 1)
    nan_fill = math.isnan(fill_value)
    meta = {
        "zarr_format": 2,
        "shape": [n],
        "chunks": [chunk],
        "dtype": "<f8",
        "compressor": None,
        "fill_value": "NaN" if nan_fill else fill_value,
        "filters": None,
        "order": "C",
    }
    (array_dir / ".zarray").write_text(json.dumps(meta))
    (array_dir / ".zattrs").write_text(json.dumps(attrs or {}))
    for i in range(math.ceil(n / chunk)):
        part = [float(v) for v in values[i * chunk : (i + 1) * chunk]]
        if all(math.isnan(v) if nan_fill else v == fill_value for v in part):
            continue
        part += [fill_value] * (chunk - len(part))
        (array_dir / str(i)).write_bytes(struct.pack(f"<{chunk}d", *part))


def write_raw_store(
    path: Path | str,
    *,
    ms: list | None = None,
    epoch: list | None = None,
    chunks: int | None = None,
) -> Path:
    """A tiny raw channel store holding only its time arrays: ``t`` (rig ms, fill 0.0)
    and/or ``t_epoch_s`` (receiver, fill NaN). No zarr needed to write or to read it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (path / ".zattrs").write_text(json.dumps({}))
    if ms is not None:
        write_f8_array(
            path / "t", ms, chunks=chunks, fill_value=0.0, attrs={"units": "ms"}
        )
    if epoch is not None:
        write_f8_array(path / "t_epoch_s", epoch, chunks=chunks, fill_value=math.nan)
    return path


# ---------------------------------------------------------------------------
# resolve_output_base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputBaseCase:
    name: str
    leaf_dir: Path
    expected: Path
    note: str = ""


def make_unwritable(path: Path) -> bool:
    """chmod ``path`` read-only; True if `os.access` now agrees (False for root)."""
    os.chmod(path, 0o555)
    return not os.access(path, os.W_OK)


def restore_permissions(root: Path) -> None:
    """Make every directory under ``root`` writable again so it can be deleted."""
    for p in [root, *root.rglob("*")]:
        if p.is_dir():
            os.chmod(p, 0o755)


def output_base_cases(workdir: Path, mirror_root: Path) -> list[OutputBaseCase]:
    """Directory layouts under ``workdir`` and what ``resolve_output_base(leaf_dir)``
    must return for each, with ``OPYM_OUTPUT_MIRROR_ROOT`` set to ``mirror_root``.

    POSIX only (the mirror reproduces an absolute POSIX path). The unwritable cases
    are left out when `os.access` ignores the chmod (running as root); call
    `restore_permissions(workdir)` before deleting ``workdir``.
    """

    def mirrored(leaf: Path) -> Path:
        return mirror_root / str(leaf.resolve()).lstrip("/")

    cases: list[OutputBaseCase] = []
    writable = workdir / "writable"
    (writable / "day").mkdir(parents=True)

    leaf = writable / "day" / "Cell_001"
    leaf.mkdir()
    cases.append(
        OutputBaseCase(
            "writable leaf", leaf, leaf, "the normal case: output next to the raw data"
        )
    )

    leaf = (
        writable / "day" / "WT_Stim"
    )  # a flat folder's `<day>/<base>`, which nothing creates
    cases.append(
        OutputBaseCase(
            "leaf does not exist yet",
            leaf,
            leaf,
            "writable nearest ancestor: in place; never created",
        )
    )

    leaf = writable / "day" / "a" / "b" / "Cell_9"
    cases.append(
        OutputBaseCase(
            "ancestor rule",
            leaf,
            leaf,
            "several missing levels: the nearest EXISTING ancestor decides",
        )
    )

    leaf = writable / "day" / "Cell_002"
    mirrored(leaf).mkdir(parents=True)
    cases.append(
        OutputBaseCase(
            "already in the mirror",
            leaf,
            mirrored(leaf),
            "output processed into the mirror stays there",
        )
    )

    locked = workdir / "locked"
    (locked / "Cell_1").mkdir(parents=True)
    (locked / "sub").mkdir()
    ok = make_unwritable(locked / "Cell_1") and make_unwritable(locked / "sub")
    if ok:
        leaf = locked / "Cell_1"
        cases.append(
            OutputBaseCase(
                "existing unwritable leaf",
                leaf,
                mirrored(leaf),
                "read-only upload: mirrored",
            )
        )
        leaf = locked / "sub" / "Cell_2"
        cases.append(
            OutputBaseCase(
                "missing leaf under unwritable ancestor",
                leaf,
                mirrored(leaf),
                "ancestor not writable: mirrored",
            )
        )
    return cases
