"""When each timepoint was acquired: read it from the raw store, keep it processed.

The rig records the time of every volume. Two raw layouts carry it:

* ``t`` -- the rig's own writer: float64 milliseconds since the acquisition began,
  one value per volume (present in stores copied over by the SSH backfill).
* ``t_epoch_s`` -- the stream receiver's raw mirror: the FRAME ``timestamp`` of each
  volume, unix seconds on the rig's clock, NaN until that volume arrives (see
  ``rawmirror.write_epoch``).

The processed (DSR) store used to drop all of it: its time axis was a placeholder
1.0 s, or one nominal interval for the whole run. ``sync_processed`` copies the
per-volume times into the processed store's root ``.zattrs["opym_time"]`` and sets
the time axis scale of every multiscales block to the real median interval, so a
viewer (or Cell Cinema) can read real elapsed seconds without the raw store.

``opym_time`` is ``{"times_s": [...], "interval_s": x, "unit": "s", "origin":
"first volume", "source": "...", "complete": bool}``; ``times_s`` has ``None`` for a
timepoint whose time is not known yet. Times are seconds since the first volume, so
the first is 0.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import struct
from pathlib import Path

import numpy as np

RAW_MS = "t"
RAW_EPOCH = "t_epoch_s"
ATTR = "opym_time"
# The backfill mirrors a dataset's absolute path below this folder.
OUTPUTS_MARKER = "/opym_backfill/outputs/"


# A real unix time is far above this (1.79e9 in 2026); a client that never sets the
# FRAME timestamp sends 0.0, which must read as "not recorded", not as 1970.
_MIN_EPOCH_S = 1e9


def _known(v: float, i: int, raw: str) -> bool:
    if not math.isfinite(v):
        return False
    if raw == RAW_EPOCH:
        return v > _MIN_EPOCH_S
    return not (i > 0 and v == 0.0)  # unwritten ms entries are 0


class ArrayNeedsZarrError(ValueError):
    """A 1-D array that the zarr-free reader refuses (compressed, filtered, not
    little-endian float64, zarr v3...) and no ``zarr`` is installed to read it."""


def read_f8_array(array_dir) -> list | None:
    """A 1-D little-endian float64 zarr v2 array read WITHOUT zarr, or None if
    there is no such array (no ``.zarray`` and no ``zarr.json``).

    Light services (the dashboard) and ChimeraX read the rig's ``t`` and the
    receiver's ``t_epoch_s`` arrays this way: both are written uncompressed, one
    chunk file per ``chunks`` values. Missing or partly written chunks read as the
    array's fill value (NaN for ``t_epoch_s``, 0.0 for ``t``), exactly as zarr
    would. Anything else raises `ArrayNeedsZarrError` naming why: a compressed
    array is never guessed at.
    """
    array_dir = Path(array_dir)
    zarray = array_dir / ".zarray"
    if not zarray.is_file():
        if (array_dir / "zarr.json").is_file():
            raise ArrayNeedsZarrError(
                f"{array_dir}: zarr v3 array, reading it needs zarr"
            )
        return None
    try:
        meta = json.loads(zarray.read_text())
    except (OSError, ValueError) as e:
        raise ArrayNeedsZarrError(f"{array_dir}: unreadable .zarray ({e})") from None
    if meta.get("compressor") or meta.get("filters"):
        raise ArrayNeedsZarrError(
            f"{array_dir}: compressed array (compressor={meta.get('compressor')!r}, "
            f"filters={meta.get('filters')!r}), reading it needs zarr"
        )
    shape, chunks = meta.get("shape") or [], meta.get("chunks") or []
    sep = meta.get("dimension_separator", ".")
    if (
        meta.get("dtype") != "<f8"
        or len(shape) != 1
        or len(chunks) != 1
        or sep not in (".", "/")
    ):
        raise ArrayNeedsZarrError(
            f"{array_dir}: not a 1-D little-endian float64 v2 array "
            f"(dtype={meta.get('dtype')!r}, shape={shape!r}), reading it needs zarr"
        )
    n, chunk = int(shape[0]), int(chunks[0])
    if chunk < 1:
        raise ArrayNeedsZarrError(f"{array_dir}: bad chunk size {chunk}")
    fill_value = meta.get("fill_value")
    fill = float("nan") if fill_value in ("NaN", None) else float(fill_value)
    out: list = []
    for i in range(math.ceil(n / chunk)):
        want = min(chunk, n - i * chunk)
        try:
            raw = (array_dir / str(i)).read_bytes()
        except OSError:
            out += [fill] * want
            continue
        if len(raw) < 8 * chunk:  # partly written: whole chunks are stored padded
            out += [fill] * want
            continue
        out += list(struct.unpack(f"<{chunk}d", raw[: 8 * chunk]))[:want]
    return out


def read_1d(array_dir, *, use_zarr: bool = True) -> list | None:
    """The values of a small 1-D float array: the zarr-free reader first, ``zarr``
    when the array is one it refuses and zarr is installed. None: no such array.
    Raises `ArrayNeedsZarrError` when the array needs zarr and zarr is absent
    (or ``use_zarr`` is False); an array zarr cannot read is None.
    """
    array_dir = Path(array_dir)
    try:
        return read_f8_array(array_dir)
    except ArrayNeedsZarrError as refusal:
        if not use_zarr:
            raise
        try:
            import zarr
        except ImportError:
            raise refusal from None
        try:
            vals = np.asarray(zarr.open(str(array_dir), mode="r")[:])
            return [float(v) for v in vals.ravel()]
        except Exception:  # noqa: BLE001 - an unreadable array is "no times"
            return None


def read_raw_times(raw_store, *, strict: bool = True) -> list | None:
    """Seconds of each timepoint in the raw store's own origin (None: not yet known).

    Returns None if the store has no times at all. Works without zarr for the
    stores the rig and the receiver write (see `read_f8_array`); a compressed one
    uses zarr when installed, else raises `ArrayNeedsZarrError` -- or, with
    ``strict=False``, reads as "no times" (a page that must render anyway).
    """
    raw_store = Path(raw_store)
    for name, scale in ((RAW_MS, 1e-3), (RAW_EPOCH, 1.0)):
        try:
            vals = read_1d(raw_store / name)
        except ArrayNeedsZarrError:
            if strict:
                raise
            continue
        if vals is None:
            continue
        out = [
            round(float(v) * scale, 4) if _known(float(v), i, name) else None
            for i, v in enumerate(vals)
        ]
        if any(v is not None for v in out):
            return out
    return None


def combine(per_store: list) -> list:
    """One time per timepoint from several channels: the earliest (volume start)."""
    n = max((len(s) for s in per_store), default=0)
    out = []
    for i in range(n):
        known = [s[i] for s in per_store if i < len(s) and s[i] is not None]
        out.append(min(known) if known else None)
    return out


def relative(times: list) -> list:
    """Seconds since the first known timepoint (so the first is 0)."""
    base = next((v for v in times if v is not None), None)
    if base is None:
        return list(times)
    return [None if v is None else round(v - base, 4) for v in times]


def interval_s(times: list) -> float | None:
    """The median spacing between consecutive known timepoints.

    Taken per timepoint, so a gap of several timepoints does not skew it.
    """
    idx = [(i, v) for i, v in enumerate(times) if v is not None]
    steps = [
        (b - a) / (j - i)
        for (i, a), (j, b) in zip(idx, idx[1:], strict=False)
        if j > i and b > a
    ]
    return round(statistics.median(steps), 4) if steps else None


def nominal_times(n_frames: int, step_s: float) -> list:
    """``n_frames`` evenly spaced times from the store's declared time-axis step.

    Nominal, not measured: a measured time (the ``opym_time`` record, the raw
    store) always beats it, and callers label the source ``"nominal"``.
    """
    return [round(i * float(step_s), 4) for i in range(int(n_frames))]


def from_raw_stores(raw_stores, *, strict: bool = True) -> list | None:
    """Times (seconds since the first volume) from whichever raw stores have them."""
    lists = [t for t in (read_raw_times(s, strict=strict) for s in raw_stores) if t]
    if not lists:
        return None
    times = relative(combine(lists))
    return times if any(v is not None for v in times) else None


def resolve_times(
    attrs: dict | None,
    raw_stores=(),
    *,
    n_frames: int | None = None,
    axis_step_s: float | None = None,
    strict: bool = False,
) -> tuple[list | None, str | None]:
    """``(times_s, source)`` for a processed store, first that exists:

    1. the store's ``opym_time`` record (``attrs`` = its root attributes):
       source ``"store:<record source>"``;
    2. the raw stores' own measured per-volume times: ``"raw"``;
    3. the store's declared time-axis step (``axis_step_s``) times the frame
       number, ``n_frames`` long: ``"nominal"``.

    ``(None, None)`` when nothing is known (the page then shows no time stamp).
    ``strict=False`` (default, for pages that must render): a raw store that needs
    zarr when none is installed counts as having no times; pass True to raise
    `ArrayNeedsZarrError` instead.
    """
    rec = (attrs or {}).get(ATTR)
    if isinstance(rec, dict) and rec.get("times_s"):
        return rec["times_s"], "store:" + str(rec.get("source") or ATTR)
    raw = from_raw_stores([Path(s) for s in raw_stores], strict=strict)
    if raw is not None:
        return raw, "raw"
    if axis_step_s is not None and axis_step_s > 0 and n_frames:
        return nominal_times(n_frames, axis_step_s), "nominal"
    return None, None


# -- the processed store ------------------------------------------------------------


def _read_attrs(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _write_attrs(path: Path, attrs: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(attrs, indent=2))
    os.replace(tmp, path)


def _set_time_scale(attrs: dict, interval: float) -> bool:
    """Set the t scale of every multiscales dataset with a time axis (True: changed)."""
    changed = False
    for ms in attrs.get("multiscales") or []:
        names = [a.get("name") for a in ms.get("axes", [])]
        if "t" not in names:
            continue
        k = names.index("t")
        for ds in ms.get("datasets", []):
            for tr in ds.get("coordinateTransformations", []):
                scale = tr.get("scale", [])
                if (
                    tr.get("type") == "scale"
                    and len(scale) > k
                    and scale[k] != interval
                ):
                    scale[k] = interval
                    changed = True
        for a in ms.get("axes", []):
            if a.get("name") == "t" and a.get("unit") != "second":
                a["unit"] = "second"
                changed = True
    return changed


def write_times(
    store, times_s: list, *, source: str, complete: bool | None = None
) -> bool:
    """Record per-timepoint times in a processed store.

    False if it is not a zarr group or there is nothing known to write.
    """
    store = Path(store)
    if not (store / ".zgroup").is_file() and not (store / "zarr.json").is_file():
        return False
    if not any(v is not None for v in times_s):
        return False
    itv = interval_s(times_s)
    root = store / ".zattrs"
    attrs = _read_attrs(root)
    record = {
        "times_s": times_s,
        "unit": "s",
        "origin": "first volume",
        "source": source,
        "complete": (
            all(v is not None for v in times_s) if complete is None else bool(complete)
        ),
    }
    if itv is not None:
        record["interval_s"] = itv
    changed = attrs.get(ATTR) != record
    attrs[ATTR] = record
    if itv is not None:
        changed |= _set_time_scale(attrs, itv)
    if changed:
        _write_attrs(root, attrs)
    if itv is not None:  # series groups of the bioformats2raw layout have their own
        for sub in sorted(p for p in store.iterdir() if (p / ".zattrs").is_file()):
            sub_attrs = _read_attrs(sub / ".zattrs")
            if _set_time_scale(sub_attrs, itv):
                _write_attrs(sub / ".zattrs", sub_attrs)
    return True


def read_times(store) -> dict | None:
    """The ``opym_time`` record of a processed store, or None."""
    rec = _read_attrs(Path(store) / ".zattrs").get(ATTR)
    return rec if isinstance(rec, dict) and rec.get("times_s") else None


def sync_processed(processed_store, raw_stores, *, source: str | None = None) -> bool:
    """Copy the raw stores' per-volume times into a processed store.

    True if times were written.
    """
    raw_stores = [Path(s) for s in raw_stores if s]
    times = from_raw_stores(raw_stores)
    if times is None:
        return False
    names = sorted(s.name for s in raw_stores if read_raw_times(s))
    return write_times(
        processed_store, times, source=source or "raw store " + ", ".join(names)
    )


# -- finding the raw stores of a processed store ------------------------------------


def dataset_dir(processed_store) -> Path:
    """<dataset>/viewer/<base>_dsr.ome.zarr -> <dataset>"""
    p = Path(processed_store)
    return p.parent.parent if p.parent.name == "viewer" else p.parent


def original_dir(path) -> Path:
    """Undo the backfill's mirroring (outputs/<original absolute path>)."""
    s = str(path)
    if OUTPUTS_MARKER in s:
        return Path("/" + s.split(OUTPUTS_MARKER, 1)[1].lstrip("/"))
    return Path(s)


def _channel_labels(processed_store) -> list:
    """Channel labels from a processed store's omero block (root or first series)."""
    p = Path(processed_store)
    for attrs_path in (p / ".zattrs", p / "0" / ".zattrs"):
        chans = (_read_attrs(attrs_path).get("omero") or {}).get("channels") or []
        labels = [c.get("label") for c in chans if c.get("label")]
        if labels:
            return labels
    return []


# A raw channel store is <base>_<ChannelName>_<wavelength>: after the base, exactly
# one underscore, then the wavelength. Anything with more (Gapvd1_Unstim_001_mNG_488nm
# next to the base Gapvd1_Unstim) belongs to a different dataset.
_CHANNEL_TAIL = re.compile(r"^[A-Za-z][A-Za-z0-9-]*_\d+(nm)?$")


def raw_stores_for(processed_store) -> list:
    """The raw channel stores of a processed store.

    They sit next to the dataset folder: <session>/<cell>_<channel>.ome.zarr.
    """
    processed_store = Path(processed_store)
    labels = _channel_labels(processed_store)
    here = dataset_dir(processed_store)
    for ds in dict.fromkeys([here, original_dir(here)]):
        # channel labels are often the raw store names
        named = [ds.parent / f"{label}.ome.zarr" for label in labels]
        named = [p for p in named if p.is_dir()]
        if named:
            return named
        base = ds.name
        tail = slice(len(base) + 1, -len(".ome.zarr"))
        found = sorted(
            p
            for p in ds.parent.glob(f"{base}_*.ome.zarr")
            if p.is_dir() and _CHANNEL_TAIL.match(p.name[tail])
        )
        if found:
            return found
    return []
