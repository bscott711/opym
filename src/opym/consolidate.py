"""
OME-Zarr consolidation for opym pipeline output.

After MATLAB finishes Decon→DSR per-frame zarrs, this module assembles them
into a single OME-NGFF v0.4 store without copying any data — only hardlinks
(or symlinks as fallback) are created.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path


def consolidate_to_ome_zarr(
    decon_dir: Path,
    base_name: str,
    z_step_um: float,
    xy_pixel_um: float,
    t_interval_s: float,
    channel_names: list[str],
) -> bool:
    """
    Assemble per-T×C zarr arrays into one OME-NGFF v0.4 store.

    Reads  decon_dir/<base_name>_T{t:04d}_C{c}.zarr  (Z, Y, X)
    Writes decon_dir/<base_name>.ome.zarr             (T, C, Z, Y, X)

    No data is copied. Chunk files are hardlinked with T and C indices
    prepended to match zarr v2's t.c.iz.iy.ix naming, falling back to
    symlinks if the source and destination are on different filesystems.

    Returns True on success, False if no matching zarrs were found.
    """
    pattern = re.compile(rf"^{re.escape(base_name)}_T(\d+)_C(\d+)\.zarr$")
    entries: dict[tuple[int, int], Path] = {}
    for p in decon_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            entries[(int(m.group(1)), int(m.group(2)))] = p

    if not entries:
        print(f"[consolidate] No matching zarrs in {decon_dir} for '{base_name}'")
        return False

    all_t = sorted({t for t, _ in entries})
    all_c = sorted({c for _, c in entries})
    num_t, num_c = len(all_t), len(all_c)

    # Read source .zarray to get exact dtype/compressor/chunks — no transcoding
    first_path = next(iter(entries.values()))
    zarray_src_path = first_path / ".zarray"
    if not zarray_src_path.exists():
        print(f"[consolidate] ERROR: {zarray_src_path} not found — invalid zarr array")
        return False
    with open(zarray_src_path) as f:
        src_meta = json.load(f)

    zyx_shape: list[int] = src_meta["shape"]
    zyx_chunks: list[int] = src_meta["chunks"]
    print(
        f"[consolidate] Linking {num_t}T × {num_c}C, each {zyx_shape} {src_meta['dtype']} "
        f"(chunks {zyx_chunks}) — no data copy"
    )

    out_path = decon_dir / f"{base_name.removesuffix('.ome')}.ome.zarr"
    arr_path = out_path / "0"
    arr_path.mkdir(parents=True, exist_ok=True)

    (out_path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))

    # 5D array: chunks [1, 1, Z, Y, X] so every T×C source maps to exactly one
    # "super-chunk" and file names iz.iy.ix → t.c.iz.iy.ix require no reshaping.
    zarray_5d = {
        "zarr_format": 2,
        "shape": [num_t, num_c, *zyx_shape],
        "chunks": [1, 1, *zyx_chunks],
        "dtype": src_meta["dtype"],
        "compressor": src_meta.get("compressor"),
        "fill_value": src_meta.get("fill_value", 0),
        "order": src_meta.get("order", "C"),
        "filters": src_meta.get("filters"),
    }
    (arr_path / ".zarray").write_text(json.dumps(zarray_5d))

    n_linked = 0
    use_symlinks = False
    for t_idx, t in enumerate(all_t):
        for c_idx, c in enumerate(all_c):
            if (t, c) not in entries:
                print(f"[consolidate]   WARNING: missing T={t} C={c}")
                continue
            src_zarr = entries[(t, c)]
            for chunk_file in src_zarr.iterdir():
                if chunk_file.name.startswith("."):
                    continue
                dst = arr_path / f"{t_idx}.{c_idx}.{chunk_file.name}"
                if dst.exists():
                    continue
                if not use_symlinks:
                    try:
                        os.link(chunk_file, dst)
                        n_linked += 1
                        continue
                    except OSError:
                        use_symlinks = True
                dst.symlink_to(chunk_file.resolve())
                n_linked += 1
        print(f"[consolidate]   Linked T={t:04d}")

    # OME-NGFF v0.4. After DSR the output is ~isotropic at xy_pixel_um in all
    # three spatial dimensions; z_step_um describes the pre-DSR raw data only.
    ch_labels = channel_names if len(channel_names) == num_c else [f"C{c}" for c in all_c]
    (out_path / ".zattrs").write_text(json.dumps({
        "multiscales": [{
            "version": "0.4",
            "name": base_name,
            "axes": [
                {"name": "t", "type": "time",    "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space",   "unit": "micrometer"},
                {"name": "y", "type": "space",   "unit": "micrometer"},
                {"name": "x", "type": "space",   "unit": "micrometer"},
            ],
            "datasets": [{
                "path": "0",
                "coordinateTransformations": [{
                    "type": "scale",
                    "scale": [t_interval_s, 1.0, xy_pixel_um, xy_pixel_um, xy_pixel_um],
                }],
            }],
        }],
        "omero": {
            "channels": [
                {"label": name, "active": True, "color": col}
                for name, col in zip(
                    ch_labels, ["00FF00", "FF00FF", "0000FF", "FF0000"]
                )
            ]
        },
    }, indent=2))

    link_type = "symlinks" if use_symlinks else "hardlinks"
    print(f"[consolidate] Done: {out_path} ({n_linked} chunks via {link_type})")
    return True


def run_pending_consolidations(base_dir: Path) -> int:
    """
    Scan completed and failed job tickets for datasets awaiting consolidation.

    Each ticket's `dataDir` is checked for a `.opym_consolidate.json` sidecar
    written by the pipeline CLI.  If present, consolidation runs and the sidecar
    is removed so subsequent calls skip already-consolidated datasets.

    Returns the number of datasets successfully consolidated.
    """
    completed_dir = base_dir / "completed"
    failed_dir = base_dir / "failed"

    # Collect unique dataDir values from all tickets
    data_dirs: set[Path] = set()
    for ticket_dir in (completed_dir, failed_dir):
        if not ticket_dir.exists():
            continue
        for ticket in ticket_dir.glob("*.json"):
            try:
                with open(ticket) as f:
                    payload = json.load(f)
                data_dir = payload.get("dataDir")
                if data_dir:
                    data_dirs.add(Path(data_dir))
            except Exception:
                pass

    n_consolidated = 0
    for data_dir in data_dirs:
        sidecar = data_dir / ".opym_consolidate.json"
        if not sidecar.exists():
            continue

        try:
            with open(sidecar) as f:
                params = json.load(f)
        except Exception as e:
            print(f"[consolidate] Failed to read sidecar {sidecar}: {e}")
            continue

        expected = set(params.get("expected_zarrs", []))
        try:
            present = {
                p.name for p in data_dir.iterdir()
                if p.suffix == ".zarr" and p.is_dir() and not p.name.endswith(".ome.zarr")
            }
        except Exception as e:
            print(f"[consolidate] Cannot list {data_dir}: {e}")
            continue

        missing = expected - present
        if missing:
            print(
                f"[consolidate] {data_dir.name}: {len(missing)}/{len(expected)} zarrs still missing — "
                "consolidating what's present"
            )

        log_path = data_dir / "consolidation.log"
        start = time.time()
        try:
            with open(log_path, "a") as log_fh:
                import sys
                _orig_stdout = sys.stdout
                sys.stdout = _TeeWriter(sys.stdout, log_fh)
                try:
                    success = consolidate_to_ome_zarr(
                        decon_dir=data_dir,
                        base_name=params["base_name"],
                        z_step_um=params["z_step_um"],
                        xy_pixel_um=params["xy_pixel_um"],
                        t_interval_s=params["t_interval_s"],
                        channel_names=params.get("channel_names", []),
                    )
                finally:
                    sys.stdout = _orig_stdout
        except Exception as e:
            print(f"[consolidate] ERROR consolidating {data_dir.name}: {e}")
            success = False

        elapsed = time.time() - start
        if success:
            sidecar.unlink(missing_ok=True)
            n_consolidated += 1
            print(f"[consolidate] {data_dir.name}: done in {elapsed:.1f}s")
        else:
            print(f"[consolidate] {data_dir.name}: failed after {elapsed:.1f}s")

    return n_consolidated


class _TeeWriter:
    """Write to two streams simultaneously (stdout + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()
