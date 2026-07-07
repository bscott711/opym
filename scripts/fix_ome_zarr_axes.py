#!/usr/bin/env python3
"""
Patch existing consolidated .ome.zarr stores whose axes are wrong.

Per-frame zarrs coming out of Decon->DSR are laid out (X, Y, Z), not
(Z, Y, X): orient_zyx_for_dsr (opym.utils) reorders each crop into
PetaKit5D's [ny, nx, nz] convention before staging it for MATLAB, and
Decon->DSR preserves that axis order on the way out. consolidate.py used
to hardcode axes=[t,c,z,y,x] regardless, mislabeling the real depth axis.

Relabeling the NGFF axes metadata alone does not fix this: most consumers
(napari's builtin reader, napari-ome-zarr, plain zarr/dask) ignore axis
semantics and just trust an array's physical axis order when slicing. So
this script makes the true (T, C, Z, Y, X) order physically real, with zero
data movement, using a zarr v2 identity: a "C"-ordered chunk of shape
(X, Y, Z) is byte-for-byte identical to an "F"-ordered chunk of shape
(Z, Y, X). It reverses the declared shape/chunks, flips "order" C<->F, and
renames chunk files' spatial index components to match -- no pixel data is
read, copied, or moved.

Idempotent and safe to re-run: a store whose array order is already "F" is
treated as fixed and skipped, regardless of what its axes names currently
say (this also repairs stores hit by an earlier, label-only version of this
script, which corrected axes names but not the underlying array order).

Usage:
    python fix_ome_zarr_axes.py /path/to/dir_or_store [...] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CANONICAL_AXES = ["t", "c", "z", "y", "x"]


def find_ome_zarr_stores(path: Path) -> list[Path]:
    if path.name.endswith(".ome.zarr"):
        return [path]
    if not path.is_dir():
        return []
    return sorted(p for p in path.rglob("*.ome.zarr") if p.is_dir())


def fix_store(store: Path, dry_run: bool) -> str:
    zattrs_path = store / ".zattrs"
    arr_path = store / "0"
    zarray_path = arr_path / ".zarray"
    if not zattrs_path.exists() or not zarray_path.exists():
        return "skip (missing .zattrs or 0/.zarray)"

    zarray = json.loads(zarray_path.read_text())
    order = zarray.get("order", "C")
    if order == "F":
        return "skip (already correct: order='F')"
    if order != "C":
        return f"skip (unrecognized order {order!r})"

    shape: list[int] = zarray["shape"]
    chunks: list[int] = zarray["chunks"]
    if len(shape) != 5:
        return f"skip (expected 5D array, got shape {shape})"

    new_shape = shape[:2] + list(reversed(shape[2:]))
    new_chunks = chunks[:2] + list(reversed(chunks[2:]))

    attrs = json.loads(zattrs_path.read_text())
    multiscales = attrs.get("multiscales")
    if not multiscales:
        return "skip (no multiscales)"
    axes = multiscales[0].get("axes", [])
    if len(axes) != 5:
        return f"skip (expected 5 axes, got {len(axes)})"

    if not dry_run:
        zarray["shape"] = new_shape
        zarray["chunks"] = new_chunks
        zarray["order"] = "F"
        zarray_path.write_text(json.dumps(zarray))

        for entry, new_name in zip(axes, CANONICAL_AXES, strict=True):
            entry["name"] = new_name
        zattrs_path.write_text(json.dumps(attrs, indent=2))

        for chunk_file in arr_path.iterdir():
            if chunk_file.name.startswith("."):
                continue
            parts = chunk_file.name.split(".")
            t_idx, c_idx, *spatial = parts
            new_name = ".".join([t_idx, c_idx, *reversed(spatial)])
            if new_name != chunk_file.name:
                chunk_file.rename(arr_path / new_name)

    verb = "would fix" if dry_run else "fixed"
    return f"{verb}: shape {shape}->{new_shape}, order 'C'->'F', axes->{CANONICAL_AXES}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Store(s) or directories to scan")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    stores: list[Path] = []
    for p in args.paths:
        stores.extend(find_ome_zarr_stores(p))

    if not stores:
        print("No .ome.zarr stores found.")
        return 1

    n_fixed = 0
    for store in stores:
        result = fix_store(store, args.dry_run)
        print(f"[fix_ome_zarr_axes] {store}: {result}")
        if result.startswith(("fixed", "would fix")):
            n_fixed += 1

    verb = "would fix" if args.dry_run else "fixed"
    print(f"[fix_ome_zarr_axes] {n_fixed}/{len(stores)} store(s) {verb}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
