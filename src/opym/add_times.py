"""opym-add-times: give processed datasets their real acquisition times.

    opym-add-times PATH [PATH ...]            show what would change (default)
    opym-add-times --apply PATH [PATH ...]    write it

PATH is a processed store (``*_dsr.ome.zarr``), a dataset folder (with ``viewer/``),
or a session folder holding several datasets. For each store the raw channel stores
beside the dataset are read (``t`` from the rig, or ``t_epoch_s`` from the stream
receiver) and the per-volume times and the real interval are written into the
processed store's metadata (see ``opym.timeline``). Pixel data are never touched, and
the ``.zattrs`` files that change are first copied to ``.zattrs.before-times``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import timeline


def find_stores(path: Path) -> list[Path]:
    """The processed stores at PATH: itself, its viewer/, or its datasets."""
    path = Path(path)
    if path.name.endswith(".ome.zarr") and (path / ".zgroup").exists():
        return [path]
    own = sorted((path / "viewer").glob("*_dsr.ome.zarr"))
    if own:
        return own
    return sorted(path.glob("*/viewer/*_dsr.ome.zarr"))


def _backup(store: Path) -> None:
    """Keep the metadata files this tool may change, once (never overwrite)."""
    for attrs in [store / ".zattrs", *store.glob("*/.zattrs")]:
        keep = attrs.with_name(attrs.name + ".before-times")
        if attrs.is_file() and not keep.exists():
            shutil.copy2(attrs, keep)


def describe(store: Path) -> tuple[str, list | None]:
    """(a one-line verdict, the times that would be written or None)."""
    have = timeline.read_times(store)
    raws = timeline.raw_stores_for(store)
    times = timeline.from_raw_stores(raws)
    if times is None:
        where = f"{len(raws)} raw store(s)" if raws else "no raw store found"
        keep = f" (keeps {have['source']})" if have else ""
        return f"no real times to add ({where} without times){keep}", None
    n_known = sum(v is not None for v in times)
    itv = timeline.interval_s(times)
    if have and have.get("times_s") == times:
        return f"already has its {n_known} times (interval {itv} s)", None
    names = ", ".join(r.name for r in raws)
    return f"{n_known}/{len(times)} times, interval {itv} s, from {names}", times


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="opym-add-times",
        description="Add real per-timepoint acquisition times to processed stores.",
    )
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument(
        "--apply", action="store_true", help="write the changes (default: dry run)"
    )
    args = ap.parse_args(argv)

    stores = [s for p in args.paths for s in find_stores(p)]
    if not stores:
        print("no processed stores (*_dsr.ome.zarr) found", file=sys.stderr)
        return 2
    wrote = 0
    for store in stores:
        verdict, times = describe(store)
        action = ""
        if times is not None:
            if args.apply:
                _backup(store)
                raws = timeline.raw_stores_for(store)
                if timeline.sync_processed(store, raws):
                    wrote += 1
                    action = "  -> written"
            else:
                action = "  (dry run; --apply to write)"
        print(f"{store}\n    {verdict}{action}")
    print(f"{len(stores)} store(s); {wrote} written" if args.apply else "dry run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
