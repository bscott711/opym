#!/usr/bin/env python3
"""List the datasets whose deskew-only output has the old, mirrored handedness.

READ-ONLY dry run (ROADMAP 3a, PROC-F1). Before bioimaging PR #39 a zarr dataset
deskewed without a PSF was written to `<leaf>/zarr_mirror/DSR_nodecon`, a
y-mirror of every decon and live product. This lists every registry dataset
whose `deskew` stage still records that path, so the user can decide what to
re-run. See docs/handedness-rerun.md.

What it touches:
  * the registry, through `opym.registry.RegistryReader` (opened `mode=ro`);
  * `os.stat` on a handful of paths per candidate (is the old output still
    there, has a re-run already written the new one). No directory is walked.
It writes nothing: no registry change, no file, no move. Output goes to stdout;
redirect it to keep a manifest of the old output paths (the registry overwrites
`output_path` when the re-run finishes). Any flag that sounds like a write is
refused.

    python scripts/handedness_rerun_candidates.py [--registry PATH]
        [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--format table|csv|json]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import PurePath

from opym.registry import RegistryReader, registry_path

# The directory pair that marks the old output: `<leaf>/zarr_mirror/DSR_nodecon`.
LEGACY_PARENT = "zarr_mirror"
LEGACY_LEAF = "DSR_nodecon"
# Where bioimaging PR #39 writes the same dataset: deskew-only, and decon.
NEW_DIR = "decon_stage"

# Flags that would make this a writer. Refused by name so the intent is clear.
WRITE_FLAGS = frozenset(
    {
        "--apply", "--write", "--reset", "--execute", "--exec", "--run",
        "--fix", "--force", "--yes", "--delete", "--remove", "--move",
        "--commit", "--rerun", "--submit", "--migrate", "--in-place",
    }
)

COLUMNS = (
    "dataset_key",
    "leaf_dir",
    "deskew_status",
    "finished_at",
    "decon_psf",
    "old_output",
    "old_output_exists",
    "viewer_exists",
    "mip_movies_exists",
    "new_nodecon_exists",
    "new_decon_exists",
    "note",
)


def is_legacy_output(output_path: str | None) -> bool:
    """Whether `output_path` is a `<...>/zarr_mirror/DSR_nodecon` directory."""
    if not output_path:
        return False
    p = PurePath(output_path)
    return p.name == LEGACY_LEAF and p.parent.name == LEGACY_PARENT


def _exists(path: PurePath) -> bool:
    try:
        os.stat(path)
    except OSError:
        return False
    return True


def _note(row: dict) -> str:
    if row["new_nodecon_exists"] or row["new_decon_exists"]:
        return "new output already on disk (a re-run has written it)"
    if row["deskew_status"] != "done":
        return f"deskew is {row['deskew_status']}"
    if not row["old_output_exists"]:
        return "old output path is missing on disk"
    if not row["decon_psf"]:
        return "decon_psf NULL: differs from the service's PSF (provenance mismatch)"
    return "decon_psf set: the service may treat this as current"


def candidates(
    reader: RegistryReader, since: str | None = None, until: str | None = None
) -> list[dict]:
    """One row per dataset whose recorded deskew output is the legacy path.

    `since`/`until` are inclusive `YYYY-MM-DD` bounds on the deskew stage's
    `finished_at`; a row with no finish time is dropped when either is given.
    """
    datasets = {d["dataset_key"]: d for d in reader.datasets()}
    rows = []
    for stage in reader.stage_rows():
        if stage["stage"] != "deskew" or not is_legacy_output(stage["output_path"]):
            continue
        finished = (stage.get("finished_at") or "")[:10]
        if (since or until) and not finished:
            continue
        if since and finished < since:
            continue
        if until and finished > until:
            continue
        key = stage["dataset_key"]
        ds = datasets.get(key, {})
        old = PurePath(stage["output_path"])
        base = old.parent.parent  # the dataset's output base
        row = {
            "dataset_key": key,
            "leaf_dir": ds.get("leaf_dir") or "",
            "deskew_status": stage["status"],
            "finished_at": stage.get("finished_at") or "",
            "decon_psf": ds.get("decon_psf") or "",
            "old_output": str(old),
            "old_output_exists": _exists(old),
            "viewer_exists": _exists(base / "viewer"),
            "mip_movies_exists": _exists(base / "mip_movies"),
            "new_nodecon_exists": _exists(base / NEW_DIR / LEGACY_LEAF),
            "new_decon_exists": _exists(base / NEW_DIR / "Decon" / "DSR_decon"),
        }
        row["note"] = _note(row)
        rows.append(row)
    rows.sort(key=lambda r: (r["finished_at"], r["dataset_key"]))
    return rows


def _refuse_write_flags(argv: list[str]) -> None:
    for arg in argv:
        flag = arg.split("=", 1)[0]
        if flag in WRITE_FLAGS:
            sys.exit(
                f"refusing {flag}: this lister is read-only. The registry reset "
                "and the moves are separate, deliberate steps "
                "(docs/handedness-rerun.md)."
            )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], allow_abbrev=False
    )
    p.add_argument(
        "--registry",
        help="registry.sqlite3 (default: $OPYM_BACKFILL_REGISTRY_PATH, else the "
        "production path)",
    )
    p.add_argument("--since", help="deskew finished on or after YYYY-MM-DD")
    p.add_argument("--until", help="deskew finished on or before YYYY-MM-DD")
    p.add_argument("--format", choices=("table", "csv", "json"), default="table")
    return p


def _print(rows: list[dict], fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(rows, indent=1))
    elif fmt == "csv":
        w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    else:
        for r in rows:
            flags = "".join(
                ch if r[k] else "-"
                for ch, k in (
                    ("O", "old_output_exists"),
                    ("V", "viewer_exists"),
                    ("M", "mip_movies_exists"),
                    ("N", "new_nodecon_exists"),
                    ("D", "new_decon_exists"),
                )
            )
            print(
                f"{r['finished_at'][:10] or '----------'}  {r['deskew_status']:<8} "
                f"{flags}  psf={'set' if r['decon_psf'] else 'NULL'}  "
                f"{r['dataset_key']}"
            )
        print(
            "\nflags: O old output on disk, V viewer/ dir, M mip_movies/ dir, "
            "N new deskew-only output, D new decon output"
        )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    _refuse_write_flags(argv)
    args = _build_parser().parse_args(argv)
    path = args.registry or registry_path()
    with RegistryReader(path) as reader:
        if not reader.exists:
            print(f"no registry at {reader.path}", file=sys.stderr)
            return 1
        rows = candidates(reader, args.since, args.until)
    _print(rows, args.format)
    print(f"{len(rows)} candidate(s) in {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
