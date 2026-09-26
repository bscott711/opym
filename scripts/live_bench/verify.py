"""Check a replay end to end: every (t, c) of the raw store on Argus
bit-identical to what was sent (the replay report's SHA-1s), and every
(t, c) processed into the view store and archived to GPFS.

usage: verify.py NAME   (reads ~/NAME.replay.json; prints one JSON line)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import zarr

from opym.ome_zarr_writer import read_progress

LV = Path("/dev/shm/opym_lv")
RAW_ROOT = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_lv/raw")


def newest(root: Path, pattern: str) -> Path | None:
    found = [p for p in root.glob(pattern) if p.is_dir()] if root.is_dir() else []
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def main() -> None:
    name = sys.argv[1]
    rep = json.loads((Path.home() / f"{name}.replay.json").read_text())
    base = rep["base_name"]
    out: dict = {"name": name, "timepoints": rep["timepoints"]}
    ok, bad = 0, []
    for c, ch in enumerate(rep["channels"]):
        store = newest(LV / "stage", f"{base}*_{ch}.ome.zarr") or newest(
            RAW_ROOT, f"{base}*_{ch}.ome.zarr"
        )
        if store is None:
            bad.append(f"no raw store for {ch}")
            continue
        arr = zarr.open(str(store / "p0"), mode="r")
        for t in range(rep["timepoints"]):
            got = hashlib.sha1(np.ascontiguousarray(arr[t]).tobytes()).hexdigest()
            if got == rep["sha1"].get(f"{t},{c}"):
                ok += 1
            else:
                bad.append(f"{t},{c}")
    out["raw_bit_identical"] = f"{ok}/{len(rep['sha1'])}"
    out["raw_bad"] = bad[:20]
    session = newest(LV / "view", "*")
    view = next(session.glob("*_dsr.ome.zarr"), None) if session else None
    want = rep["timepoints"] * len(rep["channels"])
    if view is not None:
        prog = read_progress(view) or {}
        out["view_store"] = str(view)
        out["view_done"] = f"{len(prog.get('done', []))}/{want}"
        out["view_state"] = prog.get("state")
        latest = json.loads((LV / "jobs" / "live_latest.json").read_text())
        archive = Path(latest["store"])
        aprog = read_progress(archive) or {}
        out["archive_store"] = str(archive)
        out["archive_done"] = f"{len(aprog.get('done', []))}/{want}"
    print(json.dumps(out))


if __name__ == "__main__":
    main()
