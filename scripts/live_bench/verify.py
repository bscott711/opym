"""Check a replay end to end: every (t, c) of the raw store on Argus
bit-identical to what was sent (the replay report's SHA-1s), and every
(t, c) processed into the view store and archived to GPFS.

usage: verify.py NAME [--session ID-PREFIX]
  NAME     the replay's --name (reads ~/NAME.replay.json)
  session  default: the newest session in the test stack's trace

Stores are found through the session, not by name: the receiver renames a
session whose name is taken (Cell_001 -> Cell_001_001), and the trace
records the name it used. Prints one JSON line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

from opym.ome_zarr_writer import read_progress
from opym.stream import trace, trace_report

LV = Path("/dev/shm/opym_lv")
RAW_ROOT = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_lv/raw")


def first_existing(*paths: Path) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--session")
    args = ap.parse_args()
    rep = json.loads((Path.home() / f"{args.name}.replay.json").read_text())
    events = trace.read(jobs=LV / "jobs")
    session = trace_report.pick_session(events, args.session, None)
    base = next(
        e["base_name"]
        for e in events
        if e.get("ev") == "frame" and e.get("session_id") == session
    )
    out: dict = {
        "name": args.name,
        "session_id": session,
        "base_name": base,
        "timepoints": rep["timepoints"],
    }
    ok, bad = 0, []
    for c, ch in enumerate(rep["channels"]):
        store = first_existing(
            LV / "stage" / f"{base}_{ch}.ome.zarr",
            RAW_ROOT / args.name / f"{base}_{ch}.ome.zarr",  # PC replay: raw_root/NAME
            RAW_ROOT / f"{base}_{ch}.ome.zarr",  # local replay
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
    want = rep["timepoints"] * len(rep["channels"])
    view = next((LV / "view" / session).glob("*_dsr.ome.zarr"), None)
    if view is not None:
        prog = read_progress(view) or {}
        out["view_done"] = f"{len(prog.get('done', []))}/{want}"
        out["view_state"] = prog.get("state")
    archive = first_existing(
        RAW_ROOT / args.name / base / "viewer" / f"{base}_dsr.ome.zarr",
        RAW_ROOT / base / "viewer" / f"{base}_dsr.ome.zarr",
    )
    if archive is not None:
        aprog = read_progress(archive) or {}
        out["archive_done"] = f"{len(aprog.get('done', []))}/{want}"
        out["archive_state"] = aprog.get("state")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
