"""Replay a saved run into the test stack from Argus itself, paced like the rig.

The network hop is measured from the PC (pymmcore-gui's replay); this drives
everything after it without the PC: the same slab FRAMEs the rig's sender
makes (11-plane slabs, blosc), straight into the test receiver.

Pacing is the rig's, measured from the 2026-09-02 MDA saves' frame times:
the whole 488 stack, a gap, the whole 561 stack, then the next timepoint;
the first plane comes `--setup-s` after the run starts. Per plane 35.7 ms,
0.83 s between the stacks, 0.88 s from the end of a timepoint to the next.

usage: local_replay.py NAME [--timepoints N] [--source DIR --base NAME]

Writes ~/NAME.replay.json (SHA-1 per (t, c) and the pacing), like the PC's
replay, for verify.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import zarr

from opym.stream.client import StreamSender

SOURCE = Path(
    "/mmfs1/scratch/jacks.local/microscopy/"
    "20260925-SVO-memNG-mScar2xFYVE-FLM-Macropinocytosis"
)
CHANNELS = ["GFP_488", "mScarlet_561"]
PLANE_S, GAP_S, NEXT_T_S, SETUP_S = 0.0357, 0.83, 0.88, 5.14
SLAB_BYTES = 16 * 1024 * 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--base", default="Cell_001_001")
    ap.add_argument("--timepoints", type=int, default=10)
    ap.add_argument("--plane-s", type=float, default=PLANE_S)
    ap.add_argument("--gap-s", type=float, default=GAP_S)
    ap.add_argument("--setup-s", type=float, default=SETUP_S)
    ap.add_argument("--interval", type=float, default=None)
    ap.add_argument("--z-step", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=5602)
    ap.add_argument(
        "--raw-root", default="/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_lv/raw"
    )
    ap.add_argument("--readahead", type=int, default=2)
    args = ap.parse_args()

    arrays = [
        zarr.open(str(args.source / f"{args.base}_{ch}.ome.zarr" / "p0"), mode="r")
        for ch in CHANNELS
    ]
    n_t = min(args.timepoints, *(a.shape[0] for a in arrays))
    nz, ny, nx = arrays[0].shape[1:]
    stack_s = nz * args.plane_s
    c_off = [c * (stack_s + args.gap_s) for c in range(len(arrays))]
    interval = args.interval or (c_off[-1] + stack_s + NEXT_T_S)
    slab = max(1, SLAB_BYTES // (ny * nx * 2))

    order = [(t, c) for t in range(n_t) for c in range(len(arrays))]
    q: queue.Queue = queue.Queue(maxsize=args.readahead * len(arrays))
    sha1: dict[str, str] = {}

    def read() -> None:
        for t, c in order:
            vol = np.ascontiguousarray(arrays[c][t])
            sha1[f"{t},{c}"] = hashlib.sha1(vol.tobytes()).hexdigest()
            q.put(vol)

    threading.Thread(target=read, daemon=True).start()

    header = {
        "base_name": args.name,
        "raw_root": args.raw_root,
        "dtype": "uint16",
        "shape_zyx": [nz, ny, nx],
        "num_timepoints": n_t,
        "channels": list(range(len(arrays))),
        "channel_names": CHANNELS,
        "z_step_um": args.z_step,
        "xy_pixel_size": 0.136,
        "t_interval_s": interval,
        "accepts": ["qc"],
    }
    stalls, late = [], []
    with StreamSender(f"tcp://127.0.0.1:{args.port}", compress=True) as s:
        run_start = time.time()
        s.session_start(header)
        print("session", s.session_id, flush=True)
        fi = 0
        for t, c in order:
            waited = time.time()
            vol = q.get()
            stalls.append(time.time() - waited)
            first = run_start + args.setup_s + t * interval + c_off[c]
            for z0 in range(0, nz, slab):
                n = min(slab, nz - z0)
                due = first + (z0 + n) * args.plane_s  # its last plane exposed
                ahead = due - time.time()
                if ahead > 0:
                    time.sleep(ahead)
                else:
                    late.append(-ahead)
                now = time.time()
                h = {
                    "t": t,
                    "c": c,
                    "frame_index": fi,
                    "timestamp": now,
                    "camera_id": c,
                    "dtype": "uint16",
                    "acq_first_s": first,
                    "acq_last_s": due,
                    "queued_s": now,
                    "sent_s": time.time(),
                    "clock_offset_s": 0.0,
                }
                s.send_slab(h, vol, z0, n)
                fi += 1
            print(f"t={t} c={c} last plane at +{due - run_start:.2f} s", flush=True)
        last_plane = time.time()
        acked = s.wait_for_all_acked(timeout_s=300)
        s.session_end("complete")
        time.sleep(2.0)
    report = {
        "name": args.name,
        "base_name": args.name,
        "source": str(args.source / args.base),
        "links": 0,
        "timepoints": n_t,
        "channels": CHANNELS,
        "shape_zyx": [nz, ny, nx],
        "pace": "rig",
        "setup_s": args.setup_s,
        "interval_s": interval,
        "channel_start_s": dict(enumerate(c_off)),
        "channel_stack_s": stack_s,
        "slab_planes": slab,
        "read_stall_s": {"total": round(sum(stalls), 3), "max": round(max(stalls), 3)},
        "planes_late_s": {
            "n_over_10ms": sum(x > 0.01 for x in late),
            "max": round(max(late, default=0.0), 3),
        },
        "all_acked": acked,
        "all_acked_after_last_plane_s": round(time.time() - last_plane, 2),
        "sha1": sha1,
    }
    out = Path.home() / f"{args.name}.replay.json"
    out.write_text(json.dumps(report))
    print(json.dumps({k: v for k, v in report.items() if k != "sha1"}, indent=1))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
