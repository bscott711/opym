"""Time naparym-live's interaction on a real display: rotate, scrub, rest.

Opens naparym-live on the test stack's newest session (as the user would)
and drives it from inside the process: camera rotations and time-slider
steps, one at a time, each timed from the action to the frame that shows it
(the canvas's frameSwapped). Runs against whichever opym is on PYTHONPATH,
so the same script measures the old viewer and the new one.

Sequence (after the session has loaded and painted):
  rotate_newest  20 rotations of 6 degrees at the newest timepoint
  scrub_cold     15 steps back, right away
  (wait for the prefetch to settle, if this viewer has one; else 20 s)
  scrub_warm     15 steps forward
  (rest 2 s: full resolution comes back)
  rotate_rest    20 rotations
Writes a JSON summary (p50 / p95 / max seconds per phase) and screenshots.

usage: DISPLAY=:2 XAUTHORITY=... PETAKIT_JOBS_DIR=/dev/shm/opym_lv/jobs \\
       PYTHONPATH=<opym>/src python viewer_bench.py --out RESULT.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def stats(values: list[float]) -> dict:
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50": round(statistics.median(values), 3),
        "p95": round(values[min(len(values) - 1, round(0.95 * (len(values) - 1)))], 3),
        "max": round(values[-1], 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="naparym-live [BENCH]")
    args = ap.parse_args()

    from opym import lanes, live_view

    if hasattr(live_view, "no_hugepage_stalls"):
        live_view.no_hugepage_stalls()
    import os

    for name in ("PREFETCH_IN_FLIGHT", "PREFETCH_RADIUS", "PREFETCH_READ_THREADS"):
        if os.environ.get("LB_" + name):
            setattr(live_view, name, int(os.environ["LB_" + name]))
    import napari
    from qtpy.QtCore import QTimer

    viewer = napari.Viewer(title=args.title, ndisplay=3)
    watcher = live_view.SessionWatcher(
        viewer, jobs=lanes.jobs_dir(), follow=True, title=args.title
    )
    poll = QTimer()
    poll.timeout.connect(watcher.poll)
    poll.start(100)
    canvas = viewer.window._qt_viewer.canvas

    results: dict[str, list] = {}
    state = {"pending": None, "next_at": 0.0, "plan": [], "phase": "load"}
    t_open = time.perf_counter()

    def shot(name: str) -> None:
        img = viewer.window._qt_window.grab()
        img.save(str(args.out.with_name(f"{args.out.stem}_{name}.png")))

    def on_frame() -> None:
        pending = state["pending"]
        if pending is None:
            return
        label, t0 = pending
        results.setdefault(label, []).append(time.perf_counter() - t0)
        state["pending"] = None
        state["next_at"] = time.perf_counter() + 0.05

    canvas.native.frameSwapped.connect(on_frame)

    def rotate(label: str):
        def act() -> None:
            a = list(viewer.camera.angles)
            a[1] = (a[1] + 6.0) % 360
            viewer.camera.angles = tuple(a)

        return ("act", label, act)

    def step(label: str, delta: int):
        def act() -> None:
            t = int(viewer.dims.current_step[0])
            viewer.dims.set_current_step(0, max(0, t + delta))

        return ("act", label, act)

    def settled() -> bool:
        follower = watcher.follower
        cache = getattr(follower, "cache", None)
        if cache is None:
            return time.perf_counter() - state["warm_from"] > 20
        size = cache.full.bytes + cache.small.bytes
        now = time.perf_counter()
        if size != state.get("warm_size"):
            state["warm_size"], state["warm_at"] = size, now
        return now - state["warm_at"] > 2 or now - state["warm_from"] > 90

    def tick() -> None:
        now = time.perf_counter()
        if state["phase"] == "load":
            follower = watcher.follower
            if follower is not None and follower.layers and now - t_open > 8:
                state["phase"] = "run"
                results["open_s"] = [now - t_open]
                shot("loaded")
                state["plan"] = (
                    [rotate("rotate_newest") for _ in range(20)]
                    + [step("scrub_cold", -1) for _ in range(15)]
                    + [("warm", None, None)]
                    + [step("scrub_warm", +1) for _ in range(15)]
                    + [("rest", 2.0, None)]
                    + [rotate("rotate_rest") for _ in range(20)]
                    + [("done", None, None)]
                )
            return
        pending = state["pending"]
        if pending is not None:
            if now - pending[1] > 8:  # no frame came: count as a timeout
                results.setdefault(pending[0] + "_timeouts", []).append(1)
                state["pending"] = None
            return
        if now < state["next_at"] or not state["plan"]:
            return
        kind, label, act = state["plan"][0]
        if kind == "warm":
            state.setdefault("warm_from", now)
            if not settled():
                return
            results["warm_wait_s"] = [now - state["warm_from"]]
            state["plan"].pop(0)
            shot("warm")
            return
        if kind == "rest":
            state["plan"].pop(0)
            state["next_at"] = now + label
            return
        if kind == "done":
            state["plan"].pop(0)
            driver.stop()
            shot("end")
            summary = {
                "opym": live_view.__file__,
                **{k: stats(v) for k, v in results.items()},
            }
            args.out.write_text(json.dumps(summary, indent=1))
            print(json.dumps(summary, indent=1), flush=True)
            viewer.close()
            return
        state["plan"].pop(0)
        state["pending"] = (label, now)
        act()

    driver = QTimer()
    driver.timeout.connect(tick)
    driver.start(10)
    napari.run()


if __name__ == "__main__":
    main()
