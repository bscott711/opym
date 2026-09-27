"""Time naparym-live's interaction on a real display: rotate, scrub, rest.

Opens naparym-live on the test stack's newest session (as the user would)
and drives it from inside the process: camera rotations and time-slider
steps, one at a time, each timed from the action to the frame that shows it
(the canvas's frameSwapped). Runs against whichever opym is on PYTHONPATH,
so the same script measures the old viewer and the new one.

Sequence (after the session has loaded and painted):
  rotate_newest  20 rotations of 6 degrees at the newest timepoint
  (idle 25 s: time for a viewer that fills VRAM in the background)
  scrub_preloaded 10 steps back, to timepoints never shown yet
  scrub_cold     15 steps further back
  (wait for the prefetch to settle, if this viewer has one; else 20 s)
  scrub_warm     15 steps forward
  (rest 2 s: full resolution comes back)
  rotate_rest    20 rotations
Writes a JSON summary (p50 / p95 / max seconds per phase) and screenshots.
With the VRAM texture cache on, each step also records the texture path
every channel's volume took (hit: a cached texture rebound; spare: into the
pre-touched spare; recycle: into a reused cached texture; own: through the
node's own texture) and the time spent deleting evicted textures, and the
summary splits each phase by path (`<phase>_by_path`).

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
    extra = {}
    bits = int(os.environ.get("LB_BITS", "16"))
    if "bits" in live_view.ViewCache.__init__.__code__.co_varnames:
        extra["cache"] = live_view.ViewCache(bits=bits)
    # LB_VRAM_GB: GB, "auto" (all that is free beyond LB_VRAM_RESERVE_GB),
    # or 0 for off.
    vram_arg = os.environ.get("LB_VRAM_GB", "48")
    vram_gb = float("inf") if vram_arg == "auto" else float(vram_arg)
    if hasattr(live_view, "TextureCache") and vram_gb > 0:
        reserve = os.environ.get("LB_VRAM_RESERVE_GB")
        kw = {"reserve_bytes": float(reserve) * 1e9} if reserve else {}
        extra["vram"] = live_view.TextureCache(vram_gb * 1e9, **kw)
    watcher = live_view.SessionWatcher(
        viewer, jobs=lanes.jobs_dir(), follow=True, title=args.title, **extra
    )
    poll = QTimer()
    poll.timeout.connect(watcher.poll)
    poll.start(100)
    canvas = viewer.window._qt_viewer.canvas

    results: dict[str, list] = {}
    state = {"pending": None, "next_at": 0.0, "plan": [], "phase": "load"}
    t_open = time.perf_counter()

    cache_states: dict[str, dict] = {}

    def cache_state(label: str) -> None:
        """The texture cache's free-VRAM reading, capacity and contents."""
        vram = extra.get("vram")
        if vram is None or not hasattr(vram, "capacity"):
            return
        vram._free_at = float("-inf")  # a fresh reading
        free = vram.free_bytes()
        probe = []
        for layer in getattr(watcher.follower, "layers", []):
            node = getattr(layer, "live_node", None)
            try:
                canvas = None if node is None else node.canvas
                if canvas is not None:
                    from vispy.gloo import gl

                    canvas.set_current()
                    probe.append(repr(gl.glGetParameter(0x9049)))
                else:
                    probe.append("no canvas")
            except Exception as exc:  # noqa: BLE001
                probe.append(f"{type(exc).__name__}: {exc}")
        probe.append(f"hook={getattr(vram, 'free_vram', None)!r}")
        import subprocess

        smi = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.free",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            .stdout.strip()
            .replace("\n", " | ")
        )
        probe.append(f"smi: {smi}")
        cache_states[label] = {
            "free_gb": None if free is None else round(free / 1e9, 2),
            "capacity_gb": round(vram.capacity() / 1e9, 2),
            "cached_gb": round(vram.bytes / 1e9, 2),
            "textures": len(vram._items),
            "probe": probe,
        }

    def shot(name: str) -> None:
        img = viewer.window._qt_window.grab()
        img.save(str(args.out.with_name(f"{args.out.stem}_{name}.png")))

    paths: list[str] = []  # texture paths taken since the pending action
    by_path: dict[str, list] = {}
    deletes: list[float] = []
    # per step: action -> first set_data (napari slicing: the read), time in
    # set_data, time in GL flushes (the uploads), and the rest of the frame
    parts: dict = {"first_set": None, "set_s": 0.0, "flush_s": 0.0}
    parts_by_label: dict[str, dict[str, list]] = {}

    def instrument(node) -> None:
        """Record which path each served volume takes through the node's
        `set_data`, and time the cache's texture deletes."""
        cls = type(node)
        if getattr(cls, "_bench_wrapped", False):
            return
        cls._bench_wrapped = True
        set_data = cls.set_data

        def timed_set_data(self, vol, *a, **k):
            t0 = time.perf_counter()
            if parts["first_set"] is None:
                parts["first_set"] = t0
            try:
                return _set_data(self, vol, *a, **k)
            finally:
                parts["set_s"] += time.perf_counter() - t0

        def _set_data(self, vol, *a, **k):
            vram = getattr(self, "_vram", None)
            if vram is not None and getattr(self, "_own_texture", None) is not None:
                key = live_view.served_key(vol)
                before = self._texture
                own = self._own_texture
                cached = key is not None and key in vram
                spare = getattr(self, "_spare", None)
                out = set_data(self, vol, *a, **k)
                if key is not None:
                    if cached:
                        paths.append("hit")
                    elif spare is not None and self._texture is spare[0]:
                        paths.append("spare")
                    elif self._texture is own:
                        paths.append("own")
                    elif self._texture is not before or key in vram:
                        paths.append("recycle")
                return out
            return set_data(self, vol, *a, **k)

        cls.set_data = timed_set_data
        context = node.canvas.context
        flush = context.flush_commands

        def timed_flush(*a, **k):
            t0 = time.perf_counter()
            try:
                return flush(*a, **k)
            finally:
                parts["flush_s"] += time.perf_counter() - t0

        context.flush_commands = timed_flush
        cache_cls = live_view.TextureCache
        delete = cache_cls._delete

        def timed_delete(self, item):
            t0 = time.perf_counter()
            delete(self, item)
            deletes.append(time.perf_counter() - t0)

        cache_cls._delete = timed_delete

    def on_frame() -> None:
        pending = state["pending"]
        if pending is None:
            return
        label, t0 = pending
        dt = time.perf_counter() - t0
        results.setdefault(label, []).append(dt)
        if parts["first_set"] is not None:
            split = parts_by_label.setdefault(
                label, {"read": [], "set_data": [], "flush": [], "rest": []}
            )
            read = parts["first_set"] - t0
            split["read"].append(read)
            split["set_data"].append(parts["set_s"])
            split["flush"].append(parts["flush_s"])
            split["rest"].append(dt - read - parts["set_s"] - parts["flush_s"])
        if paths:
            by_path.setdefault(f"{label}_by_path", {}).setdefault(
                "+".join(sorted(paths)), []
            ).append(dt)
        paths.clear()
        parts.update(first_set=None, set_s=0.0, flush_s=0.0)
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
                cache_state("loaded")
                for layer in follower.layers:
                    if getattr(layer, "live_node", None) is not None:
                        instrument(layer.live_node)
                shot("loaded")
                state["plan"] = (
                    [rotate("rotate_newest") for _ in range(20)]
                    + [("rest", 25.0, None)]
                    + [step("scrub_preloaded", -1) for _ in range(10)]
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
            cache_state(f"rest_at_{len(state['plan'])}")
            state["plan"].pop(0)
            state["next_at"] = now + label
            return
        if kind == "done":
            cache_state("end")
            state["plan"].pop(0)
            driver.stop()
            shot("end")
            summary = {
                "opym": live_view.__file__,
                "vram_gb": vram_arg,
                "bits": bits,
                "vram_reserve_gb": os.environ.get("LB_VRAM_RESERVE_GB"),
                **{k: stats(v) for k, v in results.items()},
                **{
                    k: {p: stats(v) for p, v in sorted(d.items())}
                    for k, d in by_path.items()
                },
                "texture_deletes": stats(deletes),
                "cache_states": cache_states,
                **{
                    f"{label}_split": {k: stats(v) for k, v in split.items()}
                    for label, split in parts_by_label.items()
                },
            }
            args.out.write_text(json.dumps(summary, indent=1))
            print(json.dumps(summary, indent=1), flush=True)
            viewer.close()
            return
        state["plan"].pop(0)
        paths.clear()
        parts.update(first_set=None, set_s=0.0, flush_s=0.0)
        state["pending"] = (label, now)
        act()

    driver = QTimer()
    driver.timeout.connect(tick)
    driver.start(10)
    napari.run()


if __name__ == "__main__":
    main()
