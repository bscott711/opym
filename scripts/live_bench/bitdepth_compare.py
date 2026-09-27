"""16-bit vs 8-bit textures, side by side on the DCV display.

One timepoint of a live session's view store, both channels, in one napari
window split into three cells that share one camera (rotate one, all turn):

  1. 16-bit   uint16 volumes, as naparym-live uploads them today
  2. 8-bit    uint8, mapped linearly from the display window (the same
              0.5-99.9 percentile contrast naparym-live sets); the smallest
              and fastest, but the contrast can't later be widened past it
  3. 8-bit wide  uint8, mapped from 0 to the 99.99th percentile; its
              contrast is set to show the same window as 1 and 2, so there
              is room to change it later, at a coarser step

Same colors (green / magenta), additive blending and rendering as
naparym-live, so any difference is quantization only. Also times the
uint16 -> uint8 conversion and each cell's texture upload (a refresh, to
the frame on screen), and prints them. The window stays open to look at.

usage: DISPLAY=:2 XAUTHORITY=... PYTHONPATH=<opym>/src \\
       python bitdepth_compare.py [STORE] [--t T] [--out RESULT.json]
  STORE: a *_dsr.ome.zarr view store (default: the newest test session's)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

COLORS = {0: "green", 1: "magenta"}


def to_uint8(vol: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """`vol` mapped linearly from [lo, hi] to 0..255 (clipped), through a
    65536-entry lookup table, a z-slab per thread."""
    ramp = (np.arange(65536, dtype=np.float32) - lo) * (255.0 / max(hi - lo, 1.0))
    lut = np.clip(np.rint(ramp), 0, 255).astype(np.uint8)
    out = np.empty(vol.shape, np.uint8)
    step = max(1, vol.shape[0] // 32)

    def one(z0: int) -> None:
        np.take(lut, vol[z0 : z0 + step], out=out[z0 : z0 + step])

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(one, range(0, vol.shape[0], step)))
    return out


def display_window(vol: np.ndarray) -> tuple[float, float]:
    """naparym-live's contrast: the 0.5-99.9 percentiles of every 4th
    nonzero voxel."""
    sub = vol[::4, ::4, ::4]
    sub = sub[sub > 0]
    lo, hi = np.percentile(sub, [0.5, 99.9]) if sub.size else (0.0, 300.0)
    return float(lo), float(max(hi, lo + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("store", nargs="?", type=Path)
    ap.add_argument("--t", type=int, default=15)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import zarr
    from opym import lanes

    store = args.store
    if store is None:
        latest = json.loads((lanes.jobs_dir() / "live_latest.json").read_text())
        store = Path(latest.get("view_store") or latest["store"])
    arr = zarr.open(str(store / "0" / "0"), mode="r")  # (t, c, z, y, x)
    scale = json.loads((store / "0" / ".zattrs").read_text())["multiscales"][0][
        "datasets"
    ][0]["coordinateTransformations"][0]["scale"][2:]
    report: dict = {"store": str(store), "t": args.t, "channels": {}}
    cells: list[list[tuple]] = [[], [], []]  # per cell: (name, data, clim, color)
    for c in range(arr.shape[1]):
        vol = np.ascontiguousarray(arr[args.t, c])
        lo, hi = display_window(vol)
        t0 = time.perf_counter()
        v8 = to_uint8(vol, lo, hi)
        convert_s = time.perf_counter() - t0
        sub = vol[::4, ::4, ::4]
        whi = float(np.percentile(sub[sub > 0], 99.99)) if sub.any() else hi
        w8 = to_uint8(vol, 0.0, whi)
        wide_clim = (lo * 255 / whi, min(255.0, hi * 255 / whi))
        report["channels"][c] = {
            "window": [lo, hi],
            "wide_window": [0.0, whi],
            "wide_levels_in_display_window": round(wide_clim[1] - wide_clim[0], 1),
            "convert_s": round(convert_s, 3),
            "bytes_16": vol.nbytes,
            "bytes_8": v8.nbytes,
        }
        color = COLORS.get(c, "gray")
        cells[0].append((f"C{c} 16-bit", vol, (lo, hi), color))
        cells[1].append((f"C{c} 8-bit", v8, (0.0, 255.0), color))
        cells[2].append((f"C{c} 8-bit wide", w8, wide_clim, color))
    print(json.dumps(report, indent=1), flush=True)

    import napari
    from qtpy.QtCore import QTimer

    viewer = napari.Viewer(
        title=f"16-bit | 8-bit | 8-bit wide -- t={args.t} {store.name}", ndisplay=3
    )
    # napari fills the grid from the END of the layer list, so add the
    # cells in reverse to read left to right: 16-bit, 8-bit, 8-bit wide.
    layers: dict[int, list] = {}
    for i in (2, 1, 0):
        for name, data, clim, color in cells[i]:
            layer = viewer.add_image(
                data,
                name=name,
                scale=scale,
                colormap=color,
                blending="additive",
                contrast_limits=clim,
            )
            layers.setdefault(i, []).append(layer)
    viewer.grid.enabled = True
    viewer.grid.stride = len(cells[0])
    viewer.grid.shape = (1, 3)
    # Label the cells by where napari actually put them.
    n = len(viewer.layers)
    col_of = {
        i: viewer.grid.position(n - 1 - viewer.layers.index(layers[i][0]), n)[1]
        for i in layers
    }
    names = ["16-bit", "8-bit", "8-bit wide"]
    order = sorted(layers, key=col_of.get)
    report["cells_left_to_right"] = [names[i] for i in order]
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = "   |   ".join(names[i] for i in order)
    viewer.window._qt_window.showMaximized()
    viewer.reset_view()

    canvas = viewer.window._qt_viewer.canvas
    timings: dict[str, list[float]] = {name: [] for name in names}
    plan = [i for _ in range(5) for i in (0, 1, 2)]
    state = {"pending": None, "next_at": time.perf_counter() + 6.0}

    def on_frame() -> None:
        if state["pending"] is None:
            return
        i, t0 = state["pending"]
        timings[names[i]].append(time.perf_counter() - t0)
        state["pending"] = None
        state["next_at"] = time.perf_counter() + 0.3

    canvas.native.frameSwapped.connect(on_frame)

    def tick() -> None:
        now = time.perf_counter()
        if state["pending"] is not None or now < state["next_at"]:
            return
        if not plan:
            driver.stop()
            report["upload_to_frame_s"] = {
                k: {"p50": round(statistics.median(v), 3), "max": round(max(v), 3)}
                for k, v in timings.items()
                if v
            }
            print(json.dumps(report["upload_to_frame_s"], indent=1), flush=True)
            viewer.reset_view()
            if args.out:
                args.out.write_text(json.dumps(report, indent=1))
                shot = viewer.window._qt_window.grab()
                shot.save(str(args.out.with_suffix(".png")))
            return
        i = plan.pop(0)
        state["pending"] = (i, now)
        for layer in layers[i]:
            layer.refresh()  # re-slice, so the texture is uploaded again

    driver = QTimer()
    driver.timeout.connect(tick)
    driver.start(20)
    napari.run()


if __name__ == "__main__":
    main()
