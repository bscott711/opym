"""The 8-bit display before and after the square-root encoding, beside 16-bit.

One timepoint of a store, every channel, in one napari window split into
three cells that share one camera (rotate one, all turn):

  1. before   the old 8-bit: min(v >> shift, 255), the shift fixed by the
              first volume's 99.9th voxel percentile (or the session's
              recorded one), contrast 0.5-99.9 percentile of the voxels
  2. sqrt     the square-root codes naparym-live holds now, decoded on the
              GPU by its own layer (LiveImage), contrast from the Z-MIP
              (live_view.default_contrast)
  3. 16-bit   the stored counts, the same contrast as 2

Every cell is a LiveImage, so 2 is drawn by the real decoding shader. Once
drawn, each cell is drawn alone in the same place (grid off: vispy jitters
rays by screen position) and compared with 3: the mean and 99th-percentile
difference in screen grey levels, and how much of the cell is at full
brightness. With --out, the side-by-side screenshot and numbers are saved;
without --quit, the window stays open to look at.

usage: DISPLAY=:N XAUTHORITY=... PYTHONPATH=<opym>/src \\
       python sqrt8_compare.py [STORE] [--t T] [--out RESULT.json] [--quit]
  STORE: a *_dsr.ome.zarr (default: the newest live session's view store)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

COLORS = {0: "green", 1: "magenta"}
NAMES = ["8-bit before", "8-bit sqrt (now)", "16-bit"]


def old_shift(vol: np.ndarray) -> int:
    """The shift the old servers fixed per session: 120 grey levels below
    the 99.9th percentile of every 4th nonzero voxel."""
    sub = vol[::4, ::4, ::4]
    sub = sub[sub > 0]
    if not sub.size:
        return 0
    return max(0, math.floor(math.log2(max(float(np.percentile(sub, 99.9)), 1) / 120)))


def old_contrast(codes: np.ndarray, shift: int) -> tuple[float, float]:
    """The old default contrast (0.5-99.9 percentile of every 4th nonzero
    voxel of the 8-bit data), in counts."""
    sub = codes[::4, ::4, ::4]
    sub = sub[sub > 0]
    lo, hi = np.percentile(sub, [0.5, 99.9]) if sub.size else (0.0, 255.0)
    return float(lo) * 2**shift, float(max(hi, lo + 1)) * 2**shift


def screen_stats(shots: list[np.ndarray], ref: int) -> list[dict]:
    """Each (H, W, 4) grab against grab `ref`. Each is the canvas with one
    cell's layers alone and the grid off, so every one is drawn in the same
    place: vispy jitters each ray's start by its screen position, and two
    grid cells of the same noisy volume differ by ~3 grey levels anyway."""
    imgs = [shot[..., :3].astype(np.int16) for shot in shots]
    signal = imgs[ref].max(axis=2) > 24  # where the 16-bit view shows the cell
    out = []
    for img in imgs:
        diff = np.abs(img - imgs[ref]).max(axis=2)[signal]
        bright = img.max(axis=2)[signal]
        out.append(
            {
                "mean_abs_diff_vs_16bit": round(float(diff.mean()), 2),
                "p99_abs_diff_vs_16bit": float(np.percentile(diff, 99)),
                "frac_full_brightness": round(float((bright >= 250).mean()), 4),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("store", nargs="?", type=Path)
    ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--quit", action="store_true", help="close once grabbed")
    ap.add_argument("--interpolation", default="linear", help="3D interpolation")
    args = ap.parse_args()

    from opym import lanes, live_view
    from opym.ome_zarr_writer import image_group

    store = args.store
    if store is None:
        latest = json.loads((lanes.jobs_dir() / "live_latest.json").read_text())
        store = Path(latest.get("view_store") or latest["store"])
    group = image_group(store)
    arr = group["0"]  # (t, c, z, y, x)
    scale = group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][
        0
    ]["scale"][2:]
    report: dict = {"store": str(store), "t": args.t, "channels": {}}
    cells: list[list[tuple]] = [[], [], []]  # per cell: (name, data, enc, clim, color)
    for c in range(arr.shape[1]):
        vol = np.ascontiguousarray(arr[args.t, c])
        shift = old_shift(vol)
        before = live_view.DisplayEncoding.linear(shift)
        old8 = before.lut()[vol]
        # the session's curve: fixed by its first volume, as the server does
        sqrt = live_view.DisplayEncoding.sqrt_for(np.asarray(arr[0, c]))
        new8 = live_view.to_8bit(vol, sqrt)
        clim_before = old_contrast(old8, shift)
        clim_now = live_view.default_contrast(new8, sqrt)
        nz = vol[vol > 0]
        report["channels"][c] = {
            "shift_before": shift,
            "sqrt_vmax": sqrt.vmax,
            "max_count": int(vol.max()),
            "frac_voxels_clipped_before": round(float((nz > (255 << shift)).mean()), 5),
            "contrast_before": [round(v, 1) for v in clim_before],
            "contrast_now": [round(v, 1) for v in clim_now],
            "contrast_now_from_16bit": [
                round(v, 1) for v in live_view.default_contrast(vol)
            ],
        }
        color = COLORS.get(c, "gray")
        cells[0].append((f"C{c} before", old8, before, clim_before, color))
        cells[1].append((f"C{c} sqrt", new8, sqrt, clim_now, color))
        cells[2].append((f"C{c} 16-bit", vol, None, clim_now, color))
    print(json.dumps(report, indent=1), flush=True)

    import napari
    from qtpy.QtCore import QTimer

    viewer = napari.Viewer(
        title=f"{' | '.join(NAMES)} -- t={args.t} {store.name}", ndisplay=3
    )
    live_image = live_view.live_image_class()
    # napari fills the grid from the END of the layer list: add the cells in
    # reverse to read left to right.
    layers: dict[int, list] = {}
    for i in (2, 1, 0):
        for name, data, enc, clim, color in cells[i]:
            layer = live_image(
                data,
                name=name,
                scale=scale,
                colormap=color,
                blending="additive",
                contrast_limits=[0, enc.vmax if enc is not None else 4095],
                encoding=(lambda e=enc: e),
            )
            layer.contrast_limits = clim
            layer.interpolation3d = args.interpolation
            viewer.add_layer(layer)
            layers.setdefault(i, []).append(layer)
    viewer.grid.enabled = True
    viewer.grid.stride = len(cells[0])
    viewer.grid.shape = (1, 3)
    viewer.text_overlay.visible = True
    viewer.window._qt_window.showMaximized()
    viewer.reset_view()

    def column_of(i: int) -> int:
        """The grid column cell i is drawn in, found by drawing it alone
        (napari's own grid order has changed between versions)."""
        for k, group in layers.items():
            for layer in group:
                layer.visible = k == i
        shot = viewer.screenshot(canvas_only=True, flash=False)[..., :3]
        w = shot.shape[1] // 3
        return int(np.argmax([shot[:, j * w : (j + 1) * w].sum() for j in range(3)]))

    def show_only(i: int | None) -> None:
        for k, group in layers.items():
            for layer in group:
                layer.visible = i is None or k == i

    def grab() -> None:
        col = {i: column_of(i) for i in layers}
        viewer.grid.enabled = False
        alone = []
        for i in range(3):
            show_only(i)
            alone.append(viewer.screenshot(canvas_only=True, flash=False))
        stats = screen_stats(alone, 2)
        report["screen"] = {NAMES[i]: stats[i] for i in range(3)}
        viewer.grid.enabled = True
        show_only(None)
        order = sorted(layers, key=col.get)
        report["cells_left_to_right"] = [NAMES[i] for i in order]
        viewer.text_overlay.text = "   |   ".join(NAMES[i] for i in order)
        shot = viewer.screenshot(canvas_only=True, flash=False)
        print(json.dumps(report["screen"], indent=1), flush=True)
        if args.out:
            args.out.write_text(json.dumps(report, indent=1))
            from PIL import Image

            Image.fromarray(shot).save(args.out.with_suffix(".png"))
        if args.quit:
            viewer.close()

    QTimer.singleShot(8000, grab)
    napari.run()


if __name__ == "__main__":
    main()
