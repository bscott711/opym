# Ruff style: Compliant
"""Follow a live acquisition in napari (`naparym-live`).

The live lane (opym.stream.live) writes each deconvolved + deskewed
timepoint into the dataset's pyramidal OME-Zarr the moment it's finished,
and records which (t, c) are written in `.opym_live.json`. This opens that
store with one layer per channel and polls the progress file every few
seconds. When a timepoint completes, the layers are refreshed and, in follow
mode, the time slider jumps to it. The status line shows how many timepoints
are done and how far the view lags the newest one.

With no argument it opens the newest session (`<jobs>/live_latest.json`).
Any finished dataset's `viewer/*_dsr.ome.zarr` opens the same way.

napari's dask cache is turned off: it would keep serving the zeros read from
a timepoint before it was written.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from opym import lanes
from opym.ome_zarr_writer import complete_timepoints, read_progress

POLL_S = 2.0
# omero hex colour -> napari colormap name (see ome_zarr_writer.CHANNEL_COLORS).
_COLORMAPS = {"00FF00": "green", "FF3D3D": "red", "00B3FF": "cyan", "FFC400": "yellow"}


def resolve_store(arg: str | None, jobs: Path | None = None) -> Path:
    """A store path, a dataset directory holding `viewer/*_dsr.ome.zarr`, or
    (no argument) the newest live session's store."""
    if arg:
        p = Path(arg)
        if p.suffix == ".zarr" or (p / ".zgroup").exists():
            return p
        found = sorted((p / "viewer").glob("*_dsr.ome.zarr"))
        if found:
            return found[0]
        raise FileNotFoundError(f"No *_dsr.ome.zarr store at or under {p}")
    latest = (jobs or lanes.jobs_dir()) / "live_latest.json"
    try:
        return Path(json.loads(latest.read_text())["store"])
    except (OSError, ValueError, KeyError) as exc:
        raise FileNotFoundError(
            f"No live session recorded yet ({latest}); pass a store path."
        ) from exc


def status_text(name: str, progress: dict | None, now: float | None = None) -> str:
    if not progress:
        return f"{name}: waiting for the first timepoint"
    done = complete_timepoints(progress)
    n_t = int(progress.get("n_t", 0))
    state = progress.get("state", "running")
    text = f"{name}: {len(done)}/{n_t} timepoints · {state}"
    if state == "running" and done:
        age = (time.time() if now is None else now) - float(
            progress.get("updated_at", 0)
        )
        text += f" · newest t={done[-1]}, updated {age:.0f}s ago"
    return text


class LiveFollower:
    """Keeps a napari viewer's layers in step with a growing store."""

    def __init__(self, viewer, store: Path, *, follow: bool = True) -> None:
        import dask.array as da
        import zarr

        self.viewer = viewer
        self.store = Path(store)
        self.follow = follow
        self.name = self.store.name.removesuffix("_dsr.ome.zarr").removesuffix(
            ".ome.zarr"
        )
        root = zarr.open_group(str(self.store), mode="r")
        ms = root.attrs["multiscales"][0]
        channels = root.attrs.get("omero", {}).get("channels", [])
        levels = [da.from_zarr(root[d["path"]]) for d in ms["datasets"]]
        scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
        n_c = levels[0].shape[1]
        self.layers = viewer.add_image(
            levels,
            channel_axis=1,
            multiscale=True,
            name=[
                channels[c]["label"] if c < len(channels) else f"C{c}"
                for c in range(n_c)
            ],
            colormap=[
                _COLORMAPS.get(channels[c].get("color", ""), "gray")
                if c < len(channels)
                else "gray"
                for c in range(n_c)
            ],
            blending="additive",
            scale=[scale[0]] + scale[2:],
            contrast_limits=[0, 300],
        )
        self.layers = self.layers if isinstance(self.layers, list) else [self.layers]
        self._levels = levels
        self._shown: list[int] = []
        self._contrast_set = False
        viewer.text_overlay.visible = True
        self.poll()

    def poll(self) -> None:
        progress = read_progress(self.store)
        self.viewer.text_overlay.text = status_text(self.name, progress)
        done = complete_timepoints(progress)
        if done == self._shown:
            return
        new = [t for t in done if t not in self._shown]
        self._shown = done
        if not self._contrast_set and done:
            self._set_contrast(done[0])
        for layer in self.layers:
            layer.refresh()
        if self.follow and new:
            self.viewer.dims.set_current_step(0, max(new))

    def _set_contrast(self, t: int) -> None:
        """From the coarsest level of the first real timepoint (the empty
        store has no range to go on)."""
        import numpy as np

        coarse = self._levels[-1]
        for c, layer in enumerate(self.layers):
            vol = np.asarray(coarse[t, c])
            lo, hi = np.percentile(vol[vol > 0], [0.5, 99.9]) if vol.any() else (0, 300)
            layer.contrast_limits = (float(lo), float(max(hi, lo + 1)))
        self._contrast_set = True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="naparym-live", description=__doc__.splitlines()[0]
    )
    ap.add_argument(
        "store",
        nargs="?",
        help="OME-Zarr store or dataset dir (default: the newest live session)",
    )
    ap.add_argument(
        "--no-follow", action="store_true", help="don't jump to new timepoints"
    )
    ap.add_argument("--poll", type=float, default=POLL_S, help="seconds between checks")
    args = ap.parse_args(argv)

    import napari
    from napari.utils import resize_dask_cache
    from qtpy.QtCore import QTimer

    resize_dask_cache(0)
    store = resolve_store(args.store)
    viewer = napari.Viewer(title=f"naparym-live: {store.name}", ndisplay=3)
    follower = LiveFollower(viewer, store, follow=not args.no_follow)

    @viewer.bind_key("f")
    def _toggle_follow(_viewer):
        follower.follow = not follower.follow
        _viewer.status = f"follow {'on' if follower.follow else 'off'}"

    timer = QTimer()
    timer.timeout.connect(follower.poll)
    timer.start(int(args.poll * 1000))
    napari.run()


if __name__ == "__main__":
    main()
