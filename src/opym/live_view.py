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

Live QC: when the dataset has `<leaf>/qc/live_qc.jsonl` (CORE's
`celldet-live-qc`), each timepoint's cell box is drawn as a wireframe coloured
by its verdict (green ok, orange warn, red act), and the status line adds the
verdict, flags and first piece of advice for the timepoint on screen.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from opym import lanes
from opym.ome_zarr_writer import complete_timepoints, read_progress

POLL_S = 2.0
QC_LOG_NAME = "live_qc.jsonl"
QC_COLORS = {"ok": "lime", "warn": "orange", "act": "red", "no_cell": "gray"}
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


def qc_dir_for(store: Path) -> Path:
    """`<leaf>/viewer/<name>_dsr.ome.zarr` -> `<leaf>/qc` (opym.stream.live)."""
    return Path(store).parent.parent / "qc"


def box_edges(t: int, z0, y0, x0, z1, y1, x1) -> list:
    """The 12 edges of a box as (2, 4) (t, z, y, x) paths."""
    import numpy as np

    corners = [(z, y, x) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    edges = []
    for i, a in enumerate(corners):
        for b in corners[i + 1 :]:
            if sum(u != v for u, v in zip(a, b, strict=True)) == 1:
                edges.append(np.array([[t, *a], [t, *b]], dtype=float))
    return edges


def qc_summary(rec: dict | None) -> str:
    if not rec:
        return ""
    text = f"QC t={rec['t']}: {rec['verdict']}"
    if rec.get("flags"):
        text += " · " + ", ".join(rec["flags"])
    advice = rec.get("advice") or []
    if advice:
        text += "\n" + advice[0]["text"]
    return text


class QCOverlay:
    """Reads the live QC log as it grows and draws each timepoint's box."""

    def __init__(self, viewer, qc_dir: Path, scale, n_z: int) -> None:
        self.viewer = viewer
        self.path = Path(qc_dir) / QC_LOG_NAME
        self.scale = list(scale)
        self.n_z = n_z
        self.offset = 0
        self.session: str | None = None
        self.raw: dict[int, dict] = {}
        self.boxed: set[int] = set()
        self.layer = None

    def _reset(self, session: str | None) -> None:
        self.session = session
        self.raw.clear()
        self.boxed.clear()
        if self.layer is not None:
            self.layer.data = []

    def poll(self) -> bool:
        """Read new complete lines; True if anything changed."""
        try:
            with open(self.path) as f:
                f.seek(self.offset)
                chunk = f.read()
        except OSError:
            return False
        end = chunk.rfind("\n")
        if end < 0:
            return False
        self.offset += len(chunk[: end + 1].encode())
        changed = False
        for line in chunk[:end].splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("stage") == "session":
                if rec.get("session_id") != self.session:
                    self._reset(rec.get("session_id"))
                    changed = True
                continue
            if self.session is not None and rec.get("session_id") != self.session:
                continue
            if rec.get("stage") == "raw":
                self.raw[rec["t"]] = rec
                changed = True
            elif rec.get("stage") == "dsr" and rec["t"] not in self.boxed:
                changed |= self._add_box(rec)
        return changed

    def _add_box(self, rec: dict) -> bool:
        found = [b["bbox_zyx"] for b in rec.get("boxes", {}).values() if b.get("found")]
        if not found:
            return False
        lo = [min(b[i] for b in found if b[i] is not None) for i in (1, 2)]
        hi = [max(b[i] for b in found if b[i] is not None) for i in (4, 5)]
        zs = [b[i] for b in found for i in (0, 3) if b[i] is not None]
        z0, z1 = (min(zs), max(zs)) if zs else (0, self.n_z - 1)
        edges = box_edges(rec["t"], z0, lo[0], lo[1], z1, hi[0], hi[1])
        color = QC_COLORS.get(self.raw.get(rec["t"], rec).get("verdict"), "white")
        if self.layer is None:
            self.layer = self.viewer.add_shapes(
                edges,
                shape_type="path",
                edge_color=color,
                edge_width=2,
                scale=self.scale,
                name="QC box",
                ndim=4,
            )
        else:
            self.layer.add_paths(edges, edge_color=color, edge_width=2)
        self.boxed.add(rec["t"])
        return True

    def summary(self, t: int) -> str:
        return qc_summary(self.raw.get(t))


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
        self._status = ""
        self.qc = QCOverlay(
            viewer, qc_dir_for(self.store), [scale[0]] + scale[2:], levels[0].shape[2]
        )
        viewer.text_overlay.visible = True
        viewer.dims.events.current_step.connect(lambda _e: self._show_text())
        self.poll()

    def _show_text(self) -> None:
        qc = self.qc.summary(int(self.viewer.dims.current_step[0]))
        self.viewer.text_overlay.text = self._status + (f"\n{qc}" if qc else "")

    def poll(self) -> None:
        progress = read_progress(self.store)
        self._status = status_text(self.name, progress)
        self.qc.poll()
        self._show_text()
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
