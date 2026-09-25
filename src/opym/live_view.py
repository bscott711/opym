# Ruff style: Compliant
"""Follow a live acquisition in napari (`naparym-live`).

The live lane (opym.stream.live) writes each deconvolved + deskewed
timepoint into the dataset's pyramidal OME-Zarr the moment it's finished,
and records which (t, c) are written in `.opym_live.json`. This opens that
store with one layer per channel and polls the progress file every few
seconds. When a timepoint completes, the layers are refreshed and, in follow
mode, the time slider jumps to it. The status line shows how many timepoints
are done and how far the view lags the newest one.

With no argument the napari window opens immediately (not once data shows
up -- napari's own startup cost is worth paying exactly once, then leaving it
open) and follows whichever live session is newest, switching feeds itself
-- tearing down the old layers and loading the new store -- every time a
fresh one starts. Run it once, before or during an acquisition, and it keeps
up: a quick single-timepoint alignment snap, then the real time-lapse right
after, both show up in the same window with no restart in between. An
explicit store path is loaded once and stays put -- a typo there fails fast,
not by waiting. Any finished dataset's `viewer/*_dsr.ome.zarr` opens the same
way.

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
import logging
import re
import time
from pathlib import Path

from opym import lanes
from opym.ome_zarr_writer import complete_timepoints, read_progress

logger = logging.getLogger(__name__)

# napari's own layer-name uniquification suffix ("GFP 488" -> "GFP 488 [1]"),
# stripped back off once the layer it collided with is gone -- see
# SessionWatcher._switch.
_NAPARI_DEDUP_SUFFIX = re.compile(r" \[\d+\]$")

POLL_S = 2.0
LIVE_LATEST_NAME = "live_latest.json"
QC_LOG_NAME = "live_qc.jsonl"
QC_COLORS = {"ok": "lime", "warn": "orange", "act": "red", "no_cell": "gray"}
# omero hex colour -> napari colormap name (see ome_zarr_writer.CHANNEL_COLORS).
_COLORMAPS = {"00FF00": "green", "FF3D3D": "red", "00B3FF": "cyan", "FFC400": "yellow"}


def resolve_store(arg: str | None, jobs: Path | None = None) -> Path:
    """A store path, a dataset directory holding `viewer/*_dsr.ome.zarr`, or
    (no argument) the newest live session's store, checked once. For the
    no-argument case, `main()` uses `SessionWatcher` instead, which keeps
    checking as sessions come and go -- this is for an explicit path, or a
    one-shot lookup, where "nothing there yet" is a real error, not something
    to wait out.
    """
    if arg:
        p = Path(arg)
        if p.suffix == ".zarr" or (p / ".zgroup").exists():
            return p
        found = sorted((p / "viewer").glob("*_dsr.ome.zarr"))
        if found:
            return found[0]
        raise FileNotFoundError(f"No *_dsr.ome.zarr store at or under {p}")
    latest = (jobs or lanes.jobs_dir()) / LIVE_LATEST_NAME
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
        self.viewer = viewer
        self.store = Path(store)
        self.follow = follow
        self.name = self.store.name.removesuffix("_dsr.ome.zarr").removesuffix(
            ".ome.zarr"
        )
        root, ms, levels = self._open_levels()
        self._channels = root.attrs.get("omero", {}).get("channels", [])
        scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
        self._scale = [scale[0]] + scale[2:]
        self._n_c = levels[0].shape[1]
        self._levels = levels
        self.layers = self._add_layers(levels, contrast=[[0, 300]] * self._n_c)
        self._shown: list[int] = []
        self._contrast_set = False
        self._status = ""
        self.qc = QCOverlay(
            viewer, qc_dir_for(self.store), self._scale, levels[0].shape[2]
        )
        viewer.text_overlay.visible = True
        viewer.dims.events.current_step.connect(lambda _e: self._show_text())
        self.poll()

    def _open_levels(self):
        """Fresh dask arrays wrapping the store's current zarr metadata."""
        import dask.array as da
        import zarr

        root = zarr.open_group(str(self.store), mode="r")
        ms = root.attrs["multiscales"][0]
        return root, ms, [da.from_zarr(root[d["path"]]) for d in ms["datasets"]]

    def _add_layers(self, levels, contrast):
        layers = self.viewer.add_image(
            levels,
            channel_axis=1,
            multiscale=True,
            name=[
                self._channels[c]["label"] if c < len(self._channels) else f"C{c}"
                for c in range(self._n_c)
            ],
            colormap=[
                _COLORMAPS.get(self._channels[c].get("color", ""), "gray")
                if c < len(self._channels)
                else "gray"
                for c in range(self._n_c)
            ],
            blending="additive",
            scale=self._scale,
            contrast_limits=contrast,
        )
        return layers if isinstance(layers, list) else [layers]

    def _show_text(self) -> None:
        qc = self.qc.summary(int(self.viewer.dims.current_step[0]))
        self.viewer.text_overlay.text = self._status + (f"\n{qc}" if qc else "")

    def poll(self) -> None:
        """Rebuild the image layers from scratch on every new timepoint,
        rather than update the existing ones in place.

        Found live on Argus (2026-09-25): the volume stayed black in the
        open window while the store kept growing, and only closing and
        reopening naparym-live ever showed the real data -- a fresh process
        always works because it builds its layers via `add_image` from
        scratch. Two narrower fixes (fresh dask arrays reassigned onto the
        existing layers' `.data`; then adding an explicit `layer.refresh()`
        alongside that, once tracing napari's event wiring showed the vispy
        renderer only listens for `events.set_data`, fired only from
        `refresh()`) each turned out to be necessary but not sufficient --
        `set_data` was confirmed (by a listener attached in the test) to
        already fire even without either fix, meaning whatever the real
        remaining gap is sits deeper in napari's rendering pipeline than
        either fix reached, in a part not reachable from here to inspect (no
        working GPU/software-GL context on Argus to render a real frame and
        compare pixels -- confirmed while investigating this). Since a full
        restart is the one thing actually confirmed to work, every timepoint
        now gets that same fresh construction, just without restarting the
        process: remove the old layers only after the new ones are built and
        added successfully (so a construction failure never leaves the
        viewer blank), carrying over whatever contrast_limits was showing.
        """
        progress = read_progress(self.store)
        self._status = status_text(self.name, progress)
        self.qc.poll()
        self._show_text()
        done = complete_timepoints(progress)
        if done == self._shown:
            return
        new = [t for t in done if t not in self._shown]
        self._shown = done
        old_layers = self.layers
        limits = [layer.contrast_limits for layer in old_layers]
        _root, _ms, self._levels = self._open_levels()
        self.layers = self._add_layers(self._levels, contrast=limits)
        for layer in old_layers:
            self.viewer.layers.remove(layer)
        # The new image layers land at the end of the layer list (added
        # after the QC box, which was only ever added once, in __init__) --
        # move the box back on top so it isn't hidden under them.
        if self.qc.layer is not None and self.qc.layer in self.viewer.layers:
            self.viewer.layers.move(
                self.viewer.layers.index(self.qc.layer), len(self.viewer.layers)
            )
        # The new layers were added while the old ones (often the same
        # channel names) were still present, so napari auto-suffixed any
        # collision ("GFP 488 [1]") to keep names unique at that instant.
        # Now that the old ones are gone, restore the clean name.
        for layer in self.layers:
            layer.name = _NAPARI_DEDUP_SUFFIX.sub("", layer.name)
        if not self._contrast_set and done:
            self._set_contrast(done[0])
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


class SessionWatcher:
    """Keeps one already-open napari viewer following whichever live session
    is newest, switching feeds -- discarding the old layers and loading the
    new store -- every time a fresh session starts. Repeated single-timepoint
    test snaps, then a real time-lapse right after, all show up in turn with
    no restart of napari itself in between: it opens once and stays ready.

    With `explicit_store` there's nothing to watch for: it loads once, on the
    first `poll()`, and stays put for the life of the viewer.
    """

    def __init__(
        self,
        viewer,
        *,
        explicit_store: Path | None = None,
        follow: bool = True,
        jobs: Path | None = None,
    ) -> None:
        self.viewer = viewer
        self.explicit_store = explicit_store
        self.follow = follow
        self.jobs = jobs or lanes.jobs_dir()
        self.session_id: str | None = None
        self.follower: LiveFollower | None = None
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            "naparym-live: waiting for a live acquisition to start..."
        )

    def _latest(self) -> tuple[str | None, Path | None]:
        try:
            d = json.loads((self.jobs / LIVE_LATEST_NAME).read_text())
            return d.get("session_id"), Path(d["store"])
        except (OSError, ValueError, KeyError):
            return None, None

    def _switch(self, store: Path, session_id: str | None) -> None:
        """Try to switch to `store`. `live_latest.json` is written the
        moment a session starts (SESSION_START), well before the first
        timepoint's zarr group actually exists on disk -- that only happens
        once its decon+DSR ticket finishes and copies out, which can be
        several seconds later for a real acquisition (a quick test snap
        never exposed this: its own first ticket usually finishes before the
        next poll tick). So a not-yet-existing store here is normal, not an
        error: build the new follower BEFORE touching any existing layers,
        and if that fails because it isn't ready yet, leave everything
        (`self.session_id` included) exactly as it was, so the next poll
        naturally retries the same target instead of raising into napari's
        event loop or leaving the viewer with no layers at all.
        """
        old_layers = list(self.viewer.layers)
        try:
            follower = LiveFollower(self.viewer, store, follow=self.follow)
        except (OSError, KeyError, ValueError) as exc:
            logger.debug("naparym-live: %s not ready yet (%r); retrying", store, exc)
            self.viewer.text_overlay.text = (
                f"naparym-live: session found, waiting for its first "
                f"timepoint ({store.name})..."
            )
            return
        for layer in old_layers:
            self.viewer.layers.remove(layer)
        # The new follower's layers were added while the old ones (often the
        # same channel names -- "GFP 488" showing up in nearly every
        # acquisition) were still present, so napari auto-suffixed any
        # collision ("GFP 488 [1]") to keep names unique at that instant.
        # Now that the old ones are gone, that suffix no longer means
        # anything -- restore the clean name.
        for layer in follower.layers:
            layer.name = _NAPARI_DEDUP_SUFFIX.sub("", layer.name)
        self.session_id = session_id
        self.follower = follower
        self.viewer.title = f"naparym-live: {store.name}"

    def poll(self) -> None:
        if self.explicit_store is not None:
            if self.follower is None:
                self._switch(self.explicit_store, session_id="explicit")
            else:
                self.follower.poll()
            return
        session_id, store = self._latest()
        if store is None:
            return  # nothing new; keep showing the waiting message
        if session_id != self.session_id:
            self._switch(store, session_id)
        elif self.follower is not None:
            self.follower.poll()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="naparym-live", description=__doc__.splitlines()[0]
    )
    ap.add_argument(
        "store",
        nargs="?",
        help="OME-Zarr store or dataset dir (default: follow whichever live "
        "session is newest, switching automatically as new ones start)",
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
    # A typo'd explicit path should still fail fast, before anything opens --
    # but the no-argument "whatever's newest" case never has a bad path to
    # fail on, so it's the SessionWatcher's job, not this lookup's.
    explicit_store = resolve_store(args.store) if args.store else None
    viewer = napari.Viewer(title="naparym-live", ndisplay=3)
    watcher = SessionWatcher(
        viewer, explicit_store=explicit_store, follow=not args.no_follow
    )

    @viewer.bind_key("f")
    def _toggle_follow(_viewer):
        watcher.follow = not watcher.follow
        if watcher.follower is not None:
            watcher.follower.follow = watcher.follow
        _viewer.status = f"follow {'on' if watcher.follow else 'off'}"

    timer = QTimer()
    timer.timeout.connect(watcher.poll)
    timer.start(int(args.poll * 1000))
    watcher.poll()  # pick up an already-running session now, not after the
    # first --poll-second tick
    napari.run()


if __name__ == "__main__":
    main()
