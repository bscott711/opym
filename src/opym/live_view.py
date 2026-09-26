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

The whole live path, camera to screen, with its timings and design
choices: docs/live-view-pipeline.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

from opym import lanes
from opym.ome_zarr_writer import complete_timepoints, image_group, read_progress
from opym.stream import trace
from opym.stream.live_zarr import buffer_name, parse_buffer_name

logger = logging.getLogger(__name__)

# napari's own layer-name uniquification suffix ("GFP 488" -> "GFP 488 [1]"),
# stripped back off once the layer it collided with is gone -- see
# SessionWatcher._switch.
_NAPARI_DEDUP_SUFFIX = re.compile(r" \[\d+\]$")

# Reading a progress file and listing a RAM-disk directory is cheap; a slow
# poll was up to 2 s of the live view's lag.
POLL_S = 0.1
LIVE_LATEST_NAME = "live_latest.json"
QC_LOG_NAME = "live_qc.jsonl"
QC_COLORS = {"ok": "lime", "warn": "orange", "act": "red", "no_cell": "gray"}
# omero hex colour -> napari colormap name (see ome_zarr_writer.CHANNEL_COLORS).
# FF3D3D is the red that stores written before 2026-09-26 name for channel 1:
# shown in magenta too.
_COLORMAPS = {
    "00FF00": "green",
    "FF00FF": "magenta",
    "FF3D3D": "magenta",
    "00B3FF": "cyan",
    "FFC400": "yellow",
}
# Display settings each rebuilt layer takes over from the one it replaces, so
# a new timepoint never resets what the user dialed in.
_CARRIED = (
    "contrast_limits",
    "colormap",
    "gamma",
    "opacity",
    "blending",
    "rendering",
    "interpolation3d",
)
_READ_POOL = None


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


def status_text(
    name: str,
    progress: dict | None,
    now: float | None = None,
    done: list[int] | None = None,
) -> str:
    """`done` overrides the progress file's list (live buffers land first)."""
    if not progress:
        return f"{name}: waiting for the first timepoint"
    done = complete_timepoints(progress) if done is None else done
    n_t = int(progress.get("n_t", 0))
    state = progress.get("state", "running")
    text = f"{name}: {len(done)}/{n_t} timepoints · {state}"
    if state == "running" and done:
        age = (time.time() if now is None else now) - float(
            progress.get("updated_at", 0)
        )
        text += f" · newest t={done[-1]}, updated {age:.0f}s ago"
    return text


def read_volume(arr, t: int, c: int):
    """`arr[t, c]` of a (t, c, z, y, x) zarr array, one z-slab of chunks per
    thread (blosc releases the GIL): two to three times faster than zarr's
    own chunk-by-chunk read of a 1 GB volume."""
    import numpy as np

    global _READ_POOL
    if _READ_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _READ_POOL = ThreadPoolExecutor(16, thread_name_prefix="naparym-read")
    out = np.empty(arr.shape[2:], arr.dtype)
    cz = arr.chunks[2]

    def read(z0: int) -> None:
        out[z0 : z0 + cz] = arr[t, c, z0 : z0 + cz]

    list(_READ_POOL.map(read, range(0, out.shape[0], cz)))
    return out


def map_buffer(path: Path):
    """A view buffer (.npy) mapped read-only with every page faulted in up
    front (MAP_POPULATE: 0.03 s for 1 GB), rather than one fault per 4 KB
    page as napari's texture upload touches it."""
    import math
    import mmap

    import numpy as np
    from numpy.lib import format as npf

    with open(path, "rb") as f:
        version = npf.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = npf.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = npf.read_array_header_2_0(f)
        if fortran:
            raise ValueError(f"{path}: Fortran-order view buffer")
        offset = f.tell()
        m = mmap.mmap(
            f.fileno(),
            0,
            flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
            prot=mmap.PROT_READ,
        )
    return np.frombuffer(m, dtype, count=math.prod(shape), offset=offset).reshape(shape)


class BufferedSeries:
    """One channel of a live session as a (t, z, y, x) array for napari,
    read a whole timepoint at a time: its RAM-disk view buffer, mapped, while
    that exists, else level 0 of the store (`read_volume`) once the store's
    progress file lists it, else zeros (nothing to decode yet).

    Not dask: dask copies every result it computes, a fresh 1 GB allocation
    per channel per timepoint shown, 0.3-2.7 s each on Argus with its free
    memory low. napari takes this mapping as is.
    """

    def __init__(self, store: Path, level0, c: int, buffers_dir: Path) -> None:
        self._store = store
        self._level0 = level0
        self._c = c
        self._dir = Path(buffers_dir)
        self.shape = (level0.shape[0], *level0.shape[2:])
        self.dtype = level0.dtype
        self.ndim = len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def volume(self, t: int):
        import numpy as np

        try:
            return map_buffer(self._dir / buffer_name(t, self._c))
        except (OSError, ValueError):
            pass
        if [t, self._c] in (read_progress(self._store) or {}).get("done", []):
            return read_volume(self._level0, t, self._c)
        return np.zeros(self.shape[1:], self.dtype)  # the kernel's zero page

    def __getitem__(self, key):
        import numpy as np

        key = key if isinstance(key, tuple) else (key,)
        t, rest = key[0], key[1:]
        if isinstance(t, slice):
            ts = range(*t.indices(self.shape[0]))
            if len(ts) == 1:  # napari slices t:t+1, even in 3D: no copy
                return self.volume(ts[0])[rest][np.newaxis]
            return np.stack([self.volume(i)[rest] for i in ts])
        t = int(t)
        return self.volume(t + self.shape[0] if t < 0 else t)[rest]


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


class PaintClock:
    """Traces `painted`: when a timepoint naparym-live has shown is actually
    on screen -- the first frame swapped after its layers were updated.

    `shown` is recorded when `LiveFollower.poll` returns, before Qt has
    painted anything: uploading the new volumes to the GPU and drawing them
    happens in the paint that follows, and that is a hop of its own
    (opym-live-trace's `paint`). vispy's Qt canvas is a QOpenGLWidget, whose
    `frameSwapped` fires once a frame is on the window; without it (another
    backend), the canvas's draw event stands in.
    """

    def __init__(self, jobs: Path | None = None) -> None:
        self.jobs = jobs
        self._pending: list[tuple[float, dict]] = []

    def attach(self, viewer) -> bool:
        try:
            canvas = viewer.window._qt_viewer.canvas
        except AttributeError:  # no Qt window (headless ViewerModel)
            return False
        swapped = getattr(canvas.native, "frameSwapped", None)
        if swapped is not None:
            swapped.connect(self.frame)
        else:
            canvas._scene_canvas.events.draw.connect(lambda _event: self.frame())
        return True

    def expect(self, **fields) -> None:
        """A timepoint was just shown; trace the next frame as its paint."""
        self._pending.append((time.time(), fields))

    def frame(self) -> None:
        if not self._pending:
            return
        now = time.time()
        pending, self._pending = self._pending, []
        for shown_at, fields in pending:
            trace.record(
                "painted",
                name=trace.VIEW_TRACE_NAME,
                jobs=self.jobs,
                paint_s=now - shown_at,
                **fields,
            )


class LiveFollower:
    """Keeps a napari viewer's layers in step with a growing store.

    One single-resolution layer per channel, full resolution: napari's 3D
    view only ever renders the coarsest level of a multiscale layer, which
    for a pyramid of three is a quarter of the resolution. A timepoint is
    read from its uncompressed view buffer (`buffers_dir`, the one-format
    live lane's RAM-disk copy, memory-mapped: no decode) while that exists,
    else from level 0 of the store.

    `replaces`: a previous feed's layers, taken down once this one's are up.
    `painter`: traces when each shown timepoint is painted (`PaintClock`).
    """

    def __init__(
        self,
        viewer,
        store: Path,
        *,
        follow: bool = True,
        session_id: str | None = None,
        jobs: Path | None = None,
        buffers_dir: Path | None = None,
        qc_dir: Path | None = None,
        replaces: list | None = None,
        painter: PaintClock | None = None,
    ) -> None:
        self.viewer = viewer
        self.painter = painter
        self.store = Path(store)
        self.follow = follow
        self.session_id = session_id
        self.jobs = jobs
        self.buffers_dir = Path(buffers_dir) if buffers_dir else None
        self.name = self.store.name.removesuffix("_dsr.ome.zarr").removesuffix(
            ".ome.zarr"
        )
        group, ms, series = self._open()
        self._channels = group.attrs.get("omero", {}).get("channels", [])
        scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
        self._scale = [scale[0]] + scale[2:]
        self._n_c = len(series)
        self._series = series
        self.layers: list = []
        self._replaces = list(replaces or [])
        self._shown: list[int] = []
        self._contrast_set = False
        self._status = ""
        self.qc = QCOverlay(
            viewer,
            Path(qc_dir) if qc_dir else qc_dir_for(self.store),
            self._scale,
            series[0].shape[1],
        )
        viewer.text_overlay.visible = True
        viewer.dims.events.current_step.connect(lambda _e: self._show_text())
        self.poll()

    def _open(self):
        """The DSR image group, its multiscales and, per channel, a fresh
        full-resolution (t, z, y, x) array over the store's current
        metadata: a `BufferedSeries` for a live session, else dask."""
        import dask.array as da

        group = image_group(self.store)
        ms = group.attrs["multiscales"][0]
        level0 = group[ms["datasets"][0]["path"]]
        n_c = level0.shape[1]
        if self.buffers_dir is None:
            full = da.from_zarr(level0)
            return group, ms, [full[:, c] for c in range(n_c)]
        return (
            group,
            ms,
            [
                BufferedSeries(self.store, level0, c, self.buffers_dir)
                for c in range(n_c)
            ],
        )

    def _done(self, progress: dict | None) -> list[int]:
        """Timepoints with every channel viewable: in the store, or with
        every channel's buffer on the RAM disk (those land first)."""
        done = set(complete_timepoints(progress))
        if self.buffers_dir is not None:
            have: dict[int, set[int]] = {}
            try:
                names = [p.name for p in self.buffers_dir.iterdir()]
            except OSError:
                names = []
            for name in names:
                tc = parse_buffer_name(name)
                if tc is not None:
                    have.setdefault(tc[0], set()).add(tc[1])
            done |= {t for t, cs in have.items() if len(cs) >= self._n_c}
        return sorted(done)

    def _add_layer(self, c: int, data, before, visible: bool, contrast, phases: dict):
        """Channel `c`, with `before`'s display settings if it replaces one,
        read once: at the viewer's timepoint.

        A layer reads its data as it's constructed, at t=0 wherever the
        slider is -- during a live session, T0's buffer is long gone, so a
        1 GB decode from the store per channel per timepoint. Built hidden
        it reads nothing; `_slice_dims` (napari 0.7's own per-layer slicing
        hook, private) points it at the viewer's timepoint before it's
        shown. It must be shown before it's added: napari's 3D renderer
        rejects a layer that has never been read.
        """
        from napari.layers import Image

        meta = self._channels
        if before is not None:
            settings = {attr: getattr(before, attr) for attr in _CARRIED}
        else:
            color = meta[c].get("color", "") if c < len(meta) else ""
            settings = {
                "colormap": _COLORMAPS.get(color, "gray"),
                "blending": "additive",
                "contrast_limits": [0, 300],
            }
        if contrast:  # the first real timepoint's, over the empty store's
            settings["contrast_limits"] = contrast[c]
        layer = Image(
            data,
            multiscale=False,
            rgb=False,
            name=meta[c]["label"] if c < len(meta) else f"C{c}",
            scale=self._scale,
            visible=False,
            **settings,
        )
        t0 = time.perf_counter()
        layer._slice_dims(self.viewer.dims)
        layer.visible = True
        t1 = time.perf_counter()
        self.viewer.layers.append(layer)
        if not visible:  # setting it True again would read it again
            layer.visible = False
        phases[f"slice_c{c}"] = t1 - t0
        phases[f"append_c{c}"] = time.perf_counter() - t1
        return layer

    def _remove(self, layers: list) -> None:
        """In one batch: napari runs a full `gc.collect()` and a GPU sync
        after each layer removed outside one."""
        with self.viewer.layers.batched_update():
            for layer in layers:
                if layer is not None:
                    self.viewer.layers.remove(layer)

    def _time_axis_stub(self):
        """A one-voxel stand-in layer spanning this session's timepoints, so
        the slider can be moved before a session's first real layer is in
        (with no time axis yet, napari reads that layer at t=0, then again
        wherever it centres the new slider)."""
        import numpy as np
        from napari.layers import Image

        n_t = self._series[0].shape[0]
        stub = Image(
            np.zeros((n_t, 1, 1, 1), np.uint8), scale=self._scale, name="time axis"
        )
        self.viewer.layers.append(stub)
        return stub

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
        viewer blank), carrying over the display settings (`_CARRIED`).

        The first call builds the layers even with nothing done yet, so a
        session's channels show up as soon as it's found.
        """
        seen_s = time.time()
        progress = read_progress(self.store)
        done = self._done(progress)
        self._status = status_text(self.name, progress, done=done)
        self.qc.poll()
        self._show_text()
        if self.layers and done == self._shown:
            return
        new = [t for t in done if t not in self._shown]
        phases = self._rebuild(done, new)
        self._shown = done
        if not new:
            return
        # Layers are built and the step set; Qt paints once this returns.
        trace.record(
            "shown",
            name=trace.VIEW_TRACE_NAME,
            jobs=self.jobs,
            session_id=self.session_id,
            store=str(self.store),
            timepoints=new,
            seen_s=seen_s,
            build_s=time.time() - seen_s,
            phases={k: round(v, 4) for k, v in phases.items()},
        )
        if self.painter is not None:
            self.painter.expect(session_id=self.session_id, timepoints=new)

    def _rebuild(self, done: list[int], new: list[int]) -> dict[str, float]:
        """Swap in fresh layers, reading each channel once.

        napari reads (slices) a visible layer on every change that touches
        it, a whole 1 GB volume per channel in 3D. Adding the new layers
        read the timepoint already on screen, then moving the time slider
        read the new one, for the old layers as well; opening a session
        read its first timepoint seven times over, three of them from the
        compressed store, 25 s before the window could first paint (Argus,
        2026-09-25). So the old layers are hidden and the slider moved
        first, and each new layer is read once, at the timepoint shown
        (`_add_layer`; `_time_axis_stub` for a session's first layers).

        Returns how long each phase took, in seconds (traced with `shown`).
        """
        phases: dict[str, float] = {}
        tick = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal tick
            now = time.perf_counter()
            phases[name] = now - tick
            tick = now

        prev = self.layers
        old = prev + self._replaces
        target = max(new) if self.follow and new else None
        _group, _ms, self._series = self._open()
        lap("open")
        contrast = None
        if not self._contrast_set and done:
            contrast = self._contrast(done, target)
            lap("contrast")
        visible = [layer.visible for layer in prev]
        hidden = [layer for layer in old if layer.visible]
        fresh = not self.viewer.layers
        layers: list = []
        stub = None
        try:
            for layer in hidden:
                layer.visible = False
            if target is not None:
                if not prev:
                    stub = self._time_axis_stub()
                self.viewer.dims.set_current_step(0, target)
            lap("slider")
            for c, data in enumerate(self._series):
                before = prev[c] if c < len(prev) else None
                shown = visible[c] if c < len(visible) else True
                layers.append(self._add_layer(c, data, before, shown, contrast, phases))
            tick = time.perf_counter()
        except Exception:
            self._remove([*layers, stub])
            for layer in hidden:
                layer.visible = True
            raise
        self._remove([*old, stub])
        lap("remove")
        if fresh:
            self.viewer.reset_view()  # napari fit the view to the stub
        self.layers = layers
        self._replaces = []
        self._contrast_set |= contrast is not None
        # The new image layers land at the end of the layer list, after the
        # QC box -- move the box back on top so it isn't hidden under them.
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
        lap("tidy")
        return phases

    def _contrast(self, done: list[int], target: int | None) -> list:
        """Per channel, from every 4th voxel of the timepoint about to be
        shown (the empty store has no range to go on): the newest one while
        following, so it's read from its RAM-disk buffer, not decoded."""
        import numpy as np

        t = int(self.viewer.dims.current_step[0]) if target is None else target
        if t not in done:
            t = done[-1]
        limits = []
        for series in self._series:
            vol = np.asarray(series[t][::4, ::4, ::4])
            lo, hi = np.percentile(vol[vol > 0], [0.5, 99.9]) if vol.any() else (0, 300)
            limits.append((float(lo), float(max(hi, lo + 1))))
        return limits


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
        painter: PaintClock | None = None,
        title: str = "naparym-live",
    ) -> None:
        self.viewer = viewer
        self.painter = painter
        self.title = title
        self.explicit_store = explicit_store
        self.follow = follow
        self.jobs = jobs or lanes.jobs_dir()
        self.session_id: str | None = None
        self.follower: LiveFollower | None = None
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            "naparym-live: waiting for a live acquisition to start..."
        )

    def _latest(self) -> tuple[str | None, Path | None, dict]:
        """The newest session, the store to show it from -- its RAM-disk
        copy while that exists (one-format lane), else the GPFS one -- and
        the follower's other inputs."""
        try:
            d = json.loads((self.jobs / LIVE_LATEST_NAME).read_text())
            store = Path(d["store"])
        except (OSError, ValueError, KeyError):
            return None, None, {}
        view = d.get("view_store")
        if view and Path(view).exists():
            extra = {"buffers_dir": d.get("buffers_dir"), "qc_dir": d.get("qc_dir")}
            return d.get("session_id"), Path(view), extra
        return d.get("session_id"), store, {"qc_dir": d.get("qc_dir")}

    def _switch(
        self, store: Path, session_id: str | None, extra: dict | None = None
    ) -> None:
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
        event loop or leaving the viewer with no layers at all. The follower
        takes the old layers down itself (`replaces`), once its own are up.
        """
        try:
            follower = LiveFollower(
                self.viewer,
                store,
                follow=self.follow,
                session_id=session_id,
                jobs=self.jobs,
                replaces=list(self.viewer.layers),
                painter=self.painter,
                **(extra or {}),
            )
        except (OSError, KeyError, ValueError) as exc:
            logger.debug("naparym-live: %s not ready yet (%r); retrying", store, exc)
            self.viewer.text_overlay.text = (
                f"naparym-live: session found, waiting for its first "
                f"timepoint ({store.name})..."
            )
            return
        self.session_id = session_id
        self.follower = follower
        self.viewer.title = f"{self.title}: {store.name}"

    def poll(self) -> None:
        if self.explicit_store is not None:
            if self.follower is None:
                self._switch(self.explicit_store, session_id="explicit")
            else:
                self.follower.poll()
            return
        session_id, store, extra = self._latest()
        if store is None:
            return  # nothing new; keep showing the waiting message
        if session_id != self.session_id:
            self._switch(store, session_id, extra)
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
    ap.add_argument(
        "--title", default="naparym-live", help="window title (e.g. for a test stack)"
    )
    args = ap.parse_args(argv)

    import napari
    from napari.utils import resize_dask_cache
    from qtpy.QtCore import QTimer

    resize_dask_cache(0)
    # A typo'd explicit path should still fail fast, before anything opens --
    # but the no-argument "whatever's newest" case never has a bad path to
    # fail on, so it's the SessionWatcher's job, not this lookup's.
    explicit_store = resolve_store(args.store) if args.store else None
    viewer = napari.Viewer(title=args.title, ndisplay=3)
    painter = PaintClock(jobs=lanes.jobs_dir())
    painter.attach(viewer)
    watcher = SessionWatcher(
        viewer,
        explicit_store=explicit_store,
        follow=not args.no_follow,
        painter=painter,
        title=args.title,
    )

    @viewer.bind_key("f")
    def _toggle_follow(_viewer):
        watcher.follow = not watcher.follow
        if watcher.follower is not None:
            watcher.follower.follow = watcher.follow
        _viewer.status = f"follow {'on' if watcher.follow else 'off'}"

    timer = QTimer()
    timer.timeout.connect(watcher.poll)
    # The first tick picks up an already-running session. Not a poll here:
    # before napari.run() the window can't paint, so it stays black while
    # that session loads.
    timer.start(int(args.poll * 1000))
    napari.run()


if __name__ == "__main__":
    main()
