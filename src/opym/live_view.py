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

Fast to scrub and rotate: each channel is one multiscale layer rendered at
full resolution at rest and at half resolution while the time slider moves,
with every timepoint prefetched into this process's RAM (`--cache-gb`) in
the background, so the UI thread only ever uploads. While following, each
channel shows its newest timepoint as soon as it lands (GFP about one stack
before mScarlet: the channels are acquired one after the other).

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
import math
import re
import threading
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
_READ_POOL = None
_LIVE_IMAGE = None
# View buffers a channel keeps mapped (each timepoint is mapped once for its
# slice, its thumbnail level and its contrast, not three times).
MAPPED_BUFFERS = 4
# Threads decoding one volume's chunks (Argus has 96 cores): for what is on
# screen, and, apart so it never queues behind them, for prefetching.
READ_THREADS = 32
PREFETCH_READ_THREADS = 8
# Prefetch jobs in flight at once, nearest the time slider first.
PREFETCH_IN_FLIGHT = 3
_PREFETCH_READ_POOL = None
# Scrubbing: level 1 while the time slider moves; full resolution once it
# has rested this long.
REFINE_AFTER_S = 0.35


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


def no_hugepage_stalls() -> None:
    """Stop numpy asking for transparent huge pages on big arrays.

    numpy madvises every large allocation for huge pages; with Argus's THP
    defrag set to "madvise", each such page fault waits on direct memory
    compaction. Measured 2026-09-26: 30 GB of 127 MB arrays took 101 s,
    single allocations up to 1.9 s, against 15 s and ~65 ms each without
    (the kernel has logged tens of millions of compaction stalls). The
    viewer allocates a volume per channel per timepoint and fills a RAM
    cache of many GB, so it opts out. Same as NUMPY_MADVISE_HUGEPAGE=0."""
    try:
        from numpy._core import multiarray
    except ImportError:  # numpy < 2
        from numpy.core import multiarray
    multiarray._set_madvise_hugepage(False)


def _read_pool(background: bool = False):
    """The decode threads: the UI's own, or the prefetcher's (apart, so a
    read for the screen never waits behind a prefetch)."""
    global _READ_POOL, _PREFETCH_READ_POOL
    from concurrent.futures import ThreadPoolExecutor

    if background:
        if _PREFETCH_READ_POOL is None:
            _PREFETCH_READ_POOL = ThreadPoolExecutor(
                PREFETCH_READ_THREADS, thread_name_prefix="naparym-prefetch-read"
            )
        return _PREFETCH_READ_POOL
    if _READ_POOL is None:
        _READ_POOL = ThreadPoolExecutor(READ_THREADS, thread_name_prefix="naparym-read")
    return _READ_POOL


def read_volume(arr, t: int, c: int, *, background: bool = False):
    """`arr[t, c]` of a (t, c, z, y, x) zarr array, one z-slab of chunks per
    thread (blosc releases the GIL): two to three times faster than zarr's
    own chunk-by-chunk read of a 1 GB volume."""
    import numpy as np

    out = np.empty(arr.shape[2:], arr.dtype)
    cz = arr.chunks[2]

    def read(z0: int) -> None:
        out[z0 : z0 + cz] = arr[t, c, z0 : z0 + cz]

    list(_read_pool(background).map(read, range(0, out.shape[0], cz)))
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


class VolumeCache:
    """Decoded volumes kept in this process's memory, least recently used
    out first, up to `budget_bytes`. In the viewer's own memory, not on the
    RAM disk: /dev/shm is capped (252 GB on Argus) and already holds the raw
    and processed stores. Thread-safe: the prefetcher fills it while the UI
    thread reads."""

    def __init__(self, budget_bytes: float) -> None:
        import threading
        from collections import OrderedDict

        self.budget = float(budget_bytes)
        self.bytes = 0
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            arr = self._items.get(key)
            if arr is not None:
                self._items.move_to_end(key)
            return arr

    def put(self, key, arr) -> None:
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self.bytes -= old.nbytes
            if arr.nbytes > self.budget:
                return
            self._items[key] = arr
            self.bytes += arr.nbytes
            while self.bytes > self.budget:
                _key, dropped = self._items.popitem(last=False)
                self.bytes -= dropped.nbytes

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._items

    def drop(self, match) -> None:
        """Forget every entry whose key `match(key)` accepts."""
        with self._lock:
            for key in [k for k in self._items if match(k)]:
                self.bytes -= self._items.pop(key).nbytes


class ViewCache:
    """The viewer's two caches: full-resolution volumes (level 0, the big
    ones, `full_gb`) and every coarser level (the scrubbing and thumbnail
    levels, 1/8 and 1/64 the size, `small_gb`), so a long session's full
    resolution can't push out the levels that make scrubbing fast."""

    def __init__(self, full_gb: float = 100.0, small_gb: float = 40.0) -> None:
        self.full = VolumeCache(full_gb * 1e9)
        self.small = VolumeCache(small_gb * 1e9)

    def of(self, level: int) -> VolumeCache:
        return self.full if level == 0 else self.small

    def drop_store(self, store: Path) -> None:
        for cache in (self.full, self.small):
            cache.drop(lambda key: key[0] == str(store))


class ChannelSource:
    """Channel `c` of a session, at every level of its pyramid, one whole
    timepoint at a time. A timepoint is served from, in order:

    - this viewer's cache (`ViewCache`);
    - at full resolution, its uncompressed view buffer on the RAM disk,
      memory-mapped, while the lane keeps it (the newest few);
    - the store, once its progress file lists (t, c), decoded (and cached);
    - at a coarser level, the full-resolution buffer subsampled, for the
      moment between the buffer landing and the store being written;
    - during a live session, for a timepoint another channel already has
      but this one hasn't reached, the channel's newest timepoint: the
      channels are acquired one after the other (the 488 stack, then the
      561 stack), so each shows what it has -- GFP about one stack ahead of
      mScarlet;
    - else zeros (the kernel's zero page).

    Not dask: dask copies every result it computes, a fresh 1 GB allocation
    per channel per timepoint shown. `update` is called once per poll with
    what the lane has produced; `served` says which timepoint the last read
    at each level actually returned (a newer one's stand-in, or itself).
    """

    def __init__(
        self,
        store: Path,
        levels: list,
        c: int,
        buffers_dir: Path | None,
        cache: ViewCache,
    ) -> None:
        self.store = Path(store)
        self._levels = levels
        self.c = c
        self._dir = Path(buffers_dir) if buffers_dir else None
        self._cache = cache
        self._maps: dict[int, object] = {}  # newest buffers, mapped once
        self._maps_lock = threading.Lock()
        self.stored: set[int] = set()  # (t, c) in the store
        self.buffered: set[int] = set()  # a view buffer on the RAM disk
        self.live = True
        self.lead: int | None = None  # the newest timepoint any channel has
        self.served: dict[int, int | None] = {}
        n_t = levels[0].shape[0]
        self.shapes = [(n_t, *arr.shape[2:]) for arr in levels]
        self.dtype = levels[0].dtype

    @property
    def has(self) -> set[int]:
        return self.stored | self.buffered

    def update(
        self, stored: set[int], buffered: set[int], live: bool, lead: int | None = None
    ) -> None:
        self.stored, self.buffered, self.live = set(stored), set(buffered), live
        self.lead = lead
        with self._maps_lock:  # the lane trimmed these: let their pages go
            for t in [t for t in self._maps if t not in self.buffered]:
                del self._maps[t]

    def _key(self, t: int, level: int) -> tuple:
        return (str(self.store), self.c, t, level)

    def _buffer(self, t: int):
        """(t)'s view buffer, mapped once and kept for the newest few."""
        if self._dir is None:
            return None
        with self._maps_lock:
            mapped = self._maps.get(t)
        if mapped is not None:
            return mapped
        try:
            mapped = map_buffer(self._dir / buffer_name(t, self.c))
        except (OSError, ValueError):
            return None
        with self._maps_lock:
            self._maps[t] = mapped
            while len(self._maps) > MAPPED_BUFFERS:
                del self._maps[min(self._maps)]
        return mapped

    def _subsample(self, full, level: int):
        import numpy as np

        f = 2**level
        z, y, x = self.shapes[level][1:]
        return np.ascontiguousarray(full[::f, ::f, ::f][:z, :y, :x])

    def load(self, t: int, level: int, *, cache: bool = True, background=False):
        """(t, level) from the cache, the buffer or the store; None if this
        channel hasn't produced t yet."""
        cached = self._cache.of(level).get(self._key(t, level))
        if cached is not None:
            return cached
        if level == 0:
            buf = self._buffer(t)
            if buf is not None:
                return buf
        if t in self.stored:
            if background:
                arr = read_volume(self._levels[level], t, self.c, background=True)
            else:
                arr = read_volume(self._levels[level], t, self.c)
            if cache:
                self._cache.of(level).put(self._key(t, level), arr)
            return arr
        if level > 0:
            buf = self._buffer(t)
            if buf is not None:
                return self._subsample(buf, level)
        return None

    def volume(self, t: int, level: int):
        """What napari shows for (t, level); records the timepoint used."""
        import numpy as np

        arr = self.load(t, level)
        used: int | None = t
        if arr is None:
            ahead = self.live and self.lead is not None and t <= self.lead
            older = [u for u in self.has if u < t] if ahead else []
            used = max(older) if older else None
            arr = self.load(used, level) if used is not None else None
        if arr is None:
            used = None
            arr = np.zeros(self.shapes[level][1:], self.dtype)
        self.served[level] = used
        return arr

    def prefetch(self, t: int, level: int) -> None:
        """Make (t, level) a cache hit; runs on the prefetch pool. A full-
        resolution buffer is copied into the cache, so the timepoint stays
        instant after the lane trims its buffer."""
        import numpy as np

        key = self._key(t, level)
        if key in self._cache.of(level):
            return
        if level == 0:
            buf = self._buffer(t)
            if buf is not None:
                self._cache.full.put(key, np.array(buf))
                return
        if t in self.stored:
            self.load(t, level, background=True)


class LevelSeries:
    """One level of a `ChannelSource` as a (t, z, y, x) array for napari,
    read a whole timepoint at a time."""

    def __init__(self, source: ChannelSource, level: int) -> None:
        self._source = source
        self._level = level
        self.shape = source.shapes[level]
        self.dtype = source.dtype
        self.ndim = len(self.shape)
        self.size = math.prod(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def volume(self, t: int):
        return self._source.volume(t, self._level)

    def __getitem__(self, key):
        import numpy as np

        key = key if isinstance(key, tuple) else (key,)
        t, rest = key[0], key[1:]
        if isinstance(t, slice):
            ts = range(*t.indices(self.shape[0]))
            if len(ts) == 1:  # napari slices t:t+1, even in 3D: no copy
                return self.volume(ts[0])[rest][np.newaxis]
            return _SeriesView(self, ts, rest)
        t = int(t)
        t = t + self.shape[0] if t < 0 else t
        if not 0 <= t < self.shape[0]:
            raise IndexError(f"timepoint {t} out of range 0..{self.shape[0] - 1}")
        return self.volume(t)[rest]


def _sliced_shape(shape: tuple, key: tuple) -> tuple:
    """The shape of `array[key]` for `array.shape == shape`, where `key`
    holds ints and slices, without making the array."""
    out = []
    for i, n in enumerate(shape):
        k = key[i] if i < len(key) else slice(None)
        if isinstance(k, slice):
            out.append(len(range(*k.indices(n))))
    return tuple(out)


class _SeriesView:
    """Several timepoints of a `LevelSeries`, read only once one is picked.

    napari's multiscale slicing indexes the whole time axis first
    (`data[:, :, a:b, c:d]`) and picks the timepoint after
    (`_project_thick_slice`): harmless for lazy dask, but a `LevelSeries`
    reads whole volumes, so taking every timepoint would read them all."""

    def __init__(self, series: LevelSeries, ts: range, rest: tuple) -> None:
        self._series = series
        self._ts = ts
        self._rest = rest
        self.shape = (len(ts), *_sliced_shape(series.shape[1:], rest))
        self.dtype = series.dtype
        self.ndim = len(self.shape)
        self.size = math.prod(self.shape)

    def _one(self, t: int, more: tuple):
        return self._series.volume(t)[self._rest][more]

    def __getitem__(self, key):
        import numpy as np

        key = key if isinstance(key, tuple) else (key,)
        tk, more = key[0], key[1:]
        if isinstance(tk, slice):
            return np.stack([self._one(t, more) for t in self._ts[tk]])
        return self._one(self._ts[int(tk)], more)

    def __array__(self, dtype=None, copy=None):
        import numpy as np

        arr = np.stack([self._one(t, ()) for t in self._ts])
        return arr if dtype is None else arr.astype(dtype)


def live_image_class():
    """`LiveImage`, a napari Image subclass (napari is imported lazily).

    napari renders a multiscale layer in 3D at its coarsest level only: its
    slice request takes the last level (`len(data) - 1`), and
    `Layer._update_draw` pins `_data_level` there on every draw. That is
    why the live layers used to be single-resolution. A `LiveImage` renders
    the level the follower picks instead -- full resolution at rest, level 1
    (1/8 the bytes) while the time slider moves -- through its own slicing
    state (the extension point napari's Labels layer uses), which hands the
    request the levels up to that one. Its thumbnail comes from napari's
    thumbnail level (the coarsest with an axis of 64 or more), not from a
    1 GB full-resolution max. `before_slice(layer, dims)` runs before each
    dims-driven slice, so the level is set before napari reads anything.
    """
    global _LIVE_IMAGE
    if _LIVE_IMAGE is None:
        import dataclasses

        from napari.layers import Image
        from napari.layers.image.image import _ImageSlicingState

        class _LevelsUpTo:
            """A layer's levels as a 3D slice request sees them: it renders
            the last one, so this ends at the level to render; coarser
            levels stay reachable by index (the thumbnail level)."""

            def __init__(self, levels, level: int) -> None:
                self._levels = levels
                self._level = level
                self.dtype = levels.dtype
                self.shape = levels.shape

            def __len__(self) -> int:
                return self._level + 1

            def __getitem__(self, i: int):
                return self._levels[i]

        class _LiveSlicingState(_ImageSlicingState):
            def _make_slice_request_internal(self, **kwargs):
                request = super()._make_slice_request_internal(**kwargs)
                if request.multiscale and request.slice_input.ndisplay == 3:
                    request = dataclasses.replace(
                        request,
                        data=_LevelsUpTo(request.data, self.layer.data_level),
                        id=request.id,
                    )
                return request

        class LiveImage(Image):
            def __init__(self, *args, before_slice=None, **kwargs) -> None:
                self._before_slice = before_slice
                self._live_level = 0
                super().__init__(*args, **kwargs)
                self._data_level = 0

            def _get_layer_slicing_state(self, data, cache):
                return _LiveSlicingState(self, data, cache)

            def _update_draw(self, *args, **kwargs) -> None:
                super()._update_draw(*args, **kwargs)
                self._data_level = self._live_level

            def _slice_dims(self, dims, force: bool = False) -> None:
                if self._before_slice is not None:
                    self._before_slice(self, dims)
                super()._slice_dims(dims, force)

            def set_level(self, level: int) -> bool:
                """Render `level` from the next slice on; True if it changed."""
                self._live_level = level
                if self._data_level == level:
                    return False
                self._data_level = level
                return True

        _LIVE_IMAGE = LiveImage
    return _LIVE_IMAGE


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

    One `LiveImage` layer per channel over the store's whole pyramid (a
    `ChannelSource` per channel): full resolution at rest, level 1 while
    the time slider moves, refined to full resolution in the background
    once it has rested `REFINE_AFTER_S` (the UI thread then only uploads).
    While following, each channel shows its newest processed timepoint the
    moment it lands -- the channels are acquired one after the other, so
    GFP about one stack ahead of mScarlet. Every timepoint the lane
    produces is prefetched into this viewer's cache (`ViewCache`) in the
    background: its coarse levels from the store, its full resolution
    copied from the RAM-disk buffer while that exists, so moving through the
    session never decodes on the UI thread.

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
        cache: ViewCache | None = None,
        prefetch_pool=None,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

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
        group = image_group(self.store)
        ms = group.attrs["multiscales"][0]
        levels = [group[d["path"]] for d in ms["datasets"]]
        self._channels = group.attrs.get("omero", {}).get("channels", [])
        scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
        self._scale = [scale[0]] + scale[2:]
        self.cache = cache if cache is not None else ViewCache()
        self._pool = prefetch_pool or ThreadPoolExecutor(
            4, thread_name_prefix="naparym-prefetch"
        )
        self.sources = [
            ChannelSource(self.store, levels, c, self.buffers_dir, self.cache)
            for c in range(levels[0].shape[1])
        ]
        self._n_levels = len(levels)
        self.layers: list = []
        self._replaces = list(replaces or [])
        self._shown: list[int] = []  # every channel shown with its own data
        self._shown_c: dict[int, set[int]] = {src.c: set() for src in self.sources}
        self._submitted: set[tuple[int, int, int]] = set()  # (c, t, level)
        self._inflight: list = []
        self._contrast_set: set[int] = set()
        self._followed: int | None = None
        self._driving = False
        self._slider_t: int | None = None
        self._scrub_at: float | None = None
        self._refine: tuple | None = None
        self._status = ""
        self.qc = QCOverlay(
            viewer,
            Path(qc_dir) if qc_dir else qc_dir_for(self.store),
            self._scale,
            self.sources[0].shapes[0][1],
        )
        viewer.text_overlay.visible = True
        viewer.dims.events.current_step.connect(self._on_step)
        self.poll()

    def close(self) -> None:
        """Stop following (a newer session took over) and free its cache."""
        self.viewer.dims.events.current_step.disconnect(self._on_step)
        self.cache.drop_store(self.store)

    # --- what the lane has produced ----------------------------------------

    def _update_sources(self, progress: dict | None) -> None:
        """Per channel: timepoints in the store and with a buffer. A live
        session's store has no progress file until its first timepoint is
        written: nothing is in it yet. An explicitly opened store without
        one is a finished dataset: all of it is there."""
        buffered: dict[int, set[int]] = {src.c: set() for src in self.sources}
        if self.buffers_dir is not None:
            try:
                names = [p.name for p in self.buffers_dir.iterdir()]
            except OSError:
                names = []
            for name in names:
                tc = parse_buffer_name(name)
                if tc is not None and tc[1] in buffered:
                    buffered[tc[1]].add(tc[0])
        n_t = self.sources[0].shapes[0][0]
        finished = progress is None and self.buffers_dir is None
        done = (progress or {}).get("done", [])
        live = not finished and (progress or {}).get("state", "running") == "running"
        stored = {
            src.c: set(range(n_t)) if finished else {t for t, c in done if c == src.c}
            for src in self.sources
        }
        lead = max(
            (max(stored[c] | buffered[c]) for c in stored if stored[c] | buffered[c]),
            default=None,
        )
        for src in self.sources:
            src.update(stored[src.c], buffered[src.c], live, lead)

    def _newest(self) -> int | None:
        has = [max(src.has) for src in self.sources if src.has]
        return max(has) if has else None

    def _prefetch_new(self) -> None:
        """Keep `PREFETCH_IN_FLIGHT` prefetch jobs going: first every full-
        resolution buffer not yet copied (the lane trims them), then coarse
        levels nearest the time slider, so where the user is scrubbing
        fills first."""
        self._inflight = [f for f in self._inflight if not f.done()]
        free = PREFETCH_IN_FLIGHT - len(self._inflight)
        if free <= 0:
            return
        here = int(self.viewer.dims.current_step[0])
        jobs = []
        for src in self.sources:
            jobs += [(0, t, src, 0) for t in src.buffered]
            jobs += [
                (abs(t - here), t, src, level)
                for t in src.stored
                for level in range(1, self._n_levels)
            ]
        jobs = [j for j in jobs if (j[2].c, j[1], j[3]) not in self._submitted]
        for _d, t, src, level in sorted(jobs, key=lambda j: (j[0], j[3], -j[1]))[:free]:
            self._submitted.add((src.c, t, level))
            self._inflight.append(self._pool.submit(src.prefetch, t, level))

    # --- polling ------------------------------------------------------------

    def _show_text(self) -> None:
        qc = self.qc.summary(int(self.viewer.dims.current_step[0]))
        self.viewer.text_overlay.text = self._status + (f"\n{qc}" if qc else "")

    def _status_text(self, progress: dict | None) -> str:
        every = set.intersection(*(src.has for src in self.sources))
        text = status_text(self.name, progress, done=sorted(every))
        newest = [max(src.has) if src.has else None for src in self.sources]
        if None not in newest and len(set(newest)) > 1:
            text += " · " + ", ".join(
                f"{self._label(src.c)} t={n}"
                for src, n in zip(self.sources, newest, strict=True)
            )
        return text

    def _label(self, c: int) -> str:
        return self._channels[c]["label"] if c < len(self._channels) else f"C{c}"

    def poll(self) -> None:
        """Show what the lane has produced since the last poll.

        The first call builds a session's layers (`_build`), even with
        nothing done yet, so its channels show up as soon as it's found.
        After that, following, a new timepoint only moves the time slider:
        napari re-slices the same layers, each reading that timepoint once
        (its mapped view buffer, or the cache), and a channel that reaches
        the timepoint on screen later is re-read in place (`_follow`).

        Until 2026-09-26 every timepoint rebuilt the layers from scratch: on
        2026-09-25 the volume stayed black in an open window while dask
        arrays were reassigned onto existing layers, and a fresh build was
        the one thing known to work. That cost about 1 s per timepoint --
        two new layers, then tearing down the old two -- and a full-volume
        thumbnail per channel on every step, all on the UI thread, which is
        what made rotating and scrubbing sluggish. Nothing is reassigned
        now: the layers' data reads each timepoint afresh.
        """
        seen_s = time.time()
        progress = read_progress(self.store)
        self._update_sources(progress)
        self._status = self._status_text(progress)
        self.qc.poll()
        self._show_text()
        self._prefetch_new()
        phases = self._follow() if self.layers else self._build()
        self._refine_step()
        for t, channels in self._newly_shown().items():
            trace.record(
                "shown",
                name=trace.VIEW_TRACE_NAME,
                jobs=self.jobs,
                session_id=self.session_id,
                store=str(self.store),
                timepoints=[t],
                channels=channels,
                seen_s=seen_s,
                build_s=time.time() - seen_s,
                phases={k: round(v, 4) for k, v in phases.items()},
            )
            if self.painter is not None:
                self.painter.expect(
                    session_id=self.session_id, timepoints=[t], channels=channels
                )

    def _newly_shown(self) -> dict[int, list[int]]:
        """Timepoints a channel now shows with its own data, for the first
        time (not a stand-in): what `shown` traces."""
        t = int(self.viewer.dims.current_step[0])
        out: dict[int, list[int]] = {}
        for src, layer in zip(self.sources, self.layers, strict=False):
            if src.served.get(layer.data_level) == t and t not in self._shown_c[src.c]:
                self._shown_c[src.c].add(t)
                out.setdefault(t, []).append(src.c)
        if out:
            self._shown = sorted(set.intersection(*self._shown_c.values()))
        return out

    # --- following ------------------------------------------------------------

    def _drive(self, t: int) -> None:
        """Move the time slider for the follower: full resolution."""
        self._driving = True
        try:
            self.viewer.dims.set_current_step(0, t)
        finally:
            self._driving = False
        self._slider_t = t

    def _before_slice(self, layer, dims) -> None:
        """Pick the level a dims-driven slice reads, before it reads: full
        resolution when the follower moves the slider, level 1 when the user
        does (until it rests), unchanged otherwise (rotating, toggling)."""
        t = int(dims.current_step[0])
        if self._driving or self._slider_t is None:
            layer.set_level(0)
        elif t != self._slider_t:
            layer.set_level(1 if self._n_levels > 1 else 0)
            self._scrub_at = time.monotonic()
            self._refine = None

    def _on_step(self, _event=None) -> None:
        self._slider_t = int(self.viewer.dims.current_step[0])
        self._show_text()

    def _follow(self) -> dict[str, float]:
        """Follow the newest timepoint any channel has, and re-read in place
        a channel that has just reached the timepoint on screen."""
        phases: dict[str, float] = {}
        t0 = time.perf_counter()
        newest = self._newest()
        if self.follow and newest is not None and newest != self._followed:
            self._followed = newest
            if newest != int(self.viewer.dims.current_step[0]):
                self._drive(newest)
                phases["slice"] = time.perf_counter() - t0
        t = int(self.viewer.dims.current_step[0])
        for src, layer in zip(self.sources, self.layers, strict=False):
            if src.c not in self._contrast_set and src.has:
                self._set_contrast(src, layer)
            if layer.visible and t in src.has and src.served.get(layer.data_level) != t:
                t1 = time.perf_counter()
                layer.refresh()
                phases[f"refresh_c{src.c}"] = time.perf_counter() - t1
        return phases

    def _refine_step(self) -> None:
        """Once the slider has rested, read the timepoint's full resolution
        in the background, then show it (only the upload is left)."""
        if self._scrub_at is None:
            return
        t = int(self.viewer.dims.current_step[0])
        if self._refine is None:
            if time.monotonic() - self._scrub_at < REFINE_AFTER_S:
                return
            futures = [
                self._pool.submit(src.prefetch, t, 0)
                for src in self.sources
                if t in src.has
            ]
            self._refine = (t, futures, time.perf_counter())
            return
        rt, futures, started = self._refine
        if rt != t or not all(f.done() for f in futures):
            return
        for layer in self.layers:
            if layer.set_level(0) and layer.visible:
                layer.refresh()
        self._refine = None
        self._scrub_at = None
        trace.record(
            "refined",
            name=trace.VIEW_TRACE_NAME,
            jobs=self.jobs,
            session_id=self.session_id,
            t=t,
            read_s=time.perf_counter() - started,
        )

    def _set_contrast(self, src: ChannelSource, layer) -> None:
        """From every 4th voxel of the channel's newest timepoint (the empty
        store it started from has no range to go on)."""
        import numpy as np

        vol = src.load(max(src.has), 0)
        if vol is None:
            return
        sub = np.asarray(vol[::4, ::4, ::4])
        lo, hi = np.percentile(sub[sub > 0], [0.5, 99.9]) if sub.any() else (0, 300)
        layer.contrast_limits = (float(lo), float(max(hi, lo + 1)))
        self._contrast_set.add(src.c)

    # --- the session's first build ----------------------------------------

    def _add_layer(self, src: ChannelSource):
        """Channel `src.c`'s layer, read once: at the viewer's timepoint.

        A layer reads its data as it's constructed, at t=0 wherever the
        slider is. Built hidden it reads nothing; `_slice_dims` (napari
        0.7's own per-layer slicing hook, private) points it at the viewer's
        timepoint before it's shown. It must be shown before it's added:
        napari's 3D renderer rejects a layer that has never been read.
        """
        meta = self._channels
        color = meta[src.c].get("color", "") if src.c < len(meta) else ""
        levels = [LevelSeries(src, level) for level in range(self._n_levels)]
        layer = live_image_class()(
            levels if len(levels) > 1 else levels[0],
            multiscale=len(levels) > 1,
            rgb=False,
            name=self._label(src.c),
            scale=self._scale,
            visible=False,
            colormap=_COLORMAPS.get(color, "gray"),
            blending="additive",
            contrast_limits=[0, 300],
            before_slice=self._before_slice,
        )
        layer._slice_dims(self.viewer.dims)
        layer.visible = True
        self.viewer.layers.append(layer)
        if src.has:
            self._set_contrast(src, layer)
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

        n_t = self.sources[0].shapes[0][0]
        stub = Image(
            np.zeros((n_t, 1, 1, 1), np.uint8), scale=self._scale, name="time axis"
        )
        self.viewer.layers.append(stub)
        return stub

    def _build(self) -> dict[str, float]:
        """Build the session's layers, each reading one timepoint once (the
        newest while following), then take a previous session's down.

        Opening a session used to read its first timepoint seven times
        over, three of them from the compressed store, 25 s before the
        window could first paint (Argus, 2026-09-25). So the old layers are
        hidden and the slider moved first, and each new layer is read once,
        at the timepoint shown (`_add_layer`; `_time_axis_stub` so the
        slider can move before any layer is in)."""
        phases: dict[str, float] = {}
        t0 = time.perf_counter()
        old = self._replaces
        target = self._newest() if self.follow else None
        hidden = [layer for layer in old if layer.visible]
        fresh = not self.viewer.layers
        layers: list = []
        stub = None
        try:
            for layer in hidden:
                layer.visible = False
            if target is not None:
                stub = self._time_axis_stub()
                self._drive(target)
                self._followed = target
            self._slider_t = int(self.viewer.dims.current_step[0])
            for src in self.sources:
                layers.append(self._add_layer(src))
        except Exception:
            self._remove([*layers, stub])
            for layer in hidden:
                layer.visible = True
            raise
        self._remove([*old, stub])
        if fresh:
            self.viewer.reset_view()  # napari fit the view to the stub
        self.layers = layers
        self._replaces = []
        # The new image layers land at the end of the layer list, after the
        # QC box -- move the box back on top so it isn't hidden under them.
        if self.qc.layer is not None and self.qc.layer in self.viewer.layers:
            self.viewer.layers.move(
                self.viewer.layers.index(self.qc.layer), len(self.viewer.layers)
            )
        # Added while the old ones (often the same channel names) were still
        # present, napari auto-suffixed any collision ("GFP 488 [1]"); now
        # that the old ones are gone, restore the clean name.
        for layer in self.layers:
            layer.name = _NAPARI_DEDUP_SUFFIX.sub("", layer.name)
        phases["build"] = time.perf_counter() - t0
        return phases


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
        cache: ViewCache | None = None,
        prefetch_pool=None,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.viewer = viewer
        self.painter = painter
        self.title = title
        self.cache = cache if cache is not None else ViewCache()
        self._pool = prefetch_pool or ThreadPoolExecutor(
            4, thread_name_prefix="naparym-prefetch"
        )
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
                cache=self.cache,
                prefetch_pool=self._pool,
                **(extra or {}),
            )
        except (OSError, KeyError, ValueError) as exc:
            logger.debug("naparym-live: %s not ready yet (%r); retrying", store, exc)
            self.viewer.text_overlay.text = (
                f"naparym-live: session found, waiting for its first "
                f"timepoint ({store.name})..."
            )
            return
        if self.follower is not None:
            self.follower.close()
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
    ap.add_argument(
        "--cache-gb",
        type=float,
        default=100.0,
        help="RAM for full-resolution timepoints kept for scrubbing (coarse "
        "levels get another 40 GB)",
    )
    args = ap.parse_args(argv)

    no_hugepage_stalls()
    import napari
    from qtpy.QtCore import QTimer

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
        cache=ViewCache(full_gb=args.cache_gb),
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
