"""Tests for opym.live_view (`naparym-live`)."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from opym import live_view
from opym import ome_zarr_writer as w


def _store(tmp_path, n_t=3):
    out = tmp_path / "Cell_005" / "viewer" / "Cell_005_dsr.ome.zarr"
    out.parent.mkdir(parents=True)
    w.create_store(
        out,
        n_t=n_t,
        n_c=2,
        shape_zyx=(8, 16, 12),
        dtype=np.uint16,
        channel_labels=["GFP 488", "mScarlet 561"],
    )
    return out


class _NoPrefetch:
    """A prefetch pool that does nothing, so a test counts only the reads
    the UI thread makes."""

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future

        future = Future()
        future.set_result(None)
        return future


def _shown_data(layer):
    """What napari holds for display: the layer's current slice."""
    return np.asarray(layer._slice.image.raw)


def _finish(store, t, done, n_t=3, state="running"):
    for c in range(2):
        vol = np.full((8, 16, 12), 100 * (t + 1) + 10 * c, dtype=np.uint16)
        w.write_timepoint(store, t, c, vol)
    done.extend([[t, 0], [t, 1]])
    w.write_progress(store, n_t=n_t, n_c=2, done=done, state=state)


def test_resolve_store_from_path_dataset_dir_or_latest_session(tmp_path):
    store = _store(tmp_path)
    assert live_view.resolve_store(str(store)) == store
    assert live_view.resolve_store(str(tmp_path / "Cell_005")) == store
    (tmp_path / "live_latest.json").write_text(json.dumps({"store": str(store)}))
    assert live_view.resolve_store(None, jobs=tmp_path) == store
    with pytest.raises(FileNotFoundError):
        live_view.resolve_store(None, jobs=tmp_path / "nothing")
    with pytest.raises(FileNotFoundError):
        live_view.resolve_store(str(tmp_path / "empty_dataset"))


def test_status_text():
    assert "waiting" in live_view.status_text("Cell_005", None)
    running = {
        "state": "running",
        "n_t": 100,
        "n_c": 2,
        "updated_at": 1000.0,
        "done": [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]],
    }
    text = live_view.status_text("Cell_005", running, now=1004.0)
    assert "2/100 timepoints" in text and "newest t=1" in text and "4s ago" in text
    complete = dict(running, state="complete")
    assert live_view.status_text("Cell_005", complete).endswith("complete")


def test_follower_tracks_new_timepoints(tmp_path):
    """Against napari's ViewerModel: the follower only uses layers, dims and
    the text overlay, so no Qt/OpenGL is needed."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel
    from napari.utils import resize_dask_cache

    resize_dask_cache(0)
    store = _store(tmp_path)
    done: list = []
    _finish(store, 0, done)

    viewer = ViewerModel()
    follower = live_view.LiveFollower(viewer, store)
    assert [layer.name for layer in follower.layers] == ["GFP 488", "mScarlet 561"]
    assert viewer.dims.current_step[0] == 0
    lo, hi = follower.layers[0].contrast_limits
    assert lo <= 100 <= hi  # from the first real timepoint, not the empty store

    first_layer = follower.layers[1]
    user_limits = (5.0, 900.0)
    follower.layers[0].contrast_limits = user_limits

    _finish(store, 1, done)
    follower.poll()
    assert viewer.dims.current_step[0] == 1  # follow mode jumped to it
    assert "2/3 timepoints" in viewer.text_overlay.text
    assert [layer.name for layer in follower.layers] == ["GFP 488", "mScarlet 561"]
    # What napari reads for t=1 is the new data, not cached zeros.
    assert int(_shown_data(follower.layers[1]).max()) == 210
    # Full resolution at rest: napari alone would render a multiscale layer
    # at its coarsest level in 3D.
    assert follower.layers[1].data_level == 0
    assert tuple(follower.layers[1].level_shapes[0]) == (3, 8, 16, 12)
    # The same layers, re-sliced at t=1 (no rebuild per timepoint: see
    # poll()'s docstring), and what napari holds for display is t=1's data.
    assert follower.layers[1] is first_layer and first_layer in viewer.layers
    assert int(np.asarray(first_layer._slice.image.raw).max()) == 210
    # A new timepoint never touches what the user dialed in.
    assert follower.layers[0].contrast_limits == list(user_limits)

    follower.follow = False
    _finish(store, 2, done, state="complete")
    follower.poll()
    assert viewer.dims.current_step[0] == 1
    assert viewer.text_overlay.text.endswith("complete")


def test_follower_rereads_the_timepoint_on_screen_when_it_comes_in(tmp_path):
    """Not following, parked on a timepoint that isn't processed yet: when
    it comes in, the layers on screen re-read it in place."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel
    from napari.utils import resize_dask_cache

    resize_dask_cache(0)
    store = _store(tmp_path)
    done: list = []
    _finish(store, 0, done)
    viewer = ViewerModel()
    follower = live_view.LiveFollower(viewer, store, follow=False)
    viewer.dims.set_current_step(0, 2)
    layer = follower.layers[0]
    assert int(np.asarray(layer._slice.image.raw).max()) == 0  # not in yet
    _finish(store, 2, done)
    follower.poll()
    assert follower.layers[0] is layer
    assert int(np.asarray(layer._slice.image.raw).max()) == 300


def _qc_line(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def test_box_edges_make_a_closed_box():
    edges = live_view.box_edges(3, 0, 0, 0, 2, 4, 6)
    assert len(edges) == 12
    assert all(e.shape == (2, 4) and (e[:, 0] == 3).all() for e in edges)


def test_qc_boxes_and_verdicts_follow_the_log(tmp_path):
    pytest.importorskip("napari")
    from napari.components import ViewerModel
    from napari.utils import resize_dask_cache

    resize_dask_cache(0)
    store = _store(tmp_path)
    done: list = []
    _finish(store, 0, done)
    qc = live_view.qc_dir_for(store)
    assert qc == tmp_path / "Cell_005" / "qc"
    qc.mkdir()
    log = qc / "live_qc.jsonl"
    _qc_line(log, {"stage": "session", "session_id": "s1"})
    _qc_line(log, {"session_id": "other", "t": 0, "stage": "raw", "verdict": "ok"})
    raw = {
        "session_id": "s1",
        "t": 0,
        "stage": "raw",
        "verdict": "act",
        "flags": ["clipped_scan_high"],
        "advice": [{"text": "Shift the scan window."}],
    }
    _qc_line(log, raw)
    box = {"found": True, "bbox_zyx": [1, 2, 3, 6, 10, 9]}
    _qc_line(log, {"session_id": "s1", "t": 0, "stage": "dsr", "boxes": {"GFP": box}})

    viewer = ViewerModel()
    follower = live_view.LiveFollower(viewer, store)
    layer = follower.qc.layer
    assert layer is not None and layer.nshapes == 12
    assert list(layer.scale) == list(follower.layers[0].scale)
    np.testing.assert_allclose(layer.edge_color[0], [1, 0, 0, 1])  # act -> red
    assert (
        "act" in viewer.text_overlay.text
        and "Shift the scan" in viewer.text_overlay.text
    )

    # A half-written line is left for the next poll.
    with open(log, "a") as f:
        f.write('{"session_id": "s1", "t": 1, "stage": "raw", "verdict": "ok"')
    follower.poll()
    assert 1 not in follower.qc.raw
    with open(log, "a") as f:
        f.write("}\n")
    _qc_line(log, {"session_id": "s1", "t": 1, "stage": "dsr", "boxes": {"GFP": box}})
    _finish(store, 1, done)
    follower.poll()
    assert layer.nshapes == 24
    np.testing.assert_allclose(layer.edge_color[-1], [0, 1, 0, 1])  # ok -> lime
    assert "t=1: ok" in viewer.text_overlay.text
    # poll() rebuilds the image layers, appending them after whatever's
    # already in the viewer -- including the QC box, added once in __init__
    # -- so without re-asserting z-order the box would end up hidden
    # underneath the (additive-blended) image data instead of on top of it.
    assert viewer.layers.index(layer) == len(viewer.layers) - 1


def _write_latest(jobs, store, session_id):
    jobs.mkdir(exist_ok=True)
    (jobs / live_view.LIVE_LATEST_NAME).write_text(
        json.dumps({"store": str(store), "session_id": session_id})
    )


def test_watcher_opens_with_nothing_and_picks_up_the_first_session(tmp_path):
    """The window is meant to open before any data exists -- that's the
    whole point (pay napari's startup cost once, then leave it open)."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    viewer = ViewerModel()
    watcher = live_view.SessionWatcher(viewer, jobs=jobs)
    assert "waiting for a live acquisition" in viewer.text_overlay.text
    assert watcher.follower is None

    watcher.poll()  # still nothing
    assert watcher.follower is None

    store = _store(tmp_path)
    _write_latest(jobs, store, "s1")
    watcher.poll()
    assert watcher.follower is not None
    assert watcher.session_id == "s1"
    assert [layer.name for layer in viewer.layers] == ["GFP 488", "mScarlet 561"]


def test_watcher_survives_a_session_recorded_before_its_store_exists(tmp_path):
    """Real bug hit live on Argus: live_latest.json is written at
    SESSION_START, well before the first timepoint's zarr group is actually
    created on disk (that waits on its decon+DSR ticket). A poll landing in
    that gap must retry quietly, not raise GroupNotFoundError into napari's
    event loop, and it must not blank out whatever was already showing."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    jobs = tmp_path / "jobs"
    not_yet = tmp_path / "StreamMe" / "viewer" / "StreamMe_dsr.ome.zarr"
    _write_latest(jobs, not_yet, "real-acq")

    viewer = ViewerModel()
    watcher = live_view.SessionWatcher(viewer, jobs=jobs)
    watcher.poll()  # must not raise
    assert watcher.follower is None
    assert watcher.session_id is None  # so the same target is retried
    assert "waiting for its first timepoint" in viewer.text_overlay.text

    watcher.poll()  # still not there; retrying must not raise either
    assert watcher.follower is None

    not_yet.parent.mkdir(parents=True)
    _store(tmp_path).rename(not_yet)  # the ticket finishes; the store appears
    watcher.poll()
    assert watcher.follower is not None
    assert watcher.session_id == "real-acq"
    assert [layer.name for layer in viewer.layers] == ["GFP 488", "mScarlet 561"]


def test_watcher_keeps_the_old_feed_visible_while_a_switch_target_is_not_ready(
    tmp_path,
):
    """A session already showing must not vanish while the watcher waits for
    the NEXT session's store to become ready."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    jobs = tmp_path / "jobs"
    first = _store(tmp_path, n_t=1)
    _write_latest(jobs, first, "s1")
    viewer = ViewerModel()
    watcher = live_view.SessionWatcher(viewer, jobs=jobs)
    watcher.poll()
    assert [layer.name for layer in viewer.layers] == ["GFP 488", "mScarlet 561"]

    not_yet = tmp_path / "StreamMe" / "viewer" / "StreamMe_dsr.ome.zarr"
    _write_latest(jobs, not_yet, "s2")
    watcher.poll()  # s2's store isn't there yet
    assert watcher.session_id == "s1"  # still showing the first session
    assert [layer.name for layer in viewer.layers] == ["GFP 488", "mScarlet 561"]


def test_watcher_switches_feeds_when_a_new_session_starts(tmp_path):
    """A quick alignment snap, then the real acquisition right after -- same
    window, no restart, old layers gone before the new ones load."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    jobs = tmp_path / "jobs"
    first = _store(tmp_path, n_t=1)
    _write_latest(jobs, first, "snap-1")

    viewer = ViewerModel()
    watcher = live_view.SessionWatcher(viewer, jobs=jobs)
    watcher.poll()
    assert len(viewer.layers) == 2
    first_layer = viewer.layers[0]

    second = tmp_path / "Cell_006" / "viewer" / "Cell_006_dsr.ome.zarr"
    second.parent.mkdir(parents=True)
    w.create_store(
        second,
        n_t=5,
        n_c=1,
        shape_zyx=(8, 16, 12),
        dtype=np.uint16,
        channel_labels=["GFP 488"],
    )
    _write_latest(jobs, second, "real-acq")
    watcher.poll()

    assert watcher.session_id == "real-acq"
    assert [layer.name for layer in viewer.layers] == ["GFP 488"]
    assert first_layer not in viewer.layers  # the old session's layers are gone
    assert "Cell_006" in viewer.title


def test_watcher_with_an_explicit_store_loads_once_and_ignores_live_latest(tmp_path):
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    store = _store(tmp_path)
    jobs = tmp_path / "jobs"
    other = tmp_path / "Cell_006" / "viewer" / "Cell_006_dsr.ome.zarr"
    other.parent.mkdir(parents=True)
    w.create_store(
        other,
        n_t=1,
        n_c=1,
        shape_zyx=(8, 16, 12),
        dtype=np.uint16,
        channel_labels=["GFP 488"],
    )
    _write_latest(jobs, other, "s-other")

    viewer = ViewerModel()
    watcher = live_view.SessionWatcher(viewer, explicit_store=store, jobs=jobs)
    watcher.poll()
    assert [layer.name for layer in viewer.layers] == ["GFP 488", "mScarlet 561"]
    watcher.poll()  # a second poll must not switch to "other" via live_latest.json
    assert [layer.name for layer in viewer.layers] == ["GFP 488", "mScarlet 561"]


def test_resolve_store_no_argument_is_a_one_shot_check_not_a_wait(tmp_path):
    """The no-argument case's waiting is SessionWatcher's job now; a direct
    call here still fails immediately if nothing is there yet."""
    with pytest.raises(FileNotFoundError, match="No live session recorded"):
        live_view.resolve_store(None, jobs=tmp_path / "nothing")


def test_resolve_store_bad_explicit_path_fails_immediately():
    with pytest.raises(FileNotFoundError, match="No \\*_dsr.ome.zarr store"):
        live_view.resolve_store("/nonexistent/path/typo")


def test_follower_traces_each_newly_shown_timepoint(tmp_path):
    """opym-live-trace's last hop: when naparym-live built the layers for t."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    from opym.stream import trace

    jobs = tmp_path / "jobs"
    store = _store(tmp_path)
    _write_latest(jobs, store, "sess-v")
    watcher = live_view.SessionWatcher(ViewerModel(), jobs=jobs)
    watcher.poll()
    done: list = []
    _finish(store, 0, done)
    watcher.poll()
    watcher.poll()  # nothing new: no second event
    [shown] = trace.read(trace.VIEW_TRACE_NAME, jobs=jobs)
    assert shown["ev"] == "shown" and shown["session_id"] == "sess-v"
    assert shown["timepoints"] == [0]
    assert shown["build_s"] >= 0 and shown["seen_s"] <= shown["at"]
    # The layers were built when the session was found; t=0 is read into
    # them in place (slider moved to it, or re-read if already there).
    assert shown["channels"] == [0, 1]
    assert set(shown["phases"]) in ({"slice"}, {"refresh_c0", "refresh_c1"})


def test_paint_clock_traces_the_first_frame_after_a_timepoint_is_shown(tmp_path):
    """`painted` closes opym-live-trace's view path: the paint after `shown`
    (texture upload + draw), traced once per shown batch."""
    from opym.stream import trace

    clock = live_view.PaintClock(jobs=tmp_path)
    clock.frame()  # a repaint with nothing shown: not traced
    clock.expect(session_id="sess-p", timepoints=[4])
    clock.frame()
    clock.frame()  # later frames (a rotation, say) are not paints of t=4
    [painted] = trace.read(trace.VIEW_TRACE_NAME, jobs=tmp_path)
    assert painted["ev"] == "painted" and painted["timepoints"] == [4]
    assert painted["session_id"] == "sess-p" and painted["paint_s"] >= 0


def test_paint_clock_needs_a_qt_window():
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    assert live_view.PaintClock().attach(ViewerModel()) is False


def test_watcher_hands_its_paint_clock_to_each_follower(tmp_path):
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    from opym.stream import trace

    jobs = tmp_path / "jobs"
    store = _store(tmp_path)
    _write_latest(jobs, store, "sess-w")
    clock = live_view.PaintClock(jobs=jobs)
    watcher = live_view.SessionWatcher(ViewerModel(), jobs=jobs, painter=clock)
    watcher.poll()
    done: list = []
    _finish(store, 0, done)
    watcher.poll()
    clock.frame()
    evs = [e["ev"] for e in trace.read(trace.VIEW_TRACE_NAME, jobs=jobs)]
    assert evs == ["shown", "painted"]


def test_follower_shows_a_timepoint_from_its_buffers_before_the_store(tmp_path):
    """The one-format lane drops each (t, c) as an uncompressed .npy before
    writing the store: the view must not wait for the store."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel
    from napari.utils import resize_dask_cache

    from opym.stream.live_zarr import buffer_name

    resize_dask_cache(0)
    store = tmp_path / "view" / "Cell_040_dsr.ome.zarr"
    w.create_processed_store(
        store,
        n_t=3,
        n_c=2,
        shape_zyx=(8, 16, 12),
        channel_labels=["GFP 488", "mScarlet 561"],
    )
    buffers = tmp_path / "view" / "buffers"
    buffers.mkdir()
    viewer = ViewerModel()
    follower = live_view.LiveFollower(
        viewer, store, buffers_dir=buffers, prefetch_pool=_NoPrefetch()
    )
    assert follower._shown == []

    np.save(buffers / buffer_name(0, 0), np.full((8, 16, 12), 50, np.uint16))
    follower.poll()
    assert follower._shown == []  # one channel of two: not yet
    np.save(buffers / buffer_name(0, 1), np.full((8, 16, 12), 60, np.uint16))
    follower.poll()
    assert follower._shown == [0]
    assert int(_shown_data(follower.layers[1]).max()) == 60  # from the buffer
    lo, hi = follower.layers[1].contrast_limits
    assert lo <= 60 <= hi < 300  # set from the first real timepoint

    # Buffer trimmed after the store has it: read from the store instead.
    w.write_timepoint(store, 0, 1, np.full((8, 16, 12), 60, np.uint16))
    w.write_progress(store, n_t=3, n_c=2, done=[[0, 0], [0, 1]], state="running")
    (buffers / buffer_name(0, 1)).unlink()
    follower.poll()
    assert int(follower.sources[1].load(0, 0).max()) == 60


def test_watcher_prefers_the_ram_disk_store(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    view = tmp_path / "shm" / "s1" / "Cell_041_dsr.ome.zarr"
    view.mkdir(parents=True)
    (jobs / "live_latest.json").write_text(
        json.dumps(
            {
                "session_id": "s1",
                "store": str(tmp_path / "gpfs" / "Cell_041_dsr.ome.zarr"),
                "view_store": str(view),
                "buffers_dir": str(view.parent / "buffers"),
                "qc_dir": str(tmp_path / "gpfs_qc"),
            }
        )
    )
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    watcher = live_view.SessionWatcher(ViewerModel(), jobs=jobs)
    sid, store, extra = watcher._latest()
    assert (sid, store) == ("s1", view)
    assert extra["buffers_dir"] == str(view.parent / "buffers")
    shutil.rmtree(view)  # evicted from the RAM disk: fall back to GPFS
    assert watcher._latest()[1] == tmp_path / "gpfs" / "Cell_041_dsr.ome.zarr"


def _buffered_session(tmp_path, n_t=4, done_t=3):
    """A one-format session with `done_t` timepoints in RAM-disk buffers."""
    from opym.stream.live_zarr import buffer_name

    store = tmp_path / "view" / "Cell_042_dsr.ome.zarr"
    w.create_processed_store(
        store,
        n_t=n_t,
        n_c=2,
        shape_zyx=(8, 16, 12),
        channel_labels=["GFP 488", "mScarlet 561"],
    )
    buffers = tmp_path / "view" / "buffers"
    buffers.mkdir()
    for t in range(done_t):
        for c in range(2):
            vol = np.full((8, 16, 12), 100 * (t + 1) + 10 * c, np.uint16)
            np.save(buffers / buffer_name(t, c), vol)
    return store, buffers


def _count_reads(monkeypatch):
    """(t, c) of every volume napari reads, from a buffer or the store."""
    from opym.stream.live_zarr import parse_buffer_name

    reads = []
    real_map, real_read = live_view.map_buffer, live_view.read_volume

    def map_buffer(path):
        reads.append(parse_buffer_name(path.name))
        return real_map(path)

    def read(arr, t, c, **kwargs):
        reads.append((t, c))
        return real_read(arr, t, c, **kwargs)

    monkeypatch.setattr(live_view, "map_buffer", map_buffer)
    monkeypatch.setattr(live_view, "read_volume", read)
    return reads


def test_opening_a_session_reads_each_channel_once_at_the_newest_timepoint(
    tmp_path, monkeypatch
):
    """Found on Argus (2026-09-25): opening a finished session read its first
    timepoint seven times, three from the compressed store, before the
    window could first paint (25 s of black)."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel
    from napari.utils import resize_dask_cache

    resize_dask_cache(0)
    store, buffers = _buffered_session(tmp_path)
    reads = _count_reads(monkeypatch)
    viewer = ViewerModel(ndisplay=3)
    follower = live_view.LiveFollower(
        viewer, store, buffers_dir=buffers, prefetch_pool=_NoPrefetch()
    )
    assert viewer.dims.current_step[0] == 2
    # One mapping per channel serves its slice, its thumbnail level and its
    # contrast.
    assert sorted(reads) == [(2, 0), (2, 1)]
    assert [layer.name for layer in viewer.layers] == ["GFP 488", "mScarlet 561"]
    assert all(layer.visible for layer in follower.layers)
    lo, hi = follower.layers[1].contrast_limits
    assert lo <= 310 <= hi  # from the timepoint on screen

    reads.clear()
    np.save(buffers / "T0003_C0.npy", np.full((8, 16, 12), 7, np.uint16))
    np.save(buffers / "T0003_C1.npy", np.full((8, 16, 12), 8, np.uint16))
    follower.poll()
    assert viewer.dims.current_step[0] == 3
    assert sorted(reads) == [(3, 0), (3, 1)]  # not t=2 again, not the old layers


def test_the_red_channel_is_magenta_and_display_settings_survive_a_rebuild(
    tmp_path,
):
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    store, buffers = _buffered_session(tmp_path, done_t=1)
    viewer = ViewerModel(ndisplay=3)
    follower = live_view.LiveFollower(viewer, store, buffers_dir=buffers)
    assert [layer.colormap.name for layer in follower.layers] == ["green", "magenta"]

    follower.layers[0].visible = False
    follower.layers[1].colormap = "gray"
    follower.layers[1].rendering = "attenuated_mip"
    np.save(buffers / "T0001_C0.npy", np.full((8, 16, 12), 7, np.uint16))
    np.save(buffers / "T0001_C1.npy", np.full((8, 16, 12), 8, np.uint16))
    follower.poll()
    assert viewer.dims.current_step[0] == 1
    assert [layer.visible for layer in follower.layers] == [False, True]
    assert follower.layers[1].colormap.name == "gray"
    assert follower.layers[1].rendering == "attenuated_mip"
    assert [layer.name for layer in viewer.layers] == ["GFP 488", "mScarlet 561"]


def test_a_session_switch_does_not_reread_the_old_feed(tmp_path, monkeypatch):
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    first, first_buffers = _buffered_session(tmp_path / "a", done_t=2)
    jobs = tmp_path / "jobs"
    jobs.mkdir()

    def latest(store, buffers, sid):
        (jobs / live_view.LIVE_LATEST_NAME).write_text(
            json.dumps(
                {
                    "session_id": sid,
                    "store": str(tmp_path / "gpfs" / store.name),
                    "view_store": str(store),
                    "buffers_dir": str(buffers),
                }
            )
        )

    latest(first, first_buffers, "s1")
    viewer = ViewerModel(ndisplay=3)
    watcher = live_view.SessionWatcher(viewer, jobs=jobs, prefetch_pool=_NoPrefetch())
    watcher.poll()
    assert viewer.dims.current_step[0] == 1

    second, second_buffers = _buffered_session(tmp_path / "b", n_t=6, done_t=4)
    latest(second, second_buffers, "s2")
    reads = _count_reads(monkeypatch)
    watcher.poll()
    assert watcher.session_id == "s2"
    assert viewer.dims.current_step[0] == 3
    assert sorted(reads) == [(3, 0), (3, 1)]
    assert [layer.name for layer in viewer.layers] == ["GFP 488", "mScarlet 561"]
    assert all(layer.visible for layer in viewer.layers)


def test_read_volume_matches_zarr(tmp_path):
    import zarr

    arr = zarr.open(
        str(tmp_path / "a.zarr"),
        mode="w",
        shape=(2, 2, 150, 20, 10),
        chunks=(1, 1, 64, 8, 8),
        dtype=np.uint16,
    )
    arr[:] = np.random.default_rng(0).integers(0, 4000, arr.shape, dtype=np.uint16)
    np.testing.assert_array_equal(live_view.read_volume(arr, 1, 0), arr[1, 0])


def test_a_session_with_nothing_processed_yet_decodes_nothing(tmp_path, monkeypatch):
    """Its layers show up at once, from zeros: decoding the empty store's
    fill values took 1.7 s, just as its first timepoint was due."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    store, buffers = _buffered_session(tmp_path, done_t=0)
    decoded = []
    monkeypatch.setattr(
        live_view, "read_volume", lambda a, t, c, **kw: decoded.append(t)
    )
    viewer = ViewerModel(ndisplay=3)
    follower = live_view.LiveFollower(viewer, store, buffers_dir=buffers)
    assert [layer.name for layer in viewer.layers] == ["GFP 488", "mScarlet 561"]
    assert int(np.asarray(follower.layers[0].data[0]).max()) == 0
    assert decoded == []


class _InlinePool:
    """A prefetch pool that runs each job at once, on the calling thread."""

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future

        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


def test_each_channel_shows_its_newest_timepoint_as_soon_as_it_lands(tmp_path):
    """The 488 stack is acquired before the 561 stack: GFP's timepoint is
    shown the moment it's processed, mScarlet keeps its newest until its own
    lands, and each channel's arrival is traced."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    from opym.stream import trace

    store, buffers = _buffered_session(tmp_path, done_t=1)
    jobs = tmp_path / "jobs"
    viewer = ViewerModel(ndisplay=3)
    follower = live_view.LiveFollower(
        viewer, store, buffers_dir=buffers, jobs=jobs, prefetch_pool=_NoPrefetch()
    )
    gfp, msc = follower.layers
    np.save(buffers / "T0001_C0.npy", np.full((8, 16, 12), 7, np.uint16))
    follower.poll()
    assert viewer.dims.current_step[0] == 1
    assert int(_shown_data(gfp).max()) == 7  # GFP at t=1
    assert int(_shown_data(msc).max()) == 110  # mScarlet still at t=0
    assert "mScarlet 561 t=0" in viewer.text_overlay.text
    np.save(buffers / "T0001_C1.npy", np.full((8, 16, 12), 8, np.uint16))
    follower.poll()
    assert int(_shown_data(msc).max()) == 8
    assert follower._shown == [0, 1]
    shown = [
        (e["timepoints"], e["channels"])
        for e in trace.read(trace.VIEW_TRACE_NAME, jobs=jobs)
        if e["ev"] == "shown"
    ]
    assert shown[-2:] == [([1], [0]), ([1], [1])]


def test_scrubbing_shows_half_resolution_then_refines_when_it_rests(
    tmp_path, monkeypatch
):
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    monkeypatch.setattr(live_view, "REFINE_AFTER_S", 0.0)
    store, buffers = _buffered_session(tmp_path, done_t=3)
    viewer = ViewerModel(ndisplay=3)
    follower = live_view.LiveFollower(
        viewer, store, buffers_dir=buffers, prefetch_pool=_InlinePool()
    )
    layer = follower.layers[0]
    assert viewer.dims.current_step[0] == 2 and layer.data_level == 0
    viewer.dims.set_current_step(0, 0)  # the user moves the slider
    assert layer.data_level == 1
    assert _shown_data(layer).shape == follower.sources[0].shapes[1][1:]
    follower.poll()  # rested: the full resolution is read in the background
    follower.poll()  # ... and shown
    assert layer.data_level == 0
    assert _shown_data(layer).shape == (8, 16, 12)
    assert int(_shown_data(layer).max()) == 100


def test_the_follower_moving_the_slider_stays_at_full_resolution(tmp_path):
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    store, buffers = _buffered_session(tmp_path, done_t=1)
    viewer = ViewerModel(ndisplay=3)
    follower = live_view.LiveFollower(
        viewer, store, buffers_dir=buffers, prefetch_pool=_NoPrefetch()
    )
    for c in range(2):
        np.save(buffers / f"T0001_C{c}.npy", np.full((8, 16, 12), 5, np.uint16))
    follower.poll()
    assert viewer.dims.current_step[0] == 1
    assert [layer.data_level for layer in follower.layers] == [0, 0]


def test_prefetch_caches_every_level_so_moving_back_reads_nothing(
    tmp_path, monkeypatch
):
    """Everything the lane produced is in the viewer's RAM: moving back to
    an earlier timepoint reads neither a buffer nor the store."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    monkeypatch.setattr(live_view, "PREFETCH_IN_FLIGHT", 100)
    store, buffers = _buffered_session(tmp_path, done_t=3)
    for t in range(3):
        for c in range(2):
            w.write_timepoint(store, t, c, np.full((8, 16, 12), 10 * t + c, np.uint16))
    w.write_progress(
        store,
        n_t=4,
        n_c=2,
        done=[[t, c] for t in range(3) for c in range(2)],
        state="running",
    )
    viewer = ViewerModel(ndisplay=3)
    follower = live_view.LiveFollower(
        viewer, store, buffers_dir=buffers, prefetch_pool=_InlinePool()
    )
    follower.poll()
    cache = follower.cache
    assert cache.full.bytes > 0 and cache.small.bytes > 0
    reads = _count_reads(monkeypatch)
    viewer.dims.set_current_step(0, 0)  # half resolution while moving
    assert reads == []


def test_the_volume_cache_keeps_to_its_budget():
    cache = live_view.VolumeCache(budget_bytes=3 * 800)
    for t in range(5):
        cache.put(("s", 0, t, 0), np.zeros(400, np.uint16))  # 800 bytes each
    assert cache.bytes == 2400
    assert cache.get(("s", 0, 0, 0)) is None  # the oldest went first
    assert cache.get(("s", 0, 4, 0)) is not None
    cache.drop(lambda key: key[0] == "s")
    assert cache.bytes == 0
