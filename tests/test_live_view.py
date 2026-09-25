"""Tests for opym.live_view (`naparym-live`)."""

from __future__ import annotations

import json

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

    first_level0 = follower.layers[1].data[0]
    user_limits = (5.0, 900.0)
    follower.layers[0].contrast_limits = user_limits

    _finish(store, 1, done)
    follower.poll()
    assert viewer.dims.current_step[0] == 1  # follow mode jumped to it
    assert "2/3 timepoints" in viewer.text_overlay.text
    # What napari reads for t=1 is the new data, not cached zeros.
    assert int(np.asarray(follower.layers[1].data[0][1]).max()) == 210
    # Real bug (2026-09-25): napari's own chunk-loading caches a fetched
    # chunk by the SOURCE ARRAY'S IDENTITY, separate from dask's own cache
    # (resize_dask_cache(0), already off above) -- reusing the same dask
    # array and merely calling layer.refresh() left the canvas showing
    # zeros a growing store had already written past, fixed only by
    # closing and reopening naparym-live (fresh array objects). A poll
    # that saw new data must hand each layer a genuinely new array, not
    # the one from construction.
    assert follower.layers[1].data[0] is not first_level0
    # A data swap must not silently wipe out the current contrast, whether
    # it's the auto-set value above or something the user dialed in by hand.
    assert follower.layers[0].contrast_limits == list(user_limits)

    follower.follow = False
    _finish(store, 2, done, state="complete")
    follower.poll()
    assert viewer.dims.current_step[0] == 1
    assert viewer.text_overlay.text.endswith("complete")


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
