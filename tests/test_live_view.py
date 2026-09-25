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

    _finish(store, 1, done)
    follower.poll()
    assert viewer.dims.current_step[0] == 1  # follow mode jumped to it
    assert "2/3 timepoints" in viewer.text_overlay.text
    # What napari reads for t=1 is the new data, not cached zeros.
    assert int(np.asarray(follower.layers[1].data[0][1]).max()) == 210

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
