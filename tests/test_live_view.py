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
