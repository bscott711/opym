"""Tests for opym.ome_zarr_writer, the one OME-Zarr layout shared by the live
lane and the backfill's viewer export."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from opym import ome_zarr_writer as w


def _float_downsample(vol):
    """The original float32 implementation; downsample2 must match it exactly."""
    z, y, x = (s - (s % 2) for s in vol.shape)
    v = vol[:z, :y, :x].astype(np.float32)
    return (
        v.reshape(z // 2, 2, y // 2, 2, x // 2, 2)
        .mean(axis=(1, 3, 5))
        .astype(vol.dtype)
    )


@pytest.mark.parametrize(
    "shape", [(8, 10, 12), (7, 9, 5), (3, 4, 4), (41, 64, 30), (2, 2, 2)]
)
def test_downsample_is_bit_identical_to_the_float_mean(shape):
    vol = np.random.default_rng(0).integers(0, 65535, shape, dtype=np.uint16)
    assert np.array_equal(w.downsample2(vol), _float_downsample(vol))


def test_downsample_keeps_the_float_path_for_other_dtypes():
    vol = np.random.default_rng(1).random((4, 6, 8)).astype(np.float32)
    assert np.allclose(w.downsample2(vol), _float_downsample(vol))


def test_store_layout_round_trip_and_progress(tmp_path):
    out = tmp_path / "Cell_002_dsr.ome.zarr"
    w.create_store(
        out,
        n_t=3,
        n_c=2,
        shape_zyx=(9, 20, 30),
        dtype=np.uint16,
        channel_labels=["GFP 488", "mScarlet 561"],
    )
    root = zarr.open_group(str(out), mode="r")
    assert [root[str(lvl)].shape for lvl in range(3)] == [
        (3, 2, 9, 20, 30),
        (3, 2, 4, 10, 15),
        (3, 2, 2, 5, 7),
    ]
    assert root["0"].chunks == (1, 1, 9, 20, 30)
    ms = root.attrs["multiscales"][0]
    assert [a["name"] for a in ms["axes"]] == list("tczyx")
    assert ms["datasets"][1]["coordinateTransformations"][0]["scale"] == [
        1.0,
        1.0,
        0.272,
        0.272,
        0.272,
    ]
    assert w.read_progress(out)["state"] == "running"
    assert w.complete_timepoints(w.read_progress(out)) == []

    vol = np.random.default_rng(2).integers(0, 4000, (9, 20, 30), dtype=np.uint16)
    w.write_timepoint(out, 1, 0, vol)
    w.write_timepoint(out, 1, 1, vol + 1)
    assert np.array_equal(root["0"][1, 0], vol)
    assert np.array_equal(root["1"][1, 1], w.downsample2(vol + 1))
    assert not root["0"][0].any() and not root["0"][2].any()

    w.write_progress(out, n_t=3, n_c=2, done=[[1, 0], [1, 1], [2, 0]], state="running")
    assert w.complete_timepoints(w.read_progress(out)) == [1]
