"""Tests for opym.ome_zarr_writer, the one OME-Zarr layout shared by the live
lane and the backfill's viewer export."""

from __future__ import annotations

import json

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


# --- processed store (bioformats2raw layout 3) ------------------------------


def _processed(tmp_path, **kw):
    out = tmp_path / "Cell_009" / "Cell_009_dsr.ome.zarr"
    arrays = w.create_processed_store(
        out,
        n_t=3,
        n_c=2,
        shape_zyx=(10, 20, 30),
        channel_labels=["GFP 488", "mScarlet 561"],
        time_interval_s=9.5,
        **kw,
    )
    return out, arrays


def test_processed_store_layout(tmp_path):
    import zarr

    out, arrays = _processed(tmp_path)
    root = zarr.open_group(str(out), mode="r")
    assert root.attrs["bioformats2raw.layout"] == 3
    assert root["OME"].attrs["series"] == ["0", "1"]
    assert [str(p.relative_to(out)) for p in arrays.levels] == ["0/0", "0/1", "0/2"]
    assert str(arrays.mip.relative_to(out)) == "1/0"
    assert root["0/0"].shape == (3, 2, 10, 20, 30)
    assert root["0/2"].shape == (3, 2, 2, 5, 7)
    assert root["1/0"].shape == (3, 2, 1, 20, 30)
    for arr in (root["0/0"], root["1/0"]):
        # What opymWriteZarrBlock requires: chunk 1 on t and c, "/" keys, C
        # order, compressed, and zeros for the chunks it skips.
        meta = json.loads((out / arr.path / ".zarray").read_text())
        assert meta["chunks"][:2] == [1, 1] and meta["dimension_separator"] == "/"
        assert meta["order"] == "C" and meta["compressor"]["cname"] == "lz4"
        assert meta["fill_value"] == 0 and meta["dtype"] == "<u2"
    ms = root["0"].attrs["multiscales"][0]
    assert ms["datasets"][1]["coordinateTransformations"][0]["scale"] == [
        9.5,
        1.0,
        w.DSR_VOXEL_UM * 2,
        w.DSR_VOXEL_UM * 2,
        w.DSR_VOXEL_UM * 2,
    ]
    assert [ch["label"] for ch in root["0"].attrs["omero"]["channels"]] == [
        "GFP 488",
        "mScarlet 561",
    ]
    assert w.read_progress(out)["done"] == []


def test_processed_store_ome_xml_is_valid_and_describes_both_images(tmp_path):
    from ome_types import from_xml

    out, _ = _processed(tmp_path)
    ome = from_xml((out / "OME" / "METADATA.ome.xml").read_text(), validate=True)
    dsr, mip = ome.images
    assert dsr.name == "Cell_009" and mip.name == "Cell_009 Z-MIP"
    px = dsr.pixels
    assert (px.size_x, px.size_y, px.size_z, px.size_c, px.size_t) == (30, 20, 10, 2, 3)
    assert px.physical_size_x == px.physical_size_z == w.DSR_VOXEL_UM
    assert px.time_increment == 9.5
    assert [c.name for c in px.channels] == ["GFP 488", "mScarlet 561"]
    assert [c.excitation_wavelength for c in px.channels] == [488, 561]
    assert mip.pixels.size_z == 1


def test_image_group_reads_both_layouts(tmp_path):
    out, _ = _processed(tmp_path)
    legacy = tmp_path / "legacy_dsr.ome.zarr"
    w.create_store(legacy, n_t=1, n_c=1, shape_zyx=(4, 8, 8), dtype=np.uint16)
    assert "multiscales" in w.image_group(out).attrs
    assert w.image_group(out, "1").attrs["multiscales"][0]["name"].endswith("Z-MIP")
    assert "multiscales" in w.image_group(legacy).attrs
    with pytest.raises(KeyError):
        w.image_group(legacy, "1")


def test_python_writer_fills_a_processed_store(tmp_path):
    import zarr

    out, _ = _processed(tmp_path)
    vol = np.arange(10 * 20 * 30, dtype=np.uint16).reshape(10, 20, 30)
    w.write_timepoint(out, 1, 1, vol)
    root = zarr.open_group(str(out), mode="r")
    np.testing.assert_array_equal(root["0/0"][1, 1], vol)
    np.testing.assert_array_equal(root["0/1"][1, 1], w.downsample2(vol))


def test_ome_zarr_py_opens_the_store_and_its_mip_series(tmp_path):
    """napari-ome-zarr (ome-zarr-py) doesn't walk bioformats2raw series; the
    root's own multiscales (pointing into series "0") make the store open as
    the DSR image, and the MIP series opens on its own."""
    reader_mod = pytest.importorskip("ome_zarr.reader")
    from ome_zarr.io import parse_url

    out, _ = _processed(tmp_path)

    def shapes(path):
        nodes = reader_mod.Reader(parse_url(str(path)))()
        return [n.data[0].shape for n in nodes if n.data]

    assert shapes(out) == [(3, 2, 10, 20, 30)]
    assert shapes(out / "1") == [(3, 2, 1, 20, 30)]
