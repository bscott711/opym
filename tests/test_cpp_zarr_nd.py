"""The patched cpp-zarr mex (opym/patches/cpp-zarr) against zarr-python:
reading one timepoint out of an N-D store (`parallelReadZarr(...,
'leadingIndex', ...)`) and writing a 3-D block into one
(`opymWriteZarrBlock`). The live pipeline's raw store is (T, Z, Y, X) and its
processed OME-Zarr (T, C, Z, Y, X); PetaKit5D's own zarr code is 3-D only.

Needs MATLAB (the `matlab_engine` fixture, `-m gpu`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from numcodecs import Blosc

pytestmark = pytest.mark.gpu

LZ4 = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
MEX_DIR = Path(__file__).resolve().parents[1] / "src/opym/patches/cpp-zarr/linux"


@pytest.fixture(scope="module")
def eng(matlab_engine):
    matlab_engine.addpath(str(MEX_DIR), "-begin", nargout=0)
    return matlab_engine


@pytest.mark.parametrize("compressor", [None, LZ4], ids=["uncompressed", "blosc"])
def test_read_one_timepoint_of_a_raw_store(eng, tmp_path, compressor):
    import matlab

    raw = np.random.default_rng(0).integers(0, 60000, (3, 5, 7, 9), dtype=np.uint16)
    store = tmp_path / "p0"
    arr = zarr.open(
        str(store),
        mode="w",
        shape=raw.shape,
        chunks=(1, 1, 7, 9),
        dtype="uint16",
        compressor=compressor,
        dimension_separator="/",
    )
    arr[:] = raw
    got = eng.parallelReadZarr(str(store), "leadingIndex", matlab.double([3]))
    np.testing.assert_array_equal(np.asarray(got), raw[2])


def test_write_a_block_into_a_5d_store(eng, tmp_path):
    import matlab

    store = tmp_path / "0"
    # Ragged edges on every trailing axis (70 = 64 + 6, 300 = 256 + 44, ...).
    shape = (2, 2, 70, 300, 270)
    p = zarr.open(
        str(store),
        mode="w",
        shape=shape,
        chunks=(1, 1, 64, 256, 256),
        dtype="uint16",
        compressor=LZ4,
        dimension_separator="/",
        fill_value=0,
    )
    p[0, 0] = 7
    z, y, x = np.meshgrid(*(np.arange(n) for n in shape[2:]), indexing="ij")
    block = ((z * 7 + y * 3 + x) % 60000).astype(np.uint16)
    block[:64, :256, :256] = 0  # an all-zero chunk is skipped, reads as 0
    zarray_before = (store / ".zarray").read_text()

    eng.opymWriteZarrBlock(
        str(store), matlab.uint16(block), matlab.double([2, 2]), nargout=0
    )

    p = zarr.open(str(store), mode="r")
    np.testing.assert_array_equal(p[1, 1], block)
    assert (p[0, 0] == 7).all() and not p[0, 1].any() and not p[1, 0].any()
    assert (store / ".zarray").read_text() == zarray_before
    assert not (store / "1" / "1" / "0" / "0" / "0").exists()
    assert json.loads(zarray_before)["shape"] == list(shape)


def test_write_rejects_a_wrong_size_or_index(eng, tmp_path):
    import matlab

    store = tmp_path / "0"
    zarr.open(
        str(store),
        mode="w",
        shape=(1, 1, 4, 5, 6),
        chunks=(1, 1, 4, 5, 6),
        dtype="uint16",
        compressor=LZ4,
        dimension_separator="/",
    )
    block = np.zeros((4, 5, 6), np.uint16)
    with pytest.raises(Exception, match="size\\(data\\)"):
        eng.opymWriteZarrBlock(
            str(store), matlab.uint16(block[:3]), matlab.double([1, 1]), nargout=0
        )
    with pytest.raises(Exception, match="leadingIndex:range"):
        eng.opymWriteZarrBlock(
            str(store), matlab.uint16(block), matlab.double([2, 1]), nargout=0
        )


@pytest.mark.parametrize("compressor", [None, LZ4], ids=["uncompressed", "blosc"])
def test_read_oriented_for_decon(eng, tmp_path, compressor):
    """'orientForDecon' returns flip(permute(zyx, [3 2 1]), 1) directly:
    a(i, j, z) = v(z, j, X + 1 - i), each C-order row reversed."""
    import matlab

    raw = np.random.default_rng(1).integers(0, 60000, (2, 5, 7, 9), dtype=np.uint16)
    store = tmp_path / "p0"
    arr = zarr.open(
        str(store),
        mode="w",
        shape=raw.shape,
        chunks=(1, 1, 7, 9),
        dtype="uint16",
        compressor=compressor,
        dimension_separator="/",
    )
    arr[:] = raw
    got = np.asarray(
        eng.parallelReadZarr(
            str(store), "leadingIndex", matlab.double([2]), "orientForDecon", True
        )
    )
    assert got.shape == (9, 7, 5)
    np.testing.assert_array_equal(got, np.flip(raw[1].transpose(2, 1, 0), axis=0))


def _processed(tmp_path, shape):
    from opym.ome_zarr_writer import create_processed_store

    return create_processed_store(
        tmp_path / "S_dsr.ome.zarr", n_t=2, n_c=2, shape_zyx=shape
    )


def test_live_outputs_match_every_reference(eng, tmp_path):
    """opymWriteLiveOutputs from a (Y, X, Z) DSR volume: the C-order
    (Z, Y, X) view buffer, level 0, each 2x level (opym's downsample2) and
    the Z-MIP -- at ragged, odd sizes, twice into one directory (the second
    call writes into the buffer readied after the first)."""
    import matlab

    from opym.ome_zarr_writer import downsample2

    shape = (70, 301, 259)  # (Z, Y, X): odd trailing edges on every axis
    arrs = _processed(tmp_path, shape)
    rng = np.random.default_rng(2)
    (tmp_path / "view").mkdir()
    for t, c in ((0, 1), (1, 0)):
        vol = rng.integers(0, 4000, shape, dtype=np.uint16)
        vol[:64, :256, :256] = 0  # an all-zero chunk: skipped, reads as 0
        npy = tmp_path / "view" / f"T{t}_C{c}.npy"
        secs = eng.opymWriteLiveOutputs(
            matlab.uint16(np.ascontiguousarray(vol.transpose(1, 2, 0))),
            str(npy),
            [str(p) for p in arrs.levels],
            str(arrs.mip),
            matlab.double([t + 1, c + 1]),
        )
        assert np.asarray(secs).size == 2
        view = np.load(npy, mmap_mode="r")
        assert view.flags.c_contiguous
        np.testing.assert_array_equal(view, vol)
        out = zarr.open_group(str(arrs.mip.parents[1]), mode="r")
        ref = vol
        for lvl in range(len(arrs.levels)):
            np.testing.assert_array_equal(out[f"0/{lvl}"][t, c], ref)
            ref = downsample2(ref)
        np.testing.assert_array_equal(out["1/0"][t, c, 0], vol.max(axis=0))
    # No partial buffers left behind; only the hidden next buffer, readied.
    left = [f.name for f in (tmp_path / "view").iterdir() if f.name.endswith(".tmp")]
    assert len(left) == 1 and left[0].startswith(".opym_prep_")


def test_live_outputs_without_a_view_buffer(eng, tmp_path):
    import matlab

    shape = (8, 12, 10)
    arrs = _processed(tmp_path, shape)
    vol = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    eng.opymWriteLiveOutputs(
        matlab.uint16(np.ascontiguousarray(vol.transpose(1, 2, 0))),
        "",
        [str(p) for p in arrs.levels],
        str(arrs.mip),
        matlab.double([2, 2]),
        nargout=1,
    )
    out = zarr.open_group(str(arrs.mip.parents[1]), mode="r")
    np.testing.assert_array_equal(out["0/0"][1, 1], vol)
    np.testing.assert_array_equal(out["1/0"][1, 1, 0], vol.max(axis=0))
