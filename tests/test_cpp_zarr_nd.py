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
