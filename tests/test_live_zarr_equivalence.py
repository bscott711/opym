"""The one-format live job (run_live_zarr.m) against the TIFF live job it
replaces (run_live_frames.m), on real data: the processed OME-Zarr's full
resolution level and Z-MIP must be bit-identical to the DSR and MIP TIFFs,
and its pyramid levels bit-identical to ome_zarr_writer.downsample2.

Both jobs get their tickets from the same writers the live lane uses and
run through MATLAB exactly as the GPU server runs them (jsondecode of the
ticket, then the job function). Needs MATLAB + a GPU (`-m gpu`) and the real
Cell_001_001 raw stores and PSF on GPFS; skipped otherwise."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
from numcodecs import Blosc

from opym.decon_config import deskew_decon_kwargs
from opym.ome_zarr_writer import downsample2, level_shapes
from opym.petakit import (
    live_zarr_parameters,
    submit_live_frames_job,
    submit_live_zarr_job,
)
from opym.utils import dsr_shape_zyx, write_decon_staged_tiff

pytestmark = pytest.mark.gpu

SRC = Path(
    "/mmfs1/scratch/jacks.local/microscopy/"
    "20260925-SVO-memNG-mScar2xFYVE-FLM-Macropinocytosis"
)
STORES = [
    SRC / f"Cell_001_001_{ch}.ome.zarr" / "p0" for ch in ("GFP_488", "mScarlet_561")
]
PSF = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260910_averaged_psf.tif")
MEX_DIR = Path(__file__).resolve().parents[1] / "src/opym/patches/cpp-zarr/linux"
DZ = 0.5
T = 1  # one timepoint, both channels
C = 2


@pytest.fixture(scope="module")
def eng(matlab_engine):
    if not (all(s.exists() for s in STORES) and PSF.exists()):
        pytest.skip("real Cell_001_001 data / PSF not reachable")
    matlab_engine.addpath(str(MEX_DIR), "-begin", nargout=0)
    return matlab_engine


def _run(eng, ticket: Path, fn: str) -> None:
    eng.eval(
        f"job = jsondecode(fileread('{ticket}')); {fn}(job.parameters, 16);", nargout=0
    )


def test_zarr_path_is_bit_identical_to_the_tiff_path(eng, tmp_path):
    kw = deskew_decon_kwargs(PSF)

    frames_dir = tmp_path / "tiff" / "decon_stage"
    frames_dir.mkdir(parents=True)
    frames = []
    for t in range(T):
        for c in range(C):
            f = frames_dir / f"Eq_C{c}_T{t:03d}.tif"
            write_decon_staged_tiff(
                np.asarray(zarr.open(str(STORES[c]), mode="r")[t]), f
            )
            frames.append(f)
    tiff_ticket = submit_live_frames_job(
        frames,
        tmp_path / "tiff" / "Decon",
        frames_dir / "Eq_C0_T000.tif",
        [PSF] * C,
        [f"_C{c}_T" for c in range(C)],
        DZ,
        ticket_name="Eq",
        queue_dir=tmp_path / "q",
        **kw,
    )

    shape = dsr_shape_zyx(zarr.open(str(STORES[0]), mode="r").shape[1:], DZ)
    lz4 = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    store = tmp_path / "zarr" / "Eq_dsr.ome.zarr"
    root = zarr.open_group(str(store), mode="w")
    levels = []
    for lvl, shp in enumerate(level_shapes(T, C, shape, 3)):
        root.create_dataset(
            f"0/{lvl}",
            shape=shp,
            chunks=(1, 1, *(min(n, m) for n, m in zip((64, 256, 256), shp[2:]))),
            dtype="uint16",
            compressor=lz4,
            dimension_separator="/",
            fill_value=0,
        )
        levels.append(store / "0" / str(lvl))
    root.create_dataset(
        "1/0",
        shape=(T, C, 1, *shape[1:]),
        chunks=(1, 1, 1, *shape[1:]),
        dtype="uint16",
        compressor=lz4,
        dimension_separator="/",
        fill_value=0,
    )

    _run(eng, tiff_ticket, "run_live_frames")

    # As in production: the session's warm-up first (PSF into the shared
    # cache, a throwaway decon + DSR of this shape, the first view buffer
    # readied), then the real tickets through that same cache.
    (tmp_path / "view").mkdir()
    cache = tmp_path / "psf_cache" / "k"
    raw_shape = zarr.open(str(STORES[0]), mode="r").shape[1:]
    warm = live_zarr_parameters(
        STORES[0],
        0,
        0,
        mask_store=STORES[0],
        levels=levels,
        mip=store / "1" / "0",
        psf_path=PSF,
        decon_dir=tmp_path / "zarr" / "Decon",
        z_step_um=DZ,
        psf_cache_dir=cache,
        **kw,
    )
    warm.update(
        warmup=True,
        raw_shape_zyx=list(raw_shape),
        dsr_shape_zyx=list(shape),
        view_dir=str(tmp_path / "view"),
    )
    spec = tmp_path / "warmup.json"
    spec.write_text(json.dumps({"parameters": warm}))
    _run(eng, spec, "run_live_zarr")
    assert list((cache / "psfgen").glob("*back_projector*.tif"))
    assert not zarr.open_group(str(store), mode="r")["0/0"][:].any()  # nothing written

    for t in range(T):
        for c in range(C):
            ticket = submit_live_zarr_job(
                STORES[c],
                t,
                c,
                mask_store=STORES[0],
                levels=levels,
                mip=store / "1" / "0",
                psf_path=PSF,
                decon_dir=tmp_path / "zarr" / "Decon",
                z_step_um=DZ,
                ticket_name=f"Eq_T{t}_C{c}",
                queue_dir=tmp_path / "q",
                view_npy=tmp_path / "view" / f"T{t}_C{c}.npy",
                psf_cache_dir=cache,
                **kw,
            )
            _run(eng, ticket, "run_live_zarr")

    dsr_dir = tmp_path / "tiff" / "Decon" / "DSR_decon"
    out = zarr.open_group(str(store), mode="r")
    for t in range(T):
        for c in range(C):
            name = f"Eq_C{c}_T{t:03d}"
            level0 = out["0/0"][t, c]
            np.testing.assert_array_equal(
                level0, tifffile.imread(dsr_dir / f"{name}.tif")
            )
            np.testing.assert_array_equal(
                out["1/0"][t, c, 0],
                tifffile.imread(dsr_dir / "MIPs" / f"{name}_MIP_z.tif"),
            )
            # The live viewer's copy: the same volume, memory-mapped as is.
            view = np.load(tmp_path / "view" / f"T{t}_C{c}.npy", mmap_mode="r")
            assert view.flags.c_contiguous and view.dtype == np.uint16
            np.testing.assert_array_equal(view, level0)
            level1 = downsample2(level0)
            np.testing.assert_array_equal(out["0/1"][t, c], level1)
            np.testing.assert_array_equal(out["0/2"][t, c], downsample2(level1))
