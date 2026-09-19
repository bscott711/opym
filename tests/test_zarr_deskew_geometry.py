"""Geometry contract for the zarr-precropped deskew path.

These lock in the two things that silently produced wrong-sized DSR output
for every zarr dataset: the axis order the ticket declares, and the z step
it carries. Both were "working" in the sense that nothing errored -- the
pipeline reported `deskew: done` on output that was ~8x too short in Y --
so the only defence is an explicit assertion on the numbers.

Pure Python: no MATLAB, no GPU, no real data (unlike the gpu-marked
test_deskew_only_regression.py next door), so these run in a normal
`pytest tests/` pass.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import zarr

from opym.metadata import (
    parse_zarr_z_step_from_store,
    resolve_zarr_z_step,
)
from opym.petakit import submit_remote_deskew_job

# A real mirror store from the 20260902 macropinocytosis upload: one
# timepoint of a 4D (t, z, y, x) acquisition, so (z, y, x) on disk.
MIRROR_SHAPE_ZYX = (301, 490, 1458)
# What PetaKit5D must end up holding. Note the acquisition's axis names are
# the opposite way round from PetaKit5D's: the store's `y` (490) is the
# TILTED axis, which PetaKit5D calls x and shears, and the store's `x`
# (1458) lies along the coverslip and is the invariant one it calls y.
# Confirmed against real projections -- see the comment in petakit.py.
EXPECTED_YXZ = (1458, 490, 301)


# --------------------------------------------------------------------------
# Axis order
# --------------------------------------------------------------------------


def _axis_order_mapping(input_order: str, output_order: str = "yxz") -> list[int]:
    """Faithful port of PetaKit5D's utils/axis_order_mapping.m:

        order_mat(i) = strfind(outputAxisOrder, inputAxisOrder(i))

    Ported rather than mocked because its exact (inverse) semantics are the
    whole reason the correct string is counterintuitive.
    """
    return [output_order.index(c) + 1 for c in input_order]


def _matlab_permute(shape: tuple[int, ...], order: list[int]) -> tuple[int, ...]:
    """MATLAB `permute` semantics: size(B)(i) == size(A)(order(i))."""
    return tuple(shape[o - 1] for o in order)


def _apply_input_axis_order(shape: tuple[int, ...], input_order: str) -> tuple[int, ...]:
    """What XR_deskewRotateFrame.m does to a loaded frame (see its
    `input_order_mat` block): map, then permute unless already sorted.
    """
    order = _axis_order_mapping(input_order)
    if order == sorted(order):
        return shape
    return _matlab_permute(shape, order)


def test_zarr_input_axis_order_yields_yxz_frame():
    """'zxy' puts the long coverslip axis invariant and shears the short
    tilted one, which is what PetaKit5D's geometry assumes."""
    assert _apply_input_axis_order(MIRROR_SHAPE_ZYX, "zxy") == EXPECTED_YXZ


def test_xzy_shears_the_wrong_camera_axis():
    """'xzy' keeps the *tilted* 490 axis invariant and shears the coverslip
    axis -- the inverted-geometry result that shipped briefly: it produced a
    171 um-deep volume whose XY projection read as a side view."""
    assert _apply_input_axis_order(MIRROR_SHAPE_ZYX, "xzy") == (490, 1458, 301)
    assert _apply_input_axis_order(MIRROR_SHAPE_ZYX, "xzy") != EXPECTED_YXZ


def test_zxy_is_self_inverse_so_the_mapping_quirk_cannot_bite():
    """'zyx'/'xzy' are a 3-cycle pair that PetaKit5D's inverse-returning
    `axis_order_mapping` silently swaps. 'zxy' -> [3,2,1] is its own
    inverse, so the order we ship is immune to that trap."""
    order = _axis_order_mapping("zxy")
    assert order == [3, 2, 1]
    assert _matlab_permute(tuple(_matlab_permute(MIRROR_SHAPE_ZYX, order)), order) == MIRROR_SHAPE_ZYX


def test_yxz_is_the_original_bug():
    """'yxz' is a no-op permute, which is how the scan planes ended up
    being sheared as if they were image rows."""
    assert _apply_input_axis_order(MIRROR_SHAPE_ZYX, "yxz") == MIRROR_SHAPE_ZYX


def _ticket_params(tmp_path: Path, **kwargs) -> dict:
    data_dir = tmp_path / "zarr_mirror"
    data_dir.mkdir()
    ticket = submit_remote_deskew_job(
        input_target=data_dir,
        z_step_um=0.5,
        queue_dir=tmp_path / "queue",
        **kwargs,
    )
    return json.loads(Path(ticket).read_text())["parameters"]


def test_ticket_declares_zxy_for_zarr_input(tmp_path):
    params = _ticket_params(tmp_path, zarr_input=True)
    assert params["zarr_input"] is True
    assert params["input_axis_order"] == "zxy"
    assert params["output_axis_order"] == "yxz"
    # The ticket's own string must survive the MATLAB-side mapping.
    assert _apply_input_axis_order(MIRROR_SHAPE_ZYX, params["input_axis_order"]) == EXPECTED_YXZ


def test_ticket_keeps_yxz_for_tiff_input(tmp_path):
    """The TIFF path must not move: MATLAB loads a TIFF stack as
    (rows, cols, planes), which is already (y, x, z)."""
    params = _ticket_params(tmp_path, zarr_input=False)
    assert params["input_axis_order"] == "yxz"
    assert params["output_axis_order"] == "yxz"


# --------------------------------------------------------------------------
# Output size
# --------------------------------------------------------------------------


def dsr_output_size(ny: int, nx: int, nz: int, dz: float, xy_pixel_size: float = 0.136,
                    skew_angle: float = 60.0) -> tuple[int, int, int]:
    """PetaKit5D's DSR output size for the galvo (objectiveScan=false) case,
    from patches/XR_deskewRotateFrame.m's `outSize`. Returned as the TIFF
    (pages, rows, cols) the pipeline actually writes.
    """
    theta = math.radians(skew_angle)
    out_y = ny
    out_x = round((nx - 1) * math.cos(theta) + (nz - 1) * dz / xy_pixel_size)
    out_z = round((nx - 1) * math.sin(theta) - 4)
    return (out_z, out_y, out_x)


@pytest.mark.parametrize(
    "ny,nx,nz,dz,expected",
    [
        # Legacy TIFF, 20260505_py_FLM_2XFyve_mSca_mem_NG/cell -- measured
        # from the real DSR_nodecon output. Guards the untouched TIFF path.
        ((1152), 576, 175, 0.3, (494, 1152, 671)),
        # The zarr bug as it shipped: (301, 490, 1458) read as (y, x, z)
        # with the defaulted 0.3 um step. Measured from 20260902.../cell_003.
        ((301), 490, 1458, 0.3, (419, 301, 3458)),
    ],
)
def test_formula_reproduces_real_outputs(ny, nx, nz, dz, expected):
    """The formula is only trustworthy as a spec because it reproduces
    real on-disk output exactly -- including the broken case."""
    assert dsr_output_size(ny, nx, nz, dz) == expected


def test_fixed_zarr_path_output_size():
    """With ny=1458 (coverslip axis) and nx=490 (tilted axis): the 20260902
    and 20260917 acquisitions scan the same 30 um range at different
    sampling, so they must produce identical DSR dimensions."""
    # 20260902: 301 planes @ 0.1 um
    assert dsr_output_size(1458, 490, 301, 0.1) == (419, 1458, 465)
    # 20260917: 61 planes @ 0.5 um
    assert dsr_output_size(1458, 490, 61, 0.5) == (419, 1458, 465)


def test_output_size_matches_the_real_zxy_run():
    """Measured from the live 'zxy' run of 20260916 Cell_002 T000 -- the one
    whose projections were confirmed correct by eye (41 planes @ 0.5 um)."""
    assert dsr_output_size(1458, 490, 41, 0.5) == (419, 1458, 392)


def test_depth_is_never_the_largest_dimension():
    """The sanity check the wrong orientation failed. Correct ('zxy') gives
    57 um deep against 198 x 53 um laterally -- depth comparable to the
    short lateral axis, well under the long one. The inverted 'xzy' gave
    171 um of depth against 67 and 119 um: deeper than the volume is wide,
    which no coverslip-mounted sample can be.

    Not "depth is smallest": 57 > 53 here, and that is fine.
    """
    good = [d * 0.136 for d in dsr_output_size(1458, 490, 41, 0.5)]
    assert good[0] < max(good[1], good[2])

    bad = [d * 0.136 for d in dsr_output_size(490, 1458, 41, 0.5)]
    assert bad[0] > max(bad[1], bad[2])


# --------------------------------------------------------------------------
# Z step
# --------------------------------------------------------------------------


def _make_store(path: Path, z_values=None, scale_placeholder=True) -> Path:
    """Minimal stand-in for a pymmcore MDA store: a `z` coordinate array,
    plus the placeholder NGFF scale the real writer emits."""
    root = zarr.open(str(path), mode="w")
    if z_values is not None:
        arr = root.create_dataset("z", shape=(len(z_values),), dtype="<f8")
        arr[:] = np.asarray(z_values, dtype="<f8")
        arr.attrs["units"] = "um"
    if scale_placeholder:
        root.attrs["multiscales"] = [
            {
                "axes": [{"name": n} for n in "tzyx"],
                "datasets": [
                    {"path": "p0", "coordinateTransformations": [
                        {"type": "scale", "scale": [1, 1, 1, 1]}]}
                ],
            }
        ]
    return path


def test_z_step_read_from_coordinate_array(tmp_path):
    store = _make_store(tmp_path / "c.ome.zarr", np.linspace(-15, 15, 61))
    assert parse_zarr_z_step_from_store(store) == 0.5


def test_z_step_survives_float_noise(tmp_path):
    """301 planes over 30 um averages to 0.09999999999999964 in float64."""
    store = _make_store(tmp_path / "c.ome.zarr", np.linspace(-15, 15, 301))
    assert parse_zarr_z_step_from_store(store) == 0.1


def test_placeholder_ngff_scale_is_not_used(tmp_path):
    """The real stores all carry scale [1,1,1,1]; reading it would give a
    1 um step. The z array must win."""
    store = _make_store(tmp_path / "c.ome.zarr", np.linspace(-15, 15, 61))
    assert parse_zarr_z_step_from_store(store) != 1.0


def test_missing_z_array_returns_none(tmp_path):
    store = _make_store(tmp_path / "c.ome.zarr", None)
    assert parse_zarr_z_step_from_store(store) is None


def test_ragged_z_axis_returns_none(tmp_path):
    store = _make_store(tmp_path / "c.ome.zarr", [0.0, 0.5, 1.0, 9.0])
    assert parse_zarr_z_step_from_store(store) is None


def test_single_plane_returns_none(tmp_path):
    store = _make_store(tmp_path / "c.ome.zarr", [0.0])
    assert parse_zarr_z_step_from_store(store) is None


def test_resolve_prefers_store_over_sidecar(tmp_path):
    store = _make_store(tmp_path / "c.ome.zarr", np.linspace(-15, 15, 61))
    sidecar = tmp_path / "MDA_settings.yaml"
    sidecar.write_text("z_plan:\n  step: 0.3\n")
    step, source = resolve_zarr_z_step([store], sidecar)
    assert step == 0.5
    assert "c.ome.zarr/z" in source


def test_resolve_falls_back_to_sidecar(tmp_path):
    store = _make_store(tmp_path / "c.ome.zarr", None)
    sidecar = tmp_path / "MDA_settings.yaml"
    sidecar.write_text("z_plan:\n  step: 0.25\n")
    step, source = resolve_zarr_z_step([store], sidecar)
    assert step == 0.25
    assert source == "MDA_settings.yaml"


def test_resolve_default_is_flagged_as_unmeasured(tmp_path):
    """The sidecar genuinely does not exist for these acquisitions, so this
    is the path that silently shipped 0.3 um. It must be loud."""
    step, source = resolve_zarr_z_step([], tmp_path / "MDA_settings.yaml")
    assert step == 0.3
    assert "NOT measured" in source


def test_resolve_refuses_to_guess_when_asked(tmp_path):
    """An interrupted acquisition has no `z` array yet. Deskewing it at a
    made-up step just produces another silently wrong-sized volume that
    reports success, so the caller must be able to tell "no idea" apart
    from a real measurement."""
    store = _make_store(tmp_path / "c.ome.zarr", None)
    step, source = resolve_zarr_z_step(
        [store], tmp_path / "MDA_settings.yaml", default_z_step=None
    )
    assert step is None
    assert source == "unmeasured"


def test_strict_mode_still_returns_a_real_measurement(tmp_path):
    """Strictness must not suppress a step that *is* measurable."""
    store = _make_store(tmp_path / "c.ome.zarr", np.linspace(-15, 15, 61))
    step, _ = resolve_zarr_z_step(
        [store], tmp_path / "MDA_settings.yaml", default_z_step=None
    )
    assert step == 0.5
