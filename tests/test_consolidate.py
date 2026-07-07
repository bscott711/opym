import json

import numpy as np
import zarr

from opym.consolidate import consolidate_to_ome_zarr


def test_consolidate_reorders_axes_with_zero_data_copy(tmp_path):
    # Per-frame zarrs are staged (X, Y, Z) by orient_zyx_for_dsr, "C"-order —
    # use a distinctly asymmetric shape so a wrong axis order would produce
    # an obviously-wrong shape/values rather than silently passing.
    x, y, z = 7, 5, 3
    base_name = "sample"
    volumes: dict[tuple[int, int], np.ndarray] = {}

    for t in range(2):
        for c in range(2):
            # Distinctive per-true-z-slice pattern so a reordering bug (not
            # just a shape bug) would be caught: value encodes (t, c, z).
            vol = np.zeros((x, y, z), dtype=np.uint16)
            for zi in range(z):
                vol[:, :, zi] = t * 100 + c * 10 + zi
            volumes[(t, c)] = vol
            zarr.save_array(
                str(tmp_path / f"{base_name}_T{t:04d}_C{c}.zarr"), vol, chunks=vol.shape
            )

    ok = consolidate_to_ome_zarr(
        decon_dir=tmp_path,
        base_name=base_name,
        z_step_um=0.3,
        xy_pixel_um=0.116,
        t_interval_s=1.0,
        channel_names=["C0", "C1"],
    )
    assert ok

    out_path = tmp_path / f"{base_name}.ome.zarr"

    zattrs = json.loads((out_path / ".zattrs").read_text())
    axes_names = [a["name"] for a in zattrs["multiscales"][0]["axes"]]
    assert axes_names == ["t", "c", "z", "y", "x"]

    zarray = json.loads((out_path / "0" / ".zarray").read_text())
    assert zarray["shape"] == [2, 2, z, y, x]
    assert zarray["chunks"] == [1, 1, z, y, x]
    assert zarray["order"] == "F"

    consolidated = np.asarray(zarr.open(str(out_path / "0"), mode="r"))
    assert consolidated.shape == (2, 2, z, y, x)
    for (t, c), vol in volumes.items():
        expected = np.transpose(vol, (2, 1, 0))  # (X,Y,Z) -> (Z,Y,X)
        np.testing.assert_array_equal(consolidated[t, c], expected)
