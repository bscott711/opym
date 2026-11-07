import json

import numpy as np
import pytest
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


def test_consolidate_rejects_mismatched_channel_shapes(tmp_path):
    # Every acquisition is expected to produce same-sized per-channel
    # volumes (fixed at capture time) -- if that's ever violated (upstream
    # bug, misconfigured ROI, etc.), the hardlink-no-copy trick this
    # function relies on would otherwise silently declare one shape for
    # the whole unified array while linking in chunk data whose actual
    # size doesn't match, corrupting the array for any later reader. This
    # must fail loudly here instead.
    base_name = "sample"

    vol_c0 = np.zeros((7, 5, 3), dtype=np.uint16)
    zarr.save_array(
        str(tmp_path / f"{base_name}_T0000_C0.zarr"), vol_c0, chunks=vol_c0.shape
    )

    vol_c1 = np.zeros((9, 11, 3), dtype=np.uint16)  # different Y/X than C0
    zarr.save_array(
        str(tmp_path / f"{base_name}_T0000_C1.zarr"), vol_c1, chunks=vol_c1.shape
    )

    with pytest.raises(ValueError, match="Shape/dtype mismatch"):
        consolidate_to_ome_zarr(
            decon_dir=tmp_path,
            base_name=base_name,
            z_step_um=0.3,
            xy_pixel_um=0.116,
            t_interval_s=1.0,
            channel_names=["C0", "C1"],
        )

    # Refuses to write anything, rather than a partially-correct store.
    assert not (tmp_path / f"{base_name}.ome.zarr").exists()


def test_the_log_tee_only_takes_the_consolidating_threads_output(tmp_path):
    """BF-08: consolidation runs on a worker thread while the supervisor's own
    thread keeps printing to the same sys.stdout: those lines are not
    consolidation log lines."""
    import io
    import threading

    from opym.consolidate import _TeeWriter

    out, log = io.StringIO(), io.StringIO()
    tee = _TeeWriter(out, log)
    tee.write("mine\n")
    t = threading.Thread(target=lambda: tee.write("supervisor\n"))
    t.start()
    t.join()
    assert out.getvalue() == "mine\nsupervisor\n"
    assert log.getvalue() == "mine\n"


def _frame_zarrs(root, base, ts, cs=(0,), shape=(4, 3, 2)):
    for t in ts:
        for c in cs:
            vol = np.full(shape, t * 10 + c + 1, dtype=np.uint16)
            zarr.save_array(str(root / f"{base}_T{t:04d}_C{c}.zarr"), vol, chunks=shape)


def _consolidate(root, base, **kw):
    return consolidate_to_ome_zarr(
        decon_dir=root, base_name=base, z_step_um=0.3, xy_pixel_um=0.116,
        t_interval_s=1.0, channel_names=["C0"], **kw,
    )


def test_a_missing_timepoint_leaves_a_blank_frame_not_a_shifted_series(tmp_path):
    """BF-09: time indices came from the entries present, so T=1 missing made T=2
    land on index 1, one frame early, and everything after it too."""
    _frame_zarrs(tmp_path, "s", ts=[0, 2, 3])
    assert _consolidate(tmp_path, "s")
    arr = np.asarray(zarr.open(str(tmp_path / "s.ome.zarr" / "0"), mode="r"))
    assert arr.shape[:2] == (4, 1)
    assert [int(arr[t, 0].max()) for t in range(4)] == [1, 0, 21, 31]


def test_pending_consolidation_keeps_its_sidecar_while_zarrs_are_missing(tmp_path):
    """BF-09: the sidecar went away after the first pass, so a gap was never repaired."""
    from opym.consolidate import run_pending_consolidations

    jobs = tmp_path / "jobs"
    (jobs / "completed").mkdir(parents=True)
    data = tmp_path / "data"
    data.mkdir()
    (jobs / "completed" / "t.json").write_text(json.dumps({"dataDir": str(data)}))
    expected = [f"s_T{t:04d}_C0.zarr" for t in range(3)]
    (data / ".opym_consolidate.json").write_text(json.dumps({
        "base_name": "s", "z_step_um": 0.3, "xy_pixel_um": 0.116,
        "t_interval_s": 1.0, "channel_names": ["C0"], "expected_zarrs": expected,
    }))
    _frame_zarrs(data, "s", ts=[0, 2])
    assert run_pending_consolidations(jobs) == 1
    assert (data / ".opym_consolidate.json").exists()

    _frame_zarrs(data, "s", ts=[1])  # the late frame arrives
    assert run_pending_consolidations(jobs) == 1
    assert not (data / ".opym_consolidate.json").exists()
    arr = np.asarray(zarr.open(str(data / "s.ome.zarr" / "0"), mode="r"))
    assert [int(arr[t, 0].max()) for t in range(3)] == [1, 11, 21]
