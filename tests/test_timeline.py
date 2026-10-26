"""opym.timeline: acquisition times from raw stores, carried into processed ones."""

import json

import numpy as np
import pytest
import zarr

from opym import timeline as tl


def raw_store(path, *, ms=None, epoch=None, n=5):
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update({"multiscales": [{"axes": [{"name": "t"}], "datasets": []}]})
    if ms is not None:
        a = root.create_dataset("t", shape=(n,), dtype="f8", fill_value=0.0)
        a[: len(ms)] = ms
        a.attrs["units"] = "ms"
    if epoch is not None:
        a = root.create_dataset("t_epoch_s", shape=(n,), dtype="f8", fill_value=np.nan)
        a[: len(epoch)] = epoch
    return path


def processed_store(path, labels=None, layout="legacy", n_t=5):
    """A minimal processed store with a placeholder 1.0 s time scale."""
    root = zarr.open_group(str(path), mode="w")
    ms = [
        {
            "axes": [{"name": "t", "type": "time", "unit": "second"}, {"name": "c"}],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0]}
                    ],
                }
            ],
        }
    ]
    attrs = {"omero": {"channels": [{"label": x} for x in labels]}} if labels else {}
    if layout == "legacy":
        attrs["multiscales"] = ms
        root.attrs.update(attrs)
    else:  # bioformats2raw: the series group carries the multiscales
        root.attrs.update(attrs)
        sub = root.create_group("0")
        sub.attrs.update({"multiscales": ms})
    return path


def test_ms_array_becomes_seconds_since_the_first_volume(tmp_path):
    p = raw_store(
        tmp_path / "a.ome.zarr", ms=[740.8436, 9012.079, 16550.8723, 24020.3426], n=4
    )
    raw = tl.read_raw_times(p)
    assert raw == pytest.approx([0.7408, 9.012, 16.5509, 24.0203], abs=1e-3)
    rel = tl.from_raw_stores([p])
    assert rel[0] == 0 and rel[1] == pytest.approx(8.2712, abs=1e-3)
    assert tl.interval_s(rel) == pytest.approx(7.5, abs=0.8)


def test_unwritten_timepoints_are_unknown_not_zero(tmp_path):
    p = raw_store(
        tmp_path / "live.ome.zarr", ms=[500.0, 8000.0], n=5
    )  # only 2 of 5 volumes received
    assert tl.read_raw_times(p) == [0.5, 8.0, None, None, None]
    e = raw_store(tmp_path / "mirror.ome.zarr", epoch=[1.79e9, 1.79e9 + 7.5], n=4)
    assert tl.read_raw_times(e) == [1.79e9, 1.79e9 + 7.5, None, None]
    assert tl.relative(tl.read_raw_times(e)) == [0.0, 7.5, None, None]


def test_an_unset_frame_timestamp_is_not_1970(tmp_path):
    p = raw_store(tmp_path / "zeros.ome.zarr", epoch=[0.0, 0.0, 0.0], n=3)
    assert tl.read_raw_times(p) is None and tl.from_raw_stores([p]) is None


def test_no_times_in_the_store_is_none(tmp_path):
    p = raw_store(
        tmp_path / "old.ome.zarr"
    )  # like the Sep-28 streamed stores: no t array
    assert tl.read_raw_times(p) is None and tl.from_raw_stores([p]) is None
    assert tl.read_raw_times(tmp_path / "missing.ome.zarr") is None


def test_channels_combine_to_the_volume_start(tmp_path):
    a = raw_store(tmp_path / "a.ome.zarr", ms=[100.0, 8100.0, 16100.0], n=3)
    b = raw_store(
        tmp_path / "b.ome.zarr", ms=[3600.0, 11600.0, 0.0], n=3
    )  # channel 2 comes later in each volume
    rel = tl.from_raw_stores([a, b])
    assert rel == [0.0, 8.0, 16.0]
    assert (
        tl.interval_s([0.0, None, 15.0, 22.5]) == 7.5
    )  # a gap does not skew the interval


def test_sync_writes_times_and_sets_the_time_axis_scale(tmp_path):
    raw = raw_store(
        tmp_path / "Cell_X_GFP_488.ome.zarr",
        ms=[740.0, 8240.0, 15740.0, 23240.0, 30740.0],
    )
    for layout in ("legacy", "series"):
        proc = processed_store(tmp_path / f"{layout}.ome.zarr", layout=layout)
        assert tl.sync_processed(proc, [raw]) is True
        rec = tl.read_times(proc)
        assert (
            rec["times_s"] == [0.0, 7.5, 15.0, 22.5, 30.0]
            and rec["interval_s"] == 7.5
            and rec["complete"] is True
        )
        assert "Cell_X_GFP_488" in rec["source"]
        where = (
            json.loads((proc / ".zattrs").read_text())
            if layout == "legacy"
            else json.loads((proc / "0" / ".zattrs").read_text())
        )
        assert where["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
            "scale"
        ] == [7.5, 1.0]
        assert where["multiscales"][0]["axes"][0]["unit"] == "second"
        before = (proc / ".zattrs").read_text()
        tl.sync_processed(proc, [raw])  # again: nothing changes
        assert (proc / ".zattrs").read_text() == before


def test_partial_times_are_recorded_as_incomplete_and_filled_in_later(tmp_path):
    raw = raw_store(tmp_path / "r_GFP_488.ome.zarr", ms=[0.5, 8000.5], n=4)
    proc = processed_store(tmp_path / "p.ome.zarr")
    tl.sync_processed(proc, [raw])
    rec = tl.read_times(proc)
    assert rec["times_s"] == [0.0, 8.0, None, None] and rec["complete"] is False
    z = zarr.open_group(str(raw), mode="a")
    z["t"][2:4] = [16000.5, 24000.5]
    tl.sync_processed(proc, [raw])
    assert (
        tl.read_times(proc)["times_s"] == [0.0, 8.0, 16.0, 24.0]
        and tl.read_times(proc)["complete"] is True
    )


def test_nothing_known_writes_nothing(tmp_path):
    proc = processed_store(tmp_path / "p.ome.zarr")
    before = (proc / ".zattrs").read_text() if (proc / ".zattrs").exists() else ""
    assert tl.write_times(proc, [None, None], source="x") is False
    assert tl.sync_processed(proc, []) is False
    assert (
        (proc / ".zattrs").read_text() if (proc / ".zattrs").exists() else ""
    ) == before
    assert (
        tl.write_times(tmp_path / "not-a-store", [0.0, 1.0], source="x") is False
    )  # not a zarr group


def test_raw_stores_are_matched_to_their_own_dataset(tmp_path):
    sess = tmp_path / "20261023_Rab5"
    (sess / "Gapvd1_Unstim" / "viewer").mkdir(parents=True)
    (sess / "Gapvd1_Unstim_001" / "viewer").mkdir(parents=True)
    for name in (
        "Gapvd1_Unstim_mNG_488nm",
        "Gapvd1_Unstim_mScarlet_561nm",
        "Gapvd1_Unstim_001_mNG_488nm",
        "Gapvd1_Unstim_001_mScarlet_561nm",
    ):
        (sess / f"{name}.ome.zarr").mkdir()
    proc = sess / "Gapvd1_Unstim" / "viewer" / "Gapvd1_Unstim_dsr.ome.zarr"
    proc.mkdir()
    names = [p.name for p in tl.raw_stores_for(proc)]
    assert names == [
        "Gapvd1_Unstim_mNG_488nm.ome.zarr",
        "Gapvd1_Unstim_mScarlet_561nm.ome.zarr",
    ], names
    proc1 = sess / "Gapvd1_Unstim_001" / "viewer" / "Gapvd1_Unstim_001_dsr.ome.zarr"
    proc1.mkdir()
    assert [p.name for p in tl.raw_stores_for(proc1)] == [
        "Gapvd1_Unstim_001_mNG_488nm.ome.zarr",
        "Gapvd1_Unstim_001_mScarlet_561nm.ome.zarr",
    ]


def test_channel_labels_that_name_the_raw_stores_are_used_exactly(tmp_path):
    sess = tmp_path / "s"
    sess.mkdir()
    for name in ("cell_9_GFP_488", "cell_9_alt_GFP_488", "cell_9_RFP_561"):
        (sess / f"{name}.ome.zarr").mkdir()
    (sess / "cell_9" / "viewer").mkdir(parents=True)
    proc = processed_store(
        sess / "cell_9" / "viewer" / "cell_9_dsr.ome.zarr",
        labels=["cell_9_GFP_488", "cell_9_RFP_561"],
    )
    assert [p.name for p in tl.raw_stores_for(proc)] == [
        "cell_9_GFP_488.ome.zarr",
        "cell_9_RFP_561.ome.zarr",
    ]


def test_backfill_outputs_mirror_finds_the_original_raw_stores(tmp_path):
    orig = tmp_path / "mmfs1" / "scratch" / "lab" / "sess"
    (orig / "Cell_3").mkdir(parents=True)
    (orig / "Cell_3_mNG_488nm.ome.zarr").mkdir()
    mirror = (
        tmp_path
        / "mmfs2"
        / "opym_backfill"
        / "outputs"
        / str(orig).lstrip("/")
        / "Cell_3"
        / "viewer"
    )
    mirror.mkdir(parents=True)
    proc = mirror / "Cell_3_dsr.ome.zarr"
    proc.mkdir()
    assert tl.original_dir(mirror.parent) == orig / "Cell_3"
    assert [p.name for p in tl.raw_stores_for(proc)] == ["Cell_3_mNG_488nm.ome.zarr"]


# -- the live lanes keep the processed store in step with the raw stores ----------


def _raw_with_times(tmp_path):
    raw = tmp_path / "Cell_9_GFP_488.ome.zarr"
    return raw_store(raw, ms=[740.0, 8240.0, 15740.0], n=3)


def test_the_zarr_live_lane_records_times_at_each_progress_write(tmp_path):
    from types import SimpleNamespace

    from opym.ome_zarr_writer import create_processed_store
    from opym.stream.live_zarr import ZarrLiveLane

    raw = _raw_with_times(tmp_path)
    store = tmp_path / "view" / "Cell_9_dsr.ome.zarr"
    create_processed_store(store, n_t=3, n_c=1, shape_zyx=(4, 8, 8))
    # the placeholder is what every earlier dataset carried
    assert tl.read_times(store) is None
    lane = ZarrLiveLane.__new__(ZarrLiveLane)  # only _write_progress is exercised
    session = SimpleNamespace(
        num_timepoints=3, n_channels=1, end=None, raw_arrays=[raw / "p0"]
    )
    lane._write_progress(session, store, {(0, 0)}, "running")
    rec = tl.read_times(store)
    assert rec["times_s"] == [0.0, 7.5, 15.0] and rec["interval_s"] == 7.5
    scales = [
        json.loads((p / ".zattrs").read_text())["multiscales"][0]["datasets"][0][
            "coordinateTransformations"
        ][0]["scale"][0]
        for p in [store / "0"]
        if (p / ".zattrs").exists()
    ] or [
        json.loads((store / ".zattrs").read_text())["multiscales"][0]["datasets"][0][
            "coordinateTransformations"
        ][0]["scale"][0]
    ]
    assert scales == [7.5]


def test_the_legacy_live_lane_records_times_and_the_nominal_interval(tmp_path):
    from types import SimpleNamespace

    from opym import ome_zarr_writer
    from opym.stream.live import LiveLane

    raw = _raw_with_times(tmp_path)
    store = tmp_path / "view" / "Cell_9_dsr.ome.zarr"
    ome_zarr_writer.create_store(
        store, n_t=3, n_c=1, shape_zyx=(4, 8, 8), dtype=np.uint16, time_interval_s=5.0
    )
    nominal = json.loads((store / ".zattrs").read_text())["multiscales"][0]
    assert nominal["datasets"][0]["coordinateTransformations"][0]["scale"][0] == 5.0
    lane = LiveLane.__new__(LiveLane)
    session = SimpleNamespace(
        raw_stores=[raw], zarr_created=True, superseded=False, zarr_path=store
    )
    lane._sync_times(session)
    assert tl.read_times(store)["times_s"] == [0.0, 7.5, 15.0]
    # the measured interval replaces the nominal one on the time axis
    measured = json.loads((store / ".zattrs").read_text())["multiscales"][0]
    assert measured["datasets"][0]["coordinateTransformations"][0]["scale"][0] == 7.5


def test_a_lane_without_raw_times_changes_nothing(tmp_path):
    from types import SimpleNamespace

    from opym import ome_zarr_writer
    from opym.stream.live import LiveLane

    old = raw_store(tmp_path / "old_GFP_488.ome.zarr")  # no t array
    store = tmp_path / "view" / "X_dsr.ome.zarr"
    ome_zarr_writer.create_store(
        store, n_t=3, n_c=1, shape_zyx=(4, 8, 8), dtype=np.uint16, time_interval_s=5.0
    )
    before = (store / ".zattrs").read_text()
    lane = LiveLane.__new__(LiveLane)
    session = SimpleNamespace(
        raw_stores=[old], zarr_created=True, superseded=False, zarr_path=store
    )
    lane._sync_times(session)
    assert (store / ".zattrs").read_text() == before and tl.read_times(store) is None


# -- opym-add-times ----------------------------------------------------------------


def _dataset(tmp_path, name="Cell_5", with_times=True):
    """<session>/<name>/viewer/<name>_dsr.ome.zarr beside <session>/<name>_GFP_488."""
    sess = tmp_path / "sess"
    sess.mkdir(exist_ok=True)
    kw = {"ms": [900.0, 8400.0, 15900.0, 23400.0]} if with_times else {}
    raw_store(sess / f"{name}_GFP_488.ome.zarr", n=4, **kw)
    (sess / name / "viewer").mkdir(parents=True)
    proc = processed_store(sess / name / "viewer" / f"{name}_dsr.ome.zarr")
    return sess, proc


def test_add_times_dry_run_changes_nothing_and_apply_writes_with_a_backup(
    tmp_path, capsys
):
    from opym import add_times

    sess, proc = _dataset(tmp_path)
    before = (proc / ".zattrs").read_text()
    assert add_times.main([str(sess)]) == 0
    out = capsys.readouterr().out
    assert "4/4 times, interval 7.5 s" in out and "dry run" in out
    assert (proc / ".zattrs").read_text() == before
    assert add_times.main(["--apply", str(sess)]) == 0
    assert tl.read_times(proc)["times_s"] == [0.0, 7.5, 15.0, 22.5]
    assert (proc / ".zattrs.before-times").read_text() == before  # the original, kept
    capsys.readouterr()
    add_times.main(["--apply", str(proc)])  # again, by the store itself
    assert "already has its 4 times" in capsys.readouterr().out
    assert (proc / ".zattrs.before-times").read_text() == before  # never overwritten


def test_add_times_reports_datasets_with_no_raw_times(tmp_path, capsys):
    from opym import add_times

    sess, proc = _dataset(tmp_path, with_times=False)
    before = (proc / ".zattrs").read_text()
    assert add_times.main(["--apply", str(sess / "Cell_5")]) == 0  # by the dataset
    assert "no real times to add" in capsys.readouterr().out
    assert (proc / ".zattrs").read_text() == before
    assert not (proc / ".zattrs.before-times").exists()  # nothing changed, no backup


def test_add_times_with_nothing_found_is_an_error(tmp_path, capsys):
    from opym import add_times

    assert add_times.main([str(tmp_path)]) == 2
