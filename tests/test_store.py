"""opym.store: the layout-3 names and the zarr-free readers, on the golden
store (opym.testing.store), plus the old import paths on ome_zarr_writer."""

from __future__ import annotations

import json

import numpy as np
import pytest
import zarr

from opym import decon_config, store
from opym import ome_zarr_writer as w
from opym.testing import store as golden


@pytest.fixture(scope="module")
def real_store(tmp_path_factory):
    return golden.write_golden_store(root=tmp_path_factory.mktemp("golden"))


def test_names():
    assert store.LAYOUT_KEY == "bioformats2raw.layout"
    assert (store.DSR_SERIES, store.MIP_SERIES) == ("0", "1")
    assert store.PROGRESS_NAME == ".opym_live.json"
    assert store.PYRAMID_LEVELS == 3
    assert store.CHUNKS_ZYX == (64, 256, 256)
    assert store.DSR_VOXEL_UM == decon_config.XY_PIXEL_SIZE_UM == 0.136


def test_old_import_paths_are_the_same_objects():
    for name in (
        "DSR_VOXEL_UM",
        "PYRAMID_LEVELS",
        "PROGRESS_NAME",
        "BF2RAW_LAYOUT",
        "DSR_SERIES",
        "MIP_SERIES",
        "CHUNKS_ZYX",
        "image_group",
        "read_progress",
        "complete_timepoints",
        "is_processed_store",
    ):
        assert getattr(w, name) is getattr(store, name)
    assert w.CHANNEL_COLORS == ("00FF00", "FF00FF", "00B3FF", "FFC400")


def test_golden_attrs_are_what_create_processed_store_writes(real_store):
    for rel, attrs in golden.golden_layout3_attrs().items():
        got = store.read_attrs(real_store / rel)
        # JSON round trip: zarr stores lists where the helpers build tuples
        assert json.loads(json.dumps(attrs)) == got, rel


def test_readers_agree_on_real_store_and_json_only_store(real_store, tmp_path):
    light = golden.write_golden_attrs(tmp_path / "light.ome.zarr")
    for path in (real_store, light):
        assert store.is_processed_store(path)
        assert store.voxel_size_um(path) == golden.GOLDEN_VOXEL_UM
        assert store.voxel_size_um(path, level=1) == golden.GOLDEN_VOXEL_UM * 2
        assert store.voxel_size_um(path, store.MIP_SERIES) == golden.GOLDEN_VOXEL_UM
        assert store.time_interval_s(path) == golden.GOLDEN_TIME_INTERVAL_S
        assert store.channel_labels(path) == list(golden.GOLDEN_LABELS)
        assert store.channel_colors(path) == ["00FF00", "FF00FF"]
        assert store.channel_labels(path, store.MIP_SERIES) == list(
            golden.GOLDEN_LABELS
        )
        prog = store.read_progress(path)
        assert prog["n_t"] == 3 and prog["n_c"] == 2
        assert store.complete_timepoints(prog) == [0, 1]


def test_legacy_red_channel_is_remapped_on_read_only(tmp_path):
    path = golden.write_golden_attrs(tmp_path / "old.ome.zarr", legacy_red=True)
    on_disk = json.loads((path / "0" / ".zattrs").read_text())
    assert on_disk["omero"]["channels"][1]["color"] == "FF3D3D"
    assert store.channel_colors(path) == ["00FF00", "FF00FF"]


def test_level_path_and_image_group(real_store):
    assert store.level_path(real_store, "0", 2) == real_store / "0" / "2"
    assert store.level_path(real_store) == real_store / "0" / "0"
    assert store.level_path(real_store, store.MIP_SERIES) == real_store / "1" / "0"
    arr = zarr.open_array(str(store.level_path(real_store, "0", 0)), mode="r")
    assert arr.shape == (3, 2, 8, 16, 16)
    np.testing.assert_array_equal(arr[1, 1], golden.golden_volume(1, 1))
    assert store.image_group(real_store).attrs["multiscales"]
    assert store.image_group(real_store, store.MIP_SERIES)["0"].shape[2] == 1


def test_legacy_store_has_only_the_dsr_image(tmp_path):
    out = tmp_path / "legacy.ome.zarr"
    w.create_store(
        out,
        n_t=2,
        n_c=2,
        shape_zyx=(4, 8, 8),
        dtype=np.uint16,
        channel_labels=["a", "b"],
        time_interval_s=3.0,
    )
    assert not store.is_processed_store(out)
    assert store.voxel_size_um(out) == 0.136
    assert store.time_interval_s(out) == 3.0
    assert store.channel_labels(out) == ["a", "b"]
    assert store.channel_colors(out) == ["00FF00", "FF00FF"]
    assert store.voxel_size_um(out, store.MIP_SERIES) is None
    assert store.image_group(out).attrs["multiscales"]
    with pytest.raises(KeyError):
        store.image_group(out, store.MIP_SERIES)


def test_readers_on_missing_or_odd_stores(tmp_path):
    assert not store.is_processed_store(None)
    assert not store.is_processed_store(tmp_path / "nope")
    assert store.read_attrs(tmp_path / "nope") == {}
    assert store.read_progress(tmp_path) is None
    assert store.voxel_size_um(tmp_path) is None
    assert store.time_interval_s(tmp_path) is None
    assert store.channel_labels(tmp_path) == []
    assert store.channel_colors(tmp_path) == []
    (tmp_path / ".zattrs").write_text("[]")
    assert store.read_attrs(tmp_path) == {}


def test_read_attrs_reads_zarr_v3_with_the_ome_namespace_merged(tmp_path):
    (tmp_path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"ome": {"multiscales": [{"name": "x"}]}, "other": 1},
            }
        )
    )
    attrs = store.read_attrs(tmp_path)
    assert attrs["multiscales"] == [{"name": "x"}] and attrs["other"] == 1


def test_complete_timepoints():
    assert store.complete_timepoints(None) == []
    prog = {"n_c": 2, "done": [[2, 0], [0, 1], [0, 0], [1, 0], [2, 1]]}
    assert store.complete_timepoints(prog) == [0, 2]


def test_mip_stacks_reads_completed_timepoints_only(real_store):
    got = w.mip_stacks(real_store)
    assert set(got) == {0, 1} and got[0].shape == (2, 16, 16)
