from opym.discovery import discover_leaf_datasets


def _store(path):
    path.mkdir(parents=True)
    (path / ".zgroup").write_text("{}")


def test_stores_in_a_root_do_not_hide_acquisitions_below_it(tmp_path):
    """2026-10-14: 22 stores landed directly in the microscopy root, which made
    the root itself a zarr leaf, and discovery stopped there -- nothing under any
    dated acquisition folder was found for 9 days."""
    root = tmp_path / "microscopy"
    _store(root / "Cell_001_GFP_488.ome.zarr")
    _store(root / "Cell_001_mScarlet_561.ome.zarr")
    exp = root / "20261023_LCK-NG_JB"
    _store(exp / "Pak2_Stim_mNG_488nm.ome.zarr")
    # What the live lane writes beside the raw store: none of it is raw data.
    _store(exp / "Pak2_Stim" / "viewer" / "Pak2_Stim_dsr.ome.zarr")
    (exp / "Pak2_Stim" / "qc" / "rawproj").mkdir(parents=True)
    _store(root / "Cell_001" / "zarr_mirror" / "Cell_001_C0_T000.ome.zarr")

    keys = sorted(ds.dataset_key for ds in discover_leaf_datasets([root]))

    assert keys == [str(exp / "Pak2_Stim"), str(root / "Cell_001")]


def test_a_sample_named_dir_that_holds_raw_stores_is_not_walked(tmp_path):
    """The per-sample dir beside a store (`<prefix>/`) is the dataset's own
    output namespace, so whatever is inside it is never a second dataset."""
    exp = tmp_path / "exp"
    _store(exp / "Cell_002_GFP_488.ome.zarr")
    _store(exp / "Cell_002" / "Cell_002_GFP_488.ome.zarr")

    keys = [ds.dataset_key for ds in discover_leaf_datasets([tmp_path])]

    assert keys == [str(exp / "Cell_002")]


def test_channel_labels_with_a_unit_group_under_the_sample_name(tmp_path):
    """Since 2026-10-22 the channel presets read "mNG_488nm"; the dataset must
    still be `<sample>`, the folder the live lane writes its output into."""
    exp = tmp_path / "exp"
    _store(exp / "Cell_001_mNG_488nm.ome.zarr")
    _store(exp / "Cell_001_mScarlet_561nm.ome.zarr")
    _store(exp / "Cell_002_GFP_488.ome.zarr")

    found = {ds.dataset_key: len(ds.channel_zarr_paths) for ds in discover_leaf_datasets([tmp_path])}

    assert found == {str(exp / "Cell_001"): 2, str(exp / "Cell_002"): 1}
