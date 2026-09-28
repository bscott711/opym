from opym.discovery import discover_leaf_datasets


def test_superseded_runs_are_never_discovered(tmp_path):
    """opym.stream.receiver moves an earlier run whose name was reused into
    .superseded/; the backfill must not process it."""
    exp = tmp_path / "exp"
    (exp / "Cell_001_GFP_488.ome.zarr").mkdir(parents=True)
    old = exp / ".superseded" / "Cell_001-20260928-120000"
    (old / "Cell_001_GFP_488.ome.zarr").mkdir(parents=True)
    keys = [ds.dataset_key for ds in discover_leaf_datasets([tmp_path])]
    assert keys == [str(exp / "Cell_001")]
