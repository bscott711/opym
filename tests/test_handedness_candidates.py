"""The read-only lister of deskew-only datasets with the old mirrored
handedness (ROADMAP 3a). Registries here are built in `tmp_path`; the
production registry is never opened."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

from opym.registry import StatusRegistry

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "handedness_rerun_candidates.py"
)
_spec = importlib.util.spec_from_file_location("handedness_rerun_candidates", SCRIPT)
lister = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lister)


def _add(reg, tmp_path, name, *, output, finished, status="done", psf=None):
    leaf = tmp_path / "data" / name
    reg.register_dataset(
        str(leaf),
        root=str(tmp_path / "data"),
        leaf_dir=str(leaf),
        master_file=str(leaf / "m.zarr"),
    )
    if psf:
        reg.set_decon_psf(str(leaf), psf)
    reg.finish_stage(str(leaf), "deskew", status=status, output_path=str(output))
    with reg._cursor() as cur:  # pin the finish time the test is about
        cur.execute(
            "UPDATE stage_status SET finished_at=? "
            "WHERE dataset_key=? AND stage='deskew'",
            (finished, str(leaf)),
        )
    return leaf


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "registry.sqlite3"
    reg = StatusRegistry(path)
    old = lambda leaf: leaf / "zarr_mirror" / "DSR_nodecon"  # noqa: E731
    d = tmp_path / "data"
    a = _add(
        reg, tmp_path, "a", output=old(d / "a"), finished="2026-09-24T10:00:00+00:00"
    )
    _add(
        reg,
        tmp_path,
        "b",
        output=old(d / "b"),
        finished="2026-09-28T23:00:00+00:00",
        psf="/psf.tif",
    )
    _add(reg, tmp_path, "c", output=old(d / "c"), finished="2026-10-02T10:00:00+00:00")
    # not candidates: new deskew-only location, decon output, TIFF-path output
    _add(
        reg,
        tmp_path,
        "new",
        output=d / "new" / "decon_stage" / "DSR_nodecon",
        finished="2026-11-06T10:00:00+00:00",
    )
    decon_out = d / "decon" / "decon_stage" / "Decon" / "DSR_decon"
    _add(
        reg,
        tmp_path,
        "decon",
        output=decon_out,
        finished="2026-09-25T10:00:00+00:00",
        psf="/psf.tif",
    )
    _add(
        reg,
        tmp_path,
        "tiff",
        output=d / "tiff" / "processed" / "DSR_nodecon",
        finished="2026-09-25T10:00:00+00:00",
    )
    # a legacy path on a stage other than deskew is not a candidate
    reg.finish_stage(
        str(d / "tiff"), "mip_encode", status="done", output_path=str(old(d / "tiff"))
    )
    reg.close()
    return path, a


def _keys(rows):
    return [Path(r["dataset_key"]).name for r in rows]


def test_lists_only_legacy_outputs_in_finish_order(registry):
    path, _ = registry
    from opym.registry import RegistryReader

    with RegistryReader(path) as reader:
        rows = lister.candidates(reader)
    assert _keys(rows) == ["a", "b", "c"]


def test_date_window_is_inclusive(registry):
    path, _ = registry
    from opym.registry import RegistryReader

    with RegistryReader(path) as reader:
        rows = lister.candidates(reader, "2026-09-24", "2026-09-28")
        assert _keys(rows) == ["a", "b"]
        assert _keys(lister.candidates(reader, since="2026-09-29")) == ["c"]
        assert _keys(lister.candidates(reader, until="2026-09-23")) == []


def test_stat_flags_and_notes(registry):
    path, a = registry
    (a / "zarr_mirror" / "DSR_nodecon").mkdir(parents=True)
    (a / "viewer").mkdir()
    from opym.registry import RegistryReader

    with RegistryReader(path) as reader:
        by_key = {Path(r["dataset_key"]).name: r for r in lister.candidates(reader)}
    ra, rb = by_key["a"], by_key["b"]
    assert (ra["old_output_exists"], ra["viewer_exists"], ra["mip_movies_exists"]) == (
        True,
        True,
        False,
    )
    assert "provenance mismatch" in ra["note"]
    assert rb["old_output_exists"] is False
    assert rb["note"] == "old output path is missing on disk"

    (a / "decon_stage" / "Decon" / "DSR_decon").mkdir(parents=True)
    with RegistryReader(path) as reader:
        ra = {Path(r["dataset_key"]).name: r for r in lister.candidates(reader)}["a"]
    assert ra["new_decon_exists"] is True
    assert "already on disk" in ra["note"]


def test_main_is_read_only_and_csv_parses(registry, capsys):
    path, _ = registry
    tmp = path.parent
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    assert lister.main(["--registry", str(path), "--format", "csv"]) == 0
    out = capsys.readouterr().out
    rows = list(csv.DictReader(io.StringIO(out)))
    assert _keys(rows) == ["a", "b", "c"]
    assert tuple(rows[0]) == lister.COLUMNS
    assert lister.main(["--registry", str(path), "--format", "json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 3
    assert lister.main(["--registry", str(path)]) == 0  # table
    # The database file is untouched. (SQLite may leave its `-shm`/`-wal`
    # sidecars next to a WAL database that was merely read; they hold no data.)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    extra = {p.name for p in tmp.iterdir()} - {path.name}
    assert extra <= {path.name + "-shm", path.name + "-wal"}


@pytest.mark.parametrize(
    "flag", ["--apply", "--reset", "--delete", "--move", "--force", "--apply=1"]
)
def test_write_flags_are_refused(registry, flag):
    path, _ = registry
    before = path.read_bytes()
    with pytest.raises(SystemExit) as exc:
        lister.main(["--registry", str(path), flag])
    assert "read-only" in str(exc.value)
    assert path.read_bytes() == before


def test_unknown_flag_is_an_error(registry):
    path, _ = registry
    with pytest.raises(SystemExit) as exc:
        lister.main(["--registry", str(path), "--bogus"])
    assert exc.value.code == 2


def test_missing_registry_is_reported_not_created(tmp_path, capsys):
    path = tmp_path / "nope.sqlite3"
    assert lister.main(["--registry", str(path)]) == 1
    assert "no registry" in capsys.readouterr().err
    assert not path.exists()
