"""opym.timeline's public hooks for a consumer with its own array reader (no
private `_known`, no copy of the loop), on the golden vectors of
opym.testing.readers."""

from __future__ import annotations

import sys

import pytest

from opym import timeline as tl
from opym.testing import readers as g


def test_times_from_arrays_matches_the_golden_vectors():
    assert tl.times_from_arrays(t_ms=g.RAW_MS) == g.RAW_MS_SECONDS
    assert tl.times_from_arrays(t_epoch_s=g.RAW_EPOCH) == g.RAW_EPOCH_SECONDS
    # ms wins when both have times; an array with none falls through
    assert tl.times_from_arrays(g.RAW_MS, g.RAW_EPOCH) == g.RAW_MS_SECONDS
    unwritten = [float("nan")] * 3
    assert tl.times_from_arrays(unwritten, g.RAW_EPOCH) == g.RAW_EPOCH_SECONDS
    assert tl.times_from_arrays(t_epoch_s=g.RAW_EPOCH_UNSET) is None
    assert tl.times_from_arrays() is None


def test_known_time_is_public_and_the_old_name_still_works():
    assert tl.known_time(0.0, 0, tl.RAW_MS) and not tl.known_time(0.0, 1, tl.RAW_MS)
    assert not tl.known_time(0.0, 0, tl.RAW_EPOCH)
    assert tl._known is tl.known_time


def test_record_from_arrays_equals_what_sync_writes(tmp_path):
    rec = tl.record_from_arrays(
        [(g.RAW_MS, None)], source=f"raw store {g.RAW_STORE_NAME}"
    )
    assert rec == g.EXPECTED_OPYM_TIME_MS
    rec = tl.record_from_arrays(
        [(None, g.RAW_EPOCH)], source=f"raw store {g.RAW_STORE_NAME}"
    )
    assert rec == g.EXPECTED_OPYM_TIME_EPOCH
    assert tl.record_from_arrays([(None, None)], source="x") is None
    assert tl.record_from_arrays([], source="x") is None


def test_record_from_arrays_combines_channels_earliest_first():
    a = [1000.0, 9000.0, 0.0]
    b = [900.0, 9100.0, 17000.0]
    rec = tl.record_from_arrays([(a, None), (b, None)], source="s")
    assert rec["times_s"] == [0.0, 8.1, 16.1]
    assert rec["complete"] is True and rec["unit"] == "s"


def test_reader_hook_replaces_the_zarr_free_reader(tmp_path):
    seen = []

    def reader(array_dir):
        seen.append(array_dir.name)
        return {"t": g.RAW_MS}.get(array_dir.name)

    # nothing on disk: only the caller's reader supplies the arrays
    assert tl.read_raw_times(tmp_path, reader=reader) == g.RAW_MS_SECONDS
    assert seen == ["t"]  # ms had times, epoch is not even read
    assert tl.from_raw_stores([tmp_path], reader=reader) == g.RAW_MS_RELATIVE
    times, source = tl.resolve_times({}, [tmp_path], reader=reader)
    assert (times, source) == (g.RAW_MS_RELATIVE, "raw")


def test_reader_may_fall_back_on_the_refusal(tmp_path, monkeypatch):
    """The rig's compressed ``t``: opym's reader refuses, the host reads it itself."""
    monkeypatch.setitem(sys.modules, "zarr", None)  # as in a venv without zarr
    g.write_raw_store(tmp_path / "s.ome.zarr", ms=g.RAW_MS)
    (tmp_path / "s.ome.zarr" / "t" / ".zarray").write_text(
        '{"compressor": {"id": "blosc"}, "dtype": "<f8", "shape": [5], "chunks": [5]}'
    )
    store = tmp_path / "s.ome.zarr"
    with pytest.raises(tl.ArrayNeedsZarrError):
        tl.read_raw_times(store)
    assert tl.read_raw_times(store, strict=False) is None

    def host_reader(array_dir):
        try:
            return tl.read_1d(array_dir, use_zarr=False)
        except tl.ArrayNeedsZarrError:
            return g.RAW_MS if array_dir.name == "t" else None

    assert tl.read_raw_times(store, reader=host_reader) == g.RAW_MS_SECONDS


def test_a_refusing_reader_obeys_strict(tmp_path):
    def refuse(array_dir):
        raise tl.ArrayNeedsZarrError("no")

    with pytest.raises(tl.ArrayNeedsZarrError):
        tl.read_raw_times(tmp_path, reader=refuse)
    assert tl.read_raw_times(tmp_path, strict=False, reader=refuse) is None
