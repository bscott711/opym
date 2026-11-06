"""The read-side contracts: registry reader, zarr-free raw times, z-step env rule,
backfill run records. Golden vectors live in `opym.testing.readers`."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from opym import metadata, registry, timeline, utils
from opym.stream import backfill_runs
from opym.testing import readers as golden

# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


@pytest.fixture
def reg_path(tmp_path):
    p = tmp_path / "registry.sqlite3"
    expected = golden.build_registry(p)
    return p, expected


def test_overall_status_covers_every_status_and_matches_the_golden_registry(reg_path):
    path, expected = reg_path
    with registry.RegistryReader(path) as r:
        rows = r.dashboard_rows()
    assert {row["dataset_key"]: row["overall_status"] for row in rows} == expected
    # every derivable overall status is represented, none outside the contract
    assert set(expected.values()) | {"acquiring"} == set(registry.OVERALL_STATUSES)
    # every stage key is present (None where there is no row), in STAGES order
    assert all(tuple(row["stages"]) == registry.STAGES for row in rows)
    # newest discovered first
    discovered = [row["discovered_at"] for row in rows]
    assert discovered == sorted(discovered, reverse=True)


def test_acquiring_comes_from_the_live_map(reg_path):
    path, _ = reg_path
    key = "/data/raw/day1/done"
    live = {key: {"done": 3, "declared": 10}}
    with registry.RegistryReader(path) as r:
        row = r.find_dashboard_row(key, live)
        assert row["overall_status"] == "acquiring" and row["live"] == live[key]
        assert r.find_dashboard_row(key)["overall_status"] == "done"
        assert r.find_dashboard_row("/nope") is None


def test_columns_are_the_contract(reg_path):
    path, _ = reg_path
    with registry.RegistryReader(path) as r:
        ds = r.datasets()[0]
        st = r.stage_rows()[0]
    assert tuple(ds) == registry.DATASET_COLUMNS
    assert tuple(st) == registry.STAGE_COLUMNS
    # the writer's schema agrees with the constants the reader publishes
    with registry.connect_ro(path) as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(datasets)")]
        scols = [row[1] for row in conn.execute("PRAGMA table_info(stage_status)")]
    assert (
        tuple(cols) == registry.DATASET_COLUMNS
        and tuple(scols) == registry.STAGE_COLUMNS
    )
    with registry.RegistryReader(path) as r:
        statuses = {s["status"] for s in r.stage_rows()}
    assert statuses <= set(registry.STAGE_STATUSES)
    assert {s for s in ds if s == "signal_flag"} == {"signal_flag"}


def test_signal_and_triage_columns_are_read(reg_path):
    path, _ = reg_path
    with registry.RegistryReader(path) as r:
        row = r.get_dataset(golden.AWKWARD_DATASET_KEY)
        flags = {d["signal_flag"] for d in r.datasets()}
    assert row["signal_flag"] == "partial"
    assert (row["expected_timepoints"], row["actual_timepoints"]) == (100, 40)
    assert flags <= set(registry.SIGNAL_FLAGS) | {None}


def test_keyed_queries_read_only_those_datasets(reg_path):
    path, _ = reg_path
    keys = ["/data/raw/day1/done", "/data/raw/day1/failed", "/data/raw/day1/done"]
    with registry.RegistryReader(path) as r:
        assert sorted(d["dataset_key"] for d in r.datasets(keys)) == sorted(set(keys))
        assert {s["dataset_key"] for s in r.stage_rows(keys)} == set(keys)
        assert [x["dataset_key"] for x in r.dashboard_rows(keys=[])] == []
        many = [f"/k/{i}" for i in range(1200)] + keys  # more than one SQL chunk
        assert len(r.datasets(many)) == 2
        assert r.leaf_dir("/data/raw/day1/done") == "/data/raw/day1/done"
        assert r.leaf_dir("/nope") is None


def test_stage_lookups(reg_path):
    path, _ = reg_path
    with registry.RegistryReader(path) as r:
        assert (
            r.done_stage_output("/data/raw/day1/done", "deskew")
            == "/data/raw/day1/done/out/deskew"
        )
        assert (
            r.done_stage_output("/data/raw/day1/running", "crop_zarr") is None
        )  # not done
        assert r.done_stage_output("/data/raw/day1/failed", "deskew") is None  # no row
        assert (
            r.get_stage("/data/raw/day1/failed", "roi_detect")["error_message"]
            == "corrupt TIFF"
        )
        assert r.get_stage("/data/raw/day1/failed", "deskew") is None


def test_downstream_queries_match_what_core_ran(reg_path):
    path, _ = reg_path
    with registry.RegistryReader(path) as r:
        deskewed = {d["dataset_key"]: d["dsr_dir"] for d in r.deskewed_datasets()}
        mip = {d["dataset_key"]: d for d in r.mip_done_datasets()}
    assert deskewed == {
        "/data/raw/day1/done": "/data/raw/day1/done/out/deskew",
        "/data/raw/day1/queued": "/data/raw/day1/queued/out/deskew",
        # 'stale': deskew was restarted (running), 'blocked': deskew failed -> neither
    }
    assert set(mip) == {
        "/data/raw/day1/done",
        "/data/raw/day1/stale",
    }  # mip_encode 'done' rows
    assert mip["/data/raw/day1/done"]["dsr_dir"] == "/data/raw/day1/done/out/deskew"
    assert mip["/data/raw/day1/done"]["leaf_dir"] == "/data/raw/day1/done"
    assert (
        mip["/data/raw/day1/stale"]["dsr_dir"] == "/data/raw/day1/stale/out/deskew"
    )  # LEFT JOIN: any status


def test_change_version_and_changed_keys(reg_path):
    path, _ = reg_path
    with registry.RegistryReader(path) as r:
        newest = r.change_version()
        assert newest and r.changed_keys(newest) == set()
        assert "/data/raw/day1/done" in r.changed_keys("")
    # strings of UTC isoformat compare: one more write moves the version
    with registry.StatusRegistry(path) as w:
        w.start_stage("/data/raw/day1/untouched", "roi_detect")
    with registry.RegistryReader(path) as r:
        assert r.change_version() > newest
        assert r.changed_keys(newest) == {"/data/raw/day1/untouched"}


def test_awkward_keys_survive_the_url_key_and_the_sqlite_uri(tmp_path):
    key = golden.AWKWARD_DATASET_KEY
    assert registry.decode_key(registry.encode_key(key)) == key
    assert "/" not in registry.encode_key(key) and "=" not in registry.encode_key(key)
    # the registry file itself lives under a path with URI-special characters
    weird = tmp_path / "a b#c?d%e"
    weird.mkdir()
    expected = golden.build_registry(weird / "registry.sqlite3")
    with registry.RegistryReader(weird / "registry.sqlite3") as r:
        assert {d["dataset_key"] for d in r.datasets()} == set(expected)


def test_missing_registry_is_empty_and_is_never_created(tmp_path):
    path = tmp_path / "nowhere" / "registry.sqlite3"
    with registry.RegistryReader(path) as r:
        assert not r.exists
        assert r.datasets() == [] and r.stage_rows() == [] and r.dashboard_rows() == []
        assert r.get_dataset("x") is None and r.leaf_dir("x") is None
        assert r.change_version() == "" and r.changed_keys("") == set()
        assert r.deskewed_datasets() == [] and r.mip_done_datasets() == []
    with registry.connect_ro(path) as conn:
        assert conn is None
    assert not path.exists() and not path.parent.exists()


def test_a_registry_that_appears_later_is_picked_up(tmp_path):
    path = tmp_path / "registry.sqlite3"
    r = registry.RegistryReader(path)
    assert r.datasets() == []
    expected = golden.build_registry(path)
    assert {d["dataset_key"] for d in r.datasets()} == set(expected)
    r.close()


def test_the_reader_cannot_write(reg_path):
    path, _ = reg_path
    r = registry.RegistryReader(path)
    r.datasets()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        r._connection().execute("DELETE FROM datasets")
    r.close()
    with (
        registry.connect_ro(path) as conn,
        pytest.raises(sqlite3.OperationalError, match="readonly"),
    ):
        conn.execute("UPDATE datasets SET signal_flag='ok'")
    with registry.StatusRegistry(path) as w:  # nothing was lost
        assert len(w.all_datasets()) == 9


@pytest.mark.skipif(os.name != "posix", reason="chmod")
def test_a_read_only_file_is_readable(reg_path):
    path, expected = reg_path
    path.chmod(0o444)
    try:
        if os.access(path, os.W_OK):
            pytest.skip("permissions are not enforced (root)")
        with registry.RegistryReader(path) as r:
            assert {d["dataset_key"] for d in r.datasets()} == set(expected)
    finally:
        path.chmod(0o644)


def test_the_reader_does_not_migrate_an_old_registry(tmp_path):
    """A registry from before `signal_flag` & co: read as it is, schema untouched."""
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """CREATE TABLE datasets (
               dataset_key TEXT PRIMARY KEY, root TEXT NOT NULL, leaf_dir TEXT NOT NULL,
               master_file TEXT NOT NULL, has_legacy_decon INTEGER NOT NULL DEFAULT 0,
               discovered_at TEXT NOT NULL);
           CREATE TABLE stage_status (
               dataset_key TEXT NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL,
               started_at TEXT, finished_at TEXT, output_path TEXT, ticket_path TEXT,
               error_message TEXT, PRIMARY KEY (dataset_key, stage));
           INSERT INTO datasets VALUES
               ('/a/b', '/a', '/a/b', '/a/b/m.tif', 0, '2026-01-01T00:00:00+00:00');"""
    )
    conn.commit()
    conn.close()
    with registry.RegistryReader(path) as r:
        rows = r.dashboard_rows()
    assert rows[0]["overall_status"] == "pending" and "signal_flag" not in rows[0]
    with sqlite3.connect(path) as c:
        assert "signal_flag" not in [
            x[1] for x in c.execute("PRAGMA table_info(datasets)")
        ]


def test_overall_status_rules_directly():
    done = {
        "status": "done",
        "started_at": "2026-01-01T00:00:01",
        "finished_at": "2026-01-01T00:00:02",
        "output_path": "x",
    }
    mip = {
        **done,
        "started_at": "2026-01-01T00:00:03",
        "finished_at": "2026-01-01T00:00:04",
    }
    assert registry.overall_status({"deskew": done, "mip_encode": mip}) == "done"
    late = {**done, "started_at": "2026-01-01T00:00:09"}  # restarted after the MIPs
    assert (
        registry.overall_status({"deskew": late, "mip_encode": mip}) == "reprocessing"
    )
    failed = {"status": "failed"}
    assert registry.overall_status({"deskew": failed, "mip_encode": mip}) == "failed"
    assert registry.overall_status({"deskew": failed}, "blocked") == "blocked"
    assert (
        registry.overall_status({"deskew": done}, "blocked") == "in_progress"
    )  # blocked needs a failed stage
    assert registry.overall_status({s: None for s in registry.STAGES}) == "pending"
    assert (
        registry.overall_status({"roi_detect": {"status": "pending"}}) == "in_progress"
    )
    assert (
        registry.overall_status(
            {"mip_encode": {"status": "pending", "output_path": "o"}}
        )
        == "reprocessing"
    )


def test_path_context():
    assert registry.path_context("/r/a/b/Cell", "/r") == "a/b"
    assert registry.path_context("/r/Cell", "/r") == ""
    assert registry.path_context("/r/a/b/c/d/e/f/Cell", "/r") == ".../c/d/e/f"
    # leaf outside root: the dashboard's long-standing behaviour, kept as it was
    # (the absolute anchor shows as a leading "//"; harmless in a table cell)
    assert registry.path_context("/x/y/Cell", "/elsewhere") == "//x/y"


def test_registry_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv(registry.REGISTRY_PATH_ENV, str(tmp_path / "r.sqlite3"))
    assert registry.registry_path() == tmp_path / "r.sqlite3"
    monkeypatch.delenv(registry.REGISTRY_PATH_ENV)
    assert str(registry.registry_path()) == registry.DEFAULT_REGISTRY_PATH
    assert registry.RegistryReader().path == registry.registry_path()


# ---------------------------------------------------------------------------
# timeline: zarr-free raw times
# ---------------------------------------------------------------------------


def _nan_equal(a, b):
    return len(a) == len(b) and all(
        (x == y) or (np.isnan(x) and np.isnan(y)) for x, y in zip(a, b)
    )


@pytest.mark.parametrize("chunks", [None, 2, 1, 4])
def test_zarr_free_reader_reads_both_golden_vectors(tmp_path, chunks):
    store = golden.write_raw_store(
        tmp_path / golden.RAW_STORE_NAME,
        ms=golden.RAW_MS,
        epoch=golden.RAW_EPOCH,
        chunks=chunks,
    )
    assert timeline.read_f8_array(store / "t") == golden.RAW_MS
    assert _nan_equal(timeline.read_f8_array(store / "t_epoch_s"), golden.RAW_EPOCH)
    assert timeline.read_raw_times(store) == golden.RAW_MS_SECONDS  # the rig's `t` wins
    only_epoch = golden.write_raw_store(
        tmp_path / "e.ome.zarr", epoch=golden.RAW_EPOCH, chunks=chunks
    )
    assert timeline.read_raw_times(only_epoch) == golden.RAW_EPOCH_SECONDS


def test_golden_relative_times_and_records(tmp_path):
    ms = golden.write_raw_store(tmp_path / golden.RAW_STORE_NAME, ms=golden.RAW_MS)
    assert timeline.from_raw_stores([ms]) == golden.RAW_MS_RELATIVE
    assert timeline.interval_s(golden.RAW_MS_RELATIVE) == golden.RAW_MS_INTERVAL_S
    ep = golden.write_raw_store(
        tmp_path / "ep" / golden.RAW_STORE_NAME, epoch=golden.RAW_EPOCH
    )
    assert timeline.from_raw_stores([ep]) == golden.RAW_EPOCH_RELATIVE
    assert timeline.interval_s(golden.RAW_EPOCH_RELATIVE) == golden.RAW_EPOCH_INTERVAL_S
    for raw, want in (
        (ms, golden.EXPECTED_OPYM_TIME_MS),
        (ep, golden.EXPECTED_OPYM_TIME_EPOCH),
    ):
        proc = tmp_path / f"{raw.parent.name}_proc.ome.zarr"
        proc.mkdir()
        (proc / ".zgroup").write_text("{}")
        assert timeline.sync_processed(proc, [raw]) is True
        assert timeline.read_times(proc) == want


def test_unset_epoch_timestamp_is_not_1970(tmp_path):
    store = golden.write_raw_store(
        tmp_path / "z.ome.zarr", epoch=golden.RAW_EPOCH_UNSET
    )
    assert timeline.read_raw_times(store) is None


def test_zarr_free_and_zarr_agree_on_the_golden_store(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = golden.write_raw_store(
        tmp_path / golden.RAW_STORE_NAME,
        ms=golden.RAW_MS,
        epoch=golden.RAW_EPOCH,
        chunks=2,
    )
    g = zarr.open_group(str(store), mode="r")
    assert _nan_equal([float(v) for v in g["t"][:]], golden.RAW_MS)
    assert _nan_equal([float(v) for v in g["t_epoch_s"][:]], golden.RAW_EPOCH)


def _compressed_store(tmp_path):
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "c.ome.zarr"
    root = zarr.open_group(str(path), mode="w")
    a = root.create_dataset(
        "t", shape=(4,), dtype="f8", fill_value=0.0
    )  # default compressor
    a[:] = [100.0, 8100.0, 16100.0, 24100.0]
    return path


def test_compressed_arrays_use_zarr_when_installed(tmp_path):
    store = _compressed_store(tmp_path)
    with pytest.raises(timeline.ArrayNeedsZarrError, match="compressed"):
        timeline.read_f8_array(store / "t")
    assert timeline.read_raw_times(store) == [0.1, 8.1, 16.1, 24.1]


def test_compressed_arrays_without_zarr_fail_clearly(tmp_path, monkeypatch):
    store = _compressed_store(tmp_path)
    monkeypatch.setitem(sys.modules, "zarr", None)  # `import zarr` -> ImportError
    with pytest.raises(timeline.ArrayNeedsZarrError, match="compressed.*needs zarr"):
        timeline.read_raw_times(store)
    assert timeline.read_raw_times(store, strict=False) is None
    assert timeline.from_raw_stores([store], strict=False) is None
    # ...and the zarr-free path is untouched by the missing import
    plain = golden.write_raw_store(tmp_path / "plain.ome.zarr", ms=golden.RAW_MS)
    assert timeline.read_raw_times(plain) == golden.RAW_MS_SECONDS


def test_odd_arrays_are_refused_not_misread(tmp_path):
    base = tmp_path / "s.ome.zarr"
    golden.write_raw_store(base, ms=golden.RAW_MS)
    meta_path = base / "t" / ".zarray"
    meta = json.loads(meta_path.read_text())
    for patch in (
        {"dtype": ">f8"},
        {"dtype": "<f4"},
        {"shape": [2, 2], "chunks": [2, 2]},
        {"filters": [{"id": "delta"}]},
        {"compressor": {"id": "blosc"}},
    ):
        meta_path.write_text(json.dumps({**meta, **patch}))
        with pytest.raises(timeline.ArrayNeedsZarrError):
            timeline.read_f8_array(base / "t")
    (base / "t" / ".zarray").unlink()
    (base / "t" / "zarr.json").write_text("{}")
    with pytest.raises(timeline.ArrayNeedsZarrError, match="v3"):
        timeline.read_f8_array(base / "t")


def test_no_array_and_no_store_are_none(tmp_path):
    empty = golden.write_raw_store(tmp_path / "empty.ome.zarr")
    assert timeline.read_f8_array(empty / "t") is None
    assert timeline.read_raw_times(empty) is None
    assert timeline.read_raw_times(tmp_path / "missing.ome.zarr") is None


def test_resolve_times_order_and_sources(tmp_path):
    raw = golden.write_raw_store(tmp_path / golden.RAW_STORE_NAME, ms=golden.RAW_MS)
    rec = {"times_s": [0.0, 7.5], "source": "raw store X"}
    # 1. the processed store's own record beats everything
    assert timeline.resolve_times(
        {"opym_time": rec}, [raw], n_frames=5, axis_step_s=1.0
    ) == ([0.0, 7.5], "store:raw store X")
    assert (
        timeline.resolve_times({"opym_time": {"times_s": [0.0]}}, [], n_frames=1)[1]
        == "store:opym_time"
    )
    # 2. then the raw store
    assert timeline.resolve_times({}, [raw], n_frames=5, axis_step_s=1.0) == (
        golden.RAW_MS_RELATIVE,
        "raw",
    )
    assert timeline.resolve_times(None, [raw]) == (golden.RAW_MS_RELATIVE, "raw")
    # 3. then the nominal axis step
    assert timeline.resolve_times({}, [], n_frames=3, axis_step_s=7.5) == (
        [0.0, 7.5, 15.0],
        "nominal",
    )
    # nothing known: never invented
    assert timeline.resolve_times({}, [], n_frames=3) == (None, None)
    assert timeline.resolve_times({}, [], n_frames=3, axis_step_s=0.0) == (None, None)
    assert timeline.resolve_times(
        {"opym_time": {"times_s": []}}, [tmp_path / "gone"], n_frames=0
    ) == (None, None)


def test_nominal_times():
    assert timeline.nominal_times(4, 0.3) == [0.0, 0.3, 0.6, 0.9]
    assert timeline.nominal_times(0, 1.0) == []


def test_resolve_times_is_soft_about_compressed_raw_stores_by_default(
    tmp_path, monkeypatch
):
    store = _compressed_store(tmp_path)
    monkeypatch.setitem(sys.modules, "zarr", None)
    assert timeline.resolve_times({}, [store], n_frames=2, axis_step_s=8.0) == (
        [0.0, 8.0],
        "nominal",
    )
    with pytest.raises(timeline.ArrayNeedsZarrError):
        timeline.resolve_times({}, [store], n_frames=2, axis_step_s=8.0, strict=True)


# ---------------------------------------------------------------------------
# z step (DUP-08)
# ---------------------------------------------------------------------------


def test_z_step_defaults_are_named_once():
    assert (metadata.LEGACY_DEFAULT_Z_STEP_UM, metadata.GUESSED_Z_STEP_UM) == (1.0, 0.3)
    assert metadata.ZARR_DEFAULT_Z_STEP_ENV == "OPYM_ZARR_DEFAULT_Z_STEP"
    assert metadata.ZARR_ALLOW_DEFAULT_Z_STEP_ENV == "OPYM_ZARR_ALLOW_DEFAULT_Z_STEP"
    import inspect

    assert (
        inspect.signature(metadata.parse_z_step).parameters["default_z_step"].default
        == 1.0
    )
    assert (
        inspect.signature(metadata.parse_zarr_z_step)
        .parameters["default_z_step"]
        .default
        == 0.3
    )
    assert (
        inspect.signature(metadata.resolve_zarr_z_step)
        .parameters["default_z_step"]
        .default
        == 0.3
    )


@pytest.mark.parametrize(
    "env, fallback, want",
    [
        ({}, None, None),  # the backfill: refuse to guess
        ({}, 0.3, 0.3),  # the stream client: its old literal
        ({"OPYM_ZARR_ALLOW_DEFAULT_Z_STEP": "1"}, None, 0.3),
        (
            {"OPYM_ZARR_ALLOW_DEFAULT_Z_STEP": "0"},
            None,
            0.3,
        ),  # any non-empty string allows
        ({"OPYM_ZARR_ALLOW_DEFAULT_Z_STEP": ""}, None, None),
        ({"OPYM_ZARR_DEFAULT_Z_STEP": "0.5"}, None, 0.5),  # the production unit
        (
            {"OPYM_ZARR_DEFAULT_Z_STEP": "0.5", "OPYM_ZARR_ALLOW_DEFAULT_Z_STEP": "1"},
            None,
            0.5,
        ),  # explicit wins
        (
            {"OPYM_ZARR_DEFAULT_Z_STEP": "", "OPYM_ZARR_ALLOW_DEFAULT_Z_STEP": "1"},
            None,
            0.3,
        ),
        ({"OPYM_ZARR_DEFAULT_Z_STEP": "0.1"}, 0.3, 0.1),
    ],
)
def test_zarr_fallback_z_step_is_bioimagings_rule(env, fallback, want):
    assert metadata.zarr_fallback_z_step(env, fallback=fallback) == want


def test_zarr_fallback_z_step_reads_the_real_environment(monkeypatch):
    monkeypatch.delenv("OPYM_ZARR_DEFAULT_Z_STEP", raising=False)
    monkeypatch.delenv("OPYM_ZARR_ALLOW_DEFAULT_Z_STEP", raising=False)
    assert metadata.zarr_fallback_z_step() is None
    monkeypatch.setenv("OPYM_ZARR_DEFAULT_Z_STEP", "0.25")
    assert metadata.zarr_fallback_z_step() == 0.25
    monkeypatch.setenv("OPYM_ZARR_DEFAULT_Z_STEP", "not a number")
    with pytest.raises(ValueError):
        metadata.zarr_fallback_z_step()


def test_parse_z_step_reads_acqsettings_and_falls_back(tmp_path, capsys):
    meta = tmp_path / "x_metadata.txt"
    assert metadata.parse_z_step(meta) == 1.0  # no AcqSettings.txt: the legacy default
    assert metadata.parse_z_step(meta, default_z_step=0.3) == 0.3
    (tmp_path / "AcqSettings.txt").write_text(json.dumps({"stepSizeUm": 0.5}))
    assert metadata.parse_z_step(meta) == 0.5
    (tmp_path / "AcqSettings.txt").write_text(json.dumps({"zStep_um": 0.2}))
    assert (
        metadata.parse_z_step(meta) == 0.2
    )  # the alias run_pipeline_cli.get_z_step never read


@pytest.fixture
def running_receiver():
    zmq = pytest.importorskip("zmq")
    from opym.stream.receiver import StreamReceiver

    recv = StreamReceiver(
        bind_addr="tcp://127.0.0.1:0", ack_every_n_frames=1, ack_every_sec=0.05
    )
    endpoint = recv._socket.getsockopt(zmq.LAST_ENDPOINT).decode()
    stop = threading.Event()

    def pump():
        while not stop.is_set():
            recv._run_once()

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    yield endpoint
    stop.set()
    thread.join(timeout=2)
    recv.close()


def _stream_store_without_z(base: Path) -> Path:
    zarr = pytest.importorskip("zarr")
    store = base / "Cell_5_GFP_488.ome.zarr"
    store.mkdir(parents=True)
    (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (store / ".zattrs").write_text(
        json.dumps(
            {
                "multiscales": [
                    {
                        "axes": [{"name": n} for n in "tzyx"],
                        "datasets": [{"path": "p0"}],
                    }
                ]
            }
        )
    )
    arr = zarr.open(
        str(store / "p0"),
        mode="a",
        shape=(1, 2, 3, 4),
        chunks=(1, 1, 3, 4),
        dtype="uint16",
        compressor=None,
        dimension_separator="/",
    )
    arr[0] = np.ones((2, 3, 4), dtype="uint16")
    for f in store.rglob("*"):
        if f.is_file():
            os.utime(f, (1.0, 1.0))  # long settled
    return store


@pytest.mark.parametrize(
    "env, want", [({}, 0.3), ({"OPYM_ZARR_DEFAULT_Z_STEP": "0.5"}, 0.5)]
)
def test_stream_client_uses_the_shared_fallback(
    tmp_path, monkeypatch, running_receiver, env, want
):
    """Unchanged with no env (0.3); the production override now applies too."""
    pytest.importorskip("zarr")
    from opym.stream.client import watch_and_stream

    monkeypatch.delenv("OPYM_ZARR_DEFAULT_Z_STEP", raising=False)
    monkeypatch.delenv("OPYM_ZARR_ALLOW_DEFAULT_Z_STEP", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    local = tmp_path / "acq"
    store = _stream_store_without_z(local)
    raw_root = tmp_path / "raw"
    assert metadata.parse_zarr_z_step_from_store(store) is None
    watch_and_stream(
        {0: store},
        {0: "GFP_488"},
        base_name="Cell_5",
        raw_root=str(raw_root),
        connect_addr=running_receiver,
        poll_interval_s=0.05,
        stability_polls=1,
        idle_exit_after_s=5,
    )
    mirrored = raw_root / "Cell_5_GFP_488.ome.zarr"
    assert metadata.parse_zarr_z_step_from_store(mirrored) == pytest.approx(want)


# ---------------------------------------------------------------------------
# resolve_output_base golden cases
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.name != "posix", reason="the mirror reproduces an absolute POSIX path"
)
def test_output_base_contract_vectors(tmp_path, monkeypatch):
    monkeypatch.setenv(utils.MIRROR_ROOT_ENV, str(tmp_path / "mirror"))
    work = tmp_path / "work"
    cases = golden.output_base_cases(work, tmp_path / "mirror")
    try:
        names = {c.name for c in cases}
        assert {
            "writable leaf",
            "leaf does not exist yet",
            "ancestor rule",
            "already in the mirror",
        } <= names
        for c in cases:
            assert utils.resolve_output_base(c.leaf_dir) == c.expected, c.name
        # resolving never creates the leaf
        assert not next(
            c for c in cases if c.name == "leaf does not exist yet"
        ).leaf_dir.exists()
        if os.access(work, os.W_OK) and "existing unwritable leaf" not in names:
            pytest.skip("permissions are not enforced (root): unwritable cases skipped")
        assert names >= {
            "existing unwritable leaf",
            "missing leaf under unwritable ancestor",
        }
    finally:
        golden.restore_permissions(work)


def test_output_base_vectors_catch_a_broken_rule(tmp_path, monkeypatch):
    """Negative control: the pre-ancestor-rule behaviour (a missing leaf is never
    writable) must fail the ancestor-rule vector."""
    monkeypatch.setenv(utils.MIRROR_ROOT_ENV, str(tmp_path / "mirror"))
    cases = golden.output_base_cases(tmp_path / "work", tmp_path / "mirror")
    try:

        def old_rule(leaf):
            leaf = Path(leaf)
            if os.access(leaf, os.W_OK):
                return leaf
            return tmp_path / "mirror" / str(leaf.resolve()).lstrip("/")

        broken = [c.name for c in cases if old_rule(c.leaf_dir) != c.expected]
        assert "leaf does not exist yet" in broken and "ancestor rule" in broken
    finally:
        golden.restore_permissions(tmp_path / "work")


# ---------------------------------------------------------------------------
# backfill run records
# ---------------------------------------------------------------------------


def test_run_record_readers(tmp_path, monkeypatch):
    d = tmp_path / "runs"
    d.mkdir()
    now = 1_000_000.0
    docs = {
        "20261001T000000-a": {
            "run_id": "a",
            "state": "running",
            "argus_last_seen": now - 10,
        },
        "20261002T000000-b": {
            "run_id": "b",
            "state": "running",
            "argus_last_seen": now - 700,
        },
        "20261003T000000-c": {
            "run_id": "c",
            "state": "complete",
            "argus_last_seen": 0,
            "updated_at": 5,
        },
    }
    for name, doc in docs.items():
        (d / f"{name}.json").write_text(json.dumps(doc))
    (d / "20261004T000000-bad.json").write_text("not json")
    (d / "20261005T000000-list.json").write_text("[1]")
    monkeypatch.setenv(backfill_runs.RUNS_DIR_ENV, str(d))
    assert backfill_runs.runs_dir() == d
    # the newest `limit` FILES, newest first; unreadable ones are skipped (as the
    # dashboard always did)
    assert [r["run_id"] for r in backfill_runs.recent_runs(4)] == ["c", "b"]
    assert backfill_runs.recent_runs(2) == []
    assert [r["run_id"] for r in backfill_runs.recent_runs(10)] == ["c", "b", "a"]
    assert not backfill_runs.is_stale(docs["20261001T000000-a"], now)
    assert backfill_runs.is_stale(
        docs["20261002T000000-b"], now
    )  # silent for > STALE_S
    assert not backfill_runs.is_stale(
        docs["20261003T000000-c"], now
    )  # finished runs are never stale
    assert backfill_runs.last_seen(docs["20261003T000000-c"]) == 5
    assert backfill_runs.recent_runs(directory=tmp_path / "none") == []
    monkeypatch.delenv(backfill_runs.RUNS_DIR_ENV)
    assert str(backfill_runs.runs_dir()) == backfill_runs.ARGUS_RUNS_DIR


def test_the_receiver_sweep_uses_the_same_staleness_rule(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    import time

    old = time.time() - backfill_runs.STALE_S - 5
    (d / "r1.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "state": "running",
                "argus_last_seen": old,
                "current": {"key": "k"},
            }
        )
    )
    (d / "r2.json").write_text(
        json.dumps({"run_id": "r2", "state": "running", "argus_last_seen": time.time()})
    )
    recs = backfill_runs.RunRecords(d)  # sweeps on construction
    assert json.loads((d / "r1.json").read_text())["state"] == "interrupted"
    assert json.loads((d / "r1.json").read_text())["interrupted_during"] == {"key": "k"}
    assert json.loads((d / "r2.json").read_text())["state"] == "running"
    assert recs.sweep(force=True) == 0


def test_the_vocabulary_matches_the_sender():
    from opym.stream.bfill_run import DONE_RESULTS

    assert backfill_runs.DONE == DONE_RESULTS
    assert backfill_runs.WORK == backfill_runs.SENDING | backfill_runs.CHECKING
    assert not backfill_runs.SENDING & backfill_runs.CHECKING
    assert (
        "planning" in backfill_runs.RUN_STATES
        and "interrupted" in backfill_runs.RUN_STATES
    )
