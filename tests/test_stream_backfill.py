"""SSH backfill: a loopback receiver and sender over real ZMQ sockets and real zarr stores.

The contract under test:
  * the copy is pixel-identical and only marked complete after a readback match;
  * nothing existing is superseded, overwritten or modified while something else writes it;
  * writes hold while a live acquisition holds the lease;
  * an interrupted run resumes by sending only what is missing.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import zarr
import zmq
from opym import lanes
from opym.stream import backfill_client as bc
from opym.stream import bfill
from opym.stream.backfill_receiver import BackfillReceiver

NZ, NY, NX = 4, 5, 6


def make_store(path: Path, *, declared_t=3, written=(0, 1, 2), seed=0, shape_yx=(NY, NX)) -> Path:
    """A stand-in for the acquisition writer's store: per-plane chunks, `/` separator,
    blosc, a small coordinate array, and `.zattrs` multiscales."""
    rng = np.random.default_rng(seed)
    root = zarr.open_group(str(path), mode="w")
    arr = root.create_dataset("p0", shape=(declared_t, NZ, *shape_yx), chunks=(1, 1, *shape_yx),
                              dtype="<u2", dimension_separator="/")
    for t in written:
        arr[t] = rng.integers(1, 4000, size=(NZ, *shape_yx), dtype=np.uint16)
    root.create_dataset("z", data=np.arange(NZ) * 0.5)
    root.attrs["multiscales"] = [{"axes": [{"name": n} for n in "tzyx"],
                                  "datasets": [{"path": "p0"}], "version": "0.4"}]
    return path


@pytest.fixture
def world(tmp_path):
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    receiver = BackfillReceiver(bind_addr="tcp://127.0.0.1:0", raw_roots=[remote], settle_seconds=0, jobs_dir=jobs)
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            receiver._run_once(20)

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    clients = []

    def make_client(**kw):
        kw.setdefault("settle_s", 0)
        kw.setdefault("reply_timeout_s", 10)
        c = bc.BackfillClient([receiver.endpoint], [(local, str(remote))], **kw)
        clients.append(c)
        return c

    class W:
        pass

    w = W()
    w.local, w.remote, w.jobs, w.receiver, w.client = local, remote, jobs, receiver, make_client
    yield w
    stop.set()
    th.join(timeout=2)
    for c in clients:
        c.close()
    receiver.close()


def leaf_report(reports, name):
    (rep,) = reports
    return rep["stores"][name]


def remote_store(w, rel="exp/cell_001_GFP_488.ome.zarr") -> Path:
    return w.remote / rel


def arrays_equal(a: Path, b: Path) -> bool:
    za, zb = zarr.open_array(str(a / "p0"), mode="r"), zarr.open_array(str(b / "p0"), mode="r")
    return za.shape == zb.shape and np.array_equal(za[...], zb[...])


# ---------------------------------------------------------------- the copy


def test_copy_is_pixel_identical_marked_and_zattrs_written_last(world):
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    reports = world.client().run_once()
    assert leaf_report(reports, src.name)["result"] == "complete"
    dst = remote_store(world)
    assert arrays_equal(src, dst)
    # small files are copied verbatim
    for rel in (".zattrs", ".zgroup", "p0/.zarray", "z/.zarray"):
        assert (src / rel).read_bytes() == (dst / rel).read_bytes(), rel
    marker = bfill.read_marker(dst)
    assert marker and marker["written_t"] == [0, 1, 2] and marker["verified"] == "readback-sha256"
    assert not (dst / bfill.LOCK_NAME).exists()


def test_second_run_sends_nothing(world):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    world.client().run_once()
    sent = world.receiver.stats["volumes"]
    reports = world.client().run_once()
    assert leaf_report(reports, "cell_001_GFP_488.ome.zarr")["result"] == "already complete"
    assert world.receiver.stats["volumes"] == sent


def test_aborted_acquisition_copies_only_written_timepoints(world):
    """Declares 5 timepoints, wrote 3: the copy declares 5 and holds 3, like the source."""
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", declared_t=5, written=(0, 1, 2))
    world.client().run_once()
    dst = remote_store(world)
    assert zarr.open_array(str(dst / "p0"), mode="r").shape[0] == 5
    assert bfill.read_marker(dst)["written_t"] == [0, 1, 2]
    assert sorted(p.name for p in (dst / "p0").iterdir() if p.name.isdigit()) == ["0", "1", "2"]
    assert arrays_equal(src, dst)


def test_missing_planes_stay_missing(world):
    """A timepoint stopped mid-write (some z chunks absent locally) is not padded on the copy."""
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", written=(0, 1))
    for z in (2, 3):                                        # t=1 lost its last two planes
        shutil.rmtree(src / "p0" / "1" / str(z))
    reports = world.client().run_once()
    assert leaf_report(reports, src.name)["result"] == "complete"
    dst = remote_store(world)
    present = sorted(int(e.name) for e in (dst / "p0" / "1").iterdir() if e.name.isdigit())
    assert present == [0, 1]                                # not padded with zero chunks
    assert arrays_equal(src, dst)


# ---------------------------------------------------------------- resume


def test_interrupted_run_resumes_sending_only_what_is_missing(world, monkeypatch):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    c1 = world.client(slab_bytes=NY * NX * 2)            # one plane per slab: 12 slabs in all
    sent_slabs = []
    real = c1._send_slab

    def flaky(key, header, payload, inflight, attempts):
        if len(sent_slabs) >= 5:
            raise bc.BackfillError("simulated tunnel failure")
        sent_slabs.append(key)
        real(key, header, payload, inflight, attempts)

    monkeypatch.setattr(c1, "_send_slab", flaky)
    res = leaf_report(c1.run_once(), "cell_001_GFP_488.ome.zarr")
    assert res["result"] == "refused" and "simulated tunnel failure" in res["reason"]
    dst = remote_store(world)
    assert bfill.read_marker(dst) is None
    assert not (dst / ".zattrs").exists()                   # an incomplete store never looks ready
    before = world.receiver.stats["volumes"]

    reports = world.client().run_once()
    assert leaf_report(reports, "cell_001_GFP_488.ome.zarr")["result"] == "complete"
    resent = world.receiver.stats["volumes"] - before
    assert 0 < resent < 3 * NZ                              # not all 12 slabs were sent again
    assert bfill.read_marker(dst)


def test_a_store_that_grew_since_it_was_marked_is_topped_up(world):
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", declared_t=3, written=(0,))
    world.client().run_once()
    dst = remote_store(world)
    assert bfill.read_marker(dst)["written_t"] == [0]
    arr = zarr.open_array(str(src / "p0"), mode="r+")        # the acquisition kept going
    arr[1] = np.full((NZ, NY, NX), 7, np.uint16)
    arr[2] = np.full((NZ, NY, NX), 9, np.uint16)
    before = world.receiver.stats["volumes"]
    world.client().run_once()
    assert bfill.read_marker(dst)["written_t"] == [0, 1, 2]
    assert world.receiver.stats["volumes"] - before == 2        # one slab each for t=1 and t=2 only
    assert arrays_equal(src, dst)


# ---------------------------------------------------------------- never damage existing data


def test_existing_store_with_a_different_array_is_a_conflict_and_untouched(world):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    other = make_store(world.remote / "exp" / "cell_001_GFP_488.ome.zarr", shape_yx=(NY + 1, NX), seed=5)
    snapshot = {p: p.read_bytes() for p in other.rglob("*") if p.is_file()}
    reports = world.client().run_once()
    res = leaf_report(reports, "cell_001_GFP_488.ome.zarr")
    assert res["result"] == bfill.CONFLICT
    assert {p: p.read_bytes() for p in other.rglob("*") if p.is_file()} == snapshot
    assert not list((world.remote / "exp").glob(".superseded*"))


def test_a_store_that_differs_only_in_declared_length_is_a_conflict(world):
    """Same chunks, same dtype; only the declared timepoints differ. Different acquisition, not ours to fill."""
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", declared_t=3)
    other = make_store(world.remote / "exp" / "cell_001_GFP_488.ome.zarr", declared_t=4, written=(0,))
    snapshot = {p: p.read_bytes() for p in other.rglob("*") if p.is_file()}
    reports = world.client().run_once()
    assert leaf_report(reports, "cell_001_GFP_488.ome.zarr")["result"] == bfill.CONFLICT
    assert {p: p.read_bytes() for p in other.rglob("*") if p.is_file()} == snapshot


def test_a_store_something_else_is_writing_is_busy_and_untouched(world):
    world.receiver.settle_s = 900
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    other = make_store(world.remote / "exp" / "cell_001_GFP_488.ome.zarr", written=(0,))   # just written
    snapshot = {p: p.read_bytes() for p in other.rglob("*") if p.is_file()}
    reports = world.client().run_once()
    assert leaf_report(reports, "cell_001_GFP_488.ome.zarr")["result"] == bfill.BUSY
    assert {p: p.read_bytes() for p in other.rglob("*") if p.is_file()} == snapshot


def test_a_store_locked_by_another_machines_sender_is_busy(world):
    """Two senders must not write one store. A lock held by a different client id blocks; our own does not."""
    world.receiver.settle_s = 0
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    other = make_store(world.remote / "exp" / "cell_001_GFP_488.ome.zarr", written=(0,))
    (other / bfill.LOCK_NAME).write_text("some-other-pc")
    reports = world.client().run_once()
    res = leaf_report(reports, src.name)
    assert res["result"] == bfill.BUSY and "some-other-pc" in res["reason"]
    (other / bfill.LOCK_NAME).write_text(world.client().client_id)       # our own lock: carry on
    assert leaf_report(world.client().run_once(), src.name)["result"] == "complete"


def test_the_same_store_is_taken_once_it_has_been_idle(world):
    """Negative control for the busy test: with the settle window off it is completed."""
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    make_store(world.remote / "exp" / "cell_001_GFP_488.ome.zarr", written=(0,), seed=0)
    reports = world.client().run_once()
    assert leaf_report(reports, src.name)["result"] == "complete"


def test_destination_outside_the_allowlist_is_refused(world, tmp_path):
    outside = tmp_path / "elsewhere"
    c = world.client()
    c.root_map = [(world.local, str(outside))]
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    (rep,) = c.run_once()
    assert "outside the allowed raw roots" in rep["error"]
    assert not outside.exists()


def _raw(world):
    sock = zmq.Context.instance().socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, b"raw-test")
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(world.receiver.endpoint)

    def call(msg_type, header, payload=None):
        sock.send_multipart(bfill.pack(msg_type, "raw-test", {**header, "client": "raw-test"}, payload))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if sock.poll(50):
                return bfill.unpack(sock.recv_multipart())[2]
        raise AssertionError("no reply")

    return call, sock


def test_path_traversal_in_metadata_is_refused(world):
    call, sock = _raw(world)
    try:
        reply = call(bfill.BF_META, {"leaf": str(world.remote / "exp"), "store": "x_GFP_488.ome.zarr",
                                     "files": [{"rel": "../../escape", "data": b"x"}]})
        assert not reply["ok"] and "unsafe" in reply["error"]
        assert not (world.remote / "escape").exists() and not (world.remote.parent / "escape").exists()
    finally:
        sock.close(0)


def test_a_volume_corrupted_in_transit_is_rejected_not_written(world):
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", written=(0,))
    world.client().run_once()          # creates + completes the store (so the array exists)
    call, sock = _raw(world)
    try:
        slab = np.full((1, NY, NX), 4242, np.uint16)
        reply = call(bfill.BF_VOLUME, {"leaf": str(world.remote / "exp"), "store": src.name, "t": 0, "z0": 0,
                                       "nz": 1, "shape_yx": [NY, NX], "sha256": "0" * 64}, slab.tobytes())
        assert not reply["ok"] and reply.get("retry")
        assert not (zarr.open_array(str(remote_store(world) / "p0"), mode="r")[0, 0] == 4242).any()
    finally:
        sock.close(0)


# ---------------------------------------------------------------- verification


def test_readback_mismatch_blocks_the_completion_marker(world, monkeypatch):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    real_commit = world.receiver._commit

    def tamper_then_commit(header):
        dst = Path(header["leaf"]) / header["store"]
        arr = zarr.open_array(str(dst / "p0"), mode="r+")     # disk changes after the volume was acked
        arr[1, 0] = arr[1, 0] + 1
        return real_commit(header)

    monkeypatch.setattr(world.receiver, "_commit", tamper_then_commit)
    res = leaf_report(world.client().run_once(), "cell_001_GFP_488.ome.zarr")
    assert res["result"] == "refused" and "differ after readback" in res["reason"]
    dst = remote_store(world)
    assert bfill.read_marker(dst) is None and not (dst / ".zattrs").exists()


# ---------------------------------------------------------------- live priority


def test_nothing_is_written_while_a_live_acquisition_holds_the_lease(world, monkeypatch):
    monkeypatch.setattr(bc, "PAUSE_POLL_S", 0.05)
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    lanes.write_lease(["live-session"], world.jobs)
    dst = remote_store(world)

    def release():
        time.sleep(0.5)
        assert not (dst / "p0").exists() or not any(p.name.isdigit() for p in (dst / "p0").iterdir())
        lanes.release_lease(world.jobs)

    th = threading.Thread(target=release)
    th.start()
    t0 = time.monotonic()
    reports = world.client().run_once()
    th.join()
    assert leaf_report(reports, "cell_001_GFP_488.ome.zarr")["result"] == "complete"
    assert time.monotonic() - t0 >= 0.4          # it waited for the lease to go


def test_the_server_itself_refuses_volume_writes_during_a_live_session(world):
    """Even a sender that ignores the live flag cannot write: the hold is enforced server-side."""
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", written=(0,))
    world.client().run_once()
    lanes.write_lease(["live-session"], world.jobs)
    call, sock = _raw(world)
    try:
        slab = np.zeros((1, NY, NX), np.uint16)
        reply = call(bfill.BF_VOLUME, {"leaf": str(world.remote / "exp"), "store": src.name, "t": 0, "z0": 0,
                                       "nz": 1, "shape_yx": [NY, NX], "sha256": bfill.volume_digest(slab)},
                     slab.tobytes())
        assert reply["hold"] == "live" and not reply["ok"]
    finally:
        sock.close(0)
        lanes.release_lease(world.jobs)


def test_pause_file_stops_sending(world, tmp_path):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    pause = tmp_path / "PAUSE"
    pause.write_text("x")
    c = world.client(pause_file=pause)
    assert c.paused()
    pause.unlink()
    assert c.paused() is None


# ---------------------------------------------------------------- local side


def test_a_store_still_being_written_locally_is_skipped(world):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    reports = world.client(settle_s=900).run_once()
    assert leaf_report(reports, "cell_001_GFP_488.ome.zarr")["result"] == "settling"
    assert not (world.remote / "exp").exists()


def test_dry_run_reports_and_sends_nothing(world):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    reports = world.client().run_once(dry_run=True)
    res = leaf_report(reports, "cell_001_GFP_488.ome.zarr")
    assert res["result"] == "would send" and res["missing_timepoints"] == 3
    assert not (world.remote / "exp").exists()


def test_two_channels_in_one_folder_and_nested_folders(world):
    make_store(world.local / "day1" / "cell_1" / "cell_1_GFP_488.ome.zarr", seed=1)
    make_store(world.local / "day1" / "cell_1" / "cell_1_mScarlet_561.ome.zarr", seed=2)
    make_store(world.local / "day1" / "cell_2" / "cell_2_GFP_488.ome.zarr", seed=3)
    reports = world.client().run_once()
    assert len(reports) == 2
    for rel in ("day1/cell_1/cell_1_GFP_488.ome.zarr", "day1/cell_1/cell_1_mScarlet_561.ome.zarr",
                "day1/cell_2/cell_2_GFP_488.ome.zarr"):
        assert bfill.read_marker(world.remote / rel)["verified"] == "readback-sha256"


def test_the_marker_is_what_the_backfill_checks_for():
    """The marker format other code reads: pinned so a change is deliberate."""
    assert bfill.MARKER_NAME == ".opym_complete.json" and bfill.MARKER_FORMAT == 1
    assert bfill.read_marker(Path("/nonexistent/store")) is None


# ---------------------------------------------------------------- one copy of everything (2026-10-24)


def _as_stream_copy(store: Path, session_id: str = "0eff19b1") -> Path:
    """What the live stream leaves when a run stops early: the same pixel layout, its own
    root attributes tagged with its session, no coordinate extras, no backfill marker."""
    root = zarr.open_group(str(store), mode="r+")
    root.attrs.put({"multiscales": [{"version": "0.4", "datasets": [{"path": "p0"}]}],
                    "opym": {"session_id": session_id, "output_format": "both"}})
    return store


def test_the_live_streams_partial_copy_is_completed_in_place(world):
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", written=(0, 1, 2), seed=7)
    part = _as_stream_copy(make_store(remote_store(world), written=(0,), seed=7))
    t0_chunk = part / "p0" / "0" / "0" / "0" / "0"
    before = t0_chunk.read_bytes()
    reports = world.client().run_once()
    res = leaf_report(reports, src.name)
    assert res["result"] == "complete" and res["sent_timepoints"] == 2  # only what was missing
    assert arrays_equal(src, part)
    assert t0_chunk.read_bytes() == before  # the stream's pixels are never rewritten
    marker = bfill.read_marker(part)
    assert marker["written_t"] == [0, 1, 2]
    assert marker["completed_stream_copy"]["session_id"] == "0eff19b1"
    attrs = zarr.open_group(str(part), mode="r").attrs.asdict()
    assert attrs["opym"]["session_id"] == "0eff19b1"  # the stream's tags stay
    assert attrs["multiscales"][0]["axes"]  # the acquisition's attributes win
    assert sorted(p.name for p in (world.remote / "exp").iterdir()) == [src.name]  # one copy


def test_dry_run_says_it_completes_the_stream_copy(world):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", seed=7)
    _as_stream_copy(make_store(remote_store(world), written=(0,), seed=7))
    res = leaf_report(world.client().run_once(dry_run=True), "cell_001_GFP_488.ome.zarr")
    assert res == {"result": "would send", "missing_timepoints": 2, "of": 3, "completes_stream_copy": True}


def test_a_stream_copy_with_a_different_pixel_layout_is_still_a_conflict(world):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", seed=7)
    part = _as_stream_copy(make_store(remote_store(world), written=(0,), seed=7))
    zarray = part / "p0" / ".zarray"
    meta = json.loads(zarray.read_text())
    meta["compressor"] = {"id": "zstd", "level": 1}  # same shape, different bytes on disk
    zarray.write_text(json.dumps(meta))
    res = leaf_report(world.client().run_once(), "cell_001_GFP_488.ome.zarr")
    assert res["result"] == "refused" and "different pixel array" in res["reason"]


def test_a_store_that_is_not_the_streams_with_other_metadata_is_still_refused(world):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", seed=7)
    other = make_store(remote_store(world), written=(0,), seed=7)
    zarr.open_group(str(other), mode="r+").attrs["note"] = "someone else's"
    snapshot = (other / ".zattrs").read_bytes()
    res = leaf_report(world.client().run_once(), "cell_001_GFP_488.ome.zarr")
    assert res["result"] == "refused" and ".zattrs exists and differs" in res["reason"]
    assert (other / ".zattrs").read_bytes() == snapshot


@pytest.fixture
def indexed(world, tmp_path):
    """A second receiver that knows the registry's folders (for duplicates elsewhere)."""
    import sqlite3

    reg = tmp_path / "registry.sqlite3"
    conn = sqlite3.connect(reg)
    conn.execute("CREATE TABLE datasets (dataset_key TEXT, leaf_dir TEXT, master_file TEXT)")
    conn.commit()
    conn.close()
    receiver = BackfillReceiver(bind_addr="tcp://127.0.0.1:0", raw_roots=[world.remote], settle_seconds=0,
                                jobs_dir=world.jobs, registry_path=reg)
    stop = threading.Event()
    th = threading.Thread(target=lambda: [receiver._run_once(20) for _ in iter(stop.is_set, True)], daemon=True)
    th.start()

    def register(store: Path) -> None:
        c = sqlite3.connect(reg)
        leaf = str(store.parent / "x")
        c.execute("INSERT INTO datasets VALUES (?, ?, ?)", (leaf, leaf, str(store)))
        c.commit()
        c.close()

    def client(**kw):
        kw.setdefault("settle_s", 0)
        kw.setdefault("reply_timeout_s", 10)
        return bc.BackfillClient([receiver.endpoint], [(world.local, str(world.remote))], **kw)

    yield receiver, register, client
    stop.set()
    th.join(timeout=2)
    receiver.close()


def test_the_same_data_in_another_folder_is_not_sent_again(world, indexed):
    """E.g. a run that reached Argus by Globus under DataUpload/: never a second copy."""
    _receiver, register, client = indexed
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", seed=9)
    elsewhere = make_store(world.remote / "globus" / "exp" / "cell_001_GFP_488.ome.zarr", seed=9)
    register(elsewhere)
    c = client()
    try:
        res = leaf_report(c.run_once(dry_run=True), src.name)
        assert res == {"result": "already on Argus", "at": str(elsewhere), "has": 3, "of": 3}
        res = leaf_report(c.run_once(), src.name)
        assert res["result"] == "already on Argus"
    finally:
        c.close()
    assert not remote_store(world).exists()


def test_a_different_run_with_the_same_name_elsewhere_is_sent(world, indexed):
    """Cell_001 recurs across experiments: the name alone proves nothing."""
    _receiver, register, client = indexed
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", seed=9)
    register(make_store(world.remote / "older" / "cell_001_GFP_488.ome.zarr", seed=10))
    c = client()
    try:
        assert leaf_report(c.run_once(), src.name)["result"] == "complete"
    finally:
        c.close()
    assert arrays_equal(src, remote_store(world))


def test_a_partial_copy_elsewhere_is_reported_not_duplicated(world, indexed):
    _receiver, register, client = indexed
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", seed=9)
    register(make_store(world.remote / "globus" / "exp" / "cell_001_GFP_488.ome.zarr", written=(0,), seed=9))
    c = client()
    try:
        res = leaf_report(c.run_once(), src.name)
    finally:
        c.close()
    assert res["result"] == "partial copy elsewhere" and res["has"] == 1 and res["of"] == 3
    assert not remote_store(world).exists()


def test_finding_the_pixel_array_never_walks_its_chunks(tmp_path, monkeypatch):
    """Rig 2026-10-24: a whole-drive dry run sat for minutes: every store's chunk files
    (tens of thousands per run) were walked just to find its pixel array."""
    store = make_store(tmp_path / "cell_GFP_488.ome.zarr", declared_t=3, written=(0, 1, 2))
    walked = []
    real_walk = os.walk

    def counting_walk(top, *a, **kw):
        for dirpath, dirnames, filenames in real_walk(top, *a, **kw):
            walked.append(Path(dirpath))
            yield dirpath, dirnames, filenames

    monkeypatch.setattr(bfill.os, "walk", counting_walk)
    assert bfill.main_array_rel(store) == "p0"
    assert all(len(p.relative_to(store).parts) <= 1 for p in walked)  # root and array dirs only
