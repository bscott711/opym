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


def test_a_different_run_under_a_name_argus_holds_is_sent_renamed_and_the_old_one_untouched(world):
    """2026-10-25: the acquisition software reused names (20260928 Cell_005, 20260924
    Cell_001/007); a store whose volumes have another size is certainly another run."""
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    other = make_store(world.remote / "exp" / "cell_001_GFP_488.ome.zarr", shape_yx=(NY + 1, NX), seed=5)
    snapshot = {p: p.read_bytes() for p in other.rglob("*") if p.is_file()}
    res = leaf_report(world.client().run_once(), "cell_001_GFP_488.ome.zarr")
    assert res["result"] == "complete" and res["as"] == "cell_001_pc2_GFP_488.ome.zarr"
    assert {p: p.read_bytes() for p in other.rglob("*") if p.is_file()} == snapshot
    renamed = world.remote / "exp" / "cell_001_pc2_GFP_488.ome.zarr"
    assert arrays_equal(src, renamed)
    assert bfill.read_marker(renamed)["renamed_from"] == "cell_001_GFP_488.ome.zarr"
    assert not list((world.remote / "exp").glob(".superseded*"))

    again = leaf_report(world.client().run_once(), "cell_001_GFP_488.ome.zarr")
    assert again["result"] == "already on Argus" and again["as"] == "cell_001_pc2_GFP_488.ome.zarr"
    assert not (world.remote / "exp" / "cell_001_pc3_GFP_488.ome.zarr").exists()


def test_every_channel_of_a_renamed_run_gets_the_same_number(world):
    for ch in ("GFP_488", "mScarlet_561"):
        make_store(world.local / "exp" / f"cell_001_{ch}.ome.zarr")
        make_store(world.remote / "exp" / f"cell_001_{ch}.ome.zarr", shape_yx=(NY + 1, NX), seed=5)
    make_store(world.remote / "exp" / "cell_001_pc2_mScarlet_561.ome.zarr", shape_yx=(NY + 2, NX), seed=6)
    (rep,) = world.client().run_once()
    assert {st["as"] for st in rep["stores"].values()} == {"cell_001_pc3_GFP_488.ome.zarr",
                                                           "cell_001_pc3_mScarlet_561.ome.zarr"}


def test_a_conflict_in_the_declared_length_only_is_not_renamed(world):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", declared_t=3)
    make_store(world.remote / "exp" / "cell_001_GFP_488.ome.zarr", declared_t=5, seed=5)
    res = leaf_report(world.client().run_once(), "cell_001_GFP_488.ome.zarr")
    assert res["result"] == bfill.CONFLICT and "as" not in res
    assert not (world.remote / "exp" / "cell_001_pc2_GFP_488.ome.zarr").exists()


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


def test_the_same_zattrs_with_other_spacing_is_not_a_conflict(world):
    """2026-10-25: 10 stores on Argus held the acquisition's `.zattrs` written compact; the PC's is
    indented. Same document, same pixels: the copy must be verified, not refused."""
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", seed=7)
    there = world.remote / "exp" / "cell_001_GFP_488.ome.zarr"
    shutil.copytree(src, there)
    attrs = json.loads((there / ".zattrs").read_text())
    (there / ".zattrs").write_text(json.dumps(attrs, separators=(",", ":")))
    assert (there / ".zattrs").read_bytes() != (src / ".zattrs").read_bytes()
    res = leaf_report(world.client().run_once(), src.name)
    assert res["result"] == "complete", res
    assert bfill.read_marker(there) is not None


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
        assert res["result"] == "would verify" and res["at"] == str(elsewhere)
        res = leaf_report(c.run_once(), src.name)
        assert res["result"] == "complete" and res["checked_only"] and res["at"] == str(elsewhere)
        assert bfill.read_marker(elsewhere)["verified"] == "readback-sha256"   # checked, then marked
        res = leaf_report(c.run_once(), src.name)
        assert res["result"] == "already on Argus"                           # and not checked twice
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


def test_a_partial_copy_elsewhere_is_completed_where_it_is(world, indexed):
    """Never a second copy: the run's other copy on Argus is filled in, in its own folder."""
    _receiver, register, client = indexed
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", seed=9)
    there = make_store(world.remote / "globus" / "exp" / "cell_001_GFP_488.ome.zarr", written=(0,), seed=9)
    register(there)
    c = client()
    try:
        dry = leaf_report(c.run_once(dry_run=True), src.name)
        assert dry == {"result": "would complete elsewhere", "at": str(there), "missing_timepoints": 2, "of": 3}
        res = leaf_report(c.run_once(), src.name)
    finally:
        c.close()
    assert res["result"] == "complete" and res["at"] == str(there) and res["sent_timepoints"] == 2
    assert arrays_equal(src, there) and bfill.read_marker(there)
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


def test_a_conflict_says_what_differs(world):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", declared_t=3)
    make_store(remote_store(world), declared_t=4, written=(0,))
    res = leaf_report(world.client().run_once(), "cell_001_GFP_488.ome.zarr")
    assert res["result"] == bfill.CONFLICT
    assert "shape on Argus [4, 4, 5, 6], on the PC [3, 4, 5, 6]" in res["reason"]
    assert "only the declared number of timepoints differs" in res["reason"]


def test_a_second_inventory_of_an_unchanged_store_is_not_recounted(world, monkeypatch):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    _as_stream_copy(make_store(remote_store(world), written=(0, 1), seed=0))
    c = world.client()
    c.run_once(dry_run=True)
    counts = []
    real = bfill.chunk_count_for_t
    monkeypatch.setattr(bfill, "chunk_count_for_t", lambda *a: counts.append(a) or real(*a))
    res = leaf_report(c.run_once(dry_run=True), "cell_001_GFP_488.ome.zarr")
    assert res["missing_timepoints"] == 1 and counts == []


def _single_volume(path: Path, seed: int) -> Path:
    """How the PC saves a one-timepoint run: a 3-D [z, y, x] array."""
    rng = np.random.default_rng(seed)
    root = zarr.open_group(str(path), mode="w")
    arr = root.create_dataset("p0", shape=(NZ, NY, NX), chunks=(1, NY, NX), dtype="<u2", dimension_separator="/")
    arr[...] = rng.integers(1, 4000, size=(NZ, NY, NX), dtype=np.uint16)
    return path


def _stream_single_volume(path: Path, like: Path) -> Path:
    """The live stream's copy of it: [1, z, y, x], stream-tagged."""
    vol = zarr.open_array(str(like / "p0"), mode="r")[...]
    root = zarr.open_group(str(path), mode="w")
    arr = root.create_dataset("p0", shape=(1, NZ, NY, NX), chunks=(1, 1, NY, NX), dtype="<u2", dimension_separator="/")
    arr[0] = vol
    return _as_stream_copy(path)


def test_a_single_volume_the_stream_brought_in_is_already_on_argus(world):
    """52 of 60 'conflicts' in the rig's first whole-drive dry run (2026-10-24)."""
    src = _single_volume(world.local / "exp" / "test_001_GFP_488.ome.zarr", seed=3)
    there = _stream_single_volume(world.remote / "exp" / "test_001_GFP_488.ome.zarr", like=src)
    snapshot = {p: p.read_bytes() for p in there.rglob("*") if p.is_file()}
    c = world.client()
    assert leaf_report(c.run_once(dry_run=True), src.name)["result"] == "would verify"
    res = leaf_report(c.run_once(), src.name)
    assert res["result"] == "complete" and res["checked_only"]
    after = {p: p.read_bytes() for p in there.rglob("*") if p.is_file() and p.name != bfill.MARKER_NAME}
    assert after == snapshot                                  # only the pixels were checked
    assert bfill.read_marker(there)["written_t"] == [0]
    assert leaf_report(c.run_once(), src.name)["result"] == "already complete"


def test_a_single_volume_with_other_data_in_the_streams_copy_is_a_conflict(world):
    src = _single_volume(world.local / "exp" / "test_001_GFP_488.ome.zarr", seed=3)
    other = _single_volume(world.local / "other" / "x_GFP_488.ome.zarr", seed=4)
    _stream_single_volume(world.remote / "exp" / "test_001_GFP_488.ome.zarr", like=other)
    res = leaf_report([r for r in world.client().run_once(dry_run=True) if r["leaf"].endswith("exp")],
                      src.name)
    assert res["result"] == bfill.CONFLICT and "different data" in res["reason"]


def test_probes_read_one_plane_not_the_whole_volume(world, indexed, monkeypatch):
    """The whole-drive dry run read every candidate's whole volume on both ends."""
    _receiver, register, client = indexed
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr", seed=9)
    register(make_store(world.remote / "globus" / "exp" / "cell_001_GFP_488.ome.zarr", seed=9))
    digested = []
    real = bfill.volume_digest
    monkeypatch.setattr(bfill, "volume_digest", lambda v: digested.append(np.asarray(v).shape) or real(v))
    c = client()
    try:
        assert leaf_report(c.run_once(dry_run=True), src.name)["result"] == "would verify"
    finally:
        c.close()
    assert digested and all(len(shape) == 2 for shape in digested)  # planes only, both ends


# ---------------------------------------------------------------- the run record (sidecar)
# 2026-10-24: a whole-drive run stalled at 22:54 and never finished; neither end kept a
# record of what it meant to send or how far it got.
from opym.stream import backfill_runs, bfill_run  # noqa: E402


def _argus_runs(w) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((w.remote / ".opym_backfill" / "runs").glob("*.json"))]


def test_the_plan_is_recorded_before_the_first_volume_and_each_result_after(world, tmp_path, monkeypatch):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    make_store(world.local / "exp" / "cell_002_GFP_488.ome.zarr", seed=2)
    runs = tmp_path / "runs"
    seen_at_first_volume = []
    real_volume = world.receiver._volume

    def spy(header, payload):
        if not seen_at_first_volume:
            seen_at_first_volume.append([json.loads(p.read_text()) for p in runs.glob("*.json")])
        return real_volume(header, payload)

    monkeypatch.setattr(world.receiver, "_volume", spy)
    world.client(runs_dir=runs).run_once()

    (local_at_start,) = seen_at_first_volume[0]
    assert local_at_start["state"] == "running"
    assert {e["name"] for e in local_at_start["plan"]} == {"cell_001_GFP_488.ome.zarr", "cell_002_GFP_488.ome.zarr"}
    assert all(e["action"] == "send" and e["send_t"] == 3 for e in local_at_start["plan"])

    (local,) = [json.loads(p.read_text()) for p in runs.glob("*.json")]
    (argus,) = _argus_runs(world)
    for doc in (local, argus):
        assert doc["state"] == "complete" and doc["counts"] == {"complete": 2}
    key = f"{world.remote}/exp/cell_001_GFP_488.ome.zarr"
    assert argus["stores"][key]["result"] == "complete"
    assert argus["stores"][key]["argus"]["state"] == "verified"   # what the receiver saw itself
    assert bfill.read_marker(remote_store(world))["run_id"] == argus["run_id"]


def _dies_at_t1(c):
    real = c._send_slab

    def dies(key, header, payload, inflight, attempts):
        if key == (1, 0):
            c.runlog.heartbeat(f"{header['leaf']}/{header['store']}", 1, force=True)
            raise KeyboardInterrupt            # the PC's window closed, the PC slept, ...
        real(key, header, payload, inflight, attempts)

    return dies


def test_a_run_that_fails_says_so(world, tmp_path, monkeypatch):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    c = world.client(runs_dir=tmp_path / "runs", slab_bytes=NY * NX * 2)
    monkeypatch.setattr(c, "_send_slab", _dies_at_t1(c))
    with pytest.raises(KeyboardInterrupt):
        c.run_once()
    (doc,) = _argus_runs(world)
    assert doc["state"] == "failed" and "KeyboardInterrupt" in doc["error"]


def test_a_run_that_stops_reporting_is_marked_interrupted_where_it_was(world, tmp_path, monkeypatch):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    c = world.client(runs_dir=tmp_path / "runs", slab_bytes=NY * NX * 2)
    monkeypatch.setattr(c, "_send_slab", _dies_at_t1(c))
    real_send_run = c._send_run
    monkeypatch.setattr(c, "_send_run", lambda h: None if h["op"] == "end" else real_send_run(h))  # killed
    with pytest.raises(KeyboardInterrupt):
        c.run_once()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not (_argus_runs(world) and _argus_runs(world)[0].get("current")):
        time.sleep(0.05)                       # the heartbeat is fire-and-forget
    (doc,) = _argus_runs(world)
    assert doc["state"] == "running"

    monkeypatch.setattr(backfill_runs, "STALE_S", 0.0)
    assert world.receiver.runs.sweep(force=True) == 1
    (marked,) = _argus_runs(world)
    assert marked["state"] == "interrupted"
    assert marked["interrupted_during"]["key"].endswith("cell_001_GFP_488.ome.zarr")
    assert marked["interrupted_during"]["t"] == 1
    assert world.receiver.runs.sweep(force=True) == 0      # once


def test_an_older_receiver_without_run_records_still_gets_the_data(world, tmp_path, monkeypatch):
    from opym.stream import backfill_receiver

    monkeypatch.setattr(backfill_receiver.bfill, "FEATURES", ["blosc", "links", "plane_probe"])
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    res = leaf_report(world.client(runs_dir=tmp_path / "runs").run_once(), "cell_001_GFP_488.ome.zarr")
    assert res["result"] == "complete"
    assert _argus_runs(world) == []                                # nothing sent it can't read
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1      # the PC keeps its own


def test_resume_skips_what_the_last_run_finished(world, tmp_path):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    runs = tmp_path / "runs"
    world.client(runs_dir=runs).run_once()
    make_store(world.local / "exp" / "cell_002_GFP_488.ome.zarr", seed=2)
    asked = []
    real = world.receiver._inventory
    world.receiver._inventory = lambda h: (asked.extend(s["name"] for s in h.get("stores", [])), real(h))[1]

    prev = bfill_run.load_run(runs)
    (rep,) = world.client(runs_dir=runs).run_once(resume=prev)
    assert rep["stores"]["cell_001_GFP_488.ome.zarr"] == {"result": "already complete", "reason": "previous run"}
    assert rep["stores"]["cell_002_GFP_488.ome.zarr"]["result"] == "complete"
    assert "cell_001_GFP_488.ome.zarr" not in asked


def test_a_dry_run_records_its_plan_and_marks_itself(world, tmp_path):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    world.client(runs_dir=tmp_path / "runs").run_once(dry_run=True)
    (doc,) = _argus_runs(world)
    assert doc["dry_run"] is True and doc["state"] == "complete"
    assert [e["action"] for e in doc["plan"]] == ["send"]
    assert bfill_run.load_run(tmp_path / "runs") is None           # resume ignores dry runs


# ---------------------------------------------------------------- a faithful copy
# 2026-10-25: "Your number one job is to make sure we have a faithful copy of the S drive
# on Argus." Copies the stream or Globus made had only ever been counted, never compared.

def test_a_copy_the_stream_made_whole_is_checked_against_the_pc_and_then_marked(world):
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    there = _as_stream_copy(make_store(remote_store(world)))
    before = world.receiver.stats["volumes"]
    res = leaf_report(world.client().run_once(), "cell_001_GFP_488.ome.zarr")
    assert res["result"] == "complete" and res["checked_only"] and res["sent_timepoints"] == 0
    assert world.receiver.stats["volumes"] == before          # nothing was sent, only checked
    assert bfill.read_marker(there)["written_t"] == [0, 1, 2]
    assert arrays_equal(src, there)


def test_a_copy_whose_pixels_differ_from_the_pc_is_reported_and_left_alone(world):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    there = _as_stream_copy(make_store(remote_store(world)))
    arr = zarr.open_array(str(there / "p0"), mode="r+")
    arr[1, 2] = arr[1, 2] + 1                                  # one plane silently wrong on Argus
    chunks = {p: p.read_bytes() for p in (there / "p0").rglob("*") if p.is_file()}
    res = leaf_report(world.client().run_once(), "cell_001_GFP_488.ome.zarr")
    assert res["result"] == "refused" and "differ after readback" in res["reason"]
    assert bfill.read_marker(there) is None                    # never counted as a faithful copy
    assert {p: p.read_bytes() for p in (there / "p0").rglob("*") if p.is_file()} == chunks  # untouched


def test_no_verify_trusts_whole_copies_as_before(world):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    _as_stream_copy(make_store(remote_store(world)))
    res = leaf_report(world.client(verify=False).run_once(), "cell_001_GFP_488.ome.zarr")
    assert res["result"] == "already on Argus"
    assert bfill.read_marker(remote_store(world)) is None



def test_a_store_cut_off_before_its_pixel_array_metadata_is_finished(world):
    """2026-10-25: a run stopped after a store's `z` array landed but before `p0/.zarray`.
    Its largest array was then the coordinate array, so it read as a conflict forever."""
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    dst = remote_store(world)
    shutil.copytree(src, dst, ignore=lambda d, names: [n for n in names if n in ("p0", ".zattrs")])
    lock = dst / bfill.LOCK_NAME
    lock.write_text("another-pc")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    assert bfill.main_array_rel(dst) == "z"                     # what made it look like a conflict
    res = leaf_report(world.client().run_once(), "cell_001_GFP_488.ome.zarr")
    assert res["result"] == "complete", res
    assert arrays_equal(src, dst) and bfill.read_marker(dst)["written_t"] == [0, 1, 2]


def test_a_store_without_our_lock_whose_largest_array_is_smaller_is_still_a_conflict(world):
    make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    dst = remote_store(world)
    zarr.open_group(str(dst), mode="w").create_dataset("z", data=np.arange(NZ) * 0.5)
    res = leaf_report(world.client().run_once(), "cell_001_GFP_488.ome.zarr")
    assert res["result"] == bfill.CONFLICT
    assert not (dst / "p0").exists()


def test_a_late_copy_of_a_metadata_request_does_not_unmark_a_finished_store(world):
    """A sender resends a request on another tunnel when its tunnel goes silent; the
    first copy can arrive after the store was committed. It must not run again: a
    metadata request removes the completion marker."""
    src = make_store(world.local / "exp" / "cell_001_GFP_488.ome.zarr")
    c = world.client()
    sent = []
    real = c.link.send

    def keep(parts, *a, **k):
        if parts[0] == bfill.BF_META:
            sent.append(list(parts))
        return real(parts, *a, **k)

    c.link.send = keep
    assert leaf_report(c.run_once(), "cell_001_GFP_488.ome.zarr")["result"] == "complete"
    assert bfill.read_marker(remote_store(world))
    c.link.send = real
    c.link.send(sent[0])                                   # the stale copy arrives now
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        got = c.link.recv_any(200)
        if got is not None:
            break
    assert bfill.read_marker(remote_store(world)), "the finished store kept its marker"
    assert arrays_equal(src, remote_store(world))


def test_an_untagged_4d_copy_of_a_single_volume_is_checked_pixel_for_pixel(world):
    """12 'conflicts' in the rig's 2026-10-25 dry run: single volumes the stream wrote
    4-D on 23-24 Sept, before it tagged its stores."""
    src = _single_volume(world.local / "exp" / "test_001_GFP_488.ome.zarr", seed=3)
    there = _stream_single_volume(world.remote / "exp" / "test_001_GFP_488.ome.zarr", like=src)
    attrs = json.loads((there / ".zattrs").read_text())
    attrs.pop("opym", None)
    (there / ".zattrs").write_text(json.dumps(attrs))
    snapshot = {p: p.read_bytes() for p in there.rglob("*") if p.is_file()}
    res = leaf_report(world.client().run_once(), src.name)
    assert res["result"] == "complete" and res["checked_only"], res
    after = {p: p.read_bytes() for p in there.rglob("*") if p.is_file() and p.name != bfill.MARKER_NAME}
    assert after == snapshot
    assert bfill.read_marker(there)["written_t"] == [0]


def test_an_untagged_4d_copy_with_other_pixels_is_still_a_conflict(world):
    src = _single_volume(world.local / "exp" / "test_001_GFP_488.ome.zarr", seed=3)
    other = _single_volume(world.local.parent / "other" / "test_001_GFP_488.ome.zarr", seed=4)
    there = _stream_single_volume(world.remote / "exp" / "test_001_GFP_488.ome.zarr", like=other)
    attrs = json.loads((there / ".zattrs").read_text())
    attrs.pop("opym", None)
    (there / ".zattrs").write_text(json.dumps(attrs))
    res = leaf_report(world.client().run_once(), src.name)
    assert res["result"] == bfill.CONFLICT
    assert bfill.read_marker(there) is None


def test_a_cut_off_3d_volume_is_finished_not_taken_for_complete(world):
    """2026-10-25: a 3-D copy holding 4 of 301 planes counted as held, so nothing was sent and
    the readback refused it on every run."""
    src = _single_volume(world.local / "exp" / "test_001_GFP_488.ome.zarr", seed=5)
    there = world.remote / "exp" / "test_001_GFP_488.ome.zarr"
    shutil.copytree(src, there)
    for z in range(1, NZ):                                   # only plane 0 made it
        shutil.rmtree(there / "p0" / str(z))
    (there / "p0" / "2").mkdir()                              # another tool's empty directory
    assert not arrays_equal(src, there)
    c = world.client()
    assert leaf_report(c.run_once(dry_run=True), src.name)["result"] == "would send"
    res = leaf_report(c.run_once(), src.name)
    assert res["result"] == "complete"
    assert arrays_equal(src, there)
    assert bfill.read_marker(there) is not None
