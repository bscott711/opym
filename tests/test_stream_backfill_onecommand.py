"""The one-command SSH backfill sender: drive-to-microscopy mapping, whole-drive scans, data that
is already on Argus, the scan cache, supervised parallel tunnels, and the background service.

The end-to-end test runs the real `main()` against a loopback receiver through a fake `ssh`
(a small TCP forwarder), so the tunnel supervisor, the round-robin over links and the status
files are all exercised without a network.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath, PureWindowsPath

import pytest
from test_stream_backfill import make_store, world  # noqa: F401  (the loopback receiver fixture)

from opym.stream import backfill_client as bc
from opym.stream import backfill_service as svc
from opym.stream import bfill
from opym.stream import tunnels as tn

GFP = "cell_001_GFP_488.ome.zarr"


# ---------------------------------------------------------------- mapping


@pytest.mark.parametrize(
    ("local", "expected"),
    [
        (r"S:\\", "/mmfs1/m"),
        (r"S:\20261023_LCK-NG_JB", "/mmfs1/m/20261023_LCK-NG_JB"),
        (r"S:\20261023_LCK-NG_JB\Cell_001", "/mmfs1/m/20261023_LCK-NG_JB/Cell_001"),
        (r"D:\acq\day1", "/mmfs1/m/acq/day1"),
        (r"\\scope\share\day1", "/mmfs1/m/day1"),
    ],
)
def test_a_windows_folder_maps_relative_to_its_drive_root(local, expected):
    _, remote = bc.map_for_root(PureWindowsPath(local), "/mmfs1/m/")
    assert remote == expected


def test_giving_the_drive_or_one_session_folder_lands_data_in_the_same_place():
    _, whole = bc.map_for_root(PureWindowsPath("S:\\"), "/mmfs1/m")
    _, one = bc.map_for_root(PureWindowsPath("S:\\20261023_Rab5_Akt_JB"), "/mmfs1/m")
    stores = PureWindowsPath("S:\\20261023_Rab5_Akt_JB\\Pak2_Stim_001_mNG_488nm.ome.zarr")
    via_whole = "/".join([whole, *stores.relative_to("S:\\").parts[:-1]])
    via_one = "/".join([one, *stores.relative_to("S:\\20261023_Rab5_Akt_JB").parts[:-1]])
    assert via_whole == via_one == "/mmfs1/m/20261023_Rab5_Akt_JB"


def test_a_posix_folder_without_a_drive_is_the_mirror_of_the_remote_root():
    assert bc.map_for_root(PurePosixPath("/data/acq"), "/mmfs1/m") == (PurePosixPath("/data/acq"), "/mmfs1/m")


def test_default_roots_come_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYM_BACKFILL_LOCAL_ROOTS", os.pathsep.join([str(tmp_path), str(tmp_path / "b")]))
    assert bc.default_local_roots() == [tmp_path, tmp_path / "b"]
    monkeypatch.setenv("OPYM_BACKFILL_REMOTE_ROOT", "/x/y/")
    assert bc.default_remote_root() == "/x/y"
    monkeypatch.delenv("OPYM_BACKFILL_REMOTE_ROOT")
    assert bc.default_remote_root() == "/mmfs1/scratch/jacks.local/microscopy"


def test_no_folder_given_and_no_default_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPYM_BACKFILL_LOCAL_ROOTS", raising=False)
    monkeypatch.setattr(bc, "default_local_roots", lambda: [])
    with pytest.raises(SystemExit, match="no folder to back up"):
        bc.resolve_maps(bc.build_parser().parse_args([]))


def test_a_missing_drive_is_a_clear_error(tmp_path):
    a = bc.build_parser().parse_args([str(tmp_path / "not_there")])
    with pytest.raises(SystemExit, match="is not a folder"):
        bc.resolve_maps(a)


# ---------------------------------------------------------------- scanning a whole drive


def test_a_whole_drive_scan_finds_every_session_and_skips_system_folders(tmp_path):
    for session in ("20231023_2xfyve-NG_AKTph_mscarlet", "20261023_LCK-NG_JB", "20261023_Rab5_Akt_JB"):
        make_store(tmp_path / session / GFP)
    make_store(tmp_path / "20261023_mem_2xfyve_aktph" / "Cell_001" / GFP)
    make_store(tmp_path / "$RECYCLE.BIN" / "S-1-5" / GFP)
    make_store(tmp_path / "System Volume Information" / GFP)
    make_store(tmp_path / ".hidden" / GFP)
    make_store(tmp_path / "d1" / "d2" / "d3" / "d4" / "d5" / "d6" / "d7" / GFP)  # deeper than 6 levels
    found = {lf.path.relative_to(tmp_path).as_posix() for lf in bc.scan_local_root(tmp_path)}
    assert found == {
        "20231023_2xfyve-NG_AKTph_mscarlet", "20261023_LCK-NG_JB", "20261023_Rab5_Akt_JB",
        "20261023_mem_2xfyve_aktph/Cell_001",
    }


def test_a_store_name_the_receiver_would_refuse_is_reported_not_ignored(tmp_path):
    make_store(tmp_path / "s" / GFP)
    (tmp_path / "s" / "cell (2)_GFP.ome.zarr").mkdir()
    (leaf,) = bc.scan_local_root(tmp_path)
    assert [s.name for s in leaf.stores] == [GFP] and leaf.invalid == ["cell (2)_GFP.ome.zarr"]


def test_the_pass_report_names_the_remote_folder_and_the_bad_store(world):  # noqa: F811
    make_store(world.local / "day1" / GFP)
    (world.local / "day1" / "bad name(1).ome.zarr").mkdir()
    (rep,) = world.client().run_once(dry_run=True)
    assert rep["remote"] == f"{world.remote}/day1" and rep["invalid"] == ["bad name(1).ome.zarr"]
    assert rep["stores"][GFP]["result"] == "would send"


# ---------------------------------------------------------------- data already on Argus


def test_a_store_already_on_argus_without_a_marker_is_verified_not_resent(world):  # noqa: F811
    src = make_store(world.local / "exp" / GFP)
    shutil.copytree(src, world.remote / "exp" / GFP)        # what a Globus transfer leaves: no marker
    (rep,) = world.client().run_once()
    info = rep["stores"][GFP]
    assert info["result"] == "complete" and info["sent_timepoints"] == 0 and info["sent_gb"] == 0
    assert bfill.read_marker(world.remote / "exp" / GFP)["verified"] == "readback-sha256"


def test_a_store_on_argus_with_the_right_shape_but_wrong_content_is_flagged_not_marked(world):  # noqa: F811
    src = make_store(world.local / "exp" / GFP)
    dst = world.remote / "exp" / GFP
    shutil.copytree(src, dst)
    import zarr

    arr = zarr.open_array(str(dst / "p0"), mode="r+")
    arr[1, 2] = arr[1, 2] + 1                                # same chunk files, different pixels
    (rep,) = world.client().run_once()
    info = rep["stores"][GFP]
    assert info["result"] == "failed" and "differ after readback" in info["reason"]
    assert bfill.read_marker(dst) is None


def test_one_refused_store_does_not_stop_its_siblings(world):  # noqa: F811
    make_store(world.local / "exp" / "a_GFP.ome.zarr", seed=1)
    make_store(world.local / "exp" / "b_RFP.ome.zarr", seed=2)
    bad = make_store(world.remote / "exp" / "a_GFP.ome.zarr", seed=1)
    (bad / ".zattrs").write_text(json.dumps({"multiscales": "something else"}))   # differs from the PC's
    (rep,) = world.client().run_once()
    assert rep["stores"]["a_GFP.ome.zarr"]["result"] == "failed"
    assert rep["stores"]["b_RFP.ome.zarr"]["result"] == "complete"
    assert bfill.read_marker(world.remote / "exp" / "b_RFP.ome.zarr")


# ---------------------------------------------------------------- the scan cache


def test_verified_stores_are_skipped_until_they_change(tmp_path):
    root = tmp_path / "drive"
    store = make_store(root / "day1" / GFP)
    maps = [(root, "/remote")]
    state = bc.ScanState(tmp_path / "state.json", maps)
    (leaf,) = bc.scan_local_root(root, known=state)
    state.record(leaf.stores[0])
    state.save()

    stats: dict = {}
    assert bc.scan_local_root(root, known=bc.ScanState(tmp_path / "state.json", maps), stats=stats) == []
    assert stats["unchanged"] == 1                           # remembered across a restart

    make_store(store, declared_t=3, written=(0, 1, 2, 2))    # rewritten in place: new mtimes
    import zarr

    zarr.open_array(str(store / "p0"), mode="r+")[2, 0] = 7
    os.utime(store / "p0" / "2", (time.time() + 5, time.time() + 5))
    assert len(bc.scan_local_root(root, known=bc.ScanState(tmp_path / "state.json", maps))) == 1


def test_a_store_that_gains_a_timepoint_is_picked_up_again(tmp_path):
    root = tmp_path / "drive"
    store = make_store(root / "day1" / GFP, declared_t=4, written=(0, 1))
    maps = [(root, "/remote")]
    state = bc.ScanState(tmp_path / "state.json", maps)
    (leaf,) = bc.scan_local_root(root, known=state)
    state.record(leaf.stores[0])
    assert bc.scan_local_root(root, known=state) == []
    import zarr

    zarr.open_array(str(store / "p0"), mode="r+")[2] = 5
    (again,) = bc.scan_local_root(root, known=state)
    assert again.stores[0].written_t == [0, 1, 2]


def test_a_changed_mapping_or_rescan_forgets_what_was_verified(tmp_path):
    root = tmp_path / "drive"
    make_store(root / "day1" / GFP)
    state = bc.ScanState(tmp_path / "state.json", [(root, "/remote")])
    (leaf,) = bc.scan_local_root(root, known=state)
    state.record(leaf.stores[0])
    state.save()
    assert len(bc.scan_local_root(root, known=bc.ScanState(tmp_path / "state.json", [(root, "/elsewhere")]))) == 1
    assert len(bc.scan_local_root(root, known=bc.ScanState(tmp_path / "state.json", [(root, "/remote")],
                                                           fresh=True))) == 1


def test_a_second_pass_over_a_synced_drive_sends_nothing_and_skips_the_scan(world, tmp_path):  # noqa: F811
    make_store(world.local / "day1" / GFP)
    state = bc.ScanState(tmp_path / "st.json", [(world.local, str(world.remote))])
    c = world.client(state=state)
    (rep,) = c.run_once()
    assert rep["stores"][GFP]["result"] == "complete"
    assert c.run_once() == []                                # nothing changed: not even loaded
    assert "1 unchanged since last verified" in c.last_pass["summary"]


# ---------------------------------------------------------------- the prefetch pipeline


def test_prefetch_keeps_order_and_reports_the_end():
    p = bc._Prefetch(iter(range(20)), depth=3)
    assert list(p) == list(range(20))


def test_prefetch_raises_the_generators_error_in_the_consumer():
    def gen():
        yield 1
        raise ValueError("boom")

    p = bc._Prefetch(gen(), depth=2)
    assert next(p) == 1
    with pytest.raises(ValueError, match="boom"):
        next(p)


def test_closing_a_prefetch_releases_a_producer_blocked_on_a_full_queue():
    p = bc._Prefetch(iter(range(1000)), depth=1)
    time.sleep(0.2)
    p.close()
    p._thread.join(timeout=3)
    assert not p._thread.is_alive()


def test_poll_returns_empty_while_the_next_item_is_not_ready():
    gate = threading.Event()

    def gen():
        gate.wait(5)
        yield "x"

    p = bc._Prefetch(gen(), depth=1)
    assert p.poll(0.05) is bc._Prefetch.EMPTY
    gate.set()
    assert p.poll(2) == "x"


# ---------------------------------------------------------------- tunnels


_next_start = [20000]


def _free_ports(n: int) -> int:
    """First of `n` consecutive free local ports (never the same range twice in one run)."""
    for base in range(_next_start[0], 60000, 7):
        socks = []
        try:
            for k in range(n):
                s = socket.socket()
                s.bind(("127.0.0.1", base + k))
                socks.append(s)
            _next_start[0] = base + n + 7
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no free ports")


def test_the_default_tunnel_ports_stay_clear_of_the_live_streams():
    live = set(range(5555, 5555 + 8))                        # 8 links is the most the live stream is tuned for
    backfill = set(range(tn.DEFAULT_LOCAL_PORT, tn.DEFAULT_LOCAL_PORT + tn.DEFAULT_LINKS))
    assert not live & backfill and tn.DEFAULT_REMOTE_PORT == 5556
    assert svc.DEFAULT_LOCK_PORT not in live | backfill


def test_each_link_is_its_own_ssh_process_with_the_live_streams_options():
    ts = tn.TunnelSet("Argus", 5656, 5556, 4, ssh_args=["-i", "key"])
    cmds = ts.commands()
    assert len(cmds) == 4 and ts.endpoints == [f"tcp://127.0.0.1:{5656 + k}" for k in range(4)]
    for k, cmd in enumerate(cmds):
        assert cmd[0] == "ssh" and cmd[-1] == "Argus" and "-N" in cmd
        assert cmd[cmd.index("-L") + 1] == f"127.0.0.1:{5656 + k}:127.0.0.1:5556"
        joined = " ".join(cmd)
        for opt in ("BatchMode=yes", "ExitOnForwardFailure=yes", "ServerAliveInterval=5", "Ciphers=^aes128-gcm"):
            assert opt in joined
        assert cmd.count("-L") == 1 and "-i" in cmd


def test_a_port_someone_else_already_serves_is_reused_not_fought_over(tmp_path):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    ts = tn.TunnelSet("Argus", port, 5556, 1, ssh_exe=str(tmp_path / "must-not-run"))
    ts.start()
    try:
        assert ts.wait_ready(5)
        time.sleep(0.3)
        t = ts.tunnels[0]
        assert t.reused and t.proc is None
        assert ts.status() == [{"port": port, "up": True, "reused": True, "restarts": 0, "error": ""}]
    finally:
        ts.stop()
        srv.close()


def test_a_tunnel_that_dies_is_restarted_with_its_error_kept(tmp_path):
    fake = tmp_path / "ssh"
    fake.write_text("#!/bin/sh\necho 'connection refused' >&2\nexit 3\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    ts = tn.TunnelSet("Argus", _free_ports(1), 5556, 1, ssh_exe=str(fake))
    ts.start()
    try:
        deadline = time.time() + 6
        while ts.tunnels[0].restarts < 1 and time.time() < deadline:
            time.sleep(0.1)
        assert ts.tunnels[0].restarts >= 1
        assert "connection refused" in ts.tunnels[0].last_error
    finally:
        ts.stop()


# ---------------------------------------------------------------- one instance, status, background task


def test_a_second_sender_on_the_same_pc_cannot_start(tmp_path, monkeypatch):
    monkeypatch.setenv("OPYM_BACKFILL_STATE_DIR", str(tmp_path))
    port = _free_ports(1)
    with svc.SingleInstance(port):
        with pytest.raises(svc.AlreadyRunning, match="already running"), svc.SingleInstance(port):
            pass
    with svc.SingleInstance(port):                           # released with the first one
        pass


def test_status_says_running_while_fresh_and_not_running_when_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("OPYM_BACKFILL_STATE_DIR", str(tmp_path))
    assert "never started" in svc.format_status()
    svc.write_status({"activity": "S:\\x\\a.ome.zarr", "sent_gb": 1.5,
                      "tunnels": [{"port": 5656, "up": True, "reused": False, "restarts": 2, "error": ""},
                                  {"port": 5657, "up": False, "reused": False, "restarts": 0, "error": "refused"}],
                      "last_pass": {"finished_at": time.time(), "seconds": 12, "summary": "3 verified"}})
    text = svc.format_status()
    assert "RUNNING" in text and "tunnel 5656: up" in text and "tunnel 5657: DOWN" in text
    assert "3 verified" in text and "1.50 GB" in text
    st = svc.read_status()
    st["updated_at"] -= 600
    (tmp_path / "status.json").write_text(json.dumps(st))
    assert "not running" in svc.format_status()


def test_the_logon_task_xml_is_valid_hidden_low_priority_and_restarts_on_failure():
    xml = svc.task_xml(["--tunnels", "6", "--remote-root", "/a&b/<c>"], python=r"C:\Py & Co\pythonw.exe",
                       user=r"SCOPE\opm", workdir=r"C:\Users\opm")
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    root = ET.fromstring(xml.replace('encoding="UTF-16"', ""))   # ET refuses a str with a declared encoding
    assert root.find(".//t:Command", ns).text == r"C:\Py & Co\pythonw.exe"
    args = root.find(".//t:Arguments", ns).text
    assert args.startswith("-m opym.stream.backfill_client --tunnels 6") and "/a&b/<c>" in args
    assert root.find(".//t:LogonTrigger/t:UserId", ns).text == r"SCOPE\opm"
    assert root.find(".//t:Settings/t:Priority", ns).text == "7"
    assert root.find(".//t:Settings/t:ExecutionTimeLimit", ns).text == "PT0S"
    assert root.find(".//t:Settings/t:MultipleInstancesPolicy", ns).text == "IgnoreNew"
    assert root.find(".//t:Settings/t:RestartOnFailure/t:Count", ns).text == "999"
    assert root.find(".//t:Principal/t:RunLevel", ns).text == "LeastPrivilege"


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows refusal")
@pytest.mark.parametrize("flag", ["--install", "--uninstall"])
def test_installing_the_task_is_refused_off_windows(flag):
    with pytest.raises(SystemExit, match="only available on the Windows"):
        bc.main([flag])


# ---------------------------------------------------------------- the whole thing, one command

FAKE_SSH = """#!{python}
import os, socket, sys, threading
args = sys.argv[1:]
bind_host, lport, rhost, rport = args[args.index("-L") + 1].split(":")
log = os.environ["FAKE_SSH_LOG"]
srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((bind_host, int(lport)))
srv.listen(16)

def pump(a, b, count):
    try:
        while True:
            data = a.recv(1 << 16)
            if not data:
                break
            if count:
                with open(log, "a") as fh:
                    fh.write(f"{{lport}} {{len(data)}}\\n")
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

while True:
    c, _ = srv.accept()
    r = socket.create_connection((rhost, int(rport)))
    threading.Thread(target=pump, args=(c, r, True), daemon=True).start()
    threading.Thread(target=pump, args=(r, c, False), daemon=True).start()
"""


@pytest.fixture
def onecommand(world, tmp_path, monkeypatch):  # noqa: F811
    fake = tmp_path / "fake-ssh"
    fake.write_text(FAKE_SSH.format(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    sshlog = tmp_path / "ssh.log"
    state = tmp_path / "state"
    monkeypatch.setenv("FAKE_SSH_LOG", str(sshlog))
    monkeypatch.setenv("OPYM_BACKFILL_STATE_DIR", str(state))
    monkeypatch.setenv("OPYM_BACKFILL_LOCK_PORT", str(_free_ports(1)))
    base = _free_ports(2)
    remote_port = int(world.receiver.endpoint.rsplit(":", 1)[1])

    def run(*extra, folder=None):
        argv = ["--once", "--tunnels", "2", "--base-port", str(base), "--remote-port", str(remote_port),
                "--ssh-exe", str(fake), "--normal-priority", "--settle", "0",
                "--remote-root", str(world.remote), *extra, str(folder or world.local)]
        return bc.main(argv)

    run.sshlog, run.state = sshlog, state
    return run


def test_one_command_copies_a_drive_through_parallel_tunnels(world, onecommand):  # noqa: F811
    for i, session in enumerate(("20261023_LCK-NG_JB", "20261023_Rab5_Akt_JB", "20231023_2xfyve-NG_AKTph_mscarlet")):
        make_store(world.local / session / GFP, seed=i)
        make_store(world.local / session / "cell_001_RFP_561.ome.zarr", seed=10 + i)
    assert onecommand() == 0
    for session in ("20261023_LCK-NG_JB", "20261023_Rab5_Akt_JB", "20231023_2xfyve-NG_AKTph_mscarlet"):
        for name in (GFP, "cell_001_RFP_561.ome.zarr"):
            assert bfill.read_marker(world.remote / session / name)["verified"] == "readback-sha256"
    per_port: dict[str, int] = {}
    for line in onecommand.sshlog.read_text().splitlines():
        port, n = line.split()
        per_port[port] = per_port.get(port, 0) + int(n)
    assert len(per_port) == 2 and min(per_port.values()) > 1000     # both tunnels carried the slabs
    st = json.loads((onecommand.state / "status.json").read_text())
    assert st["last_pass"]["summary"].startswith("6 complete") and [t["up"] for t in st["tunnels"]] == [True, True]
    assert (onecommand.state / "scan-state.json").exists() and (onecommand.state / "backfill-send.log").exists()


def test_running_it_again_finds_everything_verified_and_unchanged(world, onecommand):  # noqa: F811
    make_store(world.local / "20261023_LCK-NG_JB" / GFP)
    assert onecommand() == 0
    assert onecommand() == 0
    log = (onecommand.state / "backfill-send.log").read_text()
    assert "1 unchanged since last verified" in log


def test_dry_run_through_the_one_command_sends_and_records_nothing(world, onecommand):  # noqa: F811
    make_store(world.local / "20261023_LCK-NG_JB" / GFP)
    assert onecommand("--dry-run") == 0
    assert not (world.remote / "20261023_LCK-NG_JB").exists()
    assert not (onecommand.state / "scan-state.json").exists()
    assert "1 to send" in (onecommand.state / "backfill-send.log").read_text()


def test_a_second_sender_started_while_one_runs_is_refused_but_a_dry_run_is_not(world, onecommand):  # noqa: F811
    make_store(world.local / "d" / GFP)
    with svc.SingleInstance():
        assert onecommand() == 2
        assert onecommand("--dry-run") == 0
