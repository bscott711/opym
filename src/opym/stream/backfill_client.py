"""`opym-backfill-send`: the acquisition-PC side of the SSH backfill.

Scans local acquisition folders for finished `*.ome.zarr` stores, asks the backfill receiver
(`opym-receive-backfill`, reached through an SSH tunnel) what it already holds, and sends only
what is missing: each store's small files verbatim, then pixel volumes as slabs. It finishes a
store with a per-volume SHA-256 manifest the server verifies by reading every volume back.

It never touches the live stream: it talks to a different port, pauses while the server reports
a live acquisition (and while a `--pause-file` exists), and skips stores written to recently.

    ssh -N -L 5556:127.0.0.1:5556 <user>@argus.sdstate.edu          # one tunnel
    opym-backfill-send --map D:\\acq=/mmfs1/scratch/SDSMT.LOCAL/bscott/DataUpload --watch 600
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zarr
import zmq

from opym.stream import bfill

logger = logging.getLogger(__name__)

DEFAULT_CONNECT = "tcp://127.0.0.1:5556"
REPLY_TIMEOUT_S = 180.0
PAUSE_POLL_S = 10.0
MAX_RETRIES = 3


class BackfillError(RuntimeError):
    pass


@dataclass
class LocalStore:
    name: str
    path: Path
    main_rel: str
    shape: list[int]
    chunks: list[int]
    dtype: str
    sep: str
    written_t: list[int]

    @classmethod
    def load(cls, path: Path) -> LocalStore | None:
        main_rel = bfill.main_array_rel(path)
        if main_rel is None:
            return None
        meta = json.loads((bfill.array_dir(path, main_rel) / ".zarray").read_text())
        shape = list(meta["shape"])
        return cls(
            name=path.name, path=path, main_rel=main_rel, shape=shape, chunks=list(meta["chunks"]),
            dtype=meta["dtype"], sep=meta.get("dimension_separator") or ".",
            written_t=bfill.written_timepoints(path, main_rel, len(shape)),
        )

    def spec(self) -> dict:
        return {"name": self.name, "shape": self.shape, "chunks": self.chunks, "dtype": self.dtype,
                "written_t": self.written_t}

    def array(self) -> zarr.Array:
        return zarr.open_array(str(bfill.array_dir(self.path, self.main_rel)), mode="r")

    def newest_mtime(self) -> float:
        return bfill.newest_mtime(self.path, self.main_rel)

    def planes_present(self, t: int, nz: int) -> list[int]:
        """z indices whose chunk exists on disk. Only `/`-separated per-plane layouts
        (the acquisition writer's) are checked; anything else sends every plane."""
        if len(self.shape) < 4 or self.sep != "/" or self.chunks[1] != 1 or self.chunks[2:] != self.shape[2:]:
            return list(range(nz))
        d = bfill.array_dir(self.path, self.main_rel) / str(t)
        try:
            return sorted(int(e.name) for e in os.scandir(d) if e.name.isdigit())
        except OSError:
            return []


@dataclass
class LocalLeaf:
    path: Path
    stores: list[LocalStore] = field(default_factory=list)


def scan_local_root(root: Path, max_depth: int = 6) -> list[LocalLeaf]:
    """Every directory under `root` that directly holds `*.ome.zarr` stores."""
    leaves: list[LocalLeaf] = []
    root = Path(root)

    def walk(d: Path, depth: int) -> None:
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            return
        stores = []
        for e in entries:
            if e.is_dir() and e.name.endswith(".ome.zarr") and bfill.valid_store_name(e.name):
                s = LocalStore.load(Path(e.path))
                if s is not None:
                    stores.append(s)
        if stores:
            leaves.append(LocalLeaf(d, stores))
        if depth < max_depth:
            for e in entries:
                if e.is_dir() and not e.name.endswith(".ome.zarr") and not e.name.startswith("."):
                    walk(Path(e.path), depth + 1)

    walk(root, 0)
    return leaves


class _Link:
    """DEALER sockets to the receiver: one per tunnel, so parallel tunnels add up."""

    def __init__(self, addrs: list[str], client_id: str) -> None:
        self.ctx = zmq.Context.instance()
        self.socks = []
        for k, addr in enumerate(addrs):
            s = self.ctx.socket(zmq.DEALER)
            s.setsockopt(zmq.IDENTITY, f"{client_id}#{k}".encode())
            s.setsockopt(zmq.LINGER, 0)
            s.connect(addr)
            self.socks.append(s)
        self.poller = zmq.Poller()
        for s in self.socks:
            self.poller.register(s, zmq.POLLIN)
        self._rr = 0

    def send(self, parts: list[bytes]) -> None:
        s = self.socks[self._rr % len(self.socks)]
        self._rr += 1
        s.send_multipart(parts)

    def recv_any(self, timeout_ms: int) -> tuple[dict, bytes | None] | None:
        ready = dict(self.poller.poll(timeout_ms))
        for s in self.socks:
            if s in ready:
                msg_type, _sid, header, payload = bfill.unpack(s.recv_multipart())
                if msg_type == bfill.BF_ACK:
                    return header, payload
        return None

    def close(self) -> None:
        for s in self.socks:
            s.close(0)


class BackfillClient:
    def __init__(
        self,
        connect: list[str],
        root_map: list[tuple[Path, str]],
        *,
        client_id: str | None = None,
        settle_s: float = 900.0,
        window: int = 8,
        slab_bytes: int = 16 << 20,
        compress: bool = True,
        max_mbps: float | None = None,
        pause_file: Path | None = None,
        order: str = "newest",
        reply_timeout_s: float = REPLY_TIMEOUT_S,
    ) -> None:
        self.root_map = [(Path(a), b.rstrip("/")) for a, b in root_map]
        # `client_id` owns the per-store lock, so it is stable across restarts of this machine's
        # sender (a crashed run resumes at once); the ZMQ identity is unique per process.
        self.client_id = client_id or socket.gethostname()
        self.link = _Link(connect, f"{self.client_id}-{os.getpid()}-{uuid.uuid4().hex[:6]}")
        self.settle_s, self.window, self.slab_bytes = settle_s, window, slab_bytes
        self.compress, self.max_mbps, self.pause_file = compress, max_mbps, pause_file
        self.order, self.reply_timeout_s = order, reply_timeout_s
        self.features: list[str] = []
        self.live_active = False
        self.sent_bytes = 0
        self._t0 = time.monotonic()
        self._stash: list[dict] = []   # volume ACKs that arrived while a control request was waiting

    def close(self) -> None:
        self.link.close()

    # ---- transport helpers -----------------------------------------------------------------
    def _note(self, reply: dict) -> None:
        self.features = reply.get("features", self.features)
        if "live_active" in reply:
            self.live_active = bool(reply["live_active"])

    def _request(self, msg_type: bytes, header: dict, expect: str, payload: bytes | None = None) -> dict:
        """One control request; waits for the reply of kind `expect` carrying our echo token."""
        token = uuid.uuid4().hex
        header = {**header, "client": self.client_id, "echo": token}
        deadline = time.monotonic() + self.reply_timeout_s
        self.link.send(bfill.pack(msg_type, self.client_id, header, payload))
        while time.monotonic() < deadline:
            got = self.link.recv_any(500)
            if got is None:
                continue
            reply, _ = got
            self._note(reply)
            if reply.get("echo") == token:
                return reply
            if reply.get("kind") == "volume":
                self._stash.append(reply)
        raise BackfillError(f"no {expect} reply from the receiver within {self.reply_timeout_s:.0f} s "
                            "(is the SSH tunnel up?)")

    def ping(self) -> bool:
        self._request(bfill.BF_INVENTORY, {"leaf": "/", "stores": []}, "inventory")
        return self.live_active

    def paused(self) -> str | None:
        if self.pause_file is not None and Path(self.pause_file).exists():
            return f"pause file {self.pause_file} exists"
        if self.live_active:
            return "a live acquisition is active on Argus"
        return None

    def wait_if_paused(self) -> None:
        why = self.paused()
        while why:
            logger.info("paused: %s", why)
            time.sleep(PAUSE_POLL_S)
            try:
                self.ping()
            except BackfillError:
                pass
            why = self.paused()

    def _throttle(self, nbytes: int) -> None:
        self.sent_bytes += nbytes
        if self.max_mbps:
            due = self.sent_bytes / (self.max_mbps * 1e6)
            ahead = due - (time.monotonic() - self._t0)
            if ahead > 0:
                time.sleep(ahead)

    # ---- mapping ---------------------------------------------------------------------------
    def remote_leaf(self, local_leaf: Path) -> str:
        for local_root, remote_root in self.root_map:
            try:
                rel = Path(local_leaf).relative_to(local_root)
            except ValueError:
                continue
            return "/".join([remote_root, *rel.parts]) if rel.parts else remote_root
        raise BackfillError(f"{local_leaf} is under no --map root")

    # ---- one leaf --------------------------------------------------------------------------
    def sync_leaf(self, leaf: LocalLeaf, dry_run: bool = False) -> dict:
        report: dict = {"leaf": str(leaf.path), "stores": {}}
        remote = self.remote_leaf(leaf.path)
        ready = []
        for s in leaf.stores:
            age = time.time() - s.newest_mtime()
            if age < self.settle_s:
                report["stores"][s.name] = {"result": "settling", "idle_s": int(age)}
            elif not s.written_t:
                report["stores"][s.name] = {"result": "empty"}
            else:
                ready.append(s)
        if not ready:
            return report
        inv = self._request(bfill.BF_INVENTORY, {"leaf": remote, "stores": [s.spec() for s in ready]}, "inventory")
        if not inv.get("ok"):
            raise BackfillError(f"inventory refused for {remote}: {inv.get('error')}")
        for s in ready:
            info = inv["stores"].get(s.name, {})
            state = info.get("state")
            if state == bfill.MARKED:
                report["stores"][s.name] = {"result": "already complete"}
            elif state in (bfill.BUSY, bfill.CONFLICT):
                report["stores"][s.name] = {"result": state, "reason": info.get("reason")}
            elif dry_run:
                missing = [t for t in s.written_t if t not in set(info.get("have_t", []))]
                report["stores"][s.name] = {"result": "would send", "missing_timepoints": len(missing),
                                            "of": len(s.written_t)}
            else:
                report["stores"][s.name] = self._sync_store(remote, s, set(info.get("have_t", [])))
        return report

    def _sync_store(self, remote: str, s: LocalStore, have: set[int]) -> dict:
        t_start = time.monotonic()
        before = self.sent_bytes
        self._send_meta(remote, s)
        arr = s.array()
        digests: dict[int, str] = {}
        todo = [t for t in s.written_t if t not in have]
        pending = self._slab_stream(remote, s, arr, todo, digests)
        self._pump(pending)
        for t in s.written_t:                       # volumes the server already held: hash locally
            if t not in digests:
                digests[t] = bfill.volume_digest(np.asarray(arr[t] if arr.ndim >= 4 else arr[...]))
        root_zattrs = (s.path / ".zattrs").read_bytes() if (s.path / ".zattrs").exists() else None
        while True:
            self.wait_if_paused()
            reply = self._request(bfill.BF_COMMIT, {
                "leaf": remote, "store": s.name, "digests": {str(t): d for t, d in digests.items()},
                "declared_t": s.shape[0] if len(s.shape) >= 4 else 1, "root_zattrs": root_zattrs,
                "source": f"{socket.gethostname()}:{s.path}",
            }, "commit")
            if reply.get("hold"):
                self.live_active = True
                continue
            break
        if not reply.get("ok"):
            raise BackfillError(f"{s.name}: commit failed: {reply.get('error')} {reply.get('mismatch', '')}")
        sent = self.sent_bytes - before
        secs = max(time.monotonic() - t_start, 1e-6)
        return {"result": "complete", "timepoints": len(s.written_t), "sent_timepoints": len(todo),
                "sent_gb": round(sent / 1e9, 3), "mb_per_s": round(sent / secs / 1e6, 1),
                "verified": reply.get("verified")}

    def _send_meta(self, remote: str, s: LocalStore) -> None:
        files = []
        for rel in bfill.meta_files(s.path, s.main_rel):
            files.append({"rel": rel, "data": (s.path / rel).read_bytes()})
        reply = self._request(bfill.BF_META, {"leaf": remote, "store": s.name, "files": files}, "meta")
        if not reply.get("ok"):
            raise BackfillError(f"{s.name}: metadata refused: {reply.get('error')}")

    # ---- volumes ---------------------------------------------------------------------------
    def _slab_runs(self, s: LocalStore, t: int, nz: int) -> list[tuple[int, int]]:
        present = s.planes_present(t, nz) if len(s.shape) >= 4 else list(range(nz))
        plane_bytes = int(np.prod(s.shape[-2:])) * np.dtype(s.dtype).itemsize
        per_slab = max(1, self.slab_bytes // max(plane_bytes, 1))
        runs, start, prev = [], None, None
        for z in present:
            if start is None:
                start = prev = z
            elif z == prev + 1 and (z - start) < per_slab:
                prev = z
            else:
                runs.append((start, prev - start + 1))
                start = prev = z
        if start is not None:
            runs.append((start, prev - start + 1))
        return runs

    def _slab_stream(self, remote: str, s: LocalStore, arr: zarr.Array, todo: list[int], digests: dict[int, str]):
        for t in todo:
            vol = np.asarray(arr[t] if arr.ndim >= 4 else arr[...])
            digests[t] = bfill.volume_digest(vol)
            for z0, nz in self._slab_runs(s, t, vol.shape[0]):
                slab = vol[z0:z0 + nz]
                header = {"leaf": remote, "store": s.name, "t": t, "z0": z0, "nz": nz,
                          "shape_yx": list(vol.shape[1:]), "sha256": bfill.volume_digest(slab)}
                if self.compress and "blosc" in self.features:
                    from numcodecs import blosc

                    payload = blosc.compress(np.ascontiguousarray(slab), b"lz4", 5, blosc.BITSHUFFLE)
                    header["codec"] = "blosc"
                else:
                    payload = np.ascontiguousarray(slab).tobytes()
                yield (t, z0), header, payload

    def _pump(self, stream) -> None:
        """Send slabs with at most `window` unacknowledged, resending any that fail or time out."""
        inflight: dict[tuple, tuple[dict, bytes, float, int]] = {}
        exhausted = False
        while True:
            while not exhausted and len(inflight) < self.window and not self.paused():
                try:
                    key, header, payload = next(stream)
                except StopIteration:
                    exhausted = True
                    break
                self._send_slab(key, header, payload, inflight, attempts=0)
            if exhausted and not inflight:
                return
            if not inflight:                      # paused with nothing outstanding
                self.wait_if_paused()
                continue
            if self._stash:
                got = (self._stash.pop(0), None)
            else:
                got = self.link.recv_any(500)
            now = time.monotonic()
            if got is not None:
                reply, _ = got
                self._note(reply)
                if reply.get("kind") == "volume":
                    key = (reply.get("t"), reply.get("z0"))
                    entry = inflight.get(key)
                    if entry is None:
                        continue
                    header, payload, _sent, attempts = entry
                    if reply.get("ok"):
                        del inflight[key]
                    elif reply.get("hold"):
                        del inflight[key]
                        self.live_active = True
                        self.wait_if_paused()
                        self._send_slab(key, header, payload, inflight, attempts)
                    elif reply.get("retry") and attempts < MAX_RETRIES:
                        del inflight[key]
                        self._send_slab(key, header, payload, inflight, attempts + 1)
                    else:
                        raise BackfillError(f"volume t={key[0]} z0={key[1]} refused: {reply.get('error')}")
            for key, (header, payload, sent, attempts) in list(inflight.items()):
                if now - sent > self.reply_timeout_s:
                    if attempts >= MAX_RETRIES:
                        raise BackfillError(f"volume t={key[0]} z0={key[1]} got no reply (is the tunnel up?)")
                    del inflight[key]
                    self._send_slab(key, header, payload, inflight, attempts + 1)

    def _send_slab(self, key, header, payload, inflight, attempts) -> None:
        header = {**header, "client": self.client_id}
        self.link.send(bfill.pack(bfill.BF_VOLUME, self.client_id, header, payload))
        inflight[key] = (header, payload, time.monotonic(), attempts)
        self._throttle(len(payload))

    # ---- everything ------------------------------------------------------------------------
    def run_once(self, dry_run: bool = False) -> list[dict]:
        self.ping()
        reports = []
        leaves: list[LocalLeaf] = []
        for local_root, _ in self.root_map:
            leaves += scan_local_root(local_root)
        leaves.sort(key=lambda lf: max(s.newest_mtime() for s in lf.stores), reverse=self.order == "newest")
        for leaf in leaves:
            self.wait_if_paused()
            try:
                rep = self.sync_leaf(leaf, dry_run=dry_run)
            except BackfillError as exc:
                logger.error("%s: %s", leaf.path, exc)
                rep = {"leaf": str(leaf.path), "error": str(exc)}
            logger.info("%s", json.dumps(rep))
            reports.append(rep)
        return reports


def _parse_map(text: str) -> tuple[Path, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--map expects LOCAL=REMOTE")
    local, remote = text.split("=", 1)
    return Path(local), remote


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", dest="maps", action="append", type=_parse_map, required=True,
                   help="LOCAL=REMOTE root mapping (repeatable). The path below LOCAL is kept below REMOTE.")
    p.add_argument("--connect", action="append", default=None,
                   help=f"receiver endpoint; repeat for parallel SSH tunnels (default {DEFAULT_CONNECT})")
    p.add_argument("--settle", type=float, default=900.0, help="skip stores written to within this many seconds")
    p.add_argument("--window", type=int, default=8, help="slabs in flight")
    p.add_argument("--max-mbps", type=float, default=None, help="cap the send rate (MB/s)")
    p.add_argument("--pause-file", type=Path, default=None, help="while this file exists, send nothing")
    p.add_argument("--order", choices=("newest", "oldest"), default="newest")
    p.add_argument("--dry-run", action="store_true", help="report what would be sent; send nothing")
    p.add_argument("--watch", type=float, default=None, metavar="SECONDS", help="repeat every SECONDS")
    a = p.parse_args(argv)
    client = BackfillClient(a.connect or [DEFAULT_CONNECT], a.maps, settle_s=a.settle, window=a.window,
                            max_mbps=a.max_mbps, pause_file=a.pause_file, order=a.order)
    try:
        while True:
            try:
                client.run_once(dry_run=a.dry_run)
            except BackfillError as exc:
                logger.error("%s", exc)
            if a.watch is None:
                return 0
            time.sleep(a.watch)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
