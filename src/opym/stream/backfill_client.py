"""`opym-backfill-send`: the acquisition-PC side of the SSH backfill.

Scans local acquisition folders for finished `*.ome.zarr` stores, asks the backfill receiver
(`opym-receive-backfill`, reached through SSH tunnels) what it already holds, and sends only
what is missing: each store's small files verbatim, then pixel volumes as slabs. It finishes a
store with a per-volume SHA-256 manifest the server verifies by reading every volume back.

One command does everything. It opens its own SSH tunnels (4 `ssh` processes by default, so the
transfer is not capped at one connection's 33-63 MB/s), maps the drive onto the microscopy
folder on Argus, and keeps watching for new and growing acquisitions:

    opym-backfill-send                  # S:\\ (or $OPYM_BACKFILL_LOCAL_ROOTS) -> Argus, forever
    opym-backfill-send --dry-run        # one pass, report what would be sent, send nothing
    opym-backfill-send --install        # Windows: start at every logon, hidden, restarted if it dies
    opym-backfill-send --status         # what is it doing?   (--stop, --uninstall)

Mapping. A folder you give (or the default drive) lands under the microscopy root on Argus with
the same relative path: `S:\\20261023_LCK-NG_JB\\Cell_001\\x.ome.zarr` goes to
`<remote root>/20261023_LCK-NG_JB/Cell_001/x.ome.zarr`. On Windows the path is taken relative
to the drive root, so giving `S:\\` or `S:\\20261023_LCK-NG_JB` places the same data in the same
place. `--map LOCAL=REMOTE` still sets an explicit mapping.

It never touches the live stream: separate tunnels on separate local ports (the live links use
5555-5558), the lowest CPU/IO priority, and it pauses while the server reports a live
acquisition (and while a `--pause-file` exists) and skips stores written to recently.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import socket
import sys
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePath

import numpy as np
import zarr
import zmq

from opym.stream import bfill

logger = logging.getLogger(__name__)

DEFAULT_CONNECT = "tcp://127.0.0.1:5556"
DEFAULT_REMOTE_ROOT = "/mmfs1/scratch/jacks.local/microscopy"
DEFAULT_WINDOWS_ROOT = "S:\\"
DEFAULT_WATCH_S = 300.0
RETRY_AFTER_ERROR_S = 30.0
PREFETCH_SLABS = 8
REPLY_TIMEOUT_S = 180.0
PAUSE_POLL_S = 10.0
PAUSE_LOG_EVERY_S = 300.0
MAX_RETRIES = 3
_REMOTE_ROOT_ENV = "OPYM_BACKFILL_REMOTE_ROOT"
_LOCAL_ROOTS_ENV = "OPYM_BACKFILL_LOCAL_ROOTS"
# Directories a whole-drive scan must never descend into.
_SKIP_DIRS = frozenset({"system volume information", "$recycle.bin", "recycler"})


class BackfillError(RuntimeError):
    pass


class StoreRefused(BackfillError):
    """The receiver refused one store (differing metadata, a bad volume, a failed readback).
    Unlike a dead tunnel it says nothing about the other stores, so the pass carries on."""


def default_remote_root() -> str:
    return (os.environ.get(_REMOTE_ROOT_ENV) or DEFAULT_REMOTE_ROOT).rstrip("/")


def default_local_roots() -> list[Path]:
    """`$OPYM_BACKFILL_LOCAL_ROOTS` (os.pathsep-separated), else the S:\\ drive when it exists."""
    env = os.environ.get(_LOCAL_ROOTS_ENV)
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p.strip()]
    if sys.platform == "win32" and Path(DEFAULT_WINDOWS_ROOT).exists():
        return [Path(DEFAULT_WINDOWS_ROOT)]
    return []


def map_for_root(local: PurePath, remote_root: str) -> tuple[PurePath, str]:
    """(local, remote) for a folder to back up under `remote_root`.

    On Windows the drive root mirrors `remote_root`: `S:\\a\\b` maps to `<remote_root>/a/b`, so
    pointing the sender at the whole drive or at one session folder puts the same data in the
    same place. Elsewhere (no drive letter) the folder itself is the mirror of `remote_root`.
    """
    remote_root = remote_root.rstrip("/")
    if local.drive:
        rel = local.relative_to(local.anchor).parts
        return local, "/".join([remote_root, *rel])
    return local, remote_root


def scan_summary(reports: list[dict], unchanged: int = 0) -> str:
    """One line for the log and the status file: what a pass found and did."""
    c: Counter[str] = Counter()
    missing = sent_gb = 0.0
    for rep in reports:
        if "error" in rep:
            c["errors"] += 1
        if rep.get("invalid"):
            c["with a name the receiver refuses"] += len(rep["invalid"])
        for info in rep.get("stores", {}).values():
            r = info.get("result", "?")
            c[{"would send": "to send", "already complete": "verified"}.get(r, r)] += 1
            missing += info.get("missing_timepoints", 0)
            sent_gb += info.get("sent_gb", 0.0)
    bits = [f"{n} {k}" for k, n in sorted(c.items())]
    if unchanged:
        bits.append(f"{unchanged} unchanged since last verified")
    if missing:
        bits.append(f"{int(missing)} timepoints to send")
    if sent_gb:
        bits.append(f"{sent_gb:.2f} GB sent")
    return ", ".join(bits) or "nothing to do"


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
    sig: list = field(default_factory=list)  # `signature()` as of loading, before anything is sent

    @classmethod
    def load(cls, path: Path) -> LocalStore | None:
        main_rel = bfill.main_array_rel(path)
        if main_rel is None:
            return None
        meta = json.loads((bfill.array_dir(path, main_rel) / ".zarray").read_text())
        shape = list(meta["shape"])
        store = cls(
            name=path.name, path=path, main_rel=main_rel, shape=shape, chunks=list(meta["chunks"]),
            dtype=meta["dtype"], sep=meta.get("dimension_separator") or ".",
            written_t=bfill.written_timepoints(path, main_rel, len(shape)),
        )
        try:
            store.sig = store.signature()
        except OSError:
            store.sig = []
        return store

    def spec(self) -> dict:
        return {"name": self.name, "shape": self.shape, "chunks": self.chunks, "dtype": self.dtype,
                "written_t": self.written_t}

    def array(self) -> zarr.Array:
        return zarr.open_array(str(bfill.array_dir(self.path, self.main_rel)), mode="r")

    def newest_mtime(self) -> float:
        return bfill.newest_mtime(self.path, self.main_rel)

    def signature(self) -> list:
        """Cheap change detector: newest mtime (follows a writer adding timepoints) and how many
        timepoints exist. Taken when the store is loaded, so a store that grows while it is being
        sent no longer matches on the next pass and is sent again."""
        return [round(self.newest_mtime(), 3), len(self.written_t)]

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
    invalid: list[str] = field(default_factory=list)  # `*.ome.zarr` dirs whose names the receiver rejects


class ScanState:
    """Stores already verified complete on Argus, remembered across passes and restarts.

    Walking a store to load it costs a pass over every chunk file, and a whole-drive scan every
    few minutes would hammer the acquisition disk. A store recorded here is skipped, without
    even being loaded, for as long as its signature (see `LocalStore.signature`) is unchanged.
    Only verified-complete stores are recorded: anything busy, conflicting or failed is looked at
    again on every pass. The record is dropped if the folder mapping changes.
    """

    VERSION = 1

    def __init__(self, path: Path | None, maps: list[tuple[PurePath, str]], fresh: bool = False) -> None:
        self.path = Path(path) if path else None
        self.maps = [[str(a), b] for a, b in maps]
        self.stores: dict[str, dict] = {}
        self.dirty = fresh   # a fresh start overwrites what was remembered at the first save
        if not fresh:
            self._load()

    def _load(self) -> None:
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if data.get("version") == self.VERSION and data.get("maps") == self.maps:
            self.stores = dict(data.get("stores", {}))

    def save(self) -> None:
        if self.path is None or not self.dirty:
            return
        tmp = self.path.with_name(self.path.name + ".part")
        try:
            tmp.write_text(json.dumps({"version": self.VERSION, "maps": self.maps, "stores": self.stores}),
                           encoding="utf-8")
            os.replace(tmp, self.path)
            self.dirty = False
        except OSError:
            logger.debug("could not save the scan state", exc_info=True)

    def record(self, store: LocalStore) -> None:
        if not store.sig:  # could not be read when loaded: do not vouch for it
            return
        self.stores[str(store.path)] = {
            "main_rel": store.main_rel, "ndim": len(store.shape), "sig": store.sig, "at": time.time(),
        }
        self.dirty = True

    def unchanged(self, path: Path) -> bool:
        """True when `path` was verified complete and has not changed since."""
        entry = self.stores.get(str(path))
        if not entry:
            return False
        try:
            rel = entry["main_rel"]
            newest = bfill.newest_mtime(path, rel)
            n = len(bfill.written_timepoints(path, rel, entry["ndim"]))
        except (OSError, KeyError, ValueError):
            return False
        return entry["sig"] == [round(newest, 3), n]


def scan_local_root(root: Path, max_depth: int = 6, *, known: ScanState | None = None,
                    stats: dict | None = None) -> list[LocalLeaf]:
    """Every directory under `root` that directly holds `*.ome.zarr` stores.

    Skips hidden folders, the Windows system folders and (via `known`) stores already verified
    complete and unchanged; `stats["unchanged"]` counts those. A store directory whose name the
    receiver would refuse is listed in `LocalLeaf.invalid` rather than silently ignored.
    """
    leaves: list[LocalLeaf] = []
    root = Path(root)

    def skip_dir(name: str) -> bool:
        return name.startswith((".", "$")) or name.lower() in _SKIP_DIRS

    def walk(d: Path, depth: int) -> None:
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            return
        stores, invalid = [], []
        for e in entries:
            try:
                is_dir = e.is_dir()
            except OSError:
                continue
            if not (is_dir and e.name.endswith(".ome.zarr")):
                continue
            if not bfill.valid_store_name(e.name):
                invalid.append(e.name)
            elif known is not None and known.unchanged(Path(e.path)):
                if stats is not None:
                    stats["unchanged"] = stats.get("unchanged", 0) + 1
            else:
                s = LocalStore.load(Path(e.path))
                if s is not None:
                    stores.append(s)
        if stores or invalid:
            leaves.append(LocalLeaf(d, stores, invalid))
        if depth < max_depth:
            for e in entries:
                try:
                    descend = e.is_dir() and not e.name.endswith(".ome.zarr") and not skip_dir(e.name)
                except OSError:
                    continue
                if descend:
                    walk(Path(e.path), depth + 1)

    walk(root, 0)
    return leaves


class _Prefetch:
    """Runs a generator on a worker thread, up to `depth` items ahead of the consumer.

    Reading a volume, hashing it and compressing its slabs is CPU and disk work that releases
    the GIL; doing it beside the send/ack loop keeps the tunnels full instead of idling while the
    next slab is prepared. An exception in the generator is re-raised to the consumer.
    """

    _END = object()

    def __init__(self, gen, depth: int = PREFETCH_SLABS) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self._stop = threading.Event()
        self._exc: BaseException | None = None
        self._thread = threading.Thread(target=self._work, args=(gen,), daemon=True, name="backfill-prefetch")
        self._thread.start()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _work(self, gen) -> None:
        try:
            for item in gen:
                if not self._put(item):
                    return
        except BaseException as exc:  # noqa: BLE001 - handed to the consumer
            self._exc = exc
        finally:
            self._put(self._END)

    EMPTY = object()

    def poll(self, timeout: float):
        """The next item, `EMPTY` if none is ready within `timeout`, StopIteration at the end.
        Lets the send loop keep processing ACKs while the next slab is still being prepared."""
        try:
            item = self._q.get(timeout=timeout)
        except queue.Empty:
            return self.EMPTY
        if item is self._END:
            if self._exc is not None:
                raise self._exc
            raise StopIteration
        return item

    def __iter__(self) -> _Prefetch:
        return self

    def __next__(self):
        item = self.poll(timeout=3600)
        if item is self.EMPTY:
            raise TimeoutError("prefetch produced nothing for an hour")
        return item

    def close(self) -> None:
        self._stop.set()


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
        state: ScanState | None = None,
    ) -> None:
        self.root_map = [(Path(a), b.rstrip("/")) for a, b in root_map]
        self.state = state
        self.activity = "starting"          # shown by --status
        self.last_pass: dict | None = None  # {"finished_at", "seconds", "summary"} of the latest pass
        self._pause_logged_at = float("-inf")
        self._echo = ""
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
            self.activity = f"paused: {why}"
            now = time.monotonic()
            if now - self._pause_logged_at >= PAUSE_LOG_EVERY_S:  # a long live run must not flood the log
                logger.info("paused: %s", why)
                self._pause_logged_at = now
            time.sleep(PAUSE_POLL_S)
            try:
                self.ping()
            except BackfillError:
                pass
            why = self.paused()
        self._pause_logged_at = float("-inf")

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
        report["remote"] = remote
        if leaf.invalid:
            report["invalid"] = list(leaf.invalid)  # store names the receiver would refuse: not sent
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
            self.activity = f"{s.path}"
            if state == bfill.MARKED:
                report["stores"][s.name] = {"result": "already complete"}
                self._remember(s, dry_run)
            elif state in (bfill.BUSY, bfill.CONFLICT):
                report["stores"][s.name] = {"result": state, "reason": info.get("reason")}
            elif dry_run:
                missing = [t for t in s.written_t if t not in set(info.get("have_t", []))]
                report["stores"][s.name] = {"result": "would send", "missing_timepoints": len(missing),
                                            "of": len(s.written_t)}
            else:
                try:
                    report["stores"][s.name] = self._sync_store(remote, s, set(info.get("have_t", [])))
                except StoreRefused as exc:  # this store only: its siblings still go
                    logger.error("%s: %s", s.path, exc)
                    report["stores"][s.name] = {"result": "failed", "reason": str(exc)}
                else:
                    self._remember(s, dry_run)
        return report

    def _remember(self, s: LocalStore, dry_run: bool) -> None:
        """Note a store verified complete on Argus, so later passes skip it while it is unchanged."""
        if self.state is not None and not dry_run:
            self.state.record(s)

    def _sync_store(self, remote: str, s: LocalStore, have: set[int]) -> dict:
        t_start = time.monotonic()
        before = self.sent_bytes
        self._send_meta(remote, s)
        arr = s.array()
        digests: dict[int, str] = {}
        todo = [t for t in s.written_t if t not in have]
        # Every slab carries this token and the receiver echoes it: an ACK that arrives late from a
        # store that failed earlier can never be mistaken for one of this store's.
        self._echo = uuid.uuid4().hex
        pending = _Prefetch(self._slab_stream(remote, s, arr, todo, digests))
        try:
            self._pump(pending)
        finally:
            pending.close()
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
            raise StoreRefused(f"{s.name}: commit failed: {reply.get('error')} {reply.get('mismatch', '')}")
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
            raise StoreRefused(f"{s.name}: metadata refused: {reply.get('error')}")

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
                          "shape_yx": list(vol.shape[1:]), "sha256": bfill.volume_digest(slab),
                          "echo": self._echo}
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
                    item = stream.poll(0.05 if inflight else 0.5)
                except StopIteration:
                    exhausted = True
                    break
                if item is _Prefetch.EMPTY:
                    break                         # next slab not ready: go handle ACKs meanwhile
                key, header, payload = item
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
                    if reply.get("echo") != self._echo:
                        continue                  # a late ACK from an earlier store
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
                        raise StoreRefused(f"volume t={key[0]} z0={key[1]} refused: {reply.get('error')}")
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
        t0 = time.monotonic()
        self.activity = "checking the receiver"
        self.ping()
        reports: list[dict] = []
        stats: dict = {}
        leaves: list[LocalLeaf] = []
        self.activity = "scanning for new and changed stores"
        for local_root, _ in self.root_map:
            leaves += scan_local_root(local_root, known=self.state, stats=stats)
        leaves.sort(key=lambda lf: max((s.newest_mtime() for s in lf.stores), default=0.0),
                    reverse=self.order == "newest")
        for leaf in leaves:
            self.wait_if_paused()
            try:
                rep = self.sync_leaf(leaf, dry_run=dry_run)
            except BackfillError as exc:
                logger.error("%s: %s", leaf.path, exc)
                rep = {"leaf": str(leaf.path), "error": str(exc)}
            logger.info("%s", json.dumps(rep))
            reports.append(rep)
            if self.state is not None:
                self.state.save()   # a restart mid-pass keeps what is already verified
        summary = scan_summary(reports, stats.get("unchanged", 0))
        for line in pass_table(reports, self.root_map):
            logger.info("%s", line)
        logger.info("pass done in %.0f s: %s", time.monotonic() - t0, summary)
        self.last_pass = {"finished_at": time.time(), "seconds": time.monotonic() - t0, "summary": summary}
        self.activity = "idle until the next pass"
        return reports


def pass_table(reports: list[dict], root_map: list[tuple[Path, str]]) -> list[str]:
    """One line per top-level folder under each root that has something to report: where it
    maps to on Argus and what became of its stores."""
    groups: dict[tuple[str, str], Counter] = {}
    for rep in reports:
        leaf = Path(rep["leaf"])
        for local_root, remote_root in root_map:
            try:
                parts = leaf.relative_to(local_root).parts
            except ValueError:
                continue
            top = (str(local_root / parts[0]) if parts else str(local_root),
                   "/".join([remote_root, parts[0]]) if parts else remote_root)
            c = groups.setdefault(top, Counter())
            if "error" in rep:
                c["error"] += 1
            for info in rep.get("stores", {}).values():
                c[info.get("result", "?")] += 1
            if rep.get("invalid"):
                c["bad name"] += len(rep["invalid"])
            break
    return [f"  {local}  ->  {remote}   " + ", ".join(f"{n} {k}" for k, n in sorted(c.items()))
            for (local, remote), c in sorted(groups.items())]


def _parse_map(text: str) -> tuple[Path, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--map expects LOCAL=REMOTE")
    local, remote = text.split("=", 1)
    return Path(local), remote


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="opym-backfill-send", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="*", type=Path, metavar="FOLDER",
                   help="folder(s) to back up, each mapped under --remote-root with the same relative "
                        "path (default: $OPYM_BACKFILL_LOCAL_ROOTS, else S:\\ on Windows)")
    p.add_argument("--remote-root", default=None, metavar="PATH",
                   help=f"where FOLDERs land on Argus (default {DEFAULT_REMOTE_ROOT}, or $OPYM_BACKFILL_REMOTE_ROOT)")
    p.add_argument("--map", dest="maps", action="append", type=_parse_map, default=None,
                   help="explicit LOCAL=REMOTE mapping (repeatable); the path below LOCAL is kept below REMOTE")
    g = p.add_argument_group("tunnels (opened and kept up automatically)")
    g.add_argument("--tunnels", type=int, default=4, metavar="N",
                   help="parallel SSH connections to Argus (default 4: one is capped near 33-63 MB/s)")
    g.add_argument("--ssh-host", default="Argus", help="ssh Host alias of Argus (default Argus)")
    g.add_argument("--ssh-exe", default=os.environ.get("OPYM_BACKFILL_SSH") or "ssh",
                   help="the ssh program (default ssh, or $OPYM_BACKFILL_SSH); give its full path if the "
                        "background task cannot find it")
    g.add_argument("--base-port", type=int, default=5656,
                   help="first local port; tunnel k listens on base+k (default 5656; the live stream uses 5555-5558)")
    g.add_argument("--remote-port", type=int, default=5556, help="the backfill receiver's port on Argus")
    g.add_argument("--ssh-arg", action="append", default=[], metavar="ARG",
                   help="extra ssh argument, e.g. --ssh-arg=-i --ssh-arg=C:\\keys\\id (repeatable)")
    g.add_argument("--no-tunnels", action="store_true",
                   help=f"do not open tunnels; use the one already listening on {DEFAULT_CONNECT}")
    g.add_argument("--connect", action="append", default=None,
                   help="receiver endpoint(s) you tunnel yourself (implies --no-tunnels); repeat for several")
    g = p.add_argument_group("behaviour")
    g.add_argument("--watch", type=float, default=DEFAULT_WATCH_S, metavar="SECONDS",
                   help=f"seconds between passes (default {DEFAULT_WATCH_S:.0f})")
    g.add_argument("--once", action="store_true", help="one pass, then exit")
    g.add_argument("--dry-run", action="store_true", help="one pass: report what would be sent, send nothing")
    g.add_argument("--rescan", action="store_true",
                   help="re-verify every store, even ones already verified and unchanged")
    g.add_argument("--settle", type=float, default=900.0, help="skip stores written to within this many seconds")
    g.add_argument("--window", type=int, default=None, help="slabs in flight (default 8 per tunnel)")
    g.add_argument("--max-mbps", type=float, default=None, help="cap the send rate (MB/s)")
    g.add_argument("--pause-file", type=Path, default=None, help="while this file exists, send nothing")
    g.add_argument("--order", choices=("newest", "oldest"), default="newest")
    g.add_argument("--normal-priority", action="store_true",
                   help="do not drop to the lowest CPU/IO priority (default: background priority)")
    g = p.add_argument_group("running in the background (Windows)")
    g.add_argument("--install", action="store_true",
                   help="start at every logon (hidden, restarted if it dies) with the other options given, and start now")
    g.add_argument("--uninstall", action="store_true", help="stop it and remove the logon task")
    g.add_argument("--status", action="store_true", help="show whether it is running and what it is doing")
    g.add_argument("--stop", action="store_true", help="stop the running sender")
    return p


def resolve_maps(a: argparse.Namespace) -> list[tuple[PurePath, str]]:
    maps: list[tuple[PurePath, str]] = list(a.maps or [])
    remote_root = (a.remote_root or default_remote_root()).rstrip("/")
    roots = list(a.roots) or ([] if maps else default_local_roots())
    if not maps and not roots:
        raise SystemExit("no folder to back up: give one (opym-backfill-send S:\\), set "
                         f"{_LOCAL_ROOTS_ENV}, or use --map LOCAL=REMOTE")
    for r in roots:
        maps.append(map_for_root(Path(r), remote_root))
    for local, _ in maps:
        if not Path(local).is_dir():
            raise SystemExit(f"{local} is not a folder (is the drive connected?)")
    return maps


def _write_status(client: BackfillClient, tunnels, started: float, state: str = "running") -> None:
    from opym.stream import backfill_service as svc

    svc.write_status({
        "state": state, "started_at": started, "activity": client.activity,
        "sent_gb": client.sent_bytes / 1e9, "last_pass": client.last_pass,
        "tunnels": tunnels.status() if tunnels is not None else [],
    })


def _status_loop(client: BackfillClient, tunnels, stop: threading.Event, started: float) -> None:
    while not stop.is_set():
        _write_status(client, tunnels, started)
        stop.wait(5.0)


def main(argv: list[str] | None = None) -> int:
    from contextlib import nullcontext

    from opym.stream import backfill_service as svc

    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    if a.status:
        print(svc.format_status())
        return 0
    if a.stop:
        print(svc.stop_running())
        return 0
    if a.uninstall:
        print(svc.uninstall_task())
        return 0
    if a.install:
        print(svc.install_task([x for x in argv if x != "--install"]))
        return 0

    log = svc.setup_logging(console=sys.stderr is not None)
    maps = resolve_maps(a)
    once = a.once or a.dry_run
    try:
        guard = nullcontext() if a.dry_run else svc.SingleInstance()  # a dry run only reads
        with guard:
            if not a.normal_priority:
                logger.info("process priority: %s", svc.lower_priority())
            tunnels = None
            if a.connect:
                endpoints = a.connect
            elif a.no_tunnels:
                endpoints = [DEFAULT_CONNECT]
            else:
                from opym.stream.tunnels import TunnelSet

                tunnels = TunnelSet(a.ssh_host, a.base_port, a.remote_port, a.tunnels, ssh_args=a.ssh_arg,
                                    ssh_exe=a.ssh_exe)
                endpoints = tunnels.endpoints
                tunnels.start()
                ready = tunnels.wait_ready(20.0)
                logger.info("%d tunnel(s) to %s on local ports %d-%d: %s", len(endpoints), a.ssh_host,
                            a.base_port, a.base_port + len(endpoints) - 1,
                            "all listening" if ready else "not all listening yet (retrying in the background)")
            logger.info("mapping: %s", "; ".join(f"{loc} -> {rem}" for loc, rem in maps))
            logger.info("log: %s", log)
            state = ScanState(None if a.dry_run else svc.scan_state_path(), maps, fresh=a.rescan)
            client = BackfillClient(endpoints, maps, settle_s=a.settle, window=a.window or 8 * len(endpoints),
                                    max_mbps=a.max_mbps, pause_file=a.pause_file, order=a.order, state=state)
            stop = threading.Event()
            started = time.time()
            if not a.dry_run:
                threading.Thread(target=_status_loop, args=(client, tunnels, stop, started),
                                 daemon=True, name="backfill-status").start()
            try:
                while True:
                    wait = a.watch
                    try:
                        client.run_once(dry_run=a.dry_run)
                    except BackfillError as exc:
                        logger.error("%s", exc)
                        client.activity = f"waiting to retry: {exc}"
                        wait = min(a.watch, RETRY_AFTER_ERROR_S)
                    if once:
                        return 0
                    time.sleep(wait)
            except KeyboardInterrupt:
                logger.info("stopped")
                return 0
            finally:
                stop.set()
                if not a.dry_run:
                    client.activity = "stopped"
                    _write_status(client, tunnels, started, state="stopped")
                client.close()
                if tunnels is not None:
                    tunnels.stop()
    except svc.AlreadyRunning as exc:
        print(exc, file=sys.stderr)
        return 2
    finally:
        svc.close_logging()


if __name__ == "__main__":
    raise SystemExit(main())
