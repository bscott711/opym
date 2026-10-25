"""`opym-receive-backfill`: the Argus side of the SSH backfill.

A second, independent receiver (own process, port and systemd unit) that lands finished
acquisitions from the acquisition PC into the raw layout a Globus transfer produces. It
shares no state with the live receiver (`receiver.py`): the live path is not modified.

Guarantees:
  * It never moves, supersedes or overwrites existing data. A store whose pixel array
    disagrees with the sender's is a `conflict` and is left alone; a store that something
    else is still writing (recent modification, no lock of ours) is `busy`.
  * It holds every volume write while a live acquisition (or a warm-up for one) holds the
    lease (`opym.lanes`), so backfill never competes with a live session for Argus's disk.
  * A store is only marked complete after the server has read every volume back from disk
    and matched the sender's per-volume SHA-256. The root `.zattrs` is written after that, then
    `.opym_complete.json`, so an incomplete store never looks ready.
  * Destinations must lie under `OPYM_BACKFILL_RAW_ROOTS` (a colon-separated allowlist).

Two exceptions keep "all of the data, once" true (2026-10-24):
  * The live stream's own partial copy of a run (a store whose root `.zattrs` carries the
    stream's `opym.session_id`, with no completion marker) is completed in place instead of
    refused: its pixel array must match the sender's exactly, its pixel chunks are never
    rewritten (only missing timepoints are added), the sender's small metadata files replace
    the stream's, and the root `.zattrs` keeps the stream's `opym` tags. Every timepoint is
    still read back against the sender's digests before the marker is written.
  * A store that is absent where it is being sent but already lies elsewhere under the raw
    roots (the same name, pixel array and, for a sampled timepoint, the same SHA-256) is
    reported as a duplicate, so the sender skips it. Candidates come from the backfill
    registry's folders (`OPYM_BACKFILL_REGISTRY_PATH`), listed at most every 10 min.

Requests are handled on a worker pool; only the socket stays on the main thread
(2026-10-25). Before, every GPFS write and every commit's whole-store readback ran in
the one receive loop, so four tunnels shared one writer and a long readback held up
every other request. The run records (`backfill_runs.py`) and a log line per store
and commit say what each run did.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import socket
import sqlite3
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import zarr
import zmq

from opym import lanes
from opym.stream import bfill
from opym.stream.backfill_runs import RunRecords, default_runs_dir

logger = logging.getLogger(__name__)

DEFAULT_BIND = "tcp://127.0.0.1:5556"
_RAW_ROOTS_ENV = "OPYM_BACKFILL_RAW_ROOTS"
_BIND_ENV = "OPYM_BACKFILL_BIND"
_SETTLE_ENV = "OPYM_BACKFILL_SETTLE_SECONDS"
_REGISTRY_ENV = "OPYM_BACKFILL_REGISTRY_PATH"
INDEX_TTL_S = 600.0
# Which timepoints of a store are complete is found by counting its chunk files: ~13,000
# for a 100-timepoint run, 8.3 s one by one on GPFS (2026-10-24). GPFS is slow per request,
# not per byte, so they are counted in parallel, and the answer is kept until the store
# changes.
_COUNT_WORKERS = 32
# What the live stream's store tagging looks like (opym.stream.rawmirror).
_STREAM_TAG = "opym"
# Pixel-array .zarray keys that decide what the chunk files mean.
_PIXEL_KEYS = ("shape", "chunks", "dtype", "compressor", "filters", "order", "dimension_separator")
DEFAULT_SETTLE_S = 900.0
LOCK_STALE_S = 600.0
# Requests in flight at once: slabs from every link, plus a control request or two.
_REQUEST_WORKERS = 16
# A request that takes this long is logged: the 2026-10-24 stall left no trace at all.
SLOW_REQUEST_S = 30.0
# While requests are in flight the loop polls this often, to send their replies promptly.
_BUSY_POLL_MS = 2
# A sender resends a request on another tunnel when its tunnel goes silent; the first copy
# can still arrive later. Metadata and commit requests are remembered by their echo token
# so a late copy gets the first reply instead of running again: a stale metadata request
# would otherwise remove the completion marker of a store finished since (2026-10-25).
_ONCE = frozenset({bfill.BF_META, bfill.BF_COMMIT})
_REMEMBER = 4096


def _env_list(name: str) -> list[str]:
    return [p for p in os.environ.get(name, "").split(":") if p.strip()]


class BackfillReceiver:
    def __init__(
        self,
        bind_addr: str | None = None,
        raw_roots: list[Path | str] | tuple[Path | str, ...] = (),
        settle_seconds: float | None = None,
        jobs_dir: Path | None = None,
        registry_path: Path | str | None = None,
        runs_dir: Path | str | None = None,
    ) -> None:
        self.raw_roots = [Path(r).resolve() for r in raw_roots]
        if not self.raw_roots:
            raise ValueError(f"no raw roots allowed: set {_RAW_ROOTS_ENV}")
        env_settle = os.environ.get(_SETTLE_ENV)
        self.settle_s = settle_seconds if settle_seconds is not None else float(env_settle or DEFAULT_SETTLE_S)
        self.jobs_dir = jobs_dir
        self.registry_path = Path(registry_path) if registry_path else (
            Path(os.environ[_REGISTRY_ENV]) if os.environ.get(_REGISTRY_ENV) else None)
        self._index: dict[str, list[Path]] = {}
        self._index_at = float("-inf")
        self._adopted: dict[Path, dict[str, Any]] = {}
        self._have_cache: dict[Path, tuple[tuple, list[int], list[int]]] = {}
        self._pool = ThreadPoolExecutor(max_workers=_COUNT_WORKERS, thread_name_prefix="bf-count")
        # Separate from `_pool`: a store's inspection waits on that pool's timepoint counts.
        self._store_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="bf-store")
        # Requests; and, separately, run-record updates, kept in the order they came.
        self._req_pool = ThreadPoolExecutor(max_workers=_REQUEST_WORKERS, thread_name_prefix="bf-req")
        self._run_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bf-run")
        self._replies: queue.SimpleQueue[tuple[bytes, str, dict[str, Any]]] = queue.SimpleQueue()
        self._inflight = 0
        self._lock = threading.Lock()   # stats, the per-array write locks, the store index
        self._write_locks: dict[Path, threading.Lock] = {}
        self._index_lock = threading.Lock()
        # Commit readback: whole volumes, ~190 MB each at today's run sizes; 8 at a time.
        self._readback_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="bf-readback")
        self._started: dict[Path, float] = {}
        self._answered: OrderedDict[str, threading.Event | dict[str, Any]] = OrderedDict()
        self.runs = RunRecords(Path(runs_dir) if runs_dir else default_runs_dir(self.raw_roots))
        self.host = socket.gethostname()
        self._pending_zattrs: dict[Path, bytes] = {}
        self._arrays: dict[Path, zarr.Array] = {}
        self._main_rel_cache: dict[Path, str] = {}
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(bind_addr or os.environ.get(_BIND_ENV) or DEFAULT_BIND)
        self.endpoint = self._sock.getsockopt(zmq.LAST_ENDPOINT).decode()
        self.stats = {"volumes": 0, "bytes": 0, "commits": 0, "held": 0, "rejected": 0}

    # ---- lifecycle -----------------------------------------------------------------------
    def __enter__(self) -> BackfillReceiver:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._sock.close(0)
        for pool in (self._req_pool, self._run_pool, self._readback_pool, self._store_pool, self._pool):
            pool.shutdown(wait=False)

    def run_forever(self) -> None:
        logger.info("backfill receiver on %s, raw roots %s, run records in %s", self.endpoint,
                    [str(r) for r in self.raw_roots], self.runs.runs_dir)
        while True:
            self._run_once(1000)

    def _run_once(self, timeout_ms: int = 100) -> bool:
        """Receive at most one request and hand it to a worker; send whatever replies are
        ready. The socket is only ever used from this thread."""
        self._send_replies()
        got = bool(self._sock.poll(_BUSY_POLL_MS if self._inflight else timeout_ms))
        if got:
            frames = self._sock.recv_multipart()
            identity, parts = frames[0], frames[1:]
            try:
                msg_type, sid, header, payload = bfill.unpack(parts)
            except Exception:  # noqa: BLE001 - malformed input is dropped, never answered
                self._count("rejected")
                logger.warning("dropped a malformed message")
            else:
                self._inflight += 1
                pool = self._run_pool if msg_type == bfill.BF_RUN else self._req_pool
                pool.submit(self._handle, identity, msg_type, sid, header, payload)
        self._send_replies()
        self.runs.sweep()
        return got

    def _handle(self, identity: bytes, msg_type: bytes, sid: str, header: dict, payload: bytes | None) -> None:
        t0 = time.monotonic()
        once = f"{header.get('client', '')}/{header['echo']}" if msg_type in _ONCE and header.get("echo") else None
        first = self._claim(once) if once else None
        if once and first is not None:
            logger.info("repeated %s request for %s answered from the first", _kind(msg_type), header.get("store"))
            reply = dict(first)
        else:
            try:
                reply = self._dispatch(msg_type, header, payload)
            except Exception as exc:  # noqa: BLE001 - one bad request must not stop the receiver
                logger.exception("request failed")
                reply = {"kind": _kind(msg_type), "ok": False, "error": f"server error: {exc}"}
            if once:
                self._remember(once, reply)
        took = time.monotonic() - t0
        if took > SLOW_REQUEST_S:
            logger.warning("slow %s request: %.0f s (store %s, t %s)", _kind(msg_type), took,
                           header.get("store") or header.get("leaf"), header.get("t"))
        try:
            reply.setdefault("live_active", self.live_active())
        except Exception:  # noqa: BLE001
            reply.setdefault("live_active", False)
        reply["server_time_s"] = time.time()
        reply["features"] = bfill.FEATURES
        if "echo" in header:
            reply["echo"] = header["echo"]
        self._replies.put((identity, sid, reply))

    def _claim(self, key: str) -> dict[str, Any] | None:
        """None if this request is new (and now ours to run); else the first copy's reply,
        waiting for it if that copy is still running."""
        with self._lock:
            seen = self._answered.get(key)
            if seen is None:
                self._answered[key] = threading.Event()
                while len(self._answered) > _REMEMBER:
                    self._answered.popitem(last=False)
                return None
        if isinstance(seen, threading.Event):
            seen.wait()
            with self._lock:
                seen = self._answered.get(key)
        return seen if isinstance(seen, dict) else {"ok": False, "error": "repeated request"}

    def _remember(self, key: str, reply: dict[str, Any]) -> None:
        with self._lock:
            pending = self._answered.get(key)
            self._answered[key] = dict(reply)
        if isinstance(pending, threading.Event):
            pending.set()

    def _send_replies(self) -> None:
        while True:
            try:
                identity, sid, reply = self._replies.get_nowait()
            except queue.Empty:
                return
            self._inflight -= 1
            self._sock.send_multipart([identity, *bfill.pack(bfill.BF_ACK, sid, reply)])

    def _count(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.stats[key] += n

    # ---- policy ----------------------------------------------------------------------------
    def live_active(self) -> bool:
        return lanes.live_lease_active(self.jobs_dir) or lanes.warm_lease_active(self.jobs_dir)

    def _dispatch(self, msg_type: bytes, header: dict, payload: bytes | None) -> dict[str, Any]:
        if msg_type == bfill.BF_INVENTORY:
            return self._inventory(header)
        if msg_type == bfill.BF_META:
            return self._meta(header)
        if msg_type == bfill.BF_VOLUME:
            return self._volume(header, payload or b"")
        if msg_type == bfill.BF_COMMIT:
            return self._commit(header)
        if msg_type == bfill.BF_RUN:
            return self.runs.apply(header)
        self._count("rejected")
        return {"kind": "error", "ok": False, "error": "unexpected message type"}

    def _dest(self, header: dict) -> tuple[Path | None, str | None]:
        leaf = bfill.safe_leaf(header.get("leaf", ""), self.raw_roots)
        if leaf is None:
            return None, "leaf is outside the allowed raw roots"
        return leaf, None

    def _store_dir(self, header: dict) -> tuple[Path | None, str | None]:
        leaf, err = self._dest(header)
        if err:
            return None, err
        name = header.get("store", "")
        if not bfill.valid_store_name(name):
            return None, f"bad store name {name!r}"
        return leaf / name, None

    # ---- inventory -------------------------------------------------------------------------
    def _inventory(self, header: dict) -> dict[str, Any]:
        leaf, err = self._dest(header)
        if err:
            self._count("rejected")
            return {"kind": "inventory", "ok": False, "error": err}
        out: dict[str, Any] = {}
        specs = []
        for spec in header.get("stores", []):
            name = spec.get("name", "")
            if not bfill.valid_store_name(name):
                out[name] = {"state": bfill.CONFLICT, "reason": "bad store name"}
            else:
                specs.append(spec)
        client = header.get("client", "")
        # Stores are inspected side by side (a folder of 150-timepoint runs took longer
        # than the sender's 180 s reply timeout one by one, 2026-10-24).
        infos = list(self._store_pool.map(lambda sp: self._inspect_store(leaf / sp["name"], sp, client), specs))
        for spec, info in zip(specs, infos):
            if info["state"] == bfill.ABSENT:
                info.update(self._elsewhere(leaf / spec["name"], spec))
            out[spec["name"]] = info
        return {"kind": "inventory", "ok": True, "stores": out}

    # ---- the same data elsewhere ---------------------------------------------------------
    def _store_index(self) -> dict[str, list[Path]]:
        """Store name -> every `*.ome.zarr` with that name in the registry's folders."""
        with self._index_lock:
            return self._store_index_locked()

    def _store_index_locked(self) -> dict[str, list[Path]]:
        if time.monotonic() - self._index_at < INDEX_TTL_S:
            return self._index
        self._index_at = time.monotonic()
        if self.registry_path is None or not self.registry_path.exists():
            logger.info("no backfill registry: duplicates elsewhere on Argus are not checked")
            self._index = {}
            return self._index
        try:
            conn = sqlite3.connect(f"file:{self.registry_path}?mode=ro", uri=True, timeout=10)
            try:
                rows = conn.execute(
                    "SELECT master_file FROM datasets WHERE master_file LIKE '%.ome.zarr'"
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.Error:
            logger.exception("could not read the backfill registry")
            return self._index
        # A zarr dataset's channel stores sit beside its master store; TIFF folders (slow to
        # list, thousands of files) hold no stores.
        folders = {Path(master).parent for (master,) in rows if master}
        index: dict[str, list[Path]] = {}
        for folder in folders:
            try:
                entries = list(os.scandir(folder))
            except OSError:
                continue
            for e in entries:
                if e.name.endswith(".ome.zarr") and e.is_dir():
                    index.setdefault(e.name, []).append(Path(e.path))
        self._index = index
        logger.info("store index: %d names in %d folders", len(index), len(folders))
        return index

    def _complete_ts(self, dest: Path, main_rel: str, spec: dict) -> list[int]:
        """Timepoints of `written_t` whose every chunk is on disk (and non-empty)."""
        written = [int(t) for t in spec.get("written_t", [])]
        if len(spec["shape"]) < 4:
            d = bfill.array_dir(dest, main_rel)
            return written if any(not e.name.startswith(".") for e in os.scandir(d)) else []
        per_t = bfill.expected_chunks_per_t(spec["shape"], spec["chunks"])
        try:
            sig = (bfill.newest_mtime(dest, main_rel), per_t)
        except OSError:
            sig = None
        hit = self._have_cache.get(dest)
        if sig is not None and hit and hit[0] == sig and set(written) <= set(hit[1]) | set(hit[2]):
            return [t for t in written if t in hit[1]]

        def complete(t: int) -> bool:
            n, smallest = bfill.chunk_count_for_t(dest, main_rel, t)
            return n == per_t and smallest > 0

        flags = list(self._pool.map(complete, written))
        have = [t for t, ok in zip(written, flags) if ok]
        if sig is not None:
            self._have_cache[dest] = (sig, have, [t for t in written if t not in have])
        return have

    def _elsewhere(self, dest: Path, spec: dict) -> dict[str, Any]:
        """Copies of this store under another folder: name and pixel array match. With the
        sender's `probe` ({"t", "sha256"}), the one whose volume t matches is `duplicate_of`."""
        found = []
        for path in self._store_index().get(dest.name, []):
            if path == dest or not bfill.safe_leaf(str(path.parent), self.raw_roots):
                continue
            main_rel = bfill.main_array_rel(path)
            meta = self._read_zarray(path, main_rel) if main_rel else None
            if meta is None or any(list(meta.get(k) or []) != list(spec.get(k) or [])
                                   for k in ("shape", "chunks")) or meta.get("dtype") != spec.get("dtype"):
                continue
            marker = bfill.read_marker(path)
            have = marker.get("written_t", []) if marker else self._complete_ts(path, main_rel, spec)
            found.append((path, main_rel, have, marker is not None))
        if not found:
            return {}
        out: dict[str, Any] = {"elsewhere": [{"path": str(p), "have_t": h, "marked": m} for p, _r, h, m in found]}
        probe = spec.get("probe")
        if probe:
            t = int(probe["t"])
            for path, main_rel, have, _m in found:
                if t not in have:
                    continue
                arr = zarr.open_array(str(bfill.array_dir(path, main_rel)), mode="r")
                if _probe_matches(arr, t, probe):
                    out["duplicate_of"] = str(path)
                    out["duplicate_have_t"] = have
                    break
        return out

    # ---- the live stream's partial copy --------------------------------------------------
    @staticmethod
    def _stream_tag(dest: Path) -> dict | None:
        """The live stream's tags if `dest` is its copy of a run (and not yet marked)."""
        if bfill.read_marker(dest) is not None:
            return None
        try:
            attrs = json.loads((dest / ".zattrs").read_text())
        except (OSError, ValueError):
            return None
        tag = attrs.get(_STREAM_TAG) if isinstance(attrs, dict) else None
        return tag if isinstance(tag, dict) and tag.get("session_id") else None

    def _inspect_store(self, dest: Path, spec: dict, client: str) -> dict[str, Any]:
        if not dest.exists():
            return {"state": bfill.ABSENT, "have_t": []}
        main_rel = bfill.main_array_rel(dest)
        written = list(spec.get("written_t", []))
        marker = bfill.read_marker(dest)
        if marker and set(written) <= set(marker.get("written_t", [])):
            return {"state": bfill.MARKED, "have_t": marker.get("written_t", []), "from": marker.get("from"),
                    "renamed_from": marker.get("renamed_from")}
        if main_rel is None:
            # a store directory with no pixel array yet: ours to create unless it is being written
            return self._guard_busy(dest, None, client) or {"state": bfill.PARTIAL, "have_t": []}
        meta = self._read_zarray(dest, main_rel)
        if (meta is not None and len(meta.get("shape") or []) < len(spec["shape"])
                and (dest / bfill.LOCK_NAME).exists() and not (dest / ".zattrs").exists()):
            # A backfill stopped before the pixel array's own metadata landed (2026-10-25:
            # WT_Unstim_..._001_mScarlet was cut off after its `z` array): the largest array
            # here is a coordinate array, not a different run. Ours to finish.
            return self._guard_busy(dest, None, client) or {"state": bfill.PARTIAL, "have_t": [],
                                                            "backfill_started": True}
        if meta is not None and _is_one_timepoint_of(meta, spec) and self._stream_tag(dest):
            return self._inspect_single_volume(dest, main_rel, meta, spec, client)
        if meta is None or list(meta.get("shape", [])) != list(spec["shape"]) or meta.get("dtype") != spec["dtype"] \
                or list(meta.get("chunks", [])) != list(spec["chunks"]):
            return {"state": bfill.CONFLICT, "reason": _array_difference(meta, spec),
                    "argus_shape": (meta or {}).get("shape"), "argus_dtype": (meta or {}).get("dtype")}
        busy = self._guard_busy(dest, main_rel, client)
        if busy:
            return busy
        have = self._complete_ts(dest, main_rel, spec)
        out: dict[str, Any] = {"state": bfill.PARTIAL, "have_t": have}
        if self._stream_tag(dest):
            out["stream_copy"] = True
        if (dest / bfill.LOCK_NAME).exists():
            # A backfill began this store and never committed it: whole or not, it still
            # needs its readback and marker.
            out["backfill_started"] = True
        return out

    def _inspect_single_volume(self, dest: Path, main_rel: str, meta: dict, spec: dict,
                               client: str) -> dict[str, Any]:
        """The PC saved a single volume as [z, y, x]; the live stream always writes
        [t, z, y, x], so its copy is [1, z, y, x]: the same bytes in the same order. It is
        "already on Argus" once its one timepoint is complete and, with the sender's probe,
        holds the same data. Never written to: the stream's metadata matches its array."""
        busy = self._guard_busy(dest, main_rel, client)
        if busy:
            return busy
        spec4 = {**spec, "shape": meta["shape"], "chunks": meta["chunks"], "written_t": [0]}
        if not self._complete_ts(dest, main_rel, spec4):
            return {"state": bfill.CONFLICT, "reason": "the live stream's copy of this single volume is "
                                                         "incomplete (it is 4-D, the PC's 3-D)"}
        probe = spec.get("probe")
        if not probe:
            return {"state": bfill.PARTIAL, "have_t": [], "stream_copy": True, "needs_probe": True}
        arr = zarr.open_array(str(bfill.array_dir(dest, main_rel)), mode="r")
        if not _probe_matches(arr, 0, probe):
            return {"state": bfill.CONFLICT, "reason": "the live stream's copy of this single volume holds "
                                                         "different data"}
        return {"state": bfill.PARTIAL, "have_t": list(spec.get("written_t", [])), "stream_copy": True,
                "single_volume_as_4d": True}

    def _main(self, dest: Path) -> str | None:
        """The store's pixel-array path, cached: finding it walks the store."""
        rel = self._main_rel_cache.get(dest)
        if rel is None:
            rel = bfill.main_array_rel(dest)
            if rel is not None:
                self._main_rel_cache[dest] = rel
        return rel

    def _read_zarray(self, dest: Path, main_rel: str) -> dict | None:
        import json

        try:
            return json.loads((bfill.array_dir(dest, main_rel) / ".zarray").read_text())
        except (OSError, ValueError):
            return None

    def _guard_busy(self, dest: Path, main_rel: str | None, client: str) -> dict[str, Any] | None:
        """`busy` when something other than our own recent session is writing the store."""
        lock = dest / bfill.LOCK_NAME
        try:
            lock_age = time.time() - lock.stat().st_mtime
            owner = lock.read_text().strip()
        except OSError:
            lock_age, owner = None, ""
        if lock_age is not None and lock_age < LOCK_STALE_S and owner and owner != client:
            return {"state": bfill.BUSY, "reason": f"locked by another backfill session ({owner})"}
        if lock_age is not None and lock_age < LOCK_STALE_S:
            return None  # our own session, still going
        try:
            newest = bfill.newest_mtime(dest, main_rel) if main_rel else dest.stat().st_mtime
        except OSError:
            return None
        if time.time() - newest < self.settle_s:
            return {"state": bfill.BUSY, "reason": "written to within the last "
                                                  f"{int(self.settle_s)} s (a transfer may still be running)"}
        return None

    def _touch_lock(self, dest: Path, client: str) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / bfill.LOCK_NAME).write_text(client)

    # ---- metadata --------------------------------------------------------------------------
    def _meta(self, header: dict) -> dict[str, Any]:
        dest, err = self._store_dir(header)
        if err:
            self._count("rejected")
            return {"kind": "meta", "ok": False, "error": err}
        client = header.get("client", "")
        # re-check ownership: a store another process took since the inventory is not ours
        if dest.exists():
            busy = self._guard_busy(dest, self._main(dest), client)
            if busy:
                return {"kind": "meta", "ok": False, "error": busy["reason"], "state": busy["state"]}
        tag = self._stream_tag(dest) if dest.exists() else None
        main_rel = self._main(dest) if tag else None
        self._touch_lock(dest, client)
        self._started[dest] = time.monotonic()
        logger.info("receiving %s from %s%s", dest, client, " (completing the live stream's copy)" if tag else "")
        self.runs.note(header.get("run_id"), _run_key(header), state="receiving")
        marker = dest / bfill.MARKER_NAME
        if marker.exists():  # about to modify a store that was marked complete: it is not, any more
            marker.unlink()
        for f in header.get("files", []):
            rel, data = f["rel"], f["data"]
            if ".." in Path(rel).parts or Path(rel).is_absolute():
                return {"kind": "meta", "ok": False, "error": f"unsafe path {rel!r}"}
            target = dest / rel
            if rel == ".zattrs":
                if target.exists() and target.read_bytes() != data and not tag:
                    return {"kind": "meta", "ok": False, "error": ".zattrs exists and differs", "state": bfill.CONFLICT}
                self._pending_zattrs[dest] = data   # written last, at commit
                continue
            if target.exists():
                if target.read_bytes() != data:
                    if not tag:
                        return {"kind": "meta", "ok": False, "error": f"{rel} exists and differs",
                                "state": bfill.CONFLICT}
                    # The live stream's copy of this run: its pixel array must mean the same
                    # bytes; its other small files give way to the acquisition's own.
                    if main_rel and rel == f"{main_rel}/.zarray":
                        try:
                            mine, theirs = json.loads(target.read_text()), json.loads(data)
                        except ValueError:
                            mine, theirs = None, {}
                        if mine is None or any(mine.get(k) != theirs.get(k) for k in _PIXEL_KEYS):
                            return {"kind": "meta", "ok": False, "state": bfill.CONFLICT,
                                    "error": "the live stream's copy has a different pixel array"}
                        continue  # same pixels: keep it
                    bfill.atomic_write(target, data)
                    self._adopted.setdefault(dest, {"session_id": tag["session_id"], "replaced": []})[
                        "replaced"].append(rel)
                continue
            bfill.atomic_write(target, data)
            if tag:
                self._adopted.setdefault(dest, {"session_id": tag["session_id"], "replaced": []})
        # Where the pixel array is may have changed (a store cut off before its .zarray landed
        # looked like its coordinate array was the pixel array): find it afresh.
        self._main_rel_cache.pop(dest, None)
        for key in [k for k in self._arrays if k == dest or dest in k.parents]:
            self._arrays.pop(key, None)
        return {"kind": "meta", "ok": True, "store": header["store"]}

    # ---- volumes ---------------------------------------------------------------------------
    def _array(self, dest: Path, main_rel: str) -> zarr.Array:
        key = bfill.array_dir(dest, main_rel)
        arr = self._arrays.get(key)
        if arr is None:
            arr = zarr.open_array(str(key), mode="r+")
            self._arrays[key] = arr
        return arr

    def _volume(self, header: dict, payload: bytes) -> dict[str, Any]:
        base = {"kind": "volume", "store": header.get("store"), "t": header.get("t"), "z0": header.get("z0")}
        dest, err = self._store_dir(header)
        if err:
            self._count("rejected")
            return {**base, "ok": False, "error": err}
        if self.live_active():
            self._count("held")
            return {**base, "ok": False, "hold": "live", "error": "live acquisition active"}
        main_rel = self._main(dest)
        if main_rel is None:
            return {**base, "ok": False, "error": "store has no pixel array: send its metadata first"}
        arr = self._array(dest, main_rel)
        nz, shape_yx = int(header["nz"]), tuple(header["shape_yx"])
        if header.get("codec") == "blosc":
            from numcodecs import blosc

            raw = bytes(blosc.decompress(payload))
        else:
            raw = payload
        slab = np.frombuffer(raw, dtype=np.dtype(arr.dtype)).reshape((nz, *shape_yx))
        if bfill.volume_digest(slab) != header.get("sha256"):
            return {**base, "ok": False, "error": "digest mismatch in transit", "retry": True}
        t, z0 = int(header["t"]), int(header["z0"])
        lock = self._write_lock(bfill.array_dir(dest, main_rel), arr)
        if lock is None:
            _write_slab(arr, t, z0, nz, slab)
        else:
            with lock:
                _write_slab(arr, t, z0, nz, slab)
        (dest / bfill.LOCK_NAME).write_text(header.get("client", ""))
        with self._lock:
            self.stats["volumes"] += 1
            self.stats["bytes"] += slab.nbytes
        return {**base, "ok": True}

    def _write_lock(self, key: Path, arr: zarr.Array) -> threading.Lock | None:
        """None when slabs can be written side by side: a slab is whole planes, so when
        each chunk is one plane (the acquisition writer's layout) no two slabs share a
        chunk. Any other chunking writes one slab at a time per array."""
        chunks, shape = list(arr.chunks), list(arr.shape)
        if all(c == 1 for c in chunks[:-2]) and chunks[-2:] == shape[-2:]:
            return None
        with self._lock:
            return self._write_locks.setdefault(key, threading.Lock())

    # ---- commit ----------------------------------------------------------------------------
    def _commit(self, header: dict) -> dict[str, Any]:
        base = {"kind": "commit", "store": header.get("store")}
        dest, err = self._store_dir(header)
        if err:
            return {**base, "ok": False, "error": err}
        if self.live_active():
            return {**base, "ok": False, "hold": "live", "error": "live acquisition active"}
        main_rel = self._main(dest)
        if main_rel is None:
            return {**base, "ok": False, "error": "store has no pixel array"}
        digests = {int(t): d for t, d in header.get("digests", {}).items()}
        arr = self._array(dest, main_rel)
        run_id, key = header.get("run_id"), _run_key(header)
        t0 = time.monotonic()

        def matches(t: int) -> bool:
            vol = np.asarray(arr[t] if arr.ndim >= 4 else arr[...])
            return bfill.volume_digest(vol) == digests[t]

        ts = sorted(digests)
        bad = [t for t, ok in zip(ts, self._readback_pool.map(matches, ts)) if not ok]
        readback_s = time.monotonic() - t0
        if bad:
            logger.error("readback mismatch in %s: %d of %d timepoint(s) differ (first %s)", dest, len(bad), len(ts),
                         bad[:5])
            self.runs.note(run_id, key, state="readback mismatch", mismatch=bad[:20])
            return {**base, "ok": False, "error": f"{len(bad)} volume(s) differ after readback", "mismatch": bad[:20]}
        zattrs = header.get("root_zattrs") or self._pending_zattrs.get(dest)
        adopted = self._adopted.pop(dest, None)
        tag = self._stream_tag(dest)
        if zattrs is not None and tag:
            # Completing the live stream's copy: the acquisition's attributes, keeping the
            # stream's tags (other code finds the run's session by them).
            try:
                attrs = json.loads(zattrs)
                attrs[_STREAM_TAG] = tag
                zattrs = json.dumps(attrs, indent=4).encode()
            except ValueError:
                pass
            adopted = adopted or {"session_id": tag["session_id"], "replaced": []}
        if zattrs is not None:
            bfill.atomic_write(dest / ".zattrs", zattrs)

        marker = {
            "format": bfill.MARKER_FORMAT,
            "state": "complete",
            "source": "ssh-backfill",
            "host": self.host,
            "client": header.get("client", ""),
            "from": header.get("source", ""),
            "finished_at": time.time(),
            "written_t": sorted(digests),
            "declared_t": header.get("declared_t"),
            "manifest_sha256": bfill.manifest_digest(digests),
            "verified": "readback-sha256",
        }
        if adopted:
            # Completed the live stream's partial copy in place (no second copy).
            marker["completed_stream_copy"] = adopted
        if header.get("renamed_from"):
            # A different run under a name Argus already holds: sent as `<base>_pc<N>_...`.
            marker["renamed_from"] = header["renamed_from"]
        if run_id:
            marker["run_id"] = run_id
        bfill.atomic_write(dest / bfill.MARKER_NAME, json.dumps(marker, indent=1).encode())
        (dest / bfill.LOCK_NAME).unlink(missing_ok=True)
        self._pending_zattrs.pop(dest, None)
        self._arrays.pop(bfill.array_dir(dest, main_rel), None)
        self._main_rel_cache.pop(dest, None)
        self._count("commits")
        started = self._started.pop(dest, None)
        gb = (arr.nbytes / max(arr.shape[0], 1) * len(ts) if arr.ndim >= 4 else arr.nbytes) / 1e9
        logger.info("verified %s: %d timepoint(s), %.2f GB read back in %.0f s%s", dest, len(ts), gb, readback_s,
                    f", {time.monotonic() - started:.0f} s since its metadata" if started else "")
        self.runs.note(run_id, key, state="verified", verified_t=len(ts), readback_s=round(readback_s, 1))
        return {**base, "ok": True, "verified": len(digests), "manifest_sha256": marker["manifest_sha256"]}


def _write_slab(arr: zarr.Array, t: int, z0: int, nz: int, slab: np.ndarray) -> None:
    if arr.ndim >= 4:
        arr[t, z0:z0 + nz] = slab
    else:
        arr[z0:z0 + nz] = slab


def _run_key(header: dict) -> str:
    """Where a store lands, as the sender names it in its run record."""
    return f"{header.get('leaf', '')}/{header.get('store', '')}"


def _probe_matches(arr: zarr.Array, t: int, probe: dict) -> bool:
    """Whether Argus's volume t (or, with `z`, its one plane z) has the probe's SHA-256.
    A plane identifies an acquisition as surely (camera noise) for 1/Z of the reading:
    whole-volume probes made the rig's dry run read ~190 MB per candidate (2026-10-24)."""
    z = probe.get("z")
    if arr.ndim >= 4:
        sel = arr[t] if z is None else arr[t, int(z)]
    else:
        sel = arr[...] if z is None else arr[int(z)]
    return bfill.volume_digest(np.asarray(sel)) == probe["sha256"]


def _is_one_timepoint_of(meta: dict, spec: dict) -> bool:
    """Argus's array is [1, *the sender's shape]: one timepoint of the same volume."""
    shape, chunks = list(meta.get("shape") or []), list(meta.get("chunks") or [])
    want_shape, want_chunks = list(spec.get("shape") or []), list(spec.get("chunks") or [])
    return (len(want_shape) == 3 and shape == [1, *want_shape] and chunks == [1, *want_chunks]
            and meta.get("dtype") == spec.get("dtype"))


def _array_difference(meta: dict | None, spec: dict) -> str:
    """Why Argus's store and the sender's are not the same array, in words."""
    if meta is None:
        return "pixel array differs from the sender's: Argus's store has no readable .zarray"
    parts = []
    for key, label in (("shape", "shape"), ("chunks", "chunks"), ("dtype", "dtype")):
        mine, theirs = meta.get(key), spec.get(key)
        if (list(mine or []) if key != "dtype" else mine) != (list(theirs or []) if key != "dtype" else theirs):
            parts.append(f"{label} on Argus {mine}, on the PC {theirs}")
    shape_a, shape_b = list(meta.get("shape") or []), list(spec.get("shape") or [])
    if len(parts) == 1 and len(shape_a) == len(shape_b) >= 4 and shape_a[1:] == shape_b[1:]:
        parts[0] += " (only the declared number of timepoints differs)"
    elif len(parts) == 1 and len(shape_a) == len(shape_b) >= 4 and shape_a[0] == shape_b[0] \
            and shape_a[2:] == shape_b[2:]:
        parts[0] += " (only the number of z planes differs)"
    return "pixel array differs from the sender's: " + "; ".join(parts)


def _kind(msg_type: bytes) -> str:
    return {bfill.BF_INVENTORY: "inventory", bfill.BF_META: "meta", bfill.BF_VOLUME: "volume",
            bfill.BF_COMMIT: "commit"}.get(msg_type, "error")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with BackfillReceiver(raw_roots=_env_list(_RAW_ROOTS_ENV)) as receiver:
        receiver.run_forever()


if __name__ == "__main__":
    main()
