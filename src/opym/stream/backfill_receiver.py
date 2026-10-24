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
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr
import zmq

from opym import lanes
from opym.stream import bfill

logger = logging.getLogger(__name__)

DEFAULT_BIND = "tcp://127.0.0.1:5556"
_RAW_ROOTS_ENV = "OPYM_BACKFILL_RAW_ROOTS"
_BIND_ENV = "OPYM_BACKFILL_BIND"
_SETTLE_ENV = "OPYM_BACKFILL_SETTLE_SECONDS"
_REGISTRY_ENV = "OPYM_BACKFILL_REGISTRY_PATH"
INDEX_TTL_S = 600.0
# What the live stream's store tagging looks like (opym.stream.rawmirror).
_STREAM_TAG = "opym"
# Pixel-array .zarray keys that decide what the chunk files mean.
_PIXEL_KEYS = ("shape", "chunks", "dtype", "compressor", "filters", "order", "dimension_separator")
DEFAULT_SETTLE_S = 900.0
LOCK_STALE_S = 600.0


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

    def run_forever(self) -> None:
        logger.info("backfill receiver on %s, raw roots %s", self.endpoint, [str(r) for r in self.raw_roots])
        while True:
            self._run_once(1000)

    def _run_once(self, timeout_ms: int = 100) -> bool:
        if not self._sock.poll(timeout_ms):
            return False
        frames = self._sock.recv_multipart()
        identity, parts = frames[0], frames[1:]
        try:
            msg_type, sid, header, payload = bfill.unpack(parts)
        except Exception:  # noqa: BLE001 - malformed input is dropped, never answered
            self.stats["rejected"] += 1
            logger.warning("dropped a malformed message")
            return True
        try:
            reply = self._dispatch(msg_type, header, payload)
        except Exception as exc:  # noqa: BLE001 - one bad request must not stop the receiver
            logger.exception("request failed")
            reply = {"kind": _kind(msg_type), "ok": False, "error": f"server error: {exc}"}
        reply.setdefault("live_active", self.live_active())
        reply["server_time_s"] = time.time()
        reply["features"] = bfill.FEATURES
        if "echo" in header:
            reply["echo"] = header["echo"]
        self._sock.send_multipart([identity, *bfill.pack(bfill.BF_ACK, sid, reply)])
        return True

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
        self.stats["rejected"] += 1
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
            self.stats["rejected"] += 1
            return {"kind": "inventory", "ok": False, "error": err}
        out: dict[str, Any] = {}
        for spec in header.get("stores", []):
            name = spec.get("name", "")
            if not bfill.valid_store_name(name):
                out[name] = {"state": bfill.CONFLICT, "reason": "bad store name"}
                continue
            info = self._inspect_store(leaf / name, spec, header.get("client", ""))
            if info["state"] == bfill.ABSENT:
                info.update(self._elsewhere(leaf / name, spec))
            out[name] = info
        return {"kind": "inventory", "ok": True, "stores": out}

    # ---- the same data elsewhere ---------------------------------------------------------
    def _store_index(self) -> dict[str, list[Path]]:
        """Store name -> every `*.ome.zarr` with that name in the registry's folders."""
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
        per_t = bfill.expected_chunks_per_t(spec["shape"], spec["chunks"]) if len(spec["shape"]) >= 4 else None
        have = []
        for t in spec.get("written_t", []):
            if per_t is None:
                continue
            n, smallest = bfill.chunk_count_for_t(dest, main_rel, t)
            if n == per_t and smallest > 0:
                have.append(t)
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
                vol = np.asarray(arr[t] if arr.ndim >= 4 else arr[...])
                if bfill.volume_digest(vol) == probe["sha256"]:
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
            return {"state": bfill.MARKED, "have_t": marker.get("written_t", [])}
        if main_rel is None:
            # a store directory with no pixel array yet: ours to create unless it is being written
            return self._guard_busy(dest, None, client) or {"state": bfill.PARTIAL, "have_t": []}
        meta = self._read_zarray(dest, main_rel)
        if meta is None or list(meta.get("shape", [])) != list(spec["shape"]) or meta.get("dtype") != spec["dtype"] \
                or list(meta.get("chunks", [])) != list(spec["chunks"]):
            return {"state": bfill.CONFLICT, "reason": "pixel array differs from the sender's"}
        busy = self._guard_busy(dest, main_rel, client)
        if busy:
            return busy
        per_t = bfill.expected_chunks_per_t(spec["shape"], spec["chunks"]) if len(spec["shape"]) >= 4 else None
        have = []
        for t in written:
            if per_t is None:
                n = sum(1 for _ in bfill.array_dir(dest, main_rel).iterdir() if not _.name.startswith("."))
                if n:
                    have.append(t)
                continue
            n, smallest = bfill.chunk_count_for_t(dest, main_rel, t)
            if n == per_t and smallest > 0:
                have.append(t)
        out: dict[str, Any] = {"state": bfill.PARTIAL, "have_t": have}
        if self._stream_tag(dest):
            out["stream_copy"] = True
        return out

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
            self.stats["rejected"] += 1
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
            self.stats["rejected"] += 1
            return {**base, "ok": False, "error": err}
        if self.live_active():
            self.stats["held"] += 1
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
        if arr.ndim >= 4:
            arr[t, z0:z0 + nz] = slab
        else:
            arr[z0:z0 + nz] = slab
        (dest / bfill.LOCK_NAME).write_text(header.get("client", ""))
        self.stats["volumes"] += 1
        self.stats["bytes"] += slab.nbytes
        return {**base, "ok": True}

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
        bad = []
        for t in sorted(digests):
            vol = np.asarray(arr[t] if arr.ndim >= 4 else arr[...])
            if bfill.volume_digest(vol) != digests[t]:
                bad.append(t)
        if bad:
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
        bfill.atomic_write(dest / bfill.MARKER_NAME, json.dumps(marker, indent=1).encode())
        (dest / bfill.LOCK_NAME).unlink(missing_ok=True)
        self._pending_zattrs.pop(dest, None)
        self._arrays.pop(bfill.array_dir(dest, main_rel), None)
        self._main_rel_cache.pop(dest, None)
        self.stats["commits"] += 1
        return {**base, "ok": True, "verified": len(digests), "manifest_sha256": marker["manifest_sha256"]}


def _kind(msg_type: bytes) -> str:
    return {bfill.BF_INVENTORY: "inventory", bfill.BF_META: "meta", bfill.BF_VOLUME: "volume",
            bfill.BF_COMMIT: "commit"}.get(msg_type, "error")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with BackfillReceiver(raw_roots=_env_list(_RAW_ROOTS_ENV)) as receiver:
        receiver.run_forever()


if __name__ == "__main__":
    main()
