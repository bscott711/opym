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
import re
import socket
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr
import zmq

from opym.stream import bfill
from opym.stream import bfill_run

logger = logging.getLogger(__name__)

DEFAULT_CONNECT = "tcp://127.0.0.1:5556"
REPLY_TIMEOUT_S = 180.0
PAUSE_POLL_S = 10.0
MAX_RETRIES = 3
# Commit waits for the receiver to read the whole store back; allow it this rate at worst.
COMMIT_READBACK_MBPS = 50.0
# `<prefix>_<Channel>_<wavelength>[nm]`, as opym.discovery groups a run's channel stores.
_CHANNEL_SUFFIX = re.compile(r"^(?P<prefix>.+)_(?P<channel>[A-Za-z]+)_(?P<wavelength>\d+(?:nm)?)$")
MAX_RENAME = 9


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
        runs_dir: Path | None = None,
        verify: bool = True,
    ) -> None:
        self.runs_dir = runs_dir
        # Check copies that came another way (live stream, Globus) against this PC before
        # they count as done; see `_decide`.
        self.verify = verify
        self.runlog: bfill_run.RunLog | None = None
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

    def _request(self, msg_type: bytes, header: dict, expect: str, payload: bytes | None = None,
                 timeout_s: float | None = None) -> dict:
        """One control request; waits for the reply of kind `expect` carrying our echo token."""
        token = uuid.uuid4().hex
        header = {**header, "client": self.client_id, "echo": token}
        if self.runlog is not None:
            header["run_id"] = self.runlog.run_id
        timeout_s = max(timeout_s or 0.0, self.reply_timeout_s)
        deadline = time.monotonic() + timeout_s
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
        raise BackfillError(f"no {expect} reply from the receiver within {timeout_s:.0f} s "
                            "(is the SSH tunnel up?)")

    def _send_run(self, header: dict) -> None:
        """Mirror the run record to the receiver. Its start and end wait for the reply (the
        end is the last thing sent before the sockets close, which drops anything unsent);
        heartbeats and per-store results don't."""
        if "run_log" not in self.features:
            return
        if header.get("op") in ("start", "end"):
            try:
                self._request(bfill.BF_RUN, header, "run")
            except BackfillError as exc:
                logger.warning("the run record did not reach the receiver: %s", exc)
        else:
            self.link.send(bfill.pack(bfill.BF_RUN, self.client_id, {**header, "client": self.client_id}))

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
        """Plan and carry out one folder (what `run_once` does for every folder)."""
        remote, decisions = self.plan_leaf(leaf)
        report: dict = {"leaf": str(leaf.path), "remote": remote, "stores": {}}
        for d in decisions:
            report["stores"][d["store"].name] = self._carry_out(d, dry_run)
        return report

    # ---- planning ------------------------------------------------------------------------
    def plan_leaf(self, leaf: LocalLeaf, skip: set[str] = frozenset()) -> tuple[str, list[dict]]:
        """What to do with each store of `leaf`, decided from Argus's inventory: one
        decision per store, `action` in PLAN_ACTIONS. Sends nothing but questions."""
        remote = self.remote_leaf(leaf.path)
        decisions: list[dict] = []
        ready = []
        for s in leaf.stores:
            d = {"store": s, "remote": remote, "name": s.name}
            age = time.time() - s.newest_mtime()
            if str(s.path) in skip:
                decisions.append({**d, "action": "skip", "result": "already complete", "reason": "previous run"})
            elif age < self.settle_s:
                decisions.append({**d, "action": "wait", "result": "settling", "idle_s": int(age)})
            elif not s.written_t:
                decisions.append({**d, "action": "wait", "result": "empty"})
            elif len(s.shape) < 3:
                decisions.append({**d, "action": "report", "result": "no pixel data",
                                  "reason": f"no image array on this PC (largest is {s.shape} {s.dtype})"})
            else:
                ready.append(s)
        if not ready:
            return remote, decisions
        inv = self._request(bfill.BF_INVENTORY, {"leaf": remote, "stores": [s.spec() for s in ready]}, "inventory")
        if not inv.get("ok"):
            raise BackfillError(f"inventory refused for {remote}: {inv.get('error')}")
        infos = inv["stores"]
        self._probe_elsewhere(remote, ready, infos)
        for s in ready:
            decisions.append(self._decide(remote, s, infos.get(s.name, {})))
        self._resolve_renames(remote, decisions)
        return remote, decisions

    def _decide(self, remote: str, s: LocalStore, info: dict) -> dict:
        d: dict[str, Any] = {"store": s, "remote": remote, "name": s.name}
        state = info.get("state")
        have = set(info.get("have_t", []))
        if state == bfill.MARKED:
            return {**d, "action": "skip", "result": "already complete"}
        if state == bfill.BUSY:
            return {**d, "action": "report", "result": state, "reason": info.get("reason")}
        if state == bfill.CONFLICT:
            return {**d, "action": "report", "result": state, "reason": str(info.get("reason") or ""),
                    "different_run": _different_run(s, info)}
        if info.get("duplicate_of"):
            # The same data is already on Argus under another folder: never a second copy.
            # A partial one there is completed in place.
            at = info["duplicate_of"]
            dup_have = set(info.get("duplicate_have_t", []))
            elsewhere = {e.get("path"): e for e in info.get("elsewhere") or []}
            if set(s.written_t) <= dup_have:
                if not self.verify or elsewhere.get(at, {}).get("marked"):
                    return {**d, "action": "skip", "result": "already on Argus", "at": at}
                return {**d, "action": "verify elsewhere", "remote": str(Path(at).parent), "have": dup_have, "at": at}
            return {**d, "action": "complete elsewhere", "remote": str(Path(at).parent), "have": dup_have, "at": at}
        if set(s.written_t) <= have and not info.get("backfill_started"):
            # The live stream (or Globus) brought it in whole: nothing to send. Its chunk files
            # are all there, which says nothing about their contents, so unless told not to it
            # is checked like any backfilled store: this PC's per-volume SHA-256 against Argus's
            # readback, then the completion marker (2026-10-25: a faithful copy of S:\ is the
            # point, and 118 such stores had never been compared with it).
            if not self.verify:
                return {**d, "action": "skip", "result": "already on Argus"}
            # A single volume the stream wrote 4-D: its metadata is the stream's own and must
            # stay, so only the pixels are checked.
            return {**d, "action": "verify", "have": have, "pixels_only": bool(info.get("single_volume_as_4d"))}
        action = "complete stream copy" if info.get("stream_copy") else "send"
        return {**d, "action": action, "have": have}

    def _resolve_renames(self, remote: str, decisions: list[dict]) -> None:
        """A different run under a name Argus already holds (the acquisition software
        reused it) is sent as `<prefix>_pc<N>_<channel>`, every channel of the run with
        the same N so they stay one dataset; N is the first that is free, or that holds
        this very store from an earlier run. Updates `decisions` in place."""
        groups: dict[str, list[dict]] = {}
        for d in decisions:
            m = _CHANNEL_SUFFIX.match(d["name"].removesuffix(".ome.zarr"))
            groups.setdefault(m.group("prefix") if m else d["name"], []).append(d)
        for prefix, members in groups.items():
            differ = [d for d in members if d.get("different_run")]
            if not differ:
                continue
            if len(differ) != len(members):
                for d in differ:
                    d["reason"] += ("; not renamed: other channels of this run are not conflicts, "
                                    "so a person has to decide")
                continue
            if "marker_from" not in self.features:
                continue  # an older receiver can't say whether a renamed copy is this run
            for n in range(2, MAX_RENAME + 1):
                names = {}
                for d in differ:
                    m = _CHANNEL_SUFFIX.match(d["name"].removesuffix(".ome.zarr"))
                    tail = f"{m.group('channel')}_{m.group('wavelength')}" if m else ""
                    names[d["name"]] = f"{prefix}_pc{n}_{tail}.ome.zarr" if tail else \
                        f"{d['name'].removesuffix('.ome.zarr')}_pc{n}.ome.zarr"
                specs = [{**d["store"].spec(), "name": names[d["name"]]} for d in differ]
                inv = self._request(bfill.BF_INVENTORY, {"leaf": remote, "stores": specs}, "inventory")
                if not inv.get("ok"):
                    break
                infos = {d["name"]: inv["stores"].get(names[d["name"]], {}) for d in differ}
                source = f"{socket.gethostname()}:"
                usable = all(
                    i.get("state") == bfill.ABSENT
                    or (i.get("state") == bfill.PARTIAL and not i.get("stream_copy") and not i.get("elsewhere"))
                    or (i.get("state") == bfill.MARKED and i.get("from") == f"{source}{d['store'].path}")
                    for d, i in ((d, infos[d["name"]]) for d in differ))
                if not usable:
                    continue
                for d in differ:
                    i, new = infos[d["name"]], names[d["name"]]
                    d.update(name=new, renamed_from=d["name"], reason=None, different_run=False)
                    if i.get("state") == bfill.MARKED:
                        d.update(action="skip", result="already on Argus", at=f"{remote}/{new}")
                    else:
                        d.update(action="send renamed", result=None, have=set(i.get("have_t", [])))
                break

    def plan_entry(self, d: dict) -> dict:
        """The decision as the run record lists it."""
        s = d["store"]
        per_t = math.prod(s.shape[-3:]) * np.dtype(s.dtype).itemsize if len(s.shape) >= 3 else 0
        send_t = len([t for t in s.written_t if t not in d.get("have", set(s.written_t))])
        verify_t = len(s.written_t) if d["action"] in ("send", "complete stream copy", "complete elsewhere",
                                                       "send renamed", "verify", "verify elsewhere") else 0
        entry = {"key": f"{d['remote']}/{d['name']}", "leaf": d["remote"], "name": d["name"], "local": str(s.path),
                 "action": d["action"], "timepoints": len(s.written_t), "send_t": send_t, "bytes": send_t * per_t,
                 "verify_t": verify_t, "verify_bytes": verify_t * per_t}
        for k in ("result", "reason", "renamed_from", "at"):
            if d.get(k):
                entry[k] = d[k]
        return entry

    # ---- carrying it out -----------------------------------------------------------------
    def _carry_out(self, d: dict, dry_run: bool) -> dict:
        """The store's report entry (what used to be computed in `sync_leaf`)."""
        s, action = d["store"], d["action"]
        out: dict[str, Any]
        if action in ("skip", "wait", "report"):
            out = {k: d[k] for k in ("result", "reason", "at", "idle_s") if d.get(k) is not None}
            if d.get("renamed_from"):
                out["as"] = d["name"]
        elif dry_run:
            missing = len([t for t in s.written_t if t not in d.get("have", set())])
            result = {"send": "would send", "complete stream copy": "would send",
                      "complete elsewhere": "would complete elsewhere", "send renamed": "would send renamed",
                      "verify": "would verify", "verify elsewhere": "would verify"}[action]
            out = {"result": result, "missing_timepoints": missing, "of": len(s.written_t)}
            if action == "complete stream copy":
                out["completes_stream_copy"] = True
            if d.get("at"):
                out["at"] = d["at"]
            if d.get("renamed_from"):
                out["as"] = d["name"]
        else:
            try:
                out = self._sync_store(d["remote"], s, d.get("have", set()), name=d["name"],
                                       renamed_from=d.get("renamed_from"), meta=not d.get("pixels_only"))
                if action.startswith("verify"):
                    out["checked_only"] = True
            except BackfillError as exc:
                # One refused store must not hide what the others did.
                logger.error("%s: %s", s.name, exc)
                out = {"result": "refused", "reason": str(exc)}
            if d.get("at"):
                out["at"] = d["at"]
            if d.get("renamed_from"):
                out["as"] = d["name"]
        if self.runlog is not None and not dry_run:
            self.runlog.store(f"{d['remote']}/{d['name']}", out)
        return out

    def _probe_elsewhere(self, remote: str, stores: list[LocalStore], infos: dict) -> None:
        """For stores Argus has under another folder, send one volume's digest so it can tell
        the same data from a different run that shares the name. Updates `infos` in place."""
        probes = []
        for s in stores:
            info = infos.get(s.name, {})
            cands = info.get("elsewhere") or []
            t = next((t for t in s.written_t if any(t in c.get("have_t", []) for c in cands)), None)
            if info.get("needs_probe") and s.written_t:
                t = s.written_t[0]  # e.g. the stream's 4-D copy of a single 3-D volume
            if t is None:
                continue
            probes.append({**s.spec(), "probe": self._probe_for(s, t)})
        if not probes:
            return
        inv = self._request(bfill.BF_INVENTORY, {"leaf": remote, "stores": probes}, "inventory")
        if inv.get("ok"):
            infos.update(inv["stores"])

    def _probe_for(self, s: LocalStore, t: int) -> dict:
        """The SHA-256 Argus compares with its copy: of one plane that is on disk here (the
        middle one) when Argus takes plane probes, else of the whole volume t."""
        arr = s.array()
        if "plane_probe" in self.features:
            nz = s.shape[1] if len(s.shape) >= 4 else s.shape[0]
            present = s.planes_present(t, nz) if len(s.shape) >= 4 else list(range(nz))
            if present:
                z = present[len(present) // 2]
                plane = np.asarray(arr[t, z] if arr.ndim >= 4 else arr[z])
                return {"t": t, "z": z, "sha256": bfill.volume_digest(plane)}
        vol = np.asarray(arr[t] if arr.ndim >= 4 else arr[...])
        return {"t": t, "sha256": bfill.volume_digest(vol)}

    def _sync_store(self, remote: str, s: LocalStore, have: set[int], name: str | None = None,
                    renamed_from: str | None = None, meta: bool = True) -> dict:
        name = name or s.name
        t_start = time.monotonic()
        before = self.sent_bytes
        if meta:
            self._send_meta(remote, s, name)
        arr = s.array()
        digests: dict[int, str] = {}
        todo = [t for t in s.written_t if t not in have]
        pending = self._slab_stream(remote, s, arr, todo, digests, name)
        self._pump(pending)
        for t in s.written_t:                       # volumes the server already held: hash locally
            if t not in digests:
                digests[t] = bfill.volume_digest(np.asarray(arr[t] if arr.ndim >= 4 else arr[...]))
        root_zattrs = (s.path / ".zattrs").read_bytes() if meta and (s.path / ".zattrs").exists() else None
        # The receiver reads every volume back before it answers (~19 GB for a 100-timepoint run).
        readback_s = len(digests) * math.prod(s.shape[-3:]) * np.dtype(s.dtype).itemsize / (COMMIT_READBACK_MBPS * 1e6)
        commit = {"leaf": remote, "store": name, "digests": {str(t): d for t, d in digests.items()},
                  "declared_t": s.shape[0] if len(s.shape) >= 4 else 1, "root_zattrs": root_zattrs,
                  "source": f"{socket.gethostname()}:{s.path}"}
        if renamed_from:
            commit["renamed_from"] = renamed_from
        while True:
            self.wait_if_paused()
            if self.runlog is not None:
                self.runlog.heartbeat(f"{remote}/{name}", None, force=True)
            reply = self._request(bfill.BF_COMMIT, commit, "commit", timeout_s=readback_s)
            if reply.get("hold"):
                self.live_active = True
                continue
            break
        if not reply.get("ok"):
            raise BackfillError(f"{name}: commit failed: {reply.get('error')} {reply.get('mismatch', '')}")
        sent = self.sent_bytes - before
        secs = max(time.monotonic() - t_start, 1e-6)
        return {"result": "complete", "timepoints": len(s.written_t), "sent_timepoints": len(todo),
                "sent_gb": round(sent / 1e9, 3), "mb_per_s": round(sent / secs / 1e6, 1),
                "verified": reply.get("verified")}

    def _send_meta(self, remote: str, s: LocalStore, name: str) -> None:
        files = []
        for rel in bfill.meta_files(s.path, s.main_rel):
            files.append({"rel": rel, "data": (s.path / rel).read_bytes()})
        reply = self._request(bfill.BF_META, {"leaf": remote, "store": name, "files": files}, "meta")
        if not reply.get("ok"):
            raise BackfillError(f"{name}: metadata refused: {reply.get('error')}")

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

    def _slab_stream(self, remote: str, s: LocalStore, arr: zarr.Array, todo: list[int], digests: dict[int, str],
                     name: str):
        for t in todo:
            if self.runlog is not None:
                self.runlog.heartbeat(f"{remote}/{name}", t)
            vol = np.asarray(arr[t] if arr.ndim >= 4 else arr[...])
            digests[t] = bfill.volume_digest(vol)
            for z0, nz in self._slab_runs(s, t, vol.shape[0]):
                slab = vol[z0:z0 + nz]
                header = {"leaf": remote, "store": name, "t": t, "z0": z0, "nz": nz,
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
    def run_once(self, dry_run: bool = False, resume: dict | None = None) -> list[dict]:
        """Scan, plan every folder, record the plan, then carry it out. `resume`: a previous
        run's record (`bfill_run.load_run`); stores it finished are not looked at again."""
        self.ping()
        leaves: list[LocalLeaf] = []
        for local_root, _ in self.root_map:
            leaves += scan_local_root(local_root)
        leaves.sort(key=lambda lf: max(s.newest_mtime() for s in lf.stores), reverse=self.order == "newest")
        skip = bfill_run.done_locals(resume)
        runlog = bfill_run.RunLog(self.runs_dir, client=self.client_id, dry_run=dry_run,
                                  maps=[(str(a), b) for a, b in self.root_map], send=self._send_run)
        self.runlog = runlog
        try:
            plans: list[tuple[LocalLeaf, str | None, list[dict], str | None]] = []
            for leaf in leaves:
                self.wait_if_paused()
                try:
                    remote, decisions = self.plan_leaf(leaf, skip)
                    plans.append((leaf, remote, decisions, None))
                except BackfillError as exc:
                    logger.error("%s: %s", leaf.path, exc)
                    plans.append((leaf, None, [], str(exc)))
            runlog.plan([self.plan_entry(d) for _leaf, _r, ds, _e in plans for d in ds])
            reports = []
            for leaf, remote, decisions, error in plans:
                if error is not None:
                    reports.append({"leaf": str(leaf.path), "error": error})
                    continue
                rep = {"leaf": str(leaf.path), "remote": remote, "stores": {}}
                for d in decisions:
                    self.wait_if_paused()
                    rep["stores"][d["store"].name] = self._carry_out(d, dry_run)
                logger.info("%s", json.dumps(rep))
                reports.append(rep)
            if not dry_run:
                self._retry_refused(plans, reports)
            runlog.end("complete")
            return reports
        except BaseException as exc:
            runlog.end("failed", error=repr(exc))
            raise
        finally:
            self.runlog = None

    def _retry_refused(self, plans: list, reports: list[dict]) -> None:
        """Once more, at the end, for each store that was refused (a tunnel that dropped,
        a receiver restarted): re-planned, so it resumes from what Argus now holds."""
        for (leaf, _remote, decisions, error), rep in zip(plans, reports):
            if error is not None:
                continue
            failed = {d["store"].name for d in decisions if rep["stores"][d["store"].name].get("result") == "refused"}
            if not failed:
                continue
            retry_leaf = LocalLeaf(leaf.path, [s for s in leaf.stores if s.name in failed])
            try:
                _remote, again = self.plan_leaf(retry_leaf)
            except BackfillError as exc:
                logger.error("%s: retry: %s", leaf.path, exc)
                continue
            for d in again:
                self.wait_if_paused()
                out = self._carry_out(d, dry_run=False)
                out["retried"] = True
                rep["stores"][d["store"].name] = out


def _different_run(s: LocalStore, info: dict) -> bool:
    """Whether Argus's store of this name is certainly another acquisition: its volumes
    have another z/y/x size or pixel type (one run can't change those). Differing only in
    the declared number of timepoints may be the same run, so that stays a conflict."""
    shape, dtype = info.get("argus_shape"), info.get("argus_dtype")
    if not shape or len(shape) < 3 or len(s.shape) < 3:
        return False
    return list(shape[-3:]) != list(s.shape[-3:]) or dtype != s.dtype


def summary_counts(reports: list[dict]) -> Counter:
    return Counter(st.get("result") for rep in reports for st in rep.get("stores", {}).values())


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
    p.add_argument("--runs-dir", type=Path, default=bfill_run.default_runs_dir(),
                   help="where this run's record is kept (default %(default)s)")
    p.add_argument("--resume", nargs="?", const="latest", default=None, metavar="RUN_ID",
                   help="skip what the latest (or the named) run already finished")
    p.add_argument("--no-verify", dest="verify", action="store_false",
                   help="trust copies the live stream or Globus made instead of checking them against this PC")
    a = p.parse_args(argv)
    client = BackfillClient(a.connect or [DEFAULT_CONNECT], a.maps, settle_s=a.settle, window=a.window,
                            max_mbps=a.max_mbps, pause_file=a.pause_file, order=a.order, runs_dir=a.runs_dir, verify=a.verify)
    try:
        while True:
            try:
                resume = bfill_run.load_run(a.runs_dir, a.resume) if a.resume else None
                client.run_once(dry_run=a.dry_run, resume=resume)
            except BackfillError as exc:
                logger.error("%s", exc)
            if a.watch is None:
                return 0
            time.sleep(a.watch)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
