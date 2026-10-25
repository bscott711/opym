"""Argus's copy of each SSH backfill run's record (see `bfill_run.py` for the document).

The sender mirrors its record here (`BF_RUN`); the receiver adds what only it sees: when
it last heard from the run, and each store's metadata, write and verified commit. A run
that stops reporting for `STALE_S` is marked `interrupted`, with the store it was on, so
a run that dies (2026-10-24: a 2 h stall at 22:54, then silence) still says where it
stopped. Files: `<runs dir>/<run_id>.json`, read by opym-dashboard.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RUNS_DIR_ENV = "OPYM_BACKFILL_RUNS_DIR"
STALE_S = 600.0
SWEEP_S = 60.0
_RUN_ID = re.compile(r"^[\w.\-]{1,120}$")


def default_runs_dir(raw_roots: list[Path]) -> Path:
    env = os.environ.get(RUNS_DIR_ENV)
    return Path(env) if env else Path(raw_roots[0]) / ".opym_backfill" / "runs"


class RunRecords:
    """Thread-safe: the receiver updates records from its request workers."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self._lock = threading.Lock()
        self._docs: dict[str, dict[str, Any]] = {}
        self._last_sweep = 0.0
        self.sweep(force=True)

    def _path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def _load(self, run_id: str) -> dict[str, Any] | None:
        doc = self._docs.get(run_id)
        if doc is None:
            try:
                doc = json.loads(self._path(run_id).read_text())
            except (OSError, ValueError):
                return None
            self._docs[run_id] = doc
        return doc

    def _save(self, doc: dict[str, Any]) -> None:
        path = self._path(doc["run_id"])
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.part")
            tmp.write_text(json.dumps(doc, indent=1))
            os.replace(tmp, path)
        except OSError:
            logger.exception("could not write the run record %s", path)

    # ---- from the sender --------------------------------------------------------------------
    def apply(self, header: dict[str, Any]) -> dict[str, Any]:
        run_id = str(header.get("run_id", ""))
        if not _RUN_ID.match(run_id):
            return {"kind": "run", "ok": False, "error": "bad run_id"}
        op = header.get("op")
        now = time.time()
        with self._lock:
            if op in ("start", "end"):
                doc = dict(header.get("doc") or {})
                prev = self._load(run_id) or {}
                doc["run_id"] = run_id
                # what the receiver recorded itself survives the sender's copy
                for key, st in (prev.get("stores") or {}).items():
                    if "argus" in st:
                        doc.setdefault("stores", {}).setdefault(key, {})["argus"] = st["argus"]
            else:
                doc = self._load(run_id)
                if doc is None:
                    return {"kind": "run", "ok": False, "error": "unknown run (no start)"}
                if op == "store":
                    key = str(header.get("key", ""))
                    entry = dict(header.get("entry") or {})
                    argus = (doc.get("stores", {}).get(key) or {}).get("argus")
                    if argus:
                        entry["argus"] = argus
                    doc.setdefault("stores", {})[key] = entry
                    doc["current"] = None
                elif op == "heartbeat":
                    doc["current"] = header.get("current")
                else:
                    return {"kind": "run", "ok": False, "error": f"unknown op {op!r}"}
            doc["argus_last_seen"] = now
            if doc.get("state") == "interrupted" and op != "end":
                doc["state"] = "running"  # it came back
            self._docs[run_id] = doc
            self._save(doc)
        if op == "start":
            logger.info("run %s from %s: %d store(s) planned%s", run_id, doc.get("client"),
                        len(doc.get("plan") or []), " (dry run)" if doc.get("dry_run") else "")
        elif op == "end":
            logger.info("run %s ended %s: %s", run_id, doc.get("state"), doc.get("counts"))
        elif op == "store":
            entry = doc["stores"][str(header.get("key"))]
            logger.info("run %s: %s -> %s%s", run_id, header.get("key"), entry.get("result"),
                        f" ({entry['reason']})" if entry.get("reason") else "")
        return {"kind": "run", "ok": True}

    # ---- what only the receiver sees ----------------------------------------------------------
    def note(self, run_id: str | None, key: str, **argus: Any) -> None:
        """Record the receiver's own view of one store of a run (no-op without a run)."""
        if not run_id or not _RUN_ID.match(str(run_id)):
            return
        with self._lock:
            doc = self._load(str(run_id))
            if doc is None:
                return
            st = doc.setdefault("stores", {}).setdefault(key, {})
            st["argus"] = {**(st.get("argus") or {}), **argus, "at": time.time()}
            doc["argus_last_seen"] = time.time()
            self._save(doc)

    def sweep(self, force: bool = False) -> int:
        """Mark runs that stopped reporting `interrupted`. Returns how many it marked."""
        now = time.time()
        if not force and now - self._last_sweep < SWEEP_S:
            return 0
        self._last_sweep = now
        marked = 0
        try:
            paths = list(self.runs_dir.glob("*.json"))
        except OSError:
            return 0
        with self._lock:
            for p in paths:
                run_id = p.stem
                doc = self._docs.get(run_id)
                if doc is None:
                    try:
                        doc = json.loads(p.read_text())
                    except (OSError, ValueError):
                        continue
                if doc.get("state") not in ("planning", "running"):
                    self._docs.pop(run_id, None)  # finished: no need to keep it in memory
                    continue
                self._docs[run_id] = doc
                seen = float(doc.get("argus_last_seen") or doc.get("updated_at") or 0)
                if now - seen > STALE_S:
                    doc["state"] = "interrupted"
                    doc["interrupted_at"] = seen
                    doc["interrupted_during"] = doc.get("current")
                    self._save(doc)
                    marked += 1
                    logger.warning("run %s: no word from the sender for %.0f min, marked interrupted (during %s)",
                                   run_id, (now - seen) / 60, doc.get("current"))
        return marked
