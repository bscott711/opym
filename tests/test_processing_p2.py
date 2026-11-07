# Ruff style: Compliant
"""Processing P2s (docs/review/2026-11-review.md PROC-F3, F4, F9, F10, F11, F12).

CPU only: tickets are written to a tmp queue and read back; MATLAB and the GPU
are never involved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from opym import petakit


# --------------------------------------------------------------------------
# PROC-F3: a ticket is never visible half-written
# --------------------------------------------------------------------------


def test_ticket_is_published_atomically(tmp_path, monkeypatch):
    """The server claims the oldest non-dot `*.json`; at the moment the final
    name appears it must already hold the whole ticket."""
    queue = tmp_path / "queue"
    queue.mkdir()
    seen = {}
    real_replace = os.replace

    def spy(src, dst):
        src, dst = Path(src), Path(dst)
        seen["src_name"] = src.name
        seen["dst_existed"] = dst.exists()
        seen["visible_json"] = [
            p.name for p in queue.glob("*.json") if not p.name.startswith(".")
        ]
        seen["body"] = json.loads(src.read_text())
        real_replace(src, dst)

    monkeypatch.setattr(petakit.os, "replace", spy)
    job = petakit._write_ticket({"jobType": "x", "n": 1}, "name", "LIVE", queue)

    assert seen["src_name"].startswith(".") and seen["src_name"].endswith(".tmp")
    assert not seen["dst_existed"]
    assert seen["visible_json"] == [], "the ticket was claimable before it was whole"
    assert seen["body"]["n"] == 1
    assert json.loads(job.read_text())["jobType"] == "x"
    assert list(queue.iterdir()) == [job], "no temp file left behind"


def test_failed_ticket_write_leaves_no_file(tmp_path):
    queue = tmp_path / "queue"
    queue.mkdir()
    with pytest.raises(TypeError):
        petakit._write_ticket({"bad": object()}, "name", "LIVE", queue)
    assert list(queue.iterdir()) == []
