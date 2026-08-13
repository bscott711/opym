# Ruff style: Compliant
"""
Real-time frame-streaming receiver for opym.

Bridges a ZeroMQ frame stream from an acquisition workstation into the
existing `/dev/shm/petakit_jobs` ticket pipeline (see `opym.petakit`,
`opym.local_gpu_worker`, `opym.consolidate`) -- those modules are unmodified;
this package only produces the same shape of staged zarr + JSON ticket the
batch path already produces, per frame instead of per dataset.

See `opym_local/docs/STREAMING_PROTOCOL.md` for the wire-protocol spec.
"""

from __future__ import annotations

from .protocol import (
    MSG_ACK,
    MSG_FRAME,
    MSG_RESUME,
    MSG_SESSION_END,
    MSG_SESSION_START,
    pack_message,
    unpack_message,
)
from .receiver import StreamReceiver, main

__all__ = [
    "MSG_ACK",
    "MSG_FRAME",
    "MSG_RESUME",
    "MSG_SESSION_END",
    "MSG_SESSION_START",
    "pack_message",
    "unpack_message",
    "StreamReceiver",
    "main",
]
