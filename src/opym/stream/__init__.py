# Ruff style: Compliant
"""
Real-time frame-streaming receiver for opym.

Bridges a ZeroMQ frame stream from an acquisition workstation into the
existing `/dev/shm/petakit_jobs` ticket pipeline (see `opym.petakit`,
`opym.local_gpu_worker`, `opym.consolidate`) -- those modules are unmodified;
this package only produces the same shape of staged zarr + JSON ticket the
batch path already produces, per frame instead of per dataset.

See `opym_local/docs/STREAMING_PROTOCOL.md` for the wire-protocol spec.

The package namespace is lazy (`from opym.stream import StreamReceiver` and the
`MSG_*` / `pack_message` names still work): `import opym.stream.protocol` must
not drag in the receiver's zmq / numcodecs / zarr stack. See
`docs/adr/0001-light-base.md`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # static analysis / IDEs only; never executed
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

# Public name -> submodule that defines it.
_LAZY: dict[str, str] = {
    "MSG_ACK": "protocol",
    "MSG_FRAME": "protocol",
    "MSG_RESUME": "protocol",
    "MSG_SESSION_END": "protocol",
    "MSG_SESSION_START": "protocol",
    "pack_message": "protocol",
    "unpack_message": "protocol",
    "StreamReceiver": "receiver",
    "main": "receiver",
}

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


def __getattr__(name: str) -> Any:
    submodule = _LAZY.get(name)
    if submodule is not None:
        value = getattr(importlib.import_module(f".{submodule}", __name__), name)
        globals()[name] = value  # cache: later lookups skip __getattr__
        return value
    if not name.startswith("_"):
        # `opym.stream.live`, ... as attribute access, as the eager package allowed
        # for the submodules it happened to import.
        try:
            return importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as exc:
            if exc.name != f"{__name__}.{name}":
                raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
