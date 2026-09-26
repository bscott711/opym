# Ruff style: Compliant
"""Receiving end of the acquisition-PC link benchmark.

`pymmcore_gui._argus_stream.linkbench` (run on the acquisition PC) pushes
data through one or more SSH tunnels to this sink and times what it
confirms. The sink is deliberately trivial so the numbers measure the
links, not the receiver: every DATA message is answered with a receipt
carrying its size, PING with PONG, and RESULT records are appended to
`<jobs dir>/profiling/linkbench.jsonl` together with the sink's own view
of each link (bytes and seconds between its first and last message).

It binds loopback only, like `opym-receive`, and is started by hand for a
test window, from the bioimaging checkout:

    systemd-run --user --unit opym-linkbench \
        uv run python -m opym.stream.linkbench

Messages (ZMQ multipart, after the ROUTER identity):
    [b"PING", seq]            -> [b"PONG", seq]
    [b"DATA", seq, payload]   -> [b"RCPT", seq, int64 payload length]
    [b"RESULT", json]         -> [b"OK"]
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import time
from pathlib import Path

import zmq

from opym import lanes

logger = logging.getLogger(__name__)

DEFAULT_BIND = "tcp://127.0.0.1:5599"


class _LinkStats:
    __slots__ = ("bytes", "first", "last", "messages")

    def __init__(self) -> None:
        self.bytes = 0
        self.messages = 0
        self.first = 0.0
        self.last = 0.0


def results_path() -> Path:
    return lanes.jobs_dir() / "profiling" / "linkbench.jsonl"


def serve(bind: str = DEFAULT_BIND, out: Path | None = None) -> None:
    out = out or results_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    sock = zmq.Context.instance().socket(zmq.ROUTER)
    sock.bind(bind)
    stats: dict[bytes, _LinkStats] = {}
    logger.info("Link benchmark sink on %s, results -> %s", bind, out)
    try:
        while True:
            identity, kind, *rest = sock.recv_multipart(copy=False)
            ident = identity.bytes
            kind = kind.bytes
            if kind == b"DATA":
                now = time.monotonic()
                s = stats.setdefault(ident, _LinkStats())
                if not s.messages:
                    s.first = now
                s.last = now
                s.messages += 1
                size = len(rest[1].buffer)
                s.bytes += size
                sock.send_multipart(
                    [ident, b"RCPT", rest[0].bytes, struct.pack("<q", size)]
                )
            elif kind == b"PING":
                sock.send_multipart([ident, b"PONG", rest[0].bytes])
            elif kind == b"RESULT":
                record = json.loads(rest[0].bytes)
                record["recorded_at"] = time.time()
                prefix = ident.rsplit(b"-", 1)[0]
                record["sink_links"] = {
                    k.decode(): {
                        "mb_s": round(v.bytes / max(v.last - v.first, 1e-6) / 1e6, 1),
                        "messages": v.messages,
                    }
                    for k, v in stats.items()
                    if k.rsplit(b"-", 1)[0] == prefix
                }
                with open(out, "a") as f:
                    f.write(json.dumps(record) + "\n")
                logger.info(
                    "Recorded %s %s",
                    record.get("kind"),
                    record.get("name", record.get("host", "")),
                )
                sock.send_multipart([ident, b"OK"])
    except KeyboardInterrupt:
        pass
    finally:
        sock.close(linger=0)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Link benchmark sink")
    parser.add_argument("--bind", default=DEFAULT_BIND)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    serve(args.bind, args.out)


if __name__ == "__main__":
    main()
