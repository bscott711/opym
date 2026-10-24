# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "zarr<3", "pyzmq", "msgpack", "numcodecs"]
# ///
# Ruff style: Compliant
"""Run `opym-backfill-send` from a checkout of this repo, with nothing else installed.

The sender needs only numpy, zarr<3, pyzmq, msgpack and numcodecs. Importing it the normal way
(`import opym.stream...`) first runs `opym/__init__.py`, which pulls in the whole processing stack
(ipywidgets, scikit-image, ...), so this launcher registers `opym` and `opym.stream` as bare
packages pointing at this checkout's `src/` and imports the sender directly. It always runs the
code next to it, whatever else is installed.

    uv run --script scripts\\opym_backfill_send.py --dry-run
    uv run --script scripts\\opym_backfill_send.py            # S:\\ -> Argus, forever
    uv run --script scripts\\opym_backfill_send.py --install  # start at every logon (Windows)

The five packages are declared at the top of this file (inline script metadata), so `uv run --script`
builds and caches a small environment of its own the first time: nothing to create or activate, and the
acquisition software's environment is never touched. `--install` builds a persistent one with uv
(`%LOCALAPPDATA%\\opym\\backfill-send\\venv`) for the logon task, since uv's cached script environments
may be pruned.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "opym"


def _bare_package(name: str, path: Path) -> None:
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]  # lets `opym.stream.bfill` etc. import without running __init__.py
    sys.modules[name] = pkg


def main() -> int:
    if not SRC.is_dir():
        raise SystemExit(f"cannot find the opym sources next to this script ({SRC})")
    _bare_package("opym", SRC)
    _bare_package("opym.stream", SRC / "stream")
    # The logon task re-launches this same script (see backfill_service.entry_arguments).
    os.environ["OPYM_BACKFILL_ENTRY"] = str(Path(__file__).resolve())
    from opym.stream.backfill_client import main as sender_main

    return sender_main()


if __name__ == "__main__":
    raise SystemExit(main())
