"""pymmcore-gui's replay (its real ArgusStreamSession, resume included), run on
Argus against the test stack: one link straight to the test receiver's port
instead of SSH tunnels (Argus can't ssh to itself), and the report written to
~/<name>.replay.json directly instead of sent over ssh. Used for R3
(receiver restart), since only the GUI session code resumes a session.

usage (from a pymmcore-gui checkout, with its venv):
  PYTHONPATH=src QT_QPA_PLATFORM=offscreen python .../gui_replay_direct.py STORE \\
    --links 1 --local-port 5602 --remote-port 5602 --host none --pace fixed \\
    --stack-s 5.72 --interval 13.16 --argus-root /mmfs2/.../opym_lv/raw --name NAME
"""

import subprocess
import sys
import types
from pathlib import Path

from pymmcore_gui._argus_stream import replay


class _DirectTunnel:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass

    def stop(self):
        pass


def _run(cmd, input=None, **kw):
    if cmd and cmd[0] == "ssh":
        name = cmd[-1].removeprefix("cat > ")
        (Path.home() / name).write_bytes(input)
        return subprocess.CompletedProcess(cmd, 0)
    return subprocess.run(cmd, input=input, **kw)


replay.ArgusTunnelManager = _DirectTunnel
replay.subprocess = types.SimpleNamespace(
    run=_run, SubprocessError=subprocess.SubprocessError
)
sys.exit(replay.main(sys.argv[1:]))
