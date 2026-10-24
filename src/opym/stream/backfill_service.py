# Ruff style: Compliant
"""Running the SSH backfill sender unattended: state files, one instance, low priority, and a
Windows scheduled task that starts it at logon.

    opym-backfill-send --install      # start at every logon, hidden, restarted if it dies
    opym-backfill-send --status       # what is it doing right now?
    opym-backfill-send --stop         # stop the running copy
    opym-backfill-send --uninstall    # stop it and remove the logon task

Everything here is plumbing around `backfill_client.BackfillClient`; none of it touches the
live stream. The sender runs at the lowest CPU/IO priority the OS offers, so it cannot take
time from the acquisition, and the receiver on Argus already holds every write while a live
acquisition owns the lease (`opym.lanes`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess  # nosec B404 - schtasks/taskkill with fixed argv, never a shell
import sys
import tempfile
import time
from pathlib import Path
from xml.sax.saxutils import escape  # nosec B406 - only escapes text we write; nothing is parsed

logger = logging.getLogger(__name__)

TASK_NAME = "OPYM Backfill Send"
DEFAULT_LOCK_PORT = 5655  # next to the backfill links (5656+), clear of the live links (5555-5558)
_STATE_DIR_ENV = "OPYM_BACKFILL_STATE_DIR"
_LOCK_PORT_ENV = "OPYM_BACKFILL_LOCK_PORT"
STATUS_STALE_S = 60.0  # a status file older than this belongs to a sender that is gone


def state_dir() -> Path:
    """Where the sender keeps its log, status and scan state (created on demand)."""
    override = os.environ.get(_STATE_DIR_ENV)
    if override:
        base = Path(override)
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "opym" / "backfill-send"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "opym" / "backfill-send"
    base.mkdir(parents=True, exist_ok=True)
    return base


def log_path() -> Path:
    return state_dir() / "backfill-send.log"


def status_path() -> Path:
    return state_dir() / "status.json"


def scan_state_path() -> Path:
    return state_dir() / "scan-state.json"


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_status(payload: dict) -> None:
    payload = {**payload, "pid": os.getpid(), "updated_at": time.time()}
    try:
        _atomic_write_text(status_path(), json.dumps(payload, indent=1, default=str))
    except OSError:
        logger.debug("could not write the status file", exc_info=True)


def read_status() -> dict | None:
    try:
        return json.loads(status_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def setup_logging(console: bool = True) -> Path:
    """Log to a rotating file in the state dir, and to the console when there is one."""
    from logging.handlers import RotatingFileHandler

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    close_logging()  # a second call replaces our handlers, never someone else's
    path = log_path()
    handlers: list[logging.Handler] = [RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3,
                                                           encoding="utf-8")]
    if console and sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    for h in handlers:
        h.setFormatter(fmt)
        h._opym_backfill = True  # type: ignore[attr-defined]
        root.addHandler(h)
    return path


def close_logging() -> None:
    """Remove (and close) the handlers `setup_logging` installed."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_opym_backfill", False):
            root.removeHandler(h)
            h.close()


class AlreadyRunning(RuntimeError):
    pass


class SingleInstance:
    """Holds a loopback port for the life of the process: a second sender cannot start.

    Two senders would race on the same stores (the receiver's own lock would stop the second,
    but only after a wasted scan); a bound port goes away with the process, so a crash never
    leaves a stale lock behind.
    """

    def __init__(self, port: int | None = None) -> None:
        self.port = port or int(os.environ.get(_LOCK_PORT_ENV) or DEFAULT_LOCK_PORT)
        self._sock: socket.socket | None = None

    def __enter__(self) -> SingleInstance:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sys.platform == "win32":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            # Bound but never listening: the port is held, yet a connect is refused, so nothing
            # (a tunnel supervisor, say) can mistake the lock for a service.
            s.bind(("127.0.0.1", self.port))
        except OSError as exc:
            s.close()
            st = read_status() or {}
            raise AlreadyRunning(
                f"another backfill sender is already running (pid {st.get('pid', '?')}); "
                "use --status to see it or --stop to stop it"
            ) from exc
        self._sock = s
        return self

    def __exit__(self, *exc: object) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


def lower_priority() -> str:
    """Drop this process to the lowest CPU/IO priority the OS offers. Returns what was applied."""
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.GetCurrentProcess()
            # PROCESS_MODE_BACKGROUND_BEGIN lowers CPU, I/O and memory priority together.
            if kernel32.SetPriorityClass(handle, 0x00100000):
                return "background"
            if kernel32.SetPriorityClass(handle, 0x00004000):  # BELOW_NORMAL_PRIORITY_CLASS
                return "below-normal"
        except Exception:  # noqa: BLE001 - priority is a courtesy, never a reason to fail
            logger.debug("could not lower the process priority", exc_info=True)
        return "unchanged"
    try:
        os.nice(10)
        return "nice+10"
    except (OSError, AttributeError):
        return "unchanged"


# ---- Windows scheduled task --------------------------------------------------------------


def task_python() -> str:
    """The interpreter the task runs: `pythonw.exe` beside this one when it exists (no console)."""
    exe = Path(sys.executable)
    if sys.platform == "win32":
        w = exe.with_name("pythonw.exe")
        if w.exists():
            return str(w)
    return str(exe)


# What the sender imports; scripts/opym_backfill_send.py declares the same list as inline script metadata.
SENDER_REQUIREMENTS = ["numpy", "zarr<3", "pyzmq", "msgpack", "numcodecs"]


def is_ephemeral_env(python: str | None = None) -> bool:
    """True for the throwaway environments `uv run --script` builds inside uv's cache
    (`.../environments-v2/<hash>/`): fine for running by hand, but a logon task must not depend on
    something `uv cache prune` is free to delete."""
    return any(part.startswith("environments-v") for part in re.split(r"[\\/]+", str(python or sys.executable)))


def service_python() -> str:
    """The interpreter the logon task runs (`pythonw.exe` on Windows, so no console window).

    The one running this process when it lives in a normal environment. When it is one of uv's
    throwaway script environments, a persistent one is built with uv next to the sender's other
    state files, and kept (a rerun only refreshes it).
    """
    if not is_ephemeral_env():
        return task_python()
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv was not found on PATH; it is needed to build the sender's own environment")
    venv = state_dir() / "venv"
    bindir, name = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    exe = venv / bindir / name
    steps = []
    if not exe.exists():
        steps.append([uv, "venv", "--python", sys.executable, str(venv)])
    steps.append([uv, "pip", "install", "--python", str(exe), *SENDER_REQUIREMENTS])
    for argv in steps:
        done = subprocess.run(argv, capture_output=True, text=True, check=False)  # nosec B603 - uv, fixed argv
        if done.returncode != 0:
            raise SystemExit(f"could not build the sender's environment ({' '.join(argv[:3])}): "
                             f"{(done.stderr or done.stdout).strip()}")
    w = exe.with_name("pythonw.exe")
    return str(w if w.exists() else exe)


ENTRY_ENV = "OPYM_BACKFILL_ENTRY"


def entry_arguments() -> list[str]:
    """How the task re-launches the sender: `scripts/opym_backfill_send.py` when that is what
    started this process (it records its own path), else `-m opym.stream.backfill_client`."""
    script = os.environ.get(ENTRY_ENV)
    return [script] if script else ["-m", "opym.stream.backfill_client"]


def task_xml(arguments: list[str], python: str | None = None, user: str | None = None,
             workdir: str | None = None) -> str:
    """Task Scheduler XML: run at logon, hidden window, below-normal priority, restart if it
    dies, no time limit, never queued twice."""
    python = python or task_python()
    args = subprocess.list2cmdline([*entry_arguments(), *arguments])
    user_xml = f"<UserId>{escape(user)}</UserId>" if user else ""
    workdir_xml = f"<WorkingDirectory>{escape(workdir)}</WorkingDirectory>" if workdir else ""
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>OPYM SSH backfill: copies finished acquisitions to Argus in the background.</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      {user_xml}
      <Delay>PT30S</Delay>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      {user_xml}
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>999</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(python)}</Command>
      <Arguments>{escape(args)}</Arguments>
      {workdir_xml}
    </Exec>
  </Actions>
</Task>
"""


def _schtasks(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 B607 - Windows system tool, fixed argv, no shell
        ["schtasks", *args], capture_output=True, text=True, check=False)


def _require_windows(what: str) -> None:
    if sys.platform != "win32":
        raise SystemExit(f"{what} is only available on the Windows acquisition PC "
                         "(it uses the Task Scheduler)")


def install_task(arguments: list[str], start_now: bool = True) -> str:
    """Register the logon task with `arguments` and start it. Returns a one-line summary."""
    _require_windows("--install")
    user = os.environ.get("USERDOMAIN", "") + "\\" + os.environ.get("USERNAME", "")
    user = user if len(user) > 1 else None
    xml = task_xml(arguments, python=service_python(), user=user, workdir=str(Path.home()))
    xml_file = state_dir() / "task.xml"
    xml_file.write_text(xml, encoding="utf-16")  # schtasks wants UTF-16 with a BOM
    done = _schtasks("/Create", "/TN", TASK_NAME, "/XML", str(xml_file), "/F")
    if done.returncode != 0:
        raise SystemExit(f"could not create the scheduled task: {(done.stderr or done.stdout).strip()}")
    if start_now:
        ran = _schtasks("/Run", "/TN", TASK_NAME)
        if ran.returncode != 0:
            return (f"task '{TASK_NAME}' installed (starts at every logon), but did not start now: "
                    f"{(ran.stderr or ran.stdout).strip()}")
    return f"task '{TASK_NAME}' installed: starts at every logon, and is running now"


def task_state() -> str | None:
    """'Running', 'Ready', ... or None when the task is not installed (or not on Windows)."""
    if sys.platform != "win32":
        return None
    q = _schtasks("/Query", "/TN", TASK_NAME, "/FO", "LIST")
    if q.returncode != 0:
        return None
    for line in q.stdout.splitlines():
        if line.strip().lower().startswith("status:"):
            return line.split(":", 1)[1].strip()
    return "Installed"


def stop_running() -> str:
    """Stop the running sender (task and process). Returns what was done."""
    done = []
    if sys.platform == "win32" and task_state() is not None:
        _schtasks("/End", "/TN", TASK_NAME)
        done.append("ended the scheduled task")
    st = read_status() or {}
    pid = st.get("pid")
    if (pid and pid != os.getpid() and st.get("state") != "stopped"
            and time.time() - st.get("updated_at", 0) < STATUS_STALE_S):
        try:
            if sys.platform == "win32":
                subprocess.run(  # nosec B603 B607 - Windows system tool, pid from our own status file
                    ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
            else:
                os.kill(int(pid), 15)
            done.append(f"stopped pid {pid}")
        except OSError as exc:
            done.append(f"could not stop pid {pid}: {exc}")
    return "; ".join(done) or "no running backfill sender found"


def uninstall_task() -> str:
    _require_windows("--uninstall")
    msg = stop_running()
    if task_state() is None:
        return f"{msg}; the task was not installed"
    done = _schtasks("/Delete", "/TN", TASK_NAME, "/F")
    if done.returncode != 0:
        raise SystemExit(f"could not delete the scheduled task: {(done.stderr or done.stdout).strip()}")
    return f"{msg}; task '{TASK_NAME}' removed"


def format_status() -> str:
    """The `--status` text: is it running, what is it doing, how did the last pass go."""
    st = read_status()
    lines = []
    task = task_state()
    if sys.platform == "win32":
        lines.append(f"scheduled task: {task or 'not installed (run --install)'}")
    if not st:
        lines.append("sender: no status yet (never started on this PC)")
        lines.append(f"log: {log_path()}")
        return "\n".join(lines)
    age = time.time() - st.get("updated_at", 0)
    running = age < STATUS_STALE_S and st.get("state") != "stopped"
    lines.append(f"sender: {'RUNNING' if running else 'not running'} (pid {st.get('pid')}, "
                 f"status {int(age)} s old)")
    if running:
        lines.append(f"doing:  {st.get('activity', '?')}")
    for t in st.get("tunnels", []):
        note = " (reused)" if t.get("reused") else ""
        err = f"  last error: {t['error']}" if t.get("error") and not t.get("up") else ""
        lines.append(f"tunnel {t['port']}: {'up' if t.get('up') else 'DOWN'}{note}, "
                     f"restarts {t.get('restarts', 0)}{err}")
    lp = st.get("last_pass")
    if lp:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(lp.get("finished_at", 0)))
        lines.append(f"last pass ({when}, {lp.get('seconds', 0):.0f} s): {lp.get('summary', '')}")
    lines.append(f"sent this run: {st.get('sent_gb', 0):.2f} GB")
    lines.append(f"log: {log_path()}")
    return "\n".join(lines)
