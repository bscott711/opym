# Ruff style: Compliant
"""Supervised `ssh -N -L` tunnels for the SSH backfill sender.

One `ssh` process per link: a single SSH connection tops out around 33-63 MB/s (sshd's
fixed channel window allows one window per round trip, and one `ssh` process encrypts on
one thread), so the sender spreads its slabs over several connections, each its own `ssh`
process, all forwarding to the backfill receiver's one port. The same reasoning and the
same ssh options as the live stream's tunnels (`pymmcore_gui._argus_stream._tunnel`),
kept here so the sender does not need that package.

Ports. The live stream's tunnels listen on the acquisition PC at 5555 .. 5555 + links - 1
(4 links by default: 5555-5558) and forward to Argus 5555. The backfill's links must never
sit on those: a backfill forward on 5556 would catch live link 1. They start at
`DEFAULT_LOCAL_PORT` (5656) and forward to the backfill receiver's port, 5556.

A local port that already accepts connections is reused, not replaced, so a tunnel started
by hand (or by a second launch) is never fought over; the port is re-checked every few
seconds and taken over if that tunnel goes away.
"""

from __future__ import annotations

import logging
import socket
import subprocess  # nosec B404 - runs the user's own ssh, argv list, never a shell
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_SSH_HOST = "Argus"
DEFAULT_LOCAL_PORT = 5656
DEFAULT_REMOTE_PORT = 5556
DEFAULT_LINKS = 4

_POLL_INTERVAL_S = 1.0
_REUSE_RECHECK_S = 5.0
_BACKOFF_START_S = 1.0
_BACKOFF_MAX_S = 30.0
_HEALTHY_AFTER_S = 60.0  # an ssh that stayed up this long resets the backoff
_STAGGER_S = 0.5  # between process starts, so a burst stays under sshd's MaxStartups
_SERVER_ALIVE_INTERVAL_S = 5  # three unanswered probes = dead: a black-holed link is noticed in ~15 s
_SERVER_ALIVE_COUNT_MAX = 3
_CONNECT_TIMEOUT_S = 10
_STDERR_TAIL_LINES = 20
_PREFERRED_CIPHERS = "aes128-gcm@openssh.com,aes256-gcm@openssh.com"


def link_endpoints(local_port: int, links: int) -> list[str]:
    """The local ZMQ endpoint of each link, in link order."""
    return [f"tcp://127.0.0.1:{local_port + k}" for k in range(max(1, links))]


def port_listening(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    """True when something accepts TCP connections on `host:port`."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _creationflags() -> int:
    if sys.platform != "win32":
        return 0
    # No console window, and below-normal priority: encrypting the transfer must never
    # take CPU from the acquisition.
    return subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
    if proc.stderr is not None:
        proc.stderr.close()


class Tunnel:
    """One supervised `ssh -N -L local:127.0.0.1:remote host` process."""

    def __init__(
        self,
        ssh_host: str,
        local_port: int,
        remote_port: int,
        ssh_args: Sequence[str] = (),
        start_delay_s: float = 0.0,
        ssh_exe: str = "ssh",
    ) -> None:
        self.ssh_host = ssh_host
        self.local_port = local_port
        self.remote_port = remote_port
        self.ssh_args = list(ssh_args)
        self.start_delay_s = start_delay_s
        self.ssh_exe = ssh_exe
        self.proc: subprocess.Popen[bytes] | None = None
        self.restarts = 0
        self.reused = False  # someone else's listener is serving this port
        self.last_error = ""
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def command(self) -> list[str]:
        return [
            self.ssh_exe,
            "-N",
            "-o", "BatchMode=yes",
            "-o", "ExitOnForwardFailure=yes",
            "-o", f"ServerAliveInterval={_SERVER_ALIVE_INTERVAL_S}",
            "-o", f"ServerAliveCountMax={_SERVER_ALIVE_COUNT_MAX}",
            "-o", f"ConnectTimeout={_CONNECT_TIMEOUT_S}",
            # AES-GCM first, the defaults stay as fallback: the same MB/s for less CPU.
            "-o", f"Ciphers=^{_PREFERRED_CIPHERS}",
            "-L", f"127.0.0.1:{self.local_port}:127.0.0.1:{self.remote_port}",
            *self.ssh_args,
            self.ssh_host,
        ]

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"backfill-tunnel-{self.local_port}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self.proc is not None:
            _terminate(self.proc)
        self.proc = None

    def _drain_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        # Read continuously: a full, unread stderr pipe would block ssh.
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.decode(errors="replace").rstrip()
            if line:
                self._stderr_tail.append(line)

    def _run(self) -> None:
        if self._stop.wait(self.start_delay_s):
            return
        backoff = _BACKOFF_START_S
        while not self._stop.is_set():
            if port_listening(self.local_port):
                if not self.reused:
                    logger.info("local port %d already has a listener: reusing it", self.local_port)
                self.reused = True
                self._stop.wait(_REUSE_RECHECK_S)
                continue
            self.reused = False
            self._stderr_tail.clear()
            try:
                proc = subprocess.Popen(  # nosec B603 - fixed argv, no shell
                    self.command(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    creationflags=_creationflags(),
                )
            except OSError as exc:
                self.last_error = f"could not start {self.ssh_exe}: {exc}"
                logger.error("%s", self.last_error)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX_S)
                continue
            self.proc = proc
            started = time.monotonic()
            reader = threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True)
            reader.start()
            logger.info("tunnel up (pid=%s): local %d -> %s:%d",
                        proc.pid, self.local_port, self.ssh_host, self.remote_port)
            while not self._stop.is_set() and proc.poll() is None:
                self._stop.wait(_POLL_INTERVAL_S)
            if self._stop.is_set():
                _terminate(proc)
                reader.join(timeout=1)
                self.proc = None
                break
            reader.join(timeout=1)
            if proc.stderr is not None:
                proc.stderr.close()
            self.last_error = " | ".join(self._stderr_tail) or "no output"
            logger.warning("tunnel on local %d exited (code=%s): %s",
                           self.local_port, proc.returncode, self.last_error)
            self.restarts += 1
            if time.monotonic() - started >= _HEALTHY_AFTER_S:
                backoff = _BACKOFF_START_S
            self._stop.wait(backoff)
            backoff = min(backoff * 2, _BACKOFF_MAX_S)


class TunnelSet:
    """`links` tunnels on consecutive local ports, each its own `ssh` process."""

    def __init__(
        self,
        ssh_host: str = DEFAULT_SSH_HOST,
        local_port: int = DEFAULT_LOCAL_PORT,
        remote_port: int = DEFAULT_REMOTE_PORT,
        links: int = DEFAULT_LINKS,
        ssh_args: Sequence[str] = (),
        ssh_exe: str = "ssh",
    ) -> None:
        self.local_port = local_port
        self.links = max(1, links)
        self.tunnels = [
            Tunnel(ssh_host, local_port + k, remote_port, ssh_args,
                   start_delay_s=k * _STAGGER_S, ssh_exe=ssh_exe)
            for k in range(self.links)
        ]

    @property
    def endpoints(self) -> list[str]:
        return link_endpoints(self.local_port, self.links)

    def commands(self) -> list[list[str]]:
        return [t.command() for t in self.tunnels]

    def start(self) -> None:
        for t in self.tunnels:
            t.start()

    def stop(self) -> None:
        for t in self.tunnels:  # all at once, then wait for each
            t._stop.set()
        for t in self.tunnels:
            t.stop()

    def listening(self) -> int:
        return sum(port_listening(t.local_port) for t in self.tunnels)

    def wait_ready(self, timeout_s: float = 20.0) -> bool:
        """Block until every link's local port accepts connections, or the timeout."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.listening() == self.links:
                return True
            time.sleep(0.25)
        return self.listening() == self.links

    def status(self) -> list[dict]:
        return [
            {"port": t.local_port, "up": port_listening(t.local_port), "reused": t.reused,
             "restarts": t.restarts, "error": t.last_error}
            for t in self.tunnels
        ]
