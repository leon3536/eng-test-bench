"""Process safety on Windows.

Windows does not tie a child's life to its parent's. When the bench is ended
from Task Manager, crashes, or loses its console, harbor and everything it
started keep running, and no run budget watches them any more. A console
Ctrl+C goes the other way: it reaches every helper in the bench's process
group, so a second Ctrl+C during shutdown can kill the very docker command that
was removing a run's containers. And harbor cannot clean up after itself when
it is hard-killed, because Windows has no SIGTERM: the agents inside its
containers keep calling paid APIs until something removes those containers.

So this module keeps the bench to one process per profile, ties each harbor
child to the bench with a kill-on-close job object, keeps console Ctrl+C away
from helper processes, removes containers in a way that reports failure
instead of hiding it, and can hand that cleanup to a helper that outlives the
bench when its console is closing.

Importing it stays cheap (settings and the standard library only), because the
helper runs it as `python -m bench.procs cleanup <job_dir>...`.
"""

from __future__ import annotations

import functools
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .settings import ROOT, ConfigError

_WINDOWS = os.name == "nt"

# A console Ctrl+C goes to every process attached to the console, except those
# started as a new process group: Windows starts them with Ctrl+C ignored.
NO_CTRL_C: int = subprocess.CREATE_NEW_PROCESS_GROUP if _WINDOWS else 0

ALREADY_RUNNING = ("The bench is already running for this profile. "
                   "Use the dashboard URL printed by that process.")

LIST_TIMEOUT_SEC = 60
REMOVE_TIMEOUT_SEC = 300
_LOCK_RETRY_SEC = 0.5
_COMPOSE_FORMAT = '{{.ID}}\t{{.Label "com.docker.compose.project"}}'

# Windows console control events that end the process once handlers return.
CTRL_CLOSE_EVENT = 2
CTRL_LOGOFF_EVENT = 5
CTRL_SHUTDOWN_EVENT = 6
_CLOSING_EVENTS = frozenset({CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT})

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9  # JOBOBJECTINFOCLASS

# Handles and callbacks that must live as long as the process: a closed job
# handle kills every harbor in it, and a collected ctypes callback would crash
# the process when Windows calls it.
_instance_locks: dict[str, int] = {}
_instance_guard = threading.Lock()
_job_handle: int | None = None
_job_guard = threading.Lock()
_console_handlers: list[Any] = []


class AlreadyRunning(ConfigError):
    def __init__(self, message: str = ALREADY_RUNNING) -> None:
        super().__init__(message)


class LockCancelled(Exception):
    """file_lock stopped waiting because its cancel event was set."""


# Locks ---------------------------------------------------------------------------


def _try_lock(fd: int) -> None:
    """Lock byte 0 without waiting; OSError when another handle holds it."""
    if _WINDOWS:
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    try:
        if _WINDOWS:
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass  # closing the descriptor releases it anyway


def acquire_instance_lock(directory: Path, name: str = ".bench.lock") -> int:
    """Hold an exclusive lock on directory/name for the life of this process.

    A second bench for the same profile would otherwise treat the first one's
    live run as dead and remove its containers. The OS drops the lock when the
    process ends, however it ends, so a crash never leaves a stale lock.
    Asking again from the process that holds it returns the same descriptor.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    key = os.path.normcase(os.path.abspath(path))
    with _instance_guard:
        if key in _instance_locks:
            return _instance_locks[key]
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _try_lock(fd)
        except OSError:
            os.close(fd)
            raise AlreadyRunning() from None
        _instance_locks[key] = fd
        return fd


def release_instance_lock(fd: int) -> None:
    """Give an instance lock back. The bench never does; tests need it to
    delete their temp folders on Windows."""
    with _instance_guard:
        for key, held in list(_instance_locks.items()):
            if held == fd:
                del _instance_locks[key]
                _unlock(fd)
                os.close(fd)


@contextmanager
def file_lock(path: Path, timeout: float = 600, *,
              cancel: threading.Event | None = None) -> Iterator[None]:
    """Exclusive lock on path shared by every process, waited for in 0.5 s steps.

    Raises TimeoutError after `timeout` seconds, and LockCancelled as soon as
    `cancel` is set while waiting.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                _try_lock(fd)
                break
            except OSError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out after {timeout:g} s waiting for {path}") from None
                step = min(_LOCK_RETRY_SEC, remaining)
                if cancel is None:
                    time.sleep(step)
                elif cancel.wait(step):
                    raise LockCancelled(f"Stopped waiting for {path}") from None
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


# Process lifetime ------------------------------------------------------------------


@functools.cache
def _kernel32() -> Any:
    """A private kernel32 handle, so the argtypes set here cannot clash with
    other code using ctypes.windll. HANDLE everywhere: the default int
    restype would truncate handles on 64-bit Windows."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    k32.SetConsoleCtrlHandler.argtypes = [_handler_routine_type(), wintypes.BOOL]
    k32.SetConsoleCtrlHandler.restype = wintypes.BOOL
    return k32


@functools.cache
def _handler_routine_type() -> Any:
    import ctypes
    from ctypes import wintypes
    return ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)


def _extended_limit_info(limit_flags: int) -> Any:
    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    info = ExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = limit_flags
    return info


def _kill_on_close_job() -> int | None:
    """The one job object whose processes die when the bench exits: the OS
    closes the last handle to it then, however the bench ended."""
    global _job_handle
    import ctypes

    with _job_guard:
        if _job_handle is None:
            k32 = _kernel32()
            job = k32.CreateJobObjectW(None, None)
            if not job:
                return None
            info = _extended_limit_info(_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
            if not k32.SetInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                               ctypes.byref(info), ctypes.sizeof(info)):
                k32.CloseHandle(job)
                return None
            _job_handle = job
        return _job_handle


def assign_kill_on_close(proc: subprocess.Popen) -> bool:
    """Put a just-started child in the kill-on-close job. False when that is
    not possible (not Windows, or the call failed); the caller logs it.

    Harbor's console-script launcher keeps its own python in a kill-on-close
    job of its own, so killing harbor.exe takes harbor's python with it.
    """
    if not _WINDOWS:
        return False
    try:
        job = _kill_on_close_job()
        if job is None:
            return False
        return bool(_kernel32().AssignProcessToJobObject(job, int(proc._handle)))
    except Exception:
        return False


def kill_tree(pid: int, timeout: float = 30) -> None:
    """Force-end a process and its descendants. Never raises: a process that
    is already gone is simply skipped.

    Pass the pid of a Popen that is still held: Windows does not reuse a pid
    while a handle to its process is open, so the kill cannot hit a stranger.
    """
    if _WINDOWS:
        # Harbor handles only SIGTERM, which Windows cannot deliver, so this is
        # a hard kill; its containers are removed by cleanup_containers.
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                           timeout=timeout, creationflags=NO_CTRL_C)
        except (OSError, subprocess.SubprocessError):
            try:
                os.kill(pid, signal.SIGTERM)  # TerminateProcess on the root at least
            except OSError:
                pass
        return
    try:
        if os.getpgid(pid) == pid:  # leads its own group: its children go too
            os.killpg(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


# Containers ------------------------------------------------------------------------


def compose_project(name: str) -> str:
    # Same rule as harbor.environments.docker.docker._sanitize_docker_compose_project_name
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9_-]", "-", name)


def _docker(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["docker", *args], stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout,
                              creationflags=NO_CTRL_C)
    except OSError as exc:
        raise subprocess.SubprocessError(f"docker could not be started: {exc}") from exc


def _tail(text: str | None, limit: int = 300) -> str:
    return " ".join((text or "").split())[-limit:]


def _matching(kind: str, prefixes: tuple[str, ...]) -> list[str]:
    """IDs of the containers or networks whose compose project starts with a prefix."""
    list_args = ["ls", "-a"] if kind == "container" else ["ls"]
    result = _docker([kind, *list_args, "--format", _COMPOSE_FORMAT], LIST_TIMEOUT_SEC)
    if result.returncode != 0:
        # A failed listing prints nothing, which must not pass for "nothing to remove".
        raise subprocess.SubprocessError(
            f"docker {kind} ls failed (exit {result.returncode}): {_tail(result.stderr) or 'no output'}")
    ids = []
    for line in result.stdout.splitlines():
        item, tab, project = line.partition("\t")
        if tab and item.strip() and project.strip().startswith(prefixes):
            ids.append(item.strip())
    return ids


def _remove(kind: str, ids: list[str], prefixes: tuple[str, ...]) -> int:
    """Remove ids, then list again; raises if any match survives. The listing
    decides, not rm's exit code: rm also fails on items that went away by
    themselves in the meantime."""
    rm = ["rm", "-f"] if kind == "container" else ["rm"]
    result = _docker([kind, *rm, *ids], REMOVE_TIMEOUT_SEC)
    survivors = _matching(kind, prefixes)
    removed = len(set(ids) - set(survivors))
    if survivors:
        detail = f" docker {kind} rm said: {_tail(result.stderr)}" if result.returncode != 0 else ""
        raise subprocess.SubprocessError(
            f"{len(survivors)} {kind}(s) still present after removing {removed}.{detail}")
    return removed


def _trial_dirs(job_dir: Path) -> list[Path]:
    """The trial folders of a Harbor job, or of every per-task job under a
    task race's arm folder."""
    from .fsutil import harbor_jobs

    return [d for job in harbor_jobs(job_dir) for d in sorted(job.iterdir()) if d.is_dir()]


def cleanup_containers(job_dir: Path) -> int:
    """Remove the containers and networks a stopped job left behind. Returns
    how many containers were actually removed.

    Harbor names each trial's compose project after the trial, so they can be
    found from the job's trial directories. job_dir may also be a bench arm
    folder of a task race (fsutil.RACE_MARKER), which holds one Harbor job
    per task: their trial directories count, never the task folders' own
    names, which would match other runs' containers of the same task.
    Raises subprocess.SubprocessError
    when docker fails, cannot be started, or anything survives, so a failure
    is never reported as "nothing to remove" and callers need catch only one
    exception type.
    """
    job_dir = Path(job_dir)
    if not job_dir.is_dir():
        return 0
    try:
        prefixes = tuple(compose_project(d.name) for d in _trial_dirs(job_dir))
    except OSError as exc:
        raise subprocess.SubprocessError(f"the job folder could not be listed: {exc}") from exc
    if not prefixes:
        return 0
    removed = 0
    ids = _matching("container", prefixes)
    if ids:
        removed = _remove("container", ids, prefixes)
    # Networks last: docker refuses to remove one that a container still uses.
    networks = _matching("network", prefixes)
    if networks:
        _remove("network", networks, prefixes)
    return removed


def cleanup_with_retry(job_dir: Path, tries: int = 3, delay: float = 2.0) -> tuple[bool, str]:
    """cleanup_containers, retried: Docker Desktop's backend has brief hiccups,
    and a container still being created when the job was killed shows up a
    moment later. Returns (ok, a line for the arm's log)."""
    tries = max(1, tries)
    error = ""
    for attempt in range(1, tries + 1):
        try:
            return True, f"removed {cleanup_containers(job_dir)} container(s)"
        except (OSError, subprocess.SubprocessError) as exc:
            error = str(exc) or type(exc).__name__
            if attempt < tries:
                time.sleep(delay)
    return False, f"container cleanup FAILED: {error} (tried {tries} times)"


# Console close -----------------------------------------------------------------------


def _on_console_event(event: int, callback: Callable[[], None]) -> bool:
    if event not in _CLOSING_EVENTS:
        return False  # Ctrl+C and Ctrl+Break keep their usual handling
    try:
        callback()
    except BaseException as exc:  # nothing may escape into a Windows callback
        try:
            print(f"Emergency stop failed: {exc}", file=sys.stderr)
        except Exception:
            pass
    return True


def install_console_close_handler(callback: Callable[[], None]) -> bool:
    """Call callback when the console window is closed or Windows logs off or
    shuts down. Windows ends the process about 5 seconds after the event and
    no Python cleanup runs, so callback must finish within that time.
    Returns False when not on Windows or registration failed."""
    if not _WINDOWS:
        return False
    try:
        handler = _handler_routine_type()(lambda event: _on_console_event(event, callback))
        if not _kernel32().SetConsoleCtrlHandler(handler, True):
            return False
    except Exception:
        return False
    _console_handlers.append(handler)
    return True


def _remove_console_close_handlers() -> None:
    """Unregister every handler installed here (tests only)."""
    while _console_handlers:
        handler = _console_handlers.pop()
        try:
            _kernel32().SetConsoleCtrlHandler(handler, False)
        except Exception:
            pass


_CREDENTIAL_LIKE = re.compile(r"KEY|SECRET|TOKEN(?!S)|PASSWORD|CREDENTIAL|AUTH|_BASE_URL$|_API_BASE$", re.I)


def _helper_env() -> dict[str, str]:
    """The helper only runs docker: no credentials, and the project on its path."""
    try:
        from . import keys  # lazy: keeps `python -m bench.procs` light
        env = keys.child_env(None)
    except Exception:
        # Removing the containers matters more than a tidy env, but the
        # helper still gets no credentials or endpoint overrides.
        env = {k: v for k, v in os.environ.items() if not _CREDENTIAL_LIKE.search(k)}
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p and os.path.normcase(p) != os.path.normcase(str(ROOT))]
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *parts])
    if "BENCH_PROFILE" in os.environ:
        env["BENCH_PROFILE"] = os.environ["BENCH_PROFILE"]
    return env


def spawn_detached_cleanup(job_dirs: list[Path]) -> bool:
    """Start `python -m bench.procs cleanup <job_dir>...` so that it outlives
    this process. Returns True when it started (or there was nothing to do).

    The helper gets a console of its own without a window (CREATE_NO_WINDOW)
    rather than none (DETACHED_PROCESS): the venv's python.exe launcher answers
    DETACHED_PROCESS by opening a visible console for the real interpreter, and
    a user who closes that window kills the cleanup. Either way the helper is
    not attached to the bench's console, so closing that console cannot reach
    it. It breaks away from any job the bench is in where that is allowed.
    """
    dirs = [os.fspath(Path(d).absolute()) for d in job_dirs]
    if not dirs:
        return True
    argv = [sys.executable, "-m", "bench.procs", "cleanup", *dirs]
    env = _helper_env()
    if _WINDOWS:
        base = NO_CTRL_C | subprocess.CREATE_NO_WINDOW
        attempts = (base | subprocess.CREATE_BREAKAWAY_FROM_JOB, base)
    else:
        attempts = (0,)
    for flags in attempts:
        try:
            subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=flags, start_new_session=not _WINDOWS)
            return True
        except OSError:
            continue  # a job that forbids breakaway refuses the first attempt
    return False


def main(argv: list[str] | None = None) -> int:
    """python -m bench.procs cleanup <job_dir>...: exit 0 when every job's
    containers are gone, 1 otherwise."""
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] != "cleanup":
        print("usage: python -m bench.procs cleanup <job_dir>...", file=sys.stderr)
        return 1
    ok = True
    for job_dir in args[1:]:
        done, message = cleanup_with_retry(Path(job_dir))
        print(f"{job_dir}: {message}", file=sys.stderr)
        ok = ok and done
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
