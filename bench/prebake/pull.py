"""Pulling task images before they are needed.

A bake needs its task image locally (it is built FROM it and never pulls),
and so does a trial's layer check. A first pull inside a trial competes with
Harbor's 600 s environment build limit: hf-model-inference (15 GB) took 387 s
and mteb-retrieve (21.6 GB) 518 s in the live runs. ensure_local() pulls what
is missing beforehand, a few at a time, and stops when its cancel event is set.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import IO

from .. import keys, procs
from .inventory import DockerError, forget_images, local_images, normalize_ref, tail

PULL_TIMEOUT_SEC = 3600
_POLL_SEC = 0.5


class PullCancelled(Exception):
    """ensure_local stopped because its cancel event was set."""


def missing(task_images: Iterable[str], refresh: bool = True) -> list[str]:
    """The task images that are not local, in order. Raises DockerError."""
    local = local_images(refresh)
    return [i for i in dict.fromkeys(task_images) if i and normalize_ref(i) not in local]


def _write(log: IO[str] | None, line: str, lock: threading.Lock) -> None:
    if log is None:
        return
    with lock:
        try:
            log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}\n")
            log.flush()
        except (OSError, ValueError):
            pass


def _pull(image: str, log: IO[str] | None, lock: threading.Lock, cancel: threading.Event | None) -> str | None:
    """Pull one image; returns the error, None when it pulled."""
    _write(log, f"pulling {image}", lock)
    started = time.monotonic()
    try:
        proc = subprocess.Popen(["docker", "pull", image], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                env=keys.child_env(None), creationflags=procs.NO_CTRL_C)
    except OSError as exc:
        return f"docker could not be started: {exc}"
    output: list[str] = []
    reader = threading.Thread(target=lambda: output.extend(proc.stdout or []), daemon=True)
    reader.start()
    while proc.poll() is None:
        if cancel is not None and cancel.wait(_POLL_SEC):
            procs.kill_tree(proc.pid)
            proc.wait()
            _write(log, f"cancelled the pull of {image}", lock)
            return "cancelled"
        if cancel is None:
            time.sleep(_POLL_SEC)
        if time.monotonic() - started > PULL_TIMEOUT_SEC:
            procs.kill_tree(proc.pid)
            proc.wait()
            return f"did not finish within {PULL_TIMEOUT_SEC} s"
    reader.join(5)
    if proc.returncode != 0:
        return f"docker pull exited with {proc.returncode}: {tail(''.join(output))}"
    _write(log, f"pulled {image} in {time.monotonic() - started:.0f} s", lock)
    return None


def ensure_local(images: Iterable[str], log: IO[str] | None = None, cancel: threading.Event | None = None,
                 jobs: int = 2) -> dict[str, list[str] | dict[str, str]]:
    """Pull every image that is not local, `jobs` at a time. Returns
    {"present": [...], "pulled": [...], "failed": {image: error}}. Raises
    PullCancelled when cancel is set, DockerError when Docker cannot list."""
    wanted = list(dict.fromkeys(i for i in images if i))
    todo = missing(wanted)
    result: dict[str, list[str] | dict[str, str]] = {
        "present": [i for i in wanted if i not in todo], "pulled": [], "failed": {}}
    if not todo:
        return result
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, jobs), thread_name_prefix="prebake-pull") as pool:
        errors = dict(zip(todo, pool.map(lambda image: _pull(image, log, lock, cancel), todo)))
    forget_images()
    if cancel is not None and cancel.is_set():
        raise PullCancelled("Cancelled while pulling task images")
    for image, error in errors.items():
        if error is None:
            result["pulled"].append(image)  # type: ignore[union-attr]
        else:
            result["failed"][image] = error  # type: ignore[index]
    return result


__all__ = ["DockerError", "PullCancelled", "ensure_local", "missing"]
