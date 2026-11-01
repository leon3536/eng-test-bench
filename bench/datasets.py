"""Local dataset snapshots.

Harbor's registry pins Terminal-Bench 2.0 to a branch rather than a commit,
so every `harbor run --dataset` re-downloads its tasks into a shared cache,
deleting the old copy first. With several harbor processes running, one
deletes files another's containers are using; on Windows that crashes the
process and kills those containers. Arms started hours apart could also get
different task versions.

The bench downloads each dataset once into datasets/ and runs every arm from
that fixed copy with `harbor run --path`. Harbor uses local tasks in place.

datasets/ is shared by every profile and by `python -m bench.sample_tasks`,
so downloads are serialised across processes with a lock file, and a finished
snapshot is never deleted to make way for another copy: one bench's arms may
be building from it. The download itself can be cancelled within about a
second, and dies with the bench.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import tomllib
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from . import keys, procs
from .settings import CONFIG_DIR, ROOT, harbor_executable

DATASETS_DIR = ROOT / "datasets"
TASKS_FILE = CONFIG_DIR / "tasks.txt"
DOWNLOAD_TIMEOUT_SEC = 1800
# Another process's download is bounded by its own deadline, so wait that long.
LOCK_TIMEOUT_SEC = DOWNLOAD_TIMEOUT_SEC + 120
_POLL_SEC = 0.5
# Windows refuses to rename or delete a folder while any file in it is open,
# as antivirus and the search indexer do briefly with new files. Five tries.
_RETRY_BACKOFF = (0.5, 1.0, 1.5, 2.0)

_lock = threading.Lock()
_meta_cache: dict[str, tuple[tuple[str, str], dict[str, dict[str, Any]]]] = {}

_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*([GMK])(?:I?B)?", re.IGNORECASE)
_GB_PER_UNIT = {"G": 1.0, "M": 1 / 1024, "K": 1 / 1024 / 1024}
_SAMPLE_HEADER = re.compile(r"#\s*Task (?:sample|set) for (\S+?),")  # drawn by sample_tasks, or chosen by hand
_SAMPLE_COMMENT = re.compile(r"\s*([A-Za-z][\w-]*)\s*(?:,\s*(\d+(?:\.\d+)?)\s*s)?\s*")


class DownloadCancelled(Exception):
    """The download was stopped because its run was cancelled."""

    def __init__(self, dataset: str = "") -> None:
        super().__init__(f"Cancelled while downloading {dataset}" if dataset
                         else "Cancelled during the dataset download")
        self.dataset = dataset

    def __reduce__(self) -> tuple[type, tuple[str]]:
        return type(self), (self.dataset,)


def _root(dataset: str) -> Path:
    return DATASETS_DIR / dataset.replace("/", "_")


def info(dataset: str) -> dict[str, Any] | None:
    manifest = _root(dataset) / "snapshot.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    path = _root(dataset) / data["tasks_dir"]
    if not path.is_dir():
        return None
    return {**data, "path": str(path.relative_to(ROOT))}


def path(dataset: str) -> Path | None:
    snapshot = info(dataset)
    return ROOT / snapshot["path"] if snapshot else None


def ensure(dataset: str, log: IO[str] | None = None,
           cancel: threading.Event | None = None) -> Path:
    """Download the dataset once and return its tasks folder.

    Safe to call from several threads and several bench processes. Raises
    DownloadCancelled within about a second of `cancel` being set (also while
    waiting for another process's download), and RuntimeError when the
    download fails or runs past DOWNLOAD_TIMEOUT_SEC.
    """
    with _lock:
        existing = path(dataset)
        if existing:
            return existing
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        with contextlib.ExitStack() as held:
            try:
                held.enter_context(procs.file_lock(DATASETS_DIR / ".lock", LOCK_TIMEOUT_SEC, cancel=cancel))
            except procs.LockCancelled:
                raise DownloadCancelled(dataset) from None
            except TimeoutError:
                raise RuntimeError(f"Another bench process has been downloading datasets for over "
                                   f"{LOCK_TIMEOUT_SEC // 60} minutes. Try again when it finishes.") from None
            existing = path(dataset)  # another process may have finished it while this one waited
            if existing:
                return existing
            if cancel is not None and cancel.is_set():
                raise DownloadCancelled(dataset)
            return _download(dataset, log, cancel)


def _download(dataset: str, log: IO[str] | None, cancel: threading.Event | None) -> Path:
    _remove_stale_staging()
    staging = Path(tempfile.mkdtemp(prefix=".download-", dir=DATASETS_DIR))
    try:
        _run_download(dataset, staging, log, cancel)
        subdirs = [p for p in staging.iterdir() if p.is_dir()]
        if len(subdirs) != 1:
            raise RuntimeError(f"Unexpected layout after downloading {dataset}")
        tasks = [p for p in subdirs[0].iterdir() if (p / "task.toml").exists()]
        manifest = {
            "dataset": dataset,
            "tasks_dir": subdirs[0].name,
            "tasks": len(tasks),
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "harbor": importlib.metadata.version("harbor"),
        }
        (staging / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")
        existing = path(dataset)
        if existing:
            # Finished by a process that took no lock. Its arms may be using
            # it, so keep it and drop this copy.
            return existing
        target = _root(dataset)

        def remove_incomplete() -> None:
            # No valid manifest, so this is an earlier attempt that never finished.
            if target.exists():
                shutil.rmtree(target)

        _with_retry(remove_incomplete, f"remove the incomplete {target.name} folder")
        _with_retry(lambda: staging.replace(target), f"move the download into {target.name}")
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    result = path(dataset)
    if result is None:
        raise RuntimeError(f"Downloaded {dataset}, but its snapshot could not be read")
    return result


def _remove_stale_staging() -> None:
    """Staging folders left by a bench that died mid-download. Every download
    holds the lock, so none of them is in use."""
    for stale in DATASETS_DIR.glob(".download-*"):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)


def _with_retry(action: Callable[[], Any], what: str) -> None:
    for delay in (*_RETRY_BACKOFF, None):
        try:
            action()
            return
        except PermissionError as exc:
            if delay is None:
                raise RuntimeError(f"Could not {what}: a file in it is open in another program "
                                   f"(antivirus or the search indexer?). Try again. ({exc})") from exc
            time.sleep(delay)


def _run_download(dataset: str, staging: Path, log: IO[str] | None,
                  cancel: threading.Event | None) -> None:
    cmd = [harbor_executable(), "datasets", "download", dataset, "-o", str(staging)]
    with contextlib.ExitStack() as stack:
        if log is not None:
            log.flush()  # keep the bench's own lines ahead of harbor's
            sink: IO[str] = log
        else:
            # A file, not a pipe: a pipe nobody reads fills up and stalls harbor.
            sink = stack.enter_context(tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace"))
        proc = subprocess.Popen(cmd, cwd=ROOT, env=keys.child_env(None), stdin=subprocess.DEVNULL,
                                stdout=sink, stderr=subprocess.STDOUT, creationflags=procs.NO_CTRL_C)
        procs.assign_kill_on_close(proc)
        stopped = _wait(proc, cancel)
        if stopped == "cancelled":
            raise DownloadCancelled(dataset)
        if stopped == "timeout":
            raise RuntimeError(f"Downloading {dataset} timed out after {DOWNLOAD_TIMEOUT_SEC // 60} minutes")
        if proc.returncode != 0:
            tail = ""
            if log is None:
                sink.seek(0)
                tail = sink.read()[-500:]
            raise RuntimeError(f"Downloading {dataset} failed (exit {proc.returncode}). {tail}".rstrip())


def _wait(proc: subprocess.Popen, cancel: threading.Event | None) -> str | None:
    """Wait for the download. None when it exited by itself; otherwise it has
    been killed and the reason is returned."""
    deadline = time.monotonic() + DOWNLOAD_TIMEOUT_SEC
    reason = None
    try:
        while reason is None:
            try:
                proc.wait(timeout=_POLL_SEC)
                return None
            except subprocess.TimeoutExpired:
                pass
            if cancel is not None and cancel.is_set():
                reason = "cancelled"
            elif time.monotonic() >= deadline:
                reason = "timeout"
    except BaseException:
        # Ctrl+C in sample_tasks, for one: the download runs in its own process
        # group, so it would carry on after this process is gone.
        _kill(proc)
        raise
    _kill(proc)
    return reason


def _kill(proc: subprocess.Popen) -> None:
    # Harbor runs git; ending only harbor would leave git behind on Windows.
    procs.kill_tree(proc.pid)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)


# Task facts -----------------------------------------------------------------------


def task_meta(dataset: str) -> dict[str, dict[str, Any]]:
    """Facts for planning, per task name: difficulty, category,
    agent_timeout_sec, memory_gb, cpus and expert_time_estimate_min, each
    None when unknown.

    Read from the snapshot's task.toml files and cached until the snapshot is
    replaced. Without a snapshot, the comments in config/tasks.txt
    ("# medium, 900s") give difficulty and time limit. Never raises.
    """
    try:
        snapshot = info(dataset)
    except Exception:
        snapshot = None
    if snapshot:
        try:
            tasks_dir = ROOT / snapshot["path"]
            key = (str(tasks_dir), str(snapshot.get("downloaded_at", "")))
            cached = _meta_cache.get(dataset)
            if cached is None or cached[0] != key:
                rows, complete = _snapshot_meta(tasks_dir)
                cached = (key, rows)
                if complete:  # a file that could not be read is tried again next time
                    _meta_cache[dataset] = cached
            return {name: dict(row) for name, row in cached[1].items()}
        except Exception:
            pass
    try:
        return _sample_meta(dataset)
    except Exception:
        return {}


def _snapshot_meta(tasks_dir: Path) -> tuple[dict[str, dict[str, Any]], bool]:
    rows: dict[str, dict[str, Any]] = {}
    complete = True
    for task_dir in sorted(p for p in tasks_dir.iterdir() if (p / "task.toml").is_file()):
        try:
            data = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        except OSError:
            data, complete = {}, False
        except (UnicodeDecodeError, tomllib.TOMLDecodeError):
            data = {}
        rows[task_dir.name] = _task_row(data)
    return rows, complete


def _task_row(data: dict[str, Any]) -> dict[str, Any]:
    meta, agent, env = (_table(data, k) for k in ("metadata", "agent", "environment"))
    cpus = _number(env.get("cpus"))
    return {
        "difficulty": _text(meta.get("difficulty")),
        "category": _text(meta.get("category")),
        "agent_timeout_sec": _number(agent.get("timeout_sec")),
        "memory_gb": _memory_gb(env),
        "cpus": int(cpus) if cpus is not None and cpus.is_integer() else cpus,
        "expert_time_estimate_min": _number(meta.get("expert_time_estimate_min")),
    }


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def _memory_gb(env: dict[str, Any]) -> float | None:
    # Harbor reads the newer memory_mb, or the older memory = "2G" (1024 MB per G).
    mb = _number(env.get("memory_mb"))
    if mb is not None:
        return round(mb / 1024, 3)
    size = env.get("memory")
    match = _SIZE.fullmatch(size.strip()) if isinstance(size, str) else None
    if not match:
        return None
    return round(float(match.group(1)) * _GB_PER_UNIT[match.group(2).upper()], 3)


def _sample_meta(dataset: str) -> dict[str, dict[str, Any]]:
    """Difficulty and time limit from the comments sample_tasks writes."""
    if not TASKS_FILE.exists():
        return {}
    lines = TASKS_FILE.read_text(encoding="utf-8").splitlines()
    header = next((_SAMPLE_HEADER.match(line) for line in lines if line.strip()), None)
    if header and header.group(1) != dataset:
        return {}  # a sample drawn from another dataset says nothing about this one
    rows: dict[str, dict[str, Any]] = {}
    for line in lines:
        name, _, comment = line.partition("#")
        name = name.strip()
        if not name:
            continue
        match = _SAMPLE_COMMENT.fullmatch(comment) if comment else None
        rows[name] = {
            "difficulty": match.group(1) if match else None,
            "category": None,
            "agent_timeout_sec": float(match.group(2)) if match and match.group(2) else None,
            "memory_gb": None,
            "cpus": None,
            "expert_time_estimate_min": None,
        }
    return rows
