"""How busy the host's CPU was while a run went on.

Other work on the machine (other projects, builds, test suites) slows every
trial running beside it, and a trial slowed enough hits Harbor's agent
timeout. That time is the host's, not the model's. While a run is active a
Sampler appends the whole machine's CPU busy share every SAMPLE_SEC to
runs/<run_id>/hostload.jsonl; extract.py reads it back over each trial's
agent execution phase, so a contended trial's time stays out of speed
figures and its timeout out of the fails.

Each line is {"t": "<UTC ISO>", "cpu": <0-100>}: the busy share over the
SAMPLE_SEC that ended at t. No third-party dependency: Windows'
GetSystemTimes through ctypes, /proc/stat elsewhere.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HOSTLOAD_FILE = "hostload.jsonl"
SAMPLE_SEC = 5.0
# A sample at or above this busy share means trials were competing for the CPU.
HOT_CPU = 90.0
# A trial is contended when at least this share of its agent time had a hot host...
CONTENDED_SHARE = 0.25
# ...over at least this many samples: fewer is too little time to say.
MIN_SAMPLES = 3

log = logging.getLogger(__name__)

Counters = tuple[int, int]  # (busy, total) CPU time, any unit, since boot


def _windows_counters() -> Counters:
    import ctypes
    from ctypes import wintypes

    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        raise OSError("GetSystemTimes failed")

    def value(ft: Any) -> int:
        return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

    # Kernel time includes idle time.
    total = value(kernel) + value(user)
    return total - value(idle), total


def _proc_stat_counters(path: str = "/proc/stat") -> Counters:
    with open(path, encoding="ascii") as f:
        fields = [int(x) for x in f.readline().split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)  # idle + iowait
    total = sum(fields[:8])  # guest time is already in user
    return total - idle, total


def cpu_counters() -> Counters:
    return _windows_counters() if os.name == "nt" else _proc_stat_counters()


def busy_fraction(before: Counters, after: Counters) -> float | None:
    """The machine's CPU busy share (0..1) between two counter readings, or
    None when no CPU time passed between them."""
    busy, total = after[0] - before[0], after[1] - before[1]
    if total <= 0:
        return None
    return min(1.0, max(0.0, busy / total))


class Sampler:
    """Appends one hostload line every `interval` seconds until stop().
    Never raises into the run: errors are logged and sampling goes on."""

    def __init__(self, path: Path, interval: float = SAMPLE_SEC,
                 counters: Callable[[], Counters] = cpu_counters,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.path = Path(path)
        self.interval = interval
        self._counters = counters
        self._now = now
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> Sampler:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._loop, name="hostload", daemon=True)
            self._thread.start()
        except Exception:
            log.warning("Host load sampling did not start", exc_info=True)
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _read(self) -> Counters | None:
        try:
            return self._counters()
        except Exception:
            log.warning("Host CPU counters unreadable", exc_info=True)
            return None

    def _loop(self) -> None:
        before = self._read()
        warned = False
        while not self._stop.wait(self.interval):
            after = self._read()
            share = busy_fraction(before, after) if before is not None and after is not None else None
            before = after
            if share is None:
                continue
            line = json.dumps({"t": self._now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "cpu": round(100 * share, 1)})
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                if not warned:  # held by a viewer or antivirus: skip the sample, say so once
                    log.warning("Host load sample not written to %s", self.path, exc_info=True)
                    warned = True


def start(path: Path) -> Sampler | None:
    """A started Sampler, or None when it could not be made."""
    try:
        return Sampler(path).start()
    except Exception:
        log.warning("Host load sampling did not start", exc_info=True)
        return None


# Reading ----------------------------------------------------------------------

Sample = tuple[datetime, float]
_cache: dict[str, tuple[tuple[int, int], list[Sample]]] = {}
_CACHE_FILES = 8


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def parse_lines(lines: Iterable[str]) -> list[Sample]:
    """Samples from hostload lines; a torn or foreign line is skipped."""
    out: list[Sample] = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        t, cpu = _parse_time(row.get("t")), row.get("cpu")
        if t is not None and isinstance(cpu, (int, float)) and not isinstance(cpu, bool):
            out.append((t, float(cpu)))
    return out


def read_samples(path: Path) -> list[Sample] | None:
    """A hostload file's samples, or None when there is no file. Cached by
    size and mtime: a rescan reads one run's file for each of its trials."""
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return None
    key, stamp = os.path.normcase(os.path.abspath(path)), (st.st_mtime_ns, st.st_size)
    hit = _cache.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            samples = parse_lines(f)
    except OSError:
        return None
    if len(_cache) >= _CACHE_FILES:
        _cache.pop(next(iter(_cache)))
    _cache[key] = (stamp, samples)
    return samples


def contention(samples: Sequence[Sample] | None,
               windows: Sequence[tuple[datetime, datetime]]) -> dict[str, Any]:
    """The host_cpu_* flags for the samples inside the windows (the trial's
    agent execution), or {} when nothing is known: no file, or no sample
    there. A sample covers the SAMPLE_SEC before its time, so it counts in a
    window that holds its midpoint."""
    if not samples or not windows:
        return {}
    half = timedelta(seconds=SAMPLE_SEC / 2)
    inside = [cpu for t, cpu in samples if any(start <= t - half <= end for start, end in windows)]
    if not inside:
        return {}
    hot = sum(1 for cpu in inside if cpu >= HOT_CPU) / len(inside)
    return {
        "host_cpu_mean": round(sum(inside) / len(inside), 1),
        "host_cpu_hot_share": round(hot, 3),
        "host_cpu_samples": len(inside),
        "cpu_contended": len(inside) >= MIN_SAMPLES and hot >= CONTENDED_SHARE,
    }


HOST_FLAGS = ("host_cpu_mean", "host_cpu_hot_share", "host_cpu_samples", "cpu_contended")


def is_contended(flags: Any) -> bool:
    """A trial's flags (a dict or flags_json text) say its host was busy."""
    if isinstance(flags, str):
        try:
            flags = json.loads(flags)
        except ValueError:
            return False
    return isinstance(flags, dict) and flags.get("cpu_contended") is True
