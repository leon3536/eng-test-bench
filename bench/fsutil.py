"""File writes that survive Windows.

On Windows, replacing a file fails with PermissionError (WinError 5) while any
other handle to it is open without FILE_SHARE_DELETE: an antivirus scan, the
search indexer, or the bench's own reader on another thread. The dev pilot hit
exactly that on run.json. So mutable files are written to a temp file and then
moved into place with a short retry; records that must never change are
created once with open("x"); logs are only ever appended to.

Everything is UTF-8 with LF endings, and hashes come from canonical JSON
strings rather than file bytes, so an editor that rewrites line endings cannot
change one.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

_RETRY_SECONDS = 0.1

WINDOWS_RESERVED: frozenset[str] = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{c}" for c in "0123456789¹²³"}
    | {f"lpt{c}" for c in "0123456789¹²³"}
)
_BAD_CHARS = set('<>:"/\\|?*') | {chr(c) for c in range(32)}


def atomic_write_text(path: Path, text: str, patience: float = 2.0) -> bool:
    """Write text to path through a temp file in the same folder.

    Returns False, leaving the old file in place and no temp file behind, when
    the target stayed locked for `patience` seconds. Other errors propagate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        deadline = time.monotonic() + patience
        while True:
            try:
                tmp.replace(path)
                return True
            except PermissionError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(_RETRY_SECONDS)
    except BaseException:
        _discard(tmp)
        raise
    _discard(tmp)
    return False


def atomic_write_json(path: Path, data: Any, patience: float = 2.0, indent: int | None = 2) -> bool:
    text = json.dumps(data, indent=indent, ensure_ascii=False) + "\n"
    return atomic_write_text(path, text, patience)


def create_exclusive_json(path: Path, data: Any) -> None:
    """Create a write-once record. FileExistsError if it is already there."""
    path = Path(path)
    # Serialize first so a value that cannot be written never leaves a
    # half-written file that open("x") would then refuse to replace.
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def append_jsonl(path: Path, records: list[dict], patience: float = 2.0) -> None:
    """Append records as one write, then fsync.

    Not safe for concurrent writers to the same file; callers serialize with
    their own lock. Raises PermissionError if the file stayed locked.
    """
    if not records:
        return
    path = Path(path)
    # Encoded here, so the file is opened in binary mode: no newline
    # translation, and the same handle can check how the file ends.
    data = "".join(_jsonl_line(r) for r in records).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + patience
    while True:
        try:
            f = open(path, "a+b")
            break
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(_RETRY_SECONDS)
    with f:
        # A crash mid-append can leave a last line without its newline. Start
        # on a fresh line so that fragment cannot swallow the first new record.
        if f.seek(0, os.SEEK_END) > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                data = b"\n" + data
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _jsonl_line(record: dict) -> str:
    # JSON allows the Unicode line and paragraph separators raw inside strings,
    # but many line readers split on them; escaping keeps one record per line
    # for any reader, and json.loads turns the escapes back into the same text.
    text = json.dumps(record, ensure_ascii=False)
    return text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029").replace("\x85", "\\u0085") + "\n"


def read_jsonl(path: Path) -> list[dict]:
    """Every record in the file. Lines that do not parse are dropped: the only
    way to get one is a crash mid-append, which leaves a partial line."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        text = f.read()
    records = []
    # Split on LF only: str.splitlines() also breaks at U+0085, U+2028 and
    # U+2029, which older lines (or other writers) may carry raw inside a
    # string, and both halves would then be dropped as unparseable.
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_safe_segment(s: str) -> bool:
    """A name that is safe as one path segment on Windows and elsewhere."""
    if not isinstance(s, str) or not s or len(s) > 255 or s in (".", ".."):
        return False
    if any(c in _BAD_CHARS for c in s) or s[-1] in ". ":
        return False
    # Windows treats "con.txt" like "con", so only the part before the first dot counts.
    return s.split(".", 1)[0].rstrip(" ").lower() not in WINDOWS_RESERVED


def _discard(tmp: Path) -> None:
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass  # a scanner may hold it for a moment; a stray *.tmp is harmless


# Job folders -------------------------------------------------------------------
# A bench arm's folder, jobs/<run>/<arm>/, is one Harbor job (schedule arms) or
# holds one Harbor job per task (schedule tasks: jobs/<run>/<arm>/<task>/). A
# task race writes RACE_MARKER into the arm folder before its first job starts,
# so the two layouts are told apart without guessing from folder names.

RACE_MARKER = "race.json"


def harbor_jobs(arm_dir: Path) -> list[Path]:
    """The Harbor job folders of a bench arm: the arm folder itself, or under
    a task race (RACE_MARKER) its per-task job folders."""
    arm_dir = Path(arm_dir)
    if not (arm_dir / RACE_MARKER).is_file():
        return [arm_dir]
    try:
        return sorted(d for d in arm_dir.iterdir() if d.is_dir())
    except OSError:
        return []


def mark_race(arm_dir: Path, run_id: str, arm: str) -> None:
    """Marks arm_dir as a task race's arm folder (one Harbor job per task)."""
    arm_dir = Path(arm_dir)
    arm_dir.mkdir(parents=True, exist_ok=True)
    if not (arm_dir / RACE_MARKER).is_file():
        atomic_write_json(arm_dir / RACE_MARKER, {"schedule": "tasks", "run": run_id, "arm": arm,
                                                  "layout": "<arm>/<task>/<trial>"})
