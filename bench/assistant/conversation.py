"""Conversation files, and the live event log the dashboard streams.

Per conversation, under ASSISTANT_DIR/conversations/<cid>/:
- meta.json: rewritten atomically; the copy in memory is authoritative while
  the bench runs, so a scanner holding the file cannot lose an update.
- messages.jsonl: the API history, append-only, one message per line, stored
  exactly as sent and replayed verbatim. Opus 5.5 binds thinking blocks to
  the conversation before them, so an edited line would make every later
  thinking block fail to replay. An assistant message that calls tools is
  written in the same write as the user message holding all its results, so
  a crash can never leave a call without its answer except as a torn last
  line, which load_messages repairs.
- transcript.jsonl: durable dashboard events, append-only. Streamed text
  deltas live only in memory (EventLog) and are replaced on screen by the
  durable event with the same block_key.

A conversation locks (read-only) when the system prompt, the tool list or
the model it was created with has changed, or when its context has grown
past LOCK_CONTEXT_TOKENS. The person then continues in a new conversation
seeded with recap(): plain text, no thinking, made without a model call.
"""

from __future__ import annotations

import bisect
import collections
import copy
import heapq
import json
import os
import re
import secrets
import threading
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import fsutil, settings
from .redact import redact, redact_obj

CID_PATTERN = re.compile(r"c-\d{8}-\d{6}-[0-9a-f]{4}")
LOCK_CONTEXT_TOKENS = 150_000
EPHEMERAL_LIMIT = 5000
RECAP_ENTRIES = 10
RECAP_CHARS = 2000
INTERRUPTED = "interrupted: the bench stopped before this tool ran"
READ_PATIENCE = 2.0


def valid_cid(cid: Any) -> bool:
    return isinstance(cid, str) and CID_PATTERN.fullmatch(cid) is not None


def lock_reason(meta: dict[str, Any], prompt_hash: str, tools_hash: str, model: str) -> str | None:
    """Why this conversation can no longer take a message, or None."""
    if meta.get("status") == "locked":
        return meta.get("locked_reason") or "This conversation is locked."
    if meta.get("prompt_hash") != prompt_hash:
        return "The assistant's instructions were updated since this conversation started."
    if meta.get("tools_hash") != tools_hash:
        return "The assistant's tools were updated since this conversation started."
    if meta.get("model") != model:
        return "The assistant's model changed since this conversation started."
    if (meta.get("last_context_tokens") or 0) > LOCK_CONTEXT_TOKENS:
        return "This conversation is long."
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else settings.ASSISTANT_DIR / "conversations"
        # _lock guards the in-memory metas only and is never held across a
        # write: the SSE route reads meta on the event loop, and a write can
        # spend seconds in PermissionError retries. Writes are ordered by
        # their own locks; _io_lock may be taken while holding _lock, never
        # the other way round.
        self._lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._msg_lock = threading.Lock()
        self._metas: dict[str, dict[str, Any]] = {}
        self._meta_ver: dict[str, int] = {}
        self._written_ver: dict[str, int] = {}

    # Paths -----------------------------------------------------------------

    def _dir(self, cid: str) -> Path:
        if not valid_cid(cid):
            raise KeyError(f"No conversation {cid!r}")
        return self.root / cid

    def _new_cid(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        while True:
            cid = f"c-{stamp}-{secrets.token_hex(2)}"
            if not (self.root / cid).exists() and cid not in self._metas:
                return cid

    # Meta --------------------------------------------------------------------

    def create(self, effort: str, fallbacks: str, title: str | None, design: dict[str, Any] | None,
               seed_text: str | None = None, continued_from: str | None = None, *,
               model: str, prompt_hash: str, tools_hash: str) -> dict[str, Any]:
        with self._lock:
            cid = self._new_cid()
            now = _now()
            meta = {
                "id": cid,
                "title": redact((title or "").strip()[:80]) or "New conversation",
                "created_at": now,
                "updated_at": now,
                "model": model,
                "effort": effort,
                "fallbacks": fallbacks,
                "prompt_hash": prompt_hash,
                "tools_hash": tools_hash,
                "status": "idle",
                "locked_reason": None,
                "turns": 0,
                "requests": 0,
                "spend_usd": 0.0,
                "last_context_tokens": 0,
                "next_seq": 1,
                "designs": [design] if design else [],
                "continued_from": continued_from,
                "last_turn": None,
            }
            directory = self._dir(cid)
            directory.mkdir(parents=True, exist_ok=False)
            self._metas[cid] = meta
            snapshot = self._snapshot(cid)
        self._write_meta(cid, *snapshot)
        with self._lock:
            if seed_text:
                self.commit(cid, {"role": "user", "content": [{"type": "text", "text": seed_text}]})
            return copy.deepcopy(meta)

    def _snapshot(self, cid: str) -> tuple[dict[str, Any], int]:
        """A copy of the meta and its change number; call holding _lock."""
        ver = self._meta_ver.get(cid, 0) + 1
        self._meta_ver[cid] = ver
        return copy.deepcopy(self._metas[cid]), ver

    def _write_meta(self, cid: str, data: dict[str, Any], ver: int) -> None:
        # In memory is authoritative; a locked file is written again next
        # time. A snapshot older than one already written is skipped.
        with self._io_lock:
            if ver < self._written_ver.get(cid, 0):
                return
            fsutil.atomic_write_json(self._dir(cid) / "meta.json", data)
            self._written_ver[cid] = ver

    def _read_meta(self, cid: str) -> dict[str, Any]:
        path = self._dir(cid) / "meta.json"
        deadline = time.monotonic() + READ_PATIENCE
        while True:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                break
            except FileNotFoundError:
                raise KeyError(f"No conversation {cid}") from None
            except PermissionError as exc:
                # A scanner or the atomic replace holds it for a moment.
                if time.monotonic() >= deadline:
                    raise KeyError(f"Could not read conversation {cid}: {exc}") from None
                time.sleep(0.1)
            except (OSError, ValueError) as exc:
                raise KeyError(f"Could not read conversation {cid}: {exc}") from None
        if not isinstance(data, dict) or data.get("id") != cid:
            raise KeyError(f"Could not read conversation {cid}")
        return data

    def meta(self, cid: str) -> dict[str, Any]:
        with self._lock:
            if cid not in self._metas:
                self._metas[cid] = self._read_meta(cid)
            return copy.deepcopy(self._metas[cid])

    def update_meta(self, cid: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            self.meta(cid)
            meta = self._metas[cid]
            meta.update(copy.deepcopy(fields))
            meta["updated_at"] = _now()
            snapshot, ver = self._snapshot(cid)
        self._write_meta(cid, snapshot, ver)
        return copy.deepcopy(snapshot)

    def list(self) -> list[dict[str, Any]]:
        """Every readable conversation, newest first. Unreadable ones are
        listed as such, never fatal."""
        out = []
        if not self.root.exists():
            return out
        for entry in self.root.iterdir():
            if not entry.is_dir() or not valid_cid(entry.name):
                continue
            try:
                out.append(self.meta(entry.name))
            except KeyError as exc:
                out.append({"id": entry.name, "title": "Could not read this conversation", "status": "archived",
                            "error": str(exc).strip("'\""), "created_at": "", "updated_at": ""})
        out.sort(key=lambda m: (m.get("created_at") or "", m["id"]), reverse=True)
        return out

    def archive(self, cid: str) -> dict[str, Any]:
        return self.update_meta(cid, status="archived")

    # Messages ----------------------------------------------------------------

    def _messages_path(self, cid: str) -> Path:
        return self._dir(cid) / "messages.jsonl"

    def commit(self, cid: str, *messages: dict[str, Any]) -> None:
        """Append messages in one write. Raises if the file stayed locked:
        a message that is not on disk must not be treated as sent."""
        self.meta(cid)
        with self._msg_lock:
            fsutil.append_jsonl(self._messages_path(cid), list(messages))

    def load_messages(self, cid: str) -> list[dict[str, Any]]:
        """The history to replay, repaired after a crash: a torn last line is
        cut off, and a tool call left without results is answered with
        "interrupted" error results (appended, never edited in place)."""
        self.meta(cid)
        with self._msg_lock:
            path = self._messages_path(cid)
            _cut_torn_tail(path)
            messages = [m for m in fsutil.read_jsonl(path)
                        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), (list, str))]
            dangling = _dangling_tool_uses(messages)
            if dangling:
                closing = {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tool_id, "is_error": True,
                     "content": json.dumps({"error": "interrupted", "detail": INTERRUPTED})}
                    for tool_id in dangling]}
                fsutil.append_jsonl(path, [closing])
                messages.append(closing)
            return messages

    def read_messages(self, cid: str) -> list[dict[str, Any]]:
        """The history as stored, without repairs (for a recap)."""
        self.meta(cid)
        with self._msg_lock:
            return [m for m in fsutil.read_jsonl(self._messages_path(cid)) if m.get("role") in ("user", "assistant")]

    # Transcript --------------------------------------------------------------

    def append_transcript(self, cid: str, events: list[dict[str, Any]]) -> None:
        # No store lock: EventLog's write lock already orders these writes.
        fsutil.append_jsonl(self._dir(cid) / "transcript.jsonl", events)

    def transcript(self, cid: str, after: int = 0) -> list[dict[str, Any]]:
        events = fsutil.read_jsonl(self._dir(cid) / "transcript.jsonl")
        return [e for e in events if isinstance(e.get("seq"), int) and e["seq"] > after]

    # Recap -------------------------------------------------------------------

    def recap(self, cid: str) -> str:
        meta = self.meta(cid)
        entries: list[str] = []
        for message in self.read_messages(cid):
            role = "You" if message["role"] == "user" else "Assistant"
            for block in _blocks(message.get("content")):
                if block.get("type") != "text" or not isinstance(block.get("text"), str):
                    continue
                text = block["text"].strip()
                if not text:
                    continue
                if len(text) > RECAP_CHARS:
                    text = text[:RECAP_CHARS] + " [...]"
                entries.append(f"{role}: {text}")
        lines = [f'This conversation continues "{meta.get("title") or cid}" ({cid}). '
                 "A plain-text recap of where it ended follows; tool results and reasoning are not included, "
                 "so call the tools again for any figure you need."]
        if not entries:
            lines.append("(The earlier conversation had no messages.)")
        else:
            lines.append("")
            lines.extend(entries[-RECAP_ENTRIES:])
        designs = _design_refs(meta.get("designs") or [])
        if designs:
            lines.append("")
            lines.append("Designs it touched: " + ", ".join(designs) + ".")
        return redact("\n".join(lines))


def _blocks(content: Any) -> Iterable[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _design_refs(designs: list[Any]) -> list[str]:
    out = []
    for ref in designs:
        if isinstance(ref, dict) and ref.get("id"):
            text = f"{ref['id']} v{ref['version']}" if ref.get("version") else str(ref["id"])
            if text not in out:
                out.append(text)
    return out


def _dangling_tool_uses(messages: list[dict[str, Any]]) -> list[str]:
    """Ids of tool calls in the last assistant message that no later user
    message answers."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] != "assistant":
            continue
        calls = [b.get("id") for b in _blocks(messages[i]["content"]) if b.get("type") == "tool_use"]
        answered = {b.get("tool_use_id") for m in messages[i + 1:] if m["role"] == "user"
                    for b in _blocks(m["content"]) if b.get("type") == "tool_result"}
        return [c for c in calls if isinstance(c, str) and c not in answered]
    return []


def _cut_torn_tail(path: Path) -> None:
    """Truncate a last line that a crash left without its newline. Lines are
    written whole, so a line without "\\n" is a torn write and never valid."""
    try:
        with open(path, "r+b") as f:
            size = f.seek(0, os.SEEK_END)
            if size == 0:
                return
            f.seek(size - 1)
            if f.read(1) == b"\n":
                return
            chunk = 65536
            pos = size
            while pos > 0:
                chunk_start = max(0, pos - chunk)
                f.seek(chunk_start)
                data = f.read(pos - chunk_start)
                nl = data.rfind(b"\n")
                if nl >= 0:
                    f.truncate(chunk_start + nl + 1)
                    return
                pos = chunk_start
            f.truncate(0)
    except FileNotFoundError:
        return
    except PermissionError:
        # read_jsonl drops the fragment and append_jsonl starts a fresh line,
        # so leaving it costs nothing but bytes.
        return


def _seq(event: dict[str, Any]) -> int:
    return event["seq"]


class EventLog:
    """Dashboard events of one conversation, in sequence order.

    Durable events are written to transcript.jsonl as well, and all of them
    are held here, earlier ones loaded once when the log is built, so reading
    never goes to disk: the SSE stream polls since() on the event loop.
    Ephemeral events (streamed deltas, status lines) are kept here only, the
    last EPHEMERAL_LIMIT of them, and dropped when a new turn starts. Every
    event is redacted before it is kept or emitted. Sequence numbers continue
    from start_seq, so a browser reconnecting with Last-Event-ID resumes
    where it left off.
    """

    def __init__(self, store: ConversationStore, cid: str, start_seq: int,
                 history: list[dict[str, Any]] | None = None) -> None:
        self._store = store
        self._cid = cid
        self._cond = threading.Condition()
        # Taken before _cond by durable appends only: transcript lines stay in
        # sequence order while the write itself (fsync, retries while a
        # scanner holds the file) never blocks a reader.
        self._write_lock = threading.Lock()
        if history is None:
            history = store.transcript(cid, 0)
        self._durable: list[dict[str, Any]] = sorted(
            (dict(e) for e in history if isinstance(e.get("seq"), int) and e["seq"] < start_seq), key=_seq)
        self._ephemeral: collections.deque[dict[str, Any]] = collections.deque(maxlen=EPHEMERAL_LIMIT)
        self._next_seq = start_seq
        self._turn = 0

    @property
    def next_seq(self) -> int:
        with self._cond:
            return self._next_seq

    @property
    def last_seq(self) -> int:
        with self._cond:
            return self._next_seq - 1

    def start_turn(self, turn: int) -> None:
        with self._cond:
            self._turn = turn
            self._ephemeral.clear()

    def _new_event(self, type: str, turn: Any, data: dict[str, Any]) -> dict[str, Any]:
        event = {"seq": self._next_seq, "type": type, "at": _now(),
                 "turn": self._turn if turn is None else turn, **data}
        self._next_seq += 1
        return event

    def append(self, type: str, durable: bool, **data: Any) -> dict[str, Any]:
        turn = data.pop("turn", None)
        data = redact_obj(data)
        if not durable:
            with self._cond:
                event = self._new_event(type, turn, data)
                self._ephemeral.append(event)
                self._cond.notify_all()
            return dict(event)
        with self._write_lock:
            with self._cond:
                event = self._new_event(type, turn, data)
                self._durable.append(event)
                self._cond.notify_all()
            try:
                self._store.append_transcript(self._cid, [event])
            except OSError:
                pass  # still shown live; only a reload would miss it
        return dict(event)

    def since(self, seq: int) -> list[dict[str, Any]]:
        with self._cond:
            durable = self._durable[bisect.bisect_right(self._durable, seq, key=_seq):]
            live = []
            for event in reversed(self._ephemeral):
                if event["seq"] <= seq:
                    break
                live.append(event)
        live.reverse()
        return [dict(e) for e in heapq.merge(durable, live, key=_seq)]

    def wait(self, seq: int, timeout: float) -> list[dict[str, Any]]:
        with self._cond:
            self._cond.wait_for(lambda: self._next_seq - 1 > seq, timeout=timeout)
        return self.since(seq)
