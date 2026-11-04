"""The assistant's turn loop: Claude Opus 5.5, streamed, with the bench's tools.

A turn starts when the person sends a message and runs in its own daemon
thread ("assistant-<cid>"), one turn per process. It is the manual streamed
tool loop from the claude-api skill (python/claude-api/streaming.md,
"Streaming with Tool Use"), not the SDK's tool runner, because the bench must:
- keep the history append-only (Opus 5.5 binds thinking blocks to the
  conversation before them), so a refused or truncated reply adds nothing
  and an error never edits what was sent;
- check the per-turn and daily spend caps before every request;
- stop mid-stream when the person presses Stop;
- drop what a declined model wrote before a server-side fallback took over.

Each step:
1. Stop flag, then caps (spend so far this turn; today's spend plus the
   worst case of the next request), before any request.
2. Stream the request. Deltas go to the dashboard as they arrive; a ValueError
   from the stream is tool-input JSON the SDK could not parse at all, so the
   same request is sent again, at most twice. Typed API errors propagate.
3. Price the reply per usage iteration and record it.
4. refusal: nothing is appended and no tool runs. max_tokens with a tool
   call: the same (a cut-off input parses as a valid partial object). An
   empty reply: the same (the API rejects an empty assistant message once
   another message follows it).
5. Otherwise the reply is echoed back verbatim, minus blocks a declined
   model wrote before the last fallback block. Tool calls run one by one
   through the injected executor, which validates input before running;
   all results go in one user message, committed in the same write as the
   reply that asked for them.

Errors are classified most specific first and shown with what to do next.
Two arrive untyped and are classified here: an error event sent after the
stream started (a plain APIStatusError on the 200 response, so its body's
error type decides), and a connection that drops or times out while the
body is read (the transport's own exception; the SDK does not wrap it).
Nothing here sends a role "system" message: bench notices are dashboard
events only.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import re
import sys
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx2
from fastapi.sse import ServerSentEvent

from .. import fsutil
from ..settings import ASSISTANT_EFFORTS, AssistantSettings, Config, Price
from . import client as client_mod
from . import costs
from .conversation import ConversationStore, EventLog, lock_reason, valid_cid
from .prompt import PROMPT_HASH, SYSTEM_PROMPT
from .redact import contains_secret, redact
from .types import SAVES_PER_TURN, ToolContext, ToolOutcome

MAX_TEXT_CHARS = 8000
JSON_RETRIES = 2            # re-issues after unparseable tool-input JSON
STREAM_SECONDS = 300.0      # an SSE connection ends after this; the browser reconnects
BATCH_SECONDS = 0.1
RECONNECT_MS = 1000
CHARS_PER_TOKEN = 3         # deliberately low, so the cap reserve over- rather than under-estimates
# After a crash, sequence numbers restart this far past the last one on
# disk: the crashed turn's streamed deltas were never saved, and a page still
# open resumes from the last one it saw. Every delta carries at least one
# token, so one turn emits at most about steps x max_tokens x attempts.
RECOVERY_SEQ_GAP = 10_000_000
KEY_TEXT_MESSAGE = "This looks like an API key. Set keys on Setup; never paste them into chat."
EMPTY_REPLY_MESSAGE = "Claude returned an empty reply; nothing was added. Send your message again."
NO_VISIBLE_ANSWER = "Claude finished without a visible answer; ask it to answer again."
NOT_RUN = json.dumps({"error": "not_run", "detail": "not run: stopped"})

_DESIGN_ID = re.compile(r"d-[a-z0-9][a-z0-9-]{0,40}")
# A 400 about these means the stored history can no longer be replayed
# (preserved thinking), and retrying the same body never clears it.
_REPLAY_400 = re.compile(r"thinking|signature|prefix", re.I)
_TOO_LONG_400 = re.compile(r"prompt is too long|too many tokens|context window", re.I)
# The turn_error kind of an error event that arrives mid-stream, by its
# body's error.type (anthropic.types.shared.ErrorType). Anything else is
# treated as a server error: retrying replays the same, valid history.
_STREAM_ERROR_KINDS = {
    "overloaded_error": "server",
    "api_error": "server",
    "timeout_error": "timeout",
    "rate_limit_error": "rate_limited",
    "invalid_request_error": "bad_request",
    "billing_error": "bad_request",
    "authentication_error": "auth",
    "permission_error": "permission",
    "not_found_error": "model_not_found",
}
# Server-side tool calls. Before the last fallback block a declined model's
# call is echoed only together with its result (claude-api skill, refusal
# section); none are declared today, so this is defensive.
_SERVER_USE = ("server_tool_use", "mcp_tool_use")


class _InvalidJson(Exception):
    pass


class _CapReached(Exception):
    pass


class _Stopped(Exception):
    pass


@dataclass
class _Turn:
    cid: str
    turn: int
    started_at: str
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    usd: float = 0.0
    requests: int = 0
    step: int = 0


@dataclass
class _StreamState:
    started: bool = False
    model: str | None = None
    start_usage: Any = None
    output_tokens: int = 0
    # message_delta (the only output count after message_start) comes at the
    # very end, so a stream that ends early is priced from what it streamed.
    streamed_chars: int = 0


class _Incomplete(Exception):
    """The stream ended cleanly but before the reply finished."""


def echo_kept(content: list[dict[str, Any]]) -> list[int]:
    """Indices of the reply blocks to echo back. Before the last "fallback"
    block, a declined model's thinking, redacted_thinking and tool_use blocks
    are dropped, as are server-tool calls without their result and any other
    model-internal block; text, paired server-tool blocks, earlier fallback
    markers and everything from the boundary on are kept."""
    boundary = max((i for i, b in enumerate(content) if b.get("type") == "fallback"), default=None)
    if boundary is None:
        return list(range(len(content)))
    result_ids = {b.get("tool_use_id") for b in content if b.get("tool_use_id")}
    paired = {b.get("id") for b in content[:boundary]
              if b.get("type") in _SERVER_USE and b.get("id") in result_ids}
    kept = []
    for i, block in enumerate(content):
        kind = block.get("type")
        if i >= boundary or kind in ("text", "fallback"):
            kept.append(i)
        elif kind in _SERVER_USE and block.get("id") in paired:
            kept.append(i)
        elif kind != "tool_result" and block.get("tool_use_id") in paired:
            kept.append(i)
    return kept


def echo_filter(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [content[i] for i in echo_kept(content)]


def _design_ref(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("A design reference is {id, version}")
    design_id, version = value.get("id"), value.get("version")
    if not (isinstance(design_id, str) and _DESIGN_ID.fullmatch(design_id)):
        raise ValueError("Not a design id")
    if not (isinstance(version, int) and not isinstance(version, bool) and 0 < version < 100000):
        raise ValueError("A design version is a positive whole number")
    return {"id": design_id, "version": version}


def _check_tool_defs(tool_defs: list[dict[str, Any]]) -> None:
    names = [t.get("name") for t in tool_defs]
    if len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        raise ValueError("Assistant tool names must be unique, non-empty strings")
    for tool in tool_defs:
        # Streamed requests with client tools stream inputs eagerly; the
        # executor validates every input, since the API then does not.
        if tool.get("eager_input_streaming") is not True:
            raise ValueError(f"Tool {tool.get('name')} must set eager_input_streaming: true")


def _api_detail(exc: anthropic.APIStatusError) -> str:
    body = exc.body
    message = None
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        message = body["error"].get("message")
    return redact(str(message or exc.message))[:400]


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    try:
        value = exc.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _status_kind(exc: anthropic.APIStatusError) -> str | None:
    """The turn_error kind of an API error status, most specific first; None
    for a 4xx status with no kind of its own."""
    if exc.status_code < 400:
        # An error event after the stream started: the SDK raises it on the
        # 200 response, so only the body's error type tells what it was.
        return _STREAM_ERROR_KINDS.get(exc.type or "", "server")
    if isinstance(exc, anthropic.AuthenticationError):
        return "auth"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "permission"
    if isinstance(exc, anthropic.NotFoundError):
        return "model_not_found"
    if isinstance(exc, anthropic.RateLimitError):
        return "rate_limited"
    if isinstance(exc, anthropic.BadRequestError):
        return "bad_request"
    if exc.status_code >= 500:
        return "server"
    return None


def _status_error(exc: anthropic.APIStatusError) -> dict[str, Any]:
    code = exc.status_code
    kind = _status_kind(exc)
    detail = _api_detail(exc)
    label = f" ({code})" if code >= 400 else ""
    if kind == "auth":
        return {"kind": "auth", "retryable": False,
                "message": f"{client_mod.KEY_NAME} was rejected{label}. Replace it on Setup."}
    if kind == "permission":
        return {"kind": "permission", "retryable": False,
                "message": f"This API key may not use {client_mod.MODEL}{label}: {detail}"}
    if kind == "model_not_found":
        return {"kind": "model_not_found", "retryable": False,
                "message": f"{client_mod.MODEL} is not available to this API key{label}: {detail}"}
    if kind == "rate_limited":
        wait = _retry_after(exc) if code >= 400 else None
        return {"kind": "rate_limited", "retryable": True, "retry_after_s": wait,
                "message": f"Rate limited by the API{label}."
                           + (f" Retry in {wait:.0f} s." if wait is not None else " Retry in a minute.")}
    if kind == "bad_request":
        error = {"kind": "bad_request", "retryable": False,
                 "message": f"The API rejected the request{label}: {detail}"}
        if _REPLAY_400.search(detail):
            error["lock"] = "This conversation's history can no longer be replayed."
        elif _TOO_LONG_400.search(detail):
            error["lock"] = "This conversation is too long."
        return error
    if kind == "server":
        return {"kind": "server", "retryable": True,
                "message": f"The API had a server error{label}: {detail.rstrip('. ')}. Retry in a moment."}
    if kind == "timeout":
        return _timeout_error()
    return {"kind": "bad_request", "retryable": code in (408, 409),
            "message": f"The API returned {code}: {detail}"}


def _timeout_error() -> dict[str, Any]:
    return {"kind": "timeout", "retryable": True, "message": "The API did not answer in time. Retry."}


def _connection_error() -> dict[str, Any]:
    return {"kind": "connection", "retryable": True,
            "message": "Could not reach the Anthropic API. Check the connection and retry."}


def _blank(content: list[dict[str, Any]]) -> bool:
    """True for a reply with nothing to echo: no blocks, or only blank text
    and fallback markers (an audit marker, not content). The API rejects a
    non-final assistant message like that, so committing it would make
    every later request in the conversation a 400."""
    return all(b.get("type") == "fallback"
               or (b.get("type") == "text" and not str(b.get("text") or "").strip()) for b in content)


def _visible(content: list[dict[str, Any]]) -> bool:
    """Whether the person sees an answer: non-blank text or a tool call."""
    return any(b.get("type") == "tool_use"
               or (b.get("type") == "text" and str(b.get("text") or "").strip()) for b in content)


def _request_id(stream: Any) -> str | None:
    try:
        value = stream.request_id
    except Exception:  # no response headers on a fake or a failed stream
        return None
    return value if isinstance(value, str) else None


def _log(message: str) -> None:
    print(f"[assistant] {redact(message)}", file=sys.stderr)


class AssistantEngine:
    """Runs assistant turns and serves their events.

    ctx_factory(cid) builds the read-only ToolContext for a turn (the server
    owns what it holds); the engine replaces its spend view and saves_left
    with this turn's own. tool_defs and execute_tool come from tools.py, so
    the tool list is pinned per conversation by TOOLS_HASH, computed here.
    """

    def __init__(self, store: ConversationStore, spend: costs.SpendBook,
                 ctx_factory: Callable[[str], ToolContext],
                 settings_getter: Callable[[], AssistantSettings],
                 cfg_getter: Callable[[], Config],
                 tool_defs: list[dict[str, Any]],
                 execute_tool: Callable[[ToolContext, str, Any], ToolOutcome],
                 client_factory: Callable[[], anthropic.Anthropic] = client_mod.make_client) -> None:
        _check_tool_defs(tool_defs)
        self._store = store
        self._spend = spend
        self._ctx_factory = ctx_factory
        self._settings = settings_getter
        self._cfg = cfg_getter
        self._tool_defs = copy.deepcopy(list(tool_defs))
        self._tool_names = frozenset(t["name"] for t in self._tool_defs)
        self._execute_tool = execute_tool
        self._client_factory = client_factory
        self.tools_hash = "sha256:" + fsutil.sha256_hex(fsutil.canonical_json(self._tool_defs))
        self._fixed_chars = len(SYSTEM_PROMPT) + len(fsutil.canonical_json(self._tool_defs))
        self._lock = threading.RLock()
        self._gate = threading.Lock()          # held for the whole of the one active turn
        self._active: _Turn | None = None
        self._logs: dict[str, EventLog] = {}
        self._closing = False
        self._recover()

    # Startup -------------------------------------------------------------------

    def _recover(self) -> None:
        """A bench that stopped mid-turn left the conversation busy and maybe
        a tool call without results: make it idle and repair the history.
        meta.next_seq is saved only when a turn finishes, so sequence numbers
        restart past both it and the transcript, by more than one turn's
        worth of unsaved live events a browser may have seen."""
        s = self._settings()
        gap = max(RECOVERY_SEQ_GAP, s.max_steps * s.max_tokens * (JSON_RETRIES + 1) + 10_000)
        for meta in self._store.list():
            if meta.get("status") != "busy" or meta.get("error"):
                continue
            cid = meta["id"]
            try:
                self._store.load_messages(cid)
                durable = self._store.transcript(cid, 0)
                first_seq = max([int(meta.get("next_seq") or 1)] + [e["seq"] + 1 for e in durable])
                self._store.update_meta(cid, status="idle", next_seq=first_seq + gap)
            except (KeyError, OSError) as exc:
                _log(f"Could not recover conversation {cid}: {exc}")

    # Views ---------------------------------------------------------------------

    def _log_for(self, cid: str) -> EventLog:
        """The conversation's event log, built once with its durable history
        loaded, so reading events never goes to disk again."""
        with self._lock:
            log = self._logs.get(cid)
        if log is not None:
            return log
        meta = self._store.meta(cid)
        durable = self._store.transcript(cid, 0)
        first_seq = max([int(meta.get("next_seq") or 1)] + [e["seq"] + 1 for e in durable])
        with self._lock:
            # Nothing appends before the log is registered, so a log another
            # thread registered meanwhile is just as complete.
            log = self._logs.get(cid)
            if log is None:
                log = EventLog(self._store, cid, first_seq, history=durable)
                self._logs[cid] = log
            return log

    def _active_view(self) -> dict[str, Any] | None:
        with self._lock:
            t = self._active
            return {"conversation_id": t.cid, "turn": t.turn, "started_at": t.started_at} if t else None

    def _view(self, meta: dict[str, Any]) -> dict[str, Any]:
        view = dict(meta)
        with self._lock:
            active = self._active is not None and self._active.cid == meta.get("id")
        view["active"] = active
        if not active and view.get("status") != "archived" and not view.get("error"):
            reason = lock_reason(meta, PROMPT_HASH, self.tools_hash, client_mod.MODEL)
            if reason:
                view["status"] = "locked"
                view["locked_reason"] = reason
        return view

    def status(self) -> dict[str, Any]:
        s = self._settings()
        key_set = client_mod.key_available()
        today = self._spend.today_usd()
        reason = None
        if not s.enabled:
            reason = "The assistant is turned off (assistant.enabled in config)."
        elif not key_set:
            reason = f"Set {client_mod.KEY_NAME} on Setup to use the assistant."
        elif today >= s.daily_cap_usd:
            reason = (f"Assistant daily cap ${s.daily_cap_usd:.2f} reached; raise assistant.daily_cap_usd "
                      "in config or wait until tomorrow.")
        else:
            # start_turn refuses once today's spend plus the next request's worst
            # case passes the cap; the smallest such case is a new conversation.
            reserve = self._reserve_usd(self._cfg(), s, {"fallbacks": s.fallbacks, "last_context_tokens": 0}, [])
            if today + reserve > s.daily_cap_usd:
                reason = (f"Assistant daily cap ${s.daily_cap_usd:.2f} nearly reached: ${today:.2f} spent today "
                          f"and a message could cost up to ${reserve:.2f}.")
        return {
            "enabled": s.enabled,
            "available": reason is None,
            "reason": reason,
            "model": client_mod.MODEL,
            "effort_default": s.effort,
            "fallbacks": s.fallbacks,
            "key_set": key_set,
            "today_usd": round(today, 6),
            "daily_cap_usd": s.daily_cap_usd,
            "spend": {"today_usd": round(today, 6), "daily_cap_usd": s.daily_cap_usd,
                      "per_turn_cap_usd": s.per_turn_cap_usd},
            "active_turn": self._active_view(),
        }

    def require(self, cid: str) -> None:
        """Raise KeyError unless the conversation exists. The SSE route calls
        this from a dependency: FastAPI runs a generator endpoint's body only
        after the 200 response has started, too late to answer 404."""
        self._store.meta(cid)

    def conversations(self) -> list[dict[str, Any]]:
        return [self._view(m) for m in self._store.list()]

    def conversation(self, cid: str, after: int = 0) -> dict[str, Any]:
        """The GET /conversations/{cid} payload: meta, events after `after`,
        the next sequence number and the active turn."""
        meta = self._store.meta(cid)
        log = self._log_for(cid)
        return {"conversation": self._view(meta), "events": log.since(int(after)),
                "next_seq": log.next_seq, "active_turn": self._active_view()}

    def events(self, cid: str, after: int = 0) -> list[dict[str, Any]]:
        self._store.meta(cid)
        return self._log_for(cid).since(int(after))

    def usage(self, days: int = 30) -> dict[str, Any]:
        rows = self._spend.days(days)
        return {"days": rows, "total_usd": round(sum(r["usd"] for r in rows), 6)}

    # Conversations -------------------------------------------------------------

    def create_conversation(self, title: str | None = None, effort: str | None = None,
                            design: dict[str, Any] | None = None) -> dict[str, Any]:
        s = self._settings()
        if not s.enabled:
            raise RuntimeError("The assistant is turned off (assistant.enabled in config).")
        effort = effort or s.effort
        if effort not in ASSISTANT_EFFORTS:
            raise ValueError(f"Effort must be one of {', '.join(ASSISTANT_EFFORTS)}")
        if title is not None and not isinstance(title, str):
            raise ValueError("A title is text")
        meta = self._store.create(effort, s.fallbacks, title, _design_ref(design), model=client_mod.MODEL,
                                  prompt_hash=PROMPT_HASH, tools_hash=self.tools_hash)
        return self._view(meta)

    def continue_conversation(self, cid: str) -> dict[str, Any]:
        """A new conversation whose first message is a plain-text recap of
        this one. No model call: nothing is spent."""
        s = self._settings()
        if not s.enabled:
            raise RuntimeError("The assistant is turned off (assistant.enabled in config).")
        old = self._store.meta(cid)
        recap = self._store.recap(cid)
        title = f"{old.get('title') or 'Conversation'} (continued)"
        effort = old.get("effort") if old.get("effort") in ASSISTANT_EFFORTS else s.effort
        meta = self._store.create(effort, s.fallbacks, title, None, seed_text=recap, continued_from=cid,
                                  model=client_mod.MODEL, prompt_hash=PROMPT_HASH, tools_hash=self.tools_hash)
        meta = self._store.update_meta(meta["id"], designs=old.get("designs") or [])
        self._log_for(meta["id"]).append("user_message", True, text=recap, attach_design=None, recap=True)
        return self._view(meta)

    def archive(self, cid: str) -> dict[str, Any]:
        with self._lock:
            if self._active is not None and self._active.cid == cid:
                raise RuntimeError("Stop the running turn before archiving this conversation.")
            return self._view(self._store.archive(cid))

    # Turns ---------------------------------------------------------------------

    def _check_open(self, s: AssistantSettings, meta: dict[str, Any]) -> None:
        if not s.enabled:
            raise RuntimeError("The assistant is turned off (assistant.enabled in config).")
        if not client_mod.key_available():
            raise RuntimeError(f"Set {client_mod.KEY_NAME} on Setup to use the assistant.")
        if meta.get("status") == "archived":
            raise RuntimeError("This conversation is archived.")
        reason = lock_reason(meta, PROMPT_HASH, self.tools_hash, client_mod.MODEL)
        if reason:
            raise RuntimeError(f"{reason} Continue in a new conversation.")

    def start_turn(self, cid: str, text: str, attach_design: dict[str, Any] | None = None) -> dict[str, Any]:
        s = self._settings()
        meta = self._store.meta(cid)
        self._check_open(s, meta)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Type a message first.")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"Messages are limited to {MAX_TEXT_CHARS} characters.")
        if contains_secret(text):
            raise ValueError(KEY_TEXT_MESSAGE)
        attach = _design_ref(attach_design)
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if attach:
            content.append({"type": "text", "text": f"The user has design {attach['id']} version "
                                                    f"{attach['version']} open in the dashboard."})
        user_message = {"role": "user", "content": content}

        def prepare(meta: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
            self._store.commit(cid, user_message)
            fields: dict[str, Any] = {}
            if attach and attach not in (meta.get("designs") or []):
                fields["designs"] = (meta.get("designs") or []) + [attach]
            if meta.get("turns", 0) == 0 and meta.get("title") in (None, "", "New conversation"):
                fields["title"] = redact(text.strip().splitlines()[0][:60])
            return fields

        return self._begin(cid, s, prepare, extra=[user_message],
                           first_event=("user_message", {"text": text, "attach_design": attach}))

    def retry_turn(self, cid: str) -> dict[str, Any]:
        """Run the loop again from the stored history, after a turn that
        ended in an error. Nothing new is added to the conversation."""
        s = self._settings()
        meta = self._store.meta(cid)
        self._check_open(s, meta)
        if (meta.get("last_turn") or {}).get("status") != "error":
            raise RuntimeError("Retry is only possible right after a turn that ended in an error.")

        def prepare(meta: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
            if not history or history[-1].get("role") != "user":
                raise RuntimeError("Nothing to retry: the conversation does not end with your message.")
            return {}

        return self._begin(cid, s, prepare, extra=[], first_event=None)

    def _begin(self, cid: str, s: AssistantSettings,
               prepare: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]],
               extra: list[dict[str, Any]], first_event: tuple[str, dict[str, Any]] | None) -> dict[str, Any]:
        with self._lock:
            if self._closing:
                raise RuntimeError("The bench is shutting down.")
            if not self._gate.acquire(blocking=False):
                raise RuntimeError("The assistant is already answering a message. Wait for it, or press Stop.")
            busy = False
            try:
                meta = self._store.meta(cid)
                history = self._store.load_messages(cid)
                problem = self._daily_cap_problem(self._cfg(), s, meta, history + extra)
                if problem:
                    raise ValueError(problem)
                fields = prepare(meta, history)
                turn = int(meta.get("turns") or 0) + 1
                self._store.update_meta(cid, status="busy", turns=turn, **fields)
                busy = True
                log = self._log_for(cid)
                log.start_turn(turn)
                accepted = None
                if first_event:
                    accepted = log.append(first_event[0], True, **first_event[1])
                started_event = log.append("turn_started", True, **({} if first_event else {"retry": True}))
                t = _Turn(cid=cid, turn=turn, started_at=started_event["at"])
                t.thread = threading.Thread(target=self._run_turn, args=(t,), name=f"assistant-{cid}", daemon=True)
                self._active = t
                t.thread.start()
                return {"turn": turn, "accepted_seq": (accepted or started_event)["seq"]}
            except BaseException:
                # Refused or failed before the thread took over: nothing runs,
                # so the gate and the conversation are freed here.
                self._active = None
                self._gate.release()
                if busy:
                    try:
                        self._store.update_meta(cid, status="idle")
                    except Exception:
                        pass
                raise

    def stop(self, cid: str) -> None:
        """Ask the active turn of this conversation to stop. The stream is
        closed at the next event; a running tool finishes first."""
        with self._lock:
            if self._active is not None and self._active.cid == cid:
                self._active.stop.set()

    def wait(self, timeout: float = 10.0) -> bool:
        """Wait for the active turn to finish (tests, shutdown). True if idle."""
        with self._lock:
            t = self._active
        if t is None or t.thread is None:
            return True
        t.thread.join(timeout)
        return not t.thread.is_alive()

    def shutdown(self, timeout: float = 10.0) -> None:
        with self._lock:
            self._closing = True
            t = self._active
            if t is not None:
                t.stop.set()
        if t is not None and t.thread is not None:
            t.thread.join(timeout)

    # Caps ----------------------------------------------------------------------

    def _rates(self, cfg: Config, meta: dict[str, Any]) -> Price:
        """The dearest rates the next request could be billed at: the model,
        plus the fallback models when fallbacks are on."""
        models = [client_mod.MODEL]
        if meta.get("fallbacks", "default") == "default":
            models += list(client_mod.FALLBACK_MODELS)
        prices = [costs.model_price(cfg, m)[0] for m in models]
        return Price(input=max(p.input for p in prices), output=max(p.output for p in prices),
                     cache_read=None)

    def _reserve_usd(self, cfg: Config, s: AssistantSettings, meta: dict[str, Any],
                     history: list[dict[str, Any]]) -> float:
        """Worst case of the next request: its whole prompt written to the
        cache, plus max_tokens of output."""
        prompt_chars = self._fixed_chars + len(json.dumps(history, ensure_ascii=False))
        tokens = max(int(meta.get("last_context_tokens") or 0), prompt_chars // CHARS_PER_TOKEN)
        rates = self._rates(cfg, meta)
        return (tokens * rates.input * costs.CACHE_WRITE_MULT + s.max_tokens * rates.output) / 1e6

    def _daily_cap_problem(self, cfg: Config, s: AssistantSettings, meta: dict[str, Any],
                           history: list[dict[str, Any]]) -> str | None:
        today = self._spend.today_usd()
        reserve = self._reserve_usd(cfg, s, meta, history)
        if today + reserve > s.daily_cap_usd:
            return (f"Assistant daily cap ${s.daily_cap_usd:.2f} reached: ${today:.2f} spent today, and the next "
                    f"request could cost up to ${reserve:.2f}.")
        return None

    def _cap_problem(self, cfg: Config, s: AssistantSettings, meta: dict[str, Any], t: _Turn,
                     history: list[dict[str, Any]]) -> str | None:
        if t.usd >= s.per_turn_cap_usd:
            return (f"This message reached the per-message cap of ${s.per_turn_cap_usd:.2f} "
                    f"(${t.usd:.2f} spent). Send another message to continue.")
        return self._daily_cap_problem(cfg, s, meta, history)

    def _spend_view(self, t: _Turn) -> dict[str, Any]:
        s = self._settings()
        return {"turn_usd": round(t.usd, 6), "today_usd": round(self._spend.today_usd(), 6),
                "per_turn_cap_usd": s.per_turn_cap_usd, "daily_cap_usd": s.daily_cap_usd}

    # The turn ------------------------------------------------------------------

    def _run_turn(self, t: _Turn) -> None:
        status = "error"
        error: dict[str, Any] | None = None
        try:
            status = self._turn_body(t)
        except _Incomplete:
            error = {"kind": "connection", "retryable": True,
                     "message": "The reply ended before it finished; nothing was run. Retry."}
        except _InvalidJson:
            error = {"kind": "invalid_json", "retryable": True,
                     "message": "The reply's tool input could not be read three times in a row. "
                                "Retry, or ask for a smaller step."}
        except client_mod.AssistantUnavailable as exc:
            error = {"kind": "auth", "retryable": False, "message": str(exc)}
        except anthropic.APIStatusError as exc:
            error = _status_error(exc)
        except anthropic.APITimeoutError:
            error = _timeout_error()
        except anthropic.APIConnectionError:
            error = _connection_error()
        # The SDK wraps transport failures only while it makes the request; one
        # while the streamed body is read arrives as the transport's exception.
        except httpx2.TimeoutException:
            error = _timeout_error()
        except httpx2.TransportError:
            error = _connection_error()
        except Exception as exc:  # the turn thread must always finish its turn
            _log(f"Turn {t.turn} of {t.cid} failed: {type(exc).__name__}: {exc}")
            error = {"kind": "internal", "retryable": False,
                     "message": f"The assistant hit an internal error ({type(exc).__name__}): {redact(str(exc))[:300]}"}
        finally:
            self._finish(t, status, error)

    def _finish(self, t: _Turn, status: str, error: dict[str, Any] | None) -> None:
        try:
            log = self._log_for(t.cid)
            fields: dict[str, Any] = {"status": "idle"}
            if error is not None:
                status = "error"
                lock = error.pop("lock", None)
                error.setdefault("retry_after_s", None)
                log.append("turn_error", True, **error)
                if lock:
                    message = f"{lock} Continue in a new conversation."
                    log.append("notice", True, level="warn", code="locked", message=message)
                    fields.update(status="locked", locked_reason=lock)
            log.append("turn_finished", True, status=status, cost_usd=round(t.usd, 6), requests=t.requests)
            fields["next_seq"] = log.next_seq
            fields["last_turn"] = {"turn": t.turn, "status": status,
                                   "error_kind": error["kind"] if error else None,
                                   "retryable": bool(error and error.get("retryable"))}
            self._store.update_meta(t.cid, **fields)
        except Exception as exc:
            _log(f"Could not finish turn {t.turn} of {t.cid}: {type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                if self._active is t:
                    self._active = None
            self._gate.release()

    def _turn_body(self, t: _Turn) -> str:
        s = self._settings()
        cfg = self._cfg()
        log = self._log_for(t.cid)
        ctx = dataclasses.replace(self._ctx_factory(t.cid), spend=lambda: self._spend_view(t),
                                  saves_left=[SAVES_PER_TURN])
        api = self._client_factory()
        history = self._store.load_messages(t.cid)
        for step in range(1, s.max_steps + 1):
            t.step = step
            meta = self._store.meta(t.cid)
            if t.stop.is_set():
                return self._stopped(log)
            problem = self._cap_problem(cfg, s, meta, t, history)
            if problem:
                log.append("notice", True, level="warn", code="cap", message=problem)
                return "cap_reached"
            kwargs = client_mod.request_kwargs(s, meta, self._tool_defs, history)
            try:
                message = self._stream_step(api, kwargs, t, step, cfg, s, history)
            except _Stopped:
                return self._stopped(log)
            except _CapReached as exc:
                log.append("notice", True, level="warn", code="cap", message=str(exc))
                return "cap_reached"

            stop_reason = message.stop_reason
            if stop_reason is None:
                # The body ended before message_delta: a truncated tool input
                # still parses, so nothing of this reply may be kept or run.
                raise _Incomplete()
            if stop_reason == "refusal":
                details = getattr(message, "stop_details", None)
                category = getattr(details, "category", None) if details is not None else None
                about = f" (category: {category})" if category else ""
                log.append("notice", True, level="warn", code="refusal", category=category,
                           message=f"Claude declined this request{about}. Nothing was run or added to the "
                                   "conversation; rephrase and try again.")
                return "refused"

            raw = message.to_dict(mode="json")["content"]
            kept = echo_kept(raw)
            content = [raw[i] for i in kept]
            if _blank(content):
                # Nothing is appended (consecutive user messages are valid),
                # so the conversation stays replayable.
                log.append("notice", True, level="warn", code="empty", message=EMPTY_REPLY_MESSAGE)
                return "done"
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            cut_off = stop_reason in ("max_tokens", "model_context_window_exceeded")
            if cut_off and tool_uses:
                log.append("notice", True, level="warn", code="truncated",
                           message="Reply cut off; nothing was run. Ask for a smaller design or one step at a time.")
                return "truncated"
            reply = {"role": "assistant", "content": content}
            if stop_reason == "pause_turn":
                # Only server tools pause a turn, and none are declared; kept
                # so a paused reply is echoed back for the server to resume.
                self._store.commit(t.cid, reply)
                history.append(reply)
                self._emit_blocks(log, t, step, raw, kept)
                continue
            if not tool_uses:
                self._store.commit(t.cid, reply)
                history.append(reply)
                self._emit_blocks(log, t, step, raw, kept)
                if not _visible(content):
                    # Only thinking: part of the history, but the person saw no answer.
                    log.append("notice", True, level="warn", code="empty", message=NO_VISIBLE_ANSWER)
                if cut_off:
                    log.append("notice", True, level="warn", code="truncated",
                               message="Reply cut off at the length limit.")
                    return "truncated"
                return "done"

            results, outcomes, stopped = self._execute_tools(ctx, t, log, tool_uses)
            answer = {"role": "user", "content": results}
            self._store.commit(t.cid, reply, answer)
            history.extend([reply, answer])
            self._emit_blocks(log, t, step, raw, kept)
            self._emit_results(log, t, outcomes)
            if stopped:
                return self._stopped(log)
        log.append("notice", True, level="info", code="max_steps",
                   message=f"Stopped after {s.max_steps} steps; ask me to continue.")
        return "max_steps"

    def _stopped(self, log: EventLog) -> str:
        log.append("notice", True, level="info", code="stopped", message="Stopped. Anything already run is kept.")
        return "stopped"

    def _stream_step(self, api: anthropic.Anthropic, kwargs: dict[str, Any], t: _Turn, step: int,
                     cfg: Config, s: AssistantSettings, history: list[dict[str, Any]]) -> Any:
        """One request, re-issued unchanged (at most JSON_RETRIES times) when
        the SDK cannot parse a tool input at all: it raised before the block
        completed, so there is no tool_use_id to answer."""
        failures = 0
        log = self._log_for(t.cid)
        while True:
            try:
                return self._stream_once(api, kwargs, t, step, cfg)
            except ValueError:
                failures += 1
                if failures > JSON_RETRIES:
                    raise _InvalidJson() from None
                log.append("notice", True, level="info", code="retry",
                           message=f"The reply's tool input could not be read; asking again "
                                   f"(attempt {failures + 1} of {JSON_RETRIES + 1}).")
            if t.stop.is_set():
                raise _Stopped()
            problem = self._cap_problem(cfg, s, self._store.meta(t.cid), t, history)
            if problem:
                raise _CapReached(problem)

    def _stream_once(self, api: anthropic.Anthropic, kwargs: dict[str, Any], t: _Turn, step: int,
                     cfg: Config) -> Any:
        log = self._log_for(t.cid)
        state = _StreamState()
        request_id = None
        message = None
        try:
            with api.beta.messages.stream(**kwargs) as stream:
                # The response headers are in once the stream is entered, so
                # a stream that fails part-way is still logged with its id.
                request_id = _request_id(stream)
                for event in stream:
                    if t.stop.is_set():
                        break
                    self._on_event(event, state, log, t, step)
                if not t.stop.is_set():
                    message = stream.get_final_message()
        except Exception:
            # Unparseable tool input (ValueError) or an error event mid-stream:
            # whatever was generated before it is billed, so it is recorded.
            self._record_partial(t, cfg, step, state, request_id)
            raise
        if message is None:
            # Stopped: leaving the with block closed the HTTP stream. What was
            # generated so far is billed, so it is recorded, then discarded.
            self._record_partial(t, cfg, step, state, request_id)
            raise _Stopped()
        self._record(t, step, costs.price_message(cfg, message, client_mod.MODEL), request_id, message.stop_reason)
        return message

    def _on_event(self, event: Any, state: _StreamState, log: EventLog, t: _Turn, step: int) -> None:
        kind = getattr(event, "type", None)
        if kind == "message_start":
            state.started = True
            state.model = event.message.model
            state.start_usage = event.message.usage
            state.output_tokens = event.message.usage.output_tokens or 0
        elif kind == "content_block_start":
            block = event.content_block
            if block.type == "fallback":
                source = getattr(getattr(block, "from_", None), "model", None) or client_mod.MODEL
                target = getattr(getattr(block, "to", None), "model", None) or "a fallback model"
                category = getattr(getattr(block, "trigger", None), "category", None)
                about = f" ({category})" if category else ""
                log.append("notice", True, level="info", code="fallback", from_model=source, to_model=target,
                           message=f"{source} declined this request{about}; {target} answered instead.")
            elif block.type == "thinking":
                log.append("status", False, text="Thinking…")
            elif block.type == "text":
                log.append("status", False, text="Writing…")
            elif block.type == "tool_use":
                log.append("status", False, text=f"Preparing {block.name}…")
        elif kind == "content_block_delta":
            delta = event.delta
            key = f"{t.turn}:{step}:{event.index}"
            if delta.type == "text_delta":
                state.streamed_chars += len(delta.text or "")
                log.append("text_delta", False, block_key=key, text=delta.text)
            elif delta.type == "thinking_delta":
                state.streamed_chars += len(delta.thinking or "")
                log.append("thinking_delta", False, block_key=key, text=delta.thinking)
            elif delta.type == "input_json_delta":
                state.streamed_chars += len(getattr(delta, "partial_json", "") or "")
            elif delta.type == "signature_delta":
                state.streamed_chars += len(getattr(delta, "signature", "") or "")
        elif kind == "message_delta":
            if event.usage is not None and event.usage.output_tokens is not None:
                state.output_tokens = event.usage.output_tokens

    def _record(self, t: _Turn, step: int, priced: dict[str, Any], request_id: str | None,
                stop_reason: str | None) -> None:
        line = {"at": costs.now_iso(), "conversation_id": t.cid, "turn": t.turn, "step": step,
                "request_id": request_id, "requested_model": client_mod.MODEL,
                "served_model": priced["served_model"], "fallback_served": priced["fallback_served"],
                "stop_reason": stop_reason, "partial": priced["partial"],
                "iterations": priced["iterations"], "usd": priced["usd"]}
        self._spend.record(line)
        t.usd += priced["usd"]
        t.requests += 1
        meta = self._store.meta(t.cid)
        context = costs.context_tokens(priced) or int(meta.get("last_context_tokens") or 0)
        meta = self._store.update_meta(t.cid, requests=int(meta.get("requests") or 0) + 1,
                                       spend_usd=round(float(meta.get("spend_usd") or 0.0) + priced["usd"], 6),
                                       last_context_tokens=context)
        self._log_for(t.cid).append(
            "usage", True, request_usd=round(priced["usd"], 6), turn_usd=round(t.usd, 6),
            conversation_usd=meta["spend_usd"], today_usd=round(self._spend.today_usd(), 6),
            served_model=priced["served_model"], fallback_served=priced["fallback_served"],
            partial=priced["partial"])

    def _record_partial(self, t: _Turn, cfg: Config, step: int, state: _StreamState, request_id: str | None) -> None:
        if not state.started:
            return  # nothing came back, so nothing was generated
        # A floor: summarized thinking streams fewer characters than were billed.
        output = max(state.output_tokens, -(-state.streamed_chars // CHARS_PER_TOKEN))
        priced = costs.price_partial(cfg, state.start_usage, output, state.model, client_mod.MODEL)
        self._record(t, step, priced, request_id, None)

    # Tools ---------------------------------------------------------------------

    def _execute_tools(self, ctx: ToolContext, t: _Turn, log: EventLog, tool_uses: list[dict[str, Any]]
                       ) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], ToolOutcome]], bool]:
        """Run tool calls in order. After Stop, the rest get "not run" error
        results, so every call still has its answer."""
        results = []
        outcomes = []
        stopped = False
        for block in tool_uses:
            name = block.get("name")
            if t.stop.is_set():
                stopped = True
                outcome = ToolOutcome(False, NOT_RUN, "Not run: stopped")
            elif name not in self._tool_names:
                outcome = ToolOutcome(False, json.dumps({"error": "unknown_tool",
                                                         "detail": f"There is no tool named {name!r}."}),
                                      f"Unknown tool {name}")
            else:
                log.append("status", False, text=f"Calling {name}…")
                try:
                    outcome = self._execute_tool(ctx, name, block.get("input"))
                except Exception as exc:
                    outcome = ToolOutcome(False, json.dumps({"error": type(exc).__name__,
                                                             "detail": redact(str(exc))[:500]}), f"{name} failed")
                if not isinstance(outcome, ToolOutcome):
                    outcome = ToolOutcome(False, json.dumps({"error": "internal", "detail": "no result"}),
                                          f"{name} failed")
            text = outcome.content if isinstance(outcome.content, str) else json.dumps(outcome.content)
            result: dict[str, Any] = {"type": "tool_result", "tool_use_id": block.get("id"),
                                      "content": redact(text) or "{}"}
            if not outcome.ok:
                result["is_error"] = True
            results.append(result)
            outcomes.append((block, outcome))
        return results, outcomes, stopped

    def _emit_blocks(self, log: EventLog, t: _Turn, step: int, raw: list[dict[str, Any]], kept: list[int]) -> None:
        for i in kept:
            block = raw[i]
            key = f"{t.turn}:{step}:{i}"
            kind = block.get("type")
            if kind == "thinking" and block.get("thinking"):
                log.append("thinking_summary", True, block_key=key, text=block["thinking"])
            elif kind == "text" and block.get("text"):
                log.append("assistant_text", True, block_key=key, text=block["text"])
            elif kind == "tool_use":
                log.append("tool_call", True, tool_use_id=block.get("id"), name=block.get("name"),
                           input=block.get("input"))

    def _emit_results(self, log: EventLog, t: _Turn, outcomes: list[tuple[dict[str, Any], ToolOutcome]]) -> None:
        drafts = []
        for block, outcome in outcomes:
            draft = outcome.draft if isinstance(outcome.draft, dict) else None
            log.append("tool_result", True, tool_use_id=block.get("id"), name=block.get("name"), ok=outcome.ok,
                       summary=outcome.summary, draft=draft)
            if draft and draft.get("design_id") and draft.get("version"):
                drafts.append({"id": draft["design_id"], "version": draft["version"]})
        if drafts:
            designs = self._store.meta(t.cid).get("designs") or []
            self._store.update_meta(t.cid, designs=designs + [d for d in drafts if d not in designs])

    # Server-sent events --------------------------------------------------------

    def stream(self, cid: str, after: int = 0) -> AsyncIterator[ServerSentEvent]:
        """Events of a conversation as server-sent events: a "batch" every
        100 ms when there are new events (id = the last sequence number, so
        Last-Event-ID resumes without duplicates), then after 300 s a "bye"
        comment with retry 1000, so the browser reconnects and no stream
        outlives a shutdown for long. Raises KeyError for an unknown
        conversation id when called. It takes no lock and reads nothing (it
        runs on the event loop), so a well-formed id of a conversation that
        does not exist is caught only by require(), which a route must call
        from a sync dependency (in the threadpool) to answer 404."""
        if not valid_cid(cid):
            raise KeyError(f"No conversation {cid!r}")
        return self._sse(cid, int(after))

    async def _sse(self, cid: str, after: int) -> AsyncIterator[ServerSentEvent]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + STREAM_SECONDS
        last = after
        # Building the log may read the transcript once; after that every read
        # is in memory and never waits for a transcript write, so polling on
        # the event loop is cheap.
        log = await asyncio.to_thread(self._log_for, cid)
        events = log.since(last)
        while True:
            if events:
                last = events[-1]["seq"]
                yield ServerSentEvent(event="batch", id=str(last), data=events)
            if self._closing or loop.time() >= deadline:
                break
            await asyncio.sleep(BATCH_SECONDS)
            events = log.since(last)
        yield ServerSentEvent(comment="bye", retry=RECONNECT_MS)
