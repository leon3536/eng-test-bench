"""Scripted stand-ins for the Anthropic SDK, for the assistant's tests.

FakeAnthropic.beta.messages.stream(**kwargs) records a deep copy of the
kwargs and plays the next scripted Reply as the SDK would:
- a typed SDK error raised when the request is made (on entering the
  stream), built from httpx2 objects like the SDK's own;
- otherwise raw stream events built with the SDK's beta event models
  (model_validate), optionally a ValueError part-way through (the SDK's
  "Unable to parse tool parameter JSON"), then get_final_message() as a
  BetaMessage built from the same blocks.

It also holds a tiny tool registry (two tools with pydantic validation and
eager_input_streaming) standing in for bench/assistant/tools.py, so the
engine can be tested without the bench's real tools, and EngineCase, a test
base that wires an AssistantEngine to both over a temp ASSISTANT_DIR and
checks every request against the one shape the assistant may send.

Nothing here touches the network.
"""

from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
from anthropic.types.beta import (
    BetaMessage,
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawContentBlockStopEvent,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStartEvent,
    BetaRawMessageStopEvent,
)
from pydantic import BaseModel, ConfigDict, ValidationError

from bench import fsutil
from bench.assistant import client as client_mod
from bench.assistant import prompt
from bench.assistant.conversation import ConversationStore
from bench.assistant.costs import SpendBook
from bench.assistant.engine import AssistantEngine
from bench.assistant.types import ToolContext, ToolOutcome
from bench.settings import ASSISTANT_EFFORTS, AssistantSettings
from tests.support import MINIMAL_YAML, BenchTestCase

MODEL = "claude-opus-5-5"
API_URL = "https://api.anthropic.com/v1/messages"
_ids = itertools.count(1)


# Content blocks --------------------------------------------------------------

def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def thinking(value: str, signature: str = "EqQBCkYIBxgCKkDsig") -> dict[str, Any]:
    return {"type": "thinking", "thinking": value, "signature": signature}


def tool_use(tool_id: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


def fallback(source: str = MODEL, target: str = "claude-opus-5", category: str | None = "cyber") -> dict[str, Any]:
    return {"type": "fallback", "from": {"model": source}, "to": {"model": target},
            "trigger": {"type": "refusal", "category": category}}


def _chunks(value: str, size: int = 7) -> list[str]:
    return [value[i:i + size] for i in range(0, len(value), size)] or [""]


# Replies ---------------------------------------------------------------------

@dataclass
class Reply:
    """One scripted response to one request."""
    blocks: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = "end_turn"
    model: str = MODEL
    input_tokens: int = 1000
    cache_creation: int = 0
    cache_read: int = 0
    output_tokens: int = 50
    iterations: list[dict[str, Any]] | None = None
    stop_details: dict[str, Any] | None = None
    error: BaseException | None = None              # raised when the request is made
    fail_after: int | None = None                   # ValueError after this many events
    fail_error: BaseException | None = None         # raised there instead of the ValueError
    on_event: Callable[[int], None] | None = None   # called before event i is yielded

    @classmethod
    def raising(cls, error: BaseException) -> "Reply":
        return cls(error=error)

    def usage(self, final: bool) -> dict[str, Any]:
        usage = {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens if final else 1,
                 "cache_creation_input_tokens": self.cache_creation, "cache_read_input_tokens": self.cache_read}
        if final and self.iterations is not None:
            usage["iterations"] = copy.deepcopy(self.iterations)
        return usage

    def events(self) -> list[Any]:
        message = {"id": f"msg_{next(_ids):04d}", "type": "message", "role": "assistant", "model": self.model,
                   "content": [], "stop_reason": None, "stop_sequence": None, "usage": self.usage(False)}
        out: list[Any] = [BetaRawMessageStartEvent.model_validate({"type": "message_start", "message": message})]
        for index, block in enumerate(self.blocks):
            kind = block["type"]
            start: dict[str, Any]
            deltas: list[dict[str, Any]] = []
            if kind == "text":
                start = {"type": "text", "text": ""}
                deltas = [{"type": "text_delta", "text": c} for c in _chunks(block["text"])]
            elif kind == "thinking":
                start = {"type": "thinking", "thinking": "", "signature": ""}
                deltas = [{"type": "thinking_delta", "thinking": c} for c in _chunks(block["thinking"])]
                deltas.append({"type": "signature_delta", "signature": block["signature"]})
            elif kind == "tool_use":
                start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
                deltas = [{"type": "input_json_delta", "partial_json": c}
                          for c in _chunks(json.dumps(block["input"]))]
            else:
                start = copy.deepcopy(block)
            out.append(BetaRawContentBlockStartEvent.model_validate(
                {"type": "content_block_start", "index": index, "content_block": start}))
            for delta in deltas:
                out.append(BetaRawContentBlockDeltaEvent.model_validate(
                    {"type": "content_block_delta", "index": index, "delta": delta}))
            out.append(BetaRawContentBlockStopEvent.model_validate({"type": "content_block_stop", "index": index}))
        delta_usage = {"output_tokens": self.output_tokens}
        if self.iterations is not None:
            delta_usage["iterations"] = copy.deepcopy(self.iterations)
        out.append(BetaRawMessageDeltaEvent.model_validate({
            "type": "message_delta",
            "delta": {"stop_reason": self.stop_reason, "stop_sequence": None, "stop_details": self.stop_details},
            "usage": delta_usage}))
        out.append(BetaRawMessageStopEvent.model_validate({"type": "message_stop"}))
        return out

    def final(self) -> BetaMessage:
        return BetaMessage.model_validate({
            "id": f"msg_{next(_ids):04d}", "type": "message", "role": "assistant", "model": self.model,
            "content": copy.deepcopy(self.blocks), "stop_reason": self.stop_reason, "stop_sequence": None,
            "stop_details": self.stop_details, "usage": self.usage(True)})


class FakeStream:
    def __init__(self, owner: "FakeAnthropic", reply: Reply) -> None:
        self._owner = owner
        self._reply = reply
        self.closed = False
        self.request_id = f"req_fake_{len(owner.calls)}"
        self.events_yielded = 0

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True

    def __iter__(self):
        for i, event in enumerate(self._reply.events()):
            if self._reply.fail_after is not None and i >= self._reply.fail_after:
                if self._reply.fail_error is not None:
                    raise self._reply.fail_error
                raise ValueError("Unable to parse tool parameter JSON from model. Please retry your request "
                                 "or adjust your prompt. Error: expected `:` at line 1 column 9.")
            if self._reply.on_event is not None:
                self._reply.on_event(i)
            self.events_yielded += 1
            yield event

    def get_final_message(self) -> BetaMessage:
        return self._reply.final()


class _StreamManager:
    def __init__(self, owner: "FakeAnthropic", reply: Reply) -> None:
        self._owner = owner
        self._reply = reply

    def __enter__(self) -> FakeStream:
        if self._reply.error is not None:
            raise self._reply.error
        stream = FakeStream(self._owner, self._reply)
        self._owner.streams.append(stream)
        return stream

    def __exit__(self, *exc: Any) -> None:
        if self._owner.streams:
            self._owner.streams[-1].close()


class _Messages:
    def __init__(self, owner: "FakeAnthropic") -> None:
        self._owner = owner

    def stream(self, **kwargs: Any) -> _StreamManager:
        self._owner.calls.append(copy.deepcopy(kwargs))
        if not self._owner.script:
            raise AssertionError("The engine made a request the test did not script")
        return _StreamManager(self._owner, self._owner.script.pop(0))


class FakeAnthropic:
    """Stands in for anthropic.Anthropic: only beta.messages.stream exists."""

    def __init__(self, script: list[Reply] | None = None) -> None:
        self.script: list[Reply] = list(script or [])
        self.calls: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []
        self.beta = SimpleNamespace(messages=_Messages(self))


# SDK errors ------------------------------------------------------------------

_STATUS_CLASSES: dict[int, type[anthropic.APIStatusError]] = {
    400: anthropic.BadRequestError,
    401: anthropic.AuthenticationError,
    403: anthropic.PermissionDeniedError,
    404: anthropic.NotFoundError,
    429: anthropic.RateLimitError,
    500: anthropic.InternalServerError,
}
_ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
                404: "not_found_error", 429: "rate_limit_error", 500: "api_error"}


def _request() -> httpx2.Request:
    return httpx2.Request("POST", API_URL)


def status_error(status: int, message: str = "Something went wrong",
                 headers: dict[str, str] | None = None) -> anthropic.APIStatusError:
    """The exception the SDK raises for an HTTP error status."""
    body = {"type": "error", "error": {"type": _ERROR_TYPES.get(status, "api_error"), "message": message}}
    response = httpx2.Response(status, request=_request(), headers=headers or {}, json=body)
    cls = _STATUS_CLASSES.get(status, anthropic.InternalServerError if status >= 500 else anthropic.APIStatusError)
    return cls(f"Error code: {status} - {body}", response=response, body=body)


def stream_error(error_type: str, message: str) -> anthropic.APIStatusError:
    """What the SDK raises for an `event: error` after the stream started
    (anthropic/_streaming.py): the response is the 200 stream, so it is a
    plain APIStatusError with status 200 and the body's error type."""
    body = {"type": "error", "error": {"type": error_type, "message": message}}
    response = httpx2.Response(200, request=_request(), headers={"content-type": "text/event-stream"})
    return anthropic.APIStatusError(f"{body}", response=response, body=body)


def timeout_error() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=_request())


def connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=_request())


# A tiny tool registry --------------------------------------------------------

class LookupCostIn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    family: str
    model: str


class SaveNoteIn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: str
    text: str


TOOL_DEFS: list[dict[str, Any]] = [
    {"name": "lookup_cost", "description": "Look up a harness family's cost per trial on a model.",
     "eager_input_streaming": True,
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["family", "model"],
                      "properties": {"family": {"type": "string"}, "model": {"type": "string"}}}},
    {"name": "save_note", "description": "Save a draft note for the person to review.",
     "eager_input_streaming": True,
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["title", "text"],
                      "properties": {"title": {"type": "string"}, "text": {"type": "string"}}}},
]
INPUT_MODELS: dict[str, type[BaseModel]] = {"lookup_cost": LookupCostIn, "save_note": SaveNoteIn}


class FakeTools:
    """execute(ctx, name, raw_input) like tools.execute: validates the input
    with pydantic first and never calls a handler on invalid input."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.contexts: list[ToolContext] = []
        self.hooks: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self.results: dict[str, str] = {}

    def execute(self, ctx: ToolContext, name: str, raw_input: Any) -> ToolOutcome:
        model = INPUT_MODELS.get(name)
        if model is None:
            return ToolOutcome(False, json.dumps({"error": "unknown_tool", "detail": name}), f"Unknown tool {name}")
        try:
            args = model.model_validate(raw_input).model_dump()
        except ValidationError as exc:
            errors = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
            return ToolOutcome(False, json.dumps({"INVALID_JSON": json.dumps(raw_input), "errors": errors}),
                               f"Invalid input for {name}")
        self.calls.append((name, args))
        self.contexts.append(ctx)
        hook = self.hooks.get(name)
        if hook is not None:
            hook(args)
        if name == "save_note":
            draft = {"design_id": "d-note-ab12", "version": len(self.calls), "spec_hash": "sha256:" + "0" * 64,
                     "title": args["title"], "expected_usd": 0.4, "p90_usd": 0.7, "problems": 0}
            return ToolOutcome(True, json.dumps({"saved": True, **draft}), f"Draft saved: {args['title']}", draft)
        content = self.results.get(name) or json.dumps({"family": args["family"], "model": args["model"],
                                                        "usd_per_trial": 0.42, "basis": "measured n=6"})
        return ToolOutcome(True, content, f"Cost of {args['family']} on {args['model']}")


def tool_context(cfg: Any, cid: str) -> ToolContext:
    """A ToolContext with empty snapshots, for the fake tools."""
    return ToolContext(cfg=cfg, readiness=lambda: [], key_flags=lambda: {}, runs_snapshot=lambda: [],
                       active_run=lambda: None, queue_snapshot=lambda: {}, ledger=None,  # type: ignore[arg-type]
                       drafts=None, conversation_id=cid, spend=lambda: {}, saves_left=[0])  # type: ignore[arg-type]


# A test base with an engine wired to the fakes -------------------------------

class EngineCase(BenchTestCase):
    """An AssistantEngine over a temp ASSISTANT_DIR, FakeAnthropic and
    FakeTools. Every request the fake receives can be checked against
    the one request shape the assistant may send."""

    def setUp(self) -> None:
        super().setUp()
        self.data = self.isolate_data_dir()
        self.cfg = self.load_test_config(MINIMAL_YAML)
        self.settings = AssistantSettings()
        self.fake = FakeAnthropic()
        self.tools = FakeTools()
        self.engine = self.new_engine()
        # Every request any test makes is checked against the invariant part
        # of the request shape, not only in tests that call assert_requests.
        self.addCleanup(self.assert_request_invariants)

    def new_engine(self, client_factory: Callable[[], Any] | None = None) -> AssistantEngine:
        self.store = ConversationStore()
        self.book = SpendBook()
        engine = AssistantEngine(self.store, self.book, lambda cid: tool_context(self.cfg, cid),
                                 lambda: self.settings, lambda: self.cfg, TOOL_DEFS, self.tools.execute,
                                 client_factory or (lambda: self.fake))
        self.addCleanup(engine.shutdown, 5)
        return engine

    def script(self, *replies: Reply) -> None:
        self.fake.script.extend(replies)

    def conversation(self, **kwargs: Any) -> str:
        return self.engine.create_conversation(**kwargs)["id"]

    def turn(self, cid: str, text: str = "What does pi cost on Opus 5.5?",
             attach: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.engine.start_turn(cid, text, attach)
        # wait() returns as soon as the turn ends; the cap only matters on a loaded machine.
        self.assertTrue(self.engine.wait(60), "the turn did not finish")
        return result

    def events(self, cid: str, kind: str | None = None) -> list[dict[str, Any]]:
        events = self.engine.events(cid, 0)
        return [e for e in events if kind is None or e["type"] == kind]

    def finished(self, cid: str) -> str:
        done = self.events(cid, "turn_finished")
        self.assertTrue(done, "no turn_finished event")
        return done[-1]["status"]

    def messages(self, cid: str) -> list[dict[str, Any]]:
        return fsutil.read_jsonl(self.store.root / cid / "messages.jsonl")

    def usage_lines(self) -> list[dict[str, Any]]:
        return fsutil.read_jsonl(self.book.path)

    def assert_request(self, kw: dict[str, Any], effort: str = "medium", fallbacks: bool = True) -> None:
        expected = {"model", "max_tokens", "system", "tools", "messages", "thinking", "output_config",
                    "cache_control"}
        if fallbacks:
            expected |= {"fallbacks", "betas"}
        self.assertEqual(set(kw), expected)
        self.assertEqual(kw["model"], "claude-opus-5-5")
        self.assertEqual(kw["max_tokens"], self.settings.max_tokens)
        self.assertEqual(kw["system"], prompt.SYSTEM_PROMPT)
        self.assertEqual(kw["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(kw["output_config"], {"effort": effort})
        self.assertEqual(kw["cache_control"], {"type": "ephemeral"})
        if fallbacks:
            self.assertEqual(kw["fallbacks"], "default")
            self.assertEqual(kw["betas"], [client_mod.FALLBACK_BETA])
        dumped = json.dumps(kw)
        for banned in ("temperature", "top_p", "top_k", "budget_tokens", "tool_choice"):
            self.assertNotIn(banned, dumped)
        self.assertTrue(kw["tools"])
        for tool in kw["tools"]:
            self.assertIs(tool["eager_input_streaming"], True)
        self.assertTrue(kw["messages"])
        for message in kw["messages"]:
            self.assertIn(message["role"], ("user", "assistant"))
        self.assertEqual(kw["messages"][0]["role"], "user")
        self.assertEqual(kw["messages"][-1]["role"], "user", "a request must never end in an assistant prefill")

    def assert_request_invariants(self) -> None:
        allowed = {"model", "max_tokens", "system", "tools", "messages", "thinking", "output_config",
                   "cache_control", "fallbacks", "betas"}
        for kw in self.fake.calls:
            self.assertLessEqual(set(kw), allowed)
            self.assertEqual(kw["model"], "claude-opus-5-5")
            self.assertEqual(kw["max_tokens"], self.settings.max_tokens)
            self.assertEqual(kw["system"], prompt.SYSTEM_PROMPT)
            self.assertEqual(kw["thinking"], {"type": "adaptive", "display": "summarized"})
            self.assertEqual(kw["cache_control"], {"type": "ephemeral"})
            self.assertIn(kw["output_config"]["effort"], ASSISTANT_EFFORTS)
            self.assertEqual("fallbacks" in kw, "betas" in kw, "fallbacks and their beta go together")
            dumped = json.dumps(kw)
            for banned in ("temperature", "top_p", "top_k", "budget_tokens", "tool_choice"):
                self.assertNotIn(banned, dumped)
            self.assertTrue(all(tool["eager_input_streaming"] is True for tool in kw["tools"]))
            self.assertTrue(all(m["role"] in ("user", "assistant") for m in kw["messages"]))
            self.assertEqual(kw["messages"][-1]["role"], "user")

    def assert_requests(self, effort: str = "medium", fallbacks: bool = True, reissued: int = 0) -> None:
        """Check every recorded request, and that within one conversation
        each request only appends to the one before (`reissued` requests
        may repeat the previous one exactly)."""
        calls = self.fake.calls
        for kw in calls:
            self.assert_request(kw, effort, fallbacks)
        repeats = 0
        for before, after in zip(calls, calls[1:]):
            self.assertEqual(json.dumps(after["tools"]), json.dumps(before["tools"]))
            self.assertEqual(after["system"], before["system"])
            prefix = after["messages"][:len(before["messages"])]
            self.assertEqual(prefix, before["messages"], "history was edited between requests")
            if len(after["messages"]) == len(before["messages"]):
                repeats += 1
        self.assertEqual(repeats, reissued)
