"""The assistant through the real Anthropic SDK, with no network.

anthropic.Anthropic is given a DefaultHttpxClient over an httpx2.MockTransport,
so the request the engine builds goes through the SDK's own serialization and
headers, and canned server-sent-event bodies come back through the SDK's own
stream parser. This catches what a fake client cannot: a keyword the SDK
renames or drops, a beta header it does not send, an event it parses
differently, or a tool-input JSON failure it raises somewhere else.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any
from unittest import mock

import anthropic
import httpx2

from bench.assistant import client as client_mod
from bench.assistant import engine as engine_mod
from tests.fakes_anthropic import EngineCase, LookupCostIn

FAKE_KEY = "sk-test-0000000000000000abcd"
ASSISTANT_DIR = Path(__file__).resolve().parents[1] / "bench" / "assistant"


def sse(*events: dict[str, Any]) -> bytes:
    out = []
    for event in events:
        out.append(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n")
    return "".join(out).encode("utf-8")


def message_start(model: str = "claude-opus-5-5", input_tokens: int = 2000, cache_creation: int = 1500) -> dict:
    return {"type": "message_start", "message": {
        "id": "msg_wire", "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": 1, "cache_creation_input_tokens": cache_creation,
                  "cache_read_input_tokens": 0}}}


def block_start(index: int, block: dict) -> dict:
    return {"type": "content_block_start", "index": index, "content_block": block}


def delta(index: int, payload: dict) -> dict:
    return {"type": "content_block_delta", "index": index, "delta": payload}


def block_stop(index: int) -> dict:
    return {"type": "content_block_stop", "index": index}


def message_delta(stop_reason: str, output_tokens: int) -> dict:
    return {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens}}


TOOL_INPUT = {"family": "pi", "model": "claude-opus-5-5"}
SIGNATURE = "EqQBCkYIBxgCKkBwiresig"

TOOL_REPLY = sse(
    message_start(),
    {"type": "ping"},
    block_start(0, {"type": "thinking", "thinking": "", "signature": ""}),
    delta(0, {"type": "thinking_delta", "thinking": "Checking the ledger first."}),
    delta(0, {"type": "signature_delta", "signature": SIGNATURE}),
    block_stop(0),
    block_start(1, {"type": "text", "text": ""}),
    delta(1, {"type": "text_delta", "text": "I'll look "}),
    delta(1, {"type": "text_delta", "text": "that up."}),
    block_stop(1),
    block_start(2, {"type": "tool_use", "id": "toolu_wire_1", "name": "lookup_cost", "input": {}}),
    delta(2, {"type": "input_json_delta", "partial_json": '{"family": '}),
    delta(2, {"type": "input_json_delta", "partial_json": '"pi", "model"'}),
    delta(2, {"type": "input_json_delta", "partial_json": ': "claude-opus-5-5"}'}),
    block_stop(2),
    message_delta("tool_use", 120),
    {"type": "message_stop"},
)

TEXT_REPLY = sse(
    message_start(input_tokens=200, cache_creation=0),
    block_start(0, {"type": "text", "text": ""}),
    delta(0, {"type": "text_delta", "text": "Pi costs $0.42 per trial (measured, n=6)."}),
    block_stop(0),
    message_delta("end_turn", 30),
    {"type": "message_stop"},
)

MALFORMED_REPLY = sse(
    message_start(),
    block_start(0, {"type": "tool_use", "id": "toolu_bad", "name": "lookup_cost", "input": {}}),
    delta(0, {"type": "input_json_delta", "partial_json": '{"family" "pi"}'}),
    block_stop(0),
    message_delta("tool_use", 40),
    {"type": "message_stop"},
)


PARTIAL = sse(
    message_start(input_tokens=2500, cache_creation=0),
    block_start(0, {"type": "text", "text": ""}),
    delta(0, {"type": "text_delta", "text": "Partial answer before "}),
)

OVERLOADED_MID_STREAM = PARTIAL + sse({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})


class FailingBody(httpx2.SyncByteStream):
    """A streamed body that sends some bytes and then fails as the transport
    would: a read timeout or a dropped connection."""

    def __init__(self, head: bytes, error: Exception) -> None:
        self.head = head
        self.error = error

    def __iter__(self):
        yield self.head
        raise self.error


class MockApi:
    """Answers POST /v1/messages with canned SSE bodies (bytes, or a
    FailingBody), in order, and keeps what each request carried for the test
    to check afterwards (the handler runs on the turn thread, where a failed
    assertion would be lost)."""

    instances: list["MockApi"] = []

    def __init__(self, bodies: list[bytes | httpx2.SyncByteStream]) -> None:
        self.bodies = list(bodies)
        self.requests: list[dict[str, Any]] = []
        MockApi.instances.append(self)

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append({"method": request.method, "path": request.url.path, "host": request.url.host,
                              "headers": dict(request.headers), "body": json.loads(request.content)})
        if not self.bodies:
            return httpx2.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "unscripted"}})
        body = self.bodies.pop(0)
        headers = {"content-type": "text/event-stream", "request-id": "req_wire"}
        if isinstance(body, httpx2.SyncByteStream):
            return httpx2.Response(200, headers=headers, stream=body)
        return httpx2.Response(200, headers=headers, content=body)

    def client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(api_key=FAKE_KEY, max_retries=0,
                                   http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(self)))


class SdkWire(EngineCase):
    def setUp(self) -> None:
        super().setUp()
        MockApi.instances.clear()
        self.addCleanup(self.check_every_wire_request)

    def check_every_wire_request(self) -> None:
        for api in MockApi.instances:
            for seen in api.requests:
                self.check_wire_request(seen, fallbacks="fallbacks" in seen["body"])
        MockApi.instances.clear()

    def check_wire_request(self, seen: dict[str, Any], fallbacks: bool = True) -> None:
        self.assertEqual((seen["method"], seen["path"]), ("POST", "/v1/messages"))
        body = seen["body"]
        if fallbacks:
            self.assertIn(client_mod.FALLBACK_BETA, seen["headers"].get("anthropic-beta", ""))
            self.assertEqual(body["fallbacks"], "default")
        else:
            self.assertNotIn(client_mod.FALLBACK_BETA, seen["headers"].get("anthropic-beta", ""))
            self.assertNotIn("fallbacks", body)
        self.assertEqual(body["model"], "claude-opus-5-5")
        self.assertIs(body["stream"], True)
        self.assertEqual(body["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(body["output_config"], {"effort": "medium"})
        self.assertEqual(body["cache_control"], {"type": "ephemeral"})
        self.assertTrue(all(tool["eager_input_streaming"] is True for tool in body["tools"]))
        for banned in ("temperature", "top_p", "top_k", "tool_choice"):
            self.assertNotIn(banned, body)
        self.assertNotIn("budget_tokens", json.dumps(body["thinking"]))
        self.assertNotIn("betas", body)  # sent as the anthropic-beta header, not in the body
        self.assertTrue(all(m["role"] in ("user", "assistant") for m in body["messages"]))
        self.assertEqual(body["messages"][-1]["role"], "user")
        self.assertNotIn(FAKE_KEY, json.dumps(body))

    def test_tool_round_through_the_real_sdk(self) -> None:
        api = MockApi([TOOL_REPLY, TEXT_REPLY])
        self.engine = self.new_engine(client_factory=api.client)
        cid = self.conversation()
        self.turn(cid)

        self.assertEqual(self.finished(cid), "done")
        self.assertEqual(len(api.requests), 2)
        for seen in api.requests:
            self.check_wire_request(seen)
        self.assertEqual(self.tools.calls, [("lookup_cost", TOOL_INPUT)])
        LookupCostIn.model_validate(self.tools.calls[0][1])

        stored = self.messages(cid)
        self.assertEqual(stored[1], {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Checking the ledger first.", "signature": SIGNATURE},
            {"type": "text", "text": "I'll look that up."},
            {"type": "tool_use", "id": "toolu_wire_1", "name": "lookup_cost", "input": TOOL_INPUT}]})
        # The second request replays the reply exactly as stored.
        self.assertEqual(api.requests[1]["body"]["messages"], stored[:3])
        self.assertEqual(stored[2]["content"][0]["tool_use_id"], "toolu_wire_1")

        deltas = [e for e in self.events(cid) if e["type"] in ("text_delta", "thinking_delta")]
        self.assertEqual("".join(e["text"] for e in deltas if e["type"] == "thinking_delta"),
                         "Checking the ledger first.")
        self.assertIn("Pi costs $0.42 per trial (measured, n=6).",
                      "".join(e["text"] for e in deltas if e["type"] == "text_delta"))
        lines = self.usage_lines()
        self.assertEqual([line["request_id"] for line in lines], ["req_wire", "req_wire"])
        first = lines[0]["iterations"][0]
        self.assertEqual((first["input_tokens"], first["cache_creation_input_tokens"], first["output_tokens"]),
                         (2000, 1500, 120))
        self.assertAlmostEqual(lines[0]["usd"], (2000 * 4.0 + 1500 * 4.0 * 1.25 + 120 * 20.0) / 1e6)

    def test_malformed_tool_input_raises_value_error_from_the_iterator(self) -> None:
        api = MockApi([MALFORMED_REPLY])
        kwargs = client_mod.request_kwargs(self.settings, {"effort": "medium", "fallbacks": "default"},
                                           self.engine._tool_defs, [{"role": "user", "content": "hi"}])
        with api.client().beta.messages.stream(**kwargs) as stream:
            with self.assertRaises(ValueError):
                for _ in stream:
                    pass

    def test_malformed_tool_input_is_reissued_then_a_turn_error(self) -> None:
        api = MockApi([MALFORMED_REPLY] * 3)
        self.engine = self.new_engine(client_factory=api.client)
        cid = self.conversation()
        self.turn(cid)
        self.assertEqual(len(api.requests), 3)
        self.assertEqual(api.requests[0]["body"], api.requests[2]["body"])
        self.assertEqual(self.events(cid, "turn_error")[-1]["kind"], "invalid_json")
        self.assertEqual(self.tools.calls, [])
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user"])
        self.assertTrue(all(line["partial"] for line in self.usage_lines()))

    def check_failed_mid_stream(self, cid: str, kind: str) -> None:
        error = self.events(cid, "turn_error")[-1]
        self.assertEqual((error["kind"], error["retryable"]), (kind, True))
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user"])
        self.assertEqual(self.tools.calls, [])
        line = self.usage_lines()[-1]
        self.assertTrue(line["partial"])
        self.assertEqual(line["request_id"], "req_wire")
        self.assertEqual(line["iterations"][0]["input_tokens"], 2500)
        streamed = "".join(e["text"] for e in self.events(cid, "text_delta"))
        self.assertEqual(streamed, "Partial answer before ")

    def test_error_event_mid_stream_is_a_retryable_server_error(self) -> None:
        api = MockApi([OVERLOADED_MID_STREAM])
        self.engine = self.new_engine(client_factory=api.client)
        cid = self.conversation()
        self.turn(cid)
        self.check_failed_mid_stream(cid, "server")
        error = self.events(cid, "turn_error")[-1]
        self.assertIn("Overloaded", error["message"])
        self.assertNotIn("200", error["message"])
        self.assertTrue(self.store.meta(cid)["last_turn"]["retryable"])

    def test_connection_lost_mid_stream_is_retryable(self) -> None:
        cases = [
            ("read timeout", httpx2.ReadTimeout("timed out"), "timeout"),
            ("pool timeout", httpx2.PoolTimeout("timed out"), "timeout"),
            ("peer closed", httpx2.RemoteProtocolError("peer closed connection without sending complete message body"),
             "connection"),
            ("read error", httpx2.ReadError("connection reset by peer"), "connection"),
        ]
        for label, error, kind in cases:
            with self.subTest(label):
                api = MockApi([FailingBody(PARTIAL, error)])
                self.engine = self.new_engine(client_factory=api.client)
                cid = self.conversation()
                with mock.patch.object(engine_mod, "_log") as logged:
                    self.turn(cid)
                self.check_failed_mid_stream(cid, kind)
                logged.assert_not_called()  # an expected failure, not an internal one

    def test_retry_after_a_mid_stream_failure_replays_the_same_request(self) -> None:
        api = MockApi([FailingBody(PARTIAL, httpx2.ReadTimeout("timed out")), TEXT_REPLY])
        self.engine = self.new_engine(client_factory=api.client)
        cid = self.conversation()
        self.turn(cid)
        self.engine.retry_turn(cid)
        self.assertTrue(self.engine.wait(10))
        self.assertEqual(self.finished(cid), "done")
        self.assertEqual(api.requests[0]["body"], api.requests[1]["body"])
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user", "assistant"])

    def test_empty_reply_through_the_real_sdk_is_not_committed(self) -> None:
        empty = sse(message_start(input_tokens=200, cache_creation=0), message_delta("end_turn", 2),
                    {"type": "message_stop"})
        api = MockApi([empty, TEXT_REPLY])
        self.engine = self.new_engine(client_factory=api.client)
        cid = self.conversation()
        self.turn(cid)
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user"])
        self.assertEqual(self.events(cid, "notice")[-1]["code"], "empty")
        self.turn(cid, "Again?")
        self.assertEqual([m["role"] for m in api.requests[1]["body"]["messages"]], ["user", "user"])
        for seen in api.requests:
            self.check_wire_request(seen)

    def test_a_body_that_ends_before_message_delta_runs_nothing(self) -> None:
        cut = sse(
            message_start(),
            block_start(0, {"type": "tool_use", "id": "toolu_cut", "name": "lookup_cost", "input": {}}),
            delta(0, {"type": "input_json_delta", "partial_json": '{"family": "pi", "model": "claude-opus-5-5"}'}),
            block_stop(0),
        )
        api = MockApi([cut])
        self.engine = self.new_engine(client_factory=api.client)
        cid = self.conversation()
        self.turn(cid)
        error = self.events(cid, "turn_error")[-1]
        self.assertEqual((error["kind"], error["retryable"]), ("connection", True))
        self.assertEqual(self.tools.calls, [])
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user"])
        self.assertEqual(len(self.usage_lines()), 1)  # the request is still priced

    def test_fallbacks_off_sends_neither_the_parameter_nor_the_header(self) -> None:
        import dataclasses

        self.settings = dataclasses.replace(self.settings, fallbacks="off")
        api = MockApi([TEXT_REPLY])
        self.engine = self.new_engine(client_factory=api.client)
        cid = self.conversation()
        self.turn(cid)
        seen = api.requests[0]
        self.assertNotIn("fallbacks", seen["body"])
        self.assertNotIn(client_mod.FALLBACK_BETA, seen["headers"].get("anthropic-beta", ""))


class MakeClient(EngineCase):
    def test_make_client_uses_the_key_the_pinned_endpoint_and_bounded_retries(self) -> None:
        with mock.patch.dict("os.environ", {"ANTHROPIC_BASE_URL": "http://gateway.invalid"}):
            api = client_mod.make_client()
        try:
            self.assertEqual(api.api_key, "sk-test-1111111111111111abcd")
            self.assertEqual(str(api.base_url).rstrip("/"), "https://api.anthropic.com")
            self.assertEqual(api.max_retries, 2)
            self.assertEqual(api.timeout, 300.0)
        finally:
            api.close()

    def test_make_client_without_a_key_is_unavailable(self) -> None:
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            self.assertFalse(client_mod.key_available())
            with self.assertRaises(client_mod.AssistantUnavailable):
                client_mod.make_client()


class SecretBoundaries(EngineCase):
    """Only client.py reads the key and only redact.py lists secret values."""

    def names_used(self, path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.alias):
                names.add(node.name.split(".")[-1])
        return names

    def test_key_and_secret_value_access_stay_in_their_modules(self) -> None:
        files = sorted(ASSISTANT_DIR.glob("*.py"))
        self.assertTrue(any(f.name == "client.py" for f in files))
        for path in files:
            names = self.names_used(path)
            with self.subTest(path.name):
                if path.name != "client.py":
                    self.assertNotIn("get_secret", names)
                if path.name != "redact.py":
                    self.assertNotIn("secret_values", names)
                for forbidden in ("set_secret", "clear_secret", "start_spec", "add_listener"):
                    self.assertNotIn(forbidden, names)


if __name__ == "__main__":
    import unittest

    unittest.main()
