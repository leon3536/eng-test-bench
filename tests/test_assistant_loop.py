"""The assistant's turn loop against a scripted Anthropic client.

Every request the engine makes is checked against the one request shape the
assistant may send (EngineCase.assert_requests). The scenarios cover what
the loop must never do: run a tool from a refused or cut-off reply, edit the
history, send a request past a spend cap, or keep streaming after Stop.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import threading
from pathlib import Path
from typing import Any
from unittest import mock

from bench import fsutil
from bench.assistant import client as client_mod
from bench.assistant import conversation as conv_mod
from bench.assistant import costs
from bench.assistant import engine as engine_mod
from tests.fakes_anthropic import (
    EngineCase,
    Reply,
    connection_error,
    fallback,
    status_error,
    stream_error,
    text,
    thinking,
    timeout_error,
    tool_use,
)

LOOKUP = {"family": "pi", "model": "claude-opus-5-5"}


class TextAndToolTurns(EngineCase):
    def test_text_only_reply_is_committed_verbatim(self) -> None:
        self.script(Reply([thinking("Weighing the ledger.", "sig-123"), text("Pi costs about a third.")]))
        cid = self.conversation()
        result = self.turn(cid)

        self.assertEqual(self.finished(cid), "done")
        self.assert_requests()
        messages = self.messages(cid)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertEqual(messages[0]["content"], [{"type": "text", "text": "What does pi cost on Opus 5.5?"}])
        self.assertEqual(messages[1]["content"], [
            {"type": "thinking", "thinking": "Weighing the ledger.", "signature": "sig-123"},
            {"type": "text", "text": "Pi costs about a third."}])
        types = [e["type"] for e in self.events(cid)]
        for kind in ("user_message", "turn_started", "thinking_delta", "text_delta", "thinking_summary",
                     "assistant_text", "usage", "turn_finished"):
            self.assertIn(kind, types)
        self.assertEqual(result["accepted_seq"], self.events(cid, "user_message")[0]["seq"])
        streamed = "".join(e["text"] for e in self.events(cid, "text_delta"))
        self.assertEqual(streamed, "Pi costs about a third.")
        summary = self.events(cid, "assistant_text")[0]
        self.assertEqual(summary["block_key"], "1:1:1")
        self.assertEqual({e["block_key"] for e in self.events(cid, "text_delta")}, {"1:1:1"})
        meta = self.store.meta(cid)
        self.assertEqual((meta["status"], meta["turns"], meta["requests"]), ("idle", 1, 1))
        self.assertGreater(meta["spend_usd"], 0)
        self.assertEqual(meta["title"], "What does pi cost on Opus 5.5?")
        lines = self.usage_lines()
        self.assertEqual(len(lines), 1)
        self.assertFalse(lines[0]["partial"])
        self.assertEqual(lines[0]["request_id"], "req_fake_1")

    def test_one_tool_round(self) -> None:
        self.script(Reply([text("Checking."), tool_use("toolu_1", "lookup_cost", LOOKUP)], stop_reason="tool_use"),
                    Reply([text("About $0.42 per trial.")]))
        cid = self.conversation()
        self.turn(cid)

        self.assertEqual(self.finished(cid), "done")
        self.assert_requests()
        self.assertEqual(self.tools.calls, [("lookup_cost", LOOKUP)])
        second = self.fake.calls[1]["messages"]
        self.assertEqual([m["role"] for m in second], ["user", "assistant", "user"])
        result = second[2]["content"][0]
        self.assertEqual((result["type"], result["tool_use_id"]), ("tool_result", "toolu_1"))
        self.assertNotIn("is_error", result)
        self.assertEqual(json.loads(result["content"])["usd_per_trial"], 0.42)
        self.assertEqual(len(self.messages(cid)), 4)
        kinds = [e["type"] for e in self.events(cid) if e["type"] in ("tool_call", "tool_result")]
        self.assertEqual(kinds, ["tool_call", "tool_result"])
        self.assertEqual(self.events(cid, "tool_call")[0]["input"], LOOKUP)
        ctx = self.tools.contexts[0]
        self.assertEqual((ctx.conversation_id, ctx.saves_left), (cid, [5]))
        spend = ctx.spend()
        self.assertGreater(spend["turn_usd"], 0)
        self.assertEqual((spend["per_turn_cap_usd"], spend["daily_cap_usd"]), (1.0, 5.0))

    def test_two_tool_uses_answered_in_one_user_message_and_one_write(self) -> None:
        other = {"family": "claude-code", "model": "claude-opus-5-5"}
        self.script(Reply([tool_use("toolu_a", "lookup_cost", LOOKUP), tool_use("toolu_b", "lookup_cost", other)],
                          stop_reason="tool_use"),
                    Reply([text("Done.")]))
        cid = self.conversation()
        commits: list[int] = []
        real_commit = self.store.commit

        def counting_commit(conversation: str, *messages: dict) -> None:
            commits.append(len(messages))
            real_commit(conversation, *messages)

        with mock.patch.object(self.store, "commit", counting_commit):
            self.turn(cid)

        self.assert_requests()
        self.assertEqual([c[1] for c in self.tools.calls], [LOOKUP, other])
        results = self.fake.calls[1]["messages"][2]
        self.assertEqual(results["role"], "user")
        self.assertEqual([b["tool_use_id"] for b in results["content"]], ["toolu_a", "toolu_b"])
        # user message, then the reply with tool calls and its results together, then the answer
        self.assertEqual(commits, [1, 2, 1])

    def test_invalid_tool_input_returns_invalid_json_and_never_runs(self) -> None:
        missing_field = {"family": "pi"}
        truncated = {}  # what the SDK's tolerant parser makes of {"family": "pi
        self.script(Reply([tool_use("toolu_1", "lookup_cost", missing_field),
                           tool_use("toolu_2", "lookup_cost", truncated)], stop_reason="tool_use"),
                    Reply([text("Let me fix that.")]))
        cid = self.conversation()
        self.turn(cid)

        self.assert_requests()
        self.assertEqual(self.tools.calls, [])
        results = self.fake.calls[1]["messages"][2]["content"]
        for result in results:
            self.assertIs(result["is_error"], True)
            self.assertIn("INVALID_JSON", json.loads(result["content"]))
        self.assertEqual([e["ok"] for e in self.events(cid, "tool_result")], [False, False])

    def test_unknown_tool_gets_an_error_result_and_never_runs(self) -> None:
        self.script(Reply([tool_use("toolu_1", "start_run", {"design_id": "d-x-0001"})], stop_reason="tool_use"),
                    Reply([text("I cannot start runs.")]))
        cid = self.conversation()
        self.turn(cid)
        self.assertEqual(self.tools.calls, [])
        result = self.fake.calls[1]["messages"][2]["content"][0]
        self.assertTrue(result["is_error"])
        self.assertEqual(json.loads(result["content"])["error"], "unknown_tool")

    def test_saved_draft_reaches_events_and_meta(self) -> None:
        self.script(Reply([tool_use("toolu_1", "save_note", {"title": "Block 1", "text": "pi vs cc"})],
                          stop_reason="tool_use"),
                    Reply([text("Saved as a draft.")]))
        cid = self.conversation()
        self.turn(cid)
        draft = self.events(cid, "tool_result")[0]["draft"]
        self.assertEqual((draft["design_id"], draft["version"]), ("d-note-ab12", 1))
        self.assertIn({"id": "d-note-ab12", "version": 1}, self.store.meta(cid)["designs"])


class RefusalsAndTruncation(EngineCase):
    def test_max_tokens_with_tool_use_appends_nothing_and_runs_nothing(self) -> None:
        self.script(Reply([text("Here is the design"), tool_use("toolu_1", "save_note", {"title": "x", "text": "y"})],
                          stop_reason="max_tokens"))
        cid = self.conversation()
        self.turn(cid)

        self.assertEqual(self.finished(cid), "truncated")
        self.assertEqual(self.tools.calls, [])
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user"])
        self.assertEqual(self.events(cid, "notice")[-1]["code"], "truncated")
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(len(self.usage_lines()), 1)  # it was still billed

    def test_refusal_appends_nothing_and_the_next_turn_is_valid(self) -> None:
        self.script(Reply([text("Partial answer"), tool_use("toolu_1", "lookup_cost", LOOKUP)],
                          stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber"}),
                    Reply([text("Here is a safer framing.")]))
        cid = self.conversation()
        self.turn(cid, "Tell me about the openssl task")
        self.assertEqual(self.finished(cid), "refused")
        self.assertEqual(self.tools.calls, [])
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user"])
        notice = self.events(cid, "notice")[-1]
        self.assertEqual((notice["code"], notice["category"]), ("refusal", "cyber"))

        self.turn(cid, "What does pi cost instead?")
        self.assertEqual(self.finished(cid), "done")
        self.assert_requests()
        self.assertEqual([m["role"] for m in self.fake.calls[1]["messages"]], ["user", "user"])

    def test_empty_reply_appends_nothing_and_the_next_turn_is_valid(self) -> None:
        for label, blocks in (("no blocks", []), ("blank text", [text("  \n")])):
            with self.subTest(label):
                self.fake.calls.clear()
                self.script(Reply(blocks, stop_reason="end_turn"), Reply([text("Here it is.")]))
                cid = self.conversation()
                self.turn(cid)
                self.assertEqual(self.finished(cid), "done")
                self.assertEqual([m["role"] for m in self.messages(cid)], ["user"])
                notice = self.events(cid, "notice")[-1]
                self.assertEqual((notice["code"], notice["level"]), ("empty", "warn"))
                self.assertIn("nothing was added", notice["message"])
                self.assertEqual(self.events(cid, "assistant_text"), [])

                self.turn(cid, "Try again please.")
                self.assertEqual(self.finished(cid), "done")
                self.assert_requests()
                self.assertEqual([m["role"] for m in self.fake.calls[1]["messages"]], ["user", "user"])
                self.assertEqual([m["role"] for m in self.messages(cid)], ["user", "user", "assistant"])

    def test_empty_reply_after_a_tool_round_keeps_the_tool_results(self) -> None:
        self.script(Reply([tool_use("toolu_1", "lookup_cost", LOOKUP)], stop_reason="tool_use"),
                    Reply([], stop_reason="end_turn"))
        cid = self.conversation()
        self.turn(cid)
        self.assertEqual(self.finished(cid), "done")
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user", "assistant", "user"])
        self.assertEqual(self.messages(cid)[-1]["content"][0]["tool_use_id"], "toolu_1")
        self.assert_requests()

    def test_text_reply_cut_off_is_kept_and_marked(self) -> None:
        self.script(Reply([text("A long answer that")], stop_reason="max_tokens"))
        cid = self.conversation()
        self.turn(cid)
        self.assertEqual(self.finished(cid), "truncated")
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user", "assistant"])

    def test_fallback_blocks_before_the_boundary_are_not_echoed_or_run(self) -> None:
        after = {"family": "claude-code", "model": "claude-opus-5-5"}
        iterations = [
            {"type": "message", "model": "claude-opus-5-5", "input_tokens": 1000, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0, "output_tokens": 200},
            {"type": "fallback_message", "model": "claude-opus-5", "input_tokens": 1200,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 300},
        ]
        self.script(Reply([thinking("declined thoughts", "sig-old"), text("Before the switch."),
                           tool_use("toolu_old", "lookup_cost", LOOKUP), fallback(), text("Answer from Opus 5."),
                           tool_use("toolu_new", "lookup_cost", after)],
                          stop_reason="tool_use", model="claude-opus-5", iterations=iterations),
                    Reply([text("Done.")], model="claude-opus-5"))
        cid = self.conversation()
        self.turn(cid)

        self.assert_requests()
        self.assertEqual(self.tools.calls, [("lookup_cost", after)])
        echoed = self.messages(cid)[1]["content"]
        self.assertEqual([b["type"] for b in echoed], ["text", "fallback", "text", "tool_use"])
        self.assertEqual(echoed[-1]["id"], "toolu_new")
        results = self.messages(cid)[2]["content"]
        self.assertEqual([r["tool_use_id"] for r in results], ["toolu_new"])
        notice = [e for e in self.events(cid, "notice") if e["code"] == "fallback"][0]
        self.assertEqual((notice["from_model"], notice["to_model"]), ("claude-opus-5-5", "claude-opus-5"))

        line = self.usage_lines()[0]
        self.assertTrue(line["fallback_served"])
        self.assertEqual(line["served_model"], "claude-opus-5")
        self.assertEqual([i["model"] for i in line["iterations"]], ["claude-opus-5-5", "claude-opus-5"])
        expected = (1000 * 4.00 + 200 * 20.00) / 1e6 + (1200 * 5.00 + 300 * 25.00) / 1e6
        self.assertAlmostEqual(line["usd"], expected, places=9)
        self.assertEqual({i["priced"] for i in line["iterations"]}, {"exact"})

    def test_echo_filter_keeps_paired_server_tools_and_drops_unpaired(self) -> None:
        content = [
            {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {}},
            {"type": "web_search_tool_result", "tool_use_id": "srv_1", "content": []},
            {"type": "server_tool_use", "id": "srv_2", "name": "web_search", "input": {}},
            {"type": "redacted_thinking", "data": "xyz"},
            {"type": "text", "text": "partial"},
            fallback(),
            {"type": "thinking", "thinking": "kept", "signature": "s"},
        ]
        kept = engine_mod.echo_filter(content)
        self.assertEqual([b.get("id") or b["type"] for b in kept],
                         ["srv_1", "web_search_tool_result", "text", "fallback", "thinking"])
        no_fallback = [text("a"), {"type": "thinking", "thinking": "b", "signature": "s"}]
        self.assertEqual(engine_mod.echo_filter(no_fallback), no_fallback)


class JsonRetries(EngineCase):
    def test_value_error_twice_then_success(self) -> None:
        self.script(Reply([tool_use("toolu_1", "lookup_cost", LOOKUP)], fail_after=3),
                    Reply([tool_use("toolu_1", "lookup_cost", LOOKUP)], fail_after=3),
                    Reply([text("Recovered.")]))
        cid = self.conversation()
        self.turn(cid)

        self.assertEqual(self.finished(cid), "done")
        self.assert_requests(reissued=2)
        self.assertEqual(len(self.fake.calls), 3)
        self.assertEqual(self.fake.calls[0], self.fake.calls[2])
        self.assertEqual(len([e for e in self.events(cid, "notice") if e["code"] == "retry"]), 2)
        self.assertEqual([line["partial"] for line in self.usage_lines()], [True, True, False])

    def test_value_error_three_times_is_a_turn_error_then_retry_works(self) -> None:
        self.script(*[Reply([tool_use("toolu_1", "lookup_cost", LOOKUP)], fail_after=3) for _ in range(3)])
        cid = self.conversation()
        self.turn(cid)

        self.assertEqual(len(self.fake.calls), 3)
        self.assertEqual(self.finished(cid), "error")
        error = self.events(cid, "turn_error")[-1]
        self.assertEqual((error["kind"], error["retryable"]), ("invalid_json", True))
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user"])
        self.assertEqual(self.tools.calls, [])

        self.script(Reply([text("Second time lucky.")]))
        self.engine.retry_turn(cid)
        self.assertTrue(self.engine.wait(10))
        self.assertEqual(self.finished(cid), "done")
        self.assertEqual(self.fake.calls[3]["messages"], self.fake.calls[0]["messages"])
        self.assert_requests(reissued=3)

    def test_retry_is_refused_after_a_good_turn(self) -> None:
        self.script(Reply([text("Fine.")]))
        cid = self.conversation()
        self.turn(cid)
        with self.assertRaises(RuntimeError):
            self.engine.retry_turn(cid)


class TypedErrors(EngineCase):
    CASES = [
        ("auth", status_error(401, "invalid x-api-key"), "auth", False, None),
        ("permission", status_error(403, "Access denied"), "permission", False, None),
        ("not_found", status_error(404, "model: claude-opus-5-5"), "model_not_found", False, None),
        ("bad_request", status_error(400, "max_tokens: too large"), "bad_request", False, None),
        ("replay", status_error(400, "messages.1.content.0: Invalid `signature` in `thinking` block. The block "
                                     "is bound to a different conversation."), "bad_request", False, "locked"),
        ("rate_limited", status_error(429, "slow down", headers={"retry-after": "7"}), "rate_limited", True, None),
        ("server", status_error(500, "boom"), "server", True, None),
        ("overloaded", status_error(529, "overloaded"), "server", True, None),
        ("timeout", timeout_error(), "timeout", True, None),
        ("connection", connection_error(), "connection", True, None),
    ]

    def test_each_error_maps_to_a_turn_error_and_edits_nothing(self) -> None:
        for label, error, kind, retryable, status in self.CASES:
            with self.subTest(label):
                self.script(Reply.raising(error))
                cid = self.conversation()
                self.turn(cid)
                event = self.events(cid, "turn_error")[-1]
                self.assertEqual((event["kind"], event["retryable"]), (kind, retryable))
                self.assertEqual([m["role"] for m in self.messages(cid)], ["user"])
                meta = self.store.meta(cid)
                self.assertEqual(meta["status"], status or "idle")
                self.assertEqual(meta["last_turn"]["status"], "error")
                if label == "rate_limited":
                    self.assertEqual(event["retry_after_s"], 7.0)
                if label == "auth":
                    self.assertIn("Replace it on Setup", event["message"])
                if status == "locked":
                    self.assertEqual(self.engine.conversation(cid)["conversation"]["status"], "locked")
                    with self.assertRaises(RuntimeError):
                        self.engine.start_turn(cid, "again?", None)
        self.assertEqual(self.usage_lines(), [])  # no reply came back, so nothing was billed

    def test_error_event_mid_stream_records_what_was_generated(self) -> None:
        self.script(Reply([text("Partial answer before the error.")], fail_after=4, input_tokens=3000,
                          fail_error=stream_error("overloaded_error", "Overloaded")))
        cid = self.conversation()
        self.turn(cid)
        error = self.events(cid, "turn_error")[-1]
        self.assertEqual((error["kind"], error["retryable"]), ("server", True))
        self.assertIn("Overloaded", error["message"])
        self.assertNotIn("200", error["message"])
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user"])
        line = self.usage_lines()[0]
        self.assertTrue(line["partial"])
        self.assertEqual(line["iterations"][0]["input_tokens"], 3000)

    MID_STREAM = [
        ("overloaded_error", "Overloaded", "server", True, None),
        ("api_error", "Internal server error", "server", True, None),
        ("timeout_error", "Request timed out", "timeout", True, None),
        ("rate_limit_error", "Rate limited", "rate_limited", True, None),
        ("invalid_request_error", "messages.1.content.0: Invalid `signature` in `thinking` block",
         "bad_request", False, "locked"),
        ("invalid_request_error", "max_tokens: too large", "bad_request", False, None),
        ("authentication_error", "invalid x-api-key", "auth", False, None),
        ("permission_error", "Access denied", "permission", False, None),
        ("not_found_error", "model: claude-opus-5-5", "model_not_found", False, None),
        ("brand_new_error", "Something new", "server", True, None),
    ]

    def test_error_events_mid_stream_are_classified_by_their_type(self) -> None:
        for error_type, message, kind, retryable, status in self.MID_STREAM:
            with self.subTest(f"{error_type}: {message[:20]}"):
                self.script(Reply([text("Some text first.")], fail_after=3, fail_error=stream_error(error_type, message)))
                cid = self.conversation()
                self.turn(cid)
                event = self.events(cid, "turn_error")[-1]
                self.assertEqual((event["kind"], event["retryable"]), (kind, retryable))
                self.assertIsNone(event["retry_after_s"])
                self.assertNotIn("(200)", event["message"])
                self.assertEqual([m["role"] for m in self.messages(cid)], ["user"])
                self.assertEqual(self.store.meta(cid)["status"], status or "idle")

    def test_key_removed_before_the_turn_runs_is_an_auth_error(self) -> None:
        def no_key() -> None:
            raise client_mod.AssistantUnavailable("Set ANTHROPIC_API_KEY on Setup to use the assistant.")

        self.engine = self.new_engine(client_factory=no_key)
        cid = self.conversation()
        self.turn(cid)
        error = self.events(cid, "turn_error")[-1]
        self.assertEqual((error["kind"], error["retryable"]), ("auth", False))
        self.assertIn("Setup", error["message"])
        self.assertEqual(self.store.meta(cid)["status"], "idle")

    def test_unexpected_exception_is_an_internal_error(self) -> None:
        self.script(Reply.raising(KeyError("boom")))
        cid = self.conversation()
        with mock.patch.object(engine_mod, "_log"):
            self.turn(cid)
        self.assertEqual(self.events(cid, "turn_error")[-1]["kind"], "internal")
        self.assertEqual(self.store.meta(cid)["status"], "idle")


class Caps(EngineCase):
    def test_per_turn_cap_stops_before_the_next_request(self) -> None:
        self.settings = dataclasses.replace(self.settings, per_turn_cap_usd=0.01)
        self.script(Reply([tool_use("toolu_1", "lookup_cost", LOOKUP)], stop_reason="tool_use", input_tokens=5000),
                    Reply([text("never sent")]))
        cid = self.conversation()
        self.turn(cid)

        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(self.finished(cid), "cap_reached")
        self.assertEqual(self.events(cid, "notice")[-1]["code"], "cap")
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user", "assistant", "user"])
        self.assertEqual(len(self.fake.script), 1)

    def test_daily_cap_with_reserve_stops_mid_turn(self) -> None:
        self.settings = dataclasses.replace(self.settings, max_tokens=1000, daily_cap_usd=0.10)
        self.script(Reply([tool_use("toolu_1", "lookup_cost", LOOKUP)], stop_reason="tool_use", input_tokens=12000),
                    Reply([text("never sent")]))
        cid = self.conversation()
        self.turn(cid)
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(self.finished(cid), "cap_reached")
        self.assertIn("daily cap", self.events(cid, "notice")[-1]["message"])

    def test_daily_cap_refuses_a_new_message_and_stores_nothing(self) -> None:
        self.book.record({"conversation_id": "c-20261029-000000-0000", "usd": 4.99})
        cid = self.conversation()
        with self.assertRaises(ValueError) as caught:
            self.engine.start_turn(cid, "hello", None)
        self.assertIn("daily cap $5.00", str(caught.exception))
        self.assertEqual(self.messages(cid), [])
        self.assertEqual(self.events(cid), [])
        self.assertEqual(self.fake.calls, [])
        # Refused by the reserve before the cap itself: status says so too.
        status = self.engine.status()
        self.assertFalse(status["available"])
        self.assertIn("daily cap $5.00 nearly reached", status["reason"])
        self.assertIn("$4.99 spent today", status["reason"])
        self.book.record({"conversation_id": "c-20261029-000000-0000", "usd": 0.02})
        status = self.engine.status()
        self.assertFalse(status["available"])
        self.assertIn("daily cap $5.00 reached", status["reason"])

    def test_status_is_available_whenever_a_new_message_would_be_accepted(self) -> None:
        cid = self.conversation()
        reserve = self.engine._reserve_usd(self.cfg, self.settings,
                                           {"fallbacks": self.settings.fallbacks, "last_context_tokens": 0}, [])
        self.assertGreater(reserve, 0.5)
        self.book.record({"conversation_id": "c-20261029-000000-0000", "usd": round(4.9 - reserve, 6)})
        status = self.engine.status()
        self.assertTrue(status["available"], status["reason"])
        self.assertIsNone(status["reason"])
        self.script(Reply([text("Still within the cap.")]))
        self.turn(cid)
        self.assertEqual(self.finished(cid), "done")

    def test_reserve_uses_the_dearest_rates_when_fallbacks_are_on(self) -> None:
        cid = self.conversation()
        meta = self.store.meta(cid)
        with_fallbacks = self.engine._reserve_usd(self.cfg, self.settings, meta, [])
        without = self.engine._reserve_usd(self.cfg, self.settings, {**meta, "fallbacks": "off"}, [])
        self.assertAlmostEqual(without * 25 / 20, with_fallbacks, delta=0.05)
        self.assertGreater(with_fallbacks, self.settings.max_tokens * 25 / 1e6)


class InvisibleReplies(EngineCase):
    def test_a_reply_with_only_a_fallback_marker_is_empty(self) -> None:
        self.script(Reply([fallback()]))
        cid = self.conversation()
        self.turn(cid)
        self.assertEqual(self.finished(cid), "done")
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user"])
        self.assertEqual(self.events(cid, "notice")[-1]["code"], "empty")

    def test_a_thinking_only_reply_is_kept_and_flagged(self) -> None:
        self.script(Reply([thinking("The whole answer ended up in here.")]))
        cid = self.conversation()
        self.turn(cid)
        self.assertEqual(self.finished(cid), "done")
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user", "assistant"])
        notice = self.events(cid, "notice")[-1]
        self.assertEqual((notice["code"], notice["level"]), ("empty", "warn"))
        self.assertIn("without a visible answer", notice["message"])
        self.assertEqual(self.events(cid, "assistant_text"), [])

    def test_a_reply_without_a_stop_reason_runs_nothing(self) -> None:
        self.script(Reply([tool_use("toolu_cut", "lookup_cost", LOOKUP)], stop_reason=None))
        cid = self.conversation()
        self.turn(cid)
        error = self.events(cid, "turn_error")[-1]
        self.assertEqual((error["kind"], error["retryable"]), ("connection", True))
        self.assertEqual(self.tools.calls, [])
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user"])


class StreamNeverWaitsOnDisk(EngineCase):
    def test_opening_a_stream_does_not_wait_for_a_transcript_write(self) -> None:
        import asyncio
        import time as _time

        cid = self.conversation()
        log = self.engine._log_for(cid)
        real = fsutil.append_jsonl
        entered, release = threading.Event(), threading.Event()

        def slow(path: Any, records: list, patience: float = 2.0) -> None:
            if Path(path).name == "transcript.jsonl":
                entered.set()
                release.wait(10)
            real(path, records, patience)

        with mock.patch.object(fsutil, "append_jsonl", slow):
            writer = threading.Thread(target=log.append, args=("notice", True),
                                      kwargs={"level": "info", "code": "x", "message": "slow"})
            writer.start()
            try:
                self.assertTrue(entered.wait(10))

                async def open_stream() -> float:
                    began = _time.monotonic()
                    self.engine.stream(cid, 0)
                    self.store.meta(cid)
                    return _time.monotonic() - began

                self.assertLess(asyncio.run(open_stream()), 0.1)
            finally:
                release.set()
                writer.join(10)


class StopAndSteps(EngineCase):
    def test_stop_mid_stream_records_partial_cost_and_discards_the_reply(self) -> None:
        holder: dict[str, str] = {}

        def stop_at(i: int) -> None:
            if i == 4:
                self.engine.stop(holder["cid"])

        self.script(Reply([text("This answer will be cut off part-way through.")], on_event=stop_at,
                          input_tokens=2000, output_tokens=400))
        cid = self.conversation()
        holder["cid"] = cid
        self.turn(cid)

        self.assertEqual(self.finished(cid), "stopped")
        self.assertEqual([m["role"] for m in self.messages(cid)], ["user"])
        self.assertTrue(self.fake.streams[0].closed)
        self.assertEqual(self.fake.streams[0].events_yielded, 5)
        line = self.usage_lines()[0]
        self.assertTrue(line["partial"])
        self.assertEqual(line["iterations"][0]["input_tokens"], 2000)
        # message_delta never came, so output is estimated from the characters
        # streamed (3 per token, rounded up), not message_start's count of 1.
        streamed = sum(len(e["text"]) for e in self.engine._log_for(cid).since(0) if e["type"] == "text_delta")
        output = line["iterations"][0]["output_tokens"]
        self.assertGreater(streamed, 0)
        self.assertEqual(output, -(-streamed // 3))
        self.assertAlmostEqual(line["usd"], (2000 * 4.00 + output * 20.00) / 1e6)
        self.assertEqual(self.events(cid, "notice")[-1]["code"], "stopped")

    def test_stop_between_tools_answers_the_rest_as_not_run(self) -> None:
        other = {"family": "claude-code", "model": "claude-opus-5-5"}
        self.script(Reply([tool_use("toolu_a", "lookup_cost", LOOKUP), tool_use("toolu_b", "lookup_cost", other)],
                          stop_reason="tool_use"))
        cid = self.conversation()
        self.tools.hooks["lookup_cost"] = lambda args: self.engine.stop(cid)
        self.turn(cid)

        self.assertEqual(self.finished(cid), "stopped")
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(self.tools.calls, [("lookup_cost", LOOKUP)])
        messages = self.messages(cid)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        first, second = messages[2]["content"]
        self.assertNotIn("is_error", first)
        self.assertTrue(second["is_error"])
        self.assertIn("not run: stopped", second["content"])

    def test_max_steps(self) -> None:
        self.settings = dataclasses.replace(self.settings, max_steps=2)
        self.script(Reply([tool_use("toolu_1", "lookup_cost", LOOKUP)], stop_reason="tool_use"),
                    Reply([tool_use("toolu_2", "lookup_cost", LOOKUP)], stop_reason="tool_use"))
        cid = self.conversation()
        self.turn(cid)
        self.assertEqual(self.finished(cid), "max_steps")
        self.assertEqual(len(self.fake.calls), 2)
        self.assertEqual(self.events(cid, "notice")[-1]["code"], "max_steps")
        self.assertEqual(self.messages(cid)[-1]["role"], "user")
        self.assert_requests()

    def test_one_turn_at_a_time_across_conversations(self) -> None:
        release = threading.Event()
        entered = threading.Event()

        def hold(i: int) -> None:
            if i == 0:
                entered.set()
                release.wait(10)

        self.script(Reply([text("slow")], on_event=hold))
        first = self.conversation()
        second = self.conversation()
        self.engine.start_turn(first, "one", None)
        self.assertTrue(entered.wait(10))
        try:
            with self.assertRaises(RuntimeError):
                self.engine.start_turn(second, "two", None)
            self.assertEqual(self.engine.status()["active_turn"]["conversation_id"], first)
        finally:
            release.set()
        self.assertTrue(self.engine.wait(10))
        self.assertEqual(self.messages(second), [])


class RecoveryAndLocks(EngineCase):
    def test_crash_recovery_closes_a_dangling_tool_use_and_drops_a_partial_line(self) -> None:
        cid = self.conversation()
        path = self.store.root / cid / "messages.jsonl"
        lines = [
            {"role": "user", "content": [{"type": "text", "text": "price block 1"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_x", "name": "lookup_cost",
                                               "input": LOOKUP}]},
        ]
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(json.dumps(m) + "\n" for m in lines))
            f.write('{"role": "user", "content": [{"type": "tool_res')
        self.store.update_meta(cid, status="busy", next_seq=7)

        engine = self.new_engine()
        meta = self.store.meta(cid)
        self.assertEqual(meta["status"], "idle")
        self.assertGreater(meta["next_seq"], 7)
        repaired = self.messages(cid)
        self.assertEqual([m["role"] for m in repaired], ["user", "assistant", "user"])
        self.assertEqual(repaired[2]["content"][0]["tool_use_id"], "toolu_x")
        self.assertTrue(repaired[2]["content"][0]["is_error"])
        raw = path.read_text(encoding="utf-8")
        self.assertTrue(raw.endswith("\n"))
        stored = [line for line in raw.split("\n") if line]
        self.assertEqual(len(stored), 3)  # the torn fragment is gone, not just skipped
        for line in stored:
            json.loads(line)

        self.script(Reply([text("Picking up again.")]))
        engine.start_turn(cid, "where were we?", None)
        self.assertTrue(engine.wait(10))
        self.assertEqual([m["role"] for m in self.fake.calls[0]["messages"]], ["user", "assistant", "user", "user"])
        self.assert_requests()

    def test_recovery_restarts_sequence_numbers_past_all_a_crashed_turn_emitted(self) -> None:
        # More streamed deltas than the in-memory limit, none of them on disk.
        long_text = "abcdefg" * (conv_mod.EPHEMERAL_LIMIT + 500)
        self.script(Reply([text(long_text)]))
        cid = self.conversation()
        old_next = self.store.meta(cid)["next_seq"]
        engine = self.engine

        def crash(t: object, status: str, error: object) -> None:
            # The process died: meta stays busy and next_seq is never saved.
            with engine._lock:
                engine._active = None
            engine._gate.release()

        with mock.patch.object(engine, "_finish", crash):
            self.turn(cid)
        seen = engine._log_for(cid).last_seq  # the last event an open page could have received
        emitted = seen - old_next + 1
        self.assertGreater(emitted, conv_mod.EPHEMERAL_LIMIT)
        self.assertEqual(self.store.meta(cid)["status"], "busy")

        self.engine = self.new_engine()
        meta = self.store.meta(cid)
        self.assertEqual(meta["status"], "idle")
        self.assertGreater(meta["next_seq"], old_next + emitted)
        self.script(Reply([text("After the restart.")]))
        accepted = self.turn(cid, "Are you back?")["accepted_seq"]
        self.assertGreater(accepted, seen)
        # A page reconnecting with Last-Event-ID = seen gets the whole new turn.
        resumed = [e["type"] for e in self.engine.events(cid, seen)]
        self.assertEqual(resumed[:2], ["user_message", "turn_started"])
        self.assertIn("assistant_text", resumed)

    def test_changed_prompt_locks_and_continue_recaps_without_a_model_call(self) -> None:
        self.script(Reply([thinking("PRIVATE-REASONING", "sig"), text("Pi is cheaper.")]))
        cid = self.conversation(title="Pi costs")
        self.turn(cid, "Is pi cheaper than Claude Code?")
        calls = len(self.fake.calls)

        with mock.patch.object(engine_mod, "PROMPT_HASH", "sha256:" + "f" * 64):
            view = self.engine.conversation(cid)["conversation"]
            self.assertEqual(view["status"], "locked")
            self.assertIn("instructions were updated", view["locked_reason"])
            with self.assertRaises(RuntimeError):
                self.engine.start_turn(cid, "and now?", None)
            new = self.engine.continue_conversation(cid)

        self.assertEqual(len(self.fake.calls), calls)
        self.assertEqual(new["continued_from"], cid)
        seeded = self.messages(new["id"])
        self.assertEqual(len(seeded), 1)
        recap = seeded[0]["content"][0]["text"]
        self.assertIn("Is pi cheaper than Claude Code?", recap)
        self.assertIn("Pi is cheaper.", recap)
        self.assertNotIn("PRIVATE-REASONING", recap)
        self.assertTrue(self.events(new["id"], "user_message")[0]["recap"])

    def test_changed_tools_lock_old_conversations(self) -> None:
        cid = self.conversation()
        changed = copy.deepcopy(self.engine._tool_defs)
        changed[0]["description"] += " Now with more detail."
        engine = engine_mod.AssistantEngine(self.store, self.book, lambda c: None, lambda: self.settings,
                                            lambda: self.cfg, changed, self.tools.execute, lambda: self.fake)
        self.assertNotEqual(engine.tools_hash, self.engine.tools_hash)
        with self.assertRaises(RuntimeError) as caught:
            engine.start_turn(cid, "hi", None)
        self.assertIn("tools were updated", str(caught.exception))

    def test_tools_without_eager_input_streaming_are_rejected(self) -> None:
        bad = [{"name": "x", "description": "x", "input_schema": {"type": "object"}}]
        with self.assertRaises(ValueError):
            engine_mod.AssistantEngine(self.store, self.book, lambda c: None, lambda: self.settings,
                                       lambda: self.cfg, bad, self.tools.execute, lambda: self.fake)

    def test_effort_and_fallbacks_are_fixed_per_conversation(self) -> None:
        cid = self.conversation(effort="high")
        self.settings = dataclasses.replace(self.settings, effort="low", fallbacks="off")
        later = self.conversation()
        self.script(Reply([text("high effort")]), Reply([text("low effort")]))
        self.turn(cid)
        self.turn(later)
        self.assert_request(self.fake.calls[0], effort="high", fallbacks=True)
        self.assert_request(self.fake.calls[1], effort="low", fallbacks=False)
        with self.assertRaises(ValueError):
            self.engine.create_conversation(effort="extreme")

    def test_key_like_text_is_refused_and_nothing_is_stored(self) -> None:
        cid = self.conversation()
        for message in ("my key is sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA", "use sk-test-1111111111111111abcd"):
            with self.subTest(message[:12]):
                with self.assertRaises(ValueError) as caught:
                    self.engine.start_turn(cid, message, None)
                self.assertIn("Set keys on Setup", str(caught.exception))
        self.assertEqual(self.messages(cid), [])
        self.assertEqual(self.events(cid), [])
        self.assertEqual(self.fake.calls, [])

    def test_no_key_or_disabled_refuses_with_runtime_error(self) -> None:
        cid = self.conversation()
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            with self.assertRaises(RuntimeError):
                self.engine.start_turn(cid, "hello", None)
            self.assertFalse(self.engine.status()["key_set"])
        self.settings = dataclasses.replace(self.settings, enabled=False)
        with self.assertRaises(RuntimeError):
            self.engine.start_turn(cid, "hello", None)
        self.assertIn("turned off", self.engine.status()["reason"])
        self.assertEqual(self.fake.calls, [])

    def test_attached_design_adds_a_second_text_block(self) -> None:
        self.script(Reply([text("Looking at it.")]))
        cid = self.conversation()
        self.turn(cid, "Is this design sound?", {"id": "d-block-1-ab12", "version": 3})
        content = self.fake.calls[0]["messages"][0]["content"]
        self.assertEqual(content[1]["text"], "The user has design d-block-1-ab12 version 3 open in the dashboard.")
        self.assertIn({"id": "d-block-1-ab12", "version": 3}, self.store.meta(cid)["designs"])
        with self.assertRaises(ValueError):
            self.engine.start_turn(cid, "again", {"id": "../../config", "version": 1})


class ServerSentEvents(EngineCase):
    def collect(self, cid: str, after: int) -> list:
        async def run() -> list:
            return [item async for item in self.engine.stream(cid, after)]
        return asyncio.run(run())

    def test_batches_resume_without_duplicates_and_end_with_bye(self) -> None:
        self.script(Reply([text("Streaming works.")]))
        cid = self.conversation()
        self.turn(cid)
        with mock.patch.object(engine_mod, "STREAM_SECONDS", 0.25):
            items = self.collect(cid, 0)
        batches = [i for i in items if i.event == "batch"]
        self.assertTrue(batches)
        seqs = [e["seq"] for b in batches for e in b.data]
        self.assertEqual(seqs, sorted(set(seqs)))
        self.assertEqual(batches[-1].id, str(seqs[-1]))
        self.assertEqual((items[-1].comment, items[-1].retry), ("bye", 1000))
        self.assertEqual(seqs, [e["seq"] for e in self.engine.events(cid, 0)])

        middle = seqs[len(seqs) // 2]
        with mock.patch.object(engine_mod, "STREAM_SECONDS", 0.15):
            resumed = self.collect(cid, middle)
        again = [e["seq"] for i in resumed if i.event == "batch" for e in i.data]
        self.assertEqual(again, [s for s in seqs if s > middle])

    def test_fastapi_encodes_the_stream_as_server_sent_events(self) -> None:
        from fastapi import FastAPI, Header
        from fastapi.sse import EventSourceResponse
        from fastapi.testclient import TestClient

        app = FastAPI()
        engine = self.engine

        @app.get("/stream/{cid}", response_class=EventSourceResponse)
        async def stream(cid: str, after: int = 0, last_event_id: str | None = Header(default=None)):
            start = int(last_event_id) if last_event_id and last_event_id.isdigit() else after
            async for item in engine.stream(cid, start):
                yield item

        self.script(Reply([text("Over the wire.")]))
        cid = self.conversation()
        self.turn(cid)
        last = self.engine.events(cid, 0)[-1]["seq"]
        with mock.patch.object(engine_mod, "STREAM_SECONDS", 0.2), TestClient(app) as http:
            body = http.get(f"/stream/{cid}").text
            resumed = http.get(f"/stream/{cid}", headers={"Last-Event-ID": str(last - 1)}).text
        self.assertIn("event: batch", body)
        self.assertIn(f"id: {last}", body)
        data = [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]
        self.assertEqual([e["seq"] for e in data[0]], [e["seq"] for e in self.engine.events(cid, 0)])
        self.assertIn(": bye", body)
        self.assertIn("retry: 1000", body)
        resumed_data = [json.loads(line[len("data: "):]) for line in resumed.splitlines() if line.startswith("data: ")]
        self.assertEqual([e["seq"] for batch in resumed_data for e in batch], [last])

    def test_unknown_conversation_is_rejected_before_streaming(self) -> None:
        # stream() runs on the event loop, so it only rejects malformed ids;
        # require() (called from a sync dependency) catches the rest.
        with self.assertRaises(KeyError):
            self.engine.stream("../../etc", 0)
        for bad in ("c-20261029-000000-ffff", "../../etc"):
            with self.subTest(bad):
                with self.assertRaises(KeyError):
                    self.engine.require(bad)
        self.engine.require(self.conversation())

    def test_a_dependency_calling_require_answers_404_before_the_stream_starts(self) -> None:
        from fastapi import Depends, FastAPI, Request
        from fastapi.responses import JSONResponse
        from fastapi.sse import EventSourceResponse
        from fastapi.testclient import TestClient

        app = FastAPI()
        engine = self.engine

        @app.exception_handler(KeyError)
        async def not_found(request: Request, exc: KeyError) -> JSONResponse:
            return JSONResponse({"detail": "Not found"}, status_code=404)

        def existing(cid: str) -> str:
            engine.require(cid)
            return cid

        @app.get("/stream/{cid}", response_class=EventSourceResponse)
        async def stream(conversation: str = Depends(existing), after: int = 0):
            async for item in engine.stream(conversation, after):
                yield item

        self.script(Reply([text("Known.")]))
        cid = self.conversation()
        self.turn(cid)
        with mock.patch.object(engine_mod, "STREAM_SECONDS", 0.15), TestClient(app) as http:
            missing = http.get("/stream/c-20261029-000000-ffff")
            known = http.get(f"/stream/{cid}")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(known.status_code, 200)
        self.assertIn("event: batch", known.text)

    def test_stream_after_recovery_reads_the_transcript_at_most_once(self) -> None:
        self.script(Reply([text("Before the restart.")]))
        cid = self.conversation()
        self.turn(cid)
        durable = [e["seq"] for e in self.store.transcript(cid)]
        self.store.update_meta(cid, status="busy")
        self.engine = self.new_engine()  # recovery jumps next_seq far past the transcript

        reads: list[int] = []
        real = self.store.transcript

        def counting(*args: object, **kwargs: object) -> list:
            reads.append(1)
            return real(*args, **kwargs)  # type: ignore[arg-type]

        with mock.patch.object(self.store, "transcript", counting), \
                mock.patch.object(engine_mod, "STREAM_SECONDS", 1.0):
            items = self.collect(cid, 0)
        self.assertLessEqual(len(reads), 1)
        seqs = [e["seq"] for i in items if i.event == "batch" for e in i.data]
        self.assertEqual(seqs, durable)

    def test_a_slow_transcript_write_never_blocks_a_reader(self) -> None:
        cid = self.conversation()
        log = self.engine._log_for(cid)
        log.append("turn_started", True)
        entered = threading.Event()
        release = threading.Event()
        real = self.store.append_transcript

        def slow(conversation: str, events: list) -> None:
            entered.set()
            release.wait(10)  # a scanner holding transcript.jsonl
            real(conversation, events)

        with mock.patch.object(self.store, "append_transcript", slow):
            writer = threading.Thread(target=log.append, args=("notice", True),
                                      kwargs={"level": "info", "code": "x", "message": "slow"})
            writer.start()
            try:
                self.assertTrue(entered.wait(10))
                done = threading.Event()
                got: list = []
                reader = threading.Thread(target=lambda: (got.extend(log.since(0)), done.set()))
                reader.start()
                self.assertTrue(done.wait(2), "since() waited for the transcript write")
                self.assertEqual([e["type"] for e in got], ["turn_started", "notice"])
                log.append("status", False, text="Thinking…")  # ephemeral appends do not wait either
            finally:
                release.set()
                writer.join(10)
        self.assertEqual([e["type"] for e in self.store.transcript(cid)], ["turn_started", "notice"])

    def test_usage_summary(self) -> None:
        self.script(Reply([text("Hi.")]))
        cid = self.conversation()
        self.turn(cid)
        usage = self.engine.usage(7)
        self.assertEqual(len(usage["days"]), 7)
        self.assertAlmostEqual(usage["total_usd"], self.usage_lines()[0]["usd"], places=6)
        self.assertEqual(usage["days"][-1]["requests"], 1)
        self.assertAlmostEqual(self.engine.status()["today_usd"], costs.SpendBook(self.book.path).today_usd())


if __name__ == "__main__":
    import unittest

    unittest.main()
