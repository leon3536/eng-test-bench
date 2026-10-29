"""Conversation files, the event log, assistant costs and redaction.

The conversation history is replayed verbatim to an API that binds thinking
blocks to everything before them, so these tests pin down that it is only
ever appended to, that a crash is repaired by appending, and that what the
dashboard and the disk see never holds a secret.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bench import fsutil, settings
from bench.assistant import client as client_mod
from bench.assistant import conversation as conv_mod
from bench.assistant import costs, redact
from bench.assistant.conversation import ConversationStore, EventLog, lock_reason
from bench.assistant.prompt import PROMPT_HASH
from tests.fakes_anthropic import EngineCase, Reply, status_error, text, tool_use
from tests.support import MINIMAL_YAML, BenchTestCase

PLANTED = "planted-0123456789-secret"


def user(value: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": value}]}


def assistant(*blocks: dict) -> dict:
    return {"role": "assistant", "content": list(blocks)}


class StoreCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.data = self.isolate_data_dir()
        self.store = ConversationStore()

    def create(self, **kwargs) -> str:
        meta = self.store.create(kwargs.pop("effort", "medium"), kwargs.pop("fallbacks", "default"),
                                 kwargs.pop("title", None), kwargs.pop("design", None), model=client_mod.MODEL,
                                 prompt_hash=PROMPT_HASH, tools_hash="sha256:tools", **kwargs)
        return meta["id"]

    def path(self, cid: str) -> Path:
        return self.store.root / cid / "messages.jsonl"


class Messages(StoreCase):
    def test_root_follows_the_profile_data_dir(self) -> None:
        self.assertEqual(self.store.root, settings.ASSISTANT_DIR / "conversations")
        self.assertTrue(str(self.store.root).startswith(str(self.data)))

    def test_ids_are_validated_before_any_path_is_built(self) -> None:
        cid = self.create()
        self.assertRegex(cid, r"^c-\d{8}-\d{6}-[0-9a-f]{4}$")
        for bad in ("../../config", "c-20261029-000000-ZZZZ", "C-20261029-000000-abcd", "", None):
            with self.subTest(bad), self.assertRaises(KeyError):
                self.store.meta(bad)  # type: ignore[arg-type]

    def test_commit_only_appends(self) -> None:
        cid = self.create()
        self.store.commit(cid, user("one"))
        before = self.path(cid).read_bytes()
        self.store.commit(cid, assistant(text("two")), user("three"))
        after = self.path(cid).read_bytes()
        self.assertTrue(after.startswith(before))
        self.assertEqual(after.count(b"\n"), 3)
        self.assertEqual([m["role"] for m in self.store.load_messages(cid)], ["user", "assistant", "user"])

    def test_a_torn_last_line_is_cut_off(self) -> None:
        cid = self.create()
        self.store.commit(cid, user("one"), assistant(text("two")))
        good = self.path(cid).read_bytes()
        with open(self.path(cid), "ab") as f:
            f.write(b'{"role": "user", "content": [{"type": "te')
        self.assertEqual(len(self.store.load_messages(cid)), 2)
        self.assertEqual(self.path(cid).read_bytes(), good)
        self.store.commit(cid, user("three"))
        self.assertEqual([m["content"][0]["text"] for m in fsutil.read_jsonl(self.path(cid))], ["one", "two", "three"])

    def test_a_dangling_tool_use_is_answered_as_interrupted_once(self) -> None:
        cid = self.create()
        self.store.commit(cid, user("price it"),
                          assistant(text("Checking."), tool_use("toolu_1", "lookup_cost", {}),
                                    tool_use("toolu_2", "lookup_cost", {})))
        repaired = self.store.load_messages(cid)
        closing = repaired[-1]
        self.assertEqual(closing["role"], "user")
        self.assertEqual([b["tool_use_id"] for b in closing["content"]], ["toolu_1", "toolu_2"])
        for block in closing["content"]:
            self.assertTrue(block["is_error"])
            self.assertIn("interrupted", block["content"])
        self.assertEqual(self.store.load_messages(cid), repaired)  # persisted, not added twice
        self.assertEqual(len(fsutil.read_jsonl(self.path(cid))), 3)

    def test_answered_tool_use_is_left_alone(self) -> None:
        cid = self.create()
        answer = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "{}"}]}
        self.store.commit(cid, user("go"), assistant(tool_use("toolu_1", "lookup_cost", {})), answer)
        self.assertEqual(len(self.store.load_messages(cid)), 3)

    def test_meta_survives_a_locked_file_and_reloads(self) -> None:
        cid = self.create(title="Coverage on GLM")
        with mock.patch.object(fsutil, "atomic_write_json", return_value=False):
            self.store.update_meta(cid, turns=3)
        self.assertEqual(self.store.meta(cid)["turns"], 3)  # memory is authoritative
        self.store.update_meta(cid, requests=2)
        fresh = ConversationStore().meta(cid)
        self.assertEqual((fresh["turns"], fresh["requests"], fresh["title"]), (3, 2, "Coverage on GLM"))

    def test_list_is_newest_first_and_skips_nothing_silently(self) -> None:
        first = self.create()
        second = self.create()
        self.store.update_meta(first, created_at="2026-10-01T00:00:00+00:00")
        broken = self.store.root / "c-20261029-010101-beef"
        broken.mkdir(parents=True)
        (broken / "meta.json").write_text("{not json", encoding="utf-8")
        listed = ConversationStore().list()
        readable = [m["id"] for m in listed if "error" not in m]
        self.assertEqual(readable, [second, first])
        bad = [m for m in listed if m["id"] == broken.name][0]
        self.assertIn("error", bad)


class Recap(StoreCase):
    def test_recap_is_plain_text_of_the_last_ten_texts(self) -> None:
        cid = self.create(title="Pi pricing", design={"id": "d-pi-block-1-ab12", "version": 2})
        self.store.commit(cid, user("first question"))
        thinking = {"type": "thinking", "thinking": "PRIVATE-REASONING", "signature": "sig"}
        for i in range(8):
            self.store.commit(cid, assistant(thinking, text(f"answer {i}"), tool_use(f"toolu_{i}", "lookup_cost", {})),
                              {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"toolu_{i}",
                                                            "content": "TOOL-RESULT-TEXT"}]})
        self.store.commit(cid, user("last question"), assistant(text("last answer")))
        recap = self.store.recap(cid)
        self.assertNotIn("PRIVATE-REASONING", recap)
        self.assertNotIn("TOOL-RESULT-TEXT", recap)
        self.assertNotIn("first question", recap)  # older than the last ten
        self.assertIn("You: last question", recap)
        self.assertIn("Assistant: last answer", recap)
        self.assertEqual(sum(1 for line in recap.splitlines() if line.startswith(("You:", "Assistant:"))), 10)
        self.assertIn("d-pi-block-1-ab12 v2", recap)
        self.assertIn(cid, recap)


class Locks(StoreCase):
    def test_lock_reasons(self) -> None:
        meta = self.store.meta(self.create())
        self.assertIsNone(lock_reason(meta, PROMPT_HASH, "sha256:tools", client_mod.MODEL))
        self.assertIn("instructions", lock_reason(meta, "sha256:new", "sha256:tools", client_mod.MODEL))
        self.assertIn("tools", lock_reason(meta, PROMPT_HASH, "sha256:other", client_mod.MODEL))
        self.assertIn("model", lock_reason(meta, PROMPT_HASH, "sha256:tools", "claude-opus-6"))
        long = {**meta, "last_context_tokens": conv_mod.LOCK_CONTEXT_TOKENS + 1}
        self.assertIn("long", lock_reason(long, PROMPT_HASH, "sha256:tools", client_mod.MODEL))
        locked = {**meta, "status": "locked", "locked_reason": "History can no longer be replayed."}
        self.assertEqual(lock_reason(locked, PROMPT_HASH, "sha256:tools", client_mod.MODEL),
                         "History can no longer be replayed.")


class EventLogs(StoreCase):
    def test_sequence_ephemeral_limit_and_turn_reset(self) -> None:
        cid = self.create()
        with mock.patch.object(conv_mod, "EPHEMERAL_LIMIT", 5):
            log = EventLog(self.store, cid, 1)
        log.start_turn(1)
        first = log.append("user_message", True, text="hi", attach_design=None)
        self.assertEqual((first["seq"], first["turn"]), (1, 1))
        for i in range(4):
            log.append("text_delta", False, block_key="1:1:0", text=str(i))
        log.append("assistant_text", True, block_key="1:1:0", text="0123")
        for i in range(4, 8):
            log.append("text_delta", False, block_key="1:1:0", text=str(i))
        live = log.since(0)
        self.assertEqual([e["seq"] for e in live], [1, 5, 6, 7, 8, 9, 10])  # merged in sequence order
        self.assertEqual([e["text"] for e in live if e["type"] == "text_delta"], ["3", "4", "5", "6", "7"])
        self.assertEqual([e["seq"] for e in log.since(6)], [7, 8, 9, 10])
        self.assertEqual(log.since(10), [])
        log.start_turn(2)
        self.assertEqual([e["type"] for e in log.since(0)], ["user_message", "assistant_text"])
        self.assertEqual(log.next_seq, 11)
        self.assertEqual([e["seq"] for e in self.store.transcript(cid)], [1, 6])

    def test_older_durable_events_come_from_the_transcript(self) -> None:
        cid = self.create()
        old = EventLog(self.store, cid, 1)
        for i in range(3):
            old.append("assistant_text", True, block_key=f"1:1:{i}", text=f"t{i}")
        reloaded = EventLog(self.store, cid, 4)
        reloaded.append("notice", True, level="info", code="stopped", message="Stopped.")
        self.assertEqual([e["seq"] for e in reloaded.since(0)], [1, 2, 3, 4])
        self.assertEqual([e["seq"] for e in reloaded.since(2)], [3, 4])

    def test_preloaded_history_means_reads_never_touch_the_disk(self) -> None:
        cid = self.create()
        old = EventLog(self.store, cid, 1)
        for i in range(3):
            old.append("assistant_text", True, block_key=f"1:1:{i}", text=f"t{i}")
        history = self.store.transcript(cid, 0)
        # A restart far past the transcript, as crash recovery does.
        reloaded = EventLog(self.store, cid, 10_000_004, history=history)
        with mock.patch.object(self.store, "transcript", side_effect=AssertionError("read the transcript")):
            reloaded.append("notice", True, level="info", code="stopped", message="Stopped.")
            self.assertEqual([e["seq"] for e in reloaded.since(0)], [1, 2, 3, 10_000_004])
            self.assertEqual([e["seq"] for e in reloaded.since(3)], [10_000_004])
            self.assertEqual(reloaded.since(10_000_004), [])

    def test_wait_returns_when_an_event_arrives_or_times_out(self) -> None:
        cid = self.create()
        log = EventLog(self.store, cid, 1)
        self.assertEqual(log.wait(0, 0.05), [])
        log.append("status", False, text="Thinking…")
        self.assertEqual(len(log.wait(0, 1)), 1)


class Costs(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cfg = self.load_test_config(MINIMAL_YAML)

    def test_price_iteration_with_cache_writes_and_dated_ids(self) -> None:
        usd, priced = costs.price_iteration(self.cfg, "claude-opus-5-5", 1000, 2000, 3000, 400)
        self.assertEqual(priced, "exact")
        self.assertAlmostEqual(usd, (1000 * 4.0 + 2000 * 4.0 * 1.25 + 3000 * 0.20 + 400 * 20.0) / 1e6)
        dated, priced = costs.price_iteration(self.cfg, "claude-opus-5-5-20261101", 1000, 2000, 3000, 400)
        self.assertEqual((dated, priced), (usd, "exact"))

    def test_unknown_models_are_priced_at_the_dearest_rates(self) -> None:
        usd, priced = costs.price_iteration(self.cfg, "claude-mystery-9", 1_000_000, 0, 0, 1_000_000)
        worst = costs.worst_price(self.cfg)
        self.assertEqual(priced, "estimated")
        self.assertAlmostEqual(usd, worst.input + worst.output)
        table = self.cfg.pricing["anthropic"]
        self.assertEqual(worst.output, max(p.output for p in table.values()))

    def test_price_message_without_iterations_uses_the_serving_model(self) -> None:
        usage = SimpleNamespace(input_tokens=100, output_tokens=10, cache_creation_input_tokens=None,
                                cache_read_input_tokens=50, iterations=None)
        message = SimpleNamespace(model="claude-opus-5", usage=usage)
        priced = costs.price_message(self.cfg, message, "claude-opus-5-5")
        self.assertFalse(priced["fallback_served"])
        self.assertEqual(priced["served_model"], "claude-opus-5")
        self.assertAlmostEqual(priced["usd"], (100 * 5.0 + 50 * 0.5 + 10 * 25.0) / 1e6)
        self.assertEqual(costs.context_tokens(priced), 150)

    def test_price_partial_adds_the_last_output_count(self) -> None:
        start = SimpleNamespace(input_tokens=300, output_tokens=1, cache_creation_input_tokens=100,
                                cache_read_input_tokens=0)
        priced = costs.price_partial(self.cfg, start, 250, "claude-opus-5-5", "claude-opus-5-5")
        self.assertTrue(priced["partial"])
        self.assertAlmostEqual(priced["usd"], (300 * 4.0 + 100 * 4.0 * 1.25 + 250 * 20.0) / 1e6)


class Spend(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.path = self.tmpdir() / "usage.jsonl"

    def test_totals_by_local_day_and_conversation(self) -> None:
        book = costs.SpendBook(self.path)
        now = datetime.now().astimezone()
        midnight = datetime.combine(now.date(), dtime(0, 1), tzinfo=now.tzinfo)
        yesterday = midnight - timedelta(hours=2)
        book.record({"at": midnight.astimezone(timezone.utc).isoformat(), "conversation_id": "c-a", "usd": 0.25})
        book.record({"at": now.astimezone(timezone.utc).isoformat(), "conversation_id": "c-b", "usd": 0.50})
        book.record({"at": yesterday.astimezone(timezone.utc).isoformat(), "conversation_id": "c-a", "usd": 1.00})
        self.assertAlmostEqual(book.today_usd(), 0.75)
        self.assertAlmostEqual(book.conversation_usd("c-a"), 1.25)
        days = book.days(3)
        self.assertEqual([d["date"] for d in days],
                         [(now.date() - timedelta(days=k)).isoformat() for k in (2, 1, 0)])
        self.assertEqual([d["requests"] for d in days], [0, 1, 2])

        reloaded = costs.SpendBook(self.path)
        self.assertAlmostEqual(reloaded.today_usd(), 0.75)
        self.assertEqual(len(fsutil.read_jsonl(self.path)), 3)

    def test_a_record_counts_even_if_the_file_is_locked(self) -> None:
        book = costs.SpendBook(self.path)
        with mock.patch.object(fsutil, "append_jsonl", side_effect=PermissionError("locked")):
            book.record({"conversation_id": "c-a", "usd": 0.40})
        self.assertAlmostEqual(book.today_usd(), 0.40)

    def test_default_path_is_under_the_profile_assistant_dir(self) -> None:
        self.isolate_data_dir()
        self.assertEqual(costs.SpendBook().path, settings.ASSISTANT_DIR / "usage.jsonl")


class Redaction(BenchTestCase):
    def test_values_and_shapes_are_masked(self) -> None:
        with mock.patch.dict(os.environ, {"BENCH_FAKE_SERVICE_TOKEN": PLANTED}):
            masked = redact.redact(f"token {PLANTED} and sk-ant-api03-ABCDEFGHIJKLMNOPQRST and "
                                   f"fw_ABCDEFGHIJKLMNOPQRST and Bearer abcdefghijklmnopqrstu")
            self.assertNotIn(PLANTED, masked)
            self.assertEqual(masked.count(redact.MASK), 4)
            self.assertTrue(redact.contains_secret(f"x{PLANTED}x"))
            nested = redact.redact_obj({"a": [PLANTED, {"b": PLANTED}], PLANTED: 1, "n": 2})
            self.assertNotIn(PLANTED, json.dumps(nested))
            self.assertEqual(nested["n"], 2)
        self.assertFalse(redact.contains_secret("glm-5.3 costs $0.07 per million input tokens"))
        self.assertEqual(redact.redact("short sk-abc"), "short sk-abc")

    def test_hyphenated_names_are_not_mistaken_for_keys(self) -> None:
        for name in ("task-deps-installer-for-mips", "flask-api-endpoint-tests",
                     "disk-compression-recovery-benchmark", "risk-assessment-model-eval",
                     "rawfw_ABCDEFGHIJKLMNOPQRST"):
            with self.subTest(name):
                self.assertFalse(redact.contains_secret(f"run {name} on GLM"))
                self.assertEqual(redact.redact(f"run {name} on GLM"), f"run {name} on GLM")
        for text_with_key in ("key=sk-ant-api03-AAAAAAAAAAAAAAAAAAAA", "sk-or-v1-0123456789abcdef0123",
                              "(sk-ant-api03-AAAAAAAAAAAAAAAAAAAA)", "token:fw_ABCDEFGHIJKLMNOPQRST"):
            with self.subTest(text_with_key):
                self.assertTrue(redact.contains_secret(text_with_key))
                self.assertIn(redact.MASK, redact.redact(text_with_key))


class RedactedSession(EngineCase):
    def test_a_planted_key_reaches_no_event_request_or_file(self) -> None:
        os.environ["BENCH_FAKE_SERVICE_TOKEN"] = PLANTED  # restored by BenchTestCase
        self.tools.results["lookup_cost"] = json.dumps({"usd_per_trial": 0.42, "note": f"seen {PLANTED}"})

        def failing_save(args: dict) -> None:
            raise RuntimeError(f"could not save with {PLANTED}")

        self.tools.hooks["save_note"] = failing_save
        self.script(
            Reply([tool_use("toolu_1", "lookup_cost", {"family": "pi", "model": "claude-opus-5-5"}),
                   tool_use("toolu_2", "save_note", {"title": "t", "text": "x"})], stop_reason="tool_use"),
            Reply([text("Here is what I found.")]),
            Reply.raising(status_error(400, f"bad request mentioning {PLANTED}")),
        )
        cid = self.conversation()
        self.turn(cid)
        self.turn(cid, "And on GLM?")
        self.assertEqual(self.events(cid, "turn_error")[-1]["kind"], "bad_request")

        results = self.messages(cid)[2]["content"]
        self.assertIn(redact.MASK, results[0]["content"])
        self.assertTrue(results[1]["is_error"])
        self.assertIn(redact.MASK, results[1]["content"])
        self.assertNotIn(PLANTED, json.dumps(self.fake.calls))
        self.assertNotIn(PLANTED, json.dumps(self.engine.events(cid, 0)))
        self.assertNotIn(PLANTED, json.dumps(self.engine.conversations()))
        scanned = 0
        for path in settings.ASSISTANT_DIR.rglob("*"):
            if path.is_file():
                scanned += 1
                self.assertNotIn(PLANTED.encode(), path.read_bytes(), path.name)
        self.assertGreaterEqual(scanned, 4)  # meta, messages, transcript, usage


if __name__ == "__main__":
    import unittest

    unittest.main()
