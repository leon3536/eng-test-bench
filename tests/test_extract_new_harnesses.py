"""Cline CLI, OpenCode, Kilo Code and Goose usage from their own logs.

Each log here is in the shape the 2026-10-30 dev pilots wrote it (cline.txt's
usage events, opencode.txt's step_finish events, Goose's sessions.db
usage_ledger), with Harbor's totals as its 0.25.0 adapter sums them: Cline's
and OpenCode's leave reasoning tokens out of the output, OpenCode's leaves
cache writes out of the prompt. Priced from the log, every trial costs exactly
what its calls billed.
"""

from __future__ import annotations

import json
import sqlite3
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

from tests import ledger_fixtures as lf
from tests.support import BenchTestCase

from bench import extract, settings
from bench.agents.opencode import finished_despite_epipe

PREFIXES = {"openrouter/": "openrouter", "anthropic/": "anthropic"}
SMOKE_JOBS = settings.ROOT / "profiles" / "dev" / "jobs"
# (fresh input, cache reads, 5-minute writes, output, reasoning) per call
CALLS = ((2, 0, 3000, 400, 120), (3, 2900, 500, 250, 0), (4, 3500, 200, 90, 30))
RATES = {"input": 4.0, "read": 0.2, "write_5m": 5.0, "output": 20.0}


def billed(call: Sequence[int]) -> float:
    fresh, read, w5, out, reasoning = call
    return (fresh * RATES["input"] + read * RATES["read"] + w5 * RATES["write_5m"]
            + (out + reasoning) * RATES["output"]) / 1e6


def ctx() -> extract.RunContext:
    return extract.RunContext(run_key="dev/r1", profile="dev", run_id="r1", source="bench", arm="arm",
                              provider="anthropic", model_prefix="anthropic/", effort_label="medium",
                              dataset="terminal-bench@2.0", local_snapshot=None)


def write_cline(agent: Path, calls: Sequence[Sequence[int]]) -> tuple[int, int, int]:
    """cline.txt; returns Harbor's totals, from messages.json, which drops reasoningTokenCount."""
    lines = [{"ts": "t", "type": "agent_event", "event": {"type": "iteration_start", "iteration": 1}}]
    for fresh, read, w5, out, reasoning in calls:
        event = {"type": "usage", "inputTokens": fresh + read + w5, "outputTokens": out,
                 "cost": billed((fresh, read, w5, out, reasoning))}
        if read:
            event["cacheReadTokens"] = read
        if w5:
            event["cacheWriteTokens"] = w5
        if reasoning:
            event["reasoningTokenCount"] = reasoning
        lines.append({"ts": "t", "type": "agent_event", "event": event})
    lf._jsonl(agent / "cline.txt", lines + [{"type": "run_result"}], cut=False)
    return (sum(c[0] + c[1] + c[2] for c in calls), sum(c[1] for c in calls), sum(c[3] for c in calls))


def write_opencode(agent: Path, calls: Sequence[Sequence[int]], name: str = "opencode.txt") -> tuple[int, int, int]:
    """opencode.txt (or Kilo's kilo.txt, the same stream); returns Harbor's totals, which leave out the
    writes and the reasoning."""
    lines = []
    for fresh, read, w5, out, reasoning in calls:
        lines += [{"type": "step_start", "part": {}},
                  {"type": "step_finish", "part": {"reason": "tool-calls", "cost": billed((fresh, read, w5, out,
                                                                                             reasoning)),
                                                   "tokens": {"input": fresh, "output": out, "reasoning": reasoning,
                                                              "cache": {"read": read, "write": w5}}}}]
    lf._jsonl(agent / name, lines, cut=False)
    return (sum(c[0] + c[1] for c in calls), sum(c[1] for c in calls), sum(c[3] for c in calls))


def write_kilo(agent: Path, calls: Sequence[Sequence[int]]) -> tuple[int, int, int]:
    return write_opencode(agent, calls, "kilo.txt")


def write_goose(agent: Path, calls: Sequence[Sequence[int]], keep_open: bool = False) -> sqlite3.Connection | None:
    """sessions.db with its usage_ledger, in write-ahead-log mode as Goose keeps it."""
    db = agent / "goose" / "xdg-data" / "goose" / "sessions" / "sessions.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("CREATE TABLE usage_ledger (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
                 "created_timestamp INTEGER NOT NULL, model TEXT, input_tokens INTEGER, output_tokens INTEGER, "
                 "total_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER, cost REAL, "
                 "cost_source TEXT, is_compaction INTEGER DEFAULT 0)")
    for i, (fresh, read, w5, out, reasoning) in enumerate(calls):
        n_in = fresh + read + w5
        conn.execute("INSERT INTO usage_ledger (session_id, created_timestamp, model, input_tokens, output_tokens, "
                     "total_tokens, cache_read_tokens, cache_write_tokens, cost, cost_source) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     ("s1", i, "claude-opus-5-5", n_in, out + reasoning, n_in + out + reasoning, read, w5,
                      billed((fresh, read, w5, out, reasoning)), "canonical"))
    conn.commit()
    if keep_open:
        return conn
    conn.close()
    return None


class NewHarnessUsage(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        prices = self.tmpdir() / "pricing.yaml"
        prices.write_text(lf.OPUS_PRICING_YAML, encoding="utf-8", newline="\n")
        patcher = mock.patch.object(settings, "PRICING", prices)
        patcher.start()
        self.addCleanup(patcher.stop)

    def finished(self, family: str, harbor: tuple[int, int, int], d: Path) -> extract.TrialFacts:
        usage = {"anthropic/claude-opus-5-5": {"n_input_tokens": harbor[0], "n_cache_tokens": harbor[1],
                                               "n_output_tokens": harbor[2], "cost_usd": self.exact()}}
        lf.make_trial(d, family=family, model_name="anthropic/claude-opus-5-5", tokens=harbor, usage=usage,
                      cost=self.exact())
        facts = extract.parse_trial(d, ctx(), PREFIXES)
        assert facts is not None
        return facts

    def exact(self, calls: Sequence[Sequence[int]] = CALLS) -> float:
        return sum(billed(c) for c in calls)

    def check_full(self, facts: extract.TrialFacts, where: str) -> None:
        self.assertEqual((facts.n_input, facts.n_cache, facts.n_output), (10109, 6400, 890))
        (entry,) = json.loads(facts.usage_json).values()
        self.assertEqual((entry["n_cache_write_5m"], entry["n_cache_write_1h"]), (3700, 0))
        self.assertEqual(entry["recovered_from"], where)
        self.assertAlmostEqual(facts.cost_at_ingest_usd, self.exact(), places=12)
        self.assertAlmostEqual(facts.reported_cost_usd, self.exact(), places=12)
        self.assertEqual(json.loads(facts.flags_json), {})

    def test_cline_output_gains_its_reasoning_tokens(self) -> None:
        d = self.tmpdir() / "job" / "t"
        harbor = write_cline(d / "agent", CALLS)
        facts = self.finished("cline-cli", harbor, d)
        self.check_full(facts, "agent/cline.txt")
        (entry,) = json.loads(facts.usage_json).values()
        self.assertEqual(entry["harbor_reported"], {"n_input": 10109, "n_cache": 6400, "n_output": 740})

    def test_opencode_prompt_gains_its_cache_writes_and_output_its_reasoning(self) -> None:
        d = self.tmpdir() / "job" / "t"
        harbor = write_opencode(d / "agent", CALLS)
        facts = self.finished("opencode", harbor, d)
        self.check_full(facts, "agent/opencode.txt")
        (entry,) = json.loads(facts.usage_json).values()
        self.assertEqual(entry["harbor_reported"], {"n_input": 6409, "n_cache": 6400, "n_output": 740})

    def test_kilo_reads_the_same_step_finish_stream_from_kilo_txt(self) -> None:
        d = self.tmpdir() / "job" / "t"
        harbor = write_kilo(d / "agent", CALLS)
        facts = self.finished("kilo-code", harbor, d)
        self.check_full(facts, "agent/kilo.txt")
        (entry,) = json.loads(facts.usage_json).values()
        self.assertEqual(entry["harbor_reported"], {"n_input": 6409, "n_cache": 6400, "n_output": 740})

    def test_goose_reads_its_session_database_with_its_write_ahead_log(self) -> None:
        d = self.tmpdir() / "job" / "t"
        conn = write_goose(d / "agent", CALLS, keep_open=True)  # nothing checkpointed: the rows are in -wal
        assert conn is not None
        self.addCleanup(conn.close)
        self.assertTrue((d / "agent" / extract._GOOSE_DB).with_name("sessions.db-wal").stat().st_size > 0)
        facts = self.finished("goose", (10109, 6400, 890), d)
        self.check_full(facts, "agent/goose/xdg-data/goose/sessions/sessions.db")

    def test_goose_without_its_database_takes_the_complete_events_writes(self) -> None:
        d = self.tmpdir() / "job" / "t"
        lf._jsonl(d / "agent" / "goose.txt", [
            {"type": "message", "message": {"role": "assistant", "content": []}},
            {"type": "complete", "total_tokens": 10999, "input_tokens": 10109, "output_tokens": 890,
             "cache_read_input_tokens": 6400, "cache_write_input_tokens": 3700, "cost_usd": self.exact()}],
            cut=False)
        facts = self.finished("goose", (10109, 6400, 890), d)
        (entry,) = json.loads(facts.usage_json).values()
        self.assertEqual((entry["n_cache_write_5m"], entry["n_cache_write_1h"]), (3700, 0))
        self.assertNotIn("harbor_reported", entry)  # the same totals Harbor read: nothing replaced
        self.assertAlmostEqual(facts.cost_at_ingest_usd, self.exact(), places=12)

    def test_killed_trials_recover_their_completed_calls(self) -> None:
        for family, write in (("cline-cli", write_cline), ("opencode", write_opencode), ("kilo-code", write_kilo),
                              ("goose", write_goose)):
            with self.subTest(family):
                d = self.tmpdir() / "job" / "t"
                lf.make_trial(d, family=family, model_name="anthropic/claude-opus-5-5", finished=False)
                write(d / "agent", CALLS[:2])
                facts = extract.parse_incomplete(d, ctx(), PREFIXES)
                assert facts is not None
                self.assertEqual((facts.n_input, facts.n_cache, facts.n_output), (6405, 2900, 770))
                self.assertAlmostEqual(facts.cost_at_ingest_usd, self.exact(CALLS[:2]), places=12)

    def test_a_replaced_usage_survives_its_folder(self) -> None:
        d = self.tmpdir() / "job" / "t"
        harbor = write_cline(d / "agent", CALLS)
        facts = self.finished("cline-cli", harbor, d)
        stored = {"usage": json.loads(facts.usage_json), "flags": json.loads(facts.flags_json)}
        result = extract.read_json(d / "result.json")
        again = extract.facts_from_result(result, None, None, trial_dir="job/t", ctx=ctx(), prefixes=PREFIXES,
                                          log=None, stored=stored)
        self.assertEqual((again.n_input, again.n_cache, again.n_output), (10109, 6400, 890))
        self.assertAlmostEqual(again.cost_at_ingest_usd, self.exact(), places=12)
        self.assertTrue(json.loads(again.flags_json)["trial_folder_missing"])

    def test_a_log_with_no_calls_leaves_harbors_usage(self) -> None:
        d = self.tmpdir() / "job" / "t"
        write_cline(d / "agent", ())
        facts = self.finished("cline-cli", (100, 0, 10), d)
        self.assertEqual((facts.n_input, facts.n_output), (100, 10))
        (entry,) = json.loads(facts.usage_json).values()
        self.assertNotIn("harbor_reported", entry)

    @unittest.skipUnless((SMOKE_JOBS / "20261030-152750-pilot").is_dir(), "the 2026-10-30 dev smoke jobs are absent")
    def test_the_smoke_trials_logs_add_up_to_the_harness_costs(self) -> None:
        # Read-only. Cline's per-call costs are OpenRouter's bill; its tokens priced at the bill's rates
        # come out exact only with the reasoning tokens counted as output.
        trial = next((SMOKE_JOBS / "20261030-152750-pilot" / "cline-deepseek").glob("*__*"))
        log = extract.read_harness_log(trial)
        assert log is not None
        self.assertEqual(log.where, "agent/cline.txt")
        result = extract.read_json(trial / "result.json")
        harbor = result["agent_result"]
        self.assertEqual(sum(c[1] for c in log.calls), harbor["n_input_tokens"])
        self.assertGreater(sum(c[3] for c in log.calls), harbor["n_output_tokens"])
        priced = sum(((c[1] - c[2]) * 0.045 + c[2] * 0.006 + c[3] * 0.50) / 1e6 for c in log.calls)
        self.assertAlmostEqual(priced, harbor["cost_usd"], places=9)


class OpenCodeBunEpipe(unittest.TestCase):
    STEP = json.dumps({"type": "step_finish", "part": {"reason": "stop"}})
    TRACE = ("45 |       return applyHandlers(handlers, emitter, args), !0;\n50 |     throw er;\n"
             "                ^\nerror: EPIPE: broken pipe, write\n      at emitError (node:events:50:13)\n")

    def test_only_a_finished_run_with_nothing_but_the_epipe_trace_passes(self) -> None:
        start = json.dumps({"type": "step_start"})
        self.assertTrue(finished_despite_epipe(f"{start}\n{self.TRACE}{self.STEP}\n"))
        self.assertFalse(finished_despite_epipe(f"{start}\n{self.STEP}\n"))  # exit 1 for another reason
        self.assertFalse(finished_despite_epipe(f"{start}\n{self.TRACE}"))  # never finished
        unfinished = json.dumps({"type": "step_finish", "part": {"reason": "tool-calls"}})
        self.assertFalse(finished_despite_epipe(f"{self.TRACE}{unfinished}\n"))
        error = json.dumps({"type": "error", "error": {"name": "APIError"}})
        self.assertFalse(finished_despite_epipe(f"{error}\n{self.TRACE}{self.STEP}\n"))
        self.assertFalse(finished_despite_epipe(f"{self.TRACE}Killed\n{self.STEP}\n"))

    def test_run_keeps_only_that_exit_and_reads_the_containers_copy(self) -> None:
        import asyncio
        import tempfile

        from harbor.agents.installed.base import ApiRateLimitError, NonZeroAgentExitCodeError
        from harbor.agents.installed.opencode import OpenCode
        from harbor.environments.base import ExecResult

        from bench.agents.opencode import OpenCodeBunEpipe as Agent

        finished = f"{self.TRACE}{self.STEP}\n"

        class Env:
            def __init__(self, text: str) -> None:
                self.text, self.commands = text, []

            async def exec(self, command: str, **_: object) -> ExecResult:
                self.commands.append(command)
                return ExecResult(stdout=self.text, stderr="", return_code=0)

        def run(error: Exception, container_text: str) -> Env:
            agent = Agent(logs_dir=Path(tempfile.mkdtemp()), model_name="openrouter/deepseek/deepseek-v4.1-flash")
            env = Env(container_text)
            with mock.patch.object(OpenCode, "run", mock.AsyncMock(side_effect=error)):
                asyncio.run(agent.run("task", env, mock.Mock()))  # type: ignore[arg-type]
            return env

        env = run(NonZeroAgentExitCodeError("Command failed (exit 1): opencode ..."), finished)
        self.assertEqual(env.commands, ["cat /logs/agent/opencode.txt"])  # the host's copy is not trusted first
        for error, text in ((NonZeroAgentExitCodeError("Command failed (exit 1): x"), self.TRACE),
                            (NonZeroAgentExitCodeError("Command failed (exit 137): x"), finished),
                            (ApiRateLimitError("Command failed (exit 1): x"), finished)):
            with self.subTest(error=repr(error)), self.assertRaises(NonZeroAgentExitCodeError):
                run(error, text)
