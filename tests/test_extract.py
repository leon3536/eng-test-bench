"""Parsing Harbor trial folders into ledger facts, and finding them."""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import re
import unittest
from pathlib import Path
from unittest import mock

from tests import ledger_fixtures as lf
from tests.support import FIXTURES, BenchTestCase, fixture_schema

from bench import classify, estimates, extract, report, runs, settings, spec
from bench.extract import RunContext, ScanRoot

PREFIXES = {"openrouter/": "openrouter", "anthropic/": "anthropic",
            "fireworks_ai/accounts/fireworks/models/": "fireworks"}
PILOT_3_JOBS = settings.ROOT / "profiles" / "dev" / "jobs" / "20261029-055852-pilot"
OPUS_PILOT_MINI = (settings.ROOT / "jobs" / "20261029-122236-pilot" / "mini-swe-agent"
                   / "adaptive-rejection-sampler__saBMoJj")


def bench_ctx(provider: str = "openrouter", prefix: str | None = "openrouter/", **kw: object) -> RunContext:
    values = dict(run_key="dev/r1", profile="dev", run_id="r1", source="bench", arm="arm", provider=provider,
                  model_prefix=prefix, effort_label="medium", dataset="terminal-bench@2.0", local_snapshot=None)
    values.update(kw)
    return RunContext(**values)  # type: ignore[arg-type]


def external_ctx() -> RunContext:
    return RunContext(run_key="external/jobs/x", profile="external", run_id="x", source="external", arm="x",
                      provider=None, model_prefix=None, effort_label=None, dataset=None, local_snapshot=None)


def pilot_ctx(run_id: str, arm: str) -> RunContext:
    run = json.loads((lf.FIXTURE_DEV / "runs" / run_id / "run.json").read_text(encoding="utf-8"))
    return extract.context_for_bench_arm(run, arm, "dev")


def pilot_trial(run_id: str, arm: str) -> Path:
    return next(p for p in (lf.FIXTURE_DEV / "jobs" / run_id / arm).iterdir() if p.is_dir())


class ModelNames(unittest.TestCase):
    def test_split_model(self) -> None:
        ctx = external_ctx()
        self.assertEqual(extract.split_model("openrouter/z-ai/glm-5.3", ctx, PREFIXES, None),
                         ("openrouter", "z-ai/glm-5.3"))
        self.assertEqual(extract.split_model("anthropic/claude-opus-5-5", ctx, PREFIXES, None),
                         ("anthropic", "claude-opus-5-5"))
        self.assertEqual(extract.split_model("fireworks_ai/accounts/fireworks/models/x", ctx, PREFIXES, None),
                         ("fireworks", "x"))
        # Bench runs take the provider from the run, not from the name.
        self.assertEqual(extract.split_model("openrouter/z-ai/glm-5.3", bench_ctx(), {}, None),
                         ("openrouter", "z-ai/glm-5.3"))
        # Unknown prefix: Harbor's model_info, then the first path segment.
        self.assertEqual(extract.split_model("acme/m1", ctx, {}, {"provider": "acme-ai", "name": "m1"}),
                         ("acme-ai", "m1"))
        self.assertEqual(extract.split_model("acme/m1", ctx, {}, None), ("acme", "m1"))

    def test_usage_keys_canonicalise_alike(self) -> None:
        ctx = bench_ctx()
        pi = extract.canonical_model("openrouter/deepseek/deepseek-v4.1-flash", ctx, PREFIXES)
        cc = extract.canonical_model("deepseek/deepseek-v4.1-flash", ctx, PREFIXES)
        self.assertEqual(pi, cc)
        self.assertEqual(pi, "deepseek/deepseek-v4.1-flash")

    def test_is_primary_never_prefix_matches(self) -> None:
        self.assertTrue(extract.is_primary("z-ai/glm-5.3", "z-ai/glm-5.3"))
        self.assertFalse(extract.is_primary("z-ai/glm-5.3-flash", "z-ai/glm-5.3"))
        self.assertTrue(extract.is_primary("claude-haiku-5-5-20261001", "claude-haiku-5-5"))
        self.assertFalse(extract.is_primary("claude-opus-5-5", "claude-opus-5"))


class ConfigIdentity(BenchTestCase):
    def _hash(self, d: Path, **kw: object) -> str:
        lf.make_job(d.parent, n_concurrent=int(kw.pop("n_concurrent", 2)))  # type: ignore[arg-type]
        lf.make_trial(d, **kw)  # type: ignore[arg-type]
        facts = extract.parse_trial(d, bench_ctx(arm=d.parent.name), PREFIXES, root=d.parents[2])
        assert facts is not None
        return facts.config_hash

    def test_hash_ignores_arm_versions_ids_and_paths(self) -> None:
        tmp = self.tmpdir()
        base = self._hash(tmp / "a" / "claude-code" / "t1", family="claude-code",
                          kwargs={"reasoning_effort": "medium", "max_budget_usd": "1"})
        same = [
            self._hash(tmp / "b" / "claude-code-deepseek" / "t2", family="claude-code", version="9.9.9",
                       job_id="22222222-2222-4222-8222-222222222222", n_concurrent=5,
                       kwargs={"reasoning_effort": "medium", "max_budget_usd": 1}),
            self._hash(tmp / "c" / "cc" / "t3", family="claude-code", kwargs={"max_budget_usd": 1.0,
                                                                              "reasoning_effort": "medium"}),
        ]
        self.assertEqual(same, [base, base])

    def test_hash_changes_with_what_is_measured(self) -> None:
        tmp = self.tmpdir()
        base = self._hash(tmp / "a" / "x" / "t", kwargs={"reasoning_effort": "medium", "max_turns": 150})
        changed = {
            "effort": self._hash(tmp / "b" / "x" / "t", kwargs={"reasoning_effort": "high", "max_turns": 150}),
            "max_turns": self._hash(tmp / "c" / "x" / "t", kwargs={"reasoning_effort": "medium", "max_turns": 50}),
            "model": self._hash(tmp / "d" / "x" / "t", model_name="openrouter/deepseek/deepseek-v4.1-flash"),
            "import_path": self._hash(tmp / "e" / "x" / "t", import_path="bench.agents.pi:Custom"),
        }
        for what, h in changed.items():
            self.assertNotEqual(h, base, what)

    def test_identity_and_label(self) -> None:
        cfg = {"agent": {"kwargs": {"reasoning_effort": "medium", "max_turns": "150", "flag": False}},
               "timeout_multiplier": 1.0}
        h, identity, label = extract.config_identity(cfg, "pi", "builtin", "openrouter", "z-ai/glm-5.3")
        self.assertEqual(len(h), 16)
        self.assertEqual(identity["options"], {"flag": False, "max_turns": 150})
        self.assertEqual(identity["timeouts"], {})
        self.assertEqual(label, "pi · openrouter:z-ai/glm-5.3 · effort=medium · flag=false · max_turns=150")
        h2, identity2, label2 = extract.config_identity({**cfg, "timeout_multiplier": 2}, "pi", "x.y:Z",
                                                        "openrouter", "z-ai/glm-5.3")
        self.assertEqual(identity2["timeouts"], {"timeout_multiplier": 2})
        self.assertTrue(label2.startswith("pi* · "))
        self.assertNotEqual(h, h2)

    def test_import_path_sets_implementation_not_family(self) -> None:
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d, family="mini-swe-agent", import_path="bench.agents.mini_swe_agent:MiniSweAgentWithMcp")
        facts = extract.parse_trial(d, bench_ctx(), PREFIXES)
        assert facts is not None
        self.assertEqual(facts.family, "mini-swe-agent")
        self.assertEqual(facts.implementation, "bench.agents.mini_swe_agent:MiniSweAgentWithMcp")
        self.assertIn("mini-swe-agent* ·", facts.config_label)
        # A module:Class agent name counts as an import path too.
        d2 = self.tmpdir() / "job" / "t"
        lf.make_trial(d2, family="mini-swe-agent", agent_name="bench.agents.mini_swe_agent:MiniSweAgentWithMcp")
        facts2 = extract.parse_trial(d2, bench_ctx(), PREFIXES)
        assert facts2 is not None
        self.assertEqual(facts2.implementation, "bench.agents.mini_swe_agent:MiniSweAgentWithMcp")

    def test_family_prefers_the_runs_resolved_arm(self) -> None:
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d, family="mini-swe-agent")
        facts = extract.parse_trial(d, bench_ctx(family="mini-swe-agent-mcp"), PREFIXES)
        assert facts is not None
        self.assertEqual(facts.family, "mini-swe-agent-mcp")


class TaskIdentity(BenchTestCase):
    def test_digest_from_lock_else_dirhash(self) -> None:
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d, lock_digest="sha256:" + "b" * 64)
        facts = extract.parse_trial(d, bench_ctx(), PREFIXES)
        assert facts is not None
        self.assertEqual(facts.task_digest, "sha256:" + "b" * 64)
        d2 = self.tmpdir() / "job" / "t"
        lf.make_trial(d2, lock_digest=None, task_checksum="f" * 64)
        facts2 = extract.parse_trial(d2, bench_ctx(), PREFIXES)
        assert facts2 is not None
        self.assertEqual(facts2.task_digest, "dirhash:" + "f" * 64)

    def test_pilot_digests_and_snapshots(self) -> None:
        p1 = extract.parse_trial(pilot_trial(lf.PILOT_1, "strands-glm"), pilot_ctx(lf.PILOT_1, "strands-glm"),
                                 PREFIXES)
        p2 = extract.parse_trial(pilot_trial(lf.PILOT_2, "pi-glm"), pilot_ctx(lf.PILOT_2, "pi-glm"), PREFIXES)
        assert p1 is not None and p2 is not None
        self.assertTrue(p1.task_digest.startswith("sha256:e20725"))
        self.assertTrue(p2.task_digest.startswith("sha256:b4c601"))
        self.assertEqual(p1.dataset_snapshot, "git:69671fbaac6d")
        self.assertEqual(p2.dataset_snapshot, lf.PILOT_2_SNAPSHOT)
        self.assertEqual((p1.task_source_type, p2.task_source_type), ("git", "local"))
        self.assertEqual(p2.dataset, "terminal-bench@2.0")

    def test_package_and_plain_path_snapshots(self) -> None:
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d, task_type="package", lock_digest="sha256:" + "d" * 64)
        facts = extract.parse_trial(d, bench_ctx(), PREFIXES)
        assert facts is not None
        self.assertEqual(facts.dataset_snapshot, "pkg:org/task-a@v1")
        d2 = self.tmpdir() / "job" / "t"
        lf.make_trial(d2)
        facts2 = extract.parse_trial(d2, bench_ctx(), PREFIXES)
        assert facts2 is not None
        self.assertRegex(facts2.dataset_snapshot or "", r"^path:[0-9a-f]{8}$")


class TokensAndExceptions(BenchTestCase):
    def _facts(self, **kw: object) -> extract.TrialFacts:
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d, **kw)  # type: ignore[arg-type]
        facts = extract.parse_trial(d, bench_ctx(), PREFIXES)
        assert facts is not None
        return facts

    def test_none_stays_none_and_zero_stays_zero(self) -> None:
        none = self._facts(tokens=None)
        self.assertEqual((none.n_input, none.n_cache, none.n_output), (None, None, None))
        self.assertEqual(json.loads(none.usage_json), {})
        zero = self._facts(tokens=(0, 0, 0), cost=0.0)
        self.assertEqual((zero.n_input, zero.n_cache, zero.n_output), (0, 0, 0))
        self.assertEqual(zero.reported_cost_usd, 0.0)
        usage = json.loads(zero.usage_json)
        self.assertTrue(usage["z-ai/glm-5.3"]["synthesized"])

    def test_steps_are_summed_when_agent_result_is_null(self) -> None:
        steps = [
            {"step_name": "a", "agent_result": {"n_input_tokens": 10, "n_cache_tokens": None, "n_output_tokens": 3,
                                                "cost_usd": 0.5}},
            {"step_name": "b", "agent_result": {"n_input_tokens": 20, "n_cache_tokens": 5, "n_output_tokens": 4,
                                                "cost_usd": None},
             "exception_info": {"exception_type": "ApiRateLimitError", "exception_message": "429",
                                "exception_traceback": "", "occurred_at": "2026-10-29T05:40:00Z"}},
        ]
        facts = self._facts(steps=steps)
        self.assertEqual((facts.n_input, facts.n_cache, facts.n_output, facts.reported_cost_usd), (30, 5, 7, 0.5))
        # No trial-level exception: the first step's is used.
        self.assertEqual(facts.exception_type, "ApiRateLimitError")
        self.assertEqual(facts.harbor_cause, "model_provider_failed")

    def test_usage_is_canonicalised_and_primary_flagged(self) -> None:
        facts = self._facts(model_name="openrouter/deepseek/deepseek-v4.1-flash", tokens=(130, 50, 10), usage={
            "openrouter/deepseek/deepseek-v4.1-flash": {"n_input_tokens": 100, "n_cache_tokens": 50,
                                                        "n_output_tokens": 10, "cost_usd": 0.01},
            "z-ai/glm-5.3-flash": {"n_input_tokens": 30, "n_cache_tokens": 0, "n_output_tokens": 0},
        })
        usage = json.loads(facts.usage_json)
        self.assertTrue(usage["deepseek/deepseek-v4.1-flash"]["primary"])
        self.assertFalse(usage["z-ai/glm-5.3-flash"]["primary"])
        self.assertAlmostEqual(facts.primary_share, 110 / 140)

    def test_cap_marker_found_past_character_1400(self) -> None:
        message = "Command failed (exit 1): " + "x" * 1500 + ' {"subtype":"error_max_budget_usd"}'
        facts = self._facts(family="claude-code", exception_type="NonZeroAgentExitCodeError", message=message,
                            kwargs={"max_budget_usd": "1"})
        self.assertEqual(facts.cap_marker, "error_max_budget_usd")
        self.assertLessEqual(len(facts.exception_head or ""), extract.HEAD_CHARS)
        # The stored raw copy keeps the whole message.
        raw = extract.inflate(facts.raw_result_z)
        self.assertIn("error_max_budget_usd", raw["exception_info"]["exception_message"])
        self.assertLessEqual(len(raw["exception_info"]["exception_traceback"]), extract.TRACEBACK_KEEP)

    def test_cap_markers_only_count_for_their_family(self) -> None:
        facts = self._facts(family="pi", exception_type="NonZeroAgentExitCodeError", message="error_max_turns")
        self.assertIsNone(facts.cap_marker)

    def test_agent_started_false_for_empty_timing(self) -> None:
        facts = self._facts(agent_started=False, exception_type="NonZeroAgentExitCodeError", tokens=None)
        self.assertEqual(facts.agent_started, 0)
        self.assertIsNone(facts.agent_exec_sec)

    def test_times_are_utc_z_and_phases_measured(self) -> None:
        facts = self._facts(exec_sec=123.5)
        self.assertTrue(facts.started_at and facts.started_at.endswith("Z"))
        self.assertAlmostEqual(facts.agent_exec_sec or 0, 123.5)
        self.assertAlmostEqual(facts.total_sec or 0, 90 + 123.5)


class UsageAddsUp(BenchTestCase):
    """Per-model usage adds up to Harbor's totals: a call Harbor counted but
    attributed to no model is never dropped from the priced figures."""

    def setUp(self) -> None:
        super().setUp()
        prices = self.tmpdir() / "pricing.yaml"
        prices.write_text("anthropic:\n  claude-opus-5-5: {input: 4.00, output: 20.00, cache_read: 0.20}\n"
                          "  claude-haiku-5-5: {input: 0.10, output: 0.50, cache_read: 0.01}\n",
                          encoding="utf-8", newline="\n")
        patcher = mock.patch.object(settings, "PRICING", prices)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def sums(usage: dict) -> tuple[int, int, int]:
        return tuple(sum(u[k] or 0 for u in usage.values()) for k in ("n_input", "n_cache", "n_output"))  # type: ignore[return-value]

    def _facts(self, **kw: object) -> extract.TrialFacts:
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d, model_name="anthropic/claude-opus-5-5", **kw)  # type: ignore[arg-type]
        facts = extract.parse_trial(d, bench_ctx(provider="anthropic", prefix="anthropic/"), PREFIXES)
        assert facts is not None
        return facts

    def test_a_format_error_turn_counts_for_its_model(self) -> None:
        d = lf.make_format_error_trial(self.tmpdir() / "job" / "t")
        harbor = json.loads((d / "result.json").read_text(encoding="utf-8"))["agent_result"]
        self.assertEqual(harbor["model_usage"]["anthropic/claude-opus-5-5"]["n_input_tokens"], 4500)  # call 2 missing
        facts = extract.parse_trial(d, bench_ctx(provider="anthropic", prefix="anthropic/"), PREFIXES)
        assert facts is not None
        usage = json.loads(facts.usage_json)
        (entry,) = usage.values()
        self.assertEqual(list(usage), ["claude-opus-5-5"])
        self.assertEqual((entry["n_input"], entry["n_cache"], entry["n_output"]), (7500, 5400, 400))
        self.assertEqual(self.sums(usage), (facts.n_input, facts.n_cache, facts.n_output))
        self.assertAlmostEqual(entry["reported_cost_usd"], 0.04)
        self.assertTrue(entry["primary"])
        self.assertFalse(entry["synthesized"])
        gap = entry["unattributed"]
        self.assertEqual((gap["n_input"], gap["n_cache"], gap["n_output"]), (3000, 2400, 250))
        self.assertAlmostEqual(gap["cost_usd"], 0.04 * 250 / 400)
        # Priced on every call: 2100 fresh input, 5400 cache reads, 400 output.
        self.assertAlmostEqual(facts.cost_at_ingest_usd, (2100 * 4.00 + 5400 * 0.20 + 400 * 20.00) / 1e6)
        # The harness's own log, call by call, gives the same figures.
        where, calls = extract.recover_calls(d)  # type: ignore[misc]
        self.assertEqual((where, len(calls)), ("agent/mini-swe-agent.trajectory.json", 3))
        self.assertEqual(tuple(sum(c[i] for c in calls) for i in (1, 2, 3)), (7500, 5400, 400))

    def test_a_shortfall_across_models_goes_to_the_configured_model(self) -> None:
        opus = {"n_input_tokens": 100, "n_cache_tokens": 50, "n_output_tokens": 10, "cost_usd": 0.01}
        haiku = {"n_input_tokens": 30, "n_cache_tokens": 0, "n_output_tokens": 5, "cost_usd": 0.001}
        facts = self._facts(tokens=(200, 80, 20), cost=0.05, usage={"anthropic/claude-opus-5-5": opus,
                                                                     "claude-haiku-5-5-20261001": haiku})
        usage = json.loads(facts.usage_json)
        self.assertEqual(self.sums(usage), (200, 80, 20))
        self.assertEqual((usage["claude-opus-5-5"]["n_input"], usage["claude-opus-5-5"]["n_output"]), (170, 15))
        self.assertAlmostEqual(usage["claude-opus-5-5"]["reported_cost_usd"], 0.049)
        self.assertNotIn("unattributed", usage["claude-haiku-5-5-20261001"])
        # No reported model is the configured one: it gets an entry of its own.
        facts = self._facts(tokens=(200, 80, 20), usage={"claude-haiku-5-5-20261001": haiku,
                                                         "claude-sonnet-5-5": opus})
        usage = json.loads(facts.usage_json)
        self.assertEqual(self.sums(usage), (200, 80, 20))
        entry = usage["claude-opus-5-5"]
        self.assertEqual((entry["n_input"], entry["n_cache"], entry["n_output"]), (70, 30, 5))
        self.assertTrue(entry["primary"] and entry["synthesized"])
        self.assertNotIn("cost_usd", entry["unattributed"])  # Harbor reported no total cost
        # Reported under a dated name, the configured model still takes it: no second primary entry.
        usage = json.loads(self._facts(tokens=(200, 80, 20), usage={"anthropic/claude-opus-5-5-20261015": opus,
                                                                     "claude-haiku-5-5-20261001": haiku}).usage_json)
        self.assertEqual(sorted(usage), ["claude-haiku-5-5-20261001", "claude-opus-5-5-20261015"])
        self.assertEqual(usage["claude-opus-5-5-20261015"]["unattributed"], {"n_input": 70, "n_cache": 30, "n_output": 5})

    def test_a_lone_model_takes_the_shortfall_even_when_not_configured(self) -> None:
        haiku = {"n_input_tokens": 30, "n_cache_tokens": 0, "n_output_tokens": 5}
        usage = json.loads(self._facts(tokens=(50, 10, 8), usage={"claude-haiku-5-5-20261001": haiku}).usage_json)
        self.assertEqual(list(usage), ["claude-haiku-5-5-20261001"])  # no configured-model entry
        self.assertEqual(usage["claude-haiku-5-5-20261001"]["unattributed"], {"n_input": 20, "n_cache": 10, "n_output": 3})

    def test_a_cost_gap_needs_a_cost_for_every_model(self) -> None:
        opus = {"n_input_tokens": 100, "n_cache_tokens": 50, "n_output_tokens": 10, "cost_usd": 0.01}
        haiku = {"n_input_tokens": 30, "n_cache_tokens": 0, "n_output_tokens": 5}  # no cost of its own
        facts = self._facts(tokens=(200, 80, 20), cost=0.05, usage={"anthropic/claude-opus-5-5": opus,
                                                                     "claude-haiku-5-5-20261001": haiku})
        entry = json.loads(facts.usage_json)["claude-opus-5-5"]
        self.assertEqual(entry["unattributed"], {"n_input": 70, "n_cache": 30, "n_output": 5})
        self.assertAlmostEqual(entry["reported_cost_usd"], 0.01)
        # A cost that already adds up is no gap either.
        opus = {"n_input_tokens": 100, "n_cache_tokens": 0, "n_output_tokens": 10, "cost_usd": 0.05}
        facts = self._facts(tokens=(100, 60, 10), cost=0.05, usage={"anthropic/claude-opus-5-5": opus})
        self.assertEqual(json.loads(facts.usage_json)["claude-opus-5-5"]["unattributed"], {"n_cache": 60})

    def test_a_step_without_model_usage_still_counts(self) -> None:
        steps = [
            {"step_name": "a", "agent_result": {"n_input_tokens": 10, "n_cache_tokens": 0, "n_output_tokens": 3,
                                                "model_usage": {"anthropic/claude-opus-5-5": {
                                                    "n_input_tokens": 10, "n_cache_tokens": 0,
                                                    "n_output_tokens": 3}}}},
            {"step_name": "b", "agent_result": {"n_input_tokens": 20, "n_cache_tokens": 5, "n_output_tokens": 4,
                                                "model_usage": None}},
        ]
        usage = json.loads(self._facts(steps=steps).usage_json)
        self.assertEqual(self.sums(usage), (30, 5, 7))
        self.assertEqual(usage["claude-opus-5-5"]["unattributed"], {"n_input": 20, "n_cache": 5, "n_output": 4})

    def test_a_surplus_is_noted_not_subtracted(self) -> None:
        usage = json.loads(self._facts(tokens=(100, 50, 10), usage={"anthropic/claude-opus-5-5": {
            "n_input_tokens": 120, "n_cache_tokens": 50, "n_output_tokens": 10}}).usage_json)
        entry = usage["claude-opus-5-5"]
        self.assertEqual((entry["n_input"], entry["unattributed"]), (120, {"n_input": -20}))

    def test_usage_that_adds_up_is_left_as_reported(self) -> None:
        for trial in lf.fixture_trial_dirs():
            run_id, arm = trial.parent.parent.name, trial.parent.name
            facts = extract.parse_trial(trial, pilot_ctx(run_id, arm), PREFIXES)
            if facts is None:
                continue
            usage = json.loads(facts.usage_json)
            with self.subTest(arm=f"{run_id}/{arm}"):
                if usage:
                    self.assertEqual(self.sums(usage), tuple(v or 0 for v in (facts.n_input, facts.n_cache,
                                                                                 facts.n_output)))
                self.assertFalse(any("unattributed" in u for u in usage.values()))

    @unittest.skipUnless(OPUS_PILOT_MINI.is_dir(), "the Opus pilot's mini-swe-agent trial is not on this machine")
    def test_the_opus_pilot_mini_swe_agent_trial(self) -> None:
        # Read-only. Message 18 is a FormatError turn: 20558 in, 19995 cached, 1221 out.
        trial = OPUS_PILOT_MINI
        facts = extract.parse_trial(trial, bench_ctx(provider="anthropic", prefix="anthropic/"), PREFIXES)
        assert facts is not None
        usage = json.loads(facts.usage_json)
        totals = (facts.n_input, facts.n_cache, facts.n_output)
        self.assertEqual(self.sums(usage), totals)
        _, calls = extract.recover_calls(trial)  # type: ignore[misc]
        self.assertEqual(tuple(sum(c[i] for c in calls) for i in (1, 2, 3)), totals)
        gap = usage["claude-opus-5-5"]["unattributed"]
        self.assertEqual((gap["n_input"], gap["n_cache"], gap["n_output"]), (20558, 19995, 1221))


class Unreadable(BenchTestCase):
    def test_partial_unfinished_or_locked_result_is_none(self) -> None:
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d)
        result = (d / "result.json").read_text(encoding="utf-8")
        (d / "result.json").write_text(result[: len(result) // 2], encoding="utf-8")
        self.assertIsNone(extract.parse_trial(d, bench_ctx(), PREFIXES))
        data = json.loads(result)
        data["finished_at"] = None
        lf.write_json(d / "result.json", data)
        self.assertIsNone(extract.parse_trial(d, bench_ctx(), PREFIXES))
        lf.write_json(d / "result.json", json.loads(result))
        self.assertIsNotNone(extract.parse_trial(d, bench_ctx(), PREFIXES))
        real = Path.read_bytes

        def locked(self: Path) -> bytes:
            if self.name == "result.json":
                raise PermissionError(13, "The process cannot access the file")
            return real(self)

        with mock.patch.object(Path, "read_bytes", locked):
            self.assertIsNone(extract.parse_trial(d, bench_ctx(), PREFIXES))

    def test_parse_incomplete_on_pilot_1_pi_glm(self) -> None:
        trial = pilot_trial(lf.PILOT_1, "pi-glm")
        self.assertFalse((trial / "result.json").exists())
        ctx = pilot_ctx(lf.PILOT_1, "pi-glm")
        self.assertIsNone(extract.parse_trial(trial, ctx, PREFIXES))
        facts = extract.parse_incomplete(trial, ctx, PREFIXES)
        assert facts is not None
        self.assertEqual(facts.trial_key, "57542d33-e0ae-4bcf-9413-15e0582b136e/adaptive-rejection-sampler__XSHkG4A")
        self.assertEqual((facts.family, facts.provider, facts.model), ("pi", "openrouter", "z-ai/glm-5.3"))
        self.assertEqual(facts.task, "adaptive-rejection-sampler")
        self.assertTrue((facts.task_digest or "").startswith("sha256:e20725"))
        self.assertIsNone(facts.raw_result_z)
        self.assertIsNone(facts.finished_at)


class KilledTrials(BenchTestCase):
    """A trial killed with the harbor tree writes no result.json; the calls
    that completed are recovered from the harness's own log in agent/."""

    CALLS = ((1000, 600, 100), (3000, 2400, 250))

    def setUp(self) -> None:
        super().setUp()
        prices = self.tmpdir() / "pricing.yaml"
        prices.write_text("openrouter:\n  glm-5.3: {input: 0.07, output: 1.40, cache_read: 0.069}\n",
                          encoding="utf-8", newline="\n")
        patcher = mock.patch.object(settings, "PRICING", prices)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _killed(self, harness: str, **kw: object) -> extract.TrialFacts:
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d, family=harness, finished=False)
        lf.write_agent_log(d, harness, self.CALLS, **kw)  # type: ignore[arg-type]
        facts = extract.parse_incomplete(d, bench_ctx(family=harness), PREFIXES)
        assert facts is not None
        return facts

    def test_each_harness_log_gives_its_completed_calls(self) -> None:
        own_cost = {"pi": 0.002, "mini-swe-agent": 0.002, "terminus-2": 0.002, "claude-code": None, "strands": None}
        for harness, cost in own_cost.items():
            with self.subTest(harness):
                facts = self._killed(harness)
                self.assertEqual((facts.n_input, facts.n_cache, facts.n_output), (4000, 3000, 350))
                self.assertEqual(facts.agent_started, 1)
                self.assertIsNone(facts.finished_at)
                if cost is None:
                    self.assertIsNone(facts.reported_cost_usd)
                else:
                    self.assertAlmostEqual(facts.reported_cost_usd, cost)
                (entry,) = json.loads(facts.usage_json).values()
                self.assertTrue(entry["primary"] and entry["synthesized"])
                self.assertTrue(entry["recovered_from"].startswith("agent/"))
                # Priced like any trial: 1000 fresh input, 3000 cache reads, 350 output.
                self.assertAlmostEqual(facts.cost_at_ingest_usd, (1000 * 0.07 + 3000 * 0.069 + 350 * 1.40) / 1e6)
                f = dataclasses.asdict(facts)
                self.assertEqual(classify.classify(f)[0], "incomplete")
                self.assertEqual(classify.usage_status(f), "lower_bound")

    def test_a_started_log_without_a_completed_call_is_unmeasured(self) -> None:
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d, finished=False)
        lf.write_agent_log(d, "pi", calls=())
        self.assertEqual(extract.recover_calls(d), ("agent/pi/sessions", []))
        facts = extract.parse_incomplete(d, bench_ctx(), PREFIXES)
        assert facts is not None
        self.assertEqual((facts.n_input, facts.n_output, facts.usage_json, facts.agent_started), (None, None, "{}", 1))
        self.assertIsNone(facts.cost_at_ingest_usd)
        self.assertEqual(classify.usage_status(dataclasses.asdict(facts)), "unmeasured")

    def test_no_log_leaves_a_killed_trial_unmeasured(self) -> None:
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d, finished=False)
        (d / "agent" / "setup").mkdir(parents=True)  # pilot 1's pi-glm: killed before pi started
        self.assertIsNone(extract.recover_calls(d))
        facts = extract.parse_incomplete(d, bench_ctx(), PREFIXES)
        assert facts is not None
        self.assertEqual((facts.n_input, facts.usage_json, facts.agent_started), (None, "{}", 0))
        self.assertEqual(classify.usage_status(dataclasses.asdict(facts)), "unmeasured")

    def test_claude_code_keeps_each_models_usage(self) -> None:
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d, family="claude-code", model_name="anthropic/claude-opus-5-5", finished=False)
        log = lf.write_agent_log(d, "claude-code", self.CALLS, cut=False)
        sub = {"type": "assistant", "uuid": "sub-1", "message": {
            "id": "msg_sub", "model": "claude-haiku-5-5-20261001",
            "usage": {"input_tokens": 50, "cache_read_input_tokens": 0, "output_tokens": 5}}}
        lf._jsonl(log.parent / "s1" / "subagents" / "agent-1.jsonl", [sub], cut=False)
        facts = extract.parse_incomplete(d, bench_ctx(provider="anthropic", prefix="anthropic/"), PREFIXES)
        assert facts is not None
        usage = json.loads(facts.usage_json)
        self.assertEqual(sorted(usage), ["claude-haiku-5-5-20261001", "z-ai/glm-5.3"])
        self.assertEqual(usage["claude-haiku-5-5-20261001"]["n_input"], 50)
        self.assertEqual((facts.n_input, facts.n_output), (4050, 355))

    def test_terminus_2_counts_its_summarization_subagents_once(self) -> None:
        # Each episode's dump holds the main chat's totals only; the finally
        # block (a timeout or cancel runs it too) dumps them once more with the
        # subagents' totals added. Harbor's own total is the latter.
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d, family="terminus-2", finished=False)
        log = lf.write_agent_log(d, "terminus-2", self.CALLS)  # chat: 4000 in, 3000 cached, 350 out
        lf.write_json(log.parent / "trajectory.summarization-1-summary.json", {"final_metrics": {
            "total_prompt_tokens": 300, "total_cached_tokens": 0, "total_completion_tokens": 30}})
        lf.write_json(log.parent / "trajectory.summarization-1-answers.json", {"final_metrics": {
            "total_prompt_tokens": 200, "total_cached_tokens": 100, "total_completion_tokens": 20}})

        def totals() -> tuple[int, int, int]:
            where, calls = extract.recover_calls(d)  # type: ignore[misc]
            self.assertEqual(where, "agent/trajectory.json")
            return tuple(sum(c[i] for c in calls) for i in (1, 2, 3))  # type: ignore[return-value]

        self.assertEqual(totals(), (4500, 3100, 400))  # killed mid-loop: the subagents are added
        trajectory = json.loads(log.read_text(encoding="utf-8"))
        for in_flight in ((0, 0, 0), (900, 0, 60)):  # the call the timeout cut off may have completed first
            with self.subTest(in_flight=in_flight):
                trajectory["final_metrics"] = {"total_prompt_tokens": 4000 + in_flight[0] + 500,
                                               "total_cached_tokens": 3000 + 100,
                                               "total_completion_tokens": 350 + in_flight[2] + 50}
                lf.write_json(log, trajectory)
                self.assertEqual(totals(), (4500 + in_flight[0], 3100, 400 + in_flight[2]))
        # Without per-step metrics the two dumps cannot be told apart: the
        # subagents are left out, an undercount rather than a double count.
        trajectory["steps"] = []
        trajectory["final_metrics"] = {"total_prompt_tokens": 4000, "total_cached_tokens": 3000,
                                       "total_completion_tokens": 350}
        lf.write_json(log, trajectory)
        self.assertEqual(totals(), (4000, 3000, 350))

    def test_odd_step_results_read_as_none(self) -> None:
        # A result.json that parses can hold anything; the live view reads
        # judged_facts on every poll, so it must not raise.
        for steps in (5, "x", {"a": 1}, [None, 3, {"agent_execution": {"started_at": "2026-10-29T06:00:00Z"}}]):
            with self.subTest(steps=steps):
                facts = extract.judged_facts({"step_results": steps, "agent_result": None})
                self.assertEqual(facts["agent_started"], int(isinstance(steps, list)))
                self.assertIsNone(facts["n_input"])
                self.assertIsNone(facts["exception_type"])

    @unittest.skipUnless(PILOT_3_JOBS.is_dir(), "the dev pilot 3 job folders are not on this machine")
    def test_recipes_reproduce_harbors_totals_on_pilot_3(self) -> None:
        # Read-only: every finished pilot-3 trial's log, against what Harbor's own
        # post-run step recorded from it. mini-swe-agent's cost is OpenRouter's bill,
        # which Harbor records as 0.0, so costs are not compared.
        checked = 0
        for trial in sorted(PILOT_3_JOBS.glob("*/*__*")):
            result = extract.read_json(trial / "result.json")
            found = extract.recover_calls(trial)
            if not isinstance(result, dict) or found is None:
                continue
            harbor = result["agent_result"]
            sums = tuple(sum(c[i] for c in found[1]) for i in (1, 2, 3))
            with self.subTest(trial.parent.name):
                self.assertEqual(sums, (harbor["n_input_tokens"], harbor["n_cache_tokens"], harbor["n_output_tokens"]))
            checked += 1
        self.assertGreaterEqual(checked, 10)


class CacheWrites(BenchTestCase):
    """Harbor folds cache writes into n_input and result.json keeps no count
    of them, so each harness's own log gives them, per model, and they are
    priced at their own rates; what only the log can see becomes a flag."""

    # (fresh input, cache reads, 5-minute writes, 1-hour writes, output) per call
    CALLS = ((2, 0, 3000, 0, 400), (3, 2900, 500, 100, 250), (4, 3500, 200, 0, 90))

    def setUp(self) -> None:
        super().setUp()
        prices = self.tmpdir() / "pricing.yaml"
        prices.write_text(lf.OPUS_PRICING_YAML, encoding="utf-8", newline="\n")
        patcher = mock.patch.object(settings, "PRICING", prices)
        patcher.start()
        self.addCleanup(patcher.stop)

    def parse(self, harness: str, calls: tuple = CALLS, **kw: object) -> tuple[extract.TrialFacts, dict, dict]:
        d = self.tmpdir() / "job" / "t"
        lf.make_cache_trial(d, harness, calls, **kw)  # type: ignore[arg-type]
        return self.facts(d)

    def facts(self, d: Path) -> tuple[extract.TrialFacts, dict, dict]:
        facts = extract.parse_trial(d, bench_ctx(provider="anthropic", prefix="anthropic/"), PREFIXES)
        assert facts is not None
        return facts, json.loads(facts.usage_json), json.loads(facts.flags_json)

    def exact(self, calls: tuple = CALLS) -> float:
        return sum(lf.call_cost(c) for c in calls)

    def test_every_harness_log_gives_its_writes_and_the_exact_cost(self) -> None:
        for harness in ("claude-code", "pi", "mini-swe-agent"):
            with self.subTest(harness):
                facts, usage, flags = self.parse(harness)
                (entry,) = usage.values()
                self.assertEqual((entry["n_cache_write_5m"], entry["n_cache_write_1h"]), (3700, 100))
                self.assertAlmostEqual(facts.cost_at_ingest_usd, self.exact(), places=12)
                self.assertEqual(flags, {})  # the harness's own figure agrees, and nothing is missing
        # strands keeps the write total only; its cache config says 5-minute, the default.
        calls = tuple((f, r, w5 + w1, 0, o) for f, r, w5, w1, o in self.CALLS)
        facts, usage, flags = self.parse("strands", calls)
        self.assertEqual((usage["claude-opus-5-5"]["n_cache_write_5m"], usage["claude-opus-5-5"]["n_cache_write_1h"]),
                         (3800, 0))
        self.assertAlmostEqual(facts.cost_at_ingest_usd, self.exact(calls), places=12)
        self.assertEqual(flags, {})

    def test_writes_are_priced_at_their_own_rates_not_as_fresh_input(self) -> None:
        facts, _usage, _flags = self.parse("claude-code")
        as_input = sum(((f + w5 + w1) * 4.0 + r * 0.2 + o * 20.0) / 1e6 for f, r, w5, w1, o in self.CALLS)
        self.assertAlmostEqual(facts.cost_at_ingest_usd - as_input, (3700 * 1.0 + 100 * 4.0) / 1e6, places=12)

    def test_strands_one_hour_cache_config_makes_its_writes_one_hour(self) -> None:
        d = self.tmpdir() / "job" / "t"
        calls = ((2, 0, 3000, 0, 400),)
        lf.make_cache_trial(d, "strands", calls)
        lf.write_json(d / "agent" / "strands-result.json", {"observed_model": {"config": {
            "cache_config": "CacheConfig(strategy='auto', ttl='1h', system_prompt_ttl=True)"}}})
        _facts, usage, _flags = self.facts(d)
        self.assertEqual((usage["claude-opus-5-5"]["n_cache_write_5m"], usage["claude-opus-5-5"]["n_cache_write_1h"]),
                         (0, 3000))

    def test_claude_code_declined_attempts_are_a_flag_with_their_cost(self) -> None:
        # The first request ran two attempts: Opus 5.5 declined, then Opus 5 answered. Only usage.iterations
        # lists the declined one; Claude Code's own total counts a refused request the log never shows.
        d = self.tmpdir() / "job" / "t"
        lf.make_cache_trial(d, "claude-code", self.CALLS, model="claude-opus-5", reported=0.5)
        log = d / "agent" / "sessions" / "projects" / "-app" / "s1.jsonl"
        events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        declined = {"input_tokens": 2, "output_tokens": 0, "cache_read_input_tokens": 12635,
                    "cache_creation_input_tokens": 4699, "type": "message", "model": "claude-opus-5-5",
                    "cache_creation": {"ephemeral_5m_input_tokens": 4699, "ephemeral_1h_input_tokens": 0}}
        for e in events[:2]:
            e["message"]["usage"]["iterations"] = [declined, {"type": "fallback_message", "model": "claude-opus-5"}]
        lf._jsonl(log, events, cut=False)
        facts, usage, flags = self.facts(d)
        self.assertEqual(list(usage), ["claude-opus-5"])  # the cost stays the log's calls at Opus 5 rates
        opus_5 = {"input": 5.0, "read": 0.5, "write_5m": 6.25, "write_1h": 10.0, "output": 25.0}
        self.assertAlmostEqual(facts.cost_at_ingest_usd, sum(lf.call_cost(c, opus_5) for c in self.CALLS), places=12)
        attempt = flags["declined_attempts"]
        self.assertEqual(attempt["attempts"], 1)  # logged twice, counted once
        self.assertEqual(attempt["models"]["claude-opus-5-5"], {"n_input": 17336, "n_cache": 12635, "n_output": 0,
                                                                 "n_cache_write_5m": 4699, "n_cache_write_1h": 0})
        self.assertAlmostEqual(attempt["cost_usd_max"], (2 * 4.0 + 12635 * 0.2 + 4699 * 5.0) / 1e6)
        mismatch = flags["reported_cost_mismatch"]
        self.assertEqual(mismatch["reported_usd"], 0.5)
        self.assertAlmostEqual(mismatch["computed_usd"], facts.cost_at_ingest_usd)

    def test_a_reported_cost_within_half_a_percent_is_no_flag(self) -> None:
        for factor, flagged in ((1.004, False), (0.996, False), (1.006, True), (0.99, True)):
            with self.subTest(factor=factor):
                _facts, _usage, flags = self.parse("pi", reported=self.exact() * factor)
                self.assertEqual("reported_cost_mismatch" in flags, flagged)

    def test_mini_swe_agent_counts_format_error_writes_and_flags_unsaved_calls(self) -> None:
        d = self.tmpdir() / "job" / "t"
        lf.make_cache_trial(d, "mini-swe-agent", self.CALLS)
        trajectory = json.loads((d / "agent" / "mini-swe-agent.trajectory.json").read_text(encoding="utf-8"))
        trajectory["messages"][3]["role"] = "user"  # a FormatError reply is logged on the 'user' message
        lf.write_json(d / "agent" / "mini-swe-agent.trajectory.json", trajectory)
        steps = "".join(f"mini-swe-agent (step {n}, ${c:.2f})\n" for n, c in ((1, 0.03), (2, 0.04), (3, 0.05)))
        (d / "agent" / "mini-swe-agent.txt").write_text(steps, encoding="utf-8")
        _facts, usage, flags = self.facts(d)
        self.assertEqual(usage["claude-opus-5-5"]["n_cache_write_5m"], 3700)
        self.assertEqual(flags, {})  # three printed steps, three saved calls
        # A timeout during the fourth step's command: its reply was printed, never saved.
        (d / "agent" / "mini-swe-agent.txt").write_text(steps + "mini-swe-agent (step 4, $0.06)\n", encoding="utf-8")
        _facts, _usage, flags = self.facts(d)
        self.assertEqual(flags["unsaved_calls"]["calls"], 1)
        self.assertAlmostEqual(flags["unsaved_calls"]["cost_usd_max"], 0.065 - self.exact())

    def test_pi_flags_a_call_started_and_never_finished(self) -> None:
        d = self.tmpdir() / "job" / "t"
        lf.make_cache_trial(d, "pi", self.CALLS)
        start = '{"type":"message_start","message":{"role":"assistant","content":[]}}\n'
        end = '{"type":"message_end","message":{"role":"assistant","usage":{}}}\n'
        (d / "agent" / "pi.txt").write_text(start + end + start + end + start, encoding="utf-8")
        _facts, _usage, flags = self.facts(d)
        self.assertEqual(flags["unfinished_calls"]["calls"], 1)
        self.assertAlmostEqual(flags["unfinished_calls"]["cost_usd_max"], max(lf.call_cost(c) for c in self.CALLS))

    def test_terminus_2_recovers_writes_from_each_calls_cost(self) -> None:
        # terminus-2 keeps no write count; LiteLLM's per-call cost at these rates gives it back, whole tokens.
        d = self.tmpdir() / "job" / "t"
        lf.make_cache_trial(d, "terminus-2", ((2, 0, 3000, 0, 400), (3, 2900, 600, 0, 250), (4, 3500, 200, 0, 90)))
        trajectory = json.loads((d / "agent" / "trajectory.json").read_text(encoding="utf-8"))
        trajectory["steps"].pop()  # the last call finished but never became a step: only the totals hold it
        lf.write_json(d / "agent" / "trajectory.json", trajectory)
        facts, usage, flags = self.facts(d)
        self.assertEqual((usage["claude-opus-5-5"]["n_cache_write_5m"], usage["claude-opus-5-5"]["n_cache_write_1h"]),
                         (3800, 0))  # 1-hour writes cannot be told apart from a cost: all read as 5-minute
        self.assertEqual(flags, {})
        # A cost at other rates does not come out a whole number of tokens: those writes stay unknown.
        for step in trajectory["steps"][1:]:
            step["metrics"]["cost_usd"] *= 1.01
        lf.write_json(d / "agent" / "trajectory.json", trajectory)
        _facts, usage, flags = self.facts(d)
        self.assertNotIn("n_cache_write_5m", usage["claude-opus-5-5"])
        self.assertEqual(flags["cache_writes_unknown"]["calls"], 3)  # the step-less call's cost is off too

    def test_no_log_leaves_anthropic_writes_unknown_and_open_models_unflagged(self) -> None:
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d, model_name="anthropic/claude-opus-5-5", tokens=(10000, 6000, 500))
        facts, usage, flags = self.facts(d)
        self.assertNotIn("n_cache_write_5m", usage["claude-opus-5-5"])
        self.assertEqual(flags["cache_writes_unknown"], {"calls": 0, "models": ["claude-opus-5-5"],
                                                         "reason": "no harness log"})
        self.assertAlmostEqual(facts.cost_at_ingest_usd, (4000 * 4.0 + 6000 * 0.2 + 500 * 20.0) / 1e6)
        d = self.tmpdir() / "job2" / "t"
        lf.make_trial(d, tokens=(10000, 6000, 500))  # GLM on OpenRouter: no write premium to miss
        facts = extract.parse_trial(d, bench_ctx(), PREFIXES)
        assert facts is not None
        self.assertEqual(json.loads(facts.flags_json), {})

    def test_killed_trials_keep_their_writes_too(self) -> None:
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d, family="pi", model_name="anthropic/claude-opus-5-5", finished=False)
        lf.write_cache_log(d, "pi", self.CALLS)
        facts = extract.parse_incomplete(d, bench_ctx(provider="anthropic", prefix="anthropic/"), PREFIXES)
        assert facts is not None
        (entry,) = json.loads(facts.usage_json).values()
        self.assertEqual((entry["n_cache_write_5m"], entry["n_cache_write_1h"]), (3700, 100))
        self.assertAlmostEqual(facts.cost_at_ingest_usd, self.exact(), places=12)
        self.assertAlmostEqual(facts.reported_cost_usd, self.exact(), places=12)


class Secrets(BenchTestCase):
    def test_redact_masks_loaded_secrets_and_patterns(self) -> None:
        secret = "zq9-very-private-value-1234"
        fireworks, aws = "fw_" + "3Kq9" * 6, "AKIA" + "ABCDEFGHIJ234567"
        with mock.patch.dict(os.environ, {"TEST_API_KEY": secret}):
            text = extract.redact(f"key={secret}; sk-test1234567890abcdef; Authorization: Bearer abc.def; "
                                  f"OPENROUTER_API_KEY=xyz123; key '{fireworks}' and id {aws} rejected")
        self.assertNotIn(secret, text)
        self.assertNotIn("sk-test1234567890abcdef", text)
        self.assertNotIn("abc.def", text)
        self.assertNotIn("xyz123", text)
        self.assertIn("OPENROUTER_API_KEY=[redacted]", text)
        # Shapes the CLI must catch without the keys loaded (it never reads .env).
        self.assertNotIn(fireworks, text)
        self.assertNotIn(aws, text)
        self.assertIn("key 'fw_[redacted]' and id AKIA[redacted] rejected", text)
        # Ordinary words that merely start alike stay readable.
        self.assertEqual(extract.redact("fw_short and AKIAlower and fireworks_ai/x"),
                         "fw_short and AKIAlower and fireworks_ai/x")

    def test_option_values_are_redacted_wherever_they_are_stored(self) -> None:
        secret = "zq9-very-private-option-5678"
        d = self.tmpdir() / "job" / "t"
        with mock.patch.dict(os.environ, {"TEST_API_KEY": secret}):
            lf.make_trial(d, kwargs={"reasoning_effort": "medium", "api_key": secret,
                                     "headers": {"Authorization": "Bearer abc.def"}, "max_turns": 150})
            facts = extract.parse_trial(d, bench_ctx(), PREFIXES)
            _, identity, label = extract.config_identity(
                {"agent": {"kwargs": {}, "base_url": "https://proxy.example/v1?key=sk-abcdefghijklmnop"}},
                "pi", "builtin", "openrouter", "z-ai/glm-5.3")
        assert facts is not None
        for text in (facts.options_json, facts.config_json, facts.config_label):
            self.assertNotIn(secret, text)
            self.assertNotIn("abc.def", text)
        options = json.loads(facts.options_json)
        self.assertEqual(options["api_key"], "[redacted]")
        self.assertEqual(options["headers"], {"Authorization": "Bearer [redacted]"})
        self.assertEqual(options["max_turns"], 150)
        self.assertEqual(identity["extras"]["base_url"], "https://proxy.example/v1?key=sk-[redacted]")
        self.assertNotIn("sk-abcdefghijklmnop", label)

    def test_trim_result_drops_env_and_rollouts(self) -> None:
        d = self.tmpdir() / "job" / "t"
        lf.make_trial(d, env={"SOME_TOKEN": "not-for-the-ledger"})
        result = json.loads((d / "result.json").read_text(encoding="utf-8"))
        self.assertIn("env", result["config"]["agent"])
        trimmed = extract.trim_result(result)
        text = json.dumps(trimmed)
        self.assertNotIn('"env"', text)
        self.assertNotIn("rollout_details", text)
        self.assertNotIn("not-for-the-ledger", text)


class Fixtures(unittest.TestCase):
    def test_pilot_fixtures_hold_no_secrets(self) -> None:
        files = sorted(lf.FIXTURE_DEV.rglob("*.json"))
        self.assertEqual(len(files), 98)
        for path in files:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(lf.FIXTURE_DEV).as_posix()):
                self.assertNotRegex(text, r"sk-[A-Za-z0-9_-]{6,}")
                self.assertNotIn("Bearer ", text)
                self.assertNotRegex(text, r'"env"\s*:')
                self.assertNotIn("rollout_details", text)
                # Long mixed-case tokens are the shape of API keys. The only
                # ones here are trial names and Claude Code field names.
                for token in re.findall(r"[A-Za-z0-9_\-]{32,}", text):
                    if re.search(r"[A-Z]", token) and re.search(r"[a-z]", token) and re.search(r"[0-9]", token):
                        self.assertRegex(token, r"^([a-z0-9-]+__[A-Za-z0-9]{7}(__agent)?|cacheCreation\w+)$")


class Paths(BenchTestCase):
    def test_trial_dir_is_relative_with_forward_slashes(self) -> None:
        tmp = self.tmpdir()
        d = tmp / "jobs" / "run" / "arm" / "t"
        lf.make_trial(d)
        facts = extract.parse_trial(d, bench_ctx(), PREFIXES, root=tmp)
        assert facts is not None
        self.assertEqual(facts.trial_dir, "jobs/run/arm/t")
        outside = extract.parse_trial(d, bench_ctx(), PREFIXES, root=tmp / "elsewhere")
        assert outside is not None
        self.assertNotIn("\\", outside.trial_dir)
        self.assertTrue(outside.trial_dir.endswith("jobs/run/arm/t"))

    def test_discover_bench_and_external_layouts(self) -> None:
        tmp = self.tmpdir()
        default, dev, ext = tmp / "root", tmp / "profiles" / "dev", tmp / "ext"
        arms = [{"name": "pi-glm", "agent": "pi", "provider": "openrouter", "model": "z-ai/glm-5.3"}]
        lf.make_bench_run(default, "r-default", arms, status="running")
        lf.make_bench_run(dev, "r-dev", arms)
        lf.make_external_job(ext, "2026-10-29__10-00-00")
        (ext / "not-a-job" / "deeper").mkdir(parents=True)
        lf.make_external_job(default / "jobs", "plain-harbor-job", finished=False)
        roots = [ScanRoot(default / "jobs", "default", default / "runs"), ScanRoot(dev / "jobs", "dev", dev / "runs"),
                 ScanRoot(ext, "external", None)]
        skipped: list[dict[str, str]] = []
        found = {(ctx.run_key, ctx.arm, terminal) for _, ctx, _, terminal in extract.discover(roots, skipped,
                                                                                          root=tmp)}
        self.assertEqual(found, {
            ("default/r-default", "pi-glm", False),
            ("dev/r-dev", "pi-glm", True),
            ("external/ext/2026-10-29__10-00-00", "2026-10-29__10-00-00", True),
            ("external/root/jobs/plain-harbor-job", "plain-harbor-job", False),
        })
        self.assertEqual(skipped, [])

    def test_context_for_old_and_new_run_json(self) -> None:
        old = pilot_ctx(lf.PILOT_2, "claude-code-glm")
        self.assertEqual((old.run_key, old.provider, old.model_prefix, old.effort_label, old.family),
                         (f"dev/{lf.PILOT_2}", "openrouter", "openrouter/", "medium", None))
        self.assertEqual(old.local_snapshot, lf.PILOT_2_SNAPSHOT)
        self.assertIsNone(pilot_ctx(lf.PILOT_1, "pi-glm").local_snapshot)
        run = lf.run_dict("r1", [{"name": "msa", "agent": "bench.agents.mini_swe_agent:MiniSweAgentWithMcp",
                                  "family": "mini-swe-agent", "provider": "anthropic", "model": "claude-opus-5-5"}])
        new = extract.context_for_bench_arm(run, "msa", "default")
        self.assertEqual((new.provider, new.model_prefix, new.family, new.effort_label, new.dataset),
                         ("anthropic", "anthropic/", "mini-swe-agent", "medium", "terminal-bench@2.0"))

    def test_unreadable_run_json_is_reported_not_fatal(self) -> None:
        tmp = self.tmpdir()
        data = tmp / "dev"
        lf.make_bench_run(data, "good", [{"name": "a", "agent": "pi", "provider": "openrouter", "model": "m"}])
        lf.make_bench_run(data, "bad", [{"name": "a", "agent": "pi", "provider": "openrouter", "model": "m"}])
        (data / "runs" / "bad" / "run.json").write_text("{not json", encoding="utf-8")
        skipped: list[dict[str, str]] = []
        found = [ctx.run_key for _, ctx, _, _ in extract.discover([ScanRoot(data / "jobs", "dev", data / "runs")],
                                                                  skipped)]
        self.assertEqual(found, ["dev/good"])
        self.assertEqual(len(skipped), 1)
        self.assertIn("run.json", skipped[0]["reason"])

    def test_odd_run_json_shapes_cost_only_that_run(self) -> None:
        tmp = self.tmpdir()
        data = tmp / "dev"
        arms = [{"name": "a", "agent": "pi", "provider": "openrouter", "model": "m"}]
        lf.make_bench_run(data, "aaa-odd", arms)
        lf.make_bench_run(data, "good", arms)
        path = data / "runs" / "aaa-odd" / "run.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        run["config"]["providers"] = {"openrouter": "not a mapping"}
        run["resolved_arms"] = 7
        path.write_text(json.dumps(run), encoding="utf-8", newline="\n")
        skipped: list[dict[str, str]] = []
        roots = [ScanRoot(data / "jobs", "dev", data / "runs")]
        found = [ctx.run_key for _, ctx, _, _ in extract.discover(roots, skipped)]
        self.assertIn("dev/good", found)
        with mock.patch.object(extract, "context_for_bench_arm", side_effect=AttributeError("odd")):
            skipped = []
            self.assertEqual(list(extract.discover(roots, skipped)), [])
        self.assertEqual(len(skipped), 2)
        self.assertIn("unexpected shape", skipped[0]["reason"])

    def test_default_roots(self) -> None:
        from bench import settings
        extra = self.tmpdir()
        roots = extract.default_roots([extra])
        self.assertEqual(roots[0], ScanRoot(settings.ROOT / "jobs", "default", settings.ROOT / "runs"))
        by_profile = {r.profile: r for r in roots}
        if (settings.ROOT / "profiles" / "dev").is_dir():
            self.assertEqual(by_profile["dev"].jobs_dir, settings.ROOT / "profiles" / "dev" / "jobs")
        # Tests keep run data in a temp folder; this process's own data is always scanned.
        self.assertIn(Path(settings.DATA_DIR) / "jobs", [r.jobs_dir for r in roots])
        self.assertEqual(roots[-1], ScanRoot(extra, "external", None))

    def test_provider_prefixes_skip_non_bench_configs(self) -> None:
        d = self.tmpdir()
        (d / "bench.yaml").write_text("providers:\n  fireworks: {env_key: K, model_prefix: 'fw/'}\n", encoding="utf-8")
        (d / "pricing.yaml").write_text("providers:\n  bogus: {model_prefix: 'bogus/'}\n", encoding="utf-8")
        (d / "dev.overrides.yaml").write_text("providers:\n  other: {model_prefix: 'o/'}\n", encoding="utf-8")
        self.assertEqual(extract.provider_prefixes(d), {"fw/": "fireworks"})


# Identity guard ------------------------------------------------------------------
# A trial's identity must not move when it runs on a pre-baked image or with a
# setup safety net. tests/fixtures/identity holds trials Harbor wrote: stock/
# the 2026-10-29 Opus pilot's five arms (and fix-git from the 27-task run),
# prebaked/ the research prototype's install_only trials of the same arms on
# pre-baked fix-git images (environment.import_path and kwargs set).

IDENTITY = FIXTURES / "identity"
ARM_HASHES = {"claude-code": "c701410eb884a0d9", "terminus-2": "a06ee084038fa5e2",
              "mini-swe-agent": "e6d727483679d869", "pi": "dd7f23a86e387f25", "strands": "9c7311ba1b25de57"}


def _identity_ctx(arm: str) -> RunContext:
    return RunContext(run_key="default/r1", profile="default", run_id="r1", source="bench", arm=arm,
                      provider="anthropic", model_prefix="anthropic/", effort_label="medium",
                      dataset="terminal-bench@2.0", local_snapshot="local:terminal-bench@2.0@2026-10-29T05:29:58+00:00",
                      family=arm)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class IdentityGuard(BenchTestCase):
    def hash_of(self, trial_cfg: dict, arm: str) -> str:
        return extract.config_identity(trial_cfg, arm, extract._implementation(trial_cfg["agent"]), "anthropic",
                                       "claude-opus-5-5")[0]

    def facts(self, folder: Path, arm: str) -> extract.TrialFacts:
        return extract.facts_from_result(_read(folder / "result.json"), None, None, trial_dir=folder.name,
                                         ctx=_identity_ctx(arm), prefixes=PREFIXES)

    def test_the_five_arm_hashes_on_real_stock_trials(self) -> None:
        for arm, expected in ARM_HASHES.items():
            folder = IDENTITY / "stock" / arm
            with self.subTest(arm=arm):
                self.assertEqual(self.facts(folder, arm).config_hash, expected)               # as ingested
                self.assertEqual(self.hash_of(_read(folder / "config.json"), arm), expected)  # a killed trial
        fix_git = IDENTITY / "stock" / "claude-code-fix-git"
        self.assertEqual(self.facts(fix_git, "claude-code").config_hash, ARM_HASHES["claude-code"])

    def test_harbor_written_prebaked_trials_keep_the_hash_group_and_task(self) -> None:
        o = report.parse_options(None, self.load_shipped_config("bench"))
        for arm, expected in ARM_HASHES.items():
            with self.subTest(arm=arm):
                folder = IDENTITY / "prebaked" / arm
                self.assertTrue(_read(folder / "config.json")["environment"]["import_path"])
                self.assertTrue(_read(folder / "prebake.json")["used"])
                prebaked = self.facts(folder, arm)
                stock = self.facts(IDENTITY / "stock" / arm, arm)
                self.assertEqual(prebaked.config_hash, expected)
                for column in ("config_json", "config_label", "family", "implementation", "provider", "model",
                               "effort", "options_json", "cautions_json", "dataset_snapshot"):
                    self.assertEqual(getattr(prebaked, column), getattr(stock, column), column)
                self.assertEqual(report.group_key(dataclasses.asdict(prebaked), o),
                                 report.group_key(dataclasses.asdict(stock), o))
        stock_fix_git = self.facts(IDENTITY / "stock" / "claude-code-fix-git", "claude-code")
        prebaked_fix_git = self.facts(IDENTITY / "prebaked" / "claude-code", "claude-code")
        self.assertEqual((prebaked_fix_git.task, prebaked_fix_git.task_digest),
                         ("fix-git", stock_fix_git.task_digest))

    def test_environment_fields_never_change_the_hash(self) -> None:
        stock = _read(IDENTITY / "stock" / "claude-code-fix-git" / "config.json")
        environments = [
            {"type": None, "import_path": "bench.environments.prebaked:PrebakedDockerEnvironment",
             "kwargs": {"prebake_harness": "claude-code", "prebake_version": "2.1.312", "prebake_recipe": "a" * 64,
                        "prebake_prefix": "benchbake", "prebake_required": True}},
            {"type": "docker", "import_path": None, "kwargs": {}, "extra_docker_compose": []},
            {"type": "docker", "extra_docker_compose": ["overlay.yaml"], "force_build": False, "delete": True},
        ]
        for environment in environments:
            with self.subTest(environment=environment):
                self.assertEqual(self.hash_of({**stock, "environment": environment}, "claude-code"),
                                 ARM_HASHES["claude-code"])
        # The resource overrides still count: they change what the agent has to work with.
        self.assertNotEqual(self.hash_of({**stock, "environment": {"override_cpus": 4}}, "claude-code"),
                            ARM_HASHES["claude-code"])

    def test_setup_timeout_keys_never_change_the_hash(self) -> None:
        stock = _read(IDENTITY / "stock" / "claude-code-fix-git" / "config.json")
        agent = stock["agent"]
        for changed in ({**stock, "agent_setup_timeout_multiplier": 2.0},
                        {**stock, "agent_setup_timeout_multiplier": 1.0},
                        {**stock, "agent": {**agent, "override_setup_timeout_sec": 720}},
                        {**stock, "agent_setup_timeout_multiplier": 2, "environment_build_timeout_multiplier": 1.5}):
            with self.subTest(added={k: v for k, v in changed.items() if k not in stock}):
                h, identity, label = extract.config_identity(changed, "claude-code", "builtin", "anthropic",
                                                             "claude-opus-5-5")
                self.assertEqual(h, ARM_HASHES["claude-code"])
                self.assertEqual(identity["timeouts"], {})
                self.assertNotIn("setup", label)
        # Limits that can change a trial that ran still count.
        for changed in ({**stock, "agent_timeout_multiplier": 2.0}, {**stock, "timeout_multiplier": 2.0},
                        {**stock, "agent": {**agent, "override_timeout_sec": 600}}):
            self.assertNotEqual(self.hash_of(changed, "claude-code"), ARM_HASHES["claude-code"])

    def test_config_hash_for_still_predicts_the_arms(self) -> None:
        cfg = self.load_shipped_config("bench")
        run_spec = spec.spec_from_config(cfg, "pilot", list(ARM_HASHES))
        for arm in run_spec.arms:
            with self.subTest(arm=arm.name):
                kwargs, _ = runs.agent_kwargs(cfg.run, arm, fixture_schema(arm.agent))
                self.assertEqual(estimates.config_hash_for(arm, kwargs, cfg.run.timeout_multiplier),
                                 ARM_HASHES[arm.family])

    def test_the_identity_names_no_setup_mode_or_environment(self) -> None:
        facts = self.facts(IDENTITY / "prebaked" / "strands", "strands")
        identity = json.loads(facts.config_json)
        self.assertEqual(sorted(identity), ["effort", "extras", "family", "implementation", "model", "options",
                                            "provider", "timeouts"])
        self.assertNotIn("prebake", facts.config_json)
        self.assertNotIn("environment", facts.config_json)
        from bench import reuse

        for fn in (extract.config_identity, extract.task_identity, report.group_key, report._pair_key):
            source = inspect.getsource(fn)
            self.assertNotIn("setup_mode", source, fn.__name__)
            self.assertNotIn("prebake_", source, fn.__name__)
        self.assertNotIn("setup_mode", inspect.getsource(reuse))

    def test_identity_fixtures_hold_no_secrets(self) -> None:
        files = sorted(IDENTITY.rglob("*.json"))
        self.assertEqual(len(files), 27)
        for path in files:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(IDENTITY).as_posix()):
                self.assertEqual(extract.redact(text), text)
                self.assertNotRegex(text, r'"env"\s*:\s*\{\s*"')


# Setup provenance ------------------------------------------------------------------

BAKED = "benchbake/claude-code-2.1.312-r0123456789/alexgshaw/fix-git:20251031"
PREBAKED_ENVIRONMENT = {"import_path": extract.PREBAKED_ENV,
                        "kwargs": {"prebake_harness": "claude-code", "prebake_version": "2.1.312",
                                   "prebake_recipe": "0123456789" + "a" * 54, "prebake_required": False}}


def prebake_record(used: bool = True, **kw: object) -> dict:
    """prebake.json as PrebakedDockerEnvironment._record writes it."""
    record = {"bake_format": 1, "status": "ready" if used else "missing", "used": used,
              "reason": "pre-baked image verified" if used else f"no pre-baked image {BAKED}", "image": BAKED,
              "image_id": "sha256:" + "b" * 64 if used else None, "task_image_id": "sha256:" + "c" * 64,
              "baked_at": "2026-10-29T22:42:35.462308509Z" if used else None,
              "recipe": "0123456789" + "a" * 54 if used else None, "noop": False, "added_layers": 7 if used else None,
              "skip_execs": [], "skip_commands": [], "harness": "claude-code", "version": "2.1.312",
              "wanted_recipe": "0123456789" + "a" * 54, "prefix": "benchbake", "required": False,
              "task_image": "alexgshaw/fix-git:20251031",
              "ran_image": BAKED if used else "alexgshaw/fix-git:20251031", "skipped_execs": [],
              "written_at": "2026-10-29T23:32:14.648697+00:00"}
    record.update(kw)
    return record


class SetupProvenance(BenchTestCase):
    def test_the_constants_match_the_environment_class(self) -> None:
        from bench.environments import prebaked

        self.assertEqual((extract.PREBAKED_ENV, extract.PREBAKE_RECORD), (prebaked.PREBAKED_ENV, prebaked.RECORD_NAME))
        self.assertEqual(extract.DEFAULT_SETUP_TIMEOUT_SEC, 360.0)  # Harbor 0.25.0's BaseAgent default

    def test_the_four_modes(self) -> None:
        cfg = {"agent": {"name": "claude-code"}, "environment": PREBAKED_ENVIRONMENT}
        cases = [({"agent": {"name": "claude-code"}}, None, "stock"),
                 ({"agent": {"name": "claude-code"}, "environment": {"type": "docker", "import_path": None}}, None,
                  "stock"),
                 (cfg, None, None),  # asked for an image, killed before the class wrote its record
                 (cfg, prebake_record(), "prebaked"),
                 (cfg, prebake_record(used=False), "fallback"),
                 # The research prototype's class and record (no recipe, no wanted_recipe).
                 ({"environment": {"import_path": "prebake_env:PrebakedDockerEnvironment"}}, None, None),
                 ({}, _read(IDENTITY / "prebaked" / "strands" / "prebake.json"), "prebaked")]
        for trial_cfg, record, mode in cases:
            with self.subTest(trial_cfg=trial_cfg, record=bool(record)):
                self.assertEqual(extract.setup_provenance(trial_cfg, record)["setup_mode"], mode)
        used = extract.setup_provenance(cfg, prebake_record())
        self.assertEqual({k: v for k, v in used.items() if k != "prebake_json"}, {
            "setup_mode": "prebaked", "agent_setup_limit_sec": 360.0, "prebake_image": BAKED,
            "prebake_image_id": "sha256:" + "b" * 64, "prebake_recipe": "0123456789" + "a" * 54,
            "prebake_baked_at": "2026-10-29T22:42:35.462308Z", "prebake_reason": "pre-baked image verified"})
        self.assertEqual(json.loads(used["prebake_json"]), prebake_record())
        missing = extract.setup_provenance(cfg, prebake_record(used=False))
        self.assertEqual((missing["prebake_image"], missing["prebake_image_id"], missing["prebake_baked_at"]),
                         (BAKED, None, None))
        self.assertEqual(missing["prebake_recipe"], "0123456789" + "a" * 54)  # what the arm asked for
        stock = extract.setup_provenance({}, None)
        self.assertEqual([k for k, v in stock.items() if v is not None], ["setup_mode", "agent_setup_limit_sec"])

    def test_a_record_is_stored_redacted(self) -> None:
        record = prebake_record(used=False, reason="docker said: Authorization: Bearer abc.def.ghi")
        out = extract.setup_provenance({"environment": PREBAKED_ENVIRONMENT}, record)
        self.assertNotIn("abc.def.ghi", out["prebake_json"])
        self.assertNotIn("abc.def.ghi", out["prebake_reason"])
        self.assertEqual(json.loads(out["prebake_json"])["used"], False)  # still valid JSON

    def test_the_setup_limit_harbor_computes(self) -> None:
        cases = [({}, 360.0), ({"timeout_multiplier": 2.0}, 720.0),
                 ({"timeout_multiplier": 3.0, "agent_setup_timeout_multiplier": 2}, 720.0),
                 ({"agent": {"override_setup_timeout_sec": 100}}, 100.0),
                 ({"agent": {"override_setup_timeout_sec": 100}, "agent_setup_timeout_multiplier": 1.5}, 150.0),
                 ({"agent": {"name": "hermes"}}, 1200.0), ({"agent_setup_timeout_multiplier": "x"}, None),
                 ({"timeout_multiplier": 0}, None)]
        for trial_cfg, expected in cases:
            with self.subTest(trial_cfg=trial_cfg):
                self.assertEqual(extract.agent_setup_limit(trial_cfg), expected)

    def test_harbor_written_trials(self) -> None:
        for arm in ARM_HASHES:
            with self.subTest(arm=arm):
                stock = extract.facts_from_result(_read(IDENTITY / "stock" / arm / "result.json"), None, None,
                                                  trial_dir="t", ctx=_identity_ctx(arm), prefixes=PREFIXES)
                self.assertEqual((stock.setup_mode, stock.agent_setup_limit_sec, stock.prebake_json),
                                 ("stock", 360.0, None))
                folder = IDENTITY / "prebaked" / arm
                baked = extract.facts_from_result(_read(folder / "result.json"), None, None, trial_dir="t",
                                                  ctx=_identity_ctx(arm), prefixes=PREFIXES,
                                                  prebake=extract.read_prebake_record(folder))
                self.assertEqual(baked.setup_mode, "prebaked")
                self.assertTrue(baked.prebake_image.startswith(f"benchprobe/{arm}-"))
                self.assertEqual(baked.config_hash, stock.config_hash)
                # The prototype's class without its record: asked, but nothing says what happened.
                unknown = extract.facts_from_result(_read(folder / "result.json"), None, None, trial_dir="t",
                                                    ctx=_identity_ctx(arm), prefixes=PREFIXES)
                self.assertIsNone(unknown.setup_mode)

    def test_parse_trial_and_parse_incomplete_read_the_record(self) -> None:
        job = self.tmpdir() / "job"
        lf.make_job(job)
        done = lf.make_trial(job / "done", environment=PREBAKED_ENVIRONMENT, prebake=prebake_record(), setup_sec=0.5)
        killed = lf.make_trial(job / "killed", environment=PREBAKED_ENVIRONMENT, prebake=prebake_record(used=False),
                               finished=False)
        early = lf.make_trial(job / "early", environment=PREBAKED_ENVIRONMENT, finished=False)
        ctx = bench_ctx()
        facts = extract.parse_trial(done, ctx, PREFIXES)
        self.assertEqual((facts.setup_mode, facts.prebake_image, facts.agent_setup_sec), ("prebaked", BAKED, 0.5))
        self.assertEqual(extract.parse_incomplete(killed, ctx, PREFIXES).setup_mode, "fallback")
        self.assertIsNone(extract.parse_incomplete(early, ctx, PREFIXES).setup_mode)
        self.assertEqual(extract.parse_trial(lf.make_trial(job / "stock"), ctx, PREFIXES).setup_mode, "stock")


if __name__ == "__main__":
    unittest.main()
