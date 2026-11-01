"""Per-trial cost and reward reading, the budget's spend rule, per-arm means
over measured trials, model matching, and run results for both run.json
shapes. Trials are synthetic result.json files in temp folders, plus copies
of the dev pilot's files read from disk (never modified)."""

from __future__ import annotations

import copy
import json
import shutil
import unittest
from pathlib import Path
from typing import Any

from tests import ledger_fixtures as lf
from tests.support import BenchTestCase

from bench import results, settings
from bench.settings import Arm

PRICING = """\
anthropic:
  claude-opus-5-5:  {input: 4.00, output: 20.00, cache_read: 0.20}
  claude-haiku-5-5: {input: 0.10, output: 0.50, cache_read: 0.01}
  claude-fable-5-1: {input: 10.00, output: 50.00, cache_read: 0.25}
fireworks: {}
openrouter:
  deepseek-v4.1-flash: {input: 0.045, output: 0.50, cache_read: 0.006}
  glm-5.3:             {input: 0.07,  output: 1.40, cache_read: 0.069}
"""

REFERENCE = """\
defaults:
  direction: harness_over_baseline
  estimator: ratio_of_means
  tolerance: 1.5
  applies_to: ['anthropic:claude-*']
consensus:
  cost_per_trial: {low: 1/3, high: 1/2, source: consensus}
arms:
  pi:
    tokens_per_trial: {value: 1/3.1, source: S10}
  strands:
    tolerance: 1.1
    cost_per_trial: {value: 1/2.1, source: S8}
"""
# The same figures as written before multiples were harness / baseline.
REFERENCE_BASELINE_OVER_HARNESS = (REFERENCE.replace("harness_over_baseline", "baseline_over_harness")
                                   .replace("{low: 1/3, high: 1/2,", "{low: 2.0, high: 3.0,")
                                   .replace("{value: 1/3.1,", "{value: 3.1,").replace("{value: 1/2.1,", "{value: 2.1,"))

YAML = """\
dataset: terminal-bench@2.0
providers:
  anthropic: {env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}
  openrouter: {env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}
  fireworks: {env_key: FIREWORKS_AI_API_KEY, model_prefix: fireworks_ai/}
modes:
  pilot: {n_tasks: 1}
run:
  budget_usd: 10
baseline_arm: claude-code
arms:
  - {name: claude-code, agent: claude-code, provider: anthropic, model: claude-opus-5-5}
"""

DEEPSEEK = "deepseek/deepseek-v4.1-flash"
PILOT_ID = "20261028-222952-pilot"
PILOT_RUN = settings.ROOT / "profiles" / "dev" / "runs" / PILOT_ID / "run.json"
PILOT_JOBS = settings.ROOT / "profiles" / "dev" / "jobs" / PILOT_ID


def arm(name: str, agent: str, provider: str, model: str, family: str = "") -> Arm:
    return Arm(name=name, agent=agent, provider=provider, model=model, family=family or agent)


def usage(n_in: int, n_cache: int, n_out: int, cost: float | None = None) -> dict[str, Any]:
    return {"n_input_tokens": n_in, "n_cache_tokens": n_cache, "n_output_tokens": n_out, "cost_usd": cost}


def trial(*, n_in: int | None = 0, n_cache: int | None = 0, n_out: int | None = 0,
          cost_usd: float | None = None, model_usage: dict | None = None, reward: float | None = 1.0,
          exception: str | None = None, agent_result: bool = True, task: str = "task-a") -> dict[str, Any]:
    return {
        "task_name": task,
        "trial_name": f"{task}__x",
        "started_at": "2026-10-29T05:29:58+00:00",
        "finished_at": "2026-10-29T05:37:58+00:00",
        "agent_result": ({"n_input_tokens": n_in, "n_cache_tokens": n_cache, "n_output_tokens": n_out,
                          "cost_usd": cost_usd, "model_usage": model_usage} if agent_result else None),
        "verifier_result": None if reward is None else {"rewards": {"reward": reward}},
        "exception_info": ({"exception_type": exception, "exception_message": "boom " * 200}
                           if exception else None),
    }


class ResultsTestCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cfg = self.load_test_config(YAML, pricing=PRICING, reference=REFERENCE)
        self.jobs = self.tmpdir()
        self.count = 0

    def write(self, job: str, data: dict[str, Any]) -> Path:
        self.count += 1
        path = self.jobs / job / f"trial{self.count}" / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
        return path

    def read(self, a: Arm, data: dict[str, Any]) -> dict[str, Any]:
        row = results.read_trial(self.write(a.name, data), self.cfg, a)
        assert row is not None
        return row


# Tokens and cost of the dev pilot's claude-code-deepseek trial.
CC_DS = dict(n_in=92805, n_cache=61056, n_out=49969, cost_usd=1.438498)
CC_DS_COMPUTED = ((92805 - 61056) * 0.045 + 61056 * 0.006 + 49969 * 0.50) / 1e6


class SpendTests(ResultsTestCase):
    def test_synthetic_zero_token_entry_is_ignored(self) -> None:
        a = arm("cc-ds", "claude-code", "openrouter", DEEPSEEK)
        row = self.read(a, trial(**CC_DS, model_usage={
            DEEPSEEK: usage(92805, 61056, 49969), "<synthetic>": usage(0, 0, 0)}))
        self.assertEqual(row["unpriced_models"], [])
        self.assertEqual(row["models"], [DEEPSEEK])
        self.assertAlmostEqual(row["computed_cost"], CC_DS_COMPUTED)
        self.assertTrue(row["reported_mispriced"])
        self.assertAlmostEqual(results.spend(row), CC_DS_COMPUTED)

    def test_claude_code_on_an_unpriced_model_never_shows_its_own_figure(self) -> None:
        a = arm("cc-x", "claude-code", "openrouter", "z-ai/glm-5.3-flash")
        row = self.read(a, trial(n_in=100_000, n_cache=50_000, n_out=5_000, cost_usd=1.44,
                                 model_usage={"z-ai/glm-5.3-flash": usage(100_000, 50_000, 5_000)}))
        self.assertIsNone(row["computed_cost"])
        self.assertTrue(row["reported_mispriced"])
        self.assertLess(results.cost(row), 0.10)  # the openrouter top-rate floor, not Claude's rates
        self.assertGreater(results.cost(row), 0.0)
        self.assertAlmostEqual(results.spend(row), 1.44)  # budgets stay conservative

    def test_claude_code_off_anthropic_trusts_computed(self) -> None:
        a = arm("cc-ds", "claude-code", "openrouter", DEEPSEEK)
        row = self.read(a, trial(**CC_DS, model_usage={DEEPSEEK: usage(92805, 61056, 49969)}))
        self.assertAlmostEqual(results.spend(row), CC_DS_COMPUTED)
        self.assertAlmostEqual(results.cost(row), CC_DS_COMPUTED)

    def test_family_decides_for_import_path_claude_code(self) -> None:
        a = arm("cc-sub", "bench.agents.cc:ClaudeCodeVariant", "openrouter", DEEPSEEK, family="claude-code")
        row = self.read(a, trial(**CC_DS, model_usage={DEEPSEEK: usage(92805, 61056, 49969)}))
        self.assertEqual(row["family"], "claude-code")
        self.assertAlmostEqual(results.spend(row), CC_DS_COMPUTED)

    def test_claude_code_on_anthropic_takes_the_higher_figure(self) -> None:
        a = arm("cc", "claude-code", "anthropic", "claude-opus-5-5")
        tokens = usage(1_000_000, 800_000, 20_000)
        computed = (200_000 * 4.0 + 800_000 * 0.2 + 20_000 * 20.0) / 1e6
        row = self.read(a, trial(n_in=1_000_000, n_cache=800_000, n_out=20_000, cost_usd=computed + 0.25,
                                 model_usage={"claude-opus-5-5": tokens}))
        self.assertFalse(row["reported_mispriced"])
        self.assertAlmostEqual(row["computed_cost"], computed)
        self.assertAlmostEqual(results.spend(row), computed + 0.25)
        self.assertAlmostEqual(results.cost(row), computed)  # display stays like-for-like

    def test_pi_on_openrouter_takes_the_higher_figure(self) -> None:
        a = arm("pi-ds", "pi", "openrouter", DEEPSEEK)
        tokens = dict(n_in=1360013, n_cache=1252992, n_out=59144)
        computed = ((1360013 - 1252992) * 0.045 + 1252992 * 0.006 + 59144 * 0.5) / 1e6
        model_usage = {f"openrouter/{DEEPSEEK}": usage(1360013, 1252992, 59144)}
        low = self.read(a, trial(**tokens, cost_usd=0.0126, model_usage=model_usage))
        self.assertAlmostEqual(results.spend(low), computed)
        self.assertAlmostEqual(low["reported_cost"], 0.0126)  # kept visible next to computed
        high = self.read(a, trial(**tokens, cost_usd=0.09, model_usage=model_usage))
        self.assertAlmostEqual(results.spend(high), 0.09)

    def test_unpriced_tokens_never_count_as_zero(self) -> None:
        a = arm("strands-fw", "strands", "fireworks", "accounts/fireworks/models/glm-5p3")
        row = self.read(a, trial(n_in=3_000_000, n_cache=0, n_out=80_000, cost_usd=None))
        self.assertIsNone(row["computed_cost"])
        self.assertEqual(row["unpriced_models"], ["accounts/fireworks/models/glm-5p3"])
        # fireworks has no rows, so every provider's top rates apply
        floor = (3_000_000 * 10.0 + 80_000 * 50.0) / 1e6
        self.assertAlmostEqual(results.spend(row), floor)
        self.assertEqual(results.cost(row), 0.0)

        reported = self.read(a, trial(n_in=3_000_000, n_cache=0, n_out=80_000, cost_usd=0.5))
        self.assertAlmostEqual(results.spend(reported), 0.5)

    def test_unpriced_model_on_a_priced_provider_uses_that_providers_top_rate(self) -> None:
        a = arm("pi-x", "pi", "openrouter", "z-ai/glm-5.3-flash")
        row = self.read(a, trial(n_in=1_000_000, n_cache=0, n_out=100_000, cost_usd=0.0))
        self.assertEqual(row["unpriced_models"], ["z-ai/glm-5.3-flash"])  # not priced as glm-5.3
        self.assertAlmostEqual(results.spend(row), (1_000_000 * 0.07 + 100_000 * 1.40) / 1e6)

    def test_zero_token_setup_failure_spends_nothing(self) -> None:
        a = arm("strands-fw", "strands", "fireworks", "glm-5p3")
        row = self.read(a, trial(agent_result=False, reward=None, exception="AgentSetupTimeoutError"))
        self.assertEqual(results.spend(row), 0.0)

    def test_partial_cost_uses_the_higher_figure(self) -> None:
        a = arm("cc", "claude-code", "anthropic", "claude-opus-5-5")
        model_usage = {"claude-opus-5-5": usage(100_000, 0, 1_000), "claude-opus-9": usage(50_000, 0, 500)}
        computed = (100_000 * 4.0 + 1_000 * 20.0) / 1e6
        row = self.read(a, trial(n_in=150_000, n_out=1_500, cost_usd=0.9, model_usage=model_usage))
        self.assertTrue(row["cost_partial"])
        self.assertEqual(row["unpriced_models"], ["claude-opus-9"])
        self.assertAlmostEqual(row["computed_cost"], computed)
        self.assertAlmostEqual(results.cost(row), 0.9)
        self.assertAlmostEqual(results.spend(row), 0.9)

    def test_partial_cost_on_a_mispricing_harness_keeps_its_figure_out_of_cost(self) -> None:
        a = arm("cc-ds", "claude-code", "openrouter", DEEPSEEK)
        model_usage = {DEEPSEEK: usage(92805, 61056, 49969), "z-ai/glm-9-air": usage(10_000, 0, 500)}
        row = self.read(a, trial(n_in=102805, n_cache=61056, n_out=50469, cost_usd=1.44,
                                 model_usage=model_usage))
        self.assertTrue(row["cost_partial"])
        self.assertTrue(row["reported_mispriced"])
        floor = (10_000 * 0.07 + 500 * 1.40) / 1e6  # openrouter's top rates
        self.assertAlmostEqual(results.cost(row), CC_DS_COMPUTED + floor)
        self.assertLess(results.cost(row), 0.1)
        self.assertAlmostEqual(results.spend(row), 1.44)  # the budget stays on the safe side

    def test_partial_cost_without_reported_adds_the_floor(self) -> None:
        a = arm("cc", "claude-code", "anthropic", "claude-opus-5-5")
        model_usage = {"claude-opus-5-5": usage(100_000, 0, 1_000), "claude-opus-9": usage(50_000, 0, 500)}
        row = self.read(a, trial(n_in=150_000, n_out=1_500, cost_usd=None, model_usage=model_usage))
        expected = (100_000 * 4.0 + 1_000 * 20.0) / 1e6 + (50_000 * 10.0 + 500 * 50.0) / 1e6
        self.assertAlmostEqual(results.spend(row), expected)

    def test_a_call_missing_from_model_usage_still_counts(self) -> None:
        # The Opus pilot's mini-swe-agent trial: its totals hold a FormatError turn that model_usage leaves out.
        a = arm("mini", "mini-swe-agent", "anthropic", "claude-opus-5-5")
        row = self.read(a, trial(n_in=124959, n_cache=104191, n_out=16604, cost_usd=0.4567342, model_usage={
            "anthropic/claude-opus-5-5": usage(104401, 84196, 15383, 0.42314757)}))
        computed = ((124959 - 104191) * 4.0 + 104191 * 0.2 + 16604 * 20.0) / 1e6  # what the ledger prices
        self.assertAlmostEqual(row["computed_cost"], computed)
        self.assertAlmostEqual(results.cost(row), computed)
        self.assertEqual(row["models"], ["anthropic/claude-opus-5-5"])
        # On OpenRouter mini-swe-agent reports $0, so the budget rests on the computed figure.
        a = arm("mini-ds", "mini-swe-agent", "openrouter", DEEPSEEK)
        row = self.read(a, trial(n_in=3000, n_cache=2000, n_out=300, cost_usd=0.0, model_usage={
            f"openrouter/{DEEPSEEK}": usage(2000, 1500, 200, 0.0)}))
        self.assertAlmostEqual(results.spend(row), (1000 * 0.045 + 2000 * 0.006 + 300 * 0.50) / 1e6)

    def test_a_shortfall_across_models_goes_to_the_arms_model(self) -> None:
        a = arm("cc", "claude-code", "anthropic", "claude-opus-5-5")
        haiku = (30 * 0.10 + 5 * 0.50) / 1e6
        row = self.read(a, trial(n_in=200, n_cache=80, n_out=20, model_usage={
            "claude-opus-5-5-20261015": usage(100, 50, 10), "claude-haiku-5-5": usage(30, 0, 5)}))
        self.assertEqual(row["models"], ["claude-haiku-5-5", "claude-opus-5-5-20261015"])
        self.assertAlmostEqual(row["computed_cost"], ((170 - 80) * 4.0 + 80 * 0.2 + 15 * 20.0) / 1e6 + haiku)
        # No reported model is the arm's: the arm's model takes it.
        row = self.read(a, trial(n_in=200, n_cache=80, n_out=20, model_usage={
            "claude-fable-5-1": usage(100, 50, 10), "claude-haiku-5-5": usage(30, 0, 5)}))
        self.assertEqual(row["models"], ["claude-fable-5-1", "claude-haiku-5-5", "claude-opus-5-5"])
        fable = (50 * 10.0 + 50 * 0.25 + 10 * 50.0) / 1e6
        self.assertAlmostEqual(row["computed_cost"], fable + haiku + (40 * 4.0 + 30 * 0.2 + 5 * 20.0) / 1e6)

    def test_unchanged_cost_formula(self) -> None:
        price = settings.Price(input=4.0, output=20.0, cache_read=None)
        self.assertAlmostEqual(results._cost(price, 1000, 400, 10), (600 * 4 + 400 * 4 + 10 * 20) / 1e6)
        self.assertAlmostEqual(results._cost(price, 100, 400, 0), (400 * 4) / 1e6)  # fresh never negative

    def test_cache_writes_from_the_harness_log_are_priced_at_their_rates(self) -> None:
        # result.json holds no write counts; the trial folder's harness log does, as for the ledger.
        cfg = self.load_test_config(YAML, pricing=lf.OPUS_PRICING_YAML, reference=REFERENCE)
        calls = ((2, 0, 3000, 0, 400), (3, 2900, 500, 100, 250))
        a = arm("cc", "claude-code", "anthropic", "claude-opus-5-5")
        path = self.jobs / "cc" / "trial-w" / "result.json"
        n_in, n_cache, n_out, exact = lf.write_cache_log(path.parent, "claude-code", calls)
        path.write_text(json.dumps(trial(n_in=n_in, n_cache=n_cache, n_out=n_out, cost_usd=exact, model_usage={
            "claude-opus-5-5": usage(n_in, n_cache, n_out, exact)})), encoding="utf-8", newline="\n")
        row = results.read_trial(path, cfg, a)
        assert row is not None
        self.assertAlmostEqual(row["computed_cost"], exact, places=12)
        self.assertAlmostEqual(results.spend(row), exact, places=12)
        price = cfg.price("anthropic", "claude-opus-5-5")
        assert price is not None
        self.assertAlmostEqual(results._cost(price, 10000, 6000, 100, 3000, 500),
                               (500 * 4.0 + 6000 * 0.2 + 3000 * 5.0 + 500 * 8.0 + 100 * 20.0) / 1e6)


class RewardTests(ResultsTestCase):
    def test_sandbox_failure_reward_is_invalid(self) -> None:
        a = arm("pi", "pi", "anthropic", "claude-opus-5-5")
        row = self.read(a, trial(n_in=10, n_out=1, reward=0.0, exception="SandboxLikelyOutOfMemoryError"))
        self.assertIsNone(row["reward"])
        self.assertIsNone(row["passed"])
        self.assertFalse(row["reward_valid"])
        self.assertEqual(row["failure_cause"], "sandbox_failed")
        self.assertEqual(row["error"], "SandboxLikelyOutOfMemoryError")
        self.assertEqual(len(row["error_message"]), 400)

    def test_provider_failure_reward_is_invalid(self) -> None:
        a = arm("pi", "pi", "anthropic", "claude-opus-5-5")
        row = self.read(a, trial(n_in=10, n_out=1, reward=0.0, exception="ApiRateLimitError"))
        self.assertIsNone(row["passed"])
        self.assertEqual(row["failure_cause"], "model_provider_failed")

    def test_harness_exit_and_timeouts_stay_scored(self) -> None:
        a = arm("cc", "claude-code", "anthropic", "claude-opus-5-5")
        crashed = self.read(a, trial(n_in=10, n_out=1, reward=0.0, exception="NonZeroAgentExitCodeError"))
        self.assertTrue(crashed["reward_valid"])
        self.assertIs(crashed["passed"], False)
        timed_out = self.read(a, trial(n_in=10, n_out=1, reward=1.0, exception="AgentTimeoutError"))
        self.assertIs(timed_out["passed"], True)
        clean = self.read(a, trial(n_in=10, n_out=1, reward=1.0))
        self.assertTrue(clean["reward_valid"])
        self.assertIsNone(clean["failure_cause"])

    def test_pass_rate_leaves_out_invalid_rewards(self) -> None:
        a = arm("pi", "pi", "anthropic", "claude-opus-5-5")
        rows = [self.read(a, trial(n_in=10, n_out=1, reward=1.0)),
                self.read(a, trial(n_in=10, n_out=1, reward=0.0)),
                self.read(a, trial(n_in=10, n_out=1, reward=0.0, exception="ApiConnectionError"))]
        s = results.summarize(a, rows)
        self.assertEqual((s["scored"], s["passed"], s["errors"]), (2, 1, 1))
        self.assertEqual(s["pass_rate"], 0.5)


class UnrecordedUsageTests(ResultsTestCase):
    """The ledger's usage rule on the per-run results page: a timeout with
    Harbor's 0/0/0 is left out of the means and its spend is unknown."""

    def timed_out(self, tokens: int | None) -> dict[str, Any]:
        data = trial(n_in=tokens, n_cache=0 if tokens is not None else None, n_out=tokens, reward=0.0,
                     exception="AgentTimeoutError")
        data["agent_execution"] = {"started_at": "2026-10-29T13:14:33Z", "finished_at": "2026-10-29T13:29:33Z"}
        return data

    def test_a_timeout_with_nothing_recorded_is_unmeasured(self) -> None:
        a = arm("pi-glm", "pi", "openrouter", "z-ai/glm-5.3")
        real = self.read(a, trial(n_in=707943, n_cache=605450, n_out=60178, reward=0.0))
        for tokens in (0, None):
            with self.subTest(tokens=tokens):
                lost = self.read(a, self.timed_out(tokens))
                self.assertEqual((lost["usage_status"], results.spend(lost)), ("unmeasured", 0.0))
                s = results.summarize(a, [real, lost])
                self.assertEqual((s["measured"], s["excluded"], s["unmeasured"], s["lower_bound"]), (1, 1, 1, 0))
                self.assertAlmostEqual(s["mean_cost"], results.cost(real))
                self.assertEqual((s["scored"], s["pass_rate"]), (2, 0.0))  # its reward still counts
        self.assertEqual(real["usage_status"], "full")

    def test_a_timeout_with_usage_is_a_lower_bound(self) -> None:
        a = arm("strands-ds", "strands", "openrouter", DEEPSEEK)
        row = self.read(a, self.timed_out(1000))
        self.assertEqual(row["usage_status"], "lower_bound")
        s = results.summarize(a, [row])
        self.assertEqual((s["measured"], s["unmeasured"], s["lower_bound"]), (1, 0, 1))

    def test_a_setup_failure_spent_nothing_rather_than_an_unknown_amount(self) -> None:
        a = arm("strands-fw", "strands", "fireworks", "glm-5p3")
        row = self.read(a, trial(agent_result=False, reward=None, exception="AgentSetupTimeoutError"))
        self.assertEqual(row["usage_status"], "full")
        self.assertEqual(results.summarize(a, [row])["unmeasured"], 0)

    def test_a_provider_failure_after_real_work_counts_as_unknown_spend(self) -> None:
        # A rate limit 40 minutes in, from an adapter that wrote null: the run
        # manager must not treat its $0 as all it spent.
        a = arm("pi-glm", "pi", "openrouter", "z-ai/glm-5.3")
        for minutes, status in ((40, "unmeasured"), (0, "full")):
            with self.subTest(minutes=minutes):
                data = trial(n_in=None, n_cache=None, n_out=None, reward=None, exception="ApiRateLimitError")
                data["agent_execution"] = {"started_at": "2026-10-29T13:00:00Z",
                                           "finished_at": f"2026-10-29T13:{minutes:02d}:30Z"}
                row = self.read(a, data)
                self.assertEqual(row["usage_status"], status)
                self.assertEqual(results.summarize(a, [row])["unmeasured"], int(status == "unmeasured"))

    def test_odd_step_results_do_not_stop_the_live_view(self) -> None:
        # read_trial runs on every poll of a live run, where an exception would end the run.
        a = arm("pi-glm", "pi", "openrouter", "z-ai/glm-5.3")
        data = trial(n_in=1000, n_cache=0, n_out=100)
        data["step_results"] = 5
        self.assertEqual(self.read(a, data)["usage_status"], "full")


class SummaryTests(ResultsTestCase):
    def test_means_cover_measured_trials_only(self) -> None:
        a = arm("pi", "pi", "anthropic", "claude-opus-5-5")
        rows = [
            self.read(a, trial(n_in=1_000_000, n_cache=500_000, n_out=10_000)),
            self.read(a, trial(n_in=3_000_000, n_cache=1_500_000, n_out=30_000)),
            self.read(a, trial(agent_result=False, reward=None, exception="AgentSetupTimeoutError")),
            self.read(a, trial(n_in=None, n_cache=None, n_out=None, reward=0.0,
                               exception="NonZeroAgentExitCodeError")),
        ]
        s = results.summarize(a, rows)
        self.assertEqual((s["trials"], s["measured"], s["excluded"]), (4, 2, 2))
        self.assertEqual(s["family"], "pi")
        self.assertEqual(s["mean_tokens"], (1_010_000 + 3_030_000) / 2)
        self.assertEqual(s["mean_input"], 2_000_000)
        cost_1 = (500_000 * 4 + 500_000 * 0.2 + 10_000 * 20) / 1e6
        cost_2 = (1_500_000 * 4 + 1_500_000 * 0.2 + 30_000 * 20) / 1e6
        self.assertAlmostEqual(s["mean_cost"], (cost_1 + cost_2) / 2)
        self.assertAlmostEqual(s["total_cost"], cost_1 + cost_2)
        self.assertEqual(s["cache_share"], 0.5)

    def test_nothing_measured_gives_no_means(self) -> None:
        a = arm("strands", "strands", "anthropic", "claude-opus-5-5")
        rows = [self.read(a, trial(agent_result=False, reward=None, exception="AgentSetupTimeoutError"))]
        s = results.summarize(a, rows)
        self.assertEqual((s["measured"], s["excluded"]), (0, 1))
        self.assertIsNone(s["mean_tokens"])
        self.assertIsNone(s["mean_cost"])

    def test_empty_arm(self) -> None:
        s = results.summarize(arm("x", "pi", "anthropic", "claude-opus-5-5"), [])
        self.assertEqual((s["trials"], s["measured"], s["excluded"]), (0, 0, 0))


class ModelMatchTests(BenchTestCase):
    def test_same_model_accepts_only_date_and_version_tags(self) -> None:
        cases = [
            ("claude-haiku-5-5-20261001", "claude-haiku-5-5", True),
            ("claude-haiku-5-5", "claude-haiku-5-5", True),
            ("openrouter/z-ai/glm-5.3", "z-ai/glm-5.3", True),
            ("anthropic/claude-opus-5-5@2026-10-01", "claude-opus-5-5", True),
            ("claude-opus-5-5", "claude-opus-5", False),
            ("z-ai/glm-5.3-flash", "z-ai/glm-5.3", False),
            ("claude-opus-5-5-fast", "claude-opus-5-5", False),
        ]
        for reported, model, expected in cases:
            with self.subTest(reported=reported, model=model):
                a = arm("a", "pi", "anthropic", model)
                self.assertIs(results._same_model(reported, a), expected)


class OtherModelTests(ResultsTestCase):
    def test_a_longer_model_id_is_another_model(self) -> None:
        a = arm("cc", "claude-code", "anthropic", "claude-opus-5")
        row = self.read(a, trial(n_in=1000, n_out=100, model_usage={"claude-opus-5-5": usage(1000, 0, 100)}))
        self.assertEqual(row["other_model_tokens"], 1100)


class CompareTests(ResultsTestCase):
    def summary(self, a: Arm, tokens: int, n: int = 2, failures: int = 0) -> dict[str, Any]:
        rows = [self.read(a, trial(n_in=tokens, n_out=0)) for _ in range(n)]
        rows += [self.read(a, trial(agent_result=False, reward=None, exception="AgentSetupTimeoutError"))
                 for _ in range(failures)]
        return results.summarize(a, rows)

    def test_reference_lookup_by_family_and_verdict(self) -> None:
        base = self.summary(arm("claude-code", "claude-code", "anthropic", "claude-opus-5-5"), 3_100_000)
        pi = self.summary(arm("pi-sub", "bench.agents.p:PiVariant", "anthropic", "claude-opus-5-5",
                              family="pi"), 1_000_000, failures=3)
        (row,) = results.compare(self.cfg, [base, pi], "claude-code")
        self.assertEqual(row["family"], "pi")
        self.assertAlmostEqual(row["tokens_multiple"], 1 / 3.1)   # pi / claude-code; setup failures do not shrink it
        self.assertEqual(row["published_tokens"]["value"], 1 / 3.1)
        self.assertEqual(row["tokens_verdict"], "in line")
        self.assertEqual(row["excluded"], 3)
        self.assertEqual(row["baseline_excluded"], 0)
        self.assertIsNone(row["verdict_note"])

    def test_tolerance_from_reference(self) -> None:
        base = self.summary(arm("claude-code", "claude-code", "anthropic", "claude-opus-5-5"), 3_000_000)
        strands = self.summary(arm("strands", "strands", "anthropic", "claude-opus-5-5"), 1_250_000)
        (row,) = results.compare(self.cfg, [base, strands], "claude-code")
        self.assertAlmostEqual(row["cost_multiple"], 1 / 2.4)
        self.assertEqual(row["cost_verdict"], "lower")        # 1/2.4 < (1/2.1) / 1.1: relatively cheaper

    def test_a_reference_written_baseline_over_harness_reads_the_same(self) -> None:
        old = self.load_test_config(YAML, pricing=PRICING, reference=REFERENCE_BASELINE_OVER_HARNESS)
        base = self.summary(arm("claude-code", "claude-code", "anthropic", "claude-opus-5-5"), 3_000_000)
        others = [self.summary(arm("strands", "strands", "anthropic", "claude-opus-5-5"), 1_250_000),
                  self.summary(arm("pi", "pi", "anthropic", "claude-opus-5-5"), 1_000_000),
                  self.summary(arm("t2", "terminus-2", "anthropic", "claude-opus-5-5"), 2_500_000)]
        new_rows = results.compare(self.cfg, [base, *others], "claude-code")
        self.assertEqual(results.compare(old, [base, *others], "claude-code"), new_rows)
        t2 = next(r for r in new_rows if r["family"] == "terminus-2")
        self.assertEqual((t2["published_cost"]["low"], t2["published_cost"]["high"]), (1 / 3, 0.5))
        self.assertEqual(t2["cost_verdict"], "higher")  # 2.5/3 against the published 0.333-0.5

    def test_verdicts_withheld_for_non_claude_models(self) -> None:
        base = self.summary(arm("claude-code", "claude-code", "openrouter", DEEPSEEK), 3_100_000)
        pi = self.summary(arm("pi", "pi", "openrouter", DEEPSEEK), 1_000_000)
        (row,) = results.compare(self.cfg, [base, pi], "claude-code")
        self.assertAlmostEqual(row["tokens_multiple"], 1 / 3.1)
        self.assertIsNone(row["tokens_verdict"])
        self.assertIsNone(row["cost_verdict"])
        self.assertIn("Claude models", row["verdict_note"])

    def test_missing_baseline(self) -> None:
        pi = self.summary(arm("pi", "pi", "anthropic", "claude-opus-5-5"), 1_000_000)
        self.assertEqual(results.compare(self.cfg, [pi], "claude-code"), [])


class RunShapeTests(ResultsTestCase):
    def test_resolved_arms_win_over_the_config_snapshot(self) -> None:
        run = {
            "id": "20261029-101010-sample",
            "config": {"arms": [{"name": "mini", "agent": "mini-swe-agent", "provider": "openrouter",
                                 "model": DEEPSEEK}]},
            "resolved_arms": [{
                "name": "mini", "agent": "bench.agents.mini_swe_agent:MiniSweAgentWithMcp",
                "family": "mini-swe-agent", "kind": "import_path", "provider": "openrouter",
                "model": DEEPSEEK, "harbor_model": f"openrouter/{DEEPSEEK}",
                "effort": {"setting": "inherit", "value": "medium", "label": "medium"},
                "kwargs": [{"name": "cost_limit", "value": "1", "source": "run cap"},
                           {"name": "config", "value": "cfg.yaml", "source": "arm option"}],
                "options_fingerprint": "sha256:x", "tasks": ["t"], "trials": 1, "command": "harbor run",
            }],
        }
        (a,) = results.arms_for_run(run)
        self.assertEqual(a.agent, "bench.agents.mini_swe_agent:MiniSweAgentWithMcp")
        self.assertEqual(a.family, "mini-swe-agent")
        self.assertEqual(a.kind, "import_path")
        self.assertEqual(a.effort, "inherit")
        self.assertEqual(a.options, {"config": "cfg.yaml"})

    def test_legacy_snapshot_fills_family(self) -> None:
        run = {"id": "r", "config": {"arms": [
            {"name": "s", "agent": "strands", "provider": "openrouter", "model": DEEPSEEK, "effort": "omit"},
            {"name": "m", "agent": "x.y:Z", "family": "mini-swe-agent", "provider": "openrouter",
             "model": DEEPSEEK}]}}
        s, m = results.arms_for_run(run)
        self.assertEqual((s.family, s.effort), ("strands", "omit"))
        self.assertEqual(m.family, "mini-swe-agent")

    def test_run_results_reads_resolved_arms_and_spec_baseline(self) -> None:
        jobs = self.isolate_data_dir() / "jobs"
        run_id = "20261029-101010-design-v1"
        resolved = [
            {"name": "base", "agent": "claude-code", "family": "claude-code", "provider": "anthropic",
             "model": "claude-opus-5-5"},
            {"name": "pi-sub", "agent": "bench.agents.p:PiVariant", "family": "pi", "provider": "anthropic",
             "model": "claude-opus-5-5"},
        ]
        for name, tokens in (("base", 3_100_000), ("pi-sub", 1_000_000)):
            path = jobs / run_id / name / "t__1" / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(trial(n_in=tokens, n_out=0)), encoding="utf-8", newline="\n")
        run = {"id": run_id, "spec": {"baseline_arm": "base"}, "resolved_arms": resolved,
               "arms": [{"name": "base"}, {"name": "pi-sub"}, {"name": "gone"}]}
        out = results.run_results(self.cfg, run)
        self.assertEqual(out["baseline"], "base")
        self.assertEqual([s["arm"] for s in out["arms"]], ["base", "pi-sub"])
        (row,) = out["comparison"]
        self.assertEqual(row["family"], "pi")
        self.assertEqual(row["tokens_verdict"], "in line")


@unittest.skipUnless(PILOT_RUN.exists() and PILOT_JOBS.exists(), "dev pilot data not present")
class LegacyPilotTests(ResultsTestCase):
    """The dev pilot's run.json predates resolved_arms and family."""

    def setUp(self) -> None:
        super().setUp()
        # Real pricing and reference, as the pilot was costed with.
        self.cfg = self.load_test_config(YAML)
        self.run = copy.deepcopy(json.loads(PILOT_RUN.read_text(encoding="utf-8")))
        jobs = self.isolate_data_dir() / "jobs"
        for source in PILOT_JOBS.glob("*/*/result.json"):
            target = jobs / PILOT_ID / source.relative_to(PILOT_JOBS)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def test_arms_for_legacy_run(self) -> None:
        arms = results.arms_for_run(self.run)
        self.assertEqual(len(arms), len(self.run["config"]["arms"]))
        self.assertTrue(all(a.family == a.agent for a in arms))
        self.assertIn("mini-swe-agent", {a.family for a in arms})

    def test_run_results_still_summarize(self) -> None:
        out = results.run_results(self.cfg, self.run)
        self.assertEqual(out["baseline"], "claude-code-deepseek")
        by_arm = {s["arm"]: s for s in out["arms"]}
        self.assertEqual(len(by_arm), len(self.run["arms"]))
        cc = by_arm["claude-code-deepseek"]
        self.assertEqual((cc["trials"], cc["measured"]), (1, 1))
        self.assertEqual(by_arm["strands-deepseek"]["excluded"], 1)
        self.assertIsNone(by_arm["strands-deepseek"]["mean_tokens"])
        rows = {t["arm"]: t for t in out["trials"]}
        self.assertAlmostEqual(results.spend(rows["claude-code-deepseek"]), CC_DS_COMPUTED, places=4)
        comparison = {c["arm"]: c for c in out["comparison"]}
        self.assertIn("pi-deepseek", comparison)
        self.assertIsNone(comparison["pi-deepseek"]["tokens_verdict"])  # not a Claude model
        self.assertTrue(comparison["pi-deepseek"]["verdict_note"])
        self.assertTrue(PILOT_RUN.exists())


if __name__ == "__main__":
    unittest.main()
