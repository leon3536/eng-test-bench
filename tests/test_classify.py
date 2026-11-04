"""Outcome classification: the pilots' golden outcomes, every Harbor failure
cause, and the validity of each outcome under the report's policies."""

from __future__ import annotations

import json
import unittest
from collections import Counter
from dataclasses import asdict
from typing import Any

from harbor.models.trial.error_policy import _POLICY_GROUPS

from tests import ledger_fixtures as lf
from tests.support import BenchTestCase

from bench import classify, extract

PREFIXES = {"openrouter/": "openrouter", "anthropic/": "anthropic"}

CAUSE_TO_OUTCOME = {
    "sandbox_failed": "infra",
    "harbor_failed": "infra",
    "cancelled": "infra",
    "model_provider_failed": "provider_failure",
    "verifier_failed": "verifier_failure",
    "agent_exhausted_budget": "agent_limit",
    "agent_refused_task": "agent_limit",
    "unclassified": "agent_crash",
}


def facts(**over: Any) -> dict[str, Any]:
    """A finished trial that used the model, then failed with `exception_type`."""
    base = {"finished_at": "2026-10-29T05:40:00.000000Z", "exception_type": None, "harbor_cause": None,
            "reward": 0.0, "agent_started": 1, "n_input": 1000, "n_cache": 0, "n_output": 100,
            "agent_exec_sec": 300.0, "setup_marker": None, "cap_marker": None, "options_json": "{}",
            "family": "pi", "provider": "openrouter", "model": "z-ai/glm-5.3"}
    base.update(over)
    if base["exception_type"] and "harbor_cause" not in over:
        base["harbor_cause"] = classify.harbor_cause(base["exception_type"])
    return base


class Golden(unittest.TestCase):
    def test_pilots_classify_to_the_golden_counts(self) -> None:
        got: dict[tuple[str, str], str] = {}
        for trial in lf.fixture_trial_dirs():
            run_id, arm = trial.parent.parent.name, trial.parent.name
            run = json.loads((lf.FIXTURE_DEV / "runs" / run_id / "run.json").read_text(encoding="utf-8"))
            ctx = extract.context_for_bench_arm(run, arm, "dev")
            f = extract.parse_trial(trial, ctx, PREFIXES) or extract.parse_incomplete(trial, ctx, PREFIXES)
            self.assertIsNotNone(f, trial)
            got[(run_id, arm)] = classify.classify(asdict(f))[0]
        self.assertEqual(got, lf.GOLDEN)
        self.assertEqual(dict(Counter(got.values())), lf.GOLDEN_COUNTS)


class HarborCauses(unittest.TestCase):
    def test_every_policy_name_maps_to_its_documented_outcome(self) -> None:
        names = 0
        for policy, exception_names in _POLICY_GROUPS.items():
            expected = CAUSE_TO_OUTCOME[policy.cause.value]
            for name in exception_names:
                names += 1
                with self.subTest(name=name):
                    self.assertEqual(classify.harbor_cause(name), policy.cause.value)
                    unscored = {"reward": None} if expected == "provider_failure" else {}
                    self.assertEqual(classify.classify(facts(exception_type=name, **unscored))[0], expected)
        self.assertGreater(names, 40)

    def test_no_exception_means_no_cause(self) -> None:
        self.assertIsNone(classify.harbor_cause(None))
        self.assertIsNone(classify.harbor_cause(""))


class Rules(unittest.TestCase):
    def outcome(self, **over: Any) -> str:
        return classify.classify(facts(**over))[0]

    def test_named_cases(self) -> None:
        self.assertEqual(self.outcome(exception_type="AgentTimeoutError"), "agent_limit")
        self.assertEqual(self.outcome(exception_type="VerifierTimeoutError"), "verifier_failure")
        self.assertEqual(self.outcome(exception_type="ApiRateLimitError", reward=None), "provider_failure")
        self.assertEqual(self.outcome(exception_type="CancelledError"), "infra")
        self.assertEqual(self.outcome(exception_type="SomethingNewError"), "agent_crash")

    def test_a_timeout_on_a_busy_host_is_infra(self) -> None:
        busy = {"host_cpu_mean": 91.2, "host_cpu_hot_share": 0.62, "host_cpu_samples": 120, "cpu_contended": True}
        quiet = {**busy, "host_cpu_hot_share": 0.1, "cpu_contended": False}
        for given in ({"flags": busy}, {"flags_json": json.dumps(busy)}):
            outcome, reason = classify.classify(facts(exception_type="AgentTimeoutError", **given))
            self.assertEqual(outcome, "infra")
            self.assertEqual(reason, "AgentTimeoutError while the host CPU was busy (62% of the agent's time at >=90%)")
        self.assertEqual(classify.validity("infra"), classify.Validity(False, False, False))  # rerun, never counted
        self.assertEqual(self.outcome(exception_type="AgentTimeoutError", flags=quiet), "agent_limit")
        self.assertEqual(self.outcome(exception_type="AgentTimeoutError", flags={}), "agent_limit")  # unknown
        self.assertEqual(self.outcome(exception_type="AgentTimeoutError", flags_json="not json"), "agent_limit")
        # Only the timeout: a busy host does not excuse a crash or a failed task.
        self.assertEqual(self.outcome(exception_type="SomethingNewError", flags=busy), "agent_crash")
        self.assertEqual(self.outcome(reward=0.0, flags=busy), "failed")

    def test_no_exception(self) -> None:
        self.assertEqual(classify.classify(facts(reward=1.0)), ("passed", "reward 1"))
        self.assertEqual(self.outcome(reward=0.0), "failed")
        self.assertEqual(classify.classify(facts(reward=None)), ("verifier_failure", "no reward recorded"))

    def test_incomplete_without_a_result(self) -> None:
        self.assertEqual(classify.classify(facts(finished_at=None)), ("incomplete", "killed before finishing"))

    def test_setup_failures(self) -> None:
        nz = "NonZeroAgentExitCodeError"
        self.assertEqual(classify.classify(facts(exception_type=nz, agent_started=0)),
                         ("setup_failure", "agent never started"))
        self.assertEqual(classify.classify(facts(exception_type=nz, n_input=0, n_cache=0, n_output=0)),
                         ("setup_failure", "no model tokens"))
        self.assertEqual(self.outcome(exception_type=nz, n_input=None, n_cache=None, n_output=None,
                                      agent_exec_sec=2.6), "setup_failure")
        self.assertEqual(self.outcome(exception_type=nz, n_input=None, n_cache=None, n_output=None,
                                      agent_exec_sec=600, setup_marker="No module named"), "setup_failure")
        crash = classify.classify(facts(exception_type=nz, n_input=None, n_cache=None, n_output=None,
                                        agent_exec_sec=600))
        self.assertEqual(crash[0], "agent_crash")
        self.assertIn("tokens not reported", crash[1])

    def test_claude_code_cap_marker_depends_on_provider(self) -> None:
        cap = dict(exception_type="NonZeroAgentExitCodeError", family="claude-code", cap_marker="error_max_budget_usd",
                   options_json=json.dumps({"max_budget_usd": "1"}))
        self.assertEqual(self.outcome(**cap, provider="anthropic", model="claude-opus-5-5"), "agent_limit")
        self.assertEqual(self.outcome(**cap, provider="openrouter"), "misconfigured_cap")
        turns = dict(cap, cap_marker="error_max_turns", options_json=json.dumps({"max_turns": 150}))
        self.assertEqual(classify.classify(facts(**turns, provider="openrouter")),
                         ("agent_limit", "stopped by its own max_turns"))
        # A marker for an option the trial never set is just a crash.
        self.assertEqual(self.outcome(**dict(cap, options_json="{}"), provider="openrouter"), "agent_crash")

    def test_cap_misprices_delegates_to_harnesses(self) -> None:
        self.assertTrue(classify.cap_misprices("claude-code", "openrouter", "z-ai/glm-5.3", "max_budget_usd"))
        self.assertFalse(classify.cap_misprices("claude-code", "anthropic", "claude-opus-5-5", "max_budget_usd"))
        self.assertFalse(classify.cap_misprices("mini-swe-agent", "openrouter", "z-ai/glm-5.3", "cost_limit"))


class ValidityTable(unittest.TestCase):
    def test_default_policy(self) -> None:
        table = {o: tuple(asdict(classify.validity(o)).values()) for o in classify.OUTCOMES}
        self.assertEqual(table, {
            "passed": (True, True, True), "failed": (True, True, True), "answer_lookup": (True, True, True),
            "agent_limit": (True, True, True),
            "agent_crash": (True, True, True), "misconfigured_cap": (False, False, False),
            "verifier_failure": (True, False, True), "provider_failure": (False, False, False),
            "setup_failure": (False, False, False), "infra": (False, False, False),
            "incomplete": (False, False, False),
        })

    def test_policy_switches(self) -> None:
        self.assertEqual(classify.validity("agent_crash", crashes="exclude"), classify.Validity(False, False, False))
        self.assertEqual(classify.validity("misconfigured_cap", misconfigured_cap="include"),
                         classify.Validity(True, True, True))
        self.assertEqual(classify.validity("passed", crashes="exclude"), classify.Validity(True, True, True))
        with self.assertRaises(ValueError):
            classify.validity("nonsense")
        with self.assertRaises(ValueError):
            classify.validity("passed", crashes="maybe")


class UsageStatus(unittest.TestCase):
    """Whether a trial's recorded usage is all it spent. Pilot 3's pi-glm
    timed out after 900 s with Harbor's 0/0/0: its first model call never
    completed, so its cost is unknown, not $0."""

    def status(self, **over: Any) -> str:
        return classify.usage_status(facts(**over))

    def test_a_timeout_with_nothing_recorded_is_unmeasured_but_keeps_its_reward(self) -> None:
        # The reason names what Harbor wrote: zeros, or null.
        for tokens, tail in (((0, 0, 0), "no usage recorded"), ((None, None, None), "tokens not reported")):
            with self.subTest(tokens=tokens):
                f = facts(exception_type="AgentTimeoutError", n_input=tokens[0], n_cache=tokens[1],
                          n_output=tokens[2], agent_exec_sec=900.07)
                outcome, reason = classify.classify(f)
                self.assertEqual(outcome, "agent_limit")  # never a setup failure
                self.assertEqual(reason, f"AgentTimeoutError: the agent ran out of its budget; {tail}")
                self.assertFalse(classify.usage_recorded(f))
                self.assertEqual(classify.usage_status(f), "unmeasured")
                self.assertEqual(classify.validity(outcome, usage="unmeasured"), classify.Validity(False, True, True))

    def test_a_timeout_with_usage_is_a_lower_bound_and_still_counts(self) -> None:
        f = facts(exception_type="AgentTimeoutError", n_input=353896, n_cache=320384, n_output=82917)
        self.assertEqual(classify.classify(f), ("agent_limit", "AgentTimeoutError: the agent ran out of its budget"))
        self.assertEqual(classify.usage_status(f), "lower_bound")
        self.assertEqual(classify.validity("agent_limit", usage="lower_bound"), classify.Validity(True, True, True))

    def test_statuses(self) -> None:
        zero = dict(n_input=0, n_cache=0, n_output=0)
        self.assertEqual(self.status(), "full")
        self.assertEqual(self.status(reward=1.0, **zero), "unmeasured")  # the agent ran; nothing was recorded
        self.assertEqual(self.status(exception_type="NonZeroAgentExitCodeError", **zero), "full")  # setup failure
        self.assertEqual(self.status(exception_type="ApiRateLimitError", agent_exec_sec=5.0, reward=None, **zero),
                         "full")  # never answered
        self.assertEqual(self.status(exception_type="CancelledError"), "lower_bound")
        self.assertEqual(self.status(exception_type="CancelledError", **zero), "unmeasured")
        self.assertEqual(self.status(exception_type="CancelledError", agent_started=0, **zero), "full")
        self.assertEqual(self.status(exception_type="ContextWindowExceededError"), "full")
        self.assertEqual(self.status(exception_type="NonZeroAgentExitCodeError", n_input=None, n_cache=None,
                                     n_output=None, agent_exec_sec=600), "unmeasured")  # agent_crash
        # Killed: usage recovered from the harness's log is a floor; none is unknown.
        self.assertEqual(self.status(finished_at=None, agent_started=1), "lower_bound")
        self.assertEqual(self.status(finished_at=None, agent_started=0, n_input=None, n_cache=None, n_output=None),
                         "unmeasured")
        # A stored outcome is used as it is.
        self.assertEqual(classify.usage_status({"outcome": "failed", "n_input": 0, "n_output": 0,
                                                "agent_started": True}), "unmeasured")
        self.assertEqual(set(classify.USAGE_STATUSES), {"full", "lower_bound", "unmeasured"})

    def test_an_outcome_reached_after_the_agent_ran_is_unmeasured_without_its_timing(self) -> None:
        # An external job or an older Harbor may write no agent_execution: a
        # reward, a timeout or a verifier run still means the agent ran.
        zero = dict(n_input=0, n_cache=0, n_output=0, agent_started=0, agent_exec_sec=None)
        for over in (dict(reward=1.0), dict(reward=0.0), dict(reward=None),
                     dict(exception_type="AgentTimeoutError"), dict(exception_type="VerifierTimeoutError")):
            with self.subTest(**over):
                f = facts(**{**zero, **over})
                self.assertEqual(classify.usage_status(f), "unmeasured")
                self.assertTrue(classify.classify(f)[1].endswith("; no usage recorded"))
                self.assertFalse(classify.validity(classify.classify(f)[0], usage="unmeasured").tokens)

    def test_a_provider_error_the_verifier_scored_keeps_its_score(self) -> None:
        # Harbor's phrases also match the agent's own tool output (two verified passes, 2026-11-01), and a
        # stalled provider fails the task like any other failure.
        self.assertEqual(classify.classify(facts(exception_type="NetworkConnectionError", reward=1.0)),
                         ("passed", "reward 1 after NetworkConnectionError"))
        self.assertEqual(classify.classify(facts(exception_type="ApiRateLimitError", reward=0.0))[0], "failed")

    def test_a_provider_error_before_the_agent_started_is_setup(self) -> None:
        # The harness install could not resolve github.com (three Grok trials, 2026-11-02): no model was called.
        f = facts(exception_type="NetworkConnectionError", agent_started=0, n_input=None, n_output=None,
                  agent_exec_sec=None, reward=None)
        self.assertEqual(classify.classify(f), ("setup_failure", "NetworkConnectionError before the agent started"))
        self.assertEqual(classify.classify(facts(exception_type="NetworkConnectionError", reward=None))[0],
                         "provider_failure")

    def test_a_provider_failure_after_real_work_is_unmeasured(self) -> None:
        # The exception type alone makes it a provider failure, so a harness
        # may have worked for 40 minutes before a rate limit stopped it.
        for exec_sec, status in ((5.0, "full"), (59.9, "full"), (60.0, "unmeasured"), (2400.0, "unmeasured"),
                                 (None, "unmeasured")):
            with self.subTest(exec_sec=exec_sec):
                f = facts(exception_type="ApiRateLimitError", n_input=None, n_cache=None, n_output=None,
                          agent_exec_sec=exec_sec, reward=None)
                outcome, reason = classify.classify(f)
                self.assertEqual(outcome, "provider_failure")
                self.assertEqual(classify.usage_status(f), status)
                self.assertEqual(reason.endswith("; tokens not reported"), status == "unmeasured")
        never = facts(exception_type="ApiRateLimitError", n_input=None, n_output=None, agent_started=0,
                      agent_exec_sec=None, reward=None)
        self.assertEqual(classify.usage_status(never), "full")

    def test_reasons_say_when_nothing_was_recorded(self) -> None:
        zero = dict(n_input=0, n_cache=0, n_output=0)
        null = dict(n_input=None, n_cache=None, n_output=None)
        self.assertEqual(classify.classify(facts(reward=0.0, **zero)), ("failed", "reward 0; no usage recorded"))
        self.assertEqual(classify.classify(facts(reward=0.0, **null)), ("failed", "reward 0; tokens not reported"))
        self.assertEqual(classify.classify(facts(reward=None, **zero)),
                         ("verifier_failure", "no reward recorded; no usage recorded"))
        # Every finished branch that leaves the usage unmeasured says so.
        self.assertEqual(classify.classify(facts(exception_type="VerifierTimeoutError", **zero)),
                         ("verifier_failure", "VerifierTimeoutError: the verifier failed; no usage recorded"))
        self.assertEqual(classify.classify(facts(exception_type="CancelledError", **zero)),
                         ("infra", "CancelledError (cancelled); no usage recorded"))
        self.assertEqual(classify.classify(facts(exception_type="NonZeroAgentExitCodeError", agent_exec_sec=600,
                                                 **null)),
                         ("agent_crash", "NonZeroAgentExitCodeError after using the model; tokens not reported"))
        # A trial that never reached the model, or recorded usage, says nothing more.
        self.assertEqual(classify.classify(facts(exception_type="CancelledError", agent_started=0, **zero)),
                         ("infra", "CancelledError (cancelled)"))
        self.assertEqual(classify.classify(facts(exception_type="NonZeroAgentExitCodeError", **zero)),
                         ("setup_failure", "no model tokens"))
        self.assertEqual(classify.classify(facts(exception_type="VerifierTimeoutError")),
                         ("verifier_failure", "VerifierTimeoutError: the verifier failed"))

    def test_validity_gates_tokens_only(self) -> None:
        for outcome in classify.OUTCOMES:
            with self.subTest(outcome):
                full = classify.validity(outcome)
                lost = classify.validity(outcome, usage="unmeasured")
                self.assertEqual(lost, classify.Validity(False, full.reward, full.included))
                self.assertEqual(classify.validity(outcome, usage="lower_bound"), full)
        with self.assertRaises(ValueError):
            classify.validity("passed", usage="partial")


class Versions(BenchTestCase):
    def test_versions_are_integers(self) -> None:
        self.assertIsInstance(classify.CLASSIFIER_VERSION, int)
        self.assertEqual(len(set(classify.OUTCOMES)), 11)

    def test_unrecorded_usage_rules_bumped_the_versions(self) -> None:
        # Version 2 judges usage (classifier) and recovers killed trials' usage
        # (parser), so ledgers written before it re-derive on open.
        self.assertGreaterEqual(classify.CLASSIFIER_VERSION, 2)
        self.assertGreaterEqual(extract.PARSER_VERSION, 2)
        # Version 3: outcomes that mean the agent ran are unmeasured without
        # their timing, a long provider failure too, and every such reason says
        # so (classifier); Terminus-2's subagents are counted once (parser).
        self.assertGreaterEqual(classify.CLASSIFIER_VERSION, 3)
        self.assertGreaterEqual(extract.PARSER_VERSION, 3)
        # Version 4: per-model usage adds up to Harbor's totals, so the
        # mini-swe-agent call Harbor attributed to no model is priced (parser).
        self.assertGreaterEqual(extract.PARSER_VERSION, 4)


if __name__ == "__main__":
    unittest.main()
