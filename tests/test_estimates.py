"""bench.estimates: cost and time estimates for a design from ledger figures.

Rows are synthetic analysis rows (report.analysis_rows' shape), so every
basis in the chain can be forced and checked against hand computation:
measured, measured_any_options, borrowed (repriced at this model's price)
and the three priors. Harness schemas come from tests/fixtures/schemas and
task metadata from a fixed table, so nothing reads a snapshot or Docker.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any
from unittest import mock

from tests.support import BenchTestCase

from bench import checks, datasets, designs, estimates, pricing, report, schemas, settings
from bench.spec import ArmSpec

EVIL = "evil.module:Agent"

GLM = ("openrouter", "z-ai/glm-5.3")
DEEPSEEK = ("openrouter", "deepseek/deepseek-v4.1-flash")
OPUS = ("anthropic", "claude-opus-5-5")

SAMPLE = [f"task-{i:02d}" for i in range(20)]
META = {t: {"difficulty": ("easy", "medium", "hard")[i % 3], "category": "test", "agent_timeout_sec": 900.0,
            "memory_gb": 2.0, "cpus": 1, "expert_time_estimate_min": 30.0} for i, t in enumerate(SAMPLE)}

PRICING_YAML = """\
anthropic:
  claude-opus-5-5: {input: 5.00, output: 25.00, cache_read: 0.50}
openrouter:
  glm-5.3: {input: 0.07, output: 1.40, cache_read: 0.069}
  deepseek-v4.1-flash: {input: 0.045, output: 0.50, cache_read: 0.006}
"""

REFERENCE_YAML = """\
defaults:
  direction: harness_over_baseline
  estimator: ratio_of_means
  tolerance: 1.5
  applies_to: ['anthropic:claude-*']
consensus:
  cost_per_trial: {low: 1/3, high: 1/2, source: "S1-S5"}
  pass_rate_delta_pp: {low: -5, high: 5, source: "S1-S5"}
arms:
  pi:
    tokens_per_trial: {value: 1/3.1, source: "S10 via S9"}
  terminus-2:
    note: "no cost figures"
"""
# The same figures as written before multiples were harness / baseline.
REFERENCE_YAML_BASELINE_OVER_HARNESS = (REFERENCE_YAML.replace("harness_over_baseline", "baseline_over_harness")
                                        .replace("{low: 1/3, high: 1/2,", "{low: 2.0, high: 3.0,")
                                        .replace("{value: 1/3.1,", "{value: 3.1,"))

CONFIG_YAML = """\
dataset: terminal-bench@2.0
providers:
  anthropic: {env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}
  openrouter: {env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}
modes:
  pilot: {n_tasks: 1}
run: {budget_usd: 5, effort: medium, n_concurrent: 2, parallel_arms: 2, per_trial_cap_usd: 1, max_turns: 150}
baseline_arm: claude-code-glm
designs: {max_budget_usd: 50}
arms:
  - {name: claude-code-glm, agent: claude-code, provider: openrouter, model: z-ai/glm-5.3}
  - {name: pi-glm, agent: pi, provider: openrouter, model: z-ai/glm-5.3}
  - {name: terminus-2-glm, agent: terminus-2, provider: openrouter, model: z-ai/glm-5.3}
  - name: mini-glm
    agent: bench.agents.mini_swe_agent:MiniSweAgentWithMcp
    family: mini-swe-agent
    provider: openrouter
    model: z-ai/glm-5.3
  - {name: claude-code-opus, agent: claude-code, provider: anthropic, model: claude-opus-5-5}
"""

_SEQ = iter(range(1, 10 ** 9))


def arow(task: str, family: str, cost: float | None, *, where: tuple[str, str] = GLM, config_hash: str = "c" * 16,
         tokens: tuple[int, int, int] | None = (100000, 60000, 5000), outcome: str = "passed",
         minutes: float | None = 10.0, digest: str | None = None, run_key: str = "dev/r1",
         excluded_reason: str | None = None, exception_type: str | None = None, reported: float | None = None,
         effort: str | None = "medium") -> dict[str, Any]:
    """An analysis row (report.analysis_rows) for one trial."""
    seq = next(_SEQ)
    infra = outcome in report.RELIABILITY_OUTCOMES
    manual = bool(excluded_reason and excluded_reason.startswith(("excluded by hand", "run excluded by hand")))
    measured = tokens is not None and not infra
    n_input, n_cache, n_output = tokens if tokens is not None else (None, None, None)
    return {
        "trial_key": f"job-{family}-{seq}/{task}__{seq}", "run_key": run_key, "profile": run_key.split("/")[0],
        "run_id": run_key.split("/")[-1], "arm": family, "family": family, "implementation": "builtin",
        "config_hash": config_hash, "config_label": f"{family} {config_hash[:6]}", "provider": where[0],
        "model": where[1], "effort": effort, "options": {"max_turns": 150}, "task": task,
        "task_digest": digest or "sha256:" + hashlib.sha256(task.encode()).hexdigest(), "outcome": outcome,
        "outcome_reason": "test", "harbor_cause": None, "exception_type": exception_type,
        "valid_tokens": measured and not manual, "valid_cost": measured and cost is not None and not manual,
        "valid_reward": not infra and not manual, "excluded_reason": excluded_reason,
        "n_input": n_input if measured else None, "n_cache": n_cache if measured else None,
        "n_output": n_output if measured else None,
        "tokens_total": (n_input + n_output) if measured else None,
        "tokens_uncached": (n_input - n_cache + n_output) if measured else None,
        "cost_usd": cost if measured else None, "reported_cost_usd": reported, "cost_at_ingest_usd": cost,
        "passed": outcome == "passed" if not infra else None, "agent_minutes": minutes, "total_minutes": minutes,
        "started_at": f"2026-10-29T06:{seq % 60:02d}:00.000000Z",
    }


def paired(family: str, base: str, ratios: Sequence[float], *, tasks: Sequence[str] = SAMPLE,
           where: tuple[str, str] = GLM, base_cost: float = 0.3) -> list[dict[str, Any]]:
    """Rows where the baseline costs ratios[i] times the family on task i: a multiple (family / baseline) of
    1 / ratios[i]."""
    rows = []
    for i, k in enumerate(ratios):
        b = base_cost * (1 + 0.1 * i)
        rows.append(arow(tasks[i], base, b, where=where))
        rows.append(arow(tasks[i], family, b / k, where=where))
    return rows


def arm_spec(name: str, agent: str, *, family: str | None = None, where: tuple[str, str] = GLM,
             tasks: Sequence[str] = (), options: Mapping[str, Any] | None = None) -> ArmSpec:
    kind = "import_path" if ":" in agent else "name"
    return ArmSpec(name=name, agent=agent, family=family or agent, kind=kind, provider=where[0], model=where[1],
                   effort="inherit", options=dict(options or {}), tasks=tuple(tasks))


class EstimateCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cfg = self.load_test_config(CONFIG_YAML, pricing=PRICING_YAML, reference=REFERENCE_YAML, tasks=SAMPLE)
        for target, name, value in ((datasets, "task_meta", lambda dataset: META),
                                    (datasets, "info", lambda dataset: None),
                                    (datasets, "path", lambda dataset: None),
                                    (checks, "docker_capacity", lambda: {"ncpu": 8, "mem_gb": 16.0})):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def spec(self, arms: list[dict[str, Any]], tasks: Sequence[str] = SAMPLE[:4], budget: float = 5.0,
             **run: Any) -> dict[str, Any]:
        return designs.normalize(self.cfg, {"tasks": list(tasks), "run": {"budget_usd": budget, **run},
                                            "arms": arms})

    def estimate(self, spec: dict[str, Any], rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        check = designs.validate(self.cfg, spec)
        self.assertEqual(check["problems"], [])
        return estimates.estimate(self.cfg, spec, rows, check["plan"], **kw)

    def arm_hash(self, spec: dict[str, Any], name: str) -> str:
        """The config hash the ledger would give the named arm's trials."""
        plan = designs.validate(self.cfg, spec)["plan"]
        entry = next(a for a in plan["arms"] if a["name"] == name)
        arm = next(a for a in estimates.spec_from_design(spec, None).arms if a.name == name)
        kwargs = {k["name"]: k["value"] for k in entry["kwargs"]}
        return estimates.config_hash_for(arm, kwargs, spec["run"]["timeout_multiplier"])


ARM_PI = {"name": "pi", "agent": "pi", "provider": "openrouter", "model": "z-ai/glm-5.3"}
ARM_CC = {"name": "cc", "agent": "claude-code", "provider": "openrouter", "model": "z-ai/glm-5.3"}


class Determinism(EstimateCase):
    def test_identical_inputs_give_byte_identical_output(self) -> None:
        rows = paired("pi", "claude-code", [3.0, 2.2, 4.1, 2.8, 3.3])
        spec = self.spec([ARM_CC, ARM_PI], tasks=SAMPLE[:8])
        first = self.estimate(spec, rows)
        again = self.estimate(json.loads(json.dumps(spec)), [dict(r) for r in rows])
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(again, sort_keys=True))
        fingerprint = report.ledger_fingerprint(rows)
        self.assertEqual(first["ledger_fingerprint"], fingerprint)
        expected = int(hashlib.sha256((first["spec_hash"] + fingerprint).encode()).hexdigest()[:12], 16)
        self.assertEqual(first["seed"], expected)
        self.assertEqual(first["replicates"], 2000)

    def test_another_ledger_changes_the_seed(self) -> None:
        rows = paired("pi", "claude-code", [3.0, 2.2, 4.1])
        spec = self.spec([ARM_CC, ARM_PI])
        a = self.estimate(spec, rows, replicates=200)
        b = self.estimate(spec, rows + [arow("task-09", "pi", 0.05)], replicates=200)
        self.assertNotEqual(a["seed"], b["seed"])


class BasisChain(EstimateCase):
    def model(self, rows: list[dict[str, Any]], arm: ArmSpec, config_hash: str | None = "x" * 16
              ) -> estimates.CellModel:
        return estimates.cell_model(self.cfg, rows, arm, config_hash=config_hash, meta=META)

    def test_measured_needs_three_tasks_of_this_configuration(self) -> None:
        spec = self.spec([ARM_PI], tasks=SAMPLE[:3])
        chash = self.arm_hash(spec, "pi")
        rows = [arow(t, "pi", 0.05 * (i + 1), config_hash=chash) for i, t in enumerate(SAMPLE[:3])]
        est = self.estimate(spec, rows)
        arm = est["arms"][0]
        self.assertEqual((arm["basis"], arm["confidence"]), ("measured", "medium"))
        self.assertEqual(arm["cell"]["config_hash"], chash)
        self.assertEqual(arm["evidence"]["design_tasks_observed"], 3)
        two = self.estimate(spec, rows[:2])
        self.assertEqual(two["arms"][0]["basis"], "measured_any_options")  # 2 tasks: the family level

    def test_eight_measured_tasks_are_high_confidence(self) -> None:
        arm = arm_spec("pi", "pi")
        rows = [arow(t, "pi", 0.05, config_hash="h" * 16) for t in SAMPLE[:8]]
        m = self.model(rows, arm, config_hash="h" * 16)
        self.assertEqual((m.basis, m.confidence, m.n_tasks), ("measured", "high", 8))

    def test_other_options_fall_back_to_the_family_on_this_model(self) -> None:
        arm = arm_spec("pi", "pi")
        rows = [arow(t, "pi", 0.04, config_hash="o" * 16) for t in SAMPLE[:3]]
        m = self.model(rows, arm)
        self.assertEqual((m.basis, m.confidence), ("measured_any_options", "medium"))
        one = self.model(rows[:1], arm)
        self.assertEqual((one.basis, one.confidence), ("measured_any_options", "low"))

    def test_borrowed_reprices_another_models_tokens(self) -> None:
        arm = arm_spec("pi", "pi")
        tokens = (400000, 300000, 20000)
        rows = [arow("task-00", "pi", 0.01, where=DEEPSEEK, tokens=tokens)]
        m = self.model(rows, arm)
        self.assertEqual((m.basis, m.confidence), ("borrowed", "low"))
        price = self.cfg.price(*GLM)
        expected = ((400000 - 300000) * price.input + 300000 * price.cache_read + 20000 * price.output) / 1e6
        self.assertAlmostEqual(m.per_task["task-00"][0], expected, places=12)
        self.assertAlmostEqual(expected, pricing.unit_cost(price, *tokens), places=12)
        self.assertIn("deepseek", m.detail)

    def test_borrowed_prefers_the_same_provider(self) -> None:
        arm = arm_spec("pi", "pi")
        rows = [arow("task-00", "pi", 1.0, where=OPUS, tokens=(1000, 0, 100)),
                arow("task-01", "pi", 0.01, where=DEEPSEEK, tokens=(2000, 0, 200))]
        m = self.model(rows, arm)
        self.assertEqual(sorted(m.per_task), ["task-01"])

    def test_prior_from_the_baseline_and_the_reference_multiple(self) -> None:
        arm = arm_spec("pi", "pi")
        rows = [arow(t, "claude-code", 0.62) for t in SAMPLE[:3]]
        m = self.model(rows, arm)
        self.assertEqual((m.basis, m.confidence), ("prior", "low"))
        self.assertTrue(all(math.isclose(v, 0.62 / 3.1) for v in m.pool))
        self.assertIn("claude-code on openrouter:z-ai/glm-5.3 x 0.323 (reference pi tokens_per_trial)", m.detail)

    def test_a_reference_written_either_way_predicts_the_same(self) -> None:
        # The figures restated as harness / baseline (1/3.1, 1/3-1/2) predict exactly what 3.1 and 2-3 did.
        old = self.load_test_config(CONFIG_YAML, pricing=PRICING_YAML, reference=REFERENCE_YAML_BASELINE_OVER_HARNESS,
                                    tasks=SAMPLE)
        rows = [arow(t, "claude-code", 0.62 + 0.01 * i) for i, t in enumerate(SAMPLE[:3])]
        for agent in ("pi", "terminus-2"):
            with self.subTest(agent):
                arm = arm_spec(agent, agent)
                new_pool = self.model(rows, arm).pool
                old_pool = estimates.cell_model(old, rows, arm, config_hash="x" * 16, meta=META).pool
                self.assertEqual(new_pool, old_pool)  # bitwise
        self.assertEqual(estimates.reference_multiple(old.reference, "pi"),
                         estimates.reference_multiple(self.cfg.reference, "pi"))
        self.assertEqual(estimates.reference_multiple(self.cfg.reference, "terminus-2"),
                         (1 / 2.5, "reference consensus cost_per_trial"))
        self.assertEqual(estimates.reference_multiple({}, "pi"), (1 / 2.5, "the published 0.33x-0.5x, midpoint 0.4x"))
        # No defaults.direction: the figures are the old baseline / harness ones, so 2.1 plans at 1/2.1.
        undirected = {"consensus": {"cost_per_trial": {"low": 2.0, "high": 3.0}},
                      "arms": {"strands": {"cost_per_trial": {"value": 2.1}}}}
        self.assertEqual(estimates.reference_multiple(undirected, "strands")[0], 1 / 2.1)
        self.assertEqual(estimates.reference_multiple(undirected, "pi")[0], 1 / 2.5)

    def test_prior_uses_the_consensus_midpoint_without_a_figure(self) -> None:
        arm = arm_spec("t2", "terminus-2")
        rows = [arow(t, "claude-code", 0.5) for t in SAMPLE[:3]]
        m = self.model(rows, arm)
        self.assertTrue(all(math.isclose(v, 0.5 / 2.5) for v in m.pool))

    def test_prior_from_other_families_without_a_baseline(self) -> None:
        arm = arm_spec("t2", "terminus-2")
        rows = [arow("task-00", "pi", 0.1), arow("task-01", "pi", 0.3), arow("task-02", "mini-swe-agent", 0.2)]
        m = self.model(rows, arm)
        self.assertEqual(m.basis, "prior")
        self.assertEqual(m.pool, [0.2])
        self.assertIn("median", m.detail)

    def test_prior_from_default_trial_tokens(self) -> None:
        arm = arm_spec("t2", "terminus-2")
        m = self.model([], arm)
        self.assertEqual((m.basis, m.confidence), ("prior", "low"))
        price = self.cfg.price(*GLM)
        self.assertAlmostEqual(m.pool[0], pricing.unit_cost(price, 600000, 510000, 60000), places=12)

    def test_unpriced_model_has_no_cost(self) -> None:
        arm = arm_spec("pi", "pi", where=("openrouter", "unpriced/model-x"))
        m = self.model([], arm)
        self.assertTrue(m.unpriced)
        self.assertEqual(m.pool, [])

    def test_a_trial_with_no_usage_recorded_adds_time_and_is_drawn_high(self) -> None:
        # Pilot 3's pi-glm: a 15-minute timeout before any model call completed.
        # Left out, the task it timed out on would be priced like a typical one.
        arm = arm_spec("pi", "pi")
        real = [arow(SAMPLE[0], "pi", 0.1332, minutes=9.45), arow(SAMPLE[1], "pi", 0.05, minutes=4.0)]
        lost = dict(arow(SAMPLE[2], "pi", None, tokens=None, outcome="agent_limit", minutes=16.0,
                         exception_type="AgentTimeoutError"), usage_status="unmeasured")
        m = self.model(real + [lost], arm)
        self.assertEqual(m.basis, "measured_any_options")
        self.assertEqual(m.per_task, {SAMPLE[0]: [0.1332], SAMPLE[1]: [0.05], SAMPLE[2]: [0.1332]})
        self.assertEqual(m.n_tasks, 2)  # the confidence counts measured tasks only
        self.assertEqual(m.notes, ["1 trial that ran but recorded no usage is drawn at the most expensive measured "
                                   "trial ($0.1332): such trials often ran to the time limit."])
        self.assertEqual(sorted(m.minutes_pool), [4.0, 9.45, 16.0])
        cells = estimates.cell_stats(self.cfg, real + [lost], family="pi", match="family")
        self.assertEqual((cells[0]["cost_per_trial"]["n"], cells[0]["minutes_per_trial"]["total"]["n"]), (2, 3))

    def test_infrastructure_failures_and_hand_exclusions_are_not_costs(self) -> None:
        arm = arm_spec("pi", "pi")
        rows = [arow("task-00", "pi", 0.05), arow("task-01", "pi", None, outcome="setup_failure"),
                arow("task-02", "pi", 9.0, excluded_reason="excluded by hand: broken")]
        m = self.model(rows, arm)
        self.assertEqual(sorted(m.per_task), ["task-00"])
        self.assertAlmostEqual(m.p_infra, 0.5)


class Caps(EstimateCase):
    def test_enforced_caps_truncate_draws(self) -> None:
        mini = {"name": "mini", "agent": "bench.agents.mini_swe_agent:MiniSweAgentWithMcp",
                "provider": "openrouter", "model": "z-ai/glm-5.3"}
        rows = [arow(t, "mini-swe-agent", 5.0) for t in SAMPLE[:4]]
        est = self.estimate(self.spec([mini]), rows)
        arm = est["arms"][0]
        self.assertEqual(arm["cap_usd"], 1.0)
        self.assertLessEqual(arm["per_trial"]["p90"], 1.0 * estimates.CAP_SLACK + 1e-9)
        self.assertEqual(est["run"]["uncapped_arms"], [])
        self.assertAlmostEqual(est["run"]["worst_case_capped_usd"], 4.0)

    def test_a_mispriced_cap_never_bounds_the_estimate(self) -> None:
        rows = [arow(t, "claude-code", 5.0) for t in SAMPLE[:4]]
        est = self.estimate(self.spec([ARM_CC]), rows)
        arm = est["arms"][0]
        self.assertIsNone(arm["cap_usd"])  # the run cap is not passed to claude-code off Anthropic
        self.assertGreater(arm["per_trial"]["p90"], 2.0)
        own = dict(ARM_CC, options={"max_budget_usd": "1"})
        est = self.estimate(self.spec([own]), rows)
        self.assertIsNone(est["arms"][0]["cap_usd"])
        self.assertTrue(any("another provider's rates" in n for n in est["arms"][0]["notes"]))
        self.assertEqual(est["run"]["uncapped_arms"], ["cc"])

    def test_claude_code_on_anthropic_enforces_its_cap(self) -> None:
        arm = {"name": "cc", "agent": "claude-code", "provider": "anthropic", "model": "claude-opus-5-5"}
        rows = [arow(t, "claude-code", 5.0, where=OPUS) for t in SAMPLE[:4]]
        est = self.estimate(self.spec([arm]), rows)
        self.assertEqual(est["arms"][0]["cap_usd"], 1.0)
        self.assertLessEqual(est["arms"][0]["per_trial"]["p90"], 1.1 + 1e-9)


class Budget(EstimateCase):
    def test_p_within_budget_is_monotone_in_budget(self) -> None:
        rows = paired("pi", "claude-code", [3.0, 2.0, 4.0, 2.5, 3.5])
        p50 = self.estimate(self.spec([ARM_CC, ARM_PI], tasks=SAMPLE[:6]), rows)["run"]["total"]["p50"]
        chances = []
        for factor in (0.3, 0.8, 1.0, 1.4, 3.0):
            est = self.estimate(self.spec([ARM_CC, ARM_PI], tasks=SAMPLE[:6], budget=round(p50 * factor, 4)), rows)
            chances.append(est["run"]["p_within_budget"])
        self.assertEqual(chances, sorted(chances))
        self.assertLess(chances[0], 0.2)
        self.assertGreater(chances[-1], 0.95)

    def test_suggested_budget_rounds_p90_x_1_15_up_to_50_cents(self) -> None:
        self.assertEqual(estimates.suggested_budget(1.0), 1.5)
        self.assertEqual(estimates.suggested_budget(0.01), 0.5)
        self.assertEqual(estimates.suggested_budget(2.0 / 1.15), 2.0)
        self.assertEqual(estimates.suggested_budget(2.2), 3.0)
        self.assertIsNone(estimates.suggested_budget(None))
        est = self.estimate(self.spec([ARM_PI]), [arow(t, "pi", 0.2) for t in SAMPLE[:4]])
        self.assertEqual(est["run"]["suggested_budget_usd"], estimates.suggested_budget(est["run"]["total"]["p90"]))

    def test_inflight_allowance_formula(self) -> None:
        spec = {"tasks": SAMPLE[:5], "attempts": 2, "arms": [{"tasks": None}, {"tasks": SAMPLE[:1]}],
                "run": {"parallel_arms": 2, "n_concurrent": 3, "schedule": "arms"}}
        est = {"arms": [{"cap_usd": 1.0, "per_trial": {"p90": 0.9}}, {"cap_usd": None, "per_trial": {"p90": 0.7}}]}
        # min(2 x 3, 12 trials) busy x max(1.0 x 1.1, 2 x 0.7)
        self.assertAlmostEqual(estimates.inflight_allowance(spec, None, est), 6 * 1.4)
        small = {"tasks": SAMPLE[:1], "attempts": 1, "arms": [{"tasks": None}],
                 "run": {"parallel_arms": 4, "n_concurrent": 8, "schedule": "arms"}}
        self.assertAlmostEqual(estimates.inflight_allowance(small, None, {"arms": [{"cap_usd": 2.0}]}), 2.2)
        # A task race: the busiest task's arms, at most max_containers: 2 arms x min(3, 2 attempts) on task 1.
        race = {**spec, "run": {"parallel_arms": 2, "n_concurrent": 3, "schedule": "tasks", "max_containers": 3}}
        self.assertAlmostEqual(estimates.inflight_allowance(race, None, est), 3 * 1.4)
        rows = [arow(t, "pi", 0.2) for t in SAMPLE[:4]]
        normalized = self.spec([ARM_CC, ARM_PI])
        got = self.estimate(normalized, rows)
        self.assertAlmostEqual(got["run"]["inflight_allowance_usd"],
                               estimates.inflight_allowance(normalized, None, got))

    def test_infrastructure_failure_rates_are_flagged(self) -> None:
        rows = [arow(t, "pi", None, outcome="setup_failure", tokens=None) for t in SAMPLE[:3]]
        rows.append(arow("task-03", "pi", 0.1))
        est = self.estimate(self.spec([ARM_PI]), rows)
        self.assertAlmostEqual(est["arms"][0]["p_infra"], 0.75)
        self.assertTrue(any("fix before spending" in n for n in est["arms"][0]["notes"]))
        self.assertTrue(est["warnings"])

    def test_wall_time_and_containers(self) -> None:
        rows = [arow(t, "pi", 0.1, minutes=20.0) for t in SAMPLE[:4]]
        est = self.estimate(self.spec([ARM_CC, ARM_PI], n_concurrent=2, parallel_arms=2, schedule="arms"), rows)
        self.assertEqual(est["run"]["peak_containers"], 4)
        self.assertGreater(est["run"]["wall_minutes"]["p50"], estimates.PREFLIGHT_MINUTES)
        pi = next(a for a in est["arms"] if a["arm"] == "pi")
        self.assertGreaterEqual(pi["arm_minutes"]["p50"], 1.0 + 20.0)  # 4 trials of ~20 min on 2 slots


class Pairing(EstimateCase):
    def test_ci_factor_narrows_as_pairs_grow(self) -> None:
        rows = paired("pi", "claude-code", [3.0, 1.8, 4.5, 2.4, 3.6])
        est = self.estimate(self.spec([ARM_CC, ARM_PI], tasks=SAMPLE[:12]), rows)
        (pair,) = est["pairing"]
        self.assertEqual((pair["arm"], pair["metric"], pair["pairs_now"], pair["pairs_after"]), ("pi", "cost", 5, 12))
        self.assertEqual(pair["baseline_cell"]["arm"], "cc")
        self.assertEqual(pair["sd_source"], "observed")
        self.assertLess(pair["ci_factor_after"], pair["ci_factor_now"])
        self.assertIsNotNone(pair["multiple_now"])
        self.assertEqual(pair["multiple_now"]["n"], 5)
        # pi / claude-code: the geometric mean of 1/3.0, 1/1.8, ... (pi is the cheaper side).
        expected = math.exp(sum(math.log(1 / k) for k in (3.0, 1.8, 4.5, 2.4, 3.6)) / 5)
        self.assertAlmostEqual(pair["multiple_now"]["value"], expected, places=4)
        lo, hi = pair["multiple_now"]["ci"]
        self.assertTrue(lo < pair["multiple_now"]["value"] < hi < 1.0)

    def test_the_ledger_baseline_pairs_when_the_design_has_none(self) -> None:
        rows = paired("pi", "claude-code", [3.0, 2.0])
        est = self.estimate(self.spec([ARM_PI], tasks=SAMPLE[:4]), rows)
        (pair,) = est["pairing"]
        self.assertIsNone(pair["baseline_cell"]["arm"])
        self.assertEqual((pair["pairs_now"], pair["pairs_after"]), (2, 2))  # no baseline trials are added
        self.assertIsNone(pair["multiple_now"])  # under 3 pairs no number is shown
        self.assertEqual(pair["sd_source"], "prior")

    def test_spread_is_pooled_from_other_families_on_the_model(self) -> None:
        rows = paired("terminus-2", "claude-code", [2.0, 3.0, 2.5, 4.0])
        rows += paired("pi", "claude-code", [3.0], tasks=SAMPLE[10:])
        est = self.estimate(self.spec([ARM_CC, ARM_PI], tasks=SAMPLE[10:14]), rows)
        self.assertEqual(est["pairing"][0]["sd_source"], "pooled_model")


class QueueAdapter(EstimateCase):
    def test_queue_estimate_shape(self) -> None:
        rows = [arow(t, "pi", 0.2) for t in SAMPLE[:4]]
        spec = self.spec([ARM_PI])
        got = estimates.queue_estimate(self.cfg, spec, rows)
        self.assertEqual(set(got), {"expected_usd", "p90_usd", "inflight_allowance_usd"})
        est = self.estimate(spec, rows)
        self.assertEqual(got["expected_usd"], est["run"]["total"]["p50"])
        self.assertEqual(got["inflight_allowance_usd"], est["run"]["inflight_allowance_usd"])

    def test_queue_estimate_never_raises(self) -> None:
        def broken() -> list[dict[str, Any]]:
            raise RuntimeError("ledger busy")
        spec = self.spec([dict(ARM_PI, name="mini", agent="bench.agents.mini_swe_agent:MiniSweAgentWithMcp")],
                         schedule="arms")
        got = estimates.queue_estimate(self.cfg, spec, broken)
        self.assertIsNone(got["expected_usd"])
        self.assertAlmostEqual(got["inflight_allowance_usd"], 2 * 1.1)  # 2 busy trials x the $1 cap x 1.1
        self.assertEqual(estimates.queue_estimate(self.cfg, {"not": "a spec"}, []),
                         {"expected_usd": None, "p90_usd": None, "inflight_allowance_usd": None})


class SchemaLookups(EstimateCase):
    """Harbor imports an agent's code to print its schema, so an estimate may
    look up allowed harnesses only, and never twice for one arm."""

    def recorder(self, fail: set[str]) -> list[str]:
        calls: list[str] = []
        real = schemas.agent_schema

        def lookup(agent: str) -> dict[str, Any]:
            calls.append(agent)
            if agent in fail:
                raise AssertionError(f"looked up {agent}")
            return real(agent)
        patcher = mock.patch.object(schemas, "agent_schema", lookup)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def evil_spec(self) -> dict[str, Any]:
        spec = self.spec([ARM_PI])
        spec["arms"][0]["agent"] = EVIL  # e.g. a stored design whose import path has left the allowlist
        return spec

    def test_a_disallowed_agent_is_never_looked_up(self) -> None:
        calls = self.recorder({EVIL})
        rows = [arow(t, "pi", 0.2) for t in SAMPLE[:4]]
        spec = self.evil_spec()
        check = designs.validate(self.cfg, spec)
        self.assertEqual(check["status"], "invalid")
        self.assertIsNone(check["plan"])
        est = estimates.estimate(self.cfg, spec, rows, check["plan"])
        self.assertIsNotNone(est["run"]["total"]["p50"])
        self.assertIsNone(est["arms"][0]["cell"]["config_hash"])
        self.assertEqual(est["arms"][0]["basis"], "measured_any_options")
        got = estimates.queue_estimate(self.cfg, spec, rows)
        self.assertIsNotNone(got["expected_usd"])
        self.assertIsNone(estimates.planned_kwargs(self.cfg, estimates.spec_from_design(spec, None).arms[0]))
        self.assertNotIn(EVIL, calls)

    def test_a_failed_lookup_is_not_repeated(self) -> None:
        calls = self.recorder({"pi"})
        rows = [arow(t, "pi", 0.2) for t in SAMPLE[:4]]
        est = estimates.estimate(self.cfg, self.spec([ARM_PI]), rows, None)
        self.assertEqual(calls, ["pi"])
        self.assertEqual(est["arms"][0]["basis"], "measured_any_options")


class Bounded(EstimateCase):
    def test_replicates_are_cut_for_large_designs(self) -> None:
        self.assertEqual(estimates.effective_replicates(2000, 16), 2000)
        self.assertEqual(estimates.effective_replicates(2000, 1200), estimates.MAX_DRAWS // 1200)
        self.assertEqual(estimates.effective_replicates(2000, 10 ** 6), estimates.MIN_REPLICATES)
        self.assertEqual(estimates.effective_replicates(150, 10 ** 6), 150)  # never more than asked
        rows = [arow(t, "pi", 0.2) for t in SAMPLE[:4]]
        spec = self.spec([ARM_CC, ARM_PI], tasks=SAMPLE[:4])  # 8 trials
        with mock.patch.object(estimates, "MAX_DRAWS", 4000):
            first = self.estimate(spec, rows)
            again = self.estimate(spec, rows)
        self.assertEqual(first["replicates"], 500)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(again, sort_keys=True))

    def test_per_trial_draws_are_kept_for_the_first_replicates_only(self) -> None:
        arm = arm_spec("pi", "pi", tasks=SAMPLE[:3])
        model = estimates.cell_model(self.cfg, [arow(t, "pi", 0.2) for t in SAMPLE[:3]], arm, config_hash="x" * 16,
                                     meta=META)
        sim = estimates._sim(arm, model, None, 2, META, 1.0)
        run = estimates.spec_from_design(self.spec([ARM_PI]), None).run
        totals, walls = estimates._simulate([sim], run, 500, random.Random(1), 0.6)
        self.assertEqual((len(totals), len(walls), len(sim.totals), len(sim.arm_minutes)), (500, 500, 500, 500))
        self.assertEqual(len(sim.costs), estimates.TRIAL_DRAW_REPLICATES * 6)
        self.assertEqual(len(sim.trial_minutes), estimates.TRIAL_DRAW_REPLICATES * 6)


class DifficultyPools(EstimateCase):
    def test_a_difficulty_pool_needs_two_observed_tasks(self) -> None:
        arm = arm_spec("pi", "pi")
        # task-00 and task-03 are easy, task-01 medium, task-02 hard (META).
        rows = [arow("task-00", "pi", 1.0), arow("task-00", "pi", 1.2), arow("task-01", "pi", 5.0),
                arow("task-02", "pi", 9.0)]
        m = estimates.cell_model(self.cfg, rows, arm, config_hash="x" * 16, meta=META)
        pick = estimates._choices("task-03", m.per_task, m.by_difficulty, m.pool, META)
        self.assertEqual(sorted(pick), sorted(m.pool))  # one easy task's repeats are not an easy pool
        m = estimates.cell_model(self.cfg, rows + [arow("task-06", "pi", 2.0)], arm, config_hash="x" * 16,
                                 meta=META)
        pick = estimates._choices("task-03", m.per_task, m.by_difficulty, m.pool, META)
        self.assertEqual(sorted(pick), [1.0, 1.2, 2.0])
        self.assertEqual(estimates._choices("task-00", m.per_task, m.by_difficulty, m.pool, META), [1.0, 1.2])


class PairingRule(EstimateCase):
    """Pairs count as the REPORT tab counts them: report.match_on and
    report.same_run_only decide which trials pair."""

    def config(self, report_yaml: str) -> settings.Config:
        return self.load_test_config(CONFIG_YAML + report_yaml, pricing=PRICING_YAML, reference=REFERENCE_YAML,
                                     tasks=SAMPLE)

    def pairing(self, cfg: settings.Config, rows: list[dict[str, Any]], tasks: Sequence[str]) -> dict[str, Any]:
        spec = designs.normalize(cfg, {"tasks": list(tasks), "run": {"budget_usd": 5.0}, "arms": [ARM_CC, ARM_PI]})
        check = designs.validate(cfg, spec)
        self.assertEqual(check["problems"], [])
        (pair,) = estimates.estimate(cfg, spec, rows, check["plan"], replicates=200)["pairing"]
        return pair

    def test_match_on_name_pairs_across_task_versions(self) -> None:
        rows = []
        for i, t in enumerate(SAMPLE[:5]):
            rows.append(arow(t, "claude-code", 0.3 + 0.01 * i, digest=f"sha256:old-{t}"))
            rows.append(arow(t, "pi", 0.1 + 0.002 * i))
        by_digest = self.pairing(self.cfg, rows, SAMPLE[5:7])
        self.assertEqual((by_digest["pairs_now"], by_digest["pairs_after"]), (0, 2))
        by_name = self.pairing(self.config("report: {match_on: name}\n"), rows, SAMPLE[5:7])
        self.assertEqual((by_name["pairs_now"], by_name["pairs_after"]), (5, 7))
        self.assertIsNotNone(by_name["multiple_now"])

    def test_old_versions_that_do_not_pair_add_nothing_unless_the_design_runs_them(self) -> None:
        rows = [arow("task-00", "claude-code", 0.3, digest="sha256:v1"), arow("task-00", "pi", 0.1)]
        rows += paired("pi", "claude-code", [3.0, 2.0, 2.5], tasks=SAMPLE[1:4])
        self.assertEqual(self.pairing(self.cfg, rows, SAMPLE[4:6])["pairs_after"], 3 + 2)
        self.assertEqual(self.pairing(self.cfg, rows, SAMPLE[:1] + SAMPLE[4:6])["pairs_after"], 3 + 3)

    def test_same_run_only_pairs_within_one_run(self) -> None:
        rows = []
        for i, t in enumerate(SAMPLE[:5]):
            rows.append(arow(t, "claude-code", 0.3 + 0.01 * i, run_key="dev/r1"))
            rows.append(arow(t, "pi", 0.1 + 0.002 * i, run_key="dev/r2" if i < 3 else "dev/r1"))
        across = self.pairing(self.cfg, rows, SAMPLE[:4])
        self.assertEqual((across["pairs_now"], across["pairs_after"]), (5, 5))
        within = self.pairing(self.config("report: {same_run_only: true}\n"), rows, SAMPLE[:4])
        # Only tasks 3 and 4 share a run; the new run adds its 4 tasks, run on both sides.
        self.assertEqual((within["pairs_now"], within["pairs_after"]), (2, 6))


    def test_pairs_count_on_the_task_scope_only(self) -> None:
        # The REPORT compares on cfg.tasks by default; the design's pairs and multiple use the same tasks.
        cfg = self.load_test_config(CONFIG_YAML, pricing=PRICING_YAML, reference=REFERENCE_YAML, tasks=SAMPLE[:6])
        rows = paired("pi", "claude-code", [3.0, 2.0, 2.5, 4.0])
        rows += paired("pi", "claude-code", [0.5, 0.25], tasks=["extra-00", "extra-01"])  # outside the scope
        spec = designs.normalize(cfg, {"tasks": SAMPLE[4:8], "run": {"budget_usd": 5.0}, "arms": [ARM_CC, ARM_PI]})
        check = designs.validate(cfg, spec)
        (pair,) = estimates.estimate(cfg, spec, rows, check["plan"], replicates=200)["pairing"]
        self.assertEqual((pair["pairs_now"], pair["pairs_after"]), (4, 6))  # task-06 and task-07 never count
        expected = math.exp(sum(math.log(1 / k) for k in (3.0, 2.0, 2.5, 4.0)) / 4)
        self.assertAlmostEqual(pair["multiple_now"]["value"], expected, places=4)
        self.assertEqual(pair["scope"], {"tasks": 6, "tasks_file": report.tasks_file(cfg), "outside": 2})

    def test_pairs_at_the_arms_own_effort_as_the_report_does(self) -> None:
        # The report gives each effort a line of its own and pairs it with the baseline's line at that effort.
        def at(effort: str | None, ratios: Sequence[float]) -> list[dict[str, Any]]:
            rows = paired("pi", "claude-code", ratios)
            for r in rows:
                r["effort"] = effort
            return rows

        def pairing(rows: list[dict[str, Any]], run: Mapping[str, Any], arms: Sequence[Mapping[str, Any]]
                    ) -> dict[str, Any]:
            spec = designs.normalize(self.cfg, {"tasks": SAMPLE[:4], "run": {"budget_usd": 5.0, **run},
                                                "arms": list(arms)})
            check = designs.validate(self.cfg, spec)
            self.assertEqual(check["problems"], [])
            (pair,) = estimates.estimate(self.cfg, spec, rows, check["plan"], replicates=200)["pairing"]
            return pair

        def multiple(ratios: Sequence[float]) -> float:
            return math.exp(sum(math.log(1 / k) for k in ratios) / len(ratios))

        medium, high = [3.0, 2.0, 2.5, 4.0], [1.0, 1.5, 1.2]
        rows = at("medium", medium) + at("high", high) + at("low", [0.5, 0.25, 0.4, 0.3, 0.2])
        for run, ratios in (({}, medium), ({"effort": "high"}, high)):
            with self.subTest(run=run):
                pair = pairing(rows, run, [ARM_CC, ARM_PI])
                self.assertEqual(pair["pairs_now"], len(ratios))
                self.assertAlmostEqual(pair["multiple_now"]["value"], multiple(ratios), places=4)
        # An effort carried in the options (opencode's variant) reads like the effort field; a harness whose
        # trials record no effort reads as the harness default.
        run = self.cfg.run
        self.assertEqual(estimates._arm_effort(run, arm_spec("oc", "opencode", options={"variant": "high"}),
                                               {"high", None}), "high")
        self.assertEqual(estimates._arm_effort(run, arm_spec("t", "terminus-2"), {None}), None)
        self.assertEqual(estimates._arm_effort(run, arm_spec("pi", "pi"), {"medium", None}), "medium")
        # A harness that records no effort pairs with the baseline at the baseline effort, as the report falls back.
        pi, cc = paired("pi", "claude-code", medium), paired("pi", "claude-code", high, base_cost=0.9)
        unknown = [{**r, "effort": None} if r["family"] == "pi" else r for r in pi]
        unknown += [{**r, "effort": "high"} for r in cc if r["family"] == "claude-code"]
        pair = pairing(unknown, {}, [ARM_CC, ARM_PI])
        self.assertEqual(pair["pairs_now"], 4)
        self.assertAlmostEqual(pair["multiple_now"]["value"], multiple(medium), places=4)


class CellStats(EstimateCase):
    def test_cells_by_configuration_and_by_family(self) -> None:
        rows = [arow("task-00", "pi", 0.1, config_hash="a" * 16), arow("task-01", "pi", 0.3, config_hash="b" * 16),
                arow("task-02", "pi", None, outcome="setup_failure", tokens=None, config_hash="b" * 16,
                     exception_type="NonZeroAgentExitCodeError"),
                arow("task-00", "claude-code", 0.5)]
        cells = estimates.cell_stats(self.cfg, rows, family="pi")
        self.assertEqual([c["config_hash"] for c in cells], ["a" * 16, "b" * 16])
        (family,) = estimates.cell_stats(self.cfg, rows, family="pi", match="family", per_task=True)
        self.assertEqual((family["trials"], family["cost_valid"]), (3, 2))
        self.assertAlmostEqual(family["cost_per_trial"]["mean"], 0.2)
        self.assertEqual(family["infra_failures"], {"rate": round(1 / 3, 4), "trials": 1, "finished": 3})
        self.assertEqual(family["top_failure_causes"],
                         [{"outcome": "setup_failure", "exception_type": "NonZeroAgentExitCodeError", "n": 1}])
        self.assertEqual([t["task"] for t in family["per_task"]], ["task-00", "task-01", "task-02"])
        with self.assertRaises(ValueError):
            estimates.cell_stats(self.cfg, rows, match="harness")

    def test_conservative_spend_follows_the_budget_rule(self) -> None:
        # Claude Code prices OpenRouter models at Claude rates: its own figure is ignored there.
        cc = estimates.cell_stats(self.cfg, [arow("task-00", "claude-code", 0.03, reported=1.44)])[0]
        self.assertAlmostEqual(cc["spend_per_trial"]["mean"], 0.03)
        self.assertTrue(cc["reported_cost_per_trial"]["harness_misprices_provider"])
        # Other harnesses: the larger of computed and reported (OpenRouter's billed cost).
        pi = estimates.cell_stats(self.cfg, [arow("task-00", "pi", 0.042, reported=0.013)])[0]
        self.assertAlmostEqual(pi["spend_per_trial"]["mean"], 0.042)
        self.assertAlmostEqual(pi["reported_cost_per_trial"]["mean"], 0.013)


def set_up(row: dict[str, Any], mode: str | None, setup_sec: float | None) -> dict[str, Any]:
    """An analysis row with its harness setup: setup_sec of its total_minutes went to agent setup."""
    total = row["total_minutes"]
    return dict(row, setup_mode=mode, setup_minutes=None if setup_sec is None else setup_sec / 60.0,
                minutes_ex_setup=total - (setup_sec or 0.0) / 60.0)


class SetupPlanning(EstimateCase):
    """Stock setup installs the harness in each trial (Claude Code: minutes);
    a pre-baked image cuts it to seconds. Plans use the setup the arm will have."""

    def test_the_setup_an_arm_expects(self) -> None:
        cases = [(None, "stock"), ({"mode": "off", "setup_timeout_multiplier": None}, "stock"),
                 ({"mode": "prefer", "supported": False, "reason": "no adapter"}, "stock"),
                 ({"mode": "prefer", "supported": True, "expect_stock": True}, "stock"),
                 ({"mode": "prefer", "supported": True, "expect_stock": False}, "prebaked"),
                 ({"mode": "require", "supported": True, "expect_stock": True}, "prebaked"),
                 ({"mode": "require", "supported": False}, "stock")]
        for block, expected in cases:
            with self.subTest(block=block):
                self.assertEqual(estimates.expected_setup_mode(block), expected)

    def test_setup_figures_from_the_ledger_else_the_prior(self) -> None:
        rows = [set_up(arow(t, "claude-code", 0.2), "stock", s) for t, s in zip(SAMPLE, (300.0, 330.0, 400.0))]
        rows.append(set_up(arow("task-05", "claude-code", 0.2, minutes=6.0), "fallback", 334.0))
        timed_out = arow("task-06", "claude-code", None, outcome="setup_failure", tokens=None,
                         exception_type=estimates.SETUP_TIMEOUT, minutes=6.0)
        rows.append(set_up(timed_out, "stock", 360.0))  # never got past setup: not a setup time
        self.assertEqual(estimates.setup_figure(rows, "claude-code", "stock"),
                         {"mode": "stock", "seconds": 332.0, "basis": "measured", "trials": 4})
        self.assertEqual(estimates.setup_figure(rows, "claude-code", "prebaked"),
                         {"mode": "prebaked", "seconds": 1.0, "basis": "prior", "trials": 0})
        self.assertEqual(estimates.planned_setup_sec(rows, "pi", "prebaked"), 20.0)
        self.assertIsNone(estimates.planned_setup_sec(rows, "pi", "stock"))  # never measured: trials keep their own
        self.assertIsNone(estimates.planned_setup_sec(rows, "opencode", "prebaked"))
        rows += [set_up(arow(t, "claude-code", 0.2), "prebaked", s) for t, s in zip(SAMPLE, (0.4, 0.6))]
        self.assertEqual(estimates.setup_figure(rows, "claude-code", "prebaked"),
                         {"mode": "prebaked", "seconds": 0.5, "basis": "measured", "trials": 2})
        with self.assertRaises(ValueError):
            estimates.setup_figure(rows, "claude-code", "fallback")

    def test_trial_minutes_take_the_planned_setup(self) -> None:
        arm = arm_spec("cc", "claude-code")
        # 15-minute trials, 5 of them Claude Code's stock install; one older row says nothing of its setup.
        rows = [set_up(arow(t, "claude-code", 0.2, minutes=15.0, config_hash="x" * 16), "stock", 300.0)
                for t in SAMPLE[:3]]
        rows.append(arow("task-03", "claude-code", 0.2, minutes=15.0, config_hash="x" * 16))

        def minutes(mode: str | None) -> list[float]:
            model = estimates.cell_model(self.cfg, rows, arm, config_hash="x" * 16, meta=META, setup_mode=mode)
            self.assertEqual(model.setup_sec, None if mode is None else {"stock": 300.0, "prebaked": 1.0}[mode])
            return [round(v, 4) for v in model.minutes_pool]

        self.assertEqual(minutes(None), [15.0] * 4)
        self.assertEqual(minutes("stock"), [15.0] * 4)
        self.assertEqual(minutes("prebaked"), [round(10.0 + 1.0 / 60.0, 4)] * 3 + [15.0])

    def test_a_pre_baked_arm_leaves_stock_setup_timeouts_out_of_its_infra_rate(self) -> None:
        arm = arm_spec("cc", "claude-code")
        rows = [set_up(arow(t, "claude-code", 0.2, config_hash="x" * 16), "stock", 300.0) for t in SAMPLE[:3]]
        for i, mode in enumerate(("stock", "fallback", None, "prebaked")):
            rows.append(set_up(arow(SAMPLE[3 + i], "claude-code", None, outcome="setup_failure", tokens=None,
                                    exception_type=estimates.SETUP_TIMEOUT, config_hash="x" * 16), mode, 360.0))
        rows.append(set_up(arow("task-09", "claude-code", None, outcome="infra", tokens=None, config_hash="x" * 16,
                                exception_type="SandboxBuildFailedError"), "fallback", None))
        rate = {mode: estimates.cell_model(self.cfg, rows, arm, config_hash="x" * 16, meta=META, setup_mode=mode)
                for mode in (None, "stock", "prebaked")}
        self.assertEqual((rate[None].p_infra, rate[None].infra_finished), (5 / 8, 8))
        self.assertEqual((rate["stock"].p_infra, rate["stock"].infra_finished), (5 / 8, 8))
        # Only the timeout on a pre-baked image, and the sandbox failure, are left to count.
        self.assertEqual((rate["prebaked"].p_infra, rate["prebaked"].infra_finished), (2 / 5, 5))

    def test_the_plan_sets_each_arms_setup(self) -> None:
        spec = self.spec([ARM_CC], tasks=SAMPLE[:4])
        chash = self.arm_hash(spec, "cc")
        rows = [set_up(arow(t, "claude-code", 0.2, minutes=15.0, config_hash=chash), "stock", 300.0)
                for t in SAMPLE[:4] for _ in range(3)]  # three per task: no extra spread on the minutes
        plan = designs.validate(self.cfg, spec)["plan"]
        self.assertIn("prebake", plan["arms"][0])
        unpinned = estimates.estimate(self.cfg, spec, rows, plan, replicates=200)["arms"][0]
        self.assertEqual(unpinned["setup"]["mode"], "stock")  # no pinned version, so no pre-baked image
        self.assertEqual(unpinned["minutes_per_trial"]["p50"], 15.0)
        plan["arms"][0]["prebake"] = {"mode": "require", "supported": True, "expect_stock": True}
        baked = estimates.estimate(self.cfg, spec, rows, plan, replicates=200)["arms"][0]
        self.assertEqual(baked["setup"], {"mode": "prebaked", "seconds": 1.0, "basis": "prior", "trials": 0})
        self.assertEqual(baked["minutes_per_trial"]["p50"], 10.02)
        override = estimates.estimate(self.cfg, spec, rows, plan, replicates=200, setup_modes={"cc": "stock"})
        self.assertEqual(override["arms"][0]["minutes_per_trial"]["p50"], 15.0)
        self.assertIsNone(estimates.estimate(self.cfg, spec, rows, None, replicates=200)["arms"][0]["setup"])

    def test_cell_stats_show_minutes_without_setup(self) -> None:
        rows = [set_up(arow("task-00", "pi", 0.1, minutes=5.0), "stock", 60.0),
                set_up(arow("task-01", "pi", 0.1, minutes=4.2), "prebaked", 12.0)]
        (cell,) = estimates.cell_stats(self.cfg, rows, family="pi")
        self.assertEqual(cell["minutes_per_trial"]["ex_setup"]["median"], 4.0)
        self.assertEqual(cell["setup_modes"], {"prebaked": 1, "stock": 1})
