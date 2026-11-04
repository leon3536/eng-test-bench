"""bench.coverage: where comparisons stand, ranked gaps and suggested runs.

Every case builds a real temp ledger from synthetic Harbor trials
(tests/ledger_fixtures.make_trial), so coverage is checked against the
pooled report exactly as the REPORT tab computes it, and every suggested
design goes through designs.validate as a person's would. Harness schemas
come from tests/fixtures/schemas; task metadata and Docker capacity are
fixed tables. Nothing runs Harbor or Docker or touches real data.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest import mock

from tests import ledger_fixtures as lf
from tests.support import BenchTestCase

from bench import checks, coverage, datasets, designs, estimates, report, settings, stats
from bench.extract import ScanRoot
from bench.ledger import Filters, Ledger

GLM = ("openrouter", "z-ai/glm-5.3")
DEEPSEEK = ("openrouter", "deepseek/deepseek-v4.1-flash")
SAMPLE = [f"task-{i:02d}" for i in range(20)]
META = {t: {"difficulty": ("easy", "medium", "hard")[i % 3], "category": "test", "agent_timeout_sec": 900.0,
            "memory_gb": 2.0, "cpus": 1, "expert_time_estimate_min": 30.0} for i, t in enumerate(SAMPLE)}
MINI = "bench.agents.mini_swe_agent:MiniSweAgentWithMcp"
PLANTED_MESSAGE = "apt-get exited 137 while reading /home/agent/private-notes"

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
"""

CONFIG_YAML = """\
dataset: terminal-bench@2.0
providers:
  anthropic: {env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}
  openrouter: {env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}
modes:
  pilot: {n_tasks: 1}
run: {budget_usd: 5, effort: medium, n_concurrent: 2, parallel_arms: 3, per_trial_cap_usd: 1, max_turns: 150}
baseline_arm: claude-code-glm
designs: {max_budget_usd: 20}
arms:
  - {name: claude-code-glm, agent: claude-code, provider: openrouter, model: z-ai/glm-5.3}
  - {name: pi-glm, agent: pi, provider: openrouter, model: z-ai/glm-5.3}
  - {name: terminus-2-glm, agent: terminus-2, provider: openrouter, model: z-ai/glm-5.3}
  - name: mini-swe-agent-glm
    agent: bench.agents.mini_swe_agent:MiniSweAgentWithMcp
    family: mini-swe-agent
    provider: openrouter
    model: z-ai/glm-5.3
  - {name: pi-deepseek, agent: pi, provider: openrouter, model: deepseek/deepseek-v4.1-flash}
  - {name: terminus-2-deepseek, agent: terminus-2, provider: openrouter, model: deepseek/deepseek-v4.1-flash}
"""

AGENTS = {"claude-code": "claude-code", "pi": "pi", "terminus-2": "terminus-2", "mini-swe-agent": MINI}
# Baseline / family cost on task i (a multiple, family / baseline, of 1/ratio), varied so the per-task log
# ratios have a spread.
RATIOS = {"pi": [3.0, 2.2, 4.1, 2.8, 3.3, 2.5, 3.9, 2.1], "terminus-2": [1.6, 2.4, 1.9, 2.0, 1.5, 2.2, 1.8, 2.6],
          "mini-swe-agent": [1.2] * 8}


def digest(task: str, version: str = "") -> str:
    return "sha256:" + hashlib.sha256(f"{task}{version}".encode()).hexdigest()


class PlannerWorld:
    """A temp profile for planner tests: config, prices, task metadata and a
    ledger built from synthetic trials. Not a TestCase, so other test
    modules can import it without running these tests twice."""

    def __init__(self, case: BenchTestCase, yaml_text: str = CONFIG_YAML) -> None:
        self.case = case
        self.tmp = case.tmpdir()
        prices = self.tmp / "pricing.yaml"
        prices.write_text(PRICING_YAML, encoding="utf-8", newline="\n")
        for target, name, value in ((settings, "PRICING", prices),
                                    (datasets, "task_meta", lambda dataset: META),
                                    (datasets, "info", lambda dataset: None),
                                    (datasets, "path", lambda dataset: None),
                                    (checks, "docker_capacity", lambda: {"ncpu": 8, "mem_gb": 16.0})):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            case.addCleanup(patcher.stop)
        report.clear_cache()
        case.addCleanup(report.clear_cache)
        self.cfg = case.load_test_config(yaml_text, pricing=PRICING_YAML, reference=REFERENCE_YAML, tasks=SAMPLE)
        self.data = self.tmp / "data"
        self.trials: list[dict[str, Any]] = []
        self.ledger: Ledger | None = None

    def add(self, family: str, tasks: Sequence[int], *, where: tuple[str, str] = GLM, run: str = "r1",
            max_turns: int = 150, version: str = "", infra: bool = False, message: str = "",
            exception_type: str | None = None, cheap: float = 1.0) -> None:
        for i in tasks:
            self.trials.append({"family": family, "i": i, "where": where, "run": run, "max_turns": max_turns,
                                "version": version, "infra": infra, "message": message,
                                "exception_type": exception_type, "cheap": cheap})

    def build(self) -> coverage.ReadOnlyLedger:
        arms_by_run: dict[str, dict[str, dict[str, str]]] = {}
        for t in self.trials:
            arm = self._arm(t)
            arms_by_run.setdefault(t["run"], {})[arm] = {
                "name": arm, "agent": AGENTS[t["family"]], "family": t["family"], "provider": t["where"][0],
                "model": t["where"][1]}
        for run, arms in arms_by_run.items():
            lf.make_bench_run(self.data, run, list(arms.values()), profile="dev")
        for n, t in enumerate(self.trials):
            arm, task = self._arm(t), SAMPLE[t["i"]]
            base = 200000 * (t["i"] + 1)
            scale = 1.0 if t["family"] == "claude-code" else 1.0 / RATIOS[t["family"]][t["i"] % 8]
            scale *= t["cheap"]
            tokens = (0, 0, 0) if t["infra"] else (int(base * scale), int(base * 0.6 * scale), int(base * 0.05 * scale))
            name = f"{task}__{n}"
            lf.make_trial(self.data / "jobs" / t["run"] / arm / name, family=t["family"], task=task,
                          model_name=f"openrouter/{t['where'][1]}", lock_digest=digest(task, t["version"]),
                          kwargs={"reasoning_effort": "medium", "max_turns": t["max_turns"]}, tokens=tokens,
                          import_path=MINI if t["family"] == "mini-swe-agent" else None,
                          exception_type=(t["exception_type"] or "NonZeroAgentExitCodeError") if t["infra"] else None,
                          message=t["message"], agent_started=not t["infra"], reward=None if t["infra"] else 1.0,
                          job_id=f"{t['run']}-{arm}", trial_name=name)
        self.ledger = Ledger(self.tmp / "ledger" / "bench.sqlite",
                             roots=[ScanRoot(self.data / "jobs", "dev", self.data / "runs")], root=self.tmp)
        self.ledger.open()
        self.case.addCleanup(self.ledger.close)
        self.case.assertTrue(self.ledger.available, self.ledger.error)
        self.ledger.rescan()
        return coverage.ReadOnlyLedger(self.ledger, lambda: self.cfg)

    @staticmethod
    def _arm(t: Mapping[str, Any]) -> str:
        model = t["where"][1].split("/")[-1].split("-")[0]
        return f"{t['family']}-{model}" + ("" if t["max_turns"] == 150 else f"-t{t['max_turns']}")

    def gaps_world(self) -> coverage.ReadOnlyLedger:
        """On glm: every gap kind. On deepseek: no baseline at all."""
        self.add("claude-code", range(0, 4))
        self.add("pi", range(0, 3))
        self.add("pi", range(3, 6), max_turns=100)                # option drift
        self.add("pi", [0], version="-crlf", run="r0")              # stale task version
        self.add("terminus-2", [0, 1, 5])
        self.add("mini-swe-agent", [0, 1], infra=True, message=PLANTED_MESSAGE)
        self.add("pi", [0, 1], where=DEEPSEEK)
        self.add("terminus-2", [0, 1, 2], where=DEEPSEEK)
        return self.build()


class PlannerCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.world = PlannerWorld(self)
        self.cfg = self.world.cfg

    def cell(self, cov: Mapping[str, Any], family: str) -> dict[str, Any]:
        return next(c for c in cov["cells"] if c["family"] == family)

    def gaps(self, cov: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
        return [g for g in cov["gaps"] if g["kind"] == kind]


class Coverage(PlannerCase):
    def test_multiples_equal_the_reports_comparisons(self) -> None:
        reader = self.world.gaps_world()
        cov = coverage.coverage(self.cfg, reader, *GLM)
        rep = report.build_report(self.world.ledger, self.cfg, Filters(models=("openrouter:z-ai/glm-5.3",)),
                                  coverage.coverage_options(self.cfg))
        for family in ("pi", "terminus-2"):
            with self.subTest(family):
                comp = next(c for c in rep["comparisons"] if c["group"].split("|")[2] == family)
                cell = self.cell(cov, family)
                for metric, key in (("cost", "cost_multiple"), ("tokens_total", "tokens_multiple")):
                    m = comp["metrics"][metric]
                    if m["geomean"] is None:
                        self.assertIsNone(cell[key])
                        continue
                    self.assertEqual(cell[key]["value"], m["geomean"]["estimate"])
                    self.assertEqual(cell[key]["ci"], [m["geomean"]["lo"], m["geomean"]["hi"]])
                    self.assertEqual(cell[key]["ratio_of_means"]["value"], m["ratio_of_means"]["estimate"])
                    self.assertEqual(cell[key]["n"], m["n"])
                self.assertEqual(cell["verdict"].get("cost"), comp["metrics"]["cost"]["reference"]["verdict"])
                self.assertEqual(cell["paired_with_baseline"], comp["coverage"]["pairs"])
        self.assertEqual(self.cell(cov, "pi")["paired_with_baseline"], 4)
        self.assertIsNotNone(self.cell(cov, "pi")["cost_multiple"])
        self.assertLess(self.cell(cov, "pi")["cost_multiple"]["value"], 1.0)  # pi / claude-code: pi is cheaper
        self.assertEqual(self.cell(cov, "pi")["verdict"]["cost"], "not_applicable")  # no verdicts off Claude

    def test_counts_only_the_report_tabs_task_scope(self) -> None:
        # The assistant quotes what the REPORT tab shows, and that tab compares on the profile's task file.
        reader = self.world.gaps_world()  # claude-code on tasks 0-3, pi on 0-5
        full = coverage.coverage(self.cfg, reader, *GLM)
        narrow = dataclasses.replace(self.cfg, tasks=SAMPLE[:2])
        cov = coverage.coverage(narrow, coverage.ReadOnlyLedger(self.world.ledger, lambda: narrow), *GLM)
        self.assertEqual((self.cell(full, "pi")["paired_with_baseline"], self.cell(cov, "pi")["paired_with_baseline"]),
                         (4, 2))
        self.assertEqual((cov["scope"]["kind"], cov["scope"]["tasks"]), ("scope", 2))
        rep = report.build_report(self.world.ledger, narrow, Filters(models=("openrouter:z-ai/glm-5.3",),
                                                                      tasks=tuple(SAMPLE[:2])),
                                  coverage.coverage_options(narrow))
        comp = next(c for c in rep["comparisons"] if c["group"].split("|")[2] == "pi")
        self.assertEqual(self.cell(cov, "pi")["trials"], next(g for g in rep["groups"]
                                                               if g["key"] == comp["group"])["trials"]["total"])
        self.assertLess(self.cell(cov, "pi")["trials"], self.cell(full, "pi")["trials"])

    def test_cells_by_configuration_match_the_report_too(self) -> None:
        reader = self.world.gaps_world()
        cov = coverage.coverage(self.cfg, reader, *GLM, group_by="config")
        rep = report.build_report(self.world.ledger, self.cfg, Filters(models=("openrouter:z-ai/glm-5.3",)),
                                  coverage.coverage_options(self.cfg, group_by="config"))
        self.assertEqual(len([c for c in cov["cells"] if c["family"] == "pi"]), 2)
        for comp in rep["comparisons"]:
            cell = next(c for c in cov["cells"] if c["key"] == comp["group"])
            gm = comp["metrics"]["cost"]["geomean"]
            self.assertEqual(None if gm is None else gm["estimate"], (cell["cost_multiple"] or {}).get("value"))

    def test_every_gap_kind_is_found(self) -> None:
        reader = self.world.gaps_world()
        glm = coverage.coverage(self.cfg, reader, *GLM, target_pairs=8)
        kinds = {g["kind"] for g in glm["gaps"]}
        self.assertEqual(kinds, {"half_pair", "below_target", "infra_failing", "stale_task_version", "option_drift"})
        deepseek = coverage.coverage(self.cfg, reader, *DEEPSEEK)
        self.assertEqual([g["kind"] for g in deepseek["gaps"]], ["no_baseline"])

        (stale,) = self.gaps(glm, "stale_task_version")
        self.assertEqual(stale["tasks"], ["task-00"])
        (drift,) = self.gaps(glm, "option_drift")
        self.assertEqual(drift["family"], "pi")
        self.assertIn("max_turns", drift["detail"])
        below = {g["family"]: g for g in self.gaps(glm, "below_target")}
        self.assertEqual(set(below), {"pi", "terminus-2"})
        self.assertEqual(below["pi"]["tasks"], ["task-06", "task-07"])  # 8 - 4 paired - 2 half-pairs

    def test_a_baseline_gap_completes_pairs_for_two_families(self) -> None:
        reader = self.world.gaps_world()
        glm = coverage.coverage(self.cfg, reader, *GLM)
        base_gap = next(g for g in self.gaps(glm, "half_pair") if g["family"] == "claude-code")
        self.assertEqual(base_gap["tasks"], ["task-04", "task-05"])
        self.assertEqual(base_gap["pairs_gained"], 3)  # task-04 for pi; task-05 for pi and terminus-2
        arm_gap = next(g for g in self.gaps(glm, "half_pair") if g["family"] == "terminus-2")
        self.assertEqual(arm_gap["tasks"], ["task-02", "task-03"])
        (no_base,) = self.gaps(coverage.coverage(self.cfg, reader, *DEEPSEEK), "no_baseline")
        self.assertEqual(no_base["family"], "claude-code")
        self.assertEqual(no_base["tasks"], ["task-00", "task-01", "task-02"])
        self.assertEqual(no_base["pairs_gained"], 5)  # 2 for pi and 3 for terminus-2
        self.assertIsNotNone(no_base["est_cost_usd"])

    def test_infra_gap_names_the_exception_type_never_the_message(self) -> None:
        reader = self.world.gaps_world()
        glm = coverage.coverage(self.cfg, reader, *GLM)
        (infra,) = self.gaps(glm, "infra_failing")
        self.assertEqual(infra["family"], "mini-swe-agent")
        self.assertIn("NonZeroAgentExitCodeError", infra["detail"])
        self.assertIn("2 of 2", infra["detail"])
        text = json.dumps(glm)
        self.assertNotIn(PLANTED_MESSAGE, text)
        self.assertNotIn("private-notes", text)
        mini = self.cell(glm, "mini-swe-agent")
        self.assertEqual((mini["infra"], mini["cost_valid"]), (2, 0))

    def test_gaps_are_ranked_by_pairs_per_dollar(self) -> None:
        reader = self.world.gaps_world()
        gaps = coverage.coverage(self.cfg, reader, *GLM, target_pairs=8)["gaps"]
        priced = [g["pairs_per_usd"] for g in gaps if g["pairs_per_usd"] is not None]
        self.assertEqual(priced, sorted(priced, reverse=True))
        first_zero = next(i for i, g in enumerate(gaps) if g["pairs_gained"] == 0)
        self.assertTrue(all(g["pairs_gained"] == 0 for g in gaps[first_zero:]))
        self.assertTrue(all(g["pairs_gained"] > 0 for g in gaps[:first_zero]))

    def test_target_comes_from_the_precision_target_capped_at_the_sample(self) -> None:
        reader = self.world.gaps_world()
        glm = coverage.coverage(self.cfg, reader, *GLM)
        expected = stats.required_pairs(glm["target_sd_log"], self.cfg.stats.target_halfwidth,
                                        self.cfg.stats.confidence)
        self.assertEqual(glm["target_pairs"], min(expected, len(SAMPLE)))
        self.assertEqual(glm["target_source"], "config")
        # No pairs anywhere on Opus: the prior spread (0.6) wants 22 pairs, more than the 20-task sample.
        opus = coverage.coverage(self.cfg, reader, "anthropic", "claude-opus-5-5")
        self.assertEqual((opus["target_pairs"], opus["target_capped"], opus["target_sd_source"]), (20, True, "prior"))
        self.assertTrue(any("remaining uncertainty" in n for n in opus["notes"]))
        self.assertTrue(any("No trials" in n for n in opus["notes"]))
        given = coverage.coverage(self.cfg, reader, *GLM, target_pairs=5)
        self.assertEqual((given["target_pairs"], given["target_source"]), (5, "argument"))
        self.assertEqual(self.cell(given, "pi")["needed_more"], 1)

    def test_cells_carry_pass_rates_and_missing_positions(self) -> None:
        reader = self.world.gaps_world()
        glm = coverage.coverage(self.cfg, reader, *GLM)
        base = self.cell(glm, "claude-code")
        self.assertTrue(base["is_baseline"])
        self.assertEqual(base["missing_sample_positions"], list(range(5, 21)))
        lo, hi = base["pass_rate_ci90"]
        self.assertLess(lo, 1.0)
        self.assertEqual(hi, 1.0)
        self.assertEqual(glm["baseline"]["family"], "claude-code")


class Reader(PlannerCase):
    def test_exposes_reads_only(self) -> None:
        public = {n for n in dir(coverage.ReadOnlyLedger) if not n.startswith("_")}
        self.assertEqual(public, {"rows", "fingerprint", "coverage", "report_for", "summary"})

    def test_rows_are_cached_until_the_ledger_changes(self) -> None:
        reader = self.world.gaps_world()
        with mock.patch.object(report, "analysis_rows", wraps=report.analysis_rows) as computed:
            first = reader.rows()
            self.assertEqual(reader.rows(), first)
            self.assertEqual(computed.call_count, 1)
            self.world.ledger.add_exclusion("run", "dev/r0", "exclude", "damaged tasks")
            rows = reader.rows()
            self.assertEqual(computed.call_count, 2)
        excluded = [r for r in rows if r["run_key"] == "dev/r0"]
        self.assertTrue(excluded and all(r["excluded_reason"].startswith("run excluded by hand") for r in excluded))
        self.assertEqual(reader.summary()["excluded_by_hand"], len(excluded))

    def test_summary(self) -> None:
        reader = self.world.gaps_world()
        summary = reader.summary()
        self.assertEqual(set(summary), {"runs", "trials", "cost_valid", "infra", "excluded_by_hand",
                                        "misconfigured_cap", "first", "last", "fingerprint"})
        self.assertEqual((summary["trials"], summary["infra"]), (len(self.world.trials), 2))
        self.assertEqual(summary["fingerprint"], reader.fingerprint())
        self.assertEqual(reader.fingerprint(), report.ledger_fingerprint(reader.rows()))


class Suggestions(PlannerCase):
    def suggest(self, goal: str = "reach_target_pairs", **kw: Any) -> dict[str, Any]:
        reader = getattr(self, "_reader", None) or self.world.gaps_world()
        self._reader = reader
        return coverage.plan_suggestions(self.cfg, reader, goal, *GLM, **kw)

    def candidate(self, result: Mapping[str, Any], label: str) -> dict[str, Any]:
        return next(c for c in result["candidates"] if c["label"] == label)

    def arm_tasks(self, candidate: Mapping[str, Any]) -> dict[str, list[str]]:
        spec = candidate["spec"]
        return {a["name"]: list(spec["tasks"] if a["tasks"] is None else a["tasks"]) for a in spec["arms"]}

    def assert_valid(self, candidate: Mapping[str, Any]) -> None:
        normalized = designs.normalize(self.cfg, candidate["spec"])
        check = designs.validate(self.cfg, normalized)
        self.assertEqual(check["problems"], [], candidate["label"])
        self.assertEqual(normalized["tasks"], [t for t in SAMPLE if t in normalized["tasks"]])
        self.assertLessEqual(normalized["run"]["budget_usd"], self.cfg.designs.max_budget_usd)

    def test_half_pairs_first_then_seeded_order_sharing_the_baseline(self) -> None:
        result = self.suggest(target_pairs=8)
        full = self.candidate(result, "full")
        tasks = self.arm_tasks(full)
        self.assertEqual(tasks["claude-code-glm"], ["task-04", "task-05", "task-06", "task-07"])
        self.assertEqual(tasks["pi-glm"], ["task-06", "task-07"])
        self.assertEqual(tasks["terminus-2-glm"], ["task-02", "task-03", "task-04", "task-06", "task-07"])
        self.assertEqual(full["spec"]["baseline_arm"], "claude-code-glm")
        self.assertEqual({p["family"]: (p["before"], p["after"]) for p in full["pairs_gained"]},
                         {"pi": (4, 8), "terminus-2": (2, 8)})
        # mini-swe-agent fails before using the model, so it is left for a pilot.
        self.assertNotIn("mini-swe-agent-glm", tasks)
        self.assertTrue(any("mini-swe-agent is left out" in n for n in result["notes"]))
        for c in result["candidates"]:
            self.assert_valid(c)

    def test_extension_never_skips_ahead_to_a_cheaper_task(self) -> None:
        self.world.add("claude-code", [15], cheap=0.01)  # a cheap baseline trial late in the sample
        self.world.add("terminus-2", [15], cheap=0.01)
        result = self.suggest(target_pairs=9)
        tasks = self.arm_tasks(self.candidate(result, "full"))
        had = {"task-00", "task-01", "task-02", "task-03", "task-04", "task-05", "task-15"}
        extension = [t for t in tasks["pi-glm"] if t not in had]
        unpaired = [t for t in SAMPLE if t not in had]
        self.assertEqual(extension, unpaired[: len(extension)])
        self.assertIn("task-15", tasks["pi-glm"])  # a half-pair (pi lacks it) is completed first
        self.assertEqual(tasks["pi-glm"], [t for t in SAMPLE if t in tasks["pi-glm"]])

    def test_respects_max_budget_usd(self) -> None:
        free = self.candidate(self.suggest(target_pairs=12), "full")
        ceiling = round(free["estimate"]["p90_usd"] * 0.5, 4)
        result = self.suggest(target_pairs=12, max_budget_usd=ceiling)
        self.assertTrue(result["candidates"])
        for c in result["candidates"]:
            self.assertLessEqual(c["estimate"]["p90_usd"], ceiling)
            self.assertLessEqual(c["spec"]["run"]["budget_usd"], ceiling)
            self.assert_valid(c)
        capped = self.candidate(result, "full")
        self.assertLess(len(capped["spec"]["tasks"]), len(free["spec"]["tasks"]))
        self.assertTrue(any("budget" in n.lower() for n in capped["notes"]))

    def test_never_beyond_the_sample(self) -> None:
        result = self.suggest(target_pairs=100)
        self.assertEqual(result["target_pairs"], len(SAMPLE))
        self.assertTrue(any("remaining uncertainty" in n for n in result["notes"]))
        for c in result["candidates"]:
            self.assertTrue(set(c["spec"]["tasks"]) <= set(SAMPLE))
            self.assertTrue(all(p["after"] <= len(SAMPLE) for p in c["pairs_gained"]))
            self.assert_valid(c)

    def test_complete_half_pairs_runs_only_half_pairs(self) -> None:
        result = self.suggest("complete_half_pairs", target_pairs=8)
        (only,) = result["candidates"]
        self.assertEqual(only["label"], "half_pairs")
        self.assertEqual(self.arm_tasks(only), {"claude-code-glm": ["task-04", "task-05"],
                                                "terminus-2-glm": ["task-02", "task-03"]})
        self.assertEqual({p["family"]: p["after"] for p in only["pairs_gained"]}, {"pi": 6, "terminus-2": 5})
        self.assert_valid(only)

    def test_candidates_are_ranked_by_cost_per_pair(self) -> None:
        result = self.suggest(target_pairs=12)
        per_pair = [c["cost_per_pair_usd"] for c in result["candidates"]]
        self.assertEqual(per_pair, sorted(per_pair))
        self.assertEqual({c["label"] for c in result["candidates"]}, {"full", "staged", "half_pairs"})
        self.assertEqual(coverage.suggest_runs(self.cfg, self._reader, "reach_target_pairs", *GLM, target_pairs=12),
                         result["candidates"])

    def test_identical_candidates_are_offered_once(self) -> None:
        # At target 10 the staged half (5 pairs) is exactly the half-pair design.
        result = self.suggest(target_pairs=10)
        hashes = [c["spec_hash"] for c in result["candidates"]]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertEqual([c["label"] for c in result["candidates"]].count("half_pairs")
                         + [c["label"] for c in result["candidates"]].count("staged"), 1)

    def test_pilot_new_arms_includes_infra_failing_cells(self) -> None:
        result = self.suggest("pilot_new_arms", families=["pi", "terminus-2", "mini-swe-agent"])
        (pilot,) = result["candidates"]
        self.assertEqual(pilot["label"], "pilot")
        self.assertEqual(self.arm_tasks(pilot), {"mini-swe-agent-glm": ["task-00"]})
        self.assertTrue(any("NonZeroAgentExitCodeError" in n for n in result["notes"]))
        self.assert_valid(pilot)

    def test_a_family_without_an_allowed_harness_is_left_out(self) -> None:
        result = self.suggest(families=["pi", "openhands"], target_pairs=6)
        self.assertEqual(result["families"], ["pi"])
        self.assertTrue(any("openhands" in n for n in result["notes"]))

    def test_bad_arguments_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.suggest("run_everything")
        with self.assertRaises(ValueError):
            self.suggest(attempts=0)
        with self.assertRaises(ValueError):
            self.suggest(max_budget_usd=-1.0)

    def test_candidates_settle_on_quick_estimates_and_price_once_in_full(self) -> None:
        with mock.patch.object(coverage.estimates, "estimate", wraps=estimates.estimate) as priced:
            result = self.suggest(target_pairs=8)
        replicates = [call.kwargs.get("replicates") for call in priced.call_args_list]
        self.assertEqual(replicates.count(None), 3)  # one full estimate each for full, staged and half_pairs
        self.assertTrue(all(r in (None, coverage.GATE_REPLICATES) for r in replicates))
        self.assertGreater(replicates.count(coverage.GATE_REPLICATES), 3)
        self.assertTrue(result["candidates"])


class PairingRule(PlannerCase):
    """Half-pairs, pair counts and suggestions follow the report's pairing
    rule (report.match_on, report.same_run_only), as the REPORT tab does."""

    def use(self, report_yaml: str = "") -> None:
        self.world = PlannerWorld(self, CONFIG_YAML + report_yaml)
        self.cfg = self.world.cfg

    def suggest(self, reader: coverage.ReadOnlyLedger, goal: str, target: int) -> dict[str, Any]:
        return coverage.plan_suggestions(self.cfg, reader, goal, *GLM, families=["terminus-2"], target_pairs=target)

    @staticmethod
    def arm_tasks(candidate: Mapping[str, Any]) -> dict[str, list[str]]:
        spec = candidate["spec"]
        return {a["name"]: list(spec["tasks"] if a["tasks"] is None else a["tasks"]) for a in spec["arms"]}

    def versions_world(self) -> coverage.ReadOnlyLedger:
        """terminus-2 ran task-02 in a version the baseline never ran."""
        self.world.add("claude-code", range(0, 4))
        self.world.add("terminus-2", [0, 1])
        self.world.add("terminus-2", [2], version="-v2")
        return self.world.build()

    def report_pairs(self, family: str) -> int:
        rep = report.build_report(self.world.ledger, self.cfg, Filters(models=("openrouter:z-ai/glm-5.3",)),
                                  coverage.coverage_options(self.cfg))
        comp = next(c for c in rep["comparisons"] if c["group"].split("|")[2] == family)
        return comp["coverage"]["pairs"]

    def test_a_task_run_in_versions_that_do_not_pair_is_run_again_on_both_sides(self) -> None:
        reader = self.versions_world()
        cov = coverage.coverage(self.cfg, reader, *GLM, target_pairs=4)
        t2 = self.cell(cov, "terminus-2")
        self.assertEqual(t2["paired_with_baseline"], 2)
        self.assertEqual(t2["paired_with_baseline"], self.report_pairs("terminus-2"))
        self.assertEqual(t2["half_pair_tasks"], [{"task": "task-03", "missing": "arm"}])  # task-02 is no half-pair
        (only,) = self.suggest(reader, "complete_half_pairs", 4)["candidates"]
        self.assertEqual(self.arm_tasks(only), {"terminus-2-glm": ["task-03"]})
        full = next(c for c in self.suggest(reader, "reach_target_pairs", 4)["candidates"] if c["label"] == "full")
        self.assertEqual(self.arm_tasks(full), {"claude-code-glm": ["task-02"],
                                                "terminus-2-glm": ["task-02", "task-03"]})
        self.assertEqual([(p["before"], p["after"]) for p in full["pairs_gained"]], [(2, 4)])

    def test_match_on_name_pairs_across_versions(self) -> None:
        self.use("report: {match_on: name}\n")
        reader = self.versions_world()
        cov = coverage.coverage(self.cfg, reader, *GLM, target_pairs=4)
        t2 = self.cell(cov, "terminus-2")
        self.assertEqual(t2["paired_with_baseline"], 3)
        self.assertEqual(t2["paired_with_baseline"], self.report_pairs("terminus-2"))
        self.assertEqual(t2["half_pair_tasks"], [{"task": "task-03", "missing": "arm"}])
        full = next(c for c in self.suggest(reader, "reach_target_pairs", 4)["candidates"] if c["label"] == "full")
        self.assertEqual(self.arm_tasks(full), {"terminus-2-glm": ["task-03"]})
        self.assertEqual([(p["before"], p["after"]) for p in full["pairs_gained"]], [(3, 4)])

    def test_same_run_only_pairs_only_within_one_run(self) -> None:
        self.use("report: {same_run_only: true}\n")
        self.world.add("claude-code", range(0, 4))
        self.world.add("terminus-2", [0, 1])
        self.world.add("terminus-2", [2, 3], run="r2")
        reader = self.world.build()
        cov = coverage.coverage(self.cfg, reader, *GLM, target_pairs=5)
        t2 = self.cell(cov, "terminus-2")
        self.assertEqual(t2["paired_with_baseline"], 2)
        self.assertEqual(t2["paired_with_baseline"], self.report_pairs("terminus-2"))
        self.assertEqual(t2["half_pair_tasks"], [])  # a new run cannot complete another run's half-pair
        self.assertFalse(self.gaps(cov, "half_pair"))
        self.assertTrue(any("same_run_only" in n for n in cov["notes"]))
        self.assertEqual(self.suggest(reader, "complete_half_pairs", 5)["candidates"], [])
        full = next(c for c in self.suggest(reader, "reach_target_pairs", 5)["candidates"] if c["label"] == "full")
        # Tasks already paired in r1 are skipped while fresh sample tasks remain.
        both = ["task-02", "task-03", "task-04"]
        self.assertEqual(self.arm_tasks(full), {"claude-code-glm": both, "terminus-2-glm": both})
        self.assertEqual([(p["before"], p["after"]) for p in full["pairs_gained"]], [(2, 5)])
