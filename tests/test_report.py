"""bench.report: the pooled report over the ledger.

Two kinds of data: the real dev pilots, ingested into a temp ledger from
tests/fixtures/ledger, and synthetic trials served by an in-memory stand-in
for the ledger's read methods (query, runs, effective_exclusions, version),
so pairing, intervals, thresholds, verdicts and coverage can be checked
against hand computation. One end-to-end case writes synthetic Harbor
trials and ingests them through a real temp ledger.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import math
import statistics
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest import mock

from tests import ledger_fixtures as lf
from tests.support import BenchTestCase

from bench import classify, pricing, report, settings, stats
from bench.extract import ScanRoot
from bench.fsutil import canonical_json
from bench.ledger import Filters, Ledger

OPUS = ("anthropic", "claude-opus-5-5")
GLM = ("openrouter", "z-ai/glm-5.3")
DEEPSEEK = ("openrouter", "deepseek/deepseek-v4.1-flash")

PRICING_YAML = """\
anthropic:
  claude-opus-5-5: {input: 4.00, output: 20.00, cache_read: 0.20}
openrouter:
  deepseek-v4.1-flash: {input: 0.045, output: 0.50, cache_read: 0.006}
  glm-5.3: {input: 0.07, output: 1.40, cache_read: 0.069}
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
  strands:
    cost_per_trial: {value: 1/2.1, source: "S8"}
  mini-swe-agent:
    note: "no per-task total published"
"""
# The same figures as the file stated them before multiples were harness / baseline.
REFERENCE_YAML_BASELINE_OVER_HARNESS = """\
defaults:
  direction: baseline_over_harness
  estimator: ratio_of_means
  tolerance: 1.5
  applies_to: ['anthropic:claude-*']
consensus:
  cost_per_trial: {low: 2.0, high: 3.0, source: "S1-S5"}
  pass_rate_delta_pp: {low: -5, high: 5, source: "S1-S5"}
arms:
  pi:
    tokens_per_trial: {value: 3.1, source: "S10 via S9"}
  strands:
    cost_per_trial: {value: 2.1, source: "S8"}
  mini-swe-agent:
    note: "no per-task total published"
"""

SAMPLE = [  # config/tasks.txt as seeded on 2026-10-28
    "adaptive-rejection-sampler", "protein-assembly", "path-tracing", "openssl-selfsigned-cert",
    "cobol-modernization", "mteb-retrieve", "build-cython-ext", "filter-js-from-html",
    "feal-differential-cryptanalysis", "count-dataset-tokens",
    "caffe-cifar-10", "winning-avg-corewars", "compile-compcert", "torch-pipeline-parallelism",
    "financial-document-processor", "polyglot-c-py", "overfull-hbox", "reshard-c4-data",
    "feal-linear-cryptanalysis", "llm-inference-batching-scheduler",
]
BLOCK_1 = SAMPLE[:10]

CONFIG_YAML = """\
dataset: terminal-bench@2.0
providers:
  anthropic: {env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}
  openrouter: {env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}
modes:
  pilot: {n_tasks: 1}
run: {budget_usd: 5}
baseline_arm: BASELINE
arms:
  - {name: claude-code, agent: claude-code, provider: anthropic, model: claude-opus-5-5}
  - {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}
  - {name: claude-code-deepseek, agent: claude-code, provider: openrouter, model: deepseek/deepseek-v4.1-flash}
  - {name: pi-deepseek, agent: pi, provider: openrouter, model: deepseek/deepseek-v4.1-flash}
"""

_SEQ = itertools.count(1)
# FakeLedger paths must never repeat: the report memo keys on path and id(),
# and CPython reuses a freed object's id.
_LEDGER_SERIAL = itertools.count(1)
_PASSABLE = ("passed", "failed", "agent_limit", "agent_crash", "misconfigured_cap")


def trow(task: str, family: str, *, provider: str = "anthropic", model: str = "claude-opus-5-5",
         tokens: tuple[int, int, int] | None = (100000, 40000, 2000), outcome: str = "passed",
         passed: bool | None = None, run_key: str = "dev/r1", digest: str | None = None, effort: str | None = "medium",
         options: Mapping[str, Any] | None = None, reported: float | None = None, at_ingest: Any = "current",
         version: str | None = "1.0.0", implementation: str = "builtin", primary_share: float | None = 1.0,
         snapshot: str = "local:tb@1", total_sec: float = 600.0, exclusion: dict | None = None,
         exception_type: str | None = None, agent_started: bool = True) -> dict[str, Any]:
    """A TrialRow as Ledger.query returns it."""
    seq = next(_SEQ)
    options = dict(options) if options is not None else {"max_turns": 150}
    digest = digest or "sha256:" + hashlib.sha256(f"{snapshot}/{task}".encode()).hexdigest()
    config_hash = hashlib.sha256(canonical_json([family, implementation, provider, model, effort, options])
                                 .encode()).hexdigest()[:16]
    n_input, n_cache, n_output = tokens if tokens is not None else (None, None, None)
    usage = ({model: {"reported_as": model, "n_input": n_input, "n_cache": n_cache, "n_output": n_output,
                      "reported_cost_usd": reported, "primary": True, "synthesized": True}} if tokens else {})
    cost, unpriced = pricing.price_usage(usage, provider, pricing.load_table())
    if passed is None and outcome in _PASSABLE:
        passed = outcome == "passed"
    profile, run_id = run_key.split("/", 1)
    # Ledger._trial_row's rule: Harbor's 0/0/0 for a run that recorded nothing is not measured.
    status = classify.usage_status({"outcome": outcome, "n_input": n_input, "n_output": n_output,
                                    "exception_type": exception_type, "agent_started": agent_started,
                                    "agent_exec_sec": total_sec - 60.0})
    measured = tokens is not None and status != "unmeasured"
    return {
        "id": seq, "trial_key": f"job-{family}-{seq}/{task}__{seq}", "run_key": run_key, "profile": profile,
        "run_id": run_id, "source": "bench", "arm": family, "trial_name": f"{task}__{seq}", "attempt": 1,
        "family": family, "implementation": implementation, "harness_version": version, "harbor_version": "0.25.0",
        "provider": provider, "model": model, "effort": effort, "effort_label": effort or "harness default",
        "config_hash": config_hash, "config_label": f"{family} · {provider}:{model} · {canonical_json(options)}",
        "options": options, "cautions": [], "dataset": "terminal-bench@2.0", "dataset_snapshot": snapshot,
        "task": task, "task_digest": digest, "task_digest_short": digest.split(":", 1)[-1][:12],
        "outcome": outcome, "outcome_reason": "test", "harbor_cause": None, "exception_type": exception_type,
        "reward": None if passed is None else float(passed), "passed": passed,
        "n_input": n_input, "n_cache": n_cache, "n_output": n_output, "usage_status": status,
        "tokens_total": (n_input or 0) + (n_output or 0) if measured else None,
        "tokens_uncached": (n_input - n_cache + n_output) if measured else None,
        "cost_usd": cost if status != "unmeasured" else None,
        "cost_at_ingest_usd": None if status == "unmeasured" else cost if at_ingest == "current" else at_ingest,
        "reported_cost_usd": reported, "unpriced_models": unpriced, "usage": usage,
        "primary_share": primary_share, "started_at": f"2026-10-29T06:{seq % 60:02d}:00.000000Z",
        "finished_at": None, "phases": {"env_setup_sec": 5.0, "agent_setup_sec": 30.0,
                                        "agent_exec_sec": total_sec - 60.0, "verifier_sec": 20.0},
        "total_sec": total_sec, "exclusion": exclusion, "parser_version": 1, "agent_started": agent_started,
    }


def scaled(k: float, i: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """(arm tokens, baseline tokens) for task i, the baseline exactly k times
    the arm: the task's multiple (arm / baseline) is 1/k."""
    arm = (100000 * (i + 1), 40000 * (i + 1), 2000 * (i + 1))
    return arm, tuple(round(k * x) for x in arm)  # type: ignore[return-value]


def pair_rows(ks: Sequence[float], *, arm: str = "pi", base: str = "claude-code", where: tuple[str, str] = OPUS,
              tasks: Sequence[str] | None = None, **kw: Any) -> list[dict]:
    rows = []
    for i, k in enumerate(ks):
        task = tasks[i] if tasks else f"task-{i}"
        a, b = scaled(k, i)
        rows.append(trow(task, arm, provider=where[0], model=where[1], tokens=a, **kw))
        rows.append(trow(task, base, provider=where[0], model=where[1], tokens=b, **kw))
    return rows


class FakeLedger:
    """The four read methods the report uses, over a list of TrialRows."""

    def __init__(self, rows: Sequence[dict], *, statuses: Mapping[str, str] | None = None,
                 excluded_trials: Sequence[str] = (), excluded_runs: Sequence[str] = (),
                 ids: Sequence[int] = ()) -> None:
        self.rows = [dict(r) for r in rows]
        self.path = Path(f"memory-ledger-{next(_LEDGER_SERIAL)}")
        keys = sorted({r["run_key"] for r in rows})
        self.statuses = {k: (statuses or {}).get(k, "done") for k in keys}
        self.excl = (frozenset(excluded_trials), frozenset(excluded_runs), tuple(ids))
        self.stamp = 1
        self.queries = 0

    def query(self, f: Filters, *, outcomes: Any = None, limit: Any = None, offset: int = 0,
              order: str = "started_at DESC") -> tuple[int, list[dict]]:
        self.queries += 1
        out = []
        for r in self.rows:
            if f.models and f"{r['provider']}:{r['model']}" not in f.models:
                continue
            if f.families and r["family"] not in f.families:
                continue
            if f.runs and r["run_key"] not in f.runs:
                continue
            if r["run_key"] in f.exclude_runs:
                continue
            out.append(dict(r))
        return len(out), out

    def runs(self) -> list[dict]:
        return [{"run_key": k, "status": s} for k, s in self.statuses.items()]

    def effective_exclusions(self) -> tuple[frozenset[str], frozenset[str], tuple[int, ...]]:
        return self.excl

    def version(self) -> tuple[int, str, int]:
        # Like the real Ledger.version, it changes with run statuses.
        return (len(self.rows), f"{self.stamp}|{sorted(self.statuses.items())}", len(self.excl[2]))


class ReportCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tmp = self.tmpdir()
        self.pricing_file = self.tmp / "pricing.yaml"
        self.pricing_file.write_text(PRICING_YAML, encoding="utf-8", newline="\n")
        patcher = mock.patch.object(settings, "PRICING", self.pricing_file)
        patcher.start()
        self.addCleanup(patcher.stop)
        report.clear_cache()
        self.addCleanup(report.clear_cache)

    def config(self, *, tasks: Sequence[str] | None = None, report_yaml: str = "",
               baseline: str = "claude-code", reference: str = REFERENCE_YAML) -> settings.Config:
        text = CONFIG_YAML.replace("BASELINE", baseline) + textwrap.dedent(report_yaml)
        return self.load_test_config(text, pricing=PRICING_YAML, reference=reference,
                                     tasks=list(tasks) if tasks is not None else SAMPLE)

    def build(self, ledger: Any, cfg: settings.Config, options: Mapping | None = None,
            filters: Mapping | None = None) -> dict:
        return report.build_report(ledger, cfg, report.parse_filters(filters), report.parse_options(options, cfg))

    def comparison(self, rep: Mapping, arm: str, base: str = "claude-code", where: tuple[str, str] = OPUS) -> dict:
        prefix = f"fam:{where[0]}|{where[1]}|"
        key = f"{prefix}{arm}~{prefix}{base}"
        found = [c for c in rep["comparisons"] if c["key"] == key]
        self.assertEqual(len(found), 1, f"{key} not in {[c['key'] for c in rep['comparisons']]}")
        return found[0]


# The dev pilots ---------------------------------------------------------------------------

class DevPilots(ReportCase):
    def setUp(self) -> None:
        super().setUp()
        self.ledger = Ledger(self.tmp / "ledger" / "bench.sqlite", roots=lf.fixture_roots(), root=self.tmp)
        self.ledger.open()
        self.addCleanup(self.ledger.close)
        self.assertTrue(self.ledger.available, self.ledger.error)
        self.ledger.rescan()
        self.cfg = self.config(baseline="claude-code-deepseek")

    def test_the_seeded_exclusion_leaves_pilot_1_out_of_the_totals(self) -> None:
        rep = self.build(self.ledger, self.cfg)
        self.assertEqual(rep["totals"]["trials"], 9)
        self.assertEqual(rep["totals"]["excluded_manual"], 7)
        self.assertEqual(rep["totals"]["by_outcome"],
                         {"passed": 1, "failed": 2, "misconfigured_cap": 2, "setup_failure": 4})
        self.assertEqual(rep["provenance"]["exclusions_applied"], 1)
        self.assertEqual(rep["provenance"]["trials_used"], 9)
        self.assertEqual(rep["spec"]["options"]["ci_level"], 0.90)
        self.assertEqual(rep["spec"]["options"]["policy"], {"crashes": "include", "misconfigured_cap": "exclude",
                                                           "exclusions": True})

    def test_golden_counts_with_exclusions_off(self) -> None:
        rep = self.build(self.ledger, self.cfg, {"policy": {"exclusions": False}})
        self.assertEqual(rep["totals"]["by_outcome"], lf.GOLDEN_COUNTS)
        self.assertEqual((rep["totals"]["trials"], rep["totals"]["excluded_manual"]), (16, 0))
        self.assertEqual(rep["provenance"]["exclusions_applied"], 0)

    def test_every_comparison_is_insufficient(self) -> None:
        rep = self.build(self.ledger, self.cfg)
        self.assertEqual(len(rep["comparisons"]), 7)  # 4 harnesses on deepseek, 3 on glm (terminus-2 is pilot 1)
        for c in rep["comparisons"]:
            with self.subTest(c["key"]):
                self.assertEqual(c["pass"]["status"], "insufficient")
                for name, m in c["metrics"].items():
                    self.assertIn(m["status"], ("insufficient", "not_measured"), name)
                    self.assertIsNone(m["geomean"])
                    self.assertIsNone(m["ratio_of_means"])
                self.assertIn("more for an estimate", c["headline"])
        strands = self.comparison(rep, "strands", where=DEEPSEEK)
        self.assertEqual(strands["metrics"]["tokens_total"]["status"], "not_measured")
        self.assertEqual(strands["coverage"]["metric"], "pass")
        baselines = [g["key"] for g in rep["groups"] if g["is_baseline"]]
        self.assertEqual(baselines, [f"fam:{DEEPSEEK[0]}|{DEEPSEEK[1]}|claude-code",
                                     f"fam:{GLM[0]}|{GLM[1]}|claude-code"])

    def test_including_misconfigured_caps_pairs_pi_with_claude_code(self) -> None:
        rep = self.build(self.ledger, self.cfg, {"policy": {"misconfigured_cap": "include"}})
        c = self.comparison(rep, "pi", where=DEEPSEEK)
        self.assertEqual((c["metrics"]["cost"]["n"], c["metrics"]["cost"]["need"]), (1, 2))
        self.assertEqual(c["metrics"]["cost"]["status"], "insufficient")
        self.assertEqual(c["pairs"]["n_tasks"], 1)
        self.assertEqual(c["headline"],
                         "pi vs claude-code on openrouter:deepseek/deepseek-v4.1-flash: 1 paired task; "
                         "2 more for an estimate")
        default = self.comparison(self.build(self.ledger, self.cfg), "pi", where=DEEPSEEK)
        self.assertEqual(default["metrics"]["cost"]["n"], 0)
        self.assertEqual(default["pairs"]["dropped"]["no_baseline_valid"], 1)

    def test_coverage_completes_the_pair_then_follows_block_1(self) -> None:
        c = self.comparison(self.build(self.ledger, self.cfg), "pi", where=DEEPSEEK)
        cov = c["coverage"]
        self.assertEqual(cov["next"][0], {"task": "adaptive-rejection-sampler", "side": "baseline",
                                          "reason": "complete pair"})
        self.assertEqual([x["task"] for x in cov["next"][1:10]], BLOCK_1[1:])
        self.assertTrue(all(x["side"] == "both" and x["reason"] == "new task" for x in cov["next"][1:]))
        # The prior spread (0.6) asks for 22 pairs; the sample has 20.
        self.assertEqual((cov["sd_source"], cov["sd_log_used"]), ("prior", 0.6))
        self.assertEqual(cov["est_pairs_needed"], 20)
        self.assertTrue(cov["capped_at_sample"])
        self.assertEqual(len(cov["next"]), 20)
        self.assertTrue(any("remaining uncertainty" in x for x in c["caveats"]))
        self.assertIsNone(cov["est_cost_usd"])  # the baseline has no valid cost yet
        self.assertAlmostEqual(cov["per_trial"]["arm"]["mean_cost_usd"], 0.041905897, places=6)
        coverage = report.build_coverage(self.ledger, self.cfg, Filters(), report.parse_options(None, self.cfg))
        self.assertEqual(coverage["task_order"], SAMPLE)
        self.assertEqual(coverage["untouched_tasks"], SAMPLE[1:])
        self.assertEqual(set(coverage["comparisons"][0]), {"key", "provider", "model", "group", "baseline_group",
                                                           "headline", "coverage"})

    def test_reported_cost_caveats_on_openrouter(self) -> None:
        rep = self.build(self.ledger, self.cfg)
        c = self.comparison(rep, "pi", where=DEEPSEEK)
        self.assertIn(report.OPENROUTER_CAVEAT, c["caveats"])
        self.assertIn(report.NOT_APPLICABLE_CAVEAT, c["caveats"])
        self.assertTrue(any("claude-code prices openrouter models at its own provider's rates" in x
                            for x in c["caveats"]))
        groups = {g["key"]: g for g in rep["groups"]}
        pi = groups[f"fam:{DEEPSEEK[0]}|{DEEPSEEK[1]}|pi"]
        self.assertAlmostEqual(pi["reported_vs_computed_cost"]["median_ratio"], 0.012608063928 / 0.041905897, places=6)
        self.assertEqual(pi["reported_vs_computed_cost"]["n"], 1)
        strands = groups[f"fam:{DEEPSEEK[0]}|{DEEPSEEK[1]}|strands"]
        self.assertEqual(strands["reported_vs_computed_cost"], {"median_ratio": None, "n": 0})
        self.assertEqual(strands["setup_failure_rate"]["estimate"], 1.0)

    def test_reported_pricing_trusts_the_bill_except_for_mispriced_harnesses(self) -> None:
        o = report.parse_options({"pricing": "reported", "policy": {"exclusions": False}}, self.cfg)
        rows = {(r["run_id"], r["arm"]): r for r in report.analysis_rows(self.ledger, self.cfg, Filters(), o)}
        self.assertAlmostEqual(rows[(lf.PILOT_2, "pi-deepseek")]["cost_usd"], 0.012608063928)
        self.assertAlmostEqual(rows[(lf.PILOT_2, "claude-code-deepseek")]["cost_usd"], 0.026779541)
        self.assertAlmostEqual(rows[(lf.PILOT_2, "claude-code-deepseek")]["reported_cost_usd"], 1.438498)
        current = report.analysis_rows(self.ledger, self.cfg)
        self.assertAlmostEqual(next(r for r in current if r["arm"] == "pi-deepseek")["cost_usd"], 0.041905897)

    def test_report_pricing_setting_never_reaches_analysis_rows(self) -> None:
        # E4: a display setting must not shrink the cost that estimates and queue reserves read.
        cfg = self.config(baseline="claude-code-deepseek", report_yaml="report: {pricing: reported}\n")
        self.assertEqual(report.parse_options(None, cfg).pricing, "reported")
        pi = next(r for r in report.analysis_rows(self.ledger, cfg) if r["arm"] == "pi-deepseek")
        self.assertAlmostEqual(pi["cost_usd"], 0.041905897)
        self.assertAlmostEqual(pi["reported_cost_usd"], 0.012608063928)
        rep = report.build_report(self.ledger, cfg)  # the REPORT tab does honour the setting
        self.assertEqual(rep["provenance"]["pricing"], "reported")
        group = next(g for g in rep["groups"] if g["key"] == f"fam:{DEEPSEEK[0]}|{DEEPSEEK[1]}|pi")
        self.assertAlmostEqual(group["per_trial"]["cost"]["mean"], 0.012608063928)

    def test_analysis_rows_keep_the_contract(self) -> None:
        rows = report.analysis_rows(self.ledger, self.cfg)
        self.assertEqual(len(rows), 16)
        for r in rows:
            self.assertEqual(tuple(r), report.ANALYSIS_KEYS)
        pilot_1 = [r for r in rows if r["run_id"] == lf.PILOT_1]
        self.assertEqual(len(pilot_1), 7)
        for r in pilot_1:
            self.assertTrue(r["excluded_reason"].startswith("run excluded by hand"))
            self.assertFalse(r["valid_tokens"] or r["valid_cost"] or r["valid_reward"])
        pi = next(r for r in rows if r["arm"] == "pi-deepseek")
        self.assertEqual((pi["valid_tokens"], pi["valid_cost"], pi["valid_reward"], pi["excluded_reason"]),
                         (True, True, True, None))
        self.assertAlmostEqual(pi["agent_minutes"], 591.856053 / 60)
        self.assertEqual(pi["tokens_total"], 1360013 + 59144)
        cap = next(r for r in rows if r["arm"] == "claude-code-glm" and r["run_id"] == lf.PILOT_2)
        self.assertIn("misconfigured_cap excluded by the report policy", cap["excluded_reason"])
        self.assertEqual(report.ledger_fingerprint(rows), report.ledger_fingerprint(list(reversed(rows))))
        self.assertRegex(report.ledger_fingerprint(rows), r"^sha256:[0-9a-f]{16}$")
        # C9: over the sorted trial keys alone, so an outcome change keeps it.
        keys = "\n".join(sorted(r["trial_key"] for r in rows))
        self.assertEqual(report.ledger_fingerprint(rows),
                         "sha256:" + hashlib.sha256(keys.encode("utf-8")).hexdigest()[:16])
        self.assertEqual(report.ledger_fingerprint([dict(r, outcome="incomplete") for r in rows]),
                         report.ledger_fingerprint(rows))
        self.assertNotEqual(report.ledger_fingerprint(rows[1:]), report.ledger_fingerprint(rows))

    def test_identical_requests_give_identical_output(self) -> None:
        first = self.build(self.ledger, self.cfg)
        report.clear_cache()
        second = self.build(self.ledger, self.cfg)
        for rep in (first, second):
            rep["provenance"].pop("generated_at")
        self.assertEqual(canonical_json(first), canonical_json(second))
        first["groups"].clear()  # callers get copies, never the cached report
        self.assertTrue(self.build(self.ledger, self.cfg)["groups"])

    def test_pairs_of_one_comparison(self) -> None:
        o = report.parse_options({"policy": {"misconfigured_cap": "include"}}, self.cfg)
        rep = report.build_report(self.ledger, self.cfg, Filters(), o)
        key = self.comparison(rep, "pi", where=DEEPSEEK)["key"]
        pairs = report.build_pairs(self.ledger, self.cfg, Filters(), o, key)
        (row,) = pairs["pairs"]
        self.assertEqual((row["task"], row["digest_short"], row["same_run"]),
                         ("adaptive-rejection-sampler", "b4c60177f24d", True))
        self.assertAlmostEqual(row["ratio_cost"], 0.041905897 / 0.026779541)  # pi / claude-code
        with self.assertRaises(KeyError):
            report.build_pairs(self.ledger, self.cfg, Filters(), o, "fam:x~fam:y")
        unused = report.build_pairs(self.ledger, self.cfg, Filters(), report.parse_options(None, self.cfg), key)
        reasons = {(e["side"], e["outcome"]) for e in unused["excluded_trials"]}
        self.assertIn(("baseline", "misconfigured_cap"), reasons)


# Synthetic ledgers --------------------------------------------------------------------------

KS = [2.0, 2.5, 3.0, 2.2, 2.8, 1.8, 3.2, 2.4, 2.6, 2.1]


class Estimates(ReportCase):
    def test_geometric_mean_and_t_interval_by_hand(self) -> None:
        cfg = self.config()
        rep = self.build(FakeLedger(pair_rows(KS)), cfg)
        c = self.comparison(rep, "pi")
        logs = [math.log(1 / k) for k in KS]  # pi / claude-code on each task
        mean, sd = statistics.fmean(logs), statistics.stdev(logs)
        half = 1.833 * sd / math.sqrt(10)  # t(0.90, 9)
        for metric in ("tokens_total", "tokens_uncached", "cost"):
            with self.subTest(metric):
                m = c["metrics"][metric]
                self.assertEqual((m["status"], m["n"], m["need"]), ("ok", 10, 0))
                self.assertAlmostEqual(m["geomean"]["estimate"], math.exp(mean), places=9)
                self.assertAlmostEqual(m["geomean"]["lo"], math.exp(mean - half), places=3)
                self.assertAlmostEqual(m["geomean"]["hi"], math.exp(mean + half), places=3)
                self.assertAlmostEqual(m["geomean"]["sd_log"], sd, places=9)
                self.assertEqual(m["geomean"]["method"], "t_on_log_ratios")
        arm = [sum(scaled(k, i)[0][::2]) for i, k in enumerate(KS)]
        base = [sum(scaled(k, i)[1][::2]) for i, k in enumerate(KS)]
        rom = c["metrics"]["tokens_total"]["ratio_of_means"]
        self.assertAlmostEqual(rom["estimate"], sum(arm) / sum(base), places=9)
        self.assertLess(rom["lo"], rom["estimate"])
        self.assertGreater(rom["hi"], rom["estimate"])
        self.assertEqual(rom["method"], "task_bootstrap")
        per_task = c["metrics"]["tokens_total"]["per_task"]
        self.assertAlmostEqual(per_task["median"], statistics.median(1 / k for k in KS))
        self.assertAlmostEqual(per_task["min"], 1 / max(KS))
        self.assertAlmostEqual(per_task["max"], 1 / min(KS))
        self.assertEqual(c["pairs"]["same_run_fraction"], 1.0)
        self.assertIn(f"pi used {math.exp(mean):.2f}x the cost of claude-code", c["headline"])  # cost = tokens here
        self.assertIn("90% CI", c["headline"])
        again = self.build(FakeLedger(pair_rows(KS)), cfg)  # another ledger, the same numbers
        self.assertEqual(self.comparison(again, "pi")["metrics"]["cost"]["ratio_of_means"], c["metrics"]["cost"]
                         ["ratio_of_means"])

    def test_repeats_average_inside_a_cell(self) -> None:
        rows = pair_rows(KS)
        arm0, _ = scaled(1.0, 0)
        # task-0's first baseline trial is 2x the arm (KS[0]); two more at 3x and 4x.
        rows += [trow("task-0", "claude-code", tokens=tuple(round(f * x) for x in arm0)) for f in (3.0, 4.0)]
        rep = self.build(FakeLedger(rows), self.config(), {"include_pairs": True})
        c = self.comparison(rep, "pi")
        self.assertEqual(c["metrics"]["tokens_total"]["n"], 10)  # still one vote per task
        first = c["pairs_table"][0]
        self.assertEqual((first["task"], first["arm"]["trials"], first["baseline"]["trials"]), ("task-0", 1, 3))
        self.assertAlmostEqual(first["ratio_tokens"], 1 / 3.0)  # the arm over the mean of 2x, 3x and 4x

    def test_statuses_follow_the_thresholds(self) -> None:
        cfg = self.config()
        for n, status, need in ((2, "insufficient", 1), (4, "low_n", 4), (8, "ok", 0)):
            with self.subTest(n=n):
                m = self.comparison(self.build(FakeLedger(pair_rows(KS[:n])), cfg), "pi")["metrics"]["cost"]
                self.assertEqual((m["status"], m["n"], m["need"]), (status, n, need))
                self.assertEqual(m["geomean"] is None, status == "insufficient")
        low = self.comparison(self.build(FakeLedger(pair_rows(KS[:4])), cfg), "pi")["metrics"]["cost"]
        self.assertIsNone(low["ratio_of_means"]["lo"])  # no bootstrap below five tasks
        self.assertIsNotNone(low["ratio_of_means"]["estimate"])
        self.assertEqual(low["reference"]["verdict"], "insufficient")
        stricter = self.build(FakeLedger(pair_rows(KS[:8])), cfg, {"min_pairs": {"estimate": 5, "verdict": 10}})
        self.assertEqual(self.comparison(stricter, "pi")["metrics"]["cost"]["status"], "low_n")

    def test_pairing_on_digest_or_name(self) -> None:
        rows = []
        for i, k in enumerate(KS):
            a, b = scaled(k, i)
            rows.append(trow(f"task-{i}", "pi", tokens=a, snapshot="git:69671fbaac6d"))
            rows.append(trow(f"task-{i}", "claude-code", tokens=b, snapshot="local:tb@2"))
        cfg = self.config()
        by_digest = self.comparison(self.build(FakeLedger(rows), cfg), "pi")
        self.assertEqual(by_digest["metrics"]["cost"]["n"], 0)
        self.assertEqual(by_digest["pairs"]["arm_only_tasks"], [f"task-{i}" for i in range(10)])
        by_name = self.comparison(self.build(FakeLedger(rows), cfg, {"pairing": {"match_on": "name"}}), "pi")
        self.assertEqual(by_name["metrics"]["cost"]["n"], 10)
        self.assertTrue(any("different task snapshots" in x for x in by_name["caveats"]))

    def test_same_run_only_drops_cross_run_pairs(self) -> None:
        rows = []
        for i, k in enumerate(KS):
            a, b = scaled(k, i)
            rows.append(trow(f"task-{i}", "pi", tokens=a, run_key="dev/r1" if i < 5 else "dev/r2"))
            rows.append(trow(f"task-{i}", "claude-code", tokens=b, run_key="dev/r1"))
        cfg = self.config()
        pooled = self.comparison(self.build(FakeLedger(rows), cfg), "pi")
        self.assertEqual((pooled["metrics"]["cost"]["n"], pooled["pairs"]["same_run_fraction"]), (10, 0.5))
        self.assertTrue(any(x.startswith("50% of paired tasks were run in different runs") for x in pooled["caveats"]))
        self.assertAlmostEqual(pooled["metrics"]["cost"]["sensitivity"]["same_run_only"],
                               math.exp(statistics.fmean(math.log(1 / k) for k in KS[:5])))
        same = self.comparison(self.build(FakeLedger(rows), cfg, {"pairing": {"same_run_only": True}}), "pi")
        self.assertEqual(same["metrics"]["cost"]["n"], 5)


class References(ReportCase):
    def test_verdict_mapping(self) -> None:
        band = [2.0, 3.0]
        self.assertEqual(report.verdict((2.1, 2.9), band, "ok", True), "within")
        self.assertEqual(report.verdict((1.5, 2.5), band, "ok", True), "overlaps")
        self.assertEqual(report.verdict((3.1, 4.0), band, "ok", True), "above")
        self.assertEqual(report.verdict((1.0, 1.9), band, "ok", True), "below")
        self.assertEqual(report.verdict((2.1, 2.9), band, "low_n", True), "insufficient")
        self.assertEqual(report.verdict(None, band, "ok", True), "insufficient")
        self.assertEqual(report.verdict((2.1, 2.9), band, "ok", False), "not_applicable")

    def test_reference_lookup(self) -> None:
        cfg = self.config()
        pi = report.reference_for(cfg.reference, "pi", *OPUS, "tokens_total")
        self.assertAlmostEqual(pi["published"]["value"], 1 / 3.1)
        self.assertAlmostEqual(pi["band"][0], 1 / 4.65)  # 1/3.1 within x/÷1.5: [1/(3.1 x 1.5), 1.5/3.1]
        self.assertAlmostEqual(pi["band"][1], 1.5 / 3.1)
        self.assertEqual((pi["estimator"], pi["applies"]), ("ratio_of_means", True))
        self.assertEqual(report.reference_for(cfg.reference, "pi", *OPUS, "cost")["band"], [1 / 3, 0.5])
        strands = report.reference_for(cfg.reference, "strands", *OPUS, "cost")["band"]
        self.assertAlmostEqual(strands[0], 1 / 3.15)
        self.assertAlmostEqual(strands[1], 1 / 1.4)
        self.assertIsNone(report.reference_for(cfg.reference, "mini-swe-agent", *OPUS, "tokens_total"))
        self.assertEqual(report.reference_for(cfg.reference, "pi", *OPUS, "pass")["band"], [-5.0, 5.0])
        self.assertFalse(report.reference_for(cfg.reference, "pi", *GLM, "cost")["applies"])
        self.assertIsNone(report.reference_for(cfg.reference, "pi", *OPUS, "tokens_uncached"))

    def test_verdicts_against_published_figures(self) -> None:
        cfg = self.config()
        # ks: claude-code / pi on each task, so pi's multiples are near 1/3, 1/6 and 1/1.2. Against the
        # published 0.333-0.5 (Claude Code at 2x-3x), 1/6 is below it (relatively cheaper) and 1/1.2 above.
        cases = {
            "near 1/3": ([3.0, 3.1, 2.9, 3.2, 2.8, 3.0, 3.1, 2.95, 3.05, 3.0], "within", "overlaps"),
            "near 1/6": ([6.0, 6.2, 5.9, 6.4, 5.8, 6.1, 6.0, 6.3, 5.7, 6.0], "below", "below"),
            "near 1/1.2": ([1.2, 1.25, 1.15, 1.3, 1.1, 1.2, 1.22, 1.18, 1.24, 1.16], "above", "above"),
        }
        for name, (ks, tokens, cost) in cases.items():
            with self.subTest(name):
                c = self.comparison(self.build(FakeLedger(pair_rows(ks)), cfg), "pi")
                ref = c["metrics"]["tokens_total"]["reference"]
                self.assertEqual(ref["verdict"], tokens)
                self.assertAlmostEqual(ref["band"][0], 1 / 4.65)
                self.assertEqual(c["metrics"]["cost"]["reference"]["verdict"], cost)
                self.assertEqual(c["metrics"]["cost"]["reference"]["band"], [1 / 3, 0.5])
                self.assertIsNone(c["metrics"]["tokens_uncached"]["reference"])
                phrase = {"overlaps": "overlaps", "above": "lies above", "below": "lies below"}[cost]
                self.assertTrue(c["headline"].endswith(f"the ratio-of-means interval {phrase} the published "
                                                       "0.33-0.50x"), c["headline"])
        rows = []
        for i, k in enumerate(KS):
            a, b = scaled(k, i)
            rows.append(trow(f"task-{i}", "claude-code", tokens=a, options={"max_turns": 300}))
            rows.append(trow(f"task-{i}", "claude-code", tokens=b, options={"max_turns": 150}))
        same_family = self.build(FakeLedger(rows), cfg, {"group_by": "config"})
        self.assertEqual(len(same_family["comparisons"]), 1)
        (c,) = same_family["comparisons"]
        self.assertEqual(c["metrics"]["cost"]["status"], "ok")
        self.assertIsNone(c["metrics"]["cost"]["reference"])  # a harness against itself has no published figure
        self.assertIsNone(c["pass"]["reference"])

    def test_figures_do_not_apply_off_claude_models(self) -> None:
        c = self.comparison(self.build(FakeLedger(pair_rows([3.0, 3.1, 2.9, 3.2, 2.8, 3.0, 3.1, 2.95, 3.05, 3.0],
                                                          where=GLM)), self.config()), "pi", where=GLM)
        ref = c["metrics"]["tokens_total"]["reference"]
        self.assertEqual(c["metrics"]["tokens_total"]["status"], "ok")
        self.assertEqual((ref["verdict"], ref["applies"], ref["position"]), ("not_applicable", False, "inside"))
        self.assertEqual(c["pass"]["reference"]["verdict"], "not_applicable")
        self.assertIn(report.NOT_APPLICABLE_CAVEAT, c["caveats"])
        self.assertNotIn("Harbor reports no cache-write tokens", " ".join(c["caveats"]))
        opus = self.comparison(self.build(FakeLedger(pair_rows(KS)), self.config()), "pi")
        self.assertNotIn(report.NOT_APPLICABLE_CAVEAT, opus["caveats"])
        # Cache writes are priced from the harness logs now: no blanket caveat on Anthropic either.
        self.assertFalse(any("cache-write" in x or "cache writes" in x for x in opus["caveats"]))

    def test_a_reference_written_baseline_over_harness_gives_the_same_report(self) -> None:
        # A reference.yaml from before the flip (2-3, 3.1, 2.1) is inverted on reading: the same bands and verdicts.
        old = self.config(reference=REFERENCE_YAML_BASELINE_OVER_HARNESS)
        for ks in (KS, [6.0, 6.2, 5.9, 6.4, 5.8, 6.1, 6.0, 6.3, 5.7, 6.0]):
            with self.subTest(ks=ks[0]):
                rows = pair_rows(ks) + pair_rows(ks, arm="strands")
                new_rep, old_rep = self.build(FakeLedger(rows), self.config()), self.build(FakeLedger(rows), old)
                for arm in ("pi", "strands"):
                    for metric in ("tokens_total", "cost"):
                        self.assertEqual(self.comparison(old_rep, arm)["metrics"][metric]["reference"],
                                         self.comparison(new_rep, arm)["metrics"][metric]["reference"])
                    self.assertEqual(self.comparison(old_rep, arm)["headline"],
                                     self.comparison(new_rep, arm)["headline"])
        for family, metric in (("pi", "tokens_total"), ("strands", "cost"), ("terminus-2", "cost")):
            self.assertEqual(report.reference_for(old.reference, family, *OPUS, metric),
                             report.reference_for(self.config().reference, family, *OPUS, metric))


class Direction(ReportCase):
    """Every multiple is harness / baseline: under 1x, the harness used less than the baseline."""

    def test_a_harness_that_spends_half_reads_0_5(self) -> None:
        rows = pair_rows([2.0] * 8)  # claude-code spends twice what pi spends on every task
        rep = self.build(FakeLedger(rows), self.config(), {"include_pairs": True})
        c = self.comparison(rep, "pi")
        for metric in ("tokens_total", "tokens_uncached", "cost"):
            with self.subTest(metric):
                m = c["metrics"][metric]
                self.assertAlmostEqual(m["geomean"]["estimate"], 0.5, places=6)
                self.assertAlmostEqual(m["ratio_of_means"]["estimate"], 0.5, places=6)
                self.assertAlmostEqual(m["per_task"]["median"], 0.5, places=6)
                self.assertTrue(all(abs(v - 0.5) < 1e-6 for v in m["leave_one_out"]))
        self.assertTrue(all(abs(p["ratio_tokens"] - 0.5) < 1e-6 and abs(p["ratio_cost"] - 0.5) < 1e-6
                            for p in c["pairs_table"]))
        pairs = report.build_pairs(FakeLedger(rows), self.config(), Filters(), None, c["key"])["pairs"]
        self.assertAlmostEqual(pairs[0]["ratio_cost"], pairs[0]["arm"]["cost"] / pairs[0]["baseline"]["cost"])
        self.assertTrue(c["headline"].startswith(
            "pi vs claude-code on anthropic:claude-opus-5-5: pi used 0.50x the cost of claude-code"), c["headline"])
        self.assertEqual(rep["provenance"]["direction"], stats.DIRECTION)

    def test_the_unrecorded_sensitivities_keep_their_sides(self) -> None:
        # unrecorded_arm_high costs the arm's unrecorded trials high: the multiple rises; the baseline's: it falls.
        ks = [2.0, 2.1, 1.9, 2.0, 2.05, 1.95, 2.0, 2.0]
        timeout: dict[str, Any] = dict(outcome="agent_limit", passed=False, exception_type="AgentTimeoutError",
                                       total_sec=960.0)
        for side, other in (("pi", "claude-code"), ("claude-code", "pi")):
            with self.subTest(lost=side):
                rows = pair_rows(ks)
                rows += [trow("task-1", side, tokens=(0, 0, 0), run_key="dev/r2", **timeout),
                         trow("task-1", other, tokens=(200000, 80000, 4000), run_key="dev/r2")]
                m = self.comparison(self.build(FakeLedger(rows), self.config()), "pi")["metrics"]["cost"]
                high = m["sensitivity"]["unrecorded_arm_high" if side == "pi" else "unrecorded_baseline_high"]
                self.assertIsNone(m["sensitivity"]["unrecorded_baseline_high" if side == "pi" else
                                                   "unrecorded_arm_high"])
                if side == "pi":
                    self.assertGreater(high, m["geomean"]["estimate"])
                else:
                    self.assertLess(high, m["geomean"]["estimate"])


class TaskScope(ReportCase):
    def test_the_scope_says_which_tasks_a_report_covers(self) -> None:
        rows = pair_rows(KS, tasks=SAMPLE[:8] + ["out-of-scope-1", "out-of-scope-2"])
        cfg = self.config()
        everything = self.build(FakeLedger(rows), cfg)
        self.assertEqual(everything["scope"], {"kind": "all", "tasks": 0, "scope_tasks": 20,
                                               "tasks_file": report.tasks_file(cfg)})
        self.assertTrue(everything["scope"]["tasks_file"].endswith("tasks.txt"))
        self.assertEqual(self.comparison(everything, "pi")["metrics"]["cost"]["n"], 10)
        scoped = self.build(FakeLedger(rows), cfg, filters={"tasks": SAMPLE})
        self.assertEqual((scoped["scope"]["kind"], scoped["scope"]["tasks"]), ("scope", 20))
        self.assertEqual(scoped["spec"]["filters"]["tasks"], SAMPLE)  # the filter actually used is recorded
        chosen = self.build(FakeLedger(rows), cfg, filters={"tasks": SAMPLE[:3]})
        self.assertEqual((chosen["scope"]["kind"], chosen["scope"]["tasks"]), ("chosen", 3))
        self.assertEqual(report.task_scope(self.config(tasks=[]), Filters())["kind"], "all")

    def test_a_scoped_report_compares_on_the_scope_tasks_only(self) -> None:
        # The real ledger keeps every trial; this one serves the tasks filter as Ledger.query does.
        rows = pair_rows(KS, tasks=SAMPLE[:8] + ["out-of-scope-1", "out-of-scope-2"])

        class Scoped(FakeLedger):
            def query(self, f: Filters, **kw: Any) -> tuple[int, list[dict]]:
                total, out = super().query(f, **kw)
                out = [r for r in out if not f.tasks or r["task"] in f.tasks]
                return len(out), out

        scoped = self.build(Scoped(rows), self.config(), filters={"tasks": SAMPLE})
        c = self.comparison(scoped, "pi")
        self.assertEqual(c["metrics"]["cost"]["n"], 8)
        self.assertAlmostEqual(c["metrics"]["cost"]["geomean"]["estimate"],
                               math.exp(statistics.fmean(math.log(1 / k) for k in KS[:8])))


class SavedReports(ReportCase):
    """Snapshots saved before multiples were harness / baseline are converted when read."""

    def old_report(self) -> dict:
        metric = {
            "status": "ok", "n": 9, "need": 0,
            "geomean": {"estimate": 2.0, "lo": 1.6, "hi": 2.5, "sd_log": 0.3, "gsd": 1.35, "method": "t_on_log_ratios"},
            "ratio_of_means": {"estimate": 2.5, "lo": 2.0, "hi": 3.2, "method": "task_bootstrap"},
            "per_task": {"min": 1.25, "q1": 1.6, "median": 2.0, "q3": 2.5, "max": 4.0}, "leave_one_out": [1.9, 2.2],
            "sensitivity": {"crashes_excluded": 2.0, "misconfigured_cap_included": None, "same_run_only": 1.6,
                            "unrecorded_arm_high": 1.25, "unrecorded_baseline_high": None},
            "robust": True, "dropped": {},
            "reference": {"published": {"low": 2.0, "high": 3.0, "source": "S1-S5"}, "band": [2.0, 3.0],
                          "estimator": "ratio_of_means", "applies": True, "position": "above",
                          "verdict": "insufficient",
                          "withheld": "had those cost 2x pi's usual spend, the verdict would be 'below', not 'above'"},
        }
        tokens = copy.deepcopy(metric)
        tokens["reference"] = {"published": {"value": 3.1, "source": "S10"}, "band": [2.0667, 4.65],
                               "estimator": "ratio_of_means", "applies": True, "position": "inside",
                               "verdict": "below", "withheld": None}
        comparison = {
            "key": "fam:a|m|pi~fam:a|m|claude-code", "provider": "anthropic", "model": "claude-opus-5-5",
            "group": "fam:a|m|pi", "baseline_group": "fam:a|m|claude-code", "arm_name": "pi",
            "baseline_name": "claude-code", "headline": "pi vs claude-code: claude-code used 2.00x the cost of pi",
            "metrics": {"tokens_total": tokens, "tokens_uncached": copy.deepcopy(metric), "cost": metric},
            "pass": {"status": "ok", "n": 9, "need": 0, "delta_pp": {"estimate": -11.1, "lo": -30.0, "hi": 8.0}},
            "coverage": {"metric": "cost", "halfwidth_factor_now": 1.25},
            "caveats": ["No cost verdict: had those cost 2x pi's usual spend, the verdict would be 'below', not "
                        "'above'.", "Effort differs."],
            "pairs_table": [{"task": "t", "ratio_tokens": 4.0, "ratio_cost": 2.0}],
        }
        return {"spec": {"filters": {"tasks": ["t"]}, "options": {"ci_level": 0.9}},
                "provenance": {"fingerprint": "sha256:abc", "stats_version": 3, "trials_used": 18},
                "comparisons": [comparison], "groups": [], "totals": {"trials": 18}}

    def test_an_old_report_is_converted(self) -> None:
        old = self.old_report()
        kept = copy.deepcopy(old)
        out = report.saved_report(old, old["spec"], self.config())
        self.assertEqual(old, kept)  # never in place
        self.assertEqual((out["provenance"]["direction"], out["provenance"]["converted_from"]),
                         ("harness_over_baseline", "baseline_over_harness"))
        c = out["comparisons"][0]
        cost = c["metrics"]["cost"]
        self.assertEqual(cost["geomean"], {"estimate": 0.5, "lo": 0.4, "hi": 0.625, "sd_log": 0.3, "gsd": 1.35,
                                           "method": "t_on_log_ratios"})
        self.assertEqual((cost["ratio_of_means"]["estimate"], cost["ratio_of_means"]["lo"],
                          cost["ratio_of_means"]["hi"]), (0.4, 0.3125, 0.5))
        self.assertEqual(cost["per_task"], {"min": 0.25, "q1": 0.4, "median": 0.5, "q3": 0.625, "max": 0.8})
        self.assertEqual(cost["leave_one_out"], [1 / 2.2, 1 / 1.9])
        self.assertEqual(cost["sensitivity"], {"crashes_excluded": 0.5, "misconfigured_cap_included": None,
                                               "same_run_only": 0.625, "unrecorded_arm_high": 0.8,
                                               "unrecorded_baseline_high": None})
        ref = cost["reference"]
        self.assertEqual(ref["published"], {"low": 1 / 3, "high": 0.5, "source": "S1-S5"})
        self.assertEqual(ref["band"], [1 / 3, 0.5])
        self.assertEqual((ref["position"], ref["verdict"]), ("below", "insufficient"))
        self.assertEqual(ref["withheld"], "had those cost 2x pi's usual spend, the verdict would be 'above', not "
                                          "'below'")
        tokens = c["metrics"]["tokens_total"]["reference"]
        self.assertEqual((tokens["published"]["value"], tokens["verdict"], tokens["position"]),
                         (1 / 3.1, "above", "inside"))
        self.assertEqual(c["pass"]["delta_pp"]["estimate"], -11.1)  # points are never inverted
        self.assertEqual(c["pairs_table"][0]["ratio_tokens"], 0.25)
        self.assertEqual(c["caveats"][0], "No cost verdict: had those cost 2x pi's usual spend, the verdict would be "
                                          "'above', not 'below'.")
        self.assertTrue(c["headline"].startswith("pi vs claude-code on anthropic:claude-opus-5-5: pi used 0.50x the "
                                                 "cost of claude-code (90% CI 0.40-0.62x, 9 tasks)"), c["headline"])
        self.assertEqual((out["scope"]["kind"], out["scope"]["tasks"]), ("chosen", 1))

    def test_a_report_saved_since_is_left_as_it_is(self) -> None:
        cfg = self.config()
        rep = self.build(FakeLedger(pair_rows(KS)), cfg)
        out = report.saved_report(rep, rep["spec"], cfg)
        self.assertEqual(out, rep)
        self.assertNotIn("converted_from", out["provenance"])
        # Converting what an old version would have saved gives today's report back.
        old = copy.deepcopy(rep)
        old["provenance"].pop("direction")
        old.pop("scope")
        for c in old["comparisons"]:
            for m in c["metrics"].values():
                report._invert_metric(m)
        again = report.saved_report(old, old["spec"], cfg)
        again["provenance"].pop("converted_from")
        self.assertEqual(json_round(again), json_round(rep))
        self.assertIsNone(report.saved_report(None, None, cfg))


def json_round(value: Any) -> Any:
    """Floats to 12 significant digits, for comparing a value inverted twice."""
    if isinstance(value, float):
        return float(f"{value:.12g}")
    if isinstance(value, dict):
        return {k: json_round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_round(v) for v in value]
    return value


class PassRates(ReportCase):
    def test_newcombe_with_one_trial_a_side(self) -> None:
        arm_pass = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        base_pass = [1, 1, 1, 1, 0, 0, 1, 0, 0, 0]
        rows = []
        for i in range(10):
            rows.append(trow(f"task-{i}", "pi", outcome="passed" if arm_pass[i] else "failed"))
            rows.append(trow(f"task-{i}", "claude-code", outcome="passed" if base_pass[i] else "failed"))
        p = self.comparison(self.build(FakeLedger(rows), self.config()), "pi")["pass"]
        self.assertEqual((p["status"], p["n"]), ("ok", 10))
        self.assertAlmostEqual(p["delta_pp"]["estimate"], 10.0)
        lo, hi = stats.newcombe_paired(4, 2, 1, 3, 0.90)
        self.assertAlmostEqual(p["delta_pp"]["lo"], 100 * lo)
        self.assertAlmostEqual(p["delta_pp"]["hi"], 100 * hi)
        self.assertEqual(p["delta_pp"]["method"], "newcombe_paired")
        self.assertEqual(p["discordant"], {"arm_only": 2, "baseline_only": 1})
        self.assertEqual(p["mcnemar_p"], 1.0)
        self.assertEqual((p["arm_rate"], p["baseline_rate"]), (0.6, 0.5))
        self.assertEqual(p["reference"]["verdict"], "overlaps")
        self.assertGreater(p["mde_pp"], 10)

    def test_bootstrap_with_repeats(self) -> None:
        arm = [[1, 1], [1, 0], [0, 0], [1, 1], [0, 1], [1, 0]]
        base = [[1], [0], [0], [1], [1], [0]]
        rows = []
        for i in range(6):
            rows += [trow(f"task-{i}", "pi", outcome="passed" if x else "failed") for x in arm[i]]
            rows += [trow(f"task-{i}", "claude-code", outcome="passed" if x else "failed") for x in base[i]]
        p = self.comparison(self.build(FakeLedger(rows), self.config()), "pi")["pass"]
        self.assertEqual((p["status"], p["n"]), ("low_n", 6))
        self.assertAlmostEqual(p["delta_pp"]["estimate"], 100 * (0.5 / 6))
        self.assertEqual(p["delta_pp"]["method"], "task_bootstrap")
        self.assertLess(p["delta_pp"]["lo"], p["delta_pp"]["hi"])
        self.assertIsNone(p["discordant"])
        self.assertIsNone(p["mcnemar_p"])

    def test_all_concordant_repeats_give_no_interval_but_an_mde(self) -> None:
        rows = []
        for i in range(6):
            rows += [trow(f"task-{i}", "pi") for _ in range(2)]
            rows.append(trow(f"task-{i}", "claude-code"))
        p = self.comparison(self.build(FakeLedger(rows), self.config()), "pi")["pass"]
        self.assertEqual(p["delta_pp"], {"estimate": 0.0, "lo": None, "hi": None, "method": None})
        self.assertAlmostEqual(p["mde_pp"], stats.mde_pp(0.3, 6, 0.90))

    def test_insufficient_pass_shows_only_n(self) -> None:
        rows = [trow("task-0", "pi"), trow("task-0", "claude-code", outcome="failed")]
        p = self.comparison(self.build(FakeLedger(rows), self.config()), "pi")["pass"]
        self.assertEqual((p["status"], p["need"], p["arm_rate"], p["delta_pp"]["estimate"]),
                         ("insufficient", 2, None, None))
        self.assertAlmostEqual(p["mde_pp"], stats.mde_pp(0.5, 1, 0.90))


class Sensitivity(ReportCase):
    def test_a_crash_heavy_harness_is_not_robust(self) -> None:
        rows = []
        for i in range(10):
            a, b = scaled(10.0 if i < 5 else 2.0, i)
            rows.append(trow(f"task-{i}", "pi", tokens=a, outcome="agent_crash" if i < 5 else "passed"))
            rows.append(trow(f"task-{i}", "claude-code", tokens=b))
        m = self.comparison(self.build(FakeLedger(rows), self.config()), "pi")["metrics"]["cost"]
        primary = math.exp((5 * math.log(1 / 10) + 5 * math.log(1 / 2)) / 10)
        self.assertAlmostEqual(m["geomean"]["estimate"], primary, places=6)
        self.assertAlmostEqual(m["sensitivity"]["crashes_excluded"], 1 / 2.0, places=6)
        self.assertAlmostEqual(m["sensitivity"]["misconfigured_cap_included"], primary, places=6)
        self.assertAlmostEqual(m["sensitivity"]["same_run_only"], primary, places=6)
        self.assertLess(m["leave_one_out"][0], m["leave_one_out"][1])
        self.assertIs(m["robust"], False)
        excluded = self.comparison(self.build(FakeLedger(rows), self.config(), {"policy": {"crashes": "exclude"}}),
                                   "pi")["metrics"]["cost"]
        self.assertEqual((excluded["n"], excluded["status"]), (5, "low_n"))

    def test_a_steady_harness_is_robust(self) -> None:
        m = self.comparison(self.build(FakeLedger(pair_rows(KS)), self.config()), "pi")["metrics"]["cost"]
        self.assertIs(m["robust"], True)
        lo, hi = m["leave_one_out"]
        self.assertTrue(m["geomean"]["lo"] <= lo <= hi <= m["geomean"]["hi"])

    def test_misconfigured_cap_variant(self) -> None:
        rows = []
        for i, k in enumerate(KS):
            a, b = scaled(k, i)
            rows.append(trow(f"task-{i}", "pi", tokens=a, provider=DEEPSEEK[0], model=DEEPSEEK[1]))
            rows.append(trow(f"task-{i}", "claude-code", tokens=b, provider=DEEPSEEK[0], model=DEEPSEEK[1],
                             outcome="misconfigured_cap" if i < 8 else "passed"))
        cfg = self.config()
        m = self.comparison(self.build(FakeLedger(rows), cfg), "pi", where=DEEPSEEK)["metrics"]["cost"]
        self.assertEqual((m["status"], m["n"]), ("insufficient", 2))
        self.assertIsNone(m["sensitivity"]["misconfigured_cap_included"])  # only computed with an estimate
        included = self.comparison(self.build(FakeLedger(rows), cfg, {"policy": {"misconfigured_cap": "include"}}),
                                   "pi", where=DEEPSEEK)["metrics"]["cost"]
        self.assertEqual((included["status"], included["n"]), ("ok", 10))


class Grouping(ReportCase):
    def rows(self) -> list[dict]:
        rows = []
        for i in range(10):
            a, b = scaled(KS[i], i)
            rows.append(trow(f"task-{i}", "pi", tokens=a))
            turns = 150 if i < 4 else 300
            rows.append(trow(f"task-{i}", "claude-code", tokens=b, options={"max_turns": turns}))
            if i in (4, 5):
                rows.append(trow(f"task-{i}", "claude-code", tokens=b, options={"max_turns": 300}))
        return rows

    def test_config_diff_when_configurations_pool(self) -> None:
        rep = self.build(FakeLedger(self.rows()), self.config())
        groups = {g["family"]: g for g in rep["groups"]}
        self.assertEqual(groups["claude-code"]["config_diff"], {"max_turns": [150, 300]})
        self.assertEqual([c["trials"] for c in groups["claude-code"]["configs"]], [8, 4])
        self.assertEqual(groups["pi"]["config_diff"], {})
        c = self.comparison(rep, "pi")
        self.assertIn("claude-code pools configurations that differ in max_turns.", c["caveats"])
        self.assertEqual(rep["warnings"], [])

    def test_baseline_pick_under_group_by_config(self) -> None:
        cfg = self.config()
        rows = self.rows()
        rep = self.build(FakeLedger(rows), cfg, {"group_by": "config"})
        hashes = {r["options"]["max_turns"]: r["config_hash"] for r in rows if r["family"] == "claude-code"}
        baseline = [g for g in rep["groups"] if g["is_baseline"]]
        self.assertEqual([g["config_hash"] for g in baseline], [hashes[300]])  # the one with more valid trials
        self.assertEqual(len(rep["warnings"]), 1)
        self.assertIn("2 claude-code groups on anthropic:claude-opus-5-5", rep["warnings"][0])
        self.assertIn("Pin one with options.baseline.config", rep["warnings"][0])
        self.assertEqual({c["group"] for c in rep["comparisons"]},
                         {f"cfg:{hashes[150]}", next(g["key"] for g in rep["groups"] if g["family"] == "pi")})
        pinned = self.build(FakeLedger(rows), cfg, {"group_by": "config", "baseline": {"config": hashes[150]}})
        self.assertEqual([g["config_hash"] for g in pinned["groups"] if g["is_baseline"]], [hashes[150]])
        self.assertEqual(pinned["warnings"], [])

    def test_pinning_a_config_under_family_grouping(self) -> None:
        rows = self.rows()
        hashes = {r["options"]["max_turns"]: r["config_hash"] for r in rows if r["family"] == "claude-code"}
        rep = self.build(FakeLedger(rows), self.config(), {"baseline": {"config": hashes[150]}})
        baseline = next(g for g in rep["groups"] if g["is_baseline"])
        self.assertEqual(baseline["key"], f"cfg:{hashes[150]}")
        self.assertEqual(baseline["trials"]["total"], 4)
        arms = {c["group"] for c in rep["comparisons"]}
        self.assertIn("fam:anthropic|claude-opus-5-5|claude-code", arms)  # the rest of the family is an arm

    def test_group_by_implementation_separates_an_import_path_subclass(self) -> None:
        msa = "bench.agents.mini_swe_agent:MiniSweAgentWithMcp"
        rows = pair_rows(KS[:4], arm="mini-swe-agent")
        rows += [trow(f"task-{i}", "mini-swe-agent", implementation=msa, tokens=scaled(KS[i], i)[0]) for i in range(4)]
        cfg = self.config()
        family = self.build(FakeLedger(rows), cfg)
        msa_group = next(g for g in family["groups"] if g["family"] == "mini-swe-agent")
        self.assertEqual(msa_group["config_diff"], {"implementation": [msa, "builtin"]})
        self.assertIsNone(msa_group["implementation"])
        by_impl = self.build(FakeLedger(rows), cfg, {"group_by": "implementation"})
        names = sorted(g["name"] for g in by_impl["groups"] if g["family"] == "mini-swe-agent")
        self.assertEqual(names, ["mini-swe-agent (MiniSweAgentWithMcp)", "mini-swe-agent (builtin)"])
        self.assertEqual(len(by_impl["comparisons"]), 2)
        for c in by_impl["comparisons"]:
            self.assertEqual(c["metrics"]["cost"]["n"], 4)
            self.assertIsNone(c["metrics"]["tokens_total"]["reference"])  # mini-swe-agent has notes only

    def test_split_harness_versions(self) -> None:
        rows = pair_rows(KS[:5], version="1.0.0") + pair_rows(KS[5:], version="1.1.0",
                                                              tasks=[f"task-{i}" for i in range(5, 10)])
        rep = self.build(FakeLedger(rows), self.config())
        c = self.comparison(rep, "pi")
        self.assertTrue(any(x.startswith("Several pi versions contribute (1.0.0, 1.1.0)") for x in c["caveats"]))
        split = self.build(FakeLedger(rows), self.config(), {"split_harness_versions": True})
        keys = sorted(g["key"] for g in split["groups"])
        self.assertIn("fam:anthropic|claude-opus-5-5|pi|v1.1.0", keys)
        self.assertEqual(len(split["warnings"]), 1)  # two claude-code version groups

    def test_not_measured_when_a_harness_reports_no_tokens(self) -> None:
        rows = []
        for i in range(4):
            rows.append(trow(f"task-{i}", "strands", tokens=None, outcome="passed" if i % 2 else "failed",
                             effort=None))
            rows.append(trow(f"task-{i}", "claude-code", tokens=scaled(2.0, i)[1]))
        rep = self.build(FakeLedger(rows), self.config())
        c = self.comparison(rep, "strands")
        for metric in ("tokens_total", "tokens_uncached", "cost"):
            self.assertEqual(c["metrics"][metric]["status"], "not_measured")
        self.assertEqual((c["pass"]["status"], c["pass"]["n"]), ("low_n", 4))
        self.assertEqual(c["coverage"]["metric"], "pass")
        self.assertIsNone(c["coverage"]["est_pairs_needed"])
        self.assertIn("tokens not reported", c["headline"])
        self.assertTrue(any(x.startswith("Effort differs") for x in c["caveats"]))
        self.assertEqual(c["metrics"]["cost"]["reference"]["verdict"], "insufficient")

    def test_names_starting_with_v_are_not_versions(self) -> None:
        where = ("openrouter", "vendor/model-x")
        rows = pair_rows(KS[:4], arm="vibe", where=where)
        rep = self.build(FakeLedger(rows), self.config())
        groups = {g["family"]: g for g in rep["groups"]}
        self.assertEqual(groups["vibe"]["name"], "vibe")
        self.assertEqual(groups["vibe"]["label"], "vibe · openrouter:vendor/model-x")
        self.assertEqual(groups["claude-code"]["name"], "claude-code")
        c = self.comparison(rep, "vibe", where=where)
        self.assertTrue(c["headline"].startswith("vibe vs claude-code on openrouter:vendor/model-x: "), c["headline"])
        split = self.build(FakeLedger(rows), self.config(), {"split_harness_versions": True})
        named = {g["family"]: g for g in split["groups"]}
        self.assertEqual(named["vibe"]["name"], "vibe v1.0.0")
        self.assertEqual(named["vibe"]["label"], "vibe · openrouter:vendor/model-x v1.0.0")

    def test_models_without_a_baseline_are_named(self) -> None:
        rows = [trow("task-0", "pi", provider=GLM[0], model=GLM[1])]
        rep = self.build(FakeLedger(rows), self.config())
        self.assertEqual(rep["comparisons"], [])
        self.assertEqual(rep["warnings"], ["No claude-code trials on openrouter:z-ai/glm-5.3, so 1 group there has "
                                           "no comparison."])


class SetupModes(ReportCase):
    """How a trial got its harness (a stock install or a pre-baked image) is
    provenance: one arm's trials of either kind share its group and pairs."""

    def rows(self) -> list[dict]:
        rows = []
        for i, k in enumerate(KS[:6]):
            arm, base = scaled(k, i)
            # pi runs pre-baked on tasks 0-2 and stock on 3-5, and both ways on task 0; the baseline stock.
            rows.append(dict(trow(f"task-{i}", "pi", tokens=arm), setup_mode="prebaked" if i < 3 else "stock"))
            rows.append(dict(trow(f"task-{i}", "claude-code", tokens=base), setup_mode="stock"))
        rows.append(dict(trow("task-0", "pi", tokens=scaled(KS[0], 0)[0]), setup_mode="fallback"))
        rows.append(dict(trow("task-1", "claude-code", tokens=scaled(KS[1], 1)[1]), setup_mode="prebaked"))
        rows.append(dict(trow("task-2", "pi", tokens=scaled(KS[2], 2)[0]), setup_mode=None))
        return rows

    def test_a_pre_baked_and_a_stock_trial_of_one_arm_share_a_cell_and_pair(self) -> None:
        o = report.parse_options(None, self.config())
        rows = self.rows()
        task_0 = [r for r in rows if r["task"] == "task-0"]
        baked, stock = (next(r for r in task_0 if r["family"] == "pi" and r["setup_mode"] == mode)
                        for mode in ("prebaked", "fallback"))
        base = next(r for r in task_0 if r["family"] == "claude-code")
        self.assertEqual(report.group_key(baked, o), report.group_key(stock, o))
        self.assertEqual(report._pair_key(baked, o), report._pair_key(stock, o))
        self.assertEqual(report._pair_key(baked, o), report._pair_key(base, o))
        prepared = report.prepare_rows(rows, o, None, pricing.load_table())
        for r in prepared:
            r["group"] = report.group_key(r, o)
        cells = report.build_cells(prepared, o)
        cell = cells[(report.group_key(baked, o), report._pair_key(baked, o))]
        self.assertEqual(sorted(cell["trial_keys"]), sorted([baked["trial_key"], stock["trial_key"]]))
        rep = self.build(FakeLedger(rows), self.config())
        groups = {g["family"]: g for g in rep["groups"]}
        self.assertEqual(groups["pi"]["setup_modes"], {"fallback": 1, "prebaked": 3, "stock": 3, "unknown": 1})
        self.assertEqual(groups["claude-code"]["setup_modes"], {"prebaked": 1, "stock": 6})
        c = self.comparison(rep, "pi")
        self.assertEqual(c["metrics"]["tokens_total"]["n"], 6)  # every task pairs, whichever setup each side had

    def test_the_setup_option_filters_trials(self) -> None:
        cfg = self.config()
        rows = self.rows()
        totals = {setup: self.build(FakeLedger(rows), cfg, {"setup": setup})["totals"]["trials"]
                  for setup in report.SETUP_FILTERS}
        self.assertEqual(totals, {"any": 15, "stock": 10, "prebaked": 4})  # a fallback ran stock; None only in any
        self.assertEqual(report.parse_options(None, cfg).setup, "any")
        prebaked = self.build(FakeLedger(rows), cfg, {"setup": "prebaked"})
        self.assertEqual(prebaked["spec"]["options"]["setup"], "prebaked")
        self.assertEqual({g["family"]: g["setup_modes"] for g in prebaked["groups"]},
                         {"pi": {"prebaked": 3}, "claude-code": {"prebaked": 1}})
        self.assertEqual(len(report.analysis_rows(FakeLedger(rows), cfg)), 15)  # the estimates see every trial
        o = report.parse_options({"setup": "stock"}, cfg)
        self.assertEqual(report.parse_options(report.options_dict(o), cfg), o)
        for bad in ("baked", "", 1, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                report.parse_options({"setup": bad}, cfg)

    def test_minutes_without_setup(self) -> None:
        o = report.parse_options(None, self.config())
        row = dict(trow("task-0", "pi", total_sec=600.0), setup_mode="stock")
        never = dict(trow("task-1", "pi", total_sec=30.0), setup_mode="stock")
        never["phases"] = {**never["phases"], "agent_setup_sec": None}
        prepared = {r["task"]: r for r in report.prepare_rows([row, never], o, None, None)}
        self.assertEqual((prepared["task-0"]["setup_minutes"], prepared["task-0"]["minutes_ex_setup"]),
                         (0.5, 9.5))  # 30 s of agent setup in 600 s
        self.assertEqual((prepared["task-1"]["setup_minutes"], prepared["task-1"]["minutes_ex_setup"]), (None, 0.5))
        analysis = report._analysis_row(prepared["task-0"])
        self.assertEqual((analysis["setup_mode"], analysis["minutes_ex_setup"]), ("stock", 9.5))
        rep = self.build(FakeLedger(self.rows()), self.config())
        pi = next(g for g in rep["groups"] if g["family"] == "pi")
        self.assertEqual(pi["per_trial"]["minutes_ex_setup"]["median"], 9.5)


class UnrecordedUsage(ReportCase):
    """Pilot 3's pi-glm timed out after 900 s before its first model call
    completed; Harbor recorded 0/0/0. It used to enter the means as a $0
    trial and halve pi's cost on GLM (0.0666 over two trials against 0.1332
    for its one real trial), flipping the pair's cost ratio."""

    def pilot_rows(self, timeout_tokens: tuple[int, int, int] | None) -> list[dict]:
        real = trow("adaptive-rejection-sampler", "pi", provider=GLM[0], model=GLM[1], outcome="failed",
                    tokens=(707943, 605450, 60178), run_key="dev/pilot-2")
        timeout = trow("adaptive-rejection-sampler", "pi", provider=GLM[0], model=GLM[1], outcome="agent_limit",
                       passed=False, tokens=timeout_tokens, exception_type="AgentTimeoutError",
                       run_key="dev/pilot-3", total_sec=960.0)
        base = trow("adaptive-rejection-sampler", "claude-code", provider=GLM[0], model=GLM[1], outcome="failed",
                    tokens=(583521, 515328, 25131), run_key="dev/pilot-3")
        return [real, timeout, base]

    def test_a_timeout_with_nothing_recorded_is_not_a_cheap_trial(self) -> None:
        for tokens, reason in (((0, 0, 0), "no usage recorded"), (None, "tokens not reported")):
            with self.subTest(tokens=tokens):
                rows = self.pilot_rows(tokens)
                ledger = FakeLedger(rows)
                cfg = self.config()
                rep = self.build(ledger, cfg, {"include_pairs": True})
                pi = next(g for g in rep["groups"] if g["family"] == "pi")
                real_cost = rows[0]["cost_usd"]
                self.assertEqual(pi["per_trial"]["cost"]["n"], 1)
                self.assertAlmostEqual(pi["per_trial"]["cost"]["mean"], real_cost)
                self.assertEqual(pi["per_trial"]["tokens_total"]["n"], 1)
                self.assertEqual(pi["trials"]["usage_unmeasured"], 1)
                self.assertEqual(pi["trials"]["usage_lower_bound"], 0)
                self.assertEqual((pi["trials"]["valid_reward"], pi["trials"]["valid_cost"]), (2, 1))
                self.assertEqual(pi["pass_rate"]["tasks"], 1)  # the timeout's failure still counts
                self.assertEqual(rep["totals"]["usage_unmeasured"], 1)
                c = self.comparison(rep, "pi", where=GLM)
                (pair,) = c["pairs_table"]
                self.assertEqual((pair["arm"]["trials"], pair["arm"]["valid"]), (2, 2))
                self.assertAlmostEqual(pair["arm"]["cost"], real_cost)
                self.assertAlmostEqual(pair["ratio_cost"], real_cost / rows[2]["cost_usd"])
                self.assertIn("No usage was recorded for 1 of 2 pi trials and 0 of 1 claude-code trials; they count "
                              "for pass rate but are left out of tokens and cost. Such trials often ran to a time "
                              "limit, so the side with more of them may look cheaper than it is.", c["caveats"])
                excluded = report.build_pairs(ledger, cfg, Filters(), None, c["key"])["excluded_trials"]
                self.assertEqual([(e["trial_key"], e["reason"]) for e in excluded], [(rows[1]["trial_key"], reason)])
                row = next(r for r in report.analysis_rows(ledger, cfg) if r["trial_key"] == rows[1]["trial_key"])
                self.assertEqual((row["usage_status"], row["valid_tokens"], row["valid_cost"], row["valid_reward"]),
                                 ("unmeasured", False, False, True))
                self.assertIsNone(row["cost_usd"])
                self.assertIsNone(row["cost_at_ingest_usd"])  # not a $0 bill either
                self.assertIsNone(row["tokens_total"])

    def test_a_harness_whose_every_trial_lost_its_usage_is_not_measured(self) -> None:
        rows = [r for r in self.pilot_rows((0, 0, 0)) if r["run_key"] == "dev/pilot-3"]
        c = self.comparison(self.build(FakeLedger(rows), self.config()), "pi", where=GLM)
        self.assertEqual(c["metrics"]["cost"]["status"], "not_measured")
        self.assertEqual(c["metrics"]["cost"]["dropped"]["usage_not_recorded"], 1)
        self.assertEqual(c["metrics"]["cost"]["dropped"]["tokens_not_reported"], 0)
        self.assertEqual(c["pass"]["n"], 1)

    def test_a_timeout_with_usage_counts_as_a_lower_bound(self) -> None:
        # Above what pi's finished trial on the task recorded, so it stays as a floor.
        rows = self.pilot_rows((900000, 800000, 90000))
        rep = self.build(FakeLedger(rows), self.config())
        pi = next(g for g in rep["groups"] if g["family"] == "pi")
        self.assertEqual((pi["per_trial"]["cost"]["n"], pi["trials"]["usage_lower_bound"]), (2, 1))
        self.assertEqual(pi["trials"]["usage_unmeasured"], 0)
        caveats = self.comparison(rep, "pi", where=GLM)["caveats"]
        self.assertIn("1 pi trial hit the time limit; their usage covers completed model calls only, so their tokens "
                      "and cost are lower bounds.", caveats)
        self.assertFalse(any(x.startswith(("No usage was recorded", "1 pi trial cut short")) for x in caveats))
        # With no uncut pi trial on the task to compare against, any floor stays too.
        alone = [r for r in self.pilot_rows((6000, 0, 250)) if r["run_key"] == "dev/pilot-3"]
        pi = next(g for g in self.build(FakeLedger(alone), self.config())["groups"] if g["family"] == "pi")
        self.assertEqual((pi["per_trial"]["cost"]["n"], pi["trials"]["usage_lower_bound"]), (1, 1))

    def test_a_floor_below_every_uncut_trial_on_its_task_is_set_aside(self) -> None:
        # One completed call before the hang must not decide whether the
        # timeout halves pi's mean; nor may a bigger floor that is still below
        # what pi spent finishing the same task.
        for tokens in ((6000, 0, 250), (353896, 320384, 82917)):
            with self.subTest(tokens=tokens):
                rows = self.pilot_rows(tokens)
                ledger = FakeLedger(rows)
                cfg = self.config()
                rep = self.build(ledger, cfg, {"include_pairs": True})
                pi = next(g for g in rep["groups"] if g["family"] == "pi")
                real_cost = rows[0]["cost_usd"]
                self.assertEqual(rows[1]["usage_status"], "lower_bound")  # the ledger's TrialRow is left alone
                self.assertEqual(pi["per_trial"]["cost"]["n"], 1)
                self.assertAlmostEqual(pi["per_trial"]["cost"]["mean"], real_cost)
                self.assertEqual((pi["trials"]["usage_unmeasured"], pi["trials"]["usage_lower_bound"]), (1, 0))
                self.assertEqual(pi["trials"]["valid_reward"], 2)
                c = self.comparison(rep, "pi", where=GLM)
                self.assertAlmostEqual(c["pairs_table"][0]["ratio_cost"], real_cost / rows[2]["cost_usd"])
                self.assertIn("1 pi trial cut short by a time limit recorded fewer tokens than every uncut pi trial "
                              "on the same task, though it ran longer, so part of what it spent is missing; like "
                              "trials with no usage recorded, it counts for pass rate only.", c["caveats"])
                self.assertFalse(any(x.startswith(("No usage was recorded", "1 pi trial hit")) for x in c["caveats"]))
                excluded = report.build_pairs(ledger, cfg, Filters(), None, c["key"])["excluded_trials"]
                self.assertEqual([(e["trial_key"], e["reason"]) for e in excluded],
                                 [(rows[1]["trial_key"], "cut short below every uncut trial on its task")])
                row = next(r for r in report.analysis_rows(ledger, cfg) if r["trial_key"] == rows[1]["trial_key"])
                self.assertEqual((row["usage_status"], row["cost_usd"], row["valid_cost"], row["valid_reward"]),
                                 ("unmeasured", None, False, True))
                # What it recorded was still spent.
                self.assertAlmostEqual(rep["totals"]["spend_usd"], math.fsum(r["cost_usd"] for r in rows))

    def test_equal_counts_on_different_tasks_still_get_the_caveat(self) -> None:
        # Each side lost two trials, on different tasks: four shared tasks have no value on one side.
        rows = pair_rows([2.45, 2.55, 2.5, 2.4, 2.6, 2.5, 2.45, 2.55])
        for j in range(2):
            rows += [trow(f"big-{j}", "claude-code", tokens=(8000000, 3200000, 160000)),
                     trow(f"big-{j}", "pi", tokens=(0, 0, 0), outcome="agent_limit", passed=False,
                          exception_type="AgentTimeoutError", total_sec=960.0),
                     trow(f"small-{j}", "pi", tokens=(30000, 12000, 600)),
                     trow(f"small-{j}", "claude-code", tokens=(0, 0, 0), outcome="agent_limit", passed=False,
                          exception_type="AgentTimeoutError", total_sec=960.0)]
        c = self.comparison(self.build(FakeLedger(rows), self.config()), "pi")
        self.assertEqual(c["metrics"]["cost"]["dropped"]["usage_not_recorded"], 4)
        self.assertIn("No usage was recorded for 2 of 12 pi trials and 2 of 12 claude-code trials; they count for "
                      "pass rate but are left out of tokens and cost, which leaves 4 shared tasks without a value on "
                      "one side. Such trials often ran to a time limit, so the side with more of them may look "
                      "cheaper than it is.", c["caveats"])
        # Pooled into tasks that keep a value, an even share is still said.
        rows = pair_rows(KS[:4])
        for family in ("pi", "claude-code"):
            rows.append(trow("task-0", family, tokens=(0, 0, 0), outcome="agent_limit", passed=False,
                             exception_type="AgentTimeoutError"))
        c = self.comparison(self.build(FakeLedger(rows), self.config()), "pi")
        self.assertIn("No usage was recorded for 1 of 5 pi trials and 1 of 5 claude-code trials; they count for pass "
                      "rate but are left out of tokens and cost. Such trials often ran to a time limit, so the side "
                      "with more of them may look cheaper than it is.", c["caveats"])

    def near(self, ks: Sequence[float], run_key: str = "dev/r1") -> list[dict]:
        """pair_rows over tasks of similar size (1.0x to 1.9x), as a sample's
        tasks are closer in size than pair_rows' (1x to 10x)."""
        rows = []
        for i, k in enumerate(ks):
            s = 1 + 0.1 * i
            a = (int(100000 * s), int(40000 * s), int(2000 * s))
            b = (round(k * a[0]), round(k * a[1]), round(k * a[2]))
            rows.append(trow(f"task-{i}", "pi", tokens=a, run_key=run_key))
            rows.append(trow(f"task-{i}", "claude-code", tokens=b, run_key=run_key))
        return rows

    def test_a_verdict_that_rests_on_floors_is_withheld(self) -> None:
        # The arm's three biggest tasks timed out after one completed call:
        # the floors, kept at their recorded value, move 'above' to 'overlaps'.
        ks = [1.55, 1.65, 1.6, 1.5, 1.7, 1.6, 1.55, 1.65, 1.6, 1.6]
        cfg = self.config()
        clean = self.comparison(self.build(FakeLedger(self.near(ks)), cfg), "pi")["metrics"]["cost"]
        self.assertEqual(clean["reference"]["verdict"], "above")
        self.assertEqual((clean["sensitivity"]["unrecorded_arm_high"], clean["robust"]), (None, True))
        rows = [r for r in self.near(ks) if not (r["family"] == "pi" and r["task"] in ("task-7", "task-8", "task-9"))]
        rows += [trow(f"task-{i}", "pi", tokens=(6000, 0, 250), outcome="agent_limit", passed=False,
                      exception_type="AgentTimeoutError", total_sec=960.0) for i in (7, 8, 9)]
        c = self.comparison(self.build(FakeLedger(rows), cfg), "pi")
        cost = c["metrics"]["cost"]
        self.assertEqual((cost["status"], cost["n"]), ("ok", 10))  # the estimate is still shown
        reason = ("3 paired tasks rest on pi trials cut short or with no usage recorded; had those cost 2x pi's usual "
                  "spend on their task, the verdict would be 'above', not 'overlaps'")
        self.assertEqual((cost["reference"]["verdict"], cost["reference"]["withheld"]), ("insufficient", reason))
        self.assertIn(f"verdict withheld: {reason}", c["headline"])
        self.assertIn(f"No cost verdict: {reason}.", c["caveats"])
        self.assertGreater(cost["sensitivity"]["unrecorded_arm_high"], cost["geomean"]["hi"])  # pi dearer
        self.assertIsNone(cost["sensitivity"]["unrecorded_baseline_high"])
        self.assertIs(cost["robust"], False)

    def test_a_task_lost_on_one_side_is_checked_both_ways(self) -> None:
        # One trial a side per task: a side that recorded nothing on the three
        # biggest tasks loses them, and the other nine pairs alone say 'within'.
        easy = self.near([2.45, 2.55, 2.5, 2.4, 2.6, 2.5, 2.45, 2.55, 2.5])
        cfg = self.config()
        self.assertEqual(self.comparison(self.build(FakeLedger(easy), cfg), "pi")["metrics"]["cost"]["reference"]
                         ["verdict"], "within")
        timeout: dict[str, Any] = dict(tokens=(0, 0, 0), outcome="agent_limit", passed=False,
                                       exception_type="AgentTimeoutError", total_sec=960.0)
        for lost, kept, tokens in (("pi", "claude-code", (800000, 320000, 16000)),
                                   ("claude-code", "pi", (200000, 80000, 4000))):
            with self.subTest(lost=lost):
                rows = list(easy)
                for j in range(3):
                    rows += [trow(f"hard-{j}", kept, tokens=tokens), trow(f"hard-{j}", lost, **timeout)]
                c = self.comparison(self.build(FakeLedger(rows), cfg), "pi")
                cost = c["metrics"]["cost"]
                self.assertEqual((cost["n"], cost["dropped"]["usage_not_recorded"]), (9, 3))
                self.assertEqual(cost["reference"]["verdict"], "insufficient")
                self.assertTrue(cost["reference"]["withheld"].startswith(
                    f"3 paired tasks rest on {lost} trials cut short or with no usage recorded"), cost["reference"])
                self.assertTrue(any("which leaves 3 shared tasks without a value on one side" in x
                                    for x in c["caveats"]))

    def test_coverage_plans_a_trial_without_usage_as_an_expensive_one(self) -> None:
        # The descriptive mean leaves it out; a plan for more runs must not
        # price the tasks a harness times out on like its typical ones.
        rows = [trow("task-0", "pi", tokens=(100000, 40000, 2000)), trow("task-1", "pi", tokens=(300000, 120000, 6000)),
                trow("task-2", "pi", tokens=(0, 0, 0), outcome="agent_limit", passed=False,
                     exception_type="AgentTimeoutError", total_sec=960.0)]
        rows += [trow(f"task-{i}", "claude-code") for i in range(3)]
        rep = self.build(FakeLedger(rows), self.config())
        pi = next(g for g in rep["groups"] if g["family"] == "pi")
        cheap, dear = rows[0]["cost_usd"], rows[1]["cost_usd"]
        self.assertAlmostEqual(pi["per_trial"]["cost"]["mean"], (cheap + dear) / 2)
        plan = self.comparison(rep, "pi")["coverage"]["per_trial"]["arm"]["mean_cost_usd"]
        self.assertAlmostEqual(plan, (cheap + 2 * dear) / 3)

    def test_a_verdict_that_holds_with_the_trials_costed_high_is_kept(self) -> None:
        # A second run of every task, where the arm recorded nothing on four
        # of them: costed at 2x its usual spend there, 'within' still holds.
        ks = [2.5, 2.45, 2.55, 2.5, 2.4, 2.6, 2.5, 2.45, 2.55, 2.5, 2.5, 2.5]
        lost = ("task-0", "task-1", "task-2", "task-3")
        rows = self.near(ks) + [r for r in self.near(ks, run_key="dev/r2")
                                if not (r["family"] == "pi" and r["task"] in lost)]
        rows += [trow(task, "pi", tokens=(0, 0, 0), outcome="agent_limit", passed=False, run_key="dev/r2",
                      exception_type="AgentTimeoutError", total_sec=960.0) for task in lost]
        cost = self.comparison(self.build(FakeLedger(rows), self.config()), "pi")["metrics"]["cost"]
        self.assertEqual((cost["reference"]["verdict"], cost["reference"]["withheld"]), ("within", None))
        self.assertGreater(cost["sensitivity"]["unrecorded_arm_high"], cost["geomean"]["estimate"])


class Pricing(ReportCase):
    def test_pricing_modes_after_a_pricing_edit(self) -> None:
        rows = []
        for i in range(5):
            a, b = scaled(2.0, i)
            for family, tokens in (("pi", a), ("claude-code", b)):
                row = trow(f"task-{i}", family, tokens=tokens)
                row["reported_cost_usd"] = row["cost_usd"] / 2
                rows.append(row)
        rows.append(trow("task-0", "claude-code", provider=DEEPSEEK[0], model=DEEPSEEK[1], tokens=(100000, 0, 1000),
                         reported=5.0))
        ledger = FakeLedger(rows)
        cfg = self.config()
        old = self.build(ledger, cfg)
        doubled = self.tmp / "pricing-2.yaml"
        doubled.write_text(PRICING_YAML.replace("{input: 4.00, output: 20.00, cache_read: 0.20}",
                                                "{input: 8.00, output: 40.00, cache_read: 0.40}"),
                           encoding="utf-8", newline="\n")
        with mock.patch.object(settings, "PRICING", doubled):
            results = {mode: self.build(ledger, cfg, {"pricing": mode})
                       for mode in ("current", "as_ingested", "reported")}
            reported_rows = report.analysis_rows(ledger, cfg, Filters(), report.parse_options({"pricing": "reported"},
                                                                                               cfg))

        def pi_cost(rep: dict) -> float:
            return next(g for g in rep["groups"] if g["family"] == "pi")["per_trial"]["cost"]["mean"]

        base = pi_cost(old)
        self.assertAlmostEqual(pi_cost(results["current"]), 2 * base)
        self.assertAlmostEqual(pi_cost(results["as_ingested"]), base)
        self.assertAlmostEqual(pi_cost(results["reported"]), base / 2)
        self.assertNotEqual(results["current"]["provenance"]["pricing_sha"], old["provenance"]["pricing_sha"])
        self.assertEqual(results["reported"]["provenance"]["pricing"], "reported")
        for rep in results.values():  # tokens never move with prices
            self.assertEqual(self.comparison(rep, "pi")["metrics"]["tokens_total"]["geomean"]["estimate"],
                             self.comparison(old, "pi")["metrics"]["tokens_total"]["geomean"]["estimate"])
        off_anthropic = next(r for r in reported_rows if r["provider"] == "openrouter")
        self.assertLess(off_anthropic["cost_usd"], 1.0)  # Claude Code's own figure is not trusted off Anthropic
        self.assertEqual(off_anthropic["reported_cost_usd"], 5.0)

    def test_a_zero_reported_cost_with_tokens_falls_back_to_the_price_table(self) -> None:
        rows = []
        for i in range(10):
            a, b = scaled(2.0, i)
            for family, tokens in (("pi", a), ("claude-code", b)):
                row = trow(f"task-{i}", family, tokens=tokens)
                row["reported_cost_usd"] = 0.0 if family == "pi" else row["cost_usd"]
                rows.append(row)
        cfg = self.config()
        current = self.comparison(self.build(FakeLedger(rows), cfg), "pi")["metrics"]["cost"]
        reported = self.comparison(self.build(FakeLedger(rows), cfg, {"pricing": "reported"}), "pi")["metrics"]["cost"]
        self.assertEqual((current["status"], current["n"]), ("ok", 10))
        self.assertEqual((reported["status"], reported["n"]), ("ok", 10))

    def test_a_missing_price_is_named(self) -> None:
        where = ("openrouter", "unknown/unpriced-model")
        rows = pair_rows(KS, where=where)
        rep = self.build(FakeLedger(rows), self.config())
        c = self.comparison(rep, "pi", where=where)
        cost = c["metrics"]["cost"]
        self.assertEqual((cost["status"], cost["n"], cost["dropped"]["unpriced"]), ("not_measured", 0, 10))
        self.assertEqual(c["metrics"]["tokens_total"]["status"], "ok")
        self.assertEqual(c["coverage"]["metric"], "tokens_total")
        self.assertIn("No price in config/pricing.yaml for unknown/unpriced-model; cost multiples leave these "
                      "trials out until a price is added.", c["caveats"])
        priced = self.comparison(self.build(FakeLedger(pair_rows(KS, where=GLM)), self.config()), "pi", where=GLM)
        self.assertFalse(any(x.startswith("No price in config/pricing.yaml") for x in priced["caveats"]))


class CostBasis(ReportCase):
    """Which trials count toward a cost multiple, and on which figures."""

    def test_a_fallback_model_trial_never_counts_toward_cost(self) -> None:
        rows = pair_rows(KS)
        rows[0]["primary_share"] = 0.04  # task-0's pi trial: a fallback model answered nearly all of it
        rep = self.build(FakeLedger(rows), self.config())
        c = self.comparison(rep, "pi")
        self.assertEqual((c["metrics"]["cost"]["n"], c["metrics"]["tokens_total"]["n"]), (9, 10))
        self.assertIn("1 pi trial spent more than 5% of its tokens on models other than claude-opus-5-5 (a fallback "
                      "model answered); it counts for pass rate only.", c["caveats"])
        self.assertEqual(c["pass"]["n"], 10)
        pairs = report.build_pairs(FakeLedger(rows), self.config(), None, None, c["key"])
        (left,) = [t for t in pairs["excluded_trials"] if t["trial_key"] == rows[0]["trial_key"]]
        self.assertEqual(left["reason"], "fallback model: 96% of its tokens on models other than claude-opus-5-5")
        self.assertFalse(report.fallback_model({"primary_share": None}))
        self.assertFalse(report.fallback_model({"primary_share": report.PRIMARY_SHARE_MIN}))

    def test_reported_pricing_uses_the_harnesses_figures_only_where_both_sides_have_them(self) -> None:
        def priced(pi_factor: float | None) -> list[dict]:
            rows = pair_rows(KS)
            for r in rows:
                factor = 1.5 if r["family"] == "claude-code" else pi_factor
                r["reported_cost_usd"] = None if factor is None else r["cost_usd"] * factor
            return rows

        cfg = self.config()

        def cost(rows: list[dict], mode: str) -> float:
            rep = self.build(FakeLedger(rows), cfg, {"pricing": mode})
            return self.comparison(rep, "pi")["metrics"]["cost"]["geomean"]["estimate"]

        current = cost(priced(None), "current")
        # pi reports nothing (as strands does): both sides are priced from the table, never pi's computed figure
        # against Claude Code's own bill.
        self.assertAlmostEqual(cost(priced(None), "reported"), current)
        # Both report: their own figures.
        self.assertAlmostEqual(cost(priced(1.2), "reported"), current * 1.2 / 1.5)
        # Task by task: pi reports on half the tasks, so only those pairs use the reported figures.
        rows = priced(1.2)
        for r in rows:
            if r["family"] == "pi" and int(r["task"].split("-")[1]) % 2:
                r["reported_cost_usd"] = None
        mixed = report.build_pairs(FakeLedger(rows), cfg, None, report.parse_options({"pricing": "reported"}, cfg),
                                   self.comparison(self.build(FakeLedger(rows), cfg), "pi")["key"])["pairs"]
        plain = report.build_pairs(FakeLedger(rows), cfg, None, None,
                                   self.comparison(self.build(FakeLedger(rows), cfg), "pi")["key"])["pairs"]
        for m, p in zip(mixed, plain):
            factor = 1.0 if int(m["task"].split("-")[1]) % 2 else 1.2 / 1.5
            self.assertAlmostEqual(m["ratio_cost"], p["ratio_cost"] * factor)

    def test_flagged_trials_get_caveats_with_their_bounds(self) -> None:
        rows = pair_rows(KS)
        pi = [r for r in rows if r["family"] == "pi"]
        cc = [r for r in rows if r["family"] == "claude-code"]
        pi[0]["flags"] = {"unsaved_calls": {"calls": 1, "cost_usd_max": 0.0096}}
        pi[1]["flags"] = {"unsaved_calls": {"calls": 2, "cost_usd_max": 0.02}}
        cc[0]["flags"] = {"declined_attempts": {"attempts": 1, "cost_usd_max": 0.026},
                          "reported_cost_mismatch": {"reported_usd": 0.72, "computed_usd": 0.70, "diff_pct": 3.689}}
        cc[1]["flags"] = {"cache_writes_unknown": {"calls": 0, "models": ["claude-opus-5-5"],
                                                   "reason": "no harness log"}}
        cc[2]["flags"] = {"unfinished_calls": {"calls": 1, "cost_usd_max": None}, "trial_folder_missing": True}
        caveats = self.comparison(self.build(FakeLedger(rows), self.config()), "pi")["caveats"]
        expected = [
            "2 pi trials lost model calls that completed but were never saved (a timeout during a command): 3 calls "
            "missing from the cost (up to $0.03 in all).",
            "1 claude-code trial has model attempts declined before another model answered (a refusal fallback); "
            "only the session log lists them and whether they are billed is unknown, so the cost leaves them out "
            "(up to $0.03 in all).",
            "1 claude-code trial started model calls that never finished (a kill mid-call): 1 call missing from the "
            "cost.",
            "1 claude-code trial reports a cost of its own more than 0.5% from the cost computed from its tokens "
            "(largest +3.7%): a call the log does not show, or a price that differs.",
            "1 claude-code trial has cache writes no harness log gave; they are priced as fresh input, so the cost "
            "is low by the write premium.",
            "1 claude-code trial had no trial folder when last re-derived; their cache writes and log checks are the "
            "figures stored before.",
        ]
        for text in expected:
            self.assertIn(text, caveats)


class ModelComparisons(ReportCase):
    """compare 'models': the harness held fixed, each model against the
    baseline model on the same task content, model / baseline model."""

    def rows(self, *, deepseek_snapshot: str = "local:tb@1") -> list[dict]:
        rows = []
        for i, k in enumerate(KS):
            cheap, opus = scaled(k, i)  # DeepSeek's tokens are 1/k of Opus's on task i
            for family in ("claude-code", "pi"):
                rows.append(trow(f"task-{i}", family, tokens=opus))
                rows.append(trow(f"task-{i}", family, provider=DEEPSEEK[0], model=DEEPSEEK[1], tokens=cheap,
                                 outcome="failed" if i % 2 else "passed", snapshot=deepseek_snapshot))
        return rows

    @staticmethod
    def key(family: str, where: tuple[str, str]) -> str:
        return f"fam:{where[0]}|{where[1]}|{family}"

    def test_each_harness_compares_each_model_with_the_baseline_model(self) -> None:
        cfg = self.config()
        rep = self.build(FakeLedger(self.rows()), cfg, {"compare": "models"})
        self.assertEqual(rep["spec"]["options"]["compare"], "models")
        self.assertEqual(rep["spec"]["options"]["baseline"]["model"], "anthropic:claude-opus-5-5")
        keys = sorted(c["key"] for c in rep["comparisons"])
        self.assertEqual(keys, sorted(f"{self.key(f, DEEPSEEK)}~{self.key(f, OPUS)}" for f in ("claude-code", "pi")))
        c = next(c for c in rep["comparisons"] if c["harness"] == "claude-code")
        self.assertEqual((c["compare"], c["provider"], c["model"], c["baseline_provider"], c["baseline_model"]),
                         ("models", DEEPSEEK[0], DEEPSEEK[1], OPUS[0], OPUS[1]))
        self.assertEqual((c["arm_name"], c["baseline_name"]), (DEEPSEEK[1], OPUS[1]))
        tokens = c["metrics"]["tokens_total"]
        self.assertEqual((tokens["status"], tokens["n"]), ("ok", 10))
        # model / baseline model: DeepSeek used 1/k of Opus's tokens on each task.
        self.assertAlmostEqual(tokens["geomean"]["estimate"], math.exp(-statistics.fmean(math.log(k) for k in KS)))
        # Pass rate: candidate minus baseline, in points.
        self.assertAlmostEqual(c["pass"]["delta_pp"]["estimate"], -50.0)
        # The published figures compare harnesses: no reference and no verdict on a model comparison.
        self.assertTrue(all(m["reference"] is None for m in c["metrics"].values()))
        self.assertIsNone(c["pass"]["reference"])
        self.assertIn(report.MODELS_NO_VERDICT, c["caveats"])
        self.assertTrue(c["headline"].startswith(f"{DEEPSEEK[1]} vs {OPUS[1]} with claude-code: {DEEPSEEK[1]} used "),
                        c["headline"])
        baselines = sorted(g["key"] for g in rep["groups"] if g["is_baseline"])
        self.assertEqual(baselines, sorted(self.key(f, OPUS) for f in ("claude-code", "pi")))
        # The same rows compared the usual way: harnesses on each model.
        harnesses = self.build(FakeLedger(self.rows()), cfg)
        self.assertEqual(sorted(c["key"] for c in harnesses["comparisons"]),
                         sorted(f"{self.key('pi', w)}~{self.key('claude-code', w)}" for w in (OPUS, DEEPSEEK)))
        self.assertTrue(all(c["compare"] == "harnesses" for c in harnesses["comparisons"]))

    def test_the_baseline_model_comes_from_the_config_and_the_option_overrides_it(self) -> None:
        ledger = FakeLedger(self.rows())
        flipped = self.config(report_yaml=f"baseline_model: {DEEPSEEK[0]}:{DEEPSEEK[1]}\n")
        self.assertEqual(flipped.baseline_model, f"{DEEPSEEK[0]}:{DEEPSEEK[1]}")
        self.assertEqual(self.config().baseline_model, "anthropic:claude-opus-5-5")
        override = {"compare": "models", "baseline": {"model": f"{DEEPSEEK[0]}:{DEEPSEEK[1]}"}}
        for cfg, options in ((flipped, {"compare": "models"}), (self.config(), override)):
            with self.subTest(options=options):
                c = next(c for c in self.build(ledger, cfg, options)["comparisons"] if c["harness"] == "pi")
                self.assertEqual((c["model"], c["baseline_model"]), (OPUS[1], DEEPSEEK[1]))
                self.assertAlmostEqual(c["metrics"]["tokens_total"]["geomean"]["estimate"],
                                       math.exp(statistics.fmean(math.log(k) for k in KS)))
                self.assertAlmostEqual(c["pass"]["delta_pp"]["estimate"], 50.0)
        with self.assertRaises(settings.ConfigError):
            self.config(report_yaml="baseline_model: claude-opus-5-5\n")

    def test_pairs_need_the_same_task_content_across_models(self) -> None:
        rep = self.build(FakeLedger(self.rows(deepseek_snapshot="local:tb@2")), self.config(), {"compare": "models"})
        c = next(c for c in rep["comparisons"] if c["harness"] == "pi")
        self.assertEqual(c["metrics"]["tokens_total"]["n"], 0)
        self.assertEqual(len(c["pairs"]["arm_only_tasks"]), 10)
        by_name = self.build(FakeLedger(self.rows(deepseek_snapshot="local:tb@2")), self.config(),
                             {"compare": "models", "pairing": {"match_on": "name"}})
        c = next(c for c in by_name["comparisons"] if c["harness"] == "pi")
        self.assertEqual(c["metrics"]["tokens_total"]["n"], 10)

    def test_a_low_count_headline_never_promises_a_verdict(self) -> None:
        rows = [r for r in self.rows() if int(r["task"].split("-")[1]) < 5]
        c = next(c for c in self.build(FakeLedger(rows), self.config(), {"compare": "models"})["comparisons"]
                 if c["harness"] == "pi")
        self.assertEqual(c["metrics"]["cost"]["status"], "low_n")
        self.assertTrue(c["headline"].endswith("; 3 more for a settled figure"), c["headline"])

    def test_grouped_by_config_both_sides_name_the_harness_alike(self) -> None:
        # A config hash covers the model, so the two sides' group names differ; the harness name must not.
        rows = self.rows()
        for r in rows[:2]:
            r["primary_share"] = 0.5  # claude-code on each model: a fallback model answered one task
        rep = self.build(FakeLedger(rows), self.config(), {"compare": "models", "group_by": "config"})
        self.assertEqual(len(rep["comparisons"]), 2)
        for c in rep["comparisons"]:
            groups = {g["key"]: g for g in rep["groups"]}
            arm, base = groups[c["group"]], groups[c["baseline_group"]]
            self.assertNotEqual(arm["name"], base["name"])
            self.assertEqual((arm["harness_key"], arm["harness_name"]), (base["harness_key"], base["harness_name"]))
            self.assertEqual(c["harness"], arm["harness_name"])
            self.assertRegex(c["harness"], r"^(claude-code|pi) [0-9a-f]{6}$")
            self.assertNotIn(arm["config_hash"][:6], c["headline"])
            self.assertTrue(c["headline"].startswith(f"{DEEPSEEK[1]} vs {OPUS[1]} with {c['harness']}: "),
                            c["headline"])
            if c["harness"].startswith("claude-code"):
                # The caveats name each side by the one harness name and its model.
                fallback = sorted(t.split(" trial ")[0] for t in c["caveats"] if "a fallback model answered" in t)
                self.assertEqual(fallback, [f"1 {c['harness']} on {m}" for m in sorted((DEEPSEEK[1], OPUS[1]))])
        # Grouped by family, the harness name is the family, as before.
        family = self.build(FakeLedger(self.rows()), self.config(), {"compare": "models"})
        self.assertEqual(sorted(c["harness"] for c in family["comparisons"]), ["claude-code", "pi"])

    def test_a_harness_without_the_baseline_model_is_named(self) -> None:
        rows = [r for r in self.rows() if not (r["family"] == "pi" and r["provider"] == OPUS[0])]
        rep = self.build(FakeLedger(rows), self.config(), {"compare": "models"})
        self.assertEqual([c["harness"] for c in rep["comparisons"]], ["claude-code"])
        self.assertIn("No trials on anthropic:claude-opus-5-5 with pi, so 1 group there has no comparison.",
                      rep["warnings"])

    def test_bad_compare_options_are_refused(self) -> None:
        cfg = self.config()
        for bad in ({"compare": "arms"}, {"baseline": {"model": "claude-opus-5-5"}}, {"baseline": {"model": 5}},
                    {"baseline": {"model": ":x"}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                report.parse_options(bad, cfg)
        o = report.parse_options({"compare": "models", "baseline": {"model": " openrouter : z-ai/glm-5.3 "}}, cfg)
        self.assertEqual((o.compare, o.baseline_model), ("models", "openrouter:z-ai/glm-5.3"))
        self.assertEqual(report.parse_options(report.options_dict(o), cfg), o)
        self.assertEqual(report.parse_options(None, self.config(report_yaml="report: {compare: models}\n")).compare,
                         "models")


class Options(ReportCase):
    def test_defaults_come_from_the_config(self) -> None:
        cfg = self.config()
        o = report.parse_options(None, cfg)
        self.assertEqual((o.ci_level, o.target_ratio_halfwidth, o.prior_sd_log), (0.90, 1.25, 0.6))
        self.assertEqual((o.group_by, o.crashes, o.misconfigured_cap, o.same_run_only, o.match_on, o.pricing),
                         ("family", "include", "exclude", False, "digest", "current"))
        self.assertEqual((o.min_pairs_estimate, o.min_pairs_verdict, o.baseline_family), (3, 8, "claude-code"))
        self.assertEqual(report.parse_options(report.options_dict(o), cfg), o)
        tuned = self.config(report_yaml="""
            report:
              ci_level: 0.95
              pricing: reported
              bootstrap_resamples: 1000
              exclusions: false
              target_pass_delta_pp: 10
            """)
        t = report.parse_options({"pricing": "as_ingested"}, tuned)
        self.assertEqual((t.ci_level, t.pricing, t.resamples, t.apply_exclusions, t.target_pass_delta_pp),
                         (0.95, "as_ingested", 1000, False, 10.0))

    def test_bad_options_are_refused(self) -> None:
        cfg = self.config()
        for bad in ({"nope": 1}, {"pairing": {"match_on": "fuzzy"}}, {"pairing": {"nope": True}},
                    {"pairing": "digest"}, {"group_by": "arm"}, {"ci_level": 0.5}, {"ci_level": True},
                    {"bootstrap": {"resamples": 100}}, {"bootstrap": {"seed": 1.5}}, {"min_pairs": {"estimate": 1}},
                    {"min_pairs": {"estimate": 5, "verdict": 4}}, {"policy": {"crashes": "maybe"}},
                    {"pricing": "cheapest"}, {"coverage_target": {"ratio_halfwidth": 1.0}},
                    {"baseline": {"config": "NOT-HEX"}}, {"include_pairs": "yes"}, ["group_by"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                report.parse_options(bad, cfg)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            report.parse_options(None, self.config(report_yaml="report: {ci_levl: 0.9}\n"))
        with self.assertRaises(ValueError):
            report.parse_options(None, self.config(report_yaml="report: {pricing: free}\n"))
        for bad in ({"nope": []}, {"tasks": "a"}, {"date_from": "yesterday"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                report.parse_filters(bad)


class Coverage(ReportCase):
    def test_next_never_leaves_the_sample(self) -> None:
        sample = [f"task-{i}" for i in range(5)]
        rows = pair_rows(KS[:1]) + [trow("task-1", "pi"), trow("outside-1", "pi"), trow("outside-2", "claude-code")]
        c = self.comparison(self.build(FakeLedger(rows), self.config(tasks=sample)), "pi")
        cov = c["coverage"]
        self.assertEqual(cov["est_pairs_needed"], 5)  # the prior asks for 22
        self.assertEqual(cov["est_more_pairs"], 4)
        self.assertEqual(cov["next"], [{"task": "task-1", "side": "baseline", "reason": "complete pair"},
                                       {"task": "task-2", "side": "both", "reason": "new task"},
                                       {"task": "task-3", "side": "both", "reason": "new task"},
                                       {"task": "task-4", "side": "both", "reason": "new task"}])
        self.assertTrue(all(x["task"] in sample for x in cov["next"]))
        self.assertTrue(any("remaining uncertainty" in x for x in c["caveats"]))
        means = cov["per_trial"]
        self.assertAlmostEqual(cov["est_cost_usd"], means["baseline"]["mean_cost_usd"]
                               + 3 * (means["arm"]["mean_cost_usd"] + means["baseline"]["mean_cost_usd"]))
        self.assertEqual(c["pairs"]["arm_only_tasks"], ["task-1", "outside-1"])

    def test_observed_spread_sizes_the_plan(self) -> None:
        sample = [f"task-{i}" for i in range(20)]
        c = self.comparison(self.build(FakeLedger(pair_rows(KS)), self.config(tasks=sample)), "pi")
        cov = c["coverage"]
        sd = statistics.stdev(math.log(k) for k in KS)
        self.assertEqual((cov["sd_source"], cov["pairs"], cov["metric"]), ("observed", 10, "cost"))
        self.assertAlmostEqual(cov["sd_log_used"], sd)
        self.assertAlmostEqual(cov["halfwidth_factor_now"], math.exp(stats.t_quantile(0.9, 9) * sd / math.sqrt(10)))
        needed = stats.n_needed_ratio(sd, 1.25, 0.9)
        self.assertEqual(cov["est_pairs_needed"], min(needed, 20))
        self.assertEqual(cov["est_more_pairs"], max(0, min(needed, 20) - 10))
        self.assertEqual([x["task"] for x in cov["next"]], sample[10:10 + cov["est_more_pairs"]])
        self.assertEqual((cov["need_for_estimate"], cov["need_for_verdict"]), (0, 0))
        self.assertEqual(cov["pass_tasks_needed"], stats.n_needed_pass(0.3, 5, 0.9))

    def test_report_matrix_and_untouched_tasks(self) -> None:
        sample = [f"task-{i}" for i in range(4)]
        rows = pair_rows(KS[:2]) + [trow("task-0", "pi", outcome="setup_failure", tokens=None)]
        ledger = FakeLedger(rows)
        cfg = self.config(tasks=sample)
        rep = self.build(ledger, cfg)
        cells = rep["coverage"]["cells"]["fam:anthropic|claude-opus-5-5|pi"]
        self.assertEqual(cells["task-0"], {"valid": 1, "passed": 1, "by_outcome": {"passed": 1, "setup_failure": 1}})
        coverage = report.build_coverage(ledger, cfg, Filters(), report.parse_options(None, cfg))
        self.assertEqual(coverage["untouched_tasks"], ["task-2", "task-3"])


class Plumbing(ReportCase):
    def test_memo_follows_the_ledger_version(self) -> None:
        ledger = FakeLedger(pair_rows(KS))
        cfg = self.config()
        first = self.build(ledger, cfg)
        self.build(ledger, cfg)
        self.assertEqual(ledger.queries, 1)
        ledger.stamp += 1
        self.build(ledger, cfg)
        self.assertEqual(ledger.queries, 2)
        self.build(ledger, cfg, {"bootstrap": {"seed": 7}})
        self.assertEqual(ledger.queries, 3)
        self.assertEqual(first["provenance"]["fingerprint"], self.build(ledger, cfg)["provenance"]["fingerprint"])

    def test_hand_exclusions(self) -> None:
        rows = pair_rows(KS[:4])
        include = {"target_type": "trial", "action": "include", "reason": "checked", "created_at": "x"}
        rows.append(trow("task-9", "pi", run_key="dev/r9", exclusion=include))
        rows.append(trow("task-9", "claude-code", run_key="dev/r9",
                         exclusion={"target_type": "run", "action": "exclude", "reason": "bad run", "created_at": "x"}))
        ledger = FakeLedger(rows, excluded_trials=[rows[0]["trial_key"]], excluded_runs=["dev/r9"], ids=[3, 4])
        rep = self.build(ledger, self.config())
        self.assertEqual((rep["totals"]["trials"], rep["totals"]["excluded_manual"]), (8, 2))
        self.assertEqual(rep["provenance"]["exclusions_applied"], 2)
        reasons = {r["trial_key"]: r["excluded_reason"] for r in report.analysis_rows(ledger, self.config())}
        self.assertEqual(reasons[rows[0]["trial_key"]], "excluded by hand: trial exclusion")
        self.assertIsNone(reasons[rows[-2]["trial_key"]])  # put back inside an excluded run
        self.assertEqual(reasons[rows[-1]["trial_key"]], "run excluded by hand: bad run")
        everything = self.build(ledger, self.config(), {"policy": {"exclusions": False}})
        self.assertEqual(everything["totals"]["trials"], 10)
        self.assertNotEqual(everything["provenance"]["fingerprint"], rep["provenance"]["fingerprint"])

    def test_export_levels(self) -> None:
        ledger = FakeLedger(pair_rows(KS))
        cfg = self.config()
        o = report.parse_options(None, cfg)
        trials = list(report.export_rows(ledger, cfg, Filters(), o, "trials"))
        self.assertEqual(trials[0], [*report.ANALYSIS_KEYS, "filters"])
        self.assertEqual(len(trials), 21)
        self.assertEqual({row[-1] for row in trials[1:]}, {"{}"})
        pairs = list(report.export_rows(ledger, cfg, Filters(), o, "pairs"))
        self.assertEqual((pairs[0][0], len(pairs)), ("comparison", 11))
        comparisons = list(report.export_rows(ledger, cfg, Filters(), o, "comparisons"))
        self.assertEqual([r[5] for r in comparisons[1:]], ["tokens_total", "tokens_uncached", "cost", "pass"])
        self.assertTrue(all(isinstance(v, str) for row in comparisons for v in row))
        # Every row says how its multiples read and which filters made it.
        scoped = Filters(tasks=("task-0", "task-1", "task-2"))
        for level, rows in (("pairs", list(report.export_rows(ledger, cfg, scoped, o, "pairs"))),
                            ("comparisons", list(report.export_rows(ledger, cfg, scoped, o, "comparisons")))):
            with self.subTest(level):
                self.assertEqual(rows[0][-2:], ["direction", "filters"])
                self.assertEqual({tuple(r[-2:]) for r in rows[1:]},
                                 {("harness_over_baseline", '{"tasks":["task-0","task-1","task-2"]}')})
        head = comparisons[0]
        cost = next(r for r in comparisons[1:] if r[5] == "cost")
        c = self.comparison(self.build(ledger, cfg), "pi")["metrics"]["cost"]
        self.assertEqual(float(cost[head.index("geomean")]), c["geomean"]["estimate"])
        self.assertLess(float(cost[head.index("geomean")]), 1.0)  # pi is the cheaper side: under 1x
        with self.assertRaises(ValueError):
            report.export_rows(ledger, cfg, Filters(), o, "everything")  # fails on the call itself

    def test_active_runs_are_flagged(self) -> None:
        rep = self.build(FakeLedger(pair_rows(KS[:3]), statuses={"dev/r1": "running"}), self.config())
        self.assertTrue(rep["provenance"]["includes_active_run"])
        self.assertTrue(any("still in progress" in w for w in rep["warnings"]))
        stopped = self.build(FakeLedger(pair_rows(KS[:3]), statuses={"dev/r1": "budget_stopped"}), self.config())
        self.assertTrue(any("stopped early" in x for x in self.comparison(stopped, "pi")["caveats"]))

    def test_report_shape(self) -> None:
        rep = self.build(FakeLedger(pair_rows(KS)), self.config())
        self.assertEqual(set(rep), {"spec", "provenance", "scope", "totals", "groups", "comparisons", "coverage",
                                    "warnings"})
        self.assertEqual(set(rep["provenance"]) >= {"generated_at", "fingerprint", "trials_used", "exclusions_applied",
                                                    "pricing", "pricing_sha", "parser_versions", "classifier_version",
                                                    "stats_version", "harbor_versions", "includes_active_run",
                                                    "direction"}, True)
        self.assertEqual((rep["provenance"]["direction"], rep["provenance"]["stats_version"]),
                         ("harness_over_baseline", 4))
        group = rep["groups"][0]
        for key in ("key", "label", "provider", "model", "family", "implementation", "config_hash", "harness_version",
                    "is_baseline", "configs", "config_diff", "harness_versions", "runs", "tasks", "trials",
                    "pass_rate", "setup_failure_rate", "per_trial", "reported_vs_computed_cost"):
            self.assertIn(key, group)
        self.assertEqual(group["pass_rate"]["method"], "wilson_tasks")
        c = rep["comparisons"][0]
        for key in ("key", "provider", "model", "group", "baseline_group", "headline", "pairs", "metrics", "pass",
                    "coverage", "caveats"):
            self.assertIn(key, c)
        self.assertNotIn("pairs_table", c)
        metric = c["metrics"]["cost"]
        for key in ("status", "n", "need", "geomean", "ratio_of_means", "per_task", "leave_one_out", "sensitivity",
                    "robust", "reference"):
            self.assertIn(key, metric)
        self.assertEqual(set(c["coverage"]) >= {"pairs", "need_for_estimate", "need_for_verdict",
                                                "halfwidth_factor_now", "target_factor", "sd_log_used", "sd_source",
                                                "est_pairs_needed", "est_more_pairs", "pass_tasks_needed", "next",
                                                "est_cost_usd", "per_trial"}, True)


class RealLedger(ReportCase):
    def test_synthetic_trials_through_a_real_ledger(self) -> None:
        data = self.tmp / "data"
        arms = [{"name": "claude-code", "agent": "claude-code", "provider": "anthropic", "model": "claude-opus-5-5"},
                {"name": "pi", "agent": "pi", "provider": "anthropic", "model": "claude-opus-5-5"}]
        lf.make_bench_run(data, "20261029-070000-sample", arms, profile="dev")
        for i, k in enumerate(KS):
            task = f"task-{i}"
            arm, base = scaled(k, i)
            for a, tokens in ((arms[0], base), (arms[1], arm)):
                lf.make_trial(data / "jobs" / "20261029-070000-sample" / a["name"] / f"{task}__{a['name'][:4]}{i}",
                              family=a["agent"], model_name="anthropic/claude-opus-5-5", tokens=tokens,
                              job_id=f"00000000-0000-4000-8000-00000000000{arms.index(a)}", task=task,
                              lock_digest="sha256:" + hashlib.sha256(task.encode()).hexdigest())
        ledger = Ledger(self.tmp / "ledger.sqlite", roots=[ScanRoot(data / "jobs", "dev", data / "runs")],
                        root=self.tmp)
        ledger.open()
        self.addCleanup(ledger.close)
        self.assertEqual(ledger.rescan()["inserted"], 20)
        rep = self.build(ledger, self.config())
        c = self.comparison(rep, "pi")
        self.assertAlmostEqual(c["metrics"]["tokens_total"]["geomean"]["estimate"],
                               math.exp(statistics.fmean(math.log(1 / k) for k in KS)), places=9)
        self.assertEqual(c["metrics"]["cost"]["status"], "ok")
        self.assertEqual(rep["totals"]["trials"], 20)
        self.assertEqual(rep["provenance"]["exclusions_applied"], 1)  # the seeded pilot-1 exclusion
        self.assertEqual(len(report.analysis_rows(ledger, self.config())), 20)
