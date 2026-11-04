"""Reusing measured trials: which tasks the ledger already covers per arm,
how that is written into the spec, and what checks, estimates, LAUNCH and
the assistant's drafts do with it.

Rows go through report.prepare_rows, so which trials are usable is the
report's own decision, and one test ingests a synthetic Harbor job through
the real ledger. Schemas come from tests/fixtures/schemas, the dataset
snapshot is a fixed stand-in, and the run manager is a fake: nothing starts
Harbor or bills anyone.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import itertools
import json
from typing import Any
from unittest import mock

from tests import ledger_fixtures as lf
from tests.support import BenchTestCase
from tests.test_server import ServerCase

from bench import checks, datasets, designs, estimates, report, reuse, runs, schemas
from bench import spec as specs
from bench.designs import DesignStore, DraftWriter
from bench.extract import ScanRoot
from bench.ledger import Ledger, LedgerUnavailable

DATASET = "terminal-bench@2.0"
DOWNLOADED_AT = "2026-10-29T05:29:58+00:00"
SNAPSHOT = f"local:{DATASET}@{DOWNLOADED_AT}"
HARBOR = "0.25.0"
GLM = ("openrouter", "z-ai/glm-5.3")
TASKS = ["task-a", "task-b", "task-c", "task-d"]
META = {t: {"difficulty": "medium", "category": "test", "agent_timeout_sec": 900.0, "memory_gb": 2.0, "cpus": 1,
            "expert_time_estimate_min": 30.0} for t in TASKS}
CONVERSATION = "c-20261029-101500-ab12"

PRICING_YAML = """\
anthropic:
  claude-opus-5-5: {input: 5.00, output: 25.00, cache_read: 0.50}
openrouter:
  glm-5.3: {input: 0.07, output: 1.40, cache_read: 0.069}
"""

CONFIG_YAML = """\
dataset: terminal-bench@2.0
providers:
  anthropic: {env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}
  openrouter: {env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}
modes:
  pilot: {n_tasks: 1}
  sample: {n_tasks: 4}
run: {budget_usd: 5, effort: medium, n_concurrent: 2, parallel_arms: 2, per_trial_cap_usd: 1, max_turns: 150}
baseline_arm: cc
designs: {max_budget_usd: 50}
arms:
  - {name: cc, agent: claude-code, provider: openrouter, model: z-ai/glm-5.3, options: {version: "2.1.312"}}
  - {name: pi, agent: pi, provider: openrouter, model: z-ai/glm-5.3, options: {version: "1.4.0"}}
  - {name: t2, agent: terminus-2, provider: openrouter, model: z-ai/glm-5.3}
  - {name: pi-latest, agent: pi, provider: openrouter, model: z-ai/glm-5.3, enabled: false}
"""

CC = {"name": "cc", "agent": "claude-code", "provider": "openrouter", "model": "z-ai/glm-5.3",
      "options": {"version": "2.1.312"}}
PI = {"name": "pi", "agent": "pi", "provider": "openrouter", "model": "z-ai/glm-5.3", "options": {"version": "1.4.0"}}
T2 = {"name": "t2", "agent": "terminus-2", "provider": "openrouter", "model": "z-ai/glm-5.3"}
PI_LATEST = {"name": "pi-latest", "agent": "pi", "provider": "openrouter", "model": "z-ai/glm-5.3"}
FAMILY = {"cc": "claude-code", "pi": "pi", "t2": "terminus-2", "pi-latest": "pi"}

_SEQ = itertools.count(1)


def digest_of(task: str, version: str = "v1") -> str:
    return "sha256:" + hashlib.sha256(f"{task}@{version}".encode()).hexdigest()


def trial(task: str, chash: str, *, family: str = "claude-code", outcome: str = "passed",
          tokens: tuple[int, int, int] | None = (100000, 60000, 5000), cost: float | None = 0.2,
          exception: str | None = None, run_key: str = "default/r1", digest: str | None = None,
          snapshot: str | None = SNAPSHOT, dataset: str = DATASET, harbor: str = HARBOR,
          where: tuple[str, str] = GLM) -> dict[str, Any]:
    """A ledger TrialRow for one trial, before report.prepare_rows."""
    n = next(_SEQ)
    n_input, n_cache, n_output = tokens if tokens is not None else (None, None, None)
    return {
        "trial_key": f"job-{n}/{task}__{n}", "run_key": run_key, "profile": run_key.split("/")[0],
        "run_id": run_key.split("/")[-1], "arm": family, "family": family, "implementation": "builtin",
        "config_hash": chash, "config_label": family, "provider": where[0], "model": where[1], "effort": "medium",
        "options": {}, "task": task, "task_digest": digest or digest_of(task), "dataset": dataset,
        "dataset_snapshot": snapshot, "harbor_version": harbor, "outcome": outcome, "outcome_reason": "test",
        "harbor_cause": None, "exception_type": exception, "agent_started": True,
        "phases": {"agent_exec_sec": 300.0}, "total_sec": 400.0, "n_input": n_input, "n_cache": n_cache,
        "n_output": n_output, "cost_usd": cost, "reported_cost_usd": None, "cost_at_ingest_usd": cost,
        "passed": outcome == "passed" if outcome in ("passed", "failed", "agent_limit", "agent_crash") else None,
        "exclusion": None, "unpriced_models": [], "primary_share": 1.0 if tokens else None,
        "started_at": f"2026-10-29T{n // 60 % 24:02d}:{n % 60:02d}:00Z",
    }


class ReuseCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cfg = self.load_test_config(CONFIG_YAML, pricing=PRICING_YAML, tasks=TASKS)
        self.snapshot: dict[str, Any] | None = {"dataset": DATASET, "tasks_dir": "terminal-bench",
                                                "downloaded_at": DOWNLOADED_AT, "harbor": HARBOR}
        for target, name, value in ((datasets, "task_meta", lambda dataset: META),
                                    (datasets, "info", lambda dataset: copy.deepcopy(self.snapshot)),
                                    (datasets, "path", lambda dataset: None),
                                    (checks, "docker_capacity", lambda: None),
                                    (reuse, "harbor_version", lambda: HARBOR)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def spec(self, arms: list[dict[str, Any]], *, tasks: list[str] = TASKS, attempts: int = 1,
             reuse_mode: str = "measured") -> dict[str, Any]:
        return designs.normalize(self.cfg, {"tasks": tasks, "attempts": attempts, "run": {"budget_usd": 5.0},
                                            "arms": arms, "reuse": reuse_mode})

    def cell(self, spec: dict[str, Any], name: str) -> str:
        """The config hash the arm's trials get (estimates.config_hash_for with the planned kwargs)."""
        found = {c.arm: c for c in reuse.coverage(self.cfg, spec, [])}
        self.assertIsNotNone(found[name].config_hash, found[name].reason)
        return str(found[name].config_hash)

    def rows(self, raw: list[dict[str, Any]], *, excluded: tuple[str, ...] = (),
             crashes: str | None = None) -> list[dict[str, Any]]:
        """raw TrialRows as report.analysis_rows returns them."""
        o = dataclasses.replace(report.parse_options(None, self.cfg), pricing="current")
        if crashes is not None:
            o = dataclasses.replace(o, crashes=crashes)
        excl = (frozenset(excluded), frozenset(), ())
        return [report._analysis_row(r) for r in report.prepare_rows(raw, o, excl, None)]

    def resolve(self, spec: dict[str, Any], raw: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        return reuse.resolve(self.cfg, spec, self.rows(raw, **kw))


# Resolution ---------------------------------------------------------------------------------------


class Resolution(ReuseCase):
    def test_off_by_default_leaves_specs_and_hashes_alone(self) -> None:
        plain = designs.normalize(self.cfg, {"tasks": TASKS, "run": {"budget_usd": 5.0}, "arms": [CC, PI]})
        off = self.spec([CC, PI], reuse_mode="off")
        self.assertEqual(list(plain), list(specs.SPEC_KEYS))
        self.assertEqual(specs.spec_hash(off), specs.spec_hash(plain))
        on = self.spec([CC, PI])
        self.assertEqual((on["reuse"], on["reused"]), ("measured", {}))
        self.assertEqual(specs.structure_problems(on), [])
        self.assertNotEqual(specs.spec_hash(on), specs.spec_hash(plain))
        self.assertEqual(reuse.resolve(self.cfg, plain, []), plain)

    def test_covered_tasks_leave_the_arm_and_are_listed_with_their_trials(self) -> None:
        spec = self.spec([CC, PI])
        cc = self.cell(spec, "cc")
        a, b = trial("task-a", cc, run_key="default/old"), trial("task-c", cc, run_key="default/other")
        resolved = self.resolve(spec, [a, b])
        self.assertEqual(resolved["arms"][0]["tasks"], ["task-b", "task-d"])
        self.assertIsNone(resolved["arms"][1]["tasks"])
        self.assertEqual(resolved["reused"], {"cc": [
            {"task": "task-a", "trials": [a["trial_key"]], "runs": ["default/old"]},
            {"task": "task-c", "trials": [b["trial_key"]], "runs": ["default/other"]}]})
        self.assertEqual(specs.structure_problems(resolved), [])
        run_spec = specs.spec_from_design(resolved, None)
        self.assertEqual([a.tasks for a in run_spec.arms], [("task-b", "task-d"), tuple(TASKS)])
        self.assertEqual(designs.trial_count(resolved), 6)

    def test_only_the_same_cell_counts(self) -> None:
        spec = self.spec([CC])
        cc = self.cell(spec, "cc")
        other_options = self.cell(self.spec([{**CC, "options": {"version": "2.1.312", "max_turns": 20}}]), "cc")
        self.assertNotEqual(other_options, cc)
        raw = [
            trial("task-a", other_options),                        # other options: another configuration
            trial("task-b", cc, family="pi"),                      # another family
            trial("task-c", cc, where=("anthropic", "claude-opus-5-5")),  # another model
            trial("task-d", cc, dataset="other-bench@1"),          # another dataset
        ]
        resolved = self.resolve(spec, raw)
        self.assertEqual(resolved["reused"], {})
        self.assertIsNone(resolved["arms"][0]["tasks"])

    def test_run_settings_are_part_of_the_cell(self) -> None:
        base = self.spec([PI])
        capped = designs.normalize(self.cfg, {"tasks": TASKS, "run": {"budget_usd": 5.0, "max_turns": 40},
                                              "arms": [PI], "reuse": "measured"})
        slower = designs.normalize(self.cfg, {"tasks": TASKS, "run": {"budget_usd": 5.0, "timeout_multiplier": 2.0},
                                              "arms": [PI], "reuse": "measured"})
        hashes = {self.cell(s, "pi") for s in (base, capped, slower)}
        self.assertEqual(len(hashes), 3)
        resolved = self.resolve(capped, [trial("task-a", self.cell(base, "pi"), family="pi")])
        self.assertEqual(resolved["reused"], {})

    def test_task_content_must_be_the_snapshot_version(self) -> None:
        spec = self.spec([CC])
        cc = self.cell(spec, "cc")
        current = trial("task-a", "f" * 16, family="pi")  # any trial from the snapshot shows its digests
        stale = trial("task-a", cc, digest=digest_of("task-a", "v0"), snapshot="git:69671fbaac6d")
        same_content = trial("task-b", cc, snapshot="git:69671fbaac6d")  # older snapshot, identical digest
        never_seen = trial("task-c", cc, snapshot="git:69671fbaac6d")
        raw = [current, stale, same_content, trial("task-b", "f" * 16, family="pi"), never_seen]
        resolved = self.resolve(spec, raw)
        self.assertEqual(specs.reused_tasks(resolved, "cc"), ["task-b"])

        by_name = dataclasses.replace(report.parse_options(None, self.cfg), match_on="name")
        with mock.patch.object(estimates, "pair_options", lambda cfg: by_name):
            self.assertEqual(specs.reused_tasks(self.resolve(spec, raw), "cc"), ["task-a", "task-b", "task-c"])
        self.snapshot = None
        found = reuse.coverage(self.cfg, spec, self.rows(raw))[0]
        self.assertEqual(found.covered, {})
        self.assertIn("no local snapshot", found.reason)

    def test_same_run_only_reuses_nothing(self) -> None:
        spec = self.spec([CC])
        rows = self.rows([trial("task-a", self.cell(spec, "cc"))])
        same_run = dataclasses.replace(report.parse_options(None, self.cfg), same_run_only=True)
        with mock.patch.object(estimates, "pair_options", lambda cfg: same_run):
            found = reuse.coverage(self.cfg, spec, rows)[0]
        self.assertEqual(found.covered, {})
        self.assertIn("same_run_only", found.reason)

    def test_usable_means_counted_by_the_report_for_pass_rate_and_tokens(self) -> None:
        spec = self.spec([CC], tasks=["task-a", "task-b", "task-c", "task-d"])
        cc = self.cell(spec, "cc")
        raw = [
            trial("task-a", cc, outcome="setup_failure", exception="X", tokens=None, cost=None),
            trial("task-a", cc, outcome="verifier_failure", exception="VerifierTimeoutError"),
            trial("task-b", cc, outcome="failed", tokens=(0, 0, 0), cost=None),  # ran, recorded no usage
            trial("task-c", cc, outcome="agent_crash", exception="NonZeroAgentExitCodeError"),
            excluded := trial("task-d", cc),
        ]
        self.assertEqual(specs.reused_tasks(self.resolve(spec, raw), "cc"), ["task-c", "task-d"])
        self.assertEqual(specs.reused_tasks(self.resolve(spec, raw, excluded=(excluded["trial_key"],)), "cc"),
                         ["task-c"])
        self.assertEqual(specs.reused_tasks(self.resolve(spec, raw, crashes="exclude"), "cc"), ["task-d"])

    def test_hand_exclusions_and_model_fallbacks_never_count(self) -> None:
        # Stricter than the report, which counts both here: a reused task is never run again for the cell.
        spec = self.spec([CC], tasks=["task-a", "task-b"])
        cc = self.cell(spec, "cc")
        fallback = trial("task-a", cc)
        fallback["primary_share"] = 0.0  # Claude Code answered a refusal with another model
        excluded = trial("task-b", cc)
        excluded["exclusion"] = {"target_type": "trial", "action": "exclude", "reason": "Not an Opus 5.5 trial"}
        shown = dataclasses.replace(report.parse_options(None, self.cfg), pricing="current", apply_exclusions=False)

        def rows() -> list[dict[str, Any]]:  # as analysis_rows gives them under report.exclusions false
            return [report._analysis_row(r) for r in report.prepare_rows([fallback, excluded], shown, None, None)]

        self.assertTrue(all(r["valid_reward"] and r["valid_tokens"] for r in rows()))
        resolved = reuse.resolve(self.cfg, spec, rows())
        self.assertEqual(resolved["reused"], {})
        notes = reuse.summary(self.cfg, resolved, rows())["notes"]
        self.assertTrue(any("1 trial of these configurations" in n and "another model" in n for n in notes), notes)
        excluded["exclusion"] = {**excluded["exclusion"], "action": "include"}  # put back: the latest action wins
        self.assertEqual(specs.reused_tasks(reuse.resolve(self.cfg, spec, rows()), "cc"), ["task-b"])

    def test_a_trial_cut_short_counts_but_full_usage_is_preferred(self) -> None:
        spec = self.spec([CC], tasks=["task-a"])
        cc = self.cell(spec, "cc")
        cut = trial("task-a", cc, outcome="agent_limit", exception="AgentTimeoutError")
        rows = self.rows([cut])
        self.assertEqual(rows[0]["usage_status"], "lower_bound")
        self.assertEqual(reuse.resolve(self.cfg, spec, rows)["reused"]["cc"][0]["trials"], [cut["trial_key"]])
        full = trial("task-a", cc)
        resolved = self.resolve(spec, [cut, full])
        self.assertEqual(resolved["reused"]["cc"][0]["trials"], [full["trial_key"]])

    def test_attempts_decide_coverage_and_the_oldest_trials_are_kept(self) -> None:
        spec = self.spec([CC], tasks=["task-a", "task-b"], attempts=2)
        cc = self.cell(spec, "cc")
        first, second, third = (trial("task-a", cc) for _ in range(3))
        raw = [third, first, second, trial("task-b", cc)]
        resolved = self.resolve(spec, raw)
        self.assertEqual(resolved["reused"]["cc"], [{"task": "task-a", "runs": ["default/r1"],
                                                     "trials": [first["trial_key"], second["trial_key"]]}])
        self.assertEqual(resolved["arms"][0]["tasks"], ["task-b"])
        found = reuse.coverage(self.cfg, spec, self.rows(raw))[0]
        self.assertEqual(found.short, {"task-b": 1})

    def test_a_task_timed_only_on_a_busy_host_runs_again(self) -> None:
        spec = self.spec([CC], tasks=["task-a", "task-b"], attempts=2)
        cc = self.cell(spec, "cc")
        busy = {"cpu_contended": True, "host_cpu_samples": 10, "host_cpu_hot_share": 0.6}
        a1, a2 = (dict(trial("task-a", cc), flags=busy) for _ in range(2))
        b_busy, b_quiet, b_late = dict(trial("task-b", cc), flags=busy), trial("task-b", cc), trial("task-b", cc)
        raw = [a1, a2, b_busy, b_quiet, b_late]
        rows = self.rows(raw)
        self.assertTrue(all(reuse.usable(r) for r in rows))  # still counted for pass rate and tokens
        found = reuse.coverage(self.cfg, spec, rows)[0]
        self.assertEqual(found.host_busy, {"task-a": 2})
        # task-b has a usable time, and the quiet trials are reused before the older busy one
        self.assertEqual([r["trial_key"] for r in found.covered["task-b"]], [b_quiet["trial_key"], b_late["trial_key"]])
        resolved = reuse.resolve(self.cfg, spec, rows)
        self.assertEqual(resolved["arms"][0]["tasks"], ["task-a"])
        summary = reuse.summary(self.cfg, resolved, rows)
        self.assertEqual(summary["arms"][0]["host_busy"], [{"task": "task-a", "usable": 2}])
        self.assertTrue(any("host CPU was busy" in n and "runs again" in n for n in summary["notes"]), summary["notes"])
        quiet = self.rows([dict(r, flags=None) for r in raw])
        self.assertNotEqual(reuse.fingerprint(quiet), reuse.fingerprint(rows))

    def test_any_profile_counts(self) -> None:
        spec = self.spec([CC], tasks=["task-a"])
        resolved = self.resolve(spec, [trial("task-a", self.cell(spec, "cc"), run_key="dev/20261029-055852-pilot")])
        self.assertEqual(resolved["reused"]["cc"][0]["runs"], ["dev/20261029-055852-pilot"])

    def test_harness_version_must_be_known_to_match(self) -> None:
        spec = self.spec([PI_LATEST, T2], tasks=["task-a"])
        snapshot_seen = self.rows([trial("task-a", "f" * 16, family="pi")])  # the snapshot's digests are known
        latest, bundled = reuse.coverage(self.cfg, spec, snapshot_seen)
        self.assertEqual((latest.version, bundled.version), ("unpinned", "harbor"))
        self.assertIn("not pinned", latest.reason)
        self.assertIsNone(bundled.reason)
        raw = [trial("task-a", str(latest.config_hash), family="pi"),
               old := trial("task-a", str(bundled.config_hash), family="terminus-2", harbor="0.24.1")]
        self.assertEqual(self.resolve(spec, raw)["reused"], {})
        raw.append(now := trial("task-a", str(bundled.config_hash), family="terminus-2"))
        self.assertEqual(self.resolve(spec, raw)["reused"],
                         {"t2": [{"task": "task-a", "trials": [now["trial_key"]], "runs": ["default/r1"]}]})
        self.assertNotIn(old["trial_key"], str(self.resolve(spec, raw)))
        pinned = self.spec([PI], tasks=["task-a"])
        self.assertEqual(reuse.coverage(self.cfg, pinned, [])[0].version, "pinned")

    def test_a_pinned_harness_must_have_run_on_the_installed_harbor(self) -> None:
        # Harbor's adapter for an installed harness turns its trajectory into the usage the ledger records.
        spec = self.spec([CC], tasks=["task-a"])
        cc = self.cell(spec, "cc")
        old = trial("task-a", cc, harbor="0.24.1")
        self.assertEqual(self.resolve(spec, [old])["reused"], {})
        out = reuse.summary(self.cfg, self.resolve(spec, [old]), self.rows([old]))
        self.assertEqual(out["arms"][0]["harness_version"], "pinned")
        self.assertTrue(any(n.startswith("1 trial of these configurations ran on another Harbor than the 0.25.0")
                            for n in out["notes"]), out["notes"])
        now = trial("task-a", cc)
        self.assertEqual(self.resolve(spec, [old, now])["reused"],
                         {"cc": [{"task": "task-a", "trials": [now["trial_key"]], "runs": ["default/r1"]}]})
        with mock.patch.object(reuse, "harbor_version", lambda: ""):
            found = reuse.coverage(self.cfg, spec, self.rows([now]))[0]
        self.assertEqual((found.version, found.covered), ("pinned", {}))
        self.assertIn("installed Harbor version could not be read", found.reason)

    def test_pre_baked_and_stock_trials_cover_one_cell(self) -> None:
        # How a trial got its harness is provenance: either kind counts for the arm's cell.
        spec = self.spec([CC], tasks=["task-a"], attempts=2)
        cc = self.cell(spec, "cc")
        stock, baked = (dict(trial("task-a", cc), setup_mode=mode) for mode in ("stock", "prebaked"))
        reused = self.resolve(spec, [stock, baked])["reused"]["cc"]
        self.assertEqual(sorted(reused[0]["trials"]), sorted([stock["trial_key"], baked["trial_key"]]))

    def test_a_tag_or_range_is_not_a_pin_and_bench_code_is_not_bundled(self) -> None:
        # Harbor records options.version as given, so 'latest' trials all share one cell whatever they installed.
        for version in ("latest", "^2.1", "2.x", "2", "stable"):
            with self.subTest(version=version):
                spec = self.spec([{**CC, "options": {"version": version}}], tasks=["task-a"])
                found = reuse.coverage(self.cfg, spec, [])[0]
                self.assertEqual(found.version, "unpinned")
                self.assertIn("tag or range", found.reason)
                self.assertEqual(self.resolve(spec, [trial("task-a", str(found.config_hash))])["reused"], {})
        for version in ("2.1.312", "1.0.0rc1", "1.2.3-beta.1"):
            with self.subTest(version=version):
                spec = self.spec([{**CC, "options": {"version": version}}], tasks=["task-a"])
                self.assertEqual(reuse.coverage(self.cfg, spec, [])[0].version, "pinned")
        # An import path with no version option is code of this bench: Harbor's version does not fix it.
        arm = dataclasses.replace(specs.spec_from_design(self.spec([T2]), None).arms[0], kind="import_path",
                                  agent="bench.agents.example:Agent")
        with mock.patch.object(schemas, "agent_schema", lambda agent: {"properties": {}}):
            version, why = reuse._version(arm, {}, HARBOR)
        self.assertEqual(version, "unknown")
        self.assertIn("code of this bench", why)

    def test_the_latest_digest_counts_and_a_new_snapshot_needs_a_trial_first(self) -> None:
        spec = self.spec([CC], tasks=["task-a"])
        cc = self.cell(spec, "cc")
        before = trial("task-a", cc, digest=digest_of("task-a", "crlf"))  # the files were rewritten in place since
        after = trial("task-a", "f" * 16, family="pi")
        self.assertEqual(self.resolve(spec, [before, after])["reused"], {})
        again = trial("task-a", cc)
        self.assertEqual(specs.reused_tasks(self.resolve(spec, [before, after, again]), "cc"), ["task-a"])
        self.snapshot = {**self.snapshot, "downloaded_at": "2026-10-30T09:00:00+00:00"}
        found = reuse.coverage(self.cfg, spec, self.rows([before, after, again]))[0]
        self.assertEqual(found.covered, {})
        self.assertIn("no trial has run from the current snapshot", found.reason)

    def test_two_arms_of_one_cell_never_share_a_trial(self) -> None:
        spec = self.spec([CC, {**CC, "name": "cc2"}], tasks=["task-a", "task-b"])
        cc = self.cell(spec, "cc")
        first, second, b = trial("task-a", cc), trial("task-a", cc), trial("task-b", cc)
        resolved = self.resolve(spec, [first, second, b])
        self.assertEqual(resolved["reused"], {
            "cc": [{"task": "task-a", "trials": [first["trial_key"]], "runs": ["default/r1"]},
                   {"task": "task-b", "trials": [b["trial_key"]], "runs": ["default/r1"]}],
            "cc2": [{"task": "task-a", "trials": [second["trial_key"]], "runs": ["default/r1"]}]})
        self.assertEqual([a["tasks"] for a in resolved["arms"]], [[], ["task-b"]])
        self.assertEqual(estimates.reused_figures(resolved, self.rows([first, second, b]))["trials"], 3)

    def test_resolving_again_is_stable_and_round_trips_through_editor_input(self) -> None:
        spec = self.spec([CC, {**PI, "tasks": ["task-b", "task-c"]}])
        raw = [trial("task-a", self.cell(spec, "cc")), trial("task-b", self.cell(spec, "pi"), family="pi")]
        resolved = self.resolve(spec, raw)
        self.assertEqual(resolved["arms"][1]["tasks"], ["task-c"])
        self.assertEqual(self.resolve(resolved, raw), resolved)
        edited = designs.to_input(resolved)
        self.assertEqual(edited["reuse"], "measured")
        self.assertEqual([a["tasks"] for a in edited["arms"]], [None, ["task-b", "task-c"]])
        self.assertEqual(self.resolve(designs.normalize(self.cfg, edited), raw), resolved)
        self.assertEqual(reuse.resolve(self.cfg, resolved, [], "off"),
                         designs.normalize(self.cfg, {**edited, "reuse": "off"}))
        self.assertNotIn("reuse", designs.to_input(designs.normalize(self.cfg, {**edited, "reuse": "off"})))


# Validation, plan and estimate --------------------------------------------------------------------------


class Checks(ReuseCase):
    def test_an_arm_with_nothing_left_is_skipped(self) -> None:
        spec = self.spec([CC, PI], tasks=["task-a", "task-b"])
        cc = self.cell(spec, "cc")
        resolved = self.resolve(spec, [trial("task-a", cc), trial("task-b", cc)])
        self.assertEqual(resolved["arms"][0]["tasks"], [])
        result = designs.validate(self.cfg, resolved)
        self.assertEqual(result["problems"], [])
        self.assertTrue(any("'cc' is skipped" in w for w in result["warnings"]), result["warnings"])
        run_spec = specs.spec_from_design(resolved, None)
        self.assertEqual([a.name for a in run_spec.arms], ["pi"])
        self.assertEqual([a.name for a in run_spec.skipped_arms], ["cc"])
        plan = result["plan"]
        self.assertEqual((plan["trials"], plan["skipped_arms"], [a["name"] for a in plan["arms"]]), (2, ["cc"], ["pi"]))

    def test_nothing_left_to_run_is_a_problem(self) -> None:
        spec = self.spec([CC], tasks=["task-a"])
        resolved = self.resolve(spec, [trial("task-a", self.cell(spec, "cc"))])
        result = designs.validate(self.cfg, resolved)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any(p.startswith("Nothing to run") for p in result["problems"]), result["problems"])
        self.assertIsNone(result["plan"])
        plan = runs.plan_spec(self.cfg, specs.spec_from_launch(resolved, "pilot"))
        self.assertTrue(any(p.startswith("Nothing to run") for p in plan["problems"]), plan["problems"])
        self.assertEqual(plan["trials"], 0)

    def test_inconsistent_reuse_records_are_refused(self) -> None:
        spec = self.spec([CC], tasks=["task-a", "task-b"])
        entry = {"task": "task-a", "trials": ["job/t1"], "runs": ["default/r1"]}
        both = {**spec, "reused": {"cc": [entry]}}
        self.assertTrue(any("both reused and run" in p for p in designs.validate(self.cfg, both)["problems"]))
        stranger = {**spec, "reused": {"nope": [entry]}}
        self.assertTrue(any("does not have: nope" in p for p in designs.validate(self.cfg, stranger)["problems"]))
        outside = {**spec, "arms": [{**spec["arms"][0], "tasks": ["task-a", "task-b"]}],
                   "reused": {"cc": [{**entry, "task": "task-x"}]}}
        problems = designs.validate(self.cfg, outside)["problems"]
        self.assertTrue(any("not in the design's task list" in p for p in problems), problems)
        for bad in ({k: v for k, v in spec.items() if k != "reused"}, {**spec, "reuse": "off"},
                    {**spec, "reused": {"cc": [{**entry, "trials": []}]}}, {**spec, "reused": {"cc": "x"}},
                    {**spec, "reused": {"cc": [{"task": "task-a"}]}}):
            with self.subTest(bad=bad.get("reused"), reuse=bad.get("reuse")):
                self.assertNotEqual(specs.structure_problems(bad), [])

    def test_estimate_counts_only_trials_that_run_and_reused_pairs_as_in_hand(self) -> None:
        spec = self.spec([CC, PI])
        cc, pi = self.cell(spec, "cc"), self.cell(spec, "pi")
        raw = [trial("task-a", cc, cost=0.4), trial("task-b", cc, cost=0.2), trial("task-a", pi, family="pi", cost=0.1)]
        rows = self.rows(raw)
        resolved = reuse.resolve(self.cfg, spec, rows)
        self.assertEqual(specs.reused_tasks(resolved, "cc"), ["task-a", "task-b"])
        self.assertEqual(specs.reused_tasks(resolved, "pi"), ["task-a"])
        check = designs.validate(self.cfg, resolved)
        est = estimates.estimate(self.cfg, resolved, rows, check["plan"], replicates=200)
        self.assertEqual({a["arm"]: a["trials"] for a in est["arms"]}, {"cc": 2, "pi": 3})
        self.assertEqual(est["reuse"]["trials"], 3)
        self.assertAlmostEqual(est["reuse"]["cost_avoided_usd"], 0.7)
        self.assertIn(estimates.REUSE_ASSUMPTION, est["assumptions"])
        (pairing,) = est["pairing"]
        # task-a is paired already; the run adds task-b (pi runs it, cc's is reused) and task-c, task-d.
        self.assertEqual((pairing["pairs_now"], pairing["pairs_after"]), (1, 4))
        plain = estimates.estimate(self.cfg, self.spec([CC, PI], reuse_mode="off"), rows,
                                   designs.validate(self.cfg, self.spec([CC, PI], reuse_mode="off"))["plan"],
                                   replicates=200)
        self.assertIsNone(plain["reuse"])
        self.assertEqual(plain["pairing"][0]["pairs_after"], 4)
        self.assertEqual({a["arm"]: a["trials"] for a in plain["arms"]}, {"cc": 4, "pi": 4})

    def test_a_skipped_baseline_still_pairs(self) -> None:
        spec = self.spec([CC, PI], tasks=["task-a", "task-b"])
        cc = self.cell(spec, "cc")
        rows = self.rows([trial("task-a", cc), trial("task-b", cc)])
        resolved = reuse.resolve(self.cfg, spec, rows)
        est = estimates.estimate(self.cfg, resolved, rows, designs.validate(self.cfg, resolved)["plan"], replicates=200)
        self.assertEqual([a["arm"] for a in est["arms"]], ["pi"])
        (pairing,) = est["pairing"]
        self.assertEqual((pairing["arm"], pairing["baseline_cell"]["arm"]), ("pi", "cc"))
        self.assertEqual((pairing["pairs_now"], pairing["pairs_after"]), (0, 2))
        self.assertEqual(est["run"]["inflight_allowance_usd"],
                         estimates.inflight_allowance(resolved, None, {"arms": est["arms"]}))

    def test_summary_reports_reuse_and_what_changed_since(self) -> None:
        spec = self.spec([CC, PI_LATEST], tasks=["task-a", "task-b", "task-c"])
        cc = self.cell(spec, "cc")
        a = trial("task-a", cc, cost=0.3)
        b = trial("task-b", cc, outcome="agent_limit", exception="AgentTimeoutError", cost=0.5)
        resolved = self.resolve(spec, [a, b])
        out = reuse.summary(self.cfg, resolved, self.rows([a, b]))
        self.assertEqual((out["trials_asked"], out["trials_reused"], out["trials_to_run"]), (6, 2, 4))
        self.assertAlmostEqual(out["cost_avoided_usd"], 0.8)
        self.assertTrue(out["current"])
        arms = {x["arm"]: x for x in out["arms"]}
        self.assertEqual([r["task"] for r in arms["cc"]["reused"]], ["task-a", "task-b"])
        self.assertEqual(arms["cc"]["reused"][1]["lower_bound"], 1)
        self.assertTrue(any("lower bound" in n for n in out["notes"]))
        self.assertIn("not pinned", arms["pi-latest"]["reason"])
        self.assertEqual(arms["pi-latest"]["cost_avoided_usd"], 0.0)
        c = trial("task-c", cc)
        later = reuse.summary(self.cfg, resolved, self.rows([a, b, c], excluded=(a["trial_key"],)))
        self.assertFalse(later["current"])
        self.assertTrue(any("task-c now has enough" in x for x in later["changes"]), later["changes"])
        self.assertTrue(any("task-a no longer has" in x for x in later["changes"]), later["changes"])
        self.assertIsNone(reuse.summary(self.cfg, self.spec([CC], reuse_mode="off"), []))

    def test_drift_is_what_would_run_differently(self) -> None:
        spec = self.spec([CC], tasks=["task-a", "task-b"])
        cc = self.cell(spec, "cc")
        a = trial("task-a", cc)
        resolved = self.resolve(spec, [a])
        self.assertEqual(reuse.drift(self.cfg, resolved, self.rows([a])), [])
        # Another trial now covers task-a: the check records it, but nothing that runs changes.
        swapped = self.rows([a, trial("task-a", cc)], excluded=(a["trial_key"],))
        self.assertEqual(reuse.drift(self.cfg, resolved, swapped), [])
        self.assertTrue(any("other trials now cover task-a" in c
                            for c in reuse.summary(self.cfg, resolved, swapped)["changes"]))
        (gained,) = reuse.drift(self.cfg, resolved, self.rows([a, trial("task-b", cc)]))
        self.assertIn("task-b now has enough usable trials", gained)
        (lost,) = reuse.drift(self.cfg, resolved, self.rows([a], excluded=(a["trial_key"],)))
        self.assertIn("task-a no longer has", lost)
        self.assertEqual(reuse.drift(self.cfg, self.spec([CC], reuse_mode="off"), self.rows([a])), [])


# Through the real ledger and the assistant's draft writer --------------------------------------------------


class ThroughTheLedger(ReuseCase):
    def test_an_ingested_trial_of_the_same_configuration_is_reused(self) -> None:
        spec = self.spec([PI], tasks=["task-a", "task-b"])
        arm = specs.spec_from_design(spec, None).arms[0]
        kwargs = estimates.planned_kwargs(self.cfg, arm, specs.spec_from_design(spec, None).run)
        tmp = self.tmpdir()
        data = tmp / "data"
        lf.make_bench_run(data, "20261029-120000-sample", [{k: PI[k] for k in ("name", "agent", "provider", "model")}],
                          downloaded_at=DOWNLOADED_AT, profile="default")
        job = data / "jobs" / "20261029-120000-sample" / "pi"
        lf.make_trial(job / "t1", family="pi", model_name="openrouter/z-ai/glm-5.3", kwargs=kwargs, task="task-a",
                      lock_digest=digest_of("task-a"), version="1.4.0")
        lf.make_trial(job / "t2", family="pi", model_name="openrouter/z-ai/glm-5.3", kwargs={**kwargs, "max_turns": 9},
                      task="task-b", lock_digest=digest_of("task-b"), version="1.4.0")
        led = Ledger(tmp / "ledger" / "bench.sqlite", roots=[ScanRoot(data / "jobs", "default", data / "runs")],
                     root=tmp)
        led.open()
        self.addCleanup(led.close)
        self.assertTrue(led.available, led.error)
        led.rescan()
        rows = report.analysis_rows(led, self.cfg)
        self.assertEqual({(r["task"], r["dataset_snapshot"], r["harbor_version"]) for r in rows},
                         {("task-a", SNAPSHOT, HARBOR), ("task-b", SNAPSHOT, HARBOR)})
        resolved = reuse.resolve(self.cfg, spec, rows)
        self.assertEqual(resolved["arms"][0]["tasks"], ["task-b"])  # its max_turns made task-b another cell
        self.assertEqual(resolved["reused"]["pi"][0]["runs"], ["default/20261029-120000-sample"])

    def test_the_assistant_draft_writer_resolves_reuse(self) -> None:
        spec_in = {"tasks": ["task-a", "task-b"], "run": {"budget_usd": 5.0}, "arms": [CC], "reuse": "measured"}
        cc = self.cell(designs.normalize(self.cfg, spec_in), "cc")
        rows = self.rows([trial("task-a", cc)])
        store = DesignStore(self.tmpdir() / "designs")
        writer = DraftWriter(store, lambda: self.cfg, CONVERSATION, rows=lambda: rows)
        doc = writer.create_draft({"title": "Fill the gaps"}, spec_in)
        self.assertEqual(doc["spec"]["arms"][0]["tasks"], ["task-b"])
        self.assertEqual(specs.reused_tasks(doc["spec"], "cc"), ["task-a"])
        blind = DraftWriter(store, lambda: self.cfg, CONVERSATION)
        with self.assertRaisesRegex(ValueError, "needs the ledger"):
            blind.create_draft({"title": "No ledger"}, spec_in)
        self.assertIn("reuse", designs.DESIGN_SPEC_SCHEMA["properties"])


# Endpoints -------------------------------------------------------------------------------------------------

SERVER_YAML = CONFIG_YAML.replace("designs: {max_budget_usd: 50}", "designs: {max_budget_usd: 20}")


class FakeReader:
    """The server's ReadOnlyLedger, over fixed analysis rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.data = rows

    def rows(self) -> list[dict[str, Any]]:
        return list(self.data)

    def fingerprint(self) -> str:
        return report.ledger_fingerprint(self.data)


class Endpoints(ServerCase):
    YAML = SERVER_YAML
    TASKS = TASKS

    def setUp(self) -> None:
        super().setUp()
        for target, name, value in ((datasets, "task_meta", lambda dataset: META),
                                    (datasets, "info", lambda dataset: {"dataset": DATASET, "tasks_dir": "x",
                                                                        "downloaded_at": DOWNLOADED_AT}),
                                    (datasets, "path", lambda dataset: None),
                                    (reuse, "harbor_version", lambda: HARBOR)):
            self.patch(target, name, value)
        cfg = self.state.cfg
        spec = designs.normalize(cfg, {"tasks": TASKS, "run": {"budget_usd": 5.0}, "arms": [CC, PI],
                                       "reuse": "measured"})
        self.hashes = {c.arm: c.config_hash for c in reuse.coverage(cfg, spec, [])}
        raw = [trial("task-a", str(self.hashes["cc"]), cost=0.25)]
        self.state.reader = FakeReader(self.rows(raw))
        self.reused_key = raw[0]["trial_key"]

    def rows(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        o = dataclasses.replace(report.parse_options(None, self.state.cfg), pricing="current")
        return [report._analysis_row(r) for r in report.prepare_rows(raw, o, None, None)]

    def test_check_resolves_reuse_and_reports_what_it_saves(self) -> None:
        body = {"tasks": TASKS, "run": {"budget_usd": 5.0}, "arms": [CC, PI], "reuse": "measured"}
        checked = self.ok(self.client.post("/api/designs/check", json={"spec": body}))
        self.assertEqual(checked["validation"]["problems"], [])
        self.assertEqual(checked["plan"]["trials"], 7)
        self.assertEqual((checked["reuse"]["trials_reused"], checked["reuse"]["cost_avoided_usd"]), (1, 0.25))
        self.assertEqual(checked["estimate"]["reuse"]["trials"], 1)
        self.assertEqual(sum(a["trials"] for a in checked["estimate"]["arms"]), 7)
        off = self.ok(self.client.post("/api/designs/check", json={"spec": {**body, "reuse": "off"}}))
        self.assertEqual((off["plan"]["trials"], off["reuse"]), (8, None))

    def test_without_the_ledger_reuse_cannot_be_resolved_but_off_still_works(self) -> None:
        class Unavailable(FakeReader):
            def rows(self) -> list[dict[str, Any]]:
                raise LedgerUnavailable("The ledger could not be opened")

        self.state.reader = Unavailable([])
        body = {"tasks": TASKS, "run": {"budget_usd": 5.0}, "arms": [CC, PI], "reuse": "measured"}
        checked = self.ok(self.client.post("/api/designs/check", json={"spec": body}))
        self.assertEqual(checked["validation"]["status"], "invalid")
        self.assertIn("could not be looked up", checked["validation"]["problems"][0])
        self.assertEqual(self.client.post("/api/designs", json={"title": "Gaps", "spec": body}).status_code, 503)
        off = self.ok(self.client.post("/api/designs/check", json={"spec": {**body, "reuse": "off"}}))
        self.assertEqual((off["validation"]["status"], off["plan"]["trials"]), ("valid", 8))

    def test_saved_designs_hold_the_resolved_spec(self) -> None:
        body = {"tasks": TASKS, "run": {"budget_usd": 5.0}, "arms": [CC, PI], "reuse": "measured"}
        detail = self.ok(self.client.post("/api/designs", json={"title": "Fill gaps", "spec": body}), 201)
        stored = detail["spec"]
        self.assertEqual(stored["reused"]["cc"][0]["trials"], [self.reused_key])
        self.assertEqual(stored["arms"][0]["tasks"], ["task-b", "task-c", "task-d"])
        self.assertEqual(detail["check"]["reuse"]["trials_reused"], 1)
        from_config = self.ok(self.client.post("/api/designs/from-config",
                                               json={"mode": "sample", "arms": ["cc", "pi"], "reuse": "measured"}),
                              201)
        self.assertEqual(specs.reused_tasks(from_config["spec"], "cc"), ["task-a"])
        self.assertEqual(from_config["spec"]["arms"][0]["tasks"], ["task-b", "task-c", "task-d"])
        self.assertEqual(self.client.post("/api/designs/from-config",
                                          json={"mode": "sample", "reuse": "yes"}).status_code, 400)
        # A stored spec sent back from the editor is resolved again, to the same spec.
        again = self.ok(self.client.post(f"/api/designs/{detail['design_id']}/versions",
                                         json={"title": "Fill gaps again", "base_version": 1, "spec": stored}), 201)
        self.assertEqual(again["spec_hash"], detail["spec_hash"])

    def test_an_exclusion_makes_a_saved_reuse_check_stale(self) -> None:
        body = {"tasks": TASKS, "run": {"budget_usd": 5.0}, "arms": [CC, PI], "reuse": "measured"}
        detail = self.ok(self.client.post("/api/designs", json={"title": "Fill gaps", "spec": body}), 201)
        self.assertTrue(detail["check"]["reuse"]["current"])
        # The same trial keys, so report.ledger_fingerprint stays put; the reused trial is excluded since.
        self.state.reader = FakeReader([dict(r, excluded_by_hand=True, valid_reward=False, valid_tokens=False,
                                             excluded_reason="excluded by hand: Docker died")
                                        for r in self.state.reader.data])
        (listed,) = self.ok(self.client.get("/api/designs"))["designs"]
        self.assertTrue(listed["check_stale"])
        again = self.ok(self.client.get(f"/api/designs/{detail['design_id']}/versions/1"))["check"]["reuse"]
        self.assertFalse(again["current"])
        self.assertIn("task-a no longer has", again["changes"][0])
        (listed,) = self.ok(self.client.get("/api/designs"))["designs"]
        self.assertFalse(listed["check_stale"])  # checked again and saved
        half = self.ok(self.client.post("/api/designs/check", json={"spec": {"tasks": TASKS, "reuse": "measured"}}))
        self.assertEqual((half["validation"]["status"], half["reuse"]), ("invalid", None))

    def test_a_saved_estimate_from_before_task_races_is_stale(self) -> None:
        body = {"tasks": TASKS, "run": {"budget_usd": 5.0}, "arms": [CC, PI]}
        detail = self.ok(self.client.post("/api/designs", json={"title": "Old estimate", "spec": body}), 201)
        path = self.state.designs.root / detail["design_id"] / "v0001.check.json"
        check = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(check["estimate"]["run"]["schedule"], "tasks")
        (listed,) = self.ok(self.client.get("/api/designs"))["designs"]
        self.assertFalse(listed["check_stale"])
        # A check saved before task races: its version's spec hash is the same, but its wall time was the arms one.
        del check["estimate"]["run"]["schedule"]
        path.write_text(json.dumps(check), encoding="utf-8")
        (listed,) = self.ok(self.client.get("/api/designs"))["designs"]
        self.assertTrue(listed["check_stale"])
        again = self.ok(self.client.get(f"/api/designs/{detail['design_id']}/versions/1"))["check"]
        self.assertEqual(again["estimate"]["run"]["schedule"], "tasks")

    def test_the_queue_holds_a_version_the_ledger_has_moved_past(self) -> None:
        body = {"tasks": TASKS, "run": {"budget_usd": 5.0}, "arms": [CC, PI], "reuse": "measured"}
        detail = self.ok(self.client.post("/api/designs", json={"title": "Fill gaps", "spec": body}), 201)
        queue = self.state.queue
        item = queue.add(detail["design_id"], 1, detail["spec_hash"])
        # Saved while another run was still measuring: task-b's trial landed after the save.
        self.state.reader = FakeReader([*self.state.reader.data, *self.rows([trial("task-b", str(self.hashes["cc"]))])])
        queue.start()
        queue._step()
        self.assertEqual(self.manager.started_specs, [])
        (held,) = queue.view()["held"]
        self.assertEqual(held["id"], item["id"])
        self.assertIn("Arm 'cc': task-b now has enough usable trials", held["message"])
        queue.retry(item["id"])
        queue.start()
        queue._step()
        (started,) = self.manager.started_specs
        self.assertEqual(started.spec_hash, detail["spec_hash"])  # run as approved

    def test_launch_previews_and_starts_the_reduced_run(self) -> None:
        preview = self.ok(self.client.get("/api/plan", params={"mode": "sample", "arms": "cc,pi",
                                                               "reuse": "measured"}))
        self.assertEqual((preview["trials"], preview["reuse"]["trials_reused"]), (7, 1))
        self.assertEqual([a["trials"] for a in preview["arms"]], [3, 4])
        plain = self.ok(self.client.get("/api/plan", params={"mode": "sample", "arms": "cc,pi"}))
        self.assertEqual(plain["trials"], 8)
        body = {"mode": "sample", "arms": ["cc", "pi"], "reuse": "measured"}
        self.assertEqual(self.client.post("/api/runs", json=body).status_code, 400)
        stale = self.client.post("/api/runs", json={**body, "spec_hash": plain["spec_hash"]})
        self.assertEqual(stale.status_code, 409, stale.text)
        self.assertEqual(self.manager.started_specs, [])
        run = self.ok(self.client.post("/api/runs", json={**body, "spec_hash": preview["spec_hash"]}))
        (started,) = self.manager.started_specs
        self.assertEqual((started.source, started.label, started.spec_hash), ("config", "sample", preview["spec_hash"]))
        self.assertEqual([a.tasks for a in started.arms], [("task-b", "task-c", "task-d"), tuple(TASKS)])
        self.assertEqual(run["spec"]["reused"]["cc"][0]["trials"], [self.reused_key])
        self.assertEqual(self.client.get("/api/plan", params={"mode": "sample", "reuse": "maybe"}).status_code, 400)
