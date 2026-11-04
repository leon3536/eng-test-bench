"""Run designs: input normalization, validation, the write-once version store,
the assistant's draft-only handle, and round trips from config and past runs.

runs.plan_spec and datasets.task_meta are replaced here, so these tests run
no Harbor schema command and need no dataset snapshot.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import importlib
import json
import os
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

from pydantic import ValidationError

from tests.support import BenchTestCase

from bench import designs, fsutil, settings
from bench import spec as specs
from bench.designs import DesignNotFound, DesignStore, DesignUnreadable, DraftWriter, VersionConflict

IMPORT_PATH = "bench.agents.mini_swe_agent:MiniSweAgentWithMcp"
TASKS = ["task-a", "task-b", "task-c", "task-d"]
# The snapshot has one task outside the seeded sample.
SNAPSHOT = {t: {"difficulty": "medium"} for t in [*TASKS, "task-x"]}
CONVERSATION = "c-20261029-101500-ab12"

DESIGN_YAML = f"""\
dataset: terminal-bench@2.0
providers:
  anthropic: {{env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}}
  openrouter: {{env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}}
modes:
  pilot: {{n_tasks: 1}}
  sample: {{n_tasks: 3, n_attempts: 2}}
run:
  effort: medium
  n_concurrent: 2
  parallel_arms: 2
  budget_usd: 5
  per_trial_cap_usd: 1
  max_turns: 150
baseline_arm: cc-glm
arms:
  - {{name: cc-glm, agent: claude-code, provider: openrouter, model: z-ai/glm-5.3}}
  - {{name: pi-glm, agent: pi, provider: openrouter, model: z-ai/glm-5.3}}
  - name: mini-glm
    agent: {IMPORT_PATH}
    family: mini-swe-agent
    provider: openrouter
    model: z-ai/glm-5.3
    options: {{cost_limit: "1"}}
  - {{name: terminus-off, agent: terminus-2, provider: openrouter, model: z-ai/glm-5.3, enabled: false}}
designs:
  max_budget_usd: 10
  extra_harnesses: [strands]
"""

# A config arm that points strands at the factory project, as the shipped profiles do.
STRANDS_ARM = """\
  - name: strands-glm
    agent: strands
    provider: openrouter
    model: z-ai/glm-5.3
    effort: omit
    options:
      factory: bench_strands_factory:create_agent
      factory_path: harnesses/strands_factory
      version: "1.57.2"
"""
WITH_STRANDS_ARM = DESIGN_YAML.replace("designs:\n", STRANDS_ARM + "designs:\n")
# Not a real key: set in os.environ for one test (BenchTestCase restores it).
FAKE_KEY = "sk-or-v1-test0000fake1111value2222"

LEGACY_RUN: dict[str, Any] = {
    # The shape of a run.json written before specs were recorded (dev pilot 2).
    "id": "20261028-222952-pilot", "mode": "pilot", "created_at": "2026-10-29T05:29:52+00:00",
    "budget_usd": 5.0, "tasks": ["task-a"], "status": "done",
    "arms": [{"name": "cc-glm", "status": "done"}, {"name": "strands-glm", "status": "done"}],
    "plan": {
        "mode": "pilot", "tasks": ["task-a"], "n_attempts": 1,
        "arms": [
            {"name": "cc-glm", "agent": "claude-code", "provider": "openrouter", "model": "openrouter/z-ai/glm-5.3",
             "effort": "medium", "kwargs": [
                 {"name": "reasoning_effort", "value": "medium", "source": "effort"},
                 {"name": "max_budget_usd", "value": "1", "source": "run cap"},
                 {"name": "max_turns", "value": 150, "source": "run cap"}]},
            {"name": "strands-glm", "agent": "strands", "provider": "openrouter",
             "model": "openrouter/z-ai/glm-5.3", "effort": "harness default", "kwargs": [
                 {"name": "version", "value": "1.57.2", "source": "arm option"},
                 {"name": "factory", "value": "bench_strands_factory:create_agent", "source": "arm option"}]},
        ],
    },
    "config": {
        "dataset": "terminal-bench@2.0", "baseline_arm": "cc-glm",
        "providers": {"openrouter": {"env_key": "OPENROUTER_API_KEY", "model_prefix": "openrouter/"}},
        "modes": {"pilot": {"n_tasks": 1, "n_attempts": 1}},
        "run": {"effort": "medium", "n_concurrent": 1, "parallel_arms": 5, "budget_usd": 5,
                "per_trial_cap_usd": 1, "max_turns": 150, "timeout_multiplier": 1.0},
        "arms": [
            {"name": "cc-glm", "agent": "claude-code", "provider": "openrouter", "model": "z-ai/glm-5.3"},
            {"name": "strands-glm", "agent": "strands", "family": "strands", "provider": "openrouter",
             "model": "z-ai/glm-5.3", "effort": "omit",
             "options": {"factory": "bench_strands_factory:create_agent", "factory_path": "harnesses/x"}},
        ],
    },
}


def patch_lazy(case: BenchTestCase, module_name: str, attr: str, value: Any) -> None:
    """Patch an attribute of a module designs imports lazily. Other packages
    may be mid-edit, so a module that fails to import gets a stand-in."""
    try:
        importlib.import_module(module_name)
    except Exception:
        fake = types.ModuleType(module_name)
        package, _, leaf = module_name.rpartition(".")
        parent = sys.modules[package]
        sys.modules[module_name] = fake
        setattr(parent, leaf, fake)

        def undo() -> None:
            if sys.modules.get(module_name) is fake:
                del sys.modules[module_name]
            if getattr(parent, leaf, None) is fake:
                delattr(parent, leaf)

        case.addCleanup(undo)
    patcher = mock.patch(f"{module_name}.{attr}", value, create=True)
    patcher.start()
    case.addCleanup(patcher.stop)


def fake_plan(cfg: settings.Config, run_spec: specs.RunSpec) -> dict[str, Any]:
    return {"source": run_spec.source, "label": run_spec.label, "spec_hash": run_spec.spec_hash,
            "tasks": list(run_spec.tasks), "problems": [], "warnings": [], "peak_containers": 0,
            "arms": [{"name": a.name, "warnings": []} for a in run_spec.arms]}


def files_under(path: Path) -> set[str]:
    return {str(p.relative_to(path)) for p in path.rglob("*")}


class DesignTestCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cfg = self.load_test_config(DESIGN_YAML, tasks=TASKS)
        self.data = self.isolate_data_dir()
        self.store = DesignStore(self.data / "designs")
        self.plan_spec = mock.Mock(side_effect=fake_plan)
        patch_lazy(self, "bench.runs", "plan_spec", self.plan_spec)
        self.task_meta = mock.Mock(return_value=copy.deepcopy(SNAPSHOT))
        patch_lazy(self, "bench.datasets", "task_meta", self.task_meta)

    def spec_in(self, **changes: Any) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "tasks": ["task-a", "task-b"],
            "baseline_arm": "cc",
            "run": {"budget_usd": 3},
            "arms": [
                {"name": "cc", "agent": "claude-code", "provider": "openrouter", "model": "z-ai/glm-5.3"},
                {"name": "pi", "agent": "pi", "provider": "openrouter", "model": "z-ai/glm-5.3"},
            ],
        }
        spec.update(changes)
        return spec

    def spec(self, **changes: Any) -> dict[str, Any]:
        return designs.normalize(self.cfg, self.spec_in(**changes))

    def problems(self, spec: Any) -> list[str]:
        result = designs.validate(self.cfg, spec)
        self.assertEqual(result["status"], "invalid" if result["problems"] else "valid")
        return result["problems"]

    def create(self, title: str = "GLM pairs", **spec_changes: Any) -> dict[str, Any]:
        return self.store.create(self.cfg, {"title": title}, self.spec(**spec_changes), "user")


# Normalizing -------------------------------------------------------------------


class NormalizeTests(DesignTestCase):
    def test_fills_run_defaults_from_profile(self) -> None:
        spec = self.spec()
        self.assertEqual(spec["run"], {"budget_usd": 3.0, "parallel_arms": 2, "n_concurrent": 2, "effort": "medium",
                                       "per_trial_cap_usd": 1.0, "max_turns": 150, "timeout_multiplier": 1.0})
        self.assertIsInstance(spec["run"]["budget_usd"], float)
        self.assertEqual((spec["dataset"], spec["attempts"], spec["baseline_arm"]), ("terminal-bench@2.0", 1, "cc"))
        self.assertEqual(spec["arms"][0], {"name": "cc", "agent": "claude-code", "family": "claude-code",
                                           "provider": "openrouter", "model": "z-ai/glm-5.3", "effort": "inherit",
                                           "options": {}, "tasks": None})
        self.assertEqual(list(spec), list(specs.SPEC_KEYS))
        self.assertEqual(specs.structure_problems(spec), [])

    def test_given_run_fields_win_and_null_caps_stay_null(self) -> None:
        run = self.spec(run={"budget_usd": 3, "per_trial_cap_usd": None, "max_turns": None, "parallel_arms": 1,
                             "effort": "high", "timeout_multiplier": 2})["run"]
        self.assertIsNone(run["per_trial_cap_usd"])
        self.assertIsNone(run["max_turns"])
        self.assertEqual((run["parallel_arms"], run["effort"], run["timeout_multiplier"]), (1, "high", 2.0))

    def test_null_is_not_a_value_for_other_run_fields(self) -> None:
        for field in ("parallel_arms", "n_concurrent", "effort", "timeout_multiplier"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.spec(run={"budget_usd": 3, field: None})

    def test_strict_types(self) -> None:
        arm = self.spec_in()["arms"][0]
        bad = {
            "run effort false (YAML off)": {"run": {"budget_usd": 3, "effort": False}},
            "run effort off": {"run": {"budget_usd": 3, "effort": "off"}},
            "arm effort false": {"arms": [{**arm, "effort": False}]},
            "budget as text": {"run": {"budget_usd": "5"}},
            "budget nan": {"run": {"budget_usd": float("nan")}},
            "budget zero": {"run": {"budget_usd": 0}},
            "attempts true": {"attempts": True},
            "attempts float": {"attempts": 2.0},
            "parallel arms float": {"run": {"budget_usd": 3, "parallel_arms": 2.0}},
            "tasks as text": {"tasks": "task-a"},
            "options as list": {"arms": [{**arm, "options": []}]},
            "no run": {"run": None},
        }
        for name, change in bad.items():
            with self.subTest(case=name), self.assertRaises(ValidationError):
                self.spec(**change)
        # Pi's own "off" level is text, and text is accepted.
        self.assertEqual(self.spec(arms=[{**arm, "effort": "off"}])["arms"][0]["effort"], "off")

    def test_family_is_derived_never_given(self) -> None:
        arm = self.spec_in()["arms"][0]
        with self.assertRaises(ValidationError):
            self.spec(arms=[{**arm, "family": "claude-code"}])
        with self.assertRaises(ValidationError):
            self.spec(dataset="terminal-bench@2.0")
        spec = self.spec(arms=[
            {"name": "mini", "agent": IMPORT_PATH, "provider": "openrouter", "model": "z-ai/glm-5.3"},
            {"name": "strands", "agent": "strands", "provider": "openrouter", "model": "z-ai/glm-5.3"},
        ], baseline_arm=None)
        self.assertEqual([a["family"] for a in spec["arms"]], ["mini-swe-agent", "strands"])
        mini = specs.spec_from_design(spec, None).arms[0]
        self.assertEqual((mini.kind, mini.family), ("import_path", "mini-swe-agent"))

    def test_unknown_agent_is_a_value_error(self) -> None:
        for agent in ("opencode", "evil.module:Thing"):
            with self.subTest(agent=agent):
                with self.assertRaises(ValueError) as ctx:
                    self.spec(arms=[{"name": "x", "agent": agent, "provider": "openrouter", "model": "m"}])
                self.assertNotIsInstance(ctx.exception, ValidationError)
                self.assertIn("not a harness designs may use", str(ctx.exception))

    def test_tidies_options_tasks_and_model(self) -> None:
        arm = {"name": "pi", "agent": "pi", "provider": "openrouter", "model": "openrouter/z-ai/glm-5.3",
               "options": {"max_turns": 50, "config": {"z": 1, "a": 2}}, "tasks": ["task-z", "task-b", "task-a"]}
        out = self.spec(arms=[arm], baseline_arm=None)["arms"][0]
        self.assertEqual(list(out["options"]), ["config", "max_turns"])
        self.assertEqual(out["tasks"], ["task-a", "task-b", "task-z"])  # outsiders last, for validate to report
        self.assertEqual(out["model"], "z-ai/glm-5.3")

    def test_input_form_round_trip(self) -> None:
        spec = self.spec(attempts=2, run={"budget_usd": 3, "per_trial_cap_usd": None})
        again = designs.to_input(spec)
        designs.DesignSpecIn.model_validate(again)
        self.assertEqual(designs.normalize(self.cfg, again), spec)


class HarnessSetupRunFieldTests(DesignTestCase):
    """prebake, setup_timeout_multiplier and env_build_timeout_multiplier:
    omitted, they follow the profile; written into the spec only when they
    differ from the code defaults, so older specs keep their hash."""

    def test_omitted_fields_follow_the_profile_and_given_ones_win(self) -> None:
        self.assertEqual(list(self.spec()["run"]), list(specs.RUN_KEYS))
        run = self.spec(run={"budget_usd": 3, "prebake": "require", "setup_timeout_multiplier": 2,
                             "env_build_timeout_multiplier": 1.5})["run"]
        self.assertEqual({k: run[k] for k in specs.OPTIONAL_RUN_KEYS if k in run},
                         {"prebake": "require", "setup_timeout_multiplier": 2.0, "env_build_timeout_multiplier": 1.5})
        raced = self.spec(run={"budget_usd": 3, "schedule": "arms", "max_containers": 4})["run"]
        self.assertEqual((raced["schedule"], raced["max_containers"]), ("arms", 4))
        self.assertNotIn("setup_timeout_multiplier",
                         self.spec(run={"budget_usd": 3, "setup_timeout_multiplier": "auto"})["run"])
        off = self.load_test_config(DESIGN_YAML.replace("  max_turns: 150\n", "  max_turns: 150\n  prebake: 'off'\n"),
                                    tasks=TASKS)
        self.assertEqual(designs.normalize(off, self.spec_in())["run"]["prebake"], "off")
        self.assertEqual(designs.normalize(off, self.spec_in(run={"budget_usd": 3, "prebake": "prefer"}))["run"],
                         self.spec()["run"])

    def test_bad_values_are_refused(self) -> None:
        for run in ({"prebake": "sometimes"}, {"prebake": False}, {"prebake": None},
                    {"setup_timeout_multiplier": "fast"}, {"setup_timeout_multiplier": 0},
                    {"setup_timeout_multiplier": None}, {"env_build_timeout_multiplier": 0},
                    {"env_build_timeout_multiplier": "2"}):
            with self.subTest(run=run), self.assertRaises(ValidationError):
                self.spec(run={"budget_usd": 3, **run})

    def test_limits_are_problems(self) -> None:
        for run, message in (({"setup_timeout_multiplier": 5}, "run.setup_timeout_multiplier must be auto, or above 0"),
                             ({"env_build_timeout_multiplier": 4.5}, "run.env_build_timeout_multiplier")):
            with self.subTest(run=run):
                problems = self.problems(self.spec(run={"budget_usd": 3, **run}))
                self.assertTrue(any(message in p for p in problems), problems)
        self.assertEqual(self.problems(self.spec(run={"budget_usd": 3, "setup_timeout_multiplier": 4,
                                                      "prebake": "off"})), [])


class SchemaTests(BenchTestCase):
    def test_schema_matches_input_models(self) -> None:
        def shape(schema: dict[str, Any]) -> tuple[set[str], set[str]]:
            return set(schema["properties"]), set(schema.get("required", []))

        def model_shape(model: type) -> tuple[set[str], set[str]]:
            fields = model.model_fields  # type: ignore[attr-defined]
            return set(fields), {n for n, f in fields.items() if f.is_required()}

        schema = designs.DESIGN_SPEC_SCHEMA
        run = schema["properties"]["run"]
        arm = schema["properties"]["arms"]["items"]
        self.assertEqual(shape(schema), model_shape(designs.DesignSpecIn))
        self.assertEqual(shape(run), model_shape(designs.RunIn))
        self.assertEqual(shape(arm), model_shape(designs.ArmIn))
        for level in (schema, run, arm):
            self.assertIs(level["additionalProperties"], False)
        self.assertEqual(run["properties"]["effort"]["enum"], list(designs.RUN_EFFORTS))
        self.assertNotIn("family", arm["properties"])
        self.assertIn("family", arm["properties"]["agent"]["description"])
        self.assertNotIn("$ref", json.dumps(schema))


# Validation --------------------------------------------------------------------


class ValidateTests(DesignTestCase):
    def test_valid_design_is_planned(self) -> None:
        spec = self.spec()
        result = designs.validate(self.cfg, spec)
        self.assertEqual((result["status"], result["problems"]), ("valid", []))
        self.assertEqual(result["plan"]["label"], "design")
        self.plan_spec.assert_called_once()
        run_spec = self.plan_spec.call_args.args[1]
        self.assertEqual((run_spec.source, run_spec.spec_hash), ("design", specs.spec_hash(spec)))

    def test_problems(self) -> None:
        def arm(i: int, **fields: Any) -> Callable[[dict[str, Any]], None]:
            return lambda s: s["arms"][i].update(fields)

        def run(**fields: Any) -> Callable[[dict[str, Any]], None]:
            return lambda s: s["run"].update(fields)

        cases: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
            ("base_url option", arm(0, options={"base_url": "http://127.0.0.1:9"}), "(base_url)"),
            ("api_base option", arm(0, options={"api_base": "http://x"}), "(api_base)"),
            ("nested key", arm(1, options={"llm_kwargs": {"api_key": "x"}}), "llm_kwargs.api_key"),
            ("token option", arm(1, options={"auth_token": "x"}), "auth_token"),
            ("secret option", arm(1, options={"factory_kwargs": {"client_secret": "x"}}), "client_secret"),
            # The same things in LiteLLM's, OpenAI's and Strands' own spellings.
            ("baseUrl option", arm(0, options={"baseUrl": "http://x"}), "(baseUrl)"),
            ("baseURL option", arm(0, options={"llm_kwargs": {"baseURL": "http://x"}}), "(llm_kwargs.baseURL)"),
            ("apiBase option", arm(0, options={"apiBase": "http://x"}), "(apiBase)"),
            ("apiKey option", arm(1, options={"model_kwargs": {"apiKey": "x"}}), "model_kwargs.apiKey"),
            ("x-api-key header", arm(1, options={"extra_headers": {"x-api-key": "x"}}), "extra_headers.x-api-key"),
            ("authorization header", arm(1, options={"headers": {"Authorization": "Bearer x"}}),
             "headers.Authorization"),
            ("default headers", arm(1, options={"factory_kwargs": {"default_headers": {"Authorization": "Bearer x"}}}),
             "factory_kwargs.default_headers.Authorization"),
            ("proxy authorization", arm(1, options={"headers": {"Proxy-Authorization": "Basic x"}}),
             "headers.Proxy-Authorization"),
            ("cookie header", arm(1, options={"headers": {"Cookie": "session=x"}}), "headers.Cookie"),
            ("host file", arm(0, options={"prompt_template_path": "C:/Users/me/.ssh/id_rsa"}),
             "host files (prompt_template_path)"),
            ("strands model_spec", arm(1, options={"factory_kwargs": {"model_spec": "anthropic/claude-opus-5-5"}}),
             "factory_kwargs.model_spec is 'anthropic/claude-opus-5-5' but the arm runs 'openrouter/z-ai/glm-5.3'"),
            ("budget over limit", run(budget_usd=10.5), "designs.max_budget_usd"),
            ("attempts over limit", lambda s: s.update(attempts=6), "designs.max_attempts"),
            ("parallel arms", run(parallel_arms=13), "run.parallel_arms"),
            ("concurrency", run(n_concurrent=9), "run.n_concurrent"),
            ("timeout", run(timeout_multiplier=4.5), "run.timeout_multiplier"),
            ("run effort", run(effort="turbo"), "run.effort"),
            ("zero cap", run(per_trial_cap_usd=0), "per_trial_cap_usd"),
            ("arm task outside", arm(1, tasks=["task-c"]), "not in the design's task list"),
            ("arm task order", arm(1, tasks=["task-b", "task-a"]), "design's task order"),
            ("arm task repeated", arm(1, tasks=["task-a", "task-a"]), "more than once"),
            ("arm tasks empty", arm(1, tasks=[]), "empty task list"),
            ("unknown task", lambda s: s.update(tasks=["task-a", "task-zzz"]), "task-zzz"),
            ("repeated task", lambda s: s.update(tasks=["task-a", "task-a"]), "more than once"),
            ("no tasks", lambda s: s.update(tasks=[]), "at least one task"),
            ("no arms", lambda s: s.update(arms=[], baseline_arm=None), "at least one arm"),
            ("dataset", lambda s: s.update(dataset="other@1.0"), "this profile runs terminal-bench@2.0"),
            ("bad name", arm(0, name="Bad_Name"), "lowercase letters"),
            ("long name", arm(0, name="a" * 33), "at most 32"),
            ("device name", arm(0, name="con"), "Windows device name"),
            ("repeated name", arm(1, name="cc"), "unique"),
            ("baseline", lambda s: s.update(baseline_arm="ghost"), "baseline_arm 'ghost'"),
            ("provider", arm(0, provider="fireworks"), "no provider 'fireworks'"),
            ("import path", arm(0, agent="evil.module:Thing", family="evil"), "designs.import_paths"),
            ("agent name", arm(0, agent="opencode", family="opencode"), "designs.extra_harnesses"),
            ("too many arms", lambda s: s.update(arms=[{**s["arms"][0], "name": f"a{i}"} for i in range(13)],
                                                 baseline_arm=None), "designs.max_arms (12)"),
        ]
        for name, edit, expected in cases:
            with self.subTest(case=name):
                spec = self.spec()
                edit(spec)
                problems = self.problems(spec)
                self.assertTrue(any(expected in p for p in problems), problems)

    def test_disallowed_agents_are_never_planned(self) -> None:
        spec = self.spec()
        spec["arms"][0].update(agent="evil.module:Thing", family="evil")
        result = designs.validate(self.cfg, spec)
        self.assertEqual(result["status"], "invalid")
        self.assertIsNone(result["plan"])
        self.plan_spec.assert_not_called()

    def test_host_file_options(self) -> None:
        # Strands copies factory_path into the trial's mounted logs; mini-swe-agent reads config_file on the host.
        strands = {"name": "strands", "agent": "strands", "provider": "openrouter", "model": "z-ai/glm-5.3",
                   "effort": "omit", "options": {"factory": "bench_strands_factory:create_agent",
                                                 "factory_path": ".env"}}
        mini = {"name": "mini", "agent": IMPORT_PATH, "provider": "openrouter", "model": "z-ai/glm-5.3",
                "options": {"config_file": ".env"}}
        nested = {**mini, "options": {"agent_kwargs": {"memory_dir": "C:/Users/me"}, "configFile": "x.yaml"}}
        # Claude Code reads a text `config` as a host file and uploads it as settings.json.
        cc = {"name": "cc", "agent": "claude-code", "provider": "openrouter", "model": "z-ai/glm-5.3",
              "options": {"config": "C:/Users/me/.docker/config.json"}}
        for arm, options in ((strands, "factory_path"), (mini, "config_file"),
                             (nested, "agent_kwargs.memory_dir, configFile"), (cc, "config")):
            with self.subTest(options=options):
                problems = self.problems(self.spec(arms=[arm], baseline_arm=None))
                self.assertIn(f"Arm '{arm['name']}': designs cannot point a harness at host files ({options}); "
                              "set it on a config arm instead", problems)

        inline = {**cc, "options": {"config": {"permissions": {}}}}
        self.assertFalse(any("host files" in p for p in self.problems(self.spec(arms=[inline], baseline_arm=None))))

        # A config arm's own value is allowed for the same agent, at the same path, and nothing else.
        cfg = self.load_test_config(WITH_STRANDS_ARM, tasks=TASKS)
        shipped = {**strands, "options": {**strands["options"], "factory_path": "harnesses/strands_factory"}}
        cases = {
            "the config arm's value": (shipped, False),
            "null, which sets nothing": ({**strands, "options": {"factory_path": None}}, False),
            ".env": (strands, True),
            "a near miss": ({**strands, "options": {"factory_path": "harnesses/strands_factory/../.."}}, True),
            "the value at another path": ({**strands, "options": {"factory_kwargs": {
                "factory_path": "harnesses/strands_factory"}}}, True),
            "the value for another agent": ({**mini, "options": {"factory_path": "harnesses/strands_factory"}}, True),
        }
        for name, (arm, refused) in cases.items():
            with self.subTest(case=name):
                result = designs.validate(cfg, designs.normalize(cfg, self.spec_in(arms=[arm], baseline_arm=None)))
                self.assertEqual(any("host files" in p for p in result["problems"]), refused, result["problems"])

    def test_shipped_config_designs_pass(self) -> None:
        for profile in ("bench", "dev"):
            with self.subTest(profile=profile):
                cfg = self.load_shipped_config(profile)
                self.task_meta.return_value = {t: {} for t in cfg.tasks}
                spec = designs.design_from_config(cfg, "pilot", None)
                strands = [a for a in spec["arms"] if a["agent"] == "strands"]
                self.assertTrue(strands)
                for arm in strands:
                    self.assertEqual(arm["options"]["factory_path"], "harnesses/strands_factory")
                self.assertEqual(designs.validate(cfg, spec)["problems"], [])
                self.assertEqual(designs.normalize(cfg, designs.to_input(spec)), spec)

    def test_ordinary_option_names_pass_the_denylist(self) -> None:
        spec = self.spec()
        spec["arms"][1]["options"] = {"max_tokens": 5, "llm_call_kwargs": {"max_tokens": 10},
                                      "model_info": {"input_cost_per_token": 1e-6}, "session_id_headers": ["x"],
                                      "factory_kwargs": {"model_spec": "openrouter/z-ai/glm-5.3"}}
        self.assertEqual(self.problems(spec), [])

    def test_never_raises(self) -> None:
        good = self.spec()
        garbage = {
            "none": None, "list": [], "text": "spec", "number": 42, "empty": {},
            "arms text": {**good, "arms": "x"},
            "run null": {**good, "run": None},
            "arm null": {**good, "arms": [None]},
            "numeric tasks": {**good, "tasks": [1, 2]},
            "nan budget": {**good, "run": {**good["run"], "budget_usd": float("nan")}},
            "option keys": {**good, "arms": [{**good["arms"][0], "options": {1: 2}}]},
            "nan option": {**good, "arms": [{**good["arms"][0], "options": {"x": {"y": float("nan")}}}]},
            "set option": {**good, "arms": [{**good["arms"][0], "options": {"x": {1, 2}}}]},
            "extra field": {**good, "status": "approved"},
        }
        for name, spec in garbage.items():
            with self.subTest(case=name):
                result = designs.validate(self.cfg, spec)
                self.assertEqual(result["status"], "invalid")
                self.assertTrue(result["problems"])
                self.assertIsNone(result["plan"])
        self.plan_spec.assert_not_called()

    def test_planner_and_check_failures_become_problems(self) -> None:
        self.plan_spec.side_effect = RuntimeError("harbor agents schema pi failed")
        result = designs.validate(self.cfg, self.spec())
        self.assertIn("The run could not be planned: harbor agents schema pi failed", result["problems"])
        with mock.patch.object(designs, "_design_warnings", side_effect=KeyError("boom")):
            result = designs.validate(self.cfg, self.spec())
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(result["problems"][0].startswith("The design could not be checked"))

    def test_plan_messages_are_merged_once(self) -> None:
        self.plan_spec.side_effect = None
        self.plan_spec.return_value = {
            "problems": ["pi: OPENROUTER_API_KEY is not set", "pi: OPENROUTER_API_KEY is not set"],
            "warnings": ["Docker memory is tight"],
            "arms": [{"name": "cc", "warnings": ["max_budget_usd: Not applied"]}, {"name": "pi"}],
        }
        result = designs.validate(self.cfg, self.spec())
        self.assertEqual(result["problems"], ["pi: OPENROUTER_API_KEY is not set"])
        self.assertEqual(result["warnings"], ["Docker memory is tight", "cc: max_budget_usd: Not applied"])
        self.assertEqual(result["status"], "invalid")

    def test_container_limits(self) -> None:
        three = [*self.spec_in()["arms"],
                 {"name": "mini", "agent": IMPORT_PATH, "provider": "openrouter", "model": "z-ai/glm-5.3"}]
        warned = self.spec(arms=three, tasks=TASKS, run={"budget_usd": 3, "parallel_arms": 3, "n_concurrent": 4,
                                                             "schedule": "arms"})
        result = designs.validate(self.cfg, warned)
        self.assertEqual(result["status"], "valid")
        self.assertTrue(any("Up to 12 task containers" in w for w in result["warnings"]))
        blocked = self.spec(arms=three, tasks=TASKS, attempts=3,
                            run={"budget_usd": 3, "parallel_arms": 3, "n_concurrent": 8, "schedule": "arms"})
        self.assertTrue(any("Up to 24 task containers" in p for p in self.problems(blocked)))
        # Only containers that can exist count: one trial needs one, however wide the settings.
        small = self.spec(tasks=["task-a"], arms=three[:1], run={"budget_usd": 3, "parallel_arms": 12,
                                                                 "n_concurrent": 8})
        self.assertEqual(designs.peak_containers(small), 1)
        self.assertFalse(any("containers" in w for w in designs.validate(self.cfg, small)["warnings"]))

    def test_design_warnings(self) -> None:
        cc = self.spec_in()["arms"][0]
        cases = {
            "same configuration": self.spec(arms=[cc, {**cc, "name": "cc2"}]),
            "Pairs form only within one model": self.spec(arms=[cc, {**cc, "name": "ds",
                                                                     "model": "deepseek/deepseek-v4.1-flash"}]),
            "runs 1 of the design's 2 tasks": self.spec(arms=[cc, {**cc, "name": "pi", "agent": "pi",
                                                                   "tasks": ["task-b"]}]),
            "seeded-sample order": self.spec(tasks=["task-b", "task-a"]),
            "outside the seeded sample (task-x)": self.spec(tasks=["task-a", "task-x"]),
        }
        for expected, spec in cases.items():
            with self.subTest(warning=expected):
                result = designs.validate(self.cfg, spec)
                self.assertEqual(result["status"], "valid", result["problems"])
                self.assertTrue(any(expected in w for w in result["warnings"]), result["warnings"])

    def test_unknown_tasks_are_problems_not_sample_warnings(self) -> None:
        result = designs.validate(self.cfg, self.spec(tasks=["task-a", "task-zzz"]))
        self.assertTrue(any("task-zzz" in p for p in result["problems"]))
        self.assertFalse(any("outside the seeded sample" in w for w in result["warnings"]))

    def test_changed_family_is_a_warning(self) -> None:
        spec = self.spec()
        spec["arms"][1]["family"] = "pi-fork"
        result = designs.validate(self.cfg, spec)
        self.assertEqual(result["status"], "valid")
        self.assertTrue(any("pool apart" in w for w in result["warnings"]))

    def test_task_check_falls_back_to_the_sample(self) -> None:
        for meta in ({}, RuntimeError("no snapshot")):
            with self.subTest(task_meta=meta):
                if isinstance(meta, Exception):
                    self.task_meta.side_effect = meta
                else:
                    self.task_meta.return_value = meta
                problems = self.problems(self.spec(tasks=["task-a", "task-x"]))
                self.assertTrue(any("not in the seeded sample" in p and "task-x" in p for p in problems))


# Store -------------------------------------------------------------------------


class StoreTests(DesignTestCase):
    def test_create_writes_a_version_and_meta(self) -> None:
        d = self.store.create(self.cfg, {"title": "GLM pairs", "purpose": "Pairs on GLM"}, self.spec(), "user")
        self.assertRegex(d["design_id"], r"^d-glm-pairs-[0-9a-f]{4}$")
        self.assertEqual((d["version"], d["parent_version"], d["author"], d["conversation_id"]),
                         (1, None, "user", None))
        self.assertEqual((d["status"], d["profile"], d["purpose"]), ("draft", "default", "Pairs on GLM"))
        self.assertEqual(d["spec"], self.spec())
        self.assertEqual(d["spec_hash"], specs.spec_hash(d["spec"]))
        self.assertEqual((d["diff_from_parent"], d["check"], d["runs"], d["approved_at"]), ([], None, [], None))
        folder = self.store.root / d["design_id"]
        self.assertEqual(files_under(folder), {"v0001.json", "meta.json"})
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["id"], meta["latest_version"], meta["archived"]), (d["design_id"], 1, False))
        self.assertEqual(meta["versions"]["1"]["spec_hash"], d["spec_hash"])

    def test_default_root_is_the_profile_designs_folder(self) -> None:
        self.assertEqual(DesignStore().root, settings.DESIGNS_DIR)
        self.assertEqual(settings.DESIGNS_DIR, self.data / "designs")

    def test_ids_from_titles(self) -> None:
        for title in ("Ünïcode: Block 1/2 — GLM!!", "!!!", "x" * 80, "con", "  Pi vs CC  "):
            with self.subTest(title=title):
                design_id = designs.new_design_id(title)
                self.assertTrue(designs.valid_design_id(design_id), design_id)
                self.assertLessEqual(len(design_id), 31)
        self.assertRegex(designs.new_design_id("Ünïcode: Block 1/2 — GLM!!"),
                         r"^d-unicode-block-1-2-glm-[0-9a-f]{4}$")
        self.assertRegex(designs.new_design_id("!!!"), r"^d-design-[0-9a-f]{4}$")

    def test_versions_are_write_once(self) -> None:
        d1 = self.create()
        design_id = d1["design_id"]
        v1 = self.store.root / design_id / "v0001.json"
        before = v1.read_bytes()
        d2 = self.store.add_version(self.cfg, design_id, 1, {"title": "GLM pairs", "change_note": "More budget"},
                                    self.spec(run={"budget_usd": 4}), "user")
        self.assertEqual((d2["version"], d2["parent_version"], d2["change_note"]), (2, 1, "More budget"))
        self.assertEqual(d2["diff_from_parent"], [{"path": "run.budget_usd", "from": 3.0, "to": 4.0}])
        self.assertEqual(v1.read_bytes(), before)
        self.assertEqual(self.store.version(design_id, 1)["spec"]["run"]["budget_usd"], 3.0)
        self.assertEqual(self.store.meta(design_id)["latest_version"], 2)
        with self.assertRaises(FileExistsError):
            fsutil.create_exclusive_json(v1, {"tampered": True})
        self.assertEqual(v1.read_bytes(), before)

    def test_stale_base_version_conflicts(self) -> None:
        design_id = self.create()["design_id"]
        self.store.add_version(self.cfg, design_id, 1, {"title": "v2"}, self.spec(), "user")
        with self.assertRaises(VersionConflict) as ctx:
            self.store.add_version(self.cfg, design_id, 1, {"title": "v2b"}, self.spec(attempts=2), "assistant",
                                   CONVERSATION)
        self.assertIn("Version 2", str(ctx.exception))
        self.assertIn("since you read version 1", str(ctx.exception))
        self.assertIsInstance(ctx.exception, RuntimeError)  # the server's 409

    def test_unchanged_saves_conflict_but_title_edits_keep_the_hash(self) -> None:
        d1 = self.create()
        design_id = d1["design_id"]
        for meta in ({"title": "GLM pairs"}, {"title": "GLM pairs", "change_note": "no real change"}):
            with self.subTest(meta=meta), self.assertRaises(VersionConflict):
                self.store.add_version(self.cfg, design_id, 1, meta, self.spec(), "user")
        d2 = self.store.add_version(self.cfg, design_id, 1, {"title": "GLM pairs, block 1"}, self.spec(), "user")
        self.assertEqual(d2["spec_hash"], d1["spec_hash"])
        self.assertEqual(d2["diff_from_parent"], [{"path": "title", "from": "GLM pairs", "to": "GLM pairs, block 1"}])
        self.assertEqual(self.store.meta(design_id)["title"], "GLM pairs, block 1")

    def test_archived_designs_take_no_versions_or_approvals(self) -> None:
        design_id = self.create()["design_id"]
        self.store.archive(design_id)
        with self.assertRaises(VersionConflict):
            self.store.add_version(self.cfg, design_id, 1, {"title": "v2"}, self.spec(attempts=2), "user")
        with self.assertRaises(VersionConflict):
            self.store.mark_approved(design_id, 1)
        self.assertEqual(self.store.list(), [])
        listed = self.store.list(include_archived=True)
        self.assertEqual([(s["id"], s["status"]) for s in listed], [(design_id, "archived")])
        self.store.unarchive(design_id)
        self.assertEqual(self.store.add_version(self.cfg, design_id, 1, {"title": "v2"}, self.spec(attempts=2),
                                                "user")["version"], 2)

    def test_derived_status(self) -> None:
        design_id = self.create()["design_id"]
        self.assertEqual(self.store.version(design_id)["status"], "draft")
        self.store.mark_approved(design_id, 1)
        approved_at = self.store.version(design_id)["approved_at"]
        self.assertIsNotNone(approved_at)
        self.store.mark_approved(design_id, 1)
        self.assertEqual(self.store.version(design_id)["approved_at"], approved_at)
        self.assertEqual(self.store.version(design_id)["status"], "approved")
        for item, status in (("waiting", "queued"), ("held", "queued"), ("starting", "running"),
                             ("running", "running"), ("done", "approved")):
            with self.subTest(queue_item=item):
                queue = {(design_id, 1): item}
                self.assertEqual(self.store.version(design_id, 1, queue_status=queue)["status"], status)
                self.assertEqual(self.store.list(queue_status=queue)[0]["status"], status)
                meta = self.store.meta(design_id, queue_status=queue)
                self.assertEqual((meta["status"], meta["versions"]["1"]["status"]), (status, status))
        run_id = "20261029-101500-glm-pairs-v1"
        self.store.record_run(design_id, 1, run_id)
        self.store.record_run(design_id, 1, run_id)
        self.assertEqual(self.store.version(design_id)["runs"], [run_id])
        self.assertEqual(self.store.version(design_id)["status"], "ran")
        with self.assertRaises(ValueError):
            self.store.record_run(design_id, 1, "../escape")
        v2 = self.store.add_version(self.cfg, design_id, 1, {"title": "v2"}, self.spec(attempts=2), "user")
        self.assertEqual(v2["status"], "draft")
        meta = self.store.meta(design_id)
        self.assertEqual((meta["versions"]["1"]["status"], meta["versions"]["2"]["status"]), ("ran", "draft"))
        # A queue item is per version: v1 queued again leaves the latest a draft.
        meta = self.store.meta(design_id, queue_status={(design_id, 1): "waiting"})
        self.assertEqual((meta["status"], meta["versions"]["1"]["status"], meta["versions"]["2"]["status"]),
                         ("draft", "queued", "draft"))
        self.store.archive(design_id)
        self.assertEqual(self.store.version(design_id, 1)["status"], "archived")

    def test_record_check_is_cached_beside_the_version(self) -> None:
        design_id = self.create()["design_id"]
        check = {"validation": {"status": "valid", "problems": [], "warnings": ["w"]},
                 "estimate": {"run": {"total": {"p10": 0.1, "p50": 0.4, "p90": 0.8}}},
                 "config_fingerprint": designs.config_fingerprint(self.cfg), "ledger_fingerprint": "sha256:abc"}
        self.store.record_check(design_id, 1, check)
        cached = self.store.version(design_id)["check"]
        self.assertEqual(cached["estimate"], check["estimate"])
        self.assertIn("checked_at", cached)
        summary = self.store.list()[0]
        self.assertEqual((summary["problems"], summary["warnings"], summary["expected_usd"], summary["p90_usd"]),
                         (0, 1, 0.4, 0.8))
        self.assertEqual((summary["arms"], summary["trials"]), (2, 4))
        with self.assertRaises(DesignNotFound):
            self.store.record_check(design_id, 2, check)

    def test_bad_ids_are_rejected_before_any_path_is_built(self) -> None:
        good = self.create()["design_id"]
        before = files_under(self.data.parent)
        bad_ids = ["../../config", "..\\..\\config", "d-../../x", "d-abc/../x", "D-ABC", "d-ABC", "con", "nul",
                   "d-", "", "d-" + "a" * 32, f"{good}/", None, 7]
        operations: dict[str, Callable[[Any], Any]] = {
            "meta": lambda i: self.store.meta(i),
            "version": lambda i: self.store.version(i, 1),
            "add_version": lambda i: self.store.add_version(self.cfg, i, 1, {"title": "x"}, self.spec(), "user"),
            "record_check": lambda i: self.store.record_check(i, 1, {}),
            "mark_approved": lambda i: self.store.mark_approved(i, 1),
            "record_run": lambda i: self.store.record_run(i, 1, "run-1"),
            "archive": lambda i: self.store.archive(i),
            "unarchive": lambda i: self.store.unarchive(i),
        }
        for bad in bad_ids:
            for name, operation in operations.items():
                with self.subTest(id=bad, operation=name), self.assertRaises(ValueError):
                    operation(bad)
        self.assertEqual(files_under(self.data.parent), before)
        self.assertTrue(designs.valid_design_id("d-" + "a" * 31))

    def test_unknown_designs_and_versions(self) -> None:
        with self.assertRaises(DesignNotFound) as ctx:
            self.store.meta("d-missing-0000")
        self.assertIsInstance(ctx.exception, KeyError)
        self.assertEqual(str(ctx.exception), "No design 'd-missing-0000'")
        design_id = self.create()["design_id"]
        with self.assertRaises(DesignNotFound):
            self.store.version(design_id, 9)
        for version in (0, True, "1"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.store.version(design_id, version)  # type: ignore[arg-type]

    def test_corrupt_files_are_listed_not_fatal(self) -> None:
        good = self.create("Good one")["design_id"]
        bad_meta = self.create("Bad meta")["design_id"]
        bad_version = self.create("Bad version")["design_id"]
        (self.store.root / bad_meta / "meta.json").write_text("{not json", encoding="utf-8")
        (self.store.root / bad_version / "v0001.json").write_text("[]", encoding="utf-8")
        (self.store.root / "notes").mkdir()
        (self.store.root / "README.txt").write_text("hello", encoding="utf-8")
        listed = {s["id"]: s for s in self.store.list()}
        self.assertEqual(set(listed), {good, bad_meta, bad_version})
        self.assertEqual(listed[good]["status"], "draft")
        for design_id in (bad_meta, bad_version):
            self.assertEqual(listed[design_id]["status"], "unreadable")
            self.assertIn("Could not read", listed[design_id]["error"])
        with self.assertRaises(DesignUnreadable):
            self.store.meta(bad_meta)
        with self.assertRaises(DesignUnreadable):
            self.store.version(bad_version)

    def test_malformed_meta_records_are_listed_not_fatal(self) -> None:
        # meta.json is git-tracked: hand edits and merges can leave valid JSON of the wrong shape.
        good = self.create("Good one")["design_id"]
        changes: dict[str, Any] = {
            "null version entry": {"versions": {"1": None}},
            "text latest_version": {"latest_version": "1"},
            "bool latest_version": {"latest_version": True},
            "runs as text": {"versions": {"1": {"runs": "20261029-101500-pilot"}}},
            "a version key that is not a number": {"versions": {"1": {}, "x": {}}},
            "archived as text": {"archived": "no"},
        }
        broken = []
        for name, change in changes.items():
            design_id = self.create(name)["design_id"]
            path = self.store.root / design_id / "meta.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            meta.update(change)
            path.write_text(json.dumps(meta), encoding="utf-8")
            broken.append(design_id)
        odd = self.create("Odd date")["design_id"]
        path = self.store.root / odd / "meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["updated_at"] = 5
        path.write_text(json.dumps(meta), encoding="utf-8")

        listed = {s["id"]: s for s in self.store.list()}
        self.assertEqual(set(listed), {good, odd, *broken})
        self.assertEqual((listed[good]["status"], listed[odd]["status"]), ("draft", "draft"))
        for design_id in broken:
            with self.subTest(design=design_id):
                self.assertEqual(listed[design_id]["status"], "unreadable")
                self.assertIn("meta.json is not a design record", listed[design_id]["error"])
                with self.assertRaises(DesignUnreadable):
                    self.store.meta(design_id)
        # Anything else that breaks one summary is that design's problem, not the list's.
        with mock.patch.object(DesignStore, "_summary", side_effect=TypeError("bad entry")):
            listed = self.store.list()
        self.assertEqual({s["status"] for s in listed}, {"unreadable"})
        self.assertIn(f"Could not read design {good}: bad entry", {s["error"] for s in listed})

    def test_a_failed_version_write_leaves_no_partial_file(self) -> None:
        design_id = self.create()["design_id"]
        folder = self.store.root / design_id

        def partial(path: Path, data: Any) -> None:
            path.write_text('{"schema": 1, "spe', encoding="utf-8")
            raise OSError("disk full")

        with mock.patch.object(fsutil, "create_exclusive_json", side_effect=partial), self.assertRaises(OSError):
            self.store.add_version(self.cfg, design_id, 1, {"title": "v2"}, self.spec(attempts=2), "user")
        self.assertEqual(files_under(folder), {"v0001.json", "meta.json"})
        d2 = self.store.add_version(self.cfg, design_id, 1, {"title": "v2"}, self.spec(attempts=2), "user")
        self.assertEqual(d2["version"], 2)

    def test_an_unreadable_version_file_does_not_block_saves(self) -> None:
        design_id = self.create()["design_id"]
        folder = self.store.root / design_id
        (folder / "v0002.json").write_text('{"schema": 1, "spe', encoding="utf-8")  # left by a crash
        d = self.store.add_version(self.cfg, design_id, 1, {"title": "GLM pairs"}, self.spec(attempts=2), "user")
        self.assertEqual((d["version"], d["parent_version"]), (3, 1))
        self.assertEqual(d["diff_from_parent"], [{"path": "attempts", "from": 1, "to": 2}])
        self.assertEqual(sorted(self.store.meta(design_id)["versions"]), ["1", "3"])
        self.assertEqual(self.store.list()[0]["latest_version"], 3)
        self.assertEqual((folder / "v0002.json").read_text(encoding="utf-8"), '{"schema": 1, "spe')
        # Should another writer's half-made file take the number anyway, the conflict names it.
        (folder / "v0004.json").write_text("{", encoding="utf-8")
        with mock.patch.object(DesignStore, "_highest_on_disk", return_value=3), \
                self.assertRaises(VersionConflict) as ctx:
            self.store.add_version(self.cfg, design_id, 3, {"title": "v4"}, self.spec(attempts=3), "user")
        self.assertIn("v0004.json", str(ctx.exception))
        self.assertEqual((folder / "v0004.json").read_text(encoding="utf-8"), "{")

    def test_a_locked_meta_file_is_busy_not_missing(self) -> None:
        design_id = self.create()["design_id"]
        real = Path.read_text

        def locked(path: Path, *args: Any, **kwargs: Any) -> str:
            if path.name in ("meta.json", "v0001.json"):
                raise PermissionError(13, "The process cannot access the file")
            return real(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", locked):
            with self.assertRaises(PermissionError):
                self.store.meta(design_id)
            with self.assertRaises(PermissionError):
                self.store._read_version(design_id, 1)

    def test_credentials_are_refused_before_anything_is_written(self) -> None:
        os.environ["OPENROUTER_API_KEY"] = FAKE_KEY
        existing = self.create()["design_id"]
        before = files_under(self.store.root)
        cc, pi = self.spec_in()["arms"]
        cases: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
            "arms[1].options.api_key": ({"title": "x"}, {"options": {"api_key": "sk-x"}}),
            "arms[1].options.auth_token": ({"title": "x"}, {"options": {"auth_token": "x"}}),
            "arms[1].options.factory_kwargs.client_secret": (
                {"title": "x"}, {"options": {"factory_kwargs": {"client_secret": "x"}}}),
            "arms[1].options.headers.Authorization": (
                {"title": "x"}, {"options": {"headers": {"Authorization": "Bearer x"}}}),
            # A live key under an ordinary name, or anywhere else in the record.
            "arms[1].options.extra_body.note": ({"title": "x"}, {"options": {"extra_body": {"note": FAKE_KEY}}}),
            "arms[1].model": ({"title": "x"}, {"model": f"z-ai/{FAKE_KEY}"}),
            "purpose": ({"title": "x", "purpose": f"Uses {FAKE_KEY}"}, {}),
            # A live key used as an option name is in its own path: shown redacted.
            "arms[1].options.[redacted]": ({"title": "x"}, {"options": {FAKE_KEY: 1}}),
        }
        for path, (meta, change) in cases.items():
            spec_in = self.spec_in(arms=[cc, {**pi, **change}])
            attempts = {
                "create": lambda: self.store.create(self.cfg, meta, designs.normalize(self.cfg, spec_in), "user"),
                "create from input": lambda: self.store.create(
                    self.cfg, meta, designs.DesignSpecIn.model_validate(spec_in), "user"),
                "add_version": lambda: self.store.add_version(
                    self.cfg, existing, 1, meta, designs.normalize(self.cfg, spec_in), "user"),
                "draft": lambda: DraftWriter(self.store, lambda: self.cfg, CONVERSATION).create_draft(meta, spec_in),
            }
            for name, attempt in attempts.items():
                with self.subTest(path=path, call=name):
                    with self.assertRaises(ValueError) as ctx:
                        attempt()
                    message = str(ctx.exception)
                    self.assertTrue(message.startswith("Designs cannot store credentials ("), message)
                    self.assertIn(path, message)
                    self.assertTrue(message.endswith("Set keys on Setup."))
                    self.assertNotIn(FAKE_KEY, message)
        self.assertEqual(files_under(self.store.root), before)
        # A redirect is not a credential: it is saved, and validate keeps it from running.
        d = self.create("Redirect", arms=[cc, {**pi, "options": {"base_url": "http://127.0.0.1:9"}}])
        self.assertTrue(any("(base_url)" in p for p in self.problems(d["spec"])))

    def test_specs_that_are_not_plain_json_are_refused(self) -> None:
        pi = self.spec_in()["arms"][1]
        for name, value in (("nan", float("nan")), ("inf", float("inf")), ("set", {1, 2}),
                            ("date", datetime.date(2026, 10, 29))):
            with self.subTest(value=name):
                spec = self.spec()
                spec["arms"][1]["options"] = {"x": {"y": value}}
                self.assertIn("spec.arms[1].options must be plain JSON with finite numbers",
                              designs.validate(self.cfg, spec)["problems"])
                with self.assertRaises(ValueError):
                    self.store.create(self.cfg, {"title": "x"}, spec, "user")
                with self.assertRaises(ValueError) as ctx:
                    self.spec(arms=[{**pi, "options": {"x": value}}], baseline_arm=None)
                self.assertIn("Arm 'pi': options must be plain JSON with finite numbers", str(ctx.exception))
        self.assertFalse(self.store.root.exists() and any(self.store.root.iterdir()))
        # Anything that fails after the folder exists still takes the folder with it.
        with mock.patch.object(designs, "_version_doc", side_effect=TypeError("not JSON")):
            with self.assertRaises(TypeError):
                self.create()
        self.assertEqual(list(self.store.root.iterdir()), [])

    def test_half_made_and_broken_designs(self) -> None:
        (self.store.root / "d-being-made-0000").mkdir(parents=True)  # create() between mkdir and v0001.json
        with self.assertRaises(DesignNotFound) as ctx:
            self.store.meta("d-being-made-0000")
        self.assertNotIsInstance(ctx.exception, DesignUnreadable)
        design_id = self.create()["design_id"]
        (self.store.root / design_id / "v0001.json").unlink()
        self.assertEqual([(s["id"], s["status"]) for s in self.store.list()], [(design_id, "unreadable")])

    def test_a_failed_create_leaves_no_folder(self) -> None:
        with mock.patch.object(fsutil, "create_exclusive_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.create()
        self.assertEqual(list(self.store.root.iterdir()), [])

    def test_lagging_meta_is_rebuilt_from_version_files(self) -> None:
        design_id = self.create()["design_id"]
        folder = self.store.root / design_id
        (folder / "meta.json").unlink()  # a crash between the version file and meta.json
        meta = self.store.meta(design_id)
        self.assertEqual((meta["latest_version"], meta["title"]), (1, "GLM pairs"))
        doc = json.loads((folder / "v0001.json").read_text(encoding="utf-8"))
        doc.update(version=2, parent_version=1, title="Saved, then the meta write failed")
        fsutil.create_exclusive_json(folder / "v0002.json", doc)
        meta = self.store.meta(design_id)
        self.assertEqual((meta["latest_version"], meta["title"]), (2, "Saved, then the meta write failed"))
        with self.assertRaises(VersionConflict):
            self.store.add_version(self.cfg, design_id, 1, {"title": "x"}, self.spec(attempts=2), "user")
        self.store.add_version(self.cfg, design_id, 2, {"title": "v3"}, self.spec(attempts=2), "user")
        on_disk = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual((sorted(on_disk["versions"]), on_disk["latest_version"]), (["1", "2", "3"], 3))

    def test_a_locked_meta_file_never_loses_a_version(self) -> None:
        with mock.patch.object(fsutil, "atomic_write_json", return_value=False):
            d = self.create()
            self.assertEqual(d["version"], 1)
            with self.assertRaises(PermissionError):
                self.store.mark_approved(d["design_id"], 1)
        self.assertEqual(self.store.list()[0]["id"], d["design_id"])

    def test_inputs_are_checked(self) -> None:
        bad_meta = [{"title": ""}, {"title": "   "}, {"title": "x" * 81}, {"title": "x", "status": "approved"},
                    {"title": 5}, {"purpose": "no title"}, "title"]
        for meta in bad_meta:
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                self.store.create(self.cfg, meta, self.spec(), "user")  # type: ignore[arg-type]
        for author, conversation in (("admin", None), ("assistant", "../x"), ("user", 5)):
            with self.subTest(author=author), self.assertRaises(ValueError):
                self.store.create(self.cfg, {"title": "x"}, self.spec(), author, conversation)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.store.create(self.cfg, {"title": "x"}, {"tasks": []}, "user")
        self.assertEqual(self.store.list(), [])

    def test_create_takes_the_input_form_too(self) -> None:
        d = self.store.create(self.cfg, {"title": "x"}, designs.DesignSpecIn.model_validate(self.spec_in()), "user")
        self.assertEqual(d["spec"], self.spec())

    def test_designs_over_limits_are_saved_but_invalid(self) -> None:
        d = self.create(run={"budget_usd": 50})
        self.assertEqual(d["spec"]["run"]["budget_usd"], 50.0)
        self.assertTrue(any("designs.max_budget_usd" in p for p in self.problems(d["spec"])))


class DraftWriterTests(DesignTestCase):
    def writer(self) -> DraftWriter:
        return DraftWriter(self.store, lambda: self.cfg, CONVERSATION)

    def test_public_surface_is_draft_only(self) -> None:
        writer = self.writer()
        public = {n for n in dir(writer) if not n.startswith("_")}
        self.assertEqual(public, {"conversation_id", "create_draft", "add_draft_version", "list", "meta", "version"})
        for name in ("mark_approved", "archive", "unarchive", "record_run", "record_check", "create", "add_version",
                     "store", "root"):
            self.assertFalse(hasattr(writer, name), name)
        with self.assertRaises(AttributeError):
            writer.approve = lambda: None  # type: ignore[attr-defined]

    def test_drafts_are_authored_by_the_assistant(self) -> None:
        writer = self.writer()
        d = writer.create_draft({"title": "Pilot pi"}, self.spec_in())
        self.assertEqual((d["author"], d["conversation_id"], d["status"]), ("assistant", CONVERSATION, "draft"))
        d2 = writer.add_draft_version(d["design_id"], 1, {"title": "Pilot pi", "change_note": "More budget"},
                                      self.spec_in(run={"budget_usd": 4}))
        self.assertEqual((d2["version"], d2["author"]), (2, "assistant"))
        mine = self.create("Mine")
        revised = writer.add_draft_version(mine["design_id"], 1, {"title": "Mine"}, self.spec_in(attempts=2))
        self.assertEqual((revised["author"], revised["status"]), ("assistant", "draft"))
        self.assertEqual(writer.version(d["design_id"])["version"], 2)
        self.assertEqual(writer.meta(d["design_id"])["latest_version"], 2)
        self.assertEqual(len(writer.list()), 2)

    def test_draft_input_is_validated_before_anything_is_saved(self) -> None:
        writer = self.writer()
        with self.assertRaises(ValidationError):
            writer.create_draft({"title": "x"}, {**self.spec_in(), "status": "approved"})
        with self.assertRaises(ValidationError):
            writer.create_draft({"title": "x", "approved": True}, self.spec_in())
        with self.assertRaises(ValueError):
            writer.create_draft({"title": "x"}, self.spec_in(arms=[{"name": "x", "agent": "evil.module:Thing",
                                                                     "provider": "openrouter", "model": "m"}]))
        self.assertEqual(self.store.list(), [])

    def test_config_is_read_for_every_draft(self) -> None:
        getter = mock.Mock(return_value=self.cfg)
        writer = DraftWriter(self.store, getter, CONVERSATION)
        d = writer.create_draft({"title": "x"}, self.spec_in())
        writer.add_draft_version(d["design_id"], 1, {"title": "y"}, self.spec_in())
        self.assertEqual(getter.call_count, 2)

    def test_conversation_id_is_checked(self) -> None:
        # Required, not optional: a draft without its conversation loses where it came from.
        for bad in ("../other", None, 5, "", "c" * 65):
            with self.subTest(conversation_id=bad), self.assertRaises(ValueError):
                DraftWriter(self.store, lambda: self.cfg, bad)  # type: ignore[arg-type]

    def test_drafts_cannot_store_credentials(self) -> None:
        os.environ["OPENROUTER_API_KEY"] = FAKE_KEY
        writer = self.writer()
        cc, pi = self.spec_in()["arms"]
        for options in ({"llm_kwargs": {"api_key": "sk-x"}}, {"model_kwargs": {"extra": FAKE_KEY}}):
            with self.subTest(options=options), self.assertRaises(ValueError) as ctx:
                writer.create_draft({"title": "x"}, self.spec_in(arms=[cc, {**pi, "options": options}]))
            self.assertIn("Designs cannot store credentials", str(ctx.exception))
            self.assertNotIn(FAKE_KEY, str(ctx.exception))
        d = writer.create_draft({"title": "x"}, self.spec_in())
        before = files_under(self.store.root)
        with self.assertRaises(ValueError):
            writer.add_draft_version(d["design_id"], 1, {"title": "x", "rationale": f"key {FAKE_KEY}"},
                                     self.spec_in(attempts=2))
        self.assertEqual(files_under(self.store.root), before)
        self.assertEqual(len(self.store.list()), 1)

    def test_queue_status_reaches_draft_reads(self) -> None:
        design_id = self.create()["design_id"]
        writer = DraftWriter(self.store, lambda: self.cfg, CONVERSATION,
                             queue_status=lambda: {(design_id, 1): "waiting"})
        self.assertEqual(writer.version(design_id)["status"], "queued")
        self.assertEqual(writer.meta(design_id)["status"], "queued")
        self.assertEqual(writer.list()[0]["status"], "queued")

        def broken() -> dict[tuple[str, int], str]:
            raise RuntimeError("queue file busy")

        writer = DraftWriter(self.store, lambda: self.cfg, CONVERSATION, queue_status=broken)
        self.assertEqual(writer.version(design_id)["status"], "draft")


# Round trips and helpers -------------------------------------------------------


class RoundTripTests(DesignTestCase):
    def test_from_config(self) -> None:
        arms = ["cc-glm", "mini-glm"]
        spec = designs.design_from_config(self.cfg, "sample", arms)
        self.assertEqual(spec, specs.normalized_from_config(self.cfg, "sample", arms))
        enabled = designs.design_from_config(self.cfg, "sample", None)
        self.assertEqual([a["name"] for a in enabled["arms"]], ["cc-glm", "pi-glm", "mini-glm"])
        d = self.store.create(self.cfg, {"title": "From config"}, spec, "user")
        self.assertEqual(d["spec"], spec)
        ref = {"design_id": d["design_id"], "version": 1, "author": "user", "conversation_id": None,
               "queue_item": None}
        self.assertEqual(specs.spec_from_design(d["spec"], ref).spec_hash,
                         specs.spec_from_config(self.cfg, "sample", arms).spec_hash)
        self.assertEqual(self.problems(d["spec"]), [])
        self.assertEqual(designs.normalize(self.cfg, designs.to_input(spec)), spec)

    def test_profile_effort_outside_the_design_levels(self) -> None:
        # A profile may run an effort the design input cannot name; validate and to_input must agree on it.
        cfg = self.load_test_config(DESIGN_YAML.replace("  effort: medium\n", "  effort: minimal\n"), tasks=TASKS)
        spec = designs.design_from_config(cfg, "sample", ["cc-glm", "pi-glm"])
        self.assertEqual(spec["run"]["effort"], "minimal")
        self.assertEqual(designs.validate(cfg, spec)["problems"], [])
        again = designs.to_input(spec)
        self.assertNotIn("effort", again["run"])
        self.assertEqual(designs.normalize(cfg, again), spec)
        self.assertEqual(designs.to_input(self.spec())["run"]["effort"], "medium")  # a design level is kept

    def test_from_run_with_a_spec(self) -> None:
        spec = self.spec()
        run = {"id": "20261029-101500-glm-pairs-v1", "spec": spec, "spec_hash": specs.spec_hash(spec)}
        again = designs.design_from_run(run)
        self.assertEqual(again, spec)
        self.assertIsNot(again, spec)

    def test_from_a_legacy_run(self) -> None:
        spec = designs.design_from_run(copy.deepcopy(LEGACY_RUN))
        self.assertEqual(specs.structure_problems(spec), [])
        self.assertEqual((spec["dataset"], spec["tasks"], spec["attempts"], spec["baseline_arm"]),
                         ("terminal-bench@2.0", ["task-a"], 1, "cc-glm"))
        self.assertEqual(spec["run"], {"budget_usd": 5.0, "parallel_arms": 5, "n_concurrent": 1, "effort": "medium",
                                       "per_trial_cap_usd": 1.0, "max_turns": 150, "timeout_multiplier": 1.0})
        cc, strands = spec["arms"]
        self.assertEqual(cc, {"name": "cc-glm", "agent": "claude-code", "family": "claude-code",
                              "provider": "openrouter", "model": "z-ai/glm-5.3", "effort": "inherit",
                              "options": {}, "tasks": None})
        # The plan's 'arm option' kwargs are what ran; run caps and effort stay out.
        self.assertEqual(strands["options"], {"factory": "bench_strands_factory:create_agent", "version": "1.57.2"})
        self.assertEqual((strands["effort"], strands["family"]), ("omit", "strands"))
        self.assertEqual(self.problems(spec), [])
        d = self.store.create(self.cfg, {"title": "Run again"}, spec, "user")
        self.assertEqual(d["spec"], spec)

    def test_legacy_run_without_config_arms(self) -> None:
        run = copy.deepcopy(LEGACY_RUN)
        run["config"]["arms"] = []
        spec = designs.design_from_run(run)
        self.assertEqual([(a["agent"], a["model"]) for a in spec["arms"]],
                         [("claude-code", "z-ai/glm-5.3"), ("strands", "z-ai/glm-5.3")])
        for broken in ({}, {"plan": "x", "config": None, "arms": [None]}):
            with self.subTest(run=broken):
                self.assertEqual(designs.validate(self.cfg, designs.design_from_run(broken))["status"], "invalid")


class DiffTests(DesignTestCase):
    def test_paths(self) -> None:
        a = self.spec()
        b = copy.deepcopy(a)
        b["run"]["budget_usd"] = 4.0
        b["arms"][1]["options"]["max_turns"] = 50
        b["arms"].append({**b["arms"][0], "name": "cc2"})
        b["tasks"] = ["task-a", "task-c"]
        self.assertEqual(designs.diff(a, b), [
            {"path": "tasks", "from": ["task-a", "task-b"], "to": ["task-a", "task-c"]},
            {"path": "run.budget_usd", "from": 3.0, "to": 4.0},
            {"path": "arms[1].options.max_turns", "from": None, "to": 50},
            {"path": "arms[2]", "from": None, "to": b["arms"][2]},
        ])
        self.assertEqual(designs.diff(a, copy.deepcopy(a)), [])
        self.assertEqual(designs.diff(b, a)[-1], {"path": "arms[2]", "from": b["arms"][2], "to": None})

    def test_empty_exactly_when_the_hash_matches(self) -> None:
        a = self.spec()
        b = copy.deepcopy(a)
        b["run"]["parallel_arms"] = 2.0  # equal in Python, different JSON and a different hash
        self.assertNotEqual(specs.spec_hash(a), specs.spec_hash(b))
        self.assertEqual(designs.diff(a, b), [{"path": "run.parallel_arms", "from": 2, "to": 2.0}])


class HarnessAndFingerprintTests(DesignTestCase):
    def test_allowed_harnesses(self) -> None:
        self.assertEqual(designs.allowed_harnesses(self.cfg), [
            {"agent": "claude-code", "family": "claude-code", "kind": "name"},
            {"agent": "pi", "family": "pi", "kind": "name"},
            {"agent": IMPORT_PATH, "family": "mini-swe-agent", "kind": "import_path"},
            {"agent": "terminus-2", "family": "terminus-2", "kind": "name"},  # a disabled arm's agent
            {"agent": "strands", "family": "strands", "kind": "name"},       # designs.extra_harnesses
        ])

    def test_listed_import_path(self) -> None:
        yaml_text = DESIGN_YAML + "  import_paths:\n    pkg.mod:Agent: {family: custom}\n"
        cfg = self.load_test_config(yaml_text, tasks=TASKS)
        self.assertIn({"agent": "pkg.mod:Agent", "family": "custom", "kind": "import_path"},
                      designs.allowed_harnesses(cfg))
        spec = designs.normalize(cfg, self.spec_in(arms=[{"name": "c", "agent": "pkg.mod:Agent",
                                                         "provider": "openrouter", "model": "m"}], baseline_arm=None))
        self.assertEqual(spec["arms"][0]["family"], "custom")

    def test_config_fingerprint(self) -> None:
        fp = designs.config_fingerprint(self.cfg)
        self.assertRegex(fp, r"^sha256:[0-9a-f]{64}$")
        same = dataclasses.replace(self.cfg, effective=dict(reversed(list(self.cfg.effective.items()))))
        self.assertEqual(designs.config_fingerprint(same), fp)
        effective = copy.deepcopy(self.cfg.effective)
        effective["run"]["budget_usd"] = 6
        self.assertNotEqual(designs.config_fingerprint(dataclasses.replace(self.cfg, effective=effective)), fp)
        prices = {"openrouter": {"z-ai/glm-5.3": settings.Price(1.0, 2.0, None)}}
        repriced = dataclasses.replace(self.cfg, pricing=prices)
        self.assertNotEqual(designs.config_fingerprint(repriced), fp)
        # A check's estimate quotes multiples and plans from published figures: either changing makes it stale.
        restated = dataclasses.replace(self.cfg, reference={**(self.cfg.reference or {}), "arms": {"pi": {}}})
        self.assertNotEqual(designs.config_fingerprint(restated), fp)
        with mock.patch.object(designs.stats, "DIRECTION", "baseline_over_harness"):
            self.assertNotEqual(designs.config_fingerprint(self.cfg), fp)


if __name__ == "__main__":
    import unittest

    unittest.main()
