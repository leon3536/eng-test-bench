"""Planning: plan() and plan_spec() for config runs and run designs.

The parity tests hold the planner to what it did before run specs existed.
frozen_plan below is the pre-change planner, copied from bench/runs.py with
only names and comments changed; GOLDEN is the output of that pre-change code
for GOLDEN_YAML, captured before the change, so the frozen copy is pinned
too. For the test config and for both shipped profiles, today's plan() must
equal the frozen output except for the documented changes:

- additive keys: source, label, spec_hash, warnings, peak_containers, and per
  arm family, kind, tasks, options_fingerprint, reserve_note;
- the preview command is `harbor --% run ...` with the jobs folder shown as
  jobs\\RUN_ID (finding 13), and import-path agents use --agent-import-path;
- an unpriced model is a problem (finding 18);
- arm options are applied in name order, because the normalized spec stores
  them sorted (Harbor reads --ak flags into a mapping, so order is cosmetic).

Schemas come from tests/fixtures/schemas, the dataset snapshot is fixed, and
Docker is never asked anything.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import subprocess
from pathlib import Path
from typing import Any
from unittest import mock

from tests.support import BenchTestCase, fixture_schema

from bench import checks, datasets, keys, runs, schemas, settings, spec
from bench.harnesses import is_import_path
from bench.settings import EFFORT_INHERIT, EFFORT_OMIT, ROOT, ConfigError

SNAPSHOT = ROOT / "datasets" / "terminal-bench@2.0" / "terminal-bench"
ADDED_TOP = ("source", "label", "spec_hash", "warnings", "peak_containers")
ADDED_ARM = ("family", "kind", "tasks", "options_fingerprint", "reserve_note")


# The pre-change planner, frozen -------------------------------------------------


def _frozen_resolve_effort(cfg: settings.Config, arm: settings.Arm,
                           schema: dict[str, Any]) -> tuple[str | None, str]:
    value = cfg.run.effort if arm.effort == EFFORT_INHERIT else arm.effort
    levels = schemas.effort_levels(schema)
    if value == EFFORT_OMIT:
        return None, "harness default"
    if levels is None:
        if arm.effort != EFFORT_INHERIT:
            raise ConfigError(f"{arm.agent} does not accept an effort setting")
        return None, "not supported (model default)"
    if levels and value not in levels:
        raise ConfigError(f"effort '{value}' is not one of: {', '.join(levels)}")
    return value, value


def _frozen_agent_kwargs(cfg: settings.Config, arm: settings.Arm,
                         schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    props = schema.get("properties", {})
    kwargs: dict[str, Any] = {}
    source: dict[str, str] = {}
    effort, _ = _frozen_resolve_effort(cfg, arm, schema)
    if effort is not None:
        kwargs["reasoning_effort"], source["reasoning_effort"] = effort, "effort"
    if cfg.run.per_trial_cap_usd:
        for name in schemas.DOLLAR_CAPS:
            if name in props and not schemas.cap_misprices(arm, name):
                kwargs[name] = schemas.coerce(schema, name, f"{cfg.run.per_trial_cap_usd:g}")
                source[name] = "run cap"
    if cfg.run.max_turns:
        for name in schemas.TURN_CAPS:
            if name in props:
                kwargs[name] = schemas.coerce(schema, name, cfg.run.max_turns)
                source[name] = "run cap"
    for name, value in arm.options.items():
        kwargs[name] = schemas.coerce(schema, name, value)
        source[name] = "arm option"
    return kwargs, source


def _frozen_build_command(cfg: settings.Config, arm: settings.Arm, tasks: list[str], mode: settings.Mode,
                          job_root: Path, kwargs: dict[str, Any], dataset_path: Path,
                          dry_run: bool = False) -> list[str]:
    cmd = [
        settings.harbor_executable(), "run",
        "--path", str(dataset_path),
        "--agent", arm.agent,
        "--model", cfg.harbor_model(arm),
        "--n-attempts", str(mode.n_attempts),
        "--n-concurrent", str(cfg.run.n_concurrent),
        "--jobs-dir", str(job_root),
        "--job-name", arm.name,
        "--timeout-multiplier", f"{cfg.run.timeout_multiplier:g}",
        "--max-retries", "0",
        "--quiet",
    ]
    for task in tasks:
        cmd += ["--include-task-name", task]
    for name, value in kwargs.items():
        cmd += ["--ak", f"{name}={json.dumps(value)}"]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def frozen_plan(cfg: settings.Config, mode_name: str, arm_names: list[str]) -> dict[str, Any]:
    if mode_name not in cfg.modes:
        raise ConfigError(f"Unknown mode '{mode_name}'")
    if not arm_names:
        raise ConfigError("Pick at least one arm")
    mode = cfg.modes[mode_name]
    tasks = cfg.tasks[: mode.n_tasks]
    if not tasks:
        raise ConfigError("No task sample yet. Run: python -m bench.sample_tasks")
    per_arm = len(tasks) * mode.n_attempts
    arms, problems, uncapped = [], [], []
    worst_case = 0.0
    snapshot = datasets.path(cfg.dataset)
    display_path = snapshot.relative_to(ROOT) if snapshot else Path("datasets") / cfg.dataset
    for name in arm_names:
        arm = cfg.arm(name)
        provider = cfg.providers[arm.provider]
        entry: dict[str, Any] = {
            "name": arm.name, "agent": arm.agent, "provider": provider.name,
            "model": cfg.harbor_model(arm), "trials": per_arm,
            "effort": None, "kwargs": [], "command": "", "problems": [], "warnings": [],
        }
        if not provider.enabled:
            entry["problems"].append(f"provider '{provider.name}' is turned off in config")
        if not keys.get_secret(provider.env_key):
            entry["problems"].append(f"{provider.env_key} is not set")
        try:
            schema = schemas.agent_schema(arm.agent)
            kwargs, source = _frozen_agent_kwargs(cfg, arm, schema)
            entry["effort"] = _frozen_resolve_effort(cfg, arm, schema)[1]
            entry["kwargs"] = [{"name": k, "value": v, "source": source[k]} for k, v in kwargs.items()]
            rejected = schemas.model_warnings(arm)
            entry["warnings"] += [f"{k}: {rejected[k]}" for k in kwargs if k in rejected]
            entry["warnings"] += [f"{k}: {schemas.CAUTIONS[k]}" for k in kwargs
                                  if k in schemas.CAUTIONS and k != "version"]
            entry["warnings"] += [f"{k}: {schemas.MISPRICED_CAP}" for k in schemas.DOLLAR_CAPS
                                  if schemas.cap_misprices(arm, k) and cfg.run.per_trial_cap_usd]
            caps = [float(kwargs[k]) for k in schemas.DOLLAR_CAPS if k in kwargs and float(kwargs[k]) > 0]
            if caps:
                worst_case += per_arm * min(caps)
            else:
                uncapped.append(arm.name)
            display = _frozen_build_command(cfg, arm, tasks, mode, Path("jobs") / "<run-id>", kwargs, display_path)
            display[0] = "harbor"
            entry["command"] = subprocess.list2cmdline(display)
        except (ConfigError, ValueError, RuntimeError) as exc:
            entry["problems"].append(str(exc))
        problems += [f"{arm.name}: {p}" for p in entry["problems"]]
        arms.append(entry)
    return {
        "mode": mode_name, "tasks": tasks, "n_attempts": mode.n_attempts,
        "dataset": cfg.dataset, "dataset_snapshot": datasets.info(cfg.dataset),
        "trials": per_arm * len(arms), "budget_usd": cfg.run.budget_usd,
        "worst_case_capped_usd": worst_case, "uncapped_arms": uncapped,
        "arms": arms, "problems": problems,
    }


# The golden: frozen_plan's output for GOLDEN_YAML, captured from the pre-change
# bench/runs.py itself (datasets.path -> SNAPSHOT, datasets.info -> None).

GOLDEN_YAML = """\
dataset: terminal-bench@2.0
providers:
  anthropic: {env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}
  openrouter: {env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}
  fireworks: {env_key: FIREWORKS_AI_API_KEY, model_prefix: fireworks_ai/accounts/fireworks/models/, enabled: false}
modes:
  pilot: {n_tasks: 1}
  sample: {n_tasks: 2, n_attempts: 2}
run:
  effort: medium
  n_concurrent: 2
  parallel_arms: 2
  budget_usd: 20
  per_trial_cap_usd: 2.5
  max_turns: 100
baseline_arm: claude-code
arms:
  - {name: claude-code, agent: claude-code, provider: anthropic, model: claude-opus-5-5}
  - {name: claude-code-glm, agent: claude-code, provider: openrouter, model: z-ai/glm-5.3}
  - name: mini
    agent: bench.agents.mini_swe_agent:MiniSweAgentWithMcp
    family: mini-swe-agent
    provider: openrouter
    model: deepseek/deepseek-v4.1-flash
  - {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5, effort: omit}
  - name: terminus-2
    agent: terminus-2
    provider: anthropic
    model: claude-opus-5-5
    options: {interleaved_thinking: true, temperature: 0.2}
  - name: strands
    agent: strands
    family: strands
    provider: anthropic
    model: claude-opus-5-5
    effort: omit
    options: {factory_path: harnesses/strands_factory, factory: "bench_strands_factory:create_agent"}
  - {name: pi-fireworks, agent: pi, provider: fireworks, model: glm-5p3}
"""
GOLDEN_TASKS = ("task-a", "task-b")

GOLDEN: dict[str, Any] = json.loads(r'''
{
 "pilot": {
  "mode": "pilot",
  "tasks": ["task-a"],
  "n_attempts": 1,
  "dataset": "terminal-bench@2.0",
  "dataset_snapshot": null,
  "trials": 7,
  "budget_usd": 20.0,
  "worst_case_capped_usd": 5.0,
  "uncapped_arms": ["claude-code-glm", "pi", "terminus-2", "strands", "pi-fireworks"],
  "problems": ["pi-fireworks: provider 'fireworks' is turned off in config", "pi-fireworks: FIREWORKS_AI_API_KEY is not set"],
  "arms": [
   {"name": "claude-code", "agent": "claude-code", "provider": "anthropic", "model": "anthropic/claude-opus-5-5", "trials": 1, "effort": "medium", "kwargs": [{"name": "reasoning_effort", "value": "medium", "source": "effort"}, {"name": "max_budget_usd", "value": "2.5", "source": "run cap"}, {"name": "max_turns", "value": 100, "source": "run cap"}], "command": "harbor run --path datasets\\terminal-bench@2.0\\terminal-bench --agent claude-code --model anthropic/claude-opus-5-5 --n-attempts 1 --n-concurrent 2 --jobs-dir jobs\\<run-id> --job-name claude-code --timeout-multiplier 1 --max-retries 0 --quiet --include-task-name task-a --ak reasoning_effort=\\\"medium\\\" --ak max_budget_usd=\\\"2.5\\\" --ak max_turns=100", "problems": [], "warnings": []},
   {"name": "claude-code-glm", "agent": "claude-code", "provider": "openrouter", "model": "openrouter/z-ai/glm-5.3", "trials": 1, "effort": "medium", "kwargs": [{"name": "reasoning_effort", "value": "medium", "source": "effort"}, {"name": "max_turns", "value": 100, "source": "run cap"}], "command": "harbor run --path datasets\\terminal-bench@2.0\\terminal-bench --agent claude-code --model openrouter/z-ai/glm-5.3 --n-attempts 1 --n-concurrent 2 --jobs-dir jobs\\<run-id> --job-name claude-code-glm --timeout-multiplier 1 --max-retries 0 --quiet --include-task-name task-a --ak reasoning_effort=\\\"medium\\\" --ak max_turns=100", "problems": [], "warnings": ["max_budget_usd: Not applied: this harness prices other providers' models at its own rates, so its cap would stop trials early. The turn cap and run budget still apply."]},
   {"name": "mini", "agent": "bench.agents.mini_swe_agent:MiniSweAgentWithMcp", "provider": "openrouter", "model": "openrouter/deepseek/deepseek-v4.1-flash", "trials": 1, "effort": "medium", "kwargs": [{"name": "reasoning_effort", "value": "medium", "source": "effort"}, {"name": "cost_limit", "value": "2.5", "source": "run cap"}], "command": "harbor run --path datasets\\terminal-bench@2.0\\terminal-bench --agent bench.agents.mini_swe_agent:MiniSweAgentWithMcp --model openrouter/deepseek/deepseek-v4.1-flash --n-attempts 1 --n-concurrent 2 --jobs-dir jobs\\<run-id> --job-name mini --timeout-multiplier 1 --max-retries 0 --quiet --include-task-name task-a --ak reasoning_effort=\\\"medium\\\" --ak cost_limit=\\\"2.5\\\"", "problems": [], "warnings": []},
   {"name": "pi", "agent": "pi", "provider": "anthropic", "model": "anthropic/claude-opus-5-5", "trials": 1, "effort": "harness default", "kwargs": [{"name": "max_turns", "value": 100, "source": "run cap"}], "command": "harbor run --path datasets\\terminal-bench@2.0\\terminal-bench --agent pi --model anthropic/claude-opus-5-5 --n-attempts 1 --n-concurrent 2 --jobs-dir jobs\\<run-id> --job-name pi --timeout-multiplier 1 --max-retries 0 --quiet --include-task-name task-a --ak max_turns=100", "problems": [], "warnings": []},
   {"name": "terminus-2", "agent": "terminus-2", "provider": "anthropic", "model": "anthropic/claude-opus-5-5", "trials": 1, "effort": "medium", "kwargs": [{"name": "reasoning_effort", "value": "medium", "source": "effort"}, {"name": "max_turns", "value": 100, "source": "run cap"}, {"name": "interleaved_thinking", "value": true, "source": "arm option"}, {"name": "temperature", "value": 0.2, "source": "arm option"}], "command": "harbor run --path datasets\\terminal-bench@2.0\\terminal-bench --agent terminus-2 --model anthropic/claude-opus-5-5 --n-attempts 1 --n-concurrent 2 --jobs-dir jobs\\<run-id> --job-name terminus-2 --timeout-multiplier 1 --max-retries 0 --quiet --include-task-name task-a --ak reasoning_effort=\\\"medium\\\" --ak max_turns=100 --ak interleaved_thinking=true --ak temperature=0.2", "problems": [], "warnings": ["temperature: Current Claude models reject sampling parameters with a 400 error."]},
   {"name": "strands", "agent": "strands", "provider": "anthropic", "model": "anthropic/claude-opus-5-5", "trials": 1, "effort": "harness default", "kwargs": [{"name": "factory_path", "value": "harnesses/strands_factory", "source": "arm option"}, {"name": "factory", "value": "bench_strands_factory:create_agent", "source": "arm option"}], "command": "harbor run --path datasets\\terminal-bench@2.0\\terminal-bench --agent strands --model anthropic/claude-opus-5-5 --n-attempts 1 --n-concurrent 2 --jobs-dir jobs\\<run-id> --job-name strands --timeout-multiplier 1 --max-retries 0 --quiet --include-task-name task-a --ak factory_path=\\\"harnesses/strands_factory\\\" --ak factory=\\\"bench_strands_factory:create_agent\\\"", "problems": [], "warnings": []},
   {"name": "pi-fireworks", "agent": "pi", "provider": "fireworks", "model": "fireworks_ai/accounts/fireworks/models/glm-5p3", "trials": 1, "effort": "medium", "kwargs": [{"name": "reasoning_effort", "value": "medium", "source": "effort"}, {"name": "max_turns", "value": 100, "source": "run cap"}], "command": "harbor run --path datasets\\terminal-bench@2.0\\terminal-bench --agent pi --model fireworks_ai/accounts/fireworks/models/glm-5p3 --n-attempts 1 --n-concurrent 2 --jobs-dir jobs\\<run-id> --job-name pi-fireworks --timeout-multiplier 1 --max-retries 0 --quiet --include-task-name task-a --ak reasoning_effort=\\\"medium\\\" --ak max_turns=100", "problems": ["provider 'fireworks' is turned off in config", "FIREWORKS_AI_API_KEY is not set"], "warnings": []}
  ]
 },
 "sample": {
  "mode": "sample",
  "tasks": ["task-a", "task-b"],
  "n_attempts": 2,
  "dataset": "terminal-bench@2.0",
  "dataset_snapshot": null,
  "trials": 28,
  "budget_usd": 20.0,
  "worst_case_capped_usd": 20.0,
  "uncapped_arms": ["claude-code-glm", "pi", "terminus-2", "strands", "pi-fireworks"],
  "problems": ["pi-fireworks: provider 'fireworks' is turned off in config", "pi-fireworks: FIREWORKS_AI_API_KEY is not set"],
  "arms": [
   {"name": "claude-code", "agent": "claude-code", "provider": "anthropic", "model": "anthropic/claude-opus-5-5", "trials": 4, "effort": "medium", "kwargs": [{"name": "reasoning_effort", "value": "medium", "source": "effort"}, {"name": "max_budget_usd", "value": "2.5", "source": "run cap"}, {"name": "max_turns", "value": 100, "source": "run cap"}], "command": "harbor run --path datasets\\terminal-bench@2.0\\terminal-bench --agent claude-code --model anthropic/claude-opus-5-5 --n-attempts 2 --n-concurrent 2 --jobs-dir jobs\\<run-id> --job-name claude-code --timeout-multiplier 1 --max-retries 0 --quiet --include-task-name task-a --include-task-name task-b --ak reasoning_effort=\\\"medium\\\" --ak max_budget_usd=\\\"2.5\\\" --ak max_turns=100", "problems": [], "warnings": []},
   {"name": "claude-code-glm", "agent": "claude-code", "provider": "openrouter", "model": "openrouter/z-ai/glm-5.3", "trials": 4, "effort": "medium", "kwargs": [{"name": "reasoning_effort", "value": "medium", "source": "effort"}, {"name": "max_turns", "value": 100, "source": "run cap"}], "command": "harbor run --path datasets\\terminal-bench@2.0\\terminal-bench --agent claude-code --model openrouter/z-ai/glm-5.3 --n-attempts 2 --n-concurrent 2 --jobs-dir jobs\\<run-id> --job-name claude-code-glm --timeout-multiplier 1 --max-retries 0 --quiet --include-task-name task-a --include-task-name task-b --ak reasoning_effort=\\\"medium\\\" --ak max_turns=100", "problems": [], "warnings": ["max_budget_usd: Not applied: this harness prices other providers' models at its own rates, so its cap would stop trials early. The turn cap and run budget still apply."]},
   {"name": "mini", "agent": "bench.agents.mini_swe_agent:MiniSweAgentWithMcp", "provider": "openrouter", "model": "openrouter/deepseek/deepseek-v4.1-flash", "trials": 4, "effort": "medium", "kwargs": [{"name": "reasoning_effort", "value": "medium", "source": "effort"}, {"name": "cost_limit", "value": "2.5", "source": "run cap"}], "command": "harbor run --path datasets\\terminal-bench@2.0\\terminal-bench --agent bench.agents.mini_swe_agent:MiniSweAgentWithMcp --model openrouter/deepseek/deepseek-v4.1-flash --n-attempts 2 --n-concurrent 2 --jobs-dir jobs\\<run-id> --job-name mini --timeout-multiplier 1 --max-retries 0 --quiet --include-task-name task-a --include-task-name task-b --ak reasoning_effort=\\\"medium\\\" --ak cost_limit=\\\"2.5\\\"", "problems": [], "warnings": []},
   {"name": "pi", "agent": "pi", "provider": "anthropic", "model": "anthropic/claude-opus-5-5", "trials": 4, "effort": "harness default", "kwargs": [{"name": "max_turns", "value": 100, "source": "run cap"}], "command": "harbor run --path datasets\\terminal-bench@2.0\\terminal-bench --agent pi --model anthropic/claude-opus-5-5 --n-attempts 2 --n-concurrent 2 --jobs-dir jobs\\<run-id> --job-name pi --timeout-multiplier 1 --max-retries 0 --quiet --include-task-name task-a --include-task-name task-b --ak max_turns=100", "problems": [], "warnings": []},
   {"name": "terminus-2", "agent": "terminus-2", "provider": "anthropic", "model": "anthropic/claude-opus-5-5", "trials": 4, "effort": "medium", "kwargs": [{"name": "reasoning_effort", "value": "medium", "source": "effort"}, {"name": "max_turns", "value": 100, "source": "run cap"}, {"name": "interleaved_thinking", "value": true, "source": "arm option"}, {"name": "temperature", "value": 0.2, "source": "arm option"}], "command": "harbor run --path datasets\\terminal-bench@2.0\\terminal-bench --agent terminus-2 --model anthropic/claude-opus-5-5 --n-attempts 2 --n-concurrent 2 --jobs-dir jobs\\<run-id> --job-name terminus-2 --timeout-multiplier 1 --max-retries 0 --quiet --include-task-name task-a --include-task-name task-b --ak reasoning_effort=\\\"medium\\\" --ak max_turns=100 --ak interleaved_thinking=true --ak temperature=0.2", "problems": [], "warnings": ["temperature: Current Claude models reject sampling parameters with a 400 error."]},
   {"name": "strands", "agent": "strands", "provider": "anthropic", "model": "anthropic/claude-opus-5-5", "trials": 4, "effort": "harness default", "kwargs": [{"name": "factory_path", "value": "harnesses/strands_factory", "source": "arm option"}, {"name": "factory", "value": "bench_strands_factory:create_agent", "source": "arm option"}], "command": "harbor run --path datasets\\terminal-bench@2.0\\terminal-bench --agent strands --model anthropic/claude-opus-5-5 --n-attempts 2 --n-concurrent 2 --jobs-dir jobs\\<run-id> --job-name strands --timeout-multiplier 1 --max-retries 0 --quiet --include-task-name task-a --include-task-name task-b --ak factory_path=\\\"harnesses/strands_factory\\\" --ak factory=\\\"bench_strands_factory:create_agent\\\"", "problems": [], "warnings": []},
   {"name": "pi-fireworks", "agent": "pi", "provider": "fireworks", "model": "fireworks_ai/accounts/fireworks/models/glm-5p3", "trials": 4, "effort": "medium", "kwargs": [{"name": "reasoning_effort", "value": "medium", "source": "effort"}, {"name": "max_turns", "value": 100, "source": "run cap"}], "command": "harbor run --path datasets\\terminal-bench@2.0\\terminal-bench --agent pi --model fireworks_ai/accounts/fireworks/models/glm-5p3 --n-attempts 2 --n-concurrent 2 --jobs-dir jobs\\<run-id> --job-name pi-fireworks --timeout-multiplier 1 --max-retries 0 --quiet --include-task-name task-a --include-task-name task-b --ak reasoning_effort=\\\"medium\\\" --ak max_turns=100", "problems": ["provider 'fireworks' is turned off in config", "FIREWORKS_AI_API_KEY is not set"], "warnings": []}
  ]
 }
}
''')


# Helpers --------------------------------------------------------------------------


def sorted_options(cfg: settings.Config) -> settings.Config:
    """The config with every arm's options in name order, as the normalized
    spec holds them."""
    arms = [dataclasses.replace(a, options=dict(sorted(a.options.items()))) for a in cfg.arms]
    return dataclasses.replace(cfg, arms=arms)


def expected_from_frozen(cfg: settings.Config, old: dict[str, Any]) -> dict[str, Any]:
    """The frozen output with the documented changes applied."""
    new = copy.deepcopy(old)
    jobs = str(Path("jobs") / "RUN_ID")
    problems = []
    for entry in new["arms"]:
        arm = cfg.arm(entry["name"])
        command = entry["command"]
        if command:
            assert command.startswith("harbor run "), command
            command = "harbor --% " + command[len("harbor "):]
            command = command.replace(str(Path("jobs") / "<run-id>"), jobs)
            if is_import_path(arm.agent):
                command = command.replace(f" --agent {arm.agent} ", f" --agent-import-path {arm.agent} ")
            entry["command"] = command
        if cfg.price(arm.provider, arm.model) is None:
            # After the provider and key checks, before the schema checks.
            first = [p for p in entry["problems"]
                     if p.endswith("is turned off in config") or p.endswith("is not set")]
            rest = entry["problems"][len(first):]
            entry["problems"] = first + [
                f"no price for '{arm.model}' under '{arm.provider}' in config/pricing.yaml; "
                "the run budget could not count this arm's spend"] + rest
        problems += [f"{entry['name']}: {p}" for p in entry["problems"]]
    new["problems"] = problems
    return new


def without_additions(plan: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in plan.items() if k not in ADDED_TOP}
    out["arms"] = [{k: v for k, v in a.items() if k not in ADDED_ARM} for a in plan["arms"]]
    return out


class PlanTestCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.isolate_data_dir()
        for target, name, value in (
                (datasets, "path", lambda dataset: SNAPSHOT),
                (datasets, "info", lambda dataset: None),
                (datasets, "task_meta", lambda dataset: {}),
                (checks, "docker_capacity", lambda: None),
                (runs, "_long_paths_enabled", lambda: True)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def golden_config(self) -> settings.Config:
        return self.load_test_config(GOLDEN_YAML, tasks=GOLDEN_TASKS)

    def config(self, arms: str, run: str = "", modes: str = "pilot: {n_tasks: 1}") -> settings.Config:
        text = (
            "dataset: terminal-bench@2.0\n"
            "providers:\n"
            "  anthropic: {env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}\n"
            "  openrouter: {env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}\n"
            f"modes:\n  {modes}\n"
            "run:\n  effort: medium\n  budget_usd: 20\n" + "".join(f"  {line}\n" for line in run.splitlines() if line)
            + "arms:\n" + "".join(f"  {line}\n" for line in arms.splitlines() if line)
        )
        return self.load_test_config(text, tasks=GOLDEN_TASKS)


# Parity ------------------------------------------------------------------------


class GoldenParityTests(PlanTestCase):
    def test_frozen_copy_reproduces_the_captured_golden(self) -> None:
        cfg = self.golden_config()
        for mode in cfg.modes:
            with self.subTest(mode=mode):
                self.assertEqual(frozen_plan(cfg, mode, [a.name for a in cfg.arms]), GOLDEN[mode])

    def test_plan_matches_the_golden_except_documented_changes(self) -> None:
        cfg = self.golden_config()
        names = [a.name for a in cfg.arms]
        for mode in cfg.modes:
            with self.subTest(mode=mode):
                new = runs.plan(cfg, mode, names)
                expected = expected_from_frozen(cfg, frozen_plan(sorted_options(cfg), mode, names))
                self.assertEqual(without_additions(new), expected)
                self.assertEqual(list(new)[0], "mode")

    def test_unpriced_is_the_only_new_problem_in_the_golden(self) -> None:
        cfg = self.golden_config()
        new = runs.plan(cfg, "pilot", [a.name for a in cfg.arms])
        added = [p for p in new["problems"] if p not in GOLDEN["pilot"]["problems"]]
        self.assertEqual(added, ["pi-fireworks: no price for 'glm-5p3' under 'fireworks' in config/pricing.yaml; "
                                 "the run budget could not count this arm's spend"])

    def test_shipped_profiles_match_the_frozen_planner(self) -> None:
        for profile in ("bench", "dev"):
            cfg = self.load_shipped_config(profile)
            arms = [a.name for a in cfg.arms if a.enabled]
            for mode in cfg.modes:
                with self.subTest(profile=profile, mode=mode):
                    new = runs.plan(cfg, mode, arms)
                    expected = expected_from_frozen(cfg, frozen_plan(sorted_options(cfg), mode, arms))
                    self.assertEqual(without_additions(new), expected)
                    self.assertEqual(new["problems"], [], "the shipped profiles plan cleanly in tests")
                    # The additive keys, for the record.
                    run_spec = spec.spec_from_config(cfg, mode, arms)
                    self.assertEqual((new["source"], new["label"], new["spec_hash"]),
                                     ("config", mode, run_spec.spec_hash))
                    self.assertEqual(new["peak_containers"],
                                     min(cfg.run.parallel_arms, len(arms)) * cfg.run.n_concurrent)
                    for entry, arm in zip(new["arms"], (cfg.arm(n) for n in arms)):
                        self.assertEqual(entry["family"], arm.family)
                        self.assertEqual(entry["kind"], arm.kind)
                        self.assertEqual(entry["tasks"], cfg.modes[mode].n_tasks)
                        self.assertRegex(entry["options_fingerprint"], r"^[0-9a-f]{12}$")
                        self.assertTrue(entry["reserve_note"])

    def test_shipped_mini_swe_agent_arms_run_by_import_path(self) -> None:
        for profile in ("bench", "dev"):
            cfg = self.load_shipped_config(profile)
            mini = [a for a in cfg.arms if a.family == "mini-swe-agent"]
            self.assertTrue(mini)
            p = runs.plan(cfg, "pilot", [a.name for a in mini])
            for entry in p["arms"]:
                self.assertEqual(entry["kind"], "import_path")
                self.assertIn(f" --agent-import-path {entry['agent']} ", entry["command"])
                self.assertNotIn(" --agent bench.", entry["command"])


# Planning rules -------------------------------------------------------------------


class PlanRuleTests(PlanTestCase):
    def test_preview_command_pastes_into_powershell(self) -> None:
        cfg = self.config("- {name: cc, agent: claude-code, provider: anthropic, model: claude-opus-5-5}",
                          run="per_trial_cap_usd: 15")
        command = runs.plan(cfg, "pilot", ["cc"])["arms"][0]["command"]
        self.assertTrue(command.startswith("harbor --% run --path "), command)
        self.assertIn(f"--jobs-dir {Path('jobs') / 'RUN_ID'} ", command)
        self.assertNotIn("<", command)
        self.assertIn(r'--ak max_budget_usd=\"15\"', command)

    def test_a_non_scalar_option_value_is_an_arm_problem(self) -> None:
        cfg = self.config("- {name: cc, agent: claude-code, provider: anthropic, model: claude-opus-5-5, "
                          "options: {max_turns: [1]}}")
        p = runs.plan(cfg, "pilot", ["cc"])
        self.assertEqual(len(p["problems"]), 1, p["problems"])
        self.assertTrue(p["problems"][0].startswith("cc: "), p["problems"])

    def test_import_path_arm_uses_agent_import_path(self) -> None:
        cfg = self.config("- name: mini\n"
                          "  agent: bench.agents.mini_swe_agent:MiniSweAgentWithMcp\n"
                          "  family: mini-swe-agent\n"
                          "  provider: openrouter\n"
                          "  model: deepseek/deepseek-v4.1-flash")
        run_spec = spec.spec_from_config(cfg, "pilot", ["mini"])
        arm = run_spec.arms[0]
        cmd = runs.build_command(cfg, run_spec, arm, Path("jobs") / "x", {}, Path("datasets") / "d")
        self.assertEqual(cmd[cmd.index("--agent-import-path") + 1], "bench.agents.mini_swe_agent:MiniSweAgentWithMcp")
        self.assertNotIn("--agent", cmd)
        entry = runs.plan(cfg, "pilot", ["mini"])["arms"][0]
        self.assertEqual((entry["kind"], entry["family"]), ("import_path", "mini-swe-agent"))
        self.assertEqual(entry["problems"], [])

    def test_named_agent_uses_agent(self) -> None:
        cfg = self.config("- {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}")
        run_spec = spec.spec_from_config(cfg, "pilot", ["pi"])
        cmd = runs.build_command(cfg, run_spec, run_spec.arms[0], Path("j"), {}, Path("d"), dry_run=True)
        self.assertEqual(cmd[cmd.index("--agent") + 1], "pi")
        self.assertNotIn("--agent-import-path", cmd)
        self.assertEqual(cmd[-1], "--dry-run")

    def test_import_path_outside_designs_import_paths_is_a_problem(self) -> None:
        cfg = self.config("- {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}")
        normalized = spec.normalized_from_config(cfg, "pilot", ["pi"])
        normalized["arms"][0]["agent"] = "bench.agents.other:SomeAgent"
        with mock.patch.object(schemas, "agent_schema", lambda agent: fixture_schema("pi")):
            p = runs.plan_spec(cfg, spec.spec_from_design(normalized, None))
        self.assertIn("pi: 'bench.agents.other:SomeAgent' is not an allowed harness import path; "
                      "list it under designs.import_paths in the config", p["problems"])
        self.assertIn("--agent-import-path bench.agents.other:SomeAgent", p["arms"][0]["command"])

    def test_none_option_lifts_the_run_cap(self) -> None:
        cfg = self.config("- name: cc\n"
                          "  agent: claude-code\n"
                          "  provider: anthropic\n"
                          "  model: claude-opus-5-5\n"
                          "  options: {max_budget_usd: null}",
                          run="per_trial_cap_usd: 15\nmax_turns: 50")
        p = runs.plan(cfg, "pilot", ["cc"])
        entry = p["arms"][0]
        self.assertEqual(entry["problems"], [])
        self.assertNotIn("max_budget_usd", [k["name"] for k in entry["kwargs"]])
        self.assertIn("max_turns", [k["name"] for k in entry["kwargs"]])
        self.assertEqual(p["uncapped_arms"], ["cc"])
        self.assertEqual(p["worst_case_capped_usd"], 0.0)
        self.assertNotIn("max_budget_usd", entry["command"])

    def test_agent_kwargs_none_removes_value_and_source(self) -> None:
        cfg = self.config("- {name: cc, agent: claude-code, provider: anthropic, model: claude-opus-5-5,"
                          " options: {max_budget_usd: null, max_turns: 20}}",
                          run="per_trial_cap_usd: 15\nmax_turns: 50")
        arm = spec.spec_from_config(cfg, "pilot", ["cc"]).arms[0]
        kwargs, source = runs.agent_kwargs(cfg.run, arm, fixture_schema("claude-code"))
        self.assertEqual(kwargs, {"reasoning_effort": "medium", "max_turns": 20})
        self.assertEqual(source, {"reasoning_effort": "effort", "max_turns": "arm option"})

    def test_unpriced_model_is_a_problem(self) -> None:
        cfg = self.config("- {name: glm-flash, agent: pi, provider: openrouter, model: z-ai/glm-5.3-flash}\n"
                          "- {name: glm, agent: pi, provider: openrouter, model: z-ai/glm-5.3}")
        p = runs.plan(cfg, "pilot", ["glm-flash", "glm"])
        self.assertEqual(p["problems"], [
            "glm-flash: no price for 'z-ai/glm-5.3-flash' under 'openrouter' in config/pricing.yaml; "
            "the run budget could not count this arm's spend"])
        self.assertEqual(p["arms"][1]["problems"], [])

    def test_containers_above_max_are_a_problem(self) -> None:
        arms = "\n".join(f"- {{name: pi{i}, agent: pi, provider: anthropic, model: claude-opus-5-5}}"
                         for i in range(5))
        cfg = self.config(arms, run="parallel_arms: 5\nn_concurrent: 4")
        p = runs.plan(cfg, "pilot", [f"pi{i}" for i in range(5)])
        self.assertEqual(p["peak_containers"], 20)
        self.assertIn("Up to 20 task containers would run at once (parallel_arms x n_concurrent); "
                      "designs.max_containers is 16", p["problems"])
        self.assertTrue(all(a["problems"] == [] for a in p["arms"]))

    def test_containers_above_warn_are_a_warning(self) -> None:
        arms = "\n".join(f"- {{name: pi{i}, agent: pi, provider: anthropic, model: claude-opus-5-5}}"
                         for i in range(3))
        cfg = self.config(arms, run="parallel_arms: 3\nn_concurrent: 3")
        p = runs.plan(cfg, "pilot", [f"pi{i}" for i in range(3)])
        self.assertEqual(p["peak_containers"], 9)
        self.assertEqual(p["problems"], [])
        self.assertEqual(len(p["warnings"]), 1)
        self.assertTrue(p["warnings"][0].startswith("Up to 9 task containers would run at once. Above 8,"))

    def test_peak_counts_only_arms_that_can_run_together(self) -> None:
        cfg = self.config("- {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}",
                          run="parallel_arms: 5\nn_concurrent: 4")
        self.assertEqual(runs.plan(cfg, "pilot", ["pi"])["peak_containers"], 4)

    def test_docker_memory_warning(self) -> None:
        cfg = self.config("- {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}",
                          run="n_concurrent: 2")
        meta = {"task-a": {"memory_gb": 4.0}, "task-b": {"memory_gb": None}}
        with mock.patch.object(datasets, "task_meta", lambda dataset: meta):
            with mock.patch.object(checks, "docker_capacity", lambda: {"ncpu": 8, "mem_gb": 8.0}):
                warned = runs.plan(cfg, "pilot", ["pi"])["warnings"]
            with mock.patch.object(checks, "docker_capacity", lambda: {"ncpu": 8, "mem_gb": 16.0}):
                roomy = runs.plan(cfg, "pilot", ["pi"])["warnings"]
            with mock.patch.object(checks, "docker_capacity", lambda: None):
                unknown = runs.plan(cfg, "pilot", ["pi"])["warnings"]
        self.assertEqual(len(warned), 1)
        self.assertIn("Up to 2 containers of up to 4 GB each could need more than 80% of the 8 GB Docker has",
                      warned[0])
        self.assertEqual((roomy, unknown), ([], []))

    def test_windows_path_length_warning(self) -> None:
        cfg = self.config("- {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}")
        deep = Path("C:/") / ("x" * 120) / "jobs"
        with mock.patch.object(runs, "_WINDOWS", True), mock.patch.object(runs, "JOBS_DIR", deep):
            with mock.patch.object(runs, "_long_paths_enabled", lambda: False):
                warned = runs.plan(cfg, "pilot", ["pi"])["warnings"]
            with mock.patch.object(runs, "_long_paths_enabled", lambda: True):
                enabled = runs.plan(cfg, "pilot", ["pi"])["warnings"]
        with mock.patch.object(runs, "_WINDOWS", False), mock.patch.object(runs, "JOBS_DIR", deep):
            elsewhere = runs.plan(cfg, "pilot", ["pi"])["warnings"]
        self.assertEqual(len(warned), 1)
        self.assertIn("past Windows' 260-character path limit", warned[0])
        self.assertEqual((enabled, elsewhere), ([], []))

    def test_options_fingerprint_covers_only_arm_options(self) -> None:
        cfg = self.config(
            "- {name: t1, agent: terminus-2, provider: anthropic, model: claude-opus-5-5,"
            " options: {interleaved_thinking: true}}\n"
            "- {name: t2, agent: terminus-2, provider: openrouter, model: z-ai/glm-5.3,"
            " options: {interleaved_thinking: true}}\n"
            "- {name: t3, agent: terminus-2, provider: anthropic, model: claude-opus-5-5,"
            " options: {interleaved_thinking: false}}\n"
            "- {name: t4, agent: terminus-2, provider: anthropic, model: claude-opus-5-5}",
            run="max_turns: 50")
        p = {a["name"]: a for a in runs.plan(cfg, "pilot", ["t1", "t2", "t3", "t4"])["arms"]}
        self.assertEqual(p["t1"]["options_fingerprint"], p["t2"]["options_fingerprint"])
        self.assertNotEqual(p["t1"]["options_fingerprint"], p["t3"]["options_fingerprint"])
        self.assertNotEqual(p["t1"]["options_fingerprint"], p["t4"]["options_fingerprint"])
        self.assertRegex(p["t1"]["options_fingerprint"], r"^[0-9a-f]{12}$")
        # A run cap is not an arm option, so changing it keeps the fingerprint.
        capped = self.config("- {name: t1, agent: terminus-2, provider: anthropic, model: claude-opus-5-5,"
                             " options: {interleaved_thinking: true}}", run="max_turns: 300")
        self.assertEqual(runs.plan(capped, "pilot", ["t1"])["arms"][0]["options_fingerprint"],
                         p["t1"]["options_fingerprint"])

    def test_design_arm_with_its_own_tasks(self) -> None:
        cfg = self.config("- {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}\n"
                          "- {name: t2, agent: terminus-2, provider: anthropic, model: claude-opus-5-5}",
                          modes="sample: {n_tasks: 2, n_attempts: 3}")
        normalized = spec.normalized_from_config(cfg, "sample", ["pi", "t2"])
        normalized["arms"][1]["tasks"] = ["task-b"]
        p = runs.plan_spec(cfg, spec.spec_from_design(normalized, {"design_id": "d-subset", "version": 2}))
        pi, t2 = p["arms"]
        self.assertEqual((pi["tasks"], pi["trials"], t2["tasks"], t2["trials"]), (2, 6, 1, 3))
        self.assertEqual(p["trials"], 9)
        self.assertEqual((p["source"], p["label"]), ("design", "subset-v2"))
        self.assertIn("--include-task-name task-b", t2["command"])
        self.assertNotIn("task-a", t2["command"])
        self.assertIn("--n-attempts 3", t2["command"])

    def test_unknown_provider_in_a_design_is_a_problem(self) -> None:
        cfg = self.config("- {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}")
        normalized = spec.normalized_from_config(cfg, "pilot", ["pi"])
        normalized["arms"][0]["provider"] = "nowhere"
        p = runs.plan_spec(cfg, spec.spec_from_design(normalized, None))
        self.assertEqual(p["problems"], ["pi: unknown provider 'nowhere'"])
        self.assertEqual(p["arms"][0]["command"], "")

    def test_plan_keeps_the_old_errors(self) -> None:
        cfg = self.config("- {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}")
        for args, message in (((cfg, "nope", ["pi"]), "Unknown mode 'nope'"),
                              ((cfg, "pilot", []), "Pick at least one arm"),
                              ((cfg, "pilot", ["ghost"]), "No arm named 'ghost'")):
            with self.subTest(message=message), self.assertRaisesRegex(ConfigError, re.escape(message)):
                runs.plan(*args)

    def test_resolve_effort_takes_run_settings(self) -> None:
        cfg = self.config("- {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}\n"
                          "- {name: st, agent: strands, provider: anthropic, model: claude-opus-5-5, effort: high}")
        pi, st = spec.spec_from_config(cfg, "pilot", ["pi", "st"]).arms
        self.assertEqual(runs.resolve_effort(cfg.run, pi, fixture_schema("pi")), ("medium", "medium"))
        no_effort = {"properties": {}}
        with self.assertRaisesRegex(ConfigError, "strands does not accept an effort setting"):
            runs.resolve_effort(cfg.run, st, no_effort)
        self.assertEqual(runs.resolve_effort(cfg.run, pi, no_effort), (None, "not supported (model default)"))

    def test_mispriced_cap_warning_is_kept(self) -> None:
        cfg = self.config("- {name: cc, agent: claude-code, provider: openrouter, model: z-ai/glm-5.3}",
                          run="per_trial_cap_usd: 1")
        entry = runs.plan(cfg, "pilot", ["cc"])["arms"][0]
        self.assertEqual(entry["warnings"], [f"max_budget_usd: {schemas.MISPRICED_CAP}"])
        self.assertNotIn("max_budget_usd", [k["name"] for k in entry["kwargs"]])

    def test_reserve_note_names_the_cap(self) -> None:
        cfg = self.config("- {name: cc, agent: claude-code, provider: anthropic, model: claude-opus-5-5}\n"
                          "- {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}",
                          run="per_trial_cap_usd: 15")
        cc, pi = runs.plan(cfg, "pilot", ["cc", "pi"])["arms"]
        self.assertIn("at most $15 (its dollar cap)", cc["reserve_note"])
        self.assertIn("no dollar cap", pi["reserve_note"])
