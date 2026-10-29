"""Run designs: ad-hoc run specs that people or the assistant draft and a
person approves.

A design is a run that does not depend on the config's arms or modes. It has
its own tasks, attempts, run settings, budget and baseline, and its own arms:
any allowed harness, provider and model, with options and an optional task
subset. Designs are how runs beyond the LAUNCH tab's modes are made, by hand
or by the run-design assistant.

Every version is a write-once file, and only its normalized spec is hashed.
So what a person approves is exactly what runs: the queue admits a version by
its spec_hash, and any change is a new version that starts as a draft. The
assistant gets a DraftWriter, which can save drafts and nothing else.

Each profile keeps its designs under DATA_DIR/designs/<id>/. They are small,
readable records of intent, so they are tracked by git.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
import secrets
import threading
import unicodedata
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import fsutil, settings
from .harnesses import is_import_path
from .settings import Config, ConfigError, RunSettings
from .spec import (
    ARM_KEYS,
    RUN_KEYS,
    normalized_from_config,
    plain_json,
    run_dict,
    spec_from_design,
    spec_hash,
    structure_problems,
)

RUN_EFFORTS = ("low", "medium", "high", "xhigh", "max", "omit")
MAX_TASKS = 100
MAX_PARALLEL_ARMS = 12
MAX_CONCURRENT = 8
MAX_TIMEOUT_MULTIPLIER = 4.0
AUTHORS = ("user", "assistant")

# Lowercase only: NTFS is case-insensitive, and container cleanup matches
# Docker compose project names (always lowercase) by prefix.
ARM_NAME = re.compile(r"[a-z][a-z0-9-]{0,31}")
DESIGN_ID = re.compile(r"d-[a-z0-9][a-z0-9-]{0,30}")
_CONVERSATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_VERSION_FILE = re.compile(r"v(\d{4,})\.json")

# Options that would send an arm's traffic (and its key) somewhere else, or
# carry a credential into a design file. Checked at any depth, because
# llm_kwargs, model_kwargs and factory_kwargs pass nested keys straight on,
# in LiteLLM's, OpenAI's or Strands' spelling (baseURL, apiKey, headers).
# "per_token" is LiteLLM's price vocabulary (input_cost_per_token), not a token.
_REDIRECT_OPTION = re.compile(r"(^|_)(base_?url|api_?base)$", re.IGNORECASE)
_CREDENTIAL_OPTION = re.compile(r"(^|[-_]|api)key$|(?<!per_)token$|secret$|password$"
                                r"|^(proxy[-_]?)?authorization$|^cookie$", re.IGNORECASE)
_DENIED_OPTION = re.compile(f"{_REDIRECT_OPTION.pattern}|{_CREDENTIAL_OPTION.pattern}", re.IGNORECASE)

# Options that make Harbor read a host file or folder: strands copies
# factory_path into the trial's mounted logs folder, mini-swe-agent reads
# config_file on the host and hands its text to the trial. A design could
# point one at .env, so only the values a config arm already uses are allowed.
_HOST_FILE_OPTION = re.compile(r"(?i:(^|[-_])(paths?|files?|dirs?|directory|folder))$"
                               r"|[a-z0-9](Paths?|Files?|Dirs?|Directory|Folder)$")
# Top-level options that read a host file when their value is text: Harbor's
# installed agents treat a str `config` as a host path (Claude Code uploads
# its JSON as settings.json) and a mapping as inline JSON.
_HOST_PATH_IF_TEXT = frozenset({"config"})

_BUSY = "The design's meta.json is busy (another program has it open). Try again."


class VersionConflict(RuntimeError):
    """The design changed since the caller read it, or cannot change now."""


class DesignNotFound(KeyError):
    def __str__(self) -> str:
        # KeyError would print the message with quotes around it.
        return str(self.args[0]) if self.args else "Design not found"


class DesignUnreadable(DesignNotFound):
    """A design's files exist but could not be read."""


# Input models ------------------------------------------------------------------
# Strict, so YAML's or a model's `false` can never become an effort and "5"
# never a budget. Floats still accept ints.


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ArmIn(_Strict):
    name: str
    agent: str
    provider: str
    model: str
    effort: str = "inherit"
    options: dict[str, Any] = Field(default_factory=dict)
    tasks: list[str] | None = None


class RunIn(_Strict):
    # Omitted fields come from the profile's run settings at normalize time
    # (model_fields_set); upper limits depend on config, so validate checks them.
    budget_usd: float = Field(gt=0)
    parallel_arms: int | None = Field(default=None, ge=1)
    n_concurrent: int | None = Field(default=None, ge=1)
    effort: Literal["low", "medium", "high", "xhigh", "max", "omit"] | None = None
    per_trial_cap_usd: float | None = Field(default=None, gt=0)
    max_turns: int | None = Field(default=None, ge=1)
    timeout_multiplier: float | None = Field(default=None, gt=0)

    @field_validator("parallel_arms", "n_concurrent", "effort", "timeout_multiplier", mode="before")
    @classmethod
    def _not_null(cls, value: Any) -> Any:
        # For the two caps null means "none"; for these it would mean nothing.
        if value is None:
            raise ValueError("leave this field out to use the profile default; null is not a value here")
        return value


class DesignSpecIn(_Strict):
    tasks: list[str]
    attempts: int = Field(default=1, ge=1)
    baseline_arm: str | None = None
    run: RunIn
    arms: list[ArmIn]


class DesignMetaIn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=80)
    purpose: str = Field(default="", max_length=1000)
    rationale: str = Field(default="", max_length=1500)
    change_note: str = Field(default="", max_length=200)


# The JSON Schema the assistant's tools carry, inlined with no $ref. A test
# keeps its property names and required lists in step with DesignSpecIn.
DESIGN_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tasks", "run", "arms"],
    "properties": {
        "tasks": {
            "type": "array", "items": {"type": "string"},
            "description": "Task names from list_tasks, in seeded-sample order. Every arm runs them unless it "
                           "narrows them. 1-100, unique.",
        },
        "attempts": {
            "type": "integer",
            "description": "Attempts per task per arm, 1 to design_limits.max_attempts. Default 1. Prefer more "
                           "tasks over more attempts.",
        },
        "baseline_arm": {
            "type": ["string", "null"],
            "description": "Name of the arm the others are compared with, or null to compare with the ledger's "
                           "baseline cell on the same model.",
        },
        "run": {
            "type": "object",
            "additionalProperties": False,
            "required": ["budget_usd"],
            "properties": {
                "budget_usd": {
                    "type": "number",
                    "description": "Hard stop for this run's total spend, checked after every trial. At most "
                                   "design_limits.max_budget_usd. Use about p90 x 1.15 from estimate_cost.",
                },
                "parallel_arms": {
                    "type": "integer",
                    "description": "Arms (Harbor jobs) running at once, 1-12. Omit for the profile default.",
                },
                "n_concurrent": {
                    "type": "integer",
                    "description": "Trials at once inside each arm, 1-8. Omit for the profile default. "
                                   "parallel_arms x n_concurrent is the peak Docker container count.",
                },
                "effort": {
                    "type": "string", "enum": list(RUN_EFFORTS),
                    "description": "Effort for arms set to inherit. Omit for the profile default.",
                },
                "per_trial_cap_usd": {
                    "type": ["number", "null"],
                    "description": "Dollar cap passed to harnesses that accept one and price this provider "
                                   "correctly. null = none. Omit for the profile default.",
                },
                "max_turns": {
                    "type": ["integer", "null"],
                    "description": "Turn cap for harnesses that accept one. null = none. Omit for the profile "
                                   "default.",
                },
                "timeout_multiplier": {
                    "type": "number",
                    "description": "Task timeout multiplier, normally 1.0. Omit for the profile default.",
                },
            },
        },
        "arms": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "agent", "provider", "model"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique; lowercase letters, digits and '-', starting with a letter, at "
                                       "most 32 characters, e.g. pi-glm.",
                    },
                    "agent": {
                        "type": "string",
                        "description": "A harness from get_bench_context.harnesses: a Harbor agent name or an "
                                       "allowed import path. Its family (the label results pool and compare "
                                       "under) is derived, never given: the agent name itself, or the family "
                                       "listed for that import path.",
                    },
                    "provider": {"type": "string", "description": "A provider from get_bench_context.providers."},
                    "model": {
                        "type": "string",
                        "description": "Model id without the provider prefix, as in "
                                       "get_bench_context.priced_models, e.g. z-ai/glm-5.3.",
                    },
                    "effort": {
                        "type": "string",
                        "description": "inherit (default), omit, or a level from describe_harness.effort_levels. "
                                       "Harnesses with effort_levels null accept only inherit or omit.",
                    },
                    "options": {
                        "type": "object",
                        "description": "Harness options by name, typed as describe_harness reports. Default {}.",
                    },
                    "tasks": {
                        "type": ["array", "null"], "items": {"type": "string"},
                        "description": "Optional subset of the design's tasks for this arm, to complete "
                                       "half-pairs. null = all design tasks.",
                    },
                },
            },
        },
    },
}


# Harnesses ---------------------------------------------------------------------


def allowed_harnesses(cfg: Config) -> list[dict[str, str]]:
    """The agents a design may use, each with the family its trials pool under.

    Names: every config arm's agent plus designs.extra_harnesses, family =
    the name. Import paths: only designs.import_paths (which load_config
    fills with the config arms' own import paths), because an import path is
    code that Harbor imports and runs.
    """
    out: dict[str, dict[str, str]] = {}

    def add(agent: str, family: str) -> None:
        if agent and agent not in out:
            kind = "import_path" if is_import_path(agent) else "name"
            out[agent] = {"agent": agent, "family": family, "kind": kind}

    for arm in cfg.arms:
        if is_import_path(arm.agent):
            add(arm.agent, cfg.designs.import_paths.get(arm.agent) or arm.family or arm.agent)
        else:
            add(arm.agent, arm.agent)
    for name in cfg.designs.extra_harnesses:
        if not is_import_path(name):
            add(name, name)
    for path, family in cfg.designs.import_paths.items():
        add(path, family)
    return list(out.values())


def _families(cfg: Config) -> dict[str, str]:
    return {h["agent"]: h["family"] for h in allowed_harnesses(cfg)}


# Normalizing -------------------------------------------------------------------


def normalize(cfg: Config, spec_in: DesignSpecIn | Mapping[str, Any]) -> dict[str, Any]:
    """The NormalizedSpec for an input spec: every field explicit, so the hash
    pins the run even if the profile's defaults change later.

    Raises ValueError (pydantic's ValidationError is one) for input that does
    not parse, and for an agent whose family cannot be derived.
    """
    if not isinstance(spec_in, DesignSpecIn):
        spec_in = DesignSpecIn.model_validate(spec_in)
    families = _families(cfg)
    given = spec_in.run.model_fields_set
    run = run_dict(RunSettings(**{k: getattr(spec_in.run if k in given else cfg.run, k) for k in RUN_KEYS}))
    arms = []
    for arm in spec_in.arms:
        family = families.get(arm.agent)
        if family is None:
            allowed = ", ".join(sorted(families)) or "none"
            raise ValueError(f"Arm '{arm.name}': {arm.agent} is not a harness designs may use (allowed: {allowed})")
        if not plain_json(arm.options):
            # Any passes NaN and Python objects through, and a version file keeps them for good.
            raise ValueError(f"Arm '{arm.name}': options must be plain JSON with finite numbers")
        arms.append({
            "name": arm.name,
            "agent": arm.agent,
            "family": family,
            "provider": arm.provider,
            "model": _without_prefix(cfg, arm.provider, arm.model),
            "effort": arm.effort,
            "options": {k: copy.deepcopy(arm.options[k]) for k in sorted(arm.options)},
            "tasks": None if arm.tasks is None else _in_order(arm.tasks, spec_in.tasks),
        })
    return {
        "dataset": cfg.dataset,
        "tasks": list(spec_in.tasks),
        "attempts": spec_in.attempts,
        "baseline_arm": spec_in.baseline_arm,
        "run": run,
        "arms": arms,
    }


def _without_prefix(cfg: Config, provider: str, model: str) -> str:
    # A pasted Harbor model id would otherwise get the prefix twice.
    p = cfg.providers.get(provider)
    if p is not None and p.model_prefix and model.startswith(p.model_prefix):
        return model[len(p.model_prefix):]
    return model


def _in_order(subset: list[str], tasks: list[str]) -> list[str]:
    """An arm's tasks in the design's order. Names outside the design keep
    their place at the end, for validate to report."""
    position = {}
    for i, task in enumerate(tasks):
        position.setdefault(task, i)
    inside = sorted((t for t in subset if t in position), key=position.__getitem__)
    return inside + [t for t in subset if t not in position]


def to_input(spec: Mapping[str, Any]) -> dict[str, Any]:
    """A NormalizedSpec in DesignSpecIn form (no dataset, no derived family),
    for editors and for handing a stored or suggested spec back as input."""
    run = dict(spec["run"])
    if run.get("effort") not in RUN_EFFORTS:
        # A profile may run an effort the design input cannot name (validate
        # allows the profile's own); left out, normalize fills it back in.
        run.pop("effort", None)
    return {
        "tasks": list(spec["tasks"]),
        "attempts": spec["attempts"],
        "baseline_arm": spec["baseline_arm"],
        "run": run,
        "arms": [{k: copy.deepcopy(a[k]) for k in ARM_KEYS if k != "family"} for a in spec["arms"]],
    }


def design_from_config(cfg: Config, mode_name: str, arm_names: list[str] | None) -> dict[str, Any]:
    """A LAUNCH-tab selection as a design spec; None picks the enabled arms."""
    if arm_names is None:
        arm_names = [a.name for a in cfg.arms if a.enabled]
    return normalized_from_config(cfg, mode_name, arm_names)


def design_from_run(run: Mapping[str, Any]) -> dict[str, Any]:
    """The spec a past run executed, for "Run again". Runs from before specs
    were recorded are rebuilt from their config snapshot and plan."""
    spec = run.get("spec")
    if isinstance(spec, Mapping) and spec:
        return copy.deepcopy(dict(spec))
    return _legacy_spec(run)


def _legacy_spec(run: Mapping[str, Any]) -> dict[str, Any]:
    config = _mapping(run.get("config"))
    plan = _mapping(run.get("plan"))
    cfg_run = _mapping(config.get("run"))
    providers = _mapping(config.get("providers"))
    by_name = {a.get("name"): a for a in config.get("arms") or [] if isinstance(a, Mapping)}
    entries = [a for a in plan.get("arms") or [] if isinstance(a, Mapping)]
    if not entries:
        entries = [{"name": a.get("name")} for a in run.get("arms") or [] if isinstance(a, Mapping)]
    arms = []
    for entry in entries:
        base = _mapping(by_name.get(entry.get("name")))
        agent = base.get("agent") or entry.get("agent") or ""
        provider = base.get("provider") or entry.get("provider") or ""
        prefix = str(_mapping(providers.get(provider)).get("model_prefix") or "")
        model = base.get("model") or entry.get("model") or ""
        if prefix and isinstance(model, str) and model.startswith(prefix):
            model = model[len(prefix):]
        if isinstance(entry.get("kwargs"), list):
            # What actually ran: the plan recorded each kwarg with its source,
            # and only the arm's own options belong in the spec.
            options = {k["name"]: k.get("value") for k in entry["kwargs"]
                       if isinstance(k, Mapping) and k.get("source") == "arm option" and isinstance(k.get("name"), str)}
        else:
            options = dict(_mapping(base.get("options")))
        effort = base.get("effort")
        arms.append({
            "name": entry.get("name"),
            "agent": agent,
            "family": base.get("family") or agent,
            "provider": provider,
            "model": model,
            "effort": effort if isinstance(effort, str) and effort else settings.ARM_DEFAULTS["effort"],
            "options": {k: copy.deepcopy(options[k]) for k in sorted(options)},
            "tasks": None,
        })
    names = [a["name"] for a in arms]
    modes = _mapping(config.get("modes"))
    attempts = plan.get("n_attempts") or _mapping(modes.get(run.get("mode"))).get("n_attempts") or 1
    defaults = settings.RUN_DEFAULTS
    baseline = config.get("baseline_arm")
    return {
        "dataset": config.get("dataset") or plan.get("dataset") or _mapping(run.get("dataset")).get("dataset") or "",
        "tasks": list(run.get("tasks") or plan.get("tasks") or []),
        "attempts": _as(int, attempts),
        "baseline_arm": baseline if baseline in names else None,
        "run": {
            "budget_usd": _as(float, run.get("budget_usd") or cfg_run.get("budget_usd")),
            "parallel_arms": _as(int, cfg_run.get("parallel_arms", defaults["parallel_arms"])),
            "n_concurrent": _as(int, cfg_run.get("n_concurrent", defaults["n_concurrent"])),
            "effort": cfg_run.get("effort") or defaults["effort"],
            "per_trial_cap_usd": _as(float, cfg_run.get("per_trial_cap_usd", defaults["per_trial_cap_usd"])),
            "max_turns": _as(int, cfg_run.get("max_turns", defaults["max_turns"])),
            "timeout_multiplier": _as(float, cfg_run.get("timeout_multiplier", defaults["timeout_multiplier"])),
        },
        "arms": arms,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as(kind: type, value: Any) -> Any:
    """kind(value), or value unchanged when it does not convert, so validate
    reports the bad field instead of this failing."""
    if value is None or isinstance(value, bool):
        return value
    try:
        return kind(value)
    except (TypeError, ValueError, OverflowError):
        return value


# Validation --------------------------------------------------------------------


def validate(cfg: Config, spec: Any) -> dict[str, Any]:
    """Every check a design must pass before it can be queued, plus the
    planner's own. Returns {status, problems, warnings, plan}; status is
    "invalid" when there is any problem. Never raises for a bad spec."""
    problems: list[str] = []
    warnings: list[str] = []
    plan = None
    try:
        problems += structure_problems(spec)
        if not problems and _check(cfg, spec, problems, warnings):
            plan = _plan(cfg, spec, problems, warnings)
    except Exception as exc:  # a check that breaks is a problem to show, not an error page
        problems.append(f"The design could not be checked: {type(exc).__name__}: {exc}")
    return {"status": "invalid" if problems else "valid", "problems": problems, "warnings": warnings,
            "plan": plan}


def peak_containers(spec: Mapping[str, Any]) -> int:
    """The most trial containers that can run at once: the parallel_arms
    busiest arms, each running min(n_concurrent, its trials)."""
    run = spec["run"]
    per_arm = sorted((min(run["n_concurrent"], len(_arm_tasks(spec, a)) * spec["attempts"])
                      for a in spec["arms"]), reverse=True)
    return sum(per_arm[: max(0, run["parallel_arms"])])


def trial_count(spec: Mapping[str, Any]) -> int:
    return sum(len(_arm_tasks(spec, a)) for a in spec["arms"]) * spec["attempts"]


def _arm_tasks(spec: Mapping[str, Any], arm: Mapping[str, Any]) -> list[str]:
    return list(spec["tasks"] if arm["tasks"] is None else arm["tasks"])


def _check(cfg: Config, spec: Mapping[str, Any], problems: list[str], warnings: list[str]) -> bool:
    """Design checks on a structurally sound spec. Returns whether planning is
    safe: never for an agent outside the allowlist, because planning asks
    Harbor for its schema, which imports the agent's code."""
    plannable = True
    limits = cfg.designs
    tasks = list(spec["tasks"])
    arms = list(spec["arms"])
    run = spec["run"]

    if not arms:
        problems.append("Add at least one arm")
        plannable = False
    elif len(arms) > limits.max_arms:
        problems.append(f"{len(arms)} arms is more than designs.max_arms ({limits.max_arms})")
    if not tasks:
        problems.append("Choose at least one task")
        plannable = False
    elif len(tasks) > MAX_TASKS:
        problems.append(f"{len(tasks)} tasks is more than the {MAX_TASKS} a design may run")
    repeated = _duplicates(tasks)
    if repeated:
        problems.append(f"Tasks listed more than once: {_short(repeated)}")
    if not 1 <= spec["attempts"] <= limits.max_attempts:
        problems.append(f"attempts must be from 1 to {limits.max_attempts} (designs.max_attempts)")

    budget = run["budget_usd"]
    if budget <= 0:
        problems.append("run.budget_usd must be greater than zero")
    elif budget > limits.max_budget_usd:
        problems.append(f"The budget ${budget:.2f} is above designs.max_budget_usd (${limits.max_budget_usd:.2f})")
    if not 1 <= run["parallel_arms"] <= MAX_PARALLEL_ARMS:
        problems.append(f"run.parallel_arms must be from 1 to {MAX_PARALLEL_ARMS}")
    if not 1 <= run["n_concurrent"] <= MAX_CONCURRENT:
        problems.append(f"run.n_concurrent must be from 1 to {MAX_CONCURRENT}")
    if run["effort"] not in RUN_EFFORTS and run["effort"] != cfg.run.effort:
        problems.append(f"run.effort must be one of {', '.join(RUN_EFFORTS)}")
    if run["per_trial_cap_usd"] is not None and run["per_trial_cap_usd"] <= 0:
        problems.append("run.per_trial_cap_usd must be greater than zero, or null for none")
    if run["max_turns"] is not None and run["max_turns"] < 1:
        problems.append("run.max_turns must be 1 or more, or null for none")
    if not 0 < run["timeout_multiplier"] <= MAX_TIMEOUT_MULTIPLIER:
        problems.append(f"run.timeout_multiplier must be above 0 and at most {MAX_TIMEOUT_MULTIPLIER:g}")

    if spec["dataset"] != cfg.dataset:
        problems.append(f"The design is for {spec['dataset']}, but this profile runs {cfg.dataset}")
    known, where = _known_tasks(cfg)
    unknown: list[str] = []
    if known is None:
        warnings.append("Task names could not be checked: there is no dataset snapshot or task sample yet")
    else:
        unknown = [t for t in tasks if t not in known]
        if unknown:
            problems.append(f"Tasks {where}: {_short(unknown)}")

    families = _families(cfg)
    for arm in arms:
        label = f"Arm '{arm['name']}'"
        name = arm["name"]
        if not (ARM_NAME.fullmatch(name) and fsutil.is_safe_segment(name)):
            problems.append(f"{label}: a name uses lowercase letters, digits and '-', starts with a letter, "
                            "has at most 32 characters and is not a Windows device name (con, nul, ...)")
            plannable = False
        family = families.get(arm["agent"])
        if family is None:
            where_to = "designs.import_paths" if is_import_path(arm["agent"]) else "designs.extra_harnesses"
            problems.append(f"{label}: {arm['agent']} is not a harness designs may use. "
                            f"A person can allow it under {where_to} in the profile's config")
            plannable = False
        elif arm["family"] != family:
            warnings.append(f"{label}: its family '{arm['family']}' differs from '{family}', the family this "
                            f"profile gives {arm['agent']}, so its trials pool apart")
        provider = cfg.providers.get(arm["provider"])
        if provider is None:
            problems.append(f"{label}: there is no provider '{arm['provider']}' in this profile")
            plannable = False
        else:
            try:
                # The rule load_config applies to config arms, so a design
                # cannot call one model while pricing and recording another.
                settings.check_model_spec(name, arm["options"], provider.model_prefix + arm["model"])
            except ConfigError as exc:
                problems.append(str(exc))
        if arm["tasks"] is not None:
            _check_arm_tasks(label, list(arm["tasks"]), tasks, problems)
        denied = _denied_options(arm["options"])
        if denied:
            problems.append(f"{label}: designs cannot set options that redirect model traffic or carry "
                            f"credentials ({', '.join(denied)})")
        host_files = _host_file_options(cfg, arm)
        if host_files:
            problems.append(f"{label}: designs cannot point a harness at host files ({', '.join(host_files)}); "
                            "set it on a config arm instead")

    repeated = _duplicates([a["name"] for a in arms])
    if repeated:
        problems.append(f"Arm names must be unique: {_short(repeated)}")
    baseline = spec["baseline_arm"]
    if baseline is not None and baseline not in {a["name"] for a in arms}:
        problems.append(f"baseline_arm '{baseline}' is not one of the design's arms")

    if arms and tasks:
        peak = peak_containers(spec)
        if peak > limits.max_containers:
            problems.append(f"Up to {peak} task containers would run at once (parallel_arms x n_concurrent); "
                            f"designs.max_containers is {limits.max_containers}")
        elif peak > limits.warn_containers:
            warnings.append(f"Up to {peak} task containers would run at once. Above {limits.warn_containers}, "
                            "Docker Desktop may run short of memory and fail trials as infrastructure errors")
    _design_warnings(cfg, spec, warnings, unknown=set(unknown))
    return plannable


def _check_arm_tasks(label: str, subset: list[str], tasks: list[str], problems: list[str]) -> None:
    if not subset:
        problems.append(f"{label}: an empty task list runs nothing; use null for all of the design's tasks")
        return
    outside = [t for t in subset if t not in tasks]
    if outside:
        problems.append(f"{label}: tasks that are not in the design's task list: {_short(outside)}")
    repeated = _duplicates(subset)
    if repeated:
        problems.append(f"{label}: tasks listed more than once: {_short(repeated)}")
    inside = [t for t in subset if t in tasks]
    if inside != sorted(inside, key=tasks.index):
        problems.append(f"{label}: list its tasks in the design's task order")


def _design_warnings(cfg: Config, spec: Mapping[str, Any], warnings: list[str],
                     unknown: set[str] | frozenset[str] = frozenset()) -> None:
    arms = list(spec["arms"])
    tasks = list(spec["tasks"])
    cells: dict[tuple[str, ...], list[str]] = {}
    for arm in arms:
        key = (arm["family"], arm["provider"], arm["model"], arm["effort"], fsutil.canonical_json(arm["options"]))
        cells.setdefault(key, []).append(arm["name"])
    for names in cells.values():
        if len(names) > 1:
            warnings.append(f"Arms {', '.join(names)} run the same configuration (family, provider, model, effort "
                            "and options), so they measure one cell twice")
    models = list(dict.fromkeys(f"{a['provider']}/{a['model']}" for a in arms))
    if len(models) > 1:
        warnings.append(f"Arms use {len(models)} models ({', '.join(models)}). Pairs form only within one model "
                        "and provider, so each comparison needs its baseline on the same model")
    for arm in arms:
        if arm["tasks"] is not None:
            warnings.append(f"Arm '{arm['name']}' runs {len(arm['tasks'])} of the design's {len(tasks)} tasks")
    position = {t: i for i, t in enumerate(cfg.tasks)}
    seeded = [position[t] for t in tasks if t in position]
    if seeded != sorted(seeded):
        warnings.append("Tasks are not in seeded-sample order. Keeping that order pairs new trials with earlier "
                        "runs and keeps each block balanced by difficulty")
    outside = [t for t in tasks if t not in position and t not in unknown]  # unknown ones are already problems
    if position and outside:
        warnings.append(f"{len(outside)} task(s) are outside the seeded sample ({_short(outside)}), so they pair "
                        "only with runs that chose them too")


def _plan(cfg: Config, spec: Mapping[str, Any], problems: list[str], warnings: list[str]) -> dict[str, Any] | None:
    try:
        # Imported here: runs imports most of the bench, and only planning needs it.
        from . import runs
        plan = runs.plan_spec(cfg, spec_from_design(spec, None))
    except Exception as exc:
        problems.append(f"The run could not be planned: {exc}")
        return None
    for p in plan.get("problems") or []:
        if p not in problems:
            problems.append(p)
    for w in plan.get("warnings") or []:
        if w not in warnings:
            warnings.append(w)
    for arm in plan.get("arms") or []:
        for w in arm.get("warnings") or []:
            text = f"{arm.get('name')}: {w}"
            if text not in warnings and w not in warnings:
                warnings.append(text)
    return plan


def _known_tasks(cfg: Config) -> tuple[set[str] | None, str]:
    try:
        from . import datasets
        meta = datasets.task_meta(cfg.dataset)
    except Exception:  # no snapshot yet, or an old datasets module: fall back to the sample
        meta = None
    if meta:
        return set(meta), f"not in the {cfg.dataset} snapshot"
    if cfg.tasks:
        return set(cfg.tasks), "not in the seeded sample (no dataset snapshot yet)"
    return None, ""


def _option_entries(options: Any, prefix: str = "") -> Iterator[tuple[str, str, Any]]:
    """(path, name, value) for every option at any depth, with paths such as
    'factory_kwargs.model_spec' or 'llm_kwargs.headers[0].Authorization'."""
    if isinstance(options, Mapping):
        for key, value in options.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, str(key), value
            yield from _option_entries(value, path)
    elif isinstance(options, (list, tuple)):
        for i, value in enumerate(options):
            yield from _option_entries(value, f"{prefix}[{i}]")


def _denied_options(options: Any, pattern: re.Pattern[str] = _DENIED_OPTION) -> list[str]:
    return [path for path, name, _ in _option_entries(options) if pattern.search(name)]


def _host_file_options(cfg: Config, arm: Mapping[str, Any]) -> list[str]:
    """Host-file options whose value no config arm with this agent sets at
    the same path. A person wrote the config's; a design may only reuse them."""
    allowed: dict[str, set[str]] = {}
    for config_arm in cfg.arms:
        if config_arm.agent == arm["agent"]:
            for path, _, value in _option_entries(config_arm.options):
                allowed.setdefault(path, set()).add(_json_text(value))
    return [path for path, name, value in _option_entries(arm["options"])
            if (_HOST_FILE_OPTION.search(name)
                or (path == name and name.lower() in _HOST_PATH_IF_TEXT and isinstance(value, str)))
            and value is not None and _json_text(value) not in allowed.get(path, ())]


def _json_text(value: Any) -> str:
    # Canonical JSON rather than ==, under which 1 and True would count as the same value.
    try:
        return fsutil.canonical_json(value)
    except (TypeError, ValueError):  # a YAML date in a config option, say
        return repr(value)


def _duplicates(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return list(dict.fromkeys(x for x in items if x in seen or seen.add(x)))


def _short(names: list[str], limit: int = 8) -> str:
    more = f" and {len(names) - limit} more" if len(names) > limit else ""
    return ", ".join(names[:limit]) + more


# Comparing and fingerprints ----------------------------------------------------


def diff(a: Any, b: Any) -> list[dict[str, Any]]:
    """What changed from a to b, as [{path, from, to}] with paths such as
    'run.budget_usd' or 'arms[1].options.max_turns'. A missing key reads as
    None. Values are compared as canonical JSON, so a diff is empty exactly
    when two specs hash the same."""
    return _diff(a, b, "")


def _diff(a: Any, b: Any, path: str) -> list[dict[str, Any]]:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out = []
        for key in [*a, *(k for k in b if k not in a)]:
            sub = f"{path}.{key}" if path else str(key)
            if key not in b:
                out.append({"path": sub, "from": a[key], "to": None})
            elif key not in a:
                out.append({"path": sub, "from": None, "to": b[key]})
            else:
                out += _diff(a[key], b[key], sub)
        return out
    if (isinstance(a, list) and isinstance(b, list) and (a or b)
            and all(isinstance(x, Mapping) for x in [*a, *b])):
        out = []
        for i in range(max(len(a), len(b))):
            sub = f"{path}[{i}]"
            if i >= len(a):
                out.append({"path": sub, "from": None, "to": b[i]})
            elif i >= len(b):
                out.append({"path": sub, "from": a[i], "to": None})
            else:
                out += _diff(a[i], b[i], sub)
        return out
    if fsutil.canonical_json(a) == fsutil.canonical_json(b):
        return []
    return [{"path": path, "from": copy.deepcopy(a), "to": copy.deepcopy(b)}]


def config_fingerprint(cfg: Config) -> str:
    """Changes whenever the effective config or the prices change, so a cached
    design check can be shown as stale."""
    pricing = {p: {m: dataclasses.asdict(price) for m, price in models.items()} for p, models in cfg.pricing.items()}
    # default=str: YAML turns an unquoted date into a date object.
    text = json.dumps([cfg.effective, pricing], sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str)
    return "sha256:" + fsutil.sha256_hex(text)


def status_of(meta: Mapping[str, Any], version: int, queue_status: str | None = None) -> str:
    """Derived, never stored. queue_status is the status of this version's
    queue item, if it has one that is not finished."""
    if meta.get("archived"):
        return "archived"
    if queue_status in ("starting", "running"):
        return "running"
    if queue_status in ("waiting", "held"):
        return "queued"
    entry = _mapping(_mapping(meta.get("versions")).get(str(version)))
    if entry.get("runs"):
        return "ran"
    if entry.get("approved_at"):
        return "approved"
    return "draft"


# Store -------------------------------------------------------------------------


def valid_design_id(design_id: Any) -> bool:
    return (isinstance(design_id, str) and DESIGN_ID.fullmatch(design_id) is not None
            and fsutil.is_safe_segment(design_id))


def new_design_id(title: str) -> str:
    return f"d-{_slug(title)[:24].strip('-') or 'design'}-{secrets.token_hex(2)}"


def _slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _vname(version: int) -> str:
    return f"v{version:04d}.json"


def _version_number(version: Any) -> int:
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError("A design version is a whole number from 1")
    return version


_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(root: Path) -> threading.Lock:
    # Shared by every store on one folder, so two handles cannot both write version N+1.
    key = os.path.normcase(os.path.abspath(root))
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


class DesignStore:
    """Designs on disk: designs/<id>/v0001.json (write-once), meta.json
    (rewritten atomically) and v0001.check.json (the last check, never hashed).

    A version file is written before meta.json, and meta.json is rebuilt from
    the version files when it lags behind, so a crash or a locked file between
    the two never loses a version. Ids are validated before any path is built,
    and a spec that carries a credential is refused before anything is
    written, since version files can never be rewritten and git tracks them.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else settings.DESIGNS_DIR
        self._lock = _lock_for(self.root)

    # Reading ----------------------------------------------------------------

    def list(self, include_archived: bool = False,
             queue_status: Mapping[tuple[str, int], str] | None = None) -> list[dict[str, Any]]:
        """One summary per design, newest first. A design whose files cannot
        be read is listed with an error instead of failing the list."""
        if not self.root.is_dir():
            return []
        out = []
        for folder in self.root.iterdir():
            if not folder.is_dir() or not valid_design_id(folder.name):
                continue
            try:
                meta = self._meta(folder.name)
            except DesignUnreadable as exc:
                out.append(_unreadable(folder.name, exc))
                continue
            except DesignNotFound:
                continue  # being created, or removed while listing
            except Exception as exc:  # one broken folder never fails the list
                out.append(_unreadable(folder.name, exc))
                continue
            if meta["archived"] and not include_archived:
                continue
            try:
                latest = self._read_version(folder.name, meta["latest_version"])
                out.append(self._summary(meta, latest, queue_status or {}))
            except Exception as exc:  # also DesignNotFound: meta names a version that cannot be read
                out.append(_unreadable(folder.name, exc))
        # str(): a hand-edited updated_at of another type must not break the sort.
        out.sort(key=lambda s: str(s.get("updated_at") or ""), reverse=True)
        return out

    def meta(self, design_id: str,
             queue_status: Mapping[tuple[str, int], str] | None = None) -> dict[str, Any]:
        meta = copy.deepcopy(self._meta(design_id))
        queue_status = queue_status or {}
        meta["status"] = status_of(meta, meta["latest_version"],
                                   queue_status.get((design_id, meta["latest_version"])))
        for number, entry in meta["versions"].items():
            entry["status"] = status_of(meta, int(number), queue_status.get((design_id, int(number))))
        return meta

    def version(self, design_id: str, version: int | None = None,
                queue_status: Mapping[tuple[str, int], str] | None = None) -> dict[str, Any]:
        meta = self._meta(design_id)
        number = meta["latest_version"] if version is None else _version_number(version)
        entry = meta["versions"].get(str(number))
        if entry is None:
            raise DesignNotFound(f"Design {design_id} has no version {number}")
        doc = self._read_version(design_id, number)
        changes: list[dict[str, Any]] = []
        parent = doc.get("parent_version")
        if isinstance(parent, int) and str(parent) in meta["versions"]:
            try:
                changes = _version_diff(self._read_version(design_id, parent), doc)
            except DesignNotFound:
                changes = []
        return {
            **doc,
            "status": status_of(meta, number, (queue_status or {}).get((design_id, number))),
            "approved_at": entry.get("approved_at"),
            "runs": list(entry.get("runs") or []),
            "archived": bool(meta["archived"]),
            "latest_version": meta["latest_version"],
            "check": self._read_check(design_id, number),
            "diff_from_parent": changes,
        }

    # Writing ----------------------------------------------------------------

    def create(self, cfg: Config, meta_in: DesignMetaIn | Mapping[str, Any],
               spec: DesignSpecIn | Mapping[str, Any], author: Literal["user", "assistant"],
               conversation_id: str | None = None) -> dict[str, Any]:
        """A new design with spec as version 1. spec is a NormalizedSpec, or a
        DesignSpecIn that is normalized against cfg first. A spec with
        problems is still saved; validate says whether it can run."""
        meta_in = _meta_in(meta_in)
        spec = _stored_spec(cfg, spec, meta_in)
        _check_author(author, conversation_id)
        with self._lock:
            design_id = self._new_folder(meta_in.title)
            now = _now()
            try:
                doc = _version_doc(design_id, 1, None, author, conversation_id, now, meta_in, spec)
                fsutil.create_exclusive_json(self.root / design_id / _vname(1), doc)
            except BaseException:
                _remove_new_folder(self.root / design_id)
                raise
            meta = {"schema": 1, "id": design_id, "title": meta_in.title, "created_at": now, "updated_at": now,
                    "latest_version": 1, "archived": False, "versions": {"1": _version_entry(doc)}}
            # Not fatal if locked: meta.json is rebuilt from the version file.
            fsutil.atomic_write_json(self.root / design_id / "meta.json", meta)
        return self.version(design_id, 1)

    def add_version(self, cfg: Config, design_id: str, base_version: int,
                    meta_in: DesignMetaIn | Mapping[str, Any], spec: DesignSpecIn | Mapping[str, Any],
                    author: Literal["user", "assistant"], conversation_id: str | None = None) -> dict[str, Any]:
        """Version latest+1, as a draft. base_version is the version the caller
        read; anything newer means they would overwrite a change unseen."""
        self._folder(design_id)
        base = _version_number(base_version)
        meta_in = _meta_in(meta_in)
        spec = _stored_spec(cfg, spec, meta_in)
        _check_author(author, conversation_id)
        with self._lock:
            meta = self._meta(design_id)
            if meta["archived"]:
                raise VersionConflict(f"Design {design_id} is archived. Unarchive it before saving a new version.")
            latest = meta["latest_version"]
            if base != latest:
                raise VersionConflict(f"Version {latest} of {design_id} was saved since you read version {base}. "
                                      "Load the latest version and apply your change again.")
            current = self._read_version(design_id, latest)
            new_hash = spec_hash(spec)
            if new_hash == current.get("spec_hash") and all(
                    current.get(k) == getattr(meta_in, k) for k in ("title", "purpose", "rationale")):
                raise VersionConflict(f"Nothing changed: the spec, title, purpose and rationale match version "
                                      f"{latest}.")
            # Above every version file on disk, not just the readable ones: a
            # half-written file left by a crash must not take the number for good.
            number = max(latest, self._highest_on_disk(design_id)) + 1
            now = _now()
            doc = _version_doc(design_id, number, latest, author, conversation_id, now, meta_in, spec)
            path = self.root / design_id / _vname(number)
            try:
                fsutil.create_exclusive_json(path, doc)
            except FileExistsError:
                try:
                    self._read_version(design_id, number)
                except DesignUnreadable as exc:
                    raise VersionConflict(f"{exc}. Move {path.name} out of the design's folder and save again.") \
                        from None
                raise VersionConflict(f"Version {number} of {design_id} was saved by someone else just now. "
                                      "Load it and apply your change again.") from None
            except BaseException:
                # Ours, half written: the file was created by this call ('x' mode).
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass  # skipped as unreadable; the next save numbers above it
                raise
            meta["versions"][str(number)] = _version_entry(doc)
            meta.update(latest_version=number, title=meta_in.title, updated_at=now)
            fsutil.atomic_write_json(self.root / design_id / "meta.json", meta)
        return self.version(design_id, number)

    def record_check(self, design_id: str, version: int, check: Mapping[str, Any]) -> None:
        """Cache the last validation and estimate beside the version. Callers
        compare its fingerprints with the current ones to show it as stale."""
        number = _version_number(version)
        if not isinstance(check, Mapping):
            raise ValueError("A design check must be a JSON object")
        data = {**check}
        data.setdefault("checked_at", _now())
        with self._lock:
            meta = self._meta(design_id)
            if str(number) not in meta["versions"]:
                raise DesignNotFound(f"Design {design_id} has no version {number}")
            if not fsutil.atomic_write_json(self.root / design_id / f"v{number:04d}.check.json", data):
                raise PermissionError(f"The check file of {design_id} is busy (another program has it open). "
                                      "Try again.")

    def mark_approved(self, design_id: str, version: int) -> None:
        """A person admitted this version to the queue. Idempotent."""
        number = _version_number(version)
        with self._lock:
            meta = self._meta(design_id)
            if meta["archived"]:
                raise VersionConflict(f"Design {design_id} is archived. Unarchive it before queueing it.")
            entry = self._entry(meta, design_id, number)
            if not entry.get("approved_at"):
                entry["approved_at"] = meta["updated_at"] = _now()
                self._write_meta(design_id, meta)

    def record_run(self, design_id: str, version: int, run_id: str) -> None:
        number = _version_number(version)
        if not isinstance(run_id, str) or not fsutil.is_safe_segment(run_id):
            raise ValueError("A run id must be a plain folder name")
        with self._lock:
            meta = self._meta(design_id)
            entry = self._entry(meta, design_id, number)
            runs = entry.setdefault("runs", [])
            if run_id not in runs:
                runs.append(run_id)
                meta["updated_at"] = _now()
                self._write_meta(design_id, meta)

    def archive(self, design_id: str) -> None:
        self._set_archived(design_id, True)

    def unarchive(self, design_id: str) -> None:
        self._set_archived(design_id, False)

    # Internals --------------------------------------------------------------

    def _set_archived(self, design_id: str, archived: bool) -> None:
        with self._lock:
            meta = self._meta(design_id)
            if meta["archived"] != archived:
                meta["archived"] = archived
                meta["updated_at"] = _now()
                self._write_meta(design_id, meta)

    def _folder(self, design_id: Any) -> Path:
        if not valid_design_id(design_id):
            raise ValueError(f"'{str(design_id)[:40]}' is not a design id")
        return self.root / design_id

    def _new_folder(self, title: str) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        for _ in range(50):
            design_id = new_design_id(title)
            try:
                (self.root / design_id).mkdir()
            except FileExistsError:
                continue
            return design_id
        raise RuntimeError("Could not find a free design id; try another title")

    def _entry(self, meta: dict[str, Any], design_id: str, number: int) -> dict[str, Any]:
        entry = meta["versions"].get(str(number))
        if entry is None:
            raise DesignNotFound(f"Design {design_id} has no version {number}")
        return entry

    def _write_meta(self, design_id: str, meta: dict[str, Any]) -> None:
        if not fsutil.atomic_write_json(self.root / design_id / "meta.json", meta):
            raise PermissionError(_BUSY)

    def _meta(self, design_id: str) -> dict[str, Any]:
        folder = self._folder(design_id)
        if not folder.is_dir():
            raise DesignNotFound(f"No design '{design_id}'")
        try:
            raw = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            raw = None  # a crash between the first version file and meta.json
        except PermissionError:
            raise PermissionError(_BUSY) from None  # a scanner's lock is not a missing design
        except (OSError, ValueError) as exc:
            raise DesignUnreadable(f"Could not read design {design_id}: meta.json: {exc}") from None
        if raw is not None and not _is_meta_record(raw):
            raise DesignUnreadable(f"Could not read design {design_id}: meta.json is not a design record")
        return self._reconcile(design_id, folder, raw)

    def _highest_on_disk(self, design_id: str) -> int:
        folder = self._folder(design_id)
        return max((int(m.group(1)) for p in folder.iterdir() if (m := _VERSION_FILE.fullmatch(p.name))),
                   default=0)

    def _reconcile(self, design_id: str, folder: Path, meta: dict[str, Any] | None) -> dict[str, Any]:
        """meta.json plus any version file it does not list yet."""
        on_disk = sorted(int(m.group(1)) for p in folder.iterdir() if (m := _VERSION_FILE.fullmatch(p.name)))
        if meta is None:
            if not on_disk:
                # A folder create() has made but not filled yet is not a design.
                raise DesignNotFound(f"No design '{design_id}'")
            meta = {"schema": 1, "id": design_id, "title": "", "created_at": None, "updated_at": None,
                    "latest_version": 0, "archived": False, "versions": {}}
        meta.setdefault("id", design_id)
        meta.setdefault("archived", False)
        latest = meta.get("latest_version")
        meta["latest_version"] = latest if isinstance(latest, int) and not isinstance(latest, bool) else 0
        for number in on_disk:
            if str(number) in meta["versions"]:
                continue
            try:
                doc = self._read_version(design_id, number)
            except DesignNotFound:
                continue
            meta["versions"][str(number)] = _version_entry(doc)
            meta["created_at"] = meta.get("created_at") or doc.get("created_at")
            if number > meta["latest_version"]:
                meta.update(latest_version=number, title=doc.get("title") or meta.get("title") or "",
                            updated_at=doc.get("created_at"))
        if not meta["versions"] or str(meta["latest_version"]) not in meta["versions"]:
            raise DesignUnreadable(f"Could not read design {design_id}: no readable versions")
        return meta

    def _read_version(self, design_id: str, number: int) -> dict[str, Any]:
        path = self._folder(design_id) / _vname(number)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise DesignNotFound(f"Design {design_id} has no version {number}") from None
        except PermissionError:
            raise PermissionError(f"Design {design_id} version {number} is busy (another program has it "
                                  "open). Try again.") from None
        except (OSError, ValueError) as exc:
            raise DesignUnreadable(f"Could not read design {design_id} version {number}: {exc}") from None
        if not isinstance(doc, dict) or not isinstance(doc.get("spec"), dict):
            raise DesignUnreadable(f"Could not read design {design_id} version {number}: not a design version")
        return doc

    def _read_check(self, design_id: str, number: int) -> dict[str, Any] | None:
        path = self._folder(design_id) / f"v{number:04d}.check.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None  # never checked, or a cache that will be rewritten
        return data if isinstance(data, dict) else None

    def _summary(self, meta: dict[str, Any], latest: dict[str, Any],
                 queue_status: Mapping[tuple[str, int], str]) -> dict[str, Any]:
        number = meta["latest_version"]
        entry = meta["versions"][str(number)]
        spec = latest["spec"]
        try:
            arms, trials = len(spec["arms"]), trial_count(spec)
        except (KeyError, TypeError):
            arms, trials = None, None
        return {
            "id": meta["id"],
            "title": meta.get("title") or latest.get("title"),
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "latest_version": number,
            "archived": bool(meta["archived"]),
            "author": entry.get("author"),
            "conversation_id": entry.get("conversation_id"),
            "status": status_of(meta, number, queue_status.get((meta["id"], number))),
            "approved_at": entry.get("approved_at"),
            "runs": list(entry.get("runs") or []),
            "spec_hash": latest.get("spec_hash"),
            "arms": arms,
            "trials": trials,
            **check_figures(self._read_check(meta["id"], number)),
        }


def _is_meta_record(raw: Any) -> bool:
    """meta.json is git-tracked, so a hand edit or a merge can leave valid
    JSON of the wrong shape; such a design reads as unreadable, never as a crash."""
    if not isinstance(raw, dict) or not isinstance(raw.get("versions"), dict):
        return False
    latest = raw.get("latest_version")
    if not isinstance(latest, int) or isinstance(latest, bool):
        return False
    if not isinstance(raw.get("archived", False), bool):
        return False
    if not all(isinstance(k, str) and re.fullmatch(r"[1-9]\d*", k) for k in raw["versions"]):
        return False
    return all(isinstance(entry, dict) and isinstance(entry.get("runs", []), list)
               for entry in raw["versions"].values())


def _unreadable(design_id: str, exc: Exception) -> dict[str, Any]:
    error = str(exc) if isinstance(exc, DesignNotFound) else f"Could not read design {design_id}: {exc}"
    return {"id": design_id, "title": None, "status": "unreadable", "archived": False, "updated_at": None,
            "error": error}


def _remove_new_folder(folder: Path) -> None:
    """Undo a create() that failed before its first version was complete. The
    folder was made moments ago under the store lock, so nothing else is in it."""
    try:
        for path in folder.iterdir():
            path.unlink()
        folder.rmdir()
    except OSError:
        pass  # a partial v0001.json left behind is listed as unreadable


def check_figures(check: Mapping[str, Any] | None) -> dict[str, Any]:
    """The headline numbers of a cached check, for design lists."""
    check = _mapping(check)
    validation = _mapping(check.get("validation"))
    estimate = _mapping(check.get("estimate"))
    total = _mapping(_mapping(estimate.get("run")).get("total"))

    def count(key: str) -> int | None:
        value = validation.get(key)
        return len(value) if isinstance(value, list) else None

    return {
        "problems": count("problems"),
        "warnings": count("warnings"),
        "expected_usd": total.get("p50", estimate.get("expected_usd")),
        "p90_usd": total.get("p90", estimate.get("p90_usd")),
        "checked_at": check.get("checked_at"),
        "config_fingerprint": check.get("config_fingerprint"),
        "ledger_fingerprint": check.get("ledger_fingerprint"),
    }


def _version_doc(design_id: str, number: int, parent: int | None, author: str, conversation_id: str | None,
                 now: str, meta_in: DesignMetaIn, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": 1,
        "design_id": design_id,
        "version": number,
        "parent_version": parent,
        "author": author,
        "conversation_id": conversation_id,
        "created_at": now,
        "profile": settings.PROFILE_NAME,
        "title": meta_in.title,
        "purpose": meta_in.purpose,
        "rationale": meta_in.rationale,
        "change_note": meta_in.change_note,
        "spec": spec,
        "spec_hash": spec_hash(spec),
    }


def _version_entry(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {"author": doc.get("author"), "conversation_id": doc.get("conversation_id"),
            "created_at": doc.get("created_at"), "spec_hash": doc.get("spec_hash"),
            "change_note": doc.get("change_note", ""), "approved_at": None, "runs": []}


def _version_diff(parent: Mapping[str, Any], doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    changes = [{"path": k, "from": parent.get(k), "to": doc.get(k)}
               for k in ("title", "purpose", "rationale") if parent.get(k) != doc.get(k)]
    return changes + diff(parent.get("spec"), doc.get("spec"))


def _meta_in(meta_in: DesignMetaIn | Mapping[str, Any]) -> DesignMetaIn:
    if isinstance(meta_in, DesignMetaIn):
        return meta_in
    if not isinstance(meta_in, Mapping):
        raise ValueError("Design details must be an object with a title")
    return DesignMetaIn.model_validate(dict(meta_in))


def _stored_spec(cfg: Config, spec: DesignSpecIn | Mapping[str, Any], meta_in: DesignMetaIn) -> dict[str, Any]:
    """The spec to write, checked before any folder or file exists. Problems
    a person can fix later are saved for validate to report; a credential
    is refused, because version files are write-once and tracked by git."""
    if isinstance(spec, DesignSpecIn):
        spec = normalize(cfg, spec)
    problems = structure_problems(spec)
    if problems:
        raise ValueError("Not a valid design spec: " + "; ".join(problems))
    spec = copy.deepcopy(dict(spec))
    found = [f"arms[{i}].options.{path}" for i, arm in enumerate(spec["arms"])
             for path in _denied_options(arm["options"], _CREDENTIAL_OPTION)]
    found += _secret_paths(spec, meta_in)
    if found:
        message = f"Designs cannot store credentials ({_short(list(dict.fromkeys(found)))}). Set keys on Setup."
        # A live key used as an option name would be in its own path.
        from . import keys
        for secret in keys.secret_values():
            message = message.replace(secret, "[redacted]")
        raise ValueError(message)
    return spec


def _secret_paths(spec: Mapping[str, Any], meta_in: DesignMetaIn) -> list[str]:
    """Where a value of a key in this process's environment appears, by path
    only: the message must never repeat the key."""
    # Imported here: only saving needs keys, and reading designs should not load it.
    from . import keys
    secrets_now = keys.secret_values()
    if not secrets_now:
        return []

    def texts(value: Any, path: str) -> Iterator[tuple[str, str]]:
        if isinstance(value, str):
            yield path, value
        elif isinstance(value, Mapping):
            for key, sub in value.items():
                sub_path = f"{path}.{key}" if path else str(key)
                yield sub_path, str(key)
                yield from texts(sub, sub_path)
        elif isinstance(value, (list, tuple)):
            for i, sub in enumerate(value):
                yield from texts(sub, f"{path}[{i}]")

    record = {"title": meta_in.title, "purpose": meta_in.purpose, "rationale": meta_in.rationale,
              "change_note": meta_in.change_note, **spec}
    return [path for path, text in texts(record, "") if any(s in text for s in secrets_now)]


def _check_author(author: Any, conversation_id: Any) -> None:
    if author not in AUTHORS:
        raise ValueError("A design's author is 'user' or 'assistant'")
    if conversation_id is not None and not (isinstance(conversation_id, str)
                                            and _CONVERSATION_ID.fullmatch(conversation_id)):
        raise ValueError("Not a conversation id")


class DraftWriter:
    """The only write handle the assistant gets: it saves draft designs and
    draft versions, always as author "assistant", and reads designs. It has
    no way to approve, queue, archive or record runs."""

    __slots__ = ("_store", "_cfg_getter", "_conversation_id", "_queue_status")

    def __init__(self, store: DesignStore, cfg_getter: Callable[[], Config], conversation_id: str,
                 queue_status: Callable[[], Mapping[tuple[str, int], str]] | None = None) -> None:
        # Required: a draft without its conversation loses where it came from.
        if not (isinstance(conversation_id, str) and _CONVERSATION_ID.fullmatch(conversation_id)):
            raise ValueError("Not a conversation id")
        self._store = store
        self._cfg_getter = cfg_getter
        self._conversation_id = conversation_id
        self._queue_status = queue_status

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    def create_draft(self, meta_in: DesignMetaIn | Mapping[str, Any],
                     spec_in: DesignSpecIn | Mapping[str, Any]) -> dict[str, Any]:
        cfg = self._cfg_getter()
        spec = normalize(cfg, spec_in)
        return self._store.create(cfg, meta_in, spec, "assistant", self._conversation_id)

    def add_draft_version(self, design_id: str, base_version: int, meta_in: DesignMetaIn | Mapping[str, Any],
                          spec_in: DesignSpecIn | Mapping[str, Any]) -> dict[str, Any]:
        cfg = self._cfg_getter()
        spec = normalize(cfg, spec_in)
        return self._store.add_version(cfg, design_id, base_version, meta_in, spec, "assistant",
                                       self._conversation_id)

    def list(self, include_archived: bool = False) -> list[dict[str, Any]]:
        return self._store.list(include_archived, self._queue())

    def meta(self, design_id: str) -> dict[str, Any]:
        return self._store.meta(design_id, self._queue())

    def version(self, design_id: str, version: int | None = None) -> dict[str, Any]:
        return self._store.version(design_id, version, self._queue())

    def _queue(self) -> Mapping[tuple[str, int], str] | None:
        if self._queue_status is None:
            return None
        try:
            return self._queue_status()
        except Exception:  # a queue that cannot answer leaves statuses as stored, never a failed read
            return None
