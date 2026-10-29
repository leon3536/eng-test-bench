"""Run specs: one description of what a run executes, however it was launched.

A run used to be described by a config mode plus arm names, so its run.json
could only be read against the config of that day. A RunSpec carries
everything that changes what runs: dataset, tasks, attempts, run settings,
and every arm with its harness family and options. It is held in a
normalized, hashed form (the NormalizedSpec). Config runs (the LAUNCH tab)
and design runs (the queue) both become a RunSpec, so they execute the same
way and record the same spec. The ledger can then pool every trial, whatever
started it.

Only execution settings are hashed. A design's title, purpose and notes live
in the version file around the spec, so renaming a design never changes what
it runs or how its trials pool.

runs.py and designs.py both import this module, so it must not import them.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import fsutil
from .harnesses import is_import_path
from .settings import Arm, Config, ConfigError, RunSettings

# Key order is the order the spec is stored in; the hash sorts keys anyway.
SPEC_KEYS = ("dataset", "tasks", "attempts", "baseline_arm", "run", "arms")
RUN_KEYS = ("budget_usd", "parallel_arms", "n_concurrent", "effort", "per_trial_cap_usd", "max_turns",
            "timeout_multiplier")
ARM_KEYS = ("name", "agent", "family", "provider", "model", "effort", "options", "tasks")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    agent: str              # what Harbor runs: a registered name or module.path:Class
    family: str             # the harness label for pooling and reference lookups
    kind: str               # "name" | "import_path"
    provider: str
    model: str              # without the provider's model prefix
    effort: str             # "inherit" | "omit" | a level the harness accepts
    options: dict[str, Any]
    tasks: tuple[str, ...]  # the tasks this arm runs: the spec's, or its own subset

    def to_arm(self) -> Arm:
        return Arm(name=self.name, agent=self.agent, provider=self.provider, model=self.model,
                   enabled=True, effort=self.effort, options=copy.deepcopy(self.options),
                   family=self.family)


@dataclass(frozen=True)
class RunSpec:
    source: str             # "config" | "design"
    label: str              # mode name, or "<slug16>-v<N>" for a design
    dataset: str
    tasks: tuple[str, ...]
    n_attempts: int
    run: RunSettings
    arms: tuple[ArmSpec, ...]
    baseline_arm: str | None
    spec: dict[str, Any]    # the NormalizedSpec, exactly as hashed
    spec_hash: str
    design_ref: dict[str, Any] | None


def spec_hash(spec: Mapping[str, Any]) -> str:
    """Hash of the canonical JSON text, never of file bytes, so an editor that
    rewrites line endings or key order cannot change it."""
    return "sha256:" + fsutil.sha256_hex(fsutil.canonical_json(spec))


def run_dict(run: RunSettings) -> dict[str, Any]:
    """The spec's run section. Types are fixed here because the hash sees
    them: a budget of 5 and of 5.0 must not be two different specs."""
    return {
        "budget_usd": float(run.budget_usd),
        "parallel_arms": int(run.parallel_arms),
        "n_concurrent": int(run.n_concurrent),
        "effort": str(run.effort),
        "per_trial_cap_usd": None if run.per_trial_cap_usd is None else float(run.per_trial_cap_usd),
        "max_turns": None if run.max_turns is None else int(run.max_turns),
        "timeout_multiplier": float(run.timeout_multiplier),
    }


def arm_dict(arm: Arm, tasks: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": arm.name,
        "agent": arm.agent,
        "family": arm.family or arm.agent,
        "provider": arm.provider,
        "model": arm.model,
        "effort": arm.effort,
        "options": {k: copy.deepcopy(arm.options[k]) for k in sorted(arm.options)},
        "tasks": None if tasks is None else list(tasks),
    }


def normalized_from_config(cfg: Config, mode_name: str, arm_names: list[str]) -> dict[str, Any]:
    """The NormalizedSpec of a LAUNCH-tab run: exactly the inputs plan() has
    always used, so a config run's record says what it ran."""
    if mode_name not in cfg.modes:
        raise ConfigError(f"Unknown mode '{mode_name}'")
    if not arm_names:
        raise ConfigError("Pick at least one arm")
    mode = cfg.modes[mode_name]
    tasks = list(cfg.tasks[: mode.n_tasks])
    if not tasks:
        raise ConfigError("No task sample yet. Run: python -m bench.sample_tasks")
    # A name picked twice would be two arms writing one job folder.
    names = list(dict.fromkeys(arm_names))
    arms = [arm_dict(cfg.arm(name)) for name in names]
    return {
        "dataset": cfg.dataset,
        "tasks": tasks,
        "attempts": mode.n_attempts,
        "baseline_arm": cfg.baseline_arm if cfg.baseline_arm in names else None,
        "run": run_dict(cfg.run),
        "arms": arms,
    }


def spec_from_config(cfg: Config, mode_name: str, arm_names: list[str]) -> RunSpec:
    spec = normalized_from_config(cfg, mode_name, arm_names)
    return _run_spec(spec, source="config", label=mode_name, ref=None)


def spec_from_design(spec: Mapping[str, Any], ref: Mapping[str, Any] | None) -> RunSpec:
    """ref = {design_id, version, author, conversation_id, queue_item}, or
    None when a design is only being checked."""
    return _run_spec(spec, source="design", label=design_label(ref), ref=ref)


def design_label(ref: Mapping[str, Any] | None) -> str:
    """'<slug[:16]>-v<N>': the part of a design run's id after the time. Kept
    to [a-z0-9-] because it becomes a folder name and a Docker project prefix."""
    if not ref:
        return "design"
    slug = str(ref.get("design_id") or "")
    slug = slug[2:] if slug.startswith("d-") else slug
    slug = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")[:16].strip("-") or "design"
    version = ref.get("version")
    if isinstance(version, int) and not isinstance(version, bool):
        return f"{slug}-v{version}"
    return slug


def _run_spec(spec: Mapping[str, Any], source: str, label: str, ref: Mapping[str, Any] | None) -> RunSpec:
    problems = structure_problems(spec)
    if problems:
        raise ConfigError("Not a valid run spec: " + "; ".join(problems))
    spec = copy.deepcopy(dict(spec))
    tasks = tuple(spec["tasks"])
    run = spec["run"]
    settings = RunSettings(
        effort=run["effort"],
        n_concurrent=run["n_concurrent"],
        parallel_arms=run["parallel_arms"],
        budget_usd=float(run["budget_usd"]),
        per_trial_cap_usd=None if run["per_trial_cap_usd"] is None else float(run["per_trial_cap_usd"]),
        max_turns=run["max_turns"],
        timeout_multiplier=float(run["timeout_multiplier"]),
    )
    arms = tuple(
        ArmSpec(
            name=a["name"],
            agent=a["agent"],
            family=a["family"],
            kind="import_path" if is_import_path(a["agent"]) else "name",
            provider=a["provider"],
            model=a["model"],
            effort=a["effort"],
            options=copy.deepcopy(dict(a["options"])),
            tasks=tasks if a["tasks"] is None else tuple(a["tasks"]),
        )
        for a in spec["arms"]
    )
    return RunSpec(
        source=source,
        label=label,
        dataset=spec["dataset"],
        tasks=tasks,
        n_attempts=spec["attempts"],
        run=settings,
        arms=arms,
        baseline_arm=spec["baseline_arm"],
        spec=spec,
        spec_hash=spec_hash(spec),
        design_ref=None if ref is None else copy.deepcopy(dict(ref)),
    )


# Structure -------------------------------------------------------------------
# Shape and types only. Limits, allowed harnesses and everything that depends
# on config are designs.validate's job; this is what a RunSpec needs to exist.


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _number(v: Any) -> bool:
    # NaN fails every comparison and inf never trips a budget, so neither is a number here.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _text(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _names(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v)


def plain_json(value: Any) -> bool:
    """Whether value is stored as itself: no sets, dates or other Python
    objects, and no NaN or Infinity, which JSON responses cannot carry. A
    version file is write-once, so such a value would break it for good."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


_RUN_TYPES = {
    "budget_usd": (_number, "a number"),
    "parallel_arms": (_int, "a whole number"),
    "n_concurrent": (_int, "a whole number"),
    "effort": (_text, "an effort name"),
    "per_trial_cap_usd": (lambda v: v is None or _number(v), "a number or null"),
    "max_turns": (lambda v: v is None or _int(v), "a whole number or null"),
    "timeout_multiplier": (_number, "a number"),
}
_ARM_TYPES = {
    "name": (_text, "a name"),
    "agent": (_text, "a Harbor agent name or import path"),
    "family": (_text, "a family name"),
    "provider": (_text, "a provider name"),
    "model": (_text, "a model id"),
    "effort": (_text, "inherit, omit or an effort level"),
    "options": (lambda v: isinstance(v, Mapping) and all(isinstance(k, str) for k in v),
                "an object of options"),
    "tasks": (lambda v: v is None or _names(v), "a list of task names or null"),
}


def _fields(obj: Mapping[str, Any], keys: tuple[str, ...], label: str) -> list[str]:
    problems = [f"{label} is missing '{k}'" for k in keys if k not in obj]
    problems += [f"{label} has an unknown field '{k}'" for k in obj if k not in keys]
    return problems


def _typed(obj: Mapping[str, Any], types: dict[str, tuple[Any, str]], label: str) -> list[str]:
    return [f"{label}.{k} must be {what}" for k, (ok, what) in types.items() if k in obj and not ok(obj[k])]


def structure_problems(spec: Any) -> list[str]:
    """Why spec is not a NormalizedSpec, as plain sentences; [] when it is one."""
    if not isinstance(spec, Mapping):
        return ["The spec must be a JSON object"]
    problems = _fields(spec, SPEC_KEYS, "spec")
    problems += _typed(spec, {
        "dataset": (_text, "a dataset name"),
        "tasks": (_names, "a list of task names"),
        "attempts": (_int, "a whole number"),
        "baseline_arm": (lambda v: v is None or isinstance(v, str), "an arm name or null"),
    }, "spec")
    if "run" in spec:
        run = spec["run"]
        if isinstance(run, Mapping):
            problems += _fields(run, RUN_KEYS, "spec.run") + _typed(run, _RUN_TYPES, "spec.run")
        else:
            problems.append("spec.run must be an object")
    if "arms" in spec:
        arms = spec["arms"]
        if not isinstance(arms, (list, tuple)):
            problems.append("spec.arms must be a list")
            return problems
        for i, arm in enumerate(arms):
            label = f"spec.arms[{i}]"
            if isinstance(arm, Mapping):
                problems += _fields(arm, ARM_KEYS, label) + _typed(arm, _ARM_TYPES, label)
                options = arm.get("options")
                if isinstance(options, Mapping) and not plain_json(options):
                    problems.append(f"{label}.options must be plain JSON with finite numbers")
            else:
                problems.append(f"{label} must be an object")
    return problems
