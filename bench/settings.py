"""Paths, process environment and configuration.

Configuration has two layers. config/bench.yaml holds the hand-edited
defaults; config/overrides.yaml holds what the dashboard changes. The
dashboard only ever writes the overrides file, so bench.yaml keeps its
comments and stays the readable source of defaults.

Code defaults for settings a YAML file may leave out live here too
(RUN_DEFAULTS, MODE_DEFAULTS, ARM_DEFAULTS and the section dataclasses), so
load_config and the dashboard's "is this overridden?" diffs agree.
"""

from __future__ import annotations

import copy
import math
import os
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import fsutil, stats
from .harnesses import is_import_path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

# Names of the config files that are not bench configs, so they can never be
# a profile; "default" is what the dashboard calls the base profile.
_RESERVED_PROFILES = frozenset({"pricing", "reference", "overrides", "tasks"})


def _profile_from_env() -> str:
    name = os.environ.get("BENCH_PROFILE", "").strip()
    if name == "default":
        return ""
    if name in _RESERVED_PROFILES or name.endswith(".overrides"):
        raise SystemExit(f"Invalid profile name '{name}': config/{name}.yaml is not a bench config")
    if name and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise SystemExit(f"Invalid profile name '{name}': use lowercase letters, digits, - and _")
    return name


# A profile is a separate config (config/<profile>.yaml) with its own
# overrides file and run data under profiles/<profile>/. The default profile
# uses config/bench.yaml and keeps its data at the project root.
PROFILE = _profile_from_env()
PROFILE_NAME = PROFILE or "default"
BASE_CONFIG = CONFIG_DIR / (f"{PROFILE}.yaml" if PROFILE else "bench.yaml")
if not BASE_CONFIG.exists():
    # Otherwise a typo loads an empty config and fails on some unrelated field.
    raise SystemExit(f"No profile '{PROFILE_NAME}': config/{BASE_CONFIG.name} does not exist")
OVERRIDES = CONFIG_DIR / (f"{PROFILE}.overrides.yaml" if PROFILE else "overrides.yaml")
PRICING = CONFIG_DIR / "pricing.yaml"
REFERENCE = CONFIG_DIR / "reference.yaml"
# Verdicts by hand on agents that looked up the benchmark's answer (lookups.py); shared by every profile.
ANSWER_LOOKUPS = CONFIG_DIR / "answer_lookups.yaml"
ENV_FILE = Path(os.environ.get("BENCH_ENV_FILE") or ROOT / ".env")

# BENCH_DATA_DIR lets tests (and a second checkout) keep run data elsewhere.
DATA_DIR = (Path(os.environ["BENCH_DATA_DIR"]) if os.environ.get("BENCH_DATA_DIR")
            else ROOT / "profiles" / PROFILE if PROFILE else ROOT)
JOBS_DIR = DATA_DIR / "jobs"
RUNS_DIR = DATA_DIR / "runs"
DESIGNS_DIR = DATA_DIR / "designs"
ASSISTANT_DIR = DATA_DIR / "assistant"
QUEUE_FILE = DATA_DIR / "queue.json"
# One ledger for every profile: pairs only ever form within one model, so
# mixing profiles in the store never mixes them inside a comparison.
LEDGER_DB = Path(os.environ.get("BENCH_LEDGER") or ROOT / "ledger" / "bench.sqlite")

EFFORT_INHERIT = "inherit"  # arm follows run.effort
EFFORT_OMIT = "omit"        # pass nothing; the harness or model default applies
DEFAULT_BASELINE_MODEL = "anthropic:claude-opus-5-5"   # baseline_model when a profile names none

# run.prebake: how arms use pre-baked harness images (bench/prebake). prefer
# falls back to Harbor's stock install per trial; require fails a trial whose
# image is missing instead; off runs exactly the stock command.
PREBAKE_OFF = "off"
PREBAKE_PREFER = "prefer"
PREBAKE_REQUIRE = "require"
PREBAKE_MODES = (PREBAKE_OFF, PREBAKE_PREFER, PREBAKE_REQUIRE)
SETUP_TIMEOUT_AUTO = "auto"
# run.schedule: how a run orders its trials. tasks races every arm on one task
# at a time (its pre-baked images built just before and removed after, so the
# disk holds one task's images); arms runs each arm as one Harbor job over all
# its tasks, parallel_arms at a time (the original schedule).
SCHEDULE_ARMS = "arms"
SCHEDULE_TASKS = "tasks"
SCHEDULES = (SCHEDULE_ARMS, SCHEDULE_TASKS)

RUN_DEFAULTS: dict[str, Any] = {
    "effort": EFFORT_OMIT,
    "n_concurrent": 2,
    "parallel_arms": 1,
    "per_trial_cap_usd": None,
    "max_turns": None,
    "timeout_multiplier": 1.0,
    "prebake": PREBAKE_PREFER,
    "setup_timeout_multiplier": SETUP_TIMEOUT_AUTO,
    "env_build_timeout_multiplier": 1.0,
    "schedule": SCHEDULE_TASKS,
    # Trial containers at once in a task race (schedule tasks): one per real CPU core of the bench host.
    "max_containers": 6,
}
MODE_DEFAULTS: dict[str, Any] = {"n_attempts": 1}
ARM_DEFAULTS: dict[str, Any] = {"enabled": True, "effort": EFFORT_INHERIT}

ASSISTANT_EFFORTS = ("low", "medium", "high", "xhigh", "max")
ASSISTANT_FALLBACKS = ("default", "off")
QUEUE_ON_FAILURE = ("pause", "skip")

# Harbor clones task repositories with git. With core.autocrlf=true (the Git
# for Windows default), 833 of Terminal-Bench 2.0's 857 files arrive with CRLF
# endings, including every Dockerfile and shell script, and then fail inside
# the Linux containers. core.longpaths covers deep task paths. These apply to
# every git process the bench starts and leave the global git config alone.
GIT_OVERRIDES = {"core.autocrlf": "false", "core.longpaths": "true"}


class ConfigError(ValueError):
    pass


def prepare_process_env() -> None:
    """Called once at startup, before any harbor or git subprocess."""
    if os.environ.get("BENCH_GIT_OVERRIDES") != "1":
        start = int(os.environ.get("GIT_CONFIG_COUNT") or 0)
        for i, (key, value) in enumerate(GIT_OVERRIDES.items(), start=start):
            os.environ[f"GIT_CONFIG_KEY_{i}"] = key
            os.environ[f"GIT_CONFIG_VALUE_{i}"] = value
        os.environ["GIT_CONFIG_COUNT"] = str(start + len(GIT_OVERRIDES))
        os.environ["BENCH_GIT_OVERRIDES"] = "1"
    # Harbor prints with rich. Piped to a log file on Windows it would use the
    # cp1252 code page and fail on the first character outside it.
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ.setdefault("COLUMNS", "160")


def harbor_executable() -> str:
    scripts = Path(sys.executable).parent
    exe = scripts / ("harbor.exe" if os.name == "nt" else "harbor")
    return str(exe) if exe.exists() else "harbor"


# A reported model name may add a date or version tag to the priced id
# (claude-haiku-5-5-20261001), but anything else is a different model:
# glm-5.3-flash is not glm-5.3, and claude-opus-5-5 is not claude-opus-5.
_VERSION_SUFFIX = re.compile(r"(-\d{8}|-\d{4}-\d{2}-\d{2}|-latest|@.+|:.+)?")


def is_variant(name: str, key: str) -> bool:
    """name is key itself or key plus a date/version tag, not a different model."""
    return name.startswith(key) and _VERSION_SUFFIX.fullmatch(name[len(key):]) is not None


@dataclass(frozen=True)
class Provider:
    name: str
    env_key: str
    model_prefix: str
    enabled: bool = True


@dataclass(frozen=True)
class Arm:
    name: str
    agent: str          # what Harbor runs: a registered name or module.path:Class
    provider: str
    model: str
    enabled: bool = True
    effort: str = EFFORT_INHERIT
    options: dict[str, Any] = field(default_factory=dict)
    family: str = ""    # the harness label for pooling and reference lookups

    @property
    def kind(self) -> str:
        return "import_path" if is_import_path(self.agent) else "name"


@dataclass(frozen=True)
class Mode:
    n_tasks: int
    n_attempts: int


@dataclass(frozen=True)
class RunSettings:
    effort: str
    n_concurrent: int
    parallel_arms: int
    budget_usd: float
    per_trial_cap_usd: float | None
    max_turns: int | None
    timeout_multiplier: float
    prebake: str = PREBAKE_PREFER
    # 'auto' or a number. Harbor's setup limit is 360 s; 'auto' doubles it
    # only for arms whose trials may run Harbor's stock install (bench.prebake.setup_timeout_multiplier).
    setup_timeout_multiplier: str | float = SETUP_TIMEOUT_AUTO
    env_build_timeout_multiplier: float = 1.0
    schedule: str = SCHEDULE_TASKS
    max_containers: int = 6           # schedule tasks only: trial containers at once in one task's race


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float | None
    # None: writes of that time to live are priced at the input rate (pricing.yaml).
    cache_write_5m: float | None = None
    cache_write_1h: float | None = None


@dataclass(frozen=True)
class AssistantSettings:
    enabled: bool = True
    effort: str = "medium"
    max_tokens: int = 32000
    max_steps: int = 12
    per_turn_cap_usd: float = 1.0
    daily_cap_usd: float = 5.0
    fallbacks: str = "default"


@dataclass(frozen=True)
class DesignLimits:
    max_budget_usd: float = 150.0
    max_arms: int = 12
    max_attempts: int = 5
    warn_containers: int = 8
    max_containers: int = 16
    extra_harnesses: tuple[str, ...] = ()
    import_paths: dict[str, str] = field(default_factory=dict)  # import path -> family


@dataclass(frozen=True)
class StatsSettings:
    confidence: float = 0.90
    target_halfwidth: float = 0.25
    prior_sd_log: float = 0.6
    default_trial_tokens: dict[str, float] = field(
        default_factory=lambda: {"input": 600000, "cache_share": 0.85, "output": 60000})


@dataclass(frozen=True)
class QueueSettings:
    on_failure: str = "pause"
    global_cap_usd: float | None = None


@dataclass(frozen=True)
class LedgerSettings:
    extra_scan_roots: tuple[Path, ...] = ()


@dataclass(frozen=True)
class PrebakeSettings:
    """Pre-baked harness images (bench/prebake): how they are named, built and kept."""
    prefix: str = "benchbake"        # image repository prefix, apart from the probes' benchprobe/
    build_jobs: int = 3              # builds at once (no CPU cap: keep them within the bench's cores)
    max_age_days: int = 14           # an older image still runs, with a warning: its apt lists go stale
    keep_days: int = 14              # prune keeps an unwanted image this long
    auto_build: bool = False         # the queue builds missing images before it starts an item
    prepull: bool = True             # a run pulls missing task images before its first arm
    host_reserve_gb: float = 25.0    # free space a build leaves on the drive holding Docker's disk
    vm_reserve_gb: float = 50.0      # free space a build leaves inside Docker's VM
    keep_images: bool = False        # schedule tasks: keep a task's images after its race instead of removing them


@dataclass(frozen=True)
class Config:
    dataset: str
    tasks: list[str]
    providers: dict[str, Provider]
    modes: dict[str, Mode]
    run: RunSettings
    baseline_arm: str
    arms: list[Arm]
    pricing: dict[str, dict[str, Price]]
    reference: dict[str, Any]
    effective: dict[str, Any]
    overrides: dict[str, Any]
    assistant: AssistantSettings = field(default_factory=AssistantSettings)
    designs: DesignLimits = field(default_factory=DesignLimits)
    stats: StatsSettings = field(default_factory=StatsSettings)
    queue: QueueSettings = field(default_factory=QueueSettings)
    report: dict[str, Any] = field(default_factory=dict)
    ledger: LedgerSettings = field(default_factory=LedgerSettings)
    prebake: PrebakeSettings = field(default_factory=PrebakeSettings)
    # The model REPORT compares other models with when it holds the harness fixed ('provider:model').
    baseline_model: str = DEFAULT_BASELINE_MODEL

    def arm(self, name: str) -> Arm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise ConfigError(f"No arm named '{name}'")

    def harbor_model(self, arm: Arm) -> str:
        return self.providers[arm.provider].model_prefix + arm.model

    def price(self, provider: str, model_name: str) -> Price | None:
        """Look up a price by the name a harness reports, which may carry a
        provider prefix or a date suffix the pricing table leaves off."""
        table = self.pricing.get(provider, {})
        name = model_name.split("/")[-1]
        if name in table:
            return table[name]
        keys = [key for key in table if is_variant(name, key)]
        return table[max(keys, key=len)] if keys else None

    def price_exact(self, provider: str, model_name: str) -> Price | None:
        """Only the table's own key: for planning, where a configured model
        id should have a row of its own."""
        return self.pricing.get(provider, {}).get(model_name.split("/")[-1])


# Reading -------------------------------------------------------------------

def _read_yaml(path: Path, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ConfigError(f"{path.name} does not exist")
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must be a mapping")
    return data


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge mappings. A None in the overrides removes the key."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            # Merging into {} drops None tombstones at every level, so a removal
            # aimed at a key the base does not have can never become a value.
            out[key] = _merge({}, value) if isinstance(value, dict) else copy.deepcopy(value)
    return out


def _merge_arm(arm: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Options merge one level deep. The dashboard sends an object-valued
    option whole, so it replaces the base value whole: a key removed in the
    editor stays removed, and a null inside it is a value, not a removal."""
    over = dict(over)
    has_options = "options" in over
    opt_over = over.pop("options", None)
    out = _merge(arm, over)
    if not has_options:
        return out
    if opt_over is None:
        out.pop("options", None)
        return out
    if not isinstance(opt_over, dict):
        raise ConfigError(f"Overrides for arm '{arm.get('name')}': options must be a mapping")
    opts = dict(arm.get("options") or {})
    for key, value in opt_over.items():
        if value is None:
            opts.pop(key, None)
        else:
            opts[key] = copy.deepcopy(value)
    out["options"] = opts
    return out


def _effective(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    arm_over = over.get("arms") or {}
    if not isinstance(arm_over, dict):
        raise ConfigError("Overrides: arms must map arm names to settings")
    base_arms = base.get("arms") or []
    if not isinstance(base_arms, list) or not all(isinstance(a, dict) for a in base_arms):
        raise ConfigError("arms must be a list of mappings, one per arm")
    rest = {k: v for k, v in over.items() if k not in ("arms", "modes")}
    if "modes" in over:
        mode_over = over["modes"]
        if isinstance(mode_over, dict):
            # Like arms, overrides only reach modes the base defines. A mode
            # renamed in YAML must not come back as a ghost from its fragment.
            base_modes = base.get("modes") or {}
            rest["modes"] = {k: v for k, v in mode_over.items() if k in base_modes}
        else:
            rest["modes"] = mode_over
    merged = _merge(base, rest)
    arms = []
    for arm in base_arms:
        ov = arm_over.get(arm.get("name"))
        if ov is None:
            ov = {}
        elif not isinstance(ov, dict):
            raise ConfigError(f"Overrides for arm '{arm.get('name')}' must be a mapping")
        if "name" in ov and ov["name"] != arm.get("name"):
            # Overrides are keyed by name; a rename would cut the arm off from
            # its own entry, from baseline_arm and from orphan detection.
            raise ConfigError(f"Overrides for arm '{arm.get('name')}' cannot rename it; rename it in the YAML")
        arms.append(_merge_arm(arm, ov))
    merged["arms"] = arms
    return merged


def orphaned_overrides(overrides: Mapping[str, Any], base: Mapping[str, Any] | None = None) -> dict[str, list[str]]:
    """Override entries for arms or modes the base config no longer has, so
    the dashboard can show them before a reused name picks them up again."""
    base = base_config() if base is None else base
    arm_names = {a.get("name") for a in base.get("arms") or [] if isinstance(a, dict)}
    mode_names = set(base.get("modes") or {})
    arm_over = overrides.get("arms") if isinstance(overrides.get("arms"), dict) else {}
    mode_over = overrides.get("modes") if isinstance(overrides.get("modes"), dict) else {}
    return {
        "arms": sorted(k for k in arm_over if k not in arm_names),
        "modes": sorted(k for k in mode_over if k not in mode_names),
    }


def read_tasks(path: Path) -> list[str]:
    if not path.exists():
        return []
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            tasks.append(name)
    return tasks


# Validation ----------------------------------------------------------------

def _positive(value: Any, label: str, kind: type = float, allow_none: bool = False) -> Any:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a number")
    try:
        number = kind(value)
        # A 400-digit int converts fine but cannot become a float.
        finite = math.isfinite(number)
    except (TypeError, ValueError, OverflowError):
        raise ConfigError(f"{label} must be a number") from None
    # NaN fails every comparison and inf never trips a budget, so neither can be a limit.
    if not finite or number <= 0:
        raise ConfigError(f"{label} must be a finite number greater than zero")
    return number


def _number_in(value: Any, label: str, low: float, high: float, hint: str = "",
               kind: type = float) -> float | int:
    number = _positive(value, label, kind)
    if not low <= number <= high:
        raise ConfigError(f"{label} must be between {low:g} and {high:g}{hint}")
    return number


def _rate(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ConfigError(f"{label} must be a number") from None
    if not math.isfinite(number) or number < 0:
        raise ConfigError(f"{label} must be a finite number, zero or more")
    return number


def _flag(value: Any, default: bool, label: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be true or false")
    return value


def _choice(value: Any, default: str, choices: tuple[str, ...], label: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"{label} must be one of {', '.join(choices)} (got {value!r})")
    return value


def _effort(value: Any, default: str, label: str) -> str:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ConfigError(f"{label}: quote the value, e.g. effort: 'off' "
                          "(YAML reads unquoted off/on/yes/no as true/false)")
    return str(value)


def _check_model_spec(arm: str, options: Mapping[str, Any], harbor_model: str) -> None:
    """A strands factory_kwargs.model_spec takes precedence over the model
    Harbor builds, so if it drifts from the arm's model every trial calls one
    model while run.json, the ledger and the budget price another. Compared
    the way bench_strands_factory compares it at run time."""
    fk = options.get("factory_kwargs")
    if not isinstance(fk, Mapping) or fk.get("model_spec") is None:
        return
    model_spec = fk["model_spec"]
    if not isinstance(model_spec, str) or model_spec.strip().lower() != harbor_model.lower():
        raise ConfigError(f"Arm '{arm}': factory_kwargs.model_spec is '{model_spec}' but the arm runs "
                          f"'{harbor_model}'; change both together")


def check_model_spec(arm: str, options: Mapping[str, Any], harbor_model: str) -> None:
    """Public name for designs.validate, which applies the same rule to design arms."""
    _check_model_spec(arm, options, harbor_model)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _section(raw: dict[str, Any], name: str, known: tuple[str, ...]) -> dict[str, Any]:
    spec = _mapping(raw.get(name), name)
    unknown = sorted(set(spec) - set(known))
    if unknown:
        raise ConfigError(f"Unknown {name} settings: {', '.join(map(str, unknown))}")
    return spec


_PRICE_KEYS = ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h")


def parse_pricing(raw: Mapping[str, Any]) -> dict[str, dict[str, Price]]:
    table: dict[str, dict[str, Price]] = {}
    for provider, models in (raw or {}).items():
        if models is None:
            models = {}
        if not isinstance(models, Mapping):
            raise ConfigError(f"pricing.yaml: '{provider}' must map model ids to prices")
        rows = {}
        for model, p in models.items():
            label = f"pricing.yaml {provider}.{model}"
            if not isinstance(p, Mapping):
                raise ConfigError(f"{label} must be a mapping with input and output")
            unknown = sorted(set(map(str, p)) - set(_PRICE_KEYS))
            if unknown:
                raise ConfigError(f"{label} has unknown rates: {', '.join(unknown)}")
            optional = {key: None if p.get(key) is None else _rate(p[key], f"{label}.{key}")
                        for key in ("cache_read", "cache_write_5m", "cache_write_1h")}
            rows[str(model)] = Price(
                input=_rate(p.get("input"), f"{label}.input"),
                output=_rate(p.get("output"), f"{label}.output"),
                **optional,
            )
        table[str(provider)] = rows
    return table


def read_pricing(path: Path = PRICING) -> dict[str, dict[str, Price]]:
    return parse_pricing(_read_yaml(path))


def read_reference(path: Path = REFERENCE) -> dict[str, Any]:
    """reference.yaml as written; a direction other than the two the report
    can read is a config error, not a silent guess at which way it runs."""
    reference = _read_yaml(path)
    try:
        stats.reference_direction(reference)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    return reference


# Sections ------------------------------------------------------------------

def _assistant(raw: dict[str, Any]) -> AssistantSettings:
    d = AssistantSettings()
    spec = _section(raw, "assistant", tuple(d.__dataclass_fields__))
    fallbacks = spec.get("fallbacks")
    if isinstance(fallbacks, bool):
        # Unquoted `off`/`no` arrive as False; for this switch that can only mean off.
        fallbacks = "default" if fallbacks else "off"
    return AssistantSettings(
        enabled=_flag(spec.get("enabled"), d.enabled, "assistant.enabled"),
        effort=_choice(spec.get("effort"), d.effort, ASSISTANT_EFFORTS, "assistant.effort"),
        # Opus 5.5 returns at most 128K output tokens per request.
        max_tokens=_number_in(spec.get("max_tokens", d.max_tokens), "assistant.max_tokens", 1, 128000, kind=int),
        max_steps=_positive(spec.get("max_steps", d.max_steps), "assistant.max_steps", int),
        per_turn_cap_usd=_positive(spec.get("per_turn_cap_usd", d.per_turn_cap_usd), "assistant.per_turn_cap_usd"),
        daily_cap_usd=_positive(spec.get("daily_cap_usd", d.daily_cap_usd), "assistant.daily_cap_usd"),
        fallbacks=_choice(fallbacks, d.fallbacks, ASSISTANT_FALLBACKS, "assistant.fallbacks"),
    )


def _import_paths(raw: dict[str, Any]) -> dict[str, str]:
    """designs.import_paths: path -> family, or path -> {family: ...}."""
    spec = _mapping(_mapping(raw.get("designs"), "designs").get("import_paths"), "designs.import_paths")
    out = {}
    for path, value in spec.items():
        label = f"designs.import_paths['{path}']"
        if not is_import_path(path):
            raise ConfigError(f"{label}: not a module.path:Class import path")
        if value is None:
            family = path
        elif isinstance(value, str):
            family = value
        elif isinstance(value, dict):
            unknown = sorted(set(value) - {"family"})
            if unknown:
                raise ConfigError(f"{label}: unknown keys {', '.join(map(str, unknown))} (only family)")
            family = value.get("family") or path
        else:
            raise ConfigError(f"{label} must be a family name or {{family: ...}}")
        if not isinstance(family, str) or not family.strip():
            raise ConfigError(f"{label}: family must be a non-empty string")
        out[path] = family.strip()
    return out


def _designs(raw: dict[str, Any], import_paths: dict[str, str], arms: list[Arm]) -> DesignLimits:
    d = DesignLimits()
    spec = _section(raw, "designs", tuple(d.__dataclass_fields__))
    extra = spec.get("extra_harnesses") or []
    if not isinstance(extra, list) or not all(isinstance(x, str) and x.strip() for x in extra):
        raise ConfigError("designs.extra_harnesses must be a list of Harbor agent names")
    for name in extra:
        if is_import_path(name):
            raise ConfigError(f"designs.extra_harnesses: put the import path '{name}' "
                              "under designs.import_paths with its family")
    # Every import path a config arm already runs is allowed in designs too,
    # with the same family, so the two never pool apart.
    paths = dict(import_paths)
    for arm in arms:
        if arm.kind != "import_path":
            continue
        if paths.setdefault(arm.agent, arm.family) != arm.family:
            raise ConfigError(f"Arm '{arm.name}' gives {arm.agent} the family '{arm.family}', "
                              f"but designs.import_paths says '{paths[arm.agent]}'")
    limits = DesignLimits(
        max_budget_usd=_positive(spec.get("max_budget_usd", d.max_budget_usd), "designs.max_budget_usd"),
        max_arms=_positive(spec.get("max_arms", d.max_arms), "designs.max_arms", int),
        max_attempts=_positive(spec.get("max_attempts", d.max_attempts), "designs.max_attempts", int),
        warn_containers=_positive(spec.get("warn_containers", d.warn_containers), "designs.warn_containers", int),
        max_containers=_positive(spec.get("max_containers", d.max_containers), "designs.max_containers", int),
        extra_harnesses=tuple(x.strip() for x in extra),
        import_paths=paths,
    )
    if limits.warn_containers > limits.max_containers:
        raise ConfigError("designs.warn_containers must not be above designs.max_containers")
    return limits


def _stats(raw: dict[str, Any]) -> StatsSettings:
    d = StatsSettings()
    spec = _section(raw, "stats", tuple(d.__dataclass_fields__))
    tokens_in = _mapping(spec.get("default_trial_tokens"), "stats.default_trial_tokens")
    unknown = sorted(set(tokens_in) - set(d.default_trial_tokens))
    if unknown:
        raise ConfigError(f"Unknown stats.default_trial_tokens keys: {', '.join(map(str, unknown))}")
    tokens = {**d.default_trial_tokens, **tokens_in}
    share = tokens["cache_share"]
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 <= share < 1:
        raise ConfigError("stats.default_trial_tokens.cache_share must be a fraction from 0 up to 1")
    return StatsSettings(
        confidence=_number_in(spec.get("confidence", d.confidence), "stats.confidence", 0.80, 0.99,
                              ", e.g. 0.90 for 90%"),
        target_halfwidth=_number_in(spec.get("target_halfwidth", d.target_halfwidth), "stats.target_halfwidth",
                                    0.01, 1.0, ", e.g. 0.25 for +/-25%"),
        prior_sd_log=_positive(spec.get("prior_sd_log", d.prior_sd_log), "stats.prior_sd_log"),
        default_trial_tokens={
            "input": _positive(tokens["input"], "stats.default_trial_tokens.input", int),
            "cache_share": float(share),
            "output": _positive(tokens["output"], "stats.default_trial_tokens.output", int),
        },
    )


def _queue(raw: dict[str, Any]) -> QueueSettings:
    d = QueueSettings()
    spec = _section(raw, "queue", tuple(d.__dataclass_fields__))
    return QueueSettings(
        on_failure=_choice(spec.get("on_failure"), d.on_failure, QUEUE_ON_FAILURE, "queue.on_failure"),
        global_cap_usd=_positive(spec.get("global_cap_usd"), "queue.global_cap_usd", allow_none=True),
    )


# Docker's repository grammar: lowercase path components joined by '/'.
_IMAGE_PREFIX = re.compile(r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*(?:/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)*")


def _prebake(raw: dict[str, Any]) -> PrebakeSettings:
    d = PrebakeSettings()
    spec = _section(raw, "prebake", tuple(d.__dataclass_fields__))
    prefix = spec.get("prefix", d.prefix)
    if not isinstance(prefix, str) or not _IMAGE_PREFIX.fullmatch(prefix.strip("/")):
        raise ConfigError("prebake.prefix must be a Docker repository prefix: lowercase letters, digits, '.', "
                          "'_', '-' and '/'")
    prefix = prefix.strip("/")
    if prefix.split("/")[0] in ("benchprobe", "benchbake-staging"):
        # The probes' images and the builder's staging tags are pruned by these names.
        raise ConfigError(f"prebake.prefix '{prefix}' is reserved")
    return PrebakeSettings(
        prefix=prefix,
        build_jobs=_number_in(spec.get("build_jobs", d.build_jobs), "prebake.build_jobs", 1, 16, kind=int),
        max_age_days=_positive(spec.get("max_age_days", d.max_age_days), "prebake.max_age_days", int),
        keep_days=_positive(spec.get("keep_days", d.keep_days), "prebake.keep_days", int),
        auto_build=_flag(spec.get("auto_build"), d.auto_build, "prebake.auto_build"),
        prepull=_flag(spec.get("prepull"), d.prepull, "prebake.prepull"),
        host_reserve_gb=_positive(spec.get("host_reserve_gb", d.host_reserve_gb), "prebake.host_reserve_gb"),
        vm_reserve_gb=_positive(spec.get("vm_reserve_gb", d.vm_reserve_gb), "prebake.vm_reserve_gb"),
        keep_images=_flag(spec.get("keep_images"), d.keep_images, "prebake.keep_images"),
    )


def prebake_mode(value: Any, default: str, label: str) -> str:
    """run.prebake. YAML reads an unquoted off as false, which can only mean off."""
    if value is False:
        return PREBAKE_OFF
    if value is True:
        raise ConfigError(f"{label}: quote the value, e.g. prebake: 'prefer' (YAML reads unquoted on/yes as true)")
    return _choice(value, default, PREBAKE_MODES, label)


def setup_timeout_multiplier(value: Any, default: str | float, label: str) -> str | float:
    """run.setup_timeout_multiplier: 'auto' or a number greater than zero."""
    if value is None:
        return default
    if value == SETUP_TIMEOUT_AUTO:
        return SETUP_TIMEOUT_AUTO
    if isinstance(value, str):
        raise ConfigError(f"{label} must be auto or a number")
    return _positive(value, label)


def _ledger(raw: dict[str, Any]) -> LedgerSettings:
    spec = _section(raw, "ledger", tuple(LedgerSettings.__dataclass_fields__))
    roots = spec.get("extra_scan_roots") or []
    if not isinstance(roots, list) or not all(isinstance(r, str) and r.strip() for r in roots):
        raise ConfigError("ledger.extra_scan_roots must be a list of folder paths")
    return LedgerSettings(extra_scan_roots=tuple(ROOT / r.strip() for r in roots))


# Loading -------------------------------------------------------------------

def load_config(base_path: Path | None = None, overrides_path: Path | None = None) -> Config:
    """Load the profile's config. Paths default to this profile's files; tests
    pass their own."""
    base = _read_yaml(base_path or BASE_CONFIG, required=True)
    overrides = _read_yaml(overrides_path or OVERRIDES)
    raw = _effective(base, overrides)

    providers = {}
    for name, spec in _mapping(raw.get("providers"), "providers").items():
        spec = _mapping(spec, f"providers.{name}")
        if not spec.get("env_key"):
            raise ConfigError(f"Provider '{name}' needs an env_key")
        providers[name] = Provider(
            name=name,
            env_key=str(spec["env_key"]),
            model_prefix=str(spec.get("model_prefix") or ""),
            enabled=_flag(spec.get("enabled"), True, f"providers.{name}.enabled"),
        )

    tasks = read_tasks(ROOT / str(raw.get("tasks_file") or "config/tasks.txt"))

    modes = {}
    for name, spec in _mapping(raw.get("modes"), "modes").items():
        spec = _mapping(spec, f"modes.{name}")
        mode = Mode(
            n_tasks=_positive(spec.get("n_tasks"), f"modes.{name}.n_tasks", int),
            n_attempts=_positive(spec.get("n_attempts", MODE_DEFAULTS["n_attempts"]),
                                 f"modes.{name}.n_attempts", int),
        )
        if tasks and mode.n_tasks > len(tasks):
            raise ConfigError(
                f"modes.{name}.n_tasks is {mode.n_tasks} but the task file lists {len(tasks)}"
            )
        modes[name] = mode

    run = _mapping(raw.get("run"), "run")
    d = RUN_DEFAULTS
    settings = RunSettings(
        effort=_effort(run.get("effort"), d["effort"], "run.effort"),
        n_concurrent=_positive(run.get("n_concurrent", d["n_concurrent"]), "run.n_concurrent", int),
        parallel_arms=_positive(run.get("parallel_arms", d["parallel_arms"]), "run.parallel_arms", int),
        budget_usd=_positive(run.get("budget_usd"), "run.budget_usd"),
        per_trial_cap_usd=_positive(run.get("per_trial_cap_usd", d["per_trial_cap_usd"]),
                                    "run.per_trial_cap_usd", allow_none=True),
        max_turns=_positive(run.get("max_turns", d["max_turns"]), "run.max_turns", int, allow_none=True),
        timeout_multiplier=_positive(run.get("timeout_multiplier", d["timeout_multiplier"]),
                                     "run.timeout_multiplier"),
        prebake=prebake_mode(run.get("prebake"), d["prebake"], "run.prebake"),
        setup_timeout_multiplier=setup_timeout_multiplier(run.get("setup_timeout_multiplier"),
                                                          d["setup_timeout_multiplier"],
                                                          "run.setup_timeout_multiplier"),
        env_build_timeout_multiplier=_positive(run.get("env_build_timeout_multiplier",
                                                       d["env_build_timeout_multiplier"]),
                                               "run.env_build_timeout_multiplier"),
        schedule=_choice(run.get("schedule"), d["schedule"], SCHEDULES, "run.schedule"),
        max_containers=_positive(run.get("max_containers", d["max_containers"]), "run.max_containers", int),
    )

    import_paths = _import_paths(raw)
    arms, seen = [], set()
    for spec in raw.get("arms") or []:
        name = spec.get("name")
        if not name or not isinstance(name, str) or name in seen:
            raise ConfigError(f"Every arm needs a unique name (got '{name}')")
        seen.add(name)
        if not isinstance(spec.get("provider"), str) or spec["provider"] not in providers:
            raise ConfigError(f"Arm '{name}' uses unknown provider '{spec.get('provider')}'")
        agent = spec.get("agent")
        if not agent or not isinstance(agent, str) or not spec.get("model"):
            raise ConfigError(f"Arm '{name}' needs an agent and a model")
        if ":" in agent and not is_import_path(agent):
            raise ConfigError(f"Arm '{name}': agent '{agent}' is neither a Harbor agent name "
                              "nor a module.path:Class import path")
        family = spec.get("family")
        if family is not None and (not isinstance(family, str) or not family.strip()):
            raise ConfigError(f"Arm '{name}': family must be a non-empty name")
        options = _mapping(spec.get("options"), f"arms.{name}.options")
        if not all(isinstance(k, str) for k in options):
            raise ConfigError(f"Arm '{name}': option names must be strings")
        _check_model_spec(name, options, providers[spec["provider"]].model_prefix + str(spec["model"]))
        arms.append(Arm(
            name=name,
            agent=agent,
            provider=spec["provider"],
            model=str(spec["model"]),
            enabled=_flag(spec.get("enabled"), ARM_DEFAULTS["enabled"], f"arms.{name}.enabled"),
            effort=_effort(spec.get("effort"), ARM_DEFAULTS["effort"], f"arms.{name}.effort"),
            options=dict(options),
            family=(family or "").strip() or import_paths.get(agent) or agent,
        ))

    baseline = raw.get("baseline_arm") or (arms[0].name if arms else "")
    if baseline and baseline not in seen:
        raise ConfigError(f"baseline_arm '{baseline}' is not an arm")

    report = _mapping(raw.get("report"), "report")
    baseline_model = raw.get("baseline_model", DEFAULT_BASELINE_MODEL)
    provider, _, model = baseline_model.partition(":") if isinstance(baseline_model, str) else ("", "", "")
    if not provider.strip() or not model.strip():
        raise ConfigError(f"baseline_model must be provider:model, e.g. {DEFAULT_BASELINE_MODEL} "
                          f"(got {baseline_model!r})")

    limits = _designs(raw, import_paths, arms)
    if settings.max_containers > limits.max_containers:
        raise ConfigError(f"run.max_containers ({settings.max_containers}) must not be above designs.max_containers "
                          f"({limits.max_containers})")
    return Config(
        dataset=str(raw.get("dataset") or ""),
        tasks=tasks,
        providers=providers,
        modes=modes,
        run=settings,
        baseline_arm=baseline,
        baseline_model=f"{provider.strip()}:{model.strip()}",
        arms=arms,
        pricing=read_pricing(PRICING),
        reference=read_reference(REFERENCE),
        effective=raw,
        overrides=overrides,
        assistant=_assistant(raw),
        designs=limits,
        stats=_stats(raw),
        queue=_queue(raw),
        report=copy.deepcopy(report),
        ledger=_ledger(raw),
        prebake=_prebake(raw),
    )


def base_config() -> dict[str, Any]:
    return _read_yaml(BASE_CONFIG)


# Writing -------------------------------------------------------------------

def write_overrides(overrides: dict[str, Any]) -> Config:
    """Write the dashboard's changes and reload. If the reload fails for any
    reason, the previous file is put back and the error re-raised."""
    path = OVERRIDES
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    cleaned = _prune(overrides)
    header = f"# Written by the dashboard. Values here override config/{BASE_CONFIG.name}.\n"
    _write_or_busy(path, header + yaml.safe_dump(cleaned, sort_keys=False, allow_unicode=True))
    try:
        return load_config()
    except BaseException as exc:
        try:
            if previous is None:
                _unlink_with_retry(path)
            else:
                _write_or_busy(path, previous)
        except Exception as restore_exc:
            exc.add_note(f"{path.name} could not be restored: {restore_exc}")
        raise


def _write_or_busy(path: Path, text: str) -> None:
    if not fsutil.atomic_write_text(path, text):
        raise PermissionError(f"{path.name} is busy (another program has it open). Try again.")


def _unlink_with_retry(path: Path, patience: float = 2.0) -> None:
    deadline = time.monotonic() + patience
    while True:
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def _prune(overrides: dict[str, Any]) -> dict[str, Any]:
    """Drop empty mappings left by resets, at the structural levels only.

    Option values are never looked inside: an object-valued option the user
    set to {} is a value, not an empty override.
    """
    out: dict[str, Any] = {}
    for key, value in overrides.items():
        if key == "arms" and isinstance(value, dict):
            arms = {}
            for name, arm in value.items():
                if isinstance(arm, dict):
                    arm = {k: v for k, v in arm.items() if not (k == "options" and v == {})}
                if arm != {}:
                    arms[name] = arm
            value = arms
        elif isinstance(value, dict):
            # run, modes.<name>, and any other section: one level of empties.
            value = {k: v for k, v in value.items() if v != {}}
        if value != {}:
            out[key] = value
    return out
