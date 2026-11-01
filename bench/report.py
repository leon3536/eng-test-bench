"""The pooled report: every matching trial in the ledger, paired by task.

The user runs, runs again, and then asks "where are we?", so the report is
not one run's results but a live query over the whole ledger. Each harness
is compared with the baseline harness (Claude Code) on the same provider and
model, task by task on identical task content, pooling every run and attempt
that matches the filters. The task is the unit: repeats sharpen a task's
mean but never add a pair. With compare 'models' the other axis is held:
each harness stays fixed, and each model it ran is compared with the
baseline model (config baseline_model) on the same task content, with no
published-figure verdicts, since those figures compare harnesses.

Judgements are made here, at report time, from the ledger's facts: which
trials count (a setup failure never counts as a cheap run, and a trial that
ran but recorded no usage counts for pass rate only), what a trial cost
under today's prices, whether a figure is precise enough to show, and
whether it agrees with the published figures in config/reference.yaml. The
same request on the same ledger and prices gives the same numbers, because
every random draw is seeded from the request, and the run-design assistant
reads its coverage answer from this same code.

Pure apart from reading the ledger, which happens only through
Ledger.query, runs, effective_exclusions and version.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import re
import statistics
import threading
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from typing import Any

from . import classify, extract, harnesses, pricing, settings, stats
from .fsutil import canonical_json
from .ledger import Filters, parse_filters
from .settings import Config

__all__ = [
    "Options", "parse_options", "parse_filters", "options_dict", "filters_dict", "build_report", "build_coverage",
    "build_pairs", "export_rows", "analysis_rows", "ledger_fingerprint", "prepare_rows", "group_key",
    "pick_baselines", "build_cells", "compare_metric", "compare_pass", "reference_for", "verdict", "coverage_for",
    "caveats_for", "headline", "setup_matches", "task_scope", "tasks_file", "saved_report", "harness_key",
    "axis_key", "same_basis", "flag_caveats", "fallback_model",
]

GROUP_BY = ("family", "implementation", "config")
MATCH_ON = ("digest", "name")
PRICING_MODES = ("current", "as_ingested", "reported")
# What a comparison varies: the harness on one model, or the model under one harness.
COMPARE_AXES = ("harnesses", "models")
DEFAULT_BASELINE_MODEL = settings.DEFAULT_BASELINE_MODEL
METRICS = ("tokens_total", "tokens_uncached", "cost")
ESTIMATORS = ("ratio_of_means", "geomean")
EXPORT_LEVELS = ("trials", "pairs", "comparisons")
# Which trials by how their harness got installed (the ledger's setup_mode): stock takes a fallback too,
# since Harbor installed the harness in that trial; one with no setup recorded counts under any only.
SETUP_FILTERS = ("any", "stock", "prebaked")
_SETUP_MATCHES = {"stock": ("stock", "fallback"), "prebaked": ("prebaked",)}

# The harness the published multiples were measured against; a figure only
# applies to a comparison whose baseline is this family.
REFERENCE_BASELINE = "claude-code"
DEFAULT_APPLIES_TO = ("anthropic:claude-*",)
DEFAULT_TOLERANCE = 1.5
DEFAULT_SEED = 20261028
MIN_SD_PAIRS = 3            # pairs before the observed spread replaces the prior
PRIMARY_SHARE_MIN = 0.95
REPORTED_RATIO_BAND = (0.5, 2.0)
CACHE_SIZE = 32
# A verdict must survive each side's trials cut short or with no usage recorded
# costing this many times that side's usual spend on the task: such a trial
# often ran to the time limit, typically 1.5 to 3 times a finished trial's time.
UNRECORDED_FACTOR = 2.0
# Failures of the bench, the sandbox or the provider: a reliability figure of
# their own, never averaged into a harness's cost.
RELIABILITY_OUTCOMES = ("setup_failure", "provider_failure", "infra")
STOPPED_RUN_STATUSES = ("budget_stopped", "cancelled")

NOT_APPLICABLE_CAVEAT = "Published figures were measured on Claude models; this is shown for position only."
MODELS_NO_VERDICT = "Published figures compare harnesses, not models, so model comparisons get no verdict."
VERDICTS_MET = ("within", "overlaps", "above", "below")
OPENROUTER_CAVEAT = "OpenRouter bills the routed provider; harness-reported cost may be the truer figure"

# The row contract the planner and the assistant read (analysis_rows).
ANALYSIS_KEYS = (
    "trial_key", "run_key", "profile", "run_id", "arm", "family", "implementation", "config_hash", "config_label",
    "provider", "model", "effort", "options", "task", "task_digest", "outcome", "outcome_reason", "harbor_cause",
    "exception_type", "valid_tokens", "valid_cost", "valid_reward", "excluded_reason", "n_input", "n_cache",
    "n_output", "usage_status", "tokens_total", "tokens_uncached", "cost_usd", "reported_cost_usd",
    "cost_at_ingest_usd", "passed", "agent_minutes", "total_minutes", "started_at",
    # How the harness got installed, its setup minutes and the trial's minutes without them, so the
    # estimates can plan minutes for the setup an arm will have (a stock install or a pre-baked image).
    "setup_mode", "setup_minutes", "minutes_ex_setup",
    # What reuse.py matches a trial on beyond its cell: the dataset, its snapshot and the Harbor that ran it,
    # the share of its tokens spent on the arm's model, and a person's exclusion whatever report.exclusions says.
    "dataset", "dataset_snapshot", "harbor_version", "primary_share", "excluded_by_hand",
)


# Options ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Options:
    group_by: str = "family"
    split_harness_versions: bool = False
    baseline_family: str = REFERENCE_BASELINE
    baseline_config: str | None = None
    match_on: str = "digest"
    same_run_only: bool = False
    crashes: str = "include"
    misconfigured_cap: str = "exclude"
    apply_exclusions: bool = True
    pricing: str = "current"
    ci_level: float = 0.90
    resamples: int = 2000
    seed: int = DEFAULT_SEED
    min_pairs_estimate: int = 3
    min_pairs_verdict: int = 8
    target_ratio_halfwidth: float = 1.25
    target_pass_delta_pp: float = 5.0
    prior_sd_log: float = 0.6
    include_pairs: bool = False
    setup: str = "any"      # SETUP_FILTERS
    compare: str = "harnesses"                      # COMPARE_AXES
    baseline_model: str = DEFAULT_BASELINE_MODEL    # 'provider:model', for compare 'models'


# Where each option sits in a request body ({"pairing": {"match_on": ...}}).
_REQUEST_PATHS: dict[str, tuple[str, ...]] = {
    "group_by": ("group_by",),
    "split_harness_versions": ("split_harness_versions",),
    "baseline_family": ("baseline", "family"),
    "baseline_config": ("baseline", "config"),
    "match_on": ("pairing", "match_on"),
    "same_run_only": ("pairing", "same_run_only"),
    "crashes": ("policy", "crashes"),
    "misconfigured_cap": ("policy", "misconfigured_cap"),
    "apply_exclusions": ("policy", "exclusions"),
    "pricing": ("pricing",),
    "ci_level": ("ci_level",),
    "resamples": ("bootstrap", "resamples"),
    "seed": ("bootstrap", "seed"),
    "min_pairs_estimate": ("min_pairs", "estimate"),
    "min_pairs_verdict": ("min_pairs", "verdict"),
    "target_ratio_halfwidth": ("coverage_target", "ratio_halfwidth"),
    "target_pass_delta_pp": ("coverage_target", "pass_delta_pp"),
    "prior_sd_log": ("coverage_target", "prior_sd_log"),
    "include_pairs": ("include_pairs",),
    "setup": ("setup",),
    "compare": ("compare",),
    "baseline_model": ("baseline", "model"),
}

# The flat keys a profile's `report:` section may set. The precision target,
# the prior spread and the confidence level default from `stats:`, so the
# report and the planner quote the same figures.
_CONFIG_KEYS: dict[str, str] = {
    "group_by": "group_by", "split_harness_versions": "split_harness_versions",
    "baseline_family": "baseline_family", "baseline_config": "baseline_config",
    "crashes": "crashes", "misconfigured_cap": "misconfigured_cap", "exclusions": "apply_exclusions",
    "same_run_only": "same_run_only", "match_on": "match_on", "ci_level": "ci_level",
    "bootstrap_resamples": "resamples", "seed": "seed", "min_pairs_estimate": "min_pairs_estimate",
    "min_pairs_verdict": "min_pairs_verdict", "pricing": "pricing", "target_pass_delta_pp": "target_pass_delta_pp",
    "compare": "compare",
}


def _choice(value: Any, choices: Sequence[str], label: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{label} must be one of {', '.join(choices)} (got {value!r})")
    return value


def _flag(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false")
    return value


def _number(value: Any, label: str, low: float, high: float, *, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (integer and not isinstance(value, int)):
        raise ValueError(f"{label} must be {'a whole number' if integer else 'a number'}")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be between {low:g} and {high:g} (got {value!r})")
    return int(value) if integer else float(value)


def _coerce(name: str, value: Any, label: str) -> Any:
    if name == "group_by":
        return _choice(value, GROUP_BY, label)
    if name in ("split_harness_versions", "same_run_only", "apply_exclusions", "include_pairs"):
        return _flag(value, label)
    if name == "baseline_family":
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise ValueError(f"{label} must be a harness family name, e.g. claude-code")
        return value.strip()
    if name == "baseline_config":
        if value is None:
            return None
        if not isinstance(value, str) or not value or len(value) > 64 or any(
                ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"{label} must be a config hash (lowercase hex), or null")
        return value
    if name == "match_on":
        return _choice(value, MATCH_ON, label)
    if name == "crashes":
        return _choice(value, ("include", "exclude"), label)
    if name == "misconfigured_cap":
        return _choice(value, ("exclude", "include"), label)
    if name == "pricing":
        return _choice(value, PRICING_MODES, label)
    if name == "ci_level":
        return _number(value, label, 0.80, 0.99)
    if name == "resamples":
        return _number(value, label, 500, 10000, integer=True)
    if name == "seed":
        return _number(value, label, 0, 2 ** 63 - 1, integer=True)
    if name in ("min_pairs_estimate", "min_pairs_verdict"):
        return _number(value, label, 2, 1000, integer=True)
    if name == "target_ratio_halfwidth":
        return _number(value, label, 1.01, 10.0)
    if name == "target_pass_delta_pp":
        return _number(value, label, 0.1, 100.0)
    if name == "prior_sd_log":
        return _number(value, label, 0.01, 5.0)
    if name == "setup":
        return _choice(value, SETUP_FILTERS, label)
    if name == "compare":
        return _choice(value, COMPARE_AXES, label)
    if name == "baseline_model":
        provider, _, model = value.partition(":") if isinstance(value, str) else ("", "", "")
        if not provider.strip() or not model.strip() or len(value) > 200:
            raise ValueError(f"{label} must be provider:model, e.g. {DEFAULT_BASELINE_MODEL}")
        return f"{provider.strip()}:{model.strip()}"
    raise ValueError(f"Unknown option {label}")


def _code_defaults(cfg: Config | None) -> dict[str, Any]:
    values = {f.name: f.default for f in dataclasses.fields(Options)}
    if cfg is None:
        return values
    try:
        arm = cfg.arm(cfg.baseline_arm)
        values["baseline_family"] = arm.family or arm.agent
    except Exception:  # no arms configured: the reference baseline
        pass
    if getattr(cfg, "baseline_model", None):
        values["baseline_model"] = _coerce("baseline_model", cfg.baseline_model, "baseline_model in the config")
    st = getattr(cfg, "stats", None)
    if st is not None:
        values["ci_level"] = float(st.confidence)
        values["target_ratio_halfwidth"] = 1.0 + float(st.target_halfwidth)
        values["prior_sd_log"] = float(st.prior_sd_log)
    return values


def _request_values(body: Mapping[str, Any]) -> list[tuple[str, str, Any]]:
    """(option, label, value) for every non-null value in a request body."""
    leaves: dict[str, str] = {}
    sections: dict[str, dict[str, str]] = defaultdict(dict)
    for name, path in _REQUEST_PATHS.items():
        if len(path) == 1:
            leaves[path[0]] = name
        else:
            sections[path[0]][path[1]] = name
    unknown = sorted(str(k) for k in body if k not in leaves and k not in sections)
    if unknown:
        raise ValueError(f"Unknown option: {', '.join(unknown)}")
    out = []
    for key, value in body.items():
        if value is None:
            continue
        if key in leaves:
            out.append((leaves[key], f"options.{key}", value))
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"options.{key} must be an object")
        bad = sorted(str(k) for k in value if k not in sections[key])
        if bad:
            raise ValueError(f"Unknown option: {', '.join(f'{key}.{k}' for k in bad)}")
        for sub, inner in value.items():
            if inner is not None:
                out.append((sections[key][sub], f"options.{key}.{sub}", inner))
    return out


def parse_options(body: Mapping | None, cfg: Config | None) -> Options:
    """Options from a request body. Precedence: the request, then the
    profile's `report:` section, then `stats:`, then code defaults.
    ValueError on an unknown key, a bad value or an out-of-range number."""
    values = _code_defaults(cfg)
    report_cfg = getattr(cfg, "report", None) or {}
    if not isinstance(report_cfg, Mapping):
        raise ValueError("The config's report section must be a mapping")
    for key, value in report_cfg.items():
        if key not in _CONFIG_KEYS:
            raise ValueError(f"Unknown report setting '{key}' in the config")
        if value is not None:
            name = _CONFIG_KEYS[key]
            values[name] = _coerce(name, value, f"report.{key} in the config")
    if body is not None:
        if not isinstance(body, Mapping):
            raise ValueError("options must be an object")
        for name, label, value in _request_values(body):
            values[name] = _coerce(name, value, label)
    if values["min_pairs_verdict"] < values["min_pairs_estimate"]:
        raise ValueError("min_pairs.verdict must be at least min_pairs.estimate")
    return Options(**values)


def options_dict(o: Options) -> dict[str, Any]:
    """Options in request form, every value filled: the normalised spec."""
    out: dict[str, Any] = {}
    for name, path in _REQUEST_PATHS.items():
        value = getattr(o, name)
        if len(path) == 1:
            out[path[0]] = value
        else:
            out.setdefault(path[0], {})[path[1]] = value
    return out


def filters_dict(f: Filters) -> dict[str, Any]:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(f).items()}


# Rows ------------------------------------------------------------------------------------

def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _manual_reason(row: Mapping[str, Any], trials: frozenset[str], runs: frozenset[str]) -> str | None:
    shown = _mapping(row.get("exclusion"))
    if row.get("trial_key") in trials:
        return f"excluded by hand: {shown.get('reason') or 'trial exclusion'}"
    if row.get("run_key") in runs:
        if shown.get("target_type") == "trial" and shown.get("action") == "include":
            return None  # this trial was put back inside an excluded run
        reason = shown.get("reason") if shown.get("target_type", "run") == "run" else None
        return f"run excluded by hand: {reason or 'run exclusion'}"
    return None


def _policy_reason(row: Mapping[str, Any], included: bool) -> str | None:
    if included:
        return None
    outcome = row.get("outcome") or ""
    detail = row.get("outcome_reason") or ""
    if outcome in ("agent_crash", "misconfigured_cap"):
        return f"{outcome} excluded by the report policy" + (f" ({detail})" if detail else "")
    return f"{outcome}: {detail}" if detail else outcome


def _current_cost(row: Mapping[str, Any], table: Mapping[str, Any] | None) -> tuple[float | None, list[str]]:
    usage = row.get("usage")
    if table is not None and isinstance(usage, Mapping):
        return pricing.price_usage(usage, row.get("provider") or "", table)
    return row.get("cost_usd"), list(row.get("unpriced_models") or [])


def _usable_reported(row: Mapping[str, Any]) -> float | None:
    """The harness's (or OpenRouter's billed) figure, unless this harness
    prices this provider's models at another provider's rates. $0 for a
    trial that used tokens is a harness that could not price the model
    (LiteLLM does this), not a billed cost."""
    reported = row.get("reported_cost_usd")
    used_tokens = bool((row.get("n_input") or 0) + (row.get("n_output") or 0))
    if (reported is not None and (float(reported) > 0 or not used_tokens)
            and not harnesses.reports_mispriced_cost(row.get("family") or "", row.get("provider") or "")):
        return float(reported)
    return None


def _chosen_cost(row: Mapping[str, Any], current: float | None, mode: str) -> float | None:
    if mode == "as_ingested":
        return row.get("cost_at_ingest_usd")
    if mode == "reported":
        reported = _usable_reported(row)
        if reported is not None:
            return reported
    return current


def fallback_model(row: Mapping[str, Any]) -> bool:
    """A trial that spent more than 1 - PRIMARY_SHARE_MIN of its tokens on
    models other than its arm's (a fallback model answered): its cost is
    not the arm's model's, so it never counts toward a cost multiple."""
    share = row.get("primary_share")
    return share is not None and share < PRIMARY_SHARE_MIN


def prepare_rows(rows: Sequence[Mapping[str, Any]], o: Options,
                 excl: tuple[frozenset[str], frozenset[str], tuple[int, ...]] | None,
                 table: Mapping[str, Any] | None) -> list[dict]:
    """Ledger rows with what the report needs attached: validity under the
    policy, token metrics, cost under the chosen pricing, the models without
    a price, and why a row is left out. Missing values stay None, and so do
    the tokens and cost of a trial whose usage was not recorded: Harbor's
    0/0/0 for it is not a $0 trial (nor is its cost at ingest)."""
    trials, runs = (excl[0], excl[1]) if excl and o.apply_exclusions else (frozenset(), frozenset())
    out = []
    for source in rows:
        row = dict(source)
        manual = _manual_reason(row, trials, runs)
        phases = _mapping(row.get("phases"))
        agent_sec, total_sec = phases.get("agent_exec_sec"), row.get("total_sec")
        setup_sec = phases.get("agent_setup_sec")
        status = classify.usage_status({**row, "agent_exec_sec": row.get("agent_exec_sec", agent_sec)})
        validity = classify.validity(row["outcome"], crashes=o.crashes, misconfigured_cap=o.misconfigured_cap,
                                     usage=status)
        n_input, n_cache, n_output = row.get("n_input"), row.get("n_cache"), row.get("n_output")
        measured = status != "unmeasured" and (n_input is not None or n_output is not None)
        current, unpriced = _current_cost(row, table)
        cost = _chosen_cost(row, current, o.pricing) if status != "unmeasured" else None
        kept = manual is None
        fallback = fallback_model(row)
        row.update(
            usage_status=status, floor_set_aside=False, floor_cost=None, fallback_model=fallback,
            # Where cost came from: a pair takes the harnesses' own figures only when both sides have them.
            cost_reported=o.pricing == "reported" and cost is not None and _usable_reported(row) is not None,
            cost_at_ingest_usd=row.get("cost_at_ingest_usd") if status != "unmeasured" else None,
            tokens_total=(n_input or 0) + (n_output or 0) if measured else None,
            tokens_uncached=max((n_input or 0) - (n_cache or 0), 0) + (n_output or 0) if measured else None,
            cost_current=current, cost=cost, unpriced=unpriced,
            manual=manual, included=kept and validity.included,
            valid_tokens=kept and validity.tokens and measured,
            valid_cost=kept and validity.tokens and cost is not None and not fallback,
            valid_reward=kept and validity.reward and row.get("passed") is not None,
            excluded_reason=manual or _policy_reason(row, validity.included),
            # The latest action on the trial, else on its run (Ledger._trial_row), as _manual_reason decides it.
            excluded_by_hand=manual is not None or _mapping(row.get("exclusion")).get("action") == "exclude",
            agent_minutes=agent_sec / 60.0 if agent_sec is not None else None,
            total_minutes=total_sec / 60.0 if total_sec is not None else None,
            setup_mode=row.get("setup_mode"),
            setup_minutes=setup_sec / 60.0 if setup_sec is not None else None,
            # Without the harness's setup: 330-550 s stock for Claude Code, about 1 s pre-baked, so trials
            # set up either way compare (a trial that never reached agent setup had none).
            minutes_ex_setup=(max(total_sec - (setup_sec or 0.0), 0.0) / 60.0 if total_sec is not None else None),
        )
        out.append(row)
    _set_aside_low_floors(out, o)
    return out


def _set_aside_low_floors(rows: list[dict], o: Options) -> None:
    """A trial cut short that recorded fewer tokens than every uncut trial of
    its group on the same task ran longer than they did, yet records less:
    part of what it spent is missing (a model call hung, or the harness went
    on after the limit). Like a trial that recorded nothing, its reward
    counts and its tokens and cost do not, so the knife edge between no
    completed call and one does not decide whether a near-empty floor
    halves a mean. With no uncut trial to compare against, the floor stays,
    and the verdict check in compare_metric covers it."""
    least: dict[tuple, float] = {}
    for r in rows:
        if r["valid_tokens"] and r["usage_status"] == "full":
            k = (group_key(r, o), _pair_key(r, o))
            least[k] = min(least.get(k, math.inf), float(r["tokens_total"]))
    for r in rows:
        floor = r["valid_tokens"] and r["usage_status"] == "lower_bound"
        if floor and r["tokens_total"] < least.get((group_key(r, o), _pair_key(r, o)), -math.inf):
            # The report's view only: the TrialRow keeps 'lower_bound'. What it did record still counts as spend.
            r.update(usage_status="unmeasured", floor_set_aside=True, floor_cost=r["cost"], cost=None,
                     cost_at_ingest_usd=None, tokens_total=None, tokens_uncached=None, valid_tokens=False,
                     valid_cost=False)


def _metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    if metric == "cost":
        return row["cost"] if row.get("valid_cost") else None
    return row.get(metric) if row.get("valid_tokens") else None


def _analysis_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: row.get(k) for k in ANALYSIS_KEYS}
    out["cost_usd"] = row.get("cost")
    out["options"] = dict(_mapping(row.get("options")))
    return out


def ledger_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    """Which trials a figure was computed from, as contract C9 defines it
    (sorted trial keys only), so every package gets the same value for the
    same rows and none re-implements it. The report's own provenance
    fingerprint (_fingerprint) also hashes outcomes and exclusions, because
    a trial that finishes after an incomplete scan keeps its trial key."""
    keys = sorted(str(r.get("trial_key")) for r in rows)
    return "sha256:" + hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()[:16]


# Grouping and pairing ---------------------------------------------------------------------

def group_key(row: Mapping[str, Any], o: Options) -> str:
    """The line a trial pools into on its (provider, model). A pinned
    baseline config stands apart only when harnesses are compared."""
    provider, model, family = row["provider"], row["model"], row["family"]
    config_hash = row.get("config_hash")
    pinned = o.baseline_config and o.compare == "harnesses"
    if pinned and config_hash == o.baseline_config and o.group_by != "config":
        key = f"cfg:{config_hash}"  # a pinned baseline config stands apart from its family
    elif o.group_by == "family":
        key = f"fam:{provider}|{model}|{family}"
    elif o.group_by == "implementation":
        key = f"impl:{provider}|{model}|{family}|{row.get('implementation') or 'builtin'}"
    else:
        key = f"cfg:{config_hash}"
    if o.split_harness_versions:
        key += f"|v{row.get('harness_version') or '?'}"
    return key


def _task_id(row: Mapping[str, Any], match_on: str) -> tuple[str, str]:
    if match_on == "name":
        return ("name", row["task"])
    digest = row.get("task_digest")
    # Without a digest the content is unknown: such trials pair only with each other.
    return ("digest", digest) if digest else ("nodigest", row["task"])


def task_id(row: Mapping[str, Any], match_on: str) -> tuple[str, str]:
    """What identifies a task for pairing under report.match_on; public for
    the estimates, which must pair exactly as the report does."""
    return _task_id(row, match_on)

def _pair_key(row: Mapping[str, Any], o: Options) -> tuple[str, ...]:
    # Comparing models, the same task content pairs across models; the group key holds the model.
    task = _task_id(row, o.match_on)
    key: tuple[str, ...] = tuple(task) if o.compare == "models" else (row["provider"], row["model"], *task)
    if o.same_run_only:
        key += (row["run_key"],)
    return key


def harness_key(row: Mapping[str, Any], o: Options) -> str:
    """What compare 'models' holds fixed: the harness as the grouping sees
    it, without the model. A configuration is its family, implementation,
    effort and options, since its config hash covers the model too."""
    family = row["family"]
    if o.group_by == "family":
        key = f"fam:{family}"
    elif o.group_by == "implementation":
        key = f"impl:{family}|{row.get('implementation') or 'builtin'}"
    else:
        key = (f"cfg:{family}|{row.get('implementation') or 'builtin'}|{row.get('effort')}|"
               f"{canonical_json(_mapping(row.get('options')))}")
    if o.split_harness_versions:
        key += f"|v{row.get('harness_version') or '?'}"
    return key


def build_cells(rows: Sequence[Mapping[str, Any]], o: Options) -> dict[tuple[str, tuple], dict]:
    """One cell per (group, pair key): the trials of one harness group on
    one task. Each metric is the mean over the trials valid for it, so
    attempts and runs pool while the task keeps one vote."""
    cells: dict[tuple[str, tuple], dict] = {}
    for row in rows:
        if row.get("manual"):
            continue
        group = row.get("group") or group_key(row, o)
        pk = _pair_key(row, o)
        cell = cells.get((group, pk))
        if cell is None:
            cell = cells[(group, pk)] = {
                "group": group, "pair": pk, "task": row["task"], "tasks": set(), "digests": set(), "runs": set(),
                "trial_keys": [], "outcomes": Counter(), "included": 0, "unmeasured": 0,
                "values": {m: [] for m in METRICS}, "floors": {m: [] for m in METRICS},
                "measured": {m: False for m in METRICS},
                "passes": [], "mean": {}, "pass": None, "n_reward": 0,
                # The cost values from list prices, and whether every cost value is the harness's own
                # figure (pricing 'reported'): a pair uses reported figures only when both cells are.
                "computed": {"values": [], "floors": []}, "reported_only": True,
            }
        cell["tasks"].add(row["task"])
        if row.get("task_digest"):
            cell["digests"].add(row["task_digest"])
        cell["runs"].add(row["run_key"])
        cell["trial_keys"].append(row["trial_key"])
        cell["outcomes"][row["outcome"]] += 1
        if row.get("included"):
            cell["included"] += 1
            if row.get("usage_status") == "unmeasured":
                cell["unmeasured"] += 1
        for metric in METRICS:
            source = "cost" if metric == "cost" else metric
            if row.get(source) is not None:
                cell["measured"][metric] = True
            value = _metric_value(row, metric)
            if value is not None:
                cell["values"][metric].append(float(value))
                if row.get("usage_status") == "lower_bound":
                    cell["floors"][metric].append(float(value))  # also in values
                if metric == "cost":
                    computed = row.get("cost_current") if row.get("cost_reported") else value
                    cell["reported_only"] = cell["reported_only"] and bool(row.get("cost_reported"))
                    cell["computed"]["values"].append(None if computed is None else float(computed))
                    if row.get("usage_status") == "lower_bound":
                        cell["computed"]["floors"].append(None if computed is None else float(computed))
        if row.get("valid_reward"):
            cell["passes"].append(1.0 if row.get("passed") else 0.0)
    for cell in cells.values():
        cell["reported_only"] = cell["reported_only"] and bool(cell["values"]["cost"])
        cell["mean"] = {m: (math.fsum(v) / len(v) if v else None) for m, v in cell["values"].items()}
        cell["n_reward"] = len(cell["passes"])
        cell["pass"] = math.fsum(cell["passes"]) / len(cell["passes"]) if cell["passes"] else None
        cell["trial_keys"].sort()
    return cells


def _side(cells: Mapping[tuple[str, tuple], dict], group: str) -> dict[tuple, dict]:
    return {pk: cell for (g, pk), cell in cells.items() if g == group}


def _computed_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    """The cell with its cost at list prices instead of the harness's figures."""
    computed = cell["computed"]
    values = [v for v in computed["values"] if v is not None]
    whole = len(values) == len(computed["values"])
    out = dict(cell)
    out["values"] = {**cell["values"], "cost": values}
    out["floors"] = {**cell["floors"], "cost": [v for v in computed["floors"] if v is not None]}
    out["mean"] = {**cell["mean"], "cost": math.fsum(values) / len(values) if values and whole else None}
    out["reported_only"] = False
    return out


def same_basis(cells: Mapping[tuple[str, tuple], dict], arm: str, base: str,
               o: Options) -> Mapping[tuple[str, tuple], dict]:
    """With pricing 'reported', a task's two cells use the harnesses' own
    figures only when both are made of them; otherwise both use list prices,
    so a harness that reports nothing (strands) is never set against
    another's exact bill. The cells of this comparison's two groups only."""
    if o.pricing != "reported":
        return cells
    out = dict(cells)
    for pk in set(_side(cells, arm)) & set(_side(cells, base)):
        a, b = cells[(arm, pk)], cells[(base, pk)]
        if not (a["reported_only"] and b["reported_only"]):
            out[(arm, pk)], out[(base, pk)] = _computed_cell(a), _computed_cell(b)
    return out


def _metric_pairs(arm: Mapping[tuple, dict], base: Mapping[tuple, dict], metric: str, *,
                  arm_means: Mapping[tuple, float] | None = None, base_means: Mapping[tuple, float] | None = None
                  ) -> tuple[list[tuple[tuple, float, float]], dict[str, int]]:
    """[(pair key, baseline mean, arm mean)] for tasks where both sides have
    a valid value above zero, and why the other shared tasks were dropped.
    arm_means and base_means replace some cells' means (_costed_high). A
    pair's multiple is arm mean / baseline mean (_log_ratio)."""
    dropped = {"no_arm_valid": 0, "no_baseline_valid": 0, "usage_not_recorded": 0, "tokens_not_reported": 0,
               "unpriced": 0, "zero_value": 0}
    pairs = []
    for pk in sorted(set(arm) & set(base)):
        a, b = arm[pk], base[pk]
        av = (arm_means or {}).get(pk, a["mean"].get(metric))
        bv = (base_means or {}).get(pk, b["mean"].get(metric))
        if not a["included"]:
            dropped["no_arm_valid"] += 1
        elif not b["included"]:
            dropped["no_baseline_valid"] += 1
        elif av is None or bv is None:
            # Every counted trial on that side ran but recorded no usage.
            lost = any(v is None and c["unmeasured"] and c["mean"].get("tokens_total") is None
                       for v, c in ((av, a), (bv, b)))
            tokens_seen = (a["mean"].get("tokens_total") is not None and b["mean"].get("tokens_total") is not None)
            dropped["usage_not_recorded" if lost else "unpriced" if metric == "cost" and tokens_seen
                    else "tokens_not_reported"] += 1
        elif av <= 0 or bv <= 0:
            dropped["zero_value"] += 1
        else:
            pairs.append((pk, bv, av))
    return pairs, dropped


def _log_ratio(base: float, arm: float) -> float:
    """One task's multiple, harness / baseline, on the log scale."""
    return math.log(arm / base)


def _unrecorded(cell: Mapping[str, Any], metric: str) -> bool:
    """The cell holds a trial cut short (a floor) or one with no usage recorded."""
    return bool(cell["floors"][metric] or cell["unmeasured"])


def _costed_high(side: Mapping[tuple, dict], other: Mapping[tuple, dict], metric: str, *, arm: bool,
                 ratio: float | None) -> dict[tuple, float]:
    """The means of one side's cells holding trials cut short or with no
    usage recorded, had each of those cost UNRECORDED_FACTOR times the side's
    usual spend on that task (a floor above that keeps its own value). The
    usual spend is the cell's uncut trials, else the other side's mean on the
    task scaled by ratio (arm ÷ baseline on tasks where nothing was cut
    short), else the side's most expensive trial on the model. Deliberately
    high, since such a trial often ran to a time limit, and scaled to the task: a
    side maximum would price a small task like the biggest, and a big task
    like the biggest the side happened to finish. Only the cells this changes."""
    top = max((v for c in side.values() for v in c["values"][metric]), default=None)
    out = {}
    for pk, c in side.items():
        if not _unrecorded(c, metric):
            continue
        floors, lost, values = c["floors"][metric], c["unmeasured"], c["values"][metric]
        uncut_sum, uncut_n = math.fsum(values) - math.fsum(floors), len(values) - len(floors)
        there = (other.get(pk) or {}).get("mean", {}).get(metric)
        if uncut_n:
            high = UNRECORDED_FACTOR * uncut_sum / uncut_n
        elif there and ratio:
            high = UNRECORDED_FACTOR * (there * ratio if arm else there / ratio)
        elif top is not None:
            high = top
        else:
            continue
        out[pk] = (uncut_sum + math.fsum(max(f, high) for f in floors) + high * lost) / (len(values) + lost)
    return out


def _pass_pairs(arm: Mapping[tuple, dict], base: Mapping[tuple, dict]) -> list[tuple[tuple, float, float, int, int]]:
    out = []
    for pk in sorted(set(arm) & set(base)):
        a, b = arm[pk], base[pk]
        if a["n_reward"] and b["n_reward"]:
            out.append((pk, a["pass"], b["pass"], a["n_reward"], b["n_reward"]))
    return out


def _status(n: int, o: Options) -> tuple[str, int]:
    if n < o.min_pairs_estimate:
        return "insufficient", o.min_pairs_estimate - n
    if n < o.min_pairs_verdict:
        return "low_n", o.min_pairs_verdict - n
    return "ok", 0


def _geomean(values: Sequence[float]) -> float:
    return math.exp(math.fsum(values) / len(values))


def _rom(pairs: Sequence[tuple[float, float]]) -> float:
    """Ratio of means over (baseline, arm) pairs: the arm's mean bill over the baseline's."""
    return math.fsum(h for _, h in pairs) / math.fsum(b for b, _ in pairs)


def axis_key(group: Mapping[str, Any], o: Options) -> tuple[str, ...]:
    """What a comparison holds fixed: the (provider, model) when harnesses
    are compared, the harness (harness_key) when models are."""
    return (group["harness_key"],) if o.compare == "models" else (group["provider"], group["model"])


def _pick_model_baselines(groups: Mapping[str, Mapping[str, Any]],
                          o: Options) -> tuple[dict[tuple[str, ...], str], list[str]]:
    """The baseline group of each harness: its group on the baseline model,
    the one with the most valid trials when the grouping gives several."""
    by_harness: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for key, g in groups.items():
        by_harness[axis_key(g, o)].append(key)
    picked: dict[tuple[str, ...], str] = {}
    warnings: list[str] = []
    for axis in sorted(by_harness):
        keys = sorted(by_harness[axis])
        candidates = [k for k in keys if f"{groups[k]['provider']}:{groups[k]['model']}" == o.baseline_model]
        if not candidates:
            continue
        best = max(candidates, key=lambda k: (groups[k]["trials"]["included"], groups[k]["trials"]["total"]))
        picked[axis] = best
        if len(candidates) > 1:
            others = "; ".join(groups[k]["label"] for k in candidates if k != best)
            warnings.append(f"{len(candidates)} groups of one harness on {o.baseline_model}. The baseline is "
                            f"{groups[best]['label']} (most valid trials); the others are compared with it: {others}.")
    return picked, warnings


def pick_baselines(groups: Mapping[str, Mapping[str, Any]], o: Options) -> tuple[dict[tuple[str, ...], str], list[str]]:
    """The baseline group on each (provider, model): the pinned config when
    there is one, else the baseline family's group, else the one of its
    groups with the most valid trials, with a warning naming the rest.
    Comparing models, the baseline of each harness (_pick_model_baselines)."""
    if o.compare == "models":
        return _pick_model_baselines(groups, o)
    by_model: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, g in groups.items():
        by_model[(g["provider"], g["model"])].append(key)
    picked: dict[tuple[str, ...], str] = {}
    warnings: list[str] = []
    for pm in sorted(by_model):
        keys = sorted(by_model[pm])
        candidates: list[str] = []
        if o.baseline_config:
            pinned = f"cfg:{o.baseline_config}"
            candidates = [k for k in keys if k == pinned or k.startswith(pinned + "|")]
        if not candidates:
            candidates = [k for k in keys if groups[k]["family"] == o.baseline_family]
        if not candidates:
            continue
        best = max(candidates, key=lambda k: (groups[k]["trials"]["included"], groups[k]["trials"]["total"]))
        picked[pm] = best
        if len(candidates) > 1:
            others = "; ".join(groups[k]["label"] for k in candidates if k != best)
            warnings.append(f"{len(candidates)} {o.baseline_family} groups on {pm[0]}:{pm[1]}. The baseline is "
                            f"{groups[best]['label']} (most valid trials); the others are compared with it: "
                            f"{others}. Pin one with options.baseline.config.")
    return picked, warnings


# Groups -------------------------------------------------------------------------------------

def _summary(values: Sequence[float]) -> dict[str, Any]:
    return {"mean": math.fsum(values) / len(values) if values else None,
            "median": statistics.median(values) if values else None, "n": len(values)}


def _group_names(key: str, rows: Sequence[Mapping[str, Any]], o: Options) -> tuple[str, str]:
    """(short name for headlines, label)."""
    first = rows[0]
    family, provider, model = first["family"], first["provider"], first["model"]
    # Built the way group_key built the suffix: parsing the key would mistake
    # a model or family starting with 'v' for a version.
    version = f" v{first.get('harness_version') or '?'}" if o.split_harness_versions else ""
    where = f"{provider}:{model}"
    if key.startswith("cfg:"):
        config_hash = first.get("config_hash") or ""
        return f"{family} {config_hash[:6]}{version}", f"{first.get('config_label') or key}{version}"
    if key.startswith("impl:"):
        impl = first.get("implementation") or "builtin"
        short = impl if impl == "builtin" else impl.rsplit(":", 1)[-1]
        return f"{family} ({short}){version}", f"{family} [{impl}] · {where}{version}"
    return f"{family}{version}", f"{family} · {where}{version}"


def _harness_name(row: Mapping[str, Any], name: str, o: Options) -> str:
    """A group's harness without its model, for compare 'models': its name,
    except a configuration's, whose config hash covers the model; that one is
    named by its harness_key, so both sides of a model comparison agree."""
    if o.group_by != "config":
        return name
    version = f" v{row.get('harness_version') or '?'}" if o.split_harness_versions else ""
    digest = hashlib.sha256(harness_key(row, o).encode("utf-8")).hexdigest()[:6]
    return f"{row['family']} {digest}{version}"


def _config_diff(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Settings that differ between the configurations pooled in a group."""
    configs: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.get("config_hash") not in configs:
            settings_ = dict(_mapping(r.get("options")))
            settings_["effort"] = r.get("effort")
            settings_["implementation"] = r.get("implementation")
            configs[r.get("config_hash")] = settings_
    if len(configs) < 2:
        return {}
    keys = sorted({k for c in configs.values() for k in c})
    diff = {}
    for k in keys:
        values = {canonical_json(c.get(k)) for c in configs.values()}
        if len(values) > 1:
            diff[k] = [json.loads(v) for v in sorted(values)]
    return diff


def _group(key: str, rows: list[dict], o: Options) -> dict[str, Any]:
    first = rows[0]
    name, label = _group_names(key, rows, o)
    impls = {r.get("implementation") for r in rows}
    hashes = {r.get("config_hash") for r in rows}
    versions = Counter(r.get("harness_version") or "unknown" for r in rows)
    setups = Counter(r.get("setup_mode") or "unknown" for r in rows)
    labels: dict[str, str] = {}
    for r in rows:
        labels.setdefault(r.get("config_hash"), r.get("config_label") or "")
    configs = [{"config_hash": h, "label": labels[h], "trials": n}
               for h, n in sorted(Counter(r.get("config_hash") for r in rows).items(), key=lambda kv: (-kv[1], kv[0]))]

    by_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    task_ids = set()
    for r in rows:
        tid = _task_id(r, o.match_on)
        task_ids.add(tid)
        if r.get("valid_reward"):
            by_task[tid].append(1.0 if r.get("passed") else 0.0)
    pass_rate = None
    if by_task:
        successes = math.fsum(math.fsum(v) / len(v) for v in by_task.values())
        lo, hi = stats.wilson(successes, len(by_task), o.ci_level)  # type: ignore[misc]
        pass_rate = {"estimate": successes / len(by_task), "lo": lo, "hi": hi, "tasks": len(by_task),
                     "method": "wilson_tasks"}

    finished = [r for r in rows if r["outcome"] != "incomplete"]
    failures = sum(1 for r in finished if r["outcome"] in RELIABILITY_OUTCOMES)
    interval = stats.wilson(failures, len(finished), o.ci_level)
    setup_rate = {"estimate": failures / len(finished) if finished else None,
                  "lo": interval[0] if interval else None, "hi": interval[1] if interval else None,
                  "trials": len(finished), "method": "wilson_trials"}

    ratios = [r["reported_cost_usd"] / r["cost_current"] for r in rows
              if r.get("reported_cost_usd") is not None and r.get("cost_current")]
    outcomes = Counter(r["outcome"] for r in rows)
    return {
        "key": key, "label": label, "name": name, "provider": first["provider"], "model": first["model"],
        # What compare 'models' holds fixed across models, and its name there.
        "harness_key": harness_key(first, o), "harness_name": _harness_name(first, name, o),
        "family": first["family"], "implementation": next(iter(impls)) if len(impls) == 1 else None,
        "config_hash": next(iter(hashes)) if len(hashes) == 1 else None,
        "harness_version": next(iter(versions)) if len(versions) == 1 and "unknown" not in versions else None,
        "is_baseline": False, "configs": configs, "config_diff": _config_diff(rows),
        "harness_versions": dict(sorted(versions.items())),
        # Provenance, not a split: stock and pre-baked trials of one configuration pool in one group.
        "setup_modes": dict(sorted(setups.items())),
        "runs": len({r["run_key"] for r in rows}), "tasks": len(task_ids),
        "trials": {"total": len(rows), "by_outcome": {k: outcomes[k] for k in classify.OUTCOMES if outcomes[k]},
                   "valid_tokens": sum(1 for r in rows if r["valid_tokens"]),
                   "valid_cost": sum(1 for r in rows if r["valid_cost"]),
                   "valid_reward": sum(1 for r in rows if r["valid_reward"]),
                   "included": sum(1 for r in rows if r["included"]),
                   # Counted, but left out of tokens and cost: the agent ran and no usage was recorded,
                   # or a floor fell below every uncut trial on its task (_set_aside_low_floors).
                   "usage_unmeasured": sum(1 for r in rows if r["included"] and r["usage_status"] == "unmeasured"),
                   # In the means, though their usage covers completed calls only (a timeout).
                   "usage_lower_bound": sum(1 for r in rows
                                            if r["valid_tokens"] and r["usage_status"] == "lower_bound")},
        "pass_rate": pass_rate, "setup_failure_rate": setup_rate,
        "per_trial": {
            "tokens_total": _summary([r["tokens_total"] for r in rows if r["valid_tokens"]]),
            "tokens_uncached": _summary([r["tokens_uncached"] for r in rows if r["valid_tokens"]]),
            "cost": _summary([r["cost"] for r in rows if r["valid_cost"]]),
            "minutes": {k: v for k, v in _summary([r["total_minutes"] for r in rows
                                                   if r["included"] and r["total_minutes"] is not None]).items()
                        if k != "mean"},
            "minutes_ex_setup": {k: v for k, v in _summary([r["minutes_ex_setup"] for r in rows if r["included"]
                                                             and r["minutes_ex_setup"] is not None]).items()
                                 if k != "mean"},
        },
        # Always filled: the harness's own figure against the computed one.
        "reported_vs_computed_cost": {"median_ratio": statistics.median(ratios) if ratios else None,
                                      "n": len(ratios)},
    }


# References and verdicts ------------------------------------------------------------------

def _tolerance(*candidates: Any) -> float:
    for value in candidates:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number >= 1.0:
            return number
    return DEFAULT_TOLERANCE


def _applies(patterns: Any, provider: str, model: str) -> bool:
    if patterns is None:
        return True
    if isinstance(patterns, str):
        patterns = [patterns]
    target = f"{provider}:{model}"
    return any(fnmatchcase(target, str(p)) for p in patterns)


def reference_for(reference: Mapping[str, Any] | None, family: str, provider: str, model: str,
                  metric: str) -> dict | None:
    """The published figure for a family and metric as a band, with the
    estimator its verdict uses and whether it applies to this model. A
    multiple's figure and band are harness / baseline whichever way
    reference.yaml was written (stats.published_reference)."""
    reference = stats.published_reference(reference)
    defaults = _mapping(reference.get("defaults"))
    consensus = _mapping(reference.get("consensus"))
    entry = _mapping(_mapping(reference.get("arms")).get(family))
    if metric == "tokens_total":
        published = entry.get("tokens_per_trial")
    elif metric == "cost":
        published = entry.get("cost_per_trial") or consensus.get("cost_per_trial")
    elif metric == "pass":
        published = consensus.get("pass_rate_delta_pp")
    else:
        return None
    if not isinstance(published, Mapping):
        return None
    tolerance = _tolerance(entry.get("tolerance"), defaults.get("tolerance"))
    if metric == "pass":
        # A pass-rate figure is a band of points, never a multiple.
        if published.get("low") is None or published.get("high") is None:
            return None
        band = (float(published["low"]), float(published["high"]))
    else:
        band = stats.reference_band(published, tolerance)
        if band is None:
            return None
    estimator = entry.get("estimator") or defaults.get("estimator") or "ratio_of_means"
    if estimator not in ESTIMATORS:
        estimator = "ratio_of_means"
    patterns = entry.get("applies_to", defaults.get("applies_to", list(DEFAULT_APPLIES_TO)))
    return {
        "published": {k: published[k] for k in ("value", "low", "high", "source") if k in published},
        "band": [band[0], band[1]], "estimator": estimator, "applies": _applies(patterns, provider, model),
    }


def verdict(ci: Sequence[float | None] | None, band: Sequence[float], status: str, applies: bool) -> str:
    """Never 'confirmed': an interval is within, above, below or overlapping
    the published band, and only once there are enough pairs. With multiples
    as harness / baseline, 'below' means the harness used less, relative to
    the baseline, than the published figure says, and 'above' more."""
    if not applies:
        return "not_applicable"
    if status != "ok" or not ci or ci[0] is None or ci[1] is None:
        return "insufficient"
    lo, hi = float(ci[0]), float(ci[1])
    if lo >= band[0] and hi <= band[1]:
        return "within"
    if lo > band[1]:
        return "above"
    if hi < band[0]:
        return "below"
    return "overlaps"


def _position(point: float | None, band: Sequence[float]) -> str | None:
    if point is None:
        return None
    if point < band[0]:
        return "below"
    return "above" if point > band[1] else "inside"


def _reference_baseline(reference: Mapping[str, Any] | None) -> str:
    return str(_mapping(_mapping(reference).get("defaults")).get("baseline_family") or REFERENCE_BASELINE)


# Comparisons ------------------------------------------------------------------------------

def compare_metric(cells: Mapping[tuple[str, tuple], dict], arm: str, base: str, metric: str, o: Options, key: str,
                   *, variants: Mapping[str, Mapping[tuple[str, tuple], dict]] | None = None,
                   reference: Mapping[str, Any] | None = None,
                   names: tuple[str, str] = ("the arm", "the baseline")) -> dict:
    """Arm ÷ baseline for one metric (0.5 means the arm used half): the
    geometric mean of per-task ratios with a t interval (the headline) and
    the ratio of mean bills with a task bootstrap (like-for-like with
    published per-task averages).

    A trial cut short records a floor, and one with no usage recorded is left
    out, so either side's mean may be too low on its tasks. Each side is
    therefore also scored with those trials costed high (_costed_high), which
    brings back tasks it lost entirely: a verdict that this changes is
    withheld, and the two geomeans are sensitivity checks."""
    arm_cells, base_cells = _side(cells, arm), _side(cells, base)
    measured = (any(c["measured"][metric] for c in arm_cells.values())
                and any(c["measured"][metric] for c in base_cells.values()))
    pairs, dropped = _metric_pairs(arm_cells, base_cells, metric)
    n = len(pairs)
    out: dict[str, Any] = {
        "status": "not_measured", "n": n, "need": 0, "geomean": None, "ratio_of_means": None, "per_task": None,
        "leave_one_out": None,
        "sensitivity": {"crashes_excluded": None, "misconfigured_cap_included": None, "same_run_only": None,
                        "unrecorded_arm_high": None, "unrecorded_baseline_high": None},
        "robust": None, "reference": None, "dropped": dropped,
    }
    clean = [_log_ratio(b, h) for pk, b, h in pairs
             if not _unrecorded(arm_cells[pk], metric) and not _unrecorded(base_cells[pk], metric)]
    ratio = _geomean(clean) if clean else _geomean([_log_ratio(b, h) for _, b, h in pairs]) if pairs else None
    # (side, pairs with that side's unrecorded trials costed high, how many of them this changed)
    costed_high = []
    for side, here, there in (("arm", arm_cells, base_cells), ("baseline", base_cells, arm_cells)):
        means = _costed_high(here, there, metric, arm=side == "arm", ratio=ratio)
        if means:
            spairs, _ = _metric_pairs(arm_cells, base_cells, metric,
                                      **({"arm_means": means} if side == "arm" else {"base_means": means}))
            costed_high.append((side, spairs, sum(1 for pk, _, _ in spairs if pk in means)))
    if measured:
        out["status"], out["need"] = _status(n, o)
    if out["status"] in ("low_n", "ok"):
        logs = [_log_ratio(b, h) for _, b, h in pairs]
        gm = stats.log_ratio_summary(logs, o.ci_level)
        out["geomean"] = {"estimate": gm["estimate"], "lo": gm["lo"], "hi": gm["hi"], "sd_log": gm["sd_log"],
                          "gsd": gm["gsd"], "method": "t_on_log_ratios"}
        values = [(b, h) for _, b, h in pairs]
        ci = stats.bootstrap_ci(values, _rom, resamples=o.resamples, seed=stats.stable_seed(o.seed, key, metric),
                                level=o.ci_level)
        out["ratio_of_means"] = {"estimate": _rom(values), "lo": ci[0] if ci else None,
                                 "hi": ci[1] if ci else None, "method": "task_bootstrap"}
        out["per_task"] = stats.quartiles([h / b for b, h in values])
        loo = stats.leave_one_out(logs, _geomean)
        out["leave_one_out"] = list(loo) if loo else None
        for name, vcells in (variants or {}).items():
            vpairs, _ = _metric_pairs(_side(vcells, arm), _side(vcells, base), metric)
            if len(vpairs) >= o.min_pairs_estimate:
                out["sensitivity"][name] = _geomean([_log_ratio(b, h) for _, b, h in vpairs])
        for side, spairs, _touched in costed_high:
            if len(spairs) >= o.min_pairs_estimate:
                out["sensitivity"][f"unrecorded_{side}_high"] = _geomean([_log_ratio(b, h) for _, b, h in spairs])
        if gm["lo"] is not None:
            checks = [v for v in out["sensitivity"].values() if v is not None] + list(loo or ())
            out["robust"] = all(gm["lo"] <= v <= gm["hi"] for v in checks)
    if reference is not None:
        band = reference["band"]
        chosen = out["ratio_of_means"] if reference["estimator"] == "ratio_of_means" else out["geomean"]
        point = chosen["estimate"] if chosen else None
        ci = (chosen["lo"], chosen["hi"]) if chosen else None
        found = verdict(ci, band, out["status"], reference["applies"])
        withheld = None
        for side, spairs, touched in costed_high if found in VERDICTS_MET else ():
            # A verdict that rests on what trials cut short or unrecorded cost is not given.
            alt = _verdict_on(spairs, reference, o, key, metric)
            if alt != found:
                who = names[0] if side == "arm" else names[1]
                many = touched != 1
                withheld = (f"{touched} paired task{'s' if many else ''} rest{'' if many else 's'} on {who} trials "
                            f"cut short or with no usage recorded; had those cost {UNRECORDED_FACTOR:g}x {who}'s "
                            f"usual spend on their task, the verdict would be '{alt}', not '{found}'")
                found = "insufficient"
                break
        out["reference"] = {
            "published": reference["published"], "band": list(band), "estimator": reference["estimator"],
            "applies": reference["applies"], "position": _position(point, band), "verdict": found,
            "withheld": withheld,
        }
    return out


def _verdict_on(pairs: Sequence[tuple[tuple, float, float]], reference: Mapping[str, Any], o: Options, key: str,
                metric: str) -> str:
    """The reference verdict from these pairs, by the same rules and with the
    same bootstrap seed, so only the changed values move it."""
    status, _ = _status(len(pairs), o)
    ci: tuple[float | None, float | None] | None = None
    if status in ("low_n", "ok"):
        if reference["estimator"] == "ratio_of_means":
            ci = stats.bootstrap_ci([(b, h) for _, b, h in pairs], _rom, resamples=o.resamples,
                                    seed=stats.stable_seed(o.seed, key, metric), level=o.ci_level)
        else:
            gm = stats.log_ratio_summary([_log_ratio(b, h) for _, b, h in pairs], o.ci_level)
            ci = (gm["lo"], gm["hi"])
    return verdict(ci, reference["band"], status, reference["applies"])


def compare_pass(cells: Mapping[tuple[str, tuple], dict], arm: str, base: str, o: Options, key: str,
                 *, reference: Mapping[str, Any] | None = None) -> dict:
    """Harness minus baseline pass rate on paired tasks, in points. Newcombe's
    interval when each task has one trial a side, else a task bootstrap."""
    pairs = _pass_pairs(_side(cells, arm), _side(cells, base))
    n = len(pairs)
    status, need = _status(n, o)
    out: dict[str, Any] = {
        "status": status, "n": n, "need": need, "arm_rate": None, "baseline_rate": None,
        "delta_pp": {"estimate": None, "lo": None, "hi": None, "method": None},
        "discordant": None, "mcnemar_p": None, "mde_pp": None, "reference": None,
    }
    diffs = [pa - pb for _, pa, pb, _, _ in pairs]
    if n:
        sd_d = statistics.stdev(diffs) if n >= MIN_SD_PAIRS else 0.5
        out["mde_pp"] = stats.mde_pp(max(sd_d, 0.3), n, o.ci_level)
    ci_pp: tuple[float, float] | None = None
    if status in ("low_n", "ok"):
        arm_rate = math.fsum(pa for _, pa, _, _, _ in pairs) / n
        base_rate = math.fsum(pb for _, _, pb, _, _ in pairs) / n
        out["arm_rate"], out["baseline_rate"] = arm_rate, base_rate
        out["delta_pp"]["estimate"] = 100.0 * (arm_rate - base_rate)
        if all(na == 1 and nb == 1 for _, _, _, na, nb in pairs):
            a = sum(1 for _, pa, pb, _, _ in pairs if pa == 1 and pb == 1)
            b = sum(1 for _, pa, pb, _, _ in pairs if pa == 1 and pb == 0)
            c = sum(1 for _, pa, pb, _, _ in pairs if pa == 0 and pb == 1)
            lo, hi = stats.newcombe_paired(a, b, c, n - a - b - c, o.ci_level)
            ci_pp = (100.0 * lo, 100.0 * hi)
            out["delta_pp"].update(lo=ci_pp[0], hi=ci_pp[1], method="newcombe_paired")
            out["discordant"] = {"arm_only": b, "baseline_only": c}
            out["mcnemar_p"] = stats.mcnemar_exact(b, c)
        else:
            ci = stats.bootstrap_ci(diffs, statistics.fmean, resamples=o.resamples,
                                    seed=stats.stable_seed(o.seed, key, "pass"), level=o.ci_level)
            # No discordant tasks gives a zero-width interval: say nothing rather than a false [0, 0].
            if ci is not None and ci[1] - ci[0] > 1e-12:
                ci_pp = (100.0 * ci[0], 100.0 * ci[1])
                out["delta_pp"].update(lo=ci_pp[0], hi=ci_pp[1], method="task_bootstrap")
    if reference is not None:
        out["reference"] = {"published": reference["published"], "band": list(reference["band"]),
                            "applies": reference["applies"],
                            "verdict": verdict(ci_pp, reference["band"], status, reference["applies"])}
    return out


def _task_names(cells: Sequence[Mapping[str, Any]], order: Mapping[str, int]) -> list[str]:
    names = {t for c in cells for t in c["tasks"]}
    return sorted(names, key=lambda t: (order.get(t, len(order)), t))


def _pairs_summary(cells: Mapping[tuple[str, tuple], dict], arm: str, base: str, cost_metric: Mapping[str, Any],
                   order: Mapping[str, int]) -> dict[str, Any]:
    arm_cells, base_cells = _side(cells, arm), _side(cells, base)
    both = [pk for pk in arm_cells if pk in base_cells and arm_cells[pk]["included"] and base_cells[pk]["included"]]
    same = [pk for pk in both if arm_cells[pk]["runs"] & base_cells[pk]["runs"]]
    return {
        "n_tasks": len(both),
        "same_run_fraction": len(same) / len(both) if both else None,
        "arm_only_tasks": _task_names([c for pk, c in arm_cells.items() if pk not in base_cells], order),
        "baseline_only_tasks": _task_names([c for pk, c in base_cells.items() if pk not in arm_cells], order),
        "dropped": dict(cost_metric["dropped"]),
    }


def _side_summary(cell: Mapping[str, Any] | None) -> dict[str, Any]:
    if cell is None:
        return {"trials": 0, "valid": 0, "tokens_total": None, "cost": None, "pass": None, "trial_keys": []}
    return {"trials": len(cell["trial_keys"]), "valid": cell["included"], "tokens_total": cell["mean"]["tokens_total"],
            "cost": cell["mean"]["cost"], "pass": cell["pass"], "trial_keys": list(cell["trial_keys"])}


def _ratio(base: float | None, arm: float | None) -> float | None:
    """One task's multiple, harness / baseline."""
    return arm / base if base and arm and base > 0 and arm > 0 else None


def _pairs_table(cells: Mapping[tuple[str, tuple], dict], arm: str, base: str, o: Options,
                 order: Mapping[str, int]) -> list[dict[str, Any]]:
    arm_cells, base_cells = _side(cells, arm), _side(cells, base)

    def sort_key(pk: tuple) -> tuple:
        cell = arm_cells.get(pk) or base_cells[pk]
        return (order.get(cell["task"], len(order)), cell["task"], pk)

    rows = []
    for pk in sorted(set(arm_cells) | set(base_cells), key=sort_key):
        a, b = arm_cells.get(pk), base_cells.get(pk)
        cell = a or b
        digests = sorted((a or {}).get("digests", set()) | (b or {}).get("digests", set()))
        rows.append({
            "task": cell["task"],
            "digest_short": digests[0].split(":", 1)[-1][:12] if digests else None,
            "run_key": pk[-1] if o.same_run_only else None,
            "same_run": bool(a and b and a["runs"] & b["runs"]),
            "arm": _side_summary(a), "baseline": _side_summary(b),
            "ratio_tokens": _ratio(b["mean"]["tokens_total"] if b else None, a["mean"]["tokens_total"] if a else None),
            "ratio_cost": _ratio(b["mean"]["cost"] if b else None, a["mean"]["cost"] if a else None),
        })
    return rows


def coverage_for(comparison: Mapping[str, Any], cells: Mapping[tuple[str, tuple], dict],
                 groups: Mapping[str, Mapping[str, Any]], cfg: Config, o: Options) -> dict:
    """Where a comparison stands and what to run next: the stable answer
    the run-design assistant plans from. Never suggests tasks outside the
    seeded sample, and never asks for more pairs than the sample holds."""
    arm, base = comparison["group"], comparison["baseline_group"]
    arm_cells, base_cells = _side(cells, arm), _side(cells, base)
    metrics = comparison["metrics"]
    if metrics["cost"]["status"] != "not_measured":
        metric = "cost"
    elif metrics["tokens_total"]["status"] != "not_measured":
        metric = "tokens_total"
    else:
        metric = "pass"

    pass_diffs = [pa - pb for _, pa, pb, _, _ in _pass_pairs(arm_cells, base_cells)]
    sd_d = statistics.stdev(pass_diffs) if len(pass_diffs) >= MIN_SD_PAIRS else 0.5
    sample = list(cfg.tasks or [])
    halfwidth_now = sd_used = None
    sd_source = None
    est_needed = est_more = None
    capped = False
    note = None
    if metric == "pass":
        n = comparison["pass"]["n"]
        paired = {t for pk, *_ in _pass_pairs(arm_cells, base_cells) for t in arm_cells[pk]["tasks"]}
        arm_valid = {t for c in arm_cells.values() if c["n_reward"] for t in c["tasks"]}
        base_valid = {t for c in base_cells.values() if c["n_reward"] for t in c["tasks"]}
    else:
        pairs, _ = _metric_pairs(arm_cells, base_cells, metric)
        n = len(pairs)
        logs = [_log_ratio(b, h) for _, b, h in pairs]
        if n >= MIN_SD_PAIRS:
            sd_used, sd_source = statistics.stdev(logs), "observed"
            halfwidth_now = math.exp(stats.t_quantile(o.ci_level, n - 1) * sd_used / math.sqrt(n))
        else:
            sd_used, sd_source = o.prior_sd_log, "prior"
        raw_needed = stats.n_needed_ratio(sd_used, o.target_ratio_halfwidth, o.ci_level)
        est_needed = raw_needed
        if sample and raw_needed > len(sample):
            est_needed, capped = len(sample), True
            at_cap = (math.exp(stats.t_quantile(o.ci_level, len(sample) - 1) * sd_used / math.sqrt(len(sample)))
                      if len(sample) >= 2 else None)
            note = (f"Reaching x/÷{o.target_ratio_halfwidth:g} needs about {raw_needed} paired tasks, more than the "
                    f"{len(sample)} in the sample; at {len(sample)} pairs the remaining uncertainty is about "
                    + (f"x/÷{at_cap:.2f}." if at_cap else "unknown."))
        est_more = max(0, est_needed - n)
        paired = {t for pk, _, _ in pairs for t in arm_cells[pk]["tasks"]}
        arm_valid = {t for c in arm_cells.values() if (c["mean"].get(metric) or 0) > 0 for t in c["tasks"]}
        base_valid = {t for c in base_cells.values() if (c["mean"].get(metric) or 0) > 0 for t in c["tasks"]}

    need_estimate = max(0, o.min_pairs_estimate - n)
    need_verdict = max(0, o.min_pairs_verdict - n)
    complete, new = [], []
    for task in sample:
        if task in paired:
            continue
        has_arm, has_base = task in arm_valid, task in base_valid
        if has_arm and not has_base:
            complete.append({"task": task, "side": "baseline", "reason": "complete pair"})
        elif has_base and not has_arm:
            complete.append({"task": task, "side": "arm", "reason": "complete pair"})
        else:
            new.append({"task": task, "side": "both", "reason": "new task"})
    limit = max(est_more, need_estimate) if est_more is not None else 10
    next_tasks = (complete + new)[:limit]

    def per_trial(key: str) -> dict[str, Any]:
        g = groups[key]
        cost, lost = g["per_trial"]["cost"], g["trials"]["usage_unmeasured"]
        mean = cost["mean"]
        top = max((v for c in _side(cells, key).values() for v in c["values"]["cost"]), default=None)
        if mean is not None and lost and top is not None:
            # A plan, not a measurement: a trial that recorded no usage often ran to a time
            # limit, so each is costed at the group's most expensive trial, not left out.
            mean = (mean * cost["n"] + top * lost) / (cost["n"] + lost)
        return {"mean_cost_usd": mean, "median_minutes": g["per_trial"]["minutes"]["median"],
                "setup_failure_rate": g["setup_failure_rate"]["estimate"]}

    sides = {"arm": per_trial(arm), "baseline": per_trial(base)}
    est_cost: float | None = 0.0
    for item in next_tasks:
        for side in (("arm", "baseline") if item["side"] == "both" else (item["side"],)):
            mean_cost = sides[side]["mean_cost_usd"]
            if mean_cost is None:
                est_cost = None
                break
            est_cost += mean_cost  # type: ignore[operator]
        if est_cost is None:
            break
    return {
        "metric": metric, "pairs": n, "need_for_estimate": need_estimate, "need_for_verdict": need_verdict,
        "halfwidth_factor_now": halfwidth_now, "target_factor": o.target_ratio_halfwidth, "sd_log_used": sd_used,
        "sd_source": sd_source, "est_pairs_needed": est_needed, "est_more_pairs": est_more,
        "pass_tasks_needed": stats.n_needed_pass(max(sd_d, 0.3), o.target_pass_delta_pp, o.ci_level),
        "next": next_tasks, "est_cost_usd": est_cost, "per_trial": sides, "capped_at_sample": capped, "note": note,
    }


def caveats_for(comparison: Mapping[str, Any], arm_group: Mapping[str, Any], base_group: Mapping[str, Any],
                arm_rows: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]],
                run_status: Mapping[str, str | None], o: Options) -> list[str]:
    """Plain-sentence warnings about what the comparison's numbers may hide."""
    out: list[str] = []
    if comparison.get("compare") == "models":
        # Both sides are one harness: name each by its model.
        arm_group = {**arm_group, "name": f"{comparison['harness']} on {comparison['arm_name']}"}
        base_group = {**base_group, "name": f"{comparison['harness']} on {comparison['baseline_name']}"}
    sides = ((arm_group, [r for r in arm_rows if r["included"]]), (base_group, [r for r in base_rows if r["included"]]))
    for g, rows in sides:
        versions = sorted({r["harness_version"] for r in rows if r.get("harness_version")})
        if len(versions) > 1:
            out.append(f"Several {g['name']} versions contribute ({', '.join(versions)}); split by version or pin "
                       "the 'version' option to keep pooled runs comparable.")
    for g, _ in sides:
        if g["config_diff"]:
            out.append(f"{g['name']} pools configurations that differ in {', '.join(sorted(g['config_diff']))}.")
    efforts = [sorted({r.get("effort") or "the harness default" for r in rows}) for _, rows in sides]
    if efforts[0] and efforts[1] and efforts[0] != efforts[1]:
        out.append(f"Effort differs: {arm_group['name']} ran at {', '.join(efforts[0])}, "
                   f"{base_group['name']} at {', '.join(efforts[1])}.")
    fraction = comparison["pairs"]["same_run_fraction"]
    if fraction is not None and fraction < 1:
        out.append(f"{(1 - fraction) * 100:.0f}% of paired tasks were run in different runs; harness versions or "
                   "provider routing may have drifted between them.")
    # Any such trial, on either side: with one trial a side per task, each one
    # costs its task a value, and equal counts on different tasks still bias.
    lost = [(g, sum(1 for r in rows if r["usage_status"] == "unmeasured" and not r["floor_set_aside"]), len(rows))
            for g, rows in sides]
    if lost[0][1] or lost[1][1]:
        (ag, ak, an), (bg, bk, bn) = lost
        tasks = comparison["metrics"]["tokens_total"]["dropped"]["usage_not_recorded"]
        out.append(f"No usage was recorded for {ak} of {an} {ag['name']} trials and {bk} of {bn} {bg['name']} "
                   "trials; they count for pass rate but are left out of tokens and cost"
                   + (f", which leaves {tasks} shared task{'s' if tasks != 1 else ''} without a value on one side"
                      if tasks else "")
                   + ". Such trials often ran to a time limit, so the side with more of them may look cheaper "
                   "than it is.")
    for g, rows in sides:
        aside = sum(1 for r in rows if r["floor_set_aside"])
        if aside:
            it, many = ("they", "s") if aside != 1 else ("it", "")
            out.append(f"{aside} {g['name']} trial{many} cut short by a time limit recorded fewer tokens than "
                       f"every uncut {g['name']} trial on the same task, though {it} ran longer, so part of what "
                       f"{it} spent is missing; like trials with no usage recorded, {it} count{'' if many else 's'} "
                       "for pass rate only.")
    for name, m in comparison["metrics"].items():
        if (m.get("reference") or {}).get("withheld"):
            out.append(f"No {'cost' if name == 'cost' else 'token'} verdict: {m['reference']['withheld']}.")
    for g, rows in sides:
        floors = sum(1 for r in rows if r["valid_tokens"] and r["usage_status"] == "lower_bound")
        if floors:
            out.append(f"{floors} {g['name']} trial{'s' if floors != 1 else ''} hit the time limit; their usage "
                       "covers completed model calls only, so their tokens and cost are lower bounds.")
    every = list(arm_rows) + list(base_rows)
    if any(r["outcome"] == "incomplete" or run_status.get(r["run_key"]) in STOPPED_RUN_STATUSES for r in every):
        out.append("Some trials come from runs stopped early (budget or cancel) or never finished; expensive "
                   "trials may be under-represented.")
    for g, rows in sides:
        fallback = sum(1 for r in rows if r.get("fallback_model"))
        if fallback:
            many = fallback != 1
            out.append(f"{fallback} {g['name']} trial{'s' if many else ''} spent more than "
                       f"{100 * (1 - PRIMARY_SHARE_MIN):.0f}% of {'their' if many else 'its'} tokens on models other "
                       f"than {g['model']} (a fallback model answered); {'they count' if many else 'it counts'} "
                       "for pass rate only.")
    out += flag_caveats(sides)
    for g, _ in sides:
        rvc = g["reported_vs_computed_cost"]
        ratio = rvc["median_ratio"]
        if ratio is None or REPORTED_RATIO_BAND[0] <= ratio <= REPORTED_RATIO_BAND[1]:
            continue
        if harnesses.reports_mispriced_cost(g["family"], g["provider"]):
            out.append(f"{g['name']} prices {g['provider']} models at its own provider's rates (its reported cost "
                       f"is a median {ratio:.2g}x the computed cost); the computed cost is used.")
            continue
        if g["provider"] == "openrouter" and OPENROUTER_CAVEAT not in out:
            out.append(OPENROUTER_CAVEAT)
        out.append(f"{g['name']}'s own cost figure is a median {ratio:.2g}x the cost computed from list prices "
                   f"(n={rvc['n']}).")
    # Without this a missing price only shows as cost 'not_measured' and a
    # quiet switch to tokens; the user needs to know which price to add.
    missing = sorted({m for _, rows in sides for r in rows if r.get("cost") is None for m in (r.get("unpriced") or [])})
    if missing:
        out.append(f"No price in config/pricing.yaml for {', '.join(missing)}; cost multiples leave these trials "
                   "out until a price is added.")
    if o.match_on == "name":
        snapshots = [sorted({str(r.get("dataset_snapshot")) for r in rows}) for _, rows in sides]
        if snapshots[0] and snapshots[1] and snapshots[0] != snapshots[1]:
            out.append(f"Pairs by task name join different task snapshots ({', '.join(snapshots[0])} vs "
                       f"{', '.join(snapshots[1])}); task content may differ.")
    references = [m.get("reference") for m in comparison["metrics"].values()] + [comparison["pass"].get("reference")]
    if any(ref is not None and not ref["applies"] for ref in references):
        out.append(NOT_APPLICABLE_CAVEAT)
    if comparison.get("compare") == "models":
        out.append(MODELS_NO_VERDICT)
    note = (comparison.get("coverage") or {}).get("note")
    if note:
        out.append(note)
    return list(dict.fromkeys(out))


def _usd(value: float) -> str:
    return f"${value:.4f}" if value < 0.01 else f"${value:.2f}"


def _bound(rows: Sequence[Mapping[str, Any]], flag: str) -> str:
    """' (up to $X in all)' from the flag's cost_usd_max on every row, else ''."""
    bounds = [(r.get("flags") or {}).get(flag, {}).get("cost_usd_max") for r in rows]
    return f" (up to {_usd(math.fsum(bounds))} in all)" if bounds and None not in bounds else ""


def flag_caveats(sides: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]]) -> list[str]:
    """Sentences for the trials whose flags (extract.trial_flags, raised at
    ingest from the harness's log and the prices) say their cost may be off,
    side by side; rows are the side's counted trials. A bound is summed over
    the flagged trials: the printed running cost for calls never saved, the
    costliest completed call for calls cut off mid-stream, and today's
    prices for declined attempts."""
    out: list[str] = []
    for g, rows in sides:
        def flagged(flag: str) -> list[Mapping[str, Any]]:
            return [r for r in rows if flag in (r.get("flags") or {})]

        def lead(hit: Sequence[Mapping[str, Any]], verb: tuple[str, str]) -> str:
            one = len(hit) == 1
            return f"{len(hit)} {g['name']} trial{'' if one else 's'} {verb[0] if one else verb[1]}"

        declined = flagged("declined_attempts")
        if declined:
            out.append(f"{lead(declined, ('has', 'have'))} model attempts declined before another model answered "
                       "(a refusal fallback); only the session log lists them and whether they are billed is "
                       f"unknown, so the cost leaves them out{_bound(declined, 'declined_attempts')}.")
        for flag, verb, what in (
                ("unsaved_calls", ("lost", "lost"), "model calls that completed but were never saved (a timeout "
                                                    "during a command)"),
                ("unfinished_calls", ("started", "started"), "model calls that never finished (a kill mid-call)")):
            hit = flagged(flag)
            if hit:
                calls = sum(int(r["flags"][flag].get("calls") or 0) for r in hit)
                out.append(f"{lead(hit, verb)} {what}: {calls} call{'' if calls == 1 else 's'} missing from the "
                           f"cost{_bound(hit, flag)}.")
        mismatch = flagged("reported_cost_mismatch")
        if mismatch:
            worst = max((r["flags"]["reported_cost_mismatch"].get("diff_pct") or 0.0 for r in mismatch), key=abs)
            out.append(f"{lead(mismatch, ('reports', 'report'))} a cost of its own more than "
                       f"{100 * extract.REPORTED_COST_TOLERANCE:g}% from the cost computed from its tokens "
                       f"(largest {worst:+.1f}%): a call the log does not show, or a price that differs.")
        unknown = flagged("cache_writes_unknown")
        if unknown:
            out.append(f"{lead(unknown, ('has', 'have'))} cache writes no harness log gave; they are priced as "
                       "fresh input, so the cost is low by the write premium.")
        gone = flagged("trial_folder_missing")
        if gone:
            out.append(f"{lead(gone, ('had', 'had'))} no trial folder when last re-derived; their cache writes and "
                       "log checks are the figures stored before.")
    return out


def _x(value: float) -> str:
    """A multiple for prose: 2 decimals, or 2 significant figures under 0.1
    (0.064, not 0.06), as the dashboard shows it."""
    return f"{value:.2g}" if 0 < value < 0.1 else f"{value:.2f}"


def headline(comparison: Mapping[str, Any], level: float = 0.90) -> str:
    """One line: who against whom, how many pairs, and the multiple once
    there are enough of them."""
    arm, base = comparison["arm_name"], comparison["baseline_name"]
    if comparison.get("compare") == "models":
        who = f"{arm} vs {base} with {comparison.get('harness') or 'one harness'}"
    else:
        who = f"{arm} vs {base} on {comparison['provider']}:{comparison['model']}"
    metric = (comparison.get("coverage") or {}).get("metric") or "cost"
    if metric == "pass":
        p = comparison["pass"]
        if p["status"] == "insufficient":
            return (f"{who}: tokens not reported; {p['n']} paired task{'' if p['n'] == 1 else 's'} for pass rate; "
                    f"{p['need']} more for an estimate")
        return f"{who}: tokens not reported; pass rate {p['delta_pp']['estimate']:+.0f} points ({p['n']} tasks)"
    m = comparison["metrics"][metric]
    n = m["n"]
    if m["status"] == "insufficient":
        return f"{who}: {n} paired task{'' if n == 1 else 's'}; {m['need']} more for an estimate"
    gm = m["geomean"]
    word = "cost" if metric == "cost" else "tokens"
    text = (f"{who}: {arm} used {_x(gm['estimate'])}x the {word} of {base} "
            f"({level * 100:g}% CI {_x(gm['lo'])}-{_x(gm['hi'])}x, {n} tasks)")
    if m["status"] == "low_n":
        # A model comparison never gets a verdict; the same pair count still marks a settled figure.
        goal = "a settled figure" if comparison.get("compare") == "models" else "a verdict"
        return text + f"; {m['need']} more for {goal}"
    ref = m.get("reference")
    if ref:
        if ref.get("withheld"):
            return text + f"; verdict withheld: {ref['withheld']}"
        if ref["verdict"] == "not_applicable":
            return text + "; published figures do not apply to this model"
        published, band = ref["published"], ref["band"]
        figure = (f"{_x(published['value'])}x (read as {_x(band[0])}-{_x(band[1])}x)"
                  if published.get("value") is not None else f"{_x(band[0])}-{_x(band[1])}x")
        estimator = "ratio-of-means" if ref["estimator"] == "ratio_of_means" else "geometric-mean"
        phrase = {"within": "lies within", "overlaps": "overlaps", "above": "lies above",
                  "below": "lies below"}.get(ref["verdict"], "cannot yet be judged against")
        return text + f"; the {estimator} interval {phrase} the published {figure}"
    return text


# The analysis ----------------------------------------------------------------------------------

@dataclass
class _Analysis:
    report: dict[str, Any]
    rows: list[dict]
    pairs_tables: dict[str, list[dict]]


def tasks_file(cfg: Config) -> str:
    """Where the profile's task scope (cfg.tasks) comes from, e.g. config/tasks.txt."""
    effective = getattr(cfg, "effective", None)
    return str((effective.get("tasks_file") if isinstance(effective, Mapping) else None) or "config/tasks.txt")


def scope_label(scope: Mapping[str, Any]) -> str:
    """A task_scope for a file name: '27-scope-tasks', 'all-tasks' or '5-tasks'."""
    return {"scope": f"{scope['tasks']}-scope-tasks", "all": "all-tasks"}.get(scope["kind"], f"{scope['tasks']}-tasks")


def task_scope(cfg: Config, f: Filters) -> dict[str, Any]:
    """The tasks a report covers, against the profile's task scope (cfg.tasks,
    read from its tasks_file): kind 'scope' when the task filter is exactly
    those tasks, which the REPORT tab sends unless told otherwise, 'all' with
    no task filter, else 'chosen'. Snapshots keep it, and exports name it."""
    scope = set(cfg.tasks or [])
    chosen = set(f.tasks)
    kind = "all" if not chosen else "scope" if chosen == scope else "chosen"
    return {"kind": kind, "tasks": len(chosen), "scope_tasks": len(scope), "tasks_file": tasks_file(cfg)}


def _task_order(cfg: Config) -> dict[str, int]:
    order: dict[str, int] = {}
    for i, task in enumerate(cfg.tasks or []):
        order.setdefault(task, i)
    return order


def setup_matches(row: Mapping[str, Any], setup: str) -> bool:
    """Whether a trial's setup_mode passes the report's setup option."""
    return setup == "any" or row.get("setup_mode") in _SETUP_MATCHES.get(setup, ())


def _fetch(ledger: Any, f: Filters, o: Options, *, runs: bool = True) -> tuple[list[dict], tuple, dict, Any]:
    _total, raw = ledger.query(f, order="started_at ASC")
    raw = sorted((r for r in raw if setup_matches(r, o.setup)), key=lambda r: (r.get("started_at") or "~",
                                                                              r["trial_key"]))
    excl = ledger.effective_exclusions() if o.apply_exclusions else (frozenset(), frozenset(), ())
    status = {r["run_key"]: r.get("status") for r in ledger.runs()} if runs else {}
    return raw, excl, status, pricing.load_table()


def _compute(ledger: Any, cfg: Config, f: Filters, o: Options) -> _Analysis:
    raw, excl, run_status, table = _fetch(ledger, f, o)
    rows = prepare_rows(raw, o, excl, table)
    for r in rows:
        r["group"] = group_key(r, o)
    used = [r for r in rows if not r["manual"]]
    order = _task_order(cfg)

    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in used:
        by_group[r["group"]].append(r)
    groups = {k: _group(k, rs, o) for k, rs in by_group.items()}
    baselines, warnings = pick_baselines(groups, o)
    for g in groups.values():
        g["is_baseline"] = baselines.get(axis_key(g, o)) == g["key"]
    cells = build_cells(used, o)

    axes: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for key, g in groups.items():
        axes[axis_key(g, o)].append(key)
    comparisons: list[dict] = []
    pairs_tables: dict[str, list[dict]] = {}
    variants: dict[str, dict] | None = None
    ref_baseline = _reference_baseline(cfg.reference)
    by_models = o.compare == "models"
    for axis in sorted(axes):
        base = baselines.get(axis)
        others = [k for k in axes[axis] if k != base]
        if base is None:
            if others:
                one = len(others) == 1
                where = (f"{o.baseline_model} with {groups[others[0]]['harness_name']}" if by_models
                         else f"{axis[0]}:{axis[1]}")
                warnings.append(f"No {'trials' if by_models else o.baseline_family + ' trials'} on {where}, so "
                                f"{len(others)} group{'' if one else 's'} there {'has' if one else 'have'} "
                                "no comparison.")
            continue
        if others and variants is None:
            variants = _variant_cells(raw, excl, table, o)
        bg = groups[base]
        for arm in sorted(others, key=lambda k: (groups[k]["name"], groups[k]["provider"], groups[k]["model"], k)):
            ag = groups[arm]
            key = f"{arm}~{base}"
            # Published figures are harness multiples on one model: never a verdict on a model comparison.
            ref_ok = not by_models and ag["family"] != bg["family"] and bg["family"] == ref_baseline
            here = same_basis(cells, arm, base, o)
            there = {name: same_basis(v, arm, base, o) for name, v in (variants or {}).items()}
            if by_models:
                names = (_model_name(ag, bg), _model_name(bg, ag))
            else:
                names = (ag["name"], bg["name"])
            metrics = {}
            for metric in METRICS:
                ref = (reference_for(cfg.reference, ag["family"], ag["provider"], ag["model"], metric)
                       if ref_ok and metric != "tokens_uncached" else None)
                metrics[metric] = compare_metric(here, arm, base, metric, o, key, variants=there, reference=ref,
                                                 names=names)
            pass_ref = (reference_for(cfg.reference, ag["family"], ag["provider"], ag["model"], "pass")
                        if ref_ok else None)
            comparison: dict[str, Any] = {
                "key": key, "compare": o.compare, "provider": ag["provider"], "model": ag["model"],
                "baseline_provider": bg["provider"], "baseline_model": bg["model"],
                # The harness both sides ran (compare 'models'), else None.
                "harness": ag["harness_name"] if by_models else None,
                "group": arm, "baseline_group": base,
                "arm_name": names[0], "baseline_name": names[1], "headline": "",
                "pairs": _pairs_summary(here, arm, base, metrics["cost"], order), "metrics": metrics,
                "pass": compare_pass(here, arm, base, o, key, reference=pass_ref), "coverage": {}, "caveats": [],
            }
            comparison["coverage"] = coverage_for(comparison, here, groups, cfg, o)
            comparison["caveats"] = caveats_for(comparison, ag, bg, by_group[arm], by_group[base], run_status, o)
            comparison["headline"] = headline(comparison, o.ci_level)
            pairs_tables[key] = _pairs_table(here, arm, base, o, order)
            if o.include_pairs:
                comparison["pairs_table"] = pairs_tables[key]
            comparisons.append(comparison)

    matrix: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in used:
        entry = matrix[r["group"]].setdefault(r["task"], {"valid": 0, "passed": 0, "by_outcome": {}})
        entry["valid"] += 1 if r["included"] else 0
        entry["passed"] += 1 if r["valid_reward"] and r.get("passed") else 0
        entry["by_outcome"][r["outcome"]] = entry["by_outcome"].get(r["outcome"], 0) + 1
    ordered_groups = sorted(groups.values(), key=lambda g: (g["provider"], g["model"], not g["is_baseline"],
                                                            g["name"], g["key"]))
    active = any(run_status.get(r["run_key"]) == "running" for r in used)
    if active:
        warnings.append("A run is still in progress; these figures will change as its trials finish.")
    ids = tuple(excl[2]) if o.apply_exclusions else ()
    outcomes = Counter(r["outcome"] for r in used)
    # A floor set aside from the means was still spent.
    spend = math.fsum(next((v for v in (r["cost"], r["floor_cost"], r.get("reported_cost_usd")) if v is not None), 0.0)
                      for r in used)
    report = {
        "spec": {"filters": filters_dict(f), "options": options_dict(o)},
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "fingerprint": _fingerprint(used, ids), "trials_used": len(used), "exclusions_applied": len(ids),
            "pricing": o.pricing, "pricing_sha": pricing.table_sha(),
            "parser_versions": sorted({r["parser_version"] for r in used if r.get("parser_version") is not None}),
            "classifier_version": classify.CLASSIFIER_VERSION, "stats_version": stats.STATS_VERSION,
            "harbor_versions": sorted({r["harbor_version"] for r in used if r.get("harbor_version")}),
            "includes_active_run": active,
            # How every multiple here reads, so a saved report is never read the other way (saved_report).
            "direction": stats.DIRECTION,
        },
        "scope": task_scope(cfg, f),
        "totals": {
            "trials": len(used), "runs": len({r["run_key"] for r in used}),
            "profiles": sorted({r["profile"] for r in used if r.get("profile")}),
            "by_outcome": {k: outcomes[k] for k in classify.OUTCOMES if outcomes[k]},
            "excluded_manual": len(rows) - len(used), "spend_usd": spend,
            # Trials whose spend is unknown, so spend_usd is a lower bound.
            "usage_unmeasured": sum(1 for r in used if r["usage_status"] == "unmeasured"),
        },
        "groups": ordered_groups,
        "comparisons": comparisons,
        "coverage": {"task_order": list(cfg.tasks or []), "groups": [g["key"] for g in ordered_groups],
                     "cells": {g["key"]: matrix.get(g["key"], {}) for g in ordered_groups}},
        "warnings": warnings,
    }
    return _Analysis(report=report, rows=rows, pairs_tables=pairs_tables)


def _model_name(g: Mapping[str, Any], other: Mapping[str, Any]) -> str:
    """A side of a model comparison by its model; with its provider when the
    other side's model has the same id on another provider."""
    return g["model"] if g["model"] != other["model"] else f"{g['provider']}:{g['model']}"


def _variant_cells(raw: Sequence[Mapping[str, Any]], excl: tuple, table: Any, o: Options) -> dict[str, dict]:
    """Cells under each sensitivity variant; group keys do not depend on
    these options, so every comparison can look its pair up in them."""
    out = {}
    for name, variant in (("crashes_excluded", dataclasses.replace(o, crashes="exclude")),
                          ("misconfigured_cap_included", dataclasses.replace(o, misconfigured_cap="include")),
                          ("same_run_only", dataclasses.replace(o, same_run_only=True))):
        rows = prepare_rows(raw, variant, excl, table)
        for r in rows:
            r["group"] = group_key(r, variant)
        out[name] = build_cells([r for r in rows if not r["manual"]], variant)
    return out


def _fingerprint(rows: Sequence[Mapping[str, Any]], exclusion_ids: Sequence[int]) -> str:
    keys = sorted(f"{r['trial_key']}|{r['outcome']}" for r in rows)
    text = "\n".join(keys) + "\nexclusions:" + ",".join(str(i) for i in sorted(exclusion_ids))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# Memo ---------------------------------------------------------------------------------------------

_memo: OrderedDict[tuple, _Analysis] = OrderedDict()
_memo_lock = threading.Lock()


def _cache_key(ledger: Any, cfg: Config, f: Filters, o: Options) -> tuple:
    request = hashlib.sha256(canonical_json({"filters": filters_dict(f), "options": options_dict(o)})
                             .encode("utf-8")).hexdigest()
    context = hashlib.sha256(json.dumps({"reference": cfg.reference, "tasks": list(cfg.tasks or []),
                                         "tasks_file": tasks_file(cfg)},
                                        sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return (str(getattr(ledger, "path", id(ledger))), id(ledger), request, tuple(ledger.version()),
            pricing.table_sha(), context)


def _analysis(ledger: Any, cfg: Config, f: Filters, o: Options) -> _Analysis:
    key = _cache_key(ledger, cfg, f, o)
    with _memo_lock:
        hit = _memo.get(key)
        if hit is not None:
            _memo.move_to_end(key)
            return hit
    result = _compute(ledger, cfg, f, o)
    with _memo_lock:
        _memo[key] = result
        _memo.move_to_end(key)
        while len(_memo) > CACHE_SIZE:
            _memo.popitem(last=False)
    return result


def clear_cache() -> None:
    with _memo_lock:
        _memo.clear()


# Entry points ---------------------------------------------------------------------------------------

def build_report(ledger: Any, cfg: Config, f: Filters | None = None, o: Options | None = None) -> dict:
    """The Report for the REPORT tab: groups, comparisons, coverage."""
    f = f if f is not None else Filters()
    o = o if o is not None else parse_options(None, cfg)
    return copy.deepcopy(_analysis(ledger, cfg, f, o).report)


def build_coverage(ledger: Any, cfg: Config, f: Filters | None = None, o: Options | None = None) -> dict:
    """Only the coverage contract: what is paired and what to run next."""
    report = build_report(ledger, cfg, f, o)
    touched = {task for cells in report["coverage"]["cells"].values() for task in cells}
    return {
        "provenance": report["provenance"],
        "task_order": report["coverage"]["task_order"],
        "untouched_tasks": [t for t in report["coverage"]["task_order"] if t not in touched],
        "comparisons": [{k: c[k] for k in ("key", "provider", "model", "group", "baseline_group", "headline",
                                           "coverage")} for c in report["comparisons"]],
    }


def build_pairs(ledger: Any, cfg: Config, f: Filters | None, o: Options | None, comparison: str) -> dict:
    """One comparison's per-task table and the trials it could not use.
    KeyError for an unknown comparison key."""
    f = f if f is not None else Filters()
    o = o if o is not None else parse_options(None, cfg)
    analysis = _analysis(ledger, cfg, f, o)
    found = next((c for c in analysis.report["comparisons"] if c["key"] == comparison), None)
    if found is None:
        raise KeyError(f"No comparison '{comparison}' in this report")
    arm, base = found["group"], found["baseline_group"]
    excluded = []
    for r in analysis.rows:
        side = "arm" if r["group"] == arm else "baseline" if r["group"] == base else None
        if side is None:
            continue
        reason = r["excluded_reason"]
        if reason is None and r["floor_set_aside"]:
            reason = "cut short below every uncut trial on its task"
        elif reason is None and r["usage_status"] == "unmeasured":
            # What Harbor wrote: zeros, or null (as from a harness that does not report tokens).
            null = r.get("n_input") is None and r.get("n_output") is None
            reason = "tokens not reported" if null else "no usage recorded"
        elif reason is None and not r["valid_tokens"]:
            reason = "tokens not reported"
        elif reason is None and r.get("fallback_model") and not r["valid_cost"]:
            reason = (f"fallback model: {100 * (1 - (r.get('primary_share') or 0)):.0f}% of its tokens on models "
                      f"other than {r['model']}")
        elif reason is None and not r["valid_cost"]:
            reason = ("no price for " + ", ".join(r["unpriced"])) if r["unpriced"] else "cost not available"
        if reason is not None:
            excluded.append({"trial_key": r["trial_key"], "side": side, "task": r["task"], "outcome": r["outcome"],
                             "reason": reason})
    return {"comparison": comparison, "pairs": copy.deepcopy(analysis.pairs_tables[comparison]),
            "excluded_trials": excluded}


def analysis_rows(ledger: Any, cfg: Config, f: Filters | None = None, o: Options | None = None) -> list[dict]:
    """Every matching trial with its validity under the report policy, in the
    row shape the cost estimates and the assistant read. Hand-excluded
    trials are kept, with excluded_reason set and every valid_* false.

    Without explicit options, cost_usd is the cost computed from current
    list prices whatever report.pricing says: that setting is for the REPORT
    tab only, and estimates and queue reserves must never shrink because a
    display setting trusts the harnesses' (lower) reported figures."""
    f = f if f is not None else Filters()
    o = o if o is not None else dataclasses.replace(parse_options(None, cfg), pricing="current")
    raw, excl, _status, table = _fetch(ledger, f, o, runs=False)
    return [_analysis_row(r) for r in prepare_rows(raw, o, excl, table)]


def _csv_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    return str(value)


# Rows also say how their multiples read (direction) and which filters made them (filters), so a file read
# later, or beside one from before multiples were harness / baseline, cannot be misread.
_PAIR_COLUMNS = ("comparison", "provider", "model", "group", "baseline_group", "task", "digest_short", "run_key",
                 "same_run", "arm_trials", "arm_valid", "arm_tokens_total", "arm_cost", "arm_pass", "arm_trial_keys",
                 "baseline_trials", "baseline_valid", "baseline_tokens_total", "baseline_cost", "baseline_pass",
                 "baseline_trial_keys", "ratio_tokens", "ratio_cost", "direction", "filters")
_COMPARISON_COLUMNS = ("comparison", "provider", "model", "group", "baseline_group", "metric", "status", "n", "need",
                       "geomean", "geomean_lo", "geomean_hi", "sd_log", "ratio_of_means", "ratio_of_means_lo",
                       "ratio_of_means_hi", "delta_pp", "delta_pp_lo", "delta_pp_hi", "delta_method", "robust",
                       "band_lo", "band_hi", "estimator", "applies", "position", "verdict", "headline", "direction",
                       "filters")


def filters_text(f: Filters) -> str:
    """The filters that are set, as canonical JSON: {} for none. Every
    exported row carries it, so a file says what it was limited to."""
    return canonical_json({k: v for k, v in filters_dict(f).items() if v not in (None, [], False)})


def export_rows(ledger: Any, cfg: Config, f: Filters | None, o: Options | None, level: str) -> Iterator[list[str]]:
    """CSV rows (header first) for a reviewer to recompute everything. Fails
    on the call, before any row is produced, for a bad level or ledger."""
    if level not in EXPORT_LEVELS:
        raise ValueError(f"level must be one of {', '.join(EXPORT_LEVELS)}")
    f = f if f is not None else Filters()
    o = o if o is not None else parse_options(None, cfg)
    out: list[list[str]] = []
    used = filters_text(f)
    if level == "trials":
        out.append([*ANALYSIS_KEYS, "filters"])
        out += [[_csv_text(row[k]) for k in ANALYSIS_KEYS] + [used] for row in analysis_rows(ledger, cfg, f, o)]
        return iter(out)
    analysis = _analysis(ledger, cfg, f, o)
    comparisons = analysis.report["comparisons"]
    if level == "pairs":
        out.append(list(_PAIR_COLUMNS))
        for c in comparisons:
            head = [c["key"], c["provider"], c["model"], c["group"], c["baseline_group"]]
            for p in analysis.pairs_tables[c["key"]]:
                values = [p["task"], p["digest_short"], p["run_key"], p["same_run"]]
                for side in ("arm", "baseline"):
                    s = p[side]
                    values += [s["trials"], s["valid"], s["tokens_total"], s["cost"], s["pass"], s["trial_keys"]]
                values += [p["ratio_tokens"], p["ratio_cost"], stats.DIRECTION, used]
                out.append([_csv_text(v) for v in head + values])
        return iter(out)
    out.append(list(_COMPARISON_COLUMNS))
    for c in comparisons:
        head = [c["key"], c["provider"], c["model"], c["group"], c["baseline_group"]]
        for metric, m in c["metrics"].items():
            gm, rom, ref = m["geomean"] or {}, m["ratio_of_means"] or {}, m["reference"] or {}
            band = ref.get("band") or [None, None]
            values = [metric, m["status"], m["n"], m["need"], gm.get("estimate"), gm.get("lo"), gm.get("hi"),
                      gm.get("sd_log"), rom.get("estimate"), rom.get("lo"), rom.get("hi"), None, None, None, None,
                      m["robust"], band[0], band[1], ref.get("estimator"), ref.get("applies"), ref.get("position"),
                      ref.get("verdict"), c["headline"], stats.DIRECTION, used]
            out.append([_csv_text(v) for v in head + values])
        p = c["pass"]
        ref = p["reference"] or {}
        band = ref.get("band") or [None, None]
        d = p["delta_pp"]
        values = ["pass", p["status"], p["n"], p["need"], None, None, None, None, None, None, None, d["estimate"],
                  d["lo"], d["hi"], d["method"], None, band[0], band[1], None, ref.get("applies"), None,
                  ref.get("verdict"), c["headline"], stats.DIRECTION, used]
        out.append([_csv_text(v) for v in head + values])
    return iter(out)


# Saved reports ----------------------------------------------------------------------------------------

_SWAPPED = {"above": "below", "below": "above"}


def _inverse(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        return value
    return 1.0 / value


def _invert_interval(block: Any) -> None:
    if isinstance(block, dict):
        lo, hi = block.get("lo"), block.get("hi")
        block.update(estimate=_inverse(block.get("estimate")), lo=_inverse(hi), hi=_inverse(lo))


def _swap_quoted(text: Any) -> Any:
    """'above' and 'below' swapped inside a saved sentence (a withheld verdict's reason)."""
    if not isinstance(text, str):
        return text
    return re.sub(r"'(above|below)'", lambda m: f"'{_SWAPPED[m.group(1)]}'", text)


def _invert_metric(m: Any) -> None:
    if not isinstance(m, dict):
        return
    for key in ("geomean", "ratio_of_means"):
        _invert_interval(m.get(key))
    q = m.get("per_task")
    if isinstance(q, dict):
        # min and max invert exactly. q1, median and q3 were interpolated between ratios the other way up, and
        # interpolation does not commute with 1/x, so a fresh report's can differ (under 1% on the 2026-10-30
        # ledger, more for an even count with far-apart middles); the snapshot view says so.
        m["per_task"] = {"min": _inverse(q.get("max")), "q1": _inverse(q.get("q3")),
                         "median": _inverse(q.get("median")), "q3": _inverse(q.get("q1")),
                         "max": _inverse(q.get("min"))}
    loo = m.get("leave_one_out")
    if isinstance(loo, list) and len(loo) == 2:
        m["leave_one_out"] = [_inverse(loo[1]), _inverse(loo[0])]
    if isinstance(m.get("sensitivity"), dict):
        m["sensitivity"] = {k: _inverse(v) for k, v in m["sensitivity"].items()}
    ref = m.get("reference")
    if not isinstance(ref, dict):
        return
    band = ref.get("band")
    if isinstance(band, list) and len(band) == 2:
        ref["band"] = [_inverse(band[1]), _inverse(band[0])]
    published = ref.get("published")
    if isinstance(published, dict):
        if "value" in published:
            published["value"] = _inverse(published["value"])
        if "low" in published and "high" in published:
            published["low"], published["high"] = _inverse(published["high"]), _inverse(published["low"])
    for key in ("position", "verdict"):
        ref[key] = _SWAPPED.get(ref.get(key), ref.get(key))
    ref["withheld"] = _swap_quoted(ref.get("withheld"))


def saved_report(response: Any, request: Any, cfg: Config) -> Any:
    """A saved report (a snapshot's response) in today's reading. One saved
    before multiples were harness / baseline (no provenance.direction) is
    converted: every multiple, interval, band and published figure inverted,
    'above' and 'below' swapped, headlines rewritten, and provenance names
    converted_from, so its numbers are never read the wrong way round. Pass
    rates are harness minus baseline either way. One saved without a scope
    gets one from its request's filters."""
    if not isinstance(response, Mapping):
        return response
    out = copy.deepcopy(dict(response))
    provenance = out["provenance"] = dict(_mapping(out.get("provenance")))
    if "scope" not in out:
        filters = request.get("filters") if isinstance(request, Mapping) else None
        try:
            out["scope"] = task_scope(cfg, parse_filters(filters))
        except ValueError:
            out["scope"] = None
    if provenance.get("direction") == stats.DIRECTION:
        return out
    level = _mapping(_mapping(out.get("spec")).get("options")).get("ci_level") or 0.90
    for c in out.get("comparisons") or []:
        if not isinstance(c, dict):
            continue
        for m in _mapping(c.get("metrics")).values():
            _invert_metric(m)
        for p in c.get("pairs_table") or []:
            if isinstance(p, dict):
                p.update(ratio_tokens=_inverse(p.get("ratio_tokens")), ratio_cost=_inverse(p.get("ratio_cost")))
        c["caveats"] = [_swap_quoted(t) for t in c.get("caveats") or []]
        try:
            c["headline"] = headline(c, float(level))
        except (KeyError, TypeError, ValueError):
            c["headline"] = (f"{c.get('arm_name')} vs {c.get('baseline_name')}: saved before multiples were "
                             "harness / baseline; read the figures, not this line")
    provenance.update(direction=stats.DIRECTION, converted_from=provenance.get("direction") or "baseline_over_harness")
    return out
