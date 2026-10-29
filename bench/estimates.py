"""What a run design is likely to cost and how long it will take, from what
the ledger has measured.

A person approving a design, and the assistant proposing one, need a cost
before anything is billed. Every figure here comes from trials already in
the ledger, read through report.analysis_rows so validity (a setup failure
is not a cheap run) is decided exactly as the REPORT tab decides it. Where
the ledger has nothing for a cell, the estimate says so and falls back in a
fixed order: the same configuration, any configuration of the family on the
model, the family's tokens on another model repriced, then a prior. Each
arm's figure carries that basis and a confidence label, so a guess is never
quoted as a measurement.

The Monte Carlo is seeded from the spec hash and the ledger fingerprint, so
the same design over the same ledger gives byte-identical output, and a
cached check can be compared with a fresh one.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import datasets, designs, extract, harnesses, pricing, report, stats
from .settings import Config, Price, RunSettings
from .spec import ArmSpec, spec_from_design

REPLICATES = 2000
# The estimate runs inside tool calls, queue adds and the live design check,
# so its work is bounded: at most MAX_DRAWS trial draws in all (never under
# MIN_REPLICATES replicates), and per-trial quantiles from the first
# TRIAL_DRAW_REPLICATES replicates only. Arm and run totals use every replicate.
MAX_DRAWS = 2_000_000
MIN_REPLICATES = 200
TRIAL_DRAW_REPLICATES = 200
CAP_SLACK = 1.1             # a harness stops a trial just after its cap, not on it
PREFLIGHT_MINUTES = 1.0     # per arm: dataset check and Harbor start-up
BUDGET_MARGIN = 1.15        # suggested budget = p90 x this, rounded up
BUDGET_STEP = 0.5
MEASURED_MIN_TASKS = 3
HIGH_CONFIDENCE_TASKS = 8
MIN_POOL = 3                # fewer values than this get an extra spread
INFRA_NOTE_RATE = 0.3
DEFAULT_TIMEOUT_SEC = 900.0
CONSENSUS_MIDPOINT = 2.5    # the middle of the published 2x-3x
MATCHES = ("cell", "family")
BASES = ("measured", "measured_any_options", "borrowed", "prior")

_MANUAL_PREFIXES = ("excluded by hand", "run excluded by hand")

ASSUMPTIONS = (
    "Harbor does not report cache writes; trial cache writes are priced at the input rate.",
    "Infrastructure failures are not discounted from spend.",
    "Each trial resamples measured per-trial costs of its task; tasks never seen draw from tasks of the same "
    "difficulty, then from the whole cell.",
    "Figures from fewer than 3 trials, from another model or from a prior carry an extra spread set by "
    "stats.prior_sd_log, once per arm (the level is uncertain); a trial drawn from fewer than 3 measured values "
    "gets the same spread on its own.",
    "Pairs after the run assume the new trials succeed and run the same task versions as earlier ones; trials "
    "pair by the report's rule (report.match_on, report.same_run_only).",
    "Wall time: 1 minute of preflight per arm, trials longest first on n_concurrent slots, arms in design "
    "order on parallel_arms slots.",
)


# Rows ------------------------------------------------------------------------------------------

def is_manual(row: Mapping[str, Any]) -> bool:
    """Excluded by a person (trial or run exclusion), not by the report policy."""
    return str(row.get("excluded_reason") or "").startswith(_MANUAL_PREFIXES)


def usable(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r for r in rows if not is_manual(r)]


def is_infra(row: Mapping[str, Any]) -> bool:
    return row.get("outcome") in report.RELIABILITY_OUTCOMES


def _finished(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r for r in rows if r.get("outcome") != "incomplete"]


def cost_of(row: Mapping[str, Any]) -> float | None:
    value = row.get("cost_usd")
    return float(value) if row.get("valid_cost") and value is not None else None


def _unrecorded(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Counted trials whose usage was not recorded: no cost of their own (a
    plan draws them high, _model), but their wall time is real, and a timeout
    before any model call completed is the longest trial there is."""
    return [r for r in rows if r.get("usage_status") == "unmeasured" and r.get("valid_reward")]


def metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    """A row's value for a pairing metric, or None when it does not count."""
    if metric == "cost":
        value = cost_of(row)
    elif metric in ("tokens", "tokens_total"):
        value = row.get("tokens_total") if row.get("valid_tokens") else None
    else:
        raise ValueError(f"Unknown metric {metric!r}")
    return float(value) if value is not None and value > 0 else None


def pair_options(cfg: Config) -> report.Options:
    """The profile's report options: their pairing rule decides which trials
    pair here, so pair counts match the REPORT tab's."""
    return report.parse_options(None, cfg)


def pair_key(row: Mapping[str, Any], o: report.Options) -> tuple[str, ...]:
    """What two trials of one model pair on, by the report's rule: the task's
    content digest (its name with match_on name), plus the run with
    same_run_only."""
    key = tuple(report.task_id(row, o.match_on))
    return key + (str(row.get("run_key") or ""),) if o.same_run_only else key


def spend_of(row: Mapping[str, Any]) -> float | None:
    """What the run budget counts for a trial (results.spend's rule): the
    larger of computed and reported cost, unless the harness prices this
    provider's models at another provider's rates."""
    computed, reported = row.get("cost_usd"), row.get("reported_cost_usd")
    if computed is None and reported is None:
        return None
    if computed is not None and harnesses.reports_mispriced_cost(row.get("family") or "", row.get("provider") or ""):
        return float(computed)
    return max(float(computed or 0.0), float(reported or 0.0))


def _quantile(ordered: Sequence[float], q: float) -> float | None:
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def _money(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _mins(value: float | None) -> float | None:
    return None if value is None else round(float(value), 2)


def _sample_order(cfg: Config) -> dict[str, int]:
    order: dict[str, int] = {}
    for i, task in enumerate(cfg.tasks or []):
        order.setdefault(task, i)
    return order


def task_meta(cfg: Config, dataset: str | None = None) -> dict[str, dict[str, Any]]:
    try:
        return datasets.task_meta(dataset or cfg.dataset) or {}
    except Exception:  # planning advice never fails on missing metadata
        return {}


# Cell statistics ---------------------------------------------------------------------------------

def cell_stats(cfg: Config, rows: Sequence[Mapping[str, Any]], family: str | None = None,
               provider: str | None = None, model: str | None = None, match: str = "cell",
               per_task: bool = False) -> list[dict[str, Any]]:
    """Measured per-trial figures per configuration ("cell": family,
    provider, model and config hash) or per family on each model."""
    if match not in MATCHES:
        raise ValueError(f"match must be one of {', '.join(MATCHES)}")
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        if family and r.get("family") != family:
            continue
        if provider and r.get("provider") != provider:
            continue
        if model and r.get("model") != model:
            continue
        key = (str(r.get("provider")), str(r.get("model")), str(r.get("family")))
        if match == "cell":
            key += (str(r.get("config_hash") or ""),)
        groups[key].append(r)
    order = _sample_order(cfg)
    return [_cell_summary(cfg, key, groups[key], match, per_task, order) for key in sorted(groups)]


def _summary(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"mean": _money(math.fsum(ordered) / len(ordered)) if ordered else None,
            "median": _money(_quantile(ordered, 0.5)), "p90": _money(_quantile(ordered, 0.9)), "n": len(ordered)}


def pass_rate(rows: Sequence[Mapping[str, Any]], level: float) -> dict[str, Any] | None:
    """Task-weighted, like the report's group pass rate: each task's mean
    over its reward-valid trials."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("valid_reward"):
            by_task[str(r.get("task"))].append(1.0 if r.get("passed") else 0.0)
    if not by_task:
        return None
    successes = math.fsum(math.fsum(v) / len(v) for v in by_task.values())
    ci = stats.wilson(successes, len(by_task), level)
    return {"estimate": round(successes / len(by_task), 4), "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
            "tasks": len(by_task)}


def _cell_summary(cfg: Config, key: tuple[str, ...], rows: list[Mapping[str, Any]], match: str, per_task: bool,
                  order: Mapping[str, int]) -> dict[str, Any]:
    provider, model, family = key[0], key[1], key[2]
    used = usable(rows)
    finished = _finished(used)
    infra = [r for r in finished if is_infra(r)]
    outcomes = Counter(str(r.get("outcome")) for r in used)
    costs = [c for c in (cost_of(r) for r in used) if c is not None]
    spends = [s for s in (spend_of(r) for r in used if r.get("valid_tokens")) if s is not None]
    reported = [float(r["reported_cost_usd"]) for r in used if r.get("reported_cost_usd") is not None]
    measured = [r for r in used if r.get("valid_tokens")]
    causes = Counter((str(r.get("outcome")), r.get("exception_type") or None) for r in used
                     if r.get("outcome") not in ("passed", "failed"))
    configs = Counter(str(r.get("config_hash") or "") for r in used)
    labels = {str(r.get("config_hash") or ""): r.get("config_label") for r in used}

    def mean_of(name: str) -> float | None:
        values = [float(r[name]) for r in measured if r.get(name) is not None]
        return round(math.fsum(values) / len(values), 1) if values else None

    fresh = [max(float(r.get("n_input") or 0) - float(r.get("n_cache") or 0), 0.0) for r in measured]
    out: dict[str, Any] = {
        "family": family, "provider": provider, "model": model,
        "trials": len(rows), "excluded_by_hand": len(rows) - len(used),
        "by_outcome": {k: outcomes[k] for k in sorted(outcomes)},
        "cost_valid": len(costs), "tasks": len({r.get("task") for r in used}),
        "effort": sorted({str(r.get("effort") or "default") for r in used}),
        "cost_per_trial": _summary(costs),
        "spend_per_trial": _summary(spends),
        "reported_cost_per_trial": {"mean": _money(math.fsum(reported) / len(reported)) if reported else None,
                                    "n": len(reported),
                                    "harness_misprices_provider": harnesses.reports_mispriced_cost(family, provider)},
        "tokens_per_trial": {"fresh_input": round(math.fsum(fresh) / len(fresh), 1) if fresh else None,
                             "cache": mean_of("n_cache"), "output": mean_of("n_output"),
                             "total": mean_of("tokens_total"), "n": len(measured)},
        "minutes_per_trial": {
            "agent": _minute_summary([r.get("agent_minutes") for r in measured + _unrecorded(used)]),
            "total": _minute_summary([r.get("total_minutes") for r in measured + _unrecorded(used)]),
        },
        "pass_rate": pass_rate(used, cfg.stats.confidence),
        "infra_failures": {"rate": round(len(infra) / len(finished), 4) if finished else None,
                           "trials": len(infra), "finished": len(finished)},
        "top_failure_causes": [{"outcome": o, "exception_type": e, "n": n}
                               for (o, e), n in sorted(causes.items(), key=lambda kv: (-kv[1], str(kv[0])))[:3]],
        "unpriced_trials": sum(1 for r in measured if r.get("cost_usd") is None),
    }
    if match == "cell":
        first = used[0] if used else rows[0]
        out["config_hash"] = key[3] or None
        out["config_label"] = first.get("config_label")
        out["options"] = {k: v for k, v in sorted(dict(first.get("options") or {}).items())}
    else:
        out["configs"] = [{"config_hash": h or None, "label": labels.get(h), "trials": n}
                          for h, n in sorted(configs.items(), key=lambda kv: (-kv[1], kv[0]))]
    if per_task:
        out["per_task"] = _per_task(used, order)
    return out


def _minute_summary(values: Sequence[Any]) -> dict[str, Any]:
    ordered = sorted(float(v) for v in values if v is not None)
    return {"median": _mins(_quantile(ordered, 0.5)), "p90": _mins(_quantile(ordered, 0.9)), "n": len(ordered)}


def _per_task(rows: Sequence[Mapping[str, Any]], order: Mapping[str, int]) -> list[dict[str, Any]]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_task[str(r.get("task"))].append(r)
    out = []
    for task in sorted(by_task, key=lambda t: (order.get(t, len(order)), t)):
        rs = by_task[task]
        costs = [c for c in (cost_of(r) for r in rs) if c is not None]
        tokens = [float(r["tokens_total"]) for r in rs if r.get("valid_tokens") and r.get("tokens_total") is not None]
        minutes = sorted(float(r["total_minutes"]) for r in rs if r.get("total_minutes") is not None)
        out.append({
            "task": task, "position": order[task] + 1 if task in order else None, "trials": len(rs),
            "cost_valid": len(costs), "cost_mean": _money(math.fsum(costs) / len(costs)) if costs else None,
            "tokens_mean": round(math.fsum(tokens) / len(tokens), 1) if tokens else None,
            "passes": sum(1 for r in rs if r.get("valid_reward") and r.get("passed")),
            "infra": sum(1 for r in rs if is_infra(r)),
            "minutes_median": _mins(_quantile(minutes, 0.5)),
        })
    return out


# Cell models ---------------------------------------------------------------------------------------

@dataclass
class CellModel:
    """Where an arm's per-trial cost and minutes come from, as resampling pools."""
    basis: str                      # measured | measured_any_options | borrowed | prior
    detail: str
    confidence: str                 # high | medium | low
    per_task: dict[str, list[float]]
    pool: list[float]
    by_difficulty: dict[str, dict[str, list[float]]]    # difficulty -> task -> values
    minutes_per_task: dict[str, list[float]]
    minutes_pool: list[float]
    p_infra: float
    n_trials: int
    n_tasks: int
    runs: list[str]
    infra_finished: int = 0         # finished trials p_infra was measured over
    config_hash: str | None = None
    unpriced: bool = False
    notes: list[str] = field(default_factory=list)


def baseline_family(cfg: Config) -> str:
    """The family the report compares against: the profile baseline arm's."""
    try:
        arm = cfg.arm(cfg.baseline_arm)
        return arm.family or arm.agent
    except Exception:
        return report.REFERENCE_BASELINE


def reference_multiple(reference: Mapping[str, Any] | None, family: str) -> tuple[float, str]:
    """The published baseline/harness multiple for a family, and where it
    came from: its tokens figure, else its cost figure, else the consensus."""
    reference = reference if isinstance(reference, Mapping) else {}
    entry = (reference.get("arms") or {}).get(family) if isinstance(reference.get("arms"), Mapping) else None
    entry = entry if isinstance(entry, Mapping) else {}
    for name in ("tokens_per_trial", "cost_per_trial"):
        value = _figure(entry.get(name))
        if value is not None:
            return value, f"reference {family} {name}"
    consensus = reference.get("consensus") if isinstance(reference.get("consensus"), Mapping) else {}
    value = _figure(consensus.get("cost_per_trial"))
    if value is not None:
        return value, "reference consensus cost_per_trial"
    return CONSENSUS_MIDPOINT, "the published 2x-3x midpoint"


def _figure(published: Any) -> float | None:
    if not isinstance(published, Mapping):
        return None
    try:
        if published.get("value") is not None:
            value = float(published["value"])
        elif published.get("low") is not None and published.get("high") is not None:
            value = (float(published["low"]) + float(published["high"])) / 2.0
        else:
            return None
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def config_hash_for(arm: ArmSpec, kwargs: Mapping[str, Any], timeout_multiplier: float = 1.0) -> str:
    """The config hash the ledger will give this arm's trials: the same
    identity extract.config_identity computes from a trial's config."""
    implementation = arm.agent if arm.kind == "import_path" else "builtin"
    trial_cfg = {"agent": {"kwargs": dict(kwargs)}, "timeout_multiplier": timeout_multiplier}
    return extract.config_identity(trial_cfg, arm.family or arm.agent, implementation, arm.provider, arm.model)[0]


def planned_kwargs(cfg: Config, arm: ArmSpec, run: RunSettings | None = None) -> dict[str, Any] | None:
    """The kwargs the planner would pass this arm under these run settings;
    None when the agent is not allowed in designs, or its schema cannot be
    read or rejects the arm."""
    try:
        # Imported here: runs pulls in process supervision, and only this guess needs it.
        from . import runs, schemas
        # Harbor imports an agent's code to print its schema, so only an
        # allowed harness is ever looked up: a stored design or a narrowed
        # allowlist must not turn an estimate into code execution.
        if arm.agent not in {h["agent"] for h in designs.allowed_harnesses(cfg)}:
            return None
        kwargs, _ = runs.agent_kwargs(run or cfg.run, arm, schemas.agent_schema(arm.agent))
    except Exception:
        return None
    return kwargs


def guess_config_hash(cfg: Config, arm: ArmSpec, run: RunSettings | None = None) -> str | None:
    """config_hash_for with the kwargs the planner would pass."""
    run = run or cfg.run
    kwargs = planned_kwargs(cfg, arm, run)
    return None if kwargs is None else config_hash_for(arm, kwargs, run.timeout_multiplier)


def _values_by_task(rows: Sequence[Mapping[str, Any]], value: Callable[[Mapping[str, Any]], float | None]
                    ) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = value(r)
        if v is not None:
            out[str(r.get("task"))].append(float(v))
    return {k: out[k] for k in sorted(out)}


def _by_difficulty(per_task: Mapping[str, list[float]], meta: Mapping[str, Mapping[str, Any]]
                   ) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for task in sorted(per_task):
        difficulty = (meta.get(task) or {}).get("difficulty")
        if difficulty:
            out[str(difficulty)][task] = list(per_task[task])
    return {k: out[k] for k in sorted(out)}


def _infra_rate(rows: Sequence[Mapping[str, Any]]) -> tuple[float, int]:
    finished = _finished(rows)
    if not finished:
        return 0.0, 0
    return sum(1 for r in finished if is_infra(r)) / len(finished), len(finished)


def _minutes(row: Mapping[str, Any]) -> float | None:
    value = row.get("total_minutes")
    return float(value) if value is not None and value > 0 else None


def _model(basis: str, detail: str, rows: Sequence[Mapping[str, Any]], per_task: dict[str, list[float]],
           meta: Mapping[str, Mapping[str, Any]], p_infra: tuple[float, int], *, pool: list[float] | None = None,
           minute_rows: Sequence[Mapping[str, Any]] | None = None, config_hash: str | None = None,
           unrecorded: Sequence[Mapping[str, Any]] = ()) -> CellModel:
    """unrecorded: counted trials of these measured ones' cell that ran but
    recorded no usage. Each is drawn at the cell's most expensive measured
    trial rather than left out, which would price the tasks a harness timed
    out on like its typical ones; the confidence still counts measured tasks."""
    minute_rows = rows if minute_rows is None else minute_rows
    minutes = _values_by_task(minute_rows, _minutes)
    n_tasks = len(per_task)
    notes = []
    top = max((v for vs in per_task.values() for v in vs), default=None)
    if unrecorded and top is not None and pool is None:
        per_task = {t: list(vs) for t, vs in per_task.items()}
        for r in unrecorded:
            per_task.setdefault(str(r.get("task")), []).append(top)
        per_task = {k: per_task[k] for k in sorted(per_task)}
        k = len(unrecorded)
        notes.append(f"{k} trial{'s' if k != 1 else ''} that ran but recorded no usage {'are' if k != 1 else 'is'} "
                     f"drawn at the most expensive measured trial (${top:.4f}): such trials often ran to the time "
                     "limit.")
    values = [v for task in sorted(per_task) for v in per_task[task]] if pool is None else list(pool)
    if basis == "measured" and n_tasks >= HIGH_CONFIDENCE_TASKS:
        confidence = "high"
    elif basis in ("measured", "measured_any_options") and n_tasks >= MEASURED_MIN_TASKS:
        confidence = "medium"
    else:
        confidence = "low"
    return CellModel(
        basis=basis, detail=detail, confidence=confidence, per_task=per_task, pool=values,
        by_difficulty=_by_difficulty(per_task, meta), minutes_per_task=minutes,
        minutes_pool=[v for task in sorted(minutes) for v in minutes[task]], p_infra=p_infra[0],
        n_trials=len(rows), n_tasks=n_tasks, runs=sorted({str(r.get("run_key")) for r in rows}),
        infra_finished=p_infra[1], config_hash=config_hash, notes=notes,
    )


def cell_model(cfg: Config, rows: Sequence[Mapping[str, Any]], arm: ArmSpec, *, config_hash: str | None = None,
               meta: Mapping[str, Mapping[str, Any]] | None = None, family_only: bool = False,
               guess: bool = True) -> CellModel:
    """The first basis that applies, in order: measured (this configuration,
    at least 3 tasks), measured_any_options (the family on this model),
    borrowed (the family's tokens on another model, repriced here), prior.

    Without config_hash, guess=True guesses it from the planner's kwargs for
    the profile's run settings; a caller that already tried passes
    guess=False, so a failed lookup is not repeated. family_only skips the
    configuration level (coverage gaps)."""
    meta = task_meta(cfg) if meta is None else meta
    rows = usable(rows)
    family = arm.family or arm.agent
    provider, model = arm.provider, arm.model
    on_model = [r for r in rows if r.get("family") == family and r.get("provider") == provider
                and r.get("model") == model]
    if config_hash is None and guess and not family_only:
        config_hash = guess_config_hash(cfg, arm)
    cell_rows = [r for r in on_model if config_hash and r.get("config_hash") == config_hash]
    infra = _infra_rate(cell_rows) if _finished(cell_rows) else _infra_rate(on_model)
    where = f"{provider}:{model}"

    valid = [r for r in cell_rows if cost_of(r) is not None]
    per_task = _values_by_task(valid, cost_of)
    if not family_only and len(per_task) >= MEASURED_MIN_TASKS:
        return _model("measured", f"{len(valid)} trials on {len(per_task)} tasks of this configuration on {where}",
                      valid, per_task, meta, infra, minute_rows=valid + _unrecorded(cell_rows),
                      config_hash=config_hash, unrecorded=_unrecorded(cell_rows))

    valid = [r for r in on_model if cost_of(r) is not None]
    per_task = _values_by_task(valid, cost_of)
    if per_task:
        return _model("measured_any_options",
                      f"{len(valid)} trials on {len(per_task)} tasks of {family} on {where}, any options",
                      valid, per_task, meta, infra, minute_rows=valid + _unrecorded(on_model), config_hash=config_hash,
                      unrecorded=_unrecorded(on_model))

    price = cfg.price(provider, model)
    others = [r for r in rows if r.get("family") == family and (r.get("provider"), r.get("model")) != (provider, model)
              and r.get("valid_tokens") and r.get("n_input") is not None and r.get("n_output") is not None]
    same_provider = [r for r in others if r.get("provider") == provider]
    borrowed = same_provider or others
    if borrowed and price is not None:
        def repriced(r: Mapping[str, Any], p: Price = price) -> float:
            return pricing.unit_cost(p, int(r.get("n_input") or 0), int(r.get("n_cache") or 0),
                                     int(r.get("n_output") or 0))
        per_task = {t: [v for v in vs if v > 0] for t, vs in _values_by_task(borrowed, repriced).items()}
        per_task = {t: vs for t, vs in per_task.items() if vs}
        if per_task:
            sources = sorted({f"{r.get('provider')}:{r.get('model')}" for r in borrowed})
            return _model("borrowed", f"{len(borrowed)} trials of {family} on {', '.join(sources)}, repriced at "
                          f"{where} prices", borrowed, per_task, meta, infra, config_hash=config_hash)

    base_family = baseline_family(cfg)
    if family != base_family:
        base = [r for r in rows if r.get("family") == base_family and r.get("provider") == provider
                and r.get("model") == model and cost_of(r) is not None]
        if base:
            multiple, source = reference_multiple(cfg.reference, family)
            directions = (cfg.reference or {}).get("defaults") if isinstance(cfg.reference, Mapping) else None
            harness_over = isinstance(directions, Mapping) and directions.get("direction") == "harness_over_baseline"
            scale = multiple if harness_over else 1.0 / multiple

            def scaled(r: Mapping[str, Any], k: float = scale) -> float | None:
                c = cost_of(r)
                return None if c is None else c * k
            per_task = _values_by_task(base, scaled)
            return _model("prior", f"{base_family} on {where} {'x' if harness_over else '÷'} {multiple:g} "
                          f"({source})", base, per_task, meta, infra, config_hash=config_hash)

    others = [r for r in rows if r.get("family") != family and r.get("provider") == provider
              and r.get("model") == model and cost_of(r) is not None]
    if others:
        median = statistics.median(sorted(cost_of(r) for r in others))  # type: ignore[type-var]
        return _model("prior", f"median trial cost of other families on {where} ({len(others)} trials)", [], {},
                      meta, infra, pool=[float(median)], minute_rows=others, config_hash=config_hash)

    tokens = cfg.stats.default_trial_tokens
    n_input = int(tokens["input"])
    n_cache = int(round(n_input * float(tokens["cache_share"])))
    n_output = int(tokens["output"])
    detail = (f"default trial tokens ({n_input} in, {float(tokens['cache_share']):.0%} cached, "
              f"{n_output} out) at {where} prices")
    if price is None:
        out = _model("prior", detail, [], {}, meta, infra, pool=[], config_hash=config_hash)
        out.unpriced = True
        out.notes.append(f"No price for {model} under {provider} in config/pricing.yaml; its cost is unknown.")
        return out
    return _model("prior", detail, [], {}, meta, infra, pool=[pricing.unit_cost(price, n_input, n_cache, n_output)],
                  config_hash=config_hash)


def family_trial_cost(cfg: Config, rows: Sequence[Mapping[str, Any]], family: str, provider: str, model: str,
                      meta: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[float | None, str]:
    """(mean cost per trial, basis) for any configuration of a family on a
    model, for pricing coverage gaps."""
    arm = ArmSpec(name="gap", agent=family, family=family, kind="name", provider=provider, model=model,
                  effort="inherit", options={}, tasks=())
    m = cell_model(cfg, rows, arm, meta=meta, family_only=True)
    if not m.pool:
        return None, m.basis
    return math.fsum(m.pool) / len(m.pool), m.basis


# The estimate ----------------------------------------------------------------------------------------

def _plan_arms(plan: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    arms = (plan or {}).get("arms") if isinstance(plan, Mapping) else None
    return {a.get("name"): a for a in arms or [] if isinstance(a, Mapping) and a.get("name")}


def _kwargs(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    kwargs = entry.get("kwargs")
    if not isinstance(kwargs, list):
        return None
    return {k["name"]: k.get("value") for k in kwargs if isinstance(k, Mapping) and isinstance(k.get("name"), str)}


def enforced_cap(arm: ArmSpec, kwargs: Mapping[str, Any]) -> tuple[float | None, list[str]]:
    """The dollar cap the harness really enforces per trial, and notes on caps
    it would misprice (those never bound an estimate)."""
    caps, notes = [], []
    for name in harnesses.DOLLAR_CAPS:
        value = kwargs.get(name)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number) or number <= 0:
            continue
        if harnesses.cap_misprices_for(arm.family or arm.agent, arm.provider, name):
            notes.append(f"{name} is priced at another provider's rates on {arm.provider}, so it may stop trials "
                         "early; it does not bound this estimate.")
            continue
        caps.append(number)
    return (min(caps) if caps else None), notes


@dataclass
class _ArmSim:
    arm: ArmSpec
    model: CellModel
    cap: float | None
    # Per trial: (cost values, their count, spread them?, minute values, their count, spread them?,
    # fallback minutes).
    trials: list[tuple[list[float], int, bool, list[float], int, bool, float]]
    level_noise: bool
    costs: list[float] = field(default_factory=list)            # first TRIAL_DRAW_REPLICATES replicates
    totals: list[float] = field(default_factory=list)           # every replicate
    trial_minutes: list[float] = field(default_factory=list)    # first TRIAL_DRAW_REPLICATES replicates
    arm_minutes: list[float] = field(default_factory=list)      # every replicate

    @property
    def priced(self) -> bool:
        return bool(self.model.pool)


def _choices(task: str, per_task: Mapping[str, list[float]],
             by_difficulty: Mapping[str, Mapping[str, list[float]]], pool: list[float],
             meta: Mapping[str, Mapping[str, Any]]) -> list[float]:
    """The values a trial of this task resamples: its own, else those of at
    least 2 observed tasks of its difficulty (one task's repeats are not a
    difficulty's spread), else the whole cell's."""
    values = per_task.get(task)
    if values:
        return values
    difficulty = (meta.get(task) or {}).get("difficulty")
    same = by_difficulty.get(str(difficulty)) if difficulty else None
    if same and sum(1 for vs in same.values() if vs) >= 2:
        return [v for t in sorted(same) for v in same[t]]
    return pool


def _fallback_minutes(task: str, meta: Mapping[str, Mapping[str, Any]], multiplier: float) -> float:
    timeout = (meta.get(task) or {}).get("agent_timeout_sec")
    try:
        seconds = float(timeout) if timeout is not None else DEFAULT_TIMEOUT_SEC
    except (TypeError, ValueError):
        seconds = DEFAULT_TIMEOUT_SEC
    return 0.5 * seconds * multiplier / 60.0


def _sim(arm: ArmSpec, model: CellModel, cap: float | None, attempts: int, meta: Mapping[str, Mapping[str, Any]],
         multiplier: float) -> _ArmSim:
    trials = []
    minutes_by_difficulty = _by_difficulty(model.minutes_per_task, meta)
    for task in arm.tasks:
        costs = _choices(task, model.per_task, model.by_difficulty, model.pool, meta)
        minutes = _choices(task, model.minutes_per_task, minutes_by_difficulty, model.minutes_pool, meta)
        fallback = _fallback_minutes(task, meta, multiplier)
        # Resampling one or two values has no spread of its own, though a
        # repeat of the same task costs differently, so those draws get one.
        trials += [(costs, len(costs), len(costs) < MIN_POOL, minutes, len(minutes), len(minutes) < MIN_POOL,
                    fallback)] * attempts
    return _ArmSim(arm=arm, model=model, cap=cap, trials=trials,
                   level_noise=model.basis in ("borrowed", "prior") or len(model.pool) < MIN_POOL)


def effective_replicates(requested: int, trials: int) -> int:
    """The replicates an estimate runs: as requested, cut so the draws stay
    within MAX_DRAWS, never under MIN_REPLICATES (nor over the request)."""
    requested = max(1, int(requested))
    return min(requested, max(MIN_REPLICATES, MAX_DRAWS // max(1, trials)))


def _simulate(sims: list[_ArmSim], run: RunSettings, replicates: int, rng: random.Random, sigma: float
              ) -> tuple[list[float], list[float]]:
    totals, walls = [], []
    # Indexing by random() and skipping one-value draws keeps this loop cheap; it stays seeded.
    rnd, gauss, exp = rng.random, rng.gauss, math.exp
    shift = sigma * sigma / 2.0
    for b in range(replicates):
        keep = b < TRIAL_DRAW_REPLICATES
        run_total = 0.0
        durations = []
        for sim in sims:
            priced = sim.priced
            level = exp(sigma * gauss(0.0, 1.0) - shift) if sim.level_noise and priced else 1.0
            limit = None if sim.cap is None else sim.cap * CAP_SLACK
            total = 0.0
            minutes = []
            for costs, n_costs, spread, minute_values, n_minutes, minute_spread, fallback in sim.trials:
                if priced:
                    c = costs[0] if n_costs == 1 else costs[int(rnd() * n_costs)]
                    if spread:
                        c *= exp(sigma * gauss(0.0, 1.0) - shift)
                    c *= level
                    if limit is not None and c > limit:
                        c = limit
                    total += c
                    if keep:
                        sim.costs.append(c)
                if n_minutes:
                    m = minute_values[0] if n_minutes == 1 else minute_values[int(rnd() * n_minutes)]
                else:
                    m = fallback
                if minute_spread:
                    m *= exp(sigma * gauss(0.0, 1.0) - shift)
                minutes.append(m)
            if keep:
                sim.trial_minutes.extend(minutes)
            duration = PREFLIGHT_MINUTES + stats.lpt_makespan(minutes, run.n_concurrent)
            sim.arm_minutes.append(duration)
            durations.append(duration)
            if sim.priced:
                sim.totals.append(total)
                run_total += total
        totals.append(run_total)
        walls.append(stats.list_schedule(durations, run.parallel_arms))
    return totals, walls


def suggested_budget(p90_total: float | None) -> float | None:
    """p90 x 1.15, rounded up to the next $0.50 (at least $0.50)."""
    if p90_total is None:
        return None
    steps = math.ceil(p90_total * BUDGET_MARGIN / BUDGET_STEP - 1e-9)
    return max(BUDGET_STEP, steps * BUDGET_STEP)


def inflight_allowance(spec: Mapping[str, Any], plan: Mapping[str, Any] | None, est: Mapping[str, Any]) -> float:
    """What trials still running when the budget trips can add on top of it:
    the containers that can be busy (at most the run's trials) times the
    worst single trial, an enforced cap x 1.1 or twice an uncapped arm's p90."""
    run = spec["run"]
    trials = sum(len(spec["tasks"] if a.get("tasks") is None else a["tasks"]) for a in spec["arms"]) * spec["attempts"]
    # The planner's peak: parallel_arms beyond the number of arms never run.
    busy = min(min(int(run["parallel_arms"]), len(spec["arms"])) * int(run["n_concurrent"]), trials)
    worst = 0.0
    for arm in est.get("arms") or []:
        cap = arm.get("cap_usd")
        if cap is not None:
            worst = max(worst, float(cap) * CAP_SLACK)
        else:
            p90 = (arm.get("per_trial") or {}).get("p90")
            if p90 is not None:
                worst = max(worst, 2.0 * float(p90))
    return round(busy * worst, 6)


def _task_means(rows: Sequence[Mapping[str, Any]], family: str, provider: str, model: str, metric: str,
                o: report.Options) -> dict[tuple[str, ...], float]:
    """Mean metric per pair key (pair_key) for one family on one model."""
    values: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for r in rows:
        if r.get("family") != family or r.get("provider") != provider or r.get("model") != model:
            continue
        v = metric_value(r, metric)
        if v is not None:
            values[pair_key(r, o)].append(v)
    return {k: math.fsum(vs) / len(vs) for k, vs in sorted(values.items())}


def _spreads(rows: Sequence[Mapping[str, Any]], base_family: str, metric: str, o: report.Options
             ) -> list[tuple[str, str, str, float, int]]:
    """(provider, model, family, sd of per-task log ratios, n) for every
    family paired with the baseline on at least 3 tasks."""
    out = []
    models = sorted({(str(r.get("provider")), str(r.get("model"))) for r in rows})
    for provider, model in models:
        base = _task_means(rows, base_family, provider, model, metric, o)
        if not base:
            continue
        families = sorted({str(r.get("family")) for r in rows if r.get("provider") == provider
                           and r.get("model") == model} - {base_family})
        for family in families:
            arm = _task_means(rows, family, provider, model, metric, o)
            logs = [math.log(base[k]) - math.log(arm[k]) for k in sorted(set(base) & set(arm))]
            if len(logs) >= report.MIN_SD_PAIRS:
                out.append((provider, model, family, statistics.stdev(logs), len(logs)))
    return out


def _ci_factor(level: float, n: int, sd: float | None) -> float | None:
    if n < 2 or sd is None:
        return None
    return round(math.exp(stats.t_crit(level, n - 1) * sd / math.sqrt(n)), 4)


def _pairing(cfg: Config, spec: Any, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    level = cfg.stats.confidence
    prior = cfg.stats.prior_sd_log
    o = pair_options(cfg)
    ledger_base = baseline_family(cfg)
    arms = list(spec.arms)
    named = {a.name: a for a in arms}
    task_of: dict[tuple[str, ...], str] = {}
    for r in rows:
        task_of.setdefault(pair_key(r, o), str(r.get("task")))
    spreads: dict[tuple[str, str], list[tuple[str, str, str, float, int]]] = {}
    out = []
    for arm in arms:
        base_arm = named.get(spec.baseline_arm) if spec.baseline_arm else None
        if base_arm is not None and (base_arm.provider, base_arm.model) != (arm.provider, arm.model):
            base_arm = None
        if base_arm is None:
            base_arm = next((a for a in arms if a.family == ledger_base and a.provider == arm.provider
                             and a.model == arm.model), None)
        base_family = base_arm.family if base_arm is not None else ledger_base
        if arm is base_arm or arm.family == base_family:
            continue
        metric = "cost"
        base_vals = _task_means(rows, base_family, arm.provider, arm.model, "cost", o)
        arm_vals = _task_means(rows, arm.family, arm.provider, arm.model, "cost", o)
        if not (set(base_vals) & set(arm_vals)):
            tok_base = _task_means(rows, base_family, arm.provider, arm.model, "tokens", o)
            tok_arm = _task_means(rows, arm.family, arm.provider, arm.model, "tokens", o)
            if set(tok_base) & set(tok_arm):
                metric, base_vals, arm_vals = "tokens", tok_base, tok_arm
        # Spreads are measured against a baseline family, which can differ by arm.
        spread_key = (metric, base_family)
        if spread_key not in spreads:
            spreads[spread_key] = _spreads(rows, base_family, metric, o)
        same_model = [(sd, n) for p, m, f, sd, n in spreads[spread_key]
                      if (p, m) == (arm.provider, arm.model) and f != arm.family]
        everywhere = [(sd, n) for p, m, f, sd, n in spreads[spread_key]
                      if (p, m, f) != (arm.provider, arm.model, arm.family)]
        pooled_model, pooled_all = stats.pooled_sd(same_model), stats.pooled_sd(everywhere)
        pooled, source = ((pooled_model, "pooled_model") if pooled_model is not None
                          else (pooled_all, "pooled_all"))
        ratio = stats.paired_log_ratio(base_vals, arm_vals, pooled, prior, level=level,
                                       min_observed=report.MIN_SD_PAIRS, pooled_source=source)
        n_now = ratio.n if ratio else 0
        if ratio is not None:
            sd, sd_source = ratio.sd_log, ratio.sd_source
        elif pooled is not None:
            sd, sd_source = pooled, source
        else:
            sd, sd_source = prior, "prior"
        arm_new = set(arm.tasks)
        base_new = set(base_arm.tasks) if base_arm is not None else set()
        if o.same_run_only:
            # Earlier runs never pair with this one: only tasks it runs on both sides add pairs.
            n_after = n_now + len(arm_new & base_new)
        else:
            paired_names = {task_of[k] for k in set(base_vals) & set(arm_vals)}
            arm_after = {task_of[k] for k in arm_vals} | arm_new
            base_after = {task_of[k] for k in base_vals} | base_new
            # Only tasks this design runs can add a pair (two old versions of a task never will).
            n_after = n_now + len(((arm_after & base_after) - paired_names) & (arm_new | base_new))
        multiple = None
        if ratio is not None and ratio.n >= MEASURED_MIN_TASKS:
            multiple = {"value": round(ratio.multiple, 4),
                        "ci": [round(ratio.ci[0], 4), round(ratio.ci[1], 4)] if ratio.ci else None, "n": ratio.n}
        out.append({
            "arm": arm.name,
            "baseline_cell": {"family": base_family, "provider": arm.provider, "model": arm.model,
                              "arm": base_arm.name if base_arm is not None else None},
            "metric": metric, "pairs_now": n_now, "pairs_after": n_after, "multiple_now": multiple,
            "ci_factor_now": _ci_factor(level, n_now, sd), "ci_factor_after": _ci_factor(level, n_after, sd),
            "sd_log": None if sd is None else round(sd, 4), "sd_source": sd_source,
        })
    return out


def estimate(cfg: Config, spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
             plan: Mapping[str, Any] | None, replicates: int | None = None) -> dict[str, Any]:
    """Cost, time and pairing estimate for a NormalizedSpec. plan is the
    validation plan (per-arm kwargs give each arm's configuration and its
    enforced dollar cap); without one, kwargs are guessed from the profile,
    for allowed harnesses only. replicates is cut for large designs (see
    MAX_DRAWS); the result reports the number run.
    Raises ConfigError for a spec that is not structurally valid."""
    run_spec = spec_from_design(spec, None)
    run = run_spec.run
    meta = task_meta(cfg, run_spec.dataset)
    fingerprint = report.ledger_fingerprint(rows)  # over every row given, as ReadOnlyLedger.fingerprint()
    rows = usable(rows)
    seed = int(hashlib.sha256((run_spec.spec_hash + fingerprint).encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed)
    sigma = float(cfg.stats.prior_sd_log)
    plan_arms = _plan_arms(plan)

    sims: list[_ArmSim] = []
    meta_out: list[dict[str, Any]] = []
    for arm in run_spec.arms:
        entry = plan_arms.get(arm.name) or {}
        kwargs = _kwargs(entry)
        if kwargs is not None:
            chash: str | None = config_hash_for(arm, kwargs, run.timeout_multiplier)
        else:
            chash = guess_config_hash(cfg, arm, run)
        # guess=False: when the guess above failed, a second one would fail the same way.
        model = cell_model(cfg, rows, arm, config_hash=chash, meta=meta, guess=False)
        cap, cap_notes = enforced_cap(arm, kwargs or {})
        sims.append(_sim(arm, model, cap, run_spec.n_attempts, meta, run.timeout_multiplier))
        meta_out.append({"entry": entry, "config_hash": chash, "notes": cap_notes})

    replicates = effective_replicates(replicates or REPLICATES, sum(len(sim.trials) for sim in sims))
    totals, walls = _simulate(sims, run, replicates, rng, sigma)

    warnings: list[str] = []
    arms_out = []
    worst_case = 0.0
    uncapped = []
    for sim, extra in zip(sims, meta_out):
        arm, model = sim.arm, sim.model
        notes = list(model.notes) + list(extra["notes"])
        if model.p_infra > INFRA_NOTE_RATE and model.infra_finished:
            notes.append(f"{model.p_infra:.0%} of {model.infra_finished} earlier trials failed before using the "
                         "model; fix before spending.")
            warnings.append(f"Arm '{arm.name}': {model.p_infra:.0%} of its earlier trials were infrastructure "
                            "failures; fix before spending.")
        if not sim.priced:
            warnings.append(f"Arm '{arm.name}': no cost can be estimated; it is left out of the totals.")
        trials = len(sim.trials)
        if sim.cap is not None:
            worst_case += trials * sim.cap
        else:
            uncapped.append(arm.name)
        costs = sorted(sim.costs)
        arm_totals = sorted(sim.totals)
        trial_minutes = sorted(sim.trial_minutes)
        arm_minutes = sorted(sim.arm_minutes)
        entry = extra["entry"]
        observed = sum(1 for t in dict.fromkeys(arm.tasks) if t in model.per_task)
        arms_out.append({
            "arm": arm.name,
            "cell": {"family": arm.family, "provider": arm.provider, "model": arm.model,
                     "effort": entry.get("effort") or arm.effort, "config_hash": extra["config_hash"],
                     "options_fingerprint": entry.get("options_fingerprint")},
            "trials": trials, "basis": model.basis, "basis_detail": model.detail, "confidence": model.confidence,
            "evidence": {"trials": model.n_trials, "tasks": model.n_tasks, "design_tasks_observed": observed,
                         "runs": model.runs},
            "per_trial": {"mean": _money(math.fsum(costs) / len(costs)) if costs else None,
                          "p50": _money(_quantile(costs, 0.5)), "p90": _money(_quantile(costs, 0.9))},
            "total": {"p10": _money(_quantile(arm_totals, 0.1)), "p50": _money(_quantile(arm_totals, 0.5)),
                      "p90": _money(_quantile(arm_totals, 0.9))},
            "cap_usd": sim.cap,
            "minutes_per_trial": {"p50": _mins(_quantile(trial_minutes, 0.5)),
                                  "p90": _mins(_quantile(trial_minutes, 0.9))},
            "arm_minutes": {"p50": _mins(_quantile(arm_minutes, 0.5)), "p90": _mins(_quantile(arm_minutes, 0.9))},
            "p_infra": round(model.p_infra, 4),
            "unpriced_models": [arm.model] if model.unpriced else [],
            "notes": notes,
        })

    ordered = sorted(totals)
    walls_sorted = sorted(walls)
    budget = float(run.budget_usd)
    p90 = _quantile(ordered, 0.9)
    peak = (plan or {}).get("peak_containers") if isinstance(plan, Mapping) else None
    if not isinstance(peak, int):
        peak = min(run.parallel_arms, len(run_spec.arms)) * run.n_concurrent
    out: dict[str, Any] = {
        "spec_hash": run_spec.spec_hash, "ledger_fingerprint": fingerprint, "seed": seed, "replicates": replicates,
        "arms": arms_out,
        "run": {
            "total": {"p10": _money(_quantile(ordered, 0.1)), "p50": _money(_quantile(ordered, 0.5)),
                      "p90": _money(p90)},
            "budget_usd": budget,
            "p_within_budget": round(sum(1 for t in ordered if t <= budget) / len(ordered), 4) if ordered else None,
            "suggested_budget_usd": suggested_budget(p90),
            "worst_case_capped_usd": _money(worst_case), "uncapped_arms": uncapped,
            "inflight_allowance_usd": None,
            "wall_minutes": {"p50": _mins(_quantile(walls_sorted, 0.5)), "p90": _mins(_quantile(walls_sorted, 0.9))},
            "peak_containers": peak,
        },
        "pairing": _pairing(cfg, run_spec, rows),
        "warnings": warnings,
        "assumptions": list(ASSUMPTIONS),
    }
    out["run"]["inflight_allowance_usd"] = inflight_allowance(run_spec.spec, plan, out)
    return out


def summary(est: Mapping[str, Any] | None) -> dict[str, Any]:
    """The headline figures of an estimate, for compact results."""
    if not isinstance(est, Mapping):
        return {"p50_usd": None, "p90_usd": None, "p_within_budget": None, "suggested_budget_usd": None,
                "wall_minutes_p50": None}
    run = est.get("run") or {}
    total = run.get("total") or {}
    return {"p50_usd": total.get("p50"), "p90_usd": total.get("p90"), "p_within_budget": run.get("p_within_budget"),
            "suggested_budget_usd": run.get("suggested_budget_usd"),
            "wall_minutes_p50": (run.get("wall_minutes") or {}).get("p50")}


def queue_estimate(cfg: Config, spec: Mapping[str, Any],
                   rows: Sequence[Mapping[str, Any]] | Callable[[], Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """RunQueue.estimate_fn: {expected_usd (p50), p90_usd, inflight_allowance_usd}.
    Never raises. Without an estimate the allowance falls back to the
    enforced caps, and to None (the queue's own fallback) when an arm has none."""
    plan: Mapping[str, Any] | None = None
    try:
        plan = designs.validate(cfg, spec).get("plan")
        data = rows() if callable(rows) else rows
        est = estimate(cfg, spec, list(data), plan)
        total = est["run"]["total"]
        return {"expected_usd": total["p50"], "p90_usd": total["p90"],
                "inflight_allowance_usd": est["run"]["inflight_allowance_usd"]}
    except Exception:
        pass
    try:
        caps = []
        for entry in _plan_arms(plan).values():
            arm = next(a for a in spec_from_design(spec, None).arms if a.name == entry.get("name"))
            cap, _ = enforced_cap(arm, _kwargs(entry) or {})
            caps.append(cap)
        allowance = None
        if caps and all(c is not None for c in caps):
            fake = {"arms": [{"cap_usd": c} for c in caps]}
            allowance = inflight_allowance(spec, plan, fake)
        return {"expected_usd": None, "p90_usd": None, "inflight_allowance_usd": allowance}
    except Exception:
        return {"expected_usd": None, "p90_usd": None, "inflight_allowance_usd": None}
