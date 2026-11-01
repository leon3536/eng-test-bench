"""Where the comparisons on one model stand, and the cheapest runs that move
them forward.

The person's question after "run run run run" is "where are we?", and the
assistant's is "what should run next?". Both are answered from the pooled
report (report.build_report over every stored run), so a multiple, interval
or verdict quoted by the assistant is the one the REPORT tab shows. This
module adds what a planner needs on top: the gaps (pairs half done, families
that fail before using the model, task versions that cannot pair, pooled
configurations that drift) ranked by pairs gained per expected dollar, and
candidate designs that close them in seeded-sample order.

Suggestions never go beyond the seeded task sample (config/tasks.txt) and
never take a task out of order because it looks cheap: picking cheap tasks
would bias every multiple. Every candidate is a design input checked with
designs.validate and priced with estimates.estimate, exactly as a person's
design is, and nothing here saves or starts anything.

ReadOnlyLedger is the assistant's only view of the ledger: reads, cached by
the ledger's version, and no method that writes.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import random
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import checks, designs, estimates, harnesses, pricing, report, stats
from .ledger import Filters
from .settings import Config
from .spec import ArmSpec

GOALS = ("reach_target_pairs", "complete_half_pairs", "pilot_new_arms")
GAP_KINDS = ("no_baseline", "half_pair", "below_target", "infra_failing", "stale_task_version", "option_drift")
INFRA_FAILING_RATE = 0.5
INFRA_FAILING_MIN_TRIALS = 2
GATE_REPLICATES = 400
MAX_TRIMS = 25
MEMORY_SHARE = 0.8          # of Docker's memory that trial containers may plan to use
_LABEL_ORDER = {"full": 0, "staged": 1, "half_pairs": 2, "pilot": 3}
SCOPE_LIMIT = 500           # tasks the REPORT tab sends as its scope (report.js)


# The assistant's view of the ledger -------------------------------------------------------------

class ReadOnlyLedger:
    """LedgerReader over a Ledger: analysis rows, the report's coverage and a
    summary. Rows are cached until the ledger, the prices or the config
    change. There is deliberately no method that writes."""

    __slots__ = ("_source", "_cfg_getter", "_lock", "_cache")

    def __init__(self, ledger: Any, cfg_getter: Callable[[], Config]) -> None:
        self._source = ledger
        self._cfg_getter = cfg_getter
        self._lock = threading.Lock()
        self._cache: tuple[tuple, list[dict[str, Any]], str] | None = None

    def _rows(self) -> tuple[list[dict[str, Any]], str]:
        cfg = self._cfg_getter()
        key = (tuple(self._source.version()), pricing.table_sha(), designs.config_fingerprint(cfg))
        with self._lock:
            if self._cache is not None and self._cache[0] == key:
                return self._cache[1], self._cache[2]
        rows = report.analysis_rows(self._source, cfg)
        fingerprint = report.ledger_fingerprint(rows)
        with self._lock:
            self._cache = (key, rows, fingerprint)
        return rows, fingerprint

    def rows(self) -> list[dict[str, Any]]:
        """Every trial in report.analysis_rows form. Treat as read-only."""
        return list(self._rows()[0])

    def fingerprint(self) -> str:
        return self._rows()[1]

    def report_for(self, provider: str, model: str, baseline_family: str | None = None,
                   group_by: str = "family") -> dict[str, Any]:
        """The pooled report restricted to one model, as the REPORT tab
        computes it with these grouping and baseline options, on its task
        scope (the profile's task file)."""
        cfg = self._cfg_getter()
        return report.build_report(self._source, cfg, _model_filter(cfg, provider, model),
                                   coverage_options(cfg, baseline_family, group_by))

    def coverage(self, provider: str, model: str, baseline_family: str | None = None,
                 group_by: str = "family") -> dict[str, Any]:
        cfg = self._cfg_getter()
        return report.build_coverage(self._source, cfg, _model_filter(cfg, provider, model),
                                     coverage_options(cfg, baseline_family, group_by))

    def summary(self) -> dict[str, Any]:
        rows, fingerprint = self._rows()
        started = sorted(str(r["started_at"]) for r in rows if r.get("started_at"))
        return {
            "runs": len({r.get("run_key") for r in rows}), "trials": len(rows),
            "cost_valid": sum(1 for r in rows if r.get("valid_cost")),
            "infra": sum(1 for r in rows if estimates.is_infra(r)),
            "excluded_by_hand": sum(1 for r in rows if estimates.is_manual(r)),
            "misconfigured_cap": sum(1 for r in rows if r.get("outcome") == "misconfigured_cap"),
            "first": started[0] if started else None, "last": started[-1] if started else None,
            "fingerprint": fingerprint,
        }


def scope_tasks(cfg: Config) -> tuple[str, ...]:
    """The REPORT tab's default task scope: the profile's task file (cfg.tasks),
    as many as it sends; empty means every task."""
    return tuple(dict.fromkeys(cfg.tasks or []))[:SCOPE_LIMIT]


def _model_filter(cfg: Config, provider: str, model: str) -> Filters:
    return Filters(models=(f"{provider}:{model}",), tasks=scope_tasks(cfg))


def coverage_options(cfg: Config, baseline_family: str | None = None, group_by: str = "family") -> report.Options:
    """The profile's report options with this grouping and baseline. Harness
    versions are never split here, so groups line up with analysis rows, and
    harnesses are compared on one model whatever report.compare says."""
    if group_by not in report.GROUP_BY:
        raise ValueError(f"group_by must be one of {', '.join(report.GROUP_BY)}")
    o = report.parse_options(None, cfg)
    return dataclasses.replace(o, group_by=group_by, split_harness_versions=False, baseline_config=None,
                               include_pairs=False, baseline_family=baseline_family or o.baseline_family,
                               compare="harnesses")


# Coverage ----------------------------------------------------------------------------------------

def _valid(row: Mapping[str, Any], metric: str) -> bool:
    if metric == "pass":
        return bool(row.get("valid_reward"))
    return estimates.metric_value(row, "cost" if metric == "cost" else "tokens") is not None


def _row_metric(rows: Sequence[Mapping[str, Any]]) -> str:
    if any(_valid(r, "cost") for r in rows):
        return "cost"
    if any(_valid(r, "tokens_total") for r in rows):
        return "tokens_total"
    return "pass"


def _names(rows: Sequence[Mapping[str, Any]], metric: str) -> set[str]:
    return {str(r.get("task")) for r in rows if _valid(r, metric)}


def _task_keys(rows: Sequence[Mapping[str, Any]], metric: str, o: report.Options) -> dict[str, set[tuple[str, ...]]]:
    """Pair keys (the report's rule, estimates.pair_key) of the valid trials, by task name."""
    out: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for r in rows:
        if _valid(r, metric):
            out[str(r.get("task"))].add(estimates.pair_key(r, o))
    return out


def _pair_view(arm: Mapping[str, set[tuple[str, ...]]], base: Mapping[str, set[tuple[str, ...]]],
               o: report.Options) -> tuple[set[str], set[str]]:
    """The task names each side counts as having when planning against the
    other. With same_run_only nothing earlier pairs with a new run, so
    neither side has any. Otherwise a task both sides ran in versions that do
    not pair counts for neither, so it is run again on both, as the report's
    coverage lists it."""
    if o.same_run_only:
        return set(), set()
    unpaired = {t for t in set(arm) & set(base) if not (arm[t] & base[t])}
    return set(arm) - unpaired, set(base) - unpaired


def _already_paired(arm: Mapping[str, set[tuple[str, ...]]],
                    base: Mapping[str, set[tuple[str, ...]]]) -> set[str]:
    """Task names that already make a pair, whatever same_run_only says:
    planning re-runs them only when the sample has no other task left."""
    return {t for t in set(arm) & set(base) if arm[t] & base[t]}


def _multiple(metric: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not metric or not metric.get("geomean"):
        return None
    gm, rom = metric["geomean"], metric.get("ratio_of_means") or {}
    return {"value": gm.get("estimate"), "ci": [gm.get("lo"), gm.get("hi")], "sd_log": gm.get("sd_log"),
            "ratio_of_means": {"value": rom.get("estimate"), "ci": [rom.get("lo"), rom.get("hi")]},
            "n": metric.get("n"), "status": metric.get("status"), "sd_source": "observed"}


def _sorted_tasks(names: Any, order: Mapping[str, int]) -> list[str]:
    return sorted(names, key=lambda t: (order.get(t, len(order)), t))


def _halfwidth_at(sd: float, n: int, level: float) -> float | None:
    if n < 2:
        return None
    return math.exp(stats.t_crit(level, n - 1) * sd / math.sqrt(n))


def _fallback_groups(by_group: Mapping[str, list[Mapping[str, Any]]], comparisons: Mapping[str, Mapping[str, Any]],
                     baseline: str) -> list[dict[str, Any]]:
    """Group shells for a reader that serves only the coverage contract."""
    bases = {c.get("baseline_group") for c in comparisons.values()}
    out = []
    for key in sorted(by_group):
        rows = by_group[key]
        hashes = {r.get("config_hash") for r in rows}
        outcomes = Counter(str(r.get("outcome")) for r in rows)
        out.append({"key": key, "family": rows[0].get("family"), "label": key,
                    "is_baseline": key in bases or (not bases and rows[0].get("family") == baseline),
                    "config_hash": next(iter(hashes)) if len(hashes) == 1 else None, "configs": [],
                    "config_diff": {}, "tasks": len({r.get("task") for r in rows}),
                    "trials": {"total": len(rows), "by_outcome": dict(sorted(outcomes.items())),
                               "valid_cost": sum(1 for r in rows if r.get("valid_cost"))}})
    return out


def coverage(cfg: Config, reader: Any, provider: str, model: str, baseline_family: str | None = None,
             target_pairs: int | None = None, group_by: str = "family") -> dict[str, Any]:
    """Every family (or configuration, with group_by config) on one model:
    trials, pairs with the baseline, the report's multiples and verdicts,
    pairs still needed for the precision target, and ranked gaps. Only the
    tasks of the REPORT tab's default scope (the profile's task file) count,
    so the assistant quotes the figures that tab shows."""
    if target_pairs is not None and (isinstance(target_pairs, bool) or not isinstance(target_pairs, int)
                                     or target_pairs < 2):
        raise ValueError("target_pairs must be a whole number of at least 2")
    o = coverage_options(cfg, baseline_family, group_by)
    base_family = o.baseline_family
    level = float(cfg.stats.confidence)
    if hasattr(reader, "report_for"):
        rep = reader.report_for(provider, model, base_family, group_by)
    else:
        rep = None
    contract = rep if rep is not None else reader.coverage(provider, model, base_family, group_by)
    # The REPORT tab's task scope, which the report above was computed on.
    scope = set(scope_tasks(cfg))
    rows_all = [r for r in reader.rows() if not scope or r.get("task") in scope]
    fingerprint = reader.fingerprint()
    on_model = [r for r in rows_all if r.get("provider") == provider and r.get("model") == model]
    rows = estimates.usable(on_model)
    sample = list(cfg.tasks or [])
    order = {t: i for i, t in reversed(list(enumerate(sample)))}
    meta = estimates.task_meta(cfg)

    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_group[report.group_key(r, o)].append(r)
    comparisons = {c["group"]: c for c in contract.get("comparisons") or []}
    groups = rep["groups"] if rep is not None else _fallback_groups(by_group, comparisons, base_family)
    base_group = next((g for g in groups if g.get("is_baseline")), None)
    base_key = base_group["key"] if base_group else None
    base_rows = by_group.get(base_key, []) if base_key else []

    # The precision target: pairs for +/-target_halfwidth at stats.confidence,
    # sized by the spread observed on this model, else the prior.
    observed = [(c["coverage"].get("sd_log_used"), c["coverage"].get("pairs") or 0) for c in comparisons.values()
                if (c.get("coverage") or {}).get("sd_source") == "observed"]
    pooled = stats.pooled_sd(observed)
    sd = pooled if pooled is not None else float(cfg.stats.prior_sd_log)
    wanted = target_pairs if target_pairs is not None else stats.required_pairs(sd, cfg.stats.target_halfwidth, level)
    capped = bool(sample) and wanted > len(sample)
    target = len(sample) if capped else wanted
    notes: list[str] = []
    if capped:
        at = _halfwidth_at(sd, len(sample), level)
        notes.append(f"The precision target needs about {wanted} paired tasks, more than the {len(sample)} in the "
                     f"seeded sample; at {len(sample)} pairs the remaining uncertainty is about "
                     + (f"x/÷{at:.2f} at {level:.0%}." if at else "unknown."))
    if not rows:
        notes.append(f"No trials on {provider}:{model} yet. Start with suggest_runs goal pilot_new_arms.")
    if o.same_run_only:
        notes.append("report.same_run_only is on: trials pair only within one run, so a new run adds pairs only "
                     "on tasks it runs on both sides, and no earlier half-pair can be completed.")

    costs: dict[str, tuple[float | None, str]] = {}

    def cost(family: str) -> float | None:
        if family not in costs:
            costs[family] = estimates.family_trial_cost(cfg, rows_all, family, provider, model, meta)
        return costs[family][0]

    cells = []
    pair_state: dict[str, dict[str, Any]] = {}
    base_keys: dict[str, dict[str, set[tuple[str, ...]]]] = {}
    for g in groups:
        key = g["key"]
        rs = by_group.get(key, [])
        comp = comparisons.get(key)
        is_base = key == base_key
        metric = (comp.get("coverage") or {}).get("metric") if comp else None
        metric = metric or _row_metric(rs)
        valid_names = _names(rs, metric)
        trials = g.get("trials") or {}
        outcomes = trials.get("by_outcome") or {}
        finished = [r for r in rs if r.get("outcome") != "incomplete"]
        cell: dict[str, Any] = {
            "key": key, "family": g.get("family"), "label": g.get("label"), "is_baseline": is_base,
            "config_hash": g.get("config_hash"),
            "configs": [{"config_hash": c.get("config_hash"), "label": c.get("label"), "trials": c.get("trials")}
                        for c in g.get("configs") or []],
            "effort": sorted({str(r.get("effort") or "default") for r in rs}),
            "trials": trials.get("total", len(rs)), "cost_valid": trials.get("valid_cost"),
            "infra": sum(outcomes.get(k, 0) for k in report.RELIABILITY_OUTCOMES),
            "finished": len(finished), "tasks": g.get("tasks"),
            "passes": sum(1 for r in rs if r.get("valid_reward") and r.get("passed")),
            "pass_rate": None, "pass_rate_ci90": None,
            "outcomes": dict(outcomes),
            "failure_causes": dict(sorted(Counter(str(r["harbor_cause"]) for r in rs if r.get("harbor_cause"))
                                          .items())),
            "metric": metric, "paired_with_baseline": None, "half_pair_tasks": [],
            "missing_sample_positions": [i + 1 for i, t in enumerate(sample) if t not in valid_names],
            "cost_multiple": None, "tokens_multiple": None, "published": {}, "verdict": {},
            "pairs_needed": None, "needed_more": None, "headline": comp.get("headline") if comp else None,
        }
        rate = estimates.pass_rate(rs, level)
        if rate is not None:
            cell["pass_rate"], cell["pass_rate_ci90"] = rate["estimate"], rate["ci"]
        if comp is not None and not is_base:
            cov = comp.get("coverage") or {}
            paired = int(cov.get("pairs") or 0)
            metrics = comp.get("metrics") or {}
            cell["cost_multiple"] = _multiple(metrics.get("cost"))
            cell["tokens_multiple"] = _multiple(metrics.get("tokens_total"))
            for name, short in (("tokens_total", "tokens"), ("cost", "cost")):
                ref = (metrics.get(name) or {}).get("reference")
                if ref:
                    cell["published"][short] = ref.get("published")
                    cell["verdict"][short] = ref.get("verdict")
            if target_pairs is None and cov.get("sd_source") == "observed" and cov.get("est_pairs_needed"):
                needed = int(cov["est_pairs_needed"])
            else:
                needed = target
            cell["paired_with_baseline"] = paired
            cell["pairs_needed"] = needed
            cell["needed_more"] = max(0, needed - paired)
            if metric not in base_keys:
                base_keys[metric] = _task_keys(base_rows, metric, o)
            arm_keys = _task_keys(rs, metric, o)
            mine, theirs = _pair_view(arm_keys, base_keys[metric], o)
            already = _already_paired(arm_keys, base_keys[metric])
            missing_base = _sorted_tasks((mine - theirs) & set(sample), order)
            missing_arm = _sorted_tasks((theirs - mine) & set(sample), order)
            cell["half_pair_tasks"] = ([{"task": t, "missing": "baseline"} for t in missing_base]
                                       + [{"task": t, "missing": "arm"} for t in missing_arm])
            cell["half_pair_tasks"].sort(key=lambda h: (order.get(h["task"], len(order)), h["missing"]))
            pair_state[key] = {"valid": mine, "base": theirs, "already": already, "measured": bool(valid_names),
                               "missing_base": missing_base, "missing_arm": missing_arm, "paired": paired}
        cells.append(cell)

    gaps = _gaps(cells, by_group, rows, base_family, base_key, base_rows, pair_state, sample, order, target, cost,
                 o.same_run_only, o.match_on)
    excluded = sum(1 for r in on_model if estimates.is_manual(r))
    return {
        "provider": provider, "model": model, "group_by": group_by,
        # The tasks counted, as the REPORT tab's scope block says it (report.task_scope).
        "scope": report.task_scope(cfg, Filters(tasks=scope_tasks(cfg))),
        "baseline": {"family": base_family, "key": base_key,
                     "effort": sorted({str(r.get("effort") or "default") for r in base_rows})},
        "ledger_fingerprint": fingerprint,
        "report_fingerprint": ((rep or contract).get("provenance") or {}).get("fingerprint"),
        "target_pairs": target, "target_source": "argument" if target_pairs is not None else "config",
        "target_sd_log": round(sd, 4), "target_sd_source": "observed" if pooled is not None else "prior",
        "target_capped": capped, "sample_size": len(sample),
        "thresholds": {"estimate": o.min_pairs_estimate, "verdict": o.min_pairs_verdict, "level": level},
        "excluded": {"by_hand": excluded,
                     "misconfigured_cap": sum(1 for r in rows if r.get("outcome") == "misconfigured_cap")},
        "cells": cells, "gaps": gaps, "notes": notes,
        "warnings": list((rep or {}).get("warnings") or []),
    }


def _gap(kind: str, family: str | None, cell: str | None, detail: str, tasks: list[str], est_cost: float | None,
         pairs_gained: int) -> dict[str, Any]:
    per_usd = (pairs_gained / est_cost) if est_cost and est_cost > 0 and pairs_gained else None
    return {"kind": kind, "family": family, "cell": cell, "detail": detail, "tasks": tasks,
            "est_cost_usd": None if est_cost is None else round(est_cost, 4), "pairs_gained": pairs_gained,
            "pairs_per_usd": None if per_usd is None else round(per_usd, 2)}


def _times(n: int, each: float | None) -> float | None:
    return None if each is None else n * each


def _gaps(cells: list[dict[str, Any]], by_group: Mapping[str, list[Mapping[str, Any]]],
          rows: Sequence[Mapping[str, Any]], base_family: str, base_key: str | None,
          base_rows: Sequence[Mapping[str, Any]], pair_state: Mapping[str, Mapping[str, Any]], sample: list[str],
          order: Mapping[str, int], target: int, cost: Callable[[str], float | None],
          same_run_only: bool = False, match_on: str = "digest") -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    others = [c for c in cells if not c["is_baseline"]]
    base_has_data = bool(base_key) and any(_valid(r, m) for r in base_rows for m in ("cost", "tokens_total", "pass"))

    if not base_has_data:
        tasks: set[str] = set()
        gained = 0
        helped = []
        rerun: float | None = 0.0
        for c in others:
            names = _names(by_group.get(c["key"], []), c["metric"]) & set(sample)
            if names:
                tasks |= names
                gained += len(names)
                helped.append(c["label"] or c["family"])
                each = cost(c["family"])
                rerun = None if rerun is None or each is None else rerun + len(names) * each
        if tasks:
            ordered = _sorted_tasks(tasks, order)
            est = _times(len(ordered), cost(base_family))
            detail = (f"No usable {base_family} trials on this model, so nothing pairs. {base_family} on "
                      f"{len(ordered)} task(s) would pair {gained} trial set(s) of {', '.join(helped)}.")
            if same_run_only:
                # Earlier trials never pair with a new run: the families run those tasks again beside it.
                detail += " With report.same_run_only on, they must run those tasks again in the same run."
                est = None if est is None or rerun is None else est + rerun
            gaps.append(_gap("no_baseline", base_family, None, detail, ordered, est, gained))
    else:
        missing_base: dict[str, int] = defaultdict(int)
        helped = []
        for c in others:
            state = pair_state.get(c["key"])
            if not state:
                continue
            for t in state["missing_base"]:
                missing_base[t] += 1
            if state["missing_base"]:
                helped.append(c["label"] or c["family"])
        if missing_base:
            ordered = _sorted_tasks(missing_base, order)
            gained = sum(missing_base.values())
            gaps.append(_gap("half_pair", base_family, None,
                             f"{base_family} is missing {len(ordered)} task(s) that {', '.join(helped)} already "
                             f"ran; running them completes {gained} pair(s).",
                             ordered, _times(len(ordered), cost(base_family)), gained))
        for c in others:
            state = pair_state.get(c["key"])
            if not state or not state["measured"] or not state["missing_arm"]:
                continue
            tasks_ = state["missing_arm"]
            gaps.append(_gap("half_pair", c["family"], c["label"],
                             f"{c['label']} is missing {len(tasks_)} task(s) the baseline already ran.",
                             tasks_, _times(len(tasks_), cost(c["family"])), len(tasks_)))
        for c in others:
            state = pair_state.get(c["key"])
            # A family with nothing measured here needs fixing (infra_failing) before more pairs.
            if not state or not c["needed_more"] or not state["measured"]:
                continue
            half = len(state["missing_base"]) + len(state["missing_arm"])
            new_needed = max(0, c["needed_more"] - half)
            # Tasks already paired (possible under same_run_only) go last: a
            # repeat pair adds less than a task the comparison has not seen.
            already = state.get("already") or set()
            unseen = [t for t in sample if t not in state["valid"] and t not in state["base"]]
            fresh = [t for t in unseen if t not in already] + [t for t in unseen if t in already]
            tasks_ = fresh[:new_needed]
            per_pair = None
            arm_cost, base_cost = cost(c["family"]), cost(base_family)
            if arm_cost is not None and base_cost is not None:
                per_pair = arm_cost + base_cost
            short = (f" The sample has only {len(fresh)} task(s) neither side ran."
                     if new_needed > len(fresh) else "")
            gaps.append(_gap("below_target", c["family"], c["label"],
                             f"{c['label']} has {state['paired']} of the {c['pairs_needed']} paired tasks the "
                             f"precision target needs: {c['needed_more']} more, {min(half, c['needed_more'])} from "
                             f"half-pairs and {new_needed} from new tasks run on both sides.{short}",
                             tasks_, _times(len(tasks_), per_pair), len(tasks_)))

    for c in cells:
        rs = [r for r in by_group.get(c["key"], []) if r.get("outcome") != "incomplete"]
        failed = [r for r in rs if estimates.is_infra(r)]
        if len(rs) >= INFRA_FAILING_MIN_TRIALS and len(failed) / len(rs) >= INFRA_FAILING_RATE:
            types = Counter(str(r["exception_type"]) for r in failed if r.get("exception_type"))
            top = sorted(types.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if types else None
            gaps.append(_gap("infra_failing", c["family"], c["label"],
                             f"{len(failed)} of {len(rs)} finished trials of {c['label']} failed before the harness "
                             "used the model (setup, provider or sandbox)"
                             + (f"; most common exception type: {top}." if top else "."),
                             _sorted_tasks({str(r.get("task")) for r in failed}, order), None, 0))

    digests: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("task_digest"):
            digests[str(r.get("task"))].add(str(r["task_digest"]))
    stale = _sorted_tasks([t for t, ds in digests.items() if len(ds) > 1], order)
    if stale:
        how = ("with report.match_on name they still pair, so the multiple mixes versions"
               if match_on == "name" else "trials pair only within one version")
        gaps.append(_gap("stale_task_version", None, None,
                         f"{len(stale)} task(s) ran in more than one version (different content digests); {how}: "
                         + ", ".join(f"{t} ({len(digests[t])} versions)" for t in stale[:8]) + ".",
                         stale, None, 0))

    configs: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for r in rows:
        configs[str(r.get("family"))].setdefault(str(r.get("config_hash") or ""), r)
    for family in sorted(configs):
        if len(configs[family]) < 2:
            continue
        differ = _differences(list(configs[family].values()))
        gaps.append(_gap("option_drift", family, None,
                         f"{family} has {len(configs[family])} configurations on this model"
                         + (f" that differ in {', '.join(differ)}" if differ else "")
                         + "; pooled by family they are averaged together. Group by configuration to see each.",
                         [], None, 0))

    kind_order = {k: i for i, k in enumerate(GAP_KINDS)}
    gaps.sort(key=lambda g: (g["pairs_gained"] <= 0, g["pairs_per_usd"] is None, -(g["pairs_per_usd"] or 0.0),
                             -g["pairs_gained"], kind_order[g["kind"]], str(g["family"] or ""), str(g["cell"] or "")))
    return gaps


def _differences(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    settings_: list[dict[str, Any]] = []
    for r in rows:
        values = dict(r.get("options") or {})
        values["effort"] = r.get("effort")
        values["implementation"] = r.get("implementation")
        settings_.append(values)
    keys = sorted({k for s in settings_ for k in s})
    return [k for k in keys if len({repr(s.get(k)) for s in settings_}) > 1]


# Suggestions -----------------------------------------------------------------------------------------

@dataclass
class _Template:
    name: str
    agent: str
    family: str
    effort: str
    options: dict[str, Any]
    source: str             # "config arm <name>" | "default options"


def _arm_name(family: str, model: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", f"{family}-{model.split('/')[-1]}".lower()).strip("-") or "arm"
    if not base[0].isalpha():
        base = "a-" + base
    base = base[:32].strip("-")
    name, n = base, 2
    while name in taken:
        suffix = f"-{n}"
        name = base[: 32 - len(suffix)].strip("-") + suffix
        n += 1
    return name


def _template(cfg: Config, family: str, provider: str, model: str, taken: set[str]) -> _Template | None:
    """The config arm for this family on the model (enabled first), else an
    allowed harness of the family with default options and inherited effort."""
    arms = [a for a in cfg.arms if (a.family or a.agent) == family and a.provider == provider and a.model == model]
    arms.sort(key=lambda a: not a.enabled)
    if arms:
        a = arms[0]
        return _Template(a.name, a.agent, family, a.effort, copy.deepcopy(dict(a.options)), f"config arm {a.name}")
    allowed = [h for h in designs.allowed_harnesses(cfg) if h["family"] == family]
    if not allowed:
        return None
    return _Template(_arm_name(family, model, taken), allowed[0]["agent"], family, "inherit", {}, "default options")


def _arm_spec(t: _Template, provider: str, model: str) -> ArmSpec:
    kind = "import_path" if harnesses.is_import_path(t.agent) else "name"
    return ArmSpec(name=t.name, agent=t.agent, family=t.family, kind=kind, provider=provider, model=model,
                   effort=t.effort, options=copy.deepcopy(t.options), tasks=())


def _planning_keys(rows: Sequence[Mapping[str, Any]], family: str, o: report.Options
                   ) -> dict[str, set[tuple[str, ...]]]:
    """Pair keys, by task, of a family's usable measurements: cost-valid, or
    reward-valid when the harness reports no tokens at all."""
    mine = [r for r in rows if r.get("family") == family]
    keys = _task_keys(mine, "cost", o)
    if keys:
        return keys
    if not any(r.get("valid_tokens") for r in mine):
        return _task_keys(mine, "pass", o)
    return {}


class _Gate:
    """A quick Monte Carlo of the total cost while a design is built, to stop
    adding tasks before the p90 passes the ceiling. Draws for each (family,
    task) come from their own seeded stream, so the order of questions does
    not change the answers."""

    def __init__(self, models: Mapping[str, estimates.CellModel], caps: Mapping[str, float | None],
                 meta: Mapping[str, Mapping[str, Any]], sigma: float, attempts: int, seed: int,
                 replicates: int = GATE_REPLICATES) -> None:
        self.models, self.caps, self.meta = models, caps, meta
        self.sigma, self.attempts, self.seed, self.replicates = sigma, attempts, seed, replicates
        self.totals = [0.0] * replicates
        rng = random.Random(stats.stable_seed(seed, "levels"))
        self.levels = {}
        for family in sorted(models):
            m = models[family]
            noisy = m.basis in ("borrowed", "prior") or len(m.pool) < estimates.MIN_POOL
            self.levels[family] = [math.exp(sigma * rng.gauss(0.0, 1.0) - sigma * sigma / 2.0) if noisy else 1.0
                                   for _ in range(replicates)]
        self._cache: dict[tuple[str, str], list[float]] = {}

    def _draws(self, family: str, task: str) -> list[float]:
        key = (family, task)
        if key not in self._cache:
            m = self.models[family]
            values = estimates._choices(task, m.per_task, m.by_difficulty, m.pool, self.meta)
            out = [0.0] * self.replicates
            if values:
                rng = random.Random(stats.stable_seed(self.seed, family, task))
                small = len(values) < estimates.MIN_POOL
                cap = self.caps.get(family)
                limit = None if cap is None else cap * estimates.CAP_SLACK
                levels = self.levels[family]
                for b in range(self.replicates):
                    total = 0.0
                    for _ in range(self.attempts):
                        c = rng.choice(values)
                        if small:
                            c *= math.exp(self.sigma * rng.gauss(0.0, 1.0) - self.sigma * self.sigma / 2.0)
                        c *= levels[b]
                        total += min(c, limit) if limit is not None else c
                    out[b] = total
            self._cache[key] = out
        return self._cache[key]

    def try_add(self, task: str, families: Sequence[str], ceiling: float) -> bool:
        new = list(self.totals)
        for family in families:
            for b, v in enumerate(self._draws(family, task)):
                new[b] += v
        p90 = estimates._quantile(sorted(new), 0.9) or 0.0
        if p90 > ceiling:
            return False
        self.totals = new
        return True


@dataclass
class _Build:
    sched: dict[str, list[str]]
    commits: list[tuple[str, list[str]]]
    stopped_by_budget: bool


def _greedy(target: int, steps: Sequence[int], base: str, others: list[str], valid: Mapping[str, set[str]],
            base_for: Mapping[str, set[str]], before: Mapping[str, int], sample: list[str], gate: _Gate,
            ceiling: float, can_add_base: bool, already: Mapping[str, set[str]] | None = None) -> _Build:
    """Complete half-pairs in seeded order (step 1), then extend through the
    sample in order, sharing each baseline trial across every family still
    short of the target (step 2). Never skips ahead to a cheaper task.

    valid[h] and base_for[h] are the tasks family h and the baseline count
    as having against each other (_pair_view); before[h] is h's pair count
    now, as the report counts it. already[h] holds tasks h has paired that
    _pair_view does not count (same_run_only): step 2 reaches them only
    after every other sample task."""
    already = already or {}
    sched: dict[str, list[str]] = {f: [] for f in [base] + others}

    def has(family: str, task: str) -> bool:
        return task in valid[family] or task in sched[family]

    def base_has(family: str, task: str) -> bool:
        return task in base_for[family] or task in sched[base]

    paired = {h: int(before[h]) for h in others}
    commits: list[tuple[str, list[str]]] = []

    def commit(task: str, families: list[str], gainers: list[str]) -> bool:
        if not gate.try_add(task, families, ceiling):
            return False
        for f in families:
            sched[f].append(task)
        for h in gainers:
            paired[h] += 1
        commits.append((task, families))
        return True

    if 1 in steps:
        for task in sample:
            if all(paired[h] >= target for h in others):
                break
            families: list[str] = []
            gainers: list[str] = []
            waiting = [h for h in others if paired[h] < target and task in valid[h] and not base_has(h, task)]
            if waiting and can_add_base:
                families.append(base)
                gainers += waiting
            for h in others:
                if paired[h] < target and task in base_for[h] and not has(h, task):
                    families.append(h)
                    gainers.append(h)
            if families and not commit(task, families, gainers):
                return _Build(sched, commits, True)
    if 2 in steps:
        passes = [(task, False) for task in sample] + [(task, True) for task in sample]
        for task, repeat in passes:
            short = [h for h in others if paired[h] < target]
            if not short:
                break
            gainers = [h for h in short if not (has(h, task) and base_has(h, task))
                       and (repeat or task not in already.get(h, ()))]
            needs_base = [h for h in gainers if not base_has(h, task)]
            if needs_base and not can_add_base:
                gainers = [h for h in gainers if h not in needs_base]
                needs_base = []
            if not gainers:
                continue
            families = [base] if needs_base else []
            families += [h for h in gainers if not has(h, task)]
            if not commit(task, families, gainers):
                return _Build(sched, commits, True)
    return _Build(sched, commits, False)


def _trim(build: _Build) -> bool:
    """Drop the last task added; False when nothing is left to drop."""
    if not build.commits:
        return False
    task, families = build.commits.pop()
    for f in families:
        build.sched[f].remove(task)
    return True


def docker_capacity() -> dict[str, Any] | None:
    """checks.docker_capacity (a read-only `docker info`, cached), or None
    when Docker cannot answer: sizing advice never fails a tool."""
    try:
        capacity = checks.docker_capacity()
    except Exception:
        return None
    return dict(capacity) if isinstance(capacity, Mapping) else None


def _parallel_arms(cfg: Config, n_arms: int, n_concurrent: int, tasks: Sequence[str],
                   meta: Mapping[str, Mapping[str, Any]]) -> int:
    limit = min(n_arms, cfg.run.parallel_arms, designs.MAX_PARALLEL_ARMS,
                max(1, cfg.designs.max_containers // max(1, n_concurrent)))
    capacity = docker_capacity()
    memory = [float(m) for m in ((meta.get(t) or {}).get("memory_gb") for t in tasks)
              if isinstance(m, (int, float)) and not isinstance(m, bool) and m > 0]
    if capacity and capacity.get("mem_gb") and memory:
        limit = min(limit, math.floor(MEMORY_SHARE * float(capacity["mem_gb"]) / (max(memory) * n_concurrent)))
    return max(1, limit)


def plan_suggestions(cfg: Config, reader: Any, goal: str, provider: str, model: str,
                     families: Sequence[str] | None = None, baseline_family: str | None = None,
                     target_pairs: int | None = None, max_budget_usd: float | None = None,
                     attempts: int = 1) -> dict[str, Any]:
    """suggest_runs with the reasons behind it: {candidates, notes,
    target_pairs, baseline_family, families}."""
    if goal not in GOALS:
        raise ValueError(f"goal must be one of {', '.join(GOALS)}")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= cfg.designs.max_attempts:
        raise ValueError(f"attempts must be from 1 to {cfg.designs.max_attempts} (designs.max_attempts)")
    if max_budget_usd is not None and (isinstance(max_budget_usd, bool) or not isinstance(max_budget_usd, (int, float))
                                       or not math.isfinite(max_budget_usd) or max_budget_usd <= 0):
        raise ValueError("max_budget_usd must be a positive number")
    cov = coverage(cfg, reader, provider, model, baseline_family, target_pairs, "family")
    notes = list(cov["notes"])
    base = cov["baseline"]["family"]
    target = int(cov["target_pairs"])
    out = {"candidates": [], "notes": notes, "target_pairs": target, "baseline_family": base, "families": []}
    sample = list(cfg.tasks or [])
    if not sample:
        notes.append("There is no seeded task sample (config/tasks.txt); suggestions only use it.")
        return out
    order = {t: i for i, t in reversed(list(enumerate(sample)))}
    rows_all = reader.rows()
    rows = [r for r in estimates.usable(rows_all) if r.get("provider") == provider and r.get("model") == model]
    meta = estimates.task_meta(cfg)

    if families:
        wanted = list(dict.fromkeys(f for f in families if f != base))
    else:
        wanted = list(dict.fromkeys((a.family or a.agent) for a in cfg.arms
                                    if a.provider == provider and a.model == model and (a.family or a.agent) != base))
        if not wanted:
            wanted = sorted({str(r.get("family")) for r in rows} - {base})
    taken = {a.name for a in cfg.arms}
    templates: dict[str, _Template] = {}
    for family in [base] + wanted:
        t = _template(cfg, family, provider, model, taken)
        if t is None:
            notes.append(f"No harness of family '{family}' is allowed in designs, so it is left out. A person can "
                         "allow one under designs.extra_harnesses or designs.import_paths.")
            continue
        taken.add(t.name)
        templates[family] = t
    others = [f for f in wanted if f in templates]
    out["families"] = others
    can_add_base = base in templates
    if not can_add_base:
        notes.append(f"The baseline family '{base}' has no allowed harness, so suggestions only complete pairs on "
                     "tasks it already ran.")
    # Pairs are counted by the report's rule (match_on, same_run_only): what
    # each family and the baseline have against each other, and the pair
    # counts now as the report gives them.
    o = coverage_options(cfg, base, "family")
    keys = {f: _planning_keys(rows, f, o) for f in [base] + others}
    measured = {f: set(keys[f]) for f in keys}
    valid: dict[str, set[str]] = {base: measured[base]}
    base_for: dict[str, set[str]] = {}
    for h in others:
        valid[h], base_for[h] = _pair_view(keys[h], keys[base], o)
    already = {h: _already_paired(keys[h], keys[base]) for h in others}
    reported = {c["family"]: c["paired_with_baseline"] for c in cov["cells"]
                if not c["is_baseline"] and c.get("paired_with_baseline") is not None}
    before = {h: int(reported.get(h, len(valid[h] & base_for[h]))) for h in others}

    arm_specs = {f: _arm_spec(templates[f], provider, model) for f in templates}
    models, caps = {}, {}
    for f, arm in arm_specs.items():
        kwargs = estimates.planned_kwargs(cfg, arm)
        chash = None if kwargs is None else estimates.config_hash_for(arm, kwargs, cfg.run.timeout_multiplier)
        models[f] = estimates.cell_model(cfg, rows_all, arm, config_hash=chash, meta=meta, guess=False)
        caps[f] = estimates.enforced_cap(arm, kwargs or {})[0]
    if not others and goal != "pilot_new_arms":
        notes.append(f"No family to compare with {base} on {provider}:{model}. Name some in families.")
        return out

    limit = cfg.designs.max_budget_usd
    if max_budget_usd is not None:
        limit = min(limit, float(max_budget_usd))
    ceiling = min(float(max_budget_usd) if max_budget_usd is not None else math.inf,
                  cfg.designs.max_budget_usd / estimates.BUDGET_MARGIN)
    seed = stats.stable_seed(cov["ledger_fingerprint"], provider, model, goal, attempts)
    sigma = float(cfg.stats.prior_sd_log)

    def gate() -> _Gate:
        return _Gate(models, caps, meta, sigma, attempts, seed)

    builds: list[tuple[str, _Build, int]] = []
    if goal == "pilot_new_arms":
        first = sample[0]
        pilots = [f for f in [base] + others if f in templates and not _names([r for r in rows if r.get("family") == f],
                                                                              "cost")]
        if not pilots:
            notes.append("Every family already has cost-valid trials on this model; no pilot is needed.")
        else:
            g = gate()
            build = _Build({f: [] for f in [base] + others}, [], False)
            if g.try_add(first, pilots, ceiling):
                for f in pilots:
                    build.sched[f].append(first)
                build.commits.append((first, pilots))
            else:
                build.stopped_by_budget = True
            builds.append(("pilot", build, target))
            for gap in cov["gaps"]:
                if gap["kind"] == "infra_failing" and gap["family"] in pilots:
                    notes.append(f"{gap['detail']} The pilot checks the fix on one task.")
    else:
        failing = {g["family"] for g in cov["gaps"] if g["kind"] == "infra_failing" and g["family"] in others
                   and not measured.get(g["family"])}
        for family in sorted(failing):
            notes.append(f"{family} is left out: its trials on this model fail before using the model. Run "
                         "suggest_runs with goal pilot_new_arms to debug it on one task first.")
        others = [h for h in others if h not in failing]
        out["families"] = others
        if not others:
            return out
        if all(before[h] >= target for h in others):
            notes.append(f"Every family already has {target} paired tasks with {base}; the target is reached.")
            return out
        if goal == "reach_target_pairs":
            builds.append(("full", _greedy(target, (1, 2), base, others, valid, base_for, before, sample, gate(),
                                           ceiling, can_add_base, already), target))
            staged = max(report.parse_options(None, cfg).min_pairs_estimate, math.ceil(target / 2))
            if staged < target:
                builds.append(("staged", _greedy(staged, (1, 2), base, others, valid, base_for, before, sample,
                                                 gate(), ceiling, can_add_base, already), staged))
        builds.append(("half_pairs", _greedy(target, (1,), base, others, valid, base_for, before, sample, gate(),
                                             ceiling, can_add_base), target))

    seen: set[str] = set()
    candidates = []
    for label, build, goal_pairs in builds:
        candidate = _finish(cfg, label, build, goal_pairs, target, base, others, templates, valid, base_for, before,
                            order, rows_all, meta, provider, model, attempts, limit, max_budget_usd)
        if candidate is None:
            continue
        if "dropped" in candidate:
            notes.append(candidate["dropped"])
            continue
        if candidate["spec_hash"] in seen:
            continue
        seen.add(candidate["spec_hash"])
        candidates.append(candidate)
    candidates.sort(key=lambda c: (c["cost_per_pair_usd"] is None, c["cost_per_pair_usd"] or 0.0,
                                   c["estimate"]["p50_usd"] or 0.0, _LABEL_ORDER.get(c["label"], 9)))
    out["candidates"] = candidates
    return out


def suggest_runs(cfg: Config, reader: Any, goal: str, provider: str, model: str,
                 families: Sequence[str] | None = None, baseline_family: str | None = None,
                 target_pairs: int | None = None, max_budget_usd: float | None = None,
                 attempts: int = 1) -> list[dict[str, Any]]:
    """Ranked candidate designs (cheapest per pair first) for a goal on one
    model. Nothing is saved; each candidate's spec is design input."""
    return plan_suggestions(cfg, reader, goal, provider, model, families, baseline_family, target_pairs,
                            max_budget_usd, attempts)["candidates"]


def _spec_input(cfg: Config, build: _Build, base: str, others: list[str], templates: Mapping[str, _Template],
                order: Mapping[str, int], provider: str, model: str, attempts: int, budget: float,
                cap: float | None, meta: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    tasks = sorted({t for ts in build.sched.values() for t in ts}, key=lambda t: order.get(t, len(order)))
    arms = []
    for family in [base] + others:
        mine = sorted(set(build.sched.get(family) or []), key=lambda t: order.get(t, len(order)))
        if not mine or family not in templates:
            continue
        t = templates[family]
        arms.append({"name": t.name, "agent": t.agent, "provider": provider, "model": model, "effort": t.effort,
                     "options": copy.deepcopy(t.options), "tasks": None if mine == tasks else mine})
    n_concurrent = max(1, min(cfg.run.n_concurrent, designs.MAX_CONCURRENT))
    run: dict[str, Any] = {
        "budget_usd": float(budget), "parallel_arms": _parallel_arms(cfg, len(arms), n_concurrent, tasks, meta),
        "n_concurrent": n_concurrent, "per_trial_cap_usd": cap, "max_turns": cfg.run.max_turns,
        "timeout_multiplier": float(cfg.run.timeout_multiplier),
    }
    if cfg.run.effort in designs.RUN_EFFORTS:
        run["effort"] = cfg.run.effort
    in_design = {a["name"] for a in arms}
    baseline_arm = templates[base].name if base in templates and templates[base].name in in_design else None
    return {"tasks": tasks, "attempts": attempts, "baseline_arm": baseline_arm, "run": run, "arms": arms}


def _evaluate(cfg: Config, spec_in: dict[str, Any], rows: Sequence[Mapping[str, Any]],
              replicates: int | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    normalized = designs.normalize(cfg, spec_in)
    validation = designs.validate(cfg, normalized)
    est = None
    if not validation["problems"]:
        est = estimates.estimate(cfg, normalized, rows, validation["plan"], replicates=replicates)
    return normalized, validation, est


def _trial_cap(cfg: Config, est: Mapping[str, Any]) -> float | None:
    """min(the profile's cap, 3 x the p90 trial) for arms whose cap is enforced."""
    cap = cfg.run.per_trial_cap_usd
    if cap is None:
        return None
    p90s = [a["per_trial"]["p90"] for a in est.get("arms") or [] if a.get("cap_usd") is not None
            and (a.get("per_trial") or {}).get("p90") is not None]
    if not p90s:
        return float(cap)
    return float(min(cap, max(0.01, math.ceil(3 * max(p90s) * 100 - 1e-9) / 100)))


def _finish(cfg: Config, label: str, build: _Build, goal_pairs: int, target: int, base: str, others: list[str],
            templates: Mapping[str, _Template], valid: Mapping[str, set[str]], base_for: Mapping[str, set[str]],
            before: Mapping[str, int], order: Mapping[str, int], rows: Sequence[Mapping[str, Any]],
            meta: Mapping[str, Mapping[str, Any]], provider: str, model: str, attempts: int, limit: float,
            max_budget_usd: float | None) -> dict[str, Any] | None:
    """Validate, price and settle one candidate: per-trial cap, then the
    budget at p90 x 1.15, trimming the last tasks while the p90 is over
    max_budget_usd. Settling uses quick estimates (GATE_REPLICATES); the
    full estimate runs once on the settled design. None when there is
    nothing to run."""
    notes: list[str] = []

    def spec_for(budget: float, cap: float | None) -> dict[str, Any]:
        return _spec_input(cfg, build, base, others, templates, order, provider, model, attempts, budget, cap, meta)

    def dropped(validation: Mapping[str, Any]) -> dict[str, Any]:
        return {"dropped": f"The {label} candidate failed validation and was dropped: "
                           + "; ".join(validation["problems"][:3])}

    def over(est: Mapping[str, Any]) -> bool:
        p90 = est["run"]["total"]["p90"]
        return max_budget_usd is not None and p90 is not None and p90 > max_budget_usd

    for _ in range(MAX_TRIMS):
        if not any(build.sched.values()):
            if build.stopped_by_budget:
                return {"dropped": f"No {label} candidate fits: even the first task's p90 cost is over the "
                                   f"${limit:.2f} ceiling."}
            return None
        cap = cfg.run.per_trial_cap_usd
        normalized, validation, quick = _evaluate(cfg, spec_for(limit, cap), rows, GATE_REPLICATES)
        if quick is None:
            return dropped(validation)
        new_cap = _trial_cap(cfg, quick)
        if new_cap != cap:
            cap = new_cap
            normalized, validation, quick = _evaluate(cfg, spec_for(limit, cap), rows, GATE_REPLICATES)
            if quick is None:
                return dropped(validation)
        suggested = quick["run"]["suggested_budget_usd"] or limit
        budget = min(suggested, limit)
        spec_in = spec_for(budget, cap)
        normalized, validation, quick = _evaluate(cfg, spec_in, rows, GATE_REPLICATES)
        if quick is None:
            return dropped(validation)
        est: dict[str, Any] | None = None
        if not over(quick):
            normalized, validation, est = _evaluate(cfg, spec_in, rows)
            if est is None:
                return dropped(validation)
            if not over(est):
                break
        if not _trim(build):
            return {"dropped": f"No {label} candidate fits under max_budget_usd ${max_budget_usd:.2f}."}
        build.stopped_by_budget = True
    else:
        return {"dropped": f"The {label} candidate could not be fitted under the budget."}
    assert est is not None  # the loop only breaks on a full estimate

    if budget < suggested:
        notes.append(f"Budget capped at ${budget:.2f} (the suggestion was ${suggested:.2f}); chance of finishing "
                     f"within it: {est['run']['p_within_budget']:.0%}.")
    if build.stopped_by_budget:
        notes.append("Stopped adding tasks where the p90 cost would pass the budget ceiling; a later run can "
                     "continue in seeded order.")
    if label == "staged":
        notes.append(f"Stage 1: about half the target ({goal_pairs} pairs). Re-check get_coverage afterwards; the "
                     "observed spread then sizes the rest.")
    for warning in est.get("warnings") or []:
        notes.append(warning)

    pairs = []
    gained_total = 0
    base_sched = set(build.sched.get(base) or [])
    for h in others:
        both_after = (valid[h] | set(build.sched.get(h) or [])) & (base_for[h] | base_sched)
        gained = len(both_after) - len(valid[h] & base_for[h])
        gained_total += gained
        pairs.append({"family": h, "before": before[h], "after": before[h] + gained, "target": target})
    p50 = est["run"]["total"]["p50"]
    cost_per_pair = round(p50 / gained_total, 4) if gained_total and p50 is not None else None
    trials = sum(a["trials"] for a in est["arms"])
    family_of = {t.name: t.family for t in templates.values()}
    basis_of = {a["arm"]: a["basis"] for a in est["arms"]}
    return {
        "label": label, "spec": designs.to_input(normalized), "spec_hash": est["spec_hash"],
        "validation": {"status": validation["status"], "problems": validation["problems"],
                       "warnings": validation["warnings"]},
        "estimate": {"p10_usd": est["run"]["total"]["p10"], "p50_usd": p50, "p90_usd": est["run"]["total"]["p90"],
                     "suggested_budget_usd": est["run"]["suggested_budget_usd"],
                     "p_within_budget": est["run"]["p_within_budget"],
                     "wall_minutes_p50": est["run"]["wall_minutes"]["p50"],
                     "inflight_allowance_usd": est["run"]["inflight_allowance_usd"]},
        "trials": trials,
        "arms": [{"name": a["name"], "family": family_of.get(a["name"], a["family"]),
                  "tasks": len(a["tasks"]) if a["tasks"] is not None else len(normalized["tasks"]),
                  "basis": basis_of.get(a["name"])} for a in normalized["arms"]],
        "pairs_gained": pairs, "cost_per_pair_usd": cost_per_pair,
        "answers": _answers(cfg, pairs, provider, model, target), "notes": notes,
    }


def _answers(cfg: Config, pairs: Sequence[Mapping[str, Any]], provider: str, model: str, target: int) -> str:
    if not pairs:
        return "Prices and debugs the arms on one task; no comparison gains pairs."
    o = report.parse_options(None, cfg)
    parts = [f"{p['family']} {p['after']}/{target}" for p in pairs]
    lowest = min(p["after"] for p in pairs)
    if lowest >= target:
        reach = "every comparison reaches the precision target"
    elif lowest >= o.min_pairs_verdict:
        reach = f"every comparison has enough pairs for a verdict ({o.min_pairs_verdict}+), not yet the target"
    elif lowest >= o.min_pairs_estimate:
        reach = f"every comparison gets an estimate ({o.min_pairs_estimate}+ pairs), not yet a verdict"
    else:
        reach = f"some comparisons stay below the {o.min_pairs_estimate} pairs an estimate needs"
    ref = report.reference_for(cfg.reference, pairs[0]["family"], provider, model, "cost")
    applies = "" if ref is None or ref["applies"] else " Published figures do not apply to this model, so no verdict."
    return f"Paired tasks after the run: {', '.join(parts)}; {reach}.{applies}"
