"""Turns Harbor's per-trial result.json files into the numbers the bench
compares: tokens, cost and pass rate per arm, and each arm's multiple of the
baseline next to the published figure for it.

Pass rates on a 10-task sample are too noisy to compare. The token and cost
multiples are the point of the sample, so only trials in which the agent
actually used the model feed the per-trial means; a trial that died at setup
with no tokens would otherwise drag an arm's mean down and inflate its
multiple. Spend is real either way, so totals cover every trial. A trial
whose agent ran but recorded no usage (a timeout before any model call
completed) is left out of the means too, and its spend is unknown: totals
are then lower bounds, and usage_status says which trials they miss.

Two cost figures are kept per trial: the bench's own (pricing.yaml applied to
the token counts, the same for every arm) and the harness's. The budget uses
the higher of the two, except where the harness is known to price this
provider's models at another provider's rates.
"""

from __future__ import annotations

import json
from datetime import datetime
from fnmatch import fnmatchcase
from pathlib import Path
from statistics import mean
from typing import Any

from harbor.models.trial.error_policy import FailureCause, policy_for

from . import classify, extract, harnesses
from .settings import EFFORT_INHERIT, JOBS_DIR, Arm, Config, Price, is_variant

# Failures that end a trial before it is a task attempt. Harbor still runs the
# verifier after some of them (a provider error is a NonZeroAgentExitCodeError
# subclass) and records reward 0.0, which is not a measurement of the harness.
# A plain NonZeroAgentExitCodeError stays scored: it is the harness's own exit,
# often after minutes of real work.
_NOT_AN_ATTEMPT = frozenset({FailureCause.MODEL_PROVIDER_FAILED, FailureCause.SANDBOX_FAILED,
                             FailureCause.HARBOR_FAILED, FailureCause.CANCELLED})

_DEFAULT_TOLERANCE = 1.5


def _cost(price: Price, n_input: int, n_cache: int, n_output: int) -> float:
    # Harbor counts cache reads inside n_input, and folds cache writes into it
    # too, so writes are priced at the plain input rate. The ledger's
    # pricing.unit_cost uses this same formula.
    fresh = max(n_input - n_cache, 0)
    cache_rate = price.input if price.cache_read is None else price.cache_read
    return (fresh * price.input + n_cache * cache_rate + n_output * price.output) / 1e6


def _ceiling(cfg: Config, provider: str) -> Price | None:
    """The highest rates in the provider's price table (any provider's when it
    has none), used so tokens of an unpriced model never count as $0 spend."""
    prices = list((cfg.pricing.get(provider) or {}).values())
    if not prices:
        prices = [p for table in cfg.pricing.values() for p in table.values()]
    if not prices:
        return None
    return Price(
        input=max(p.input for p in prices),
        output=max(p.output for p in prices),
        cache_read=max(p.input if p.cache_read is None else p.cache_read for p in prices),
    )


def _minutes(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60
    except ValueError:
        return None


def _same_model(reported: str, arm: Arm) -> bool:
    return is_variant(reported.split("/")[-1], arm.model.split("/")[-1])


def _reward(data: dict[str, Any]) -> float | None:
    rewards = (data.get("verifier_result") or {}).get("rewards") or {}
    if not isinstance(rewards, dict) or not rewards:
        return None
    value = rewards.get("reward", next(iter(rewards.values()), None))
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def read_trial(path: Path, cfg: Config, arm: Arm) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # still being written
    if not isinstance(data, dict):
        return None
    ctx = data.get("agent_result") or {}
    n_input = ctx.get("n_input_tokens") or 0
    n_cache = ctx.get("n_cache_tokens") or 0
    n_output = ctx.get("n_output_tokens") or 0

    # Zero-token entries are skipped, as Harbor's own cost estimate does: Claude
    # Code records error and interrupt messages under model '<synthetic>' with
    # no usage, and listing one as unpriced would undo the spend rule below.
    usage = {}
    for model, u in (ctx.get("model_usage") or {}).items():
        u = u or {}
        counts = tuple(u.get(k) or 0 for k in ("n_input_tokens", "n_cache_tokens", "n_output_tokens"))
        if counts[0] or counts[2]:
            usage[model] = counts
    usage_by_arm = not usage  # no per-model breakdown: price the totals as the arm's model
    if usage_by_arm:
        usage = {arm.model: (n_input, n_cache, n_output)}
    # Tokens in the totals but under no model (Harbor files a mini-swe-agent
    # FormatError turn as an observation) go where the ledger puts them
    # (extract._usage): to the one model that used tokens, else the arm's.
    gap = [max(total - sum(u[i] for u in usage.values()), 0) for i, total in enumerate((n_input, n_cache, n_output))]
    if any(gap):
        primary = [m for m in usage if _same_model(m, arm)]
        target = next(iter(usage)) if len(usage) == 1 else primary[0] if len(primary) == 1 else arm.model
        usage[target] = tuple(a + b for a, b in zip(usage.get(target, (0, 0, 0)), gap))

    ceiling = _ceiling(cfg, arm.provider)
    computed, floor, unpriced, other_tokens = 0.0, 0.0, [], 0
    for model, (u_in, u_cache, u_out) in usage.items():
        if not usage_by_arm and not _same_model(model, arm):
            other_tokens += u_in + u_out
        price = cfg.price(arm.provider, model)
        if price is None:
            unpriced.append(model)
            if ceiling is not None:
                floor += _cost(ceiling, u_in, u_cache, u_out)
        else:
            computed += _cost(price, u_in, u_cache, u_out)

    reward = _reward(data)
    exc = data.get("exception_info") or {}
    error = exc.get("exception_type")
    cause = policy_for(error).cause if error else None
    if cause in _NOT_AN_ATTEMPT:
        reward = None  # still counted under errors
    return {
        "arm": arm.name,
        "family": arm.family or arm.agent,
        "task": data.get("task_name"),
        "trial": data.get("trial_name"),
        "reward": reward,
        "reward_valid": reward is not None,
        "passed": None if reward is None else reward >= 1.0,
        "error": error,
        "failure_cause": cause.value if cause else None,
        "error_message": (exc.get("exception_message") or "")[:400],  # the ledger keeps the full text
        "n_input": n_input,
        "n_cache": n_cache,
        "n_output": n_output,
        "computed_cost": None if unpriced and not computed else computed,
        "reported_cost": ctx.get("cost_usd"),
        # Some models priced, others not: the computed figure is only part of the cost.
        "cost_partial": bool(unpriced) and computed > 0,
        # The cost of the unpriced tokens at the provider's highest rates.
        "unpriced_floor_cost": floor,
        # The harness's own cost uses another provider's prices (Claude Code
        # prices DeepSeek tokens at Claude rates, about 50x too high).
        "reported_mispriced": harnesses.reports_mispriced_cost(arm.family or arm.agent, arm.provider),
        "unpriced_models": unpriced,
        "models": [] if usage_by_arm else sorted(usage),
        "other_model_tokens": other_tokens,
        "minutes": _minutes(data.get("started_at"), data.get("finished_at")),
        # The ledger's rule: 'unmeasured' when the agent ran but nothing was
        # recorded, 'lower_bound' when a timeout cut the recorded usage short.
        "usage_status": classify.usage_status(extract.judged_facts(data)),
    }


def cost(row: dict[str, Any]) -> float:
    """Like-for-like cost: our price table first, the harness's figure otherwise."""
    computed, reported = row["computed_cost"], row["reported_cost"]
    if row.get("reported_mispriced") and (row.get("cost_partial") or computed is None):
        # The harness's figure is at another provider's rates (about 50x for
        # Claude Code on DeepSeek), whether some or all of the models are
        # unpriced; the unpriced tokens get the top rate instead.
        return (computed or 0.0) + row.get("unpriced_floor_cost", 0.0)
    if row.get("cost_partial"):
        return max(computed or 0.0, reported or 0.0)
    if computed is not None:
        return computed
    return reported or 0.0


def spend(row: dict[str, Any]) -> float:
    """For budget enforcement: the higher of our figure and the harness's.
    The harness's figure includes what ours cannot see (Anthropic cache-write
    premiums, Haiku's long-prompt rate, OpenRouter's billed routing). The one
    exception is a harness that prices this provider's models at another
    provider's rates; when every model is in our table, ours is used there."""
    computed, reported = row["computed_cost"], row["reported_cost"]
    if computed is not None and not row["unpriced_models"] and row.get("reported_mispriced"):
        return computed
    value = max(computed or 0.0, reported or 0.0)
    if row["unpriced_models"] and not reported:
        # Nothing prices these tokens, so charge them at the top rate rather than $0.
        value = max(value, (computed or 0.0) + row.get("unpriced_floor_cost", 0.0))
    return value


def load_job(job_dir: Path, cfg: Config, arm: Arm) -> list[dict[str, Any]]:
    if not job_dir.exists():
        return []
    rows = (read_trial(p, cfg, arm) for p in sorted(job_dir.glob("*/result.json")))
    return [r for r in rows if r is not None]


def _measured(row: dict[str, Any]) -> bool:
    """The agent used the model. Timeouts and mid-run crashes still count:
    their tokens are a real measurement."""
    return bool(row["n_input"] or row["n_output"] or cost(row))


def summarize(arm: Arm, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "arm": arm.name, "agent": arm.agent, "family": arm.family or arm.agent,
        "provider": arm.provider, "model": arm.model,
        "trials": len(rows), "measured": 0, "excluded": 0, "unmeasured": 0, "lower_bound": 0,
    }
    if not rows:
        return out
    measured = [r for r in rows if _measured(r)]
    scored = [r for r in rows if r["passed"] is not None]
    passed = sum(1 for r in scored if r["passed"])
    costs = [cost(r) for r in rows]
    total_in = sum(r["n_input"] for r in rows)
    total_tokens = total_in + sum(r["n_output"] for r in rows)
    reported = [r["reported_cost"] for r in rows]
    minutes = [r["minutes"] for r in rows if r["minutes"] is not None]
    out.update({
        "measured": len(measured),
        "excluded": len(rows) - len(measured),
        # Of the excluded: trials that ran but recorded no usage, so their spend is not in total_cost.
        "unmeasured": sum(1 for r in rows if r.get("usage_status") == "unmeasured" and not spend(r)),
        # Measured, but cut short by a timeout: their usage covers completed calls only.
        "lower_bound": sum(1 for r in measured if r.get("usage_status") == "lower_bound"),
        "scored": len(scored),
        "passed": passed,
        "errors": sum(1 for r in rows if r["error"]),
        "pass_rate": passed / len(scored) if scored else None,
        "mean_input": mean(r["n_input"] for r in measured) if measured else None,
        "mean_cache": mean(r["n_cache"] for r in measured) if measured else None,
        "mean_output": mean(r["n_output"] for r in measured) if measured else None,
        "mean_tokens": (sum(r["n_input"] + r["n_output"] for r in measured) / len(measured)
                        if measured else None),
        "cache_share": sum(r["n_cache"] for r in rows) / total_in if total_in else None,
        "mean_cost": mean(cost(r) for r in measured) if measured else None,
        "total_cost": sum(costs),
        "reported_cost_total": sum(reported) if all(c is not None for c in reported) else None,
        "cost_per_success": sum(costs) / passed if passed else None,
        "cost_partial": any(r.get("cost_partial") for r in rows),
        "other_model_share": sum(r["other_model_tokens"] for r in rows) / total_tokens if total_tokens else 0.0,
        "models": sorted({m for r in rows for m in r["models"]}),
        "unpriced_models": sorted({m for r in rows for m in r["unpriced_models"]}),
        "mean_minutes": mean(minutes) if minutes else None,
    })
    return out


def _verdict(measured: float | None, ref: dict[str, Any] | None,
             tolerance: float = _DEFAULT_TOLERANCE) -> str | None:
    if measured is None or not ref:
        return None
    if "value" in ref:
        low, high = ref["value"] / tolerance, ref["value"] * tolerance
    elif "low" in ref and "high" in ref:
        low, high = ref["low"], ref["high"]
    else:
        return None
    if measured < low:
        return "lower"
    return "higher" if measured > high else "in line"


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a and b else None


def _tolerance(*candidates: Any) -> float:
    """The first usable tolerance: per-arm, then reference defaults."""
    for value in candidates:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value >= 1.0:
            return value
    return _DEFAULT_TOLERANCE


def _applies(patterns: Any, summary: dict[str, Any]) -> bool:
    """Whether published figures apply to this arm's model. reference.yaml's
    applies_to lists provider:model globs; none listed means every model."""
    if patterns is None:
        return True
    if isinstance(patterns, str):
        patterns = [patterns]
    target = f"{summary.get('provider') or ''}:{summary.get('model') or ''}"
    return any(fnmatchcase(target, str(p)) for p in patterns)


def compare(cfg: Config, summaries: list[dict[str, Any]], baseline: str) -> list[dict[str, Any]]:
    by_name = {s["arm"]: s for s in summaries}
    base = by_name.get(baseline)
    if not base or not base.get("trials"):
        return []
    reference = cfg.reference or {}
    defaults = reference.get("defaults") or {}
    consensus = (reference.get("consensus") or {}).get("cost_per_trial")
    refs = reference.get("arms") or {}
    rows = []
    for s in summaries:
        if s["arm"] == baseline or not s.get("trials"):
            continue
        family = s.get("family") or s["agent"]
        ref = refs.get(family) or {}
        tolerance = _tolerance(ref.get("tolerance"), defaults.get("tolerance"))
        patterns = ref.get("applies_to", defaults.get("applies_to"))
        applies = _applies(patterns, s) and _applies(patterns, base)
        tokens_x = _ratio(base.get("mean_tokens"), s.get("mean_tokens"))
        cost_x = _ratio(base.get("mean_cost"), s.get("mean_cost"))
        published_tokens = ref.get("tokens_per_trial")
        published_cost = ref.get("cost_per_trial") or consensus
        rows.append({
            "arm": s["arm"],
            "family": family,
            "tokens_multiple": tokens_x,
            "cost_multiple": cost_x,
            "published_tokens": published_tokens,
            "published_cost": published_cost,
            "tokens_verdict": _verdict(tokens_x, published_tokens, tolerance) if applies else None,
            "cost_verdict": _verdict(cost_x, published_cost, tolerance) if applies else None,
            "verdict_note": None if applies else (
                "No verdict: the published figures were measured on Claude models, "
                f"and this comparison runs {s.get('model')} on {s.get('provider')}."),
            "note": ref.get("note"),
            "excluded": s.get("excluded", 0),
            "baseline_excluded": base.get("excluded", 0),
        })
    return rows


def _arm(name: Any, agent: Any, provider: Any, model: Any, effort: Any = None,
         options: Any = None, family: Any = None) -> Arm:
    agent = str(agent or "")
    return Arm(
        name=str(name), agent=agent, provider=str(provider or ""), model=str(model or ""),
        effort=effort if isinstance(effort, str) and effort else EFFORT_INHERIT,
        options=dict(options) if isinstance(options, dict) else {},
        family=str(family or "") or agent,
    )


def arms_from_snapshot(snapshot: dict[str, Any]) -> list[Arm]:
    """Arms from the config snapshot a run.json recorded when it started."""
    return [
        _arm(a["name"], a.get("agent"), a.get("provider"), a.get("model"),
             a.get("effort"), a.get("options"), a.get("family"))
        for a in snapshot.get("arms") or [] if isinstance(a, dict) and a.get("name")
    ]


def arms_for_run(run: dict[str, Any]) -> list[Arm]:
    """The arms exactly as the run resolved them; runs recorded before
    resolved_arms existed fall back to their config snapshot."""
    resolved = run.get("resolved_arms")
    if not resolved:
        return arms_from_snapshot(run.get("config") or {})
    arms = []
    for a in resolved:
        if not isinstance(a, dict) or not a.get("name"):
            continue
        effort = a.get("effort")
        setting = effort.get("setting") if isinstance(effort, dict) else effort
        options = {k["name"]: k.get("value") for k in a.get("kwargs") or []
                   if isinstance(k, dict) and k.get("name") and k.get("source") == "arm option"}
        arms.append(_arm(a["name"], a.get("agent"), a.get("provider"), a.get("model"),
                         setting, options, a.get("family")))
    return arms


def run_results(cfg: Config, run: dict[str, Any]) -> dict[str, Any]:
    """Prices with the arm settings recorded when the run started."""
    run_arms = {a.name: a for a in arms_for_run(run)}
    summaries, trials = [], []
    for arm_run in run.get("arms") or []:
        arm = run_arms.get(arm_run.get("name"))
        if arm is None:
            continue
        rows = load_job(JOBS_DIR / run["id"] / arm.name, cfg, arm)
        trials += rows
        summaries.append(summarize(arm, rows))
    baseline = ((run.get("spec") or {}).get("baseline_arm")
                or (run.get("config") or {}).get("baseline_arm")
                or cfg.baseline_arm)
    return {
        "run_id": run["id"],
        "baseline": baseline,
        "arms": summaries,
        "comparison": compare(cfg, summaries, baseline),
        "trials": trials,
    }
