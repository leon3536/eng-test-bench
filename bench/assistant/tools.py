"""The assistant's nine tools: what the model may ask the bench, and the one
thing it may write.

Eight tools read: bench state, harness options, tasks, coverage, measured
costs, saved designs, an estimate and suggested designs. One writes, and
only a draft design through the DraftWriter, which cannot approve, queue,
archive or start anything. A person still approves a specific version in the
Designs view; a tool has no path to a run, a key or the queue.

Inputs arrive eagerly streamed, so the API does not validate them: every
input is checked with its pydantic model first, and one that fails comes back
as an INVALID_JSON error without the handler running. Every result is JSON
(sorted keys) passed through redact() and cut to 20,000 characters, and
results carry figures, names and enums only: no trial logs, exception
messages or task instructions, which a hostile task could use to steer the
model. Numbers come from estimates.py and coverage.py, the same engine the
REPORT tab uses.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .. import coverage, datasets, designs, estimates, harnesses, reuse, schemas, settings
from ..designs import DESIGN_SPEC_SCHEMA, DesignNotFound, DesignSpecIn, VersionConflict
from ..fsutil import canonical_json, sha256_hex
from ..spec import run_dict
from .redact import redact
from .types import ToolContext, ToolOutcome

MAX_RESULT_CHARS = 20_000
MAX_ERROR_DETAIL = 500
_INVALID_ECHO_CHARS = 4_000
_RECENT_RUNS = 10
_SAMPLE_BLOCK = 10

__all__ = ["TOOL_DEFS", "TOOLS_HASH", "ALLOWED_TOOL_NAMES", "INPUT_MODELS", "DESIGN_SPEC_SCHEMA", "execute"]


# Definitions -----------------------------------------------------------------------------------
# Fixed order and text: they are part of the cached prompt prefix, and a
# conversation locks when TOOLS_HASH changes.

def _tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None
          ) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = required
    return {"name": name, "description": description, "input_schema": schema, "eager_input_streaming": True}


TOOL_DEFS: list[dict[str, Any]] = [
    _tool(
        "get_bench_context",
        "Read the bench's current state: profile, dataset and snapshot, providers (enabled, and whether a key is "
        "set; you never see keys), priced models, run defaults, design limits, harnesses allowed in designs with "
        "their family, the config arms and baseline, readiness (Docker with CPUs and memory, dataset), the active "
        "run, the queue with its global spend cap, the 10 most recent runs, a ledger summary and your own spend "
        "today. Call it first in every conversation, and again before recommending a budget if the user mentions "
        "new runs.",
        {},
    ),
    _tool(
        "describe_harness",
        "Get the options a harness accepts in the pinned Harbor version: effort_levels (null = takes no effort "
        "setting, [] = any text), and every option with kind, enum, default, description, group and caution, plus "
        "the family its trials pool under. With provider and model, also says which dollar and turn caps would "
        "apply and lists warnings (a dollar cap the harness misprices on that provider, options current Claude "
        "models reject). Call it before putting an effort level or any option on an arm.",
        {"agent": {"type": "string", "description": "An agent from get_bench_context.harnesses."},
         "provider": {"type": "string"}, "model": {"type": "string"}},
        ["agent"],
    ),
    _tool(
        "list_tasks",
        "List dataset tasks with position and block in the seeded sample, difficulty, category, agent time limit, "
        "CPU and memory, expert time estimate and, with provider and model, cost-valid trials per family. Use it to "
        "choose tasks. Prefer contiguous seeded positions (1-10 is block 1, 11-20 block 2) so new trials pair with "
        "existing ones; never choose tasks because they are cheap.",
        {"scope": {"type": "string", "enum": ["sample", "all"],
                   "description": "sample (default): the seeded sample in order. all: every task in the snapshot."},
         "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
         "provider": {"type": "string"}, "model": {"type": "string"},
         "limit": {"type": "integer", "description": "Default 40, max 80."},
         "offset": {"type": "integer"}},
    ),
    _tool(
        "get_coverage",
        "Coverage of one model across every stored run, however it was launched, on the REPORT tab's task scope "
        "(the profile's task file; scope says which): for each harness family, its "
        "cost-valid trials, distinct tasks, tasks paired with the baseline, infrastructure failures, pass rate with "
        "interval, the cost and token multiples against the baseline (arm divided by baseline; 0.5 means the arm "
        "uses half, 2.0 twice as much) with 90% intervals, the published figure in the same terms and a verdict, and "
        "the paired tasks still needed for the precision target. Also returns ranked gaps. Call it before proposing "
        "any run.",
        {"provider": {"type": "string"},
         "model": {"type": "string", "description": "Model id as configured, e.g. deepseek/deepseek-v4.1-flash."},
         "baseline_family": {"type": "string", "description": "Default: the family of the profile's baseline arm."},
         "target_pairs": {"type": "integer", "description": "Default: computed from the precision target."},
         "match": {"type": "string", "enum": ["family", "cell"],
                   "description": "family (default): every configuration of a family pools together, with its "
                                  "configurations listed. cell: each exact configuration (effort and options) is its "
                                  "own line."}},
        ["provider", "model"],
    ),
    _tool(
        "get_cost_stats",
        "Measured per-trial figures for harness configurations or families in the ledger: trials by outcome, cost "
        "per trial (mean, median, p90, like-for-like and conservative spend), tokens (fresh input, cache, output), "
        "minutes, pass rate with interval, infrastructure-failure rate, top failure causes, unpriced trials and, "
        "optionally, one row per task. All filters are optional. Use it to answer what a family costs per trial and "
        "to explain an estimate.",
        {"family": {"type": "string"}, "provider": {"type": "string"}, "model": {"type": "string"},
         "match": {"type": "string", "enum": ["cell", "family"],
                   "description": "cell (default): one row per configuration. family: one row per family on each "
                                  "model."},
         "per_task": {"type": "boolean"}},
    ),
    _tool(
        "list_designs",
        "Without design_id, lists saved run designs with status (draft, approved, queued, running, ran, archived), "
        "author, latest version and last estimate. With design_id, returns that design's version history and the "
        "full spec of one version (the latest unless version is given). Read the latest version before revising a "
        "design.",
        {"design_id": {"type": "string"}, "version": {"type": "integer"}, "include_archived": {"type": "boolean"}},
    ),
    _tool(
        "estimate_cost",
        "Validate a run design with the same checks a person's run goes through (limits, tasks, allowed harnesses, "
        "keys present, providers enabled, options, effort levels, prices known, Docker capacity) and estimate its "
        "cost and time from measured ledger figures: per arm and in total p10/p50/p90 cost with the basis of each "
        "figure (measured with n, borrowed from another model's tokens, or prior), worst case under per-trial caps, "
        "chance of staying within budget, a suggested budget, wall-clock time, the in-flight overshoot allowance "
        "and the paired tasks it adds per comparison with the expected narrowing of each interval. Give exactly one "
        "of spec, or design_id with an optional version. Quote costs only from this tool. It never starts anything.",
        {"spec": copy.deepcopy(DESIGN_SPEC_SCHEMA), "design_id": {"type": "string"}, "version": {"type": "integer"}},
    ),
    _tool(
        "suggest_runs",
        "Compute the cheapest designs that reach a goal on one model from ledger coverage and measured costs. "
        "reach_target_pairs: enough paired tasks for the precision target. complete_half_pairs: only trials that "
        "finish existing half-pairs. pilot_new_arms: one task (seeded position 1) for harness families with no "
        "cost-valid trials, to price and debug them. It completes half-pairs first, then extends in seeded-sample "
        "order, shares baseline trials across families and stays within max_budget_usd. Returns ranked candidates "
        "with a spec, estimate, pairs gained and cost per pair. Candidates are not saved; pass one to "
        "save_draft_design to keep it.",
        {"goal": {"type": "string", "enum": ["reach_target_pairs", "complete_half_pairs", "pilot_new_arms"]},
         "provider": {"type": "string"}, "model": {"type": "string"},
         "families": {"type": "array", "items": {"type": "string"},
                      "description": "Default: families of the profile's arms on this model."},
         "baseline_family": {"type": "string"}, "target_pairs": {"type": "integer"},
         "max_budget_usd": {"type": "number", "description": "Skip candidates whose p90 cost exceeds this."},
         "attempts": {"type": "integer"}},
        ["goal", "provider", "model"],
    ),
    _tool(
        "save_draft_design",
        "Save a run design as a DRAFT for a person to review: a new design (omit design_id) or a new version "
        "(design_id with base_version = the latest version you read). Drafts never run. Only a person can queue or "
        "start a design, in the Designs view. Call it after estimate_cost, when the user agrees to a recommendation "
        "or asks you to keep one. Designs with problems are saved, marked invalid. If saving fails because a newer "
        "version exists, read it with list_designs and re-apply your change.",
        {"design_id": {"type": "string"}, "base_version": {"type": "integer"},
         "title": {"type": "string", "description": "Under 80 characters."},
         "purpose": {"type": "string", "description": "The question this design answers, one or two sentences."},
         "rationale": {"type": "string", "description": "Why these arms, tasks, attempts and budget, citing figures "
                                                        "from tool results. Under 1500 characters."},
         "change_note": {"type": "string", "description": "One line: what changed and why."},
         "spec": copy.deepcopy(DESIGN_SPEC_SCHEMA)},
        ["title", "purpose", "rationale", "change_note", "spec"],
    ),
]

TOOLS_HASH = "sha256:" + sha256_hex(canonical_json(TOOL_DEFS))
ALLOWED_TOOL_NAMES = frozenset({
    "get_bench_context", "describe_harness", "list_tasks", "get_coverage", "get_cost_stats", "list_designs",
    "estimate_cost", "suggest_runs", "save_draft_design",
})


# Inputs ----------------------------------------------------------------------------------------------

class _In(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


_Name = Field(default=None, min_length=1, max_length=300)


class GetBenchContextIn(_In):
    pass


class DescribeHarnessIn(_In):
    agent: str = Field(min_length=1, max_length=300)
    provider: str | None = _Name
    model: str | None = _Name


class ListTasksIn(_In):
    scope: Literal["sample", "all"] = "sample"
    difficulty: Literal["easy", "medium", "hard"] | None = None
    provider: str | None = _Name
    model: str | None = _Name
    limit: int = Field(default=40, ge=1, le=80)
    offset: int = Field(default=0, ge=0)


class GetCoverageIn(_In):
    provider: str = Field(min_length=1, max_length=300)
    model: str = Field(min_length=1, max_length=300)
    baseline_family: str | None = _Name
    target_pairs: int | None = Field(default=None, ge=2, le=1000)
    match: Literal["family", "cell"] = "family"


class GetCostStatsIn(_In):
    family: str | None = _Name
    provider: str | None = _Name
    model: str | None = _Name
    match: Literal["cell", "family"] = "cell"
    per_task: bool = False


class ListDesignsIn(_In):
    design_id: str | None = _Name
    version: int | None = Field(default=None, ge=1)
    include_archived: bool = False

    @model_validator(mode="after")
    def _version_needs_design(self) -> ListDesignsIn:
        if self.version is not None and self.design_id is None:
            raise ValueError("version needs design_id")
        return self


class EstimateCostIn(_In):
    spec: DesignSpecIn | None = None
    design_id: str | None = _Name
    version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _one_source(self) -> EstimateCostIn:
        if (self.spec is None) == (self.design_id is None):
            raise ValueError("give exactly one of spec or design_id")
        if self.version is not None and self.design_id is None:
            raise ValueError("version needs design_id")
        return self


class SuggestRunsIn(_In):
    goal: Literal["reach_target_pairs", "complete_half_pairs", "pilot_new_arms"]
    provider: str = Field(min_length=1, max_length=300)
    model: str = Field(min_length=1, max_length=300)
    families: list[str] | None = Field(default=None, max_length=12)
    baseline_family: str | None = _Name
    target_pairs: int | None = Field(default=None, ge=2, le=1000)
    max_budget_usd: float | None = Field(default=None, gt=0)
    attempts: int = Field(default=1, ge=1, le=10)


class SaveDraftDesignIn(_In):
    design_id: str | None = _Name
    base_version: int | None = Field(default=None, ge=1)
    title: str = Field(min_length=1, max_length=80)
    purpose: str = Field(max_length=1000)
    rationale: str = Field(max_length=1500)
    change_note: str = Field(max_length=200)
    spec: DesignSpecIn

    @model_validator(mode="after")
    def _version_pairs_with_design(self) -> SaveDraftDesignIn:
        if (self.design_id is None) != (self.base_version is None):
            raise ValueError("a new version needs both design_id and base_version; a new design needs neither")
        return self


INPUT_MODELS: dict[str, type[BaseModel]] = {
    "get_bench_context": GetBenchContextIn,
    "describe_harness": DescribeHarnessIn,
    "list_tasks": ListTasksIn,
    "get_coverage": GetCoverageIn,
    "get_cost_stats": GetCostStatsIn,
    "list_designs": ListDesignsIn,
    "estimate_cost": EstimateCostIn,
    "suggest_runs": SuggestRunsIn,
    "save_draft_design": SaveDraftDesignIn,
}


# Results -----------------------------------------------------------------------------------------------

class ToolError(Exception):
    """A refusal the model should read: an error code and a plain sentence."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass
class _Result:
    data: Any
    summary: str
    draft: dict[str, Any] | None = None


def _compact(value: Any) -> Any:
    """JSON-safe and short: six significant digits, no NaN or infinity."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.6g}")
    if isinstance(value, Mapping):
        return {str(k): _compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_compact(v) for v in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _compact(dataclasses.asdict(value))
    return value


def _dump(data: Any) -> str:
    return json.dumps(_compact(data), sort_keys=True, ensure_ascii=False, default=str)


def render(data: Any) -> str:
    """The tool_result text: sorted JSON, redacted, then cut to the limit
    with a hint, still valid JSON."""
    text = redact(_dump(data))
    if len(text) <= MAX_RESULT_CHARS:
        return text
    size = MAX_RESULT_CHARS - 200
    while size > 0:
        wrapped = json.dumps({"truncated": True, "chars": len(text),
                              "hint": "The result was too long; narrow the filter (provider, model, family, "
                                      "per_task off, a design_id) and ask again.",
                              "head": text[:size]}, ensure_ascii=False)
        if len(wrapped) <= MAX_RESULT_CHARS:
            return redact(wrapped)
        size = int(size * 0.9)
    return json.dumps({"truncated": True, "hint": "narrow the filter"})


def _error(code: str, detail: str) -> str:
    return render({"error": code, "detail": redact(str(detail))[:MAX_ERROR_DETAIL]})


def _invalid(raw_input: Any, exc: Exception) -> str:
    try:
        echo = json.dumps(raw_input, ensure_ascii=False, default=str)
    except (TypeError, ValueError, RecursionError):
        try:
            echo = repr(raw_input)
        except Exception:  # RecursionError on a deeply nested input; execute must never raise
            echo = "<input could not be echoed>"
    # Redact before cutting: a key split by the cut would match neither its value nor a key pattern.
    echo = redact(echo)
    if len(echo) > _INVALID_ECHO_CHARS:
        echo = echo[:_INVALID_ECHO_CHARS] + "…"
    errors: list[dict[str, str]] = []
    if isinstance(exc, ValidationError):
        for e in exc.errors()[:20]:
            # loc and msg only: pydantic's "input" field would echo values back.
            errors.append({"loc": ".".join(str(p) for p in e.get("loc", ())), "msg": str(e.get("msg", ""))[:200]})
    else:
        errors.append({"loc": "", "msg": "the input must be a JSON object"})
    return render({"INVALID_JSON": echo, "errors": errors})


def execute(ctx: ToolContext, name: str, raw_input: Any) -> ToolOutcome:
    """Run one tool call. Never raises: an unknown tool, input that does not
    validate, a refusal and a failure each come back as an error result."""
    if name not in ALLOWED_TOOL_NAMES:
        names = ", ".join(d["name"] for d in TOOL_DEFS)
        return ToolOutcome(False, _error("unknown_tool", f"There is no tool named {str(name)[:80]!r}. The tools are: "
                                                         f"{names}."), redact(f"Unknown tool {str(name)[:80]}"))
    try:
        if not isinstance(raw_input, dict):
            raise TypeError("not an object")
        args = INPUT_MODELS[name].model_validate(raw_input)
    except (ValidationError, TypeError) as exc:
        return ToolOutcome(False, _invalid(raw_input, exc), f"{name}: the input did not validate")
    try:
        result = _HANDLERS[name](ctx, args)
        # Rendered inside the try: data that cannot be rendered is a failed call, not a crashed turn.
        content = render(result.data)
        summary = redact(str(result.summary))[:200]
    except ToolError as exc:
        return ToolOutcome(False, _error(exc.code, exc.detail), redact(f"{name}: {exc.code}"))
    except Exception as exc:  # a failing tool is an answer the model can work with, never a crashed turn
        return ToolOutcome(False, _error(type(exc).__name__, str(exc)), f"{name} failed")
    return ToolOutcome(True, content, summary, result.draft)


# Helpers --------------------------------------------------------------------------------------------------

def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception as exc:  # one unreadable part of the context never fails the whole read
        return {"error": f"{type(exc).__name__}: {redact(str(exc))[:200]}"} if default is None else default


def _allowed(cfg: settings.Config) -> dict[str, dict[str, str]]:
    return {h["agent"]: h for h in designs.allowed_harnesses(cfg)}


def _money(value: Any) -> str:
    return f"${value:.2f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "$?"


def _require_design_id(design_id: str | None) -> str:
    if not designs.valid_design_id(design_id):
        raise ToolError("not_found", f"'{str(design_id)[:40]}' is not a design id; list_designs shows them.")
    return str(design_id)


def _check_figures(check: Any) -> dict[str, Any]:
    figures = designs.check_figures(check if isinstance(check, Mapping) else None)
    return {k: figures.get(k) for k in ("problems", "warnings", "expected_usd", "p90_usd", "checked_at")}


# get_bench_context -----------------------------------------------------------------------------------------

def _get_bench_context(ctx: ToolContext, args: GetBenchContextIn) -> _Result:
    cfg = ctx.cfg
    flags = _safe(ctx.key_flags, {})
    flags = {str(k): bool(v) for k, v in flags.items()} if isinstance(flags, Mapping) else {}
    snapshot = _safe(lambda: datasets.info(cfg.dataset), {})
    snapshot_out = ({k: snapshot.get(k) for k in ("tasks", "downloaded_at", "harbor")}
                    if isinstance(snapshot, Mapping) and snapshot else None)
    sample = list(cfg.tasks or [])
    blocks = [{"block": i // _SAMPLE_BLOCK + 1, "from": i + 1, "to": min(i + _SAMPLE_BLOCK, len(sample))}
              for i in range(0, len(sample), _SAMPLE_BLOCK)]
    readiness_raw = _safe(ctx.readiness, [])
    readiness = [{"name": r.get("name"), "ok": bool(r.get("ok")), "detail": r.get("detail")}
                 for r in readiness_raw if isinstance(r, Mapping)] if isinstance(readiness_raw, list) else []
    docker_row = next((r for r in readiness if r["name"] == "Docker"), None)
    docker = None
    if docker_row is not None:
        capacity = coverage.docker_capacity() if docker_row["ok"] else None
        docker = {"ok": docker_row["ok"], "ncpu": (capacity or {}).get("ncpu"), "mem_gb": (capacity or {}).get("mem_gb")}
    active = _safe(ctx.active_run, {})
    active_out = None
    if isinstance(active, Mapping) and active.get("id"):
        active_out = {"id": active.get("id"), "status": active.get("status"), "spend_usd": active.get("spend_usd"),
                      "budget_usd": active.get("budget_usd"), "design_ref": active.get("design_ref"),
                      "arms": [{"name": a.get("name"), "status": a.get("status"), "trials_done": a.get("trials_done"),
                                "trials_total": a.get("trials_total")}
                               for a in active.get("arms") or [] if isinstance(a, Mapping)]}
    queue = _safe(ctx.queue_snapshot, {})
    queue = queue if isinstance(queue, Mapping) else {}
    runs = _safe(ctx.runs_snapshot, [])
    runs = [r for r in runs if isinstance(r, Mapping)] if isinstance(runs, list) else []
    runs.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    base_arm = next((a for a in cfg.arms if a.name == cfg.baseline_arm), None)
    data = {
        "profile": settings.PROFILE_NAME,
        "dataset": {"name": cfg.dataset, "snapshot": snapshot_out},
        "sample": {"size": len(sample), "blocks": blocks},
        "providers": [{"name": p.name, "enabled": p.enabled, "key_set": flags.get(p.env_key, False),
                       "model_prefix": p.model_prefix} for p in cfg.providers.values()],
        "priced_models": {name: [{"model": m, "input": price.input, "output": price.output,
                                  "cache_read": price.cache_read} for m, price in sorted(models.items())]
                          for name, models in sorted(cfg.pricing.items()) if name in cfg.providers},
        "harnesses": designs.allowed_harnesses(cfg),
        "run_defaults": run_dict(cfg.run),
        "design_limits": {**dataclasses.asdict(cfg.designs), "extra_harnesses": list(cfg.designs.extra_harnesses)},
        "config_arms": [{"name": a.name, "agent": a.agent, "family": a.family or a.agent, "provider": a.provider,
                         "model": a.model, "effort": a.effort, "enabled": a.enabled, "options": a.options}
                        for a in cfg.arms],
        "baseline_arm": cfg.baseline_arm,
        "baseline_family": (base_arm.family or base_arm.agent) if base_arm else None,
        "readiness": readiness, "docker": docker, "active_run": active_out,
        "queue": {"state": queue.get("state"), "pause_reason": queue.get("pause_reason"),
                  "waiting": [{"id": i.get("id"), "design_id": i.get("design_id"), "version": i.get("version"),
                               "title": i.get("title"), "budget_usd": i.get("budget_usd")}
                              for i in queue.get("waiting") or [] if isinstance(i, Mapping)],
                  "global_cap_usd": queue.get("global_cap_usd"), "cap_remaining_usd": queue.get("cap_remaining_usd")},
        "recent_runs": [{"id": r.get("id"), "created_at": r.get("created_at"), "source": r.get("source"),
                         "design_ref": r.get("design_ref"), "status": r.get("status"),
                         "spend_usd": r.get("spend_usd"), "budget_usd": r.get("budget_usd"),
                         "arms": [a.get("name") for a in r.get("arms") or [] if isinstance(a, Mapping)]}
                        for r in runs[:_RECENT_RUNS]],
        "ledger": _safe(ctx.ledger.summary),
        "assistant_spend": _safe(ctx.spend, {}),
    }
    trials = data["ledger"].get("trials") if isinstance(data["ledger"], Mapping) else None
    return _Result(data, f"Read bench context · {settings.PROFILE_NAME}"
                         + (f" · {trials} trials in the ledger" if trials is not None else ""))


# describe_harness -----------------------------------------------------------------------------------------

def _describe_harness(ctx: ToolContext, args: DescribeHarnessIn) -> _Result:
    cfg = ctx.cfg
    allowed = _allowed(cfg)
    harness = allowed.get(args.agent)
    if harness is None:
        raise ToolError("not_allowed", f"{args.agent} is not a harness designs may use. Allowed: "
                                       f"{', '.join(sorted(allowed)) or 'none'}.")
    described = schemas.describe(args.agent)
    props = schemas.agent_schema(args.agent).get("properties", {})
    family = harness["family"]
    dollar = [n for n in harnesses.DOLLAR_CAPS if n in props]
    turn = [n for n in harnesses.TURN_CAPS if n in props]
    misprices = bool(args.provider) and any(harnesses.cap_misprices_for(family, args.provider or "", n)
                                            for n in dollar)
    warnings = []
    if misprices:
        warnings += [f"{n}: {harnesses.MISPRICED_CAP}" for n in dollar
                     if harnesses.cap_misprices_for(family, args.provider or "", n)]
    if args.provider and args.model:
        arm = settings.Arm(name="describe", agent=args.agent, provider=args.provider, model=args.model, family=family)
        warnings += [f"{k}: {v}" for k, v in schemas.model_warnings(arm).items() if k in props]
    if described.get("effort_levels") is None:
        warnings.append("This harness takes no effort setting: use effort inherit or omit on its arms.")
    data = {**described, "family": family, "kind": harness["kind"],
            "caps": {"dollar": dollar, "turn": turn, "dollar_cap_misprices": misprices}, "warnings": warnings}
    return _Result(data, f"Read options · {args.agent}")


# list_tasks -----------------------------------------------------------------------------------------------

def _list_tasks(ctx: ToolContext, args: ListTasksIn) -> _Result:
    cfg = ctx.cfg
    if (args.provider is None) != (args.model is None):
        raise ToolError("bad_request", "Give both provider and model, or neither.")
    meta = estimates.task_meta(cfg)
    sample = list(dict.fromkeys(cfg.tasks or []))
    position = {t: i for i, t in enumerate(sample)}
    names = sample if args.scope == "sample" else sample + sorted(t for t in meta if t not in position)
    if args.difficulty:
        names = [t for t in names if (meta.get(t) or {}).get("difficulty") == args.difficulty]
    counts: dict[str, Counter[str]] | None = None
    if args.provider and args.model:
        counts = {}
        for r in ctx.ledger.rows():
            if r.get("provider") == args.provider and r.get("model") == args.model and r.get("valid_cost"):
                counts.setdefault(str(r.get("task")), Counter())[str(r.get("family"))] += 1
    tasks = []
    for name in names[args.offset: args.offset + args.limit]:
        m = meta.get(name) or {}
        i = position.get(name)
        tasks.append({
            "position": None if i is None else i + 1, "block": None if i is None else i // _SAMPLE_BLOCK + 1,
            "name": name, "difficulty": m.get("difficulty"), "category": m.get("category"),
            "agent_timeout_sec": m.get("agent_timeout_sec"), "cpus": m.get("cpus"), "memory_gb": m.get("memory_gb"),
            "expert_time_estimate_min": m.get("expert_time_estimate_min"),
            "valid_trials_by_family": None if counts is None else dict(sorted((counts.get(name) or {}).items())),
        })
    return _Result({"tasks": tasks, "total": len(names), "offset": args.offset},
                   f"Listed {len(tasks)} of {len(names)} tasks")


# get_coverage and get_cost_stats ---------------------------------------------------------------------------

def _get_coverage(ctx: ToolContext, args: GetCoverageIn) -> _Result:
    group_by = "config" if args.match == "cell" else "family"
    data = coverage.coverage(ctx.cfg, ctx.ledger, args.provider, args.model, args.baseline_family,
                             args.target_pairs, group_by)
    return _Result(data, f"Checked coverage · {args.provider}:{args.model} · {len(data['gaps'])} gaps")


def _get_cost_stats(ctx: ToolContext, args: GetCostStatsIn) -> _Result:
    cells = estimates.cell_stats(ctx.cfg, ctx.ledger.rows(), args.family, args.provider, args.model, args.match,
                                 args.per_task)
    return _Result({"cells": cells, "ledger_fingerprint": ctx.ledger.fingerprint()},
                   f"Read cost stats · {len(cells)} {'configurations' if args.match == 'cell' else 'families'}")


# list_designs ----------------------------------------------------------------------------------------------

_SUMMARY_KEYS = ("id", "title", "status", "author", "latest_version", "archived", "updated_at", "arms", "trials",
                 "problems", "warnings", "expected_usd", "p90_usd", "runs", "conversation_id", "error")
_VERSION_KEYS = ("design_id", "version", "parent_version", "author", "conversation_id", "created_at", "title",
                 "purpose", "rationale", "change_note", "spec", "spec_hash", "status", "approved_at", "runs",
                 "latest_version", "archived", "diff_from_parent")


def _list_designs(ctx: ToolContext, args: ListDesignsIn) -> _Result:
    if args.design_id is None:
        items = [{k: d.get(k) for k in _SUMMARY_KEYS if k in d} for d in ctx.drafts.list(args.include_archived)]
        return _Result({"designs": items}, f"Listed {len(items)} designs")
    design_id = _require_design_id(args.design_id)
    try:
        meta = ctx.drafts.meta(design_id)
        doc = ctx.drafts.version(design_id, args.version)
    except DesignNotFound as exc:
        raise ToolError("not_found", str(exc)) from None
    versions = [{"version": int(n), "author": v.get("author"), "created_at": v.get("created_at"),
                 "change_note": v.get("change_note"), "spec_hash": v.get("spec_hash"), "status": v.get("status"),
                 "runs": v.get("runs") or []}
                for n, v in sorted((meta.get("versions") or {}).items(), key=lambda kv: int(kv[0]))]
    version = {k: doc.get(k) for k in _VERSION_KEYS}
    version["check"] = _check_figures(doc.get("check"))
    data = {"design": {"id": meta.get("id"), "title": meta.get("title"), "status": meta.get("status"),
                       "latest_version": meta.get("latest_version"), "archived": meta.get("archived")},
            "versions": versions, "version": version}
    return _Result(data, f"Read design {args.design_id} v{doc.get('version')}")


# estimate_cost ---------------------------------------------------------------------------------------------

def _plan_arms(plan: Any) -> list[dict[str, Any]]:
    arms = plan.get("arms") if isinstance(plan, Mapping) else None
    return [{"name": a.get("name"), "family": a.get("family"), "trials": a.get("trials"), "effort": a.get("effort"),
             "kwargs": a.get("kwargs"), "problems": a.get("problems"), "warnings": a.get("warnings")}
            for a in arms or [] if isinstance(a, Mapping)]


def _estimate_cost(ctx: ToolContext, args: EstimateCostIn) -> _Result:
    cfg = ctx.cfg
    if args.spec is not None:
        try:
            spec = designs.normalize(cfg, args.spec)
        except ValueError as exc:
            raise ToolError("invalid_design", str(exc)) from None
        try:
            # Reuse resolved as saving would resolve it, so the estimate covers only the trials that would run.
            spec = reuse.materialize(cfg, spec, ctx.ledger.rows)
        except Exception as exc:
            raise ToolError("ledger_unavailable", f"The trials to reuse could not be looked up in the ledger, so "
                                                  f"there is no estimate: {exc}") from None
        label = "the spec"
    else:
        try:
            spec = ctx.drafts.version(_require_design_id(args.design_id), args.version)["spec"]
        except DesignNotFound as exc:
            raise ToolError("not_found", str(exc)) from None
        label = f"{args.design_id}" + (f" v{args.version}" if args.version else "")
    validation = designs.validate(cfg, spec)
    checked = {"status": validation["status"], "problems": validation["problems"], "warnings": validation["warnings"]}
    try:
        rows = ctx.ledger.rows()
    except Exception as exc:
        # Validation still stands; only the figures need the ledger.
        raise ToolError("ledger_unavailable", f"The checks ran ({validation['status']}: "
                                              f"{'; '.join(validation['problems'][:3]) or 'no problems'}), but the "
                                              f"ledger could not be read for an estimate: {exc}") from None
    est = estimates.estimate(cfg, spec, rows, validation.get("plan"))
    total = est["run"]["total"]
    summary = f"Estimated {label} · {_money(total['p50'])} p50 / {_money(total['p90'])} p90"
    if validation["problems"]:
        summary += f" · {len(validation['problems'])} problems"
    return _Result({"validation": checked, "arms": _plan_arms(validation.get("plan")), "estimate": est}, summary)


# suggest_runs ----------------------------------------------------------------------------------------------

def _suggest_runs(ctx: ToolContext, args: SuggestRunsIn) -> _Result:
    data = coverage.plan_suggestions(ctx.cfg, ctx.ledger, args.goal, args.provider, args.model, args.families,
                                     args.baseline_family, args.target_pairs, args.max_budget_usd, args.attempts)
    n = len(data["candidates"])
    return _Result(data, f"Suggested {n} design{'' if n == 1 else 's'} · {args.provider}:{args.model}")


# save_draft_design -----------------------------------------------------------------------------------------

def _save_draft_design(ctx: ToolContext, args: SaveDraftDesignIn) -> _Result:
    if not ctx.saves_left or ctx.saves_left[0] <= 0:
        raise ToolError("save_limit", "The draft limit for this message is used up. Ask the person before saving "
                                      "more drafts.")
    meta_in = {"title": args.title, "purpose": args.purpose, "rationale": args.rationale,
               "change_note": args.change_note}
    try:
        if args.design_id is None:
            doc = ctx.drafts.create_draft(meta_in, args.spec)
        else:
            doc = ctx.drafts.add_draft_version(_require_design_id(args.design_id), int(args.base_version or 0),
                                               meta_in, args.spec)
    except VersionConflict as exc:
        raise ToolError("version_conflict", f"{exc} Read it with list_designs and apply your change again.") from None
    except DesignNotFound as exc:
        raise ToolError("not_found", str(exc)) from None
    except ValueError as exc:
        raise ToolError("invalid_design", str(exc)) from None
    ctx.saves_left[0] -= 1
    cfg = ctx.cfg
    validation = designs.validate(cfg, doc["spec"])
    note = "Draft saved. A person must queue or start it in the Designs view."
    try:
        est = estimates.estimate(cfg, doc["spec"], ctx.ledger.rows(), validation.get("plan"))
    except Exception as exc:  # the draft is saved either way; say why there are no figures
        est = None
        note += f" No estimate: {type(exc).__name__}."
    figures = estimates.summary(est)
    data = {"design_id": doc["design_id"], "version": doc["version"], "spec_hash": doc["spec_hash"],
            "status": "draft",
            "validation": {"status": validation["status"], "problems": validation["problems"],
                           "warnings": validation["warnings"]},
            "estimate": {k: figures[k] for k in ("p50_usd", "p90_usd", "p_within_budget", "wall_minutes_p50")},
            "saves_left": ctx.saves_left[0], "note": note}
    draft = {"design_id": doc["design_id"], "version": doc["version"], "spec_hash": doc["spec_hash"],
             "title": doc.get("title"), "expected_usd": figures["p50_usd"], "p90_usd": figures["p90_usd"],
             "problems": len(validation["problems"])}
    return _Result(data, f"Saved draft {doc.get('title')} v{doc['version']}", draft)


_HANDLERS: dict[str, Callable[[ToolContext, Any], _Result]] = {
    "get_bench_context": _get_bench_context,
    "describe_harness": _describe_harness,
    "list_tasks": _list_tasks,
    "get_coverage": _get_coverage,
    "get_cost_stats": _get_cost_stats,
    "list_designs": _list_designs,
    "estimate_cost": _estimate_cost,
    "suggest_runs": _suggest_runs,
    "save_draft_design": _save_draft_design,
}

# The registry is the safety boundary: refuse to load if any list drifts.
if not (ALLOWED_TOOL_NAMES == frozenset(d["name"] for d in TOOL_DEFS) == frozenset(INPUT_MODELS)
        == frozenset(_HANDLERS)) or len(TOOL_DEFS) != len(ALLOWED_TOOL_NAMES):
    raise RuntimeError("tools.py: TOOL_DEFS, INPUT_MODELS and the handlers name different tools")
