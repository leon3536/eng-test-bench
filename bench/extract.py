"""Reads Harbor trial folders into ledger facts, and finds them on disk.

The ledger stores facts and derives judgements later, so this module turns
one trial's result.json, lock.json and job lock into exact columns: tokens per
model, the exception and its markers, phase timings, the reward, the task's
content digest and the configuration's identity. It also keeps a trimmed,
redacted copy of the raw JSON, so a parser or classifier change can re-derive
every column without a re-run. A trial killed before Harbor wrote its result
gets the usage its harness logged before the kill (see Killed trials).

Pure apart from reading files: no database. Every Harbor file is read with a
single read_bytes(), so the bench never holds a handle that would block
Harbor's own non-atomic write of result.json on Windows.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import zlib
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import yaml

from . import classify, hostload, lookups, pricing, settings
from .fsutil import canonical_json, harbor_jobs
from .harnesses import is_import_path, reports_mispriced_cost

# 5: setup provenance (setup_mode, agent_setup_limit_sec, prebake_*); rederive gives older rows 'stock'.
# 6: cache writes per model (usage_json n_cache_write_5m/1h) and flags_json, both from the harness's own
#    log, which result.json does not hold: rederive re-reads the trial folder where it still exists.
#    Killed trials re-recovered from their logs also gain mini-swe-agent's extra.cost (reported_cost_usd)
#    and terminus-2's per-step model names (reported_as).
# 7: Cline CLI, OpenCode and Goose calls from their own logs. Cline's and OpenCode's replace Harbor's usage,
#    whose adapters leave reasoning tokens out of the output (both) and cache writes out of the prompt
#    (OpenCode); a stored row keeps the replaced usage once its folder is gone. Kilo Code's kilo.txt
#    joined later without a bump: no stored row was Kilo's, so none needs re-deriving.
#    DeepSeek Harness's dsh.jsonl joined the same way.
# 8: host CPU load over the agent's execution (flags_json host_cpu_*, cpu_contended) from the run's
#    hostload.jsonl, so time a busy host took is not counted as the model's.
# 9: answer lookups (flags_json answer_lookup) from the agent's tool calls in its ATIF trajectory, and the
#    verdict by hand from config/answer_lookups.yaml (answer_lookup_verdict); see lookups.py.
# 10: DeepSeek Harness's answer lookups judged from the tool calls and results in its dsh.jsonl, which has
#    no ATIF trajectory: they were a text search left for a person (used null).
# 11: a gateway model's usage under the upstream id its streams carry (providers.<name>.upstream_models:
#    LiteLLM's message_start names Fireworks' accounts/fireworks/models/... id) joins the arm's own model,
#    primary and priced at the gateway name's row, instead of reading as a fallback model left unpriced.
PARSER_VERSION = 11

# Text in a failed trial's message that means the harness never got going.
SETUP_SIGNATURES: tuple[str, ...] = ("No module named", "ModuleNotFoundError", "command not found",
                                     "Failed to start tmux session")
# Text a harness prints when it stops itself on its own cap: marker -> option.
# Extend as other harnesses' cap messages are seen.
CAP_SIGNATURES: dict[str, dict[str, str]] = {
    "claude-code": {"error_max_budget_usd": "max_budget_usd", "error_max_turns": "max_turns"},
}

TRACEBACK_KEEP = 4000   # the end of a traceback says where it failed
HEAD_CHARS = 1000

_FALLBACK_CAUTIONS = ("fallback_model", "base_url", "api_base", "version")
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


@dataclass(frozen=True)
class RunContext:
    """What the run, rather than the trial folder, says about a trial."""
    run_key: str
    profile: str
    run_id: str
    source: Literal["bench", "external"]
    arm: str
    provider: str | None
    model_prefix: str | None
    effort_label: str | None
    dataset: str | None
    local_snapshot: str | None
    family: str | None = None   # the arm's configured family (run.json resolved_arms), if any
    # A gateway arm's base_url as planned, ${VAR}/route (keys.base_url_template). The trial's
    # config holds the address it ran against; this stands in for it, so moving the gateway
    # keeps one configuration and its address stays out of the ledger.
    base_url: str | None = None
    # The upstream id the arm's gateway model streams back as (provider_upstreams): usage
    # under it is the arm's own model's.
    upstream_model: str | None = None


@dataclass(frozen=True)
class TrialFacts:
    """One trials row, before classification. Field names are column names."""
    trial_key: str
    job_id: str
    trial_name: str
    trial_uid: str | None
    run_key: str
    profile: str
    run_id: str
    source: str
    arm: str
    trial_dir: str
    family: str
    implementation: str
    harness_version: str | None
    harness_version_pinned: str | None
    harbor_version: str | None
    provider: str
    model: str
    harbor_model: str
    effort: str | None
    effort_label: str | None
    options_json: str
    cautions_json: str
    config_hash: str
    config_json: str
    config_label: str
    n_concurrent: int | None
    dataset: str | None
    dataset_snapshot: str | None
    task: str
    task_digest: str | None
    task_source_type: str | None
    reward: float | None
    rewards_json: str | None
    passed: int | None
    exception_type: str | None
    exception_head: str | None
    harbor_cause: str | None
    cap_marker: str | None
    setup_marker: str | None
    agent_started: int
    n_input: int | None
    n_cache: int | None
    n_output: int | None
    usage_json: str
    primary_share: float | None
    reported_cost_usd: float | None
    cost_at_ingest_usd: float | None
    pricing_sha_at_ingest: str | None
    flags_json: str     # checks only the harness's log or the prices could make (trial_flags)
    started_at: str | None
    finished_at: str | None
    env_setup_sec: float | None
    agent_setup_sec: float | None
    agent_exec_sec: float | None
    verifier_sec: float | None
    total_sec: float | None
    # Setup provenance (setup_provenance): how the harness got installed. Never identity.
    setup_mode: str | None
    agent_setup_limit_sec: float | None
    prebake_image: str | None
    prebake_image_id: str | None
    prebake_recipe: str | None
    prebake_baked_at: str | None
    prebake_reason: str | None
    prebake_json: str | None
    parser_version: int
    raw_result_z: bytes | None
    raw_lock_z: bytes | None


@dataclass(frozen=True)
class ScanRoot:
    jobs_dir: Path
    profile: str
    runs_dir: Path | None


# Files and paths ---------------------------------------------------------------

def read_json(path: Path) -> Any | None:
    """The parsed file, or None when it is missing, locked, partial or not JSON."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def relative_dir(path: Path, root: Path | None = None) -> str:
    """Relative to the project root with '/' separators, so the ledger survives
    moving the project; absolute posix form for anything outside it."""
    root = Path(root or settings.ROOT)
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(Path(os.path.abspath(root))).as_posix()
    except ValueError:
        return absolute.as_posix()


def list_dirs(path: Path) -> list[Path]:
    with os.scandir(path) as it:
        return sorted(Path(e.path) for e in it if e.is_dir())


def _pure(path: str) -> PurePosixPath | PureWindowsPath:
    # Harbor records Windows paths with backslashes; read them as such anywhere.
    if "\\" in path or re.match(r"^[A-Za-z]:", path):
        return PureWindowsPath(path)
    return PurePosixPath(path)


# Roots and discovery ----------------------------------------------------------

def default_roots(extra: Sequence[Path] = ()) -> list[ScanRoot]:
    """The default profile's jobs, every profiles/<name>/jobs, this process's
    own data folder if it lives elsewhere, and any extra plain Harbor folders."""
    root = settings.ROOT
    roots = [ScanRoot(root / "jobs", "default", root / "runs")]
    profiles = root / "profiles"
    try:
        for p in list_dirs(profiles):
            roots.append(ScanRoot(p / "jobs", p.name, p / "runs"))
    except OSError:
        pass
    data = Path(settings.DATA_DIR)
    known = {os.path.normcase(os.path.abspath(r.jobs_dir)) for r in roots}
    if os.path.normcase(os.path.abspath(data / "jobs")) not in known:
        roots.append(ScanRoot(data / "jobs", settings.PROFILE_NAME, data / "runs"))
    for x in extra:
        roots.append(ScanRoot(Path(x), "external", None))
    return roots


def _config_paths(config_dir: Path) -> list[Path]:
    """Every bench config (profile) in config_dir."""
    try:
        paths = sorted(config_dir.glob("*.yaml"))
    except OSError:
        return []
    return [p for p in paths if not p.name.endswith(".overrides.yaml")
            and p.name not in ("pricing.yaml", "reference.yaml", "overrides.yaml")]


def _provider_specs(paths: Sequence[Path]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    for p in paths:
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError, UnicodeDecodeError):
            continue
        providers = data.get("providers") if isinstance(data, dict) else None
        if not isinstance(providers, dict):
            continue
        for name, spec in providers.items():
            if isinstance(spec, dict):
                yield str(name), spec


def provider_prefixes(config_dir: Path | None = None) -> dict[str, str]:
    """{model_prefix: provider} from every bench config, for jobs that carry
    no run.json: 'fireworks_ai/accounts/fireworks/models/' -> 'fireworks'."""
    out: dict[str, str] = {}
    for name, spec in _provider_specs(_config_paths(Path(config_dir or settings.CONFIG_DIR))):
        prefix = spec.get("model_prefix")
        if isinstance(prefix, str) and prefix:
            out.setdefault(prefix, name)
    return out


# config_dir -> (its configs' (name, mtime, size), provider_upstreams' answer): a context is
# built per arm on every scan, and the files rarely change.
_UPSTREAMS: dict[str, tuple[tuple[tuple[str, int, int], ...], dict[str, dict[str, str]]]] = {}


def provider_upstreams(config_dir: Path | None = None) -> dict[str, dict[str, str]]:
    """{provider: {model: upstream id}} from every bench config's
    upstream_models: the id a gateway model's streamed replies carry. It is
    a fact about the gateway, not a run, so the current files decide it for
    a first parse and a re-derive alike (run.json's copy is not read)."""
    config_dir = Path(config_dir or settings.CONFIG_DIR)
    paths = _config_paths(config_dir)
    stamp = []
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue
        stamp.append((p.name, st.st_mtime_ns, st.st_size))
    hit = _UPSTREAMS.get(str(config_dir))
    if hit is not None and hit[0] == tuple(stamp):
        return hit[1]
    out: dict[str, dict[str, str]] = {}
    for name, spec in _provider_specs(paths):
        declared = spec.get("upstream_models")
        if isinstance(declared, dict):
            for model, upstream in declared.items():
                if isinstance(model, str) and isinstance(upstream, str) and upstream:
                    out.setdefault(name, {}).setdefault(model, upstream)
    _UPSTREAMS[str(config_dir)] = (tuple(stamp), out)
    return out


def upstream_of(provider: str | None, model: str | None) -> str | None:
    """The upstream id this provider's model streams back as, if a config declares one."""
    if not provider or not model:
        return None
    return provider_upstreams().get(provider, {}).get(model)


def _find(items: Any, name: str) -> Mapping[str, Any]:
    for item in items if isinstance(items, (list, tuple)) else []:
        if isinstance(item, Mapping) and item.get("name") == name:
            return item
    return {}


def context_for_bench_arm(run: Mapping[str, Any], arm: str, profile: str) -> RunContext:
    """run is Run.to_dict() or a run.json dict, old (plan/config only) or new
    (with resolved_arms)."""
    run_id = str(run.get("id") or "")
    config = run.get("config") if isinstance(run.get("config"), Mapping) else {}
    plan = run.get("plan") if isinstance(run.get("plan"), Mapping) else {}
    resolved = _find(run.get("resolved_arms"), arm)
    planned = _find(plan.get("arms"), arm)
    configured = _find(config.get("arms"), arm)

    provider = resolved.get("provider") or planned.get("provider") or configured.get("provider")
    providers = config.get("providers") if isinstance(config.get("providers"), Mapping) else {}
    entry = providers.get(provider) if isinstance(provider, str) else None
    model_prefix = entry.get("model_prefix") if isinstance(entry, Mapping) else None
    if model_prefix is None:
        # Derive it from the full and bare model names the run recorded.
        full = resolved.get("harbor_model") or planned.get("model")
        bare = resolved.get("model") or configured.get("model")
        if isinstance(full, str) and isinstance(bare, str) and full.endswith(bare):
            model_prefix = full[: len(full) - len(bare)]

    effort = resolved.get("effort") if resolved else planned.get("effort")
    effort_label = effort.get("label") if isinstance(effort, Mapping) else effort

    spec = run.get("spec") if isinstance(run.get("spec"), Mapping) else {}
    snapshot = run.get("dataset") if isinstance(run.get("dataset"), Mapping) else None
    dataset = config.get("dataset") or spec.get("dataset") or (snapshot or {}).get("dataset")
    local = None
    if snapshot and snapshot.get("dataset") and snapshot.get("downloaded_at"):
        local = f"local:{snapshot['dataset']}@{snapshot['downloaded_at']}"
    family = resolved.get("family") or planned.get("family") or configured.get("family") or None
    planned_url = next((k.get("value") for k in resolved.get("kwargs") or []
                        if isinstance(k, Mapping) and k.get("name") == "base_url"), None)
    bare = resolved.get("model") or configured.get("model")
    return RunContext(
        run_key=f"{profile}/{run_id}", profile=profile, run_id=run_id, source="bench", arm=arm,
        provider=str(provider) if provider else None,
        model_prefix=model_prefix if isinstance(model_prefix, str) else None,
        effort_label=str(effort_label) if effort_label is not None else None,
        dataset=str(dataset) if dataset else None, local_snapshot=local,
        family=str(family) if family else None,
        base_url=planned_url if isinstance(planned_url, str) and planned_url.startswith("${") else None,
        upstream_model=upstream_of(str(provider) if provider else None, bare if isinstance(bare, str) else None),
    )


def context_for_external(job_dir: Path, root: Path | None = None) -> RunContext:
    job_dir = Path(job_dir)
    cfg = read_json(job_dir / "config.json")
    name = cfg.get("job_name") if isinstance(cfg, dict) else None
    return RunContext(
        run_key=f"external/{relative_dir(job_dir, root)}", profile="external", run_id=job_dir.name,
        source="external", arm=str(name or job_dir.name), provider=None, model_prefix=None,
        effort_label=None, dataset=None, local_snapshot=None,
    )


def _is_job(path: Path) -> bool:
    cfg = read_json(path / "config.json")
    return isinstance(cfg, dict) and "job_name" in cfg


def external_terminal(job_dir: Path) -> bool:
    """A plain Harbor job is over once its result.json has finished_at."""
    result = read_json(Path(job_dir) / "result.json")
    return isinstance(result, dict) and bool(result.get("finished_at"))


def external_run_info(job_dir: Path) -> dict[str, Any]:
    """The runs-table fields a plain Harbor job can supply. Its job-level
    result.json uses naive local times, so only the lock's UTC time is used."""
    lock = read_json(Path(job_dir) / "lock.json")
    created = lock.get("created_at") if isinstance(lock, dict) else None
    return {"status": "finished" if external_terminal(job_dir) else "running", "created_at": created}


def discover(roots: Sequence[ScanRoot], skipped: list[dict[str, str]] | None = None,
             root: Path | None = None) -> Iterator[tuple[Path, RunContext, dict | None, bool]]:
    """(job_dir, ctx, run_dict, terminal) for every Harbor job under the roots.

    A bench job is <jobs_dir>/<run_id>/<arm>/ with runs_dir/<run_id>/run.json, or
    under a task race (run.schedule tasks) each <jobs_dir>/<run_id>/<arm>/<task>/;
    it is terminal once run.json leaves 'running'. Anything else is a plain
    Harbor job found by its config.json 'job_name', terminal once its
    result.json has finished_at. Unreadable folders go to `skipped`.
    """
    def skip(path: Path, reason: str) -> None:
        if skipped is not None:
            skipped.append({"path": str(path), "reason": reason})

    seen: set[str] = set()

    def fresh(path: Path) -> bool:
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            return False
        seen.add(key)
        return True

    def external(path: Path, depth: int) -> Iterator[tuple[Path, RunContext, dict | None, bool]]:
        if _is_job(path):
            if fresh(path):
                yield path, context_for_external(path, root), None, external_terminal(path)
            return
        if depth <= 0:
            return
        try:
            children = list_dirs(path)
        except OSError as exc:
            skip(path, f"unreadable folder: {exc}")
            return
        for child in children:
            yield from external(child, depth - 1)

    for scan in roots:
        jobs_dir = Path(scan.jobs_dir)
        if not jobs_dir.is_dir():
            continue
        if scan.runs_dir is None:
            yield from external(jobs_dir, 3)
            continue
        try:
            entries = list_dirs(jobs_dir)
        except OSError as exc:
            skip(jobs_dir, f"unreadable folder: {exc}")
            continue
        for entry in entries:
            run_json = Path(scan.runs_dir) / entry.name / "run.json"
            if not run_json.exists():
                yield from external(entry, 2)
                continue
            run = read_json(run_json)
            if not isinstance(run, dict):
                skip(run_json, "run.json is unreadable or not valid JSON")
                continue
            run.setdefault("id", entry.name)
            terminal = run.get("status") != "running"
            try:
                arms = list_dirs(entry)
            except OSError as exc:
                skip(entry, f"unreadable folder: {exc}")
                continue
            for arm_dir in arms:
                # A hand-edited or foreign run.json must cost only that run, never the rescan.
                try:
                    ctx = context_for_bench_arm(run, arm_dir.name, scan.profile)
                except Exception as exc:
                    skip(run_json, f"run.json has an unexpected shape: {type(exc).__name__}: {exc}")
                    break
                # One Harbor job per arm, or one per task under a task race (jobs/<run>/<arm>/<task>/).
                for job_dir in harbor_jobs(arm_dir):
                    if fresh(job_dir):
                        yield job_dir, ctx, run, terminal


# Models -----------------------------------------------------------------------

def split_model(harbor_model: str, ctx: RunContext, prefixes: Mapping[str, str],
                model_info: Mapping | None) -> tuple[str, str]:
    """(provider, model without the provider prefix)."""
    harbor_model = harbor_model or ""
    if ctx.provider:
        if ctx.model_prefix and harbor_model.startswith(ctx.model_prefix):
            return ctx.provider, harbor_model[len(ctx.model_prefix):]
        if ctx.model_prefix is None:
            for prefix in sorted(prefixes, key=len, reverse=True):
                if prefixes[prefix] == ctx.provider and harbor_model.startswith(prefix):
                    return ctx.provider, harbor_model[len(prefix):]
        return ctx.provider, harbor_model
    for prefix in sorted(prefixes, key=len, reverse=True):
        if prefix and harbor_model.startswith(prefix):
            return prefixes[prefix], harbor_model[len(prefix):]
    if isinstance(model_info, Mapping) and model_info.get("provider") and model_info.get("name"):
        return str(model_info["provider"]), str(model_info["name"])
    if "/" in harbor_model:
        first, rest = harbor_model.split("/", 1)
        return first, rest
    return "unknown", harbor_model


def canonical_model(reported: str, ctx: RunContext, prefixes: Mapping[str, str], model: str | None = None) -> str:
    """A model_usage key without its provider prefix, so pi's
    'openrouter/deepseek/x' and Claude Code's 'deepseek/x' are one model.
    Given the arm's model, the upstream id it streams back as
    (ctx.upstream_model) is that model, so the two names make one entry."""
    key = reported
    if ctx.model_prefix and reported.startswith(ctx.model_prefix):
        key = reported[len(ctx.model_prefix):]
    else:
        for prefix in sorted(prefixes, key=len, reverse=True):
            if prefix and reported.startswith(prefix) and (ctx.provider is None or prefixes[prefix] == ctx.provider):
                key = reported[len(prefix):]
                break
    if model and settings.streamed_as(key, ctx.upstream_model):
        return model
    return key


def _note_upstream(entry: dict[str, Any], reported: str, key: str, ctx: RunContext,
                   prefixes: Mapping[str, str]) -> None:
    """Names, in upstream_id, the upstream id whose tokens canonical_model
    joined to this entry (the arm's model). reported_as keeps the gateway's
    own name whichever came first, so a re-read in another key order (the
    stored result is sorted) stores the same entry."""
    if key != canonical_model(reported, ctx, prefixes):
        entry["upstream_id"] = reported
    elif entry.get("upstream_id") and entry["reported_as"] == entry["upstream_id"]:
        entry["reported_as"] = reported


def is_primary(canonical: str, model: str) -> bool:
    """The configured model itself, perhaps with a date or version tag; never
    a prefix match (glm-5.3-flash is not glm-5.3)."""
    if canonical == model:
        return True
    return settings.is_variant(canonical.split("/")[-1], model.split("/")[-1])


# Configuration identity ---------------------------------------------------------

def _number(value: float) -> int | float:
    return int(value) if float(value).is_integer() and abs(value) < 2 ** 53 else float(value)


def _normalize(value: Any) -> Any:
    """'1' and 1 hash alike: a string that parses as a finite number becomes it."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            n = float(value)
        except ValueError:
            return value
        return _number(n) if math.isfinite(n) else value
    if isinstance(value, (int, float)):
        return _number(value) if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return str(value)


def _redacted_options(kwargs: Any, rx: re.Pattern[str] | None) -> dict[str, Any]:
    """Agent kwargs with every string value redacted; keys and shape kept,
    so a credential passed as an option never reaches a column."""
    if not isinstance(kwargs, Mapping):
        return {}
    return {k: _scrub(v, rx, trim=False, drop_env=False) for k, v in kwargs.items()}


def _label_value(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def config_identity(trial_cfg: Mapping[str, Any], family: str, implementation: str, provider: str,
                    model: str) -> tuple[str, dict, str]:
    """(hash, identity, label). The hash covers what changes what is measured;
    arm name, versions, concurrency, ids, paths and env (but for agent_env's
    model selectors) are left out, so the same settings under two arm names
    pool, and version drift is its own column.
    Option values are redacted first: the identity is stored and exported."""
    agent = trial_cfg.get("agent") if isinstance(trial_cfg.get("agent"), Mapping) else {}
    rx = _secret_regex()
    raw = dict(agent["kwargs"]) if isinstance(agent.get("kwargs"), Mapping) else {}
    # An arm's agent_env reaches Harbor as --ae, so the trial holds it in agent.env.
    # Back among the options, as the planner has it (runs.agent_kwargs): the
    # models Claude Code picks for aliases and subagents change what is measured.
    env = agent.get("env") if isinstance(agent.get("env"), Mapping) else {}
    picked = {k: str(env[k]) for k in sorted(settings.AGENT_ENV_NAMES) if k in env}
    if picked:
        raw[settings.AGENT_ENV_OPTION] = picked
    kwargs = _redacted_options(raw, rx)
    effort = kwargs.pop("reasoning_effort", None)
    effort = None if effort is None else str(effort)
    options = {str(k): _normalize(v) for k, v in sorted(kwargs.items(), key=lambda kv: str(kv[0]))}

    # Setup limits (agent_setup_timeout_multiplier, agent.override_setup_timeout_sec)
    # are left out: a setup limit cannot change a trial that got past setup, and
    # one that did not is infra and already excluded. So the setup safety net a
    # run adds for stock setups (run.setup_timeout_multiplier) keeps its cell.
    # Nothing in `environment` beyond the resource overrides below is read either:
    # its type, import_path, kwargs and extra_docker_compose are how a trial got
    # its container (a pre-baked image or a stock install), not what it measured.
    # tests/test_extract.py pins both, with the five arms' hashes.
    timeouts: dict[str, Any] = {}
    multiplier = trial_cfg.get("timeout_multiplier")
    if multiplier is not None and _normalize(multiplier) != 1:
        timeouts["timeout_multiplier"] = _normalize(multiplier)
    if trial_cfg.get("agent_timeout_multiplier") is not None:
        timeouts["agent_timeout_multiplier"] = _normalize(trial_cfg["agent_timeout_multiplier"])
    for key in ("override_timeout_sec", "max_timeout_sec"):
        if agent.get(key) is not None:
            timeouts[key] = _normalize(agent[key])

    extras: dict[str, Any] = {}
    skills = [str(s) for s in agent.get("skills") or []]
    if skills:
        extras["skills"] = sorted(skills)
    servers = [str(s.get("name")) for s in agent.get("mcp_servers") or [] if isinstance(s, Mapping)]
    if servers:
        extras["mcp_servers"] = sorted(servers)
    instructions = trial_cfg.get("extra_instructions") or []
    if instructions:
        extras["extra_instructions"] = hashlib.sha256(canonical_json(instructions).encode("utf-8")).hexdigest()[:8]
    environment = trial_cfg.get("environment") if isinstance(trial_cfg.get("environment"), Mapping) else {}
    for key in ("override_cpus", "override_memory_mb", "override_storage_mb", "override_gpus"):
        if environment.get(key) is not None:
            extras[key] = _normalize(environment[key])
    if agent.get("base_url"):
        # Traffic to another endpoint is not the same configuration.
        extras["base_url"] = _redact_with(rx, str(agent["base_url"]))

    identity = {"family": family, "implementation": implementation, "provider": provider, "model": model,
                "effort": effort, "options": options, "timeouts": timeouts, "extras": extras}
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()[:16]
    parts = [f"{family}{'' if implementation == 'builtin' else '*'}", f"{provider}:{model}",
             f"effort={effort if effort is not None else 'default'}"]
    for group in (options, timeouts, extras):
        parts += [f"{k}={_label_value(v)}" for k, v in group.items()]
    return digest, identity, " · ".join(parts)


# Tasks and times ------------------------------------------------------------------

def _task_name(task_cfg: Mapping[str, Any]) -> str | None:
    if task_cfg.get("name"):
        return str(task_cfg["name"])
    if task_cfg.get("path"):
        return _pure(str(task_cfg["path"])).name
    return None


def task_identity(result: Mapping | None, lock: Mapping | None, ctx: RunContext,
                  config: Mapping | None = None) -> tuple[str, str | None, str | None, str | None]:
    """(task, digest, source_type, snapshot). The digest is Harbor's hash of
    the task content, so CRLF-damaged and clean copies of a task never pair."""
    result = result or {}
    lock_task = (lock or {}).get("task") if isinstance((lock or {}).get("task"), Mapping) else {}
    trial_cfg = result.get("config") if isinstance(result.get("config"), Mapping) else (config or {})
    task_cfg = trial_cfg.get("task") if isinstance(trial_cfg.get("task"), Mapping) else {}

    task = result.get("task_name") or lock_task.get("name") or _task_name(task_cfg) or "unknown"
    digest = lock_task.get("digest")
    if not digest and result.get("task_checksum"):
        digest = f"dirhash:{result['task_checksum']}"
    source_type = lock_task.get("type")
    if not source_type:
        source_type = ("package" if task_cfg.get("name") else "git" if task_cfg.get("git_url")
                       else "local" if task_cfg.get("path") else None)

    snapshot = None
    if source_type == "git":
        task_id = result.get("task_id") if isinstance(result.get("task_id"), Mapping) else {}
        commit = lock_task.get("git_commit_id") or task_cfg.get("git_commit_id") or task_id.get("git_commit_id")
        snapshot = f"git:{str(commit)[:12]}" if commit else "git:unpinned"
    elif source_type == "package":
        name = task_cfg.get("name") or lock_task.get("name") or task
        ref = task_cfg.get("ref") or (str(digest).split(":", 1)[-1][:12] if digest else None)
        snapshot = f"pkg:{name}@{ref}" if ref else f"pkg:{name}"
    elif source_type == "local":
        if ctx.local_snapshot:
            snapshot = ctx.local_snapshot
        else:
            path = task_cfg.get("path") or lock_task.get("path")
            if path:
                parent = _pure(str(path)).parent.as_posix()
                snapshot = f"path:{hashlib.sha256(parent.encode('utf-8')).hexdigest()[:8]}"
    return str(task), (str(digest) if digest else None), source_type, snapshot


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def utc_text(value: Any) -> str | None:
    """A trial-level time as fixed-width UTC '...Z', so text order is time order."""
    dt = _parse_time(value)
    return dt.strftime(_TIME_FORMAT) if dt else None


def _span(timing: Any) -> float | None:
    if not isinstance(timing, Mapping):
        return None
    start, end = _parse_time(timing.get("started_at")), _parse_time(timing.get("finished_at"))
    return (end - start).total_seconds() if start and end else None


def _steps(result: Mapping) -> list[Mapping]:
    """A multi-step trial's step results. Anything Harbor's model would not
    write there (a file that parses can hold any JSON) reads as none."""
    steps = result.get("step_results")
    return [s for s in steps if isinstance(s, Mapping)] if isinstance(steps, list) else []


def _step_span(result: Mapping, key: str) -> float | None:
    spans = [_span(s.get(key)) for s in _steps(result)]
    spans = [s for s in spans if s is not None]
    return sum(spans) if spans else None


def phase_seconds(result: Mapping) -> dict[str, float | None]:
    """Phase durations from the trial-level UTC times only; the job-level
    result.json uses naive local time and is never read for times."""
    agent_exec = _span(result.get("agent_execution"))
    verifier = _span(result.get("verifier"))
    return {
        "env_setup_sec": _span(result.get("environment_setup")),
        "agent_setup_sec": _span(result.get("agent_setup")),
        "agent_exec_sec": agent_exec if agent_exec is not None else _step_span(result, "agent_execution"),
        "verifier_sec": verifier if verifier is not None else _step_span(result, "verifier"),
        "total_sec": _span(result),
    }


def agent_windows(result: Mapping) -> list[tuple[datetime, datetime]]:
    """The agent execution phase's (start, end) in UTC: one, or one per step."""
    timings = [result.get("agent_execution")]
    if not isinstance(timings[0], Mapping):
        timings = [s.get("agent_execution") for s in _steps(result)]
    out = []
    for timing in timings:
        if isinstance(timing, Mapping):
            start, end = _parse_time(timing.get("started_at")), _parse_time(timing.get("finished_at"))
            if start and end:
                out.append((start, end))
    return out


# Host load --------------------------------------------------------------------

def hostload_path(trial_dir: Path, ctx: RunContext) -> Path | None:
    """runs/<run_id>/hostload.jsonl beside the jobs/<run_id>/... folder a
    bench trial is in, or None for a plain Harbor job."""
    if ctx.source != "bench" or not ctx.run_id:
        return None
    for parent in Path(trial_dir).parents:
        if parent.name == ctx.run_id and parent.parent.name == "jobs":
            return parent.parent.parent / "runs" / ctx.run_id / hostload.HOSTLOAD_FILE
    return None


def trial_hostload(trial_dir: Path, ctx: RunContext) -> list[hostload.Sample] | None:
    """The run's host load samples for a trial, or None when none were kept."""
    path = hostload_path(trial_dir, ctx)
    return hostload.read_samples(path) if path is not None else None


def _host_flags(flags: dict[str, Any], result: Mapping, samples: Sequence[hostload.Sample] | None,
                stored: Mapping[str, Any] | None) -> dict[str, Any]:
    """The flags with the host's CPU load over the agent's execution. With no
    samples there it is unknown and no key is set; a re-derive whose hostload
    file is gone keeps what was stored."""
    out = {k: v for k, v in flags.items() if k not in hostload.HOST_FLAGS}
    found = hostload.contention(samples, agent_windows(result))
    if not found and samples is None and stored is not None:
        found = {k: v for k, v in (stored.get("flags") or {}).items() if k in hostload.HOST_FLAGS}
    out.update(found)
    return out


# Setup provenance -------------------------------------------------------------
#
# How a trial's harness got installed: by Harbor's own install in the trial's
# container (stock), or already, in a pre-baked image that
# bench/environments/prebaked.py chose at environment start and recorded in
# <trial_dir>/prebake.json. Provenance only: config_identity, task_identity,
# report pairing and reuse never read it, so a pre-baked and a stock trial of
# one arm stay one cell.

# bench.environments.prebaked's constants, kept here because that module loads Harbor's Docker environment.
PREBAKED_ENV = "bench.environments.prebaked:PrebakedDockerEnvironment"
PREBAKE_RECORD = "prebake.json"
SETUP_MODES = ("stock", "prebaked", "fallback")
# Harbor 0.25.0's BaseAgent.DEFAULT_SETUP_TIMEOUT_SEC, and the harnesses that set their own.
DEFAULT_SETUP_TIMEOUT_SEC = 360.0
SETUP_TIMEOUT_SEC = {"hermes": 1200.0}
_PREBAKE_TEXT = 300


def read_prebake_record(trial_dir: Path) -> dict | None:
    """The environment class's prebake.json in a trial folder, or None."""
    record = read_json(Path(trial_dir) / PREBAKE_RECORD)
    return record if isinstance(record, dict) else None


def _is_prebaked_env(import_path: Any) -> bool:
    # The bench's class, or the research prototype's (prebake_env:PrebakedDockerEnvironment).
    return isinstance(import_path, str) and import_path.rsplit(":", 1)[-1] == PREBAKED_ENV.rsplit(":", 1)[-1]


def _seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) and number > 0 else None


def agent_setup_limit(trial_cfg: Mapping[str, Any]) -> float | None:
    """The agent setup timeout the trial ran under, as Harbor's Trial computes
    it: agent.override_setup_timeout_sec, else the harness's default, times
    agent_setup_timeout_multiplier, else timeout_multiplier."""
    agent = trial_cfg.get("agent") if isinstance(trial_cfg.get("agent"), Mapping) else {}
    base = _seconds(agent.get("override_setup_timeout_sec"))
    if base is None:
        base = SETUP_TIMEOUT_SEC.get(str(agent.get("name") or ""), DEFAULT_SETUP_TIMEOUT_SEC)
    multiplier = trial_cfg.get("agent_setup_timeout_multiplier")
    if multiplier is None:
        multiplier = trial_cfg.get("timeout_multiplier", 1.0)
    multiplier = _seconds(1.0 if multiplier is None else multiplier)
    return None if multiplier is None else round(base * multiplier, 3)


def _record_text(value: Any) -> str | None:
    return None if value in (None, "") else redact(str(value))[:_PREBAKE_TEXT]


def setup_provenance(trial_cfg: Mapping[str, Any], record: Mapping[str, Any] | None) -> dict[str, Any]:
    """The setup columns of one trial, from its config (result.json's, or
    config.json or lock.json for a killed one) and its prebake.json. setup_mode:
    - 'prebaked': the record says the pre-baked image was used;
    - 'fallback': the record says it was not, so Harbor installed the harness
      on the task image (or, with prebake_required, the trial failed fast);
    - 'stock': the trial did not run through the pre-baked environment class;
    - None: it did, but left no record, which the class writes before any
      container starts (the trial was killed before environment start)."""
    environment = trial_cfg.get("environment") if isinstance(trial_cfg.get("environment"), Mapping) else {}
    out: dict[str, Any] = {"setup_mode": "stock", "agent_setup_limit_sec": agent_setup_limit(trial_cfg),
                           "prebake_image": None, "prebake_image_id": None, "prebake_recipe": None,
                           "prebake_baked_at": None, "prebake_reason": None, "prebake_json": None}
    if isinstance(record, Mapping) and "used" in record:
        baked_at = record.get("baked_at")
        out.update(
            setup_mode="prebaked" if record.get("used") is True else "fallback",
            prebake_image=_record_text(record.get("image")), prebake_image_id=_record_text(record.get("image_id")),
            # What the arm asked for; a used image's own recipe label is checked equal to it.
            prebake_recipe=_record_text(record.get("wanted_recipe") or record.get("recipe")),
            prebake_baked_at=utc_text(baked_at) or _record_text(baked_at),
            prebake_reason=_record_text(record.get("reason")),
            prebake_json=canonical_json(redact_data(dict(record))))
    elif _is_prebaked_env(environment.get("import_path")):
        out["setup_mode"] = None
    return out


# Secrets ----------------------------------------------------------------------

# The CLI never loads .env, so key shapes the bench's providers use are
# masked by pattern too: a Fireworks key or an AWS key id has no sk- prefix.
_PATTERNS = (
    (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), "sk-[redacted]"),
    (re.compile(r"\bfw_[A-Za-z0-9]{16,}"), "fw_[redacted]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA[redacted]"),
    (re.compile(r"Bearer \S+"), "Bearer [redacted]"),
    (re.compile(r"([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)[A-Z0-9_]*)=\S+"), r"\1=[redacted]"),
)
_FALLBACK_SECRET = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)$", re.I)
_secret_cache: dict[tuple[str, ...], re.Pattern[str] | None] = {}


def _is_secret_name(name: str) -> bool:
    try:
        from .keys import _is_secret
    except Exception:  # keys.py mid-edit or broken: stay conservative
        return not name.upper().startswith("GIT_CONFIG_") and bool(_FALLBACK_SECRET.search(name))
    return _is_secret(name)


def _secret_regex() -> re.Pattern[str] | None:
    """One pattern for the values of the secret variables loaded right now."""
    values = tuple(sorted({v for k, v in os.environ.items() if v and len(v) >= 8 and _is_secret_name(k)},
                          key=lambda v: (-len(v), v)))
    if values not in _secret_cache:
        if len(_secret_cache) > 16:
            _secret_cache.clear()
        _secret_cache[values] = re.compile("|".join(map(re.escape, values))) if values else None
    return _secret_cache[values]


def _redact_with(rx: re.Pattern[str] | None, text: str) -> str:
    if not text:
        return text
    if rx is not None:
        text = rx.sub("[redacted]", text)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact(text: str) -> str:
    """Masks the values of loaded secret variables, then sk-, fw_, AKIA,
    Bearer and NAME_API_KEY=/_TOKEN=/_SECRET= patterns."""
    return _redact_with(_secret_regex(), text)


def _scrub(data: Any, rx: re.Pattern[str] | None, *, trim: bool, drop_env: bool = True) -> Any:
    if isinstance(data, Mapping):
        out = {}
        for key, value in data.items():
            if (drop_env and key == "env") or (trim and key == "rollout_details"):
                continue
            if trim and key == "exception_traceback" and isinstance(value, str):
                value = value[-TRACEBACK_KEEP:]
            out[key] = _scrub(value, rx, trim=trim, drop_env=drop_env)
        return out
    if isinstance(data, (list, tuple)):
        return [_scrub(v, rx, trim=trim, drop_env=drop_env) for v in data]
    if isinstance(data, str):
        return _redact_with(rx, data)
    return data


def redact_data(data: Any) -> Any:
    """A JSON-like value with every string redacted and 'env' maps dropped,
    so it can be stored as JSON text that still parses."""
    return _scrub(data, _secret_regex(), trim=False)


def trim_result(result: dict) -> dict:
    """result.json as stored: no rollout_details, no env maps, the last
    4000 characters of each traceback, the whole exception_message (it holds
    the cap markers), every string redacted."""
    return _scrub(result, _secret_regex(), trim=True)


def scrub_lock(lock: dict) -> dict:
    return redact_data(lock)


def compress(data: Any) -> bytes:
    return zlib.compress(json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8"), 6)


def inflate(blob: bytes | None) -> Any | None:
    return json.loads(zlib.decompress(blob).decode("utf-8")) if blob else None


# Trials -------------------------------------------------------------------------

def _contexts(result: Mapping) -> list[Mapping]:
    """Harbor's rule: agent_result when set, else every step's agent_result."""
    agent_result = result.get("agent_result")
    if agent_result is not None:
        return [agent_result] if isinstance(agent_result, Mapping) else []
    return [s["agent_result"] for s in _steps(result) if isinstance(s.get("agent_result"), Mapping)]


def token_totals(result: Mapping) -> tuple[int | None, int | None, int | None, float | None]:
    """Like TrialResult.compute_token_cost_totals: None (not reported) stays
    distinct from 0 (reported as zero)."""
    totals: list[Any] = [None, None, None, None]
    for ctx in _contexts(result):
        for i, key in enumerate(("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd")):
            value = ctx.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[i] = (totals[i] or 0) + value
    return totals[0], totals[1], totals[2], totals[3]


def _exception(result: Mapping) -> Mapping | None:
    exc = result.get("exception_info")
    if isinstance(exc, Mapping) and exc.get("exception_type"):
        return exc
    for step in _steps(result):
        exc = step.get("exception_info")
        if isinstance(exc, Mapping) and exc.get("exception_type"):
            return exc
    return None


def exception_message(result: Mapping) -> str | None:
    exc = _exception(result)
    return str(exc.get("exception_message") or "") if exc else None


def _agent_started(result: Mapping) -> bool:
    timing = result.get("agent_execution")
    if isinstance(timing, Mapping) and timing.get("started_at"):
        return True
    return any(isinstance(s.get("agent_execution"), Mapping) and s["agent_execution"].get("started_at")
               for s in _steps(result))


def _reward(result: Mapping) -> tuple[float | None, dict | None]:
    verifier = result.get("verifier_result") if isinstance(result.get("verifier_result"), Mapping) else {}
    rewards = verifier.get("rewards") if isinstance(verifier.get("rewards"), Mapping) else {}
    if not rewards:
        return None, None
    value = rewards.get("reward", next(iter(rewards.values()), None))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, dict(rewards)
    return float(value), dict(rewards)


def judged_facts(result: Mapping) -> dict[str, Any]:
    """The facts classify reads, from one result.json alone, for views that
    read trial folders directly rather than through the ledger (the per-run
    results). Cap markers are left out: they never change usage_status."""
    n_input, n_cache, n_output, _cost = token_totals(result)
    exc = _exception(result)
    exception_type = str(exc["exception_type"]) if exc else None
    message = str(exc.get("exception_message") or "") if exc else ""
    return {
        # Harbor writes result.json once, at the end, so one that parses is finished.
        "finished_at": result.get("finished_at") or "result.json", "exception_type": exception_type,
        "harbor_cause": classify.harbor_cause(exception_type), "reward": _reward(result)[0],
        "agent_started": int(_agent_started(result)), "n_input": n_input, "n_cache": n_cache, "n_output": n_output,
        "agent_exec_sec": phase_seconds(result)["agent_exec_sec"],
        "setup_marker": _first(SETUP_SIGNATURES, message) if exc else None,
    }


def _usage(result: Mapping, ctx: RunContext, prefixes: Mapping[str, str], model: str,
           totals: tuple[Any, Any, Any, Any]) -> dict[str, dict[str, Any]]:
    """Harbor's per-model usage, made to add up to its totals.

    The adapter sets the totals (mini-swe-agent's from its own log), but
    Harbor sums model_usage from the ATIF steps' metrics
    (trajectory_utils.compute_model_usage), so a call the ATIF conversion
    gives no step metrics is in the totals and under no model.
    mini-swe-agent logs a reply without a tool call (a FormatError) on the
    'user' message that answers it, which Harbor's converter files as an
    observation; priced per model, that trial cost less than it spent.
    The invariant: per-model sums equal the totals Harbor reported. A
    shortfall goes to the one model that reported tokens, else to the
    configured model under whatever name Harbor gave it (or a new entry, as
    totals with no breakdown get), and that entry's 'unattributed' holds
    the totals minus the per-model sums. A surplus is only noted there:
    which model over-reports is unknown, and the larger figure under-counts
    nothing. A total Harbor left out is not reconciled, so with no cache
    total a missing call's input is priced as fresh: an upper bound."""
    usage: dict[str, dict[str, Any]] = {}
    for c in _contexts(result):
        for reported, u in (c.get("model_usage") or {}).items():
            if not isinstance(u, Mapping):
                continue
            key = canonical_model(str(reported), ctx, prefixes, model)
            entry = usage.setdefault(key, {"reported_as": str(reported), "n_input": None, "n_cache": None,
                                           "n_output": None, "reported_cost_usd": None,
                                           "primary": is_primary(key, model), "synthesized": False})
            _note_upstream(entry, str(reported), key, ctx, prefixes)
            for src, dst in (("n_input_tokens", "n_input"), ("n_cache_tokens", "n_cache"),
                             ("n_output_tokens", "n_output"), ("cost_usd", "reported_cost_usd")):
                value = u.get(src)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    entry[dst] = (entry[dst] or 0) + value
    n_input, n_cache, n_output, cost = totals
    if not usage and (n_input is not None or n_output is not None):
        usage[model] = {"reported_as": None, "n_input": n_input, "n_cache": n_cache, "n_output": n_output,
                        "reported_cost_usd": cost, "primary": True, "synthesized": True}
    gap = {key: total - sum(u[key] or 0 for u in usage.values())
           for key, total in zip(("n_input", "n_cache", "n_output"), totals) if total is not None}
    gap = {key: diff for key, diff in gap.items() if diff}
    if not gap or not usage:
        return usage
    used = [k for k, u in usage.items() if u["n_input"] or u["n_output"]]
    primary = [k for k in used if usage[k]["primary"]]
    costs = [usage[k]["reported_cost_usd"] for k in used]
    if cost is not None and costs and None not in costs and cost != math.fsum(costs):
        gap["cost_usd"] = cost - math.fsum(costs)
    entry = usage.setdefault(used[0] if len(used) == 1 else primary[0] if len(primary) == 1 else model, {
        "reported_as": None, "n_input": None, "n_cache": None, "n_output": None, "reported_cost_usd": None,
        "primary": True, "synthesized": True})
    for key, diff in gap.items():
        if diff > 0:
            dst = "reported_cost_usd" if key == "cost_usd" else key
            entry[dst] = (entry[dst] or 0) + diff
    entry["unattributed"] = gap
    return usage


def _primary_share(usage: Mapping[str, Mapping[str, Any]]) -> float | None:
    total = primary = 0
    for u in usage.values():
        tokens = (u.get("n_input") or 0) + (u.get("n_output") or 0)
        total += tokens
        if u.get("primary"):
            primary += tokens
    return primary / total if total else None


# Harness logs ---------------------------------------------------------------------
#
# A trial killed with the harbor process tree (a cancel or budget stop on
# Windows) writes no result.json, so Harbor never totals its usage. The
# harness's own log survives in agent/ (a live bind mount), and holds every
# call that completed before the kill. These recipes mirror the Harbor 0.25.0
# adapters' populate_context_post_run, so a recovered figure is what Harbor
# would have recorded, minus the call in flight: a lower bound.
#
# The same logs are the only record of cache writes: Harbor folds them into
# n_input with no count of their own, and result.json keeps none. So every
# trial, finished or not, takes its 5-minute and 1-hour write counts from the
# rawest record its harness keeps (read_harness_log), and the checks that
# only a log can make (attempts no usage counts, calls never saved) become
# the trial's flags.

# (model as the harness logged it or None, prompt tokens incl. cache, cache reads, output, own cost or None,
#  5-minute cache writes, 1-hour cache writes); a write count is None when the log does not give it.
_Call = tuple[str | None, int, int, int, float | None, int | None, int | None]
# A computed cost this far from the harness's own (as a share of the computed) flags the trial.
REPORTED_COST_TOLERANCE = 0.005
# Cache writes recovered from a call's cost must come out a whole number of tokens within this.
_WRITE_RESIDUAL = 0.01


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _cost_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    """The parseable objects of a JSONL file; a line cut off by the kill is skipped."""
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, Mapping):
            out.append(event)
    return out


def _pi_calls(agent: Path) -> list[_Call] | None:
    """Pi's newest session file: usage on assistant and toolResult messages,
    and on compaction and branch_summary entries (harbor pi.py). cacheWrite
    holds every write, cacheWrite1h the 1-hour ones among them."""
    try:
        files = [(p.stat().st_mtime_ns, p.name, p) for p in (agent / "pi" / "sessions").glob("*.jsonl")]
    except OSError:
        return None
    if not files:
        return None
    calls: list[_Call] = []
    for event in _jsonl(max(files)[2]):
        message = event.get("message")
        if event.get("type") in ("compaction", "branch_summary"):
            usage = event.get("usage")
        elif (event.get("type") == "message" and isinstance(message, Mapping)
              and message.get("role") in ("assistant", "toolResult")):
            usage = message.get("usage")
        else:
            continue
        if not isinstance(usage, Mapping):
            continue
        read, write = _int(usage.get("cacheRead")), _int(usage.get("cacheWrite"))
        hour = min(_int(usage.get("cacheWrite1h")), write)
        cost = usage.get("cost")
        calls.append((None, _int(usage.get("input")) + read + write, read, _int(usage.get("output")),
                      _cost_or_none(cost.get("total") if isinstance(cost, Mapping) else None), write - hour, hour))
    return calls


def _anthropic_writes(usage: Mapping[str, Any]) -> tuple[int, int]:
    """(5-minute, 1-hour) writes from an Anthropic usage object: the
    cache_creation split, else the total, which is 5-minute (the default)."""
    split = usage.get("cache_creation")
    if isinstance(split, Mapping) and ("ephemeral_5m_input_tokens" in split or "ephemeral_1h_input_tokens" in split):
        return _int(split.get("ephemeral_5m_input_tokens")), _int(split.get("ephemeral_1h_input_tokens"))
    return _int(usage.get("cache_creation_input_tokens")), 0


def _claude_code_messages(agent: Path) -> dict[str, tuple[Any, Mapping[str, Any]]] | None:
    """{message id: (model, usage)} from Claude Code's session files and
    subagent transcripts: the last usage of each assistant message id, events
    deduplicated by uuid (harbor claude_code.py). A call is logged one to
    three times, every copy with the same final usage. Every session folder
    is read; Harbor gives up on two."""
    root = agent / "sessions" / "projects"
    try:
        files = sorted(root.rglob("*.jsonl")) if root.is_dir() else []
    except OSError:
        return None
    if not files:
        return None
    seen: set[str] = set()
    last: dict[str, tuple[Any, Mapping[str, Any]]] = {}
    for path in files:
        for event in _jsonl(path):
            uid = event.get("uuid")
            if isinstance(uid, str) and uid:
                if uid in seen:
                    continue
                seen.add(uid)
            message = event.get("message")
            if event.get("type") != "assistant" or not isinstance(message, Mapping):
                continue
            if message.get("id") and isinstance(message.get("usage"), Mapping):
                last[str(message["id"])] = (message.get("model"), message["usage"])
    return last


def _claude_code_calls(agent: Path) -> list[_Call] | None:
    """Each assistant message's final usage: one API call per message id."""
    last = _claude_code_messages(agent)
    if last is None:
        return None
    calls: list[_Call] = []
    for model, u in last.values():
        read = _int(u.get("cache_read_input_tokens"))
        calls.append((str(model) if model else None,
                      _int(u.get("input_tokens")) + read + _int(u.get("cache_creation_input_tokens")),
                      read, _int(u.get("output_tokens")), None, *_anthropic_writes(u)))
    return calls


def _claude_code_flags(agent: Path, calls: Sequence[_Call]) -> dict[str, Any]:
    """Attempts no usage counts. A message's usage.iterations lists each
    attempt of its request with that attempt's model; the top-level usage is
    the attempt that produced the message, so the others (an Opus 5.5 attempt
    its safety classifier declined before a fallback model answered) appear
    nowhere else. Whether a declined attempt is billed depends on why it was
    declined, so they are a flag with their tokens, not part of the cost."""
    models: dict[str, dict[str, int]] = {}
    count = 0
    for model, u in (_claude_code_messages(agent) or {}).values():
        attempts = [a for a in u.get("iterations") or [] if isinstance(a, Mapping)]
        for attempt in attempts[:-1]:  # the last attempt is the top-level usage
            name = str(attempt.get("model") or model or "unknown")
            entry = models.setdefault(name, {"n_input": 0, "n_cache": 0, "n_output": 0, "n_cache_write_5m": 0,
                                             "n_cache_write_1h": 0})
            read, (w5, w1) = _int(attempt.get("cache_read_input_tokens")), _anthropic_writes(attempt)
            entry["n_input"] += _int(attempt.get("input_tokens")) + read + _int(
                attempt.get("cache_creation_input_tokens"))
            entry["n_cache"] += read
            entry["n_output"] += _int(attempt.get("output_tokens"))
            entry["n_cache_write_5m"] += w5
            entry["n_cache_write_1h"] += w1
            count += 1
    return {"declined_attempts": {"attempts": count, "models": models}} if count else {}


def _strands_calls(agent: Path) -> list[_Call] | None:
    """strands-events.jsonl, which the runner flushes per message: each new
    assistant message's metadata.usage (harbor strands_trajectory.py)."""
    path = agent / "strands-events.jsonl"
    if not path.is_file():
        return None
    result = read_json(agent / "strands-result.json")
    observed = result.get("observed_model") if isinstance(result, Mapping) else None
    config = observed.get("config") if isinstance(observed, Mapping) else None
    cache = config.get("cache_config") if isinstance(config, Mapping) else None
    hour = cache is not None and re.search(r"ttl\W+1h", str(cache)) is not None
    calls: list[_Call] = []
    for event in _jsonl(path):
        message = event.get("message")
        if (event.get("kind") != "message" or event.get("is_preexisting_context") is True
                or not isinstance(message, Mapping) or message.get("role") != "assistant"):
            continue
        metadata = message.get("metadata")
        usage = metadata.get("usage") if isinstance(metadata, Mapping) else None
        if not isinstance(usage, Mapping):
            continue
        n_in, n_out = _int(usage.get("inputTokens")), _int(usage.get("outputTokens"))
        read, write = _int(usage.get("cacheReadInputTokens")), _int(usage.get("cacheWriteInputTokens"))
        total = usage.get("totalTokens")
        if total is None or n_in + n_out + read + write == total:
            n_in += read + write  # cache reported apart from input
        # strands drops Anthropic's 5-minute/1-hour split; its cache config says which applies.
        calls.append((None, n_in, read, n_out, None, 0 if hour else write, write if hour else 0))
    return calls


def _mini_swe_agent_calls(agent: Path) -> list[_Call] | None:
    """mini-swe-agent's own trajectory, saved after every step: each
    response's usage, with OpenRouter's billed cost where it gave one."""
    data = read_json(agent / "mini-swe-agent.trajectory.json")
    if not isinstance(data, Mapping):
        return None
    calls: list[_Call] = []
    for message in data.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        extra = message.get("extra") if isinstance(message.get("extra"), Mapping) else {}
        response = extra.get("response") if isinstance(extra.get("response"), Mapping) else {}
        usage = response.get("usage")
        if not usage and message.get("object") == "response":
            usage = message.get("usage")
        if not isinstance(usage, Mapping) or not usage:
            continue
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        details = details if isinstance(details, Mapping) else {}
        cost = usage.get("cost") if usage.get("cost") is not None else extra.get("cost")
        calls.append((None, _int(usage.get("prompt_tokens") or usage.get("input_tokens")),
                      _int(details.get("cached_tokens")),
                      _int(usage.get("completion_tokens") or usage.get("output_tokens")),
                      _cost_or_none(cost), *_litellm_writes(usage, details)))
    return calls


def _litellm_writes(usage: Mapping[str, Any], details: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """LiteLLM keeps Anthropic's write split under prompt_tokens_details and
    the total at the top level; neither is there for a provider without them."""
    split = details.get("cache_creation_token_details")
    if isinstance(split, Mapping) and ("ephemeral_5m_input_tokens" in split or "ephemeral_1h_input_tokens" in split):
        return _int(split.get("ephemeral_5m_input_tokens")), _int(split.get("ephemeral_1h_input_tokens"))
    total = usage.get("cache_creation_input_tokens", details.get("cache_creation_tokens"))
    return (_int(total), 0) if total is not None else (None, None)


_MINI_STEP = re.compile(rb"mini-swe-agent \(step (\d+), \$(\d+(?:\.(\d+))?)\)")


def _mini_swe_agent_flags(agent: Path, calls: Sequence[_Call]) -> dict[str, Any]:
    """Calls that completed but were never saved. mini-swe-agent prints
    'mini-swe-agent (step N, $cost so far)' after each model call and saves
    its trajectory only after the step's command returns, so a timeout during
    a command loses the reply that started it. The printed running cost,
    rounded, bounds what the lost calls cost."""
    try:
        text = (agent / "mini-swe-agent.txt").read_bytes()
    except OSError:
        return {}
    steps = [(int(m.group(1)), float(m.group(2)), len(m.group(3) or b"")) for m in _MINI_STEP.finditer(text)]
    if not steps:
        return {}
    step, shown, places = max(steps)
    missing = step - len(calls)
    if missing <= 0:
        return {}
    costs = [c[4] for c in calls]
    bound = None
    if None not in costs:
        bound = max(shown + 0.5 * 10 ** -places - math.fsum(c for c in costs if c is not None), 0.0)
    return {"unsaved_calls": {"calls": missing, "cost_usd_max": bound, "source": "agent/mini-swe-agent.txt",
                              "bound": "printed running cost"}}


_PI_EVENT = re.compile(r'^\{"type":"message_(start|end)","message":\{"role":"assistant"')


def _pi_flags(agent: Path, calls: Sequence[_Call]) -> dict[str, Any]:
    """A request pi started and never finished: an assistant message_start in
    pi.txt with no message_end, as when a kill lands mid-call. Anthropic may
    bill such a call; nothing records its size, so its bound is the trial's
    costliest completed call."""
    try:
        text = (agent / "pi.txt").read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return {}
    counts = Counter(m.group(1) for m in map(_PI_EVENT.match, text.splitlines()) if m)
    missing = counts["start"] - counts["end"]
    if missing <= 0:
        return {}
    costs = [c[4] for c in calls if c[4] is not None]
    return {"unfinished_calls": {"calls": missing, "cost_usd_max": missing * max(costs) if costs else None,
                                 "source": "agent/pi.txt", "bound": "costliest completed call"}}


def _cline_calls(agent: Path) -> list[_Call] | None:
    """cline.txt, the `cline --json` stream Harbor tees as it runs: one usage
    event per model call. Cline's inputTokens hold the cache reads and writes
    (its own cost formula subtracts both), and its outputTokens leave out the
    reasoning tokens, which it reports apart as reasoningTokenCount and the
    provider bills as output. The session's messages.json, which Harbor sums
    (harbor cline.py), drops reasoningTokenCount. Cline asks for no 1-hour
    cache, so its writes are 5-minute."""
    path = agent / "cline.txt"
    if not path.is_file():
        return None
    calls: list[_Call] = []
    for line in _jsonl(path):
        event = line.get("event")
        if line.get("type") != "agent_event" or not isinstance(event, Mapping) or event.get("type") != "usage":
            continue
        calls.append((None, _int(event.get("inputTokens")), _int(event.get("cacheReadTokens")),
                      _int(event.get("outputTokens")) + _int(event.get("reasoningTokenCount")),
                      _cost_or_none(event.get("cost")), _int(event.get("cacheWriteTokens")), 0))
    return calls


def _step_finish_calls(path: Path) -> list[_Call] | None:
    """A `run --format=json` stream that OpenCode or its fork Kilo writes and
    Harbor tees as it runs: one step_finish per model call. Both split a
    call's usage into fresh input, cache reads, cache writes, output and
    reasoning (session.getUsage, whose output leaves the reasoning out);
    Harbor's adapter adds the reads to the input but not the writes, and
    leaves the reasoning out of the output. Their Anthropic cache points carry
    no ttl, so their writes are 5-minute."""
    if not path.is_file():
        return None
    calls: list[_Call] = []
    for event in _jsonl(path):
        part = event.get("part")
        if event.get("type") != "step_finish" or not isinstance(part, Mapping):
            continue
        tokens = part.get("tokens") if isinstance(part.get("tokens"), Mapping) else {}
        cache = tokens.get("cache") if isinstance(tokens.get("cache"), Mapping) else {}
        read, write = _int(cache.get("read")), _int(cache.get("write"))
        calls.append((None, _int(tokens.get("input")) + read + write, read,
                      _int(tokens.get("output")) + _int(tokens.get("reasoning")), _cost_or_none(part.get("cost")),
                      write, 0))
    return calls


def _opencode_calls(agent: Path) -> list[_Call] | None:
    """opencode.txt, OpenCode's step_finish stream (_step_finish_calls)."""
    return _step_finish_calls(agent / "opencode.txt")


def _kilo_calls(agent: Path) -> list[_Call] | None:
    """kilo.txt, Kilo Code's step_finish stream (_step_finish_calls), which
    bench/agents/kilo.py tees under its own name. Kilo 7.8.18 reports a
    call's cost from its own catalog, or OpenRouter's bill where it has one;
    either may be 0, which leaves the call's own cost unknown."""
    return _step_finish_calls(agent / "kilo.txt")


def _dsh_steps(agent: Path) -> list[Mapping[str, Any]] | None:
    """dsh.jsonl's status events, or None without the file."""
    path = agent / "dsh.jsonl"
    if not path.is_file():
        return None
    return [e for e in _jsonl(path) if e.get("type") == "status"]


def _dsh_calls(agent: Path) -> list[_Call] | None:
    """dsh.jsonl, the `dsh --profile headless --json` stream that
    bench/agents/dsh.py writes: one step_end status event per model call,
    its usage summed over the step's retried attempts. dsh's counts are
    disjoint: inputTokens is uncached input only, the cache reads and writes
    come apart, and outputTokens holds the reasoning (pi-ai counts
    completion_tokens; reasoningTokens is a share of it, not more). dsh
    computes no cost. A step whose attempts did not all report usage has no
    usage at all, and only the top-level agent is in the stream: both are
    left out here (_dsh_flags counts the first)."""
    steps = _dsh_steps(agent)
    if steps is None:
        return None
    calls: list[_Call] = []
    for event in steps:
        usage = event.get("usage")
        if event.get("phase") != "step_end" or not isinstance(usage, Mapping):
            continue
        read, write = _int(usage.get("cacheReadTokens")), _int(usage.get("cacheWriteTokens"))
        calls.append((None, _int(usage.get("inputTokens")) + read + write, read, _int(usage.get("outputTokens")),
                      None, write, 0))
    return calls


def _dsh_flags(agent: Path, calls: Sequence[_Call]) -> dict[str, Any]:
    """Model calls dsh.jsonl does not meter: a step started and never ended
    (a kill mid-call), or one that ended without usage because an attempt
    reported none. dsh gives no cost, so nothing bounds theirs."""
    steps = _dsh_steps(agent) or []
    phases = Counter(e.get("phase") for e in steps)
    unmetered = sum(1 for e in steps if e.get("phase") == "step_end" and not isinstance(e.get("usage"), Mapping))
    missing = max(phases["step_start"] - phases["step_end"], 0) + unmetered
    if missing <= 0:
        return {}
    return {"unfinished_calls": {"calls": missing, "cost_usd_max": None, "source": "agent/dsh.jsonl",
                                 "bound": "none (dsh reports no cost)"}}


def _droid_usage(agent: Path) -> tuple[Mapping[str, Any], bool] | None:
    """(usage, finished) of a Droid trial, or None without its log. Finished:
    the stream's last `completion` (or `result`) event, Droid's own total for
    the run. Otherwise the largest tokenUsage among the session settings
    files bench/agents/droid.py copies out while the run goes: a killed run's
    total up to its last copy, a lower bound."""
    path = agent / "droid.jsonl"
    if not path.is_file():
        return None
    final = None
    for event in _jsonl(path):
        if event.get("type") in ("completion", "result") and isinstance(event.get("usage"), Mapping):
            final = event["usage"]
    if final is not None:
        return final, True
    best: dict[str, Any] = {}
    try:
        files = sorted((agent / "droid-sessions").rglob("*.settings.json"))
    except OSError:
        files = []
    for file in files:
        data = read_json(file)
        usage = (data.get("inclusiveTokenUsage") or data.get("tokenUsage")) if isinstance(data, Mapping) else None
        if isinstance(usage, Mapping):
            mapped = {"input_tokens": usage.get("inputTokens"), "output_tokens": usage.get("outputTokens"),
                      "cache_read_input_tokens": usage.get("cacheReadTokens"),
                      "cache_creation_input_tokens": usage.get("cacheCreationTokens")}
            if sum(_int(v) for v in mapped.values()) > sum(_int(v) for v in best.values()):
                best = mapped
    return best, False


def _droid_calls(agent: Path) -> list[_Call] | None:
    """droid.jsonl, the `droid exec -o stream-json` stream that
    bench/agents/droid.py writes. Droid reports usage once, for the whole
    run, so a trial is one entry. Its counts are disjoint, as in Anthropic's
    usage: input_tokens is uncached input only, and the cache reads and
    writes come apart. Droid computes no dollar cost (factory_credits is 0
    on a custom model)."""
    found = _droid_usage(agent)
    if found is None:
        return None
    usage, _finished = found
    read, write = _int(usage.get("cache_read_input_tokens")), _int(usage.get("cache_creation_input_tokens"))
    prompt, output = _int(usage.get("input_tokens")) + read + write, _int(usage.get("output_tokens"))
    if not prompt and not output:
        return []
    return [(None, prompt, read, output, None, write, 0)]


def _droid_flags(agent: Path, calls: Sequence[_Call]) -> dict[str, Any]:
    """A Droid run that never reached its final usage event (a kill): what
    was counted is a copy from mid-run, so later calls are missing."""
    found = _droid_usage(agent)
    if found is None or found[1]:
        return {}
    return {"unfinished_calls": {"calls": 1, "cost_usd_max": None, "source": "agent/droid.jsonl",
                                 "bound": "none (cut before Droid's final usage; counts are from its last "
                                          "session copy)"}}


_GOOSE_DB = Path("goose") / "xdg-data" / "goose" / "sessions" / "sessions.db"


def _goose_ledger(db: Path) -> list[tuple[Any, ...]] | None:
    """Goose's usage_ledger rows, read from a copy: the trial's database may
    still have its write-ahead log beside it (a killed trial), which only a
    writable connection replays, and the trial folder is never written."""
    with tempfile.TemporaryDirectory(prefix="bench-goose-") as tmp:
        for suffix in ("", "-wal", "-shm"):
            source = db.with_name(db.name + suffix)
            if source.is_file():
                shutil.copyfile(source, Path(tmp) / (db.name + suffix))
        conn = sqlite3.connect(Path(tmp) / db.name)
        try:
            return conn.execute("SELECT model, input_tokens, cache_read_tokens, output_tokens, cost, "
                                "cache_write_tokens FROM usage_ledger ORDER BY id").fetchall()
        except sqlite3.Error:
            return None
        finally:
            conn.close()


def _goose_calls(agent: Path) -> list[_Call] | None:
    """Goose's session database (XDG_DATA_HOME is under agent/, harbor
    goose.py): usage_ledger has a row per model call, compaction and other
    sessions included, written as the call ends, with OpenRouter's billed cost
    where it gave one. Goose's input_tokens hold the cache reads and writes.
    It sends a 1-hour cache_control only when cache_ttl is set, which Harbor
    never does, so its writes are 5-minute."""
    db = agent / _GOOSE_DB
    if not db.is_file():
        return None
    try:
        rows = _goose_ledger(db)
    except (sqlite3.Error, OSError):
        return None
    if rows is None:
        return None
    return [(str(model) if model else None, _int(n_in), _int(read), _int(n_out), _cost_or_none(cost), _int(write), 0)
            for model, n_in, read, n_out, cost, write in rows]


def _goose_complete(agent: Path) -> list[_Call] | None:
    """Without the database (an older Goose): the stream's final `complete`
    event, the totals Harbor records, as one call, for its cache writes."""
    done = [e for e in _jsonl(agent / "goose.txt") if e.get("type") == "complete"]
    if not done:
        return None
    e, write = done[-1], done[-1].get("cache_write_input_tokens")
    return [(None, _int(e.get("input_tokens", e.get("total_tokens"))), _int(e.get("cache_read_input_tokens")),
             _int(e.get("output_tokens")), _cost_or_none(e.get("cost_usd")),
             None if write is None else _int(write), None if write is None else 0)]


def _atif_writes(metrics: Mapping[str, Any]) -> tuple[int | None, int | None]:
    extra = metrics.get("extra")
    write = extra.get("cache_write_input_tokens") if isinstance(extra, Mapping) else None
    return (_int(write), 0) if write is not None else (None, None)


def _final_metrics(data: Any) -> _Call | None:
    metrics = data.get("final_metrics") if isinstance(data, Mapping) else None
    if not isinstance(metrics, Mapping):
        return None
    return (None, _int(metrics.get("total_prompt_tokens")), _int(metrics.get("total_cached_tokens")),
            _int(metrics.get("total_completion_tokens")), _cost_or_none(metrics.get("total_cost_usd")),
            *_atif_writes(metrics))


def _step_calls(data: Mapping[str, Any], main: _Call) -> list[_Call]:
    """The main chat's totals as its steps' calls, plus one for whatever the
    totals hold beyond them (a call that finished but never became a step),
    so cache writes can be recovered call by call. The totals alone when the
    steps do not fit inside them."""
    steps = data.get("steps") if isinstance(data.get("steps"), list) else []
    calls: list[_Call] = []
    for step in steps:
        m = step.get("metrics") if isinstance(step, Mapping) else None
        if not isinstance(m, Mapping):
            continue
        name = step.get("model_name")
        calls.append((str(name) if name else None, _int(m.get("prompt_tokens")), _int(m.get("cached_tokens")),
                      _int(m.get("completion_tokens")), _cost_or_none(m.get("cost_usd")), *_atif_writes(m)))
    if not calls:
        return [main]
    rest = [main[i] - sum(c[i] for c in calls) for i in (1, 2, 3)]  # type: ignore[misc]
    if min(rest) < 0:
        return [main]
    if any(rest):
        costs = [c[4] for c in calls]
        cost = (None if main[4] is None or None in costs
                else _cost_or_none(main[4] - math.fsum(c for c in costs if c is not None)))
        writes = [c[5] for c in calls]
        write = None if main[5] is None or None in writes else main[5] - sum(w for w in writes if w is not None)
        calls.append((None, rest[0], rest[1], rest[2], cost, write, None if write is None else 0))
    return calls


def _atif_calls(agent: Path) -> list[_Call] | None:
    """An ATIF trajectory the harness keeps current, plus its summarization
    subagents' files: final_metrics of each. Terminus-2 (default, non-linear
    history) rewrites trajectory.json after every episode with the main
    chat's totals, which its steps' metrics add up to; its finally block,
    which runs on a timeout or cancel too, rewrites it once more with the
    subagents' totals added in. The subagents' files count only when the
    totals do not already hold them: when unsure they are left out, since an
    undercount is still a lower bound and a double count is not."""
    try:
        summaries = sorted(agent.glob("trajectory.summarization-*.json"))
    except OSError:
        summaries = []
    data = read_json(agent / "trajectory.json")
    main = _final_metrics(data)
    subagents = [c for c in (_final_metrics(read_json(p)) for p in summaries) if c is not None]
    if main is None:
        return subagents or None
    if subagents:
        steps = data.get("steps") if isinstance(data.get("steps"), list) else []
        steps = [s["metrics"] for s in steps if isinstance(s, Mapping) and isinstance(s.get("metrics"), Mapping)]
        beyond = (main[1] - sum(_int(m.get("prompt_tokens")) for m in steps),
                  main[3] - sum(_int(m.get("completion_tokens")) for m in steps))
        if not steps or (beyond[0] >= sum(c[1] for c in subagents) and beyond[1] >= sum(c[3] for c in subagents)):
            subagents = []
    return [*_step_calls(data, main), *subagents]


# First match wins: a harness's own log, then an ATIF trajectory, which for an
# installed harness only Harbor's post-run step writes.
_RECOVERY = (("agent/pi/sessions", _pi_calls), ("agent/sessions/projects", _claude_code_calls),
             ("agent/strands-events.jsonl", _strands_calls),
             ("agent/mini-swe-agent.trajectory.json", _mini_swe_agent_calls), ("agent/cline.txt", _cline_calls),
             ("agent/opencode.txt", _opencode_calls), ("agent/kilo.txt", _kilo_calls),
             ("agent/" + _GOOSE_DB.as_posix(), _goose_calls),
             ("agent/goose.txt", _goose_complete), ("agent/dsh.jsonl", _dsh_calls),
             ("agent/droid.jsonl", _droid_calls),
             ("agent/trajectory.json", _atif_calls))

# Logs whose calls replace Harbor's usage of a finished trial, not just add its
# cache writes: Harbor's adapter sums the same calls short (see each recipe),
# or, for Goose, reads only the main run's final `complete` event, while the
# database holds every call of every session.
_REPLACES_HARBOR = frozenset({"agent/cline.txt", "agent/opencode.txt", "agent/kilo.txt",
                              "agent/" + _GOOSE_DB.as_posix(), "agent/dsh.jsonl", "agent/droid.jsonl"})


# The checks only a harness's own log can make, by where its calls came from.
_FLAG_READERS = {"agent/sessions/projects": _claude_code_flags,
                 "agent/mini-swe-agent.trajectory.json": _mini_swe_agent_flags, "agent/pi/sessions": _pi_flags,
                 "agent/dsh.jsonl": _dsh_flags, "agent/droid.jsonl": _droid_flags}


@dataclass(frozen=True)
class HarnessLog:
    """What a harness's own log says about a trial: where it was read, its
    completed calls, and the flags only the log can raise."""
    where: str
    calls: list[_Call]
    flags: dict[str, Any]

    @property
    def replaces_harbor(self) -> bool:
        """Whether its calls are the trial's usage, rather than Harbor's totals."""
        return self.where in _REPLACES_HARBOR


def _lookup_flags(agent: Path) -> dict[str, Any]:
    """The answer_lookup flag (lookups.scan), {} for none; a scan that fails
    on an unexpected file shape never stops ingestion."""
    try:
        found = lookups.scan(agent)
    except (OSError, ValueError, TypeError, AttributeError, RecursionError):
        return {}
    return {lookups.FLAG: found} if found else {}


def read_harness_log(trial_dir: Path) -> HarnessLog | None:
    """The harness's own usage log in trial_dir/agent, or None when it left
    none. Calls may be an empty list: the harness started its log, but no
    model call completed. Every harness's flags include an answer lookup
    found in its tool calls (lookups.scan)."""
    agent = Path(trial_dir) / "agent"
    if not agent.is_dir():
        return None
    for where, recipe in _RECOVERY:
        try:
            calls = recipe(agent)
        except (OSError, ValueError, TypeError, AttributeError):
            calls = None
        if calls is None:
            continue
        flags: dict[str, Any] = {}
        reader = _FLAG_READERS.get(where)
        if reader is not None:
            try:
                flags = reader(agent, calls)
            except (OSError, ValueError, TypeError, AttributeError):
                flags = {}
        return HarnessLog(where, calls, {**flags, **_lookup_flags(agent)})
    return None


def recover_calls(trial_dir: Path) -> tuple[str, list[_Call]] | None:
    """(where, completed calls) from the usage log the harness left in
    trial_dir/agent, or None when it left none."""
    log = read_harness_log(trial_dir)
    return None if log is None else (log.where, log.calls)


def _recovered_usage(found: tuple[str, list[_Call]], ctx: RunContext, prefixes: Mapping[str, str],
                     model: str) -> dict[str, dict[str, Any]]:
    """The same per-model shape as _usage; every entry says where it came from."""
    where, calls = found
    usage: dict[str, dict[str, Any]] = {}
    for reported, n_in, n_cache, n_out, cost, *_writes in calls:
        key = canonical_model(reported, ctx, prefixes, model) if reported else model
        entry = usage.setdefault(key, {"reported_as": reported, "n_input": 0, "n_cache": 0, "n_output": 0,
                                       "reported_cost_usd": None, "primary": is_primary(key, model),
                                       "synthesized": True, "recovered_from": where})
        if reported:
            _note_upstream(entry, reported, key, ctx, prefixes)
        entry["n_input"] += n_in
        entry["n_cache"] += n_cache
        entry["n_output"] += n_out
        if cost is not None:
            entry["reported_cost_usd"] = (entry["reported_cost_usd"] or 0.0) + cost
    return usage


def _replaced_usage(log: HarnessLog, harbor: Mapping[str, Mapping[str, Any]], ctx: RunContext,
                    prefixes: Mapping[str, str], model: str) -> dict[str, dict[str, Any]]:
    """A finished trial's usage from a log that replaces Harbor's (_REPLACES_HARBOR).
    Each entry keeps what Harbor reported for its model, so the difference stays visible."""
    usage = _recovered_usage((log.where, log.calls), ctx, prefixes, model)
    for key, entry in usage.items():
        theirs = harbor.get(key) or {}
        entry["synthesized"] = False
        entry["harbor_reported"] = {k: theirs.get(k) for k in ("n_input", "n_cache", "n_output")}
    return usage


def _was_replaced(usage: Any) -> bool:
    return isinstance(usage, Mapping) and any(isinstance(u, Mapping) and "harbor_reported" in u
                                              for u in usage.values())


def _usage_totals(usage: Mapping[str, Mapping[str, Any]], harbor_cost: float | None
                  ) -> tuple[int, int, int, float | None]:
    """(input, cache, output, cost) over a replaced usage: the harness's own
    cost where every model has one, else what Harbor reported."""
    costs = [u.get("reported_cost_usd") for u in usage.values()]
    cost = math.fsum(c for c in costs if c is not None) if costs and None not in costs else harbor_cost
    return (sum(_int(u.get("n_input")) for u in usage.values()), sum(_int(u.get("n_cache")) for u in usage.values()),
            sum(_int(u.get("n_output")) for u in usage.values()), cost)


def _same_name(reported: str, key: str) -> bool:
    a, b = str(reported).split("/")[-1], str(key).split("/")[-1]
    return a == b or settings.is_variant(a, b) or settings.is_variant(b, a)


def _entry_key(usage: Mapping[str, Mapping[str, Any]], reported: str | None, model: str,
               upstream: str | None = None) -> str | None:
    """The usage entry a logged call belongs to: its model's (the arm's for
    the upstream id it streams back as), else (a log without model names)
    the one model that used tokens, the configured one among several, or none."""
    if reported and model in usage and settings.streamed_as(reported, upstream):
        return model
    if reported:
        for key in usage:
            if _same_name(reported, key):
                return key
    used = [k for k, u in usage.items() if u.get("n_input") or u.get("n_output")]
    if len(used) == 1:
        return used[0]
    primary = [k for k in used if usage[k].get("primary")]
    if len(primary) == 1:
        return primary[0]
    return model if model in usage else None


def _writes_priced(price: settings.Price | None) -> bool:
    return price is not None and (price.cache_write_5m is not None or price.cache_write_1h is not None)


def _writes_from_cost(call: _Call, price: settings.Price | None) -> tuple[int, int] | None:
    """A call's 5-minute writes from its own cost, where the harness priced it
    at this table's rates (LiteLLM does for Anthropic; terminus-2 keeps no
    write count): the cost left over once fresh input, reads and output are
    priced, over the write premium. Only a whole number of tokens that fits
    in the prompt is believed; anything else means the rates differ."""
    _model, prompt, read, output, cost = call[:5]
    if cost is None or price is None or price.cache_write_5m is None or price.cache_write_5m <= price.input:
        return None
    read_rate = price.input if price.cache_read is None else price.cache_read
    base = (prompt - read) * price.input + read * read_rate + output * price.output
    writes = (cost * 1e6 - base) / (price.cache_write_5m - price.input)
    whole = round(writes)
    if abs(writes - whole) > _WRITE_RESIDUAL or not 0 <= whole <= prompt - read:
        return None
    return whole, 0


def _gap_call(usage: Mapping[str, Mapping[str, Any]], calls: Sequence[_Call], cost: float | None) -> _Call | None:
    """What Harbor's totals hold beyond the log's calls, as one call: a call
    that finished but that the log never recorded."""
    rest = [sum(u.get(k) or 0 for u in usage.values()) - sum(c[i] for c in calls)  # type: ignore[misc]
            for i, k in ((1, "n_input"), (2, "n_cache"), (3, "n_output"))]
    if min(rest) < 0 or not any(rest):
        return None
    costs = [c[4] for c in calls]
    own = None if cost is None or None in costs else _cost_or_none(cost - math.fsum(c for c in costs if c is not None))
    return (None, rest[0], rest[1], rest[2], own, None, None)


def assign_cache_writes(usage: dict[str, dict[str, Any]], calls: Sequence[_Call] | None, provider: str, model: str,
                        table: Mapping[str, Mapping[str, settings.Price]], *, cost: float | None = None,
                        missing: str = "no harness log", upstream: str | None = None) -> dict[str, Any]:
    """Sets n_cache_write_5m and n_cache_write_1h on the usage entries the
    log's calls belong to, and returns the flags this raises. A call's writes
    come from the log, else from its own cost (_writes_from_cost); Harbor's
    totals beyond the log (a call the log never recorded) are one more call,
    costed from the harness's total `cost`. An entry with a call whose writes
    stay unknown keeps the known ones; an entry the log names none for, on a
    model whose price has write rates, is flagged cache_writes_unknown with
    the reason (`missing` when there is no log at all). upstream is the id
    the arm's model streams back as (RunContext.upstream_model)."""
    calls = list(calls or [])
    if calls:
        gap = _gap_call(usage, calls, cost)
        if gap is not None:
            calls.append(gap)
    sums: dict[str, list[int]] = {}
    unknown = 0
    for call in calls:
        key = _entry_key(usage, call[0], model, upstream)
        price = pricing.lookup(table, provider, key) if key is not None else None
        writes = (call[5], call[6] or 0) if call[5] is not None else _writes_from_cost(call, price)
        if key is None or writes is None:
            if call[1] > call[2] and (key is None or _writes_priced(price)):
                unknown += 1
            continue
        entry_sums = sums.setdefault(key, [0, 0])
        entry_sums[0] += writes[0]
        entry_sums[1] += writes[1]
    flags: dict[str, Any] = {}
    for key, (w5, w1) in sums.items():
        entry = usage[key]
        if w5 + w1 > (entry.get("n_input") or 0) - (entry.get("n_cache") or 0):
            # The log and Harbor's totals disagree; writes beyond the prompt would price tokens never sent.
            flags.setdefault("cache_writes_exceed_prompt", []).append(key)
            continue
        entry["n_cache_write_5m"], entry["n_cache_write_1h"] = w5, w1
    silent = sorted(k for k, u in usage.items() if "n_cache_write_5m" not in u
                    and (u.get("n_input") or 0) > (u.get("n_cache") or 0)
                    and _writes_priced(pricing.lookup(table, provider, k)))
    if unknown or silent:
        flags["cache_writes_unknown"] = {"calls": unknown, "models": silent,
                                         "reason": missing if not calls else "not in the harness log"}
    return flags


def _stored_writes(usage: dict[str, dict[str, Any]], stored: Mapping[str, Any]) -> None:
    """The write counts an earlier parse stored, for a trial whose folder is gone."""
    for key, entry in usage.items():
        old = stored.get(key)
        if isinstance(old, Mapping) and "n_cache_write_5m" in old:
            entry["n_cache_write_5m"] = _int(old.get("n_cache_write_5m"))
            entry["n_cache_write_1h"] = _int(old.get("n_cache_write_1h"))


def trial_flags(flags: dict[str, Any], usage: Mapping[str, Mapping[str, Any]], provider: str, family: str,
                table: Mapping[str, Mapping[str, settings.Price]], reported: float | None,
                computed: float | None, model: str | None = None) -> dict[str, Any]:
    """The flags with what the prices add: each declined attempt's cost at
    today's rates (an upper bound: whether it was billed is unknown), and a
    harness-reported cost more than REPORTED_COST_TOLERANCE from the computed
    one, which is how a call the bench cannot see shows (a refused request
    only Claude Code's own total counts), and how price drift shows."""
    declined = flags.get("declined_attempts")
    if isinstance(declined, dict):
        total: float | None = 0.0
        for name, u in (declined.get("models") or {}).items():
            price = pricing.lookup(table, provider, name)
            if price is None:
                total = None
                break
            total += pricing.unit_cost(price, u.get("n_input") or 0, u.get("n_cache") or 0,  # type: ignore[operator]
                                       u.get("n_output") or 0, *pricing.cache_writes(u))
        declined["cost_usd_max"] = total
    flags.pop("reported_cost_mismatch", None)
    if (reported is not None and reported > 0 and computed is not None and computed > 0
            and not reports_mispriced_cost(family, provider, model)):
        diff = (reported - computed) / computed
        if abs(diff) > REPORTED_COST_TOLERANCE:
            flags["reported_cost_mismatch"] = {"reported_usd": reported, "computed_usd": computed,
                                               "diff_pct": round(100 * diff, 3)}
    return flags


def _cautions(agent: Mapping[str, Any], kwargs: Mapping[str, Any]) -> list[str]:
    try:
        from .schemas import CAUTIONS
        names = set(CAUTIONS)
    except Exception:  # schemas.py mid-edit: the options that change what is measured
        names = set(_FALLBACK_CAUTIONS)
    found = {k for k in kwargs if k in names and k != "version"}
    if agent.get("base_url"):
        found.add("base_url")
    return sorted(found)


def _implementation(agent: Mapping[str, Any]) -> str:
    if agent.get("import_path"):
        return str(agent["import_path"])
    name = agent.get("name")
    if isinstance(name, str) and is_import_path(name):
        return name
    return "builtin"


def _first(markers: Sequence[str], text: str) -> str | None:
    return next((m for m in markers if m in text), None)


def _priced(usage: dict[str, dict[str, Any]], provider: str, model: str, family: str,
            log: HarnessLog | None, stored: Mapping[str, Any] | None,
            reported: float | None, upstream: str | None = None) -> tuple[float | None, str | None, dict[str, Any]]:
    """(cost, pricing sha, flags) once the usage carries its cache writes:
    from the log, else (a re-derive whose trial folder is gone) as stored."""
    try:
        table = pricing.load_table()
    except Exception:  # an invalid pricing.yaml must not stop ingestion
        table = None
    flags: dict[str, Any] = dict(log.flags) if log is not None else {}
    if log is None and stored is not None:
        _stored_writes(usage, stored.get("usage") or {})
        flags = {k: v for k, v in (stored.get("flags") or {}).items() if k != "cache_writes_unknown"}
        flags["trial_folder_missing"] = True
    flags.update(assign_cache_writes(usage, log.calls if log is not None else None, provider, model, table or {},
                                     cost=reported, missing="trial folder gone" if stored is not None
                                     else "no harness log", upstream=upstream))
    if table is None:
        return None, None, flags
    cost, _unpriced = pricing.price_usage(usage, provider, table)
    return cost, pricing.table_sha(), trial_flags(flags, usage, provider, family, table, reported, cost,
                                                         model)


def _build(*, result: Mapping | None, trial_cfg: Mapping, lock: Mapping | None, job_lock: Mapping | None,
           trial_dir: str, ctx: RunContext, prefixes: Mapping[str, str], job_id: str | None,
           log: HarnessLog | None = None, prebake: Mapping[str, Any] | None = None,
           stored: Mapping[str, Any] | None = None,
           host: Sequence[hostload.Sample] | None = None) -> TrialFacts:
    """Facts from a parsed result.json (finished) or a trial config.json
    (result None: killed before finishing). The harness's own log gives a
    killed trial its usage, and every trial its cache writes and flags;
    `stored` ({"usage", "flags"} as stored before) stands in for the log of
    a finished trial whose folder is gone. prebake is the trial's prebake.json;
    host the run's host load samples (hostload.jsonl), None for none kept."""
    agent = trial_cfg.get("agent") if isinstance(trial_cfg.get("agent"), Mapping) else {}
    kwargs = agent.get("kwargs") if isinstance(agent.get("kwargs"), Mapping) else {}
    if ctx.base_url and kwargs.get("base_url"):
        kwargs = {**kwargs, "base_url": ctx.base_url}
        agent = {**agent, "kwargs": kwargs}
        trial_cfg = {**trial_cfg, "agent": agent}
    info = (result or {}).get("agent_info") if isinstance((result or {}).get("agent_info"), Mapping) else {}
    implementation = _implementation(agent)
    agent_name = agent.get("name") if isinstance(agent.get("name"), str) and not is_import_path(agent["name"]) else None
    family = ctx.family or info.get("name") or agent_name or implementation
    harbor_model = str(agent.get("model_name") or "")
    provider, model = split_model(harbor_model, ctx, prefixes, info.get("model_info"))
    config_hash, identity, label = config_identity(trial_cfg, family, implementation, provider, model)
    task, digest, source_type, snapshot = task_identity(result, lock, ctx, trial_cfg)
    options = {k: v for k, v in _redacted_options(kwargs, _secret_regex()).items() if k != "reasoning_effort"}
    effort = kwargs.get("reasoning_effort")
    version = info.get("version")
    pinned = kwargs.get("version")
    job_lock = job_lock or {}
    harbor = job_lock.get("harbor") if isinstance(job_lock.get("harbor"), Mapping) else {}
    n_concurrent = job_lock.get("n_concurrent_trials")
    trial_name = str((result or {}).get("trial_name") or trial_cfg.get("trial_name") or Path(trial_dir).name)
    if not (job_id or trial_cfg.get("job_id")):
        # Older Harbor wrote no job_id; the job folder stands in for it.
        parent = str(PurePosixPath(trial_dir).parent)
        job_id = f"dir-{hashlib.sha256(parent.encode('utf-8')).hexdigest()[:12]}"
    job_id = str(job_id or trial_cfg.get("job_id"))
    if ctx.source == "bench":
        dataset = ctx.dataset
    else:
        lock_task = (lock or {}).get("task") if isinstance((lock or {}).get("task"), Mapping) else {}
        dataset = lock_task.get("source") or (result or {}).get("source")

    common = dict(
        trial_key=f"{job_id}/{trial_name}", job_id=job_id, trial_name=trial_name,
        run_key=ctx.run_key, profile=ctx.profile, run_id=ctx.run_id, source=ctx.source, arm=ctx.arm,
        trial_dir=trial_dir, family=str(family), implementation=implementation,
        harness_version=None if version in (None, "", "unknown") else str(version),
        harness_version_pinned=None if pinned in (None, "") else str(pinned),
        harbor_version=str(harbor["version"]) if harbor.get("version") else None,
        provider=provider, model=model, harbor_model=harbor_model,
        effort=None if effort is None else str(effort), effort_label=ctx.effort_label,
        options_json=canonical_json(options), cautions_json=canonical_json(_cautions(agent, kwargs)),
        config_hash=config_hash, config_json=canonical_json(identity), config_label=label,
        n_concurrent=n_concurrent if isinstance(n_concurrent, int) else None,
        dataset=str(dataset) if dataset else None, dataset_snapshot=snapshot,
        task=task, task_digest=digest, task_source_type=source_type,
        **setup_provenance(trial_cfg, prebake),
        parser_version=PARSER_VERSION,
        raw_lock_z=compress(scrub_lock(dict(lock))) if isinstance(lock, Mapping) else None,
    )
    if result is None:
        usage = _recovered_usage((log.where, log.calls), ctx, prefixes, model) if log else {}
        sums = [None if not usage else sum(u[k] for u in usage.values()) for k in ("n_input", "n_cache", "n_output")]
        costs = [u["reported_cost_usd"] for u in usage.values() if u["reported_cost_usd"] is not None]
        reported = math.fsum(costs) if costs else None
        cost, pricing_sha, flags = None, None, dict(log.flags) if log else {}
        if usage:
            cost, pricing_sha, flags = _priced(usage, provider, model, str(family), log, None, reported,
                                               ctx.upstream_model)
        flags = lookups.with_verdict(flags, common["trial_key"])
        return TrialFacts(
            **common, trial_uid=None, reward=None, rewards_json=None, passed=None, exception_type=None,
            exception_head=None, harbor_cause=None, cap_marker=None, setup_marker=None,
            agent_started=int(log is not None),  # the harness began its log
            n_input=sums[0], n_cache=sums[1], n_output=sums[2], usage_json=canonical_json(usage),
            primary_share=_primary_share(usage), reported_cost_usd=reported,
            cost_at_ingest_usd=cost, pricing_sha_at_ingest=pricing_sha, flags_json=canonical_json(flags),
            started_at=None, finished_at=None, env_setup_sec=None, agent_setup_sec=None,
            agent_exec_sec=None, verifier_sec=None, total_sec=None, raw_result_z=None,
        )

    totals = token_totals(result)
    usage = _usage(result, ctx, prefixes, model, totals)
    if log is not None and log.replaces_harbor and log.calls:
        usage = _replaced_usage(log, usage, ctx, prefixes, model)
        totals = _usage_totals(usage, totals[3])
    elif log is None and stored is not None and _was_replaced(stored.get("usage")):
        usage = json.loads(json.dumps(stored["usage"]))
        totals = _usage_totals(usage, totals[3])
    cost, pricing_sha, flags = None, None, dict(log.flags) if log is not None else {}
    if usage:
        cost, pricing_sha, flags = _priced(usage, provider, model, str(family), log, stored, totals[3],
                                           ctx.upstream_model)
    flags = _host_flags(flags, result, host, stored)
    if log is None and stored is not None and not usage and lookups.FLAG in (stored.get("flags") or {}):
        flags[lookups.FLAG] = stored["flags"][lookups.FLAG]  # _priced keeps it when there is usage
    flags = lookups.with_verdict(flags, common["trial_key"])
    reward, rewards = _reward(result)
    exc = _exception(result)
    exception_type = str(exc["exception_type"]) if exc else None
    message = str(exc.get("exception_message") or "") if exc else ""
    phases = phase_seconds(result)
    return TrialFacts(
        **common,
        trial_uid=str(result["id"]) if result.get("id") else None,
        reward=reward, rewards_json=canonical_json(rewards) if rewards is not None else None,
        passed=None if reward is None else int(reward >= 1.0),
        exception_type=exception_type,
        exception_head=redact(message)[:HEAD_CHARS] if exc else None,
        harbor_cause=classify.harbor_cause(exception_type),
        # Markers are searched in the whole message: Claude Code's cap error
        # sits past character 900 of a long command echo.
        cap_marker=_first(tuple(CAP_SIGNATURES.get(str(family), {})), message) if exc else None,
        setup_marker=_first(SETUP_SIGNATURES, message) if exc else None,
        agent_started=int(_agent_started(result)),
        n_input=totals[0], n_cache=totals[1], n_output=totals[2],
        usage_json=canonical_json(usage), primary_share=_primary_share(usage),
        reported_cost_usd=totals[3], cost_at_ingest_usd=cost, pricing_sha_at_ingest=pricing_sha,
        flags_json=canonical_json(flags),
        started_at=utc_text(result.get("started_at")), finished_at=utc_text(result.get("finished_at")),
        **phases,
        raw_result_z=compress(trim_result(dict(result))),
    )


def facts_from_result(result: Mapping, lock: Mapping | None, job_lock: Mapping | None, *, trial_dir: str,
                      ctx: RunContext, prefixes: Mapping[str, str],
                      prebake: Mapping[str, Any] | None = None, log: HarnessLog | None = None,
                      stored: Mapping[str, Any] | None = None,
                      host: Sequence[hostload.Sample] | None = None) -> TrialFacts:
    """Facts from an already-parsed result.json and prebake.json; the
    ledger's rederive uses this on the stored copies, with the trial
    folder's harness log, or, once the folder is gone, the stored usage
    and flags ({"usage", "flags"}) for what only the log held. host is the
    run's host load samples (trial_hostload)."""
    trial_cfg = result.get("config") if isinstance(result.get("config"), Mapping) else {}
    return _build(result=result, trial_cfg=trial_cfg, lock=lock, job_lock=job_lock, trial_dir=trial_dir,
                  ctx=ctx, prefixes=prefixes, job_id=trial_cfg.get("job_id"), prebake=prebake, log=log,
                  stored=stored, host=host)


def parse_trial(trial_dir: Path, ctx: RunContext, prefixes: Mapping[str, str], *, root: Path | None = None,
                job_lock: Mapping | None = None) -> TrialFacts | None:
    """None until result.json is complete: missing, locked, partial or without
    finished_at (Harbor writes it once, non-atomically, at the end)."""
    trial_dir = Path(trial_dir)
    result = read_json(trial_dir / "result.json")
    if not isinstance(result, dict) or not result.get("finished_at"):
        return None
    lock = read_json(trial_dir / "lock.json")
    if job_lock is None:
        job_lock = read_json(trial_dir.parent / "lock.json")
    return facts_from_result(result, lock if isinstance(lock, dict) else None,
                             job_lock if isinstance(job_lock, dict) else None,
                             trial_dir=relative_dir(trial_dir, root), ctx=ctx, prefixes=prefixes,
                             prebake=read_prebake_record(trial_dir), log=read_harness_log(trial_dir),
                             host=trial_hostload(trial_dir, ctx))


def parse_incomplete(trial_dir: Path, ctx: RunContext, prefixes: Mapping[str, str], *, root: Path | None = None,
                     job_lock: Mapping | None = None) -> TrialFacts | None:
    """Facts for a trial that can no longer finish, from its config.json and
    the completed calls in its harness's own log."""
    trial_dir = Path(trial_dir)
    cfg = read_json(trial_dir / "config.json")
    if not isinstance(cfg, dict) or not isinstance(cfg.get("agent"), Mapping):
        return None
    lock = read_json(trial_dir / "lock.json")
    if job_lock is None:
        job_lock = read_json(trial_dir.parent / "lock.json")
    return _build(result=None, trial_cfg=cfg, lock=lock if isinstance(lock, dict) else None,
                  job_lock=job_lock if isinstance(job_lock, dict) else None,
                  trial_dir=relative_dir(trial_dir, root), ctx=ctx, prefixes=prefixes, job_id=cfg.get("job_id"),
                  log=read_harness_log(trial_dir), prebake=read_prebake_record(trial_dir))
