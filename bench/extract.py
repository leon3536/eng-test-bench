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
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import yaml

from . import classify, pricing, settings
from .fsutil import canonical_json
from .harnesses import is_import_path

PARSER_VERSION = 4

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
    started_at: str | None
    finished_at: str | None
    env_setup_sec: float | None
    agent_setup_sec: float | None
    agent_exec_sec: float | None
    verifier_sec: float | None
    total_sec: float | None
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


def provider_prefixes(config_dir: Path | None = None) -> dict[str, str]:
    """{model_prefix: provider} from every bench config, for jobs that carry
    no run.json: 'fireworks_ai/accounts/fireworks/models/' -> 'fireworks'."""
    config_dir = Path(config_dir or settings.CONFIG_DIR)
    out: dict[str, str] = {}
    try:
        paths = sorted(config_dir.glob("*.yaml"))
    except OSError:
        return out
    for p in paths:
        if p.name.endswith(".overrides.yaml") or p.name in ("pricing.yaml", "reference.yaml", "overrides.yaml"):
            continue
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError, UnicodeDecodeError):
            continue
        providers = data.get("providers") if isinstance(data, dict) else None
        if not isinstance(providers, dict):
            continue
        for name, spec in providers.items():
            prefix = spec.get("model_prefix") if isinstance(spec, dict) else None
            if isinstance(prefix, str) and prefix:
                out.setdefault(prefix, str(name))
    return out


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
    return RunContext(
        run_key=f"{profile}/{run_id}", profile=profile, run_id=run_id, source="bench", arm=arm,
        provider=str(provider) if provider else None,
        model_prefix=model_prefix if isinstance(model_prefix, str) else None,
        effort_label=str(effort_label) if effort_label is not None else None,
        dataset=str(dataset) if dataset else None, local_snapshot=local,
        family=str(family) if family else None,
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

    A bench job is <jobs_dir>/<run_id>/<arm>/ with runs_dir/<run_id>/run.json;
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
                if fresh(arm_dir):
                    yield arm_dir, ctx, run, terminal


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


def canonical_model(reported: str, ctx: RunContext, prefixes: Mapping[str, str]) -> str:
    """A model_usage key without its provider prefix, so pi's
    'openrouter/deepseek/x' and Claude Code's 'deepseek/x' are one model."""
    if ctx.model_prefix and reported.startswith(ctx.model_prefix):
        return reported[len(ctx.model_prefix):]
    for prefix in sorted(prefixes, key=len, reverse=True):
        if prefix and reported.startswith(prefix) and (ctx.provider is None or prefixes[prefix] == ctx.provider):
            return reported[len(prefix):]
    return reported


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
    arm name, versions, concurrency, ids, paths and env are left out, so the
    same settings under two arm names pool, and version drift is its own column.
    Option values are redacted first: the identity is stored and exported."""
    agent = trial_cfg.get("agent") if isinstance(trial_cfg.get("agent"), Mapping) else {}
    rx = _secret_regex()
    kwargs = _redacted_options(agent.get("kwargs"), rx)
    effort = kwargs.pop("reasoning_effort", None)
    effort = None if effort is None else str(effort)
    options = {str(k): _normalize(v) for k, v in sorted(kwargs.items(), key=lambda kv: str(kv[0]))}

    timeouts: dict[str, Any] = {}
    multiplier = trial_cfg.get("timeout_multiplier")
    if multiplier is not None and _normalize(multiplier) != 1:
        timeouts["timeout_multiplier"] = _normalize(multiplier)
    for key in ("agent_timeout_multiplier", "agent_setup_timeout_multiplier"):
        if trial_cfg.get(key) is not None:
            timeouts[key] = _normalize(trial_cfg[key])
    for key in ("override_timeout_sec", "override_setup_timeout_sec", "max_timeout_sec"):
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
            key = canonical_model(str(reported), ctx, prefixes)
            entry = usage.setdefault(key, {"reported_as": str(reported), "n_input": None, "n_cache": None,
                                           "n_output": None, "reported_cost_usd": None,
                                           "primary": is_primary(key, model), "synthesized": False})
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


# Killed trials --------------------------------------------------------------------
#
# A trial killed with the harbor process tree (a cancel or budget stop on
# Windows) writes no result.json, so Harbor never totals its usage. The
# harness's own log survives in agent/ (a live bind mount), and holds every
# call that completed before the kill. These recipes mirror the Harbor 0.25.0
# adapters' populate_context_post_run, so a recovered figure is what Harbor
# would have recorded, minus the call in flight: a lower bound.

# (model as the harness logged it or None, prompt tokens incl. cache, cache reads, output, own cost or None)
_Call = tuple[str | None, int, int, int, float | None]


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
    and on compaction and branch_summary entries (harbor pi.py)."""
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
        cost = usage.get("cost")
        calls.append((None, _int(usage.get("input")) + read + write, read, _int(usage.get("output")),
                      _cost_or_none(cost.get("total") if isinstance(cost, Mapping) else None)))
    return calls


def _claude_code_calls(agent: Path) -> list[_Call] | None:
    """Claude Code's session files and subagent transcripts: the last usage
    of each assistant message id, events deduplicated by uuid (harbor
    claude_code.py). Every session folder is read; Harbor gives up on two."""
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
    calls: list[_Call] = []
    for model, u in last.values():
        read = _int(u.get("cache_read_input_tokens"))
        calls.append((str(model) if model else None,
                      _int(u.get("input_tokens")) + read + _int(u.get("cache_creation_input_tokens")),
                      read, _int(u.get("output_tokens")), None))
    return calls


def _strands_calls(agent: Path) -> list[_Call] | None:
    """strands-events.jsonl, which the runner flushes per message: each new
    assistant message's metadata.usage (harbor strands_trajectory.py)."""
    path = agent / "strands-events.jsonl"
    if not path.is_file():
        return None
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
        calls.append((None, n_in, read, n_out, None))
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
        calls.append((None, _int(usage.get("prompt_tokens") or usage.get("input_tokens")),
                      _int(details.get("cached_tokens") if isinstance(details, Mapping) else None),
                      _int(usage.get("completion_tokens") or usage.get("output_tokens")),
                      _cost_or_none(usage.get("cost"))))
    return calls


def _final_metrics(data: Any) -> _Call | None:
    metrics = data.get("final_metrics") if isinstance(data, Mapping) else None
    if not isinstance(metrics, Mapping):
        return None
    return (None, _int(metrics.get("total_prompt_tokens")), _int(metrics.get("total_cached_tokens")),
            _int(metrics.get("total_completion_tokens")), _cost_or_none(metrics.get("total_cost_usd")))


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
    return [main, *subagents]


# First match wins: a harness's own log, then an ATIF trajectory, which for an
# installed harness only Harbor's post-run step writes.
_RECOVERY = (("agent/pi/sessions", _pi_calls), ("agent/sessions/projects", _claude_code_calls),
             ("agent/strands-events.jsonl", _strands_calls),
             ("agent/mini-swe-agent.trajectory.json", _mini_swe_agent_calls), ("agent/trajectory.json", _atif_calls))


def recover_calls(trial_dir: Path) -> tuple[str, list[_Call]] | None:
    """(where, completed calls) from the usage log the harness left in
    trial_dir/agent, or None when it left none. Calls may be an empty list:
    the harness started its log, but no model call completed."""
    agent = Path(trial_dir) / "agent"
    if not agent.is_dir():
        return None
    for where, recipe in _RECOVERY:
        try:
            calls = recipe(agent)
        except (OSError, ValueError, TypeError, AttributeError):
            calls = None
        if calls is not None:
            return where, calls
    return None


def _recovered_usage(found: tuple[str, list[_Call]], ctx: RunContext, prefixes: Mapping[str, str],
                     model: str) -> dict[str, dict[str, Any]]:
    """The same per-model shape as _usage; every entry says where it came from."""
    where, calls = found
    usage: dict[str, dict[str, Any]] = {}
    for reported, n_in, n_cache, n_out, cost in calls:
        key = canonical_model(reported, ctx, prefixes) if reported else model
        entry = usage.setdefault(key, {"reported_as": reported, "n_input": 0, "n_cache": 0, "n_output": 0,
                                       "reported_cost_usd": None, "primary": is_primary(key, model),
                                       "synthesized": True, "recovered_from": where})
        entry["n_input"] += n_in
        entry["n_cache"] += n_cache
        entry["n_output"] += n_out
        if cost is not None:
            entry["reported_cost_usd"] = (entry["reported_cost_usd"] or 0.0) + cost
    return usage


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


def _build(*, result: Mapping | None, trial_cfg: Mapping, lock: Mapping | None, job_lock: Mapping | None,
           trial_dir: str, ctx: RunContext, prefixes: Mapping[str, str], job_id: str | None,
           recovered: tuple[str, list[_Call]] | None = None) -> TrialFacts:
    """Facts from a parsed result.json (finished) or a trial config.json
    (result None: killed before finishing), with the usage recovered from the
    harness's own log for a killed trial."""
    agent = trial_cfg.get("agent") if isinstance(trial_cfg.get("agent"), Mapping) else {}
    kwargs = agent.get("kwargs") if isinstance(agent.get("kwargs"), Mapping) else {}
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
        parser_version=PARSER_VERSION,
        raw_lock_z=compress(scrub_lock(dict(lock))) if isinstance(lock, Mapping) else None,
    )
    if result is None:
        usage = _recovered_usage(recovered, ctx, prefixes, model) if recovered else {}
        sums = [None if not usage else sum(u[k] for u in usage.values()) for k in ("n_input", "n_cache", "n_output")]
        costs = [u["reported_cost_usd"] for u in usage.values() if u["reported_cost_usd"] is not None]
        cost, pricing_sha = None, None
        if usage:
            try:
                cost, _unpriced = pricing.price_usage(usage, provider, pricing.load_table())
                pricing_sha = pricing.table_sha()
            except Exception:  # an invalid pricing.yaml must not stop ingestion
                cost, pricing_sha = None, None
        return TrialFacts(
            **common, trial_uid=None, reward=None, rewards_json=None, passed=None, exception_type=None,
            exception_head=None, harbor_cause=None, cap_marker=None, setup_marker=None,
            agent_started=int(recovered is not None),  # the harness began its log
            n_input=sums[0], n_cache=sums[1], n_output=sums[2], usage_json=canonical_json(usage),
            primary_share=_primary_share(usage), reported_cost_usd=math.fsum(costs) if costs else None,
            cost_at_ingest_usd=cost, pricing_sha_at_ingest=pricing_sha,
            started_at=None, finished_at=None, env_setup_sec=None, agent_setup_sec=None,
            agent_exec_sec=None, verifier_sec=None, total_sec=None, raw_result_z=None,
        )

    totals = token_totals(result)
    usage = _usage(result, ctx, prefixes, model, totals)
    try:
        cost, _unpriced = pricing.price_usage(usage, provider, pricing.load_table())
        pricing_sha = pricing.table_sha()
    except Exception:  # an invalid pricing.yaml must not stop ingestion
        cost, pricing_sha = None, None
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
        started_at=utc_text(result.get("started_at")), finished_at=utc_text(result.get("finished_at")),
        **phases,
        raw_result_z=compress(trim_result(dict(result))),
    )


def facts_from_result(result: Mapping, lock: Mapping | None, job_lock: Mapping | None, *, trial_dir: str,
                      ctx: RunContext, prefixes: Mapping[str, str]) -> TrialFacts:
    """Facts from an already-parsed result.json; the ledger's rederive uses
    this on the stored copy."""
    trial_cfg = result.get("config") if isinstance(result.get("config"), Mapping) else {}
    return _build(result=result, trial_cfg=trial_cfg, lock=lock, job_lock=job_lock, trial_dir=trial_dir,
                  ctx=ctx, prefixes=prefixes, job_id=trial_cfg.get("job_id"))


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
                             trial_dir=relative_dir(trial_dir, root), ctx=ctx, prefixes=prefixes)


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
                  recovered=recover_calls(trial_dir))
