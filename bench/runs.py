"""Run manager.

One run at a time. Every run, whether started from the LAUNCH tab (a config
mode plus arms) or from the queue (a run design), is a RunSpec, and it
executes only from that spec: dataset, tasks, attempts, run settings and arms
are fixed when it starts, and run.json records the spec, its hash and every
arm exactly as resolved, so a run can be read later without the config of the
day. Arms start in order, run.parallel_arms at a time. Each arm is one harbor
job, started as a child process and supervised by its own worker thread:

1. A `harbor run --dry-run` validates the arm's options. Nothing is spent.
   The budget and cancel are checked again before the real job starts.
2. The real job starts with a scrubbed environment holding only the arm's
   provider key, tied to the bench by a kill-on-close job object on Windows,
   so harbor cannot outlive a bench that is killed or crashes.
3. Every few seconds the worker reads finished trials, adds up spend, holds
   back an estimate for the trials still running, and stops the job when
   spend plus that reserve reaches the run budget. Trials killed mid-run
   write no result, and a trial stopped before any model call completed
   records no usage, so their spend is missing and the record says so.
4. Cancelling, hitting the budget, or harbor dying by itself kills harbor's
   process tree and removes the containers its trials left behind. Agents in
   those containers keep calling paid APIs until the containers are gone, so
   a cleanup that fails is recorded (cleanup_pending) and retried at the next
   start rather than reported as done.

Only one bench process may own a profile's runs: a second one would take the
first one's live run for dead and remove its containers.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.metadata
import json
import logging
import math
import os
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from . import checks, datasets, fsutil, keys, procs, results, schemas
from .procs import cleanup_containers  # noqa: F401  re-exported: older callers and tests patch runs.cleanup_containers
from .settings import (
    EFFORT_INHERIT,
    EFFORT_OMIT,
    JOBS_DIR,
    PROFILE_NAME,
    ROOT,
    RUNS_DIR,
    Arm,
    Config,
    ConfigError,
    RunSettings,
    harbor_executable,
)
from .spec import ArmSpec, RunSpec, spec_from_config, spec_from_design  # noqa: F401  spec_from_design re-exported

POLL_SECONDS = 3
MAX_RUN_ID = 44  # keeps the deepest trial paths under Windows' MAX_PATH
_WINDOWS = os.name == "nt"
_LIVE = ("queued", "preflight", "running")
_KILLED = ("{n} trial(s) were killed mid-run; their spend is not in the total, "
           "so recorded spend is a lower bound.")
_UNMEASURED = ("{n} finished trial(s) ran but recorded no usage; their spend is not in the total, "
               "so recorded spend is a lower bound.")

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _append_log(path: Path, line: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{_now()}] {line}\n")
    except OSError:
        pass  # a log line is never worth failing a run over


class LedgerHook(Protocol):
    """What the run manager needs from the ledger (ledger.Ledger)."""

    available: bool

    def upsert_bench_run(self, run: dict[str, Any]) -> None: ...

    def ingest_bench_job(self, job_dir: Path, run: dict[str, Any], arm: str, *,
                         final: bool, via: str) -> dict[str, Any]: ...

    def set_error(self, where: str, exc: BaseException) -> None: ...


# Planning --------------------------------------------------------------------


def resolve_effort(run: RunSettings, arm: Arm | ArmSpec, schema: dict[str, Any]) -> tuple[str | None, str]:
    """Returns the value to pass (None to pass nothing) and a label for the dashboard."""
    value = run.effort if arm.effort == EFFORT_INHERIT else arm.effort
    levels = schemas.effort_levels(schema)
    if value == EFFORT_OMIT:
        return None, "harness default"
    if levels is None:
        if arm.effort != EFFORT_INHERIT:
            raise ConfigError(f"{arm.agent} does not accept an effort setting")
        return None, "not supported (model default)"
    if levels and value not in levels:
        raise ConfigError(f"effort '{value}' is not one of: {', '.join(levels)}")
    return value, value


def agent_kwargs(run: RunSettings, arm: Arm | ArmSpec,
                 schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Returns the kwargs for harbor and where each one came from. Run-level
    caps fill in wherever the harness has a matching option; the arm's own
    options always win."""
    props = schema.get("properties", {})
    kwargs: dict[str, Any] = {}
    source: dict[str, str] = {}
    effort, _ = resolve_effort(run, arm, schema)
    if effort is not None:
        kwargs["reasoning_effort"], source["reasoning_effort"] = effort, "effort"
    if run.per_trial_cap_usd:
        for name in schemas.DOLLAR_CAPS:
            if name in props and not schemas.cap_misprices(arm, name):
                kwargs[name] = schemas.coerce(schema, name, f"{run.per_trial_cap_usd:g}")
                source[name] = "run cap"
    if run.max_turns:
        for name in schemas.TURN_CAPS:
            if name in props:
                kwargs[name] = schemas.coerce(schema, name, run.max_turns)
                source[name] = "run cap"
    for name, value in arm.options.items():
        if value is None:
            # None means "pass nothing", as it does in the overrides file, so
            # `max_budget_usd: null` lifts the run cap for this arm and Harbor
            # never sees `--ak name=null`, which it rejects.
            kwargs.pop(name, None)
            source.pop(name, None)
            continue
        kwargs[name] = schemas.coerce(schema, name, value)
        source[name] = "arm option"
    return kwargs, source


def options_fingerprint(kwargs: dict[str, Any], source: dict[str, str]) -> str:
    """12 hex of the arm's own options. Effort and run caps are left out, so
    changing a cap does not split one configuration into two when pooling."""
    own = {k: v for k, v in kwargs.items() if source.get(k) == "arm option"}
    return fsutil.sha256_hex(fsutil.canonical_json(own))[:12]


def build_command(cfg: Config, spec: RunSpec, arm: ArmSpec, job_root: Path, kwargs: dict[str, Any],
                  dataset_path: Path, dry_run: bool = False) -> list[str]:
    # An import path goes through --agent-import-path, which records the
    # class as agent.import_path in the job's config.json (the ledger reads it
    # there). Harbor 0.25 still honours the flag and logs a deprecation warning.
    agent_flag = "--agent-import-path" if arm.kind == "import_path" else "--agent"
    cmd = [
        harbor_executable(), "run",
        "--path", str(dataset_path),  # the local snapshot; see datasets.py
        agent_flag, arm.agent,
        "--model", cfg.harbor_model(arm),  # type: ignore[arg-type]
        "--n-attempts", str(spec.n_attempts),
        "--n-concurrent", str(spec.run.n_concurrent),
        "--jobs-dir", str(job_root),
        "--job-name", arm.name,
        "--timeout-multiplier", f"{spec.run.timeout_multiplier:g}",
        "--max-retries", "0",
        "--quiet",
        # No --yes: it would auto-approve prompts such as host environment
        # access. With stdin closed, an unexpected prompt fails the arm instead.
    ]
    for task in arm.tasks:
        cmd += ["--include-task-name", task]
    for name, value in kwargs.items():
        cmd += ["--ak", f"{name}={json.dumps(value)}"]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def preview_command(cfg: Config, spec: RunSpec, arm: ArmSpec, kwargs: dict[str, Any], dataset_path: Path,
                    job_root: Path = Path("jobs") / "RUN_ID") -> str:
    """The harbor command as a user can paste it into PowerShell (5.1 or 7).
    The quoting is CreateProcess's (`\\"` for a quote), which PowerShell would
    mangle; `--%` makes it pass the rest of the line through untouched. In
    cmd.exe, leave out the `--%`."""
    display = build_command(cfg, spec, arm, job_root, kwargs, dataset_path)
    return "harbor --% " + subprocess.list2cmdline(display[1:])


def _dollar_caps(kwargs: dict[str, Any]) -> list[float]:
    """The dollar caps the harness will enforce on each trial. Raises
    ValueError for a cap that is not a number, which plan reports."""
    caps = [float(kwargs[k]) for k in schemas.DOLLAR_CAPS if kwargs.get(k) is not None]
    return [c for c in caps if c > 0 and math.isfinite(c)]


def _trial_cap(kwargs: dict[str, Any]) -> float | None:
    try:
        caps = _dollar_caps(kwargs)
    except (TypeError, ValueError):
        return None
    return min(caps) if caps else None


def _reserve_note(cap: float | None) -> str:
    if cap is not None:
        return (f"While its trials run, each holds back the spend of the largest finished trial, "
                f"at most ${cap:g} (its dollar cap), from the run budget.")
    return ("While its trials run, each holds back the spend of the largest finished trial from the run "
            "budget. Before any trial finishes nothing is held back, and this arm has no dollar cap, so "
            "trials in flight can take the run past its budget.")


def _container_checks(cfg: Config, peak: int, problems: list[str], warnings: list[str]) -> None:
    limits = cfg.designs
    if peak > limits.max_containers:
        problems.append(f"Up to {peak} task containers would run at once (parallel_arms x n_concurrent); "
                        f"designs.max_containers is {limits.max_containers}")
    elif peak > limits.warn_containers:
        warnings.append(f"Up to {peak} task containers would run at once. Above {limits.warn_containers}, "
                        "Docker Desktop may run short of memory and fail trials as infrastructure errors")


def _memory_warning(spec: RunSpec, tasks: list[str], peak: int) -> list[str]:
    """Docker Desktop's VM has a fixed amount of memory; oversubscribing it
    fails trials as sandbox errors that look like harness failures."""
    try:
        capacity = checks.docker_capacity()
        if not capacity or not capacity.get("mem_gb"):
            return []
        meta = datasets.task_meta(spec.dataset)
        needs = [(meta.get(t) or {}).get("memory_gb") for t in tasks]
        known = [float(m) for m in needs if _finite(m)]
        if not known:
            return []
        largest, total = max(known), float(capacity["mem_gb"])
        if peak * largest <= 0.8 * total:
            return []
        return [f"Up to {peak} containers of up to {largest:g} GB each could need more than 80% of the "
                f"{total:g} GB Docker has. Trials may fail for lack of memory; lower parallel_arms or "
                "n_concurrent, or give Docker Desktop more memory"]
    except Exception:  # an advisory check never blocks planning
        return []


def _long_paths_enabled() -> bool:
    """HKLM\\...\\FileSystem\\LongPathsEnabled, read only. Without it most
    Windows programs cannot open paths of 260 characters or more."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        return value == 1
    except (ImportError, OSError):
        return False


def _path_warning(tasks: list[str]) -> list[str]:
    if not _WINDOWS or not tasks:
        return []
    # jobs dir + run id (at most 44) + arm name (at most 32) + task folder +
    # the deepest file Harbor writes under a trial (about 99 characters).
    projected = len(str(JOBS_DIR)) + MAX_RUN_ID + 32 + max(len(t) for t in tasks) + 99
    if projected <= 250 or _long_paths_enabled():
        return []
    return [f"Trial files may reach about {projected} characters, past Windows' 260-character path limit, "
            "and long paths are not enabled. Turn on LongPathsEnabled or move the bench to a shorter folder"]


def plan_spec(cfg: Config, spec: RunSpec) -> dict[str, Any]:
    """Everything a run would do, with the problems that block it and the
    warnings worth reading first. Shared by the LAUNCH tab, run designs and
    the queue."""
    arms: list[dict[str, Any]] = []
    problems: list[str] = []
    warnings: list[str] = []
    uncapped: list[str] = []
    worst_case = 0.0
    snapshot = datasets.path(spec.dataset)
    display_path = _display_path(snapshot) if snapshot else Path("datasets") / spec.dataset
    for arm in spec.arms:
        per_arm = len(arm.tasks) * spec.n_attempts
        provider = cfg.providers.get(arm.provider)
        entry: dict[str, Any] = {
            "name": arm.name, "agent": arm.agent, "family": arm.family, "kind": arm.kind,
            "provider": arm.provider,
            "model": cfg.harbor_model(arm) if provider else arm.model,  # type: ignore[arg-type]
            "tasks": len(arm.tasks), "trials": per_arm,
            "effort": None, "kwargs": [], "options_fingerprint": None, "command": "", "reserve_note": "",
            "problems": [], "warnings": [],
        }
        if provider is None:
            entry["problems"].append(f"unknown provider '{arm.provider}'")
        else:
            if not provider.enabled:
                entry["problems"].append(f"provider '{provider.name}' is turned off in config")
            if not keys.get_secret(provider.env_key):
                entry["problems"].append(f"{provider.env_key} is not set")
            if cfg.price(arm.provider, arm.model) is None:
                entry["problems"].append(
                    f"no price for '{arm.model}' under '{provider.name}' in config/pricing.yaml; "
                    "the run budget could not count this arm's spend")
        if arm.kind == "import_path" and arm.agent not in cfg.designs.import_paths:
            entry["problems"].append(f"'{arm.agent}' is not an allowed harness import path; "
                                     "list it under designs.import_paths in the config")
        try:
            schema = schemas.agent_schema(arm.agent)
            kwargs, source = agent_kwargs(spec.run, arm, schema)
            entry["effort"] = resolve_effort(spec.run, arm, schema)[1]
            entry["kwargs"] = [{"name": k, "value": v, "source": source[k]} for k, v in kwargs.items()]
            entry["options_fingerprint"] = options_fingerprint(kwargs, source)
            rejected = schemas.model_warnings(arm)  # type: ignore[arg-type]
            entry["warnings"] += [f"{k}: {rejected[k]}" for k in kwargs if k in rejected]
            entry["warnings"] += [f"{k}: {schemas.CAUTIONS[k]}" for k in kwargs
                                  if k in schemas.CAUTIONS and k != "version"]
            entry["warnings"] += [f"{k}: {schemas.MISPRICED_CAP}" for k in schemas.DOLLAR_CAPS
                                  if schemas.cap_misprices(arm, k) and spec.run.per_trial_cap_usd]
            caps = _dollar_caps(kwargs)
            if caps:
                worst_case += per_arm * min(caps)
            else:
                uncapped.append(arm.name)
            entry["reserve_note"] = _reserve_note(min(caps) if caps else None)
            if provider is not None:
                entry["command"] = preview_command(cfg, spec, arm, kwargs, display_path)
        except (ConfigError, ValueError, TypeError, RuntimeError) as exc:
            entry["problems"].append(str(exc))
        problems += [f"{arm.name}: {p}" for p in entry["problems"]]
        arms.append(entry)

    all_tasks = list(dict.fromkeys(t for arm in spec.arms for t in arm.tasks))
    peak = min(spec.run.parallel_arms, len(spec.arms)) * spec.run.n_concurrent
    _container_checks(cfg, peak, problems, warnings)
    warnings += _memory_warning(spec, all_tasks, peak)
    warnings += _path_warning(all_tasks)
    return {
        "tasks": list(spec.tasks), "n_attempts": spec.n_attempts,
        "dataset": spec.dataset, "dataset_snapshot": datasets.info(spec.dataset),
        "trials": sum(a["trials"] for a in arms), "budget_usd": spec.run.budget_usd,
        "worst_case_capped_usd": worst_case, "uncapped_arms": uncapped,
        "arms": arms, "problems": problems,
        "source": spec.source, "label": spec.label, "spec_hash": spec.spec_hash,
        "warnings": warnings, "peak_containers": peak,
    }


def plan(cfg: Config, mode_name: str, arm_names: list[str]) -> dict[str, Any]:
    """The LAUNCH tab's preview: plan_spec for the mode and arms, plus "mode"."""
    return {"mode": mode_name, **plan_spec(cfg, spec_from_config(cfg, mode_name, arm_names))}


def _harbor_version() -> str:
    try:
        return importlib.metadata.version("harbor")
    except importlib.metadata.PackageNotFoundError:
        return ""


def resolve_arms(cfg: Config, spec: RunSpec, run_id: str) -> list[dict[str, Any]]:
    """Every arm as it will run, for run.json. Readers (results, the ledger)
    use this rather than the config, which may have changed since."""
    snapshot = datasets.path(spec.dataset)
    display_path = _display_path(snapshot) if snapshot else Path("datasets") / spec.dataset
    job_root = _display_path(JOBS_DIR / run_id)
    out = []
    for arm in spec.arms:
        schema = schemas.agent_schema(arm.agent)
        kwargs, source = agent_kwargs(spec.run, arm, schema)
        value, label = resolve_effort(spec.run, arm, schema)
        out.append({
            "name": arm.name, "agent": arm.agent, "family": arm.family, "kind": arm.kind,
            "provider": arm.provider, "model": arm.model,
            "harbor_model": cfg.harbor_model(arm),  # type: ignore[arg-type]
            "effort": {"setting": arm.effort, "value": value, "label": label},
            "kwargs": [{"name": k, "value": v, "source": source[k]} for k, v in kwargs.items()],
            "options_fingerprint": options_fingerprint(kwargs, source),
            "tasks": list(arm.tasks),
            "trials": len(arm.tasks) * spec.n_attempts,
            "command": preview_command(cfg, spec, arm, kwargs, display_path, job_root),
        })
    return out


def _running_trials(job_dir: Path) -> int:
    """Trial folders Harbor has started (config.json) but not finished
    (result.json, which it writes once, at the very end)."""
    try:
        return sum(1 for d in job_dir.iterdir()
                   if d.is_dir() and (d / "config.json").exists() and not (d / "result.json").exists())
    except OSError:
        return 0


# Runs ----------------------------------------------------------------------------


@dataclass
class ArmRun:
    name: str
    # queued, preflight, running, done, done_with_errors, failed, cancelled, skipped, budget_stopped, interrupted
    status: str = "queued"
    message: str = ""
    trials_total: int = 0
    trials_done: int = 0
    spend_usd: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    family: str = ""
    cleanup_pending: bool = False  # this arm's containers may still exist; the next start retries
    trials_running: int = 0
    reserve_usd: float = 0.0       # held back from the budget for trials_running
    trials_killed: int = 0         # stopped mid-run: their spend is not in spend_usd
    trials_unmeasured: int = 0     # finished with no usage recorded: their spend is not in spend_usd
    pid: int | None = None         # the harbor process last started for this arm


@dataclass
class Run:
    id: str
    mode: str  # the config mode, or "design"
    created_at: str
    budget_usd: float
    tasks: list[str]
    arms: list[ArmRun]
    plan: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)  # snapshot the run used
    status: str = "running"  # running, done, done_with_errors, budget_stopped, cancelled, failed, interrupted
    message: str = ""
    finished_at: str | None = None
    profile: str = ""
    source: str = "config"   # "config" (LAUNCH tab) or "design" (the queue)
    spec: dict[str, Any] = field(default_factory=dict)  # the NormalizedSpec it ran
    spec_hash: str = ""
    design_ref: dict[str, Any] | None = None
    resolved_arms: list[dict[str, Any]] = field(default_factory=list)
    harbor_version: str = ""

    @property
    def spend_usd(self) -> float:
        return sum(a.spend_usd for a in self.arms)

    @property
    def reserve_usd(self) -> float:
        return sum(a.reserve_usd for a in self.arms)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["spend_usd"] = self.spend_usd
        data["reserve_usd"] = self.reserve_usd
        return data


_RUN_FIELDS = frozenset(f.name for f in dataclasses.fields(Run)) - {"arms"}
_ARM_FIELDS = frozenset(f.name for f in dataclasses.fields(ArmRun))


class RunManager:
    def __init__(self, ledger: LedgerHook | None = None) -> None:
        self.ledger = ledger
        self._lock = threading.RLock()
        self._save_lock = threading.Lock()
        self._runs: dict[str, Run] = {}
        self._active: str | None = None
        self._announced: str | None = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._procs: set[subprocess.Popen] = set()
        self._procs_lock = threading.Lock()
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self._ledger_logged: set[str] = set()
        self._largest_trial: dict[tuple[str, str], float] = {}  # (run, arm) -> largest finished spend
        self._trial_caps: dict[tuple[str, str], float | None] = {}
        # Arm threads write and scan these together; an unguarded scan during
        # another arm's first insert would raise and get a healthy arm killed.
        self._estimates_lock = threading.Lock()
        # Set by shutdown/emergency_stop: a start racing process exit would
        # clear the cancel flag and run arms that nothing can stop.
        self._closing = False
        # Before anything touches containers: a second bench for this profile
        # would take the first one's live run for dead and remove its containers.
        self._lock_fd = procs.acquire_instance_lock(RUNS_DIR)
        self._load_history()

    # Public ---------------------------------------------------------------

    @property
    def active_id(self) -> str | None:
        return self._active

    def is_idle(self) -> bool:
        return self._active is None

    def add_listener(self, fn: Callable[[dict[str, Any]], None]) -> None:
        """fn(run_dict) is called once per run, after its final save."""
        with self._lock:
            self._listeners.append(fn)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)
            return [r.to_dict() for r in runs]

    def get(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(run_id)
            return self._runs[run_id].to_dict()

    def log_path(self, run_id: str, arm: str) -> Path:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or arm not in {a.name for a in run.arms}:
                raise KeyError(arm)
        return RUNS_DIR / run_id / f"{arm}.log"

    def start(self, cfg: Config, mode_name: str, arm_names: list[str]) -> dict[str, Any]:
        """A LAUNCH-tab run: the config's mode and arms, recorded as a spec."""
        with self._lock:
            if self._closing:
                raise RuntimeError("The bench is shutting down, so no new run can start")
            if self._active:
                raise RuntimeError("A run is already in progress")
            return self.start_spec(cfg, spec_from_config(cfg, mode_name, arm_names))

    def start_spec(self, cfg: Config, spec: RunSpec) -> dict[str, Any]:
        with self._lock:
            if self._closing:
                raise RuntimeError("The bench is shutting down, so no new run can start")
            if self._active:
                raise RuntimeError("A run is already in progress")
            p = plan_spec(cfg, spec)
            if p["problems"]:
                raise ConfigError("; ".join(p["problems"]))
            mode = spec.label if spec.source == "config" else "design"
            run_id = self._new_run_id(spec)
            run = Run(
                id=run_id,
                mode=mode,
                created_at=_now(),
                budget_usd=spec.run.budget_usd,
                tasks=list(spec.tasks),
                arms=[ArmRun(name=a.name, family=a.family, trials_total=len(a.tasks) * spec.n_attempts)
                      for a in spec.arms],
                plan={"mode": mode, **p},
                config=cfg.effective,
                profile=PROFILE_NAME,
                source=spec.source,
                spec=copy.deepcopy(spec.spec),
                spec_hash=spec.spec_hash,
                design_ref=copy.deepcopy(spec.design_ref),
                resolved_arms=resolve_arms(cfg, spec, run_id),
                harbor_version=_harbor_version(),
            )
            self._runs[run.id] = run
            self._active = run.id
            self._cancel.clear()
            # emergency_stop takes no lock and may have run while this start
            # was planning; it sets _closing before cancelling, so this check
            # after the clear cannot miss it.
            if self._closing:
                self._cancel.set()
                del self._runs[run.id]
                self._active = None
                raise RuntimeError("The bench is shutting down, so no new run can start")
            self._save(run)
            self._ledger_call(None, "upsert_bench_run", run.to_dict())
            self._thread = threading.Thread(
                target=self._execute, args=(cfg, spec, run), name=f"run-{run.id}", daemon=True
            )
            self._thread.start()
            return run.to_dict()

    def cancel(self, run_id: str) -> None:
        with self._lock:
            if self._active != run_id:
                raise RuntimeError("That run is not in progress")
            self._cancel.set()

    def shutdown(self, timeout: float = 90) -> None:
        """Called when the dashboard stops: a run in progress is stopped
        cleanly. Safe to call more than once and when nothing is running."""
        with self._lock:
            self._closing = True
            active, thread = self._active, self._thread
            announce = active is not None and self._announced != active
            if announce:
                self._announced = active
        if active is None or thread is None:
            return
        if announce:
            # uvicorn logs nothing at warning level, and silence invites a
            # second Ctrl+C, which can interrupt the container cleanup.
            try:
                print(f"Stopping run {active}: killing harbor and removing containers...", flush=True)
            except (OSError, ValueError):
                pass
        self._cancel.set()
        thread.join(timeout)

    def emergency_stop(self) -> None:
        """The console is closing (or Windows is logging off): Windows ends the
        process about 5 seconds later and no Python cleanup runs. Kill harbor
        now and hand container removal to a helper that outlives the bench.
        Returns within about 4 seconds and never raises."""
        try:
            # No lock: the console handler must not wait on a busy start.
            self._closing = True
            self._cancel.set()
            with self._procs_lock:
                live = list(self._procs)
            killers = [threading.Thread(target=procs.kill_tree, args=(p.pid, 2), daemon=True,
                                        name="emergency-kill") for p in live]
            for t in killers:
                t.start()
            deadline = time.monotonic() + 2.5
            for t in killers:
                t.join(max(0.0, deadline - time.monotonic()))
            active = self._active
            run = self._runs.get(active) if active else None
            if run is None:
                return
            job_dirs = []
            for a in run.arms:
                if a.status == "running" or a.cleanup_pending:
                    a.cleanup_pending = True  # cleared by the next start's sweep
                    job_dirs.append(JOBS_DIR / run.id / a.name)
            if job_dirs:
                procs.spawn_detached_cleanup(job_dirs)
            self._save(run, patience=0.5, wait=0.5)
        except Exception:
            pass

    # Worker ---------------------------------------------------------------

    def _new_run_id(self, spec: RunSpec) -> str:
        base = f"{datetime.now():%Y%m%d-%H%M%S}-{spec.label}"[:MAX_RUN_ID].rstrip("-")
        run_id, n = base, 2
        while run_id in self._runs or (RUNS_DIR / run_id).exists() or (JOBS_DIR / run_id).exists():
            tail = f"-{n}"
            run_id = base[: MAX_RUN_ID - len(tail)].rstrip("-") + tail
            n += 1
        return run_id

    def _budget_used(self, run: Run) -> bool:
        return run.spend_usd + run.reserve_usd >= run.budget_usd

    def _execute(self, cfg: Config, spec: RunSpec, run: Run) -> None:
        try:
            log_path = RUNS_DIR / run.id / "dataset.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(log_path, "a", encoding="utf-8") as dataset_log:
                    dataset_path = datasets.ensure(spec.dataset, dataset_log, cancel=self._cancel)
            except datasets.DownloadCancelled:
                dataset_path = None
                run.status = "cancelled"
                run.message = "Cancelled during the dataset download"
                for a in run.arms:
                    if a.status in _LIVE:
                        a.status = "cancelled"
            if dataset_path is not None:
                run.dataset = datasets.info(spec.dataset) or {}
                self._save(run)
                # Arms start in spec order, run.parallel_arms at a time. Every
                # arm checks the shared run budget before it starts and while it runs.
                with ThreadPoolExecutor(max_workers=spec.run.parallel_arms,
                                        thread_name_prefix=f"arm-{run.id}") as pool:
                    futures = [pool.submit(self._arm_slot, cfg, spec, run, arm, arm_run, dataset_path)
                               for arm, arm_run in zip(spec.arms, run.arms)]
                    for future in futures:
                        future.result()
                statuses = {a.status for a in run.arms}
                if self._cancel.is_set():
                    run.status = "cancelled"
                elif statuses & {"budget_stopped", "skipped"}:
                    run.status = "budget_stopped"
                elif statuses == {"done"}:
                    run.status = "done"
                elif statuses == {"failed"}:
                    run.status = "failed"
                    run.message = "Every arm failed. See the arm logs."
                else:
                    run.status = "done_with_errors"
        except Exception as exc:  # keep the dashboard alive and the run record honest
            run.status = "failed"
            run.message = f"{type(exc).__name__}: {exc}"
            # 'failed', not 'skipped': skipped reads as "the budget ran out".
            for a in run.arms:
                if a.status in _LIVE:
                    a.status = "failed"
                    a.message = a.message or f"Run failed: {run.message}"
                    a.finished_at = a.finished_at or _now()
        finally:
            run.finished_at = _now()
            with self._lock:
                self._active = None
            self._save(run, patience=30.0)  # the final record matters most
            self._ledger_call(None, "upsert_bench_run", run.to_dict())
            self._notify(run)

    def _notify(self, run: Run) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(run.to_dict())
            except Exception:
                log.exception("A run listener failed for run %s", run.id)

    def _arm_slot(self, cfg: Config, spec: RunSpec, run: Run, arm: ArmSpec, arm_run: ArmRun,
                  dataset_path: Path) -> None:
        if self._cancel.is_set():
            arm_run.status = "cancelled"
        elif self._budget_used(run):
            arm_run.status = "skipped"
            arm_run.message = "Run budget was used up before this arm started"
        else:
            try:
                self._run_arm(cfg, spec, run, arm, arm_run, dataset_path)
            except Exception as exc:  # one arm failing must not stop the others
                arm_run.status = "failed"
                arm_run.message = f"{type(exc).__name__}: {exc}"
                arm_run.finished_at = _now()
                # Nothing runs for this arm any more, so it must hold back no budget.
                arm_run.trials_running = 0
                arm_run.reserve_usd = 0.0
        self._save(run)

    def _run_arm(self, cfg: Config, spec: RunSpec, run: Run, arm: ArmSpec, arm_run: ArmRun,
                 dataset_path: Path) -> None:
        schema = schemas.agent_schema(arm.agent)
        kwargs, _ = agent_kwargs(spec.run, arm, schema)
        env = keys.child_env(cfg.providers[arm.provider])
        job_root = JOBS_DIR / run.id
        job_dir = job_root / arm.name
        log_path = RUNS_DIR / run.id / f"{arm.name}.log"
        with self._estimates_lock:
            self._trial_caps[(run.id, arm.name)] = _trial_cap(kwargs)
        arm_run.started_at = _now()

        arm_run.status = "preflight"
        self._save(run)
        rc = self._supervise(build_command(cfg, spec, arm, job_root, kwargs, dataset_path, dry_run=True),
                             env, log_path, run, arm_run, watch=None)
        if arm_run.status == "preflight":
            if rc != 0:
                arm_run.status = "failed"
                arm_run.message = "Preflight failed, so nothing was spent. See the log."
            elif self._cancel.is_set():
                arm_run.status, arm_run.message = "cancelled", "Cancelled from the dashboard"
            elif self._budget_used(run):
                # Another arm spent the rest while this one was in its dry run.
                arm_run.status = "skipped"
                arm_run.message = "Run budget was used up before this arm started"
        if arm_run.status != "preflight":
            arm_run.returncode, arm_run.finished_at = rc, _now()
            return

        arm_run.status = "running"
        # Recorded before harbor starts, so a bench that dies from here on
        # leaves a record that this job's containers may need removing.
        arm_run.cleanup_pending = True
        self._save(run)
        local_arm = arm.to_arm()
        rc = self._supervise(build_command(cfg, spec, arm, job_root, kwargs, dataset_path),
                             env, log_path, run, arm_run, watch=(cfg, local_arm, job_dir))
        if rc == 0:
            arm_run.cleanup_pending = False  # harbor finished and tore down its own trials
        elif arm_run.status == "running":
            # Harbor ended by itself with an error (Ctrl+Break, Task Manager, a
            # crash) and could not remove its trials' containers. Cancel and
            # budget stops changed the status and swept already.
            ok, message = procs.cleanup_with_retry(job_dir)
            arm_run.cleanup_pending = not ok
            _append_log(log_path, f"harbor exited with code {rc}; {message}")
        rows = self._refresh(cfg, local_arm, job_dir, arm_run, run, live=False)
        arm_run.returncode, arm_run.finished_at = rc, _now()
        if arm_run.status == "running":
            errored = sum(1 for r in rows if r["error"])
            if rc != 0:
                arm_run.status = "failed"
                arm_run.message = f"harbor exited with code {rc}. See the log."
            elif errored:
                # Harbor exits 0 even when trials error, so say so.
                arm_run.status = "done_with_errors"
                arm_run.message = f"{errored} of {len(rows)} trials errored before finishing"
            else:
                arm_run.status = "done"
        if arm_run.trials_unmeasured:
            note = _UNMEASURED.format(n=arm_run.trials_unmeasured)
            arm_run.message = f"{arm_run.message.rstrip('.')}. {note}".lstrip(". ")
        self._ledger_call(log_path, "ingest_bench_job", job_dir, run.to_dict(), arm.name,
                          final=True, via="finalize")

    def _supervise(self, cmd: list[str], env: dict[str, str], log_path: Path, run: Run,
                   arm_run: ArmRun, watch: tuple[Config, Arm, Path] | None) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        job_dir = watch[2] if watch else None
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{_now()}] $ {subprocess.list2cmdline(cmd)}\n")
            log_file.flush()
            proc = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT,
                creationflags=procs.NO_CTRL_C,  # console Ctrl+C reaches the bench, which stops harbor itself
            )
            with self._procs_lock:
                self._procs.add(proc)
            arm_run.pid = proc.pid
            try:
                if not procs.assign_kill_on_close(proc) and _WINDOWS:
                    log_file.write(f"[{_now()}] could not tie harbor (pid {proc.pid}) to the bench's lifetime; "
                                   "if the bench is killed, end harbor.exe in Task Manager\n")
                    log_file.flush()
                while proc.poll() is None:
                    if self._cancel.wait(POLL_SECONDS):
                        arm_run.status, arm_run.message = "cancelled", "Cancelled from the dashboard"
                        self._stop(proc, job_dir, log_file, arm_run)
                        break
                    if watch is None:
                        continue
                    cfg, arm, _ = watch
                    self._refresh(cfg, arm, job_dir, arm_run, run, log_path=log_path)  # type: ignore[arg-type]
                    if self._budget_used(run):
                        arm_run.status = "budget_stopped"
                        held = (f" plus ${run.reserve_usd:.2f} held back for running trials"
                                if run.reserve_usd > 0 else "")
                        arm_run.message = (f"Run spend ${run.spend_usd:.2f}{held} reached "
                                           f"the ${run.budget_usd:.2f} budget")
                        self._stop(proc, job_dir, log_file, arm_run)
                        break
                    self._save(run)
                rc = proc.wait()
            except BaseException:
                # Never leave harbor running unsupervised: whatever went wrong
                # here, stop the job and its containers before passing it on.
                if proc.poll() is None:
                    self._stop(proc, job_dir, log_file, arm_run)
                raise
            finally:
                with self._procs_lock:
                    self._procs.discard(proc)
            log_file.write(f"[{_now()}] exit code {rc}\n")
        return rc

    def _stop(self, proc: subprocess.Popen, job_dir: Path | None, log_file: Any, arm_run: ArmRun) -> None:
        if job_dir is not None:
            killed = _running_trials(job_dir)
            if killed:
                arm_run.trials_killed = killed
                arm_run.message = f"{arm_run.message.rstrip('.')}. {_KILLED.format(n=killed)}".lstrip(". ")
        log_file.write(f"[{_now()}] stopping harbor (pid {proc.pid}) and its containers\n")
        log_file.flush()
        # Harbor cleans up on SIGTERM, which Windows cannot deliver, so the
        # tree is killed and the containers are removed here.
        procs.kill_tree(proc.pid)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        if job_dir is not None:
            ok, message = procs.cleanup_with_retry(job_dir)
            arm_run.cleanup_pending = not ok
            log_file.write(f"[{_now()}] {message}\n")
            log_file.flush()

    def _refresh(self, cfg: Config, arm: Arm, job_dir: Path, arm_run: ArmRun, run: Run,
                 log_path: Path | None = None, live: bool = True) -> list[dict[str, Any]]:
        rows = results.load_job(job_dir, cfg, arm)
        spends = [results.spend(r) for r in rows]
        # A trial with no usage recorded shows $0, which is not what it spent.
        unknown = [r.get("usage_status") == "unmeasured" and not s for r, s in zip(rows, spends)]
        arm_run.trials_done = len(rows)
        arm_run.spend_usd = round(sum(spends), 4)
        arm_run.trials_unmeasured = sum(unknown)
        known = [s for s, u in zip(spends, unknown) if not u]
        if known:
            with self._estimates_lock:
                self._largest_trial[(run.id, arm_run.name)] = max(known)
        if live:
            running = _running_trials(job_dir)
            arm_run.trials_running = running
            arm_run.reserve_usd = round(running * self._trial_estimate(run, arm_run.name), 4) if running else 0.0
            if log_path is not None:
                self._ledger_call(log_path, "ingest_bench_job", job_dir, run.to_dict(), arm_run.name,
                                  final=False, via="live")
        else:
            # The job has ended: nothing is running, and what was killed is
            # recorded in trials_killed rather than held back.
            arm_run.trials_running = 0
            arm_run.reserve_usd = 0.0
        return rows

    def _trial_estimate(self, run: Run, arm_name: str) -> float:
        """What one running trial may be spending: the largest finished trial
        in this arm, else in the whole run, else nothing, and never more than
        the arm's own dollar cap. A flat per-trial cap would stop a cheap run
        before it started."""
        with self._estimates_lock:
            estimate = self._largest_trial.get((run.id, arm_name))
            if estimate is None:
                estimate = max((v for (rid, _), v in self._largest_trial.items() if rid == run.id),
                               default=0.0)
            cap = self._trial_caps.get((run.id, arm_name))
        return min(estimate, cap) if cap is not None else estimate

    # Ledger -----------------------------------------------------------------

    def _ledger_call(self, log_path: Path | None, fn: Callable[..., Any] | str, *args: Any, **kwargs: Any) -> Any:
        """Call the ledger without ever letting it change a run: a failure
        writes one line to the arm's log, is reported to the ledger, and the
        next rescan catches up. fn is a ledger method or its name."""
        ledger = self.ledger
        if ledger is None:
            return None
        try:
            if not ledger.available:
                return None
            target = getattr(ledger, fn) if isinstance(fn, str) else fn
            return target(*args, **kwargs)
        except Exception as exc:
            name = fn if isinstance(fn, str) else getattr(fn, "__name__", "ledger")
            if log_path is not None and str(log_path) not in self._ledger_logged:
                self._ledger_logged.add(str(log_path))
                detail = " ".join(str(exc).split())[:300]
                _append_log(log_path, f"[ledger] {type(exc).__name__}: {detail}; the next rescan will catch up")
            try:
                ledger.set_error(name, exc)
            except Exception:
                pass
            return None

    # Persistence ----------------------------------------------------------

    def _save(self, run: Run, patience: float = 2.0, wait: float | None = None) -> None:
        """Persist the run. The in-memory run is authoritative and saves are
        best-effort: on Windows, replacing a file fails while any other process
        has it open (a viewer, an indexer, antivirus), so retry, then give up
        until the next save rather than disturb the run."""
        if not self._save_lock.acquire(timeout=-1 if wait is None else wait):
            return
        try:  # arms running in parallel all save the same run
            fsutil.atomic_write_json(RUNS_DIR / run.id / "run.json", run.to_dict(), patience)
        except (OSError, TypeError, ValueError):
            pass
        finally:
            self._save_lock.release()

    def _load_history(self) -> None:
        for path in sorted(RUNS_DIR.glob("*/run.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # Fields this version does not know (or no longer has) are dropped.
                arms = [ArmRun(**{k: v for k, v in a.items() if k in _ARM_FIELDS}) for a in data.get("arms") or []]
                run = Run(arms=arms, **{k: v for k, v in data.items() if k in _RUN_FIELDS})
            except (OSError, ValueError, TypeError, AttributeError):
                continue
            if not _finite(run.budget_usd):
                continue  # a budget that can never trip, and that JSON responses cannot carry
            run.profile = run.profile or PROFILE_NAME
            changed = False
            sweep = [a for a in run.arms if a.cleanup_pending or a.status in _LIVE]
            if run.status == "running":
                run.status = "interrupted"
                run.message = "The dashboard stopped while this run was in progress."
                changed = True
            for a in run.arms:
                # Older versions could finish a run with arms still queued or
                # running; once swept, such an arm is not swept again.
                if a.status in _LIVE:
                    a.status = "interrupted"
                    changed = True
            for a in sweep:
                job_dir = JOBS_DIR / run.id / a.name
                ok, message = procs.cleanup_with_retry(job_dir, tries=1)
                if job_dir.exists() or not ok:
                    _append_log(RUNS_DIR / run.id / f"{a.name}.log", f"startup sweep: {message}")
                # A failed sweep (Docker not running) keeps the flag, so the
                # next start tries again.
                if a.cleanup_pending == ok:
                    a.cleanup_pending = not ok
                    changed = True
            if changed:
                self._save(run)
            self._runs[run.id] = run
