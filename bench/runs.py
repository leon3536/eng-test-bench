"""Run manager.

One run at a time. Every run, whether started from the LAUNCH tab (a config
mode plus arms) or from the queue (a run design), is a RunSpec, and it
executes only from that spec: dataset, tasks, attempts, run settings and arms
are fixed when it starts, and run.json records the spec, its hash and every
arm exactly as resolved, so a run can be read later without the config of the
day. run.schedule orders the trials:

- arms: arms start in order, run.parallel_arms at a time, each arm one harbor
  job over all its tasks (jobs/<run>/<arm>/).
- tasks (a series of task races): for each task in order, the pre-baked
  images every arm needs for it are built (BuildManager, the build cache it
  made pruned after), then every arm that runs the task races on it at once,
  one harbor job per arm (jobs/<run>/<arm>/<task>/), at most
  run.max_containers containers in all, and then the task's images are
  removed unless a later task, another arm or a waiting queue item needs them
  (or prebake.keep_images). The disk holds one task's images at a time, and
  builds and races never overlap: the run holds the run lock throughout.

Each harbor job is a child process supervised by its own worker thread:

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

import contextlib
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

from . import checks, datasets, fsutil, keys, prebake, procs, results, schemas
from .procs import cleanup_containers  # noqa: F401  re-exported: older callers and tests patch runs.cleanup_containers
from .prebake import ArmPrebake
from .settings import (
    EFFORT_INHERIT,
    EFFORT_OMIT,
    JOBS_DIR,
    PREBAKE_OFF,
    PREBAKE_REQUIRE,
    PROFILE_NAME,
    ROOT,
    RUNS_DIR,
    SCHEDULE_TASKS,
    Arm,
    Config,
    ConfigError,
    RunSettings,
    harbor_executable,
)
from .spec import (  # noqa: F401  spec_from_design re-exported
    ArmSpec,
    RunSpec,
    race_lane_slots,
    race_peak,
    spec_from_config,
    spec_from_design,
)

POLL_SECONDS = 3
# Between the starts of the first parallel arms, so their preflights and first installs do not all hit Docker at once.
ARM_STAGGER_SECONDS = 15
# Harbor's preflight gives `docker info` 10 s, which a busy host can exceed while Docker is up: the waits before
# each retry of a preflight that failed only on that.
PREFLIGHT_RETRY_WAITS = (20, 60)
_DOCKER_SLOW = "Docker daemon is not running"  # harbor's preflight message when `docker info` fails or times out
# Between two harbor starts in a task race: enough to spread their preflights, short because a campaign has many.
RACE_STAGGER_SECONDS = 3
# Images built at once before a task's race: one per arm, at most one per real CPU core of the bench host.
RACE_BUILD_JOBS = 6
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


def _docker_was_slow(log_path: Path) -> bool:
    """Whether harbor's last command in this log failed on its `docker info`
    check while Docker is in fact running."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return False
    return _DOCKER_SLOW in tail.rsplit("] $ ", 1)[-1] and checks.docker_ok()


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


def prebake_flags(run: RunSettings, arm_prebake: ArmPrebake | None) -> list[str]:
    """The harbor flags for how an arm's trials get their harness installed.
    With a supported arm under prefer or require, the bench's environment
    class (--env/--ek, values json-dumped like --ak) runs each trial on a
    pre-baked image when one is verified, else on stock setup. The setup and
    build limits follow run.setup_timeout_multiplier and
    run.env_build_timeout_multiplier. None of it is part of the config hash.
    Mode off with the default limits adds nothing: exactly the stock command."""
    flags: list[str] = []
    if arm_prebake is not None and arm_prebake.supported:
        from .environments.prebaked import PREBAKED_ENV
        flags += ["--env", PREBAKED_ENV]
        for name, value in arm_prebake.env_kwargs().items():
            flags += ["--ek", f"{name}={json.dumps(value)}"]
    setup = prebake.setup_timeout_multiplier(run, arm_prebake)
    if setup is not None:
        flags += ["--agent-setup-timeout-multiplier", f"{setup:g}"]
    if float(run.env_build_timeout_multiplier) != 1:
        flags += ["--environment-build-timeout-multiplier", f"{run.env_build_timeout_multiplier:g}"]
    return flags


def build_command(cfg: Config, spec: RunSpec, arm: ArmSpec, job_root: Path, kwargs: dict[str, Any],
                  dataset_path: Path, dry_run: bool = False, arm_prebake: ArmPrebake | None = None,
                  job_name: str | None = None) -> list[str]:
    """The harbor command for one arm's job: job_root/<job_name, else the arm's
    name>. A task race passes one task's arm (arm.tasks == (task,)), the arm's
    folder as job_root and the task as job_name."""
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
        "--job-name", job_name or arm.name,
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
    cmd += prebake_flags(spec.run, arm_prebake)
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def preview_command(cfg: Config, spec: RunSpec, arm: ArmSpec, kwargs: dict[str, Any], dataset_path: Path,
                    job_root: Path = Path("jobs") / "RUN_ID", arm_prebake: ArmPrebake | None = None) -> str:
    """The harbor command as a user can paste it into PowerShell (5.1 or 7).
    The quoting is CreateProcess's (`\\"` for a quote), which PowerShell would
    mangle; `--%` makes it pass the rest of the line through untouched. In
    cmd.exe, leave out the `--%`. Under schedule tasks it is the first task's
    job; every task of the arm gets the same command with its own name."""
    if spec.run.schedule == SCHEDULE_TASKS and arm.tasks:
        display = build_command(cfg, spec, dataclasses.replace(arm, tasks=arm.tasks[:1]), job_root / arm.name,
                                kwargs, dataset_path, arm_prebake=arm_prebake, job_name=arm.tasks[0])
        return "harbor --% " + subprocess.list2cmdline(display[1:])
    display = build_command(cfg, spec, arm, job_root, kwargs, dataset_path, arm_prebake=arm_prebake)
    return "harbor --% " + subprocess.list2cmdline(display[1:])


def prebake_entry(run: RunSettings, arm_prebake: ArmPrebake | None, readiness: Any = None,
                  error: str | None = None) -> dict[str, Any]:
    """An arm's harness setup as the plan and run.json show it: the prebake
    mode, whether the arm can use pre-baked images (and why not), its recipe,
    and the setup and build limit multipliers its command carries. With the
    arm's inventory.ArmReadiness, 'images' says which tasks' images are
    ready, missing or stale (with the reason), which are label-only no-ops,
    the oldest bake and what building the rest would take; error says why
    they could not be checked."""
    out: dict[str, Any] = {"mode": run.prebake} if arm_prebake is None else arm_prebake.summary()
    out["setup_timeout_multiplier"] = prebake.setup_timeout_multiplier(run, arm_prebake)
    out["env_build_timeout_multiplier"] = float(run.env_build_timeout_multiplier)
    if arm_prebake is None or not arm_prebake.supported:
        return out
    if readiness is not None:
        r = readiness
        out["images"] = {"tasks": r.tasks, "ready": list(r.ready), "missing": list(r.missing), "stale": dict(r.stale),
                         "unverifiable": list(r.unverifiable), "unknown": dict(r.unknown), "noop": list(r.noop),
                         "over_age": list(r.over_age), "oldest_baked_at": r.oldest_baked_at,
                         "build_min": r.build_min, "build_gb": r.build_gb, "all_ready": r.all_ready,
                         "error": r.docker_error}
    elif error:
        out["images"] = {"error": error}
    return out


def image_readiness(cfg: Config, spec: RunSpec, refresh: bool = False) -> tuple[dict[str, Any], str | None]:
    """inventory.readiness for spec's arms ({} under prebake off), or ({},
    why the images could not be checked). Never raises: a plan always plans."""
    if spec.run.prebake == PREBAKE_OFF or not spec.arms:
        return {}, None
    try:
        from .prebake import inventory
        return inventory.readiness(cfg, spec, refresh=refresh), None
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"


def _stock_setup_sec(family: str) -> float | None:
    """Harbor's own install of family, as measured (the adapters' figure)."""
    try:
        from .prebake.adapters import ADAPTERS
    except Exception:
        return None
    cls = ADAPTERS.get(family)
    return float(cls.stock_setup_sec) if cls is not None else None


def _prebake_issues(cfg: Config, spec: RunSpec, arm: ArmSpec, arm_prebake: ArmPrebake | None, readiness: Any,
                    error: str | None, peak: int) -> tuple[list[str], list[str]]:
    """(problems, warnings) about how the arm's trials get their harness.
    require: an image that is not ready (or not checkable) is a problem;
    prefer: a warning, since those trials fall back to Harbor's own install.
    Under schedule tasks a missing image is neither: the race builds it."""
    problems: list[str] = []
    warnings: list[str] = []
    stock = 0
    if arm_prebake is None or not arm_prebake.supported:
        stock = len(arm.tasks)  # off, or no image possible (require's problem is raised by plan_spec)
    elif error or (readiness is not None and readiness.docker_error):
        why = error or readiness.docker_error
        if arm_prebake.required:
            problems.append(f"run.prebake is require, but the pre-baked images could not be checked: {why}")
        else:
            warnings.append(f"The pre-baked images could not be checked ({why}); a trial whose image is not "
                            "ready installs the harness itself (stock setup)")
    elif readiness is not None and spec.run.schedule == SCHEDULE_TASKS:
        # A task race builds each task's missing images right before that task; only images it cannot name
        # (no docker_image) are known now to end up without one.
        unknown = len(readiness.unknown)
        if unknown and arm_prebake.required:
            problems.append(f"run.prebake is require, but {unknown} task(s) have no pre-baked image name "
                            f"({readiness.describe()})")
        elif unknown:
            stock = unknown
            warnings.append(f"{unknown} of {readiness.tasks} trials will install the harness themselves (stock "
                            "setup): their task has no docker_image to build a pre-baked image from")
        if readiness.over_age:
            warnings.append(f"{len(readiness.over_age)} pre-baked image(s) were baked more than "
                            f"{cfg.prebake.max_age_days} days ago, so their package lists may be out of date. "
                            "A task race uses a ready image as it is: rebuild them first "
                            "(python -m bench.prebake build --rebuild), or remove them")
    elif readiness is not None:
        not_ready = len(readiness.not_ready) + len(readiness.unknown)
        if not_ready and arm_prebake.required:
            problems.append(f"run.prebake is require, but {not_ready} of {readiness.tasks} pre-baked images are not "
                            f"ready ({readiness.describe()}). Build them first: Build missing on LAUNCH, or "
                            "python -m bench.prebake build")
        elif not_ready:
            stock = not_ready
            typical = _stock_setup_sec(arm.family)
            warnings.append(f"{not_ready} of {readiness.tasks} trials will install the harness themselves "
                            f"(stock setup{f', about {typical:.0f} s each' if typical else ''}): no ready pre-baked "
                            f"image ({readiness.not_ready_text()}). Build missing images first to avoid it")
        if readiness.over_age:
            warnings.append(f"{len(readiness.over_age)} pre-baked image(s) were baked more than "
                            f"{cfg.prebake.max_age_days} days ago, so their package lists may be out of date. "
                            "Rebuild them: python -m bench.prebake build --rebuild")
    if stock and arm.family == "claude-code" and peak > 3:
        warnings.append(f"Claude Code installs itself in {stock} trial(s). Its install took 330-550 s under parallel "
                        f"load, and 13 of 23 such trials hit Harbor's 360 s setup limit. With up to {peak} "
                        "containers at once, build its pre-baked images first or run fewer at once")
    return problems, warnings


def _images_summary(cfg: Config, spec: RunSpec, found: dict[str, Any], error: str | None) -> dict[str, Any]:
    """The plan's top-level 'prebake': what building the arms' missing images
    would take (de-duplicated across arms, wall time at prebake.build_jobs),
    and the task images that are not local yet (the Pull button)."""
    out: dict[str, Any] = {"mode": spec.run.prebake, "error": error, "images": 0, "build_min": 0.0,
                           "build_wall_min": 0.0, "build_gb": 0.0, "jobs": cfg.prebake.build_jobs,
                           "task_images": 0, "pull": [], "pull_error": None}
    if found:
        try:
            from .prebake import builder
            plan = builder.plan_builds(cfg, spec, found=found)
            out.update(images=len(plan.items), build_min=plan.minutes, build_gb=plan.need_gb,
                       build_wall_min=plan.wall_minutes(cfg.prebake.build_jobs))
        except Exception as exc:  # advice for the buttons; the per-arm blocks still stand
            out["error"] = out["error"] or f"{type(exc).__name__}: {exc}"
    if spec.run.schedule == SCHEDULE_TASKS:
        out["race"] = race_images(cfg, spec, found)
    try:
        from .prebake import inventory, pull
        images = [i for i in inventory.task_images(spec.dataset, (t for a in spec.arms for t in a.tasks)).values() if i]
        out["task_images"] = len(set(images))
        out["pull"] = pull.missing(images, refresh=False) if images else []
    except Exception as exc:
        out["pull_error"] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
    return out


def race_tasks(spec: RunSpec) -> list[str]:
    """The tasks a task race runs, in order: the spec's, then any an arm adds."""
    wanted = {t for a in spec.arms for t in a.tasks}
    return [t for t in dict.fromkeys([*spec.tasks, *(t for a in spec.arms for t in a.tasks)]) if t in wanted]


def race_images(cfg: Config, spec: RunSpec, found: dict[str, Any]) -> dict[str, Any]:
    """Per task of a task race: the arms that race on it, the pre-baked
    images it needs and how many must be built first (RACE_BUILD_JOBS at
    once), with their minutes and the disk one task's images take. The peak
    disk is the largest task's images; nothing else is kept between tasks."""
    try:
        from .prebake.adapters import ADAPTERS
        from .prebake.builder import DISK_FACTOR
    except Exception:  # no estimates without the adapters; the race still runs
        ADAPTERS, DISK_FACTOR = {}, 1.0
    from . import stats

    per_task = []
    for task in race_tasks(spec):
        arms = [a.name for a in spec.arms if task in a.tasks]
        images: dict[str, tuple[str, bool]] = {}   # image -> (family, needs a build)
        for name in arms:
            r = found.get(name)
            want = r.wants.get(task) if r is not None and getattr(r, "supported", False) else None
            if want is None:
                continue
            build = task not in r.ready
            family, before = images.get(want.image, (r.family, False))
            images[want.image] = (family, before or build)
        minutes = [float(getattr(ADAPTERS.get(f), "expected_build_min", 0.0)) for f, b in images.values() if b]
        gb = sum(float(getattr(ADAPTERS.get(f), "expected_gb", 0.0)) for f, _ in images.values())
        per_task.append({"task": task, "arms": arms, "images": len(images), "build": len(minutes),
                         "build_min": round(sum(minutes), 1),
                         "build_wall_min": round(stats.lpt_makespan(minutes, RACE_BUILD_JOBS), 1),
                         "disk_gb": round(gb * DISK_FACTOR, 2)})
    walls = [t["build_wall_min"] for t in per_task]
    return {"tasks": len(per_task), "arms": len(spec.arms), "lanes": sum(len(t["arms"]) for t in per_task),
            "max_containers": spec.run.max_containers, "build_jobs": RACE_BUILD_JOBS,
            "images_to_build": sum(t["build"] for t in per_task),
            "build_wall_min_total": round(sum(walls), 1),
            "build_wall_min_max": max(walls, default=0.0),
            "peak_disk_gb": max((t["disk_gb"] for t in per_task), default=0.0),
            "keep_images": bool(cfg.prebake.keep_images), "per_task": per_task}


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


def _container_checks(cfg: Config, peak: int, problems: list[str], warnings: list[str],
                      spec: RunSpec | None = None) -> None:
    limits = cfg.designs
    race = spec is not None and spec.run.schedule == SCHEDULE_TASKS
    if race:
        assert spec is not None
        lane = race_lane_slots(spec.run.n_concurrent, spec.n_attempts)
        if lane > spec.run.max_containers:
            problems.append(f"One arm's job on a task runs up to {lane} containers (n_concurrent, at most the "
                            f"attempts), above run.max_containers ({spec.run.max_containers})")
        if spec.run.max_containers > limits.max_containers:
            problems.append(f"run.max_containers ({spec.run.max_containers}) is above designs.max_containers "
                            f"({limits.max_containers})")
    rule = "the arms racing on one task, at most run.max_containers" if race else "parallel_arms x n_concurrent"
    if peak > limits.max_containers:
        problems.append(f"Up to {peak} task containers would run at once ({rule}); "
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


def _path_warning(tasks: list[str], race: bool = False) -> list[str]:
    if not _WINDOWS or not tasks:
        return []
    # jobs dir + run id (at most 44) + arm name (at most 32) + task folder +
    # the deepest file Harbor writes under a trial (about 99 characters); a
    # task race adds the task's own job folder.
    longest = max(len(t) for t in tasks)
    projected = len(str(JOBS_DIR)) + MAX_RUN_ID + 32 + longest + 99 + (longest + 1 if race else 0)
    if projected <= 250 or _long_paths_enabled():
        return []
    return [f"Trial files may reach about {projected} characters, past Windows' 260-character path limit, "
            "and long paths are not enabled. Turn on LongPathsEnabled or move the bench to a shorter folder"]


def plan_spec(cfg: Config, spec: RunSpec, found: tuple[dict[str, Any], str | None] | None = None) -> dict[str, Any]:
    """Everything a run would do, with the problems that block it and the
    warnings worth reading first. Shared by the LAUNCH tab, run designs and
    the queue. found is image_readiness(cfg, spec), when the caller has it."""
    arms: list[dict[str, Any]] = []
    problems: list[str] = []
    warnings: list[str] = []
    image_problems: list[str] = []
    uncapped: list[str] = []
    worst_case = 0.0
    snapshot = datasets.path(spec.dataset)
    display_path = _display_path(snapshot) if snapshot else Path("datasets") / spec.dataset
    race = spec.run.schedule == SCHEDULE_TASKS
    peak = (race_peak(spec.run.n_concurrent, spec.run.max_containers, spec.n_attempts,
                      [list(a.tasks) for a in spec.arms])
            if race else min(spec.run.parallel_arms, len(spec.arms)) * spec.run.n_concurrent)
    found, found_error = found if found is not None else image_readiness(cfg, spec)
    for arm in spec.arms:
        per_arm = len(arm.tasks) * spec.n_attempts
        provider = cfg.providers.get(arm.provider)
        entry: dict[str, Any] = {
            "name": arm.name, "agent": arm.agent, "family": arm.family, "kind": arm.kind,
            "provider": arm.provider,
            "model": cfg.harbor_model(arm) if provider else arm.model,  # type: ignore[arg-type]
            "tasks": len(arm.tasks), "trials": per_arm,
            "effort": None, "kwargs": [], "options_fingerprint": None, "command": "", "reserve_note": "",
            "prebake": None, "problems": [], "warnings": [], "setup_warnings": [],
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
            arm_prebake, ready = _arm_prebake(cfg, spec.run, arm, kwargs, found)
            entry["prebake"] = prebake_entry(spec.run, arm_prebake, ready, found_error)
            if arm_prebake is not None and arm_prebake.required and not arm_prebake.supported:
                entry["problems"].append(f"run.prebake is require, but this arm cannot use pre-baked images: "
                                         f"{arm_prebake.reason}")
            arm_problems, arm_warnings = _prebake_issues(cfg, spec, arm, arm_prebake, ready, found_error, peak)
            entry["problems"] += arm_problems
            entry["warnings"] += arm_warnings
            entry["setup_warnings"] = arm_warnings  # which of the warnings are about the harness setup
            image_problems += [f"{arm.name}: {p}" for p in arm_problems]
            if provider is not None:
                entry["command"] = preview_command(cfg, spec, arm, kwargs, display_path, arm_prebake=arm_prebake)
        except (ConfigError, ValueError, TypeError, RuntimeError) as exc:
            entry["problems"].append(str(exc))
        problems += [f"{arm.name}: {p}" for p in entry["problems"]]
        arms.append(entry)

    if not spec.arms:
        problems.append("Nothing to run: the ledger already holds enough usable trials of every task for every arm"
                        if spec.spec.get("reuse") else "Nothing to run: no arm has a task")
    all_tasks = list(dict.fromkeys(t for arm in spec.arms for t in arm.tasks))
    _container_checks(cfg, peak, problems, warnings, spec)
    warnings += _memory_warning(spec, all_tasks, peak)
    warnings += _path_warning(all_tasks, race)
    images = _images_summary(cfg, spec, found, found_error)
    if images["pull"]:
        warnings.append(f"{len(images['pull'])} task image(s) are not on this machine yet. "
                        + ("The run pulls them before its first arm (prebake.prepull); a 15-20 GB image took "
                           "6-9 minutes the first time." if cfg.prebake.prepull else
                           "Each trial pulls its own within Harbor's 600 s environment limit, which a 15-20 GB "
                           "image nearly used up the first time. Pull them first."))
    return {
        "tasks": list(spec.tasks), "n_attempts": spec.n_attempts,
        "dataset": spec.dataset, "dataset_snapshot": datasets.info(spec.dataset),
        "trials": sum(a["trials"] for a in arms), "budget_usd": spec.run.budget_usd,
        "worst_case_capped_usd": worst_case, "uncapped_arms": uncapped,
        "arms": arms, "problems": problems,
        "source": spec.source, "label": spec.label, "spec_hash": spec.spec_hash,
        "warnings": warnings, "peak_containers": peak,
        # Arms whose every task is reused from the ledger (spec.reused): recorded, never started.
        "skipped_arms": [a.name for a in spec.skipped_arms],
        # Building the missing pre-baked images and pulling task images; image_problems are the problems about
        # images that are not ready under require, which a design may still be queued with (designs.validate).
        "prebake": images, "image_problems": image_problems,
        # How the trials are ordered (run.schedule); under tasks, the race block: tasks x arms, images per task,
        # build minutes per task and the peak disk (prebake.race).
        "schedule": spec.run.schedule,
        "race": _race_summary(spec, images) if race else None,
    }


def _race_summary(spec: RunSpec, images: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in (images.get("race") or {}).items() if k != "per_task"}
    out.setdefault("tasks", len(race_tasks(spec)))
    out.setdefault("arms", len(spec.arms))
    out.setdefault("max_containers", spec.run.max_containers)
    out["n_concurrent"] = spec.run.n_concurrent
    out["note"] = (f"{out['tasks']} task race(s) x {out['arms']} arm(s): each task's pre-baked images are built "
                   "just before its race and removed after it"
                   + (" (kept: prebake.keep_images)" if out.get("keep_images") else "")
                   + f"; at most {spec.run.max_containers} containers race at once. parallel_arms does not apply.")
    out["per_task"] = (images.get("race") or {}).get("per_task", [])
    return out


def _arm_prebake(cfg: Config, run: RunSettings, arm: ArmSpec, kwargs: dict[str, Any],
                 found: dict[str, Any], planning: bool = True) -> tuple[ArmPrebake | None, Any]:
    """The arm's prebake and its readiness (None when it has none); every
    task's image ready means no trial expects a stock setup."""
    arm_prebake = prebake.resolve(cfg, run, arm, kwargs)
    ready = found.get(arm.name)
    if arm_prebake is None or not arm_prebake.supported or ready is None or not ready.supported:
        return arm_prebake, None
    # A task race builds every task's missing image right before its race, so only a task it cannot name an
    # image for runs stock (_run_arm checks each task's image again before its lane starts).
    built_first = planning and run.schedule == SCHEDULE_TASKS and not ready.unknown and not ready.docker_error
    if ready.all_ready or built_first:
        arm_prebake = dataclasses.replace(arm_prebake, expect_stock=False)
    return arm_prebake, ready


def plan(cfg: Config, mode_name: str, arm_names: list[str]) -> dict[str, Any]:
    """The LAUNCH tab's preview: plan_spec for the mode and arms, plus "mode"."""
    return {"mode": mode_name, **plan_spec(cfg, spec_from_config(cfg, mode_name, arm_names))}


def _harbor_version() -> str:
    try:
        return importlib.metadata.version("harbor")
    except importlib.metadata.PackageNotFoundError:
        return ""


def resolve_arms(cfg: Config, spec: RunSpec, run_id: str,
                 found: tuple[dict[str, Any], str | None] | None = None) -> list[dict[str, Any]]:
    """Every arm as it will run, for run.json. Readers (results, the ledger)
    use this rather than the config, which may have changed since. found is
    image_readiness(cfg, spec) when the caller has it (start_spec's plan)."""
    snapshot = datasets.path(spec.dataset)
    display_path = _display_path(snapshot) if snapshot else Path("datasets") / spec.dataset
    job_root = _display_path(JOBS_DIR / run_id)
    readiness, error = found if found is not None else image_readiness(cfg, spec)
    out = []
    for arm in spec.arms:
        schema = schemas.agent_schema(arm.agent)
        kwargs, source = agent_kwargs(spec.run, arm, schema)
        value, label = resolve_effort(spec.run, arm, schema)
        arm_prebake, ready = _arm_prebake(cfg, spec.run, arm, kwargs, readiness)
        out.append({
            "name": arm.name, "agent": arm.agent, "family": arm.family, "kind": arm.kind,
            "provider": arm.provider, "model": arm.model,
            "harbor_model": cfg.harbor_model(arm),  # type: ignore[arg-type]
            "effort": {"setting": arm.effort, "value": value, "label": label},
            "kwargs": [{"name": k, "value": v, "source": source[k]} for k, v in kwargs.items()],
            "options_fingerprint": options_fingerprint(kwargs, source),
            "tasks": list(arm.tasks),
            "trials": len(arm.tasks) * spec.n_attempts,
            "command": preview_command(cfg, spec, arm, kwargs, display_path, job_root, arm_prebake=arm_prebake),
            "prebake": prebake_entry(spec.run, arm_prebake, ready, error),
        })
    return out


def _readiness_line(arm_prebake: ArmPrebake | None, ready: Any, error: str | None) -> str:
    """The arm log's line on its images, written right before it launches."""
    required = arm_prebake is not None and arm_prebake.required
    if ready is None:
        return (f"not checked{f' ({error})' if error else ''}; a trial whose image is not usable "
                + ("fails fast (require)" if required else "runs stock setup"))
    text = ready.describe()
    if ready.all_ready:
        return f"{text}. Every trial starts from its pre-baked image"
    names = [*ready.missing, *ready.stale, *ready.unverifiable, *ready.unknown]
    shown = ", ".join(names[:10]) + (" ..." if len(names) > 10 else "")
    then = "fail fast as infrastructure errors (require)" if required else "run Harbor's stock install"
    return f"{text}. These trials {then}: {shown}"


def _run_lock() -> contextlib.ExitStack:
    """The profile's run lock (prebake.builder.run_lock), held until the
    stack is closed. Raises BuildRefused, a RuntimeError, while a pre-baked
    image build (from the dashboard or the CLI) is running."""
    from .prebake import builder

    stack = contextlib.ExitStack()
    stack.enter_context(builder.run_lock())
    return stack


_SETUP_WHY = {"missing": "no pre-baked image", "stale": "stale image", "unverifiable": "task image not local",
              "not-applicable": "not applicable"}


def setup_note(records: list[dict[str, Any]]) -> tuple[int, str]:
    """(trials that did not get their pre-baked image, the arm-message note)
    from the trials' prebake.json records. Under prefer such a trial ran
    Harbor's stock install; under require it failed fast as infrastructure."""
    stock: dict[str, int] = {}
    failed = 0
    for record in records:
        if record.get("used") is not False:
            continue
        if record.get("required"):
            failed += 1
        else:
            why = _SETUP_WHY.get(str(record.get("status")), str(record.get("status") or "unknown"))
            stock[why] = stock.get(why, 0) + 1
    notes = []
    if stock:
        n = sum(stock.values())
        reasons = ", ".join(f"{why} x{k}" if len(stock) > 1 else why for why, k in sorted(stock.items()))
        notes.append(f"{n} trial{'s' if n != 1 else ''} ran stock setup ({reasons})")
    if failed:
        notes.append(f"{failed} trial{'s' if failed != 1 else ''} failed fast: no usable pre-baked image "
                     "(run.prebake require)")
    return sum(stock.values()) + failed, "; ".join(notes)


class _Slots:
    """Containers a task race may run at once. Lanes take their slots in
    ticket (arm) order, whole or not at all, and two harbor starts are at
    least `stagger` seconds apart."""

    def __init__(self, total: int, stagger: float = 0.0) -> None:
        self.free = max(1, int(total))
        self.total = self.free
        self.stagger = stagger
        self.turn = 0
        self.last_start = 0.0
        self.cond = threading.Condition()

    def acquire(self, ticket: int, width: int, cancel: threading.Event) -> bool:
        width = min(width, self.total)
        with self.cond:
            while not (self.turn == ticket and self.free >= width):
                if cancel.is_set():
                    self.turn = max(self.turn, ticket + 1)
                    self.cond.notify_all()
                    return False
                self.cond.wait(0.2)
            self.free -= width
            self.turn += 1
            now = time.monotonic()
            delay = max(0.0, self.last_start + self.stagger - now) if self.last_start else 0.0
            self.last_start = now + delay
            self.cond.notify_all()
        if delay:
            cancel.wait(delay)  # a cancel meanwhile: the lane sees it and releases its slots
        return True

    def release(self, width: int) -> None:
        with self.cond:
            self.free = min(self.total, self.free + min(width, self.total))
            self.cond.notify_all()


def _interrupt_race(run: Run) -> list[str]:
    """A loaded run's race left open (the bench stopped): its open tasks and
    lanes are marked interrupted. Returns the arms whose lanes may have left
    containers, for the startup sweep."""
    arms = []
    for entry in run.race.get("tasks") or [] if isinstance(run.race, dict) else []:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") in ("building", "racing"):
            entry["status"] = "interrupted"
        for name, lane in (entry.get("lanes") or {}).items():
            if isinstance(lane, dict) and (lane.get("status") in _LIVE or lane.get("cleanup_pending")):
                if lane.get("status") in _LIVE:
                    lane["status"] = "interrupted"
                arms.append(name)
    if arms and isinstance(run.race, dict):
        run.race["current"] = None
    return list(dict.fromkeys(arms))


def images_needed(cfg: Config, specs: list[RunSpec]) -> set[str]:
    """The pre-baked images these (waiting queue) runs will look for without
    building them themselves: those of arms-schedule runs. A task race builds
    each task's images just before its race, so it needs none kept for it."""
    from .prebake import inventory

    out: set[str] = set()
    for spec in specs:
        if spec.run.schedule != SCHEDULE_TASKS and spec.run.prebake != PREBAKE_OFF:
            out |= set(inventory.wanted_images(cfg, spec).values())
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
    setup_fallbacks: int = 0       # trials that ran Harbor's stock install (or failed fast under require)
    setup_note: str = ""           # "N trials ran stock setup (why)", from the trials' prebake.json


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
    schedule: str = "arms"   # run.schedule as it ran; runs from before it existed ran as arms
    # schedule tasks: {current, max_containers, tasks: [{task, status (waiting, building, racing, done, cancelled,
    # skipped, interrupted), arms, lanes: {arm: ArmRun of that task's job}, images: {arm: how its image stands},
    # build, removed, kept, started_at, finished_at}]}.
    race: dict[str, Any] = field(default_factory=dict)

    @property
    def spend_usd(self) -> float:
        return sum(a.spend_usd for a in self.arms)

    @property
    def reserve_usd(self) -> float:
        return sum(a.reserve_usd for a in self.arms)

    def to_dict(self) -> dict[str, Any]:
        for _ in range(5):  # a task race's lanes update their state while it is copied
            try:
                data = asdict(self)
                break
            except RuntimeError:
                time.sleep(0.01)
        else:
            data = asdict(self)
        data["spend_usd"] = self.spend_usd
        data["reserve_usd"] = self.reserve_usd
        return data


_RUN_FIELDS = frozenset(f.name for f in dataclasses.fields(Run)) - {"arms"}
_ARM_FIELDS = frozenset(f.name for f in dataclasses.fields(ArmRun))


class RunManager:
    def __init__(self, ledger: LedgerHook | None = None, builds: Any = None) -> None:
        self.ledger = ledger
        # The prebake BuildManager a task race builds through (the dashboard's, so the IMAGES panel shows its
        # builds); made on first use when none is given.
        self.builds = builds
        # keep_images(cfg) -> pre-baked image names a waiting queue item still needs, so a task race keeps them.
        self.keep_images: Callable[[Config], set[str]] | None = None
        self._lanes: dict[int, tuple[ArmRun, list[ArmRun]]] = {}  # id(lane) -> (its arm's ArmRun, the arm's lanes)
        self._arm_lanes: dict[int, list[ArmRun]] = {}  # id(an arm's ArmRun) -> its task lanes, in task order
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
        self._setup_seen: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}  # (run, arm) -> trial -> prebake.json
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
            # Held for the run's life, so no pre-baked image build starts meanwhile (a build slows the trials'
            # containers and could retag an image under them). Raises (409) while a build is running.
            locks = _run_lock()
            try:
                found = image_readiness(cfg, spec, refresh=True)
                p = plan_spec(cfg, spec, found)
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
                    resolved_arms=resolve_arms(cfg, spec, run_id, found),
                    harbor_version=_harbor_version(),
                    schedule=spec.run.schedule,
                )
                if spec.run.schedule == SCHEDULE_TASKS:
                    self._race_record(spec, run)
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
                    target=self._execute, args=(cfg, spec, run, locks), name=f"run-{run.id}", daemon=True
                )
                self._thread.start()
            except BaseException:
                locks.close()
                raise
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
            builds = self.builds
            if builds is not None and hasattr(builds, "emergency_stop"):
                # A task race's build (or a manual one): its docker builds die, and its cache prune is kept for later.
                killers.append(threading.Thread(target=builds.emergency_stop, daemon=True, name="emergency-build"))
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

    def _execute(self, cfg: Config, spec: RunSpec, run: Run, locks: contextlib.ExitStack | None = None) -> None:
        try:
            log_path = RUNS_DIR / run.id / "dataset.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cancelled = ""
            try:
                with open(log_path, "a", encoding="utf-8") as dataset_log:
                    dataset_path = datasets.ensure(spec.dataset, dataset_log, cancel=self._cancel)
            except datasets.DownloadCancelled:
                dataset_path, cancelled = None, "Cancelled during the dataset download"
            # The 'images' step: task images not on this machine are pulled now, before Harbor's 600 s
            # environment build limit would count a first pull of 15-20 GB against a trial.
            if dataset_path is not None and cfg.prebake.prepull and not self._pull_images(spec, run):
                dataset_path, cancelled = None, "Cancelled while pulling task images"
            if cancelled:
                run.status = "cancelled"
                run.message = cancelled
                for a in run.arms:
                    if a.status in _LIVE:
                        a.status = "cancelled"
            if dataset_path is not None:
                run.dataset = datasets.info(spec.dataset) or {}
                self._save(run)
                if spec.run.schedule == SCHEDULE_TASKS:
                    self._race_all(cfg, spec, run, dataset_path)
                else:
                    # Arms start in spec order, run.parallel_arms at a time. Every
                    # arm checks the shared run budget before it starts and while it runs.
                    with ThreadPoolExecutor(max_workers=spec.run.parallel_arms,
                                            thread_name_prefix=f"arm-{run.id}") as pool:
                        futures = [pool.submit(self._arm_slot, cfg, spec, run, arm, arm_run, dataset_path,
                                               ARM_STAGGER_SECONDS * i if i < spec.run.parallel_arms else 0)
                                   for i, (arm, arm_run) in enumerate(zip(spec.arms, run.arms))]
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
            self._finish_race(run)
            if locks is not None:
                locks.close()  # a build may start once this run is over
            with self._lock:
                self._active = None
                for lane_id in self._lane_set(run):
                    parent = self._lanes.pop(lane_id, (None, []))[0]
                    self._arm_lanes.pop(id(parent), None)
            self._save(run, patience=30.0)  # the final record matters most
            self._ledger_call(None, "upsert_bench_run", run.to_dict())
            self._notify(run)

    def _pull_images(self, spec: RunSpec, run: Run) -> bool:
        """Pulls the spec's task images that are not local (prebake.prepull),
        logging to images.log. Returns False when the run was cancelled
        meanwhile. A pull that fails, or Docker not answering, is logged and
        the run goes on: each trial then pulls its own image, as it always has."""
        from .prebake import inventory, pull

        log_path = RUNS_DIR / run.id / "images.log"
        try:
            names = inventory.task_images(spec.dataset, (t for a in spec.arms for t in a.tasks))
            images = sorted({i for i in names.values() if i})
            if not images:
                return True
            todo = pull.missing(images)
            if not todo:
                _append_log(log_path, f"all {len(images)} task image(s) are local")
                return True
            run.message = f"Pulling {len(todo)} task image(s) before the first arm"
            self._save(run)
            with open(log_path, "a", encoding="utf-8") as log:
                result = pull.ensure_local(todo, log, cancel=self._cancel)
            failed = result.get("failed") or {}
            for image, error in failed.items():  # type: ignore[union-attr]
                _append_log(log_path, f"could not pull {image}: {error}; its trials pull it themselves")
        except pull.PullCancelled:
            return False
        except Exception as exc:  # a head start, never a reason to fail the run
            _append_log(log_path, f"task images not pulled ahead ({type(exc).__name__}: {exc}); "
                                  "each trial pulls its own image")
        finally:
            if run.message.startswith("Pulling "):
                run.message = ""
        return not self._cancel.is_set()

    def _notify(self, run: Run) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(run.to_dict())
            except Exception:
                log.exception("A run listener failed for run %s", run.id)

    def _arm_slot(self, cfg: Config, spec: RunSpec, run: Run, arm: ArmSpec, arm_run: ArmRun,
                  dataset_path: Path, delay: float = 0) -> None:
        if delay:
            self._cancel.wait(delay)
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
                 dataset_path: Path, job_root: Path | None = None, job_name: str | None = None) -> None:
        """One harbor job: the arm's (jobs/<run>/<arm>/), or in a task race
        one task's lane, with job_root jobs/<run>/<arm> and the task as job_name."""
        schema = schemas.agent_schema(arm.agent)
        kwargs, _ = agent_kwargs(spec.run, arm, schema)
        arm_prebake = prebake.resolve(cfg, spec.run, arm, kwargs)
        env = keys.child_env(cfg.providers[arm.provider])
        job_root = job_root or JOBS_DIR / run.id
        job_name = job_name or arm.name
        job_dir = job_root / job_name
        log_path = RUNS_DIR / run.id / f"{arm.name}.log"
        with self._estimates_lock:
            self._trial_caps[(run.id, arm.name)] = _trial_cap(kwargs)
        arm_run.started_at = _now()

        arm_run.status = "preflight"
        self._save(run)
        # Again right before launch: an image may have been built, pruned or gone stale since the plan.
        ready = None
        if arm_prebake is not None and arm_prebake.supported:
            found, error = image_readiness(cfg, dataclasses.replace(spec, arms=(arm,)), refresh=True)
            arm_prebake, ready = _arm_prebake(cfg, spec.run, arm, kwargs, found, planning=False)
            _append_log(log_path, f"pre-baked images: {_readiness_line(arm_prebake, ready, error)}")
        _append_log(log_path, f"harness setup: {json.dumps(prebake_entry(spec.run, arm_prebake, ready))}")
        preflight = build_command(cfg, spec, arm, job_root, kwargs, dataset_path, dry_run=True,
                                  arm_prebake=arm_prebake, job_name=job_name)
        rc = self._supervise(preflight, env, log_path, run, arm_run, watch=None)
        for wait in PREFLIGHT_RETRY_WAITS:
            if rc == 0 or arm_run.status != "preflight" or not _docker_was_slow(log_path):
                break
            _append_log(log_path, f"Docker is running but answered harbor's check too slowly; trying again in {wait} s")
            if self._cancel.wait(wait):
                arm_run.status, arm_run.message = "cancelled", "Cancelled from the dashboard"
                break
            rc = self._supervise(preflight, env, log_path, run, arm_run, watch=None)
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
        rc = self._supervise(build_command(cfg, spec, arm, job_root, kwargs, dataset_path, arm_prebake=arm_prebake,
                                           job_name=job_name),
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
        if arm_run.setup_note:
            arm_run.message = f"{arm_run.message.rstrip('.')}. {arm_run.setup_note}.".lstrip(". ")
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
                key = (run.id, arm_run.name)
                # A task race refreshes one task's job at a time: the arm's largest trial spans its tasks.
                self._largest_trial[key] = max(max(known), self._largest_trial.get(key, 0.0))
        arm_run.setup_fallbacks, arm_run.setup_note = setup_note(self._setup_records(run.id, arm_run.name, job_dir))
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
        lane = self._lanes.get(id(arm_run))
        if lane is not None:
            self._roll_up(run, lane[0])
        return rows

    def _setup_records(self, run_id: str, arm_name: str, job_dir: Path) -> list[dict[str, Any]]:
        """The prebake.json of every trial folder of the job. A trial's record
        is written once, when its environment starts, so each is read once."""
        from .extract import read_prebake_record

        seen = self._setup_seen.setdefault((run_id, arm_name, str(job_dir)), {})
        try:
            folders = [d for d in job_dir.iterdir() if d.is_dir() and d.name not in seen]
        except OSError:
            folders = []
        for folder in folders:
            record = read_prebake_record(folder)
            if record is not None and "used" in record:
                seen[folder.name] = record
        return list(seen.values())

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

    # Task races ---------------------------------------------------------------

    def _race_record(self, spec: RunSpec, run: Run) -> None:
        """run.race with every task and lane in place before the run starts, so
        saving it never meets a dict that grows."""
        parents = {a.name: a for a in run.arms}
        by_arm: dict[str, list[ArmRun]] = {a.name: [] for a in run.arms}
        tasks = []
        for task in race_tasks(spec):
            arms = [a.name for a in spec.arms if task in a.tasks]
            lanes = {}
            for name in arms:
                lane = ArmRun(name=name, family=parents[name].family, trials_total=spec.n_attempts)
                lanes[name] = lane
                by_arm[name].append(lane)
            tasks.append({"task": task, "status": "waiting", "arms": arms, "lanes": lanes,
                          "images": {name: {"status": "waiting", "image": None, "message": ""} for name in arms},
                          "build": {"id": None, "status": None, "images": 0, "message": "", "cache": {}},
                          "removed": [], "kept": {}, "started_at": None, "finished_at": None})
        for name, lanes in by_arm.items():
            self._arm_lanes[id(parents[name])] = lanes
            for lane in lanes:
                self._lanes[id(lane)] = (parents[name], lanes)
        run.race = {"current": None, "max_containers": spec.run.max_containers, "done": 0, "tasks": tasks}

    @staticmethod
    def _lane_set(run: Run) -> set[int]:
        return {id(lane) for entry in run.race.get("tasks") or [] for lane in (entry.get("lanes") or {}).values()}

    def _race_builds(self) -> Any:
        if self.builds is None:
            from .prebake.builder import BuildManager

            self.builds = BuildManager(run_active=lambda: not self.is_idle())
        return self.builds

    def _race_all(self, cfg: Config, spec: RunSpec, run: Run, dataset_path: Path) -> None:
        """The task races, one task after another: build, race, remove images."""
        log_path = RUNS_DIR / run.id / "images.log"
        arms = {a.name: a for a in spec.arms}
        parents = {a.name: a for a in run.arms}
        order = [e["task"] for e in run.race["tasks"]]
        try:
            wanted: dict[tuple[str, str], str] | None = None
            if spec.run.prebake != PREBAKE_OFF:
                from .prebake import inventory
                wanted = inventory.wanted_images(cfg, spec)
        except Exception as exc:  # then nothing is removed: which images later tasks need is not known
            wanted = None
            _append_log(log_path, f"the images each task needs could not be worked out ({type(exc).__name__}: {exc}); "
                                  "no image is removed after its race")
        for i, entry in enumerate(run.race["tasks"]):
            if self._cancel.is_set() or self._budget_used(run):
                break
            task = entry["task"]
            task_arms = [dataclasses.replace(arms[name], tasks=(task,)) for name in entry["arms"]]
            entry["started_at"] = _now()
            run.race["current"] = task
            found: dict[str, Any] = {}
            try:
                entry["status"] = "building"
                self._save(run)
                found = self._race_build(cfg, spec, run, entry, task_arms, log_path)
                if self._cancel.is_set():
                    break
                entry["status"] = "racing"
                self._save(run)
                self._race_lanes(cfg, spec, run, entry, task_arms, parents, dataset_path)
            finally:
                self._race_remove(cfg, spec, run, entry, found, wanted, order[i + 1:], log_path)
                lanes = list(entry["lanes"].values())
                entry["status"] = ("cancelled" if self._cancel.is_set()
                                   else "budget_stopped" if any(x.status == "budget_stopped" for x in lanes)
                                   else "done")
                entry["finished_at"] = _now()
                run.race["done"] = sum(1 for e in run.race["tasks"] if e["finished_at"])
                self._save(run)
        run.race["current"] = None
        cancelled = self._cancel.is_set()
        for entry in run.race["tasks"]:
            if entry["status"] == "waiting":
                entry["status"] = "cancelled" if cancelled else "skipped"
            # Every lane that never started, also those of a task stopped while
            # its images were building, before the arms' statuses are worked out.
            for lane in entry["lanes"].values():
                if lane.status != "queued":
                    continue
                lane.status = "cancelled" if cancelled else "skipped"
                lane.message = "" if cancelled else "Run budget was used up before this task's race"
        for arm_run in run.arms:
            self._arm_outcome(arm_run, [e["lanes"][arm_run.name] for e in run.race["tasks"]
                                        if arm_run.name in e["lanes"]])

    def _race_build(self, cfg: Config, spec: RunSpec, run: Run, entry: dict[str, Any], task_arms: list[ArmSpec],
                    log_path: Path) -> dict[str, Any]:
        """Builds the task's missing pre-baked images, every arm's at once, and
        waits for them. Returns the task's readiness ({} under prebake off). A
        build that cannot start or fails leaves its arms as run.prebake says:
        stock setup under prefer, a fast infrastructure failure under require."""
        task = entry["task"]
        images = entry["images"]
        if spec.run.prebake == PREBAKE_OFF:
            for name in entry["arms"]:
                images[name].update(status="off", message="run.prebake is off: stock setup")
            entry["build"]["status"] = "off"
            return {}
        task_spec = dataclasses.replace(spec, arms=tuple(task_arms))
        found, error = image_readiness(cfg, task_spec, refresh=True)
        fallback = ("fails fast as infrastructure (require)" if spec.run.prebake == PREBAKE_REQUIRE
                    else "runs stock setup")
        for name in entry["arms"]:
            r = found.get(name)
            if r is None:
                images[name].update(status="unknown", message=error or "not checked")
            elif not r.supported:
                images[name].update(status="stock", message=r.reason)
            elif task in r.wants:
                images[name]["image"] = r.wants[task].image
                images[name]["status"] = "ready" if task in r.ready else "missing"
                images[name]["message"] = "" if task in r.ready else (
                    r.stale.get(task) or ("not checked: " + r.docker_error if r.docker_error else "no image yet"))
            else:
                images[name].update(status="stock", message=r.unknown.get(task, "no image name"))
        if not found:
            return found
        try:
            from .prebake import builder
            plan = builder.plan_builds(cfg, task_spec, label=f"run {run.id}: {task}", found=found)
        except Exception as exc:
            entry["build"].update(status="refused", message=f"{type(exc).__name__}: {exc}")
            _append_log(log_path, f"{task}: no build planned ({exc}); trials without an image {fallback}")
            return found
        if not plan.items:
            entry["build"]["status"] = "ready"
            _append_log(log_path, f"{task}: every pre-baked image is ready")
            return found
        building = {name for item in plan.items for name in item.arms}
        for name in building:
            images[name].update(status="building", message="")
        entry["build"]["images"] = len(plan.items)
        manager = self._race_builds()
        try:
            started = manager.start(cfg, plan, jobs=min(RACE_BUILD_JOBS, len(plan.items)), setup="once",
                                    owner_run=run.id, prune_cache=True)
        except Exception as exc:  # BuildRefused (disk, Docker, a build elsewhere): the race still runs
            why = " ".join(str(exc).split())[:400]
            entry["build"].update(status="refused", message=why)
            for name in building:
                images[name].update(status="not built", message=why)
            _append_log(log_path, f"{task}: the build did not start ({why}); those trials {fallback}")
            return found
        build_id = str(started.get("id"))
        entry["build"].update(id=build_id, status="running")
        _append_log(log_path, f"{task}: building {len(plan.items)} image(s) (build {build_id})")
        self._save(run)
        while not manager.wait(POLL_SECONDS):
            if self._cancel.is_set():
                with contextlib.suppress(RuntimeError, KeyError):
                    manager.cancel(build_id)
            self._race_build_state(manager, build_id, entry)
            self._save(run)
        self._race_build_state(manager, build_id, entry)
        result = entry["build"]
        _append_log(log_path, f"{task}: build {build_id} {result['status']}"
                              f"{': ' + result['message'] if result['message'] else ''}; build cache "
                              f"{json.dumps(result.get('cache') or {})}")
        failed = [n for n in building if images[n]["status"] != "built"]
        if failed:
            _append_log(log_path, f"{task}: no image for {', '.join(sorted(failed))}; those trials {fallback}")
        return found

    @staticmethod
    def _race_build_state(manager: Any, build_id: str, entry: dict[str, Any]) -> None:
        """Copies the build's per-image states onto the task's lanes."""
        try:
            build = manager.status(build_id) or {}
        except KeyError:
            return
        entry["build"].update(status=build.get("status"), message=build.get("message") or "",
                              cache=dict(build.get("cache") or {}))
        for item in build.get("items") or []:
            status = item.get("status")
            shown = ("built" if status == "promoted" else "failed" if status == "failed"
                     else "cancelled" if status == "cancelled" else "building")
            for name in item.get("arms") or []:
                if name in entry["images"]:
                    entry["images"][name].update(status=shown, message=item.get("message") or "")

    def _race_lanes(self, cfg: Config, spec: RunSpec, run: Run, entry: dict[str, Any], task_arms: list[ArmSpec],
                    parents: dict[str, ArmRun], dataset_path: Path) -> None:
        """Every arm of the task at once, one harbor job each, as many as fit
        in run.max_containers; the rest start, in arm order, as slots free up."""
        task = entry["task"]
        slots = _Slots(spec.run.max_containers, RACE_STAGGER_SECONDS)
        width = race_lane_slots(spec.run.n_concurrent, spec.n_attempts)
        with ThreadPoolExecutor(max_workers=max(1, len(task_arms)), thread_name_prefix=f"race-{run.id}") as pool:
            futures = [pool.submit(self._race_lane, cfg, spec, run, arm, parents[arm.name], entry["lanes"][arm.name],
                                   dataset_path, slots, n, width, task)
                       for n, arm in enumerate(task_arms)]
            for future in futures:
                future.result()

    def _race_lane(self, cfg: Config, spec: RunSpec, run: Run, arm: ArmSpec, parent: ArmRun, lane: ArmRun,
                   dataset_path: Path, slots: _Slots, ticket: int, width: int, task: str) -> None:
        if not slots.acquire(ticket, width, self._cancel):
            lane.status = "cancelled"
            self._roll_up(run, parent)
            return
        try:
            if self._cancel.is_set():
                lane.status = "cancelled"
            elif self._budget_used(run):
                lane.status = "skipped"
                lane.message = "Run budget was used up before this arm started on the task"
            else:
                parent.status = "running"
                parent.started_at = parent.started_at or _now()
                # Before harbor starts: a bench that dies from here on leaves the arm marked for the sweep.
                parent.cleanup_pending = True
                _append_log(RUNS_DIR / run.id / f"{arm.name}.log", f"--- task race: {task} ---")
                fsutil.mark_race(JOBS_DIR / run.id / arm.name, run.id, arm.name)
                try:
                    self._run_arm(cfg, spec, run, arm, lane, dataset_path, job_root=JOBS_DIR / run.id / arm.name,
                                  job_name=task)
                except Exception as exc:  # one lane failing must not stop the others
                    lane.status = "failed"
                    lane.message = f"{type(exc).__name__}: {exc}"
                    lane.finished_at = _now()
                    lane.trials_running = 0
                    lane.reserve_usd = 0.0
        finally:
            slots.release(width)
            self._roll_up(run, parent)
            self._save(run)

    def _roll_up(self, run: Run, parent: ArmRun) -> None:
        """An arm's totals across its task lanes: what the budget, the reserve
        and the dashboard read."""
        lanes = self._arm_lanes.get(id(parent)) or []
        if not lanes:
            return
        parent.trials_done = sum(x.trials_done for x in lanes)
        parent.spend_usd = round(sum(x.spend_usd for x in lanes), 4)
        parent.trials_running = sum(x.trials_running for x in lanes)
        parent.reserve_usd = round(sum(x.reserve_usd for x in lanes), 4)
        parent.trials_killed = sum(x.trials_killed for x in lanes)
        parent.trials_unmeasured = sum(x.trials_unmeasured for x in lanes)
        records = [r for (rid, name, _), seen in list(self._setup_seen.items())
                   if rid == run.id and name == parent.name for r in seen.values()]
        parent.setup_fallbacks, parent.setup_note = setup_note(records)
        started = [x for x in lanes if x.pid is not None]
        if started:
            parent.pid = started[-1].pid
        live = [x for x in lanes if x.status in ("preflight", "running")]
        parent.cleanup_pending = bool(live) or any(x.cleanup_pending for x in lanes)
        if parent.status in _LIVE:
            done = sum(1 for x in lanes if x.status not in ("queued",))
            parent.message = (f"Task {lanes.index(live[0]) + 1} of {len(lanes)} racing" if live
                              else f"{done} of {len(lanes)} tasks finished")

    def _arm_outcome(self, parent: ArmRun, lanes: list[ArmRun]) -> None:
        """An arm's final status and message from its task lanes."""
        self._roll_up_final(parent, lanes)
        statuses = [x.status for x in lanes]
        ran = [x for x in lanes if x.status not in ("queued", "skipped", "cancelled")]
        if not lanes:
            parent.status = "skipped"
        elif "cancelled" in statuses and self._cancel.is_set():
            parent.status = "cancelled"
        elif "budget_stopped" in statuses or ("skipped" in statuses and ran):
            parent.status = "budget_stopped"
        elif "skipped" in statuses:
            parent.status = "skipped"
        elif all(x == "done" for x in statuses):
            parent.status = "done"
        elif all(x == "failed" for x in statuses):
            parent.status = "failed"
        else:
            parent.status = "done_with_errors"
        parts = []
        stopped = next((x.message for x in lanes if x.status == "budget_stopped"), "")
        if stopped:
            parts.append(stopped.split(". ")[0])
        not_run = sum(1 for x in lanes if x.status in ("skipped", "cancelled"))
        if not_run and lanes:
            parts.append(f"{not_run} of {len(lanes)} tasks not started")
        failed = sum(1 for x in lanes if x.status == "failed")
        if failed:
            parts.append(f"{failed} of {len(lanes)} tasks failed (see the log)")
        errored = sum(1 for x in lanes if x.status == "done_with_errors")
        if errored:
            parts.append(f"{errored} task(s) had trials that errored before finishing")
        if parent.trials_killed:
            parts.append(_KILLED.format(n=parent.trials_killed).rstrip("."))
        if parent.trials_unmeasured:
            parts.append(_UNMEASURED.format(n=parent.trials_unmeasured).rstrip("."))
        if parent.setup_note:
            parts.append(parent.setup_note)
        parent.message = ". ".join(parts) + ("." if parts else "")
        rcs = [x.returncode for x in lanes if x.returncode not in (None, 0)]
        parent.returncode = rcs[-1] if rcs else (0 if ran else None)
        parent.finished_at = parent.finished_at or _now()
        parent.trials_running = 0
        parent.reserve_usd = 0.0

    def _roll_up_final(self, parent: ArmRun, lanes: list[ArmRun]) -> None:
        parent.trials_done = sum(x.trials_done for x in lanes)
        parent.spend_usd = round(sum(x.spend_usd for x in lanes), 4)
        parent.trials_killed = sum(x.trials_killed for x in lanes)
        parent.trials_unmeasured = sum(x.trials_unmeasured for x in lanes)
        parent.cleanup_pending = any(x.cleanup_pending for x in lanes)
        fallbacks = sum(x.setup_fallbacks for x in lanes)
        if fallbacks:
            parent.setup_fallbacks = fallbacks
            notes = [x.setup_note for x in lanes if x.setup_note]
            parent.setup_note = parent.setup_note or "; ".join(notes)

    def _race_remove(self, cfg: Config, spec: RunSpec, run: Run, entry: dict[str, Any], found: dict[str, Any],
                     wanted: dict[tuple[str, str], str] | None, later: list[str], log_path: Path) -> None:
        """After a task's race, removes its pre-baked images, except those a
        later task of this run, or a waiting queue item, still needs. Never a
        task image, never a name outside prebake.prefix (builder.remove_images)."""
        task = entry["task"]
        # The images this task had: ready before its build, or built by it. One that never existed needs no rmi.
        names = sorted({str(info["image"]) for info in entry["images"].values()
                        if info.get("image") and info.get("status") in ("ready", "built")})
        if not names or spec.run.prebake == PREBAKE_OFF:
            return
        if cfg.prebake.keep_images:
            entry["kept"] = {n: "prebake.keep_images" for n in names}
            return
        if wanted is None:
            entry["kept"] = {n: "the images later tasks need are not known" for n in names}
            return
        needed_later = {img for (_, t), img in wanted.items() if t in set(later)}
        try:
            waiting = self.keep_images(cfg) if self.keep_images is not None else set()
        except Exception as exc:
            waiting = None
            _append_log(log_path, f"{task}: the images waiting queue items need could not be worked out "
                                  f"({type(exc).__name__}: {exc}); this task's images are kept")
        kept: dict[str, str] = {}
        remove = []
        for name in names:
            if name in needed_later:
                kept[name] = "a later task of this run needs it"
            elif waiting is None:
                kept[name] = "the queue's needs are not known"
            elif name in waiting:
                kept[name] = "a waiting queue item needs it"
            else:
                remove.append(name)
        task_images = [w.task_image for r in found.values() for w in getattr(r, "wants", {}).values()]
        result: dict[str, str] = {}
        if remove:
            try:
                from .prebake import builder
                result = builder.remove_images(remove, cfg.prebake.prefix, task_images)
            except Exception as exc:
                result = {n: f"{type(exc).__name__}: {exc}" for n in remove}
        entry["removed"] = [n for n, r in result.items() if r in ("removed", "absent")]
        kept.update({n: r for n, r in result.items() if r not in ("removed", "absent")})
        entry["kept"] = kept
        _append_log(log_path, f"{task}: removed {len(entry['removed'])} image(s)"
                              + (f"; kept {json.dumps(kept)}" if kept else ""))

    def _finish_race(self, run: Run) -> None:
        """Lanes still open when a run ends early (an exception) are closed."""
        for entry in run.race.get("tasks") or []:
            if entry.get("status") in ("building", "racing"):
                entry["status"] = "cancelled" if self._cancel.is_set() else "interrupted"
            for lane in (entry.get("lanes") or {}).values():
                if isinstance(lane, ArmRun) and lane.status in _LIVE:
                    lane.status = "cancelled" if self._cancel.is_set() else "failed"
                    lane.trials_running = 0
                    lane.reserve_usd = 0.0
        if run.race:
            run.race["current"] = None

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
            for name in _interrupt_race(run):
                changed = True
                if name not in {a.name for a in sweep}:
                    sweep += [a for a in run.arms if a.name == name]
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
