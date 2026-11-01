"""Building pre-baked images: what a selection needs, built, verified, named.

A selection is a config mode and arms, a design version, or what remains of
a run, resolved through reuse (reuse=measured drops every task the ledger
already covers), so only images some trial will use are built. plan_builds()
asks inventory.readiness() which of them are missing or stale and turns
those into BuildItems, de-duplicated by (family, version, recipe, task image)
across arms and ordered longest first.

Per item, BuildManager:
1. pulls the task image if it is not local (all of them first, two at a time);
2. records the harness's install on it (recorder) and renders the Dockerfile
   and build context into prebake/ctx/<slug>/;
3. builds with `docker build --no-cache` (never --pull, no credentials) under
   a staging name, benchbake-staging/<build>-<n>/<the final name's path>;
4. verifies that image (verify.py), with the setup and equivalence checks
   where they are due;
5. promotes it: tags the final name, then removes the staging tag;
6. appends a line to prebake/builds.jsonl.
A failed or cancelled item removes only its own staging tag, so a final name
never points at an unverified image.

Builds and runs exclude each other: a build holds prebake/.build.lock
(procs.file_lock) and refuses while a run holds its profile's run lock
(run_lock(), which RunManager takes for a run's life) or a run.json of any
profile shows a live run; a run checks the build lock the same way. Before
building, disk_check() refuses unless the drive holding Docker Desktop's disk
image and Docker's VM both keep their reserve after the build.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import heapq
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from .. import designs, fsutil, keys, procs, reuse, settings
from ..environments.prebaked import LABEL_NS, Want, bake_labels, prebaked_image_name
from ..settings import PREBAKE_OFF, PREBAKE_PREFER, PREBAKE_REQUIRE, Config, ConfigError
from ..spec import RunSpec, spec_from_design, spec_from_launch
from . import PREBAKE_DIR, inventory, pull, verify
from .inventory import ArmReadiness, DockerError, docker, tail

STAGING = "benchbake-staging"
DISK_FACTOR = 1.15            # needed space = the adapters' expected GB x this
BUILD_TIMEOUT_SEC = 3 * 3600
RUN_FRESH_SEC = 900           # a live run.json saved this recently counts as a run in progress
LOCK_WAIT_SEC = 3.0
CACHE_PENDING = "cache-pending.json"   # a cache prune the console closing cut short
CACHE_PENDING_SLACK_SEC = 120         # Docker's clock against ours, for that prune's window
PROBE_GRACE_SEC = 0.05        # a probe holds a lock for microseconds; a holder outlasts this
# auto: the first image of each harness and OS family in a build; once: the same, unless an earlier build's
# image of that recipe and OS family passed it (builds.jsonl), as a task race's per-task builds use it.
SETUP_MODES = ("auto", "once", "all", "none")
EQUIVALENCE_MODES = ("auto", "always", "none")
SAMPLES = ("per-os", "all")
_LIVE_RUN = ("queued", "preflight", "running")
_GB = 1024 ** 3
_jsonl_lock = threading.Lock()


class BuildRefused(RuntimeError):
    """A build (or a run) cannot start now; the message says why."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Selection ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """What to build for: a config mode and arms, a design version, or a run."""
    mode: str | None = None
    arms: tuple[str, ...] = ()
    design: str | None = None       # "<id>" (latest version) or "<id>@v<N>"
    run: str | None = None          # what remains of this run
    reuse: str | None = None        # off | measured; None: the design's own, measured for a run, off for a mode
    tasks: tuple[str, ...] = ()     # only these tasks

    def label(self) -> str:
        if self.design:
            base = f"design {self.design}"
        elif self.run:
            base = f"run {self.run}"
        else:
            base = f"mode {self.mode}"
        parts = [base]
        if self.arms:
            parts.append(f"arms {', '.join(self.arms)}")
        if self.tasks:
            parts.append(f"tasks {', '.join(self.tasks)}")
        if self.effective_reuse() == "measured":
            parts.append("reuse measured")
        return "; ".join(parts)

    def effective_reuse(self, spec: Mapping[str, Any] | None = None) -> str:
        if self.reuse:
            return self.reuse
        if self.run:
            return "measured"
        if self.design and spec is not None:
            return str(spec.get("reuse") or "off")
        return "off"


_DESIGN_REF = re.compile(r"(?P<id>[^@]+?)(?:@v?(?P<version>\d+))?")


def parse_design_ref(text: str) -> tuple[str, int | None]:
    match = _DESIGN_REF.fullmatch(text.strip())
    if not match or not designs.valid_design_id(match.group("id")):
        raise ConfigError(f"'{text}' is not a design id, or <id>@v<N>")
    version = match.group("version")
    return match.group("id"), None if version is None else int(version)


def ledger_rows(cfg: Config) -> list[dict[str, Any]]:
    """Every trial in report.analysis_rows form, for reuse. Opens the ledger
    as `python -m bench.ledger` does; raises BuildRefused when it cannot."""
    from .. import extract, report
    from ..ledger import Ledger

    ledger = Ledger(settings.LEDGER_DB, roots=extract.default_roots(cfg.ledger.extra_scan_roots))
    ledger.open()
    try:
        if not ledger.available:
            raise BuildRefused(f"Reusing measured trials needs the ledger: {ledger.error}")
        return report.analysis_rows(ledger, cfg)
    finally:
        ledger.close()


def selection_spec(cfg: Config, sel: Selection,
                   rows: Callable[[], Sequence[Mapping[str, Any]]] | None = None) -> RunSpec:
    """The RunSpec the selection would run now, its reuse resolved against
    the ledger. Raises ConfigError for a selection that names nothing."""
    rows = rows or (lambda: ledger_rows(cfg))
    if sum(x is not None for x in (sel.mode, sel.design, sel.run)) != 1:
        raise ConfigError("Pick one of a mode (--mode), a design (--design) or a run (--run)")
    if sel.mode is not None:
        names = list(sel.arms) or [a.name for a in cfg.arms if a.enabled]
        launch = designs.design_from_config(cfg, sel.mode, names, sel.effective_reuse())
        if sel.effective_reuse() == "measured":
            launch = reuse.resolve(cfg, launch, rows(), "measured")
        spec = spec_from_launch(launch, sel.mode)
    elif sel.design is not None:
        design_id, version = parse_design_ref(sel.design)
        doc = designs.DesignStore().version(design_id, version)
        stored = doc["spec"]
        mode = sel.effective_reuse(stored)
        resolved = reuse.resolve(cfg, stored, rows(), mode) if mode == "measured" else designs.with_reuse(stored, "off")
        spec = spec_from_design(resolved, {"design_id": design_id, "version": doc.get("version")})
    else:
        path = settings.RUNS_DIR / str(sel.run) / "run.json"
        if not fsutil.is_safe_segment(str(sel.run)) or not path.is_file():
            raise ConfigError(f"No run '{sel.run}' in {settings.RUNS_DIR}")
        run = json.loads(path.read_text(encoding="utf-8"))
        stored = designs.design_from_run(run)
        mode = sel.effective_reuse()
        resolved = reuse.resolve(cfg, stored, rows(), mode) if mode == "measured" else designs.with_reuse(stored, "off")
        spec = spec_from_launch(resolved, str(sel.run))
    return _narrowed(spec, sel)


def _narrowed(spec: RunSpec, sel: Selection) -> RunSpec:
    arms = list(spec.arms)
    if sel.arms and (sel.design or sel.run):
        unknown = sorted(set(sel.arms) - {a.name for a in (*spec.arms, *spec.skipped_arms)})
        if unknown:
            raise ConfigError(f"No arm named {', '.join(unknown)} in this {'design' if sel.design else 'run'}")
        arms = [a for a in arms if a.name in sel.arms]
    if sel.tasks:
        unknown = sorted(set(sel.tasks) - set(spec.tasks))
        if unknown:
            raise ConfigError(f"Not among the selection's tasks: {', '.join(unknown)}")
        arms = [dataclasses.replace(a, tasks=tuple(t for t in a.tasks if t in sel.tasks)) for a in arms]
    return dataclasses.replace(spec, arms=tuple(a for a in arms if a.tasks))


# Plan ---------------------------------------------------------------------------------


def _slug(text: str, limit: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")[:limit] or "image"


@dataclass
class BuildItem:
    """One image to build: a harness version and recipe on one task image."""
    family: str
    version: str
    recipe: str
    variant: str
    agent: str
    kwargs: dict[str, Any]
    model_name: str
    task_image: str
    prefix: str
    dataset: str
    tasks: list[str] = field(default_factory=list)
    arms: list[str] = field(default_factory=list)
    why: str = ""
    expected_min: float = 0.0
    expected_gb: float = 0.0

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.family, self.version, self.recipe, self.task_image)

    @property
    def image(self) -> str:
        return prebaked_image_name(self.prefix, self.family, self.version, self.recipe, self.task_image)

    def want(self, prefix: str | None = None) -> Want:
        return Want(harness=self.family, version=self.version, recipe=self.recipe, task_image=self.task_image,
                    prefix=prefix or self.prefix)

    @property
    def slug(self) -> str:
        return _slug(f"{self.family}-{self.version}-r{self.recipe[:10]}__{self.task_image}")

    def adapter(self) -> Any:
        from .adapters import adapter_for

        return adapter_for(self.agent, self.family, self.kwargs, self.model_name)

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out.pop("kwargs")
        out["image"] = self.image
        return out


@dataclass
class BuildPlan:
    label: str
    items: list[BuildItem]
    ready: list[dict[str, Any]] = field(default_factory=list)        # {arm, task, image}
    unsupported: list[dict[str, Any]] = field(default_factory=list)  # {arm, reason}
    unknown: list[dict[str, Any]] = field(default_factory=list)      # {arm, task, reason}
    notes: list[str] = field(default_factory=list)
    readiness: dict[str, ArmReadiness] = field(default_factory=dict, repr=False)
    ready_items: list[BuildItem] = field(default_factory=list, repr=False)   # the ready images, for verify

    @property
    def minutes(self) -> float:
        return round(sum(i.expected_min for i in self.items), 1)

    @property
    def need_gb(self) -> float:
        return round(sum(i.expected_gb for i in self.items) * DISK_FACTOR, 2)

    def wall_minutes(self, jobs: int) -> float:
        """Longest-first onto `jobs` builders, as the build schedules them."""
        loads = [0.0] * max(1, jobs)
        for item in sorted(self.items, key=lambda i: -i.expected_min):
            heapq.heapreplace(loads, loads[0] + item.expected_min)
        return round(max(loads), 1)

    def to_dict(self, jobs: int = 4) -> dict[str, Any]:
        return {"label": self.label, "items": [i.to_dict() for i in self.items], "ready": self.ready,
                "unsupported": self.unsupported, "unknown": self.unknown, "notes": self.notes,
                "images": len(self.items), "minutes": self.minutes, "wall_minutes": self.wall_minutes(jobs),
                "jobs": jobs, "need_gb": self.need_gb,
                "readiness": {k: v.to_dict() for k, v in self.readiness.items()}}


def plan_builds(cfg: Config, spec: RunSpec, *, label: str = "", rebuild: bool = False,
                refresh: bool = False, found: Mapping[str, ArmReadiness] | None = None) -> BuildPlan:
    """The images spec's arms would use that are not ready (all of them with
    rebuild), one item per (family, version, recipe, task image), longest
    first. An arm whose run.prebake is off is planned as under prefer. found
    is inventory.readiness(cfg, spec) when the caller has it already."""
    notes = []
    if spec.run.prebake == PREBAKE_OFF:
        notes.append("run.prebake is off for this selection, so its runs would not use these images; "
                     "they are planned as under prefer")
        spec = dataclasses.replace(spec, run=dataclasses.replace(spec.run, prebake=PREBAKE_PREFER))
        found = None
    found = dict(found) if found is not None else inventory.readiness(cfg, spec, refresh=refresh)
    plan = BuildPlan(label=label or spec.label, items=[], notes=notes, readiness=found)
    errors = sorted({r.docker_error for r in found.values() if r.docker_error})
    plan.notes += [f"Docker did not answer ({e}), so no image is known to be ready" for e in errors]
    items: dict[tuple[str, str, str, str], BuildItem] = {}
    ready: dict[tuple[str, str, str, str], BuildItem] = {}
    from .adapters import ADAPTERS

    for arm in spec.arms:
        r = found[arm.name]
        if not r.supported:
            plan.unsupported.append({"arm": arm.name, "family": arm.family, "reason": r.reason})
            continue
        for task, why in r.unknown.items():
            plan.unknown.append({"arm": arm.name, "task": task, "reason": why})
        cls = ADAPTERS[r.family]
        for task, want in r.wants.items():
            decision = r.decisions.get(task)
            used = decision is not None and decision.used
            if used:
                plan.ready.append({"arm": arm.name, "task": task, "image": want.image,
                                   "baked_at": decision.baked_at, "noop": decision.noop})  # type: ignore[union-attr]
            key = (r.family, str(r.version), str(r.recipe), want.task_image)
            target = ready if used and not rebuild else items
            item = target.get(key)
            if item is None:
                why = "rebuild" if used else (decision.reason if decision is not None else
                                              "not checked: Docker did not answer" if r.docker_error else "unknown")
                item = target[key] = BuildItem(
                    family=r.family, version=str(r.version), recipe=str(r.recipe), variant=str(r.variant),
                    agent=arm.agent, kwargs=dict(r.kwargs or {}),
                    model_name=cfg.harbor_model(arm),  # type: ignore[arg-type]
                    task_image=want.task_image, prefix=r.prefix, dataset=spec.dataset,
                    why="" if target is ready else why, expected_min=cls.expected_build_min,
                    expected_gb=cls.expected_gb)
            if task not in item.tasks:
                item.tasks.append(task)
            if arm.name not in item.arms:
                item.arms.append(arm.name)
    plan.items = sorted(items.values(), key=lambda i: (-i.expected_min, i.family, i.task_image))
    plan.ready_items = sorted(ready.values(), key=lambda i: (i.family, i.task_image))
    return plan


def verify_ready(cfg: Config, plan: BuildPlan, *, setup: bool = False, equivalence: bool = False,
                 sample: str = "per-os", log: IO[str] | None = None, cancel: threading.Event | None = None,
                 root: Path | None = None) -> list[dict[str, Any]]:
    """Checks the ready images of plan under their final names: always
    static, smoke and fingerprint; the setup and equivalence checks when
    asked, on one image per harness and OS family (sample per-os) or all.
    Each result is appended to prebake/verify.jsonl. Hold build_lock()."""
    from .recorder import arm_recipe

    root = Path(root or PREBAKE_DIR)
    out = []
    sampled: set[tuple[str, ...]] = set()

    def note(line: str) -> None:
        if log is not None:
            log.write(f"[{_now()}] {line}\n")
            log.flush()

    for item in plan.ready_items:
        if cancel is not None and cancel.is_set():
            break
        adapter = item.adapter()
        if adapter is None:
            out.append({"image": item.image, "ok": False, "error": f"no pre-bake adapter for {item.family}"})
            continue
        note(f"checking {item.image}")
        result: dict[str, Any] = {"kind": "image", "image": item.image, "family": item.family,
                                  "version": item.version, "recipe": item.recipe, "task_image": item.task_image,
                                  **verify.check_image(item.image, item.want(), adapter.smoke, root)}
        os_family = verify.os_family(item.task_image) if setup or equivalence else None
        key = (item.family, item.version, item.recipe, str(os_family))
        due = result["ok"] and (sample == "all" or key not in sampled)
        stamp = f"{datetime.now():%Y%m%d-%H%M%S}-{item.slug[:40]}"
        if due and setup:
            task_dir = inventory.task_dir(item.dataset, item.tasks[0])
            if task_dir is None:
                result["setup"] = {"ok": False, "error": f"no task folder for {item.tasks[0]}"}
            else:
                result["setup"] = verify.setup_check(
                    prefix=item.prefix, family=item.family, version=item.version, recipe=item.recipe,
                    agent=item.agent, kwargs=item.kwargs, model_name=item.model_name, task_dir=task_dir,
                    budget_sec=adapter.setup_budget_sec, expected_skips=len(arm_recipe(adapter).skip_execs),
                    work=root / "verify" / f"s{stamp}", log=log, cancel=cancel)
            note(f"setup check: ok={result['setup'].get('ok')} agent_setup_sec="
                 f"{result['setup'].get('agent_setup_sec')} "
                 f"{'; '.join(result['setup'].get('problems') or []) or result['setup'].get('error') or ''}")
        if due and equivalence:
            eq = result["equivalence"] = verify.equivalence(
                image=item.image, task_image=item.task_image, family=item.family, agent=item.agent,
                kwargs=item.kwargs, model_name=item.model_name, work=root / "verify" / f"e{stamp}", log=log,
                cancel=cancel)
            with _jsonl_lock:
                fsutil.append_jsonl(root / "verify.jsonl", [{
                    "kind": "equivalence", "ok": bool(eq.get("ok")), "family": item.family,
                    "version": item.version, "recipe": item.recipe, "os_family": os_family,
                    "task_image": item.task_image, "image": item.image,
                    "detail": eq.get("error") or "; ".join(((eq.get("files") or {}).get("violations") or [])[:5]),
                    "at": _now()}])
        if due:
            sampled.add(key)
        result["os_family"] = os_family
        result["ok"] = bool(result["ok"] and all((result.get(k) or {"ok": True}).get("ok")
                                                 for k in ("setup", "equivalence")))
        record = {k: v for k, v in result.items()
                  if k not in ("static", "smoke", "fingerprint", "setup", "equivalence")}
        record["verify"] = _verify_summary(result)
        with _jsonl_lock:
            fsutil.append_jsonl(root / "verify.jsonl", [{**record, "at": _now()}])
        out.append(result)
    return out


# Disk -----------------------------------------------------------------------------------


def _appdata(name: str, *default: str) -> Path:
    return Path(os.environ.get(name) or Path.home().joinpath("AppData", *default))


def docker_disk() -> tuple[Path | None, str]:
    """(Docker Desktop's disk image, where that came from). A moved disk image
    is found through Docker Desktop's settings store (CustomWslDistroDir)."""
    local, roaming = _appdata("LOCALAPPDATA", "Local"), _appdata("APPDATA", "Roaming")
    custom = None
    for store in (roaming / "Docker" / "settings-store.json", roaming / "Docker" / "settings.json"):
        try:
            data = json.loads(store.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        value = data.get("CustomWslDistroDir") or data.get("customWslDistroDir") if isinstance(data, dict) else None
        if isinstance(value, str) and value.strip():
            custom = Path(value.strip())
            break
    places: list[tuple[Path, str]] = []
    if custom is not None:
        for rel in (("DockerDesktopWSL", "disk"), ("disk",), (), ("DockerDesktopWSL", "data"), ("data",)):
            for name in ("docker_data.vhdx", "ext4.vhdx"):
                places.append((custom.joinpath(*rel, name), "Docker Desktop's settings (CustomWslDistroDir)"))
    for rel, name in ((("disk",), "docker_data.vhdx"), (("data",), "ext4.vhdx")):
        places.append((local.joinpath("Docker", "wsl", *rel, name), "Docker Desktop's default location"))
    for path, source in places:
        if path.is_file():
            return path, source
    if custom is not None:
        return custom, "Docker Desktop's settings (CustomWslDistroDir); no disk image file found there"
    return None, "not found"


def host_free_gb(path: Path) -> float:
    return shutil.disk_usage(path.anchor or path).free / _GB


# The last VM figure any check measured: {"free_gb", "at" (epoch seconds)}. The
# dashboard's 'Docker disk' readiness item shows it rather than start a container.
last_vm: dict[str, float] = {}


def vm_free_gb(probe_image: str) -> float:
    """Free space inside Docker's VM: df of a throwaway container's root."""
    r = docker(["run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "df", probe_image, "-Pk", "/"],
               timeout=120)
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or len(lines) < 2:
        raise DockerError(f"df in {probe_image} failed: {tail(r.stderr) or 'no output'}")
    fields = lines[-1].split()
    free, used = int(fields[3]) * 1024 / _GB, int(fields[2]) * 1024 / _GB
    last_vm.update(free_gb=round(free, 2), used_gb=round(used, 2), at=time.time())
    return free


def disk_slack_gb(disk: Path | None) -> float:
    """Space inside Docker's disk image file that new writes reuse before the
    file grows: its size on the host minus what the VM last measured in use.
    The file never shrinks by itself, so after images are pruned this can be
    most of it (2026-10-31: a 460 GB file holding 105 GB). 0 when unknown."""
    if disk is None or "used_gb" not in last_vm:
        return 0.0
    try:
        size = disk.stat().st_size / _GB
    except OSError:
        return 0.0
    return max(0.0, size - last_vm["used_gb"])


def vm_probe(items: Sequence[BuildItem]) -> str | None:
    """A local task image of items to run the VM's df in; None when there is none (or no Docker)."""
    try:
        local = inventory.local_images()
    except DockerError:
        return None
    return next((i.task_image for i in items if inventory.normalize_ref(i.task_image) in local), None)


@dataclass
class DiskCheck:
    ok: bool
    need_gb: float
    host: dict[str, Any]
    vm: dict[str, Any]
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def disk_check(cfg: Config, items: Sequence[BuildItem], probe_image: str | None = None) -> DiskCheck:
    """Whether building items keeps prebake.host_reserve_gb free on the drive
    holding Docker's disk image and prebake.vm_reserve_gb inside Docker's VM.
    The host must hold only what the disk image would grow by: when it looks
    short, the VM's df tells how much of the file is free inside and reused.
    The VM is checked only when the host passes: its check runs a container."""
    need = round(sum(i.expected_gb for i in items) * DISK_FACTOR, 2)
    disk, source = docker_disk()
    target = disk or _appdata("LOCALAPPDATA", "Local")
    host: dict[str, Any] = {"path": str(disk) if disk else None, "source": source, "drive": target.anchor,
                            "reserve_gb": cfg.prebake.host_reserve_gb}
    vm: dict[str, Any] = {"reserve_gb": cfg.prebake.vm_reserve_gb, "checked": False}
    reasons = []
    if not items:
        return DiskCheck(True, 0.0, {**host, "ok": True}, {**vm, "ok": True, "detail": "nothing to build"})
    try:
        free = host_free_gb(target)
    except OSError as exc:
        free = None
        reasons.append(f"The free space on {target.anchor} could not be read: {exc}")
    host["free_gb"] = None if free is None else round(free, 2)
    growth = need
    if free is not None and free - need < cfg.prebake.host_reserve_gb and disk is not None and disk.is_file() \
            and probe_image is not None:
        # Short on the host: the file only grows by what the free space inside it can't hold.
        try:
            vm_free_gb(probe_image)
        except DockerError:
            pass
        growth = max(0.0, need - disk_slack_gb(disk))
        host["slack_gb"] = round(disk_slack_gb(disk), 2)
    host["growth_gb"] = round(growth, 2)
    # A build the file can hold takes nothing from the host, however full it already is.
    host["ok"] = free is not None and (growth <= 0 or free - growth >= cfg.prebake.host_reserve_gb)
    if free is not None and not host["ok"]:
        where = f"{disk} ({source})" if disk else f"{target.anchor} (Docker's disk image was not found; assumed)"
        reasons.append(
            f"The drive holding Docker Desktop's disk image, {where}, has {free:.1f} GB free. This build needs about "
            f"{need:.1f} GB ({len(items)} image(s)) and must leave prebake.host_reserve_gb "
            f"({cfg.prebake.host_reserve_gb:g} GB) free, so it needs {need + cfg.prebake.host_reserve_gb:.1f} GB. "
            "Free space there, or move the disk image in Docker Desktop (Settings > Resources > Advanced > "
            "Disk image location).")
    if not host["ok"]:
        vm["detail"] = "not checked: the host drive is short already, and the check runs a container"
    elif probe_image is None:
        vm["detail"] = "not checked: no local task image to run df in"
    else:
        try:
            vm_free = vm_free_gb(probe_image)
            vm.update(checked=True, free_gb=round(vm_free, 2), ok=vm_free - need >= cfg.prebake.vm_reserve_gb)
            if not vm["ok"]:
                reasons.append(f"Docker's VM has {vm_free:.1f} GB free; this build needs about {need:.1f} GB and "
                               f"must leave prebake.vm_reserve_gb ({cfg.prebake.vm_reserve_gb:g} GB) free. Prune "
                               "images (python -m bench.prebake prune) or grow Docker Desktop's disk limit.")
        except DockerError as exc:
            vm.update(ok=False, detail=str(exc))
            reasons.append(f"The free space inside Docker's VM could not be read: {exc}")
    vm.setdefault("ok", True)
    return DiskCheck(ok=bool(host["ok"] and vm["ok"]), need_gb=need, host=host, vm=vm, reasons=reasons)


# Locks ----------------------------------------------------------------------------------


def build_lock_path() -> Path:
    return PREBAKE_DIR / ".build.lock"


def run_lock_path(profile: str | None = None) -> Path:
    return PREBAKE_DIR / f".run-{profile or settings.PROFILE_NAME}.lock"


def _held(path: Path) -> bool:
    """Whether another handle holds the lock on path. Probing takes the lock for
    a moment, so a second try after PROBE_GRACE_SEC keeps one probe (the state
    poll's, say) from reading another as a build or a run."""
    try:
        with procs.file_lock(path, timeout=PROBE_GRACE_SEC):
            return False
    except TimeoutError:
        return True


def build_active() -> bool:
    """A build (CLI or dashboard, any profile) holds the build lock."""
    return _held(build_lock_path())


def _runs_dirs() -> list[Path]:
    if os.environ.get("BENCH_DATA_DIR"):
        return [settings.RUNS_DIR]
    return [settings.ROOT / "runs", *sorted((settings.ROOT / "profiles").glob("*/runs"))]


def active_runs(in_process: Callable[[], bool] | None = None, owner_run: str | None = None) -> list[str]:
    """Runs in progress anywhere: this bench's own (in_process), any profile
    holding its run lock, or a run.json of any profile with a live status
    saved in the last RUN_FRESH_SEC (a bench from before the lock existed).
    owner_run is the run of this profile a build is for (a task race builds
    its next task's images between races): it and its lock do not count."""
    out = []
    if owner_run is None and in_process is not None and in_process():
        out.append("a run of this bench")
    own_lock = run_lock_path() if owner_run is not None else None
    for path in sorted(PREBAKE_DIR.glob(".run-*.lock")) if PREBAKE_DIR.is_dir() else []:
        if own_lock is not None and path == own_lock:
            continue
        if _held(path):
            out.append(f"a run of profile {path.name[5:-5]}")
    now = time.time()
    for runs_dir in _runs_dirs():
        for path in runs_dir.glob("*/run.json") if runs_dir.is_dir() else []:
            try:
                if now - path.stat().st_mtime > RUN_FRESH_SEC:
                    continue
                status = json.loads(path.read_text(encoding="utf-8")).get("status")
            except (OSError, ValueError, AttributeError):
                continue
            if status in _LIVE_RUN and not (owner_run is not None and path.parent.name == owner_run
                                             and runs_dir == settings.RUNS_DIR):
                out.append(f"run {path.parent.name}")
    return list(dict.fromkeys(out))


def _acquire(stack: contextlib.ExitStack, path: Path, busy: str) -> None:
    try:
        stack.enter_context(procs.file_lock(path, timeout=LOCK_WAIT_SEC))
    except TimeoutError:
        raise BuildRefused(busy) from None


# Each side takes its own lock first and then probes the other's, so a build
# and a run starting at the same moment both refuse rather than both start.

@contextlib.contextmanager
def run_lock(profile: str | None = None) -> Iterator[None]:
    """Held by a run for its whole life, so no build starts meanwhile. Raises
    BuildRefused when a build is active, or another run of the profile holds it."""
    with contextlib.ExitStack() as stack:
        _acquire(stack, run_lock_path(profile), "Another run of this profile holds the run lock")
        if build_active():
            raise BuildRefused("A pre-baked image build is running; runs wait until it ends "
                               "(python -m bench.prebake status, or the IMAGES panel)")
        yield


@contextlib.contextmanager
def build_lock(in_process_runs: Callable[[], bool] | None = None, owner_run: str | None = None) -> Iterator[None]:
    """The build lock, refused while a build or a run is active (other than
    owner_run, the task race this build is for, which holds the run lock and
    races nothing while its build runs)."""
    with contextlib.ExitStack() as stack:
        _acquire(stack, build_lock_path(), "Another build (from the CLI or the dashboard) is running")
        runs = active_runs(in_process_runs, owner_run)
        if runs:
            raise BuildRefused(f"Images are not built during a run ({', '.join(runs)}): builds slow the "
                               "trials' containers, and a run must not see its images change")
        yield


# Builds ---------------------------------------------------------------------------------


@dataclass
class ItemState:
    n: int
    family: str
    version: str
    recipe: str
    task_image: str
    image: str
    tasks: list[str]
    arms: list[str]
    why: str = ""
    status: str = "queued"        # queued, recording, building, verifying, awaiting equivalence, promoted,
                                  # failed, cancelled
    message: str = ""
    staging: str | None = None
    image_id: str | None = None
    size: int | None = None
    noop: bool | None = None
    os_family: str | None = None
    verify: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    seconds: float | None = None
    log: str = ""


@dataclass
class Build:
    id: str
    kind: str                     # build | pull
    label: str
    created_at: str
    jobs: int
    options: dict[str, Any]
    items: list[ItemState] = field(default_factory=list)
    status: str = "running"       # running, done, done_with_errors, failed, cancelled, interrupted
    message: str = ""
    disk: dict[str, Any] = field(default_factory=dict)
    pulled: dict[str, Any] = field(default_factory=dict)
    finished_at: str | None = None
    cache: dict[str, Any] = field(default_factory=dict)   # the build cache this build created, and what was pruned

    def to_dict(self) -> dict[str, Any]:
        for _ in range(5):  # item threads update their state while it is copied
            try:
                out = dataclasses.asdict(self)
                break
            except RuntimeError:
                time.sleep(0.01)
        else:
            out = dataclasses.asdict(self)
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        out["counts"] = counts
        return out


class _Equivalence:
    """One (family, version, recipe, OS family): the first image to get there
    runs the check; the others wait for its verdict before they are promoted."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state = "open"       # open, running, passed, failed
        self.verdict = ""
        self.waiting: list[Callable[[bool, str], None]] = []


class BuildManager:
    """Builds run one at a time per machine (the build lock), never during a
    run, each item on a worker thread. The dashboard drives it: start,
    cancel, status, is_idle, log_path and log_tail. run_active, when given,
    says whether this process's RunManager has a run going."""

    def __init__(self, root: Path | None = None, run_active: Callable[[], bool] | None = None) -> None:
        self.root = Path(root or PREBAKE_DIR)
        self.run_active = run_active
        self._lock = threading.RLock()
        self._builds: dict[str, Build] = {}
        self._active: str | None = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self._save_lock = threading.Lock()
        # Per build: which (family, OS family) passed a setup check, and the equivalence groups.
        self._setup_lock = threading.Lock()
        self._setup_passed: set[tuple[str, str]] = set()
        self._groups: dict[tuple[str, ...], _Equivalence] = {}
        self._setup_before: set[tuple[Any, ...]] = set()   # setup checks earlier builds passed (setup 'once')
        self._procs_lock = threading.Lock()
        self._build_procs: set[subprocess.Popen] = set()   # docker build children (emergency_stop kills them)
        self._cache_pending: tuple[str, Any, str] | None = None   # (build id, cache listing before it, started at)
        self._load_history()

    # Public -------------------------------------------------------------------------

    @property
    def active_id(self) -> str | None:
        return self._active

    def is_idle(self) -> bool:
        return self._active is None

    def add_listener(self, fn: Callable[[dict[str, Any]], None]) -> None:
        """fn(build_dict) is called once per build, after its final save."""
        with self._lock:
            self._listeners.append(fn)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [b.to_dict() for b in sorted(self._builds.values(), key=lambda b: b.created_at, reverse=True)]

    def summaries(self, limit: int = 20) -> list[dict[str, Any]]:
        """The newest builds without their items (counts only): cheap enough for every dashboard poll."""
        with self._lock:
            builds = sorted(self._builds.values(), key=lambda b: b.created_at, reverse=True)[:limit]
            out = []
            for b in builds:
                counts: dict[str, int] = {}
                for item in list(b.items):
                    counts[item.status] = counts.get(item.status, 0) + 1
                out.append({"id": b.id, "kind": b.kind, "label": b.label, "created_at": b.created_at,
                            "finished_at": b.finished_at, "status": b.status, "message": b.message, "jobs": b.jobs,
                            "items": len(b.items), "counts": counts})
            return out

    def status(self, build_id: str | None = None) -> dict[str, Any] | None:
        """One build (the active or latest without an id); None when there is none."""
        with self._lock:
            if build_id is None:
                build_id = self._active or max(self._builds, key=lambda k: self._builds[k].created_at, default=None)
            if build_id is None:
                return None
            if build_id not in self._builds:
                raise KeyError(build_id)
            return self._builds[build_id].to_dict()

    def log_path(self, build_id: str, item: int = 0) -> Path:
        """The build's own log (item 0) or one item's."""
        with self._lock:
            build = self._builds.get(build_id)
            if build is None:
                raise KeyError(build_id)
            if item == 0:
                return self.root / "logs" / build_id / "build.log"
            for state in build.items:
                if state.n == item:
                    return Path(state.log)
        raise KeyError(item)

    def log_tail(self, build_id: str, item: int = 0, lines: int = 200) -> str:
        path = self.log_path(build_id, item)
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 256 * 1024))
                text = f.read().decode("utf-8", "replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def cancel(self, build_id: str) -> None:
        with self._lock:
            if self._active != build_id:
                raise RuntimeError("That build is not in progress")
            self._cancel.set()

    def wait(self, timeout: float | None = None) -> bool:
        """True when no build is running any more."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return self.is_idle()

    def start(self, cfg: Config, plan: BuildPlan, *, jobs: int | None = None, setup: str = "auto",
              equivalence: str = "auto", sample: str = "per-os", owner_run: str | None = None,
              prune_cache: bool = False) -> dict[str, Any]:
        """Builds plan's items. Raises BuildRefused when a build or run is
        active or the disk check fails; ValueError for an unknown option.
        owner_run: the run of this profile the build is for (a task race),
        which does not count as a run in progress. prune_cache: afterwards,
        remove the Docker build cache records this build created (and only
        those: the ones that were not there when it started)."""
        if setup not in SETUP_MODES or equivalence not in EQUIVALENCE_MODES or sample not in SAMPLES:
            raise ValueError(f"setup is one of {SETUP_MODES}, equivalence of {EQUIVALENCE_MODES}, "
                             f"sample of {SAMPLES}")
        if not plan.items:
            raise BuildRefused("Nothing to build: every image this selection needs is ready")
        jobs = max(1, min(int(jobs or cfg.prebake.build_jobs), 16))
        options: dict[str, Any] = {"setup": setup, "equivalence": equivalence, "sample": sample}
        if owner_run is not None:
            options["owner_run"] = owner_run
        if prune_cache:
            options["prune_cache"] = True
        return self._begin(cfg, "build", plan.label, jobs, options, plan.items)

    def start_pull(self, cfg: Config, images: Sequence[str], label: str = "pull") -> dict[str, Any]:
        """Pulls the task images that are not local, as a build of its own."""
        if not images:
            raise BuildRefused("No task image to pull")
        return self._begin(cfg, "pull", label, 2, {"images": list(dict.fromkeys(images))}, [])

    def shutdown(self, timeout: float = 60) -> None:
        """Cancels a build in progress (the dashboard is stopping)."""
        thread = self._thread
        if self._active is not None:
            self._cancel.set()
        if thread is not None:
            thread.join(timeout)

    def emergency_stop(self, timeout: float = 2.0) -> None:
        """The console is closing: no Python cleanup runs after about 5 seconds.
        Cancels the build, kills its docker build processes and, when the build
        was to prune its build cache, writes what that prune needs to
        cache-pending.json; the next build that prunes its cache finishes it.
        Never raises."""
        try:
            self._cancel.set()
            pending = self._cache_pending
            if pending is not None:
                build_id, before, started_at = pending
                fsutil.atomic_write_json(self.root / CACHE_PENDING, {
                    "build_id": build_id, "source": before.source, "before": sorted(before.ids),
                    "images": len(self._builds[build_id].items) if build_id in self._builds else 1,
                    "started_at": started_at, "stopped_at": _now()})
            with self._procs_lock:
                live = list(self._build_procs)
            killers = [threading.Thread(target=procs.kill_tree, args=(p.pid, timeout), daemon=True,
                                        name="emergency-build-kill") for p in live]
            for t in killers:
                t.start()
            deadline = time.monotonic() + timeout
            for t in killers:
                t.join(max(0.0, deadline - time.monotonic()))
        except Exception:
            pass

    # Worker -----------------------------------------------------------------------------

    def _begin(self, cfg: Config, kind: str, label: str, jobs: int, options: dict[str, Any],
               items: Sequence[BuildItem]) -> dict[str, Any]:
        with self._lock:
            if self._active is not None:
                raise BuildRefused("A build is already running")
            stack = contextlib.ExitStack()
            stack.enter_context(build_lock(self.run_active, options.get("owner_run")))
            try:
                disk: dict[str, Any] = {}
                if kind == "build":
                    inventory.local_images(refresh=True)  # a build needs Docker: DockerError says why not
                    check = disk_check(cfg, items, vm_probe(items))
                    disk = check.to_dict()
                    if not check.ok:
                        raise BuildRefused("Not enough disk space to build: " + " ".join(check.reasons))
                build_id = f"b{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
                build = Build(id=build_id, kind=kind, label=label, created_at=_now(), jobs=jobs, options=options,
                              disk=disk)
                logs = self.root / "logs" / build_id
                for n, item in enumerate(items, 1):
                    build.items.append(ItemState(
                        n=n, family=item.family, version=item.version, recipe=item.recipe,
                        task_image=item.task_image, image=item.image, tasks=list(item.tasks), arms=list(item.arms),
                        why=item.why, log=str(logs / f"{n:02d}-{item.slug[:60]}.log")))
            except BaseException:
                stack.close()
                raise
            self._builds[build_id] = build
            self._active = build_id
            self._cancel.clear()
            self._save(build)
            self._thread = threading.Thread(target=self._execute, args=(cfg, build, list(items), stack),
                                            name=f"prebake-{build_id}", daemon=True)
            self._thread.start()
            return build.to_dict()

    def _main_log(self, build: Build) -> Path:
        path = self.root / "logs" / build.id / "build.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _log(self, build: Build, line: str) -> None:
        try:
            with open(self._main_log(build), "a", encoding="utf-8") as f:
                f.write(f"[{_now()}] {line}\n")
        except OSError:
            pass

    def _execute(self, cfg: Config, build: Build, items: list[BuildItem], stack: contextlib.ExitStack) -> None:
        try:
            with stack, open(self._main_log(build), "a", encoding="utf-8") as main_log:
                before = self._cache_before(build, main_log)
                try:
                    if not self._run_items(cfg, build, items, main_log):
                        return
                finally:
                    if before is not None:
                        self._prune_cache(build, before, main_log)
                statuses = {s.status for s in build.items}
                if self._cancel.is_set():
                    build.status = "cancelled"
                elif statuses <= {"promoted"}:
                    build.status = "done"
                elif "promoted" not in statuses:
                    build.status = "failed"
                    build.message = "No image was promoted. See the item logs."
                else:
                    build.status = "done_with_errors"
        except Exception as exc:  # keep the record honest; the dashboard stays up
            build.status = "failed"
            build.message = f"{type(exc).__name__}: {exc}"
            self._log(build, f"build failed: {build.message}")
            for state in build.items:
                if state.status not in ("promoted", "failed", "cancelled", "awaiting equivalence"):
                    state.status, state.message = "failed", state.message or f"Build failed: {build.message}"
        finally:
            for item, state in zip(items, build.items):
                if state.status == "queued":
                    state.status = "cancelled"
                elif state.status == "awaiting equivalence":  # its verdict never came
                    self._remove_staging(state)
                    self._finish(build, item, state, "cancelled" if self._cancel.is_set() else "failed",
                                 "not promoted: the equivalence check it waited for did not finish")
            build.finished_at = _now()
            inventory.forget_images()
            with self._lock:
                self._active = None
            self._save(build)
            self._notify(build)

    def _run_items(self, cfg: Config, build: Build, items: list[BuildItem], main_log: IO[str]) -> bool:
        """Pulls, then builds every item. False when the build is over already
        (a pull build, or cancelled while pulling) and its status is set."""
        images = build.options.get("images") or list(dict.fromkeys(i.task_image for i in items))
        try:
            build.pulled = pull.ensure_local(images, main_log, self._cancel)
        except pull.PullCancelled:
            build.status, build.message = "cancelled", "Cancelled while pulling task images"
            return False
        failed = build.pulled.get("failed") or {}
        if build.kind == "pull":
            build.status = "done_with_errors" if failed else "done"
            return False
        for item, state in zip(items, build.items):
            if item.task_image in failed:
                self._finish(build, item, state, "failed", f"the task image could not be pulled: "
                                                           f"{failed[item.task_image]}")
        work = [(i, s) for i, s in zip(items, build.items) if s.status == "queued"]
        self._plan_checks(build, work)
        with ThreadPoolExecutor(max_workers=build.jobs, thread_name_prefix=f"prebake-{build.id}") as pool:
            for future in [pool.submit(self._item, cfg, build, item, state) for item, state in work]:
                future.result()
        return True

    def _cache_before(self, build: Build, log: IO[str]) -> Any:
        """With prune_cache, the build cache listing taken before this build (a
        prune.CacheListing); None when it could not be listed (nothing is
        pruned then). Finishes a prune an earlier build left pending first."""
        if not build.options.get("prune_cache") or build.kind != "build":
            return None
        from .prune import build_cache_listing

        pending = self._finish_pending_prune(log)
        extra = {"pending": pending} if pending is not None else {}
        try:
            before = build_cache_listing()
        except DockerError as exc:
            build.cache = {"pruned": 0, "error": f"not pruned: the build cache could not be listed first ({exc})",
                           **extra}
            log.write(f"[{_now()}] build cache: {build.cache['error']}\n")
            log.flush()
            return None
        build.cache = {"before": len(before.ids), "source": before.source, **extra}
        self._cache_pending = (build.id, before, _now())
        return before

    def _prune_cache(self, build: Build, before: Any, log: IO[str]) -> None:
        """Removes the build cache records this build created: those listed now
        that were not there before it (prune.prune_new_cache). Every other
        record stays."""
        from .prune import prune_new_cache

        try:
            extra = {"pending": build.cache["pending"]} if "pending" in build.cache else {}
            build.cache = {**prune_new_cache(before, len(build.items)), **extra}
        except DockerError as exc:
            build.cache.update(pruned=0, error=f"not pruned: {exc}")
        finally:
            self._cache_pending = None
        log.write(f"[{_now()}] build cache: {json.dumps(build.cache)}\n")
        log.flush()

    def _finish_pending_prune(self, log: IO[str]) -> dict[str, Any] | None:
        """The prune of a build the console closing cut short (cache-pending.json):
        the records not listed before that build and created while it ran (a
        window, so records made since stay). Needs buildx, which says when each
        record was made; otherwise nothing is pruned. One attempt, unless
        Docker cannot list the cache at all. Returns the result, None when
        there was nothing to finish."""
        from .prune import CacheListing, prune_new_cache

        path = self.root / CACHE_PENDING
        if not path.is_file():
            return None
        try:
            pending = json.loads(path.read_text(encoding="utf-8"))
            slack = CACHE_PENDING_SLACK_SEC
            start = datetime.fromisoformat(pending["started_at"]).timestamp() - slack
            end = datetime.fromisoformat(pending["stopped_at"]).timestamp() + slack
            before = CacheListing(str(pending["source"]), frozenset(pending["before"]), {})
            window = (datetime.fromtimestamp(start, timezone.utc), datetime.fromtimestamp(end, timezone.utc))
            result: dict[str, Any] = {"build_id": pending.get("build_id"),
                                      **prune_new_cache(before, int(pending.get("images") or 1), window)}
        except DockerError as exc:
            log.write(f"[{_now()}] pending build cache prune: Docker could not list the cache ({exc}); "
                      "kept for the next build\n")
            log.flush()
            return None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result = {"pruned": 0, "error": f"not pruned: cache-pending.json is unreadable ({exc})"}
        log.write(f"[{_now()}] pending build cache prune: {json.dumps(result)}\n")
        log.flush()
        with contextlib.suppress(OSError):
            path.unlink()
        return result

    def _plan_checks(self, build: Build, work: list[tuple[BuildItem, ItemState]]) -> None:
        """OS family per task image, then which items run the setup and
        equivalence checks (verify.jsonl remembers passed equivalence checks)."""
        options = build.options
        self._setup_passed: set[tuple[str, str]] = set()
        self._groups: dict[tuple[str, ...], _Equivalence] = {}
        self._setup_before = set()
        if options["setup"] == "none" and options["equivalence"] == "none":
            return
        if options["setup"] == "once":
            self._setup_before = {(r.get("family"), r.get("version"), r.get("recipe"), r.get("os_family"))
                                  for r in fsutil.read_jsonl(self.root / "builds.jsonl")
                                  if r.get("status") == "promoted"
                                  and ((r.get("verify") or {}).get("setup") or {}).get("ok")}
        families = {img: verify.os_family(img) for img in dict.fromkeys(i.task_image for i, _ in work)}
        passed = {(r.get("family"), r.get("version"), r.get("recipe"), r.get("os_family"))
                  for r in fsutil.read_jsonl(self.root / "verify.jsonl")
                  if r.get("kind") == "equivalence" and r.get("ok")}
        for item, state in work:
            state.os_family = families[item.task_image]
            group = (item.family, item.version, item.recipe, state.os_family)
            due = options["equivalence"] == "always" or (options["equivalence"] == "auto" and group not in passed)
            if due:
                key = group if options["sample"] == "per-os" else (*group, item.task_image)
                self._groups.setdefault(key, _Equivalence())

    def _group(self, build: Build, item: BuildItem, state: ItemState) -> _Equivalence | None:
        key = (item.family, item.version, item.recipe, state.os_family or "unknown")
        return self._groups.get(key) or self._groups.get((*key, item.task_image))

    def _item(self, cfg: Config, build: Build, item: BuildItem, state: ItemState) -> None:
        if self._cancel.is_set():
            self._finish(build, item, state, "cancelled", "Cancelled before it started")
            return
        state.started_at = _now()
        started = time.monotonic()
        Path(state.log).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(state.log, "a", encoding="utf-8") as log:
                self._bake(cfg, build, item, state, log)
        except Exception as exc:
            self._remove_staging(state)
            self._finish(build, item, state, "cancelled" if self._cancel.is_set() else "failed",
                         f"{type(exc).__name__}: {exc}")
        finally:
            state.seconds = round(time.monotonic() - started, 1)
            self._save(build)

    def _bake(self, cfg: Config, build: Build, item: BuildItem, state: ItemState, log: IO[str]) -> None:
        from . import dockerfile, recorder

        def note(line: str) -> None:
            log.write(f"[{_now()}] {line}\n")
            log.flush()

        def stop(status: str, message: str) -> None:
            note(message)
            self._remove_staging(state)
            self._finish(build, item, state, status, message)

        disk = docker_disk()[0]
        free = host_free_gb(disk or _appdata("LOCALAPPDATA", "Local"))
        growth = max(0.0, item.expected_gb * DISK_FACTOR - disk_slack_gb(disk))
        if growth > 0 and free - growth < cfg.prebake.host_reserve_gb:
            stop("failed", f"Not started: {free:.1f} GB free on the drive holding Docker's disk image, under "
                           f"prebake.host_reserve_gb ({cfg.prebake.host_reserve_gb:g} GB) plus this image")
            return
        state.status = "recording"
        self._save(build)
        adapter = item.adapter()
        if adapter is None:
            stop("failed", f"no pre-bake adapter for {item.family}")
            return
        recipe = recorder.arm_recipe(adapter)
        if recipe.sha256 != item.recipe:
            stop("failed", f"the arm's install changed since this build was planned (recipe {recipe.sha256[:10]}, "
                           f"planned {item.recipe[:10]}); plan again")
            return
        stage = self.root / "ctx" / f"{item.slug[:60]}"
        shutil.rmtree(stage, ignore_errors=True)
        recording = recorder.record_for_image(adapter, item.task_image, stage / "stage")
        text, context = dockerfile.render(recording)
        ctx = stage / "context"
        ctx.mkdir(parents=True, exist_ok=True)
        for name, source in context.items():
            (shutil.copytree if source.is_dir() else shutil.copy2)(source, ctx / name)
        dockerfile_path = stage / "Dockerfile"
        dockerfile_path.write_text(text, encoding="utf-8", newline="\n")
        state.noop = recording.noop
        note(f"recorded {len(recording.install_steps)} install step(s) on {item.task_image}"
             f"{' (a label-only no-op image)' if recording.noop else ''}; Dockerfile: {dockerfile_path}")
        base = inventory.inspect_raw([item.task_image], refresh=True).get(item.task_image)
        if base is None:
            stop("failed", f"the task image {item.task_image} is not local")
            return
        staging_prefix = f"{STAGING}/{build.id}-{state.n}"
        staging = prebaked_image_name(staging_prefix, item.family, item.version, item.recipe, item.task_image)
        labels = bake_labels(item.want(), base_image_id=str(base.get("Id")), baked_at=_now(),
                             dockerfile_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                             skip_execs=recipe.skip_execs, skip_commands=recipe.skip_commands, noop=recording.noop)
        labels[LABEL_NS + "build-id"] = build.id
        args = ["build", "--no-cache", "--progress=plain", "-f", str(dockerfile_path),
                "--build-arg", f"BASE={item.task_image}"]
        for k, v in labels.items():
            args += ["--label", f"{k}={v}"]
        args += ["-t", staging, str(ctx)]
        state.status = "building"
        self._save(build)
        note(f"docker {' '.join(args[:6])} ... -t {staging}")
        rc = self._docker_build(args, log)
        state.staging = staging
        if self._cancel.is_set():
            stop("cancelled", "Cancelled during the build")
            return
        if rc != 0:
            stop("failed", f"docker build exited with {rc}; see the log")
            return
        state.status = "verifying"
        self._save(build)
        checks = verify.check_image(staging, item.want(staging_prefix), adapter.smoke, self.root)
        state.verify.update(checks)
        state.image_id, state.size = checks["static"].get("image_id"), checks["static"].get("size")
        note(f"checks: static {checks['static']['ok']}, smoke {(checks.get('smoke') or {}).get('ok')}, "
             f"fingerprint {bool((checks.get('fingerprint') or {}).get('ok'))}")
        if not checks["ok"]:
            stop("failed", "verification failed: " + _check_problem(checks))
            return
        if self._setup_due(build, item, state):
            task_dir = inventory.task_dir(item.dataset, item.tasks[0])
            if task_dir is None:
                stop("failed", f"the setup check needs the task folder of {item.tasks[0]} (dataset snapshot)")
                return
            note("setup check: Harbor's install_only Trial through PrebakedDockerEnvironment")
            result = verify.setup_check(prefix=staging_prefix, family=item.family, version=item.version,
                                        recipe=item.recipe, agent=item.agent, kwargs=item.kwargs,
                                        model_name=item.model_name, task_dir=task_dir,
                                        budget_sec=adapter.setup_budget_sec,
                                        expected_skips=len(recipe.skip_execs),
                                        work=self.root / "verify" / build.id / f"{state.n:02d}", log=log,
                                        cancel=self._cancel)
            state.verify["setup"] = result
            note(f"setup check: ok={result.get('ok')} agent_setup_sec={result.get('agent_setup_sec')} "
                 f"{'; '.join(result.get('problems') or []) or result.get('error') or ''}")
            if not result.get("ok"):
                stop("cancelled" if self._cancel.is_set() else "failed",
                     "setup check failed: " + ("; ".join(result.get("problems") or []) or str(result.get("error"))))
                return
            with self._setup_lock:
                self._setup_passed.add((item.family, state.os_family or "unknown"))
        group = self._group(build, item, state)
        if group is None:
            self._promote_and_finish(build, item, state, note)
            return
        with group.lock:
            if group.state == "passed":
                mine = False
            elif group.state == "failed":
                stop("failed", f"not promoted: the equivalence check of this recipe failed ({group.verdict})")
                return
            elif group.state == "running":
                state.status = "awaiting equivalence"
                group.waiting.append(lambda ok, why, s=state, i=item: self._after_equivalence(build, i, s, ok, why))
                note("waiting for the equivalence check another image of this recipe and OS family is running")
                self._save(build)
                return
            else:
                group.state, mine = "running", True
        if not mine:
            self._promote_and_finish(build, item, state, note)
            return
        note("equivalence check: Harbor's setup() in a stock container and on this image, compared")
        try:
            result = verify.equivalence(image=staging, task_image=item.task_image, family=item.family,
                                        agent=item.agent, kwargs=item.kwargs, model_name=item.model_name,
                                        work=self.root / "verify" / build.id / f"{state.n:02d}-equivalence",
                                        log=log, cancel=self._cancel)
        except Exception as exc:  # the images waiting on this verdict must still get one
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        state.verify["equivalence"] = {k: v for k, v in result.items() if k != "files"} | {
            "files": {k: v for k, v in (result.get("files") or {}).items() if k != "violations"},
            "violations": (result.get("files") or {}).get("violations", [])[:50]}
        ok = bool(result.get("ok"))
        why = "passed" if ok else (result.get("error") or "; ".join(
            ((result.get("files") or {}).get("violations") or [])[:5] + (result.get("fingerprints") or [])[:5]))
        self._record_verify({"kind": "equivalence", "ok": ok, "family": item.family, "version": item.version,
                             "recipe": item.recipe, "os_family": state.os_family, "task_image": item.task_image,
                             "image": item.image, "build_id": build.id, "detail": why, "at": _now()})
        with group.lock:
            group.state, group.verdict = ("passed" if ok else "failed"), why
            waiting, group.waiting = group.waiting, []
        try:
            if ok:
                self._promote_and_finish(build, item, state, note)
            else:
                stop("cancelled" if self._cancel.is_set() else "failed", f"equivalence check failed: {why}")
        finally:
            for fn in waiting:
                fn(ok, why)

    def _setup_due(self, build: Build, item: BuildItem, state: ItemState) -> bool:
        mode = build.options["setup"]
        if mode == "none":
            return False
        if mode == "all":
            return True
        if mode == "once" and (item.family, item.version, item.recipe, state.os_family) in self._setup_before:
            return False
        with self._setup_lock:
            return (item.family, state.os_family or "unknown") not in self._setup_passed

    def _after_equivalence(self, build: Build, item: BuildItem, state: ItemState, ok: bool, why: str) -> None:
        with open(state.log, "a", encoding="utf-8") as log:
            def note(line: str) -> None:
                log.write(f"[{_now()}] {line}\n")
                log.flush()

            if ok:
                try:
                    self._promote_and_finish(build, item, state, note)
                except Exception as exc:
                    self._remove_staging(state)
                    self._finish(build, item, state, "failed", f"{type(exc).__name__}: {exc}")
            else:
                note(f"not promoted: the equivalence check of this recipe failed ({why})")
                self._remove_staging(state)
                self._finish(build, item, state, "cancelled" if self._cancel.is_set() else "failed",
                             f"the equivalence check of this recipe failed ({why})")
        self._save(build)

    def _promote_and_finish(self, build: Build, item: BuildItem, state: ItemState,
                            note: Callable[[str], None]) -> None:
        if self._cancel.is_set():
            self._remove_staging(state)
            self._finish(build, item, state, "cancelled", "Cancelled before promotion")
            return
        promote(str(state.staging), item.image)
        state.staging = None
        note(f"promoted to {item.image}")
        self._finish(build, item, state, "promoted", "")

    def _docker_build(self, args: list[str], log: IO[str]) -> int:
        log.flush()
        proc = subprocess.Popen(["docker", *args], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                env=keys.child_env(None), creationflags=procs.NO_CTRL_C)
        procs.assign_kill_on_close(proc)
        with self._procs_lock:
            self._build_procs.add(proc)
        try:
            started = time.monotonic()
            while proc.poll() is None:
                if self._cancel.wait(1.0) or time.monotonic() - started > BUILD_TIMEOUT_SEC:
                    procs.kill_tree(proc.pid)
                    proc.wait()
                    return -1
            return proc.returncode
        finally:
            with self._procs_lock:
                self._build_procs.discard(proc)

    def _remove_staging(self, state: ItemState) -> None:
        """Only this item's own staging tag; never a final name."""
        if not state.staging:
            return
        try:
            docker(["rmi", state.staging], timeout=120)
        except DockerError:
            pass
        state.staging = None

    def _finish(self, build: Build, item: BuildItem, state: ItemState, status: str, message: str) -> None:
        state.status, state.message, state.finished_at = status, message, _now()
        fp = (state.verify.get("fingerprint") or {}) if state.verify else {}
        record = {
            "build_id": build.id, "n": state.n, "status": status, "message": message, "image": item.image,
            "image_id": state.image_id, "size": state.size, "family": item.family, "version": item.version,
            "variant": item.variant, "recipe": item.recipe, "task_image": item.task_image,
            "base_id": (state.verify.get("static") or {}).get("base_id") if state.verify else None,
            "noop": state.noop, "os_family": state.os_family, "tasks": state.tasks, "arms": state.arms,
            "seconds": _elapsed(state.started_at),
            "verify": _verify_summary(state.verify), "fingerprint_sha256": fp.get("sha256"), "log": state.log,
            "finished_at": state.finished_at,
        }
        with _jsonl_lock:
            try:
                fsutil.append_jsonl(self.root / "builds.jsonl", [record])
            except OSError:
                pass

    def _record_verify(self, record: dict[str, Any]) -> None:
        with _jsonl_lock:
            try:
                fsutil.append_jsonl(self.root / "verify.jsonl", [record])
            except OSError:
                pass

    # Persistence ------------------------------------------------------------------------

    def _notify(self, build: Build) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(build.to_dict())
            except Exception:
                pass

    def _save(self, build: Build) -> None:
        with self._save_lock:  # item threads save the same build
            try:
                fsutil.atomic_write_json(self.root / "builds" / f"{build.id}.json", build.to_dict())
            except (OSError, TypeError, ValueError, RuntimeError):
                pass

    def _load_history(self, mark_interrupted: bool = True) -> None:
        """Builds on disk: this process's past ones, and those another process
        (the CLI, another bench) runs or ran. A record left 'running' with
        nobody holding the build lock is marked interrupted."""
        folder = self.root / "builds"
        if not folder.is_dir():
            return
        live = build_active()
        for path in sorted(folder.glob("*.json")):
            if path.stem == self._active:
                continue  # this process's own build: the memory copy is the newer one
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                items = [ItemState(**{k: v for k, v in s.items() if k in _ITEM_FIELDS})
                         for s in data.get("items") or []]
                build = Build(items=items, **{k: v for k, v in data.items() if k in _BUILD_FIELDS})
            except (OSError, ValueError, TypeError, AttributeError):
                continue
            if build.status == "running" and not live and mark_interrupted:
                build.status, build.message = "interrupted", "The process running this build stopped"
                for state in build.items:
                    if state.status not in ("promoted", "failed", "cancelled"):
                        state.status = "cancelled"
                self._save(build)
            self._builds[build.id] = build

    def refresh(self) -> None:
        """Re-reads the build records, so builds started by another process
        (python -m bench.prebake build) show their progress here too."""
        with self._lock:
            self._load_history(mark_interrupted=False)


_ITEM_FIELDS = frozenset(f.name for f in dataclasses.fields(ItemState))
_BUILD_FIELDS = frozenset(f.name for f in dataclasses.fields(Build)) - {"items"}


def _elapsed(since: str | None) -> float | None:
    try:
        return round(time.time() - datetime.fromisoformat(str(since)).timestamp(), 1)
    except ValueError:
        return None


def _check_problem(checks: Mapping[str, Any]) -> str:
    static = checks.get("static") or {}
    if not static.get("ok"):
        return static.get("reason") if static.get("status") != "ready" else \
            "the image config differs from the task image's: " + "; ".join(static.get("config_diff") or [])
    smoke = checks.get("smoke") or {}
    if not smoke.get("ok"):
        return f"the smoke test failed (exit {smoke.get('rc')}): {smoke.get('output')}"
    return (checks.get("fingerprint") or {}).get("error") or "the fingerprint could not be taken"


def _verify_summary(checks: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("static", "smoke", "fingerprint", "setup", "equivalence"):
        value = checks.get(name) if checks else None
        if isinstance(value, Mapping):
            out[name] = {k: copy.deepcopy(value[k]) for k in ("ok", "status", "reason", "agent_setup_sec",
                                                              "budget_sec", "problems", "error", "violations",
                                                              "files", "setup_sec", "skipped", "sha256")
                         if k in value}
    return out


def promote(staging: str, final: str) -> None:
    """Gives the verified staging image its final name, then drops the staging tag."""
    r = docker(["tag", staging, final])
    if r.returncode != 0:
        raise DockerError(f"docker tag {staging} {final} failed: {tail(r.stderr)}")
    r = docker(["rmi", staging], timeout=120)
    if r.returncode != 0:
        raise DockerError(f"docker rmi {staging} failed after promotion: {tail(r.stderr)}")
    inventory.forget_images()


def remove_images(images: Sequence[str], prefix: str, task_images: Sequence[str] = ()) -> dict[str, str]:
    """Removes pre-baked images after a task race: `docker rmi <name>`, never
    forced, so an image a container still uses stays. Only names under
    '<prefix>/' are touched, never a task image (nor a name equal to one).
    Returns {image: removed | absent | kept: why | the error}."""
    out: dict[str, str] = {}
    keep = {inventory.normalize_ref(t) for t in task_images}
    for image in dict.fromkeys(images):
        if not prefix or not image.startswith(prefix.strip("/") + "/") or image.startswith(STAGING + "/"):
            out[image] = "kept: not a pre-baked image name"
            continue
        if inventory.normalize_ref(image) in keep:
            out[image] = "kept: a task image"
            continue
        try:
            raw = inventory.inspect_raw([image], refresh=True).get(image)
            if raw is None:
                out[image] = "absent"
                continue
            if not inventory.bake_label(raw, "bake-format"):
                out[image] = "kept: it carries no bake-format label, so the builder did not make it"
                continue
            r = docker(["rmi", image], timeout=120)
        except DockerError as exc:
            out[image] = str(exc)
            continue
        if r.returncode == 0:
            out[image] = "removed"
        elif "no such image" in (r.stderr or "").lower():
            out[image] = "absent"
        else:
            out[image] = tail(r.stderr) or f"docker rmi exited {r.returncode}"
    inventory.forget_images()
    return out


class QueueImages:
    """The run queue's view of the builder (runqueue.ImageHook): how many of
    a run's pre-baked images are not ready, and a build of them. Under
    require, 'blocking' says why the run may not start."""

    def __init__(self, manager: BuildManager) -> None:
        self.manager = manager

    def check(self, cfg: Config, spec: RunSpec) -> dict[str, Any]:
        if spec.run.prebake == PREBAKE_OFF:
            return {"todo": 0, "blocking": []}
        if spec.run.schedule == settings.SCHEDULE_TASKS:
            # A task race builds each task's images itself, just before that task: nothing to build or wait for
            # first. Under require, only an arm that can never have an image holds the item.
            found = inventory.readiness(cfg, spec, refresh=False)
            blocking = [f"{r.arm} cannot use pre-baked images ({r.reason})" for r in found.values()
                        if not r.supported] if spec.run.prebake == PREBAKE_REQUIRE else []
            return {"todo": 0, "blocking": blocking, "race": True}
        plan = plan_builds(cfg, spec, refresh=True)
        blocking = []
        if spec.run.prebake == PREBAKE_REQUIRE:
            for r in plan.readiness.values():
                if not r.supported:
                    blocking.append(f"{r.arm} cannot use pre-baked images ({r.reason})")
                elif r.docker_error or r.not_ready or r.unknown:
                    blocking.append(f"{r.arm}: {r.describe()}")
        return {"todo": len(plan.items), "blocking": blocking, "need_gb": plan.need_gb,
                "minutes": plan.wall_minutes(cfg.prebake.build_jobs)}

    def build(self, cfg: Config, spec: RunSpec) -> str:
        """Starts a build of spec's missing images, or joins the build that is
        running; returns its id. Raises BuildRefused when it cannot start."""
        active = self.manager.active_id
        if active is not None:
            return active
        plan = plan_builds(cfg, spec, label=f"queue: {spec.label}", refresh=True)
        return str(self.manager.start(cfg, plan)["id"])

    def status(self, build_id: str) -> dict[str, Any] | None:
        try:
            return self.manager.status(build_id)
        except KeyError:
            return None
