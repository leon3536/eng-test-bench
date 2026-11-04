"""What Docker holds, and what a run's arms would find in it.

Every Docker call of the builder goes through docker() here: the client gets
no credentials (keys.child_env(None)) and no console Ctrl+C, and tests replace
the one seam. local_images() is one cached `docker image ls` (30 s), and
inspect_many() one batched `docker image inspect` of the names that list
holds, so a plan never waits on one inspect per task.

readiness() runs the same pure check_candidate() a trial runs in
PrebakedDockerEnvironment.start(), per arm and task: a plan and a trial
cannot disagree about which image is ready. Missing and stale images (with
the reason), task images that are not local, label-only no-op images, the
oldest bake and what building the rest would cost all come with it.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import subprocess
import threading
import time
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .. import datasets, keys, procs, schemas, settings
from ..environments.prebaked import (
    LABEL_NS,
    MISSING,
    READY,
    STALE,
    UNVERIFIABLE,
    Decision,
    ImageInfo,
    Want,
    check_candidate,
)
from ..settings import PREBAKE_OFF, PREBAKE_PREFER, Config, ConfigError, Provider, RunSettings
from ..spec import ArmSpec, RunSpec
from . import ArmPrebake
from . import resolve as resolve_prebake

DOCKER_TIMEOUT_SEC = 60
LIST_TTL_SEC = 30
_INSPECT_BATCH = 50


class DockerError(RuntimeError):
    """Docker could not be asked, or did not answer."""


def _text(data: bytes | str | None) -> str:
    return data.decode("utf-8", "replace") if isinstance(data, bytes) else data or ""


def docker(args: Sequence[str], timeout: float = DOCKER_TIMEOUT_SEC, input: str | None = None
           ) -> subprocess.CompletedProcess:
    """`docker <args>`: no credentials in the client's environment, no console
    Ctrl+C. input goes in as UTF-8 bytes: a text-mode pipe on Windows would
    turn every newline into CRLF, and bash reading a script from stdin takes
    the CR as part of each command."""
    try:
        r = subprocess.run(["docker", *args], input=None if input is None else input.encode("utf-8"),
                           capture_output=True, timeout=timeout, env=keys.child_env(None),
                           creationflags=procs.NO_CTRL_C,
                           **({} if input is not None else {"stdin": subprocess.DEVNULL}))
        return subprocess.CompletedProcess(r.args, r.returncode, _text(r.stdout), _text(r.stderr))
    except FileNotFoundError:
        raise DockerError("the docker command was not found; install Docker Desktop") from None
    except OSError as exc:
        raise DockerError(f"docker could not be started: {exc}") from None
    except subprocess.TimeoutExpired:
        raise DockerError(f"docker {' '.join(args[:2])} did not answer within {timeout:g} s") from None


def tail(text: str | None, limit: int = 400) -> str:
    return " ".join((text or "").split())[-limit:]


# Local images ---------------------------------------------------------------------

_list_lock = threading.Lock()
_list_cache: tuple[float, dict[str, str]] | None = None


def normalize_ref(ref: str) -> str:
    """ref with Docker's implied :latest, the way `docker image ls` lists it."""
    name, sep, tag = ref.rpartition(":")
    return ref if sep and "/" not in tag else f"{ref}:latest"


def local_images(refresh: bool = False) -> dict[str, str]:
    """Every tagged local image: {repository:tag: image id}. Raises DockerError."""
    global _list_cache
    with _list_lock:
        if not refresh and _list_cache is not None and time.monotonic() - _list_cache[0] < LIST_TTL_SEC:
            return dict(_list_cache[1])
    r = docker(["image", "ls", "--no-trunc", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}"])
    if r.returncode != 0:
        raise DockerError(f"docker image ls failed: {tail(r.stderr) or 'no output'}")
    images: dict[str, str] = {}
    for line in r.stdout.splitlines():
        ref, tab, image_id = line.strip().partition("\t")
        if tab and "<none>" not in ref:
            images[ref] = image_id.strip()
    with _list_lock:
        _list_cache = (time.monotonic(), images)
    return dict(images)


def forget_images() -> None:
    """Drop the cached listing and inspections: images were built, tagged or removed."""
    global _list_cache
    with _list_lock:
        _list_cache = None
    with _inspect_lock:
        _inspected.clear()


_inspect_lock = threading.Lock()
_inspected: dict[str, dict[str, Any]] = {}   # image id -> inspect object; an id's config never changes
_INSPECT_KEEP = 4096


def inspect_raw(refs: Iterable[str], refresh: bool = False) -> dict[str, dict[str, Any] | None]:
    """{ref: one object of `docker image inspect`, or None when not local}.
    Only names the listing holds are inspected, in batches, so one missing
    image never fails the rest. An image id inspected before is answered
    from memory (its labels, layers and OS cannot change), so a plan
    inspects only images it has not seen; refresh asks Docker again."""
    wanted = list(dict.fromkeys(refs))
    local = local_images(refresh)
    out: dict[str, dict[str, Any] | None] = {r: None for r in wanted}
    present = []
    with _inspect_lock:
        for ref in wanted:
            image_id = local.get(normalize_ref(ref))
            if image_id is None:
                continue
            if image_id in _inspected and not refresh:
                out[ref] = _inspected[image_id]
            else:
                present.append(ref)
    for i in range(0, len(present), _INSPECT_BATCH):
        batch = present[i:i + _INSPECT_BATCH]
        r = docker(["image", "inspect", *batch])
        try:
            objects = json.loads(r.stdout or "[]")
        except ValueError:
            objects = []
        if r.returncode != 0 and not objects:
            raise DockerError(f"docker image inspect failed: {tail(r.stderr) or 'no output'}")
        by_tag: dict[str, dict[str, Any]] = {}
        for obj in objects if isinstance(objects, list) else []:
            for tag in obj.get("RepoTags") or []:
                by_tag[str(tag)] = obj
        with _inspect_lock:
            for ref in batch:
                obj = by_tag.get(normalize_ref(ref)) or by_tag.get(ref)
                out[ref] = obj
                if obj is not None and isinstance(obj.get("Id"), str):
                    if len(_inspected) >= _INSPECT_KEEP:
                        _inspected.clear()
                    _inspected[obj["Id"]] = obj
    return out


def inspect_many(refs: Iterable[str], refresh: bool = False) -> dict[str, ImageInfo | None]:
    return {ref: None if raw is None else ImageInfo.from_inspect(raw)
            for ref, raw in inspect_raw(refs, refresh).items()}


# Task images ----------------------------------------------------------------------


@functools.lru_cache(maxsize=4096)
def _docker_image(task_toml: Path, mtime_ns: int) -> str | None:
    try:
        data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    env = data.get("environment")
    image = env.get("docker_image") if isinstance(env, dict) else None
    return image.strip() if isinstance(image, str) and image.strip() else None


def task_images(dataset: str, tasks: Iterable[str]) -> dict[str, str | None]:
    """{task: its task.toml [environment].docker_image}, from the dataset's
    local snapshot; None for a task without one (or no snapshot yet)."""
    root = datasets.path(dataset)
    out: dict[str, str | None] = {}
    for task in dict.fromkeys(tasks):
        path = root / task / "task.toml" if root else None
        try:
            out[task] = _docker_image(path, path.stat().st_mtime_ns) if path else None
        except OSError:
            out[task] = None
    return out


def task_dir(dataset: str, task: str) -> Path | None:
    root = datasets.path(dataset)
    return root / task if root and (root / task / "task.toml").is_file() else None


# Readiness -----------------------------------------------------------------------


def arm_kwargs(run: RunSettings, arm: ArmSpec, provider: Provider | None = None) -> dict[str, Any]:
    """The agent kwargs Harbor will get for this arm (runs.agent_kwargs), as
    the planner shows them: a gateway arm's base_url still ${VAR}/route."""
    from ..runs import agent_kwargs  # runs imports this package

    return agent_kwargs(run, arm, schemas.agent_schema(arm.agent), provider)[0]


def resolve_arm(cfg: Config, run: RunSettings, arm: ArmSpec) -> tuple[ArmPrebake | None, dict[str, Any] | None,
                                                                      str]:
    """(the arm's prebake, its kwargs, why it has none). Never raises."""
    try:
        kwargs = arm_kwargs(run, arm, cfg.providers.get(arm.provider))
    except (ConfigError, ValueError, TypeError, RuntimeError) as exc:
        return None, None, f"its options did not resolve: {exc}"
    if run.prebake == PREBAKE_OFF:
        return None, kwargs, "run.prebake is off"
    return resolve_prebake(cfg, run, arm, kwargs), kwargs, ""


@dataclass
class ArmReadiness:
    """What one arm's trials would find, task by task."""
    arm: str
    family: str
    mode: str
    supported: bool
    reason: str = ""                  # why the arm uses no pre-baked image
    version: str | None = None
    recipe: str | None = None
    variant: str | None = None
    prefix: str = ""
    tasks: int = 0
    ready: list[str] = field(default_factory=list)
    noop: list[str] = field(default_factory=list)            # ready, label-only (Harbor installs nothing there)
    missing: list[str] = field(default_factory=list)
    stale: dict[str, str] = field(default_factory=dict)      # task -> why
    unverifiable: list[str] = field(default_factory=list)    # task image not local
    unknown: dict[str, str] = field(default_factory=dict)    # task -> why it has no image name
    over_age: list[str] = field(default_factory=list)        # ready, baked more than max_age_days ago
    oldest_baked_at: str | None = None
    build_min: float = 0.0            # building what is not ready, one at a time
    build_gb: float = 0.0
    docker_error: str | None = None
    # Per task, for the builder; left out of to_dict.
    wants: dict[str, Want] = field(default_factory=dict, repr=False)
    decisions: dict[str, Decision] = field(default_factory=dict, repr=False)
    kwargs: dict[str, Any] | None = field(default=None, repr=False)
    prebake: ArmPrebake | None = field(default=None, repr=False)

    @property
    def all_ready(self) -> bool:
        return self.supported and self.tasks > 0 and len(self.ready) == self.tasks

    @property
    def not_ready(self) -> list[str]:
        return [*self.missing, *self.stale, *self.unverifiable]

    def not_ready_text(self) -> str:
        """'9 missing, 1 stale (why)': the tasks without a usable image, counted."""
        parts = [f"{len(self.missing)} missing" if self.missing else "",
                 f"{len(self.stale)} stale ({next(iter(self.stale.values()))})" if self.stale else "",
                 f"{len(self.unverifiable)} with the task image not local" if self.unverifiable else "",
                 f"{len(self.unknown)} without a task image" if self.unknown else ""]
        return ", ".join(p for p in parts if p)

    def describe(self) -> str:
        """'3 of 13 ready; 9 missing, 1 stale', for plans, arm logs and the queue."""
        if not self.supported:
            return f"not pre-baked: {self.reason}"
        if self.docker_error:
            return f"not checked: Docker did not answer ({self.docker_error})"
        rest = self.not_ready_text()
        return f"{len(self.ready)} of {self.tasks} ready" + (f"; {rest}" if rest else "")

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
               if f.name not in ("wants", "decisions", "kwargs", "prebake")}
        out["all_ready"] = self.all_ready
        out["images"] = {t: w.image for t, w in self.wants.items()}
        return out

    def arm_prebake(self) -> ArmPrebake | None:
        """The arm's prebake with expect_stock cleared when every task's image
        is ready, so setup_timeout_multiplier 'auto' adds nothing."""
        if self.prebake is None or not self.all_ready:
            return self.prebake
        return dataclasses.replace(self.prebake, expect_stock=False)


def _adapter_class(family: str) -> Any:
    from .adapters import ADAPTERS

    return ADAPTERS.get(family)


def _age_days(stamp: str | None, now: datetime) -> float | None:
    try:
        when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (now - when) / timedelta(days=1)


def readiness(cfg: Config, spec: RunSpec, refresh: bool = False) -> dict[str, ArmReadiness]:
    """Per arm of spec, which tasks' pre-baked images are ready, missing or
    stale, decided by check_candidate() as a trial would. Never raises: an
    arm that cannot be resolved, or Docker not answering, is in the result."""
    images = task_images(spec.dataset, (t for arm in spec.arms for t in arm.tasks))
    out: dict[str, ArmReadiness] = {}
    for arm in spec.arms:
        prebake, kwargs, why = resolve_arm(cfg, spec.run, arm)
        r = ArmReadiness(arm=arm.name, family=arm.family, mode=spec.run.prebake, supported=False,
                         reason=why, prefix=cfg.prebake.prefix, tasks=len(arm.tasks), kwargs=kwargs, prebake=prebake)
        out[arm.name] = r
        if prebake is None:
            continue
        r.supported, r.reason = prebake.supported, prebake.reason
        r.version, r.recipe, r.variant, r.prefix = prebake.version, prebake.recipe, prebake.variant, prebake.prefix
        if not prebake.supported:
            continue
        for task in arm.tasks:
            image = images.get(task)
            if not image:
                r.unknown[task] = "the task has no docker_image (or the dataset is not downloaded yet)"
                continue
            want = Want(harness=r.family, version=str(r.version), recipe=str(r.recipe), task_image=image,
                        prefix=r.prefix)
            try:
                want.image  # noqa: B018  checks the name grammar
            except ValueError as exc:
                r.unknown[task] = str(exc)
                continue
            r.wants[task] = want
    refs = {ref for r in out.values() for w in r.wants.values() for ref in (w.image, w.task_image)}
    try:
        infos = inspect_many(refs, refresh) if refs else {}
    except DockerError as exc:
        for r in out.values():
            if r.wants:
                r.docker_error = str(exc)
                r.unverifiable = list(r.wants)
                _estimate(r)
        return out
    now = datetime.now(timezone.utc)
    for r in out.values():
        for task, want in r.wants.items():
            decision = check_candidate(infos.get(want.image), infos.get(want.task_image), want)
            r.decisions[task] = decision
            if decision.status == READY:
                r.ready.append(task)
                if decision.noop:
                    r.noop.append(task)
                age = _age_days(decision.baked_at, now)
                if age is not None and age > cfg.prebake.max_age_days:
                    r.over_age.append(task)
                if decision.baked_at and (r.oldest_baked_at is None or decision.baked_at < r.oldest_baked_at):
                    r.oldest_baked_at = decision.baked_at
            elif decision.status == MISSING:
                r.missing.append(task)
            elif decision.status == STALE:
                r.stale[task] = decision.reason
            elif decision.status == UNVERIFIABLE:
                r.unverifiable.append(task)
            else:
                r.unknown[task] = decision.reason
        _estimate(r)
    return out


def wanted_images(cfg: Config, spec: RunSpec) -> dict[tuple[str, str], str]:
    """{(arm, task): the pre-baked image name its trial would look for}, for
    spec's supported arms; {} under prebake off. Pure: it asks Docker nothing,
    so a task race can tell which images later tasks or queue items still need."""
    if spec.run.prebake == PREBAKE_OFF:
        return {}
    images = task_images(spec.dataset, (t for arm in spec.arms for t in arm.tasks))
    out: dict[tuple[str, str], str] = {}
    for arm in spec.arms:
        prebake, _, _ = resolve_arm(cfg, spec.run, arm)
        if prebake is None or not prebake.supported:
            continue
        for task in arm.tasks:
            image = images.get(task)
            if not image:
                continue
            want = Want(harness=prebake.family, version=str(prebake.version), recipe=str(prebake.recipe),
                        task_image=image, prefix=prebake.prefix)
            try:
                out[(arm.name, task)] = want.image
            except ValueError:
                continue
    return out


def _estimate(r: ArmReadiness) -> None:
    """What building the arm's images that are not ready would take, one at a time."""
    cls = _adapter_class(r.family)
    if cls is not None:
        todo = len({r.wants[t].task_image for t in r.not_ready})
        r.build_min = round(todo * cls.expected_build_min, 1)
        r.build_gb = round(todo * cls.expected_gb, 2)


# Wanted recipes --------------------------------------------------------------------


def profile_configs() -> list[tuple[str, Path, Path]]:
    """(profile, base config, overrides file) for every profile in config/."""
    out = [("default", settings.CONFIG_DIR / "bench.yaml", settings.CONFIG_DIR / "overrides.yaml")]
    for path in sorted(settings.CONFIG_DIR.glob("*.yaml")):
        name = path.stem
        if name == "bench" or name in settings._RESERVED_PROFILES or name.endswith(".overrides"):
            continue
        out.append((name, path, settings.CONFIG_DIR / f"{name}.overrides.yaml"))
    return out


def wanted_recipes(cfg: Config | None = None, all_profiles: bool = True
                   ) -> tuple[set[tuple[str, str, str]], list[str]]:
    """({(family, version, recipe)} some enabled arm would run on, notes).
    Every profile counts, whatever its run.prebake says (a run can turn it
    on), so prune never removes an image a profile can still use."""
    configs: list[tuple[str, Config]] = []
    notes: list[str] = []
    if cfg is not None:
        configs.append((settings.PROFILE_NAME, cfg))
    if all_profiles:
        for name, base, overrides in profile_configs():
            if cfg is not None and name == settings.PROFILE_NAME:
                continue
            try:
                configs.append((name, settings.load_config(base, overrides)))
            except (ConfigError, OSError, SystemExit) as exc:
                notes.append(f"profile {name}: its config could not be read ({exc}), so its arms keep nothing")
    wanted: set[tuple[str, str, str]] = set()
    for name, c in configs:
        run = dataclasses.replace(c.run, prebake=PREBAKE_PREFER)
        for arm in c.arms:
            if not arm.enabled:
                continue
            spec_arm = ArmSpec(name=arm.name, agent=arm.agent, family=arm.family or arm.agent, kind=arm.kind,
                               provider=arm.provider, model=arm.model, effort=arm.effort, options=dict(arm.options),
                               tasks=())
            prebake, _, why = resolve_arm(c, run, spec_arm)
            if prebake is not None and prebake.supported:
                wanted.add((prebake.family, str(prebake.version), str(prebake.recipe)))
            elif prebake is None and why:
                notes.append(f"profile {name}, arm {arm.name}: {why}")
    return wanted, notes


def labels_of(raw: Mapping[str, Any] | None) -> dict[str, str]:
    config = (raw or {}).get("Config") or {}
    return {str(k): str(v) for k, v in (config.get("Labels") or {}).items()}


def bake_label(raw: Mapping[str, Any] | None, key: str) -> str | None:
    return labels_of(raw).get(LABEL_NS + key)
