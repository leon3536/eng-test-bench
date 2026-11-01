"""A Harbor 0.25 Docker environment that runs a trial on a pre-baked image.

A pre-baked image is a task image with one harness's Harbor install already
run in it, built FROM the local task image by a recipe that issues Harbor's
own install commands (bench/prebake). Using one changes no file, package or
version the agent sees; it moves the install from trial time to bake time, so
the trial's unchanged setup() short-circuits or finishes in seconds.

Wiring, per arm (runs.build_command adds it; nothing else in the command changes):

    harbor run ... --env bench.environments.prebaked:PrebakedDockerEnvironment \\
        --ek prebake_harness='"claude-code"' --ek prebake_version='"2.1.312"' \\
        --ek prebake_recipe='"<sha256>"' --ek prebake_prefix='"benchbake"' --ek prebake_required=false

Harbor records --env / --ek as config.environment.import_path / .kwargs. The
bench's identity (extract.config_identity, task_identity, report pairing,
reuse) reads none of them, so a pre-baked trial and a stock trial of the same
arm are the same cell.

At start() the agent environment (session id "<trial>__env"; never a separate
verifier environment) looks for

    <prefix>/<harness>-<version>-r<recipe[:10]>/<task docker_image>
    e.g. benchbake/claude-code-2.1.312-r3f9c0a1b2c/alexgshaw/fix-git:20251031

and uses it only when check_candidate() passes: the image is local (it is
never pulled), its org.etb.prebake.* labels match the bake format, harness,
version, installed Harbor version, recipe and the local task image's id, the
task image's layers are a strict prefix of its layers (equal for a label-only
no-op image), and the OS matches. Otherwise the trial runs stock on the task
image, or with prebake_required fails before any container starts, as
SandboxBuildFailedError (Harbor's own name, so the ledger counts it as infra).
The decision goes to trial.log and to <trial_dir>/prebake.json, which is
outside every mount, so the agent never sees it.

An image may list in its skip-execs label the signatures of install commands
it already ran whose re-run is slow and adds nothing (only strands' uv venv +
pip step, which re-links ~18k files out of the image's lower layer). Such an
exec, matched exactly on command, user, cwd and env, is answered with exit
code 0 once instead of being run. Every other exec runs as in a stock trial.

DockerEnvironment internals used: _env_vars.prebuilt_image_name (the value
Harbor's prebuilt compose file reads as PREBUILT_IMAGE_NAME), _engine_cmd,
_dockerfile_path, _resolve_user and _merge_env. tests/test_prebake_env.py
fails if a Harbor upgrade moves them, and the harbor-version label makes
every image stale after an upgrade anyway.

Only Harbor and the standard library are imported: this module loads inside
the harbor process.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import Any

from harbor.environments.base import ExecResult, SandboxBuildFailedError
from harbor.environments.docker.docker import DockerEnvironment

PREBAKED_ENV = "bench.environments.prebaked:PrebakedDockerEnvironment"
LABEL_NS = "org.etb.prebake."
BAKE_FORMAT = 1
DEFAULT_PREFIX = "benchbake"
RECORD_NAME = "prebake.json"
PREFLIGHT_TIMEOUT_SEC = 60          # Harbor's own check allows `docker info` 10 s
PREFLIGHT_WAITS = (0, 15, 30)       # before each try of it

# Decision statuses. Only READY uses the image.
READY = "ready"
MISSING = "missing"
STALE = "stale"
UNVERIFIABLE = "unverifiable"     # the task image is not local, so the base cannot be checked
NOT_APPLICABLE = "not-applicable"  # verifier environment, force_build, no prebake arguments

# Docker's reference grammar (distribution/reference): lowercase path
# components joined by '/', then an optional tag.
_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_REPOSITORY = re.compile(rf"{_COMPONENT}(?:/{_COMPONENT})*")
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
# Credential-like env vars: their values never reach a signature, a label or a log.
_CREDENTIAL_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|AUTH", re.I)
_CREDENTIAL_VALUE = re.compile(r"sk-[A-Za-z0-9_-]{8,}|\bfw_[A-Za-z0-9]{16,}|\bAKIA[0-9A-Z]{16}\b|Bearer \S+")


@functools.cache
def harbor_version() -> str:
    try:
        return _dist_version("harbor")
    except PackageNotFoundError:
        return ""


def prebaked_image_name(prefix: str, family: str, version: str, recipe: str, task_image: str) -> str:
    """The pre-baked image for one task image, harness version and recipe.
    Raises ValueError when the result is not a valid Docker image name (an
    uppercase version, a digest-pinned or registry-port task image)."""
    if not isinstance(recipe, str) or not _SHA256.fullmatch(recipe):
        raise ValueError("the recipe must be a sha256 hex digest")
    repo, sep, tag = task_image.rpartition(":")
    if not sep or "/" in tag:
        repo, tag = task_image, ""
    name = f"{prefix.strip('/')}/{family}-{version}-r{recipe[:10]}/{repo}"
    if not _REPOSITORY.fullmatch(name) or len(name) > 255:
        raise ValueError(f"'{name}' is not a valid Docker image name")
    if tag and not _TAG.fullmatch(tag):
        raise ValueError(f"'{tag}' is not a valid Docker image tag")
    return f"{name}:{tag}" if tag else name


def redacted_env(env: Mapping[str, str] | None) -> dict[str, str]:
    """env with every credential-like value replaced by '<set>'."""
    return {str(k): "<set>" if _CREDENTIAL_NAME.search(str(k)) or _CREDENTIAL_VALUE.search(str(v)) else str(v)
            for k, v in (env or {}).items()}


def exec_signature(command: str, user: str | int | None, cwd: str | None,
                   env: Mapping[str, str] | None) -> str:
    """What a skippable exec is matched on: the command, the resolved user,
    the cwd and the merged env, with credential values replaced, so a
    signature can sit in an image label. The recorder computes the same four
    values for the commands a bake ran; any difference means no skip."""
    payload = {"command": command, "user": None if user is None else str(user), "cwd": cwd,
               "env": dict(sorted(redacted_env(env).items()))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def command_digest(command: str) -> str:
    """The command alone, for spotting a near miss (same command, other user or env)."""
    return hashlib.sha256(command.encode()).hexdigest()


@dataclass(frozen=True)
class Want:
    """The pre-baked image an arm asks for on one task image."""
    harness: str
    version: str
    recipe: str
    task_image: str
    harbor_version: str = field(default_factory=harbor_version)
    prefix: str = DEFAULT_PREFIX

    @property
    def image(self) -> str:
        return prebaked_image_name(self.prefix, self.harness, self.version, self.recipe, self.task_image)


@dataclass(frozen=True)
class ImageInfo:
    """What the decision reads from `docker image inspect`."""
    id: str
    os: str
    labels: Mapping[str, str]
    layers: tuple[str, ...]

    @classmethod
    def from_inspect(cls, data: Mapping[str, Any]) -> ImageInfo:
        """From one object of `docker image inspect` JSON output."""
        config = data.get("Config") if isinstance(data.get("Config"), Mapping) else {}
        rootfs = data.get("RootFS") if isinstance(data.get("RootFS"), Mapping) else {}
        return cls(id=str(data.get("Id") or ""), os=str(data.get("Os") or ""),
                   labels={str(k): str(v) for k, v in (config.get("Labels") or {}).items()},
                   layers=tuple(str(x) for x in rootfs.get("Layers") or ()))


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    image: str | None = None          # the pre-baked image looked up
    image_id: str | None = None
    task_image_id: str | None = None
    baked_at: str | None = None
    recipe: str | None = None         # the image's recipe label
    noop: bool = False
    added_layers: int | None = None
    skip_execs: tuple[str, ...] = ()
    skip_commands: tuple[str, ...] = ()

    @property
    def used(self) -> bool:
        return self.status == READY

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["used"] = self.used
        out["skip_execs"] = list(self.skip_execs)
        out["skip_commands"] = list(self.skip_commands)
        return out


def bake_labels(want: Want, *, base_image_id: str, baked_at: str, dockerfile_sha256: str,
                skip_execs: tuple[str, ...] = (), skip_commands: tuple[str, ...] = (),
                noop: bool = False) -> dict[str, str]:
    """The labels a bake puts on its image (docker build --label); check_candidate reads them back."""
    labels = {
        "bake-format": str(BAKE_FORMAT), "harness": want.harness, "harness-version": want.version,
        "harbor-version": want.harbor_version, "recipe-sha256": want.recipe, "base-image": want.task_image,
        "base-image-id": base_image_id, "baked-at": baked_at, "dockerfile-sha256": dockerfile_sha256,
        "skip-execs": json.dumps(sorted(skip_execs), separators=(",", ":")),
        "skip-commands": json.dumps(sorted(skip_commands), separators=(",", ":")),
    }
    if noop:
        labels["noop"] = "true"
    return {LABEL_NS + k: v for k, v in labels.items()}


def _digest_list(text: str | None) -> tuple[str, ...] | None:
    try:
        value = json.loads(text or "[]")
    except ValueError:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) and _SHA256.fullmatch(v) for v in value):
        return None
    return tuple(sorted(value))


def check_candidate(baked: ImageInfo | None, base: ImageInfo | None, want: Want) -> Decision:
    """Whether the pre-baked image may stand in for the task image. Pure, so
    a trial (PrebakedDockerEnvironment) and the planner's readiness check
    cannot disagree."""
    image = want.image
    if baked is None:
        return Decision(MISSING, f"no pre-baked image {image}", image=image)
    labels = baked.labels
    expected = {"bake-format": str(BAKE_FORMAT), "harness": want.harness, "harness-version": want.version,
                "harbor-version": want.harbor_version, "recipe-sha256": want.recipe}
    wrong = [f"{k}={labels.get(LABEL_NS + k)!r} (wanted {v!r})" for k, v in expected.items()
             if labels.get(LABEL_NS + k) != v]
    common = dict(image=image, image_id=baked.id, baked_at=labels.get(LABEL_NS + "baked-at"),
                  recipe=labels.get(LABEL_NS + "recipe-sha256"))
    if wrong:
        return Decision(STALE, "stale: labels differ: " + ", ".join(wrong), **common)
    if base is None:
        return Decision(UNVERIFIABLE, f"the task image {want.task_image} is not local, so the pre-baked "
                                      "image's base cannot be checked", **common)
    common["task_image_id"] = base.id
    if labels.get(LABEL_NS + "base-image-id") != base.id:
        return Decision(STALE, f"stale: baked from {labels.get(LABEL_NS + 'base-image-id')}, but the local task "
                               f"image is {base.id} (re-pulled since the bake?)", **common)
    noop = labels.get(LABEL_NS + "noop") == "true"
    prefix = baked.layers[: len(base.layers)] == base.layers
    if not prefix or (noop and len(baked.layers) != len(base.layers)) or \
            (not noop and len(baked.layers) <= len(base.layers)):
        return Decision(STALE, "stale: not built FROM the local task image (layer mismatch)", **common)
    if baked.os != base.os:
        return Decision(STALE, f"pre-baked image OS {baked.os} != task image OS {base.os}", **common)
    skips = _digest_list(labels.get(LABEL_NS + "skip-execs"))
    skip_commands = _digest_list(labels.get(LABEL_NS + "skip-commands"))
    if skips is None or skip_commands is None:
        return Decision(STALE, "stale: unreadable skip-execs or skip-commands label", **common)
    return Decision(READY, "pre-baked image verified", noop=noop,
                    added_layers=len(baked.layers) - len(base.layers), skip_execs=skips,
                    skip_commands=skip_commands, **common)


def _flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _text(value: Any) -> str | None:
    return None if value is None or str(value).strip() == "" else str(value).strip()


class PrebakedDockerEnvironment(DockerEnvironment):
    def __init__(self, *args: Any, prebake_harness: str | None = None, prebake_version: str | None = None,
                 prebake_recipe: str | None = None, prebake_prefix: str = DEFAULT_PREFIX,
                 prebake_required: bool = False, **kwargs: Any):
        self._prebake_harness = _text(prebake_harness)
        self._prebake_version = _text(prebake_version)
        self._prebake_recipe = _text(prebake_recipe)
        self._prebake_prefix = (_text(prebake_prefix) or DEFAULT_PREFIX).strip("/")
        self._prebake_required = _flag(prebake_required)
        self._decision: Decision | None = None
        self._skip_execs: set[str] = set()
        self._skip_commands: set[str] = set()
        self._near_misses: set[str] = set()
        self._skipped: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    @classmethod
    def preflight(cls) -> None:
        """Harbor's Docker check, with room for a busy daemon. Harbor gives
        `docker info` 10 s and then reports Docker as not running; with several
        harbor processes starting at once that check failed while Docker was up
        (2026-10-31, 8 of a race's first 17 lanes, at both the dry run and the
        real run). Same command and message, a longer limit and two retries."""
        import shutil
        import subprocess
        import time

        if not shutil.which("docker"):
            raise SystemExit("Docker is not installed or not on PATH. Please install Docker and try again.")
        for wait in PREFLIGHT_WAITS:
            time.sleep(wait)
            try:
                subprocess.run(cls._engine_cmd("info"), capture_output=True, timeout=PREFLIGHT_TIMEOUT_SEC, check=True)
                return
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
        raise SystemExit("Docker daemon is not running. Please start Docker and try again.")

    # Decision -----------------------------------------------------------------

    @property
    def _is_agent_environment(self) -> bool:
        # Trial names the agent environment "{trial_name}__env" and a separate
        # verifier environment "{trial_name}__verifier__{key}" (trial.py).
        return self.session_id.endswith("__env")

    @property
    def _asked(self) -> bool:
        """This environment is the trial's agent environment and the arm asked for a pre-baked image."""
        return self._is_agent_environment and self._prebake_harness is not None

    async def _inspect(self, ref: str) -> ImageInfo | None:
        # The whole JSON: a template naming .Config.Labels fails on an image without labels.
        proc = await asyncio.create_subprocess_exec(
            *type(self)._engine_cmd("image", "inspect", ref),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        try:
            return ImageInfo.from_inspect(json.loads(out.decode("utf-8", "replace"))[0])
        except (ValueError, IndexError, TypeError, AttributeError):
            return None

    async def _decide(self, force_build: bool) -> Decision:
        if not self._is_agent_environment:
            return Decision(NOT_APPLICABLE, "not the agent environment")
        if not (self._prebake_harness and self._prebake_version and self._prebake_recipe):
            return Decision(NOT_APPLICABLE, "no prebake_harness, prebake_version and prebake_recipe given")
        task_image = self.task_env_config.docker_image
        if not task_image:
            return Decision(NOT_APPLICABLE, "the task has no docker_image")
        if force_build and self._dockerfile_path.exists():
            return Decision(NOT_APPLICABLE, "force_build: Harbor builds the task image instead")
        want = Want(harness=self._prebake_harness, version=self._prebake_version, recipe=self._prebake_recipe,
                    task_image=task_image, prefix=self._prebake_prefix)
        try:
            candidate = want.image
        except ValueError as exc:
            return Decision(NOT_APPLICABLE, f"no pre-baked image name: {exc}")
        baked, base = await asyncio.gather(self._inspect(candidate), self._inspect(task_image))
        return check_candidate(baked, base, want)

    def _record(self) -> dict[str, Any]:
        decision = self._decision or Decision(NOT_APPLICABLE, "not decided yet")
        task_image = self.task_env_config.docker_image
        return {
            "bake_format": BAKE_FORMAT, **decision.to_dict(),
            "harness": self._prebake_harness, "version": self._prebake_version,
            "wanted_recipe": self._prebake_recipe, "prefix": self._prebake_prefix,
            "required": self._prebake_required, "task_image": task_image,
            "ran_image": decision.image if decision.used else task_image,
            "skipped_execs": list(self._skipped),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }

    def _write_record(self) -> None:
        if not self._asked:
            return
        try:
            path = self.trial_paths.trial_dir / RECORD_NAME
            path.write_text(json.dumps(self._record(), indent=2), encoding="utf-8")
        except OSError as exc:  # provenance only; never fail a trial for it
            self.logger.warning(f"prebake: could not write {RECORD_NAME}: {exc}")

    # Lifecycle ----------------------------------------------------------------

    async def start(self, force_build: bool):
        self._decision = await self._decide(force_build)
        if self._asked:
            self._write_record()
            decision = self._decision
            if decision.used:
                # The one value Harbor's prebuilt compose file reads:
                # services.main.image: ${PREBUILT_IMAGE_NAME}
                self._env_vars.prebuilt_image_name = decision.image
                self._skip_execs = set(decision.skip_execs)
                self._skip_commands = set(decision.skip_commands)
                self.logger.info(f"prebake: using {decision.image} ({(decision.image_id or '')[:19]})")
            elif self._prebake_required:
                raise SandboxBuildFailedError(f"prebake required but not usable: {decision.reason}")
            else:
                self.logger.warning(f"prebake: stock setup on {self.task_env_config.docker_image}: "
                                    f"{decision.reason}")
        await super().start(force_build)

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: str | int | None = None) -> ExecResult:
        if self._skip_execs:
            sig = exec_signature(command, self._resolve_user(user), cwd or self.task_env_config.workdir,
                                 self._merge_env(env))
            if sig in self._skip_execs:
                self._skip_execs.discard(sig)  # at most once per trial
                self._skipped.append({"signature": sig, "command_head": command[:120],
                                      "at": datetime.now(timezone.utc).isoformat()})
                self._write_record()
                self.logger.info(f"prebake: skipped an install exec the image already ran ({sig[:12]})")
                return ExecResult(stdout="", stderr="", return_code=0)
            digest = command_digest(command)
            if digest in self._skip_commands and digest not in self._near_misses:
                self._near_misses.add(digest)
                self.logger.warning("prebake: an install exec matches a skipped command but not its user, cwd "
                                    f"or env, so it runs ({sig[:12]})")
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)
