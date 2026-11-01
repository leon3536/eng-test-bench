"""Install recipes taken from each harness's own Harbor setup() code.

The harness's real setup() runs against a recording environment: every
command and upload is recorded in order and never run, and the read-only
probes its adapter names (`command -v ...`, the package-manager loop,
`claude --version`, `tmux -V`, ...) are answered:

- ProbeEnv(task image): for real, by `docker run --rm --pull never --network
  none -u root <task image> bash -c <probe>`, cached per image. So what a bake
  installs is exactly what a live trial would: procps is skipped on the Ubuntu
  images, curl where the image ships it, terminus-2's install where tmux and
  asciinema are already there.
- BareEnv: as a bare apt image would answer, so setup() issues the full stock
  install. BakedEnv: as the baked image will, i.e. what a trial issues there.

A probe asked again after an install step checks that step, so it gets the
baked answer. arm_recipe() hashes the BareEnv and BakedEnv transcripts (with
Harbor's version, the host platform and every uploaded byte) into the recipe
that names an arm's images: a new harness version, an edited strands factory,
record_terminal_session or a Harbor upgrade makes a new recipe, while options
install() never reads (effort, caps, max_turns, model) do not.

A task image that is not root, not apt-based or not local raises RecipeError,
and that task runs stock.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from .. import keys
from ..environments.prebaked import BAKE_FORMAT, command_digest, exec_signature, harbor_version, redacted_env
from ..fsutil import canonical_json, sha256_hex
from .adapters import Adapter

log = logging.getLogger(__name__)

PROBE_TIMEOUT_SEC = 120
_ROOT_USERS = (None, "root", "0")
_ROOT_IMAGE_USERS = ("", "root", "0", "0:0", "root:root")


class RecipeError(Exception):
    """This harness or task image cannot be pre-baked; its trials run stock setup."""


@dataclass(frozen=True)
class Step:
    """One exec Harbor issues: an install step, or a probe with its answer."""
    command: str
    user: str | None = None             # resolved: None is the image's user (root on every baked image)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)  # per-exec env, as Harbor merges it
    timeout_sec: int | None = None
    probe: str | None = None            # the probe's name; None for an install step
    rc: int | None = None               # a probe's exit code

    @property
    def signature(self) -> str:
        return exec_signature(self.command, self.user, self.cwd, self.env or None)

    def replaced(self, **changes: Any) -> Step:
        return replace(self, **changes)

    def entry(self) -> dict[str, Any]:
        out: dict[str, Any] = {"exec": self.command, "user": self.user, "cwd": self.cwd,
                               "env": redacted_env(self.env)}
        if self.probe is not None:
            out.update(probe=self.probe, rc=self.rc)
        return out


@dataclass(frozen=True)
class Upload:
    """A file or folder setup() copies into the container (docker compose cp)."""
    source: Path          # on the host, staged by the agent's own install()
    target: str
    is_dir: bool
    sha256: str

    def entry(self) -> dict[str, Any]:
        return {"upload": self.target, "dir": self.is_dir, "sha256": self.sha256}


@dataclass(frozen=True)
class Recording:
    """Everything one harness's setup() issued against one environment."""
    family: str
    version: str | None
    variant: str
    task_image: str | None
    steps: tuple[Step | Upload, ...]

    @property
    def install_steps(self) -> tuple[Step | Upload, ...]:
        return tuple(s for s in self.steps if not (isinstance(s, Step) and s.probe))

    @property
    def noop(self) -> bool:
        """Harbor would install nothing on this image (terminus-2 where tmux and asciinema exist)."""
        return not self.install_steps

    @property
    def probes(self) -> list[dict[str, Any]]:
        return [{"command": s.command, "rc": s.rc} for s in self.steps if isinstance(s, Step) and s.probe]


@dataclass(frozen=True)
class Recipe:
    """An arm's install, the same for every task: what names and labels its images."""
    sha256: str
    family: str
    agent: str
    version: str | None
    variant: str
    harbor_version: str
    bare: tuple[Step | Upload, ...]     # the full stock install, as baked
    baked: tuple[Step | Upload, ...]    # what a trial issues on the image
    skip_execs: tuple[str, ...]         # signatures of the baked steps a trial may skip
    skip_commands: tuple[str, ...]
    document: dict[str, Any]            # what sha256 hashes


def _user(user: str | int | None) -> str | None:
    return None if user is None else str(user)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_dir():
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
            h.update(f.relative_to(path).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    else:
        h.update(path.read_bytes())
    return h.hexdigest()


def run_sync(coro: Any) -> Any:
    """Runs a coroutine to the end from sync code, also from a thread with a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


# Recording environments ---------------------------------------------------------


class _RecordingEnv:
    """Just enough of Harbor's DockerEnvironment for a harness's setup().
    Uploads go through Harbor's own UnixOps, so the CRLF fix it runs after an
    upload on a Windows host is recorded too."""

    def __init__(self, adapter: Adapter, stage_dir: Path) -> None:
        from harbor.environments.docker.docker_unix import UnixOps

        self.default_user = None
        self.session_id = "prebake-recorder__env"
        self.task_env_config = SimpleNamespace(workdir=None, docker_image=None)
        self.trial_paths = SimpleNamespace(trial_dir=stage_dir, agent_dir=stage_dir / "agent")
        self.logger = log
        self.steps: list[Step | Upload] = []
        self._adapter = adapter
        self._asked: dict[str, int] = {}
        self._installs = 0
        self._ops = UnixOps(self)  # type: ignore[arg-type]

    def answer(self, name: str, command: str) -> tuple[int, str] | None:
        raise NotImplementedError

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: str | int | None = None) -> Any:
        from harbor.environments.base import ExecResult

        user = user if user is not None else self.default_user
        cwd = cwd or self.task_env_config.workdir
        name = self._adapter.probe_name(command)
        if name is not None:
            check = command in self._asked and self._installs > self._asked[command]
            self._asked[command] = self._installs
            answer = self._adapter.baked_answer(name, command) if check else self.answer(name, command)
            if answer is None:
                raise RecipeError(f"unknown probe '{name}': {command[:120]}")
            rc, out = answer
            self.steps.append(Step(command, _user(user), cwd, dict(env or {}), timeout_sec, probe=name, rc=rc))
            return ExecResult(stdout=out, stderr="", return_code=rc)
        if _user(user) not in _ROOT_USERS:
            raise RecipeError(f"an install step runs as user {user!r}; only root installs are baked")
        self.steps.append(Step(command, _user(user), cwd, dict(env or {}), timeout_sec))
        self._installs += 1
        return ExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._ops.upload_file(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._ops.upload_dir(source_dir, target_dir)

    async def _run_docker_compose_command(self, command: list[str], check: bool = True,
                                          timeout_sec: int | None = None, **_: Any) -> Any:
        """What UnixOps asks of the environment for an upload: `cp`, and on
        Windows `exec main bash -c <CRLF fix>` as the container's user."""
        from harbor.constants import MAIN_SERVICE_NAME
        from harbor.environments.base import ExecResult

        if len(command) == 3 and command[0] == "cp" and command[2].startswith(f"{MAIN_SERVICE_NAME}:"):
            source, target = command[1], command[2].split(":", 1)[1]
            is_dir = source.endswith("/.")
            path = Path(source[:-2] if is_dir else source)
            self.steps.append(Upload(path, target, is_dir, _digest(path)))
        elif len(command) == 5 and command[:4] == ["exec", MAIN_SERVICE_NAME, "bash", "-c"]:
            self.steps.append(Step(command[4]))
            self._installs += 1
        else:
            raise RecipeError(f"unexpected docker compose call during setup: {' '.join(command[:3])}")
        return ExecResult(stdout="", stderr="", return_code=0)


class BareEnv(_RecordingEnv):
    """Probes answer as on an apt-based image with nothing installed: the full stock install."""

    def answer(self, name: str, command: str) -> tuple[int, str] | None:
        return self._adapter.bare_answer(name, command)


class BakedEnv(_RecordingEnv):
    """Probes answer as on the baked image: what a trial issues there."""

    def answer(self, name: str, command: str) -> tuple[int, str] | None:
        return self._adapter.baked_answer(name, command)


class ProbeRunner(Protocol):
    def image(self, image: str) -> dict[str, str]: ...

    def run(self, image: str, command: str) -> tuple[int, str]: ...


class DockerProbes:
    """Runs probes in throwaway containers of a local task image: no network,
    no pull, no credentials in the docker client's environment. Cached."""

    def __init__(self, timeout_sec: float = PROBE_TIMEOUT_SEC) -> None:
        self.timeout_sec = timeout_sec
        self._images: dict[str, dict[str, str]] = {}
        self._answers: dict[tuple[str, str], tuple[int, str]] = {}
        self._lock = threading.Lock()

    def _docker(self, args: Sequence[str]) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", *args], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=self.timeout_sec, env=keys.child_env(None))

    def image(self, image: str) -> dict[str, str]:
        with self._lock:
            if image in self._images:
                return self._images[image]
        # The whole JSON: a template naming .Config.User fails when the image sets no user.
        r = self._docker(["image", "inspect", image])
        if r.returncode != 0:
            raise RecipeError(f"{image} is not local (recording never pulls)")
        data = json.loads(r.stdout)[0]
        config = data.get("Config") or {}
        info = {"Id": str(data.get("Id") or ""), "Os": str(data.get("Os") or ""), "User": str(config.get("User") or "")}
        if self.run(image, "true")[0] != 0:
            raise RecipeError(f"{image} has no bash")
        with self._lock:
            self._images[image] = info
        return info

    def run(self, image: str, command: str) -> tuple[int, str]:
        with self._lock:
            if (image, command) in self._answers:
                return self._answers[(image, command)]
        r = self._docker(["run", "--rm", "--pull", "never", "--network", "none", "-u", "root",
                          "--entrypoint", "bash", image, "-c", command])
        if r.returncode in (125, 126) or "executable file not found" in (r.stderr or ""):
            raise RecipeError(f"docker could not run a probe in {image}: {(r.stderr or '').strip()[-300:]}")
        with self._lock:
            self._answers[(image, command)] = (r.returncode, r.stdout)
        return r.returncode, r.stdout


PROBES = DockerProbes()


class ProbeEnv(_RecordingEnv):
    """Probes run for real in the task image, so the recording is exactly
    what a stock trial on it would install."""

    def __init__(self, adapter: Adapter, stage_dir: Path, task_image: str, probes: ProbeRunner) -> None:
        super().__init__(adapter, stage_dir)
        self.task_image = task_image
        self.task_env_config.docker_image = task_image
        self._probes = probes

    def answer(self, name: str, command: str) -> tuple[int, str] | None:
        rc, out = self._probes.run(self.task_image, command)
        if name == "package-manager" and out.strip() != "apt-get":
            raise RecipeError(f"{self.task_image} is not apt-based (package manager: {out.strip() or 'none'})")
        if name == "which" and command.startswith("which apt-get ") and rc != 0:
            raise RecipeError(f"{self.task_image} is not apt-based")
        return rc, out


# Recording ------------------------------------------------------------------------


def _record(adapter: Adapter, env: _RecordingEnv, stage_dir: Path) -> list[Step | Upload]:
    logs_dir = stage_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_sync(adapter.record(env, logs_dir))
    except RecipeError:
        raise
    except Exception as exc:
        raise RecipeError(f"{adapter.family} setup() could not be recorded: {type(exc).__name__}: {exc}") from exc
    return list(env.steps)


def record_for_image(adapter: Adapter, task_image: str, stage_dir: Path,
                     probes: ProbeRunner | None = None) -> Recording:
    """What the harness's setup() installs on this task image, as the bake
    replays it. Uploaded files are staged under stage_dir by the agent's own
    install() and stay there for the build context."""
    probes = probes or PROBES
    info = probes.image(task_image)
    if (info.get("User") or "") not in _ROOT_IMAGE_USERS:
        raise RecipeError(f"{task_image} runs as user {info.get('User')!r}; only root images are baked")
    if (info.get("Os") or "linux") != "linux":
        raise RecipeError(f"{task_image} is a {info.get('Os')} image; only linux images are baked")
    env = ProbeEnv(adapter, stage_dir, task_image, probes)
    steps = adapter.postprocess(_record(adapter, env, stage_dir))
    return Recording(adapter.family, adapter.version, adapter.variant, task_image, tuple(steps))


_recipes: dict[tuple[str, ...], Recipe] = {}
_recipes_lock = threading.Lock()


def arm_recipe(adapter: Adapter) -> Recipe:
    """The arm's recipe, the same for every task. Records against BareEnv and
    BakedEnv only, so it needs no Docker. Cached on Harbor's version, the host
    platform and everything the adapter's install depends on."""
    key = (harbor_version(), sys.platform, adapter.cache_key())
    with _recipes_lock:
        if key in _recipes:
            return _recipes[key]
    with tempfile.TemporaryDirectory(prefix="bench-prebake-") as tmp:
        bare = tuple(adapter.postprocess(_record(adapter, BareEnv(adapter, Path(tmp) / "bare"), Path(tmp) / "bare")))
        baked = tuple(_record(adapter, BakedEnv(adapter, Path(tmp) / "baked"), Path(tmp) / "baked"))
    skips = [s for s in baked if adapter.skippable(s)]
    skip_execs = tuple(sorted({s.signature for s in skips}))  # type: ignore[union-attr]
    skip_commands = tuple(sorted({command_digest(s.command) for s in skips}))  # type: ignore[union-attr]
    document = {
        "bake_format": BAKE_FORMAT, "harbor": key[0], "host_platform": key[1], "family": adapter.family,
        "agent": adapter.agent, "version": adapter.version, "variant": adapter.variant,
        "bare": [s.entry() for s in bare], "baked": [s.entry() for s in baked], "skip_execs": list(skip_execs),
    }
    recipe = Recipe(sha256=sha256_hex(canonical_json(document)), family=adapter.family, agent=adapter.agent,
                    version=adapter.version, variant=adapter.variant, harbor_version=key[0], bare=bare,
                    baked=baked, skip_execs=skip_execs, skip_commands=skip_commands, document=document)
    with _recipes_lock:
        if len(_recipes) > 64:
            _recipes.clear()
        _recipes[key] = recipe
    return recipe
