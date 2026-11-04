"""What pre-baking needs to know per harness.

An adapter builds the arm's Harbor agent exactly as a trial would and runs
its setup() against a recording environment (recorder.py), names the
read-only probes that setup() issues and how they answer on a bare apt image
and on a baked one, and says how its bake differs from a plain replay:

- ClaudeCodeAdapter, InstalledAgentAdapter (mini-swe-agent) and
  Terminus2Adapter: plain. Claude Code needs no skip, because its own
  `claude --version` check returns early on the image.
- PiAdapter (pi-warm): bakes nvm, Node and the npm cache but removes the
  global package in the same RUN, so the trial's own `npm install -g` writes
  into the container's layer. A full bake re-ran in 336-370 s, because npm
  cannot rename the old package out of the image's lower layer.
- StrandsAdapter (strands-skip): a full bake whose uv venv/pip step is marked
  skippable. Its re-run on the image took 503-590 s (uv hard-links ~18k files,
  each forcing a copy-up), past Harbor's 360 s setup limit.
- ClineAdapter (cline-warm), OpenCodeAdapter (opencode-warm) and KiloAdapter
  (kilo-warm): pi's recipe for their global npm packages (cline, opencode-ai,
  @kilocode/cli), which a trial installs again with `npm install -g` just as
  pi's does. Cline's install also copies
  GITHUB_TOKEN/GH_TOKEN from the bench's own environment into its exec; they
  are left out of its recording, so no token reaches a Dockerfile or recipe.
- GooseAdapter (goose-skip): a full bake that marks skippable the apt step
  Harbor always re-runs (libxcb1 and libgomp1 are always_install packages)
  and, for a pinned release, the binary download, which would only replace
  the baked binary with the same release.
These four are recorded but not yet built or verified: their timings and
sizes are estimates until a build measures them.

A harness with no adapter is not supported and runs stock.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from ..environments.prebaked import harbor_version
from ..settings import ROOT

# Probes every installed agent issues (BaseInstalledAgent.ensure_system_dependencies).
_COMMAND_V = re.compile(r"command -v \S+ >/dev/null 2>&1(?: && command -v \S+ >/dev/null 2>&1)*")
_PACKAGE_MANAGER = re.compile(r"for manager in [\w -]+; do .*; done")
INSTALLED_PROBES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("command-v", _COMMAND_V),
    ("package-manager", _PACKAGE_MANAGER),
)
# What a bare apt-based image says to a terminal tool or OS probe.
_OS_RELEASE = 'PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"\nID=debian\n'


def harbor_agents() -> None:
    """Call before importing Harbor's agent modules. They import LiteLLM,
    which loads the project's .env into this process on import unless
    dotenv loading is off. The bench reads .env itself (keys.py) and never
    wants that, so it is off here as in every child (keys.child_env);
    dotenv_values, which keys.py uses, ignores the switch."""
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"


class Adapter:
    family: ClassVar[str] = ""
    variant: ClassVar[str] = "plain"
    # Read-only probes this harness's setup() issues: (name, full-match pattern).
    PROBES: ClassVar[tuple[tuple[str, re.Pattern[str]], ...]] = INSTALLED_PROBES
    setup_budget_sec: ClassVar[float] = 20.0   # verify --setup refuses an image whose setup takes longer
    expected_gb: ClassVar[float] = 1.5         # added per image, for the builder's disk guard
    expected_build_min: ClassVar[float] = 3.0  # per image, measured with 3-4 builds at once
    stock_setup_sec: ClassVar[float] = 60.0    # Harbor's own install in a trial, measured median (plan warnings)

    def __init__(self, agent: str, kwargs: Mapping[str, Any], model_name: str) -> None:
        self.agent = agent
        self.kwargs = dict(kwargs)
        self.model_name = model_name

    @classmethod
    def matches(cls, agent: str) -> bool:
        """Whether the arm's agent is the Harbor harness this adapter knows."""
        raise NotImplementedError

    @property
    def version(self) -> str | None:
        """The pinned harness version, None when Harbor would install the latest."""
        value = self.kwargs.get("version")
        return None if value in (None, "", "latest") else str(value)

    @property
    def smoke(self) -> str:
        """An offline check that the bake installed the harness (verify)."""
        raise NotImplementedError

    def cache_key(self) -> str:
        """Everything recording depends on besides Harbor's own code."""
        inputs = {"family": self.family, "agent": self.agent, "kwargs": self.kwargs, "model": self.model_name,
                  "files": self.input_files_digest()}
        return json.dumps(inputs, sort_keys=True, default=str)

    def input_files_digest(self) -> str:
        """A digest of host files install() uploads, so an edited file makes a new recipe."""
        return ""

    # Probes -----------------------------------------------------------------

    def probe_name(self, command: str) -> str | None:
        for name, pattern in self.PROBES:
            if pattern.fullmatch(command):
                return name
        return None

    def bare_answer(self, name: str, command: str) -> tuple[int, str] | None:
        """(exit code, stdout) on an apt-based image with nothing installed."""
        return {"command-v": (1, ""), "package-manager": (0, "apt-get")}.get(name)

    def baked_answer(self, name: str, command: str) -> tuple[int, str] | None:
        """(exit code, stdout) on the baked image, where the install already ran."""
        return {"command-v": (0, ""), "package-manager": (0, "apt-get")}.get(name)

    # Recording --------------------------------------------------------------

    def make_agent(self, logs_dir: Path) -> Any:
        """The agent exactly as Harbor's Trial builds it from the arm's config."""
        harbor_agents()
        from harbor.agents.factory import AgentFactory

        return AgentFactory.create_agent_from_config(self._agent_config(), logs_dir=logs_dir)

    def _agent_config(self) -> Any:
        from harbor.models.trial.config import AgentConfig

        if ":" in self.agent:
            return AgentConfig(import_path=self.agent, model_name=self.model_name, kwargs=self.agent_kwargs())
        return AgentConfig(name=self.agent, model_name=self.model_name, kwargs=self.agent_kwargs())

    def agent_kwargs(self) -> dict[str, Any]:
        return dict(self.kwargs)

    async def record(self, env: Any, logs_dir: Path) -> None:
        """Runs the harness's own setup() against env."""
        await self.make_agent(logs_dir).setup(environment=env)

    def postprocess(self, steps: list[Any]) -> list[Any]:
        """The recorded install steps as the Dockerfile replays them."""
        return steps

    def skippable(self, step: Any) -> bool:
        """Whether a trial on the image may skip this exec (skip-execs label)."""
        return False


def _agent_class(agent: str) -> type | None:
    harbor_agents()
    from harbor.agents.factory import AgentFactory
    from harbor.models.trial.config import AgentConfig

    try:
        config = AgentConfig(import_path=agent) if ":" in agent else AgentConfig(name=agent)
        return AgentFactory.get_agent_class_from_config(config)
    except Exception:
        return None


@functools.lru_cache(maxsize=256)
def _subclass(agent: str, import_path: str) -> bool:
    from harbor.utils.import_path import import_class

    cls = _agent_class(agent)
    return cls is not None and issubclass(cls, import_class(import_path, label="agent"))


class ClaudeCodeAdapter(Adapter):
    family = "claude-code"
    setup_budget_sec = 10.0
    expected_gb = 1.9
    expected_build_min = 11.0
    stock_setup_sec = 330.0

    @classmethod
    def matches(cls, agent: str) -> bool:
        return _subclass(agent, "harbor.agents.installed.claude_code:ClaudeCode")

    @classmethod
    def _commands(cls) -> tuple[str, str]:
        harbor_agents()
        from harbor.agents.installed.claude_code import ClaudeCode

        return ClaudeCode._INSTALL_VERSION_COMMAND, ClaudeCode._INSTALL_CHECK_COMMAND

    def probe_name(self, command: str) -> str | None:
        version_command, check_command = self._commands()
        if command == version_command:
            return "claude-version"
        if command == check_command:
            return "claude-present"
        return super().probe_name(command)

    def bare_answer(self, name: str, command: str) -> tuple[int, str] | None:
        return {"claude-version": (127, ""), "claude-present": (1, "")}.get(name) or super().bare_answer(name, command)

    def baked_answer(self, name: str, command: str) -> tuple[int, str] | None:
        answers = {"claude-version": (0, f"{self.version} (Claude Code)\n"), "claude-present": (0, "")}
        return answers.get(name) or super().baked_answer(name, command)

    @property
    def smoke(self) -> str:
        return ('export PATH="$HOME/.local/bin:$PATH"; '
                f'claude --version | grep -q "^{re.escape(self.version or "")} " && command -v node npm curl pgrep')


class InstalledAgentAdapter(Adapter):
    """mini-swe-agent, Harbor's own or the bench's subclass that adds mcp."""
    family = "mini-swe-agent"
    setup_budget_sec = 20.0
    expected_gb = 1.7
    expected_build_min = 3.6
    stock_setup_sec = 90.0

    @classmethod
    def matches(cls, agent: str) -> bool:
        return _subclass(agent, "harbor.agents.installed.mini_swe_agent:MiniSweAgent")

    @property
    def smoke(self) -> str:
        return ('export PATH="$HOME/.local/bin:$PATH"; '
                f'uv tool list | grep -qx "mini-swe-agent v{re.escape(self.version or "")}"')


class NpmWarmAdapter(Adapter):
    """Bakes nvm, Node and the npm cache but removes the harness's global npm
    package in the same RUN, so the trial's own `npm install -g` writes into
    the container's layer instead of renaming the old package out of the
    image's lower layer (pi's full bake re-ran in 336-370 s)."""
    bin_names: ClassVar[tuple[str, ...]] = ()  # the commands the package links into npm's bin
    _GLOBAL_INSTALL: ClassVar[re.Pattern[str]]  # group 1: the package

    @property
    def smoke(self) -> str:
        harbor_agents()
        from harbor.agents.installed.node_install import DEFAULT_NODE_MAJOR

        absent = " && ".join(f"! command -v {name}" for name in self.bin_names)
        return f'. "$HOME/.nvm/nvm.sh" >/dev/null && node -v | grep -q "^v{DEFAULT_NODE_MAJOR}\\." && {absent}'

    def postprocess(self, steps: list[Any]) -> list[Any]:
        """Same RUN as the stock install, then removes in that layer exactly what
        the trial's own `npm install -g` creates again (the global package and
        its bin links) plus the bake's npm logs and Node compile cache."""
        from .recorder import RecipeError, Step

        hits = [i for i, s in enumerate(steps) if isinstance(s, Step) and self._GLOBAL_INSTALL.search(s.command)]
        if len(hits) != 1:
            raise RecipeError(f"{self.family}: expected one global npm install among the install steps, "
                              f"found {len(hits)}")
        step = steps[hits[0]]
        package = self._GLOBAL_INSTALL.search(step.command).group(1)  # type: ignore[union-attr]
        scope = package.split("/")[0]
        links = " ".join(f'"$P/bin/{name}"' for name in self.bin_names)
        gone = " && ".join(f'test ! -e "$P/bin/{name}"' for name in self.bin_names)
        unbake = ('P="$(npm config get prefix)" && '
                  f'rm -rf "$P/lib/node_modules/{scope}" {links} "$HOME/.npm/_logs" /tmp/node-compile-cache && {gone}')
        out = list(steps)
        out[hits[0]] = step.replaced(command=f"{step.command} && {unbake}")
        return out


class PiAdapter(NpmWarmAdapter):
    family = "pi"
    variant = "pi-warm"
    setup_budget_sec = 60.0
    expected_gb = 0.6
    expected_build_min = 1.5
    stock_setup_sec = 40.0
    bin_names = ("pi",)
    _GLOBAL_INSTALL = re.compile(r"npm install -g --ignore-scripts (@?[^\s@]+(?:/[^\s@]+)?)@\S+")

    @classmethod
    def matches(cls, agent: str) -> bool:
        return _subclass(agent, "harbor.agents.installed.pi:Pi")

    async def record(self, env: Any, logs_dir: Path) -> None:
        agent = self.make_agent(logs_dir)
        if agent.mcp_servers:
            # Its adapter installs into another npm prefix, which this bake does not unbake.
            raise ValueError("pi with MCP servers is not supported")
        await agent.setup(environment=env)


class ClineAdapter(NpmWarmAdapter):
    """Harbor's cline-cli, or the bench's PinnedClineCli. The version is what
    npm installs: cline_version, which PinnedClineCli sets from version."""
    family = "cline-cli"
    variant = "cline-warm"
    # The trial's own npm install of cline still runs: 83 s alone, 106 s with four builds at once.
    setup_budget_sec = 180.0
    expected_gb = 0.9
    expected_build_min = 2.0
    stock_setup_sec = 60.0
    bin_names = ("cline",)
    _GLOBAL_INSTALL = re.compile(r"npm install -g (cline)@\S+ && ")
    # Options that install Cline from somewhere other than npm, which this bake does not handle.
    _SOURCES = ("tarball_url", "tarball_path", "github_user", "commit_hash", "plugin_source", "plugin_tarball_url",
                "plugin_tarball_path")
    # Credentials Harbor's install() copies from the bench's own environment into its exec.
    _TOKENS = ("GITHUB_TOKEN", "GH_TOKEN")

    @classmethod
    def matches(cls, agent: str) -> bool:
        return _subclass(agent, "harbor.agents.installed.cline.cline:ClineCli")

    @property
    def version(self) -> str | None:
        """version, when npm installs exactly it: PinnedClineCli installs its
        version; Harbor's own adapter installs cline_version (else nightly)
        and reports version without checking it."""
        value, npm = self.kwargs.get("version"), self.kwargs.get("cline_version")
        if value in (None, "", "latest", "nightly") or any(self.kwargs.get(k) for k in self._SOURCES):
            return None
        if npm not in (None, "") and str(npm) != str(value):
            return None
        if npm in (None, "") and not _subclass(self.agent, "bench.agents.cline:PinnedClineCli"):
            return None
        return str(value)

    @property
    def smoke(self) -> str:
        return f"{super().smoke} && command -v git curl unzip >/dev/null"

    async def record(self, env: Any, logs_dir: Path) -> None:
        from .recorder import Step

        if any(self.kwargs.get(k) for k in self._SOURCES):
            raise ValueError("cline-cli from a tarball, a git source or with a plugin is not supported")
        steps = getattr(env, "steps", None)  # a recording; the equivalence check passes a live container instead
        start = len(steps) if steps is not None else 0
        await self.make_agent(logs_dir).setup(environment=env)
        for i in range(start, len(steps) if steps is not None else 0):
            step = steps[i]
            if isinstance(step, Step) and any(k in step.env for k in self._TOKENS):
                steps[i] = step.replaced(env={k: v for k, v in step.env.items() if k not in self._TOKENS})


class OpenCodeAdapter(NpmWarmAdapter):
    family = "opencode"
    variant = "opencode-warm"
    setup_budget_sec = 60.0
    expected_gb = 0.9
    expected_build_min = 2.0
    stock_setup_sec = 60.0
    bin_names = ("opencode",)
    _GLOBAL_INSTALL = re.compile(r"npm i -g (opencode-ai)@\S+ && ")

    @classmethod
    def matches(cls, agent: str) -> bool:
        # Kilo Code subclasses OpenCode but installs another package (KiloAdapter).
        return (_subclass(agent, "harbor.agents.installed.opencode:OpenCode")
                and not _subclass(agent, "bench.agents.kilo:KiloCode"))

    @property
    def smoke(self) -> str:
        return f"{super().smoke} && command -v stdbuf >/dev/null"


class KiloAdapter(NpmWarmAdapter):
    """The bench's KiloCode (bench/agents/kilo.py). Its package links two
    commands, kilo and kilocode, to one file; the unbake removes both, with
    the whole @kilocode scope and so its prebuilt binary packages."""
    family = "kilo-code"
    variant = "kilo-warm"
    # Estimates, not measured: OpenCode's scaled by Kilo's larger native packages.
    # On glibc x64 npm installs both @kilocode/cli-linux-x64 and its -baseline
    # (no libc field), 326.7 MB unpacked and 419 files each, against 214 MB and 2
    # files for each opencode-linux-x64 package. Set these from the first
    # kilo-warm build and its `verify --setup`.
    setup_budget_sec = 90.0
    expected_gb = 1.3
    expected_build_min = 3.0
    stock_setup_sec = 90.0
    bin_names = ("kilo", "kilocode")
    _GLOBAL_INSTALL = re.compile(r"npm i -g (@kilocode/cli)@\S+ && ")

    @classmethod
    def matches(cls, agent: str) -> bool:
        return _subclass(agent, "bench.agents.kilo:KiloCode")

    @property
    def smoke(self) -> str:
        return f"{super().smoke} && command -v stdbuf >/dev/null"


class GooseAdapter(Adapter):
    """Harbor's goose, or the bench's PinnedGoose (only it installs exactly
    the pinned release; see bench/agents/goose.py)."""
    family = "goose"
    variant = "goose-skip"
    setup_budget_sec = 20.0
    expected_gb = 0.5
    expected_build_min = 1.5
    stock_setup_sec = 45.0
    _ALWAYS_APT = "apt-get install -y libxcb1 libgomp1"

    @classmethod
    def matches(cls, agent: str) -> bool:
        return _subclass(agent, "harbor.agents.installed.goose:Goose")

    @property
    def version(self) -> str | None:
        """None for Harbor's own Goose: it installs the latest stable release whatever the pin."""
        if not _subclass(self.agent, "bench.agents.goose:PinnedGoose"):
            return None
        return super().version

    @property
    def smoke(self) -> str:
        return ('export PATH="$HOME/.local/bin:$PATH"; '
                f'goose --version | grep -q "{re.escape(self.version or "")}" && dpkg -s libxcb1 libgomp1 >/dev/null')

    def skippable(self, step: Any) -> bool:
        from .recorder import Step

        if not isinstance(step, Step):
            return False
        # Harbor installs these two in every trial (always_install); the bake ran the same command.
        if step.command.endswith(self._ALWAYS_APT):
            return True
        # Only for a pinned release, which the image name holds: the download would fetch the baked binary again.
        return "GOOSE_VERSION" in step.env and "/download_cli.sh | bash" in step.command


class StrandsAdapter(Adapter):
    family = "strands"
    variant = "strands-skip"
    setup_budget_sec = 15.0
    expected_gb = 1.25
    expected_build_min = 2.5
    stock_setup_sec = 35.0

    @classmethod
    def matches(cls, agent: str) -> bool:
        return _subclass(agent, "harbor.agents.installed.strands:Strands")

    def agent_kwargs(self) -> dict[str, Any]:
        # A trial resolves a relative factory_path against harbor's cwd, the
        # project root; the bench process may run elsewhere.
        kwargs = dict(self.kwargs)
        path = kwargs.get("factory_path")
        if isinstance(path, str) and path and not Path(path).is_absolute():
            kwargs["factory_path"] = str(ROOT / path)
        return kwargs

    def input_files_digest(self) -> str:
        path = self.agent_kwargs().get("factory_path")
        if not isinstance(path, str) or not Path(path).is_dir():
            return ""
        h = hashlib.sha256()
        for f in sorted(p for p in Path(path).rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            h.update(f.relative_to(path).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
        return h.hexdigest()

    @property
    def smoke(self) -> str:
        factory = str(self.kwargs.get("factory") or "")
        module = factory.split(":", 1)[0] if ":" in factory else ""
        imports = ", ".join(m for m in ("strands", "strands_harness", module) if m)
        return f'/installed-agent/strands/venv/bin/python -c "import {imports}"'

    def skippable(self, step: Any) -> bool:
        from .recorder import Step

        return isinstance(step, Step) and "uv venv " in step.command and "uv pip install " in step.command


class Terminus2Adapter(Adapter):
    """Terminus-2 ships inside Harbor; its only install is the tmux and
    asciinema check of TmuxSession._install_recording_tools. On an image that
    has both, it installs nothing and the image is a label-only no-op."""
    family = "terminus-2"
    PROBES = (
        ("tmux", re.compile(r"tmux -V")),
        ("asciinema", re.compile(r"asciinema --version")),
        ("os-release", re.compile(r"cat /etc/os-release 2>/dev/null \|\| echo 'not found'")),
        ("uname", re.compile(r"uname -s")),
        ("which", re.compile(r"which \S+ >/dev/null 2>&1")),
    )
    setup_budget_sec = 15.0
    expected_gb = 0.2
    expected_build_min = 0.6
    stock_setup_sec = 20.0

    @classmethod
    def matches(cls, agent: str) -> bool:
        return agent == "terminus-2"

    @property
    def version(self) -> str | None:
        return harbor_version() or None

    @property
    def record_terminal_session(self) -> bool:
        harbor_agents()
        from harbor.agents.terminus_2.terminus_2 import Terminus2Options

        value = self.kwargs.get("record_terminal_session")
        if value is None:
            return bool(Terminus2Options.model_fields["record_terminal_session"].default)
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)

    def cache_key(self) -> str:
        # Only record_terminal_session reaches the install.
        return json.dumps({"family": self.family, "agent": self.agent, "record": self.record_terminal_session})

    def bare_answer(self, name: str, command: str) -> tuple[int, str] | None:
        if name == "which":
            return (0, "") if command.startswith("which apt-get ") else (1, "")
        return {"tmux": (127, ""), "asciinema": (127, ""), "os-release": (0, _OS_RELEASE),
                "uname": (0, "Linux\n")}.get(name)

    def baked_answer(self, name: str, command: str) -> tuple[int, str] | None:
        if name in ("tmux", "asciinema"):
            return (0, "tmux 3.3a\n") if name == "tmux" else (0, "asciinema 2.2.0\n")
        return self.bare_answer(name, command)

    @property
    def smoke(self) -> str:
        return "tmux -V && asciinema --version" if self.record_terminal_session else "tmux -V"

    async def record(self, env: Any, logs_dir: Path) -> None:
        harbor_agents()
        from harbor.agents.terminus_2.tmux_session import TmuxSession
        from harbor.models.trial.paths import EnvironmentPaths

        remote = EnvironmentPaths.agent_dir / "recording.cast" if self.record_terminal_session else None
        session = TmuxSession(session_name="terminus-2", environment=env,
                              logging_path=EnvironmentPaths.agent_dir / "terminus_2.pane",
                              local_asciinema_recording_path=None, remote_asciinema_recording_path=remote,
                              user=env.default_user)
        # Only the install: the session, history and recording are per trial.
        await session._install_recording_tools()


ADAPTERS: dict[str, type[Adapter]] = {a.family: a for a in (ClaudeCodeAdapter, Terminus2Adapter,
                                                           InstalledAgentAdapter, PiAdapter, StrandsAdapter,
                                                           ClineAdapter, OpenCodeAdapter, KiloAdapter,
                                                           GooseAdapter)}


def adapter_for(agent: str, family: str, kwargs: Mapping[str, Any], model_name: str) -> Adapter | None:
    """The adapter for an arm, None when its harness cannot be pre-baked."""
    cls = ADAPTERS.get(family)
    if cls is None or not cls.matches(agent):
        return None
    return cls(agent, kwargs, model_name)

