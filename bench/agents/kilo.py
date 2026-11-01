"""Kilo Code CLI, an OpenCode fork, run the way Harbor runs OpenCode.

Harbor 0.25.0 has no Kilo adapter. Kilo 7.8.18 keeps OpenCode's `run
--format=json` stream (step_start, text, reasoning, tool_use, step_finish,
error), its config schema and its exit codes, so this subclass is Harbor's
OpenCode with Kilo's names:

- npm package @kilocode/cli, binary `kilo` (the package also links
  `kilocode` to the same file);
- config in ~/.config/kilo/kilo.json and skills in ~/.config/kilo/skills;
- `--auto` approves every permission not explicitly denied, as
  `--dangerously-skip-permissions` does for OpenCode (Kilo keeps that flag as
  a hidden alias of the same replies), except skill-shell and
  sandbox-escalation requests, which Kilo always rejects: a tool call refused
  that way in a trial is Kilo's rule, not a bench bug;
- KILO_FAKE_VCS and the XDG data and state dirs under logs/kilo.

Three switches are off for the bench, and only these, so the arm stays
comparable with OpenCode's: telemetry (KILO_TELEMETRY_LEVEL turns it on only
at `all`), Claude Code's config, prompt and skills (KILO_DISABLE_CLAUDE_CODE)
and the auto-update check (KILO_DISABLE_AUTOUPDATE). Everything else,
the model catalog fetch included, is Kilo's default.

It inherits OpenCodeBunEpipe's acceptance of Bun's late EPIPE (Kilo is the
same kind of Bun build) and its Fireworks provider entry. ACP target mode is not supported.

The kilo arms use it as `agent: bench.agents.kilo:KiloCode` with
`family: kilo-code`.
"""

from __future__ import annotations

import shlex
from typing import Any

from harbor.agents.installed.base import NonZeroAgentExitCodeError, with_prompt_template
from harbor.agents.installed.node_install import nvm_node_install_snippet
from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .opencode import OpenCodeBunEpipe

# Kilo's own switches for every run (see the module docstring).
KILO_ENV = {"KILO_FAKE_VCS": "git", "KILO_TELEMETRY_LEVEL": "off", "KILO_DISABLE_CLAUDE_CODE": "1",
            "KILO_DISABLE_AUTOUPDATE": "1"}
# Where Harbor's OpenCode writes its config, and where Kilo reads the same file.
_OPENCODE_CONFIG = ("mkdir -p ~/.config/opencode && echo ", " > ~/.config/opencode/opencode.json")
_KILO_CONFIG = ("mkdir -p ~/.config/kilo && echo ", " > ~/.config/kilo/kilo.json")


class KiloCode(OpenCodeBunEpipe):
    _OUTPUT_FILENAME = "kilo.txt"
    _HARNESS = "Kilo"
    capabilities = OpenCode.capabilities.model_copy(update={"bridges": frozenset()})

    @staticmethod
    def name() -> str:
        return "kilo-code"

    def get_version_command(self) -> str | None:
        return "[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; kilo --version"

    async def install(self, environment: BaseEnvironment) -> None:
        """Harbor's OpenCode install, with Kilo's package and binary."""
        await self.ensure_system_dependencies(environment, ("curl", "bash", "coreutils", "nodejs", "npm"))
        version_spec = f"@{self._version}" if self._version else "@latest"
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "if ldd --version 2>&1 | grep -qi musl || "
                "[ -f /etc/alpine-release ]; then "
                "node --version && npm --version; "
                f"else {nvm_node_install_snippet()}; fi && "
                f"npm i -g @kilocode/cli{version_spec} && "
                "kilo --version"
            ),
        )

    def acp_command(self) -> list[str]:
        raise NotImplementedError("kilo-code does not support ACP target mode")

    async def acp_install(self, environment: BaseEnvironment) -> None:
        raise NotImplementedError("kilo-code does not support ACP target mode")

    def _build_register_skills_command(self) -> str | None:
        if not self.skills_dir:
            return None
        return (f"mkdir -p ~/.config/kilo/skills && cp -r {shlex.quote(self.skills_dir)}/* "
                f"~/.config/kilo/skills/ 2>/dev/null || true")

    def _build_register_config_command(self) -> str | None:
        """Harbor's OpenCode config, written where Kilo reads it."""
        command = super()._build_register_config_command()
        if command is None:
            return None
        head, tail = _OPENCODE_CONFIG
        if not (command.startswith(head) and command.endswith(tail)):
            raise RuntimeError("kilo-code: Harbor's OpenCode config command changed; update bench/agents/kilo.py")
        return f"{_KILO_CONFIG[0]}{command[len(head):-len(tail)]}{_KILO_CONFIG[1]}"

    def _convert_events_to_trajectory(self, events: list[dict[str, Any]]) -> Any:
        trajectory = super()._convert_events_to_trajectory(events)
        if trajectory is not None:
            trajectory.agent.name = self.name()
        return trajectory

    def run_env(self) -> dict[str, str]:
        """The environment of Kilo's own execs: the model connection, then Kilo's switches."""
        logs = self.environment_logs_dir / "kilo"
        return {**self.model_connection.env, **KILO_ENV, "XDG_DATA_HOME": (logs / "xdg-data").as_posix(),
                "XDG_STATE_HOME": (logs / "xdg-state").as_posix()}

    def run_command(self, instruction: str) -> str:
        """Harbor's OpenCode command line with Kilo's binary and `--auto`."""
        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""
        resume_flag = "--continue " if self._resume else ""
        output = shlex.quote((self.environment_logs_dir / self._OUTPUT_FILENAME).as_posix())
        return ("[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; "
                f"kilo run --model={shlex.quote(self._execution_model_name())} --format=json "
                f"{resume_flag}{cli_flags_arg}--thinking --auto -- {shlex.quote(instruction)} "
                f"2>&1 </dev/null | stdbuf -oL tee {output}")

    @with_prompt_template
    async def _run_harness(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        """Harbor's OpenCode run() with Kilo's command and environment."""
        self._instruction = instruction
        if not self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        env = self.run_env()
        if skills_command := self._build_register_skills_command():
            await self.exec_as_agent(environment, command=skills_command, env=env)
        if config_command := self._build_register_config_command():
            await self.exec_as_agent(environment, command=config_command, env=env)
        await self.exec_as_agent(environment, command=self.run_command(instruction), env=env)
        if messages := self._error_messages():
            raise NonZeroAgentExitCodeError("Kilo emitted error event(s): " + "; ".join(messages[:3]))
