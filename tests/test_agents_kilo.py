"""Kilo Code as Harbor's OpenCode with Kilo's package, binary, config and switches (bench/agents/kilo.py)."""

from __future__ import annotations

import asyncio
import json
import shlex
from typing import Any
from unittest import mock

from tests.support import BenchTestCase

from bench.prebake.adapters import harbor_agents

harbor_agents()

from harbor.agents.installed.base import NonZeroAgentExitCodeError  # noqa: E402

from bench.agents.kilo import KILO_ENV, KiloCode  # noqa: E402

MODEL = "openrouter/deepseek/deepseek-v4.1-flash"
STOP = json.dumps({"type": "step_finish", "part": {"reason": "stop", "tokens": {"input": 1, "output": 2}}})
EPIPE = "error: EPIPE: broken pipe, write\n      at emitError (node:events:50:13)\n"


class Execs:
    """Records every exec; `kilo run` exits with run_rc, `cat` of kilo.txt prints container_text."""

    def __init__(self, run_rc: int = 0, container_text: str = "") -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.run_rc, self.container_text = run_rc, container_text

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: Any = None) -> Any:
        from harbor.environments.base import ExecResult

        self.calls.append((command, dict(env or {})))
        if command.startswith("for manager"):
            return ExecResult(stdout="apt-get", stderr="", return_code=0)
        if command.startswith("cat "):
            return ExecResult(stdout=self.container_text, stderr="", return_code=0)
        if " kilo run " in command:
            return ExecResult(stdout="", stderr="", return_code=self.run_rc)
        return ExecResult(stdout="", stderr="", return_code=1 if command.startswith("command -v") else 0)

    def commands(self, needle: str) -> list[tuple[str, dict[str, str]]]:
        return [c for c in self.calls if needle in c[0]]


class KiloAgent(BenchTestCase):
    def agent(self, **kwargs: Any) -> KiloCode:
        return KiloCode(logs_dir=self.tmpdir(), model_name=MODEL, **{"version": "7.8.18", **kwargs})

    def run_agent(self, agent: KiloCode, env: Execs, instruction: str = "fix it") -> None:
        asyncio.run(agent.run(instruction, env, mock.Mock()))  # type: ignore[arg-type]

    def test_install_pins_kilos_npm_package(self) -> None:
        env = Execs()
        agent = self.agent()
        asyncio.run(agent.install(env))  # type: ignore[arg-type]
        ((command, _),) = env.commands("npm i -g")
        self.assertIn(" && npm i -g @kilocode/cli@7.8.18 && kilo --version", command)
        self.assertNotIn("opencode", command)
        self.assertEqual(agent.get_version_command(), "[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; kilo --version")
        self.assertEqual((agent.name(), agent.version()), ("kilo-code", "7.8.18"))

    def test_run_is_harbors_opencode_command_with_kilos_binary_and_auto(self) -> None:
        env = Execs()
        instruction = "Fix the bug; don't touch 'tests' $HOME"
        self.run_agent(self.agent(variant="medium"), env, instruction)
        ((command, run_env),) = env.commands(" kilo run ")
        self.assertEqual(command, "set -o pipefail; [ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; "
                                  f"kilo run --model={MODEL} --format=json --variant medium --thinking --auto "
                                  f"-- {shlex.quote(instruction)} 2>&1 </dev/null "
                                  "| stdbuf -oL tee /logs/agent/kilo.txt")
        self.assertNotIn("dangerously-skip-permissions", command)
        self.assertEqual({k: run_env[k] for k in KILO_ENV}, {"KILO_FAKE_VCS": "git", "KILO_TELEMETRY_LEVEL": "off",
                                                             "KILO_DISABLE_CLAUDE_CODE": "1",
                                                             "KILO_DISABLE_AUTOUPDATE": "1"})
        self.assertEqual((run_env["XDG_DATA_HOME"], run_env["XDG_STATE_HOME"]),
                         ("/logs/agent/kilo/xdg-data", "/logs/agent/kilo/xdg-state"))
        self.assertIn("OPENROUTER_API_KEY", run_env)
        self.assertFalse([k for k in run_env if k.startswith("OPENCODE_")])
        # Only the three switches the bench chose: the rest stays at Kilo's defaults, as for OpenCode.
        self.assertFalse({k for k in run_env if k.startswith("KILO_DISABLE_")} - {"KILO_DISABLE_CLAUDE_CODE",
                                                                                   "KILO_DISABLE_AUTOUPDATE"})

    def test_config_and_skills_go_where_kilo_reads_them(self) -> None:
        env = Execs()
        skills = self.tmpdir()
        self.run_agent(self.agent(skills_dir=skills.as_posix()), env)
        ((config, _),) = env.commands("kilo.json")
        self.assertTrue(config.startswith("set -o pipefail; mkdir -p ~/.config/kilo && echo "))
        self.assertTrue(config.endswith(" > ~/.config/kilo/kilo.json"))
        written = json.loads(shlex.split(config.split(" && ", 1)[1].rsplit(" > ", 1)[0])[1])
        self.assertEqual(written, {"provider": {"openrouter": {"models": {"deepseek/deepseek-v4.1-flash": {}}}}})
        ((copy, _),) = env.commands("skills")
        self.assertTrue(copy.startswith("set -o pipefail; mkdir -p ~/.config/kilo/skills && cp -r "))
        self.assertFalse(env.commands("~/.config/opencode"))

    def test_a_late_epipe_after_a_finished_run_is_kept_and_read_from_kilo_txt(self) -> None:
        env = Execs(run_rc=1, container_text=f"{EPIPE}{STOP}\n")
        self.run_agent(self.agent(), env)
        self.assertEqual([c for c, _ in env.commands("cat ")], ["cat /logs/agent/kilo.txt"])
        with self.assertRaises(NonZeroAgentExitCodeError):
            self.run_agent(self.agent(), Execs(run_rc=1, container_text=EPIPE))  # never finished

    def test_a_github_prompt_in_the_agents_own_tool_output_is_not_an_auth_failure(self) -> None:
        # Harbor's GitHub auth pattern also matches a `git clone` the agent ran in a tool call (2026-10-31).
        from harbor.agents.installed.base import AgentAuthenticationError

        clone = json.dumps({"type": "tool_use", "part": {"state": {
            "output": "fatal: could not read Username for 'https://github.com': No such device or address"}}})
        failed = AgentAuthenticationError("Command failed (exit 1): kilo run ...")
        for text, kept in ((f"{clone}\n{EPIPE}{STOP}\n", True), (f"{clone}\n{EPIPE}", False)):
            with self.subTest(finished=kept), mock.patch.object(KiloCode, "_run_harness", side_effect=failed):
                if kept:
                    self.run_agent(self.agent(), Execs(container_text=text))
                else:
                    with self.assertRaises(AgentAuthenticationError):
                        self.run_agent(self.agent(), Execs(container_text=text))

    def test_an_error_event_fails_the_run(self) -> None:
        agent = self.agent()
        (agent.logs_dir / "kilo.txt").write_text(json.dumps({"type": "error", "error": {"name": "APIError"}}) + "\n",
                                                 encoding="utf-8")
        with self.assertRaisesRegex(NonZeroAgentExitCodeError, "Kilo emitted error event"):
            self.run_agent(agent, Execs())

    def test_the_trajectory_names_kilo(self) -> None:
        agent = self.agent()
        trajectory = agent._convert_events_to_trajectory([{"type": "step_start", "sessionID": "s"},
                                                         json.loads(STOP)])
        assert trajectory is not None
        self.assertEqual(trajectory.agent.name, "kilo-code")

    def test_acp_target_mode_is_refused(self) -> None:
        agent = self.agent()
        self.assertFalse(agent.capabilities.bridges)
        with self.assertRaises(NotImplementedError):
            agent.acp_command()
        with self.assertRaises(NotImplementedError):
            asyncio.run(agent.acp_install(Execs()))  # type: ignore[arg-type]

    def test_a_changed_harbor_config_command_fails_loudly(self) -> None:
        from harbor.agents.installed.opencode import OpenCode

        with mock.patch.object(OpenCode, "_build_register_config_command", return_value="echo {} > elsewhere"):
            with self.assertRaises(RuntimeError):
                self.agent()._build_register_config_command()
        with mock.patch.object(OpenCode, "_build_register_config_command", return_value=None):
            self.assertIsNone(self.agent()._build_register_config_command())
