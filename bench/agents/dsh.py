"""DeepSeek Harness (dsh), DeepSeek's own coding agent, run headless.

Harbor 0.25.0 has no dsh adapter, so this is a whole BaseInstalledAgent:

- npm package @deepseek-ai/dsh, binary `dsh`, pinned through `version` (all
  ~90 @deepseek-ai/dsh-* packages it pulls in are pinned to the same version
  by the package itself). Its provider library is not: dsh-llm-pi-ai asks for
  @earendil-works/pi-ai ^1.1.0, so the install puts dsh in its own npm prefix
  ($HOME/.dsh-bench/cli) whose package.json overrides pi-ai to PI_AI_VERSION,
  the release the dsh repo's lockfile pins. `npm i -g` would ignore the
  override. The pnpm patch dsh applies to pi-ai in its own repo (Gemini
  thought signatures, streaming JSON) does not reach an npm install either.
- Node >= 24.9 (the repo's engines field; the npm manifest declares none, so
  npm would not stop an older Node), installed through Harbor's nvm snippet at
  NODE_VERSION. glibc only: the install tree has native addons.
- One task per run: `dsh --profile headless --json -` reads the task from
  stdin. The task and the provider overlay are uploaded as files under
  /installed-agent, so neither the task nor the trial's name is ever on the
  long-running process's command line (an agent that ran `pkill -f` on a task
  name killed Goose that way).
- Unattended: DSH_PERMISSION_MODE=danger-full-access, from which dsh's base
  profile derives the sandbox mode and approval policy never (the headless
  command has no flags for either and rejects unknown options, exit 1, before
  the runner mounts). Headless has no approval channel and rejects anything
  that asks, and dsh's sandbox (bwrap, then Landlock) fails closed in an
  unprivileged container, so danger-full-access is the only mode that runs
  there. There is no turn or step budget; Harbor's agent timeout is the bound.
- DSH_HOME ($HOME/.dsh) is outside /logs/agent, as every data and cache dir
  must be (links there break the log download on the Windows host). After the
  run its sessions dir is copied, links resolved, to logs/dsh-sessions.

The model goes through dsh's pi-ai adapter, not its native DeepSeek one: the
native adapter speaks only Anthropic Messages to DeepSeek's own API (and asks
api.deepseek.com for its model list when it holds a DEEPSEEK_API_KEY, which
the trial never passes). The overlay (--patch) declares a Fireworks route by
hand, openai-completions on Fireworks' endpoint with the key from the bench's
FIREWORKS_AI_API_KEY, and makes it the default model. Hand-declared, it does
not depend on which models pi-ai's own Fireworks catalog lists (1.3.0 lacks
deepseek-v4p1-flash-1017). maxTokens is the model's output limit from
providers.FIREWORKS_MODELS: without it a hand-declared model gets the route's
defaultMaxTokens (32768), which reasoning at high or max can use up before the
answer or a tool call ends. Fireworks is the only provider routed so far; any other
fails the trial before it starts.

Effort: `reasoning_effort` (off | low | high | max) becomes the route's
`reasoning` level and agent-default-model's reasoningEffort. With
compat.thinkingFormat deepseek, `off` sends thinking {type: disabled}; every
other level sends thinking {type: enabled} and that reasoning_effort. Unset,
neither is written, and dsh may then send thinking off (a review of rc.5 read
it that way; unconfirmed live), so the arms always set it.

Both uploads are written with \n line ends: Harbor's _upload_config_text
writes in text mode, which on the Windows host would hand dsh a task in CRLF.

Switched off: telemetry (DSH_TELEMETRY_DISABLED; its OTLP exporter only sends
after explicit user feedback anyway) and session-title-llm, the first-prompt
title generator, which would add a model call the --json stream never counts.
dsh has no update check on the headless path.

Usage: dsh computes no cost. Every step_end status event in dsh.jsonl carries
the step's tokens, disjoint (inputTokens is uncached input only), summed over
retried attempts, and left out when an attempt reported none. The stream
covers the top-level agent only: a subagent's calls are not in it.
bench/extract.py reads it (_dsh_calls).

The dsh arms use it as `agent: bench.agents.dsh:DeepSeekHarness` with
`family: deepseek-harness`.
"""

from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from harbor.agents.agent_auth_resolution import ModelConnectionSpec, require_model_api_key
from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.agents.installed.node_install import nvm_node_install_snippet
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from . import providers

PACKAGE = "@deepseek-ai/dsh"
PI_AI_PACKAGE, PI_AI_VERSION = "@earendil-works/pi-ai", "1.3.0"   # dsh's lockfile pin, 2026-10-15
NODE_VERSION = "24.11.0"                                           # dsh needs >= 24.9.0

# The install prefix, outside /logs/agent; $HOME is the agent user's.
_PREFIX = '"$HOME/.dsh-bench/cli"'
_BIN = '"$HOME/.dsh-bench/cli/node_modules/.bin/dsh"'
_NVM = "[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; "
# Uploaded per trial: names that say nothing about the task.
TASK_FILE = "/installed-agent/dsh-task.txt"
PATCH_FILE = "/installed-agent/dsh-bench.patch.yml"

# dsh's own switches for every run (see the module docstring).
DSH_ENV = {"DSH_TELEMETRY_DISABLED": "1", "DSH_PERMISSION_MODE": "danger-full-access"}
EFFORTS = ("off", "low", "high", "max")

# dsh's names for the routes this adapter declares: LiteLLM provider -> (route id, display name).
ROUTES = {providers.FIREWORKS: ("bench-fireworks", "Fireworks AI (bench)")}


class DshOptions(InstalledAgentOptions):
    reasoning_effort: Literal["off", "low", "high", "max"] | None = Field(
        default=None, description="Model reasoning effort (the route's reasoning level). Omit for the model's default."
    )


def provider_patch(model_name: str | None, effort: str | None) -> list[dict[str, Any]]:
    """The --patch overlay: a hand-declared pi-ai route to the model, as the
    default model, and the title generator off. A Cordis config override
    replaces an entry's whole config, so agent-default-model restates both."""
    provider, model_id = providers.split(model_name)
    if provider not in ROUTES:
        raise ValueError(f"deepseek-harness: provider {provider or '(none)'!r} is not routed; "
                         f"only {', '.join(ROUTES)} is (bench/agents/dsh.py)")
    if effort is not None and effort not in EFFORTS:
        raise ValueError(f"deepseek-harness: effort {effort!r} is not one of {', '.join(EFFORTS)}")
    route_id, display = ROUTES[provider]
    model: dict[str, Any] = {"id": model_id,
                             "reasoningEfforts": {"off": None, "low": "low", "high": "high", "max": "max"}}
    limits = providers.FIREWORKS_MODELS.get(model_id)
    if limits is not None:
        model["contextWindow"] = limits.context
        model["maxTokens"] = limits.output
    route: dict[str, Any] = {"displayName": display, "apiKeyEnv": providers.KEY_ENVS[provider],
                             "api": "openai-completions", "baseURL": providers.FIREWORKS_BASE_URL,
                             "compat": {"supportsDeveloperRole": False, "maxTokensField": "max_tokens",
                                        "thinkingFormat": "deepseek"},
                             "models": [model]}
    default: dict[str, Any] = {"provider": route_id, "model": model_id}
    if effort is not None:
        route["reasoning"] = effort
        default["reasoningEffort"] = effort
    return [{"id": "llm-pi-ai", "config": {"providers": {route_id: route}}},
            {"id": "agent-default-model", "config": default},
            {"id": "session-title-llm", "disabled": True}]


def step_usage(text: str) -> list[dict[str, Any]]:
    """The usage of each step_end status event in a dsh --json stream that has one."""
    out = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if (isinstance(event, dict) and event.get("type") == "status" and event.get("phase") == "step_end"
                and isinstance(event.get("usage"), dict)):
            out.append(event["usage"])
    return out


class DeepSeekHarness(BaseInstalledAgent):
    _OUTPUT_FILENAME = "dsh.jsonl"
    _STDERR_FILENAME = "dsh.stderr.txt"
    _SESSIONS_DIRNAME = "dsh-sessions"

    capabilities = AgentCapabilities(reasoning_effort=True)
    MODEL_CONNECTION = ModelConnectionSpec(base_url_destination_names={"default": None})
    options_model = DshOptions

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Fail at construction, before any setup, on a provider or effort the overlay cannot express.
        provider_patch(self.model_name, self._effort())

    @classmethod
    def check_model_credentials(cls, connection, *, kwargs, env, explicit_env=None):
        require_model_api_key(connection, cls.name())

    @staticmethod
    def name() -> str:
        return "deepseek-harness"

    def get_version_command(self) -> str | None:
        return f"{_NVM}{_BIN} --version"

    def _effort(self) -> str | None:
        return getattr(self.options, "reasoning_effort", None)

    def install_command(self) -> str:
        version = self._version or "latest"
        package_json = {"private": True, "dependencies": {PACKAGE: version},
                        "overrides": {PI_AI_PACKAGE: PI_AI_VERSION}}
        return ("set -euo pipefail; "
                "if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then "
                "echo 'deepseek-harness needs glibc (Node 24 from nvm, native addons)' >&2; exit 1; fi; "
                f"{nvm_node_install_snippet(NODE_VERSION)} && "  # type: ignore[arg-type]
                f"mkdir -p {_PREFIX} && "
                f"printf '%s\\n' {shlex.quote(json.dumps(package_json, sort_keys=True))} "
                f"> {_PREFIX}/package.json && "
                f"npm install --prefix {_PREFIX} --no-audit --no-fund && "
                f"{_BIN} --version")

    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("curl", "bash", "coreutils"))
        await self.exec_as_agent(environment, command=self.install_command())

    async def _upload_config_text(self, environment: BaseEnvironment, *, content: str,
                                  remote_path: str, filename: str) -> None:
        """Harbor's upload, but byte-exact: \n stays \n on the Windows host."""
        with tempfile.TemporaryDirectory(prefix="harbor-agent-config-") as temp_dir:
            local_path = Path(temp_dir) / filename
            local_path.write_text(content, encoding="utf-8", newline="\n")
            await environment.upload_file(local_path, remote_path)
        if environment.default_user is not None:
            quoted = shlex.quote(remote_path)
            await self.exec_as_root(environment, command=f"chown {shlex.quote(str(environment.default_user))} "
                                                         f"{quoted} && chmod 600 {quoted}")

    def run_env(self) -> dict[str, str]:
        """The model connection (the provider's key under its own name), then dsh's switches."""
        return {**self.model_connection.env, **DSH_ENV}

    def run_command(self) -> str:
        """dsh headless on the uploaded task; the exit code is dsh's, and on a
        failure only dsh's own `dsh: <code>: <message>` lines reach Harbor's
        error patterns, never the agent's tool output."""
        logs = self.environment_logs_dir
        output = shlex.quote((logs / self._OUTPUT_FILENAME).as_posix())
        stderr = shlex.quote((logs / self._STDERR_FILENAME).as_posix())
        sessions = shlex.quote((logs / self._SESSIONS_DIRNAME).as_posix())
        return (f"{_NVM}export DSH_HOME=\"$HOME/.dsh\"; "
                f"{_BIN} --profile headless --patch {PATCH_FILE} "
                f"--json - < {TASK_FILE} > {output} 2> {stderr}; rc=$?; "
                f"if [ -d \"$DSH_HOME/sessions\" ]; then mkdir -p {sessions} && "
                f"cp -rL \"$DSH_HOME/sessions/.\" {sessions}/ 2>/dev/null; fi; "
                f"[ $rc -eq 0 ] || grep '^dsh: ' {stderr} | tail -n 3 >&2; "
                "exit $rc")

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if not self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        patch = json.dumps(provider_patch(self.model_name, self._effort()), indent=1)
        await self._upload_config_text(environment, content=patch, remote_path=PATCH_FILE,
                                       filename="dsh-bench.patch.yml")
        await self._upload_config_text(environment, content=instruction, remote_path=TASK_FILE,
                                       filename="dsh-task.txt")
        await self.exec_as_agent(environment, command=self.run_command(), env=self.run_env())

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Token totals over the step_end events; cost stays unset (dsh computes none)."""
        try:
            text = Path(self.logs_dir / self._OUTPUT_FILENAME).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        steps = step_usage(text)
        if not steps:
            return

        def total(key: str) -> int:
            return sum(int(u.get(key) or 0) for u in steps)

        read = total("cacheReadTokens")
        context.n_input_tokens = total("inputTokens") + read + total("cacheWriteTokens")
        context.n_cache_tokens = read
        context.n_output_tokens = total("outputTokens")
