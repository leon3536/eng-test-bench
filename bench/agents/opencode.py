"""OpenCode without the false failure Bun's late EPIPE causes.

OpenCode 1.19.4 moved to Bun 1.4.4, which can throw an unhandled
`EPIPE: broken pipe, write` when a child process OpenCode feeds exits before
reading its stdin. Bun prints the trace and exits 1, even after the run has
finished. The 2026-10-30 dev pilot on 1.19.5 solved its task (reward 1.0),
ended on a `step_finish` with reason `stop`, and still failed as an agent
crash. The same fix is in a Harbor fork (zendo-ai/harbor PR #33).

This subclass accepts exit 1 only when all of these hold, else the failure
stands: the exception is Harbor's plain NonZeroAgentExitCodeError or one it
named from a phrase in the output (below), opencode.txt holds Bun's EPIPE trace and
nothing else outside OpenCode's JSON events, no event is an `error`, and the
last event is a `step_finish` with reason `stop`. A timeout or kill never
reaches here as exit 1. Everything else is Harbor's OpenCode unchanged.

Every error Harbor names from its ERROR_PATTERNS is accepted on the same terms.
Harbor scans the whole output for those phrases, and they also match what the
agent's own tool calls printed: its GitHub auth pattern ("could not read
Username for 'https://github.com") matched a `git clone` (2026-10-31,
build-cython-ext on Sonnet 5.5, twice), and a network and a rate-limit pattern
matched two verified passes on hf-model-inference (2026-11-01, Kilo on DeepSeek,
OpenCode on GPT-6.1 Terra). A run that ended on `stop` with no error event
never failed at its provider.

It also registers Fireworks for an arm on a Fireworks model. OpenCode knows
Fireworks as `fireworks-ai` (models.dev), but Harbor selects the model as
`fireworks_ai/<id>` and writes an empty `fireworks_ai` provider, so OpenCode
would find neither an endpoint nor a key. The subclass adds that provider to
opencode_config: the openai-compatible SDK, Fireworks' endpoint, the key from
FIREWORKS_AI_API_KEY, and for each model the bench runs there `reasoning`,
`tool_call`, its limits and its effort variants. The variants are declared,
not left to OpenCode: its ProviderTransform.variants() (1.19.5, and Kilo
7.8.18) gives no variant at all to a model id containing `glm` (but GLM 5.2)
or `kimi`, so `--variant high` would set nothing on GLM 5.3 and Kimi K3 and
they would run at their default, max. Declared variants are merged
over the computed ones, so each of low/high/max sends that reasoningEffort
(reasoning_effort on the wire) and the arm's `variant` sets its effort. An
arm whose own opencode_config has a `fireworks_ai` provider keeps it. Kilo
Code inherits all of this.

The opencode arms use it as `agent: bench.agents.opencode:OpenCodeBunEpipe`
with `family: opencode`.
"""

from __future__ import annotations

import copy
import json
import re
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from . import providers

_EPIPE = "error: EPIPE: broken pipe, write"
# The lines of Bun's trace: the source excerpt ("45 |   ..."), its caret, the error and its frames.
_TRACE_LINE = re.compile(r"\s*(\d+ \|.*|\^|error: EPIPE: broken pipe, write|at \S.*)")


def finished_despite_epipe(text: str) -> bool:
    """Whether opencode.txt shows a finished run whose only fault is Bun's late EPIPE."""
    events = []
    saw_epipe = False
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            event = None
        if isinstance(event, dict):
            events.append(event)
            continue
        if not _TRACE_LINE.fullmatch(line):
            return False
        saw_epipe = saw_epipe or line.strip() == _EPIPE
    if not saw_epipe or not events or any(e.get("type") == "error" for e in events):
        return False
    last = events[-1]
    part = last.get("part")
    return last.get("type") == "step_finish" and isinstance(part, dict) and part.get("reason") == "stop"


def fireworks_provider(model_id: str) -> dict[str, Any]:
    """OpenCode's provider entry for Fireworks, under Harbor's provider name."""
    model: dict[str, Any] = {}
    limits = providers.FIREWORKS_MODELS.get(model_id)
    if limits is not None:
        model = {"name": model_id, "reasoning": True, "tool_call": True,
                 "limit": {"context": limits.context, "output": limits.output},
                 "variants": {effort: {"reasoningEffort": effort} for effort in providers.FIREWORKS_EFFORTS}}
    return {"npm": "@ai-sdk/openai-compatible", "name": "Fireworks AI",
            "options": {"baseURL": providers.FIREWORKS_BASE_URL,
                        "apiKey": "{env:" + providers.KEY_ENVS[providers.FIREWORKS] + "}"},
            "models": {model_id: model}}


def with_provider_routes(model_name: str | None, opencode_config: dict[str, Any] | None) -> dict[str, Any] | None:
    """opencode_config with the route to the model's provider added where OpenCode has none."""
    provider, model_id = providers.split(model_name)
    if provider != providers.FIREWORKS or (opencode_config or {}).get("provider", {}).get(provider) is not None:
        return opencode_config
    config = copy.deepcopy(opencode_config or {})
    config.setdefault("provider", {})[provider] = fireworks_provider(model_id)
    return config


class OpenCodeBunEpipe(OpenCode):
    _HARNESS = "OpenCode"  # as the EPIPE warning names it

    def __init__(self, *args: Any, opencode_config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        model_name = kwargs.get("model_name", args[1] if len(args) > 1 else None)
        if not kwargs.get("skip_model_parsing"):
            opencode_config = with_provider_routes(model_name, opencode_config)
        super().__init__(*args, opencode_config=opencode_config, **kwargs)

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        try:
            await self._run_harness(instruction, environment, context)
        except NonZeroAgentExitCodeError as exc:
            accepted = {NonZeroAgentExitCodeError, *(p.exception for p in self.ERROR_PATTERNS)}
            if type(exc) not in accepted or not str(exc).startswith("Command failed (exit 1):"):
                raise
            if not finished_despite_epipe(await self._output_text(environment)):
                raise
            self.logger.warning("%s exited 1 on Bun's late EPIPE after finishing its run; kept as finished",
                                self._HARNESS)

    async def _run_harness(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        """The harness's own run: Harbor's OpenCode here, Kilo's in bench/agents/kilo.py."""
        await OpenCode.run(self, instruction, environment, context)

    async def _output_text(self, environment: BaseEnvironment) -> str:
        """opencode.txt as the container sees it. The host's copy, through a
        Docker Desktop bind mount on Windows, can lag its last lines: on
        2026-10-30 it did, and a finished run still failed."""
        path = (self.environment_logs_dir / self._OUTPUT_FILENAME).as_posix()
        try:
            result = await environment.exec(command=f"cat {shlex.quote(path)}")
            if result.return_code == 0 and result.stdout:
                return result.stdout
        except Exception:  # the host's copy is the fallback
            pass
        try:
            return Path(self.logs_dir / self._OUTPUT_FILENAME).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
