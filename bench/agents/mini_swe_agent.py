"""mini-swe-agent with the `mcp` package added to its tool environment.

Harbor 0.25.0 installs mini-swe-agent with `--with litellm` but leaves out
LiteLLM's proxy extra, which drags in a dependency that can't be built on
Alpine images. LiteLLM 1.106 imports `mcp` from proxy code that mini-swe-agent
reaches, so on 2026-10-28 every trial failed with
`ModuleNotFoundError: No module named 'mcp'` before its first model call.
Adding just `mcp` after Harbor's own install fixes that without the rest of
the extra.

It also runs GPT-6.1 Terra through LiteLLM's Responses client: its tools work
only through the Responses API, and mini-swe-agent's default LiteLLM client
calls chat completions. On such a model, with no format and no model config
of the arm's own, the subclass sets the openai_responses format, which
Harbor turns into `model.model_class=litellm_response` and sends the effort
as `reasoning.effort`. Everything else is Harbor's MiniSweAgent unchanged.

The mini-swe-agent arms in config/bench.yaml and config/dev.yaml use it as
`agent: bench.agents.mini_swe_agent:MiniSweAgentWithMcp` with
`family: mini-swe-agent`, so the arm keeps mini-swe-agent's reference figures,
cost-cap rule and report group.
"""

from __future__ import annotations

from typing import Any

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import BaseEnvironment

from . import providers


class MiniSweAgentWithMcp(MiniSweAgent):
    def __init__(self, *args: Any, model_api_format: Any = None, **kwargs: Any) -> None:
        model_name = kwargs.get("model_name", args[1] if len(args) > 1 else None)
        own_model = kwargs.get("config") is not None or kwargs.get("config_file") is not None
        if model_api_format is None and not own_model and providers.needs_responses_api(model_name):
            model_api_format = "openai_responses"
        super().__init__(*args, model_api_format=model_api_format, **kwargs)

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                'export PATH="$HOME/.local/bin:$PATH" && '
                'uv pip install --python "$(uv tool dir)/mini-swe-agent/bin/python" mcp'
            ),
        )
