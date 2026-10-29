"""mini-swe-agent with the `mcp` package added to its tool environment.

Harbor 0.25.0 installs mini-swe-agent with `--with litellm` but leaves out
LiteLLM's proxy extra, which drags in a dependency that can't be built on
Alpine images. LiteLLM 1.106 imports `mcp` from proxy code that mini-swe-agent
reaches, so on 2026-10-28 every trial failed with
`ModuleNotFoundError: No module named 'mcp'` before its first model call.
Adding just `mcp` after Harbor's own install fixes that without the rest of
the extra. Everything else is Harbor's MiniSweAgent unchanged.

The mini-swe-agent arms in config/bench.yaml and config/dev.yaml use it as
`agent: bench.agents.mini_swe_agent:MiniSweAgentWithMcp` with
`family: mini-swe-agent`, so the arm keeps mini-swe-agent's reference figures,
cost-cap rule and report group.
"""

from __future__ import annotations

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import BaseEnvironment


class MiniSweAgentWithMcp(MiniSweAgent):
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
