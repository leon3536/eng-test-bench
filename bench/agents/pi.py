"""Pi with routes to Fireworks and to OpenAI's Responses-only models.

Harbor 0.25.0's pi adapter passes the LiteLLM provider name through as
`--provider`, and Pi (pi-ai 1.4.0) knows two of the bench's three other
providers by other names or not at all:

- Fireworks is `fireworks` to Pi, which reads its key from FIREWORKS_API_KEY;
  Harbor would send `--provider fireworks_ai` with FIREWORKS_AI_API_KEY, and
  Pi would stop on an unknown provider. This subclass sends `fireworks` and
  copies the key to FIREWORKS_API_KEY. Pi's own Fireworks catalog then
  supplies the endpoint, API style, limits, prices and session-affinity
  headers for every model the bench runs there.
- GPT-6.1 Terra is not in Pi's OpenAI catalog yet. Its tools work only
  through the Responses API, so the subclass gives Harbor OpenAI's endpoint
  and the openai_responses format: Harbor then writes a models.json for its
  harbor-endpoint provider (key $OPENAI_API_KEY, api openai-responses), and
  this subclass adds the model's context and output limits to that entry,
  which Pi would otherwise not know.
- xAI is `xai` to Pi as well, and Grok 4.7 is in its catalog: unchanged.

Anthropic arms are unchanged too. The pi arms on these providers use it as
`agent: bench.agents.pi:PiProviders` with `family: pi`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from harbor.agents.agent_auth_resolution import ResolvedModelConnection
from harbor.agents.installed.pi import Pi
from harbor.models.trial.result import AgentInfo

from . import providers

# Pi's names for the providers it spells differently, and the key variable it reads for each.
PI_PROVIDERS = {providers.FIREWORKS: ("fireworks", "FIREWORKS_API_KEY")}


class PiProviders(Pi):
    def __init__(self, *args: Any, base_url: str | None = None, model_api_format: Any = None,
                 **kwargs: Any) -> None:
        model_name = kwargs.get("model_name", args[1] if len(args) > 1 else None)
        if base_url is None and model_api_format is None and providers.needs_responses_api(model_name):
            base_url, model_api_format = providers.OPENAI_BASE_URL, "openai_responses"
        super().__init__(*args, base_url=base_url, model_api_format=model_api_format, **kwargs)

    @property
    def model_connection(self) -> ResolvedModelConnection:
        """Harbor's connection under Pi's provider name, with the key where Pi reads it."""
        connection = super().model_connection
        if self.skip_model_parsing or connection.provider not in PI_PROVIDERS:
            return connection
        name, key_env = PI_PROVIDERS[connection.provider]
        env = dict(connection.env)
        if connection.api_key is not None:
            env[key_env] = connection.api_key
        return replace(connection, provider=name, env=env)

    def to_agent_info(self) -> AgentInfo:
        """The trial's model info keeps the provider as the bench and LiteLLM name it."""
        info = super().to_agent_info()
        if info.model_info is not None and not self.skip_model_parsing:
            info.model_info.provider = super().model_connection.provider
        return info

    def _build_custom_models_json(self, access: ResolvedModelConnection, model_id: str) -> dict[str, Any] | None:
        """Harbor's custom-endpoint models.json, with the limits of a model the bench knows."""
        models_json = super()._build_custom_models_json(access, model_id)
        limits = providers.RESPONSES_MODELS.get(model_id) if providers.needs_responses_api(self.model_name) else None
        if models_json is not None and limits is not None:
            for provider in models_json.get("providers", {}).values():
                for model in provider.get("models", []):
                    if model.get("id") == model_id:
                        model.setdefault("contextWindow", limits.context)
                        model.setdefault("maxTokens", limits.output)
                        model.setdefault("input", ["text", "image"])
        return models_json
