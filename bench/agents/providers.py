"""What the harness subclasses need to know about the bench's non-Anthropic providers.

The bench names a model as Harbor does, `<LiteLLM provider>/<model id>`:
fireworks_ai/accounts/fireworks/models/glm-5p3, openai/gpt-6.1-terra,
xai/grok-4.7. Several harnesses spell those providers differently, or have no
built-in route to them, so their subclasses translate. The routes live in
code rather than in arm options because a design may not set an endpoint or a
credential option (designs._DENIED_OPTION): every arm on a provider gets the
same endpoint, and a design can only pick the model.

Endpoints are the providers' public ones. Model limits are the catalog
figures the bench's harnesses already ship (pi-ai 1.4.0's Fireworks catalog)
or, for GPT-6.1 Terra, which no bundled catalog has yet, OpenRouter's model
list on 2026-10-29. Pure data and string functions: no I/O.
"""

from __future__ import annotations

from typing import NamedTuple

FIREWORKS = "fireworks_ai"   # LiteLLM's and Harbor's name; Pi and Cline call it fireworks, OpenCode fireworks-ai
OPENAI = "openai"
XAI = "xai"

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
XAI_BASE_URL = "https://api.x.ai/v1"

# The provider's own key variable, as Harbor and LiteLLM read it.
KEY_ENVS = {FIREWORKS: "FIREWORKS_AI_API_KEY", OPENAI: "OPENAI_API_KEY", XAI: "XAI_API_KEY"}


class Limits(NamedTuple):
    context: int   # tokens of prompt the model accepts
    output: int    # tokens it may generate in one call


# Fireworks models the bench runs, by model id (pi-ai 1.4.0 providers/data/fireworks.json).
FIREWORKS_MODELS: dict[str, Limits] = {
    "accounts/fireworks/models/deepseek-v4p1-flash-1017": Limits(1_000_000, 384_000),
    "accounts/fireworks/models/glm-5p3": Limits(1_048_573, 262_144),
    "accounts/fireworks/models/kimi-k3": Limits(1_048_576, 131_072),
}

# The reasoning_effort levels those three models take on Fireworks (the
# models' pages and pi-ai's thinkingLevelMap): no medium, which Fireworks
# turns into high. With no effort sent, GLM 5.3 and Kimi K3 run at max and
# DeepSeek V4.1 Flash at high.
FIREWORKS_EFFORTS = ("low", "high", "max")

# OpenAI models whose tool calls need the Responses API, which no bundled
# harness catalog lists yet (OpenRouter's model list, 2026-10-29).
RESPONSES_MODELS: dict[str, Limits] = {
    "gpt-6.1-terra": Limits(1_050_000, 128_000),
}


def split(model_name: str | None) -> tuple[str, str]:
    """(provider, model id) of a Harbor model name; ('', name) without a provider."""
    provider, sep, model_id = (model_name or "").partition("/")
    return (provider, model_id) if sep else ("", provider)


def needs_responses_api(model_name: str | None) -> bool:
    """An OpenAI model whose tools work only through the Responses API."""
    provider, model_id = split(model_name)
    return provider == OPENAI and model_id in RESPONSES_MODELS
