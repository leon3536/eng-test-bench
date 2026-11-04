"""The Anthropic client and the one request shape the assistant sends.

This is the only module that reads ANTHROPIC_API_KEY. A client is built per
turn, so a key changed on the Setup page applies to the next message without
a restart, and the key lives only in that client.

The request (anthropic 2.3.0, client.beta.messages.stream; fallbacks exists
only on the beta namespace):
- model claude-opus-5-5, streamed, so a long reply never hits an HTTP timeout;
- adaptive thinking with summarized display, so the panel can show what the
  model is weighing instead of a long silence;
- effort fixed per conversation, sent explicitly (Opus 5.5 defaults to medium);
- top-level automatic prompt caching over a frozen system prompt and tool list;
- server-side fallbacks ("default", beta server-side-fallback-2026-07-01), so
  a classifier decline is answered by Anthropic's recommended fallback model
  instead of ending the turn. assistant.fallbacks: off turns this off.

Never sent: temperature, top_p, top_k, budget_tokens or disabled thinking
(all 400 on Opus 5.5), tool_choice (forced tool use 400s, and auto is the
default), assistant prefills, or role "system" messages. Bench notices are
dashboard events only and never enter the conversation.
"""

from __future__ import annotations

import copy
from typing import Any

import anthropic

from .. import keys
from ..settings import AssistantSettings
from . import prompt

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Where fallbacks="default" can route a decline today (claude-api skill notes);
# used only to size the daily-cap reserve, never sent.
FALLBACK_MODELS = ("claude-opus-5", "claude-opus-4-8")
KEY_NAME = keys.ASSISTANT_KEY
# Pinned rather than read from ANTHROPIC_BASE_URL: an ambient gateway would
# receive the key, and might reject beta fields like fallbacks or
# eager_input_streaming. keys.child_env strips the same variable from arms.
API_BASE_URL = "https://api.anthropic.com"
MAX_RETRIES = 2          # the SDK retries 408, 409, 429, 5xx and connection errors itself
TIMEOUT_SECONDS = 300.0


class AssistantUnavailable(RuntimeError):
    pass


def key_available() -> bool:
    return bool(keys.get_secret(KEY_NAME))


def make_client() -> anthropic.Anthropic:
    api_key = keys.get_secret(KEY_NAME)
    if not api_key:
        raise AssistantUnavailable(f"Set {KEY_NAME} on Setup to use the assistant.")
    return anthropic.Anthropic(api_key=api_key, base_url=API_BASE_URL,
                               max_retries=MAX_RETRIES, timeout=TIMEOUT_SECONDS)


def request_kwargs(settings: AssistantSettings, conv_meta: dict[str, Any], tools: list[dict[str, Any]],
                   messages: list[dict[str, Any]], system: str = prompt.SYSTEM_PROMPT) -> dict[str, Any]:
    """Keyword arguments for client.beta.messages.stream, exactly as sent.

    Effort and fallbacks come from the conversation (fixed when it was
    created), so changing config never alters a running conversation's
    cached prefix. messages are copied, so later appends cannot change what
    a recorded request held.
    """
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": settings.max_tokens,
        "system": system,
        "tools": tools,
        "messages": copy.deepcopy(messages),
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": conv_meta["effort"]},
        "cache_control": {"type": "ephemeral"},
    }
    if conv_meta.get("fallbacks", "default") == "default":
        kwargs["fallbacks"] = "default"
        kwargs["betas"] = [FALLBACK_BETA]
    return kwargs
