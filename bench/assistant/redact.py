"""Keeps secrets out of everything the assistant stores, shows or sends.

The model never needs a key, so none should reach it, the browser or a file:
tool results, events, error messages and titles all pass through redact()
before they are kept or emitted, and a user message that carries a key is
refused before anything is stored. Two nets: the exact values of every
credential-like variable in this process (keys.secret_values), and the
shapes of common keys, which catch one that is not loaded here.

This is the only module that reads keys.secret_values().

Model output already committed to a conversation is never rewritten: editing
history breaks thinking-block replay. It cannot hold a key anyway, because
the model is only ever shown redacted text.
"""

from __future__ import annotations

import re
from typing import Any

from .. import keys

MASK = "[redacted]"

# Anchored at a token start: a key always begins one, while hyphenated task
# and option names ("flask-api-endpoint-tests") contain "sk-" mid-word.
_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"),      # Anthropic, OpenRouter, OpenAI style
    re.compile(r"(?<![A-Za-z0-9])fw_[A-Za-z0-9]{16,}"),        # Fireworks
    re.compile(r"Bearer\s+\S{16,}"),
)


def redact(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    for value in keys.secret_values():  # longest first, so a key is never half-masked
        if value in text:
            text = text.replace(value, MASK)
    for pattern in _PATTERNS:
        text = pattern.sub(MASK, text)
    return text


def contains_secret(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    if any(value in text for value in keys.secret_values()):
        return True
    return any(pattern.search(text) for pattern in _PATTERNS)


def redact_obj(obj: Any) -> Any:
    """redact() applied to every string in a JSON-like value; keys included,
    since a dict key is just as visible."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {redact(k) if isinstance(k, str) else k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj
