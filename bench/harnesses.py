"""Facts about harnesses that planning, pricing and the ledger all rely on.

They live apart from schemas.py, which shells out to Harbor, and from
settings.py, which reads config, so pure modules such as the ledger's
classifier can use them without importing either. Nothing here does I/O.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any

# Options through which a harness stops itself on spend or on turns. The run's
# per-trial caps are passed through whichever of these a harness accepts.
DOLLAR_CAPS = ("max_budget_usd", "cost_limit")
TURN_CAPS = ("max_turns",)

# A harness's own dollar cap is only as good as its price table. Claude Code
# prices every model at Claude rates, so on DeepSeek its $1 cap stopped a
# trial that had really cost about $0.03. Keyed by (family, option); the value
# is the only provider whose prices the harness knows.
CAP_PRICED_FOR: dict[tuple[str, str], str] = {("claude-code", "max_budget_usd"): "anthropic"}
_CAP_PRICED_FOR = CAP_PRICED_FOR  # the name schemas.py used before the move

MISPRICED_CAP = ("Not applied: this harness prices other providers' models at its own rates, "
                 "so its cap would stop trials early. The turn cap and run budget still apply.")

# Harbor's import paths are "module.path:ClassName" (harbor/utils/import_path.py
# splits on the first ':' and takes one attribute). Registered agent names such
# as claude-code or terminus-2 contain hyphens, so they can never match.
_IMPORT_PATH = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")


def is_import_path(agent: str) -> bool:
    """True for a harness class given as module.path:Class rather than a Harbor name."""
    return isinstance(agent, str) and _IMPORT_PATH.fullmatch(agent) is not None


def cap_misprices_for(family: str, provider: str, option: str) -> bool:
    """The harness would price this cap at another provider's rates."""
    priced_for = CAP_PRICED_FOR.get((family, option))
    return priced_for is not None and provider != priced_for


def reports_mispriced_cost(family: str, provider: str) -> bool:
    """The harness's own cost figure uses another provider's prices, so it
    must not be trusted over the bench's computed cost on this provider."""
    return any(f == family and priced_for != provider
               for (f, _option), priced_for in CAP_PRICED_FOR.items())


def usable_reported_cost(row: Mapping[str, Any]) -> float | None:
    """The harness's (or OpenRouter's billed) figure, unless this harness
    prices this provider's models at another provider's rates. $0 for a
    trial that used tokens is a harness that could not price the model
    (LiteLLM does this), not a billed cost."""
    reported = row.get("reported_cost_usd")
    used_tokens = bool((row.get("n_input") or 0) + (row.get("n_output") or 0))
    if (reported is not None and (float(reported) > 0 or not used_tokens)
            and not reports_mispriced_cost(row.get("family") or "", row.get("provider") or "")):
        return float(reported)
    return None


def _options(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """A trial's options, parsed ('options') or as the ledger stores them ('options_json')."""
    options = row.get("options")
    if isinstance(options, Mapping):
        return options
    raw = row.get("options_json")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def _level(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def effective_effort(row: Mapping[str, Any]) -> str | None:
    """The reasoning effort a trial ran at, wherever its harness takes it:
    the ledger's effort, else the option that carries it for a harness
    without an effort setting (opencode's and kilo-code's variant, the
    strands factory's effort, else its model's reasoning_effort). None is
    the harness default: no level was recorded."""
    level = _level(row.get("effort"))
    if level is not None:
        return level
    options = _options(row)
    factory = options.get("factory_kwargs")
    model = options.get("model_kwargs")
    params = model.get("params") if isinstance(model, Mapping) else None
    for value in (options.get("variant"), factory.get("effort") if isinstance(factory, Mapping) else None,
                  params.get("reasoning_effort") if isinstance(params, Mapping) else None):
        level = _level(value)
        if level is not None:
            return level
    return None


def options_without_effort(row: Mapping[str, Any]) -> dict[str, Any]:
    """A trial's options less the one that carried its effective effort, so
    a configuration can be named apart from its effort. The ledger's own
    effort field is not an option and leaves the options as they are."""
    options = copy.deepcopy(dict(_options(row)))
    if _level(row.get("effort")) is not None:
        return options
    if _level(options.get("variant")) is not None:
        del options["variant"]
        return options
    factory = options.get("factory_kwargs")
    if isinstance(factory, dict) and _level(factory.get("effort")) is not None:
        del factory["effort"]
        return options
    model = options.get("model_kwargs")
    params = model.get("params") if isinstance(model, dict) else None
    if isinstance(params, dict) and _level(params.get("reasoning_effort")) is not None:
        del params["reasoning_effort"]
    return options
