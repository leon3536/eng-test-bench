"""Facts about harnesses that planning, pricing and the ledger all rely on.

They live apart from schemas.py, which shells out to Harbor, and from
settings.py, which reads config, so pure modules such as the ledger's
classifier can use them without importing either. Nothing here does I/O.
"""

from __future__ import annotations

import re

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
