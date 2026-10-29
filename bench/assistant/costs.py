"""What the assistant's own requests cost, and the daily spend book.

The assistant bills the person's Anthropic key, so its spend is capped per
turn and per local day, and every request is recorded, including streams
stopped part-way and requests answered by a fallback model.

Pricing:
- Each entry of usage.iterations is priced at its own model, so a fallback
  hop (claude-opus-5, claude-opus-4-8) is billed at that model's rates. With
  no iterations, the top-level usage is priced at the model that answered.
- Prices come from config/pricing.yaml (cfg.price, which accepts a dated
  variant of a priced id). Cache writes cost 1.25x input (5-minute TTL).
- A model with no row is priced at the highest anthropic rates in the table
  and marked "estimated", so an unknown fallback can never make a cap lenient.
- A stopped stream is priced from message_start's usage plus the last
  output token count seen.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .. import fsutil, settings
from ..settings import Config, Price

CACHE_WRITE_MULT = 1.25
PROVIDER = "anthropic"
# Used only if pricing.yaml has no anthropic rows at all: above every
# current Claude model's list price, so caps stay conservative.
_FALLBACK_WORST = Price(input=10.0, output=50.0, cache_read=1.0)


def worst_price(cfg: Config) -> Price:
    table = cfg.pricing.get(PROVIDER) or {}
    if not table:
        return _FALLBACK_WORST
    prices = list(table.values())
    return Price(input=max(p.input for p in prices),
                 output=max(p.output for p in prices),
                 cache_read=max((p.cache_read if p.cache_read is not None else p.input) for p in prices))


def model_price(cfg: Config, model: str | None) -> tuple[Price, str]:
    """(price, "exact" | "estimated") for an Anthropic model id."""
    price = cfg.price(PROVIDER, model) if model else None
    if price is None:
        return worst_price(cfg), "estimated"
    return price, "exact"


def price_iteration(cfg: Config, model: str | None, input_tokens: int, cache_creation: int, cache_read: int,
                    output_tokens: int) -> tuple[float, str]:
    price, priced = model_price(cfg, model)
    cache_read_rate = price.cache_read if price.cache_read is not None else price.input
    usd = (input_tokens * price.input
           + cache_creation * price.input * CACHE_WRITE_MULT
           + cache_read * cache_read_rate
           + output_tokens * price.output) / 1e6
    return usd, priced


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _usage_get(usage: Any, name: str) -> Any:
    return usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)


def _iteration(cfg: Config, kind: str, model: str | None, usage: Any) -> dict[str, Any]:
    tokens = {name: _int(_usage_get(usage, name)) for name in
              ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    usd, priced = price_iteration(cfg, model, tokens["input_tokens"], tokens["cache_creation_input_tokens"],
                                  tokens["cache_read_input_tokens"], tokens["output_tokens"])
    return {"type": kind, "model": model, **tokens, "usd": usd, "priced": priced}


def price_message(cfg: Config, message: Any, requested_model: str) -> dict[str, Any]:
    """Price a finished (Beta)Message. Returns the usage.jsonl fields
    requested_model, served_model, fallback_served, iterations, usd, partial."""
    usage = message.usage
    entries = _usage_get(usage, "iterations") or []
    iterations = []
    for entry in entries:
        kind = _usage_get(entry, "type") or "message"
        # Only token-bearing entries; their model is null when it is the requested one.
        if _usage_get(entry, "input_tokens") is None and _usage_get(entry, "output_tokens") is None:
            continue
        iterations.append(_iteration(cfg, kind, _usage_get(entry, "model") or requested_model, entry))
    if not iterations:
        iterations.append(_iteration(cfg, "message", getattr(message, "model", None) or requested_model, usage))
    return {
        "requested_model": requested_model,
        "served_model": getattr(message, "model", None) or requested_model,
        "fallback_served": any(_usage_get(e, "type") == "fallback_message" for e in entries),
        "iterations": iterations,
        "usd": sum(i["usd"] for i in iterations),
        "partial": False,
    }


def price_partial(cfg: Config, start_usage: Any, output_tokens: int, model: str | None,
                  requested_model: str | None = None) -> dict[str, Any]:
    """A stream that ended early: message_start's usage plus the best output
    count the engine has (a message_delta's, or one estimated from the
    characters streamed). It is a floor: summarized thinking streams fewer
    characters than the thinking that was billed."""
    tokens = {name: _int(_usage_get(start_usage, name)) if start_usage is not None else 0 for name in
              ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    tokens["output_tokens"] = max(tokens["output_tokens"], _int(output_tokens))
    iteration = _iteration(cfg, "message", model or requested_model, tokens)
    return {
        "requested_model": requested_model or model,
        "served_model": model or requested_model,
        "fallback_served": False,
        "iterations": [iteration],
        "usd": iteration["usd"],
        "partial": True,
    }


def context_tokens(priced: dict[str, Any]) -> int:
    """Prompt size of the request that produced `priced` (its last iteration),
    which bounds what the next request will carry at least."""
    iterations = priced.get("iterations") or []
    if not iterations:
        return 0
    last = iterations[-1]
    return last["input_tokens"] + last["cache_creation_input_tokens"] + last["cache_read_input_tokens"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_date(at: str) -> date | None:
    try:
        moment = datetime.fromisoformat(at)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone().date()


class SpendBook:
    """usage.jsonl for this profile: one line per API request, append-only.

    Totals by local calendar day and by conversation are kept in memory,
    loaded once at startup, so a cap check never reads the file. The daily
    cap follows the person's local date, the same day the dashboard shows.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else settings.ASSISTANT_DIR / "usage.jsonl"
        self._lock = threading.Lock()
        self._by_day: dict[date, float] = defaultdict(float)
        self._requests_by_day: dict[date, int] = defaultdict(int)
        self._by_conversation: dict[str, float] = defaultdict(float)
        for line in fsutil.read_jsonl(self.path):
            self._count(line)

    def _count(self, line: dict[str, Any]) -> None:
        usd = line.get("usd")
        if not isinstance(usd, (int, float)) or isinstance(usd, bool):
            return
        day = local_date(line.get("at", ""))
        if day is not None:
            self._by_day[day] += float(usd)
            self._requests_by_day[day] += 1
        cid = line.get("conversation_id")
        if isinstance(cid, str):
            self._by_conversation[cid] += float(usd)

    def record(self, line: dict[str, Any]) -> None:
        line = dict(line)
        line.setdefault("at", now_iso())
        with self._lock:
            # Counted before the write: spend happened whether or not the
            # file could be appended to, and the caps must see it.
            self._count(line)
            try:
                fsutil.append_jsonl(self.path, [line])
            except OSError:
                pass

    def today_usd(self) -> float:
        with self._lock:
            return self._by_day.get(datetime.now().astimezone().date(), 0.0)

    def conversation_usd(self, cid: str) -> float:
        with self._lock:
            return self._by_conversation.get(cid, 0.0)

    def days(self, n: int = 30) -> list[dict[str, Any]]:
        """The last n local days, oldest first, zero days included."""
        n = max(1, int(n))
        today = datetime.now().astimezone().date()
        with self._lock:
            out = []
            for back in range(n - 1, -1, -1):
                day = today - timedelta(days=back)
                out.append({"date": day.isoformat(), "usd": round(self._by_day.get(day, 0.0), 6),
                            "requests": self._requests_by_day.get(day, 0)})
            return out
