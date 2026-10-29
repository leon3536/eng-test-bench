"""Prices for the ledger and the report.

The ledger keeps token counts as facts and prices them when asked, so a
pricing.yaml edit changes report costs without re-reading any trial. Lookups
follow the same rule as Config.price: a reported name matches a row exactly
or with a date or version tag added, never by prefix, so glm-5.3-flash is
never priced as glm-5.3. The cost formula is results._cost's, so the ledger
and the per-run results page agree to the cent.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import settings
from .settings import Price

_lock = threading.Lock()
# path -> ((mtime_ns, size), table, sha). Re-read only when the file changes.
_cache: dict[str, tuple[tuple[int, int], dict[str, dict[str, Price]], str]] = {}
_MISSING = (-1, -1)


def _stat_key(path: Path) -> tuple[int, int]:
    try:
        st = path.stat()
    except OSError:
        return _MISSING
    return (st.st_mtime_ns, st.st_size)


def _load(path: Path | None) -> tuple[dict[str, dict[str, Price]], str]:
    path = Path(path or settings.PRICING)
    key = str(path)
    stamp = _stat_key(path)
    with _lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1], hit[2]
    if stamp == _MISSING:
        table, sha = {}, "missing"
    else:
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()[:12]
        table = settings.read_pricing(path)
    with _lock:
        _cache[key] = (stamp, table, sha)
    return table, sha


def load_table(path: Path | None = None) -> dict[str, dict[str, Price]]:
    """pricing.yaml parsed exactly as load_config parses it. Raises
    ConfigError for an invalid file; a missing file is an empty table."""
    return _load(path)[0]


def table_sha(path: Path | None = None) -> str:
    """First 12 hex of sha256 over the file bytes: which prices a figure used."""
    try:
        return _load(path)[1]
    except Exception:
        return "invalid"


def lookup(table: Mapping[str, Mapping[str, Price]], provider: str, model: str) -> Price | None:
    """The row for a model name as a harness reports it, which may carry a
    provider path or a date suffix the table leaves off."""
    rows = table.get(provider) or {}
    name = str(model).split("/")[-1]
    if name in rows:
        return rows[name]
    keys = [key for key in rows if settings.is_variant(name, key)]
    return rows[max(keys, key=len)] if keys else None


def unit_cost(price: Price, n_input: int, n_cache: int, n_output: int) -> float:
    # Harbor counts cache reads inside n_input; writes are not reported apart
    # from it, so they are priced at the input rate (see pricing.yaml).
    fresh = max(n_input - n_cache, 0)
    cache_rate = price.input if price.cache_read is None else price.cache_read
    return (fresh * price.input + n_cache * cache_rate + n_output * price.output) / 1e6


def price_usage(usage: Mapping[str, Mapping[str, Any]], provider: str,
                table: Mapping[str, Mapping[str, Price]]) -> tuple[float | None, list[str]]:
    """Cost of a trial's per-model usage, and the models that had no price.

    None when nothing was reported or any model that used tokens is unpriced:
    a partial sum would look like a complete, lower figure. Zero-token entries
    (Claude Code's '<synthetic>') are ignored, as Harbor itself ignores them.
    """
    if not usage or all(u.get("n_input") is None and u.get("n_output") is None for u in usage.values()):
        return None, []
    total, unpriced = 0.0, []
    for model, u in usage.items():
        n_input = u.get("n_input") or 0
        n_cache = u.get("n_cache") or 0
        n_output = u.get("n_output") or 0
        if not (n_input or n_output):
            continue
        price = lookup(table, provider, model)
        if price is None:
            unpriced.append(model)
            continue
        total += unit_cost(price, n_input, n_cache, n_output)
    if unpriced:
        return None, sorted(unpriced)
    return total, []
