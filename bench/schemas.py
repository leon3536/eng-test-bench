"""Harness option schemas, read from Harbor itself.

`harbor agents schema <agent> --json` describes every option a harness
accepts in the pinned Harbor version. The dashboard builds its arm editor
from this, so it only offers settings that harness will take. An agent may be
a registered Harbor name or a module.path:Class import path; the latter only
loads with the project root on PYTHONPATH, which keys.child_env provides.
"""

from __future__ import annotations

import json
import subprocess
import threading
from typing import Any

from . import harnesses, keys, procs
from .harnesses import DOLLAR_CAPS, MISPRICED_CAP, TURN_CAPS  # noqa: F401  re-exported for runs and server
from .settings import ROOT, Arm, Config, harbor_executable

# Options grouped for the arm editor. Anything not named here goes under "Other".
REASONING = ("reasoning_effort", "thinking_display", "max_thinking_tokens", "interleaved_thinking")
LIMITS = ("max_turns", "max_budget_usd", "cost_limit", "max_tokens")

# Options that change what is being measured. The editor flags them.
CAUTIONS = {
    "fallback_model": "Switches models mid-run, so the arm no longer measures one model.",
    "base_url": "Sends model traffic somewhere other than the provider's API.",
    "api_base": "Sends model traffic somewhere other than the provider's API.",
    "version": "Pins the harness version. Leave empty to use Harbor's default.",
}


def cap_misprices(arm: Any, option: str) -> bool:
    """The arm's harness would price this cap at another provider's rates.
    Keyed by family, so an import-path subclass of claude-code is covered too."""
    return harnesses.cap_misprices_for(getattr(arm, "family", "") or arm.agent, arm.provider, option)


# Current Claude models reject these request settings outright.
_CURRENT_CLAUDE = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1")
_REJECTED_ON_CURRENT_CLAUDE = {
    "temperature": "Current Claude models reject sampling parameters with a 400 error.",
    "max_thinking_tokens": "Current Claude models reject fixed thinking budgets. Use effort instead.",
}

_cache: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def agent_schema(agent: str) -> dict[str, Any]:
    with _lock:
        if agent in _cache:
            return _cache[agent]
    # The scrubbed child env: no keys (printing a schema needs none) and the
    # project root on PYTHONPATH for import-path agents.
    proc = subprocess.run(
        [harbor_executable(), "agents", "schema", agent, "--json"],
        cwd=ROOT, env=keys.child_env(None), stdin=subprocess.DEVNULL,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, creationflags=procs.NO_WINDOW,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"harbor agents schema {agent} failed: {proc.stderr.strip()[-500:]}")
    schema = json.loads(proc.stdout)
    with _lock:
        _cache[agent] = schema
    return schema


def warm(cfg: Config) -> None:
    """Fetch every schema the dashboard or a design may ask for. The first
    call per harness takes seconds, so this runs in the background at start."""
    agents = {arm.agent for arm in cfg.arms}
    agents |= set(cfg.designs.import_paths)
    agents |= set(cfg.designs.extra_harnesses)
    for agent in sorted(agents):
        try:
            agent_schema(agent)
        except Exception:  # surfaced later when the dashboard asks for it
            pass


def _resolve(variant: dict[str, Any], defs: dict[str, Any] | None) -> dict[str, Any]:
    # Pydantic puts enum classes in $defs, so terminus-2's llm_backend is
    # anyOf [{$ref: LLMBackend}, string]; follow the ref to see the enum.
    ref = variant.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/") and defs:
        target = defs.get(ref[len("#/$defs/"):])
        if isinstance(target, dict):
            return {**target, **{k: v for k, v in variant.items() if k != "$ref"}}
    return variant


def _variants(prop: dict[str, Any], defs: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    resolved = (_resolve(v, defs) for v in prop.get("anyOf", [prop]))
    return [v for v in resolved if v.get("type") != "null"]


def field_kind(prop: dict[str, Any], defs: dict[str, Any] | None = None) -> tuple[str, list[Any] | None]:
    variants = _variants(prop, defs)
    if not variants:
        return "unsupported", None
    for v in variants:
        if "enum" in v:
            return "enum", v["enum"]
    types = {v.get("type") for v in variants}
    for kind in ("boolean", "integer", "number", "string"):
        if types == {kind}:
            return kind, None
    return "json", None


def _takes_any_string(prop: dict[str, Any], defs: dict[str, Any] | None) -> bool:
    """An enum that also accepts free text, such as LLMBackend | str."""
    return any(v.get("type") == "string" and "enum" not in v for v in _variants(prop, defs))


def effort_levels(schema: dict[str, Any]) -> list[str] | None:
    """None: the harness takes no effort setting. []: any value is accepted."""
    prop = schema.get("properties", {}).get("reasoning_effort")
    if prop is None:
        return None
    kind, enum = field_kind(prop, schema.get("$defs"))
    if kind == "unsupported":
        return None
    return list(enum) if enum else []


def describe(agent: str) -> dict[str, Any]:
    schema = agent_schema(agent)
    defs = schema.get("$defs")
    fields = []
    for name, prop in schema.get("properties", {}).items():
        if name == "reasoning_effort":
            continue  # edited through the effort control
        kind, enum = field_kind(prop, defs)
        if kind == "unsupported":
            continue
        group = "reasoning" if name in REASONING else "limits" if name in LIMITS else "other"
        fields.append({
            "name": name,
            "kind": kind,
            "enum": enum,
            "default": prop.get("default"),
            "description": prop.get("description", ""),
            "group": group,
            "caution": CAUTIONS.get(name),
        })
    return {"agent": agent, "effort_levels": effort_levels(schema), "fields": fields}


def model_warnings(arm: Arm) -> dict[str, str]:
    if arm.provider == "anthropic" and arm.model in _CURRENT_CLAUDE:
        return dict(_REJECTED_ON_CURRENT_CLAUDE)
    return {}


def coerce(schema: dict[str, Any], name: str, value: Any) -> Any:
    """Match a value to the option's declared type. Several Harbor options are
    string-typed even when they hold numbers (max_budget_usd, cost_limit)."""
    prop = schema.get("properties", {}).get(name)
    if prop is None:
        raise ValueError(f"'{name}' is not an option of this harness")
    defs = schema.get("$defs")
    kind, enum = field_kind(prop, defs)
    if value is None:
        return None
    if kind == "enum":
        if value in enum or (isinstance(value, str) and _takes_any_string(prop, defs)):
            return value
        raise ValueError(f"{name} must be one of {enum}")
    if kind == "string":
        return str(value)
    if kind == "integer":
        return int(value)
    if kind == "number":
        return float(value)
    if kind == "boolean":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    # json-kind values arrive decoded already: the dashboard JSON.parses its
    # textarea and YAML gives native types. Decoding a string a second time
    # broke plain strings such as a config file path, so only inline JSON
    # written as a YAML string is decoded, and kept as text if it does not parse.
    if isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value
