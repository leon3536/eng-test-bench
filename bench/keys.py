"""API keys, kept in .env. The dashboard only ever sees them masked.

Each harbor process (and each `harbor agents schema` or dataset download)
gets a scrubbed copy of the environment from child_env():

- Every variable whose name looks like a credential is removed: any name
  containing KEY, SECRET, TOKEN (not TOKENS), PASSWORD, CREDENTIAL or AUTH,
  the same rule as Harbor's own is_sensitive_env_key. GIT_CONFIG_COUNT and
  GIT_CONFIG_KEY_n/VALUE_n survive, because they carry the CRLF fix.
- Cloud routing selectors are removed too (AWS_*, CLAUDE_CODE_USE_BEDROCK,
  CLAUDE_CODE_USE_VERTEX, CLAUDE_FORCE_OAUTH, ANTHROPIC_VERTEX_*,
  CLOUD_ML_REGION). With one of them set, Harbor's Claude Code adapter moves
  an anthropic arm to Bedrock or Vertex and forwards AWS credentials into the
  task container.
- Every endpoint override (*_BASE_URL, *_API_BASE) is removed, so an ambient
  gateway cannot silently change what an arm measures. An arm that means to
  use another endpoint says so with its base_url/api_base option, which the
  plan flags.
- Ambient Claude Code model and effort selectors are removed
  (CLAUDE_CODE_EFFORT_LEVEL, CLAUDE_CODE_MAX_TURNS, CLAUDE_CODE_MAX_OUTPUT_TOKENS,
  CLAUDE_CODE_SUBAGENT_MODEL, CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING,
  MAX_THINKING_TOKENS, ANTHROPIC_MODEL, ANTHROPIC_DEFAULT_*_MODEL,
  ANTHROPIC_SMALL_FAST_MODEL*, DISABLE_PROMPT_CACHING). Harbor's Claude Code
  adapter falls back to them, so a shell set up for Claude Code would change
  an arm's effort, thinking, turn limit, models or caching without the plan
  showing it.
  They are not secrets, so secret_values() leaves them alone.
- Every gateway URL variable a loaded config names (any provider's
  base_url_env, settings.GATEWAY_URL_VARS) is removed too, whatever its name.
- Only the key for that arm's provider is put back, and for a gateway
  provider the variable holding its URL. Harbor never reads that variable:
  the arm's base_url is written ${VAR}/route in the plan, and build_command
  fills it in from this env (expand_base_url), so the address stays out of
  plans, run.json, the arm log (hide_gateway_urls) and the config hash.
  Harbor's own job and trial config.json files under jobs/ do hold it.
- .env loading is switched off in the child (PYTHON_DOTENV_DISABLED,
  LITELLM_MODE=PRODUCTION). LiteLLM calls load_dotenv() on import, and from
  inside .venv that walks up to this project's .env and would restore every
  key removed above.
- The project root goes first on PYTHONPATH, so arms whose agent is an
  import path such as bench.agents.mini_swe_agent:MiniSweAgentWithMcp load.

This decides what the harbor process holds. What reaches a task container is
then up to Harbor's adapter for the harness, which forwards the provider key.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Callable, TypeVar
from urllib.parse import urlsplit

from dotenv import dotenv_values, set_key, unset_key

from .settings import ENV_FILE, GATEWAY_URL_VARS, ROOT, Config, ConfigError, Provider

# The run-design assistant's key. It has a Setup row of its own when no
# provider uses it (the dev profile runs everything through OpenRouter).
ASSISTANT_KEY = "ANTHROPIC_API_KEY"

_SENSITIVE = re.compile(r"KEY|SECRET|TOKEN(?!S)|PASSWORD|CREDENTIAL|AUTH", re.I)
_KEEP = re.compile(r"GIT_CONFIG_(?:COUNT|KEY_\d+|VALUE_\d+)", re.I)
_ROUTING = re.compile(
    r"AWS_.*|CLAUDE_CODE_USE_(?:BEDROCK|VERTEX)|CLAUDE_FORCE_OAUTH|ANTHROPIC_VERTEX_.*|CLOUD_ML_REGION", re.I)
_ENDPOINT_SUFFIXES = ("_BASE_URL", "_API_BASE")
_HARNESS_SELECTORS = re.compile(
    r"CLAUDE_CODE_(?:EFFORT_LEVEL|MAX_TURNS|MAX_OUTPUT_TOKENS|SUBAGENT_MODEL|DISABLE_ADAPTIVE_THINKING)"
    r"|ANTHROPIC_(?:MODEL|DEFAULT_\w+_MODEL|SMALL_FAST_MODEL\w*)|DISABLE_PROMPT_CACHING|MAX_THINKING_TOKENS",
    re.I)

# Shortest value secret_values() hands to the redactor: shorter ones ("1",
# "true") would blank out ordinary words.
_MIN_SECRET_LEN = 8

# The dashboard reads .env on every state poll from a worker thread while a key
# save replaces the file on another; on Windows that replace fails while the
# file is open. One lock serializes the bench's own access, and a short retry
# covers a second bench process or a scanner.
_ENV_LOCK = threading.RLock()
_PATIENCE = 2.0
_RETRY_SECONDS = 0.1

_T = TypeVar("_T")


def _is_secret(name: str) -> bool:
    if _KEEP.fullmatch(name):
        return False
    return bool(_SENSITIVE.search(name) or _ROUTING.fullmatch(name))


def _is_endpoint(name: str) -> bool:
    return name.upper().endswith(_ENDPOINT_SUFFIXES)


def _is_harness_selector(name: str) -> bool:
    return bool(_HARNESS_SELECTORS.fullmatch(name))


def _with_retry(fn: Callable[[], _T]) -> _T:
    """Retry fn on PermissionError for _PATIENCE seconds, then re-raise: a key
    save must never fail silently. python-dotenv removes its temp file when
    the replace fails, so trying again is safe."""
    deadline = time.monotonic() + _PATIENCE
    while True:
        try:
            return fn()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(_RETRY_SECONDS)


def _file_entries() -> dict[str, Any]:
    with _ENV_LOCK:
        if not ENV_FILE.exists():
            return {}
        # interpolate=False: set_key writes values literally, so reading them
        # back must not expand ${...} inside a key.
        return _with_retry(lambda: dict(dotenv_values(ENV_FILE, interpolate=False, encoding="utf-8")))


def _file_values() -> dict[str, str]:
    return {k: v for k, v in _file_entries().items() if v}


def load_into_process() -> None:
    """.env is the source of truth, so it wins over the launching shell."""
    os.environ.update(_file_values())


def allowed_names(cfg: Config) -> set[str]:
    names = {p.env_key for p in cfg.providers.values()}
    if cfg.assistant.enabled:
        names.add(ASSISTANT_KEY)
    return names


def set_secret(cfg: Config, name: str, value: str) -> None:
    value = value.strip()
    if name not in allowed_names(cfg):
        raise ValueError(f"{name} is not a configured provider key")
    if not value or any(c in value for c in "\r\n"):
        raise ValueError("Key must be a single non-empty line")
    with _ENV_LOCK:
        # set_key creates a missing file itself, so no separate touch to fail
        # outside the retry.
        _with_retry(lambda: set_key(str(ENV_FILE), name, value))
    os.environ[name] = value


def clear_secret(cfg: Config, name: str) -> None:
    if name not in allowed_names(cfg):
        raise ValueError(f"{name} is not a configured provider key")
    with _ENV_LOCK:
        if name in _file_entries():
            _with_retry(lambda: unset_key(str(ENV_FILE), name))
    os.environ.pop(name, None)


def get_secret(name: str) -> str | None:
    return os.environ.get(name) or None


def is_set(name: str) -> bool:
    """Whether a key is available, for checks that must not handle its value."""
    return bool(os.environ.get(name))


def mask(value: str) -> str:
    return f"…{value[-4:]}" if len(value) > 12 else "set"


def _status_row(provider: str, env_key: str, enabled: bool, in_file: dict[str, str]) -> dict[str, Any]:
    value = get_secret(env_key)
    return {
        "provider": provider,
        "env_key": env_key,
        "enabled": enabled,
        "set": bool(value),
        "masked": mask(value) if value else None,
        "source": ".env" if env_key in in_file else ("shell" if value else None),
    }


def status(cfg: Config) -> list[dict[str, Any]]:
    in_file = _file_values()
    rows = [_status_row(p.name, p.env_key, p.enabled, in_file) for p in cfg.providers.values()]
    if cfg.assistant.enabled and all(p.env_key != ASSISTANT_KEY for p in cfg.providers.values()):
        rows.append(_status_row("assistant", ASSISTANT_KEY, True, in_file))
    return rows


def secret_values() -> list[str]:
    """Values of every credential-like variable in this process, longest
    first, so a redactor replaces a key before any shorter key inside it."""
    values = {v for k, v in os.environ.items() if v and len(v) >= _MIN_SECRET_LEN and _is_secret(k)}
    return sorted(values, key=lambda v: (-len(v), v))


def base_url_template(provider: Provider, route: str | None) -> str:
    """A gateway arm's base_url as plans and run.json show it: ${VAR}/route.
    No route, "" or "/" is the gateway's root."""
    return f"${{{provider.base_url_env}}}{(route or '').rstrip('/')}"


def base_url_problem(provider: Provider) -> str | None:
    """Why a gateway provider's URL cannot be used, in words that never repeat it."""
    name = provider.base_url_env
    if not name:
        return None
    value = (os.environ.get(name) or "").strip()
    if not value:
        return f"{name} is not set"
    parts = urlsplit(value)
    if (parts.scheme not in ("http", "https") or not parts.hostname or "@" in parts.netloc
            or parts.query or parts.fragment):
        return f"{name} is not a plain http(s) URL (scheme, host, optional port and path)"
    return None


def expand_base_url(value: Any, env: dict[str, str]) -> Any:
    """base_url_template's ${VAR} filled in from a child env; anything else
    unchanged. Missing or empty is an error: Claude Code on a provider Harbor
    does not know would otherwise fall back to that provider's public URL."""
    if not (isinstance(value, str) and value.startswith("${")):
        return value
    name, _, route = value[2:].partition("}")
    root = (env.get(name) or "").strip().rstrip("/")
    if not root:
        raise ConfigError(f"{name} is not set, so this gateway arm has no endpoint")
    return root + route


def hide_gateway_urls(cmd: list[str], env: dict[str, str]) -> list[str]:
    """cmd with each gateway URL from this child env written back as ${VAR},
    for showing the command: in a log, the address would be served to anyone
    reading it."""
    roots = sorted(((name, (env.get(name) or "").strip().rstrip("/")) for name in GATEWAY_URL_VARS),
                   key=lambda pair: -len(pair[1]))
    roots = [(name, root) for name, root in roots if root]
    out = []
    for part in cmd:
        for name, root in roots:
            part = part.replace(root, f"${{{name}}}")
        out.append(part)
    return out


def child_env(provider: Provider | None) -> dict[str, str]:
    gateways = {name.upper() for name in GATEWAY_URL_VARS}
    env = {k: v for k, v in os.environ.items()
           if not _is_secret(k) and not _is_endpoint(k) and not _is_harness_selector(k)
           and k.upper() not in gateways}
    if provider is not None:
        value = get_secret(provider.env_key)
        if value:
            env[provider.env_key] = value
        if provider.base_url_env and os.environ.get(provider.base_url_env):
            # Put back by name: every gateway's variable was removed above.
            env[provider.base_url_env] = os.environ[provider.base_url_env]
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["LITELLM_MODE"] = "PRODUCTION"
    root = str(ROOT)
    rest = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != root]
    env["PYTHONPATH"] = os.pathsep.join([root, *rest])
    return env
