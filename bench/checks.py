"""Readiness checks and key tests. Key tests call read-only endpoints, so
they use no tokens and cost nothing. Docker is only ever asked read-only
questions (`docker info`)."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import threading
import time
from typing import Any, Callable

import anthropic
import httpx

from . import datasets, keys
from .settings import Config

_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_lock = threading.Lock()


def _cached(name: str, ttl: float, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    with _lock:
        hit = _cache.get(name)
        if hit and time.monotonic() - hit[0] < ttl:
            return hit[1]
    value = fn()
    with _lock:
        _cache[name] = (time.monotonic(), value)
    return value


def _docker() -> dict[str, Any]:
    try:
        proc = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                              capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        return {"ok": False, "detail": "Docker CLI not found. Install Docker Desktop."}
    except subprocess.TimeoutExpired:
        return {"ok": False, "detail": "Docker did not answer within 20 seconds."}
    if proc.returncode == 0 and proc.stdout.strip():
        return {"ok": True, "detail": f"Docker Engine {proc.stdout.strip()}"}
    return {"ok": False, "detail": "Docker Desktop is not running. Start it; this updates on its own."}


def _capacity() -> dict[str, Any]:
    try:
        proc = subprocess.run(["docker", "info", "--format", "{{.NCPU}} {{.MemTotal}}"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=20)
    except (OSError, subprocess.SubprocessError):
        return {"value": None}
    parts = proc.stdout.split() if proc.returncode == 0 and proc.stdout else []
    try:
        ncpu, mem_bytes = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return {"value": None}
    if ncpu <= 0 or mem_bytes <= 0:
        return {"value": None}
    return {"value": {"ncpu": ncpu, "mem_gb": round(mem_bytes / 1024 ** 3, 2)}}


def docker_capacity() -> dict[str, Any] | None:
    """CPUs and memory Docker can give containers, for sizing parallel arms.
    None when Docker is not answering."""
    return _cached("docker_capacity", 60, _capacity)["value"]


def _git_fix() -> dict[str, Any]:
    proc = subprocess.run(["git", "config", "--get", "core.autocrlf"],
                          capture_output=True, text=True, env=os.environ, timeout=20)
    value = proc.stdout.strip()
    if value == "false":
        return {"ok": True, "detail": "core.autocrlf=false and core.longpaths=true for bench processes"}
    return {"ok": False, "detail": f"core.autocrlf is '{value}' for bench processes; task files would get CRLF"}


def readiness(cfg: Config) -> list[dict[str, Any]]:
    return [
        {"name": "Docker", **_cached("docker", 10, _docker)},
        {"name": "Harbor", "ok": True, "detail": f"harbor {importlib.metadata.version('harbor')}"},
        {"name": "Git line endings", **_cached("git", 300, _git_fix)},
        {"name": "Task sample", "ok": bool(cfg.tasks),
         "detail": f"{len(cfg.tasks)} tasks in config/tasks.txt" if cfg.tasks
         else "No task sample. Run: python -m bench.sample_tasks"},
        {"name": "Dataset snapshot", "ok": True, "detail": _snapshot_detail(cfg.dataset)},
        {"name": "Assistant", **_assistant(cfg)},
    ]


def _assistant(cfg: Config) -> dict[str, Any]:
    if not cfg.assistant.enabled:
        return {"ok": False, "detail": "Disabled in config"}
    if not keys.is_set(keys.ASSISTANT_KEY):
        return {"ok": False, "detail": f"Set {keys.ASSISTANT_KEY} on Setup"}
    return {"ok": True, "detail": f"Claude Opus 5.5 · effort {cfg.assistant.effort}"}


def _snapshot_detail(dataset: str) -> str:
    snapshot = datasets.info(dataset)
    if snapshot is None:
        return f"{dataset} not downloaded yet; the first run downloads it once"
    return f"{dataset}: {snapshot['tasks']} tasks in {snapshot['path']}, downloaded {snapshot['downloaded_at'][:10]}"


def docker_ok() -> bool:
    return _cached("docker", 10, _docker)["ok"]


def test_key(cfg: Config, provider: str, value: str) -> dict[str, Any]:
    try:
        if provider == "anthropic":
            models = sorted({a.model for a in cfg.arms if a.provider == "anthropic" and a.enabled})
            return _test_anthropic(value, models)
        if provider == "fireworks":
            return _test_bearer("https://api.fireworks.ai/inference/v1/models", value, _count_models)
        if provider == "openrouter":
            return _test_openrouter(value)
    except httpx.HTTPError as exc:
        return {"ok": False, "detail": f"Could not reach the provider ({type(exc).__name__})"}
    return {"ok": None, "detail": "No test is defined for this provider"}


def _test_anthropic(value: str, models: list[str]) -> dict[str, Any]:
    client = anthropic.Anthropic(api_key=value, max_retries=0, timeout=20)
    try:
        for model in models or ["claude-opus-5-5"]:
            client.models.retrieve(model)
    except anthropic.AuthenticationError:
        return {"ok": False, "detail": "Rejected: invalid API key (401)"}
    except anthropic.PermissionDeniedError:
        return {"ok": False, "detail": "This key lacks permission (403)"}
    except anthropic.NotFoundError:
        return {"ok": False, "detail": f"Key works but cannot see model {model}"}
    except anthropic.APIConnectionError:
        return {"ok": False, "detail": "Could not reach api.anthropic.com"}
    except anthropic.APIStatusError as exc:
        return {"ok": False, "detail": f"Anthropic API returned HTTP {exc.status_code}"}
    return {"ok": True, "detail": f"Key accepted; can see {', '.join(models or ['claude-opus-5-5'])}"}


def _count_models(resp: httpx.Response) -> str:
    return f"Key accepted; {len(resp.json().get('data', []))} models visible"


def _test_bearer(url: str, value: str, describe: Callable[[httpx.Response], str]) -> dict[str, Any]:
    resp = httpx.get(url, headers={"Authorization": f"Bearer {value}"}, timeout=20)
    if resp.status_code in (401, 403):
        return {"ok": False, "detail": f"Rejected ({resp.status_code})"}
    if resp.status_code != 200:
        return {"ok": False, "detail": f"Provider returned HTTP {resp.status_code}"}
    return {"ok": True, "detail": describe(resp)}


def _openrouter_detail(resp: httpx.Response) -> str:
    data = resp.json().get("data") or {}
    limit = data.get("limit")
    return f"Key accepted; used ${data.get('usage', 0):.2f}" + (f" of ${limit:.2f}" if limit else ", no limit set")


def _test_openrouter(value: str) -> dict[str, Any]:
    result = _test_bearer("https://openrouter.ai/api/v1/key", value, _openrouter_detail)
    if result["detail"].endswith("HTTP 404"):
        result = _test_bearer("https://openrouter.ai/api/v1/auth/key", value, _openrouter_detail)
    return result
