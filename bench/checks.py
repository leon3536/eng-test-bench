"""Readiness checks and key tests. Key tests call read-only endpoints, so
they use no tokens and cost nothing. Docker is only ever asked read-only
questions (`docker info`)."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Callable

import anthropic
import httpx

from . import datasets, fsutil, keys
from .settings import PREBAKE_OFF, PREBAKE_REQUIRE, SCHEDULE_TASKS, Config

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
    """The dashboard's readiness rows: {name, ok, detail}, plus warn: true on
    a row that is usable but worth a look (shown amber)."""
    key = fsutil.sha256_hex(fsutil.canonical_json(cfg.effective))[:16]
    return [
        {"name": "Docker", **_cached("docker", 10, _docker)},
        {"name": "Harbor", "ok": True, "detail": f"harbor {importlib.metadata.version('harbor')}"},
        {"name": "Git line endings", **_cached("git", 300, _git_fix)},
        {"name": "Task sample", "ok": bool(cfg.tasks),
         "detail": f"{len(cfg.tasks)} tasks in config/tasks.txt" if cfg.tasks
         else "No task sample. Run: python -m bench.sample_tasks"},
        {"name": "Dataset snapshot", "ok": True, "detail": _snapshot_detail(cfg.dataset)},
        {"name": "Pre-baked images", **_cached(f"prebake:{key}", 30, lambda: _prebake_images(cfg))},
        {"name": "Docker disk", **_cached(f"docker_disk:{key}", 60, lambda: _docker_disk(cfg))},
        {"name": "Assistant", **_assistant(cfg)},
    ]


def forget(prefix: str) -> None:
    """Drop cached rows whose key starts with prefix (a build finished: 'prebake')."""
    with _lock:
        for name in [n for n in _cache if n.startswith(prefix)]:
            del _cache[name]


def _prebake_images(cfg: Config) -> dict[str, Any]:
    """Whether the images the profile's largest mode would use are ready, for
    its enabled arms. Shown only once Harbor's agent code is loaded (at start,
    in the background), so the dashboard's first poll never waits on it."""
    from . import prebake

    if cfg.run.prebake == PREBAKE_OFF:
        return {"ok": True, "detail": "Off (run.prebake): every trial installs its harness itself, as Harbor does"}
    if not prebake.recipes_ready():
        return {"ok": True, "warn": True, "detail": "Recording each harness's install; checked in a moment"}
    if not cfg.modes or not any(a.enabled for a in cfg.arms):
        return {"ok": True, "detail": "No mode or no enabled arm to check"}
    try:
        from .prebake import inventory
        from .spec import spec_from_config

        mode = max(cfg.modes, key=lambda m: cfg.modes[m].n_tasks)
        found = inventory.readiness(cfg, spec_from_config(cfg, mode, [a.name for a in cfg.arms if a.enabled]))
    except Exception as exc:
        return {"ok": True, "warn": True, "detail": f"Could not be checked: {type(exc).__name__}: {exc}"}
    arms = [r for r in found.values() if r.supported]
    if not arms:
        return {"ok": True, "detail": "No enabled arm pins its harness version, so every trial installs its harness "
                                      "itself (stock setup)"}
    errors = sorted({r.docker_error for r in arms if r.docker_error})
    if errors:
        return {"ok": True, "warn": True, "detail": f"Not checked: Docker did not answer ({errors[0]})"}
    ready, total = sum(len(r.ready) for r in arms), sum(r.tasks for r in arms)
    each = ", ".join(f"{r.family} {len(r.ready)}/{r.tasks}" for r in arms)
    old = sum(len(r.over_age) for r in arms)
    stale = f"; {old} baked more than {cfg.prebake.max_age_days} days ago, rebuild them" if old else ""
    if ready == total:
        return {"ok": True, "warn": bool(old), "detail": f"All {total} ready for the {mode} mode ({each}){stale}"}
    detail = f"{ready} of {total} ready for the {mode} mode ({each}){stale}. "
    if cfg.run.schedule == SCHEDULE_TASKS:
        # A task race builds each task's missing images just before that task's race and removes them after it.
        return {"ok": True, "warn": bool(old), "detail": detail + "Runs are task races (run.schedule tasks): each "
                                                                  "task's images are built just before its race"}
    if cfg.run.prebake == PREBAKE_REQUIRE:
        return {"ok": False, "detail": detail + "run.prebake is require, so runs wait until they are built: "
                                                "Build missing on LAUNCH"}
    return {"ok": True, "warn": True, "detail": detail + "Trials without one install their harness themselves; "
                                                         "Build missing on LAUNCH avoids that"}


def _drive(path: Any) -> str:
    return str(getattr(path, "drive", "") or getattr(path, "anchor", "") or path).rstrip("\\/")


def _roomiest_drive(besides: str) -> tuple[str, float] | None:
    """The other local drive with the most free space (Windows), for the fix text."""
    if os.name != "nt":
        return None
    best: tuple[str, float] | None = None
    for letter in "CDEFGHIJ":
        if f"{letter}:" == besides.upper():
            continue
        try:
            free = shutil.disk_usage(f"{letter}:\\").free / 1024 ** 3
        except OSError:
            continue
        if best is None or free > best[1]:
            best = (f"{letter}:", free)
    return best


def _docker_disk(cfg: Config) -> dict[str, Any]:
    """The drive holding Docker Desktop's disk image (images and builds grow
    it, and a full drive can stop Docker Desktop), and Docker's own disk as
    last measured by a build or the IMAGES panel (measuring it starts a
    container, which a status check should not)."""
    from .prebake import builder

    try:
        path, _source = builder.docker_disk()
        if path is None:
            return {"ok": True, "warn": True, "detail": "Docker Desktop's disk image was not found, so the space it "
                                                        "can grow into is not checked"}
        free = builder.host_free_gb(path)
    except OSError as exc:
        return {"ok": True, "warn": True, "detail": f"The disk image's drive could not be read: {exc}"}
    drive, reserve = _drive(path), cfg.prebake.host_reserve_gb
    vm = builder.last_vm.get("free_gb")
    inside = f"; Docker's own disk {vm:.0f} GB free" if isinstance(vm, (int, float)) else ""
    if free < reserve:
        other = _roomiest_drive(drive)
        where = f" such as {other[0]} ({other[1]:.0f} GB free)" if other and other[1] > free + reserve else ""
        return {"ok": False, "detail": (
            f"Docker's disk image is on {drive}, which has only {free:.0f} GB free; keep at least {reserve:g} GB "
            "(prebake.host_reserve_gb). Pre-baked images and pulls make it grow, and a full drive can stop Docker "
            f"Desktop. Fix: in Docker Desktop, Settings > Resources > Advanced > Disk image location, move it to a "
            f"drive with more room{where}, or free space on {drive}.")}
    if isinstance(vm, (int, float)) and vm < cfg.prebake.vm_reserve_gb:
        return {"ok": False, "detail": f"Docker's own disk has {vm:.0f} GB free; keep at least "
                                       f"{cfg.prebake.vm_reserve_gb:g} GB (prebake.vm_reserve_gb). Prune unused "
                                       "images in the IMAGES panel or with docker image prune."}
    return {"ok": True, "detail": f"Disk image on {drive}, {free:.0f} GB free{inside}"}


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
