"""Goose with its `version` option installing exactly that release.

Harbor 0.25.0 fetches `releases/download/<version>/download_cli.sh`, but that
script downloads the latest *stable* binary unless GOOSE_VERSION is set in its
environment, which Harbor never does. So a pinned arm would run whichever
release was stable when the trial started, while reporting the pin as its
version. Goose tags its releases vX.Y.Z, so `version: "1.57.0"` would also
404 on the script's URL.

This subclass takes the pin as the bench writes every pin (X.Y.Z, so reuse and
the ledger treat it as exact), fetches the script from the vX.Y.Z tag and runs
it with GOOSE_VERSION=vX.Y.Z.

It also routes the bench's Fireworks and xAI models, which Harbor's Goose
adapter knows neither of ("Unsupported provider"). Harbor is given the
provider's endpoint in Goose's host form, the openai_chat_completions format
and the provider's own key variable, so it accepts the arm and delivers the
key as OPENAI_API_KEY. An arm that sets any of the three itself is left as it
is, and so is everything else in Harbor's Goose.

Goose itself then runs a declarative custom provider this subclass writes into
the trial (~/.config/goose/custom_providers/), not its `openai` provider, for
the effort: Goose 1.57.0 puts reasoning_effort in a chat request only for
OpenAI's own reasoning models and grok-4.5, grok-4.3 and grok-3-mini
(goose-provider-types formats/openai.rs), so GOOSE_THINKING_EFFORT never
reached these models. The 2026-10-31 DeepSeek pilot's llm_request log showed
a request with model, messages, tools and stream only. A declared model's
`request_params` are merged into every chat request (goose-providers
openai.rs, apply_declared_request_params), so the provider declares the
model with reasoning_effort at the arm's level, plus the header that keeps a
trial's calls on one cache replica (Fireworks' x-session-affinity, xAI's
x-grok-conv-id), the provider's chat endpoint and OPENAI_API_KEY as its key.
A level the provider does not take fails the trial before any model call.

The provider file is written by an exec of its own before the run, and the
header carries a hash of the trial name, not the name: Harbor runs a command
as `bash -c <command>`, so whatever is in the run's command line stays in
that wrapper's /proc cmdline for the whole run, and trial names start with
the task's name. On pypi-server the agent killed every process whose cmdline
matched the server's name, Goose's wrapper with it (exit 143).

Harbor points XDG_DATA_HOME and XDG_STATE_HOME at /logs/agent/goose, so
whatever else the agent installs under them (uv's Pythons and tools, with
their symlinks) lands in the logs dir too, and copying that dir to a Windows
host fails on the symlinks, losing goose.txt. Goose runs with the default
~/.local/share and ~/.local/state instead, and only its own dirs there are
links into Harbor's, so its session database (usage, extract.py) and its
llm_request logs are still downloaded where they were.

The goose arms use it as `agent: bench.agents.goose:PinnedGoose` with
`family: goose`.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import uuid
from typing import Any, NamedTuple

from harbor.agents.installed.goose import Goose
from harbor.environments.base import BaseEnvironment

from . import providers

_SCRIPT = "/download_cli.sh"
# The endpoint Harbor is given for these providers (its OPENAI_HOST, without /v1); Goose runs CUSTOM_ROUTES.
GOOSE_HOSTS = {providers.FIREWORKS: providers.FIREWORKS_BASE_URL.removesuffix("/v1"),
               providers.XAI: providers.XAI_BASE_URL.removesuffix("/v1")}
_RUN = "goose run --recipe"
# Harbor's XDG dirs (under /logs/agent) and the default each is replaced by for the run.
_XDG_HOMES = (("XDG_DATA_HOME", ".local/share"), ("XDG_STATE_HOME", ".local/state"))


class CustomRoute(NamedTuple):
    provider_id: str           # the custom provider's name and file name
    display_name: str
    chat_url: str              # the full chat completions URL
    session_header: str        # the header that keeps a trial's calls on one cache replica
    efforts: tuple[str, ...]   # the reasoning_effort levels the provider takes


CUSTOM_ROUTES = {
    providers.FIREWORKS: CustomRoute("bench_fireworks_ai", "Fireworks AI (bench)",
                                     providers.FIREWORKS_BASE_URL + "/chat/completions", "x-session-affinity",
                                     providers.FIREWORKS_EFFORTS),
    providers.XAI: CustomRoute("bench_xai", "xAI (bench)", providers.XAI_BASE_URL + "/chat/completions",
                               "x-grok-conv-id", ("low", "medium", "high")),
}


def custom_provider(provider: str, model_id: str, effort: str | None, session: str) -> dict[str, Any]:
    """Goose's declarative provider for one model on Fireworks or xAI, with the effort in every request."""
    route = CUSTOM_ROUTES[provider]
    model: dict[str, Any] = {"name": model_id}
    limits = providers.FIREWORKS_MODELS.get(model_id) if provider == providers.FIREWORKS else None
    if limits is not None:
        model["context_limit"] = limits.context
    if effort is not None:
        if effort not in route.efforts:
            raise ValueError(f"goose on {provider}: effort {effort!r} is not one of {', '.join(route.efforts)}")
        model["request_params"] = {"reasoning_effort": effort}
    return {"name": route.provider_id, "engine": "openai", "display_name": route.display_name,
            "api_key_env": "OPENAI_API_KEY", "base_url": route.chat_url, "models": [model],
            "headers": {route.session_header: session}, "supports_streaming": True, "requires_auth": True,
            "dynamic_models": False}


def session_affinity(session: str) -> str:
    """The cache-replica header's value for a trial: stable per trial, with no task name in it."""
    return hashlib.sha256(session.encode("utf-8")).hexdigest()[:32]


def release_tag(version: str) -> str:
    """Goose's release tag for a version: 1.57.0 -> v1.57.0."""
    return "v" + version.strip().removeprefix("v")


class PinnedGoose(Goose):
    _install_env: dict[str, str] | None = None

    def __init__(self, *args: Any, base_url: str | None = None, model_api_format: Any = None,
                 model_api_key_name: str | None = None, **kwargs: Any) -> None:
        provider, model_id = providers.split(kwargs.get("model_name", args[1] if len(args) > 1 else None))
        self._custom_route: tuple[str, str] | None = None  # (provider, model id) Goose runs as a custom provider
        if provider in GOOSE_HOSTS and base_url is None and model_api_format is None and model_api_key_name is None:
            base_url, model_api_format = GOOSE_HOSTS[provider], "openai_chat_completions"
            model_api_key_name = providers.KEY_ENVS[provider]
            self._custom_route = (provider, model_id)
        super().__init__(*args, base_url=base_url, model_api_format=model_api_format,
                         model_api_key_name=model_api_key_name, **kwargs)

    async def install(self, environment: BaseEnvironment) -> None:
        pinned = self._version
        if pinned in (None, "", "stable", "latest"):
            await super().install(environment)
            return
        tag = release_tag(str(pinned))
        if not re.fullmatch(r"v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", tag):
            raise ValueError(f"goose version {pinned!r} is not a release (X.Y.Z)")
        # Harbor builds the script URL from _version; the tag is only for the install.
        self._version, self._install_env = tag, {"GOOSE_VERSION": tag}
        try:
            await super().install(environment)
        finally:
            self._version, self._install_env = pinned, None

    async def exec_as_agent(self, environment: BaseEnvironment, command: str, env: dict[str, str] | None = None,
                            cwd: str | None = None, timeout_sec: int | None = None) -> Any:
        if self._install_env and _SCRIPT in command:
            env = {**(env or {}), **self._install_env}
        if _RUN in command:
            setup, command, env = self._run_setup(command, dict(env or {}))
            if setup:
                await super().exec_as_agent(environment, setup, env=env, cwd=cwd, timeout_sec=60)
        return await super().exec_as_agent(environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec)

    def _run_setup(self, command: str, env: dict[str, str]) -> tuple[str, str, dict[str, str]]:
        """(setup, run command, env): the setup exec moves Goose's XDG dirs out of
        the logs dir (links back in) and writes the trial's custom provider, so
        nothing of the trial's name is in the run's own command line."""
        steps: list[str] = []
        exports: list[str] = []
        for var, default in _XDG_HOMES:
            logs = env.pop(var, None)
            if logs is None:
                continue
            target, link = shlex.quote(f"{logs}/goose"), f"~/{default}/goose"
            steps.append(f"mkdir -p {target} ~/{default} && rm -rf {link} && ln -s {target} {link}")
            exports.append(f'{var}="$HOME/{default}"')
        if self._custom_route is not None:
            provider, model_id = self._custom_route
            session = getattr(self, "session_id", None) or str(uuid.uuid4())
            spec = custom_provider(provider, model_id, env.get("GOOSE_THINKING_EFFORT"), session_affinity(session))
            path = f"~/.config/goose/custom_providers/{spec['name']}.json"
            steps.append(f"mkdir -p ~/.config/goose/custom_providers && "
                         f"printf '%s\\n' {shlex.quote(json.dumps(spec, sort_keys=True))} > {path}")
            env["GOOSE_PROVIDER"] = spec["name"]
        if exports:
            command = f"export {' '.join(exports)} && {command}"
        return " && ".join(steps), command, env
