"""Factory's Droid CLI, run headless with the bench's own provider keys.

Harbor 0.25.0 has no Droid adapter, so this is a whole BaseInstalledAgent:

- One self-contained binary per release, downloaded from Factory's release
  store (downloads.factory.ai/factory-cli/releases/<version>/linux/x64/droid)
  and checked against the .sha256 file published next to it, as Factory's
  own install script does. Pinned through `version`; no npm, no Node. glibc
  only (Factory: Droid does not run on Alpine). Installed outside /logs/agent,
  at $HOME/.droid-bench/droid. The install also makes sure procps is there:
  Droid follows each shell command it starts with `ps`, and in an image
  without it (fix-git, among others) every Execute call came back "terminated
  by signal: SIGKILL" within milliseconds, even `echo hi`. Reproduced with a
  stub model on 2026-11-12: 8 of 8 commands killed without procps, 8 of 8 run
  with it. Harbor's own Cline adapter installs procps for the same kind of
  reason.
- `droid exec` with `--auto high`, Droid's own unattended level (edits, local
  commands and remote writes allowed), `-o stream-json` and the task read with
  `-f` from a file under /installed-agent, so neither the task nor the trial's
  name is ever on the long-running process's command line.
- The model is a "custom model" (Droid's bring-your-own-key route): a
  settings file uploaded per trial and passed with `--settings` declares one
  entry for the arm's provider and model, and `-m custom:bench-0` selects it.
  The key is never in the file: the entry's apiKey is `${<provider key env>}`,
  which Droid expands from the environment, and the trial's env carries the
  provider key under its own name. No Factory account or FACTORY_API_KEY is
  used or needed (checked 2026-11-12: with none set, `droid exec` went
  straight to the declared endpoint).
- Routes (ROUTES): Anthropic through Droid's `anthropic` provider, OpenAI
  (Terra) through `openai` (the Responses API), and Fireworks, xAI and
  OpenRouter through `generic-chat-completion-api` at the providers' public
  endpoints (providers.py). Any other provider fails the trial before setup.
- Effort: `reasoning_effort` becomes `-r <level>` and the entry's
  reasoningEffort.

Droid reports a failure in its stream, not on stderr: an API error is an
`error` event (exit 1), and a run can also end on a `result` with is_error
true (exit 0). The run command puts that event's message on stderr as
`droid: <message>` and exits non-zero, so Harbor's error patterns (rate
limits, auth) classify it as they would for any harness.

What Droid 1.0.0 sends, read off a stub endpoint on 2026-11-12: Opus 5.5 gets
adaptive thinking with output_config.effort from `-r` (high when unset, so the
arms always set it), as Claude Code sends it; Fireworks and xAI get
reasoning_effort; Terra goes through the Responses API with reasoning.effort.

Switched off for the bench, and only these: telemetry (DO_NOT_TRACK, and
FACTORY_OTEL_ENABLED) and the auto-update (FACTORY_DROID_AUTO_UPDATE_ENABLED,
DISABLE_AUTO_UPDATE). Everything else, Droid's built-in skills and tools
included, is Droid's default. Tools that need a Factory account (Slack,
automations, missions) fail if a model calls them; that is Droid's behaviour
without an account, not a bench bug.

Logs: the stream goes to logs/agent/droid.jsonl. Droid keeps its sessions in
~/.factory/sessions, outside /logs/agent (links there break the log download
on the Windows host); they are copied, links resolved, to logs/droid-sessions
every 20 s while it runs and once more after, and its own diagnostic log to
logs/droid-logs. Usage: the stream's final `completion` event carries the
run's totals (Droid reports no per-call usage in the stream); a run killed
before it leaves the totals in its last copied session settings file, which
bench/extract.py reads as a lower bound (_droid_calls).

The droid arms use it as `agent: bench.agents.droid:Droid` with
`family: droid`.
"""

from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from harbor.agents.agent_auth_resolution import ModelConnectionSpec, require_model_api_key
from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from . import providers

RELEASES = "https://downloads.factory.ai/factory-cli/releases"
_DIR = '"$HOME/.droid-bench"'
_BIN = '"$HOME/.droid-bench/droid"'
# Uploaded per trial: names that say nothing about the task.
TASK_FILE = "/installed-agent/droid-task.txt"
SETTINGS_FILE = "/installed-agent/droid-bench.settings.json"
MODEL_ID = "custom:bench-0"   # Droid's id for the first custom model, displayName "bench"

ANTHROPIC, OPENROUTER = "anthropic", "openrouter"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# LiteLLM provider -> (Droid provider type, endpoint, key variable).
ROUTES: dict[str, tuple[str, str, str]] = {
    ANTHROPIC: ("anthropic", ANTHROPIC_BASE_URL, "ANTHROPIC_API_KEY"),
    providers.OPENAI: ("openai", providers.OPENAI_BASE_URL, providers.KEY_ENVS[providers.OPENAI]),
    providers.FIREWORKS: ("generic-chat-completion-api", providers.FIREWORKS_BASE_URL,
                          providers.KEY_ENVS[providers.FIREWORKS]),
    providers.XAI: ("generic-chat-completion-api", providers.XAI_BASE_URL, providers.KEY_ENVS[providers.XAI]),
    OPENROUTER: ("generic-chat-completion-api", OPENROUTER_BASE_URL, "OPENROUTER_API_KEY"),
}
EFFORTS = ("off", "low", "medium", "high", "max")

# Droid's own switches for every run (see the module docstring).
DROID_ENV = {"DO_NOT_TRACK": "1", "FACTORY_OTEL_ENABLED": "false",
             "FACTORY_DROID_AUTO_UPDATE_ENABLED": "false", "DISABLE_AUTO_UPDATE": "1"}


# Shell, after the run with the stream's last error or result event in $last:
# a failure prints its message (the event's "message", else "result") to
# stderr as `droid: ...` and makes the exit code non-zero.
_REPORT_FAILURE = (
    r"""if printf '%s' "$last" | grep -qE '"type":"error"|"is_error":true'; then """
    r"""msg=$(printf '%s' "$last" | sed -n 's/.*"message":"\(\([^"\\]\|\\.\)*\)".*/\1/p'); """
    r"""[ -n "$msg" ] || msg=$(printf '%s' "$last" | sed -n 's/.*"result":"\(\([^"\\]\|\\.\)*\)".*/\1/p'); """
    r"""printf 'droid: %s\n' "$(printf '%s' "$msg" | head -c 600)" >&2; [ $rc -ne 0 ] || rc=1; fi; """
)


class DroidOptions(InstalledAgentOptions):
    reasoning_effort: Literal["off", "low", "medium", "high", "max"] | None = Field(
        default=None, description="Model reasoning effort (droid exec -r). Omit for the model's default.")


def custom_model(model_name: str | None, effort: str | None) -> dict[str, Any]:
    """The settings file's one custom model: the arm's provider and model, by key variable, never by key."""
    provider, model_id = providers.split(model_name)
    if provider not in ROUTES:
        raise ValueError(f"droid: provider {provider or '(none)'!r} is not routed; "
                         f"only {', '.join(ROUTES)} are (bench/agents/droid.py)")
    if effort is not None and effort not in EFFORTS:
        raise ValueError(f"droid: effort {effort!r} is not one of {', '.join(EFFORTS)}")
    kind, base_url, key_env = ROUTES[provider]
    entry: dict[str, Any] = {"model": model_id, "displayName": "bench", "baseUrl": base_url,
                             "apiKey": "${" + key_env + "}", "provider": kind}
    limits = providers.FIREWORKS_MODELS.get(model_id) or providers.RESPONSES_MODELS.get(model_id)
    if limits is not None:
        entry["maxContextLimit"] = limits.context
        entry["maxOutputTokens"] = limits.output
    if effort is not None:
        entry["reasoningEffort"] = effort
    return {"customModels": [entry]}


def final_result(text: str) -> dict[str, Any] | None:
    """The last `completion` event of a droid stream-json log (`result` in -o json)."""
    found = None
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") in ("completion", "result"):
            found = event
    return found


class Droid(BaseInstalledAgent):
    _OUTPUT_FILENAME = "droid.jsonl"
    _STDERR_FILENAME = "droid.stderr.txt"
    _SESSIONS_DIRNAME = "droid-sessions"
    _LOGS_DIRNAME = "droid-logs"          # Droid's own diagnostic log (~/.factory/logs)

    capabilities = AgentCapabilities(reasoning_effort=True)
    MODEL_CONNECTION = ModelConnectionSpec(base_url_destination_names={"default": None})
    options_model = DroidOptions

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Fail at construction, before any setup, on a provider or effort the settings cannot express.
        custom_model(self.model_name, self._effort())

    @classmethod
    def check_model_credentials(cls, connection, *, kwargs, env, explicit_env=None):
        require_model_api_key(connection, cls.name())

    @staticmethod
    def name() -> str:
        return "droid"

    def get_version_command(self) -> str | None:
        return f"{_BIN} --version"

    def _effort(self) -> str | None:
        return getattr(self.options, "reasoning_effort", None)

    def install_command(self) -> str:
        if not self._version:
            raise ValueError("droid: set `version` (a release on downloads.factory.ai); there is no unpinned install")
        url = f"{RELEASES}/{self._version}/linux/x64/droid"
        return ("set -euo pipefail; "
                "if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then "
                "echo 'droid needs glibc (Factory ships no musl build)' >&2; exit 1; fi; "
                f"mkdir -p {_DIR} && "
                f"curl -fsSL -o {_BIN}.part {shlex.quote(url)} && "
                f"want=$(curl -fsSL {shlex.quote(url + '.sha256')} | awk '{{print $1}}') && "
                f"got=$(sha256sum {_BIN}.part | awk '{{print $1}}') && "
                "if [ -z \"$want\" ] || [ \"$want\" != \"$got\" ]; then "
                "echo 'droid: checksum mismatch' >&2; exit 1; fi && "
                f"chmod +x {_BIN}.part && mv {_BIN}.part {_BIN} && {_BIN} --version")

    async def install(self, environment: BaseEnvironment) -> None:
        # procps: Droid tracks every shell command it starts with ps; without it (many task images
        # ship none) it SIGKILLs each command at once (found 2026-11-12, see the module docstring).
        await self.ensure_system_dependencies(environment, ("curl", "bash", "coreutils", "procps", "ca_certificates"))
        await self.exec_as_agent(environment, command=self.install_command())

    async def _upload_config_text(self, environment: BaseEnvironment, *, content: str,
                                  remote_path: str, filename: str) -> None:
        """Harbor's upload, but byte-exact: \n stays \n on the Windows host."""
        with tempfile.TemporaryDirectory(prefix="harbor-agent-config-") as temp_dir:
            local_path = Path(temp_dir) / filename
            local_path.write_text(content, encoding="utf-8", newline="\n")
            await environment.upload_file(local_path, remote_path)
        if environment.default_user is not None:
            quoted = shlex.quote(remote_path)
            await self.exec_as_root(environment, command=f"chown {shlex.quote(str(environment.default_user))} "
                                                         f"{quoted} && chmod 600 {quoted}")

    def run_env(self) -> dict[str, str]:
        """The model connection (the provider's key under its own name), then Droid's switches."""
        return {**self.model_connection.env, **DROID_ENV}

    def run_command(self) -> str:
        """droid exec on the uploaded task. Droid reports a failure in its
        stream, not on stderr: an `error` event (exit 1) or a `result` with
        is_error true (exit 0). Either way the run exits non-zero with that
        message on stderr as `droid: <message>`; only that line reaches
        Harbor's error patterns, never the agent's tool output."""
        logs = self.environment_logs_dir
        output = shlex.quote((logs / self._OUTPUT_FILENAME).as_posix())
        stderr = shlex.quote((logs / self._STDERR_FILENAME).as_posix())
        sessions = shlex.quote((logs / self._SESSIONS_DIRNAME).as_posix())
        droid_logs = shlex.quote((logs / self._LOGS_DIRNAME).as_posix())
        effort = f"-r {shlex.quote(self._effort())} " if self._effort() else ""
        # While Droid runs, its session files (which carry the running token totals) are copied out
        # every 20 s, so a run the task time limit kills still leaves its usage up to the last copy.
        keep = (f"( while sleep 20; do mkdir -p {sessions} && "
                f"cp -rL \"$HOME/.factory/sessions/.\" {sessions}/ 2>/dev/null; done ) > /dev/null 2>&1 & keep=$!; ")
        return (keep +
                f"{_BIN} exec --auto high -m {MODEL_ID} {effort}--settings {SETTINGS_FILE} "
                f"-o stream-json -f {TASK_FILE} > {output} 2> {stderr}; rc=$?; "
                "kill $keep 2>/dev/null; "
                "if [ -d \"$HOME/.factory/sessions\" ]; then mkdir -p " + sessions + " && "
                f"cp -rL \"$HOME/.factory/sessions/.\" {sessions}/ 2>/dev/null; fi; "
                "if [ -d \"$HOME/.factory/logs\" ]; then mkdir -p " + droid_logs + " && "
                f"cp -rL \"$HOME/.factory/logs/.\" {droid_logs}/ 2>/dev/null; fi; "
                f"last=$(grep -E '\"type\":\"(error|result|completion)\"' {output} | tail -n 1); "
                + _REPORT_FAILURE +
                "exit $rc")

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if not self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        settings = json.dumps(custom_model(self.model_name, self._effort()), indent=1)
        await self._upload_config_text(environment, content=settings, remote_path=SETTINGS_FILE,
                                       filename="droid-bench.settings.json")
        await self._upload_config_text(environment, content=instruction, remote_path=TASK_FILE,
                                       filename="droid-task.txt")
        await self.exec_as_agent(environment, command=self.run_command(), env=self.run_env())

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Token totals from the final result event; cost stays unset (Droid
        reports Factory credits, not dollars). input_tokens is fresh input
        only, as in Anthropic's usage, so the prompt adds the cache reads and
        writes back."""
        try:
            text = Path(self.logs_dir / self._OUTPUT_FILENAME).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        result = final_result(text)
        usage = result.get("usage") if result else None
        if not isinstance(usage, dict):
            return

        def n(key: str) -> int:
            return int(usage.get(key) or 0)

        read = n("cache_read_input_tokens")
        context.n_input_tokens = n("input_tokens") + read + n("cache_creation_input_tokens")
        context.n_cache_tokens = read
        context.n_output_tokens = n("output_tokens")
