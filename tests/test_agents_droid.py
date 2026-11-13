"""Factory's Droid headless with the bench's own provider keys (bench/agents/droid.py)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest import mock

from tests.support import BenchTestCase

from bench.prebake.adapters import harbor_agents

harbor_agents()

from harbor.agents.installed.base import ApiRateLimitError, NonZeroAgentExitCodeError  # noqa: E402
from harbor.models.agent.context import AgentContext  # noqa: E402

from bench import extract, lookups  # noqa: E402
from bench.agents import droid  # noqa: E402
from bench.agents.droid import DROID_ENV, Droid  # noqa: E402

FIREWORKS = "fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash-1017"
OPUS = "anthropic/claude-opus-5-5"
KEYS = {"FIREWORKS_AI_API_KEY": "fw-test-key-not-real", "ANTHROPIC_API_KEY": "sk-ant-test-not-real",
        "OPENAI_API_KEY": "sk-oa-test-not-real", "XAI_API_KEY": "xai-test-not-real",
        "OPENROUTER_API_KEY": "sk-or-test-not-real"}
TASK = "Fix the failing build in /app; don't touch 'tests' $HOME"
TRIAL = "build-cython-ext__AbC123"   # a trial name: must appear nowhere on a command line


def result_event(*, error: bool = False, message: str = "Done.") -> dict[str, Any]:
    return {"type": "result", "subtype": "failure" if error else "success", "is_error": error,
            "result": message, "num_turns": 3,
            "usage": {"input_tokens": 1200, "output_tokens": 340, "cache_read_input_tokens": 9000,
                      "cache_creation_input_tokens": 800, "factory_credits": 0, "thinking_tokens": 120}}


class Env:
    """Records every exec and upload; the droid run exits with run_rc and prints run_stderr."""

    default_user = None

    def __init__(self, run_rc: int = 0, run_stderr: str = "") -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.uploads: dict[str, str] = {}
        self.raw: dict[str, bytes] = {}
        self.run_rc, self.run_stderr = run_rc, run_stderr

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: Any = None) -> Any:
        from harbor.environments.base import ExecResult

        self.calls.append((command, dict(env or {})))
        if command.startswith("for manager"):
            return ExecResult(stdout="apt-get", stderr="", return_code=0)
        if " exec --auto high" in command:
            return ExecResult(stdout="", stderr=self.run_stderr, return_code=self.run_rc)
        return ExecResult(stdout="", stderr="", return_code=1 if command.startswith("command -v") else 0)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploads[target_path] = Path(source_path).read_text(encoding="utf-8")
        self.raw[target_path] = Path(source_path).read_bytes()

    def commands(self, needle: str) -> list[tuple[str, dict[str, str]]]:
        return [c for c in self.calls if needle in c[0]]


class DroidAgent(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.dict(os.environ, KEYS)
        patcher.start()
        self.addCleanup(patcher.stop)

    def agent(self, model: str = FIREWORKS, **kwargs: Any) -> Droid:
        logs = self.tmpdir() / TRIAL / "agent"
        logs.mkdir(parents=True)
        return Droid(logs_dir=logs, model_name=model, **{"version": "1.0.0", **kwargs})

    def run_agent(self, agent: Droid, env: Env) -> None:
        asyncio.run(agent.run(TASK, env, AgentContext()))  # type: ignore[arg-type]

    def settings(self, env: Env) -> dict[str, Any]:
        (entry,) = json.loads(env.uploads[droid.SETTINGS_FILE])["customModels"]
        return entry

    def test_install_downloads_the_pinned_release_and_checks_its_sha256(self) -> None:
        env = Env()
        agent = self.agent()
        asyncio.run(agent.install(env))  # type: ignore[arg-type]
        ((command, _),) = env.commands("downloads.factory.ai")
        url = "https://downloads.factory.ai/factory-cli/releases/1.0.0/linux/x64/droid"
        self.assertIn(f"curl -fsSL -o \"$HOME/.droid-bench/droid\".part {url} &&", command)
        self.assertIn(f"curl -fsSL {url}.sha256", command)
        self.assertIn("checksum mismatch", command)
        self.assertIn("musl", command)
        self.assertTrue(command.endswith('"$HOME/.droid-bench/droid" --version'))
        self.assertEqual((agent.name(), agent.version()), ("droid", "1.0.0"))

    def test_install_brings_procps(self) -> None:
        # Without ps in the image Droid SIGKILLs every shell command it starts (2026-11-12).
        env = Env()
        with mock.patch.object(Droid, "ensure_system_dependencies") as deps:
            asyncio.run(self.agent().install(env))  # type: ignore[arg-type]
        (_, needed), _ = deps.call_args
        self.assertIn("procps", needed)
        self.assertIn("curl", needed)

    def test_an_unpinned_install_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "set `version`"):
            Droid(logs_dir=self.tmpdir(), model_name=FIREWORKS).install_command()

    def test_run_reads_the_task_from_a_file_and_never_names_it_the_trial_or_a_key(self) -> None:
        env = Env()
        self.run_agent(self.agent(reasoning_effort="high"), env)
        ((command, run_env),) = env.commands(" exec --auto high")
        self.assertEqual(env.uploads[droid.TASK_FILE], TASK)
        for text in (TASK, "Fix the failing build", TRIAL, "AbC123", *KEYS.values()):
            for every, _ in env.calls:
                self.assertNotIn(text, every)
            for upload in env.uploads.values():
                if upload != TASK:
                    self.assertNotIn(text, upload)
        self.assertIn('"$HOME/.droid-bench/droid" exec --auto high -m custom:bench-0 -r high '
                      f"--settings {droid.SETTINGS_FILE} -o stream-json -f {droid.TASK_FILE} "
                      "> /logs/agent/droid.jsonl 2> /logs/agent/droid.stderr.txt; rc=$?", command)
        self.assertTrue(command.endswith("exit $rc"))
        self.assertEqual({k: run_env[k] for k in DROID_ENV}, DROID_ENV)
        self.assertEqual(run_env["FIREWORKS_AI_API_KEY"], KEYS["FIREWORKS_AI_API_KEY"])
        self.assertNotIn("FACTORY_API_KEY", run_env)

    def test_uploads_keep_lf_line_ends_on_any_host(self) -> None:
        env = Env()
        asyncio.run(self.agent().run("line one\nline two\n", env, AgentContext()))  # type: ignore[arg-type]
        self.assertEqual(env.raw[droid.TASK_FILE], b"line one\nline two\n")
        self.assertNotIn(b"\r", env.raw[droid.SETTINGS_FILE])

    def test_sessions_stay_out_of_the_logs_but_are_copied_after(self) -> None:
        env = Env()
        self.run_agent(self.agent(), env)
        ((command, run_env),) = env.commands(" exec --auto high")
        self.assertFalse([v for v in run_env.values() if v.startswith("/logs/")])
        self.assertIn('cp -rL "$HOME/.factory/sessions/." /logs/agent/droid-sessions/', command)

    def test_sessions_are_copied_while_it_runs_without_holding_the_exec_open(self) -> None:
        env = Env()
        self.run_agent(self.agent(), env)
        ((command, _),) = env.commands(" exec --auto high")
        self.assertIn('( while sleep 20; do mkdir -p /logs/agent/droid-sessions && cp -rL "$HOME/.factory/sessions/." '
                      "/logs/agent/droid-sessions/ 2>/dev/null; done ) > /dev/null 2>&1 & keep=$!; ", command)
        self.assertIn("rc=$?; kill $keep 2>/dev/null; ", command)
        self.assertIn('cp -rL "$HOME/.factory/logs/." /logs/agent/droid-logs/', command)

    def test_each_provider_gets_its_route_by_key_variable(self) -> None:
        cases = {
            OPUS: ("anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY", "claude-opus-5-5"),
            FIREWORKS: ("generic-chat-completion-api", "https://api.fireworks.ai/inference/v1",
                        "FIREWORKS_AI_API_KEY", "accounts/fireworks/models/deepseek-v4p1-flash-1017"),
            "openai/gpt-6.1-terra": ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-6.1-terra"),
            "xai/grok-4.7": ("generic-chat-completion-api", "https://api.x.ai/v1", "XAI_API_KEY", "grok-4.7"),
            "openrouter/deepseek/deepseek-v4.1-flash": ("generic-chat-completion-api", "https://openrouter.ai/api/v1",
                                                        "OPENROUTER_API_KEY", "deepseek/deepseek-v4.1-flash"),
        }
        for model, (kind, url, key_env, model_id) in cases.items():
            with self.subTest(model=model):
                env = Env()
                self.run_agent(self.agent(model=model), env)
                entry = self.settings(env)
                self.assertEqual((entry["provider"], entry["baseUrl"], entry["apiKey"], entry["model"],
                                  entry["displayName"]), (kind, url, "${" + key_env + "}", model_id, "bench"))

    def test_known_models_carry_their_limits(self) -> None:
        env = Env()
        self.run_agent(self.agent(), env)
        entry = self.settings(env)
        self.assertEqual((entry["maxContextLimit"], entry["maxOutputTokens"]), (1_000_000, 384_000))
        env = Env()
        self.run_agent(self.agent(model=OPUS), env)
        self.assertNotIn("maxOutputTokens", self.settings(env))   # Droid's own Claude limits

    def test_effort_goes_to_the_flag_and_the_entry(self) -> None:
        for effort in ("low", "medium", "high"):
            with self.subTest(effort=effort):
                env = Env()
                self.run_agent(self.agent(model=OPUS, reasoning_effort=effort), env)
                ((command, _),) = env.commands(" exec --auto high")
                self.assertIn(f"-m custom:bench-0 -r {effort} --settings", command)
                self.assertEqual(self.settings(env)["reasoningEffort"], effort)
        env = Env()
        self.run_agent(self.agent(model=OPUS), env)   # unset: the model's own default
        ((command, _),) = env.commands(" exec --auto high")
        self.assertNotIn(" -r ", command)
        self.assertNotIn("reasoningEffort", self.settings(env))
        with self.assertRaises(ValueError):
            self.agent(reasoning_effort="extreme")

    def test_a_provider_without_a_route_fails_before_setup(self) -> None:
        for model in ("gemini/gemini-3-pro", "claude-opus-5-5"):
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "not routed"):
                self.agent(model=model)

    def test_a_failed_run_raises_with_droids_own_message(self) -> None:
        with self.assertRaises(ApiRateLimitError):
            self.run_agent(self.agent(), Env(run_rc=1, run_stderr="droid: 429 Too Many Requests: rate limit\n"))
        with self.assertRaises(NonZeroAgentExitCodeError):
            self.run_agent(self.agent(), Env(run_rc=1, run_stderr=""))

    def test_a_failure_in_the_stream_reaches_stderr_and_the_exit_code(self) -> None:
        env = Env()
        self.run_agent(self.agent(), env)
        ((command, _),) = env.commands(" exec --auto high")
        self.assertIn('last=$(grep -E \'"type":"(error|result|completion)"\' /logs/agent/droid.jsonl | tail -n 1); '
                      + droid._REPORT_FAILURE + "exit $rc", command)
        self.assertIn("\"type\":\"error\"|\"is_error\":true", droid._REPORT_FAILURE)
        self.assertIn("printf 'droid: %s\\n'", droid._REPORT_FAILURE)
        self.assertIn("[ $rc -ne 0 ] || rc=1", droid._REPORT_FAILURE)

    def test_context_totals_from_the_final_result(self) -> None:
        agent = self.agent()
        final = {**result_event(), "type": "completion"}   # stream-json ends on `completion`, json on `result`
        lines = [{"type": "system", "subtype": "init"}, result_event(error=True, message="first"), final]
        (agent.logs_dir / "droid.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
        context = AgentContext()
        agent.populate_context_post_run(context)
        self.assertEqual((context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens),
                         (1200 + 9000 + 800, 9000, 340))
        self.assertIsNone(context.cost_usd)

    def test_no_result_leaves_the_context_empty(self) -> None:
        agent = self.agent()
        (agent.logs_dir / "droid.jsonl").write_text('{"type":"system"}\n{"type":"mess', encoding="utf-8")
        context = AgentContext()
        agent.populate_context_post_run(context)
        self.assertIsNone(context.n_input_tokens)


class DroidLogs(BenchTestCase):
    """bench/extract.py and bench/lookups.py on a Droid trial's files."""

    def trial(self, lines: list[dict[str, Any]], sessions: dict[str, dict[str, Any]] | None = None) -> Path:
        trial = self.tmpdir()
        (trial / "agent").mkdir()
        (trial / "agent" / "droid.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
        for name, data in (sessions or {}).items():
            path = trial / "agent" / "droid-sessions" / "-app" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
        return trial

    def test_a_finished_run_is_one_entry_with_disjoint_counts_added_up(self) -> None:
        log = extract.read_harness_log(self.trial([{"type": "system"}, {**result_event(), "type": "completion"}]))
        assert log is not None
        self.assertEqual(log.where, "agent/droid.jsonl")
        self.assertTrue(log.replaces_harbor)
        # (model, prompt incl. cache, cache reads, output, cost, 5-minute writes, 1-hour writes)
        self.assertEqual(log.calls, [(None, 1200 + 9000 + 800, 9000, 340, None, 800, 0)])
        self.assertEqual(log.flags, {})

    def test_a_killed_run_counts_its_last_session_copy_and_is_flagged(self) -> None:
        usage = {"inputTokens": 500, "outputTokens": 70, "cacheReadTokens": 4000, "cacheCreationTokens": 0}
        older = {"inputTokens": 100, "outputTokens": 10, "cacheReadTokens": 0, "cacheCreationTokens": 0}
        log = extract.read_harness_log(self.trial([{"type": "system"}, {"type": "tool_call", "id": "a"}],
                                                  {"s1.settings.json": {"tokenUsage": older},
                                                   "s2.settings.json": {"inclusiveTokenUsage": usage}}))
        assert log is not None
        self.assertEqual(log.calls, [(None, 4500, 4000, 70, None, 0, 0)])
        self.assertEqual(log.flags["unfinished_calls"]["source"], "agent/droid.jsonl")

    def test_a_killed_run_with_no_session_copy_has_no_calls(self) -> None:
        log = extract.read_harness_log(self.trial([{"type": "system"}]))
        assert log is not None
        self.assertEqual(log.calls, [])
        self.assertIn("unfinished_calls", log.flags)

    def test_answer_lookups_are_read_from_the_stream(self) -> None:
        url = "https://raw.githubusercontent.com/laude-institute/terminal-bench/main/tasks/fix-git/solution.sh"
        lines = [{"type": "tool_call", "id": "c1", "toolName": "Execute", "parameters": {"command": f"curl -s {url}"}},
                 {"type": "tool_result", "id": "c1", "isError": False, "value": "#!/bin/bash\ngit cherry-pick abc\n" * 30},
                 {"type": "tool_call", "id": "c2", "toolName": "Execute",
                  "parameters": {"command": "cd /app && git commit -am fix"}},
                 {"type": "tool_result", "id": "c2", "isError": False, "value": "ok"}]
        found = lookups.scan(self.trial(lines) / "agent")
        assert found is not None
        self.assertEqual((found["used"], found["source"]), (True, "agent/droid.jsonl"))
        self.assertIsNone(lookups.scan(self.trial(lines[2:]) / "agent"))
        calls = lookups.droid_calls(json.dumps({"type": "tool_call", "id": "x", "toolName": "Execute",
                                                "parameters": {"command": "ls"}}) + "\n"
                                    + json.dumps({"type": "tool_result", "id": "x", "isError": True,
                                                  "error": {"message": "Error: boom"}}))
        assert calls is not None
        self.assertEqual((calls[0].name, calls[0].runs, calls[0].output), ("Execute", "ls", "Error: boom"))
