"""DeepSeek Harness (dsh) headless on a Fireworks route (bench/agents/dsh.py), and its usage in extract.py.

The stream fixture follows dsh 0.2.1-rc.5's --json projection
(packages/bundle/headless/src/json-stream.ts): status events per turn and
step, step_end carrying the step's disjoint usage, and a terminal `final`.
"""

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

from bench import extract  # noqa: E402
from bench.agents import dsh  # noqa: E402
from bench.agents.dsh import DSH_ENV, DeepSeekHarness  # noqa: E402

MODEL = "fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash-1017"
KEY = "fw-test-key-not-real"
TASK = "Fix the failing build in /app; don't touch 'tests' $HOME"
TRIAL = "build-cython-ext__AbC123"   # a trial name: must appear nowhere on a command line

# (fresh input, cache reads, output) per step
STEPS = ((5200, 0, 900), (310, 5100, 420), (120, 5600, 75))


def stream(steps=STEPS, *, unfinished: bool = False, unmetered: bool = False) -> str:
    events: list[dict[str, Any]] = [{"type": "session", "sessionId": "s-1", "cwd": "/app"},
                                    {"type": "status", "phase": "turn_start", "turn": 1}]
    for n, (fresh, read, out) in enumerate(steps, 1):
        usage: dict[str, Any] = {"inputTokens": fresh, "outputTokens": out, "totalTokens": fresh + read + out,
                                 "reasoningTokens": out // 3}
        if read:
            usage["cacheReadTokens"] = read
        events += [{"type": "status", "phase": "step_start", "turn": 1, "step": n},
                   {"type": "thinking", "text": "look at the build"},
                   {"type": "tool_call", "callId": f"c{n}", "tool": "bash", "input": {"command": "make"}},
                   {"type": "tool_result", "callId": f"c{n}", "status": "completed", "result": "ok"},
                   {"type": "status", "phase": "step_end", "turn": 1, "step": n, "usage": usage}]
    if unmetered:
        events += [{"type": "status", "phase": "step_start", "turn": 1, "step": 90},
                   {"type": "status", "phase": "step_end", "turn": 1, "step": 90}]
    if unfinished:
        events.append({"type": "status", "phase": "step_start", "turn": 1, "step": 99})
        return "\n".join(json.dumps(e) for e in events) + '\n{"type":"thinking","te'   # cut by the kill
    events += [{"type": "status", "phase": "turn_end", "turn": 1, "reason": {"kind": "completed"}},
               {"type": "final", "sessionId": "s-1", "reason": {"kind": "completed"}, "text": "Done."}]
    return "\n".join(json.dumps(e) for e in events) + "\n"


class Env:
    """Records every exec and upload; the dsh run exits with run_rc and prints run_stderr."""

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
        if "--profile headless" in command:
            return ExecResult(stdout="", stderr=self.run_stderr, return_code=self.run_rc)
        return ExecResult(stdout="", stderr="", return_code=1 if command.startswith("command -v") else 0)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploads[target_path] = Path(source_path).read_text(encoding="utf-8")
        self.raw[target_path] = Path(source_path).read_bytes()

    def commands(self, needle: str) -> list[tuple[str, dict[str, str]]]:
        return [c for c in self.calls if needle in c[0]]


class DshAgent(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.dict(os.environ, {"FIREWORKS_AI_API_KEY": KEY})
        patcher.start()
        self.addCleanup(patcher.stop)

    def agent(self, model: str = MODEL, **kwargs: Any) -> DeepSeekHarness:
        logs = self.tmpdir() / TRIAL / "agent"
        logs.mkdir(parents=True)
        return DeepSeekHarness(logs_dir=logs, model_name=model, **{"version": "0.2.1-rc.5", **kwargs})

    def run_agent(self, agent: DeepSeekHarness, env: Env) -> None:
        asyncio.run(agent.run(TASK, env, AgentContext()))  # type: ignore[arg-type]

    def test_install_pins_dsh_pi_ai_and_node_in_a_prefix_of_its_own(self) -> None:
        env = Env()
        agent = self.agent()
        asyncio.run(agent.install(env))  # type: ignore[arg-type]
        ((command, _),) = env.commands("npm install")
        self.assertIn("nvm install 24.11.0 && nvm alias default 24.11.0", command)
        package = json.loads(command.split("printf '%s\\n' '", 1)[1].split("' > ", 1)[0])
        self.assertEqual(package, {"private": True, "dependencies": {"@deepseek-ai/dsh": "0.2.1-rc.5"},
                                   "overrides": {"@earendil-works/pi-ai": "1.3.0"}})
        self.assertIn('npm install --prefix "$HOME/.dsh-bench/cli" --no-audit --no-fund && '
                      '"$HOME/.dsh-bench/cli/node_modules/.bin/dsh" --version', command)
        self.assertNotIn("npm i -g", command)   # overrides do not apply to a global install
        self.assertIn("musl", command)
        self.assertEqual((agent.name(), agent.version()), ("deepseek-harness", "0.2.1-rc.5"))

    def test_run_reads_the_task_from_a_file_and_never_names_it_or_the_trial(self) -> None:
        env = Env()
        self.run_agent(self.agent(reasoning_effort="high"), env)
        ((command, run_env),) = env.commands("--profile headless")
        self.assertEqual(env.uploads[dsh.TASK_FILE], TASK)
        for text in (TASK, "Fix the failing build", TRIAL, "AbC123", KEY):
            for every, _ in env.calls:
                self.assertNotIn(text, every)
        self.assertIn('"$HOME/.dsh-bench/cli/node_modules/.bin/dsh" --profile headless '
                      f"--patch {dsh.PATCH_FILE} --json - < {dsh.TASK_FILE} > /logs/agent/dsh.jsonl 2> /logs/agent/dsh.stderr.txt; rc=$?",
                      command)
        self.assertTrue(command.endswith("exit $rc"))
        self.assertNotIn("--permission-mode", command)   # dsh-headless rejects unknown options
        self.assertNotIn("--approval-policy", command)
        self.assertEqual({k: run_env[k] for k in DSH_ENV}, {"DSH_TELEMETRY_DISABLED": "1",
                                                            "DSH_PERMISSION_MODE": "danger-full-access"})
        self.assertEqual(run_env["FIREWORKS_AI_API_KEY"], KEY)   # the key reaches dsh only through the env
        self.assertNotIn("DEEPSEEK_API_KEY", run_env)            # the native adapter would list models with it
        self.assertNotIn(KEY, env.uploads[dsh.PATCH_FILE])

    def test_uploads_keep_lf_line_ends_on_any_host(self) -> None:
        env = Env()
        agent = self.agent(reasoning_effort="high")
        asyncio.run(agent.run("line one\nline two\n", env, AgentContext()))  # type: ignore[arg-type]
        self.assertEqual(env.raw[dsh.TASK_FILE], b"line one\nline two\n")
        self.assertNotIn(b"\r", env.raw[dsh.PATCH_FILE])

    def test_dsh_home_and_its_sessions_stay_out_of_the_logs_but_are_copied_after(self) -> None:
        env = Env()
        self.run_agent(self.agent(), env)
        ((command, run_env),) = env.commands("--profile headless")
        self.assertIn('export DSH_HOME="$HOME/.dsh";', command)
        self.assertFalse([v for v in run_env.values() if v.startswith("/logs/")])
        self.assertIn('cp -rL "$DSH_HOME/sessions/." /logs/agent/dsh-sessions/', command)

    def test_the_overlay_routes_fireworks_through_pi_ai_with_the_effort(self) -> None:
        env = Env()
        self.run_agent(self.agent(reasoning_effort="high"), env)
        patch = json.loads(env.uploads[dsh.PATCH_FILE])
        rows = {row["id"]: row for row in patch}
        route = rows["llm-pi-ai"]["config"]["providers"]["bench-fireworks"]
        self.assertEqual((route["api"], route["baseURL"], route["apiKeyEnv"], route["reasoning"]),
                         ("openai-completions", "https://api.fireworks.ai/inference/v1", "FIREWORKS_AI_API_KEY",
                          "high"))
        self.assertEqual(route["compat"], {"supportsDeveloperRole": False, "maxTokensField": "max_tokens",
                                           "thinkingFormat": "deepseek"})
        (model,) = route["models"]
        self.assertEqual(model["id"], "accounts/fireworks/models/deepseek-v4p1-flash-1017")
        self.assertEqual(model["contextWindow"], 1_000_000)
        self.assertEqual(model["maxTokens"], 384_000)   # not the route's 32768 default
        self.assertEqual(rows["agent-default-model"]["config"],
                         {"provider": "bench-fireworks", "model": "accounts/fireworks/models/deepseek-v4p1-flash-1017",
                          "reasoningEffort": "high"})
        self.assertEqual(rows["session-title-llm"], {"id": "session-title-llm", "disabled": True})

    def test_effort_levels(self) -> None:
        for effort in ("off", "low", "max"):
            with self.subTest(effort=effort):
                env = Env()
                self.run_agent(self.agent(reasoning_effort=effort), env)
                rows = {row["id"]: row for row in json.loads(env.uploads[dsh.PATCH_FILE])}
                self.assertEqual(rows["llm-pi-ai"]["config"]["providers"]["bench-fireworks"]["reasoning"], effort)
                self.assertEqual(rows["agent-default-model"]["config"]["reasoningEffort"], effort)
        env = Env()
        self.run_agent(self.agent(), env)   # unset: the model's own default
        rows = {row["id"]: row for row in json.loads(env.uploads[dsh.PATCH_FILE])}
        self.assertNotIn("reasoning", rows["llm-pi-ai"]["config"]["providers"]["bench-fireworks"])
        self.assertNotIn("reasoningEffort", rows["agent-default-model"]["config"])
        with self.assertRaises(ValueError):   # Fireworks has no medium
            self.agent(reasoning_effort="medium")

    def test_a_provider_without_a_route_fails_before_setup(self) -> None:
        for model in ("anthropic/claude-opus-5-5", "openai/gpt-6.1-terra", "xai/grok-4.7"):
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "not routed"):
                self.agent(model=model)

    def test_a_failed_run_raises_with_dshs_own_error_code(self) -> None:
        with self.assertRaises(ApiRateLimitError):
            self.run_agent(self.agent(), Env(run_rc=1, run_stderr="dsh: RATE_LIMIT: 429 Too Many Requests\n"))
        with self.assertRaises(NonZeroAgentExitCodeError):
            self.run_agent(self.agent(), Env(run_rc=1, run_stderr=""))

    def test_context_totals_from_step_end_usage(self) -> None:
        agent = self.agent()
        (agent.logs_dir / "dsh.jsonl").write_text(stream(), encoding="utf-8")
        context = AgentContext()
        agent.populate_context_post_run(context)
        self.assertEqual((context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens),
                         (sum(f + r for f, r, _ in STEPS), sum(r for _, r, _ in STEPS), sum(o for _, _, o in STEPS)))
        self.assertIsNone(context.cost_usd)


class DshUsage(BenchTestCase):
    def trial(self, text: str) -> Path:
        trial = self.tmpdir()
        (trial / "agent").mkdir()
        (trial / "agent" / "dsh.jsonl").write_text(text, encoding="utf-8")
        return trial

    def test_each_step_end_is_a_call_with_disjoint_counts_added_up(self) -> None:
        log = extract.read_harness_log(self.trial(stream()))
        assert log is not None
        self.assertEqual(log.where, "agent/dsh.jsonl")
        self.assertTrue(log.replaces_harbor)
        # (model, prompt incl. cache, cache reads, output incl. reasoning, cost, 5-minute writes, 1-hour writes)
        self.assertEqual(log.calls, [(None, f + r, r, o, None, 0, 0) for f, r, o in STEPS])
        self.assertEqual(log.flags, {})

    def test_cache_writes_join_the_prompt(self) -> None:
        line = {"type": "status", "phase": "step_end", "turn": 1, "step": 1,
                "usage": {"inputTokens": 10, "outputTokens": 5, "cacheReadTokens": 100, "cacheWriteTokens": 40}}
        log = extract.read_harness_log(self.trial(json.dumps(line) + "\n"))
        assert log is not None
        self.assertEqual(log.calls, [(None, 150, 100, 5, None, 40, 0)])

    def test_a_killed_or_unmetered_step_is_flagged(self) -> None:
        log = extract.read_harness_log(self.trial(stream(unfinished=True, unmetered=True)))
        assert log is not None
        self.assertEqual(len(log.calls), len(STEPS))
        self.assertEqual(log.flags, {"unfinished_calls": {"calls": 2, "cost_usd_max": None,
                                                          "source": "agent/dsh.jsonl",
                                                          "bound": "none (dsh reports no cost)"}})

    def test_a_run_that_never_reached_a_step_has_an_empty_log(self) -> None:
        log = extract.read_harness_log(self.trial(json.dumps({"type": "session", "sessionId": "s"}) + "\n"))
        assert log is not None
        self.assertEqual((log.calls, log.flags), ([], {}))
