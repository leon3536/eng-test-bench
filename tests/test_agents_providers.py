"""Every harness on every non-Anthropic model, built as a design run would build it.

For each of the eight harnesses on DeepSeek V4.1 Flash, GLM 5.3 and Kimi K3
(Fireworks), GPT-6.1 Terra (OpenAI) and Grok 4.7 (xAI), the arm comes from
tests/provider_arms.py, goes through the bench's own design normalization
and agent_kwargs (the --ak values a run passes), and becomes Harbor's
adapter with dummy keys. The test then runs the adapter against a recording
environment and checks the exact command, its environment (which key, under
which name), the provider and the effort it would send. Nothing reaches a
network or a container, and no model is called.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from tests import provider_arms
from tests.support import BenchTestCase

from bench.prebake.adapters import harbor_agents

harbor_agents()

from bench import designs, runs, schemas, settings  # noqa: E402
from bench.spec import spec_from_design  # noqa: E402

FW, OA, XA = "fw-test-key-0001", "oa-test-key-0002", "xai-test-key-0003"
KEYS = {"FIREWORKS_AI_API_KEY": FW, "OPENAI_API_KEY": OA, "XAI_API_KEY": XA}
KEY_OF = {"fireworks": ("FIREWORKS_AI_API_KEY", FW), "openai": ("OPENAI_API_KEY", OA), "xai": ("XAI_API_KEY", XA)}
INSTRUCTION = "Fix the bug; don't touch 'tests'"
PREFIX = "set -o pipefail; "
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1"
SESSION = "trial-session-1"


class Recorder:
    """A task environment that records every exec and upload and answers like a bare apt image."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.uploads: dict[str, str] = {}
        self.default_user = None
        self.task_env_config = SimpleNamespace(workdir="/app")

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: Any = None) -> Any:
        from harbor.environments.base import ExecResult

        self.calls.append((command, dict(env or {})))
        if command.startswith("for manager"):
            return ExecResult(stdout="apt-get", stderr="", return_code=0)
        return ExecResult(stdout="", stderr="", return_code=1 if command.startswith("command -v") else 0)

    async def upload_file(self, source: Any, target: str) -> None:
        self.uploads[target] = Path(source).read_text(encoding="utf-8")

    async def upload_dir(self, source: Any, target: str) -> None:
        self.uploads[target] = "<dir>"

    def find(self, needle: str) -> tuple[str, dict[str, str]]:
        found = [c for c in self.calls if needle in c[0]]
        if len(found) != 1:
            raise AssertionError(f"expected one exec with {needle!r}, got {[c for c, _ in self.calls]}")
        return found[0]


class ProviderArms(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.dict(os.environ, KEYS)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cfg = self.load_shipped_config("bench")

    # Building -------------------------------------------------------------------

    def arm_specs(self, key: str) -> dict[str, Any]:
        """{arm name: (ArmSpec, kwargs)} for one model's design, as a run plans them."""
        spec_in = provider_arms.design_input(key, budget_usd=50)["spec"]
        run_spec = spec_from_design(designs.normalize(self.cfg, spec_in), None)
        out = {}
        for arm in run_spec.arms:
            kwargs, _ = runs.agent_kwargs(run_spec.run, arm, schemas.agent_schema(arm.agent))
            # Harbor reads each --ak value back as JSON.
            out[arm.name] = (arm, {k: json.loads(json.dumps(v)) for k, v in kwargs.items()})
        return out

    def build(self, key: str, name: str) -> Any:
        from harbor.agents.factory import AgentFactory
        from harbor.models.trial.config import AgentConfig

        arm, kwargs = self.arm_specs(key)[name]
        config = AgentConfig(import_path=arm.agent) if arm.kind == "import_path" else AgentConfig(name=arm.agent)
        cls = AgentFactory.get_agent_class_from_config(config)
        return cls(logs_dir=self.tmpdir(), model_name=self.cfg.harbor_model(arm), **kwargs)

    def run_agent(self, key: str, name: str) -> tuple[Any, Recorder]:
        agent, env = self.build(key, name), Recorder()
        agent.session_id = SESSION  # Harbor sets it per trial
        asyncio.run(agent.run(INSTRUCTION, env, SimpleNamespace(metadata={})))  # type: ignore[arg-type]
        return agent, env

    @staticmethod
    def model_id(m: provider_arms.Model) -> str:
        """The model as the provider names it: Harbor's -m value without its LiteLLM provider."""
        return m.harbor.partition("/")[2]

    # The design --------------------------------------------------------------------

    def test_each_design_is_the_baseline_plus_the_eight_harnesses(self) -> None:
        for key, m in provider_arms.MODELS.items():
            with self.subTest(model=key):
                specs = self.arm_specs(key)
                self.assertEqual(list(specs), ["claude-code", *provider_arms.HARNESSES])
                baseline, _ = specs["claude-code"]
                self.assertEqual((baseline.provider, baseline.model, baseline.family, baseline.effort),
                                 ("anthropic", "claude-opus-5-5", "claude-code", "inherit"))
                for name in provider_arms.HARNESSES:
                    arm, _ = specs[name]
                    self.assertEqual((arm.provider, self.cfg.harbor_model(arm)), (m.provider, m.harbor), name)
                self.assertEqual([specs[n][0].family for n in provider_arms.HARNESSES],
                                 ["terminus-2", "mini-swe-agent", "pi", "strands", "cline-cli", "goose", "opencode",
                                  "kilo-code"])

    def test_designs_set_no_endpoint_or_credential_and_reuse_the_baseline(self) -> None:
        for key, _m in provider_arms.MODELS.items():
            with self.subTest(model=key):
                body = provider_arms.design_input(key, budget_usd=50)
                spec = designs.normalize(self.cfg, body["spec"])
                run = spec_from_design(spec, None).run  # prefer and tasks are the defaults, so not stored
                self.assertEqual((spec["reuse"], run.prebake, run.schedule, run.effort),
                                 ("measured", "prefer", "tasks", "medium"))
                self.assertEqual(len(spec["tasks"]), 27)
                for arm in spec["arms"]:
                    self.assertEqual(designs._denied_options(arm["options"]), [], arm["name"])
                    self.assertEqual(designs._host_file_options(self.cfg, arm), [], arm["name"])
                designs.DesignMetaIn.model_validate({k: v for k, v in body.items() if k != "spec"})
                # No arm pairs with the Opus baseline in the run's report: the purpose says where that comparison is.
                self.assertIn(f"compare models and baseline_model {settings.DEFAULT_BASELINE_MODEL}", body["purpose"])

    def test_every_model_has_a_price(self) -> None:
        for _key, m in provider_arms.MODELS.items():
            with self.subTest(model=_key):
                price = self.cfg.price(m.provider, m.model)
                assert price is not None, m.model
                self.assertGreater(price.output, price.input)

    def test_the_shipped_prices_have_the_research_reports_rows(self) -> None:
        shipped = self.load_shipped_config("bench")
        dated, alias = (shipped.price("fireworks", m) for m in ("deepseek-v4p1-flash-1017", "deepseek-v4p1-flash"))
        self.assertIsNotNone(alias)
        self.assertEqual(dated, alias)
        terra = shipped.price("openai", "gpt-6.1-terra")
        assert terra is not None
        # 30-minute writes (the default) at 1.25, 12-hour writes at 2.50; the bench files OpenAI's writes as 5m.
        self.assertEqual((terra.cache_write_5m, terra.cache_write_1h), (1.25, 2.50))

    # Harness by harness ------------------------------------------------------------

    def test_terminus_2_sends_litellm_its_effort_and_session_header(self) -> None:
        import litellm

        for key, m in provider_arms.MODELS.items():
            with self.subTest(model=key):
                agent = self.build(key, "terminus-2")
                agent.session_id = SESSION
                agent._reset_per_run_state()
                llm = agent._llm
                self.assertEqual((llm._model_name, llm._reasoning_effort, llm._use_responses_api, llm._api_base),
                                 (m.harbor, m.effort, False, None))
                headers = {m.session_header: SESSION} if m.session_header else None
                self.assertEqual(agent._llm_call_kwargs.get("extra_headers"), headers)
                # Harbor passes drop_params, so LiteLLM would drop an effort it thinks the model lacks.
                provider, _, model = m.harbor.partition("/")
                self.assertIn("reasoning_effort",
                              litellm.get_supported_openai_params(model=model, custom_llm_provider=provider) or [])

    def test_mini_swe_agent(self) -> None:
        q = shlex.quote(INSTRUCTION)
        for key, m in provider_arms.MODELS.items():
            with self.subTest(model=key):
                _, env = self.run_agent(key, "mini-swe-agent")
                command, run_env = env.find("mini-swe-agent --yolo")
                # Harbor's default header where the arm names none.
                header = f"-c model.model_kwargs.extra_headers.{m.session_header or 'X-Session-ID'}={SESSION} "
                if m.provider == "openai":  # tools only through Responses: LiteLLM's Responses client, effort nested
                    effort = (f"-c model.model_kwargs.reasoning.effort={m.effort} "
                              "-c model.model_kwargs.custom_llm_provider=openai -c model.model_class=litellm_response ")
                else:
                    effort = f"-c model.model_kwargs.reasoning_effort={m.effort} "
                self.assertEqual(command, (
                    f'{PREFIX}if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; '
                    f'else export PATH="$HOME/.local/bin:$PATH"; fi; mini-swe-agent --yolo --model={m.harbor} '
                    f"--task={q} --output=/logs/agent/mini-swe-agent.trajectory.json --cost-limit 15 -c mini "
                    f"-c model.model_kwargs.timeout=3600 {header}{effort}--exit-immediately 2>&1 </dev/null "
                    "| tee /logs/agent/mini-swe-agent.txt"))
                name, value = KEY_OF[m.provider]
                self.assertEqual(run_env, {name: value, "MSWEA_CONFIGURED": "true",
                                           "MSWEA_COST_TRACKING": "ignore_errors"})

    def test_pi(self) -> None:
        q = shlex.quote(INSTRUCTION)
        for key, m in provider_arms.MODELS.items():
            with self.subTest(model=key):
                agent, env = self.run_agent(key, "pi")
                command, run_env = env.find(" pi --print ")
                if m.provider == "fireworks":
                    isolated, provider, model = "", "fireworks", self.model_id(m)
                    expected_env = {"FIREWORKS_AI_API_KEY": FW, "FIREWORKS_API_KEY": FW}
                elif m.provider == "openai":
                    isolated, provider, model = "PI_CODING_AGENT_DIR=/tmp/harbor-pi-agent ", "harbor-endpoint", m.model
                    expected_env = {"OPENAI_API_KEY": OA}
                    self.assertEqual(json.loads(env.uploads["/tmp/harbor-pi-agent/models.json"]), {"providers": {
                        "harbor-endpoint": {"baseUrl": "https://api.openai.com/v1", "apiKey": "$OPENAI_API_KEY",
                                            "api": "openai-responses",
                                            "models": [{"id": "gpt-6.1-terra", "reasoning": True,
                                                        "contextWindow": 1050000, "maxTokens": 128000,
                                                        "input": ["text", "image"]}]}}})
                else:
                    isolated, provider, model = "", "xai", m.model
                    expected_env = {"XAI_API_KEY": XA}
                self.assertEqual(command, (
                    f"{PREFIX}. ~/.nvm/nvm.sh; mkdir -p /logs/agent/pi/sessions && {isolated}pi --print --mode json "
                    f"--session-dir /logs/agent/pi/sessions --provider {provider} --model {model} "
                    f"--extension /tmp/harbor-pi-agent/max-turns.ts --thinking {m.effort} {q} 2>&1 </dev/null "
                    "| grep -v '\"type\":\"message_update\"' | stdbuf -oL tee /logs/agent/pi.txt"))
                self.assertEqual(run_env, expected_env)
                # The trial still records the provider as the bench and LiteLLM name it.
                info = agent.to_agent_info().model_info
                self.assertEqual(info.provider, m.harbor.partition("/")[0])

    def test_strands(self) -> None:
        for key, m in provider_arms.MODELS.items():
            with self.subTest(model=key):
                _, env = self.run_agent(key, "strands")
                command, run_env = env.find("strands_runner.py")
                self.assertEqual(command, (
                    f"{PREFIX}rm -f /logs/agent/strands-result.json /logs/agent/strands-events.jsonl; exec "
                    "/installed-agent/strands/venv/bin/python /installed-agent/strands/strands_runner.py "
                    "--config /installed-agent/strands/config.json "
                    "--instruction /installed-agent/strands/instruction.txt "
                    "--result /logs/agent/strands-result.json --events /logs/agent/strands-events.jsonl"))
                name, value = KEY_OF[m.provider]
                provider, _, model_id = m.harbor.partition("/")
                self.assertEqual(run_env, {name: value, "PYTHONPATH": "/installed-agent/strands/factory",
                                           "HARBOR_MODEL": m.harbor, "HARBOR_MODEL_ID": model_id,
                                           "HARBOR_MODEL_PROVIDER": provider})
                if m.provider == "openai":
                    # strands-harness's openai provider: OpenAIResponsesModel with reasoning.effort.
                    factory_kwargs, model_kwargs = {"model_spec": m.harbor, "effort": m.effort}, {}
                else:
                    # The runner builds LiteLLMModel (fireworks_ai) or OpenAIModel (xai) with these params;
                    # the factory checks them and gives the xai one its endpoint (test_strands_factory.py).
                    factory_kwargs, model_kwargs = {}, {"params": {"reasoning_effort": m.effort}}
                self.assertEqual(json.loads(env.uploads["/installed-agent/strands/config.json"]), {
                    "base_url": None, "factory": "bench_strands_factory:create_agent", "factory_kwargs": factory_kwargs,
                    "mcp_servers": {}, "model_api_key_name": name,
                    "model_connection_overrides": {"api_key": True, "base_url": False}, "model_id": model_id,
                    "model_kwargs": model_kwargs, "model_name": m.harbor, "model_provider": provider,
                    "workdir": "/app"})

    def test_cline(self) -> None:
        q = shlex.quote(INSTRUCTION)
        for key, m in provider_arms.MODELS.items():
            with self.subTest(model=key):
                _, env = self.run_agent(key, "cline")
                command, run_env = env.find(" cline -P ")
                provider = {"fireworks": "fireworks", "openai": "openai-native", "xai": "xai"}[m.provider]
                self.assertEqual(command, (
                    f'{PREFIX}export NVM_DIR="$HOME/.nvm"; if [ -s "$NVM_DIR/nvm.sh" ]; then . "$NVM_DIR/nvm.sh"; '
                    f'nvm use 22 >/dev/null 2>&1 || true; fi; set -o pipefail; cline -P {provider} -k "$API_KEY" '
                    f'-m "$MODELID" --json --yolo --thinking {m.effort} -- {q} < /dev/null 2>&1 '
                    "| stdbuf -oL tee /logs/agent/cline.txt; status=${PIPESTATUS[0]}; "
                    'echo "__CLINE_EXIT=${status}" | tee -a /logs/agent/cline.txt; exit "${status}"'))
                self.assertEqual(run_env, {"PROVIDER": provider, "API_KEY": KEY_OF[m.provider][1],
                                           "MODELID": self.model_id(m), "CLINE_WRITE_PROMPT_ARTIFACTS": "1",
                                           "CLINE_PROMPT_ARTIFACT_DIR": "/logs/agent"})

    def test_goose(self) -> None:
        for key, m in provider_arms.MODELS.items():
            with self.subTest(model=key):
                _, env = self.run_agent(key, "goose")
                command, run_env = env.find("goose run ")
                run = ('export PATH="$HOME/.local/bin:$PATH" && goose run --recipe ~/runtime-recipe.yaml '
                       "--output-format stream-json --max-turns 300 2>&1 | stdbuf -oL tee /logs/agent/goose.txt")
                # Fireworks and xAI run as a custom provider declaring the effort (Harbor sees Goose's openai
                # client at the provider's host); Terra is OpenAI's own provider.
                host = {"fireworks": {"OPENAI_HOST": "https://api.fireworks.ai/inference"},
                        "xai": {"OPENAI_HOST": "https://api.x.ai"}}.get(m.provider, {})
                provider = {"fireworks": "bench_fireworks_ai", "xai": "bench_xai"}.get(m.provider, "openai")
                self.assertEqual(run_env, {**host, "OPENAI_API_KEY": KEY_OF[m.provider][1],
                                           "GOOSE_MODEL": self.model_id(m), "GOOSE_PROVIDER": provider,
                                           "XDG_DATA_HOME": "/logs/agent/goose/xdg-data",
                                           "XDG_STATE_HOME": "/logs/agent/goose/xdg-state",
                                           "GOOSE_THINKING_EFFORT": m.effort})
                if m.provider == "openai":
                    self.assertEqual(command, PREFIX + run)
                    continue
                write, _, rest = command.removeprefix(PREFIX).partition(" && printf '%s\\n' ")
                self.assertEqual(write, "mkdir -p ~/.config/goose/custom_providers")
                spec_text, _, rest = rest.partition(f" > ~/.config/goose/custom_providers/{provider}.json && ")
                self.assertEqual(rest, run)
                model: dict[str, Any] = {"name": self.model_id(m), "request_params": {"reasoning_effort": m.effort}}
                if m.provider == "fireworks":
                    model["context_limit"] = {"deepseek": 1000000, "glm": 1048573, "kimi": 1048576}[key]
                url, header, title = {"fireworks": (FIREWORKS_URL, "x-session-affinity", "Fireworks AI (bench)"),
                                      "xai": ("https://api.x.ai/v1", "x-grok-conv-id", "xAI (bench)")}[m.provider]
                self.assertEqual(json.loads(shlex.split(spec_text)[0]), {
                    "name": provider, "engine": "openai", "display_name": title,
                    "api_key_env": "OPENAI_API_KEY", "base_url": url + "/chat/completions", "models": [model],
                    "headers": {header: SESSION}, "supports_streaming": True, "requires_auth": True,
                    "dynamic_models": False})

    def test_goose_refuses_an_effort_the_provider_does_not_take(self) -> None:
        from bench.agents.goose import custom_provider

        with self.assertRaisesRegex(ValueError, "effort 'medium' is not one of low, high, max"):
            custom_provider("fireworks_ai", "accounts/fireworks/models/glm-5p3", "medium", SESSION)
        spec = custom_provider("xai", "grok-4.7", None, SESSION)
        self.assertEqual(spec["models"], [{"name": "grok-4.7"}])  # no effort: the provider's default

    def check_opencode_family(self, name: str, config_path: str, switches: str, extra_env: dict[str, str]) -> None:
        q = shlex.quote(INSTRUCTION)
        binary, output = ("opencode", "opencode.txt") if name == "opencode" else ("kilo", "kilo.txt")
        limits = {"deepseek": (1000000, 384000), "glm": (1048573, 262144), "kimi": (1048576, 131072)}
        for key, m in provider_arms.MODELS.items():
            with self.subTest(model=key):
                _, env = self.run_agent(key, name)
                command, run_env = env.find(f" {binary} ")
                launch = f"opencode --model={m.harbor} run" if binary == "opencode" else f"kilo run --model={m.harbor}"
                self.assertEqual(command, (
                    f"{PREFIX}[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; {launch} --format=json --variant {m.effort} "
                    f"{switches} -- {q} 2>&1 </dev/null | stdbuf -oL tee /logs/agent/{output}"))
                key_name, value = KEY_OF[m.provider]
                self.assertEqual(run_env, {key_name: value, **extra_env})
                config_command, _ = env.find(config_path)
                written = json.loads(shlex.split(config_command.split(" && ", 1)[1].rsplit(" > ", 1)[0])[1])
                if m.provider == "fireworks":
                    context, out = limits[key]
                    entry = {"npm": "@ai-sdk/openai-compatible", "name": "Fireworks AI",
                             "options": {"baseURL": FIREWORKS_URL, "apiKey": "{env:FIREWORKS_AI_API_KEY}"},
                             "models": {self.model_id(m): {
                                 "name": self.model_id(m), "reasoning": True, "tool_call": True,
                                 "limit": {"context": context, "output": out},
                                 # Declared: OpenCode computes none for glm and kimi ids.
                                 "variants": {"low": {"reasoningEffort": "low"},
                                              "high": {"reasoningEffort": "high"},
                                              "max": {"reasoningEffort": "max"}}}}}
                else:
                    entry = {"models": {m.model: {}}}  # OpenCode's own openai and xai providers
                self.assertEqual(written, {"provider": {m.harbor.partition("/")[0]: entry}})

    def test_opencode(self) -> None:
        self.check_opencode_family("opencode", "~/.config/opencode/opencode.json",
                                   "--thinking --dangerously-skip-permissions",
                                   {"OPENCODE_FAKE_VCS": "git", "XDG_DATA_HOME": "/logs/agent/opencode/xdg-data",
                                    "XDG_STATE_HOME": "/logs/agent/opencode/xdg-state"})

    def test_kilo(self) -> None:
        from bench.agents.kilo import KILO_ENV

        self.check_opencode_family("kilo", "~/.config/kilo/kilo.json", "--thinking --auto",
                                   {**KILO_ENV, "XDG_DATA_HOME": "/logs/agent/kilo/xdg-data",
                                    "XDG_STATE_HOME": "/logs/agent/kilo/xdg-state"})


class RoutesLeaveOtherArmsAlone(BenchTestCase):
    """The subclasses change nothing for Anthropic arms or for an arm that sets a route itself."""

    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.dict(os.environ, KEYS)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_anthropic_arms_are_harbors(self) -> None:
        from harbor.agents.installed.cline import ClineCli
        from harbor.agents.installed.goose import Goose
        from harbor.agents.installed.mini_swe_agent import MiniSweAgent
        from harbor.agents.installed.opencode import OpenCode
        from harbor.agents.installed.pi import Pi

        from bench.agents.cline import PinnedClineCli
        from bench.agents.goose import PinnedGoose
        from bench.agents.mini_swe_agent import MiniSweAgentWithMcp
        from bench.agents.opencode import OpenCodeBunEpipe
        from bench.agents.pi import PiProviders

        model = "anthropic/claude-opus-5-5"
        for ours, harbors in ((PiProviders, Pi), (PinnedClineCli, ClineCli), (PinnedGoose, Goose),
                              (OpenCodeBunEpipe, OpenCode), (MiniSweAgentWithMcp, MiniSweAgent)):
            with self.subTest(agent=ours.__name__):
                a = ours(logs_dir=self.tmpdir(), model_name=model)
                b = harbors(logs_dir=self.tmpdir(), model_name=model)
                self.assertEqual(a.model_connection, b.model_connection)
                self.assertEqual(a._parse_model_name(model), b._parse_model_name(model))
                self.assertEqual((a._base_url, a._model_api_format, a._model_api_key_name), (None, None, None))
                self.assertEqual(a.to_agent_info().model_info, b.to_agent_info().model_info)
        self.assertEqual(OpenCodeBunEpipe(logs_dir=self.tmpdir(), model_name=model)._opencode_config, {})

    def test_an_arms_own_route_wins(self) -> None:
        from bench.agents.goose import PinnedGoose
        from bench.agents.mini_swe_agent import MiniSweAgentWithMcp
        from bench.agents.opencode import OpenCodeBunEpipe
        from bench.agents.pi import PiProviders

        fw = "fireworks_ai/accounts/fireworks/models/glm-5p3"
        own = {"provider": {"fireworks_ai": {"options": {"baseURL": "https://example.test/v1"}}}}
        self.assertEqual(OpenCodeBunEpipe(logs_dir=self.tmpdir(), model_name=fw, opencode_config=own)._opencode_config,
                         own)
        goose = PinnedGoose(logs_dir=self.tmpdir(), model_name=fw, base_url="https://example.test")
        self.assertEqual((goose._base_url, goose._model_api_format, goose._model_api_key_name),
                         ("https://example.test", None, None))
        pi = PiProviders(logs_dir=self.tmpdir(), model_name="openai/gpt-6.1-terra",
                         base_url="https://example.test/v1", model_api_format="openai_chat_completions")
        self.assertEqual((pi._base_url, pi._model_api_format), ("https://example.test/v1", "openai_chat_completions"))
        mini = MiniSweAgentWithMcp(logs_dir=self.tmpdir(), model_name="openai/gpt-6.1-terra",
                                   config={"model": {"model_class": "litellm"}})
        self.assertIsNone(mini._model_api_format)
        # Only the models listed in bench/agents/providers.py get the Responses route.
        for agent in (PiProviders, MiniSweAgentWithMcp):
            other = agent(logs_dir=self.tmpdir(), model_name="openai/gpt-6.1-sol")
            self.assertEqual((other._base_url, other._model_api_format), (None, None))

    def test_unknown_fireworks_models_get_the_route_without_model_facts(self) -> None:
        from bench.agents.opencode import with_provider_routes

        config = with_provider_routes("fireworks_ai/accounts/fireworks/models/other", None)
        assert config is not None
        self.assertEqual(config["provider"]["fireworks_ai"]["models"], {"accounts/fireworks/models/other": {}})
        self.assertEqual(config["provider"]["fireworks_ai"]["options"]["baseURL"], FIREWORKS_URL)
        self.assertIsNone(with_provider_routes("xai/grok-4.7", None))


class NewProviderLogs(BenchTestCase):
    """The harness logs on these providers are formats extract.py already reads."""

    def test_mini_swe_agent_on_the_responses_client(self) -> None:
        # LitellmResponseModel (mini-swe-agent 2.4.6) saves response.model_dump() as the message itself.
        from bench import extract, pricing

        agent = self.tmpdir() / "agent"
        agent.mkdir()
        usage = {"input_tokens": 12000, "output_tokens": 900, "total_tokens": 12900,
                 "input_tokens_details": {"cached_tokens": 8000}, "output_tokens_details": {"reasoning_tokens": 600}}
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                    {"object": "response", "model": "gpt-6.1-terra", "output": [], "usage": usage,
                     "extra": {"actions": [], "cost": 0.0123}}]
        (agent / "mini-swe-agent.trajectory.json").write_text(json.dumps({"messages": messages}), encoding="utf-8")
        where, calls = extract.recover_calls(agent.parent)  # type: ignore[misc]
        self.assertEqual(where, "agent/mini-swe-agent.trajectory.json")
        self.assertEqual(calls, [(None, 12000, 8000, 900, 0.0123, None, None)])
        terra = self.load_shipped_config("bench").price("openai", "gpt-6.1-terra")
        assert terra is not None
        # Reasoning is inside output_tokens; no write counts, so the 4000 fresh tokens bill as input.
        self.assertAlmostEqual(pricing.unit_cost(terra, 12000, 8000, 900),
                               (4000 * 1.00 + 8000 * 0.05 + 900 * 5.00) / 1e6)
