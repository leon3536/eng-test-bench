"""The LiteLLM gateway: provider litellm and the five Claude Code arms on it.

The gateway's root URL lives in .env (LITELLM_URL, kept out of the repo), and each
arm picks a path on it with base_url_route. Plans, run.json and the config
hash carry ${LITELLM_URL}/route; only the harbor command a job starts holds
the address, filled in from the job's child env. These tests build every arm
as a run would, then Harbor's Claude Code adapter from that exact command
under that env, and check what Claude Code would get. Every value is fake,
nothing is started and no request is sent.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from tests.support import BenchTestCase, fixture_schema
from tests.test_agents_providers import Recorder

from bench import designs, estimates, extract, keys, results, runs, settings, spec
from bench.extract import RunContext
from bench.settings import ConfigError

KEY = "sk-fake-gateway-0000000000000000"
URL = "http://192.0.2.10:4000"
MOVED = "http://192.0.2.11:4000"
ROOT_ROUTE = "${LITELLM_URL}"
ANTHROPIC_ROUTE = "${LITELLM_URL}/anthropic"
# arm: (gateway model name, base_url as planned, effort)
ARMS = {
    "claude-code-gw-opus": ("claude-opus-5-5", ANTHROPIC_ROUTE, "medium"),
    "claude-code-gw-deepseek": ("deepseek-v4.1-flash", ROOT_ROUTE, "high"),
    "claude-code-gw-deepseek-tr": ("deepseek-v4.1-flash-translated", ROOT_ROUTE, "high"),
    "claude-code-gw-kimi": ("kimi-k3-fireworks", ROOT_ROUTE, "high"),
    "claude-code-gw-sol": ("gpt-6.1-sol", ROOT_ROUTE, "medium"),
}
IDENTITY = Path(__file__).parent / "fixtures" / "identity" / "stock" / "claude-code"

GATEWAY_YAML = """\
dataset: terminal-bench@2.0
providers:
  anthropic: {env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}
  litellm: {env_key: LITELLM_API_KEY, base_url_env: LITELLM_URL, model_prefix: litellm_proxy/}
modes:
  pilot: {n_tasks: 1}
run:
  budget_usd: 5
arms:
"""


class GatewayCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.dict(os.environ, {"LITELLM_API_KEY": KEY, "LITELLM_URL": URL})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cfg = self.load_shipped_config("bench")

    def arm(self, name: str) -> spec.ArmSpec:
        return spec.spec_from_config(self.cfg, "pilot", [name]).arms[0]

    def kwargs(self, arm: spec.ArmSpec) -> tuple[dict[str, Any], dict[str, str]]:
        return runs.agent_kwargs(self.cfg.run, arm, fixture_schema(arm.agent), self.cfg.providers[arm.provider])

    def command(self, name: str, env: dict[str, str] | None = None) -> list[str]:
        run_spec = spec.spec_from_config(self.cfg, "pilot", [name])
        arm = run_spec.arms[0]
        return runs.build_command(self.cfg, run_spec, arm, Path("jobs") / "r", self.kwargs(arm)[0],
                                  Path("datasets") / "x", env=env)


def _flags(cmd: list[str]) -> tuple[dict[str, str], list[str]]:
    single = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i] in ("--agent", "--model", "--api-key-name")}
    return single, [cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i] == "--ak"]


def _aes(cmd: list[str]) -> list[str]:
    return [cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i] == "--ae"]


# What gw-opus sets so its models match the direct claude-code arm's.
OPUS_ENV = {"ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-5-5", "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-5-5",
            "CLAUDE_CODE_SUBAGENT_MODEL": ""}


class ShippedConfigTests(GatewayCase):
    def test_the_provider_names_its_key_and_url_variables(self) -> None:
        provider = self.cfg.providers["litellm"]
        self.assertEqual((provider.env_key, provider.base_url_env, provider.model_prefix, provider.enabled),
                         ("LITELLM_API_KEY", "LITELLM_URL", "litellm_proxy/", True))

    def test_five_disabled_claude_code_arms_pinned_like_the_baseline(self) -> None:
        baseline = self.cfg.arm("claude-code")
        gateway = {a.name: a for a in self.cfg.arms if a.provider == "litellm"}
        self.assertEqual(sorted(gateway), sorted(ARMS))
        for name, (model, _, effort) in ARMS.items():
            arm = gateway[name]
            with self.subTest(arm=name):
                self.assertEqual((arm.agent, arm.family, arm.model, arm.enabled), ("claude-code", "claude-code",
                                                                                    model, False))
                self.assertEqual(arm.options["version"], baseline.options["version"])
                self.assertEqual(runs.resolve_effort(self.cfg.run, arm, fixture_schema(arm.agent))[0], effort)

    def test_gw_opus_keeps_the_direct_arms_models_and_dollar_cap(self) -> None:
        self.assertEqual(self.cfg.arm("claude-code-gw-opus").options["agent_env"], OPUS_ENV)
        direct, _ = self.kwargs(self.arm("claude-code"))
        gateway, source = self.kwargs(self.arm("claude-code-gw-opus"))
        self.assertEqual((gateway["max_budget_usd"], source["max_budget_usd"]), (direct["max_budget_usd"], "run cap"))
        for name in ("claude-code-gw-deepseek", "claude-code-gw-kimi", "claude-code-gw-sol"):
            with self.subTest(arm=name):
                # Claude Code would price these models at Claude rates.
                self.assertNotIn("max_budget_usd", self.kwargs(self.arm(name))[0])
                self.assertNotIn("agent_env", self.cfg.arm(name).options)

    def test_every_gateway_model_is_priced_at_its_underlying_list_price(self) -> None:
        underlying = {"claude-opus-5-5": ("anthropic", "claude-opus-5-5"),
                      "claude-sonnet-5-5": ("anthropic", "claude-sonnet-5-5"),
                      "claude-haiku-5-5": ("anthropic", "claude-haiku-5-5"),
                      "deepseek-v4.1-flash": ("fireworks", "deepseek-v4p1-flash-1017"),
                      "deepseek-v4.1-flash-translated": ("fireworks", "deepseek-v4p1-flash-1017"),
                      "kimi-k3-fireworks": ("fireworks", "kimi-k3"),
                      # The gateway name before 2026-11-04, kept for the smoke run's trials on it.
                      "kimi-k3": ("fireworks", "kimi-k3")}
        for model, (provider, row) in underlying.items():
            with self.subTest(model=model):
                self.assertEqual(self.cfg.price_exact("litellm", model), self.cfg.price_exact(provider, row))
        # OpenAI's pricing page, 2026-11-04.
        self.assertEqual(self.cfg.price_exact("litellm", "gpt-6.1-sol"),
                         settings.Price(input=2.0, output=10.0, cache_read=0.10, cache_write_5m=2.5,
                                        cache_write_1h=5.0))
        for name, (model, _, _) in ARMS.items():
            self.assertIsNotNone(self.cfg.price_exact("litellm", model), name)


class RouteRuleTests(BenchTestCase):
    def load(self, arm: str) -> settings.Config:
        return self.load_test_config(GATEWAY_YAML + f"  - {arm}\n")

    def test_a_route_needs_a_gateway_provider(self) -> None:
        with self.assertRaisesRegex(ConfigError, "base_url_route needs a gateway provider"):
            self.load("{name: a, agent: claude-code, provider: anthropic, model: claude-opus-5-5, "
                      "options: {base_url_route: /anthropic}}")

    def test_a_gateway_arm_sets_no_endpoint_of_its_own(self) -> None:
        with self.assertRaisesRegex(ConfigError, "set base_url_route, not base_url"):
            self.load("{name: a, agent: claude-code, provider: litellm, model: kimi-k3, "
                      "options: {base_url: 'http://elsewhere:1'}}")

    def test_a_route_is_a_plain_path(self) -> None:
        for route in ("@evil.example", "//evil.example", "/a/../b", "anthropic", "/a?x=1", "/a#b", "/a b", 5):
            with self.subTest(route=route), self.assertRaisesRegex(ConfigError, "plain path"):
                self.load("{name: a, agent: claude-code, provider: litellm, model: kimi-k3, "
                          f"options: {{base_url_route: {json.dumps(route)}}}}}")
        for route in ("/anthropic", "/", "", "/v1/x_y-z.1"):
            with self.subTest(route=route):
                self.load("{name: a, agent: claude-code, provider: litellm, model: kimi-k3, "
                          f"options: {{base_url_route: {json.dumps(route)}}}}}")

    def test_agent_env_sets_only_claude_codes_model_selectors(self) -> None:
        self.load("{name: a, agent: claude-code, provider: litellm, model: kimi-k3, "
                  "options: {agent_env: {ANTHROPIC_DEFAULT_HAIKU_MODEL: claude-haiku-5-5, "
                  "CLAUDE_CODE_SUBAGENT_MODEL: ''}}}")
        for options, problem in (("{agent_env: {ANTHROPIC_BASE_URL: 'http://x'}}", "can only set"),
                                 ("{agent_env: {LITELLM_API_KEY: x}}", "can only set"),
                                 ("{agent_env: [ANTHROPIC_DEFAULT_HAIKU_MODEL]}", "must map"),
                                 ("{agent_env: {ANTHROPIC_DEFAULT_HAIKU_MODEL: 'a b'}}", "model names or empty"),
                                 ("{agent_env: {ANTHROPIC_DEFAULT_HAIKU_MODEL: 5}}", "model names or empty")):
            with self.subTest(options=options), self.assertRaisesRegex(ConfigError, problem):
                self.load("{name: a, agent: claude-code, provider: litellm, model: kimi-k3, "
                          f"options: {options}}}")
        # designs.validate runs the same check, and the names pass its redirect and credential filter.
        with self.assertRaisesRegex(ConfigError, "can only set"):
            settings.check_agent_env("b", {"agent_env": {"HTTPS_PROXY": "x"}})
        self.assertFalse(designs._denied_options({"agent_env": OPUS_ENV}))

    def test_base_url_env_names_a_variable(self) -> None:
        with self.assertRaisesRegex(ConfigError, "base_url_env must name an environment variable"):
            self.load_test_config(GATEWAY_YAML.replace("LITELLM_URL", "'http://x'")
                                  + "  - {name: a, agent: claude-code, provider: anthropic, model: claude-opus-5-5}\n")

    def test_a_design_may_pick_a_route_but_not_an_endpoint(self) -> None:
        # designs.validate runs settings.check_route on each design arm; base_url stays denied there.
        cfg = self.load("{name: a, agent: claude-code, provider: litellm, model: kimi-k3}")
        with self.assertRaisesRegex(ConfigError, "plain path"):
            settings.check_route("b", {"base_url_route": "@evil.example"}, cfg.providers["litellm"])
        self.assertTrue(designs._denied_options({"base_url": "http://x"}))
        self.assertFalse(designs._denied_options({"base_url_route": "/anthropic"}))


class PlanTests(GatewayCase):
    def test_agent_kwargs_turn_the_route_into_a_templated_base_url(self) -> None:
        for name, (_, planned, _) in ARMS.items():
            with self.subTest(arm=name):
                kwargs, source = self.kwargs(self.arm(name))
                self.assertEqual((kwargs["base_url"], source["base_url"]), (planned, "arm option"))
                self.assertNotIn("base_url_route", kwargs)
        # Without the provider the route adds nothing.
        arm = self.arm("claude-code-gw-opus")
        kwargs, _ = runs.agent_kwargs(self.cfg.run, arm, fixture_schema(arm.agent))
        self.assertNotIn("base_url", kwargs)
        self.assertNotIn("base_url_route", kwargs)

    def test_the_preview_shows_the_variable_and_the_job_gets_the_address(self) -> None:
        for name, (model, planned, _) in ARMS.items():
            with self.subTest(arm=name):
                preview = self.command(name)
                single, aks = _flags(preview)
                self.assertEqual(single, {"--agent": "claude-code", "--model": f"litellm_proxy/{model}",
                                          "--api-key-name": "LITELLM_API_KEY"})
                self.assertIn(f"base_url={json.dumps(planned)}", aks)
                self.assertNotIn(URL, " ".join(preview))
                real = self.command(name, env=keys.child_env(self.cfg.providers["litellm"]))
                self.assertIn(f"base_url={json.dumps(planned.replace(ROOT_ROUTE, URL))}", _flags(real)[1])

    def test_agent_env_goes_to_harbor_as_ae_and_never_as_a_kwarg(self) -> None:
        cmd = self.command("claude-code-gw-opus", env=keys.child_env(self.cfg.providers["litellm"]))
        self.assertEqual(_aes(cmd), [f"{k}={v}" for k, v in sorted(OPUS_ENV.items())])
        self.assertFalse(any(a.startswith("agent_env=") for a in _flags(cmd)[1]))
        self.assertEqual(_aes(self.command("claude-code-gw-kimi")), [])
        # It is one of the arm's own options, so the fingerprint follows it.
        arm = self.arm("claude-code-gw-opus")
        kwargs, source = self.kwargs(arm)
        self.assertNotIn("agent_env", kwargs)
        self.assertNotEqual(runs.options_fingerprint(kwargs, source, runs.arm_agent_env(arm)),
                            runs.options_fingerprint(kwargs, source))

    def test_a_job_without_the_url_does_not_start(self) -> None:
        env = keys.child_env(self.cfg.providers["litellm"])
        env.pop("LITELLM_URL")
        with self.assertRaisesRegex(ConfigError, "LITELLM_URL is not set"):
            self.command("claude-code-gw-kimi", env=env)

    def test_other_providers_commands_are_unchanged(self) -> None:
        cmd = self.command("claude-code", env=keys.child_env(self.cfg.providers["anthropic"]))
        self.assertNotIn("--api-key-name", cmd)
        self.assertFalse(any(a.startswith("base_url=") for a in _flags(cmd)[1]))

    def test_the_plan_names_a_missing_or_bad_url_without_repeating_it(self) -> None:
        names = list(ARMS)
        run_spec = spec.spec_from_config(self.cfg, "pilot", names)
        plan = runs.plan_spec(self.cfg, run_spec, found=({}, None))
        self.assertFalse([p for p in plan["problems"] if "LITELLM" in p], plan["problems"])
        for entry in plan["arms"]:
            self.assertIn("base_url: Sends model traffic", " ".join(entry["warnings"]))
            self.assertIn("${LITELLM_URL}", entry["command"])
        for value, problem in (("", "LITELLM_URL is not set"),
                               ("ftp://192.0.2.10", "LITELLM_URL is not a plain http(s) URL"),
                               ("http://user:pw@192.0.2.10", "LITELLM_URL is not a plain http(s) URL")):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"LITELLM_URL": value}):
                plan = runs.plan_spec(self.cfg, run_spec, found=({}, None))
                found = [p for p in plan["problems"] if "LITELLM_URL" in p]
                self.assertEqual(len(found), len(names), plan["problems"])
                self.assertTrue(all(problem in p for p in found), found)
                if value:
                    self.assertNotIn(value, json.dumps(plan))

    def test_run_json_keeps_the_variable(self) -> None:
        run_spec = spec.spec_from_config(self.cfg, "pilot", ["claude-code-gw-opus"])
        resolved = runs.resolve_arms(self.cfg, run_spec, "r1", found=({}, None))[0]
        self.assertIn({"name": "base_url", "value": ANTHROPIC_ROUTE, "source": "arm option"}, resolved["kwargs"])
        self.assertEqual(resolved["agent_env"], OPUS_ENV)
        self.assertNotIn(URL, json.dumps(resolved))

    def test_the_arm_log_shows_the_variable_not_the_address(self) -> None:
        # RunManager._supervise writes each command it starts to the arm log, which the dashboard serves.
        env = keys.child_env(self.cfg.providers["litellm"])
        cmd = self.command("claude-code-gw-opus", env=env)
        self.assertIn(URL, " ".join(cmd))
        shown = keys.hide_gateway_urls(cmd, env)
        self.assertNotIn("192.0.2.", " ".join(shown))
        self.assertIn(f"base_url={json.dumps(ANTHROPIC_ROUTE)}", shown)

        manager = runs.RunManager.__new__(runs.RunManager)  # no lock file, no recovery
        manager._procs, manager._procs_lock, manager._cancel = set(), threading.Lock(), threading.Event()
        log_path = self.tmpdir() / "arm.log"
        arm_run = SimpleNamespace(pid=None, status="preflight", message="")
        fake = [sys.executable, "-c", "pass", f"base_url={URL}/anthropic"]
        with mock.patch.object(runs.procs, "assign_kill_on_close", lambda proc: True):
            rc = manager._supervise(fake, env, log_path, None, arm_run, watch=None)  # type: ignore[arg-type]
        self.assertEqual(rc, 0)
        text = log_path.read_text(encoding="utf-8")
        self.assertIn("base_url=${LITELLM_URL}/anthropic", text)
        self.assertNotIn("192.0.2.", text)

    def test_gateway_arms_share_the_baselines_pre_baked_image(self) -> None:
        from bench import prebake

        recipes = set()
        for name in ("claude-code", "claude-code-gw-opus", "claude-code-gw-kimi"):
            arm = self.arm(name)
            found = prebake.resolve(self.cfg, self.cfg.run, arm, self.kwargs(arm)[0])
            self.assertTrue(found.supported, found)
            recipes.add(found.recipe)
        self.assertEqual(len(recipes), 1)


class ChildEnvTests(GatewayCase):
    def test_the_gateway_key_and_url_come_back_and_no_anthropic_key(self) -> None:
        env = keys.child_env(self.cfg.providers["litellm"])
        self.assertEqual(env.get("LITELLM_API_KEY"), KEY)
        self.assertEqual(env.get("LITELLM_URL"), URL)
        self.assertTrue(os.environ.get("ANTHROPIC_API_KEY"))  # set in the test process ...
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
            self.assertNotIn(name, env)  # ... and never passed on

    def test_other_arms_and_helpers_never_get_the_gateway_url(self) -> None:
        self.assertEqual(os.environ.get("LITELLM_URL"), URL)
        for provider in (self.cfg.providers["anthropic"], self.cfg.providers["fireworks"], None):
            with self.subTest(provider=provider and provider.name):
                env = keys.child_env(provider)
                self.assertNotIn("LITELLM_URL", env)
                self.assertNotIn(URL, json.dumps(env))

    def test_a_url_variable_spelled_like_an_endpoint_is_put_back_for_its_provider_only(self) -> None:
        provider = settings.Provider(name="gw", env_key="GW_API_KEY", model_prefix="litellm_proxy/",
                                     base_url_env="GW_BASE_URL")
        with mock.patch.dict(os.environ, {"GW_BASE_URL": URL}):
            self.assertEqual(keys.child_env(provider).get("GW_BASE_URL"), URL)
            self.assertNotIn("GW_BASE_URL", keys.child_env(self.cfg.providers["anthropic"]))


class HarborAdapterTests(GatewayCase):
    """Harbor's Claude Code adapter, built from the job's own command under its child env."""

    def claude_env(self, name: str) -> tuple[dict[str, str], str]:
        from harbor.agents.factory import AgentFactory
        from harbor.cli.utils import parse_kwargs
        from harbor.models.trial.config import AgentConfig

        from harbor.cli.utils import parse_env_vars

        env = keys.child_env(self.cfg.providers["litellm"])
        cmd = self.command(name, env=env)
        single, aks = _flags(cmd)
        config = AgentConfig(name=single["--agent"], model_name=single["--model"], kwargs=parse_kwargs(aks),
                             model_api_key_name=single["--api-key-name"], env=parse_env_vars(_aes(cmd)))
        with mock.patch.dict(os.environ, env, clear=True):  # the harbor process's environment
            agent = AgentFactory.create_agent_from_config(config, logs_dir=self.tmpdir())
            recorder = Recorder()
            agent.session_id = "s"
            asyncio.run(agent.run("Fix it", recorder, SimpleNamespace(metadata={})))  # type: ignore[arg-type]
        command, run_env = recorder.find("claude --verbose")
        return run_env, command

    def test_claude_code_gets_the_route_the_bearer_key_and_the_model_name_unchanged(self) -> None:
        for name, (model, planned, effort) in ARMS.items():
            with self.subTest(arm=name):
                env, command = self.claude_env(name)
                self.assertEqual(env["ANTHROPIC_BASE_URL"], planned.replace(ROOT_ROUTE, URL))
                self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], KEY)
                self.assertEqual(env["ANTHROPIC_API_KEY"], "")
                self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "")
                self.assertEqual(env["ANTHROPIC_MODEL"], model)
                self.assertIn(f"--effort {effort} ", command)
                self.assertNotIn(os.environ["ANTHROPIC_API_KEY"], json.dumps(env))

    def test_opus_aliases_follow_the_direct_arm_and_the_others_stay_on_the_arms_model(self) -> None:
        env, _ = self.claude_env("claude-code-gw-opus")
        self.assertEqual({k: env.get(k) for k in OPUS_ENV}, OPUS_ENV)
        self.assertEqual(env["ANTHROPIC_DEFAULT_OPUS_MODEL"], "claude-opus-5-5")
        for name in ("claude-code-gw-deepseek", "claude-code-gw-kimi"):
            model = ARMS[name][0]
            with self.subTest(arm=name):
                env, _ = self.claude_env(name)
                # Harbor's own pinning keeps background and subagent calls off Claude models.
                for alias in ("ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                              "CLAUDE_CODE_SUBAGENT_MODEL"):
                    self.assertEqual(env[alias], model)


class ConfigHashTests(GatewayCase):
    def trial(self, base_url: str, model: str = "claude-opus-5-5") -> dict[str, Any]:
        result = json.loads((IDENTITY / "result.json").read_text(encoding="utf-8"))
        agent = result["config"]["agent"]
        agent["model_name"] = f"litellm_proxy/{model}"
        agent["kwargs"] = {**agent["kwargs"], "base_url": base_url}
        agent["env"] = dict(OPUS_ENV)  # where Harbor records gw-opus's --ae
        return result

    def facts(self, result: dict[str, Any], planned: str | None) -> extract.TrialFacts:
        ctx = RunContext(run_key="default/r1", profile="default", run_id="r1", source="bench",
                         arm="claude-code-gw-opus", provider="litellm", model_prefix="litellm_proxy/",
                         effort_label="medium", dataset="terminal-bench@2.0", local_snapshot=None,
                         family="claude-code", base_url=planned)
        return extract.facts_from_result(result, None, None, trial_dir="t", ctx=ctx,
                                         prefixes={"litellm_proxy/": "litellm", "anthropic/": "anthropic"})

    def test_moving_the_gateway_keeps_the_hash_and_the_route_changes_it(self) -> None:
        here = self.facts(self.trial(URL + "/anthropic"), ANTHROPIC_ROUTE)
        moved = self.facts(self.trial(MOVED + "/anthropic"), ANTHROPIC_ROUTE)
        root = self.facts(self.trial(URL), ROOT_ROUTE)
        self.assertEqual(here.config_hash, moved.config_hash)
        self.assertNotEqual(here.config_hash, root.config_hash)
        for facts in (here, moved, root):
            for text in (facts.config_json, facts.options_json, facts.config_label):
                self.assertNotIn("192.0.2.", text)
        self.assertIn('"base_url":"${LITELLM_URL}/anthropic"', here.config_json)
        self.assertIn("base_url", json.loads(here.cautions_json))

    def test_the_planner_predicts_the_hash(self) -> None:
        arm = self.arm("claude-code-gw-opus")
        kwargs, _ = self.kwargs(arm)
        predicted = estimates.config_hash_for(arm, kwargs, self.cfg.run.timeout_multiplier)
        trial = self.trial(URL + "/anthropic")
        self.assertEqual(predicted, self.facts(trial, ANTHROPIC_ROUTE).config_hash)
        # The models Claude Code picks for aliases and subagents are part of the configuration.
        trial["config"]["agent"].pop("env")
        self.assertNotEqual(predicted, self.facts(trial, ANTHROPIC_ROUTE).config_hash)

    def test_the_context_reads_the_planned_base_url_from_run_json(self) -> None:
        run_spec = spec.spec_from_config(self.cfg, "pilot", ["claude-code-gw-opus", "claude-code"])
        run = {"id": "r1", "resolved_arms": runs.resolve_arms(self.cfg, run_spec, "r1", found=({}, None))}
        self.assertEqual(extract.context_for_bench_arm(run, "claude-code-gw-opus", "default").base_url,
                         ANTHROPIC_ROUTE)
        self.assertIsNone(extract.context_for_bench_arm(run, "claude-code", "default").base_url)


# What LiteLLM 1.83.7's streams report in message_start: the upstream id, not the gateway name.
DS_UPSTREAM = "accounts/fireworks/models/deepseek-v4p1-flash"
KIMI_UPSTREAM = "accounts/fireworks/models/kimi-k3"
GW_PREFIXES = {"litellm_proxy/": "litellm", "anthropic/": "anthropic",
               "fireworks_ai/accounts/fireworks/models/": "fireworks"}


def _usage(n_in: int, n_cache: int, n_out: int, cost: float | None = None) -> dict[str, Any]:
    return {"n_input_tokens": n_in, "n_cache_tokens": n_cache, "n_output_tokens": n_out, "cost_usd": cost}


def _gw_result(model: str, model_usage: dict[str, dict[str, Any]], cost_usd: float) -> dict[str, Any]:
    """A finished gateway trial's result.json, its totals the per-model sums."""
    result = json.loads((IDENTITY / "result.json").read_text(encoding="utf-8"))
    agent = result["config"]["agent"]
    agent["model_name"] = f"litellm_proxy/{model}"
    agent["kwargs"] = {"reasoning_effort": "high", "max_turns": 300, "version": "2.1.312", "base_url": URL}
    result["agent_result"] = {
        **{k: sum(u[k] for u in model_usage.values()) for k in ("n_input_tokens", "n_cache_tokens",
                                                                "n_output_tokens")},
        "cost_usd": cost_usd, "model_usage": model_usage}
    return result


class UpstreamIdTests(GatewayCase):
    """A gateway model's usage under the upstream id its streams carry is the arm's own model's."""

    def ctx(self, arm: str, model: str) -> RunContext:
        return RunContext(run_key="default/r1", profile="default", run_id="r1", source="bench", arm=arm,
                          provider="litellm", model_prefix="litellm_proxy/", effort_label="high",
                          dataset="terminal-bench@2.0", local_snapshot=None, family="claude-code",
                          base_url=ROOT_ROUTE, upstream_model=extract.upstream_of("litellm", model))

    def facts(self, arm: str, model_usage: dict[str, dict[str, Any]], cost_usd: float = 0.78) -> extract.TrialFacts:
        model = ARMS[arm][0]
        return extract.facts_from_result(_gw_result(model, model_usage, cost_usd), None, None, trial_dir="t",
                                         ctx=self.ctx(arm, model), prefixes=GW_PREFIXES)

    def test_the_shipped_config_declares_each_streamed_id(self) -> None:
        declared = {"deepseek-v4.1-flash": DS_UPSTREAM, "deepseek-v4.1-flash-translated": DS_UPSTREAM,
                    "kimi-k3-fireworks": KIMI_UPSTREAM}
        self.assertEqual(self.cfg.providers["litellm"].upstream_models, declared)
        self.assertEqual(extract.provider_upstreams()["litellm"], declared)
        for model, upstream in declared.items():
            self.assertEqual(self.cfg.upstream_model("litellm", model), upstream)
        # Claude and Sol stream back their own names; other providers declare nothing.
        for model in ("claude-opus-5-5", "gpt-6.1-sol", "kimi-k3"):
            self.assertIsNone(self.cfg.upstream_model("litellm", model), model)
        self.assertIsNone(extract.upstream_of("fireworks", "deepseek-v4p1-flash"))

    def test_the_run_context_carries_the_arms_upstream_id(self) -> None:
        run_spec = spec.spec_from_config(self.cfg, "pilot", ["claude-code-gw-deepseek", "claude-code-gw-sol",
                                                             "claude-code"])
        run = {"id": "r1", "resolved_arms": runs.resolve_arms(self.cfg, run_spec, "r1", found=({}, None))}
        self.assertEqual(extract.context_for_bench_arm(run, "claude-code-gw-deepseek", "default").upstream_model,
                         DS_UPSTREAM)
        self.assertIsNone(extract.context_for_bench_arm(run, "claude-code-gw-sol", "default").upstream_model)
        self.assertIsNone(extract.context_for_bench_arm(run, "claude-code", "default").upstream_model)

    def test_usage_under_the_upstream_id_is_the_arms_model_and_priced(self) -> None:
        # The smoke run's gw-deepseek regex-log trial: every call streamed, so all under the upstream id.
        facts = self.facts("claude-code-gw-deepseek", {DS_UPSTREAM: _usage(171845, 138434, 21848)})
        usage = json.loads(facts.usage_json)
        self.assertEqual(list(usage), ["deepseek-v4.1-flash"])
        self.assertTrue(usage["deepseek-v4.1-flash"]["primary"])
        self.assertEqual(usage["deepseek-v4.1-flash"]["upstream_id"], DS_UPSTREAM)
        self.assertEqual(facts.primary_share, 1.0)
        expected = ((171845 - 138434) * 0.30 + 138434 * 0.006 + 21848 * 1.20) / 1e6
        self.assertAlmostEqual(facts.cost_at_ingest_usd, expected)

    def test_the_two_names_of_the_translated_arm_make_one_entry(self) -> None:
        facts = self.facts("claude-code-gw-deepseek-tr", {"deepseek-v4.1-flash-translated": _usage(33961, 33572, 13372),
                                                          DS_UPSTREAM: _usage(3728, 0, 641)})
        usage = json.loads(facts.usage_json)
        self.assertEqual(list(usage), ["deepseek-v4.1-flash-translated"])
        entry = usage["deepseek-v4.1-flash-translated"]
        self.assertEqual((entry["n_input"], entry["n_cache"], entry["n_output"]), (33961 + 3728, 33572, 13372 + 641))
        self.assertEqual(facts.primary_share, 1.0)
        self.assertIsNotNone(facts.cost_at_ingest_usd)

    def test_the_merged_entry_is_the_same_whichever_name_comes_first(self) -> None:
        # The stored result is sorted, so a re-read sees the upstream id before the gateway name.
        named = {"deepseek-v4.1-flash-translated": _usage(33961, 33572, 13372), DS_UPSTREAM: _usage(3728, 0, 641)}
        first = self.facts("claude-code-gw-deepseek-tr", named)
        reread = self.facts("claude-code-gw-deepseek-tr", dict(reversed(list(named.items()))))
        self.assertEqual(first.usage_json, reread.usage_json)
        entry = json.loads(first.usage_json)["deepseek-v4.1-flash-translated"]
        self.assertEqual((entry["reported_as"], entry["upstream_id"]), ("deepseek-v4.1-flash-translated", DS_UPSTREAM))

    def test_a_ledger_reread_keeps_the_config_hash_and_the_variable(self) -> None:
        from dataclasses import asdict
        from bench.ledger import Ledger
        model = ARMS["claude-code-gw-deepseek"][0]
        facts = self.facts("claude-code-gw-deepseek", {DS_UPSTREAM: _usage(1000, 0, 100)})
        led = Ledger.__new__(Ledger)
        led.root = Path(__file__).parent  # trial_dir "t" is no folder there: the stored usage is kept
        row = {**asdict(facts), "parser_version": 0}
        self.assertEqual(row["harbor_model"], f"litellm_proxy/{model}")
        values = led._rederived(row, GW_PREFIXES)
        self.assertEqual(values["config_hash"], facts.config_hash)
        for column in ("config_json", "options_json", "config_label"):
            self.assertEqual(values[column], getattr(facts, column), column)
            self.assertNotIn("192.0.2.", values[column])
        self.assertIn('"base_url":"${LITELLM_URL}"', values["config_json"])
        self.assertEqual(values["usage_json"], facts.usage_json)

    def test_another_models_tokens_are_still_another_model(self) -> None:
        for other in ("claude-haiku-5-5", KIMI_UPSTREAM, "moonshotai/kimi-k3"):
            with self.subTest(other=other):
                facts = self.facts("claude-code-gw-deepseek", {DS_UPSTREAM: _usage(1000, 0, 100),
                                                               other: _usage(1000, 0, 100)})
                usage = json.loads(facts.usage_json)
                self.assertTrue(usage["deepseek-v4.1-flash"]["primary"])
                self.assertFalse(usage[other]["primary"])
                self.assertNotIn("upstream_id", usage[other])
                self.assertEqual(facts.primary_share, 0.5)
        # Kimi's upstream id is Kimi's own on its arm, and DeepSeek's is not.
        facts = self.facts("claude-code-gw-kimi", {KIMI_UPSTREAM: _usage(1000, 0, 100),
                                                   DS_UPSTREAM: _usage(1000, 0, 100)})
        usage = json.loads(facts.usage_json)
        self.assertTrue(usage["kimi-k3-fireworks"]["primary"])
        self.assertFalse(usage[DS_UPSTREAM]["primary"])

    def test_an_arm_without_a_declared_id_is_unchanged(self) -> None:
        # Sol declares none, so a stray upstream id stays its own entry.
        facts = self.facts("claude-code-gw-sol", {"gpt-6.1-sol": _usage(1000, 0, 100), DS_UPSTREAM: _usage(10, 0, 1)})
        self.assertEqual(sorted(json.loads(facts.usage_json)), [DS_UPSTREAM, "gpt-6.1-sol"])

    def test_a_logged_call_under_the_upstream_id_gives_its_cache_writes_to_the_arms_model(self) -> None:
        # The OpenRouter fallback's name ends like Kimi's upstream id, so a name match alone would pick it.
        usage: dict[str, dict[str, Any]] = {
            "kimi-k3-fireworks": {"n_input": 1000, "n_cache": 0, "n_output": 10, "primary": True},
            "moonshotai/kimi-k3": {"n_input": 500, "n_cache": 0, "n_output": 10, "primary": False}}
        calls = [(KIMI_UPSTREAM, 1000, 0, 10, None, 200, 0), ("moonshotai/kimi-k3", 500, 0, 10, None, 50, 0)]
        extract.assign_cache_writes(usage, calls, "litellm", "kimi-k3-fireworks", self.cfg.pricing,
                                    upstream=KIMI_UPSTREAM)
        self.assertEqual(usage["kimi-k3-fireworks"]["n_cache_write_5m"], 200)
        self.assertEqual(usage["moonshotai/kimi-k3"]["n_cache_write_5m"], 50)

    def test_canonical_model_needs_the_arms_model_to_join_the_names(self) -> None:
        ctx = self.ctx("claude-code-gw-deepseek", "deepseek-v4.1-flash")
        self.assertEqual(extract.canonical_model(DS_UPSTREAM, ctx, GW_PREFIXES), DS_UPSTREAM)
        self.assertEqual(extract.canonical_model(DS_UPSTREAM, ctx, GW_PREFIXES, "deepseek-v4.1-flash"),
                         "deepseek-v4.1-flash")
        self.assertEqual(extract.canonical_model(DS_UPSTREAM + "-turbo", ctx, GW_PREFIXES, "deepseek-v4.1-flash"),
                         DS_UPSTREAM + "-turbo")  # a different model, not a date tag
        self.assertEqual(extract.canonical_model("litellm_proxy/deepseek-v4.1-flash", ctx, GW_PREFIXES,
                                                 "deepseek-v4.1-flash"), "deepseek-v4.1-flash")

    def test_upstream_models_must_map_names_to_ids(self) -> None:
        for bad in ("[a, b]", "{deepseek-v4.1-flash: ''}", "{deepseek-v4.1-flash: 5}"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ConfigError, "upstream_models"):
                self.load_test_config(GATEWAY_YAML.replace(
                    "model_prefix: litellm_proxy/}", f"model_prefix: litellm_proxy/, upstream_models: {bad}}}")
                    + "  - {name: a, agent: claude-code, provider: anthropic, model: claude-opus-5-5}\n")


class GatewaySpendTests(GatewayCase):
    """The run budget's figure for a gateway trial (results.spend), read from the trial folder as a run does."""

    def row(self, arm_name: str, model_usage: dict[str, dict[str, Any]], cost_usd: float) -> dict[str, Any]:
        arm = self.cfg.arm(arm_name)
        path = self.tmpdir() / "trial" / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_gw_result(arm.model, model_usage, cost_usd)), encoding="utf-8")
        row = results.read_trial(path, self.cfg, arm)
        assert row is not None
        return row

    def test_streamed_deepseek_is_priced_at_list_price_not_claude_codes_figure(self) -> None:
        # Claude Code priced this trial at $0.78 (Claude rates); Fireworks' list price is under 4 cents.
        row = self.row("claude-code-gw-deepseek", {DS_UPSTREAM: _usage(171845, 138434, 21848)}, 0.782472)
        expected = ((171845 - 138434) * 0.30 + 138434 * 0.006 + 21848 * 1.20) / 1e6
        self.assertEqual((row["models"], row["unpriced_models"], row["other_model_tokens"]),
                         (["deepseek-v4.1-flash"], [], 0))
        self.assertTrue(row["reported_mispriced"])
        self.assertAlmostEqual(row["computed_cost"], expected)
        self.assertAlmostEqual(results.spend(row), expected)

    def test_the_translated_arms_mix_is_one_model(self) -> None:
        row = self.row("claude-code-gw-deepseek-tr", {"deepseek-v4.1-flash-translated": _usage(33961, 33572, 13372),
                                                      DS_UPSTREAM: _usage(3728, 0, 641)}, 0.387696)
        self.assertEqual(row["models"], ["deepseek-v4.1-flash-translated"])
        self.assertAlmostEqual(results.spend(row), row["computed_cost"])
        self.assertLess(results.spend(row), 0.05)

    def test_an_unpriced_models_tokens_keep_the_budget_conservative(self) -> None:
        # Tokens our table cannot price: the budget keeps the harness's higher figure, as on any arm.
        row = self.row("claude-code-gw-deepseek", {DS_UPSTREAM: _usage(1000, 0, 100),
                                                   "zai/glm-9": _usage(1000, 0, 100)}, 0.5)
        self.assertEqual(row["unpriced_models"], ["zai/glm-9"])
        self.assertEqual(row["other_model_tokens"], 1100)
        self.assertAlmostEqual(results.spend(row), 0.5)

    def test_claude_on_the_gateway_still_takes_the_higher_figure(self) -> None:
        row = self.row("claude-code-gw-opus", {"claude-opus-5-5": _usage(100_000, 80_000, 2_000)}, 1.0)
        self.assertFalse(row["reported_mispriced"])
        self.assertAlmostEqual(results.spend(row), 1.0)
