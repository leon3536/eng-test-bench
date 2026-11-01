"""Checks for the Strands harness factory in harnesses/strands_factory.

The factory runs inside the task container, where Harbor installs strands-agents
and strands-harness; the bench venv has neither. So these tests read the files
(AST, TOML, YAML, bytes) and load the factory against a stub `strands_harness`.
A mistake here only shows up in a paid trial otherwise: Harbor's runner refuses
a factory that cannot bind (sandbox, model=None, **factory_kwargs), uv cannot
resolve a wrong pin, and Harbor refuses symbolic links in the folder.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import logging
import os
import sys
import tomllib
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
FACTORY_DIR = ROOT / "harnesses" / "strands_factory"
PACKAGE = "bench_strands_factory"
INIT_PY = FACTORY_DIR / PACKAGE / "__init__.py"
FACTORY_REF = f"{PACKAGE}:create_agent"  # the `factory` option arms pass to Harbor
STRANDS_HARNESS_PIN = "strands-harness==0.3.2"
MIN_STRANDS_AGENTS = (1, 57, 2)  # strands-harness 0.3.2 requires strands-agents>=1.57.2


def _source() -> str:
    return INIT_PY.read_text(encoding="utf-8")


def _create_agent_node() -> ast.FunctionDef:
    tree = ast.parse(_source())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "create_agent":
            return node
    raise AssertionError("bench_strands_factory has no top-level create_agent")


def _signature_stub() -> Any:
    """create_agent's signature on an empty body, without importing strands."""
    node = _create_agent_node()
    args = copy.deepcopy(node.args)
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
        if arg is not None:
            arg.annotation = None
    stub = ast.FunctionDef(
        name="create_agent",
        args=args,
        body=[ast.Return(ast.Constant(None))],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[]))
    namespace: dict[str, Any] = {}
    exec(compile(module, "<create_agent signature>", "exec"), namespace)
    return namespace["create_agent"]


def _load_factory() -> tuple[types.ModuleType, list[dict[str, Any]]]:
    """Import the factory with a stub strands_harness that records create_harness calls."""
    calls: list[dict[str, Any]] = []

    def create_harness(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    stub = types.ModuleType("strands_harness")
    stub.create_harness = create_harness  # type: ignore[attr-defined]
    spec = importlib.util.spec_from_file_location(PACKAGE, INIT_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # No __pycache__ inside the folder Harbor uploads.
    with mock.patch.dict(sys.modules, {"strands_harness": stub}), \
            mock.patch.object(sys, "dont_write_bytecode", True):
        spec.loader.exec_module(module)
    return module, calls


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(text).split("."))


class SignatureTests(unittest.TestCase):
    def test_binds_the_runner_call(self) -> None:
        signature = inspect.signature(_signature_stub())
        sandbox, model = object(), object()
        # The runner's own check: inspect.signature(factory).bind(sandbox, **factory_kwargs).
        signature.bind(sandbox)
        signature.bind(sandbox, model=model)
        signature.bind(sandbox, model=model, model_spec="anthropic/claude-opus-5-5",
                       effort="medium", builtin_tools={"subagent": False})

    def test_parameters_match_the_contract(self) -> None:
        params = inspect.signature(_signature_stub()).parameters
        positional = [p for p in params.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        self.assertEqual([p.name for p in positional], ["sandbox", "model"])
        self.assertIs(params["model"].default, None)
        self.assertEqual(params["model_spec"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(params["model_spec"].default, None)
        self.assertEqual(params["effort"].default, "auto")
        self.assertTrue(any(p.kind == p.VAR_KEYWORD for p in params.values()))

    def test_runtime_imports_exist_in_the_container(self) -> None:
        # Only the stdlib and strands-harness (which brings strands) are installed there.
        allowed = {"__future__", "logging", "os", "typing", "strands_harness"}
        tree = ast.parse(_source())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0], allowed)


class _FactoryCase(unittest.TestCase):
    """Runs each test without the runner's HARBOR_MODEL unless the test sets it."""

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("HARBOR_MODEL", None)


class BehaviourTests(_FactoryCase):
    def test_model_spec_passes_sandbox_and_effort(self) -> None:
        module, calls = _load_factory()
        sandbox, harbor_model = object(), object()
        module.create_agent(sandbox, model=harbor_model, model_spec="anthropic/claude-opus-5-5",
                            effort="medium", builtin_tools={"subagent": False})
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["model"], "anthropic/claude-opus-5-5")
        self.assertEqual(call["effort"], "medium")
        self.assertIs(call["sandbox"], sandbox)
        self.assertEqual(call["builtin_tools"], {"subagent": False})

    def test_state_stays_out_of_the_task_directory(self) -> None:
        module, calls = _load_factory()
        module.create_agent(object(), model_spec="anthropic/claude-opus-5-5")
        call = calls[0]
        self.assertTrue(module.STATE_DIR.startswith("/tmp/"))
        self.assertTrue(call["session"]["dir"].startswith(module.STATE_DIR))
        self.assertTrue(call["memory"]["dir"].startswith(module.STATE_DIR))

    def test_explicit_session_and_memory_win(self) -> None:
        module, calls = _load_factory()
        module.create_agent(object(), model_spec="anthropic/claude-opus-5-5",
                            session=False, memory={"dir": "/tmp/elsewhere"})
        self.assertIs(calls[0]["session"], False)
        self.assertEqual(calls[0]["memory"], {"dir": "/tmp/elsewhere"})

    def test_prebuilt_model_runs_at_auto_and_warns_once(self) -> None:
        module, calls = _load_factory()
        sandbox, harbor_model = object(), object()
        with self.assertLogs(PACKAGE, level=logging.WARNING) as logs:
            module.create_agent(sandbox, model=harbor_model, effort="medium")
        self.assertEqual(len(logs.records), 1)
        self.assertIn("effort", logs.output[0])
        call = calls[0]
        self.assertIs(call["model"], harbor_model)
        self.assertEqual(call["effort"], "auto")
        self.assertIs(call["sandbox"], sandbox)

    def test_prebuilt_model_at_auto_does_not_warn(self) -> None:
        module, calls = _load_factory()
        with self.assertNoLogs(PACKAGE, level=logging.WARNING):
            module.create_agent(object(), model=object())
        self.assertEqual(calls[0]["effort"], "auto")

    def test_yaml_off_and_null_effort(self) -> None:
        module, calls = _load_factory()
        module.create_agent(object(), model_spec="anthropic/claude-opus-5-5", effort=False)
        module.create_agent(object(), model_spec="anthropic/claude-opus-5-5", effort=None)
        self.assertEqual([c["effort"] for c in calls], ["off", "auto"])

    def test_string_model_is_treated_as_a_spec(self) -> None:
        module, calls = _load_factory()
        module.create_agent(object(), model="anthropic/claude-opus-5-5", effort="high")
        self.assertEqual(calls[0]["model"], "anthropic/claude-opus-5-5")
        self.assertEqual(calls[0]["effort"], "high")


class ModelMatchTests(_FactoryCase):
    """model_spec must be the arm's own model, which Harbor, the ledger and the budget price."""

    def test_mismatch_raises_before_building(self) -> None:
        module, calls = _load_factory()
        with mock.patch.dict(os.environ, {"HARBOR_MODEL": "anthropic/claude-sonnet-5"}):
            with self.assertRaises(ValueError) as ctx:
                module.create_agent(object(), model=object(), model_spec="anthropic/claude-opus-5-5")
        self.assertIn("anthropic/claude-sonnet-5", str(ctx.exception))
        self.assertIn("anthropic/claude-opus-5-5", str(ctx.exception))
        self.assertEqual(calls, [])

    def test_string_model_mismatch_raises(self) -> None:
        module, calls = _load_factory()
        with mock.patch.dict(os.environ, {"HARBOR_MODEL": "anthropic/claude-sonnet-5"}):
            with self.assertRaises(ValueError):
                module.create_agent(object(), model="anthropic/claude-opus-5-5")
        self.assertEqual(calls, [])

    def test_match_passes(self) -> None:
        module, calls = _load_factory()
        with mock.patch.dict(os.environ, {"HARBOR_MODEL": " Anthropic/Claude-Opus-5-5 "}):
            module.create_agent(object(), model=object(), model_spec="anthropic/claude-opus-5-5",
                                effort="medium")
        self.assertEqual(calls[0]["model"], "anthropic/claude-opus-5-5")
        self.assertEqual(calls[0]["effort"], "medium")

    def test_unset_passes(self) -> None:
        module, calls = _load_factory()
        module.create_agent(object(), model_spec="anthropic/claude-opus-5-5")
        self.assertEqual(calls[0]["model"], "anthropic/claude-opus-5-5")

    def test_prebuilt_model_is_not_checked(self) -> None:
        module, calls = _load_factory()
        harbor_model = object()
        with mock.patch.dict(os.environ, {"HARBOR_MODEL": "openrouter/deepseek/deepseek-v4"}):
            module.create_agent(object(), model=harbor_model)
        self.assertIs(calls[0]["model"], harbor_model)


class _Built:
    """Stands in for a strands Model the runner built: its config and, for OpenAIModel, client_args."""

    def __init__(self, params: dict[str, Any] | None = None, client_args: dict[str, Any] | None = None) -> None:
        self.config = {"model_id": "m", **({"params": params} if params is not None else {})}
        self.client_args = {} if client_args is None else client_args

    def get_config(self) -> dict[str, Any]:
        return self.config


class OpenAIModel(_Built):
    pass


class LiteLLMModel(_Built):
    pass


class PrebuiltModelTests(_FactoryCase):
    """Fireworks and xAI arms use the model Harbor built, with their effort in model_kwargs.params."""

    def create(self, harbor_model: str, model: Any) -> list[dict[str, Any]]:
        module, calls = _load_factory()
        with mock.patch.dict(os.environ, {"HARBOR_MODEL": harbor_model}):
            module.create_agent(object(), model=model)
        return calls

    def test_fireworks_and_xai_models_must_carry_an_effort(self) -> None:
        for harbor_model, cls in (("fireworks_ai/accounts/fireworks/models/glm-5p3", LiteLLMModel),
                                  ("xai/grok-4.7", OpenAIModel)):
            for params in (None, {}, {"max_tokens": 10}, {"reasoning_effort": ""}):
                with self.subTest(model=harbor_model, params=params), self.assertRaises(ValueError) as ctx:
                    self.create(harbor_model, cls(params))
                self.assertIn("model_kwargs.params.reasoning_effort", str(ctx.exception))
            with self.subTest(model=harbor_model):
                model = cls({"reasoning_effort": "high"})
                calls = self.create(harbor_model, model)
                self.assertIs(calls[0]["model"], model)
                self.assertEqual(calls[0]["effort"], "auto")

    def test_an_xai_openai_model_gets_xais_endpoint(self) -> None:
        model = OpenAIModel({"reasoning_effort": "medium"}, {"api_key": "xai-test"})
        self.create("xai/grok-4.7", model)
        self.assertEqual(model.client_args, {"api_key": "xai-test", "base_url": "https://api.x.ai/v1"})
        kept = OpenAIModel({"reasoning_effort": "medium"}, {"base_url": "https://example.test/v1"})
        self.create("xai/grok-4.7", kept)
        self.assertEqual(kept.client_args, {"base_url": "https://example.test/v1"})

    def test_other_providers_are_left_alone(self) -> None:
        for harbor_model in ("openrouter/deepseek/deepseek-v4", "openai/gpt-6.1-sol", "anthropic/claude-opus-5-5"):
            with self.subTest(model=harbor_model):
                model = OpenAIModel()
                self.create(harbor_model, model)
                self.assertEqual(model.client_args, {})
        fireworks = LiteLLMModel({"reasoning_effort": "high"})
        self.create("fireworks_ai/accounts/fireworks/models/kimi-k3", fireworks)
        self.assertEqual(fireworks.client_args, {})  # LiteLLM reaches Fireworks itself


class ProjectTests(unittest.TestCase):
    def test_pyproject_pins_strands_harness(self) -> None:
        with open(FACTORY_DIR / "pyproject.toml", "rb") as fh:
            data = tomllib.load(fh)
        project = data["project"]
        self.assertEqual(project["name"], "bench-strands-factory")
        self.assertEqual(project["dependencies"], [STRANDS_HARNESS_PIN])
        self.assertEqual(project["requires-python"], ">=3.10")
        self.assertEqual(data["build-system"]["build-backend"], "setuptools.build_meta")
        self.assertEqual(data["tool"]["setuptools"]["packages"], [PACKAGE])

    def test_files_are_lf_and_not_links(self) -> None:
        self.assertFalse(FACTORY_DIR.is_symlink())
        seen = 0
        for directory, dirnames, filenames in os.walk(FACTORY_DIR, followlinks=False):
            dirnames[:] = [d for d in dirnames if d != "__pycache__" and not d.endswith(".egg-info")]
            for name in [*dirnames, *filenames]:
                path = Path(directory, name)
                self.assertFalse(path.is_symlink(), f"{path} is a link; Harbor refuses links")
                self.assertFalse(name.startswith(".env"), f"{path} must not be in the factory folder")
            for name in filenames:
                seen += 1
                self.assertNotIn(b"\r", Path(directory, name).read_bytes(),
                                 f"{name} has CR bytes; the container expects LF")
        self.assertGreaterEqual(seen, 3)

    def test_package_dir_matches_the_factory_module(self) -> None:
        module, _, symbol = FACTORY_REF.partition(":")
        self.assertEqual(module, PACKAGE)
        self.assertTrue((FACTORY_DIR / module / "__init__.py").is_file())
        self.assertEqual(_create_agent_node().name, symbol)


class ConfiguredArmsTests(unittest.TestCase):
    """The strands arms in config/*.yaml point at this folder with a usable version."""

    def _strands_arms(self, name: str) -> list[dict[str, Any]]:
        raw = yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8")) or {}
        return [arm for arm in raw.get("arms") or [] if isinstance(arm, dict) and arm.get("agent") == "strands"]

    def test_configured_factories_resolve(self) -> None:
        wired = 0
        for name in ("bench.yaml", "dev.yaml"):
            arms = self._strands_arms(name)
            with_factory = [a for a in arms if (a.get("options") or {}).get("factory")]
            if not with_factory:
                continue
            self.assertEqual(len(with_factory), len(arms), f"{name}: some strands arms have no factory")
            for arm in with_factory:
                wired += 1
                options = arm["options"]
                label = f"{name}:{arm.get('name')}"
                self.assertEqual(options["factory"], FACTORY_REF, label)
                factory_path = ROOT / options["factory_path"]
                self.assertEqual(factory_path.resolve(), FACTORY_DIR.resolve(), label)
                module = options["factory"].partition(":")[0]
                self.assertTrue((factory_path / module / "__init__.py").is_file(), label)
                self.assertIn("version", options, f"{label}: Harbor's default strands-agents is too old")
                self.assertGreaterEqual(_version_tuple(options["version"]), MIN_STRANDS_AGENTS, label)
        if not wired:
            self.skipTest("No strands arm in config/bench.yaml or config/dev.yaml sets a factory yet")


if __name__ == "__main__":
    unittest.main()
