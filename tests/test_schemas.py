"""Harness schemas: the scrubbed environment for `harbor agents schema`,
option coercion, family-based cap mispricing and schema warming."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest import mock

from tests.support import MINIMAL_YAML, BenchTestCase, completed, fixture_schema, program_name

from bench import harnesses, schemas, settings

# BenchTestCase replaces agent_schema with the fixtures in setUp; keep the
# real one to test how it calls Harbor (through a fake subprocess).
REAL_AGENT_SCHEMA = schemas.agent_schema

MINI_PATH = "bench.agents.mini_swe_agent:MiniSweAgentWithMcp"


def is_schema_call(argv: list[str]) -> bool:
    return program_name(argv) == "harbor" and argv[1:3] == ["agents", "schema"]


class AgentSchemaTests(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.calls: list[tuple[list[str], dict]] = []

    def fake(self, returncode: int = 0, stdout: str = '{"properties": {}}', stderr: str = ""):
        def run(argv: list[str], **kwargs):
            self.calls.append((argv, kwargs))
            return completed(argv, stdout=stdout, stderr=stderr, returncode=returncode)
        return run

    def forget(self, agent: str) -> None:
        schemas._cache.pop(agent, None)
        self.addCleanup(schemas._cache.pop, agent, None)

    def test_import_path_schema_runs_with_child_env(self) -> None:
        self.forget(MINI_PATH)
        self.allow_subprocess(is_schema_call, self.fake(stdout=json.dumps(fixture_schema(MINI_PATH))))
        schema = REAL_AGENT_SCHEMA(MINI_PATH)
        self.assertIn("cost_limit", schema["properties"])
        argv, kwargs = self.calls[0]
        self.assertEqual(argv[1:], ["agents", "schema", MINI_PATH, "--json"])
        env = kwargs["env"]
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(settings.ROOT))
        self.assertEqual(env["PYTHON_DOTENV_DISABLED"], "1")
        names = set(env)
        for name in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
            self.assertNotIn(name, names)
        self.assertEqual(kwargs["cwd"], settings.ROOT)

    def test_cached_after_first_call(self) -> None:
        self.forget("pi")
        self.allow_subprocess(is_schema_call, self.fake(stdout=json.dumps(fixture_schema("pi"))))
        REAL_AGENT_SCHEMA("pi")
        REAL_AGENT_SCHEMA("pi")
        self.assertEqual(len(self.calls), 1)

    def test_failure_raises_with_stderr_tail(self) -> None:
        self.forget("no-such-agent")
        self.allow_subprocess(is_schema_call, self.fake(returncode=1, stdout="", stderr="Unknown agent"))
        with self.assertRaisesRegex(RuntimeError, "Unknown agent"):
            REAL_AGENT_SCHEMA("no-such-agent")
        self.assertNotIn("no-such-agent", schemas._cache)


class CoerceTests(BenchTestCase):
    def test_llm_backend_is_an_open_enum(self) -> None:
        schema = fixture_schema("terminus-2")
        prop = schema["properties"]["llm_backend"]
        self.assertEqual(schemas.field_kind(prop, schema["$defs"]), ("enum", ["litellm", "tinker"]))
        self.assertEqual(schemas.field_kind(prop), ("json", None))
        self.assertEqual(schemas.coerce(schema, "llm_backend", "tinker"), "tinker")
        self.assertEqual(schemas.coerce(schema, "llm_backend", "litellm"), "litellm")
        self.assertEqual(schemas.coerce(schema, "llm_backend", "my-backend"), "my-backend")

    def test_describe_shows_llm_backend_choices(self) -> None:
        fields = {f["name"]: f for f in schemas.describe("terminus-2")["fields"]}
        self.assertEqual(fields["llm_backend"]["kind"], "enum")
        self.assertEqual(fields["llm_backend"]["enum"], ["litellm", "tinker"])
        self.assertEqual(fields["trajectory_config"]["kind"], "json")

    def test_closed_enum_still_rejects(self) -> None:
        schema = fixture_schema("claude-code")
        with self.assertRaises(ValueError):
            schemas.coerce(schema, "permission_mode", "bogus")
        self.assertEqual(schemas.coerce(schema, "permission_mode", "plan"), "plan")

    def test_json_strings_pass_through(self) -> None:
        schema = fixture_schema("claude-code")
        self.assertEqual(schemas.field_kind(schema["properties"]["config"])[0], "json")
        for value in ("/work/claude-config.json", "tinker", "5", "", "{not json", "  [broken"):
            with self.subTest(value=value):
                self.assertEqual(schemas.coerce(schema, "config", value), value)

    def test_inline_json_strings_decode(self) -> None:
        schema = fixture_schema("strands")
        self.assertEqual(schemas.coerce(schema, "factory_kwargs", '{"model_spec": "anthropic/x", "n": 2}'),
                         {"model_spec": "anthropic/x", "n": 2})
        self.assertEqual(schemas.coerce(fixture_schema("claude-code"), "config", ' [1, 2]'), [1, 2])

    def test_native_values_unchanged(self) -> None:
        schema = fixture_schema("strands")
        value = {"effort": "medium"}
        self.assertEqual(schemas.coerce(schema, "factory_kwargs", value), value)
        self.assertIsNone(schemas.coerce(schema, "factory_kwargs", None))

    def test_scalar_kinds(self) -> None:
        schema = fixture_schema("claude-code")
        self.assertEqual(schemas.coerce(schema, "max_budget_usd", 1.5), "1.5")
        self.assertEqual(schemas.coerce(schema, "max_turns", "150"), 150)
        with self.assertRaises(ValueError):
            schemas.coerce(schema, "no_such_option", 1)

    def test_effort_levels(self) -> None:
        self.assertEqual(schemas.effort_levels(fixture_schema("claude-code")),
                         ["low", "medium", "high", "xhigh", "max"])
        self.assertIsNone(schemas.effort_levels(fixture_schema("strands")))
        self.assertEqual(schemas.effort_levels(fixture_schema("mini-swe-agent")), [])


class CapTests(BenchTestCase):
    def test_reexports_come_from_harnesses(self) -> None:
        self.assertIs(schemas.DOLLAR_CAPS, harnesses.DOLLAR_CAPS)
        self.assertIs(schemas.TURN_CAPS, harnesses.TURN_CAPS)
        self.assertIs(schemas.MISPRICED_CAP, harnesses.MISPRICED_CAP)

    def test_family_decides_for_import_path_arm(self) -> None:
        arm = settings.Arm(name="cc-sub", agent="bench.agents.cc:ClaudeCodeVariant", provider="openrouter",
                           model="z-ai/glm-5.3", family="claude-code")
        self.assertTrue(schemas.cap_misprices(arm, "max_budget_usd"))
        self.assertFalse(schemas.cap_misprices(arm, "max_turns"))
        on_anthropic = settings.Arm(name="cc-sub", agent=arm.agent, provider="anthropic",
                                    model="claude-opus-5-5", family="claude-code")
        self.assertFalse(schemas.cap_misprices(on_anthropic, "max_budget_usd"))

    def test_agent_is_the_family_when_none_given(self) -> None:
        arm = settings.Arm(name="cc", agent="claude-code", provider="openrouter", model="z-ai/glm-5.3")
        self.assertTrue(schemas.cap_misprices(arm, "max_budget_usd"))
        other = settings.Arm(name="x", agent="bench.agents.cc:ClaudeCodeVariant", provider="openrouter",
                             model="z-ai/glm-5.3")
        self.assertFalse(schemas.cap_misprices(other, "max_budget_usd"))

    def test_arm_spec_shaped_objects(self) -> None:
        spec_arm = SimpleNamespace(agent="claude-code", family="claude-code", provider="fireworks")
        self.assertTrue(schemas.cap_misprices(spec_arm, "max_budget_usd"))


class WarmTests(BenchTestCase):
    YAML = MINIMAL_YAML + """\
  - name: mini
    agent: bench.agents.mini_swe_agent:MiniSweAgentWithMcp
    family: mini-swe-agent
    provider: anthropic
    model: claude-opus-5-5
designs:
  import_paths:
    pkg.harness:Agent: my-family
  extra_harnesses: [openhands, terminus-2]
"""

    def test_warm_covers_arms_import_paths_and_extra_harnesses(self) -> None:
        cfg = self.load_test_config(self.YAML)
        seen: list[str] = []

        def fake(agent: str) -> dict:
            seen.append(agent)
            if agent == "openhands":
                raise RuntimeError("no such harness here")
            return {}

        with mock.patch.object(schemas, "agent_schema", fake):
            schemas.warm(cfg)
        self.assertEqual(sorted(seen), sorted({"claude-code", MINI_PATH, "pkg.harness:Agent",
                                               "openhands", "terminus-2"}))
        self.assertEqual(len(seen), len(set(seen)))


if __name__ == "__main__":
    import unittest

    unittest.main()
