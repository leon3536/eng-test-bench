"""Key handling: what a harbor child process inherits, how .env round-trips
values, retries on Windows sharing violations, and the assistant's key row.

Every value here is fake. Assertions compare variable names, never values,
so a failure message cannot print anything secret-looking.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

from tests.support import MINIMAL_YAML, BenchTestCase

from bench import keys, settings

OPENROUTER_ONLY = """\
dataset: terminal-bench@2.0
providers:
  openrouter: {env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}
modes:
  pilot: {n_tasks: 1}
run:
  budget_usd: 5
arms:
  - {name: pi-glm, agent: pi, provider: openrouter, model: z-ai/glm-5.3}
"""

# Names from review finding 32 (and its verdict) that the old suffix list let through.
LEAKED_BEFORE = (
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_PROFILE", "AWS_REGION", "GITHUB_TOKEN", "GH_TOKEN", "HF_TOKEN", "NPM_TOKEN", "OPENAI_KEY",
    "DB_PASSWORD", "GOOGLE_APPLICATION_CREDENTIALS", "DOCKER_AUTH_CONFIG", "CLAUDE_CODE_MESSAGING_TOKEN",
    "DATABRICKS_TOKEN",
)
ROUTING = ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_FORCE_OAUTH",
           "ANTHROPIC_VERTEX_PROJECT_ID", "CLOUD_ML_REGION")
ALWAYS_SCRUBBED = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN",
                   "FIREWORKS_AI_API_KEY", "OPENROUTER_API_KEY", "SOME_SECRET")
ENDPOINTS = ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE", "OPENAI_BASE_URL", "OPENAI_API_BASE",
             "OPENROUTER_BASE_URL")
# Ambient Claude Code settings that Harbor's adapter falls back to.
HARNESS_SELECTORS = ("CLAUDE_CODE_EFFORT_LEVEL", "CLAUDE_CODE_MAX_TURNS", "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
                     "CLAUDE_CODE_SUBAGENT_MODEL", "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                     "ANTHROPIC_DEFAULT_FABLE_MODEL", "ANTHROPIC_SMALL_FAST_MODEL",
                     "ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION", "DISABLE_PROMPT_CACHING",
                     "MAX_THINKING_TOKENS", "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING")


def fake_value(name: str) -> str:
    return f"fake-{name.lower()}-0000"


class KeysTestCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.env_file = self.tmpdir() / "keys.env"
        patcher = mock.patch.object(keys, "ENV_FILE", self.env_file)
        patcher.start()
        self.addCleanup(patcher.stop)

    def anthropic(self) -> settings.Provider:
        return settings.Provider(name="anthropic", env_key="ANTHROPIC_API_KEY", model_prefix="anthropic/")


class ChildEnvTests(KeysTestCase):
    def test_scrub_removes_credentials_and_routing(self) -> None:
        for name in LEAKED_BEFORE + ROUTING + ALWAYS_SCRUBBED:
            os.environ[name] = fake_value(name)
        env = keys.child_env(self.anthropic())
        names = set(env)
        for name in LEAKED_BEFORE + ROUTING + ALWAYS_SCRUBBED:
            with self.subTest(name=name):
                self.assertNotIn(name, names)
        self.assertIn("ANTHROPIC_API_KEY", names)
        self.assertEqual(env["ANTHROPIC_API_KEY"], os.environ["ANTHROPIC_API_KEY"])

    def test_only_the_provider_key_among_credentials(self) -> None:
        os.environ["FIREWORKS_AI_API_KEY"] = fake_value("fw")
        env = keys.child_env(self.anthropic())
        credential_like = sorted(n for n in env if keys._is_secret(n))
        self.assertEqual(credential_like, ["ANTHROPIC_API_KEY"])

    def test_no_provider_carries_no_key(self) -> None:
        env = keys.child_env(None)
        self.assertEqual(sorted(n for n in env if keys._is_secret(n)), [])

    def test_unset_provider_key_is_not_added(self) -> None:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        self.assertNotIn("ANTHROPIC_API_KEY", set(keys.child_env(self.anthropic())))

    def test_git_config_overrides_survive(self) -> None:
        os.environ.update({"GIT_CONFIG_COUNT": "2", "GIT_CONFIG_KEY_0": "core.autocrlf",
                           "GIT_CONFIG_VALUE_0": "false", "GIT_CONFIG_KEY_1": "core.longpaths",
                           "GIT_CONFIG_VALUE_1": "true"})
        env = keys.child_env(self.anthropic())
        self.assertEqual(env["GIT_CONFIG_COUNT"], "2")
        self.assertEqual(env["GIT_CONFIG_KEY_0"], "core.autocrlf")
        self.assertEqual(env["GIT_CONFIG_VALUE_1"], "true")

    def test_ordinary_variables_survive(self) -> None:
        os.environ["BENCH_MAX_TOKENS_HINT"] = "5"   # TOKENS is not TOKEN
        env = keys.child_env(None)
        self.assertIn("PATH", {n.upper() for n in env})
        self.assertEqual(env["BENCH_MAX_TOKENS_HINT"], "5")

    def test_endpoint_overrides_dropped(self) -> None:
        for name in ENDPOINTS:
            os.environ[name] = "http://127.0.0.1:9/gateway"
        env = keys.child_env(self.anthropic())
        for name in ENDPOINTS:
            with self.subTest(name=name):
                self.assertNotIn(name, set(env))

    def test_claude_code_selectors_dropped(self) -> None:
        for name in HARNESS_SELECTORS:
            os.environ[name] = "1"
        env = keys.child_env(self.anthropic())
        for name in HARNESS_SELECTORS:
            with self.subTest(name=name):
                self.assertNotIn(name, set(env))
                self.assertFalse(keys._is_secret(name))  # not handed to the redactor
        os.environ["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        self.assertIn("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", set(keys.child_env(None)))

    def test_dotenv_disabled_and_litellm_production(self) -> None:
        os.environ.pop("PYTHON_DOTENV_DISABLED", None)
        env = keys.child_env(None)
        self.assertEqual(env["PYTHON_DOTENV_DISABLED"], "1")
        self.assertEqual(env["LITELLM_MODE"], "PRODUCTION")

    def test_pythonpath_starts_with_project_root(self) -> None:
        os.environ.pop("PYTHONPATH", None)
        self.assertEqual(keys.child_env(None)["PYTHONPATH"], str(settings.ROOT))

        other = str(Path(self.tmpdir()))
        os.environ["PYTHONPATH"] = os.pathsep.join([other, str(settings.ROOT), ""])
        parts = keys.child_env(None)["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(parts, [str(settings.ROOT), other])

    def test_is_secret(self) -> None:
        for name in ("X_API_KEY", "my_token", "PGPASSWORD", "AZURE_CREDENTIAL", "SSH_AUTH_SOCK",
                     "AWS_DEFAULT_REGION", "cloud_ml_region", "ANTHROPIC_VERTEX_BASE"):
            with self.subTest(name=name):
                self.assertTrue(keys._is_secret(name))
        for name in ("PATH", "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_3", "GIT_CONFIG_VALUE_12",
                     "MAX_TOKENS", "PYTHONPATH", "COLUMNS", "ANTHROPIC_BASE_URL"):
            with self.subTest(name=name):
                self.assertFalse(keys._is_secret(name))


class EnvFileTests(KeysTestCase):
    def test_dollar_values_round_trip(self) -> None:
        cfg = self.load_test_config(MINIMAL_YAML)
        os.environ["BENCH_TEST_EXPANDS"] = "expanded"
        value = "sk-or-pre${BENCH_TEST_EXPANDS}post$PLAIN'q\\z"
        keys.set_secret(cfg, "OPENROUTER_API_KEY", value)
        self.assertTrue(keys._file_values()["OPENROUTER_API_KEY"] == value, "value changed on read-back")
        os.environ["OPENROUTER_API_KEY"] = "something-else-entirely"
        keys.load_into_process()
        self.assertTrue(os.environ["OPENROUTER_API_KEY"] == value, "value changed on reload")

    def test_set_secret_retries_permission_errors(self) -> None:
        cfg = self.load_test_config(MINIMAL_YAML)
        real = keys.set_key
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise PermissionError(13, "The process cannot access the file")
            return real(*args, **kwargs)

        with mock.patch.object(keys, "set_key", flaky), mock.patch.object(keys, "_RETRY_SECONDS", 0.01):
            keys.set_secret(cfg, "OPENROUTER_API_KEY", "sk-or-new-value-1234")
        self.assertEqual(calls["n"], 3)
        self.assertTrue(keys._file_values().get("OPENROUTER_API_KEY") == "sk-or-new-value-1234")
        self.assertTrue(os.environ["OPENROUTER_API_KEY"] == "sk-or-new-value-1234")

    def test_set_secret_gives_up_and_raises(self) -> None:
        cfg = self.load_test_config(MINIMAL_YAML)
        before = os.environ["OPENROUTER_API_KEY"]

        def locked(*args, **kwargs):
            raise PermissionError(13, "The process cannot access the file")

        with mock.patch.object(keys, "set_key", locked), mock.patch.object(keys, "_PATIENCE", 0.2), \
                mock.patch.object(keys, "_RETRY_SECONDS", 0.01):
            with self.assertRaises(PermissionError):
                keys.set_secret(cfg, "OPENROUTER_API_KEY", "sk-or-never-saved-1234")
        self.assertTrue(os.environ["OPENROUTER_API_KEY"] == before, "environment changed on a failed save")

    def test_clear_secret_retries_and_removes(self) -> None:
        cfg = self.load_test_config(MINIMAL_YAML)
        keys.set_secret(cfg, "OPENROUTER_API_KEY", "sk-or-to-remove-1234")
        real = keys.unset_key
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise PermissionError(13, "The process cannot access the file")
            return real(*args, **kwargs)

        with mock.patch.object(keys, "unset_key", flaky), mock.patch.object(keys, "_RETRY_SECONDS", 0.01):
            keys.clear_secret(cfg, "OPENROUTER_API_KEY")
        self.assertEqual(calls["n"], 2)
        self.assertNotIn("OPENROUTER_API_KEY", set(keys._file_values()))
        self.assertNotIn("OPENROUTER_API_KEY", set(os.environ))

    def test_clear_secret_removes_an_empty_line(self) -> None:
        cfg = self.load_test_config(MINIMAL_YAML)
        self.env_file.write_text("OPENROUTER_API_KEY=\nOTHER=1\n", encoding="utf-8")
        keys.clear_secret(cfg, "OPENROUTER_API_KEY")
        self.assertNotIn("OPENROUTER_API_KEY", set(keys._file_entries()))
        self.assertEqual(keys._file_values(), {"OTHER": "1"})

    def test_rejects_unknown_names_and_bad_values(self) -> None:
        cfg = self.load_test_config(MINIMAL_YAML)
        with self.assertRaises(ValueError):
            keys.set_secret(cfg, "PATH", "x")
        with self.assertRaises(ValueError):
            keys.set_secret(cfg, "OPENROUTER_API_KEY", "two\nlines")
        with self.assertRaises(ValueError):
            keys.clear_secret(cfg, "PATH")


class SecretValuesTests(KeysTestCase):
    def test_values_of_secret_names_only(self) -> None:
        # Start from no credential-like names at all, so the list can only hold
        # the fakes below, whatever the launching shell had set. BenchTestCase
        # restores the environment afterwards.
        for name in [n for n in os.environ if keys._is_secret(n)]:
            del os.environ[name]
        os.environ["HF_TOKEN"] = "hf-fake-token-value-123456"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-fake-secret-values-4321"
        os.environ["SHORT_SECRET"] = "abc"
        os.environ["BENCH_NOT_SENSITIVE"] = "an ordinary long value"
        os.environ["GIT_CONFIG_VALUE_0"] = "a-long-git-config-value"
        values = keys.secret_values()
        for present in ("hf-fake-token-value-123456", "sk-ant-fake-secret-values-4321"):
            self.assertTrue(present in values, "expected fake value missing")
        for absent in ("abc", "an ordinary long value", "a-long-git-config-value"):
            self.assertFalse(absent in values, "a value that should not be redacted was listed")
        self.assertEqual([len(v) for v in values], sorted((len(v) for v in values), reverse=True))


class AssistantKeyTests(KeysTestCase):
    def test_allowed_names(self) -> None:
        cfg = self.load_test_config(OPENROUTER_ONLY)
        self.assertEqual(keys.allowed_names(cfg), {"OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"})
        off = self.load_test_config(OPENROUTER_ONLY + "assistant: {enabled: false}\n")
        self.assertEqual(keys.allowed_names(off), {"OPENROUTER_API_KEY"})

    def test_assistant_key_can_be_saved_without_an_anthropic_provider(self) -> None:
        cfg = self.load_test_config(OPENROUTER_ONLY)
        keys.set_secret(cfg, "ANTHROPIC_API_KEY", "sk-ant-fake-assistant-key-9876")
        self.assertTrue(keys._file_values().get("ANTHROPIC_API_KEY") == "sk-ant-fake-assistant-key-9876")

    def test_status_row_for_assistant(self) -> None:
        cfg = self.load_test_config(OPENROUTER_ONLY)
        rows = keys.status(cfg)
        self.assertEqual([r["provider"] for r in rows], ["openrouter", "assistant"])
        row = rows[-1]
        self.assertEqual(row["env_key"], "ANTHROPIC_API_KEY")
        self.assertTrue(row["enabled"])
        self.assertTrue(row["set"])
        self.assertEqual(row["masked"], "…" + os.environ["ANTHROPIC_API_KEY"][-4:])
        self.assertEqual(row["source"], "shell")

        keys.set_secret(cfg, "ANTHROPIC_API_KEY", "sk-ant-fake-assistant-key-9876")
        self.assertEqual(keys.status(cfg)[-1]["source"], ".env")

    def test_no_assistant_row_when_a_provider_owns_the_key_or_disabled(self) -> None:
        with_anthropic = self.load_test_config(MINIMAL_YAML)
        self.assertEqual([r["provider"] for r in keys.status(with_anthropic)], ["anthropic", "openrouter"])
        off = self.load_test_config(OPENROUTER_ONLY + "assistant: {enabled: false}\n")
        self.assertEqual([r["provider"] for r in keys.status(off)], ["openrouter"])

    def test_assistant_key_never_reaches_other_providers(self) -> None:
        cfg = self.load_test_config(OPENROUTER_ONLY)
        env = keys.child_env(cfg.providers["openrouter"])
        self.assertEqual(sorted(n for n in env if keys._is_secret(n)), ["OPENROUTER_API_KEY"])


if __name__ == "__main__":
    import unittest

    unittest.main()
