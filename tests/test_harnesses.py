"""Harness facts: import-path detection and which caps a harness misprices."""

from __future__ import annotations

from tests.support import BenchTestCase

from bench import harnesses


class ImportPathTests(BenchTestCase):
    def test_import_paths(self) -> None:
        for agent in ("bench.agents.mini_swe_agent:MiniSweAgentWithMcp", "agent:create_agent",
                      "pkg.mod:Class", "_private.mod:_Cls"):
            with self.subTest(agent=agent):
                self.assertTrue(harnesses.is_import_path(agent))

    def test_registered_names_and_junk(self) -> None:
        for agent in ("claude-code", "terminus-2", "mini-swe-agent", "pi", "strands", "",
                      "foo:bar-baz", "a.b:c.d", ":Class", "mod:", "my-pkg.mod:Class", "mod: Class",
                      "1mod:Class", None, 42):
            with self.subTest(agent=agent):
                self.assertFalse(harnesses.is_import_path(agent))  # type: ignore[arg-type]


class CapTests(BenchTestCase):
    def test_cap_names(self) -> None:
        self.assertEqual(harnesses.DOLLAR_CAPS, ("max_budget_usd", "cost_limit"))
        self.assertEqual(harnesses.TURN_CAPS, ("max_turns",))
        self.assertIsInstance(harnesses.MISPRICED_CAP, str)
        self.assertIs(harnesses._CAP_PRICED_FOR, harnesses.CAP_PRICED_FOR)

    def test_claude_code_budget_cap_off_anthropic(self) -> None:
        self.assertTrue(harnesses.cap_misprices_for("claude-code", "openrouter", "max_budget_usd"))
        self.assertTrue(harnesses.cap_misprices_for("claude-code", "fireworks", "max_budget_usd"))
        self.assertFalse(harnesses.cap_misprices_for("claude-code", "anthropic", "max_budget_usd"))

    def test_other_caps_and_harnesses(self) -> None:
        self.assertFalse(harnesses.cap_misprices_for("claude-code", "openrouter", "max_turns"))
        self.assertFalse(harnesses.cap_misprices_for("mini-swe-agent", "openrouter", "cost_limit"))
        self.assertFalse(harnesses.cap_misprices_for("pi", "openrouter", "max_budget_usd"))

    def test_reported_cost(self) -> None:
        self.assertTrue(harnesses.reports_mispriced_cost("claude-code", "openrouter"))
        self.assertFalse(harnesses.reports_mispriced_cost("claude-code", "anthropic"))
        self.assertFalse(harnesses.reports_mispriced_cost("pi", "openrouter"))
        self.assertFalse(harnesses.reports_mispriced_cost("mini-swe-agent", "anthropic"))


if __name__ == "__main__":
    import unittest

    unittest.main()
