"""Config loading: profiles, merging of dashboard overrides, validation of
every section, price lookup and the rollback in write_overrides."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

import yaml

from tests.support import MINIMAL_YAML, BenchTestCase

from bench import settings
from bench.settings import ConfigError

BASE = MINIMAL_YAML


def with_run(**fields) -> str:
    """MINIMAL_YAML with its run section replaced."""
    run = {"budget_usd": 10, **fields}
    lines = "".join(f"  {k}: {v}\n" for k, v in run.items())
    return BASE.replace("run:\n  budget_usd: 10\n", f"run:\n{lines}")


def import_settings(profile: str | None, keep_data_dir: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(settings.ROOT)
    if profile is None:
        env.pop("BENCH_PROFILE", None)
    else:
        env["BENCH_PROFILE"] = profile
    if not keep_data_dir:
        env.pop("BENCH_DATA_DIR", None)
    code = ("import bench.settings as s; "
            "print(s.PROFILE_NAME, s.BASE_CONFIG.name, s.OVERRIDES.name, s.DATA_DIR, sep='|')")
    return subprocess.run([sys.executable, "-c", code], cwd=settings.ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=120)


class ProfileTests(BenchTestCase):
    def test_default_alias_loads_bench_yaml(self) -> None:
        proc = import_settings("default")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        name, base, overrides, _ = proc.stdout.strip().split("|")
        self.assertEqual((name, base, overrides), ("default", "bench.yaml", "overrides.yaml"))

    def test_missing_profile_exits_with_message(self) -> None:
        proc = import_settings("nosuch")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("No profile 'nosuch': config/nosuch.yaml does not exist", proc.stderr)

    def test_reserved_names_rejected(self) -> None:
        for name in ("pricing", "reference", "overrides", "tasks", "dev.overrides"):
            with self.subTest(name=name):
                proc = import_settings(name)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(f"Invalid profile name '{name}'", proc.stderr)

    def test_invalid_characters_rejected(self) -> None:
        proc = import_settings("Dev")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Invalid profile name", proc.stderr)

    def test_dev_profile_paths(self) -> None:
        proc = import_settings("dev", keep_data_dir=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        name, base, overrides, data_dir = proc.stdout.strip().split("|")
        self.assertEqual((name, base, overrides), ("dev", "dev.yaml", "dev.overrides.yaml"))
        self.assertEqual(pathlib.Path(data_dir), settings.ROOT / "profiles" / "dev")

    def test_test_process_paths_are_isolated(self) -> None:
        self.assertEqual(settings.PROFILE, "")
        self.assertEqual(settings.PROFILE_NAME, "default")
        self.assertEqual(settings.DATA_DIR, pathlib.Path(os.environ["BENCH_DATA_DIR"]))
        self.assertEqual(settings.LEDGER_DB, pathlib.Path(os.environ["BENCH_LEDGER"]))
        self.assertEqual(settings.RUNS_DIR, settings.DATA_DIR / "runs")
        self.assertEqual(settings.JOBS_DIR, settings.DATA_DIR / "jobs")
        self.assertEqual(settings.DESIGNS_DIR, settings.DATA_DIR / "designs")
        self.assertEqual(settings.ASSISTANT_DIR, settings.DATA_DIR / "assistant")
        self.assertEqual(settings.QUEUE_FILE, settings.DATA_DIR / "queue.json")
        self.assertNotEqual(settings.ENV_FILE, settings.ROOT / ".env")


class ShippedConfigTests(BenchTestCase):
    MSA = "bench.agents.mini_swe_agent:MiniSweAgentWithMcp"
    FACTORY = {"factory": "bench_strands_factory:create_agent",
               "factory_path": "harnesses/strands_factory", "version": "1.57.2"}

    def check_arms(self, cfg: settings.Config) -> None:
        self.assertTrue(cfg.arms)
        for arm in cfg.arms:
            with self.subTest(arm=arm.name):
                self.assertTrue(arm.family)
                if arm.name.startswith("mini-swe-agent"):
                    self.assertEqual((arm.agent, arm.family, arm.kind), (self.MSA, "mini-swe-agent", "import_path"))
                elif arm.name.startswith("strands"):
                    self.assertEqual((arm.agent, arm.family, arm.effort), ("strands", "strands", "omit"))
                    self.assertEqual({k: arm.options[k] for k in self.FACTORY}, self.FACTORY)
                    self.assertIsInstance(arm.options["factory_kwargs"], dict)
                else:
                    self.assertEqual((arm.family, arm.kind), (arm.agent, "name"))
        self.assertEqual(cfg.designs.import_paths, {self.MSA: "mini-swe-agent"})
        self.assertIn(cfg.baseline_arm, [a.name for a in cfg.arms])

    def test_bench_yaml(self) -> None:
        cfg = self.load_shipped_config("bench")
        self.check_arms(cfg)
        self.assertEqual(cfg.arm("strands").options["factory_kwargs"],
                         {"model_spec": "anthropic/claude-opus-5-5", "effort": "medium"})
        self.assertEqual(cfg.designs.max_budget_usd, 150)
        self.assertEqual(cfg.assistant, settings.AssistantSettings())
        self.assertEqual(cfg.stats, settings.StatsSettings())
        self.assertEqual(cfg.queue, settings.QueueSettings())
        self.assertEqual(cfg.ledger, settings.LedgerSettings())
        self.assertEqual(cfg.report["group_by"], "family")
        self.assertEqual(cfg.report["misconfigured_cap"], "exclude")
        self.assertIs(cfg.report["exclusions"], True)
        self.assertNotIn("ci_level", cfg.report)

    def test_dev_yaml(self) -> None:
        cfg = self.load_shipped_config("dev")
        self.check_arms(cfg)
        self.assertEqual(len(cfg.arms), 10)
        self.assertEqual(cfg.arm("strands-glm").options["factory_kwargs"], {})
        self.assertEqual(cfg.designs.max_budget_usd, 10)
        self.assertEqual(cfg.assistant.daily_cap_usd, 5.0)
        self.assertIs(cfg.report["exclusions"], True)

    def test_prices(self) -> None:
        dev = self.load_shipped_config("dev")
        self.assertIsNone(dev.price("openrouter", "z-ai/glm-5.3-flash"))
        self.assertEqual(dev.price("openrouter", "openrouter/z-ai/glm-5.3"), dev.pricing["openrouter"]["glm-5.3"])
        cfg = self.load_shipped_config("bench")
        haiku = cfg.pricing["anthropic"]["claude-haiku-5-5"]
        self.assertEqual(cfg.price("anthropic", "claude-haiku-5-5-20261001"), haiku)
        self.assertEqual(cfg.price("anthropic", "anthropic/claude-opus-5-5"), cfg.pricing["anthropic"]["claude-opus-5-5"])
        self.assertEqual(cfg.price("anthropic", "claude-opus-5"), cfg.pricing["anthropic"]["claude-opus-5"])
        self.assertIsNone(cfg.price("anthropic", "claude-opus-5-5-fast"))
        self.assertIn("claude-opus-4-8", cfg.pricing["anthropic"])
        self.assertIsNone(cfg.price_exact("anthropic", "claude-haiku-5-5-20261001"))
        self.assertEqual(cfg.price_exact("anthropic", "anthropic/claude-haiku-5-5"), haiku)

    def test_reference_defaults(self) -> None:
        ref = self.load_shipped_config("bench").reference
        self.assertEqual(ref["defaults"]["direction"], "baseline_over_harness")
        self.assertEqual(ref["defaults"]["estimator"], "ratio_of_means")
        self.assertEqual(ref["defaults"]["applies_to"], ["anthropic:claude-*"])
        self.assertEqual((ref["consensus"]["cost_per_trial"]["low"], ref["consensus"]["cost_per_trial"]["high"]), (2.0, 3.0))
        self.assertEqual((ref["consensus"]["pass_rate_delta_pp"]["low"], ref["consensus"]["pass_rate_delta_pp"]["high"]), (-5, 5))
        self.assertIn("Terminal-Bench 2.1", ref["arms"]["strands"]["note"])


class VariantTests(BenchTestCase):
    def test_is_variant(self) -> None:
        cases = [
            ("glm-5.3", "glm-5.3", True),
            ("glm-5.3-flash", "glm-5.3", False),
            ("claude-opus-5-5", "claude-opus-5", False),
            ("claude-opus-5", "claude-opus-5-5", False),
            ("claude-haiku-5-5-20261001", "claude-haiku-5-5", True),
            ("claude-haiku-5-5-2026-10-01", "claude-haiku-5-5", True),
            ("claude-haiku-5-5-latest", "claude-haiku-5-5", True),
            ("claude-haiku-5-5@20261001", "claude-haiku-5-5", True),
            ("deepseek-v4.1-flash:free", "deepseek-v4.1-flash", True),
            ("claude-opus-5-5-fast", "claude-opus-5-5", False),
            ("claude-haiku-5-5-2026100", "claude-haiku-5-5", False),
        ]
        for name, key, expected in cases:
            with self.subTest(name=name, key=key):
                self.assertIs(settings.is_variant(name, key), expected)

    def test_longest_key_wins(self) -> None:
        cfg = self.load_test_config(BASE, pricing="""
            anthropic:
              claude-opus-5:   {input: 5, output: 25}
              claude-opus-5-5: {input: 4, output: 20}
        """)
        self.assertEqual(cfg.price("anthropic", "claude-opus-5-5-20261001").input, 4)
        self.assertEqual(cfg.price("anthropic", "claude-opus-5-20261001").input, 5)
        self.assertIsNone(cfg.price("openrouter", "claude-opus-5"))


class PricingParseTests(BenchTestCase):
    def test_parse(self) -> None:
        table = settings.parse_pricing({"a": {"m": {"input": 1, "output": "2"}}, "b": None})
        self.assertEqual(table, {"a": {"m": settings.Price(1.0, 2.0, None)}, "b": {}})

    def test_bad_rows(self) -> None:
        for raw in ({"a": {"m": {"input": 1}}}, {"a": {"m": {"input": -1, "output": 1}}},
                    {"a": {"m": {"input": float("nan"), "output": 1}}}, {"a": {"m": 3}}, {"a": [1]},
                    {"a": {"m": {"input": 1, "output": 1, "cache_read": "x"}}}):
            with self.subTest(raw=raw):
                with self.assertRaises(ConfigError):
                    settings.parse_pricing(raw)

    def test_read_pricing_real_file(self) -> None:
        table = settings.read_pricing(settings.PRICING)
        self.assertEqual(table["anthropic"]["claude-opus-5-5"], settings.Price(4.0, 20.0, 0.2))


class EffortTests(BenchTestCase):
    def test_unquoted_off_on_arm_raises(self) -> None:
        text = BASE.replace("model: claude-opus-5-5}", "model: claude-opus-5-5, effort: off}")
        with self.assertRaises(ConfigError) as ctx:
            self.load_test_config(text)
        self.assertIn("quote", str(ctx.exception))
        self.assertIn("'off'", str(ctx.exception))
        self.assertIn("arms.claude-code.effort", str(ctx.exception))

    def test_unquoted_on_run_raises(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.load_test_config(with_run(effort="no"))
        self.assertIn("run.effort", str(ctx.exception))

    def test_quoted_off_is_kept(self) -> None:
        text = BASE.replace("model: claude-opus-5-5}", "model: claude-opus-5-5, effort: 'off'}")
        self.assertEqual(self.load_test_config(text).arm("claude-code").effort, "off")

    def test_defaults(self) -> None:
        cfg = self.load_test_config(BASE)
        self.assertEqual(cfg.run.effort, settings.RUN_DEFAULTS["effort"])
        self.assertEqual(cfg.arm("claude-code").effort, settings.ARM_DEFAULTS["effort"])
        self.assertEqual(self.load_test_config(with_run(effort="0")).run.effort, "0")


class NumberTests(BenchTestCase):
    def test_non_finite_budgets_raise(self) -> None:
        for value in (".nan", ".inf", "-.inf", "'inf'", "'nan'", "0", "-3", "abc", "true"):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError):
                    self.load_test_config(with_run(budget_usd=value))

    def test_infinite_int_is_a_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            self.load_test_config(with_run(n_concurrent=".inf"))

    def test_huge_int_is_a_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            self.load_test_config(with_run(n_concurrent="9" * 400))

    def test_run_defaults_fill_missing_fields(self) -> None:
        run = self.load_test_config(BASE).run
        for name, value in settings.RUN_DEFAULTS.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(run, name), value)
        self.assertEqual(self.load_test_config(BASE).modes["pilot"].n_attempts, settings.MODE_DEFAULTS["n_attempts"])


class MergeTests(BenchTestCase):
    ARM_WITH_OPTIONS = BASE + (
        "  - name: t2\n"
        "    agent: terminus-2\n"
        "    provider: anthropic\n"
        "    model: claude-opus-5-5\n"
        "    options:\n"
        "      llm_call_kwargs: {top_k: 5, extra_headers: {x: 1}}\n"
        "      interleaved_thinking: true\n"
    )

    def test_tombstone_without_base_options_never_leaks(self) -> None:
        cfg = self.load_test_config(BASE, overrides="""
            arms:
              claude-code:
                options: {interleaved_thinking: null, max_turns: 20}
        """)
        self.assertEqual(cfg.arm("claude-code").options, {"max_turns": 20})

    def test_nested_tombstones_dropped_when_base_lacks_key(self) -> None:
        cfg = self.load_test_config(BASE, overrides="""
            stats: {default_trial_tokens: {output: null, input: 1000}}
        """)
        self.assertEqual(cfg.stats.default_trial_tokens["input"], 1000)
        self.assertEqual(cfg.stats.default_trial_tokens["output"], 60000)

    def test_object_option_replaced_whole(self) -> None:
        cfg = self.load_test_config(self.ARM_WITH_OPTIONS, overrides="""
            arms:
              t2:
                options:
                  llm_call_kwargs: {top_k: 5, stop: null}
        """)
        self.assertEqual(cfg.arm("t2").options["llm_call_kwargs"], {"top_k": 5, "stop": None})
        self.assertTrue(cfg.arm("t2").options["interleaved_thinking"])

    def test_option_tombstone_removes_base_option(self) -> None:
        cfg = self.load_test_config(self.ARM_WITH_OPTIONS, overrides="""
            arms: {t2: {options: {interleaved_thinking: null}}}
        """)
        self.assertNotIn("interleaved_thinking", cfg.arm("t2").options)
        self.assertIn("llm_call_kwargs", cfg.arm("t2").options)

    def test_null_option_in_base_yaml_is_kept(self) -> None:
        # An explicit null in YAML lifts the run's cap for that arm (runs.agent_kwargs).
        text = BASE.replace("model: claude-opus-5-5}", "model: claude-opus-5-5, options: {max_budget_usd: null}}")
        self.assertEqual(self.load_test_config(text).arm("claude-code").options, {"max_budget_usd": None})

    def test_ghost_mode_suppressed(self) -> None:
        text = BASE.replace("pilot: {n_tasks: 1}", "pilot: {n_tasks: 1}\n  block1: {n_tasks: 2}")
        cfg = self.load_test_config(text, overrides="""
            modes:
              sample: {n_attempts: 2}
              block1: {n_tasks: 3}
        """)
        self.assertEqual(set(cfg.modes), {"pilot", "block1"})
        self.assertEqual(cfg.modes["block1"].n_tasks, 3)

    def test_mode_tombstone_still_removes(self) -> None:
        text = BASE.replace("pilot: {n_tasks: 1}", "pilot: {n_tasks: 1}\n  block1: {n_tasks: 2}")
        cfg = self.load_test_config(text, overrides="modes: {block1: null}\n")
        self.assertEqual(set(cfg.modes), {"pilot"})

    def test_non_mapping_arm_override_is_config_error(self) -> None:
        for value in ("[1, 2]", "x", "3", "[]", "0", "false", "''"):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError) as ctx:
                    self.load_test_config(BASE, overrides=f"arms: {{claude-code: {value}}}\n")
                self.assertIn("Overrides for arm 'claude-code' must be a mapping", str(ctx.exception))

    def test_override_cannot_rename_an_arm(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.load_test_config(BASE, overrides="arms: {claude-code: {name: other}}\n")
        self.assertIn("cannot rename it", str(ctx.exception))
        cfg = self.load_test_config(BASE, overrides="arms: {claude-code: {name: claude-code}}\n")
        self.assertIn("claude-code", [a.name for a in cfg.arms])

    def test_renamed_arm_drops_orphan(self) -> None:
        cfg = self.load_test_config(BASE, overrides="arms: {strands: {enabled: false}}\n")
        self.assertEqual([a.name for a in cfg.arms], ["claude-code"])
        self.assertEqual(settings.orphaned_overrides(cfg.overrides, yaml.safe_load(BASE)),
                         {"arms": ["strands"], "modes": []})

    def test_orphaned_modes_listed(self) -> None:
        over = {"modes": {"sample": {"n_tasks": 3}, "pilot": {"n_tasks": 1}}}
        self.assertEqual(settings.orphaned_overrides(over, yaml.safe_load(BASE)),
                         {"arms": [], "modes": ["sample"]})

    def test_prune_structural_levels_only(self) -> None:
        pruned = settings._prune({
            "run": {}, "modes": {"pilot": {}},
            "arms": {"a": {"options": {}}, "b": {"options": {"llm_kwargs": {}}}, "c": {"enabled": False}},
            "assistant": {"effort": "high"},
        })
        self.assertEqual(pruned, {"arms": {"b": {"options": {"llm_kwargs": {}}}, "c": {"enabled": False}},
                                  "assistant": {"effort": "high"}})


class ArmTests(BenchTestCase):
    MSA = "bench.agents.mini_swe_agent:MiniSweAgentWithMcp"

    def add_arm(self, line: str) -> str:
        return BASE + f"  - {line}\n"

    def test_import_path_family_defaults_to_agent(self) -> None:
        cfg = self.load_test_config(self.add_arm(f"{{name: msa, agent: '{self.MSA}', provider: anthropic, model: m}}"))
        arm = cfg.arm("msa")
        self.assertEqual((arm.family, arm.kind), (self.MSA, "import_path"))
        self.assertEqual(cfg.designs.import_paths, {self.MSA: self.MSA})
        self.assertEqual(cfg.arm("claude-code").family, "claude-code")
        self.assertEqual(cfg.arm("claude-code").kind, "name")

    def test_explicit_family(self) -> None:
        cfg = self.load_test_config(self.add_arm(
            f"{{name: msa, agent: '{self.MSA}', family: mini-swe-agent, provider: anthropic, model: m}}"))
        self.assertEqual(cfg.arm("msa").family, "mini-swe-agent")
        self.assertEqual(cfg.designs.import_paths, {self.MSA: "mini-swe-agent"})

    def test_family_from_designs_import_paths(self) -> None:
        text = self.add_arm(f"{{name: msa, agent: '{self.MSA}', provider: anthropic, model: m}}")
        text += f"designs:\n  import_paths:\n    '{self.MSA}': {{family: mini-swe-agent}}\n    'x.y:Z': zed\n"
        cfg = self.load_test_config(text)
        self.assertEqual(cfg.arm("msa").family, "mini-swe-agent")
        self.assertEqual(cfg.designs.import_paths, {self.MSA: "mini-swe-agent", "x.y:Z": "zed"})

    def test_family_conflict_raises(self) -> None:
        text = self.add_arm(f"{{name: msa, agent: '{self.MSA}', family: other, provider: anthropic, model: m}}")
        text += f"designs:\n  import_paths:\n    '{self.MSA}': mini-swe-agent\n"
        with self.assertRaises(ConfigError):
            self.load_test_config(text)

    def test_bad_agent_strings(self) -> None:
        for agent in ("'foo:bar-baz'", "'a.b:c.d'"):
            with self.subTest(agent=agent):
                with self.assertRaises(ConfigError):
                    self.load_test_config(self.add_arm(f"{{name: x, agent: {agent}, provider: anthropic, model: m}}"))

    def test_enabled_must_be_boolean(self) -> None:
        with self.assertRaises(ConfigError):
            self.load_test_config(self.add_arm("{name: x, agent: pi, provider: anthropic, model: m, enabled: 'false'}"))
        cfg = self.load_test_config(self.add_arm("{name: x, agent: pi, provider: anthropic, model: m, enabled: false}"))
        self.assertFalse(cfg.arm("x").enabled)

    def test_options_must_be_mapping(self) -> None:
        with self.assertRaises(ConfigError):
            self.load_test_config(self.add_arm("{name: x, agent: pi, provider: anthropic, model: m, options: [1]}"))

    STRANDS = ("  - name: s\n"
               "    agent: strands\n"
               "    provider: anthropic\n"
               "    model: claude-opus-5-5\n"
               "    options:\n"
               "      factory_kwargs: {model_spec: %s, effort: medium}\n")

    def test_model_spec_must_match_the_arm(self) -> None:
        for spec in ("anthropic/claude-opus-5-5", "' Anthropic/Claude-Opus-5-5 '", "null"):
            with self.subTest(spec=spec):
                self.assertEqual(self.load_test_config(BASE + self.STRANDS % spec).arm("s").model,
                                 "claude-opus-5-5")
        for spec in ("anthropic/claude-sonnet-5-5", "claude-opus-5-5", "openrouter/claude-opus-5-5", "3"):
            with self.subTest(spec=spec):
                with self.assertRaises(ConfigError) as ctx:
                    self.load_test_config(BASE + self.STRANDS % spec)
                self.assertIn("change both together", str(ctx.exception))

    def test_model_change_without_model_spec_is_refused(self) -> None:
        # A dashboard model change that leaves model_spec behind must not load.
        with self.assertRaises(ConfigError) as ctx:
            self.load_test_config(BASE + self.STRANDS % "anthropic/claude-opus-5-5",
                                  overrides="arms: {s: {model: claude-sonnet-5-5}}\n")
        self.assertIn("'anthropic/claude-sonnet-5-5'", str(ctx.exception))

    def test_arm_positional_fields(self) -> None:
        arm = settings.Arm("n", "claude-code", "anthropic", "m")
        self.assertEqual((arm.enabled, arm.effort, arm.options, arm.family), (True, "inherit", {}, ""))


class SectionTests(BenchTestCase):
    def test_defaults_when_absent(self) -> None:
        cfg = self.load_test_config(BASE)
        self.assertEqual(cfg.assistant, settings.AssistantSettings())
        self.assertEqual(cfg.designs.max_budget_usd, 150.0)
        self.assertEqual(cfg.designs.extra_harnesses, ())
        self.assertEqual(cfg.designs.import_paths, {})
        self.assertEqual(cfg.stats.confidence, 0.90)
        self.assertEqual(cfg.stats.default_trial_tokens, {"input": 600000, "cache_share": 0.85, "output": 60000})
        self.assertEqual(cfg.queue, settings.QueueSettings("pause", None))
        self.assertEqual(cfg.ledger.extra_scan_roots, ())
        self.assertEqual(cfg.report, {})

    def test_values_parse(self) -> None:
        cfg = self.load_test_config(BASE + textwrap.dedent("""\
            assistant: {effort: high, max_tokens: 64000, per_turn_cap_usd: 0.5, fallbacks: 'off', enabled: false}
            designs: {max_budget_usd: 10, extra_harnesses: [opencode], warn_containers: 4}
            stats: {confidence: 0.95, target_halfwidth: 0.2, default_trial_tokens: {output: 1000}}
            queue: {on_failure: skip, global_cap_usd: 40}
            ledger: {extra_scan_roots: [../other/jobs]}
            report: {group_by: config, crashes: exclude}
        """))
        self.assertEqual(cfg.assistant, settings.AssistantSettings(
            enabled=False, effort="high", max_tokens=64000, per_turn_cap_usd=0.5, fallbacks="off"))
        self.assertEqual((cfg.designs.max_budget_usd, cfg.designs.extra_harnesses, cfg.designs.warn_containers),
                         (10.0, ("opencode",), 4))
        self.assertEqual((cfg.stats.confidence, cfg.stats.target_halfwidth), (0.95, 0.2))
        self.assertEqual(cfg.stats.default_trial_tokens["output"], 1000)
        self.assertEqual(cfg.queue, settings.QueueSettings("skip", 40.0))
        self.assertEqual(cfg.ledger.extra_scan_roots, (settings.ROOT / "../other/jobs",))
        self.assertEqual(cfg.report, {"group_by": "config", "crashes": "exclude"})

    def test_unquoted_off_turns_fallbacks_off(self) -> None:
        self.assertEqual(self.load_test_config(BASE + "assistant: {fallbacks: off}\n").assistant.fallbacks, "off")

    def test_bad_values_raise(self) -> None:
        bad = [
            "assistant: {effort: ultra}",
            "assistant: {effort: off}",
            "assistant: {fallbacks: maybe}",
            "assistant: {max_tokens: 500000}",
            "assistant: {daily_cap_usd: .inf}",
            "assistant: {enabled: 'yes'}",
            "assistant: {model: claude-opus-5}",
            "queue: {on_failure: retry}",
            "queue: {global_cap_usd: -1}",
            "stats: {confidence: 95}",
            "stats: {confidence: 0.5}",
            "stats: {target_halfwidth: 25}",
            "stats: {default_trial_tokens: {cache_share: 1.5}}",
            "stats: {default_trial_tokens: {cached: 3}}",
            "designs: {max_budget_usd: .nan}",
            "designs: {warn_containers: 20}",
            "designs: {extra_harnesses: 'opencode'}",
            "designs: {extra_harnesses: ['bench.agents.x:Y']}",
            "designs: {import_paths: {not-a-path: x}}",
            "designs: {import_paths: {'a.b:C': {harness: x}}}",
            "ledger: {extra_scan_roots: '/tmp'}",
            "report: [1, 2]",
            "queue: 3",
        ]
        for line in bad:
            with self.subTest(line=line):
                with self.assertRaises(ConfigError):
                    self.load_test_config(BASE + line + "\n")


class WriteOverridesTests(BenchTestCase):
    PREVIOUS = "# kept\nrun:\n  budget_usd: 20\n"

    def setUp(self) -> None:
        super().setUp()
        self.base, self.overrides = self.use_config_files(BASE, self.PREVIOUS)

    def tmp_files(self) -> list[pathlib.Path]:
        return list(self.overrides.parent.glob("*.tmp"))

    def test_writes_pruned_overrides_and_reloads(self) -> None:
        cfg = settings.write_overrides({"run": {"budget_usd": 30}, "modes": {"pilot": {}}})
        self.assertEqual(cfg.run.budget_usd, 30)
        text = self.overrides.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# Written by the dashboard."))
        self.assertEqual(yaml.safe_load(text), {"run": {"budget_usd": 30}})
        self.assertNotIn("\r\n", self.overrides.read_bytes().decode("utf-8"))
        self.assertEqual(self.tmp_files(), [])

    def test_rollback_on_bad_value(self) -> None:
        for value in (float("nan"), float("inf"), "abc"):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError):
                    settings.write_overrides({"run": {"budget_usd": value}})
                self.assertEqual(self.overrides.read_text(encoding="utf-8"), self.PREVIOUS)
                self.assertEqual(self.tmp_files(), [])
        self.assertEqual(settings.load_config().run.budget_usd, 20)

    def test_rollback_on_any_exception(self) -> None:
        with mock.patch.object(settings, "load_config", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                settings.write_overrides({"run": {"budget_usd": 30}})
        self.assertEqual(self.overrides.read_text(encoding="utf-8"), self.PREVIOUS)

    def test_rollback_removes_file_that_did_not_exist(self) -> None:
        self.overrides.unlink()
        with self.assertRaises(ConfigError):
            settings.write_overrides({"run": {"n_concurrent": 0}})
        self.assertFalse(self.overrides.exists())
        self.assertEqual(self.tmp_files(), [])

    def test_survives_two_permission_errors(self) -> None:
        real = pathlib.Path.replace
        calls = []

        def flaky(self_path: pathlib.Path, target: object) -> pathlib.Path:
            calls.append(self_path)
            if len(calls) <= 2:
                raise PermissionError(13, "locked", str(target))
            return real(self_path, target)

        with mock.patch.object(pathlib.Path, "replace", flaky):
            cfg = settings.write_overrides({"run": {"budget_usd": 33}})
        self.assertEqual(cfg.run.budget_usd, 33)
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.tmp_files(), [])

    def test_busy_file_raises_permission_error(self) -> None:
        with mock.patch.object(settings.fsutil, "atomic_write_text", return_value=False):
            with self.assertRaises(PermissionError) as ctx:
                settings.write_overrides({"run": {"budget_usd": 33}})
        self.assertIn("busy", str(ctx.exception))
        self.assertEqual(self.overrides.read_text(encoding="utf-8"), self.PREVIOUS)

    def test_base_config_reads_patched_base(self) -> None:
        self.assertEqual(settings.base_config()["run"], {"budget_usd": 10})


class TestInfrastructureTests(BenchTestCase):
    """tests/support.py itself: every other package's tests rely on these guards.
    The commands are harmless (--version, help) in case a guard ever fails."""

    def test_harbor_docker_taskkill_blocked(self) -> None:
        for argv in ([settings.harbor_executable(), "--version"], ["harbor.exe", "--version"],
                     ["docker", "info"], [r"C:\Windows\System32\taskkill.exe", "/?"],
                     # Through a shell, and the cleanup helper that runs docker itself.
                     ["cmd", "/c", "docker", "--version"], ["cmd.exe", "/c", "docker.exe --version"],
                     ["powershell", "-NoProfile", "-Command", "& docker --version"],
                     ["pwsh", "-c", "Write-Host x; harbor --version"], ["wsl", "docker", "--version"],
                     [sys.executable, "-m", "bench.procs", "--help"], ["python", "-mbench.procs", "--help"]):
            with self.subTest(argv=argv):
                with self.assertRaises(AssertionError):
                    subprocess.run(argv, capture_output=True, timeout=5)
                with self.assertRaises(AssertionError):
                    subprocess.Popen(argv)
                with self.assertRaises(AssertionError):
                    subprocess.check_output(argv, timeout=5)

    def test_shell_command_strings_are_scanned(self) -> None:
        for command in ("echo x & docker --version", "cd . && harbor --version"):
            with self.subTest(command=command):
                with self.assertRaises(AssertionError):
                    subprocess.run(command, shell=True, capture_output=True, timeout=5)
                with self.assertRaises(AssertionError):
                    subprocess.Popen(command, shell=True)

    @unittest.skipUnless(sys.platform == "win32", "asyncio's Windows Popen")
    def test_asyncio_subprocesses_are_guarded(self) -> None:
        import asyncio

        async def start() -> None:
            await asyncio.create_subprocess_exec("docker", "--version")

        with self.assertRaises(AssertionError):
            asyncio.run(start())

    def test_allow_subprocess_routes_to_fake(self) -> None:
        from tests.support import completed

        seen = []
        self.allow_subprocess(lambda argv: argv[:2] == ["docker", "info"],
                              lambda argv, **kw: seen.append(argv) or completed(argv, stdout="8 16000000000"))
        proc = subprocess.run(["docker", "info", "--format", "x"], capture_output=True, text=True)
        self.assertEqual(proc.stdout, "8 16000000000")
        self.assertEqual(seen, [["docker", "info", "--format", "x"]])
        with self.assertRaises(AssertionError):
            subprocess.run(["docker", "ps"])

    def test_allow_subprocess_covers_the_cleanup_helper(self) -> None:
        from tests.support import completed

        argv = [sys.executable, "-m", "bench.procs", "cleanup", "x"]
        self.allow_subprocess(lambda a: "bench.procs" in a, lambda a, **kw: completed(a, returncode=0))
        self.assertEqual(subprocess.run(argv).returncode, 0)

    def test_other_programs_still_run(self) -> None:
        proc = subprocess.run([sys.executable, "-c", "print('ok')"], capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.stdout.strip(), "ok")
        if os.name == "nt":
            proc = subprocess.run(["cmd", "/c", "echo", "ok"], capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.stdout.strip(), "ok")
        self.assertIsInstance(subprocess.Popen([sys.executable, "-c", "pass"]).wait(timeout=60), int)

    def test_non_loopback_network_blocked(self) -> None:
        import socket

        from tests.support import NetworkBlocked

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            with self.assertRaises(NetworkBlocked):
                sock.connect(("203.0.113.1", 9))  # TEST-NET-3: documentation only
            with self.assertRaises(NetworkBlocked):
                sock.connect_ex(("203.0.113.1", 9))
        with self.assertRaises(OSError):
            socket.create_connection(("api.anthropic.com", 443), timeout=0.5)

    def test_loopback_allowed(self) -> None:
        import socket

        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            with socket.create_connection(server.getsockname(), timeout=5) as client:
                self.assertTrue(client.getpeername())

    def test_fake_keys_and_no_dotenv(self) -> None:
        self.assertTrue(os.environ["ANTHROPIC_API_KEY"].startswith("sk-test-"))
        self.assertTrue(os.environ["OPENROUTER_API_KEY"].startswith("sk-test-"))
        self.assertEqual(os.environ["PYTHON_DOTENV_DISABLED"], "1")
        self.assertNotIn("BENCH_PROFILE", os.environ)

    def test_schema_fixtures(self) -> None:
        from tests.support import fixture_schema

        for agent in ("claude-code", "terminus-2", "mini-swe-agent", "pi", "strands",
                      "bench.agents.mini_swe_agent:MiniSweAgentWithMcp"):
            with self.subTest(agent=agent):
                self.assertIn("properties", fixture_schema(agent))
        self.assertNotIn("reasoning_effort", {k for k, v in fixture_schema("strands")["properties"].items()
                                              if v.get("type") != "null"})
        with self.assertRaises(RuntimeError):
            fixture_schema("no-such-agent")
        try:
            from bench import schemas
        except Exception:
            self.skipTest("bench.schemas is not importable right now")
        self.assertIn("max_budget_usd", schemas.agent_schema("claude-code")["properties"])

    def test_isolate_data_dir(self) -> None:
        before = settings.RUNS_DIR
        root = self.isolate_data_dir()
        self.assertEqual(settings.RUNS_DIR, root / "runs")
        self.assertEqual(settings.QUEUE_FILE, root / "queue.json")
        self.assertNotEqual(settings.RUNS_DIR, before)


class ReadYamlTests(BenchTestCase):
    def test_invalid_yaml_is_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            self.load_test_config(BASE, overrides="run: {budget_usd: [\n")

    def test_missing_base_is_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            settings.load_config(self.tmpdir() / "none.yaml", self.tmpdir() / "none2.yaml")


if __name__ == "__main__":
    import unittest

    unittest.main()
