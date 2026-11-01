"""Run specs: a LAUNCH-tab run and a design run reduce to the same hashed spec,
and a config run's spec reproduces exactly what plan() has always run."""

from __future__ import annotations

import copy
import dataclasses
import datetime
import hashlib
import json
import re
from typing import Any

from tests.support import MINIMAL_YAML, BenchTestCase

from bench import settings
from bench import spec as specs
from bench.settings import ConfigError, RunSettings

IMPORT_PATH = "bench.agents.mini_swe_agent:MiniSweAgentWithMcp"

TWO_ARMS = MINIMAL_YAML + f"""\
  - {{name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}}
  - name: mini
    agent: {IMPORT_PATH}
    family: mini-swe-agent
    provider: anthropic
    model: claude-opus-5-5
    effort: high
    options: {{cost_limit: "2", version: "1.0"}}
"""


def reorder(value: Any) -> Any:
    """The same JSON with every object's keys in reverse order."""
    if isinstance(value, dict):
        return {k: reorder(value[k]) for k in reversed(list(value))}
    if isinstance(value, list):
        return [reorder(v) for v in value]
    return value


class ConfigSpecTests(BenchTestCase):
    def test_shipped_profiles_reproduce_plan_inputs(self) -> None:
        for profile in ("bench", "dev"):
            cfg = self.load_shipped_config(profile)
            names = [a.name for a in cfg.arms if a.enabled]
            for mode_name, mode in cfg.modes.items():
                with self.subTest(profile=profile, mode=mode_name):
                    rs = specs.spec_from_config(cfg, mode_name, names)
                    self.assertEqual((rs.source, rs.label, rs.design_ref), ("config", mode_name, None))
                    self.assertEqual(rs.dataset, cfg.dataset)
                    self.assertEqual(rs.tasks, tuple(cfg.tasks[: mode.n_tasks]))
                    self.assertEqual(rs.n_attempts, mode.n_attempts)
                    self.assertEqual(rs.run, cfg.run)
                    self.assertEqual(rs.baseline_arm, cfg.baseline_arm)
                    self.assertEqual([a.name for a in rs.arms], names)
                    for arm_spec in rs.arms:
                        arm = cfg.arm(arm_spec.name)
                        self.assertEqual(arm_spec.to_arm(), dataclasses.replace(arm, enabled=True))
                        self.assertEqual((arm_spec.kind, arm_spec.family), (arm.kind, arm.family))
                        self.assertEqual(arm_spec.tasks, rs.tasks)
                    self.assertEqual(rs.spec_hash, specs.spec_hash(rs.spec))
                    self.assertRegex(rs.spec_hash, r"^sha256:[0-9a-f]{64}$")
                    self.assertEqual(specs.structure_problems(rs.spec), [])

    def test_spec_layout(self) -> None:
        cfg = self.load_shipped_config("bench")
        spec = specs.normalized_from_config(cfg, "pilot", ["claude-code", "strands"])
        self.assertEqual(list(spec), list(specs.SPEC_KEYS))
        self.assertEqual(list(spec["run"]), list(specs.RUN_KEYS))
        self.assertEqual(spec["run"], {
            "budget_usd": 150.0, "parallel_arms": 1, "n_concurrent": 2, "effort": "medium",
            "per_trial_cap_usd": 15.0, "max_turns": 300, "timeout_multiplier": 1.0,
        })
        strands = spec["arms"][1]
        self.assertEqual(list(strands), list(specs.ARM_KEYS))
        self.assertEqual((strands["family"], strands["effort"], strands["tasks"]), ("strands", "omit", None))
        self.assertEqual(list(strands["options"]), sorted(strands["options"]))
        self.assertEqual(strands["options"]["factory_kwargs"],
                         {"model_spec": "anthropic/claude-opus-5-5", "effort": "medium"})

    def test_dev_import_path_arm(self) -> None:
        cfg = self.load_shipped_config("dev")
        rs = specs.spec_from_config(cfg, "pilot", ["mini-swe-agent-glm", "pi-glm"])
        mini, pi = rs.arms
        self.assertEqual((mini.agent, mini.kind, mini.family), (IMPORT_PATH, "import_path", "mini-swe-agent"))
        self.assertEqual((pi.kind, pi.family), ("name", "pi"))
        self.assertEqual(mini.to_arm().kind, "import_path")
        self.assertIsNone(rs.baseline_arm)  # the baseline arm was not picked

    def test_config_arm_options_are_copied(self) -> None:
        cfg = self.load_test_config(TWO_ARMS)
        rs = specs.spec_from_config(cfg, "pilot", ["mini"])
        rs.arms[0].options["cost_limit"] = "999"
        rs.spec["arms"][0]["options"]["version"] = "x"
        self.assertEqual(cfg.arm("mini").options, {"cost_limit": "2", "version": "1.0"})

    def test_arm_order_kept_and_repeats_dropped(self) -> None:
        cfg = self.load_test_config(TWO_ARMS)
        rs = specs.spec_from_config(cfg, "pilot", ["mini", "claude-code", "mini"])
        self.assertEqual([a.name for a in rs.arms], ["mini", "claude-code"])
        self.assertEqual(rs.baseline_arm, "claude-code")

    def test_errors_match_plan(self) -> None:
        cfg = self.load_test_config(TWO_ARMS)
        cases = [
            (("nope", ["pi"]), "Unknown mode 'nope'"),
            (("pilot", []), "Pick at least one arm"),
            (("pilot", ["pi", "ghost"]), "No arm named 'ghost'"),
        ]
        for args, message in cases:
            with self.subTest(args=args):
                with self.assertRaises(ConfigError) as ctx:
                    specs.spec_from_config(cfg, *args)
                self.assertEqual(str(ctx.exception), message)
        empty = self.load_test_config(TWO_ARMS, tasks=[])
        with self.assertRaises(ConfigError) as ctx:
            specs.spec_from_config(empty, "pilot", ["pi"])
        self.assertEqual(str(ctx.exception), "No task sample yet. Run: python -m bench.sample_tasks")


class HashTests(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        cfg = self.load_test_config(TWO_ARMS)
        self.spec = specs.normalized_from_config(cfg, "pilot", ["claude-code", "pi", "mini"])

    def test_stable_under_key_order_and_json_round_trip(self) -> None:
        h = specs.spec_hash(self.spec)
        self.assertEqual(specs.spec_hash(reorder(self.spec)), h)
        self.assertEqual(specs.spec_hash(json.loads(json.dumps(self.spec, indent=2))), h)
        canonical = json.dumps(self.spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.assertEqual(h, "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest())

    def test_any_change_changes_the_hash(self) -> None:
        edits = {
            "budget": lambda s: s["run"].__setitem__("budget_usd", 11.0),
            "budget type": lambda s: s["run"].__setitem__("budget_usd", 10),
            "cap": lambda s: s["run"].__setitem__("per_trial_cap_usd", 1.0),
            "task": lambda s: s["tasks"].__setitem__(0, "other-task"),
            "attempts": lambda s: s.__setitem__("attempts", 2),
            "baseline": lambda s: s.__setitem__("baseline_arm", None),
            "option": lambda s: s["arms"][2]["options"].__setitem__("cost_limit", "3"),
            "effort": lambda s: s["arms"][1].__setitem__("effort", "high"),
            "family": lambda s: s["arms"][2].__setitem__("family", "other"),
            "arm tasks": lambda s: s["arms"][0].__setitem__("tasks", list(s["tasks"])),
            "arm order": lambda s: s["arms"].reverse(),
        }
        seen = {specs.spec_hash(self.spec)}
        for name, edit in edits.items():
            with self.subTest(edit=name):
                changed = copy.deepcopy(self.spec)
                edit(changed)
                h = specs.spec_hash(changed)
                self.assertNotIn(h, seen)
                seen.add(h)


class DesignSpecTests(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        cfg = self.load_test_config(TWO_ARMS)
        self.spec = specs.normalized_from_config(cfg, "pilot", ["claude-code", "mini"])
        self.spec["tasks"] = ["t1", "t2", "t3"]

    def test_from_design(self) -> None:
        self.spec["arms"][1]["tasks"] = ["t2"]
        ref = {"design_id": "d-ds-pairs-block1-glm-7f3a", "version": 3, "author": "assistant",
               "conversation_id": None, "queue_item": "q-0007"}
        rs = specs.spec_from_design(self.spec, ref)
        self.assertEqual((rs.source, rs.label), ("design", "ds-pairs-block1-v3"))
        self.assertEqual(rs.design_ref, ref)
        self.assertIsNot(rs.design_ref, ref)
        self.assertEqual(rs.tasks, ("t1", "t2", "t3"))
        self.assertEqual(rs.arms[0].tasks, ("t1", "t2", "t3"))
        self.assertEqual(rs.arms[1].tasks, ("t2",))
        self.assertEqual((rs.arms[1].kind, rs.arms[1].family), ("import_path", "mini-swe-agent"))
        self.assertIsInstance(rs.run, RunSettings)
        self.assertEqual(rs.run.budget_usd, 10.0)
        self.assertEqual(rs.spec, self.spec)
        self.assertEqual(rs.spec_hash, specs.spec_hash(self.spec))
        self.spec["tasks"].append("t4")
        self.assertEqual(rs.tasks, ("t1", "t2", "t3"))  # the RunSpec kept its own copy

    def test_labels(self) -> None:
        self.assertEqual(specs.spec_from_design(self.spec, None).label, "design")
        self.assertEqual(specs.design_label({"design_id": "d-a-1f2e", "version": 12}), "a-1f2e-v12")
        self.assertEqual(specs.design_label({"design_id": "d-abcdefghijklmno-xyz", "version": 1}),
                         "abcdefghijklmno-v1")  # cut at 16, no trailing '-'
        self.assertRegex(specs.design_label({"design_id": "../Weird ID!", "version": 2}), r"^[a-z0-9-]+-v2$")

    def test_bad_specs_raise_config_error(self) -> None:
        bad = {
            "not an object": [],
            "missing key": {k: v for k, v in self.spec.items() if k != "attempts"},
            "unknown key": {**self.spec, "title": "x"},
            "bool attempts": {**self.spec, "attempts": True},
            "nan budget": {**self.spec, "run": {**self.spec["run"], "budget_usd": float("nan")}},
            "arm without family": {**self.spec, "arms": [{k: v for k, v in self.spec["arms"][0].items()
                                                          if k != "family"}]},
            "options not object": {**self.spec, "arms": [{**self.spec["arms"][0], "options": []}]},
        }
        # A version file is write-once and JSON responses refuse NaN, so options must be plain JSON.
        for name, value in (("nan", float("nan")), ("inf", float("-inf")), ("set", {1, 2}),
                            ("date", datetime.date(2026, 10, 29))):
            arms = copy.deepcopy(self.spec["arms"])
            arms[1]["options"]["nested"] = {"value": value}
            bad[f"{name} option"] = {**self.spec, "arms": arms}
            self.assertIn("spec.arms[1].options must be plain JSON with finite numbers",
                          specs.structure_problems(bad[f"{name} option"]))
        for name, spec in bad.items():
            with self.subTest(case=name):
                self.assertNotEqual(specs.structure_problems(spec), [])
                with self.assertRaises(ConfigError):
                    specs.spec_from_design(spec, None)  # type: ignore[arg-type]
        self.assertTrue(specs.plain_json({"a": [1, 2.5, None, True, {"b": "c"}]}))


class OptionalRunKeyTests(BenchTestCase):
    """prebake, setup_timeout_multiplier and env_build_timeout_multiplier
    are written only when they differ from RUN_DEFAULTS, so no spec stored
    before they existed changes its hash."""

    def test_every_stored_design_keeps_its_spec_hash(self) -> None:
        roots = [settings.ROOT / "designs", *sorted((settings.ROOT / "profiles").glob("*/designs"))]
        files = sorted(f for root in roots if root.is_dir() for f in root.glob("*/v*.json")
                       if re.fullmatch(r"v\d{4,}\.json", f.name))
        self.assertTrue(files, "no stored design version to check")
        for path in files:
            with self.subTest(path=path.relative_to(settings.ROOT).as_posix()):
                stored = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(specs.spec_hash(stored["spec"]), stored["spec_hash"])
                self.assertEqual(specs.structure_problems(stored["spec"]), [])
                rs = specs.spec_from_design(stored["spec"], None)
                self.assertEqual(rs.spec_hash, stored["spec_hash"])
                written = {k: stored["spec"]["run"][k] for k in specs.OPTIONAL_RUN_KEYS if k in stored["spec"]["run"]}
                # A version saved before the keys existed, or with their defaults, reads as the defaults.
                defaults = ("prefer", "auto", 1.0)
                self.assertEqual((rs.run.prebake, rs.run.setup_timeout_multiplier, rs.run.env_build_timeout_multiplier),
                                 tuple(written.get(k, d) for k, d in zip(specs.OPTIONAL_RUN_KEYS, defaults)))
                self.assertEqual(specs.run_dict(rs.run), stored["spec"]["run"])

    def test_defaults_are_left_out_and_others_written(self) -> None:
        base = dict(effort="medium", n_concurrent=2, parallel_arms=1, budget_usd=5, per_trial_cap_usd=None,
                    max_turns=None, timeout_multiplier=1.0)
        self.assertEqual(list(specs.run_dict(RunSettings(**base))), list(specs.RUN_KEYS))
        changed = specs.run_dict(RunSettings(**base, prebake="off", setup_timeout_multiplier=2,
                                             env_build_timeout_multiplier=1.5))
        self.assertEqual({k: changed[k] for k in specs.OPTIONAL_RUN_KEYS if k in changed},
                         {"prebake": "off", "setup_timeout_multiplier": 2.0, "env_build_timeout_multiplier": 1.5})
        # The schedule: tasks (a series of task races) and 6 containers are the defaults, left out.
        raced = specs.run_dict(RunSettings(**base, schedule="arms", max_containers=4))
        self.assertEqual((raced["schedule"], raced["max_containers"]), ("arms", 4))
        self.assertNotIn("schedule", specs.run_dict(RunSettings(**base, schedule="tasks", max_containers=6)))
        self.assertNotIn("setup_timeout_multiplier",
                         specs.run_dict(RunSettings(**base, setup_timeout_multiplier="auto")))
        # 1 is a number, not auto: it means "never a safety net", so it is written.
        self.assertEqual(specs.run_dict(RunSettings(**base, setup_timeout_multiplier=1))["setup_timeout_multiplier"],
                         1.0)

    def test_a_spec_with_them_round_trips_and_hashes_apart(self) -> None:
        cfg = self.load_test_config(MINIMAL_YAML)
        spec = specs.normalized_from_config(cfg, "pilot", ["claude-code"])
        changed = copy.deepcopy(spec)
        changed["run"].update(prebake="require", setup_timeout_multiplier=1.5, env_build_timeout_multiplier=2.0)
        self.assertEqual(specs.structure_problems(changed), [])
        rs = specs.spec_from_design(changed, None)
        self.assertEqual((rs.run.prebake, rs.run.setup_timeout_multiplier, rs.run.env_build_timeout_multiplier),
                         ("require", 1.5, 2.0))
        self.assertEqual(specs.run_dict(rs.run), changed["run"])
        self.assertNotEqual(specs.spec_hash(changed), specs.spec_hash(spec))
        off = self.load_test_config(MINIMAL_YAML.replace("  budget_usd: 10\n", "  budget_usd: 10\n  prebake: 'off'\n"))
        self.assertEqual(specs.normalized_from_config(off, "pilot", ["claude-code"])["run"]["prebake"], "off")
        self.assertEqual(specs.spec_from_config(off, "pilot", ["claude-code"]).run, off.run)

    def test_bad_values_are_structure_problems(self) -> None:
        cfg = self.load_test_config(MINIMAL_YAML)
        spec = specs.normalized_from_config(cfg, "pilot", ["claude-code"])
        for key, value, message in (("prebake", "sometimes", "spec.run.prebake must be one of off, prefer, require"),
                                    ("setup_timeout_multiplier", "fast",
                                     "spec.run.setup_timeout_multiplier must be auto or a number"),
                                    ("env_build_timeout_multiplier", True,
                                     "spec.run.env_build_timeout_multiplier must be a number"),
                                    ("colour", "blue", "spec.run has an unknown field 'colour'")):
            bad = copy.deepcopy(spec)
            bad["run"][key] = value
            with self.subTest(key=key):
                self.assertIn(message, specs.structure_problems(bad))


if __name__ == "__main__":
    import unittest

    unittest.main()
