"""Pruning pre-baked images: only the builder's own images are ever touched.

No Docker: FakeDocker (tests/fakes_docker.py) holds the images and records
every command.
"""

from __future__ import annotations

import io
import unittest
from typing import Any
from unittest import mock

from tests.fakes_docker import FakeDocker
from tests.support import BenchTestCase

from bench.environments.prebaked import LABEL_NS, Want, bake_labels, harbor_version
from bench.prebake import __main__ as cli
from bench.prebake import builder, inventory, prune
from bench.prebake.builder import BuildRefused

RECIPE = "a" * 64
OTHER = "f" * 64
CONFIG = """\
dataset: terminal-bench@2.0
providers:
  anthropic: {env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}
modes:
  pilot: {n_tasks: 1}
run:
  budget_usd: 10
prebake:
  prefix: benchbake
  keep_days: 14
arms:
  - {name: claude-code, agent: claude-code, provider: anthropic, model: claude-opus-5-5, options: {version: "2.1.312"}}
"""
WANTED = {("claude-code", "2.1.312", RECIPE)}


class PruneTestCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.docker = FakeDocker()
        self.allow_subprocess(self.docker.matches, self.docker)
        root = self.tmpdir() / "prebake"
        self.patch(builder, "PREBAKE_DIR", root)
        self.patch(builder, "LOCK_WAIT_SEC", 0)
        self.isolate_data_dir()
        inventory.forget_images()
        self.addCleanup(inventory.forget_images)
        self.cfg = self.load_test_config(CONFIG, tasks=["fix-git"])
        self.task = self.docker.add_image("alexgshaw/fix-git:20251031")

    def patch(self, target: Any, name: str, value: Any) -> None:
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def bake(self, prefix: str = "benchbake", recipe: str = RECIPE, baked_at: str = "2026-01-01T00:00:00Z",
             harbor: str | None = None, label: bool = True, task: str = "alexgshaw/fix-git:20251031") -> str:
        want = Want("claude-code", "2.1.312", recipe, task, harbor_version=harbor or harbor_version(),
                    prefix=prefix)
        labels = bake_labels(want, base_image_id=self.task, baked_at=baked_at, dockerfile_sha256="d" * 64)
        if not label:
            labels.pop(LABEL_NS + "bake-format")
        self.docker.add_image(want.image, layers=("sha256:l1", "sha256:l2", "sha256:x"), labels=labels)
        return want.image

    def found(self, **kw: Any) -> dict[str, str]:
        out, _ = prune.candidates(self.cfg, wanted=kw.pop("wanted", WANTED), **kw)
        return {c.ref: c.kind for c in out}


class Candidates(PruneTestCase):
    def test_only_the_builders_own_images_are_candidates(self) -> None:
        old_unwanted = self.bake(recipe=OTHER)
        wanted = self.bake()
        unlabelled = self.bake(recipe="b" * 64, label=False)
        foreign = self.bake(prefix="someoneelse", recipe=OTHER)
        probe = "benchprobe/claude-code-2.1.312/alexgshaw/fix-git:20251031"
        self.docker.add_image(probe, labels={"org.benchprobe.harness": "claude-code"})
        self.docker.add_image("ailab-web:latest")
        found = self.found()
        self.assertEqual(found, {old_unwanted: "bake"})
        for kept in (wanted, unlabelled, foreign, probe, "ailab-web:latest", "alexgshaw/fix-git:20251031"):
            self.assertNotIn(kept, found)

    def test_reasons_harbor_upgrade_age_and_staging(self) -> None:
        upgraded = self.bake(harbor="0.1.0")
        young = self.bake(recipe=OTHER, baked_at="2999-01-01T00:00:00Z")
        staging = self.bake(prefix="benchbake-staging/b1-1")
        found, _ = prune.candidates(self.cfg, wanted=WANTED)
        reasons = {c.ref: c.reason for c in found}
        self.assertIn("built for Harbor 0.1.0", reasons[upgraded])
        self.assertIn("staging tag", reasons[staging])
        self.assertNotIn(young, reasons)
        self.assertIn(young, self.found(keep_days=-1000000))

    def test_staging_tags_are_kept_while_a_build_runs(self) -> None:
        staging = self.bake(prefix="benchbake-staging/b1-1")
        with builder.build_lock():
            found, notes = prune.candidates(self.cfg, wanted=WANTED)
        self.assertNotIn(staging, {c.ref for c in found})
        self.assertIn("a build is running", notes[0])

    def test_all_takes_every_image_under_the_prefix_and_probes_need_asking(self) -> None:
        wanted = self.bake()
        probe = "benchprobe/pi-1.4.0/alexgshaw/fix-git:20251031"
        self.docker.add_image(probe)
        self.assertEqual(self.found(everything=True), {wanted: "bake"})
        self.assertEqual(self.found(everything=True, include_probes=True), {wanted: "bake", probe: "probe"})

    def test_wanted_defaults_to_every_profiles_enabled_arms(self) -> None:
        calls: list[Any] = []
        self.patch(inventory, "wanted_recipes", lambda cfg: (calls.append(cfg), (WANTED, ["a note"]))[1])
        old = self.bake(recipe=OTHER)
        found, notes = prune.candidates(self.cfg)
        self.assertEqual(([c.ref for c in found], notes, calls), ([old], ["a note"], [self.cfg]))


class Removal(PruneTestCase):
    def test_a_dry_run_removes_nothing(self) -> None:
        self.bake(recipe=OTHER)
        found, _ = prune.candidates(self.cfg, wanted=WANTED)
        result = prune.prune(found)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["removed"], [])
        self.assertEqual(self.docker.ran("rmi") + self.docker.ran("image", "prune"), [])

    def test_removal_untags_without_force_then_prunes_only_labelled_layers(self) -> None:
        old = self.bake(recipe=OTHER)
        kept = self.bake()
        found, _ = prune.candidates(self.cfg, wanted=WANTED)
        result = prune.prune(found, dry_run=False)
        self.assertEqual(result["removed"], [old])
        self.assertEqual(self.docker.ran("rmi"), [["rmi", old]])
        self.assertEqual(self.docker.ran("image", "prune"), [["image", "prune", "-f", "--filter",
                                                              f"label={LABEL_NS}bake-format"]])
        self.assertIsNone(self.docker.find(old))
        self.assertIsNotNone(self.docker.find(kept))
        self.assertIsNotNone(self.docker.find("alexgshaw/fix-git:20251031"))
        for call in self.docker.ran("rmi"):
            self.assertFalse({"-f", "--force"} & set(call))
        self.assertEqual(self.docker.ran("system") + self.docker.ran("builder"), [])

    def test_an_image_that_lost_its_label_since_listing_is_left_alone(self) -> None:
        old = self.bake(recipe=OTHER)
        found, _ = prune.candidates(self.cfg, wanted=WANTED)
        self.docker.find(old)["Config"]["Labels"].pop(LABEL_NS + "bake-format")
        result = prune.prune(found, dry_run=False)
        self.assertIn(old, result["failed"])
        self.assertEqual(self.docker.ran("rmi"), [])

    def test_prune_refuses_during_a_run(self) -> None:
        self.bake(recipe=OTHER)
        found, _ = prune.candidates(self.cfg, wanted=WANTED)
        with builder.run_lock(), self.assertRaises(BuildRefused):
            prune.prune(found, dry_run=False)
        self.assertEqual(self.docker.ran("rmi"), [])

    def test_the_cli_is_a_dry_run_until_yes(self) -> None:
        old = self.bake(recipe=OTHER)
        self.patch(inventory, "wanted_recipes", lambda cfg: (WANTED, []))

        def run(*argv: str) -> tuple[int, str]:
            out = io.StringIO()
            with mock.patch("sys.stdout", out), mock.patch.object(cli, "load", lambda args: self.cfg):
                return cli.main(list(argv)), out.getvalue()

        code, out = run("prune")
        self.assertEqual(code, 0)
        self.assertIn("would be removed (dry run)", out)
        self.assertIsNotNone(self.docker.find(old))
        code, out = run("prune", "--yes")
        self.assertEqual(code, 0)
        self.assertIn("Removed 1", out)
        self.assertIsNone(self.docker.find(old))


if __name__ == "__main__":
    unittest.main()
