"""The pre-bake builder: what a selection needs, the disk guard, the locks
that keep builds and runs apart, and the build pipeline from recording to
promotion.

No Docker: FakeDocker (tests/fakes_docker.py) answers every docker command
through BenchTestCase.allow_subprocess, and the recorder's probes come from
real answers canned in tests/fixtures/prebake/probes.json. The harnesses'
own Harbor setup() code runs for real where a recipe is recorded.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from tests.fakes_docker import CannedProbes, FakeDocker
from tests.support import BenchTestCase, completed

from bench import datasets, procs, reuse, settings
from bench.environments.prebaked import LABEL_NS, Want, bake_labels, harbor_version
from bench.prebake import ArmPrebake, adapters, builder, inventory, pull, recorder, verify
from bench.prebake import __main__ as cli
from bench.prebake.builder import BuildItem, BuildManager, BuildRefused, Selection

MODEL = "anthropic/claude-opus-5-5"
FIX_GIT = "alexgshaw/fix-git:20251031"
REGEX_LOG = "alexgshaw/regex-log:20251031"
RECIPES = {"claude-code": "a" * 64, "terminus-2": "b" * 64, "mini-swe-agent": "c" * 64}
VERSIONS = {"claude-code": "2.1.312", "terminus-2": "0.25.0", "mini-swe-agent": "2.4.6"}
CONFIG = """\
dataset: terminal-bench@2.0
providers:
  anthropic: {env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}
modes:
  pilot: {n_tasks: 2}
  full: {n_tasks: 3}
run:
  budget_usd: 10
prebake:
  prefix: benchbake-test
  host_reserve_gb: 25
  vm_reserve_gb: 50
arms:
  - {name: claude-code, agent: claude-code, provider: anthropic, model: claude-opus-5-5,
     options: {version: "2.1.312"}}
  - {name: claude-code-b, agent: claude-code, family: claude-code, provider: anthropic, model: claude-opus-5-5,
     options: {version: "2.1.312", max_turns: 100}}
  - {name: terminus-2, agent: terminus-2, provider: anthropic, model: claude-opus-5-5}
  - {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}
"""


def fake_resolve(cfg: Any, run: Any, arm: Any, kwargs: Any) -> ArmPrebake | None:
    """prebake.resolve without Harbor: fixed recipes; pi unpinned, so unsupported."""
    if run.prebake == "off":
        return None
    if arm.family not in RECIPES:
        return ArmPrebake(mode=run.prebake, family=arm.family, supported=False, prefix=cfg.prebake.prefix,
                          reason="no pinned version, so Harbor installs the latest in every trial")
    return ArmPrebake(mode=run.prebake, family=arm.family, supported=True, version=VERSIONS[arm.family],
                      recipe=RECIPES[arm.family], variant="plain", prefix=cfg.prebake.prefix)


class BuilderTestCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.docker = FakeDocker()
        self.allow_subprocess(self.docker.matches, self.docker)
        self.root = self.tmpdir() / "prebake"
        self.patch(builder, "PREBAKE_DIR", self.root)
        self.patch(verify, "PREBAKE_DIR", self.root)
        self.patch(builder, "LOCK_WAIT_SEC", 0)
        self.data = self.isolate_data_dir()
        self.dataset = self.tmpdir() / "tasks"
        for task, image in (("fix-git", FIX_GIT), ("regex-log", REGEX_LOG), ("no-image", None),
                            ("other", "alexgshaw/other:1"),
                            ("multi-source-data-merger", "alexgshaw/multi-source-data-merger:20251031")):
            (self.dataset / task).mkdir(parents=True)
            env = f'[environment]\ndocker_image = "{image}"\n' if image else "[environment]\ncpus = 1\n"
            (self.dataset / task / "task.toml").write_text(env, encoding="utf-8")
        self.patch(datasets, "path", lambda dataset: self.dataset)
        self.base_id = self.docker.add_image(FIX_GIT, layers=("sha256:l1", "sha256:l2"))
        self.docker.add_image(REGEX_LOG, layers=("sha256:r1",))
        inventory.forget_images()
        self.addCleanup(inventory.forget_images)
        self.cfg = self.load_test_config(CONFIG, tasks=["fix-git", "regex-log", "no-image"])

    def patch(self, target: Any, name: str, value: Any) -> Any:
        patcher = mock.patch.object(target, name, value)
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started

    def fake_recipes(self) -> None:
        self.patch(inventory, "resolve_prebake", fake_resolve)

    def baked(self, family: str, task_image: str = FIX_GIT, **label_changes: str) -> str:
        """A ready pre-baked image of family on task_image, as a build would leave it."""
        want = Want(family, VERSIONS[family], RECIPES[family], task_image, prefix="benchbake-test")
        base = self.docker.find(task_image)
        assert base is not None
        baked_at = label_changes.pop("baked_at", "2026-10-29T12:00:00Z")
        labels = bake_labels(want, base_image_id=base["Id"], baked_at=baked_at, dockerfile_sha256="d" * 64)
        labels.update({LABEL_NS + k: v for k, v in label_changes.items()})
        self.docker.add_image(want.image, layers=(*base["RootFS"]["Layers"], "sha256:x"), labels=labels,
                              config={k: v for k, v in base["Config"].items() if k != "Labels"})
        return want.image

    def spec(self, arms: list[str] | None = None, mode: str = "pilot", **sel: Any) -> Any:
        return builder.selection_spec(self.cfg, Selection(mode=mode, arms=tuple(arms or ()), **sel), rows=lambda: [])


class Inventory(BuilderTestCase):
    def test_listing_is_cached_and_inspect_asks_only_for_local_names(self) -> None:
        self.assertIn(FIX_GIT, inventory.local_images())
        self.docker.add_image("late/image:1")
        self.assertNotIn("late/image:1", inventory.local_images())
        self.assertIn("late/image:1", inventory.local_images(refresh=True))
        self.docker.calls.clear()
        found = inventory.inspect_raw([FIX_GIT, "absent/image:1"])
        self.assertEqual(found[FIX_GIT]["Id"], self.base_id)
        self.assertIsNone(found["absent/image:1"])
        self.assertEqual(self.docker.ran("image", "inspect"), [["image", "inspect", FIX_GIT]])

    def test_task_images_come_from_task_toml(self) -> None:
        self.assertEqual(inventory.task_images("terminal-bench@2.0", ["fix-git", "regex-log", "no-image", "gone"]),
                         {"fix-git": FIX_GIT, "regex-log": REGEX_LOG, "no-image": None, "gone": None})

    def test_readiness_runs_the_trials_own_check_per_task(self) -> None:
        self.fake_recipes()
        self.baked("claude-code")
        self.baked("terminus-2", **{"harbor-version": "0.1.0"})
        result = inventory.readiness(self.cfg, self.spec(mode="full"))
        cc = result["claude-code"]
        self.assertTrue(cc.supported)
        self.assertEqual((cc.ready, cc.missing), (["fix-git"], ["regex-log"]))
        self.assertEqual(list(cc.unknown), ["no-image"])
        self.assertAlmostEqual(cc.build_min, adapters.ClaudeCodeAdapter.expected_build_min)
        t2 = result["terminus-2"]
        self.assertEqual(list(t2.stale), ["fix-git"])
        self.assertIn("harbor-version", t2.stale["fix-git"])
        self.assertFalse(result["pi"].supported)
        self.assertIn("pinned", result["pi"].reason)
        self.assertEqual(cc.to_dict()["images"]["fix-git"],
                         "benchbake-test/claude-code-2.1.312-raaaaaaaaaa/alexgshaw/fix-git:20251031")

    def test_a_task_image_that_is_not_local_cannot_be_verified(self) -> None:
        self.fake_recipes()
        self.baked("claude-code", REGEX_LOG)
        for image_id, obj in list(self.docker.images.items()):
            if REGEX_LOG in obj["RepoTags"]:
                obj["RepoTags"].remove(REGEX_LOG)
        result = inventory.readiness(self.cfg, self.spec(["claude-code"]), refresh=True)
        self.assertEqual(result["claude-code"].unverifiable, ["regex-log"])

    def test_every_task_ready_clears_expect_stock_so_auto_adds_no_setup_limit(self) -> None:
        from bench import prebake

        self.fake_recipes()
        self.baked("claude-code")
        self.baked("claude-code", REGEX_LOG)
        spec = self.spec(["claude-code"], tasks=("fix-git", "regex-log"))
        r = inventory.readiness(self.cfg, spec)["claude-code"]
        self.assertTrue(r.all_ready)
        self.assertTrue(r.prebake.expect_stock)
        self.assertFalse(r.arm_prebake().expect_stock)
        self.assertIsNone(prebake.setup_timeout_multiplier(spec.run, r.arm_prebake()))
        self.assertEqual(prebake.setup_timeout_multiplier(spec.run, r.prebake), 2.0)

    def test_noop_and_over_age_images_are_reported(self) -> None:
        self.fake_recipes()
        base = self.docker.find(FIX_GIT)
        want = Want("terminus-2", "0.25.0", RECIPES["terminus-2"], FIX_GIT, prefix="benchbake-test")
        labels = bake_labels(want, base_image_id=base["Id"], baked_at="2026-01-01T00:00:00Z",
                             dockerfile_sha256="d" * 64, noop=True)
        self.docker.add_image(want.image, layers=tuple(base["RootFS"]["Layers"]), labels=labels,
                              config={k: v for k, v in base["Config"].items() if k != "Labels"})
        r = inventory.readiness(self.cfg, self.spec(["terminus-2"]))["terminus-2"]
        self.assertEqual((r.ready, r.noop, r.over_age), (["fix-git"], ["fix-git"], ["fix-git"]))
        self.assertEqual(r.oldest_baked_at, "2026-01-01T00:00:00Z")

    def test_docker_not_answering_is_reported_not_raised(self) -> None:
        self.fake_recipes()
        self.allow_subprocess(self.docker.matches, lambda argv, **kw: completed(
            argv, "", "Error response from daemon: Docker Desktop is unable to start", 1))
        r = inventory.readiness(self.cfg, self.spec(["claude-code"]), refresh=True)["claude-code"]
        self.assertIn("unable to start", r.docker_error)
        self.assertEqual(r.unverifiable, ["fix-git", "regex-log"])
        self.assertGreater(r.build_gb, 0)

    def test_wanted_recipes_cover_every_enabled_arm(self) -> None:
        self.fake_recipes()
        wanted, notes = inventory.wanted_recipes(self.cfg, all_profiles=False)
        self.assertEqual(wanted, {("claude-code", "2.1.312", "a" * 64), ("terminus-2", "0.25.0", "b" * 64)})
        self.assertEqual(notes, [])


class Selections(BuilderTestCase):
    def test_a_mode_takes_the_enabled_arms_and_task_narrows_it(self) -> None:
        spec = self.spec()
        self.assertEqual([a.name for a in spec.arms], ["claude-code", "claude-code-b", "terminus-2", "pi"])
        narrowed = self.spec(["terminus-2"], tasks=("regex-log",))
        self.assertEqual([(a.name, a.tasks) for a in narrowed.arms], [("terminus-2", ("regex-log",))])
        with self.assertRaises(settings.ConfigError):
            self.spec(tasks=("nope",))

    def test_a_design_resolves_its_version_through_reuse(self) -> None:
        from bench import designs

        store = designs.DesignStore()
        launch = designs.design_from_config(self.cfg, "pilot", ["claude-code", "terminus-2"])
        created = store.create(self.cfg, {"title": "t"}, launch, "user")
        design_id = created["design_id"]
        seen: list[str] = []

        def resolve(cfg: Any, spec: Any, rows: Any, mode: Any) -> Any:
            seen.append(mode)
            out = designs.with_reuse(spec, "measured")
            out["arms"][0]["tasks"] = ["regex-log"]          # the ledger covers fix-git
            out["reused"] = {"claude-code": [{"task": "fix-git", "trials": ["k"], "runs": ["r"]}]}
            return out

        self.patch(reuse, "resolve", resolve)
        spec = builder.selection_spec(self.cfg, Selection(design=f"{design_id}@v1", reuse="measured"),
                                      rows=lambda: [])
        self.assertEqual(seen, ["measured"])
        self.assertEqual({a.name: a.tasks for a in spec.arms},
                         {"claude-code": ("regex-log",), "terminus-2": ("fix-git", "regex-log")})
        plain = builder.selection_spec(self.cfg, Selection(design=design_id), rows=lambda: [])
        self.assertEqual(seen, ["measured"])  # the design's own reuse (off) reads no ledger
        self.assertEqual(len(plain.arms), 2)

    def test_a_run_builds_what_remains_by_default(self) -> None:
        from bench import designs

        launch = designs.design_from_config(self.cfg, "pilot", ["terminus-2"])
        run_dir = settings.RUNS_DIR / "20261029-000000-pilot"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({"spec": launch, "status": "done"}), encoding="utf-8")
        modes: list[str] = []
        self.patch(reuse, "resolve", lambda cfg, spec, rows, mode: modes.append(mode) or designs.with_reuse(spec, mode))
        spec = builder.selection_spec(self.cfg, Selection(run="20261029-000000-pilot"), rows=lambda: [])
        self.assertEqual(modes, ["measured"])
        self.assertEqual([a.name for a in spec.arms], ["terminus-2"])
        with self.assertRaises(settings.ConfigError):
            builder.selection_spec(self.cfg, Selection(run="..\\x"), rows=lambda: [])

    def test_design_refs(self) -> None:
        self.assertEqual(builder.parse_design_ref("d-x-1@v3"), ("d-x-1", 3))
        self.assertEqual(builder.parse_design_ref("d-x-1"), ("d-x-1", None))
        with self.assertRaises(settings.ConfigError):
            builder.parse_design_ref("not a design")


class Plans(BuilderTestCase):
    def test_items_are_deduplicated_across_arms_and_ordered_longest_first(self) -> None:
        self.fake_recipes()
        self.baked("terminus-2", REGEX_LOG)
        plan = builder.plan_builds(self.cfg, self.spec())
        keys = [(i.family, i.task_image) for i in plan.items]
        self.assertEqual(keys, [("claude-code", FIX_GIT), ("claude-code", REGEX_LOG), ("terminus-2", FIX_GIT)])
        self.assertEqual(plan.items[0].arms, ["claude-code", "claude-code-b"])
        self.assertEqual(plan.items[0].image, "benchbake-test/claude-code-2.1.312-raaaaaaaaaa/alexgshaw/"
                                              "fix-git:20251031")
        self.assertEqual([r["task"] for r in plan.ready], ["regex-log"])
        self.assertEqual([u["arm"] for u in plan.unsupported], ["pi"])
        self.assertEqual({u["task"] for u in plan.unknown}, set())  # no-image is not in the pilot mode
        self.assertAlmostEqual(plan.need_gb, round((2 * 1.9 + 0.2) * builder.DISK_FACTOR, 2))
        self.assertEqual(plan.wall_minutes(4), 11.0)
        self.assertEqual(plan.wall_minutes(1), 22.6)
        self.assertEqual([i.task_image for i in plan.ready_items], [REGEX_LOG])

    def test_rebuild_includes_ready_images(self) -> None:
        self.fake_recipes()
        self.baked("terminus-2", REGEX_LOG)
        plan = builder.plan_builds(self.cfg, self.spec(["terminus-2"]), rebuild=True)
        self.assertEqual([(i.task_image, i.why) for i in plan.items], [(FIX_GIT, f"no pre-baked image "
                                                                                 f"{plan.items[0].image}"),
                                                                        (REGEX_LOG, "rebuild")])

    def test_verify_checks_ready_images_and_samples_the_setup_check(self) -> None:
        self.fake_recipes()
        self.patch(recorder, "PROBES", CannedProbes())
        self.baked("terminus-2")
        self.baked("terminus-2", REGEX_LOG)
        setups: list[dict[str, Any]] = []
        self.patch(verify, "setup_check", lambda **kw: setups.append(kw) or {"ok": True, "agent_setup_sec": 2.0})
        plan = builder.plan_builds(self.cfg, self.spec(["terminus-2"]))
        self.assertEqual(plan.items, [])
        results = builder.verify_ready(self.cfg, plan, setup=True, root=self.root)
        self.assertEqual([(r["task_image"], r["ok"], r["os_family"]) for r in results],
                         [(FIX_GIT, True, "debian-12"), (REGEX_LOG, True, "ubuntu-24.04")])
        self.assertEqual([kw["prefix"] for kw in setups], ["benchbake-test", "benchbake-test"])
        records = [json.loads(x) for x in (self.root / "verify.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(r["kind"], r["ok"]) for r in records], [("image", True), ("image", True)])
        self.docker.run_handlers.append((lambda a, s: s is not None and s.startswith("set -o pipefail"),
                                         lambda a, s: (1, "", "no tmux")))
        self.assertFalse(any(r["ok"] for r in builder.verify_ready(self.cfg, plan, root=self.root)))

    def test_mode_off_is_planned_as_prefer_with_a_note(self) -> None:
        self.fake_recipes()
        spec = self.spec(["terminus-2"])
        spec = dataclasses.replace(spec, run=dataclasses.replace(spec.run, prebake="off"))
        plan = builder.plan_builds(self.cfg, spec)
        self.assertEqual(len(plan.items), 2)
        self.assertIn("run.prebake is off", plan.notes[0])


class Disk(BuilderTestCase):
    def item(self, family: str = "claude-code", task_image: str = FIX_GIT) -> BuildItem:
        cls = adapters.ADAPTERS[family]
        return BuildItem(family=family, version=VERSIONS[family], recipe=RECIPES[family], variant="plain",
                         agent=family, kwargs={}, model_name=MODEL, task_image=task_image, prefix="benchbake-test",
                         dataset="terminal-bench@2.0", tasks=["fix-git"], expected_min=cls.expected_build_min,
                         expected_gb=cls.expected_gb)

    def test_docker_disk_is_found_by_default_or_through_the_settings_store(self) -> None:
        local, roaming = self.tmpdir(), self.tmpdir()
        os.environ.update(LOCALAPPDATA=str(local), APPDATA=str(roaming))
        default = local / "Docker" / "wsl" / "disk" / "docker_data.vhdx"
        default.parent.mkdir(parents=True)
        default.write_bytes(b"")
        self.assertEqual(builder.docker_disk(), (default, "Docker Desktop's default location"))
        moved = self.tmpdir() / "DockerDesktopWSL" / "disk" / "docker_data.vhdx"
        moved.parent.mkdir(parents=True)
        moved.write_bytes(b"")
        (roaming / "Docker").mkdir()
        (roaming / "Docker" / "settings-store.json").write_text(
            json.dumps({"CustomWslDistroDir": str(moved.parent.parent.parent)}), encoding="utf-8")
        self.assertEqual(builder.docker_disk(), (moved, "Docker Desktop's settings (CustomWslDistroDir)"))

    def test_a_short_host_drive_refuses_and_says_why_without_running_a_container(self) -> None:
        vhdx = Path("C:/Users/x/AppData/Local/Docker/wsl/disk/docker_data.vhdx")
        self.patch(builder, "docker_disk", lambda: (vhdx, "Docker Desktop's default location"))
        self.patch(builder, "host_free_gb", lambda path: 9.2)
        check = builder.disk_check(self.cfg, [self.item()] * 3, probe_image=FIX_GIT)
        self.assertFalse(check.ok)
        self.assertEqual(check.need_gb, round(3 * 1.9 * 1.15, 2))
        self.assertIn("9.2 GB free", check.reasons[0])
        self.assertIn("prebake.host_reserve_gb (25 GB)", check.reasons[0])
        self.assertIn("docker_data.vhdx", check.reasons[0])
        self.assertFalse(check.vm["checked"])
        self.assertEqual(self.docker.ran("run"), [])

    def test_free_space_inside_the_disk_image_counts_before_the_host_drive(self) -> None:
        # 2026-10-31: D: had 16 GB free, but the 460 GB disk image held 105 GB, so builds grow nothing.
        class Disk:
            def __init__(self, gb: float) -> None:
                self.gb = gb

            def is_file(self) -> bool:
                return True

            def stat(self) -> Any:
                return mock.Mock(st_size=int(self.gb * 1024 ** 3))

            def __str__(self) -> str:
                return "D:/Docker/DockerDesktopWSL/disk/docker_data.vhdx"

            anchor = "D:\\"

        self.patch(builder, "last_vm", {})
        self.patch(builder, "host_free_gb", lambda path: 16.2)
        self.docker.vm_free_kb = 900 * 1024 ** 2
        roomy = Disk(1000 * 1024 / 1024 ** 2 + 460)  # the fake df reports 1000 KB in use
        self.patch(builder, "docker_disk", lambda: (roomy, "Docker Desktop's settings (CustomWslDistroDir)"))
        check = builder.disk_check(self.cfg, [self.item()] * 3, probe_image=FIX_GIT)
        self.assertTrue(check.ok, check.reasons)
        self.assertEqual(check.host["growth_gb"], 0.0)
        tight = Disk(1000 * 1024 / 1024 ** 3 + 2.0)  # only 2 GB free inside the file
        self.patch(builder, "docker_disk", lambda: (tight, "Docker Desktop's settings (CustomWslDistroDir)"))
        self.assertFalse(builder.disk_check(self.cfg, [self.item()] * 3, probe_image=FIX_GIT).ok)

    def test_the_vm_is_checked_with_df_in_a_local_task_image(self) -> None:
        self.patch(builder, "docker_disk", lambda: (None, "not found"))
        self.patch(builder, "host_free_gb", lambda path: 500.0)
        self.assertTrue(builder.disk_check(self.cfg, [self.item()], probe_image=FIX_GIT).ok)
        run = self.docker.ran("run")[0]
        self.assertIn("--pull", run)
        self.assertEqual(run[run.index("--entrypoint") + 1:], ["df", FIX_GIT, "-Pk", "/"])
        self.docker.vm_free_kb = 40 * 1024 ** 2
        check = builder.disk_check(self.cfg, [self.item()], probe_image=FIX_GIT)
        self.assertFalse(check.ok)
        self.assertIn("Docker's VM has 40.0 GB free", check.reasons[0])

    def test_nothing_to_build_needs_no_space(self) -> None:
        self.assertTrue(builder.disk_check(self.cfg, []).ok)


class Locks(BuilderTestCase):
    def test_a_build_refuses_during_a_run_and_a_run_during_a_build(self) -> None:
        with builder.run_lock():
            self.assertTrue(builder.active_runs())
            with self.assertRaises(BuildRefused) as caught, builder.build_lock():
                pass
            self.assertIn("during a run", str(caught.exception))
        with builder.build_lock():
            self.assertTrue(builder.build_active())
            with self.assertRaises(BuildRefused), builder.run_lock():
                pass
        self.assertFalse(builder.build_active())
        self.assertEqual(builder.active_runs(), [])

    def test_another_probe_holding_a_lock_for_a_moment_is_not_a_holder(self) -> None:
        # The state poll probes the build lock while a run start probes it too.
        for path, check in ((builder.build_lock_path(), builder.build_active),
                            (builder.run_lock_path(), lambda: bool(builder.active_runs()))):
            taken, done = threading.Event(), threading.Event()

            def probe(lock: Path = path) -> None:
                with procs.file_lock(lock, timeout=0):
                    taken.set()
                    time.sleep(builder.PROBE_GRACE_SEC / 5)
                done.set()

            thread = threading.Thread(target=probe)
            thread.start()
            self.assertTrue(taken.wait(5))
            self.assertFalse(check(), path.name)
            thread.join(5)
            self.assertTrue(done.is_set())
        with procs.file_lock(builder.build_lock_path(), timeout=0):
            self.assertTrue(builder.build_active())  # a real holder outlasts the grace
            with self.assertRaises(BuildRefused), builder.run_lock():
                pass

    def test_a_fresh_live_run_json_counts_as_a_run(self) -> None:
        run = settings.RUNS_DIR / "r1" / "run.json"
        run.parent.mkdir(parents=True)
        run.write_text(json.dumps({"status": "running"}), encoding="utf-8")
        self.assertEqual(builder.active_runs(), ["run r1"])
        old = time.time() - builder.RUN_FRESH_SEC - 60
        os.utime(run, (old, old))
        self.assertEqual(builder.active_runs(), [])  # a bench that died mid-run left it behind
        run.write_text(json.dumps({"status": "done"}), encoding="utf-8")
        self.assertEqual(builder.active_runs(), [])

    def test_the_managers_own_run_check_and_a_second_build_refuse(self) -> None:
        self.fake_recipes()
        plan = builder.plan_builds(self.cfg, self.spec(["terminus-2"]))
        manager = BuildManager(root=self.root, run_active=lambda: True)
        with self.assertRaises(BuildRefused):
            manager.start(self.cfg, plan)
        self.assertTrue(manager.is_idle())
        with builder.build_lock(), self.assertRaises(BuildRefused) as caught:
            BuildManager(root=self.root).start(self.cfg, plan)
        self.assertIn("Another build", str(caught.exception))


class Pipeline(BuilderTestCase):
    """A real terminus-2 recording on fix-git (canned probes), built, verified and promoted."""

    def setUp(self) -> None:
        super().setUp()
        self.patch(recorder, "PROBES", CannedProbes())
        self.patch(builder, "docker_disk", lambda: (None, "not found"))
        self.patch(builder, "host_free_gb", lambda path: 500.0)
        self.setup_calls: list[dict[str, Any]] = []
        self.setup_result: dict[str, Any] = {"ok": True, "agent_setup_sec": 2.6, "problems": []}
        self.patch(verify, "setup_check", self.fake_setup)
        self.equivalence_calls: list[dict[str, Any]] = []
        self.equivalence_ok = True
        self.patch(verify, "equivalence", self.fake_equivalence)
        self.adapter = adapters.adapter_for("terminus-2", "terminus-2", {}, MODEL)
        self.recipe = recorder.arm_recipe(self.adapter)
        # FakeDocker's fix-git stands in for the probed one.
        self.docker.add_image("alexgshaw/multi-source-data-merger:20251031", layers=("sha256:m1",))

    def fake_setup(self, **kw: Any) -> dict[str, Any]:
        self.setup_calls.append(kw)
        return dict(self.setup_result)

    def fake_equivalence(self, **kw: Any) -> dict[str, Any]:
        self.equivalence_calls.append(kw)
        violations = [] if self.equivalence_ok else ["/usr/bin/x: content"]
        return {"ok": self.equivalence_ok, "files": {"violations": violations}, "fingerprints": []}

    def item(self, task_image: str = FIX_GIT, task: str = "fix-git") -> BuildItem:
        return BuildItem(family="terminus-2", version=harbor_version(), recipe=self.recipe.sha256, variant="plain",
                         agent="terminus-2", kwargs={}, model_name=MODEL, task_image=task_image,
                         prefix="benchbake-test", dataset="terminal-bench@2.0", tasks=[task], arms=["terminus-2"],
                         why="missing", expected_min=0.6, expected_gb=0.2)

    def build(self, items: list[BuildItem], **options: Any) -> dict[str, Any]:
        manager = BuildManager(root=self.root)
        started = manager.start(self.cfg, builder.BuildPlan(label="test", items=items), **options)
        self.assertTrue(manager.wait(60))
        self.manager = manager
        return manager.status(started["id"])

    def test_an_image_is_built_verified_and_only_then_named(self) -> None:
        result = self.build([self.item()])
        self.assertEqual(result["status"], "done", result)
        item = result["items"][0]
        self.assertEqual(item["status"], "promoted", item)
        final = self.docker.find(item["image"])
        self.assertIsNotNone(final)
        self.assertFalse([r for r in self.docker.refs() if r.startswith(builder.STAGING)])
        labels = final["Config"]["Labels"]
        self.assertEqual(labels[LABEL_NS + "recipe-sha256"], self.recipe.sha256)
        self.assertEqual(labels[LABEL_NS + "base-image-id"], self.base_id)
        self.assertEqual(labels[LABEL_NS + "build-id"], result["id"])
        self.assertNotIn(LABEL_NS + "noop", labels)
        build = self.docker.ran("build")[0]
        self.assertIn("--no-cache", build)
        self.assertNotIn("--pull", build)
        self.assertIn(f"BASE={FIX_GIT}", build)
        staging = build[build.index("-t") + 1]
        self.assertTrue(staging.startswith(f"{builder.STAGING}/{result['id']}-1/terminus-2-"), staging)
        dockerfile = Path(build[build.index("-f") + 1]).read_text(encoding="utf-8")
        self.assertIn("apt-get install -y tmux asciinema", dockerfile)
        # Tagged with the final name only after it was verified, and the staging tag dropped after.
        order = [c[1] for c in self.docker.calls if c[1] in ("build", "run", "tag", "rmi")]
        self.assertEqual(order[-2:], ["tag", "rmi"])
        self.assertLess(order.index("run"), order.index("tag"))
        # The setup check ran through the staging name, with the arm's budget and the recipe's skips.
        self.assertEqual(len(self.setup_calls), 1)
        self.assertEqual(self.setup_calls[0]["prefix"], f"{builder.STAGING}/{result['id']}-1")
        self.assertEqual(self.setup_calls[0]["budget_sec"], adapters.Terminus2Adapter.setup_budget_sec)
        self.assertEqual(self.setup_calls[0]["expected_skips"], 0)
        self.assertEqual(len(self.equivalence_calls), 1)
        lines = [json.loads(x) for x in (self.root / "builds.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(r["status"], r["image"]) for r in lines], [("promoted", item["image"])])
        self.assertTrue(lines[0]["fingerprint_sha256"])
        self.assertEqual(lines[0]["os_family"], "debian-12")
        fingerprints = list((self.root / "fingerprints").glob("*.txt"))
        self.assertEqual(len(fingerprints), 1)
        self.assertIn("promoted to", self.manager.log_tail(result["id"], 1))

    def test_a_failed_check_removes_only_its_own_staging_tag(self) -> None:
        other = self.docker.add_image("benchbake-test/keep/me:1")

        def smoke_fails(args: list[str], stdin: str | None) -> tuple[int, str, str]:
            return 1, "", "tmux: command not found"

        self.docker.run_handlers.append((lambda a, s: s is not None and s.startswith("set -o pipefail"), smoke_fails))
        result = self.build([self.item()])
        self.assertEqual(result["status"], "failed")
        item = result["items"][0]
        self.assertEqual(item["status"], "failed")
        self.assertIn("smoke test failed", item["message"])
        self.assertIsNone(self.docker.find(item["image"]))
        self.assertFalse([r for r in self.docker.refs() if r.startswith(builder.STAGING)])
        self.assertIn(other, self.docker.images)
        self.assertEqual(self.setup_calls, [])

    def test_a_failed_setup_check_blocks_promotion(self) -> None:
        self.setup_result = {"ok": False, "agent_setup_sec": 400.0,
                             "problems": ["agent setup took 400 s, over this harness's 15 s budget"]}
        result = self.build([self.item()])
        item = result["items"][0]
        self.assertEqual(item["status"], "failed")
        self.assertIn("setup check failed", item["message"])
        self.assertIsNone(self.docker.find(item["image"]))

    def test_one_equivalence_check_per_recipe_and_os_family_gates_its_group(self) -> None:
        self.docker.add_image("alexgshaw/other:1", layers=("sha256:o1",))
        probes = CannedProbes()
        probes.data["alexgshaw/other:1"] = probes.data[FIX_GIT]  # another Debian 12 image
        self.patch(recorder, "PROBES", probes)
        self.equivalence_ok = False
        result = self.build([self.item(), self.item("alexgshaw/other:1", "other")], jobs=2)
        self.assertEqual(len(self.equivalence_calls), 1)
        self.assertEqual([i["status"] for i in result["items"]], ["failed", "failed"])
        self.assertTrue(all("equivalence" in i["message"] for i in result["items"]))
        self.assertFalse([r for r in self.docker.refs() if r.startswith(("benchbake-test/", builder.STAGING))])
        record = [json.loads(x) for x in (self.root / "verify.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(r["kind"], r["ok"], r["os_family"]) for r in record], [("equivalence", False, "debian-12")])

    def test_a_crashed_equivalence_check_still_gives_its_group_a_verdict(self) -> None:
        self.docker.add_image("alexgshaw/other:1", layers=("sha256:o1",))
        probes = CannedProbes()
        probes.data["alexgshaw/other:1"] = probes.data[FIX_GIT]
        self.patch(recorder, "PROBES", probes)
        self.patch(verify, "equivalence", mock.Mock(side_effect=RuntimeError("docker went away")))
        result = self.build([self.item(), self.item("alexgshaw/other:1", "other")], jobs=2)
        self.assertEqual([i["status"] for i in result["items"]], ["failed", "failed"])
        self.assertTrue(any("docker went away" in i["message"] for i in result["items"]))
        self.assertFalse([r for r in self.docker.refs() if r.startswith(("benchbake-test/", builder.STAGING))])

    def test_a_passed_equivalence_check_is_not_repeated_and_setup_runs_once_per_os(self) -> None:
        self.build([self.item()])
        self.equivalence_calls.clear()
        self.setup_calls.clear()
        self.docker.add_image("alexgshaw/other:1", layers=("sha256:o1",))
        probes = CannedProbes()
        probes.data["alexgshaw/other:1"] = probes.data[FIX_GIT]
        self.patch(recorder, "PROBES", probes)
        result = self.build([self.item("alexgshaw/other:1", "other"),
                             self.item("alexgshaw/multi-source-data-merger:20251031", "multi-source-data-merger")])
        self.assertEqual([i["status"] for i in result["items"]], ["promoted", "promoted"], result["items"])
        self.assertEqual(len(self.equivalence_calls), 1)  # only Debian 13 is new
        self.assertEqual(self.equivalence_calls[0]["task_image"], "alexgshaw/multi-source-data-merger:20251031")
        self.assertEqual(len(self.setup_calls), 2)           # auto: one per harness and OS family
        noop = next(i for i in result["items"] if i["task_image"].startswith("alexgshaw/multi"))
        self.assertTrue(noop["noop"])
        labels = self.docker.find(noop["image"])["Config"]["Labels"]
        self.assertEqual(labels[LABEL_NS + "noop"], "true")

    def test_options_turn_the_checks_off_or_on_for_every_image(self) -> None:
        self.build([self.item()], setup="none", equivalence="none")
        self.assertEqual((self.setup_calls, self.equivalence_calls), ([], []))
        self.docker.calls.clear()
        self.build([self.item()], setup="all", equivalence="always")
        self.assertEqual((len(self.setup_calls), len(self.equivalence_calls)), (1, 1))

    def test_cancel_stops_the_build_and_removes_the_staging_tag(self) -> None:
        self.docker.build_release = threading.Event()
        killed: list[int] = []
        self.patch(procs, "kill_tree", lambda pid, timeout=30: (killed.append(pid), self.docker.build_release.set()))
        manager = BuildManager(root=self.root)
        started = manager.start(self.cfg, builder.BuildPlan(label="test", items=[self.item()]))
        deadline = time.monotonic() + 30
        while (manager.status(started["id"])["items"][0]["status"] != "building" and time.monotonic() < deadline):
            time.sleep(0.05)
        manager.cancel(started["id"])
        self.assertTrue(manager.wait(30))
        result = manager.status(started["id"])
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["items"][0]["status"], "cancelled")
        self.assertEqual(killed, [0])
        self.assertFalse([r for r in self.docker.refs() if r.startswith(("benchbake-test/", builder.STAGING))])

    def test_a_recipe_that_changed_since_planning_is_not_built(self) -> None:
        item = dataclasses.replace(self.item(), recipe="e" * 64)
        result = self.build([item])
        self.assertIn("install changed", result["items"][0]["message"])
        self.assertEqual(self.docker.ran("build"), [])

    def test_the_disk_check_refuses_before_anything_starts(self) -> None:
        self.patch(builder, "host_free_gb", lambda path: 10.0)
        manager = BuildManager(root=self.root)
        with self.assertRaises(BuildRefused) as caught:
            manager.start(self.cfg, builder.BuildPlan(label="test", items=[self.item()]))
        self.assertIn("Not enough disk space", str(caught.exception))
        self.assertTrue(manager.is_idle())
        self.assertFalse(builder.build_active())
        self.assertEqual(self.docker.ran("build"), [])

    def test_a_missing_task_image_is_pulled_first(self) -> None:
        image = "alexgshaw/pulled:1"
        self.docker.pullable[image] = {"Id": "sha256:" + "9" * 64, "RepoTags": [image], "Os": "linux", "Size": 1,
                                       "Config": {"Labels": {}}, "RootFS": {"Layers": ["sha256:p1"]}}
        manager = BuildManager(root=self.root)
        started = manager.start_pull(self.cfg, [image, FIX_GIT])
        self.assertTrue(manager.wait(30))
        result = manager.status(started["id"])
        self.assertEqual((result["kind"], result["status"]), ("pull", "done"))
        self.assertEqual(result["pulled"], {"present": [FIX_GIT], "pulled": [image], "failed": {}})
        self.assertEqual(self.docker.ran("pull"), [["pull", image]])

    def test_history_marks_a_build_whose_process_died_as_interrupted(self) -> None:
        manager = BuildManager(root=self.root)
        build = builder.Build(id="b1", kind="build", label="x", created_at="2026-10-29T00:00:00+00:00", jobs=1,
                              options={}, items=[builder.ItemState(n=1, family="terminus-2", version="v", recipe="r",
                                                                   task_image=FIX_GIT, image="i", tasks=[], arms=[],
                                                                   status="building")])
        with builder.build_lock():  # another process is building
            manager._save(build)
            manager.refresh()
            self.assertEqual(manager.status("b1")["status"], "running")
        again = BuildManager(root=self.root)
        status = again.status("b1")
        self.assertEqual((status["status"], status["items"][0]["status"]), ("interrupted", "cancelled"))


class Pulls(BuilderTestCase):
    def test_only_missing_images_are_pulled_and_failures_reported(self) -> None:
        self.docker.pull_rc = 1
        result = pull.ensure_local([FIX_GIT, "alexgshaw/absent:1"], io.StringIO())
        self.assertEqual(result["present"], [FIX_GIT])
        self.assertIn("alexgshaw/absent:1", result["failed"])
        self.assertEqual(self.docker.ran("pull"), [["pull", "alexgshaw/absent:1"]])

    def test_cancel_kills_the_pull(self) -> None:
        self.docker.pull_release = threading.Event()
        cancel = threading.Event()
        cancel.set()
        self.patch(procs, "kill_tree", lambda pid, timeout=30: self.docker.pull_release.set())
        with self.assertRaises(pull.PullCancelled):
            pull.ensure_local(["alexgshaw/absent:1"], None, cancel)


class Cli(BuilderTestCase):
    def run_cli(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr", out), \
                mock.patch.object(cli, "load", lambda args: self.cfg):
            code = cli.main(list(argv))
        return code, out.getvalue()

    def test_build_dry_run_shows_the_plan_and_the_refusal(self) -> None:
        self.fake_recipes()
        self.patch(builder, "docker_disk", lambda: (Path("C:/docker_data.vhdx"), "Docker Desktop's default location"))
        self.patch(builder, "host_free_gb", lambda path: 9.2)
        code, out = self.run_cli("build", "--mode", "pilot", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("4 image(s) to build", out)
        self.assertIn("Disk check: REFUSED", out)
        self.assertIn("prebake.host_reserve_gb (25 GB)", out)
        self.assertIn("pi: not pre-baked, runs stock setup", out)
        self.assertEqual(self.docker.ran("build"), [])
        self.assertFalse((self.root / "builds").exists())

    def test_status_lists_each_arms_images(self) -> None:
        self.fake_recipes()
        self.baked("claude-code")
        code, out = self.run_cli("status", "--mode", "pilot", "--arms", "claude-code")
        self.assertEqual(code, 0)
        self.assertIn("claude-code", out)
        self.assertIn("1/2", out)

    def test_build_without_yes_outside_a_terminal_changes_nothing(self) -> None:
        self.fake_recipes()
        self.patch(builder, "docker_disk", lambda: (None, "not found"))
        self.patch(builder, "host_free_gb", lambda path: 500.0)
        with mock.patch("sys.stdin", io.StringIO()):
            code, out = self.run_cli("build", "--mode", "pilot", "--arms", "terminus-2")
        self.assertEqual(code, 1)
        self.assertIn("pass --yes", out)
        self.assertEqual(self.docker.ran("build"), [])

    def test_a_bad_selection_is_an_error_not_a_traceback(self) -> None:
        code, out = self.run_cli("status", "--design", "not a design")
        self.assertEqual(code, 2)
        self.assertIn("ConfigError", out)


if __name__ == "__main__":
    unittest.main()
