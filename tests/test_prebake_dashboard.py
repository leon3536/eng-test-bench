"""Pre-baked images where runs are planned, run, queued and watched: the
plan's per-arm images block with its problems and warnings, the run manager's
run lock, pre-pull, re-check and fallback count, the queue's image gate and
auto-build, the readiness rows, and the dashboard's /api/prebake routes.

Docker is FakeDocker (tests/fakes_docker.py) or a stub, and every recipe is
fixed (fake_resolve), so no Harbor agent code is loaded and nothing is built
or pulled.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

from tests.fakes_docker import FakeDocker
from tests.test_prebake_builder import FIX_GIT, REGEX_LOG, BuilderTestCase, fake_resolve
from tests.test_runqueue import QUEUE_YAML, TASKS, QueueTestCase
from tests.test_runs import RunsTestCase
from tests.test_server import ServerCase

from bench import checks, datasets, designs, estimates, prebake, procs, runqueue, runs, server, spec
from bench.prebake import ArmPrebake, builder, inventory, prune, pull
from bench.prebake.builder import BuildRefused
from bench.prebake.inventory import ArmReadiness, DockerError
from bench.settings import Config


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def with_run(yaml_text: str, *lines: str) -> str:
    """yaml_text with lines added to its run: section."""
    return yaml_text.replace("run:\n", "run:\n" + "".join(f"  {line}\n" for line in lines), 1)


# The plan --------------------------------------------------------------------------


class PlanImagesTests(BuilderTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.fake_recipes()
        self.patch(prebake, "resolve", fake_resolve)
        self.patch(datasets, "info", lambda dataset: None)
        self.patch(datasets, "task_meta", lambda dataset: {})
        self.patch(checks, "docker_capacity", lambda: None)
        self.patch(runs, "_long_paths_enabled", lambda: True)
        self.cfg = self.config()

    def config(self, *run_lines: str, tasks: list[str] | None = None) -> Config:
        from tests.test_prebake_builder import CONFIG

        # These pin the arms schedule, where images are built before a run; tests/test_runs_race.py the races.
        if not any(line.startswith("schedule:") for line in run_lines):
            run_lines = (*run_lines, "schedule: arms")
        return self.load_test_config(with_run(CONFIG, *run_lines), tasks=tasks or ["fix-git", "regex-log", "no-image"])

    def plan(self, arms: list[str], cfg: Config | None = None) -> dict[str, Any]:
        cfg = cfg or self.cfg
        return runs.plan_spec(cfg, spec.spec_from_config(cfg, "pilot", arms))

    def test_missing_images_warn_under_prefer_and_keep_the_setup_net(self) -> None:
        p = self.plan(["claude-code", "pi"])
        cc, pi = p["arms"]
        images = cc["prebake"]["images"]
        self.assertEqual((images["tasks"], images["ready"], images["missing"], images["all_ready"]),
                         (2, [], ["fix-git", "regex-log"], False))
        self.assertEqual((images["build_min"], images["build_gb"]), (22.0, 3.8))
        self.assertEqual(cc["prebake"]["setup_timeout_multiplier"], 2.0)
        self.assertTrue(cc["command"].endswith("--agent-setup-timeout-multiplier 2"), cc["command"])
        self.assertEqual(p["problems"], [])
        self.assertEqual([w for w in cc["warnings"] if "pre-baked" in w],
                         ["2 of 2 trials will install the harness themselves (stock setup, about 330 s each): no "
                          "ready pre-baked image (2 missing). Build missing images first to avoid it"])
        self.assertEqual(cc["setup_warnings"], [w for w in cc["warnings"] if "pre-baked" in w])
        # pi pins no version: it runs stock setup, says why, and needs no image.
        self.assertEqual((pi["prebake"]["supported"], "images" in pi["prebake"]), (False, False))
        self.assertIn("no pinned version", pi["prebake"]["reason"])
        top = p["prebake"]
        self.assertEqual((top["mode"], top["images"], top["build_min"], top["build_gb"], top["jobs"]),
                         ("prefer", 2, 22.0, 4.37, 4))
        self.assertEqual((top["task_images"], top["pull"], top["error"]), (2, [], None))

    def test_every_image_ready_drops_the_net_and_plans_pre_baked_minutes(self) -> None:
        self.baked("claude-code", FIX_GIT, baked_at=now())
        self.baked("claude-code", REGEX_LOG, baked_at=now())
        cc = self.plan(["claude-code"])["arms"][0]
        block = cc["prebake"]
        self.assertTrue(block["images"]["all_ready"])
        self.assertEqual(block["images"]["ready"], ["fix-git", "regex-log"])
        self.assertIsNotNone(block["images"]["oldest_baked_at"])
        self.assertEqual((block["expect_stock"], block["setup_timeout_multiplier"]), (False, None))
        self.assertNotIn("--agent-setup-timeout-multiplier", cc["command"])
        self.assertEqual([w for w in cc["warnings"] if "pre-baked" in w or "install" in w], [])
        self.assertEqual(estimates.expected_setup_mode(block), "prebaked")

    def test_a_stale_image_says_why_and_old_bakes_warn(self) -> None:
        self.baked("claude-code", FIX_GIT, **{"harbor-version": "0.1.0"})
        self.baked("claude-code", REGEX_LOG, baked_at="2026-01-01T00:00:00Z")
        cc = self.plan(["claude-code"])["arms"][0]
        images = cc["prebake"]["images"]
        self.assertEqual(list(images["stale"]), ["fix-git"])
        self.assertIn("labels differ: harbor-version", images["stale"]["fix-git"])
        self.assertEqual((images["ready"], images["over_age"]), (["regex-log"], ["regex-log"]))
        self.assertTrue(any("baked more than 14 days ago" in w for w in cc["warnings"]), cc["warnings"])
        self.assertTrue(any("1 of 2 trials will install" in w and "1 stale" in w for w in cc["warnings"]))

    def test_require_makes_images_that_are_not_ready_a_problem_a_design_may_still_be_queued_with(self) -> None:
        cfg = self.config("prebake: require")
        p = self.plan(["claude-code"], cfg)
        expected = ("claude-code: run.prebake is require, but 2 of 2 pre-baked images are not ready (0 of 2 ready; "
                    "2 missing). Build them first: Build missing on LAUNCH, or python -m bench.prebake build")
        self.assertEqual((p["problems"], p["image_problems"]), ([expected], [expected]))
        self.assertIsNone(p["arms"][0]["prebake"]["setup_timeout_multiplier"])  # a missing image fails fast
        check = designs.validate(cfg, designs.design_from_config(cfg, "pilot", ["claude-code"]))
        self.assertEqual((check["status"], check["problems"]), ("valid", []))
        self.assertIn(f"{expected}. The queue holds this design until they are ready", check["warnings"])
        # An arm that can never have an image is a real problem under require.
        p = self.plan(["pi"], cfg)
        self.assertEqual(p["image_problems"], [])
        self.assertIn("this arm cannot use pre-baked images", p["problems"][0])

    def test_docker_not_answering_warns_under_prefer_and_blocks_require(self) -> None:
        def down(refresh: bool = False) -> dict[str, str]:
            raise DockerError("docker image ls did not answer within 60 s")

        self.patch(inventory, "local_images", down)
        cc = self.plan(["claude-code"])["arms"][0]
        self.assertEqual(cc["prebake"]["images"]["error"], "docker image ls did not answer within 60 s")
        self.assertTrue(any(w.startswith("The pre-baked images could not be checked") for w in cc["warnings"]))
        p = self.plan(["claude-code"], self.config("prebake: require"))
        self.assertEqual(p["problems"], ["claude-code: run.prebake is require, but the pre-baked images could not be "
                                         "checked: docker image ls did not answer within 60 s"])

    def test_claude_code_installing_itself_under_parallel_load_warns(self) -> None:
        cc = self.plan(["claude-code"], self.config("n_concurrent: 4"))["arms"][0]
        self.assertTrue(any(w.startswith("Claude Code installs itself in 2 trial(s)") and "up to 4 containers" in w
                            for w in cc["warnings"]), cc["warnings"])
        self.assertFalse(any("Claude Code installs itself" in w for w in self.plan(["claude-code"])["arms"][0]
                             ["warnings"]))  # 2 at once

    def test_task_images_not_local_are_listed_for_the_pull(self) -> None:
        cfg = self.config(tasks=["other", "fix-git", "regex-log"])
        p = self.plan(["claude-code"], cfg)
        self.assertEqual(p["prebake"]["pull"], ["alexgshaw/other:1"])
        self.assertEqual(p["arms"][0]["prebake"]["images"]["missing"], ["other", "fix-git"])
        self.assertTrue(any(w.startswith("1 task image(s) are not on this machine yet. The run pulls them")
                            for w in p["warnings"]), p["warnings"])

    def test_off_checks_no_image_and_run_json_records_what_the_plan_saw(self) -> None:
        p = self.plan(["claude-code"], self.config("prebake: 'off'"))
        self.assertEqual(p["arms"][0]["prebake"], {"mode": "off", "setup_timeout_multiplier": None,
                                                   "env_build_timeout_multiplier": 1.0})
        self.assertEqual(self.docker.ran("image", "inspect"), [])
        self.baked("claude-code", FIX_GIT, baked_at=now())
        inventory.forget_images()  # as a build does
        run_spec = spec.spec_from_config(self.cfg, "pilot", ["claude-code"])
        found = runs.image_readiness(self.cfg, run_spec)
        planned = runs.plan_spec(self.cfg, run_spec, found)
        resolved = runs.resolve_arms(self.cfg, run_spec, "20261029-120000-pilot", found)
        self.assertEqual(resolved[0]["prebake"], planned["arms"][0]["prebake"])
        self.assertEqual(resolved[0]["prebake"]["images"]["ready"], ["fix-git"])

    def test_setup_note_counts_fallbacks_and_fast_failures(self) -> None:
        records = [{"used": True}, {"used": False, "status": "missing", "required": False},
                   {"used": False, "status": "stale", "required": False},
                   {"used": False, "status": "missing", "required": True}]
        self.assertEqual(runs.setup_note(records), (3, "2 trials ran stock setup (no pre-baked image x1, stale image "
                                                       "x1); 1 trial failed fast: no usable pre-baked image "
                                                       "(run.prebake require)"))
        self.assertEqual(runs.setup_note(records[:2]), (1, "1 trial ran stock setup (no pre-baked image)"))
        self.assertEqual(runs.setup_note([{"used": True}]), (0, ""))


# The run manager ----------------------------------------------------------------------


def terminus_resolve(cfg: Any, run: Any, arm: Any, kwargs: Any) -> ArmPrebake | None:
    """Arm a (terminus-2) can be pre-baked; the others pin nothing."""
    if run.prebake == "off":
        return None
    if arm.family != "terminus-2":
        return ArmPrebake(mode=run.prebake, family=arm.family, supported=False, reason="no pinned version")
    return ArmPrebake(mode=run.prebake, family="terminus-2", supported=True, version="0.25.0", recipe="b" * 64,
                      variant="plain", prefix=cfg.prebake.prefix)


class RunManagerImagesTests(RunsTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.prebake_dir = self.tmpdir() / "prebake"
        self.pulled: list[list[str]] = []
        for target, name, value in (
                (builder, "PREBAKE_DIR", self.prebake_dir),
                (builder, "LOCK_WAIT_SEC", 0),
                (prebake, "resolve", terminus_resolve),
                (inventory, "resolve_prebake", terminus_resolve),
                (inventory, "task_images", lambda dataset, tasks: {t: f"alexgshaw/{t}:1" for t in tasks}),
                (inventory, "local_images", lambda refresh=False: {}),
                (pull, "ensure_local", self.fake_pull)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_pull(self, images: list[str], log: Any = None, cancel: Any = None, jobs: int = 2) -> dict[str, Any]:
        self.pulled.append(list(images))
        return {"present": [], "pulled": list(images), "failed": {}}

    def test_a_run_refuses_to_start_while_a_build_holds_the_build_lock(self) -> None:
        cfg = self.config()
        manager = self.manager()
        with procs.file_lock(builder.build_lock_path(), timeout=0):
            with self.assertRaises(BuildRefused) as caught:
                manager.start(cfg, "pilot", ["a"])
        self.assertIn("A pre-baked image build is running", str(caught.exception))
        self.assertEqual((manager.list(), manager.is_idle()), ([], True))
        self.assertFalse(builder._held(builder.run_lock_path()), "a refused start keeps no run lock")
        manager.start(cfg, "pilot", ["a"])
        self.assertEqual(self.finish(manager)["status"], "done")

    def test_a_build_refuses_while_a_run_holds_the_run_lock_and_may_start_after(self) -> None:
        self.scenarios["a"] = "hang"
        manager = self.manager()
        started = manager.start(self.config(), "pilot", ["a"])
        self.wait_for(lambda: manager.get(started["id"])["arms"][0]["status"] == "running")
        with self.assertRaises(BuildRefused) as caught:
            with builder.build_lock():
                pass
        self.assertIn("a run of profile default", str(caught.exception))
        manager.cancel(started["id"])
        self.finish(manager)
        with builder.build_lock():
            pass  # the run is over, so the lock is free

    def test_missing_task_images_are_pulled_before_the_first_arm(self) -> None:
        self.patch_pull_missing(lambda images, refresh=True: list(images))
        manager = self.manager()
        manager.start(self.config(), "pilot", ["a", "b"])
        run = self.finish(manager)
        self.assertEqual(self.pulled, [["alexgshaw/task-a:1"]])
        self.assertEqual((run["status"], run["message"]), ("done", ""))
        self.assertEqual(self.calls(), ["a", "dry", "a", "real", "b", "dry", "b", "real"])

    def test_cancel_during_the_pull_cancels_the_run_before_any_arm(self) -> None:
        def cancelled(images: list[str], log: Any = None, cancel: Any = None, jobs: int = 2) -> dict[str, Any]:
            raise pull.PullCancelled("Cancelled while pulling task images")

        self.patch_pull_missing(lambda images, refresh=True: list(images))
        with mock.patch.object(pull, "ensure_local", cancelled):
            manager = self.manager()
            manager.start(self.config(), "pilot", ["a"])
            run = self.finish(manager)
        self.assertEqual((run["status"], run["message"]), ("cancelled", "Cancelled while pulling task images"))
        self.assertEqual([a["status"] for a in run["arms"]], ["cancelled"])
        self.assertEqual(self.calls(), [])

    def test_no_pull_when_prepull_is_off_or_every_image_is_local(self) -> None:
        self.patch_pull_missing(lambda images, refresh=True: [])
        manager = self.manager()
        manager.start(self.config(), "pilot", ["a"])
        run = self.finish(manager)
        self.assertEqual((self.pulled, run["status"]), ([], "done"))
        self.assertIn("are local", (self.root / "runs" / run["id"] / "images.log").read_text(encoding="utf-8"))

    def test_the_arm_log_records_its_images_right_before_launch(self) -> None:
        manager = self.manager()
        started = manager.start(self.config(), "pilot", ["a", "b"])
        run = self.finish(manager)
        log = self.arm_log(run["id"], "a")
        self.assertIn("pre-baked images: 0 of 1 ready; 1 missing. These trials run Harbor's stock install: task-a",
                      log)
        self.assertIn('harness setup: {"mode": "prefer", "family": "terminus-2", "supported": true', log)
        self.assertNotIn("pre-baked images:", self.arm_log(run["id"], "b"))  # unsupported: nothing to check
        self.assertEqual(started["resolved_arms"][0]["prebake"]["images"]["missing"], ["task-a"])

    def test_trials_that_ran_stock_setup_are_counted_in_the_arm_message(self) -> None:
        records = {"t1": {"used": False, "status": "missing", "required": False, "reason": "no pre-baked image x"}}
        with mock.patch("bench.extract.read_prebake_record", lambda folder: records.get(Path(folder).name)):
            manager = self.manager()
            manager.start(self.config(), "pilot", ["a"])
            run = self.finish(manager)
        arm = run["arms"][0]
        self.assertEqual((arm["setup_fallbacks"], arm["setup_note"]),
                         (1, "1 trial ran stock setup (no pre-baked image)"))
        self.assertTrue(arm["message"].endswith("1 trial ran stock setup (no pre-baked image)."), arm["message"])

    def patch_pull_missing(self, fn: Any) -> None:
        patcher = mock.patch.object(pull, "missing", fn)
        patcher.start()
        self.addCleanup(patcher.stop)


# The queue ------------------------------------------------------------------------------


class FakeImages:
    """runqueue.ImageHook: images to build until a build ends; a build runs for `polls` status reads."""

    def __init__(self, todo: int = 0, blocking: list[str] | None = None, polls: int = 2,
                 build_error: Exception | None = None) -> None:
        self.todo, self.blocking, self.polls, self.build_error = todo, list(blocking or []), polls, build_error
        self.checks = 0
        self.builds: list[str] = []
        self.forever = False

    def check(self, cfg: Config, run_spec: Any) -> dict[str, Any]:
        self.checks += 1
        return {"todo": self.todo, "blocking": list(self.blocking)}

    def build(self, cfg: Config, run_spec: Any) -> str:
        if self.build_error is not None:
            raise self.build_error
        self.builds.append(run_spec.label)
        return "b-1"

    def status(self, build_id: str) -> dict[str, Any]:
        if self.forever or self.polls > 0:
            self.polls -= 1
            return {"id": build_id, "status": "running", "items": [{}, {}], "counts": {"promoted": 1},
                    "finished_at": None}
        self.todo, self.blocking = 0, []
        return {"id": build_id, "status": "done", "items": [{}, {}], "counts": {"promoted": 2}, "finished_at": now()}


class QueueImageGateTests(QueueTestCase):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.object(runqueue, "WAIT_SECONDS", 0.01)
        patcher.start()
        self.addCleanup(patcher.stop)

    def started_queue(self, images: FakeImages, mode: str = "require") -> tuple[runqueue.RunQueue, dict[str, Any]]:
        q = self.queue(images=images)
        doc = self.design(run={"budget_usd": 3.0, "prebake": mode})
        item = self.add(q, doc)
        q.start()
        return q, item

    def test_require_holds_an_item_whose_images_are_missing(self) -> None:
        images = FakeImages(todo=2, blocking=["cc: 0 of 2 ready; 2 missing"])
        q, item = self.started_queue(images)
        q._step()
        held = self.item(q, item["id"])
        self.assertEqual(held["status"], "held")
        self.assertIn("run.prebake is require and its pre-baked images are not ready: cc: 0 of 2 ready; 2 missing",
                      held["message"])
        self.assertIn(f"python -m bench.prebake build --design {item['design_id']}@v{item['version']}",
                      held["message"])
        self.assertEqual((self.manager.started, images.builds), ([], []))

    def test_require_holds_a_task_race_only_for_an_arm_that_can_never_have_an_image(self) -> None:
        images = FakeImages(blocking=["cc cannot use pre-baked images (no pinned version)"])
        images.check = lambda cfg, run_spec: {"todo": 0, "blocking": list(images.blocking),  # type: ignore
                                              "race": True}
        q, item = self.started_queue(images)
        q._step()
        held = self.item(q, item["id"])
        self.assertEqual(held["status"], "held")
        self.assertIn("some of its arms can never use a pre-baked image: cc cannot use pre-baked images "
                      "(no pinned version)", held["message"])
        self.assertNotIn("Build missing", held["message"])
        self.assertEqual((self.manager.started, images.builds), ([], []))

    def test_prefer_starts_and_falls_back_without_auto_build(self) -> None:
        images = FakeImages(todo=2)
        q, item = self.started_queue(images, mode="prefer")
        q._step()
        self.assertEqual((self.item(q, item["id"])["status"], len(self.manager.started)), ("running", 1))
        self.assertEqual(images.builds, [])

    def test_off_never_asks(self) -> None:
        images = FakeImages(todo=2, blocking=["x"])
        q, item = self.started_queue(images, mode="off")
        q._step()
        self.assertEqual((images.checks, len(self.manager.started)), (0, 1))

    def test_auto_build_builds_first_then_starts(self) -> None:
        self.cfg = self.load_test_config(QUEUE_YAML + "prebake:\n  auto_build: true\n", tasks=TASKS)
        images = FakeImages(todo=2, blocking=["cc: 0 of 2 ready; 2 missing"])
        seen: list[str] = []
        real = images.status

        def watched(build_id: str) -> dict[str, Any]:
            seen.append(str(q.view()["active"]["message"]) if q.view()["active"] else "")
            return real(build_id)

        images.status = watched  # type: ignore[method-assign]
        q, item = self.started_queue(images)
        q._step()
        self.assertEqual(len(images.builds), 1)
        self.assertTrue(any(m.startswith("Building its pre-baked images first: 1 of 2 finished (build b-1)")
                            for m in seen), seen)
        active = self.item(q, item["id"])
        self.assertEqual((active["status"], active["message"], len(self.manager.started)), ("running", "", 1))

    def test_a_pause_during_the_build_puts_the_item_back_and_leaves_the_build_running(self) -> None:
        self.cfg = self.load_test_config(QUEUE_YAML + "prebake:\n  auto_build: true\n", tasks=TASKS)
        images = FakeImages(todo=2)
        images.forever = True
        q, item = self.started_queue(images, mode="prefer")
        worker = threading.Thread(target=q._step)
        worker.start()
        self.assertTrue(self.wait_building(q))
        q.pause()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        waiting = self.item(q, item["id"])
        self.assertEqual(waiting["status"], "waiting")
        self.assertIn("build b-1 goes on", waiting["message"])
        self.assertEqual(self.manager.started, [])

    def test_a_build_that_cannot_start_holds_under_require(self) -> None:
        self.cfg = self.load_test_config(QUEUE_YAML + "prebake:\n  auto_build: true\n", tasks=TASKS)
        refused = BuildRefused("Not enough disk space to build: the drive holding Docker's disk image is full")
        q, item = self.started_queue(FakeImages(todo=2, blocking=["cc: 0 of 2 ready"], build_error=refused))
        q._step()
        held = self.item(q, item["id"])
        self.assertEqual(held["status"], "held")
        self.assertIn("Its pre-baked images could not be built: Not enough disk space", held["message"])
        self.assertEqual(self.manager.started, [])

    def test_a_build_that_cannot_start_lets_prefer_run_on_stock_setup(self) -> None:
        self.cfg = self.load_test_config(QUEUE_YAML + "prebake:\n  auto_build: true\n", tasks=TASKS)
        refused = BuildRefused("Another build (from the CLI or the dashboard) is running")
        q, item = self.started_queue(FakeImages(todo=2, build_error=refused), mode="prefer")
        q._step()
        self.assertEqual((self.item(q, item["id"])["status"], len(self.manager.started)), ("running", 1))

    def test_waiting_specs_are_what_the_open_items_would_run(self) -> None:
        # A task race keeps the images an arms-schedule item waiting here will look for (runs.images_needed).
        q = self.queue(images=FakeImages())
        item = self.add(q, self.design(run={"budget_usd": 3.0, "schedule": "arms"}))
        specs = q.waiting_specs()
        self.assertEqual([(s.spec_hash, s.run.schedule) for s in specs], [(item["spec_hash"], "arms")])
        self.assertEqual(runs.images_needed(self.cfg, [dataclasses.replace(specs[0], arms=())]), set())

    def wait_building(self, q: runqueue.RunQueue) -> bool:
        from tests.test_runqueue import wait_for

        return wait_for(lambda: bool(q.view()["active"]) and "Building" in q.view()["active"]["message"])


# Readiness rows ---------------------------------------------------------------------------


class ReadinessRowTests(BuilderTestCase):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.dict(checks._cache, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.patch(builder, "last_vm", {})

    def disk(self, path: str | None, free: float, roomiest: tuple[str, float] | None = None) -> dict[str, Any]:
        self.patch(builder, "docker_disk", lambda: (Path(path) if path else None, "test"))
        self.patch(builder, "host_free_gb", lambda p: free)
        self.patch(checks, "_roomiest_drive", lambda besides: roomiest)
        return checks._docker_disk(self.cfg)

    def test_docker_disk_on_a_full_drive_says_what_is_wrong_and_how_to_fix_it(self) -> None:
        row = self.disk("C:/Users/x/AppData/Local/Docker/wsl/disk/docker_data.vhdx", 12.3, ("D:", 265.0))
        self.assertFalse(row["ok"])
        self.assertIn("Docker's disk image is on C:, which has only 12 GB free; keep at least 25 GB", row["detail"])
        self.assertIn("a full drive can stop Docker Desktop", row["detail"])
        self.assertIn("Settings > Resources > Advanced > Disk image location, move it to a drive with more room "
                      "such as D: (265 GB free), or free space on C:", row["detail"])

    def test_docker_disk_on_a_roomy_drive_is_ready_and_shows_the_last_vm_figure(self) -> None:
        self.assertEqual(self.disk("D:/Docker/DockerDesktopWSL/disk/docker_data.vhdx", 265.2),
                         {"ok": True, "detail": "Disk image on D:, 265 GB free"})
        self.patch(builder, "last_vm", {"free_gb": 869.0, "at": 1.0})
        self.assertEqual(self.disk("D:/Docker/disk/docker_data.vhdx", 265.2)["detail"],
                         "Disk image on D:, 265 GB free; Docker's own disk 869 GB free")
        self.patch(builder, "last_vm", {"free_gb": 20.0, "at": 1.0})
        row = self.disk("D:/Docker/disk/docker_data.vhdx", 265.2)
        self.assertFalse(row["ok"])
        self.assertIn("Docker's own disk has 20 GB free; keep at least 50 GB", row["detail"])

    def test_docker_disk_not_found_is_a_warning(self) -> None:
        row = self.disk(None, 0)
        self.assertEqual((row["ok"], row["warn"]), (True, True))

    def test_pre_baked_images_row(self) -> None:
        self.patch(prebake, "_loaded", threading.Event())
        self.assertIn("Recording each harness's install", checks._prebake_images(self.cfg)["detail"])
        prebake._loaded.set()
        found = {"claude-code": ArmReadiness(arm="claude-code", family="claude-code", mode="prefer", supported=True,
                                             tasks=3, ready=["a"], missing=["b", "c"]),
                 "pi": ArmReadiness(arm="pi", family="pi", mode="prefer", supported=False, reason="no pinned version")}
        self.patch(inventory, "readiness", lambda cfg, run_spec, refresh=False: found)
        tasks = ["fix-git", "regex-log", "no-image"]
        arms = self.load_test_config(with_run(self.cfg_yaml(), "schedule: arms"), tasks=tasks)
        row = checks._prebake_images(arms)
        self.assertEqual((row["ok"], row["warn"]), (True, True))
        self.assertTrue(row["detail"].startswith("1 of 3 ready for the full mode (claude-code 1/3). Trials without"))
        self.assertIn("Build missing on LAUNCH", row["detail"])
        required = self.load_test_config(with_run(self.cfg_yaml(), "prebake: require", "schedule: arms"), tasks=tasks)
        self.assertFalse(checks._prebake_images(required)["ok"])
        # A task race builds each task's images itself, so LAUNCH's Build missing (hidden then) is not the advice.
        for cfg in (self.cfg, self.load_test_config(with_run(self.cfg_yaml(), "prebake: require"), tasks=tasks)):
            row = checks._prebake_images(cfg)
            self.assertEqual((row["ok"], row["warn"]), (True, False))
            self.assertEqual(row["detail"], "1 of 3 ready for the full mode (claude-code 1/3). Runs are task races "
                                            "(run.schedule tasks): each task's images are built just before its race")
        found["claude-code"].ready, found["claude-code"].missing = ["a", "b", "c"], []
        self.assertEqual(checks._prebake_images(self.cfg),
                         {"ok": True, "warn": False, "detail": "All 3 ready for the full mode (claude-code 3/3)"})
        off = self.load_test_config(with_run(self.cfg_yaml(), "prebake: 'off'"), tasks=tasks)
        self.assertTrue(checks._prebake_images(off)["detail"].startswith("Off (run.prebake)"))

    def test_rows_come_before_the_assistant(self) -> None:
        self.patch(checks, "_docker", lambda: {"ok": True, "detail": "Docker Engine 29"})
        self.patch(checks, "_git_fix", lambda: {"ok": True, "detail": "fixed"})
        self.patch(checks, "_snapshot_detail", lambda dataset: "snapshot")
        self.patch(checks, "_prebake_images", lambda cfg: {"ok": True, "detail": "images"})
        self.patch(checks, "_docker_disk", lambda cfg: {"ok": True, "detail": "disk"})
        names = [r["name"] for r in checks.readiness(self.cfg)]
        self.assertEqual(names[-3:], ["Pre-baked images", "Docker disk", "Assistant"])

    def cfg_yaml(self) -> str:
        from tests.test_prebake_builder import CONFIG

        return CONFIG


# The dashboard's routes --------------------------------------------------------------------


def server_resolve(cfg: Any, run: Any, arm: Any, kwargs: Any) -> ArmPrebake | None:
    if run.prebake == "off":
        return None
    if arm.family != "claude-code":
        return ArmPrebake(mode=run.prebake, family=arm.family, supported=False, reason="no pinned version")
    return ArmPrebake(mode=run.prebake, family="claude-code", supported=True, version="2.1.312", recipe="a" * 64,
                      variant="plain", prefix=cfg.prebake.prefix)


class PrebakeRouteTests(ServerCase):
    def setUp(self) -> None:
        super().setUp()
        self.docker_fake = FakeDocker()
        self.allow_subprocess(self.docker_fake.matches, self.docker_fake)
        self.dataset = self.tmpdir() / "tasks"
        for task in ("task-a", "task-b", "task-c", "task-d"):
            (self.dataset / task).mkdir(parents=True)
            (self.dataset / task / "task.toml").write_text(
                f'[environment]\ndocker_image = "alexgshaw/{task}:1"\n', encoding="utf-8")
        for task in ("task-a", "task-b"):
            self.docker_fake.add_image(f"alexgshaw/{task}:1")
        self.prebake_dir = self.tmpdir() / "prebake"
        for target, name, value in ((datasets, "path", lambda dataset: self.dataset),
                                    (prebake, "resolve", server_resolve),
                                    (inventory, "resolve_prebake", server_resolve),
                                    (builder, "PREBAKE_DIR", self.prebake_dir),
                                    (builder, "LOCK_WAIT_SEC", 0)):
            self.patch(target, name, value)
        inventory.forget_images()
        self.addCleanup(inventory.forget_images)
        self.state.builds = builder.BuildManager(root=self.prebake_dir,
                                                 run_active=lambda: not self.state.manager.is_idle())

    def test_status_shows_what_a_launch_selection_needs(self) -> None:
        out = self.ok(self.client.get("/api/prebake/status", params={"mode": "sample", "arms": "claude-code,pi"}))
        cc = out["readiness"]["claude-code"]
        self.assertEqual(cc["missing"], ["task-a", "task-b", "task-c"])
        self.assertEqual([u["arm"] for u in out["unsupported"]], ["pi"])
        self.assertEqual((out["images"], out["pull"], out["blocked"], out["mode"]),
                         (3, ["alexgshaw/task-c:1"], "", "prefer"))
        self.assertEqual(out["builds"], [])
        self.assertEqual(set(out["disk"]), {"path", "source", "free_gb", "reserve_gb", "vm_free_gb", "vm_checked_at",
                                            "vm_reserve_gb"})
        default = self.ok(self.client.get("/api/prebake/status"))  # the largest mode, enabled arms
        self.assertEqual(default["readiness"]["claude-code"]["tasks"], 3)

    def test_builds_and_pulls_refuse_during_a_run_or_while_the_queue_is_active(self) -> None:
        self.manager.active_id = "20261029-120000-sample"
        for path in ("/api/prebake/build", "/api/prebake/pull"):
            response = self.client.post(path, json={"mode": "sample", "arms": ["claude-code"]})
            self.assertEqual((response.status_code, response.json()["detail"]), (409, server.BUILD_RUN_ACTIVE))
        self.manager.active_id = None
        with mock.patch.object(self.state.queue, "is_active", lambda: True):
            response = self.client.post("/api/prebake/build", json={"mode": "sample"})
            self.assertEqual((response.status_code, response.json()["detail"]), (409, server.BUILD_QUEUE_ACTIVE))
            self.assertEqual(self.ok(self.client.get("/api/prebake/status"))["blocked"], server.BUILD_QUEUE_ACTIVE)

    def test_writes_need_the_header(self) -> None:
        bare = self.make_client(self.state, x_bench=False)
        for path in ("/api/prebake/build", "/api/prebake/pull", "/api/prebake/prune", "/api/prebake/disk",
                     "/api/prebake/builds/b1/cancel"):
            self.assertEqual(bare.post(path, json={}).status_code, 403, path)

    def test_build_starts_for_the_selection(self) -> None:
        started: list[tuple[Any, Any]] = []

        def start(cfg: Config, plan: Any, *, jobs: int | None = None, **kwargs: Any) -> dict[str, Any]:
            started.append((plan, jobs))
            return {"id": "b20261029-120000-abcd", "status": "running", "items": []}

        self.state.builds.start = start  # type: ignore[method-assign]
        out = self.ok(self.client.post("/api/prebake/build", json={"mode": "sample", "arms": ["claude-code"],
                                                                   "jobs": 2}), 201)
        self.assertEqual(out["id"], "b20261029-120000-abcd")
        plan, jobs = started[0]
        self.assertEqual((sorted(i.task_image for i in plan.items), jobs),
                         (["alexgshaw/task-a:1", "alexgshaw/task-b:1", "alexgshaw/task-c:1"], 2))
        self.assertEqual(self.client.post("/api/prebake/build", json={"mode": "sample", "bogus": 1}).status_code, 400)

    def test_pull_takes_only_the_selections_missing_task_images(self) -> None:
        pulls: list[list[str]] = []
        self.state.builds.start_pull = (  # type: ignore[method-assign]
            lambda cfg, images, label="pull": pulls.append(list(images)) or {"id": "b-pull", "kind": "pull"})
        self.ok(self.client.post("/api/prebake/pull", json={"mode": "sample"}), 201)
        self.assertEqual(pulls, [["alexgshaw/task-c:1"]])
        response = self.client.post("/api/prebake/pull", json={"mode": "pilot"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("on this machine already", response.json()["detail"])

    def test_prune_is_a_dry_run_until_the_listed_images_are_confirmed(self) -> None:
        old = prune.Candidate(ref="benchbake/claude-code-2.1.312-raaaaaaaaaa/alexgshaw/task-a:1", id="sha256:1",
                              kind="bake", reason="built for Harbor 0.24.0")
        kept = prune.Candidate(ref="benchbake-staging/b1-1/x:1", id="sha256:2", kind="staging", reason="staging")
        removed: list[list[str]] = []
        self.patch(prune, "candidates", lambda cfg: ([old, kept], ["a note"]))
        self.patch(prune, "prune", lambda found, dry_run=True, run_active=None: removed.append(
            [c.ref for c in found]) or {"dry_run": dry_run, "removed": [c.ref for c in found], "failed": {}})
        dry = self.ok(self.client.post("/api/prebake/prune", json={}))
        self.assertEqual((dry["dry_run"], [c["ref"] for c in dry["candidates"]], dry["notes"]),
                         (True, [old.ref, kept.ref], ["a note"]))
        self.assertEqual(removed, [])
        done = self.ok(self.client.post("/api/prebake/prune", json={"refs": [old.ref, "alexgshaw/task-a:1"]}))
        self.assertEqual((removed, done["skipped"]), ([[old.ref]], ["alexgshaw/task-a:1"]))
        self.manager.active_id = "r"
        self.assertEqual(self.client.post("/api/prebake/prune", json={"refs": [old.ref]}).status_code, 409)

    def test_state_build_list_status_log_and_cancel(self) -> None:
        build = {"id": "b20261029-120000-abcd", "kind": "build", "label": "mode sample", "created_at": now(),
                 "jobs": 4, "options": {}, "status": "done", "message": "", "finished_at": now(),
                 "items": [{"n": 1, "family": "claude-code", "version": "2.1.312", "recipe": "a" * 64,
                            "task_image": "alexgshaw/task-a:1", "image": "benchbake/x", "tasks": ["task-a"],
                            "arms": ["claude-code"], "status": "promoted",
                            "log": str(self.prebake_dir / "logs" / "b20261029-120000-abcd" / "01-x.log")}]}
        folder = self.prebake_dir / "builds"
        folder.mkdir(parents=True)
        (folder / f"{build['id']}.json").write_text(json.dumps(build), encoding="utf-8")
        log = self.prebake_dir / "logs" / build["id"]
        log.mkdir(parents=True)
        (log / "01-x.log").write_text("step 1\nstep 2\n", encoding="utf-8")
        listed = self.ok(self.client.get("/api/prebake/builds"))
        self.assertEqual([(b["id"], b["items"], b["counts"]) for b in listed["builds"]],
                         [(build["id"], 1, {"promoted": 1})])
        self.assertEqual(self.ok(self.client.get(f"/api/prebake/builds/{build['id']}"))["items"][0]["status"],
                         "promoted")
        self.assertEqual(self.client.get(f"/api/prebake/builds/{build['id']}/log/1").text, "step 1\nstep 2")
        self.assertEqual(self.client.get("/api/prebake/builds/nope").status_code, 404)
        self.assertEqual(self.client.post(f"/api/prebake/builds/{build['id']}/cancel").status_code, 409)
        block = self.ok(self.client.get("/api/state"))["prebake"]
        self.assertEqual((block["last"]["id"], block["active"], block["elsewhere"]), (build["id"], None, False))
        self.assertTrue(block["version"].startswith(f"{build['id']}:done:"))

    def test_disk_measures_the_vm_with_a_local_task_image(self) -> None:
        self.patch(builder, "last_vm", {})
        out = self.ok(self.client.post("/api/prebake/disk"))
        self.assertEqual(out["vm_free_gb"], 900.0)
        runs_ = self.docker_fake.ran("run")
        self.assertEqual(len(runs_), 1)
        self.assertIn("--network", runs_[0])


if __name__ == "__main__":
    import unittest

    unittest.main()
