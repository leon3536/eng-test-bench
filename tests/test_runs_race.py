"""Task races (run.schedule tasks): per task, build every arm's pre-baked image,
race every arm on it at once within run.max_containers, remove the task's
images, then the next task.

Harbor never runs: RunsTestCase (tests/test_runs.py) swaps each harbor
command for a small script that writes trial folders. Docker never runs
either: the image inventory answers from a set of "local" names, the build
manager is FakeBuilds, and image removal is recorded. The builder's own
pieces a race relies on (the owner run's lock, the scoped build cache prune,
remove_images) are tested against FakeDocker at the end.
"""

from __future__ import annotations

import contextlib
import io
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

from tests.fakes_docker import FakeDocker, go_time
from tests.support import BenchTestCase
from tests.test_runs import FAKE_HARBOR, RunsTestCase

from bench import designs, estimates, extract, fsutil, procs, results, runs, settings, spec
from bench.environments.prebaked import MISSING, READY, Decision
from bench.prebake import builder, inventory, prune, pull
from bench.prebake.builder import BuildManager, BuildRefused

assert FAKE_HARBOR  # the script RunsTestCase runs in place of harbor

RACE_YAML = """\
dataset: terminal-bench@2.0
providers:
  anthropic: {{env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}}
modes:
  pilot: {{n_tasks: 1}}
  three: {{n_tasks: 3, n_attempts: {attempts}}}
run:
  effort: medium
  n_concurrent: {n_concurrent}
  parallel_arms: 1
  budget_usd: {budget}
  max_turns: 50
  max_containers: {max_containers}
  prebake: {prebake}
prebake:
  keep_images: {keep}
baseline_arm: a
arms:
  - {{name: a, agent: terminus-2, provider: anthropic, model: claude-opus-5-5}}
  - {{name: b, agent: pi, provider: anthropic, model: claude-opus-5-5}}
  - {{name: c, agent: terminus-2, provider: anthropic, model: claude-opus-5-5, options: {{max_turns: 20}}}}
"""
TASKS = ["task-a", "task-b", "task-c"]


class FakeBuilds:
    """BuildManager as a task race uses it: start, wait, status, cancel."""

    def __init__(self, case: RaceTestCase) -> None:
        self.case = case
        self.starts: list[dict[str, Any]] = []
        self.fail: set[str] = set()        # arm names whose image fails to build
        self.refuse: str | None = None     # start raises BuildRefused with this
        self.hold = threading.Event()      # cleared: wait() blocks until set or cancelled
        self.hold.set()
        self.cancelled: list[str] = []
        self.emergencies = 0
        self.builds: dict[str, dict[str, Any]] = {}

    def start(self, cfg: Any, plan: Any, **options: Any) -> dict[str, Any]:
        if self.refuse:
            raise BuildRefused(self.refuse)
        build_id = f"b{len(self.starts) + 1}"
        self.starts.append({"id": build_id, "options": options,
                            "tasks": sorted({t for i in plan.items for t in i.tasks}),
                            "images": [i.image for i in plan.items], "arms": [list(i.arms) for i in plan.items]})
        items = []
        for item in plan.items:
            failed = any(a in self.fail for a in item.arms)
            if not failed:
                self.case.local.add(item.image)
            items.append({"status": "failed" if failed else "promoted", "arms": list(item.arms),
                          "message": "docker build exited with 1" if failed else "", "image": item.image})
        self.builds[build_id] = {"id": build_id, "status": "done_with_errors" if self.fail else "done",
                                 "message": "", "items": items, "cache": {"created": 4, "pruned": 4}}
        return {"id": build_id}

    def wait(self, timeout: float | None = None) -> bool:
        return self.hold.wait(timeout)

    def status(self, build_id: str) -> dict[str, Any]:
        return self.builds[build_id]

    def cancel(self, build_id: str) -> None:
        self.cancelled.append(build_id)
        self.builds[build_id]["status"] = "cancelled"
        self.hold.set()

    def emergency_stop(self) -> None:
        self.emergencies += 1


class RaceTestCase(RunsTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.local: set[str] = {f"example/{t}:1" for t in TASKS}   # the task images are local
        self.removed: list[list[str]] = []
        self.remove_calls: list[dict[str, Any]] = []
        self.builds = FakeBuilds(self)
        for target, name, value in (
                (runs, "RACE_STAGGER_SECONDS", 0),
                (inventory, "task_images", lambda dataset, tasks: {t: f"example/{t}:1" for t in tasks}),
                (inventory, "inspect_many", lambda refs, refresh=False: {r: r if r in self.local else None
                                                                        for r in refs}),
                (inventory, "check_candidate", self.decide),
                (pull, "missing", lambda images, refresh=True: []),
                (builder, "remove_images", self.fake_remove)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def decide(self, baked: Any, base: Any, want: Any) -> Decision:
        if baked is None:
            return Decision(MISSING, f"no pre-baked image {want.image}", image=want.image)
        return Decision(READY, "ready", image=want.image, baked_at="2026-10-31T00:00:00Z")

    def fake_remove(self, images: list[str], prefix: str, task_images: Any = ()) -> dict[str, str]:
        self.remove_calls.append({"images": list(images), "prefix": prefix, "task_images": list(task_images)})
        self.removed.append(list(images))
        for image in images:
            self.local.discard(image)
        return {i: "removed" for i in images}

    def race_config(self, budget: float = 10, max_containers: int = 6, prebake: str = "prefer",
                    keep: bool = False, attempts: int = 1, n_concurrent: int = 1) -> settings.Config:
        text = RACE_YAML.format(budget=budget, max_containers=max_containers, prebake=prebake,
                                keep="true" if keep else "false", attempts=attempts, n_concurrent=n_concurrent)
        return self.load_test_config(text, tasks=TASKS)

    def race_manager(self) -> runs.RunManager:
        manager = self.manager()
        manager.builds = self.builds
        return manager

    def real(self, arm: str | None = None) -> list[list[str]]:
        """The real (not dry-run) harbor commands, in start order."""
        return [c for c in self.commands if "--dry-run" not in c
                and (arm is None or Path(c[c.index("--jobs-dir") + 1]).name == arm)]

    @staticmethod
    def where(cmd: list[str]) -> tuple[str, str]:
        """(arm, task) of a race command: --jobs-dir jobs/<run>/<arm> --job-name <task>."""
        return Path(cmd[cmd.index("--jobs-dir") + 1]).name, cmd[cmd.index("--job-name") + 1]


class RaceRunTests(RaceTestCase):
    def test_each_task_is_built_raced_and_cleared_before_the_next(self) -> None:
        manager = self.race_manager()
        run_id = manager.start(self.race_config(), "three", ["a", "b", "c"])["id"]
        final = self.finish(manager)
        self.assertEqual(final["status"], "done", final["message"])
        self.assertEqual(final["schedule"], "tasks")
        # One harbor job per (arm, task), each limited to its task, tasks strictly in order.
        real = self.real()
        self.assertEqual([self.where(c)[1] for c in real], [t for t in TASKS for _ in "abc"])
        for cmd in real:
            arm, task = self.where(cmd)
            self.assertEqual(Path(cmd[cmd.index("--jobs-dir") + 1]), settings.JOBS_DIR / run_id / arm)
            self.assertEqual([cmd[i + 1] for i, x in enumerate(cmd) if x == "--include-task-name"], [task])
            self.assertEqual(cmd[cmd.index("--n-attempts") + 1], "1")
        # Every lane had its preflight first (the same dry run as today).
        self.assertEqual(len([c for c in self.commands if "--dry-run" in c]), 9)
        # a and c share terminus-2's recipe, so one image per task serves both; pi pins no version: stock.
        self.assertEqual([b["tasks"] for b in self.builds.starts], [[t] for t in TASKS])
        for b in self.builds.starts:
            self.assertEqual(len(b["images"]), 1)
            self.assertEqual(sorted(b["arms"][0]), ["a", "c"])
            self.assertEqual(b["options"], {"jobs": 1, "setup": "once", "owner_run": run_id, "prune_cache": True})
        self.assertEqual([len(r) for r in self.removed], [1, 1, 1])
        self.assertFalse([i for i in self.local if i.startswith("benchbake/")], "no image is left behind")
        for call in self.remove_calls:
            self.assertEqual(call["prefix"], "benchbake")
            self.assertTrue(call["task_images"])
        # The image was ready when a's and c's lanes started, so no stock setup and no setup net.
        for cmd in self.real("a") + self.real("c"):
            self.assertIn("bench.environments.prebaked:PrebakedDockerEnvironment", cmd)
            self.assertNotIn("--agent-setup-timeout-multiplier", cmd)
        for cmd in self.real("b"):
            self.assertNotIn("--env", cmd)
            self.assertEqual(cmd[cmd.index("--agent-setup-timeout-multiplier") + 1], "2")
        # Arms add up their lanes.
        for arm in final["arms"]:
            self.assertEqual((arm["status"], arm["trials_done"], arm["trials_total"]), ("done", 3, 3))
            self.assertAlmostEqual(arm["spend_usd"], 0.75)
            self.assertFalse(arm["cleanup_pending"])
        self.assertAlmostEqual(final["spend_usd"], 2.25)
        record = self.on_disk(run_id)
        race = record["race"]
        self.assertEqual(race["done"], 3)
        self.assertIsNone(race["current"])
        for entry in race["tasks"]:
            self.assertEqual(entry["status"], "done")
            self.assertEqual(sorted(entry["lanes"]), ["a", "b", "c"])
            self.assertEqual({k: v["status"] for k, v in entry["lanes"].items()}, dict.fromkeys("abc", "done"))
            self.assertEqual(entry["images"]["a"]["status"], "built")
            self.assertEqual(entry["images"]["b"]["status"], "stock")
            self.assertEqual(entry["build"]["status"], "done")
            self.assertEqual(entry["build"]["cache"], {"created": 4, "pruned": 4})
            self.assertEqual(len(entry["removed"]), 1)
        # Folders stay one per (arm, task): jobs/<run>/<arm>/<task>/<trial>, marked so readers find them.
        for arm in "abc":
            arm_dir = settings.JOBS_DIR / run_id / arm
            self.assertTrue((arm_dir / fsutil.RACE_MARKER).is_file())
            self.assertEqual(fsutil.harbor_jobs(arm_dir), [arm_dir / t for t in TASKS])
            for task in TASKS:
                self.assertTrue((arm_dir / task / "t1" / "result.json").is_file())
        cfg = self.race_config()
        rows = results.load_job(settings.JOBS_DIR / run_id / "a", cfg, cfg.arm("a"))
        self.assertEqual(len(rows), 3)
        log = (settings.RUNS_DIR / run_id / "images.log").read_text(encoding="utf-8")
        self.assertIn("task-a: building 1 image(s) (build b1)", log)
        self.assertIn("task-c: removed 1 image(s)", log)
        self.assertIn("--- task race: task-b ---", self.arm_log(run_id, "a"))

    def test_the_ledger_finds_every_task_job_of_a_race(self) -> None:
        manager = self.race_manager()
        run_id = manager.start(self.race_config(), "three", ["a"])["id"]
        self.finish(manager)
        roots = [extract.ScanRoot(settings.JOBS_DIR, "default", settings.RUNS_DIR)]
        found = [(job, ctx.arm) for job, ctx, _, _ in extract.discover(roots)]
        self.assertEqual(found, [(settings.JOBS_DIR / run_id / "a" / t, "a") for t in TASKS])

    def test_max_containers_races_that_many_then_the_rest(self) -> None:
        spans: dict[tuple[str, str], list[float]] = {}
        real_run_arm = runs.RunManager._run_arm

        def timed(manager: Any, cfg: Any, spec_: Any, run: Any, arm: Any, lane: Any, dataset_path: Any,
                  job_root: Any = None, job_name: Any = None) -> None:
            start = time.monotonic()
            real_run_arm(manager, cfg, spec_, run, arm, lane, dataset_path, job_root, job_name)
            spans[(arm.name, job_name)] = [start, time.monotonic()]

        self.scenarios.update(a="slow-ok", b="slow-ok", c="slow-ok")
        with mock.patch.object(runs.RunManager, "_run_arm", timed):
            manager = self.race_manager()
            manager.start(self.race_config(max_containers=2), "three", ["a", "b", "c"])
            final = self.finish(manager, timeout=60)
        self.assertEqual(final["status"], "done")
        for task in TASKS:
            a, b, c = (spans[(x, task)] for x in "abc")
            self.assertLess(max(a[0], b[0]), min(a[1], b[1]), "a and b race together")
            self.assertGreaterEqual(c[0], min(a[1], b[1]), "c starts when a slot frees up")
        for earlier, later in zip(TASKS, TASKS[1:]):
            self.assertGreaterEqual(min(spans[(x, later)][0] for x in "abc"),
                                    max(spans[(x, earlier)][1] for x in "abc"), "tasks never overlap")

    def test_a_failed_image_build_under_prefer_runs_that_arm_stock(self) -> None:
        self.builds.fail = {"a", "c"}
        manager = self.race_manager()
        run_id = manager.start(self.race_config(), "three", ["a", "b"])["id"]
        final = self.finish(manager)
        self.assertEqual(final["status"], "done")
        for cmd in self.real("a"):
            self.assertIn("prebake_required=false", cmd)
            self.assertEqual(cmd[cmd.index("--agent-setup-timeout-multiplier") + 1], "2")  # the setup net
        entry = self.on_disk(run_id)["race"]["tasks"][0]
        self.assertEqual(entry["images"]["a"]["status"], "failed")
        self.assertIn("no image for a; those trials runs stock setup",
                      (settings.RUNS_DIR / run_id / "images.log").read_text(encoding="utf-8"))

    def test_require_still_races_an_arm_without_an_image_so_its_trial_fails_fast_as_infra(self) -> None:
        self.builds.fail = {"a"}
        manager = self.race_manager()
        run_id = manager.start(self.race_config(prebake="require"), "three", ["a"])["id"]
        self.finish(manager)
        self.assertEqual(len(self.real("a")), 3)
        for cmd in self.real("a"):
            self.assertIn("prebake_required=true", cmd)  # the env class raises before any container starts
        self.assertEqual(self.on_disk(run_id)["race"]["tasks"][1]["images"]["a"]["status"], "failed")

    def test_a_build_that_cannot_start_leaves_the_race_to_run(self) -> None:
        self.builds.refuse = "Not enough disk space to build: the drive has 3 GB free"
        manager = self.race_manager()
        run_id = manager.start(self.race_config(), "three", ["a", "b"])["id"]
        final = self.finish(manager)
        self.assertEqual(final["status"], "done")
        entry = self.on_disk(run_id)["race"]["tasks"][0]
        self.assertEqual(entry["build"]["status"], "refused")
        self.assertIn("Not enough disk space", entry["images"]["a"]["message"])
        self.assertEqual(entry["images"]["a"]["status"], "not built")
        self.assertEqual(self.removed, [], "nothing was built, and absent images are not removed")

    def test_prebake_off_builds_and_removes_nothing(self) -> None:
        manager = self.race_manager()
        run_id = manager.start(self.race_config(prebake="'off'"), "three", ["a"])["id"]
        self.finish(manager)
        self.assertEqual((self.builds.starts, self.removed), ([], []))
        self.assertEqual(self.on_disk(run_id)["race"]["tasks"][0]["images"]["a"]["status"], "off")


class RaceRemovalTests(RaceTestCase):
    def test_keep_images_keeps_every_image(self) -> None:
        manager = self.race_manager()
        run_id = manager.start(self.race_config(keep=True), "three", ["a"])["id"]
        self.finish(manager)
        self.assertEqual(self.removed, [])
        kept = self.on_disk(run_id)["race"]["tasks"][0]["kept"]
        self.assertEqual(list(kept.values()), ["prebake.keep_images"])

    def test_an_image_a_later_task_needs_is_kept_until_that_task(self) -> None:
        shared = "example/shared:1"
        with mock.patch.object(inventory, "task_images",
                               lambda dataset, tasks: {t: shared if t in ("task-a", "task-c") else f"example/{t}:1"
                                                       for t in tasks}):
            self.local.add(shared)
            manager = self.race_manager()
            run_id = manager.start(self.race_config(), "three", ["a"])["id"]
            self.finish(manager)
        tasks = self.on_disk(run_id)["race"]["tasks"]
        self.assertEqual(tasks[0]["removed"], [])
        self.assertEqual(list(tasks[0]["kept"].values()), ["a later task of this run needs it"])
        self.assertEqual(len(tasks[1]["removed"]), 1)
        self.assertEqual(tasks[2]["removed"], [tasks[0]["images"]["a"]["image"]])
        self.assertEqual(len(self.builds.starts), 2, "task-c found task-a's image ready")

    def test_an_image_a_waiting_queue_item_needs_is_kept(self) -> None:
        manager = self.race_manager()
        wanted: set[str] = set()
        manager.keep_images = lambda cfg: wanted
        self.builds.start = self.wrap_start(lambda plan: wanted.update(i.image for i in plan.items[:1])
                                            if plan.label.endswith("task-b") else None)
        run_id = manager.start(self.race_config(), "three", ["a"])["id"]
        self.finish(manager)
        tasks = self.on_disk(run_id)["race"]["tasks"]
        self.assertEqual([len(t["removed"]) for t in tasks], [1, 0, 1])
        self.assertEqual(list(tasks[1]["kept"].values()), ["a waiting queue item needs it"])

    def test_a_queue_check_that_fails_keeps_the_images(self) -> None:
        manager = self.race_manager()

        def broken(cfg: Any) -> set[str]:
            raise OSError("queue.json is busy")

        manager.keep_images = broken
        run_id = manager.start(self.race_config(), "three", ["a"])["id"]
        self.finish(manager)
        self.assertEqual(self.removed, [])
        self.assertIn("this task's images are kept",
                      (settings.RUNS_DIR / run_id / "images.log").read_text(encoding="utf-8"))

    def wrap_start(self, before: Any) -> Any:
        real = FakeBuilds.start.__get__(self.builds)

        def start(cfg: Any, plan: Any, **options: Any) -> dict[str, Any]:
            before(plan)
            return real(cfg, plan, **options)

        return start

    def test_images_needed_counts_only_arms_schedule_runs(self) -> None:
        cfg = self.race_config()
        race = spec.spec_from_config(cfg, "three", ["a"])
        arms_spec = spec.spec_from_launch({**race.spec, "run": {**race.spec["run"], "schedule": "arms"}}, "three")
        self.assertEqual(runs.images_needed(cfg, [race]), set())
        needed = runs.images_needed(cfg, [arms_spec])
        self.assertEqual(len(needed), 3)
        self.assertTrue(all(n.startswith("benchbake/terminus-2-") for n in needed))


class RaceStopTests(RaceTestCase):
    def test_the_budget_mid_race_stops_every_lane_and_skips_the_remaining_tasks(self) -> None:
        self.scenarios.update(a="spend", b="hang")
        manager = self.race_manager()
        run_id = manager.start(self.race_config(budget=10), "three", ["a", "b"])["id"]
        final = self.finish(manager, timeout=40)
        self.assertEqual(final["status"], "budget_stopped")
        arms = {a["name"]: a for a in final["arms"]}
        self.assertEqual(arms["a"]["status"], "budget_stopped")
        self.assertIn("reached the $10.00 budget", arms["a"]["message"])
        self.assertIn("2 of 3 tasks not started", arms["a"]["message"])
        self.assertIn(arms["b"]["status"], ("budget_stopped", "cancelled"))
        race = self.on_disk(run_id)["race"]
        self.assertEqual([t["status"] for t in race["tasks"]], ["budget_stopped", "skipped", "skipped"])
        self.assertEqual({k: v["status"] for k, v in race["tasks"][1]["lanes"].items()}, dict.fromkeys("ab", "skipped"))
        self.assertEqual(len(self.real()), 2, "no harbor job started after the budget ran out")
        self.assertEqual(len(self.removed), 1, "the stopped task's images are still removed")
        # Each stopped lane's own job folder was swept, never the arm folder of another task.
        self.assertEqual(sorted(self.cleanups), sorted({settings.JOBS_DIR / run_id / x / "task-a" for x in "ab"}))

    def test_cancel_mid_race(self) -> None:
        self.scenarios.update(a="hang", b="hang")
        manager = self.race_manager()
        run_id = manager.start(self.race_config(), "three", ["a", "b"])["id"]
        self.wait_for(lambda: len(self.real()) == 2 and all(
            a["trials_running"] for a in manager.get(run_id)["arms"]))
        live = manager.get(run_id)
        self.assertEqual(live["race"]["current"], "task-a")
        self.assertEqual(live["race"]["tasks"][0]["status"], "racing")
        self.assertEqual([a["status"] for a in live["arms"]], ["running", "running"])
        self.assertTrue(all(a["cleanup_pending"] for a in live["arms"]))
        manager.cancel(run_id)
        final = self.finish(manager)
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual([a["status"] for a in final["arms"]], ["cancelled", "cancelled"])
        self.assertEqual([t["status"] for t in final["race"]["tasks"]], ["cancelled", "cancelled", "cancelled"])
        self.assertEqual(len(self.removed), 1)
        self.assertEqual([a["trials_killed"] for a in final["arms"]], [1, 1])

    def test_cancel_during_a_build_cancels_it_and_races_nothing(self) -> None:
        self.builds.hold.clear()
        manager = self.race_manager()
        run_id = manager.start(self.race_config(), "three", ["a"])["id"]
        self.wait_for(lambda: manager.get(run_id)["race"]["tasks"][0]["status"] == "building"
                      and self.builds.starts)
        manager.cancel(run_id)
        final = self.finish(manager)
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual(self.builds.cancelled, ["b1"])
        self.assertEqual(self.real(), [])
        self.assertEqual(final["arms"][0]["status"], "cancelled")

    def test_cancel_during_the_build_of_a_one_task_run_cancels_the_arm(self) -> None:
        self.builds.hold.clear()
        manager = self.race_manager()
        run_id = manager.start(self.race_config(), "pilot", ["a"])["id"]
        self.wait_for(lambda: manager.get(run_id)["race"]["tasks"][0]["status"] == "building"
                      and self.builds.starts)
        manager.cancel(run_id)
        final = self.finish(manager)
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual(final["race"]["tasks"][0]["lanes"]["a"]["status"], "cancelled")
        self.assertEqual(final["arms"][0]["status"], "cancelled")
        self.assertEqual(final["arms"][0]["message"], "1 of 1 tasks not started.")

    def test_cancel_during_the_last_tasks_build_cancels_the_arm(self) -> None:
        manager = self.race_manager()
        start = self.builds.start

        def hold_the_third(cfg: Any, plan: Any, **options: Any) -> dict[str, Any]:
            if len(self.builds.starts) == 2:
                self.builds.hold.clear()
            return start(cfg, plan, **options)

        self.builds.start = hold_the_third  # type: ignore[method-assign]
        run_id = manager.start(self.race_config(), "three", ["a"])["id"]
        self.wait_for(lambda: len(self.builds.starts) == 3)
        manager.cancel(run_id)
        final = self.finish(manager)
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual([t["lanes"]["a"]["status"] for t in final["race"]["tasks"]], ["done", "done", "cancelled"])
        self.assertEqual(final["arms"][0]["status"], "cancelled")
        self.assertIn("1 of 3 tasks not started", final["arms"][0]["message"])

    def test_emergency_stop_during_a_build_stops_the_build_too(self) -> None:
        self.builds.hold.clear()
        manager = self.race_manager()
        run_id = manager.start(self.race_config(), "three", ["a"])["id"]
        self.wait_for(lambda: manager.get(run_id)["race"]["tasks"][0]["status"] == "building"
                      and self.builds.starts)
        manager.emergency_stop()
        self.assertEqual(self.builds.emergencies, 1)
        self.builds.hold.set()
        self.finish(manager)

    def test_emergency_stop_hands_the_arm_folder_to_the_helper(self) -> None:
        self.scenarios["a"] = "hang"
        manager = self.race_manager()
        run_id = manager.start(self.race_config(), "three", ["a"])["id"]
        self.wait_for(lambda: manager.get(run_id)["arms"][0]["trials_running"] == 1)
        manager.emergency_stop()
        self.assertEqual(self.spawned, [[settings.JOBS_DIR / run_id / "a"]])

    def test_a_race_left_running_is_interrupted_and_its_arms_swept_at_startup(self) -> None:
        run_id = "20261031-010101-design"
        record = {"id": run_id, "mode": "design", "created_at": "2026-10-31T01:01:01+00:00", "budget_usd": 5.0,
                  "tasks": TASKS, "status": "running", "schedule": "tasks",
                  "arms": [{"name": "a", "status": "running", "cleanup_pending": False}],
                  "race": {"current": "task-b", "tasks": [
                      {"task": "task-a", "status": "done", "lanes": {"a": {"name": "a", "status": "done"}}},
                      {"task": "task-b", "status": "racing", "lanes": {"a": {"name": "a", "status": "running"}}}]}}
        (settings.RUNS_DIR / run_id).mkdir(parents=True)
        (settings.RUNS_DIR / run_id / "run.json").write_text(json.dumps(record), encoding="utf-8")
        manager = self.manager()
        loaded = manager.get(run_id)
        self.assertEqual(loaded["status"], "interrupted")
        self.assertEqual(loaded["race"]["tasks"][1]["status"], "interrupted")
        self.assertEqual(loaded["race"]["tasks"][1]["lanes"]["a"]["status"], "interrupted")
        self.assertIsNone(loaded["race"]["current"])
        self.assertEqual(self.cleanups, [settings.JOBS_DIR / run_id / "a"])


class RacePlanTests(RaceTestCase):
    def test_the_plan_reports_the_race(self) -> None:
        cfg = self.race_config(max_containers=2)
        p = runs.plan(cfg, "three", ["a", "b", "c"])
        self.assertEqual(p["problems"], [])
        self.assertEqual(p["schedule"], "tasks")
        self.assertEqual(p["peak_containers"], 2)
        race = p["race"]
        self.assertEqual((race["tasks"], race["arms"], race["lanes"], race["max_containers"]), (3, 3, 9, 2))
        self.assertEqual(race["images_to_build"], 3)
        self.assertEqual([t["build"] for t in race["per_task"]], [1, 1, 1])
        from bench.prebake.adapters import ADAPTERS
        self.assertEqual(race["build_wall_min_max"], ADAPTERS["terminus-2"].expected_build_min)
        self.assertAlmostEqual(race["peak_disk_gb"], round(ADAPTERS["terminus-2"].expected_gb * builder.DISK_FACTOR, 2))
        self.assertIn("parallel_arms does not apply", race["note"])
        # Missing images are no warning: the race builds them. Their command is the first task's job.
        a = p["arms"][0]
        self.assertFalse([w for w in a["warnings"] if "install the harness themselves" in w])
        self.assertIn(" --job-name task-a ", a["command"])
        self.assertIn(str(Path("jobs") / "RUN_ID" / "a"), a["command"])

    def test_require_with_missing_images_is_no_plan_problem_in_a_race(self) -> None:
        p = runs.plan(self.race_config(prebake="require"), "three", ["a"])
        self.assertEqual(p["problems"], [])

    def test_a_lane_wider_than_max_containers_is_a_problem(self) -> None:
        p = runs.plan(self.race_config(max_containers=1, attempts=2, n_concurrent=2), "three", ["a"])
        self.assertTrue(any("above run.max_containers (1)" in x for x in p["problems"]), p["problems"])

    def test_estimate_wall_time_adds_up_task_races(self) -> None:
        cfg = self.race_config(max_containers=2)
        run_spec = spec.spec_from_config(cfg, "three", ["a", "b", "c"])
        p = runs.plan_spec(cfg, run_spec)
        est = estimates.estimate(cfg, run_spec.spec, [], p, replicates=200)
        race = est["run"]["race"]
        self.assertEqual((race["tasks"], race["arms"], race["max_containers"]), (3, 3, 2))
        self.assertEqual(race["peak_disk_gb"], p["race"]["peak_disk_gb"])
        self.assertEqual(est["run"]["schedule"], "tasks")
        arms_spec = {**run_spec.spec, "run": {**run_spec.spec["run"], "schedule": "arms", "parallel_arms": 3}}
        arms_est = estimates.estimate(cfg, arms_spec, [], None, replicates=200)
        # Three races of two lanes then one, each after its build: longer than three arms side by side.
        self.assertGreater(est["run"]["wall_minutes"]["p50"], arms_est["run"]["wall_minutes"]["p50"])

    def test_estimate_race_wall_is_builds_plus_lanes_per_task(self) -> None:
        run = settings.RunSettings(effort="medium", n_concurrent=1, parallel_arms=1, budget_usd=5,
                                   per_trial_cap_usd=None, max_turns=None, timeout_multiplier=1.0,
                                   schedule="tasks", max_containers=2)
        sims = [mock.Mock(tasks=["t1", "t2"]) for _ in range(3)]
        minutes = [[10.0, 1.0], [20.0, 1.0], [5.0, 1.0]]
        wall, busy = estimates._race_wall(sims, minutes, run, 1, ["t1", "t2"], {"t1": 3.0, "t2": 0.0})
        p = estimates.PREFLIGHT_MINUTES
        # t1: build 3, then lanes 11 and 21 together, 6 when the first frees up -> 3 + max(21, 11 + 6)
        # t2: lanes of 2 minutes each, two then one -> 4
        self.assertAlmostEqual(wall, 3 + max(20 + p, 10 + p + 5 + p) + (1 + p) * 2)
        self.assertEqual(busy, [10 + 1 + 2 * p, 20 + 1 + 2 * p, 5 + 1 + 2 * p])


class RaceSlotsTests(BenchTestCase):
    def test_lanes_take_their_slots_in_order_and_whole(self) -> None:
        slots = runs._Slots(3)
        cancel = threading.Event()
        order: list[int] = []

        def lane(ticket: int, width: int) -> None:
            self.assertTrue(slots.acquire(ticket, width, cancel))
            order.append(ticket)

        self.assertTrue(slots.acquire(0, 2, cancel))
        t1 = threading.Thread(target=lane, args=(1, 2))
        t2 = threading.Thread(target=lane, args=(2, 1))
        t1.start()
        t2.start()
        time.sleep(0.3)
        self.assertEqual(order, [], "ticket 1 waits for 2 slots, and 2 waits its turn")
        slots.release(2)
        t1.join(5)
        t2.join(5)
        self.assertEqual(order, [1, 2])

    def test_a_cancel_frees_every_waiting_lane(self) -> None:
        slots = runs._Slots(1)
        cancel = threading.Event()
        self.assertTrue(slots.acquire(0, 1, cancel))
        got: list[bool] = []
        threads = [threading.Thread(target=lambda n=n: got.append(slots.acquire(n, 1, cancel))) for n in (1, 2)]
        for t in threads:
            t.start()
        cancel.set()
        for t in threads:
            t.join(5)
        self.assertEqual(got, [False, False])


class RaceSettingsTests(BenchTestCase):
    BASE = "dataset: d\nproviders:\n  p: {env_key: K}\nmodes:\n  m: {n_tasks: 1}\nrun:\n  budget_usd: 1\n{run}" \
           "arms:\n  - {name: a, agent: terminus-2, provider: p, model: x}\n"

    def load(self, run: str = "") -> settings.Config:
        return self.load_test_config(self.BASE.replace("{run}", run), tasks=["t"])

    def test_tasks_and_six_containers_are_the_defaults(self) -> None:
        cfg = self.load()
        self.assertEqual((cfg.run.schedule, cfg.run.max_containers), ("tasks", 6))
        self.assertFalse(cfg.prebake.keep_images)
        self.assertEqual(self.load("  schedule: arms\n").run.schedule, "arms")

    def test_bad_values_are_refused(self) -> None:
        for run, why in (("  schedule: lanes\n", "run.schedule must be one of arms, tasks"),
                         ("  max_containers: 0\n", "run.max_containers"),
                         ("  max_containers: 17\n", "must not be above designs.max_containers")):
            with self.subTest(run=run), self.assertRaisesRegex(settings.ConfigError, why):
                self.load(run)

    def test_the_spec_writes_them_only_when_they_differ_from_the_defaults(self) -> None:
        stored = spec.run_dict(self.load().run)
        self.assertNotIn("schedule", stored)
        self.assertNotIn("max_containers", stored)
        changed = spec.run_dict(self.load("  schedule: arms\n  max_containers: 4\n").run)
        self.assertEqual((changed["schedule"], changed["max_containers"]), ("arms", 4))
        # A stored spec without the keys keeps its hash and runs the default schedule.
        old = {"dataset": "d", "tasks": ["t"], "attempts": 1, "baseline_arm": None, "run": stored,
               "arms": [{"name": "a", "agent": "terminus-2", "family": "terminus-2", "provider": "p", "model": "x",
                         "effort": "inherit", "options": {}, "tasks": None}]}
        run_spec = spec.spec_from_design(old, None)
        self.assertEqual(run_spec.spec_hash, spec.spec_hash(old))
        self.assertEqual((run_spec.run.schedule, run_spec.run.max_containers), ("tasks", 6))
        self.assertEqual(spec.structure_problems({**old, "run": {**stored, "schedule": "lanes"}}),
                         ["spec.run.schedule must be one of arms, tasks"])

    def test_peak_containers_of_a_race_design(self) -> None:
        design = {"tasks": ["t1", "t2"], "attempts": 2, "run": {"n_concurrent": 2, "parallel_arms": 1,
                                                               "schedule": "tasks", "max_containers": 6},
                  "arms": [{"tasks": None}, {"tasks": None}, {"tasks": ["t1"]}, {"tasks": None}]}
        self.assertEqual(designs.peak_containers(design), 6)       # t1: 4 lanes x 2, held to 6
        design["arms"] = [{"tasks": None}, {"tasks": ["t2"]}]
        self.assertEqual(designs.peak_containers(design), 4)
        design["run"]["schedule"] = "arms"
        self.assertEqual(designs.peak_containers(design), 2)       # parallel_arms 1 x n_concurrent 2


class RaceLayoutTests(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.arm_dir = self.tmpdir() / "jobs" / "20261031-101500-design" / "claude-code"
        for task, trial in (("fix-git", "fix-git__AbC1dEf"), ("regex-log", "regex-log__x9Y8z7w")):
            (self.arm_dir / task / trial).mkdir(parents=True)
        fsutil.mark_race(self.arm_dir, "20261031-101500-design", "claude-code")
        self.docker = FakeDocker()
        self.allow_subprocess(self.docker.matches, self.docker)

    def test_cleanup_of_a_race_arm_matches_its_trials_never_a_task_name(self) -> None:
        listed: list[list[str]] = []
        def matching(kind: str, prefixes: tuple[str, ...]) -> list[str]:
            listed.append(list(prefixes))
            return []

        with mock.patch.object(procs, "_matching", matching):
            procs.cleanup_containers(self.arm_dir)
        self.assertEqual(listed[0], ["fix-git__abc1def", "regex-log__x9y8z7w"])

    def test_an_arm_folder_without_the_marker_is_one_job_as_before(self) -> None:
        plain = self.tmpdir() / "a"
        (plain / "t1").mkdir(parents=True)
        self.assertEqual(fsutil.harbor_jobs(plain), [plain])
        self.assertEqual(fsutil.harbor_jobs(self.arm_dir), [self.arm_dir / "fix-git", self.arm_dir / "regex-log"])


class RaceBuilderTests(BenchTestCase):
    """The builder's parts a race uses, against FakeDocker."""

    def setUp(self) -> None:
        super().setUp()
        self.docker = FakeDocker()
        self.allow_subprocess(self.docker.matches, self.docker)
        self.root = self.tmpdir() / "prebake"
        for target, name, value in ((builder, "PREBAKE_DIR", self.root), (builder, "LOCK_WAIT_SEC", 0)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.isolate_data_dir()

    def test_a_race_build_passes_its_own_run_lock_and_no_other(self) -> None:
        run = settings.RUNS_DIR / "r1" / "run.json"
        run.parent.mkdir(parents=True)
        run.write_text(json.dumps({"status": "running"}), encoding="utf-8")
        with builder.run_lock():
            with self.assertRaises(BuildRefused):
                with builder.build_lock(lambda: True):
                    pass
            with builder.build_lock(lambda: True, owner_run="r1"):
                self.assertTrue(builder.build_active())
            other = settings.RUNS_DIR / "r2" / "run.json"
            other.parent.mkdir(parents=True)
            other.write_text(json.dumps({"status": "running"}), encoding="utf-8")
            with self.assertRaises(BuildRefused) as caught:
                with builder.build_lock(lambda: True, owner_run="r1"):
                    pass
            self.assertIn("run r2", str(caught.exception))

    def test_remove_images_touches_only_baked_names_under_the_prefix(self) -> None:
        from bench.environments.prebaked import LABEL_NS

        labels = {LABEL_NS + "bake-format": "1"}
        self.docker.add_image("benchbake/terminus-2-0.25.0-rabc/example/t:1", labels=labels)
        self.docker.add_image("benchbake/unlabelled/x:1")
        self.docker.add_image("example/t:1")
        out = builder.remove_images(["benchbake/terminus-2-0.25.0-rabc/example/t:1", "benchbake/unlabelled/x:1",
                                     "example/t:1", "benchbake/gone/x:1", "benchbake-staging/b1-1/x:1"],
                                    "benchbake", ["example/t:1"])
        self.assertEqual(out["benchbake/terminus-2-0.25.0-rabc/example/t:1"], "removed")
        self.assertTrue(out["benchbake/unlabelled/x:1"].startswith("kept: it carries no bake-format label"))
        self.assertEqual(out["example/t:1"], "kept: not a pre-baked image name")
        self.assertEqual(out["benchbake/gone/x:1"], "absent")
        self.assertEqual(out["benchbake-staging/b1-1/x:1"], "kept: not a pre-baked image name")
        self.assertEqual([c for c in self.docker.ran("rmi")], [["rmi", "benchbake/terminus-2-0.25.0-rabc/example/t:1"]])
        self.assertIsNotNone(self.docker.find("example/t:1"))
        self.assertTrue(all("-f" not in c for c in self.docker.ran("rmi")))

    FOREIGN = {"foreign1000000000000000001": "another project", "foreign2000000000000000002": "x"}

    def chain(self, *records: str) -> None:
        """Our build's records, each the parent of the next (as BuildKit links a build's steps)."""
        for n, record in enumerate(records):
            self.docker.cache[record] = f"a step of our build ({n})"
            if n:
                self.docker.cache_parents[record] = records[n - 1]

    def test_the_build_cache_prune_removes_only_records_the_build_made(self) -> None:
        self.docker.cache = dict(self.FOREIGN)
        before = prune.build_cache_listing()
        self.assertEqual((before.source, set(before.ids)), ("buildx", set(self.FOREIGN)))
        self.docker.cache["ours10000000000000000000001"] = "a step of our build"
        result = prune.prune_new_cache(before, 1)
        self.assertEqual(set(self.docker.cache), set(self.FOREIGN))
        self.assertEqual(result, {"before": 2, "source": "buildx", "created": 1, "pruned": 1, "left": 0,
                                  "passes": 1, "failed": 0, "error": None})
        prunes = self.docker.ran("builder", "prune")
        self.assertEqual(prunes, [["builder", "prune", "-f", "--all", "--filter", "id=ours10000000000000000000001"]])
        self.assertEqual(prune.prune_build_cache(["", "bad id;rm"]), {"ok": 0, "failed": 0, "error": None})
        self.assertEqual(len(self.docker.ran("builder", "prune")), 1, "never a prune without a plain id")

    def test_without_buildx_the_cache_is_listed_from_system_df(self) -> None:
        self.docker.buildx = False
        self.docker.cache = {"abcdef1234567890abcdef123": "x"}
        self.assertEqual(prune.build_cache_ids(), {"abcdef123456"})
        self.assertEqual(prune.build_cache_listing().source, "df")

    def test_a_listing_from_another_command_after_the_build_prunes_nothing(self) -> None:
        # buildx answered before the build and failed after it: the 12-character ids from system df match
        # none of the full ones, and every foreign record would have looked new.
        self.docker.cache = dict(self.FOREIGN)
        before = prune.build_cache_listing()
        self.docker.cache["ours10000000000000000000001"] = "ours"
        self.docker.buildx = False
        result = prune.prune_new_cache(before, 1)
        self.assertEqual(set(self.docker.cache), {*self.FOREIGN, "ours10000000000000000000001"})
        self.assertEqual(result["pruned"], 0)
        self.assertIn("listed by buildx before the build and by df after it", result["error"])
        self.assertEqual(self.docker.ran("builder", "prune"), [])

    def test_a_build_whose_second_listing_falls_back_prunes_nothing(self) -> None:
        self.docker.cache = dict(self.FOREIGN)
        manager = BuildManager(root=self.root)
        build = builder.Build(id="b1", kind="build", label="t", created_at="", jobs=1,
                              options={"prune_cache": True}, items=[])

        def fake_items(cfg: Any, b: Any, items: Any, log: Any) -> bool:
            self.docker.cache["ours10000000000000000000001"] = "ours"
            self.docker.buildx = False   # buildx fails (a timeout, say) on the listing after the build
            return True

        with mock.patch.object(manager, "_run_items", fake_items):
            manager._builds["b1"] = build
            manager._active = "b1"
            manager._execute(mock.Mock(), build, [], contextlib.ExitStack())
        self.assertTrue(set(self.FOREIGN) <= set(self.docker.cache), "another project's cache was pruned")
        self.assertEqual(build.cache["pruned"], 0)
        self.assertEqual(self.docker.ran("builder", "prune"), [])

    def test_both_listings_from_system_df_compare_on_twelve_characters(self) -> None:
        self.docker.buildx = False
        self.docker.cache = dict(self.FOREIGN)
        before = prune.build_cache_listing()
        self.docker.cache["ours10000000000000000000001"] = "ours"
        result = prune.prune_new_cache(before, 1)
        self.assertEqual(set(self.docker.cache), set(self.FOREIGN))
        self.assertEqual((result["created"], result["pruned"], result["source"]), (1, 1, "df"))
        self.assertEqual(self.docker.ran("builder", "prune"),
                         [["builder", "prune", "-f", "--all", "--filter", "id=ours10000000"]])

    def test_an_implausible_listing_prunes_nothing(self) -> None:
        self.docker.cache = dict(self.FOREIGN)
        before = prune.build_cache_listing()
        self.docker.cache = {"other0000000000000000000001": "x"}   # every record listed before has gone
        gone = prune.prune_new_cache(before, 1)
        self.assertIn("none of the 2 records listed before the build", gone["error"])
        self.docker.cache = {**self.FOREIGN, "ours10000000000000000000001": "a", "ours20000000000000000000002": "b"}
        with mock.patch.object(prune, "CACHE_MAX_NEW_PER_IMAGE", 1):
            many = prune.prune_new_cache(before, 1)
        self.assertIn("2 new records is more than a build of 1 image(s) makes (at most 1)", many["error"])
        self.assertEqual((many["created"], many["pruned"]), (2, 0))
        self.assertEqual(self.docker.ran("builder", "prune"), [])
        self.assertEqual(len(self.docker.cache), 4)

    def test_children_are_pruned_before_their_parents(self) -> None:
        self.docker.cache = dict(self.FOREIGN)
        before = prune.build_cache_listing()
        # Sorted, the parent comes first; BuildKit would keep it while its child exists.
        self.chain("ours0a000000000000000000001", "ours0b000000000000000000001", "ours0c000000000000000000001")
        result = prune.prune_new_cache(before, 1)
        self.assertEqual(set(self.docker.cache), set(self.FOREIGN))
        self.assertEqual((result["created"], result["pruned"], result["left"], result["passes"]), (3, 3, 0, 1))
        self.assertEqual([c[-1] for c in self.docker.ran("builder", "prune")],
                         ["id=ours0c000000000000000000001", "id=ours0b000000000000000000001",
                          "id=ours0a000000000000000000001"])

    def test_without_parent_links_the_prune_repeats_until_the_chain_is_gone(self) -> None:
        self.docker.buildx = False   # system df names no parents
        self.docker.cache = dict(self.FOREIGN)
        before = prune.build_cache_listing()
        self.chain("ours0a000000000000000000001", "ours0b000000000000000000001", "ours0c000000000000000000001")
        result = prune.prune_new_cache(before, 1)
        self.assertEqual(set(self.docker.cache), set(self.FOREIGN))
        self.assertEqual((result["created"], result["pruned"], result["left"], result["passes"]), (3, 3, 0, 3))

    def test_pruned_counts_what_is_gone_not_the_commands_that_succeeded(self) -> None:
        self.docker.cache = dict(self.FOREIGN)
        before = prune.build_cache_listing()
        self.chain("ours0a000000000000000000001", "ours0b000000000000000000001")
        # A build still running uses our first record: BuildKit keeps it, and every prune still exits 0.
        self.docker.cache_in_use.add("ours0a000000000000000000001")
        result = prune.prune_new_cache(before, 1)
        self.assertIn("ours0a000000000000000000001", self.docker.cache)
        self.assertNotIn("ours0b000000000000000000001", self.docker.cache)
        self.assertEqual((result["created"], result["pruned"], result["left"], result["failed"]), (2, 1, 1, 0))
        self.assertEqual(result["error"], "1 of the 2 records this build made are still there")

    def test_a_build_with_prune_cache_leaves_only_the_cache_it_found(self) -> None:
        self.docker.cache = {"foreign1000000000000000001": "another project"}
        manager = BuildManager(root=self.root)
        build = builder.Build(id="b1", kind="build", label="t", created_at="", jobs=1,
                              options={"prune_cache": True}, items=[])

        def fake_items(cfg: Any, b: Any, items: Any, log: Any) -> bool:
            self.chain("ours10000000000000000000001", "ours20000000000000000000002")
            return True

        with mock.patch.object(manager, "_run_items", fake_items):
            manager._builds["b1"] = build
            manager._active = "b1"
            manager._execute(mock.Mock(), build, [], contextlib.ExitStack())
        self.assertEqual(set(self.docker.cache), {"foreign1000000000000000001"})
        self.assertEqual(build.cache, {"before": 1, "source": "buildx", "created": 2, "pruned": 2, "left": 0,
                                       "passes": 1, "failed": 0, "error": None})
        self.assertIsNone(manager._cache_pending)

    def test_the_created_times_buildx_prints_are_read(self) -> None:
        self.assertEqual(prune._parse_created("2026-10-31 11:53:01.123456789 +0000 UTC"),
                         datetime(2026, 10, 31, 11, 53, 1, tzinfo=timezone.utc))
        self.assertEqual(prune._parse_created("2026-10-31 12:53:01 +0100 CET"),
                         datetime(2026, 10, 31, 11, 53, 1, tzinfo=timezone.utc))
        self.assertIsNone(prune._parse_created("an hour ago"))

    def test_an_emergency_stop_kills_the_builds_and_leaves_the_prune_for_the_next_build(self) -> None:
        self.docker.cache = dict(self.FOREIGN)
        manager = BuildManager(root=self.root)
        build = builder.Build(id="b1", kind="build", label="t", created_at="", jobs=1,
                              options={"prune_cache": True}, items=[])
        manager._builds["b1"] = build
        log = io.StringIO()
        before = manager._cache_before(build, log)
        self.assertIsNotNone(before)
        # The build makes its records, then the console closes: no finally block runs after this.
        self.chain("ours10000000000000000000001", "ours20000000000000000000002")
        now = go_time(datetime.now(timezone.utc))
        self.docker.cache_created.update({"ours10000000000000000000001": now, "ours20000000000000000000002": now})
        proc = mock.Mock(pid=4321)
        manager._build_procs.add(proc)
        killed: list[int] = []
        with mock.patch.object(procs, "kill_tree", lambda pid, timeout=30: killed.append(pid)):
            manager.emergency_stop()
        self.assertEqual(killed, [4321])
        self.assertTrue(manager._cancel.is_set())
        pending = json.loads((self.root / builder.CACHE_PENDING).read_text(encoding="utf-8"))
        self.assertEqual((pending["build_id"], pending["source"], sorted(pending["before"])),
                         ("b1", "buildx", sorted(self.FOREIGN)))
        # A record made long after that build stays, though it was not listed before it either.
        self.docker.cache["later0000000000000000000001"] = "made since"
        self.docker.cache_created["later0000000000000000000001"] = go_time(datetime.now(timezone.utc)
                                                                           + timedelta(hours=3))
        nxt = BuildManager(root=self.root)
        build2 = builder.Build(id="b2", kind="build", label="t", created_at="", jobs=1,
                               options={"prune_cache": True}, items=[])
        nxt._cache_before(build2, io.StringIO())
        self.assertEqual(set(self.docker.cache), {*self.FOREIGN, "later0000000000000000000001"})
        self.assertEqual((build2.cache["pending"]["build_id"], build2.cache["pending"]["pruned"]), ("b1", 2))
        self.assertFalse((self.root / builder.CACHE_PENDING).exists())

    def test_a_pending_prune_needs_buildx(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / builder.CACHE_PENDING).write_text(json.dumps({
            "build_id": "b1", "source": "df", "before": [], "images": 1,
            "started_at": "2026-10-31T10:00:00+00:00", "stopped_at": "2026-10-31T10:05:00+00:00"}), encoding="utf-8")
        self.docker.buildx = False
        self.docker.cache = {"ours10000000000000000000001": "x"}
        build = builder.Build(id="b2", kind="build", label="t", created_at="", jobs=1,
                              options={"prune_cache": True}, items=[])
        BuildManager(root=self.root)._cache_before(build, io.StringIO())
        self.assertIn("only buildx lists when each record was made", build.cache["pending"]["error"])
        self.assertEqual(self.docker.ran("builder", "prune"), [])

    def test_queue_images_in_a_race_never_build_first(self) -> None:
        cfg = self.load_test_config(RACE_YAML.format(budget=5, max_containers=6, prebake="require", keep="false",
                                                     attempts=1, n_concurrent=1), tasks=TASKS)
        run_spec = spec.spec_from_config(cfg, "three", ["a", "b"])
        with mock.patch.object(inventory, "readiness", lambda cfg, spec, refresh=False: {
                "a": inventory.ArmReadiness(arm="a", family="terminus-2", mode="require", supported=True, tasks=3),
                "b": inventory.ArmReadiness(arm="b", family="pi", mode="require", supported=False,
                                            reason="no pinned version")}):
            check = builder.QueueImages(BuildManager(root=self.root)).check(cfg, run_spec)
        self.assertEqual(check, {"todo": 0, "blocking": ["b cannot use pre-baked images (no pinned version)"],
                                 "race": True})
