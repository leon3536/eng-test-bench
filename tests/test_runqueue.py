"""Run queue: approval by spec hash, one design at a time while the run
manager is idle, and every reason the queue stops instead of spending.

The RunManager is a fake that records start_spec calls and finishes runs on
demand. Designs live in a real DesignStore under a temp folder, and
runs.plan_spec and datasets.task_meta are replaced, so no Harbor schema
command runs and no dataset snapshot is needed. Docker is a flag. Most tests
drive the worker one pass at a time (_step); one runs the real thread.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
import threading
import time
import types
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

from tests.support import BenchTestCase

from bench import designs, runqueue
from bench import spec as specs
from bench.designs import DesignNotFound, DesignStore, VersionConflict
from bench.runqueue import RunQueue
from bench.settings import Config, ConfigError

TASKS = ["task-a", "task-b", "task-c", "task-d"]
SNAPSHOT = {t: {"difficulty": "medium"} for t in TASKS}
CONVERSATION = "c-20261029-101500-ab12"

QUEUE_YAML = """\
dataset: terminal-bench@2.0
providers:
  anthropic: {env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}
  openrouter: {env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}
modes:
  pilot: {n_tasks: 1}
run:
  effort: medium
  n_concurrent: 2
  parallel_arms: 2
  budget_usd: 5
  per_trial_cap_usd: 1
  max_turns: 150
  schedule: arms
baseline_arm: cc-glm
arms:
  - {name: cc-glm, agent: claude-code, provider: openrouter, model: z-ai/glm-5.3}
  - {name: pi-glm, agent: pi, provider: openrouter, model: z-ai/glm-5.3}
designs:
  max_budget_usd: 10
"""

VIEW_KEYS = {"state", "pause_reason", "on_failure", "global_cap_usd", "cap_since", "cap_spent_usd",
             "cap_remaining_usd", "committed_usd", "active", "waiting", "held", "history"}
ITEM_KEYS = {"id", "design_id", "version", "spec_hash", "title", "budget_usd", "inflight_allowance_usd",
             "expected_usd", "p90_usd", "status", "message", "added_at", "started_at", "finished_at", "run_id",
             "spend_usd"}


def patch_lazy(case: BenchTestCase, module_name: str, attr: str, value: Any) -> None:
    """Patch an attribute of a module designs imports lazily. Other packages
    may be mid-edit, so a module that fails to import gets a stand-in."""
    try:
        importlib.import_module(module_name)
    except Exception:
        fake = types.ModuleType(module_name)
        package, _, leaf = module_name.rpartition(".")
        parent = sys.modules[package]
        sys.modules[module_name] = fake
        setattr(parent, leaf, fake)

        def undo() -> None:
            if sys.modules.get(module_name) is fake:
                del sys.modules[module_name]
            if getattr(parent, leaf, None) is fake:
                delattr(parent, leaf)

        case.addCleanup(undo)
    patcher = mock.patch(f"{module_name}.{attr}", value, create=True)
    patcher.start()
    case.addCleanup(patcher.stop)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


class FakeRunManager:
    """start_spec records the RunSpec and makes its run active; finish() ends
    it and calls the listeners, as RunManager does after its final save."""

    def __init__(self) -> None:
        self.started: list[specs.RunSpec] = []
        self.listeners: list[Callable[[dict[str, Any]], None]] = []
        self.runs: dict[str, dict[str, Any]] = {}
        self.active: str | None = None
        self.start_error: Exception | None = None
        self.finish_during_start: str | None = None

    def start_spec(self, cfg: Config, spec: specs.RunSpec) -> dict[str, Any]:
        if self.active is not None:
            raise RuntimeError("A run is already in progress")
        if self.start_error is not None:
            raise self.start_error
        run_id = f"20261029-1200{len(self.runs):02d}-{spec.label}"
        self.runs[run_id] = {
            "id": run_id, "created_at": now(), "status": "running", "message": "", "finished_at": None,
            "spend_usd": 0.0, "budget_usd": spec.run.budget_usd, "source": spec.source,
            "spec_hash": spec.spec_hash, "design_ref": copy.deepcopy(spec.design_ref),
        }
        self.active = run_id
        self.started.append(spec)
        if self.finish_during_start:
            self.finish(run_id, self.finish_during_start, 0.5)
        return copy.deepcopy(self.runs[run_id])

    def is_idle(self) -> bool:
        return self.active is None

    def add_listener(self, fn: Callable[[dict[str, Any]], None]) -> None:
        self.listeners.append(fn)

    def list(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(r) for r in self.runs.values()]

    def add_manual(self, spend: float, created_at: str, status: str = "done") -> str:
        run_id = f"manual-{len(self.runs)}"
        self.runs[run_id] = {"id": run_id, "created_at": created_at, "status": status, "spend_usd": spend,
                             "design_ref": None, "source": "config"}
        return run_id

    def finish(self, run_id: str, status: str, spend: float = 0.0, message: str = "") -> None:
        run = self.runs[run_id]
        run.update(status=status, spend_usd=spend, finished_at=now(), message=message)
        if self.active == run_id:
            self.active = None
        for fn in list(self.listeners):
            fn(copy.deepcopy(run))


class QueueTestCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cfg = self.load_test_config(QUEUE_YAML, tasks=TASKS)
        self.data = self.isolate_data_dir()
        self.store = DesignStore(self.data / "designs")
        self.path = self.data / "queue.json"
        self.plan_problems: list[str] = []
        patch_lazy(self, "bench.runs", "plan_spec", self.fake_plan)
        patch_lazy(self, "bench.datasets", "task_meta", lambda dataset: copy.deepcopy(SNAPSHOT))
        self.manager = FakeRunManager()
        self.docker = True
        self.estimate: dict[str, Any] | None = None
        self.estimate_calls: list[tuple[Config, dict[str, Any]]] = []

    def fake_plan(self, cfg: Config, run_spec: specs.RunSpec) -> dict[str, Any]:
        return {"source": run_spec.source, "label": run_spec.label, "spec_hash": run_spec.spec_hash,
                "tasks": list(run_spec.tasks), "problems": list(self.plan_problems), "warnings": [],
                "peak_containers": designs.peak_containers(run_spec.spec),
                "arms": [{"name": a.name, "warnings": []} for a in run_spec.arms]}

    def estimate_fn(self, cfg: Config, spec: dict[str, Any]) -> dict[str, Any]:
        self.estimate_calls.append((cfg, spec))
        assert self.estimate is not None
        return dict(self.estimate)

    def queue(self, **kwargs: Any) -> RunQueue:
        if "estimate_fn" not in kwargs and self.estimate is not None:
            kwargs["estimate_fn"] = self.estimate_fn
        q = RunQueue(self.path, self.manager, self.store, lambda: self.cfg, lambda: self.docker, **kwargs)
        self.addCleanup(q.shutdown, 1)
        return q

    def design(self, title: str = "GLM pairs", budget: float = 3.0, author: str = "user",
               conversation_id: str | None = None, **changes: Any) -> dict[str, Any]:
        spec_in: dict[str, Any] = {
            "tasks": ["task-a", "task-b"],
            "baseline_arm": "cc",
            "run": {"budget_usd": budget},
            "arms": [
                {"name": "cc", "agent": "claude-code", "provider": "openrouter", "model": "z-ai/glm-5.3"},
                {"name": "pi", "agent": "pi", "provider": "openrouter", "model": "z-ai/glm-5.3"},
            ],
        }
        spec_in.update(changes)
        spec = designs.normalize(self.cfg, spec_in)
        return self.store.create(self.cfg, {"title": title}, spec, author, conversation_id)

    def add(self, q: RunQueue, doc: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return q.add(doc["design_id"], doc["version"], doc["spec_hash"], **kwargs)

    def version_file(self, doc: dict[str, Any]) -> Path:
        return self.store.root / doc["design_id"] / f"v{doc['version']:04d}.json"

    def tamper(self, doc: dict[str, Any], **run_changes: Any) -> None:
        """Hand-edit a write-once version file, leaving its stored hash alone."""
        path = self.version_file(doc)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["spec"]["run"].update(run_changes)
        path.write_text(json.dumps(raw), encoding="utf-8", newline="\n")

    def saved(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def item(self, q: RunQueue, item_id: str) -> dict[str, Any]:
        view = q.view()
        for item in [*([view["active"]] if view["active"] else []), *view["waiting"], *view["held"],
                     *view["history"]]:
            if item["id"] == item_id:
                return item
        raise AssertionError(f"{item_id} is not in the queue")

    def write_queue(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8", newline="\n")

    def stored_item(self, doc: dict[str, Any], item_id: str, status: str, **fields: Any) -> dict[str, Any]:
        return {"id": item_id, "design_id": doc["design_id"], "version": doc["version"],
                "spec_hash": doc["spec_hash"], "title": doc["title"], "budget_usd": 3.0,
                "inflight_allowance_usd": 4.0, "expected_usd": None, "p90_usd": None, "status": status,
                "message": "", "added_at": "2026-10-29T10:00:00+00:00", "started_at": None, "finished_at": None,
                "run_id": None, "spend_usd": 0.0, **fields}


# Loading -------------------------------------------------------------------------


class LoadTests(QueueTestCase):
    def test_new_queue_is_paused_and_writes_nothing(self) -> None:
        q = self.queue()
        view = q.view()
        self.assertEqual(set(view), VIEW_KEYS)
        self.assertEqual((view["state"], view["pause_reason"]), ("paused", runqueue.NOT_STARTED))
        self.assertEqual((view["on_failure"], view["global_cap_usd"]), ("pause", None))
        self.assertFalse(self.path.exists())
        self.assertFalse(q.is_active())

    def test_file_saying_running_loads_paused_with_active_items_interrupted(self) -> None:
        a, b, c = self.design("Alpha"), self.design("Beta"), self.design("Gamma")
        self.write_queue({
            "schema": 1, "state": "running", "pause_reason": "", "next_seq": 4, "global_cap_usd": 7.5,
            "cap_since": "2026-10-01T00:00:00+00:00", "on_failure": "skip",
            "items": [self.stored_item(a, "q-0001", "running", run_id="20261029-120000-alpha-v1",
                                       started_at="2026-10-29T10:01:00+00:00"),
                      self.stored_item(b, "q-0002", "starting"),
                      self.stored_item(c, "q-0003", "waiting")],
        })
        q = self.queue()
        view = q.view()
        self.assertEqual((view["state"], view["pause_reason"]), ("paused", runqueue.PAUSED_AFTER_RESTART))
        self.assertEqual((view["on_failure"], view["global_cap_usd"], view["cap_since"]),
                         ("skip", 7.5, "2026-10-01T00:00:00+00:00"))
        self.assertIsNone(view["active"])
        self.assertEqual(self.item(q, "q-0001")["status"], "interrupted")
        self.assertIn("in progress", self.item(q, "q-0001")["message"])
        self.assertEqual(self.item(q, "q-0002")["status"], "interrupted")
        self.assertIn("before this run started", self.item(q, "q-0002")["message"])
        self.assertEqual([i["id"] for i in view["waiting"]], ["q-0003"])
        # The file is rewritten paused, so a crash right now still loads paused.
        self.assertEqual(self.saved()["state"], "paused")
        self.assertEqual({i["id"]: i["status"] for i in self.saved()["items"]},
                         {"q-0001": "interrupted", "q-0002": "interrupted", "q-0003": "waiting"})
        # Nothing starts until a person presses Start.
        self.assertFalse(q._step())
        self.assertEqual(self.manager.started, [])
        self.assertEqual(self.add(q, self.design("Delta"))["id"], "q-0004")

    def test_restart_after_a_real_session(self) -> None:
        a, b = self.design("Alpha"), self.design("Beta")
        q = self.queue()
        first = self.add(q, a)
        self.add(q, b)
        q.start()
        q._step()
        self.assertEqual(self.item(q, first["id"])["status"], "running")
        again = RunQueue(self.path, FakeRunManager(), self.store, lambda: self.cfg, lambda: True)
        self.addCleanup(again.shutdown, 1)
        view = again.view()
        self.assertEqual(view["state"], "paused")
        self.assertEqual(self.item(again, first["id"])["status"], "interrupted")
        self.assertEqual(len(view["waiting"]), 1)
        self.assertFalse(again._step())

    def test_corrupt_file_is_kept_aside_and_the_queue_starts_empty(self) -> None:
        self.write_queue({"schema": 1})
        self.path.write_text("{not json", encoding="utf-8")
        with mock.patch("sys.stderr"):
            q = self.queue()
        self.assertEqual(q.view()["waiting"], [])
        copies = list(self.path.parent.glob("queue.json.corrupt-*"))
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0].read_text(encoding="utf-8"), "{not json")
        self.assertEqual(self.saved()["items"], [])
        self.assertEqual(self.saved()["state"], "paused")

    def test_locked_corrupt_file_is_copied_not_renamed(self) -> None:
        self.write_queue({"schema": 1})
        self.path.write_text("[1, 2", encoding="utf-8")
        real_replace = Path.replace

        def locked_rename(path: Path, target: Any) -> Path:
            if ".corrupt-" in Path(target).name:
                raise PermissionError(13, "The process cannot access the file")
            return real_replace(path, target)

        with mock.patch.object(Path, "replace", locked_rename), mock.patch("sys.stderr"):
            q = self.queue()
        copies = list(self.path.parent.glob("queue.json.corrupt-*"))
        self.assertEqual([c.read_text(encoding="utf-8") for c in copies], ["[1, 2"])
        self.assertEqual(q.view()["waiting"], [])

    def test_a_locked_valid_file_is_copied_and_ids_continue(self) -> None:
        a = self.design("Alpha")
        self.write_queue({"schema": 1, "state": "paused", "pause_reason": "", "next_seq": 7,
                          "items": [self.stored_item(a, "q-0006", "waiting")]})
        original = self.path.read_text(encoding="utf-8")
        real_read = Path.read_text

        def locked(path: Path, *args: Any, **kwargs: Any) -> str:
            if path == self.path:
                raise PermissionError(13, "The process cannot access the file")
            return real_read(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", locked), mock.patch.object(runqueue, "SAVE_PATIENCE", 0.05), \
                mock.patch("sys.stderr"):
            q = self.queue()
        copies = list(self.path.parent.glob("queue.json.corrupt-*"))
        self.assertEqual([c.read_text(encoding="utf-8") for c in copies], [original])
        # Ids never restart at q-0001 after a load that kept nothing.
        self.assertEqual(self.add(q, self.design("Beta"))["id"], "q-0007")

    def test_an_uncopyable_locked_file_is_never_overwritten_unseen(self) -> None:
        a = self.design("Alpha")
        self.write_queue({"schema": 1, "next_seq": 3, "items": [self.stored_item(a, "q-0002", "waiting")]})
        original = self.path.read_text(encoding="utf-8")
        real_read = Path.read_text

        def locked(path: Path, *args: Any, **kwargs: Any) -> str:
            if path == self.path:
                raise PermissionError(13, "The process cannot access the file")
            return real_read(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", locked), mock.patch.object(runqueue, "SAVE_PATIENCE", 0.05), \
                mock.patch.object(runqueue.shutil, "copyfile", side_effect=PermissionError(13, "locked")), \
                mock.patch("sys.stderr"):
            q = self.queue()
            self.add(q, self.design("Beta"))
        self.assertEqual(self.path.read_text(encoding="utf-8"), original)
        # Once the lock is gone the original is copied first, then replaced.
        with mock.patch("sys.stderr"):
            q._step()
        copies = list(self.path.parent.glob("queue.json.corrupt-*"))
        self.assertEqual([c.read_text(encoding="utf-8") for c in copies], [original])
        self.assertEqual([i["title"] for i in self.saved()["items"]], ["Beta"])

    def test_unreadable_items_are_skipped_and_the_file_kept(self) -> None:
        a = self.design("Alpha")
        self.write_queue({"schema": 1, "state": "paused", "pause_reason": "", "next_seq": 9,
                          "global_cap_usd": "lots", "cap_since": "yesterday", "on_failure": "explode",
                          "items": [self.stored_item(a, "q-0002", "waiting"),
                                    {"id": "../../etc", "status": "waiting"},
                                    self.stored_item(a, "q-0003", "teleported"),
                                    self.stored_item(a, "q-0002", "held")]})
        with mock.patch("sys.stderr"):
            q = self.queue()
        view = q.view()
        self.assertEqual([i["id"] for i in view["waiting"]], ["q-0002"])
        self.assertEqual(view["held"], [])
        self.assertEqual(len(list(self.path.parent.glob("queue.json.corrupt-*"))), 1)
        # Invalid settings fall back to the profile's; ids continue after next_seq.
        self.assertEqual((view["global_cap_usd"], view["on_failure"]), (None, "pause"))
        self.assertIsNotNone(datetime.fromisoformat(view["cap_since"]))
        self.assertEqual(self.add(q, self.design("Beta"))["id"], "q-0009")

    def test_profile_settings_seed_a_new_queue(self) -> None:
        self.cfg = self.load_test_config(QUEUE_YAML + "queue:\n  on_failure: skip\n  global_cap_usd: 20\n",
                                         tasks=TASKS)
        view = self.queue().view()
        self.assertEqual((view["on_failure"], view["global_cap_usd"]), ("skip", 20.0))


# Admission -------------------------------------------------------------------------


class AddTests(QueueTestCase):
    def test_add_approves_the_version_and_records_figures(self) -> None:
        doc = self.design("Alpha")
        q = self.queue()
        item = self.add(q, doc)
        self.assertEqual(set(item), ITEM_KEYS)
        self.assertEqual((item["id"], item["status"], item["design_id"], item["version"], item["spec_hash"]),
                         ("q-0001", "waiting", doc["design_id"], 1, doc["spec_hash"]))
        self.assertEqual((item["title"], item["budget_usd"]), ("Alpha", 3.0))
        # No estimator: 4 containers can be busy (2 arms x 2 trials) at the $1 per-trial cap.
        self.assertEqual((item["inflight_allowance_usd"], item["expected_usd"], item["p90_usd"]), (4.0, None, None))
        self.assertIsNotNone(self.store.version(doc["design_id"], 1)["approved_at"])
        self.assertEqual([i["id"] for i in self.saved()["items"]], ["q-0001"])
        self.assertEqual(q.committed(), 7.0)
        self.assertEqual(q.view()["committed_usd"], 7.0)

    def test_add_uses_the_estimate(self) -> None:
        self.estimate = {"expected_usd": 1.2, "p90_usd": 2.5, "inflight_allowance_usd": 0.8}
        doc = self.design("Alpha")
        q = self.queue()
        item = self.add(q, doc)
        self.assertEqual((item["expected_usd"], item["p90_usd"], item["inflight_allowance_usd"]), (1.2, 2.5, 0.8))
        cfg, spec = self.estimate_calls[0]
        self.assertIs(cfg, self.cfg)
        self.assertEqual(specs.spec_hash(spec), doc["spec_hash"])

    def test_estimate_failure_falls_back_to_the_caps(self) -> None:
        def broken(cfg: Config, spec: dict[str, Any]) -> dict[str, Any]:
            raise ZeroDivisionError("no ledger rows")

        q = self.queue(estimate_fn=broken)
        with mock.patch("sys.stderr"):
            item = self.add(q, self.design(run={"budget_usd": 3, "per_trial_cap_usd": 0.5}))
        self.assertEqual((item["inflight_allowance_usd"], item["expected_usd"]), (2.0, None))
        # Without a per-trial cap there is nothing to bound a trial by.
        self.estimate = {"expected_usd": "soon", "inflight_allowance_usd": -1}
        self.path = self.data / "queue-2.json"
        q2 = self.queue(estimate_fn=self.estimate_fn)
        item = self.add(q2, self.design("No cap", run={"budget_usd": 3, "per_trial_cap_usd": None}))
        self.assertEqual((item["inflight_allowance_usd"], item["expected_usd"]), (0.0, None))

    def test_add_requires_the_approved_hash(self) -> None:
        doc = self.design("Alpha")
        q = self.queue()
        with self.assertRaises(VersionConflict) as ctx:
            q.add(doc["design_id"], 1, "sha256:" + "0" * 64)
        self.assertIn("approved", str(ctx.exception))
        self.assertEqual(q.view()["waiting"], [])
        self.assertIsNone(self.store.version(doc["design_id"], 1)["approved_at"])
        self.assertFalse(self.path.exists())

    def test_a_hand_edited_version_file_cannot_be_queued(self) -> None:
        doc = self.design("Alpha")
        self.tamper(doc, budget_usd=9.0)
        q = self.queue()
        with self.assertRaises(VersionConflict) as ctx:
            self.add(q, doc)
        self.assertIn("edited", str(ctx.exception))
        self.assertEqual(q.view()["waiting"], [])

    def test_validation_problems_refuse_with_a_config_error(self) -> None:
        doc = self.design("Alpha")
        q = self.queue()
        self.plan_problems = ["No price for openrouter/z-ai/glm-5.3 in config/pricing.yaml"]
        with self.assertRaises(ConfigError) as ctx:
            self.add(q, doc)
        self.assertIn("No price", str(ctx.exception))
        self.assertEqual(q.view()["waiting"], [])
        self.assertIsNone(self.store.version(doc["design_id"], 1)["approved_at"])

    def test_a_version_is_queued_once_until_it_has_run(self) -> None:
        doc = self.design("Alpha")
        q = self.queue()
        self.add(q, doc)
        with self.assertRaises(RuntimeError):
            self.add(q, doc)
        q.start()
        q._step()
        # Running is not waiting or held: the same version may be queued again, as a second run.
        again = self.add(q, doc)
        self.assertEqual(again["id"], "q-0002")
        with self.assertRaises(RuntimeError):
            self.add(q, doc)

    def test_archived_and_unknown_designs(self) -> None:
        doc = self.design("Alpha")
        q = self.queue()
        self.store.archive(doc["design_id"])
        with self.assertRaises(VersionConflict):
            self.add(q, doc)
        with self.assertRaises(DesignNotFound):
            q.add("d-nothing-here-0000", 1, doc["spec_hash"])
        with self.assertRaises(DesignNotFound):
            q.add(doc["design_id"], 2, doc["spec_hash"])
        for bad_id in ("../../config", "D-UPPER", "con"):
            with self.subTest(design_id=bad_id), self.assertRaises(ValueError):
                q.add(bad_id, 1, doc["spec_hash"])

    def test_bad_arguments(self) -> None:
        doc = self.design("Alpha")
        q = self.queue()
        with self.assertRaises(ValueError):
            self.add(q, doc, position="middle")
        with self.assertRaises(ValueError):
            q.add(doc["design_id"], 1, None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            q.add(doc["design_id"], True, doc["spec_hash"])  # type: ignore[arg-type]

    def test_front_goes_before_every_waiting_item(self) -> None:
        a, b, c = self.design("Alpha"), self.design("Beta"), self.design("Gamma")
        q = self.queue()
        self.add(q, a)
        self.add(q, b)
        self.add(q, c, position="front")
        self.assertEqual([i["title"] for i in q.view()["waiting"]], ["Gamma", "Alpha", "Beta"])
        self.assertEqual([i["position"] for i in q.view()["waiting"]], [0, 1, 2])


# The worker ------------------------------------------------------------------------


class WorkerTests(QueueTestCase):
    def test_items_run_in_order_one_at_a_time_and_only_when_idle(self) -> None:
        a, b, c = self.design("Alpha"), self.design("Beta"), self.design("Gamma")
        q = self.queue()
        for doc in (a, b, c):
            self.add(q, doc)
        self.assertFalse(q._step())  # paused
        self.assertEqual(self.manager.started, [])
        q.start()
        self.assertTrue(q.is_active())
        manual = self.manager.add_manual(0.0, now(), status="running")
        self.manager.active = manual
        self.assertFalse(q._step())
        self.assertEqual(self.manager.started, [])
        self.manager.finish(manual, "done")

        titles = []
        for doc in (a, b, c):
            q._step()
            self.assertEqual(len(self.manager.started), len(titles) + 1)
            started = self.manager.started[-1]
            self.assertEqual(started.design_ref["design_id"], doc["design_id"])
            titles.append(doc["title"])
            # Nothing else starts while this one runs.
            self.assertFalse(q._step())
            self.assertEqual(len(self.manager.started), len(titles))
            self.assertEqual(q.view()["active"]["title"], doc["title"])
            self.manager.finish(self.manager.active, "done", spend=1.0)
        self.assertFalse(q._step())
        view = q.view()
        self.assertEqual(view["state"], "paused")
        self.assertIn("Queue finished", view["pause_reason"])
        self.assertEqual([i["title"] for i in view["history"]], ["Gamma", "Beta", "Alpha"])
        self.assertFalse(q.is_active())

    def test_start_runs_exactly_the_approved_spec(self) -> None:
        doc = self.store.create(self.cfg, {"title": "Pairs on GLM"}, designs.normalize(self.cfg, {
            "tasks": ["task-a"], "run": {"budget_usd": 2},
            "arms": [{"name": "pi", "agent": "pi", "provider": "openrouter", "model": "z-ai/glm-5.3"}],
        }), "assistant", CONVERSATION)
        q = self.queue()
        item = self.add(q, doc)
        q.start()
        q._step()
        run_spec = self.manager.started[0]
        self.assertEqual((run_spec.source, run_spec.spec_hash, run_spec.spec), ("design", doc["spec_hash"],
                                                                                 doc["spec"]))
        self.assertEqual(run_spec.design_ref, {"design_id": doc["design_id"], "version": 1, "author": "assistant",
                                               "conversation_id": CONVERSATION, "queue_item": item["id"]})
        self.assertTrue(run_spec.label.endswith("-v1"))
        active = q.view()["active"]
        self.assertEqual((active["status"], active["run_id"]), ("running", self.manager.active))
        self.assertIsNotNone(active["started_at"])
        self.assertEqual(self.store.version(doc["design_id"], 1)["runs"], [self.manager.active])
        self.assertEqual(self.saved()["items"][0]["status"], "running")

    def test_hash_mismatch_at_start_holds_and_pauses(self) -> None:
        a, b = self.design("Alpha"), self.design("Beta")
        q = self.queue()
        first = self.add(q, a)
        self.add(q, b)
        self.tamper(a, budget_usd=9.0)
        q.start()
        q._step()
        self.assertEqual(self.manager.started, [])
        held = self.item(q, first["id"])
        self.assertEqual(held["status"], "held")
        self.assertIn("edited", held["message"])
        self.assertEqual(q.view()["state"], "paused")
        self.assertIn("Alpha v1", q.view()["pause_reason"])

    def test_hash_mismatch_under_skip_moves_on(self) -> None:
        a, b = self.design("Alpha"), self.design("Beta")
        q = self.queue()
        q.update_settings(on_failure="skip")
        first = self.add(q, a)
        self.add(q, b)
        # The stored hash itself was changed: not the version that was approved.
        path = self.version_file(a)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["spec_hash"] = "sha256:" + "1" * 64
        path.write_text(json.dumps(raw), encoding="utf-8")
        q.start()
        while q._step():
            pass
        self.assertEqual(self.item(q, first["id"])["status"], "held")
        self.assertIn("not the", self.item(q, first["id"])["message"])
        self.assertEqual([s.design_ref["design_id"] for s in self.manager.started], [b["design_id"]])
        self.assertEqual(q.view()["state"], "running")

    def test_validation_problem_at_start_follows_on_failure(self) -> None:
        for policy, state, started in (("pause", "paused", 0), ("skip", "running", 1)):
            with self.subTest(on_failure=policy):
                self.manager = FakeRunManager()
                self.path = self.data / f"queue-{policy}.json"
                self.plan_problems = []
                q = self.queue()
                q.update_settings(on_failure=policy)
                first = self.add(q, self.design("Alpha"))
                self.add(q, self.design("Beta", budget=2.0))
                q.start()
                # The config changed after approval: only Alpha's arms are now unpriced.
                self.plan_problems = []
                original = self.fake_plan

                def plan(cfg: Config, run_spec: specs.RunSpec) -> dict[str, Any]:
                    out = original(cfg, run_spec)
                    if run_spec.spec["run"]["budget_usd"] == 3.0:
                        out["problems"] = ["No price for openrouter/z-ai/glm-5.3 in config/pricing.yaml"]
                    return out

                with mock.patch("bench.runs.plan_spec", plan, create=True):
                    while q._step():
                        pass
                held = self.item(q, first["id"])
                self.assertEqual(held["status"], "held")
                self.assertIn("No price", held["message"])
                self.assertEqual(q.view()["state"], state)
                self.assertEqual(len(self.manager.started), started)

    def test_a_reuse_version_the_ledger_moved_past_is_held_until_retried(self) -> None:
        drift = ["Arm 'cc': task-b now has enough usable trials; save a new version to reuse them"]
        asked: list[str] = []

        def reuse_drift(cfg: Config, spec: dict[str, Any]) -> list[str]:
            asked.append(spec["reuse"])
            return list(drift)

        q = self.queue(reuse_drift=reuse_drift)
        plain = self.add(q, self.design("Plain"))  # reuses nothing: never looked up
        gaps = self.add(q, self.design("Gaps", reuse="measured"))
        q.start()
        q._step()
        self.manager.finish(self.manager.active, "done")
        q._step()
        held = self.item(q, gaps["id"])
        self.assertEqual((held["status"], len(self.manager.started)), ("held", 1))
        self.assertIn("task-b now has enough", held["message"])
        self.assertIn("retry this item to run it as approved", held["message"])
        q.retry(gaps["id"])
        drift.append("Arm 'pi': task-a now has enough usable trials; save a new version to reuse them")
        q.start()
        q._step()
        self.assertEqual(self.item(q, gaps["id"])["status"], "held")  # the ledger moved on again since the retry
        q.retry(gaps["id"])
        q.start()
        q._step()
        self.assertEqual([s.design_ref["queue_item"] for s in self.manager.started], [plain["id"], gaps["id"]])
        self.assertEqual(self.manager.started[1].spec_hash, held["spec_hash"])  # exactly as approved
        self.assertEqual(asked, ["measured"] * 3)

    def test_docker_down_holds_and_pauses_even_under_skip(self) -> None:
        q = self.queue()
        q.update_settings(on_failure="skip")
        first = self.add(q, self.design("Alpha"))
        self.add(q, self.design("Beta"))
        q.start()
        self.docker = False
        q._step()
        self.assertEqual(self.manager.started, [])
        self.assertEqual(self.item(q, first["id"])["status"], "held")
        view = q.view()
        self.assertEqual((view["state"], view["pause_reason"]), ("paused", runqueue.DOCKER_DOWN))
        self.assertEqual(len(view["waiting"]), 1)

    def test_global_cap_counts_the_in_flight_allowance(self) -> None:
        self.estimate = {"expected_usd": 2.0, "p90_usd": 2.8, "inflight_allowance_usd": 1.5}
        q = self.queue()
        q.update_settings(global_cap_usd=5.0, on_failure="skip")
        self.manager.add_manual(1.0, q.view()["cap_since"])
        item = self.add(q, self.design("Alpha", budget=3.0))
        q.start()
        q._step()
        # 1.00 spent + 3.00 budget + 1.50 allowance > 5.00, though the budget alone fits.
        self.assertEqual(self.manager.started, [])
        held = self.item(q, item["id"])
        self.assertEqual(held["status"], "held")
        self.assertIn("in-flight allowance", held["message"])
        view = q.view()
        self.assertEqual(view["state"], "paused")
        self.assertIn("Global cap", view["pause_reason"])
        self.assertEqual((view["cap_spent_usd"], view["cap_remaining_usd"]), (1.0, 4.0))
        # Exactly at the cap is allowed.
        q.update_settings(global_cap_usd=5.5)
        q.retry(item["id"])
        q.start()
        q._step()
        self.assertEqual(len(self.manager.started), 1)

    def test_start_spec_error_holds_with_its_message(self) -> None:
        q = self.queue()
        item = self.add(q, self.design("Alpha"))
        q.start()
        self.manager.start_error = ConfigError("Arm 'pi': harbor agents schema failed")
        q._step()
        held = self.item(q, item["id"])
        self.assertEqual(held["status"], "held")
        self.assertIn("harbor agents schema failed", held["message"])
        self.assertEqual(q.view()["state"], "paused")

    def test_a_config_that_no_longer_loads_holds_the_item(self) -> None:
        broken = {"on": False}

        def cfg_getter() -> Config:
            if broken["on"]:
                raise ConfigError("bench.yaml is not valid YAML")
            return self.cfg

        q = RunQueue(self.path, self.manager, self.store, cfg_getter, lambda: True)
        self.addCleanup(q.shutdown, 1)
        item = self.add(q, self.design("Alpha"))
        q.start()
        broken["on"] = True
        q._step()
        held = self.item(q, item["id"])
        self.assertEqual(held["status"], "held")
        self.assertIn("not valid YAML", held["message"])
        self.assertEqual(self.manager.started, [])

    def test_an_unexpected_error_holds_the_item_instead_of_blocking_the_queue(self) -> None:
        q = self.queue()
        q.update_settings(on_failure="skip")
        first = self.add(q, self.design("Alpha"))
        self.add(q, self.design("Beta"))
        q.start()
        real_validate = designs.validate
        calls = {"n": 0}

        def breaks_once(cfg: Config, spec: Any) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return real_validate(cfg, spec)

        with mock.patch.object(designs, "validate", breaks_once):
            while q._step():
                pass
        held = self.item(q, first["id"])
        self.assertEqual(held["status"], "held")
        self.assertIn("boom", held["message"])
        self.assertEqual(len(self.manager.started), 1)
        self.assertEqual(q.view()["active"]["title"], "Beta")

    def test_pause_while_an_item_is_checked_keeps_its_place(self) -> None:
        q = self.queue()
        item = self.add(q, self.design("Alpha"))
        q.start()
        real_validate = designs.validate

        def pausing(cfg: Config, spec: Any) -> dict[str, Any]:
            result = real_validate(cfg, spec)
            q.pause()
            return result

        with mock.patch.object(designs, "validate", pausing):
            q._step()
        self.assertEqual(self.manager.started, [])
        self.assertEqual(self.item(q, item["id"])["status"], "waiting")
        self.assertEqual((q.view()["state"], q.view()["pause_reason"]), ("paused", runqueue.PAUSED_BY_YOU))

    def test_a_run_that_ends_before_start_returns_is_not_left_running(self) -> None:
        q = self.queue()
        item = self.add(q, self.design("Alpha"))
        q.start()
        self.manager.finish_during_start = "done"
        q._step()
        done = self.item(q, item["id"])
        self.assertEqual((done["status"], done["spend_usd"]), ("done", 0.5))
        self.assertEqual(done["run_id"], next(iter(self.manager.runs)))
        self.assertIsNone(q.view()["active"])


class ListenerTests(QueueTestCase):
    def test_a_manager_without_listeners_is_refused(self) -> None:
        class NoListeners:
            def is_idle(self) -> bool:
                return True

            def list(self) -> list[dict[str, Any]]:
                return []

        with self.assertRaises(TypeError):
            RunQueue(self.path, NoListeners(), self.store, lambda: self.cfg, lambda: True)

    def test_a_lost_finish_is_recovered_from_the_run_list(self) -> None:
        q = self.queue()
        first = self.add(q, self.design("Alpha"))
        q.start()
        q._step()
        run_id = self.manager.active
        self.manager.listeners.clear()  # the notification is lost
        self.manager.finish(run_id, "done", spend=0.75)
        self.assertEqual(self.item(q, first["id"])["status"], "running")
        q._step()
        item = self.item(q, first["id"])
        self.assertEqual((item["status"], item["spend_usd"]), ("done", 0.75))
        self.assertFalse(q.is_active())

    def test_a_start_refused_because_the_bench_is_closing_keeps_the_item_waiting(self) -> None:
        q = self.queue()
        first = self.add(q, self.design("Alpha"))
        q.start()
        self.manager.start_error = RuntimeError("The bench is shutting down, so no new run can start")
        self.manager._closing = True
        q._step()
        self.assertEqual(self.item(q, first["id"])["status"], "waiting")
        self.assertEqual(q.view()["state"], "running")


class OutcomeTests(QueueTestCase):
    CASES = [
        # run status, on_failure, item status, queue carries on
        ("done", "pause", "done", True),
        ("done_with_errors", "pause", "done", True),
        ("budget_stopped", "pause", "budget_stopped", True),
        ("failed", "pause", "failed", False),
        ("interrupted", "pause", "interrupted", False),
        ("cancelled", "pause", "cancelled", False),
        ("failed", "skip", "failed", True),
        ("interrupted", "skip", "interrupted", True),
        ("cancelled", "skip", "cancelled", False),
        ("something-new", "skip", "failed", False),
    ]

    def test_outcome_mapping(self) -> None:
        a, b = self.design("Alpha"), self.design("Beta")
        for n, (status, policy, item_status, carries_on) in enumerate(self.CASES):
            with self.subTest(status=status, on_failure=policy):
                self.manager = FakeRunManager()
                self.path = self.data / f"queue-{n}.json"
                q = self.queue()
                q.update_settings(on_failure=policy)
                first = self.add(q, a)
                self.add(q, b)
                q.start()
                q._step()
                run_id = self.manager.active
                self.manager.finish(run_id, status, spend=1.25, message="" if status != "failed" else "Arm pi failed")
                item = self.item(q, first["id"])
                self.assertEqual((item["status"], item["spend_usd"], item["run_id"]), (item_status, 1.25, run_id))
                self.assertIsNotNone(item["finished_at"])
                if status == "failed":
                    self.assertEqual(item["message"], "Arm pi failed")
                view = q.view()
                if carries_on:
                    self.assertEqual(view["state"], "running")
                    q._step()
                    self.assertEqual(len(self.manager.started), 2)
                else:
                    self.assertEqual(view["state"], "paused")
                    self.assertIn("Alpha v1", view["pause_reason"])
                    self.assertFalse(q._step())
                    self.assertEqual(len(self.manager.started), 1)
                self.assertEqual(self.saved()["items"][0]["status"], item_status)

    def test_other_runs_are_ignored(self) -> None:
        q = self.queue()
        first = self.add(q, self.design("Alpha"))
        before = q.view()
        manual = self.manager.add_manual(2.0, now(), status="running")
        self.manager.finish(manual, "failed")
        # A run naming a waiting item is not that item's run.
        q.on_run_finished({"id": "x", "status": "cancelled", "design_ref": {"queue_item": first["id"]}})
        q.on_run_finished("not a run")  # type: ignore[arg-type]
        after = q.view()
        self.assertEqual(after["waiting"], before["waiting"])
        self.assertEqual(after["state"], "paused")
        self.assertEqual(after["pause_reason"], runqueue.NOT_STARTED)

    def test_a_finish_is_recorded_once(self) -> None:
        q = self.queue()
        q.update_settings(on_failure="skip")
        self.add(q, self.design("Alpha"))
        q.start()
        q._step()
        run = self.manager.runs[self.manager.active]
        self.manager.finish(run["id"], "done", spend=1.0)
        # A second listener call (registered twice) changes nothing.
        q.on_run_finished({**self.manager.runs[run["id"]], "status": "cancelled", "spend_usd": 9.0})
        self.assertEqual(q.view()["history"][0]["status"], "done")
        self.assertEqual(q.view()["history"][0]["spend_usd"], 1.0)


# Editing and settings ------------------------------------------------------------------


class EditTests(QueueTestCase):
    def test_remove_and_move_only_waiting_or_held_items(self) -> None:
        a, b, c = self.design("Alpha"), self.design("Beta"), self.design("Gamma")
        q = self.queue()
        ia, ib, ic = (self.add(q, d)["id"] for d in (a, b, c))
        q.move(ic, 0)
        self.assertEqual([i["id"] for i in q.view()["waiting"]], [ic, ia, ib])
        q.move(ia, 99)
        self.assertEqual([i["id"] for i in q.view()["waiting"]], [ic, ib, ia])
        q.move(ia, -5)
        self.assertEqual([i["id"] for i in q.view()["waiting"]], [ia, ic, ib])
        with self.assertRaises(ValueError):
            q.move(ia, True)  # type: ignore[arg-type]
        q.start()
        q._step()
        self.assertEqual(q.view()["active"]["id"], ia)
        for action in (lambda: q.remove(ia), lambda: q.move(ia, 1)):
            with self.assertRaises(RuntimeError):
                action()
        q.remove(ib)
        self.assertEqual([i["id"] for i in q.view()["waiting"]], [ic])
        self.manager.finish(self.manager.active, "done")
        with self.assertRaises(RuntimeError):
            q.remove(ia)
        with self.assertRaises(KeyError):
            q.remove("q-9999")
        self.assertNotIn(ib, [i["id"] for i in self.saved()["items"]])

    def test_retry_checks_the_version_again(self) -> None:
        a = self.design("Alpha")
        q = self.queue()
        item = self.add(q, a)
        q.start()
        self.docker = False
        q._step()
        self.assertEqual(self.item(q, item["id"])["status"], "held")
        self.plan_problems = ["Arm 'pi': no OPENROUTER_API_KEY"]
        with self.assertRaises(ConfigError):
            q.retry(item["id"])
        held = self.item(q, item["id"])
        self.assertEqual(held["status"], "held")
        self.assertIn("OPENROUTER_API_KEY", held["message"])
        self.plan_problems = []
        q.retry(item["id"])
        self.assertEqual(self.item(q, item["id"])["status"], "waiting")
        self.assertEqual(q.view()["state"], "paused")  # retry never starts the queue
        with self.assertRaises(RuntimeError):
            q.retry(item["id"])
        self.docker = True
        q.start()
        self.tamper(a, budget_usd=8.0)
        q._step()
        self.assertEqual(self.item(q, item["id"])["status"], "held")
        with self.assertRaises(VersionConflict):
            q.retry(item["id"])

    def test_start_needs_work_and_docker(self) -> None:
        q = self.queue()
        with self.assertRaises(RuntimeError) as ctx:
            q.start()
        self.assertIn("Nothing is waiting", str(ctx.exception))
        item = self.add(q, self.design("Alpha"))
        self.docker = False
        with self.assertRaises(RuntimeError) as ctx:
            q.start()
        self.assertIn(runqueue.DOCKER_DOWN, str(ctx.exception))
        self.assertEqual(q.view()["state"], "paused")
        self.docker = True
        q.start()
        self.docker = False
        q._step()
        self.assertEqual(self.item(q, item["id"])["status"], "held")
        self.docker = True
        with self.assertRaises(RuntimeError) as ctx:
            q.start()
        self.assertIn("held", str(ctx.exception))

    def test_pause_lets_the_current_run_finish(self) -> None:
        q = self.queue()
        self.add(q, self.design("Alpha"))
        self.add(q, self.design("Beta"))
        q.start()
        q._step()
        view = q.pause()
        self.assertEqual((view["state"], view["pause_reason"]), ("paused", runqueue.PAUSED_BY_YOU))
        self.assertTrue(q.is_active())  # its run is still going
        self.manager.finish(self.manager.active, "done")
        self.assertFalse(q._step())
        self.assertEqual(len(self.manager.started), 1)
        self.assertFalse(q.is_active())
        q.start()
        q._step()
        self.assertEqual(len(self.manager.started), 2)

    def test_update_settings(self) -> None:
        q = self.queue()
        for bad in (0, -1, "5", True, float("nan"), float("inf")):
            with self.subTest(cap=bad), self.assertRaises(ValueError):
                q.update_settings(global_cap_usd=bad)
        with self.assertRaises(ValueError):
            q.update_settings(on_failure="stop")
        with self.assertRaises(ValueError):
            q.update_settings(reset_cap="yes")  # type: ignore[arg-type]
        view = q.update_settings(global_cap_usd=12)
        self.assertEqual(view["global_cap_usd"], 12.0)
        self.assertEqual(q.update_settings(on_failure="skip")["global_cap_usd"], 12.0)  # UNSET keeps it
        self.assertIsNone(q.update_settings(global_cap_usd=None)["global_cap_usd"])
        self.assertEqual(self.saved()["on_failure"], "skip")

    def test_cap_spent_counts_every_run_since_cap_since(self) -> None:
        self.write_queue({"schema": 1, "state": "paused", "pause_reason": "", "next_seq": 1,
                          "global_cap_usd": 10, "cap_since": "2000-01-01T00:00:00+00:00", "on_failure": "pause",
                          "items": []})
        self.manager.add_manual(50.0, "1999-12-31T23:00:00+00:00")
        self.manager.add_manual(1.25, "2001-01-01T00:00:00+00:00")
        self.manager.add_manual(0.5, "not a time")  # counted: the cap errs on the side of stopping
        self.manager.runs["queued"] = {"id": "queued", "created_at": "2000-06-01T00:00:00", "spend_usd": 2.0,
                                       "status": "done", "design_ref": {"queue_item": "q-0001"}}
        q = self.queue()
        self.assertAlmostEqual(q.cap_spent(), 3.75)
        view = q.view()
        self.assertEqual((view["cap_spent_usd"], view["cap_remaining_usd"]), (3.75, 6.25))
        view = q.update_settings(reset_cap=True)
        self.assertNotEqual(view["cap_since"], "2000-01-01T00:00:00+00:00")
        self.assertAlmostEqual(q.cap_spent(), 0.5)

    def test_version_statuses_feed_the_design_store(self) -> None:
        a, b = self.design("Alpha"), self.design("Beta")
        q = self.queue()
        self.add(q, a)
        self.add(q, b)
        q.start()
        q._step()
        statuses = q.version_statuses()
        self.assertEqual(statuses, {(a["design_id"], 1): "running", (b["design_id"], 1): "waiting"})
        self.assertEqual(self.store.version(a["design_id"], 1, queue_status=statuses)["status"], "running")
        self.assertEqual(self.store.version(b["design_id"], 1, queue_status=statuses)["status"], "queued")

    def test_summary_for_the_state_endpoint(self) -> None:
        q = self.queue()
        q.update_settings(global_cap_usd=20)
        self.add(q, self.design("Alpha"))
        summary = q.summary()
        self.assertEqual(summary, {"state": "paused", "pause_reason": runqueue.NOT_STARTED, "active_item": None,
                                   "waiting": 1, "held": 0, "cap_remaining_usd": 20.0})


# Persistence and the thread -----------------------------------------------------------


class PersistenceTests(QueueTestCase):
    def test_saves_through_brief_permission_errors(self) -> None:
        q = self.queue()
        real_replace = Path.replace
        failures = {"left": 2}

        def flaky(path: Path, target: Any) -> Path:
            if Path(target).name == "queue.json" and failures["left"]:
                failures["left"] -= 1
                raise PermissionError(13, "Access is denied")
            return real_replace(path, target)

        with mock.patch.object(Path, "replace", flaky):
            item = self.add(q, self.design("Alpha"))
        self.assertEqual(failures["left"], 0)
        self.assertEqual([i["id"] for i in self.saved()["items"]], [item["id"]])

    def test_memory_stays_authoritative_while_the_file_is_locked(self) -> None:
        q = self.queue()
        self.add(q, self.design("Alpha"))
        real_replace = Path.replace

        def locked(path: Path, target: Any) -> Path:
            if Path(target).name == "queue.json":
                raise PermissionError(13, "Access is denied")
            return real_replace(path, target)

        with mock.patch.object(runqueue, "SAVE_PATIENCE", 0.05), mock.patch.object(Path, "replace", locked), \
                mock.patch("sys.stderr"):
            item = self.add(q, self.design("Beta"))
            self.assertEqual(len(q.view()["waiting"]), 2)
            self.assertEqual(len(self.saved()["items"]), 1)
        self.assertEqual(list(self.path.parent.glob(".queue.json.*.tmp")), [])
        q._step()  # the worker's next pass saves what is owed
        self.assertEqual([i["id"] for i in self.saved()["items"]], ["q-0001", item["id"]])

    def test_shutdown_persists_paused_and_refuses_new_starts(self) -> None:
        q = self.queue()
        self.add(q, self.design("Alpha"))
        self.add(q, self.design("Beta"))
        q.start()
        q._step()
        self.assertEqual(self.saved()["state"], "running")
        q.shutdown()
        q.shutdown()  # idempotent
        saved = self.saved()
        self.assertEqual((saved["state"], saved["pause_reason"]), ("paused", runqueue.STOPPED))
        with self.assertRaises(RuntimeError):
            q.start()
        self.assertFalse(q._step())
        # The run stopped by RunManager.shutdown is still recorded.
        self.manager.finish(self.manager.active, "cancelled")
        self.assertEqual(self.saved()["items"][0]["status"], "cancelled")


class ThreadTests(QueueTestCase):
    def test_worker_thread_runs_the_queue(self) -> None:
        q = self.queue()
        q.start_worker()
        q.start_worker()  # one thread only
        self.add(q, self.design("Alpha"))
        self.add(q, self.design("Beta"))
        self.assertEqual([t.name for t in threading.enumerate()].count("run-queue"), 1)
        q.start()
        self.assertTrue(wait_for(lambda: len(self.manager.started) == 1))
        self.manager.finish(self.manager.active, "done", spend=0.4)
        self.assertTrue(wait_for(lambda: len(self.manager.started) == 2))
        self.manager.finish(self.manager.active, "budget_stopped", spend=3.1)
        self.assertTrue(wait_for(lambda: q.view()["state"] == "paused"))
        self.assertIn("Queue finished", q.view()["pause_reason"])
        self.assertEqual([i["status"] for i in q.view()["history"]], ["budget_stopped", "done"])
        thread = q._thread
        q.shutdown(timeout=5)
        self.assertIsNotNone(thread)
        self.assertFalse(thread.is_alive())  # type: ignore[union-attr]
