"""The run manager: one execution path for config and design runs, and the
supervision that keeps a run within its budget and its containers accounted
for.

Harbor never runs. RunManager._supervise is real, but the command it starts is
swapped for a small python script (FAKE_HARBOR) that writes trial folders the
way Harbor does, sleeps, or exits with a code. Container cleanup, process-tree
kills and the detached cleanup helper are fakes; docker and taskkill are
refused by BenchTestCase. Run data lives in a temp DATA_DIR.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from unittest import mock

from tests.support import BenchTestCase

from bench import checks, datasets, procs, runs, settings, spec

POLL = 0.05

FAKE_HARBOR = r'''
import json, os, sys, time
job_dir, phase, scenario, calls = sys.argv[1:5]
with open(calls, "a", encoding="utf-8") as f:
    f.write(f"{os.path.basename(job_dir)} {phase}\n")

def trial(name, cost=None):
    d = os.path.join(job_dir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"trial_name": name}, f)
    if cost is not None:
        result = {"task_name": "task-a", "trial_name": name,
                  "agent_result": {"n_input_tokens": 0, "n_cache_tokens": 0, "n_output_tokens": 0,
                                   "cost_usd": cost},
                  "verifier_result": {"rewards": {"reward": 1.0}}}
        tmp = os.path.join(d, "result.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f)
        os.replace(tmp, os.path.join(d, "result.json"))

def timed_out(name):
    # Harbor's record of a timeout before any model call completed: 0/0/0, no cost.
    d = os.path.join(job_dir, name)
    os.makedirs(d, exist_ok=True)
    result = {"task_name": "task-a", "trial_name": name,
              "agent_result": {"n_input_tokens": 0, "n_cache_tokens": 0, "n_output_tokens": 0, "cost_usd": None},
              "agent_execution": {"started_at": "2026-10-29T13:14:33Z", "finished_at": "2026-10-29T13:29:33Z"},
              "exception_info": {"exception_type": "AgentTimeoutError", "exception_message": "timed out"},
              "verifier_result": {"rewards": {"reward": 0.0}}}
    with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f)

if phase == "dry":
    if scenario == "fail-dry":
        sys.exit(1)
    if scenario == "slow-dry":
        time.sleep(2.5)
    sys.exit(0)
if scenario == "ok":
    trial("t1", 0.25)
    sys.exit(0)
if scenario == "unrecorded":
    trial("t1", 0.25)
    timed_out("t2")
    sys.exit(0)
if scenario == "slow-ok":
    trial("t1", 0.25)
    time.sleep(0.6)
    sys.exit(0)
if scenario == "spend":
    trial("t1", 50.0)
    time.sleep(30)
if scenario == "exit3":
    trial("t1")
    sys.exit(3)
if scenario == "reserve":
    trial("t2")
    trial("t1", 4.0)
    time.sleep(30)
if scenario == "hang":
    trial("t1")
    time.sleep(30)
sys.exit(9)
'''

YAML = """\
dataset: terminal-bench@2.0
providers:
  anthropic: {{env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}}
  openrouter: {{env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}}
modes:
  pilot: {{n_tasks: 1}}
run:
  effort: medium
  n_concurrent: 1
  parallel_arms: {parallel}
  budget_usd: {budget}
  max_turns: 50
baseline_arm: a
arms:
  - {{name: a, agent: terminus-2, provider: anthropic, model: claude-opus-5-5}}
  - {{name: b, agent: pi, provider: anthropic, model: claude-opus-5-5}}
  - name: m
    agent: bench.agents.mini_swe_agent:MiniSweAgentWithMcp
    family: mini-swe-agent
    provider: openrouter
    model: deepseek/deepseek-v4.1-flash
"""

# Read-only copies of the two dev pilot run.json files (kept by the ledger tests).
DEV_RUNS = Path(__file__).resolve().parent / "fixtures" / "ledger" / "dev" / "runs"
CHILD_SECOND_BENCH = r'''
from bench import procs, runs
calls = []
def fake(job_dir):
    calls.append(str(job_dir))
    return 0
procs.cleanup_containers = fake
runs.cleanup_containers = fake
try:
    runs.RunManager()
except procs.AlreadyRunning as exc:
    print("refused:", exc)
else:
    print("started")
print("cleanups:", len(calls))
'''


class RunsTestCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = self.isolate_data_dir()
        self.work = self.tmpdir()
        self.script = self.work / "fake_harbor.py"
        self.script.write_text(FAKE_HARBOR, encoding="utf-8")
        self.calls_file = self.work / "calls.txt"
        self.scenarios: dict[str, str] = {}
        self.commands: list[list[str]] = []
        self.cleanups: list[Path] = []
        self.cleanup_mode = "ok"
        self.release = threading.Event()
        self.killed: list[int] = []
        self.spawned: list[list[Path]] = []
        self.managers: list[runs.RunManager] = []

        real_supervise = runs.RunManager._supervise
        case = self

        def supervise(manager: runs.RunManager, cmd: list[str], env: dict[str, str], log_path: Path,
                      run: runs.Run, arm_run: runs.ArmRun, watch: Any) -> int:
            case.commands.append(list(cmd))
            job_dir = Path(cmd[cmd.index("--jobs-dir") + 1]) / cmd[cmd.index("--job-name") + 1]
            phase = "dry" if "--dry-run" in cmd else "real"
            fake = [sys.executable, str(case.script), str(job_dir), phase,
                    case.scenarios.get(arm_run.name, "ok"), str(case.calls_file)]
            rc = real_supervise(manager, fake, env, log_path, run, arm_run, watch)
            if phase == "dry" and case.scenarios.get(arm_run.name) == "cancel-after-dry":
                manager._cancel.set()  # a cancel landing just as the dry run exits
            return rc

        real_retry = procs.cleanup_with_retry
        snapshot = self.work / "snapshot"
        snapshot.mkdir()
        for target, name, value in (
                (runs, "POLL_SECONDS", POLL),
                (runs.RunManager, "_supervise", supervise),
                (procs, "cleanup_containers", self._fake_cleanup),
                (runs, "cleanup_containers", self._fake_cleanup),
                (procs, "cleanup_with_retry", lambda job_dir, tries=3, delay=2.0: real_retry(job_dir, tries, 0.0)),
                (procs, "kill_tree", self._fake_kill),
                (procs, "assign_kill_on_close", lambda proc: True),
                (procs, "spawn_detached_cleanup", self._fake_spawn),
                (datasets, "ensure", lambda dataset, log=None, cancel=None: snapshot),
                (datasets, "info", lambda dataset: {"dataset": dataset,
                                                    "downloaded_at": "2026-10-29T05:29:58+00:00"}),
                (datasets, "path", lambda dataset: None),
                (datasets, "task_meta", lambda dataset: {}),
                (checks, "docker_capacity", lambda: None),
                (runs, "_long_paths_enabled", lambda: True)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    # Fakes ------------------------------------------------------------------

    def _fake_cleanup(self, job_dir: Path) -> int:
        self.cleanups.append(Path(job_dir))
        if self.cleanup_mode == "fail":
            raise subprocess.SubprocessError("docker container ls failed (exit 1): error during connect")
        if self.cleanup_mode == "block":
            self.release.wait(10)
        return 0

    def _fake_kill(self, pid: int, timeout: float = 30) -> None:
        self.killed.append(pid)
        try:
            os.kill(pid, signal.SIGTERM)  # TerminateProcess on Windows; only our own sleepers
        except OSError:
            pass

    def _fake_spawn(self, job_dirs: list[Path]) -> bool:
        self.spawned.append([Path(d) for d in job_dirs])
        return True

    # Helpers ------------------------------------------------------------------

    def config(self, budget: float = 10, parallel: int = 1) -> settings.Config:
        return self.load_test_config(YAML.format(budget=budget, parallel=parallel), tasks=["task-a", "task-b"])

    def manager(self, ledger: Any = None) -> runs.RunManager:
        manager = runs.RunManager(ledger=ledger)
        self.managers.append(manager)
        self.addCleanup(self._close, manager)
        return manager

    def _close(self, manager: runs.RunManager) -> None:
        manager._cancel.set()
        self.release.set()
        if manager._thread is not None:
            manager._thread.join(20)
        procs.release_instance_lock(manager._lock_fd)

    def finish(self, manager: runs.RunManager, timeout: float = 20) -> dict[str, Any]:
        thread = manager._thread
        assert thread is not None
        thread.join(timeout)
        self.assertFalse(thread.is_alive(), "the run did not finish")
        return manager.get(manager.list()[0]["id"])

    def wait_for(self, predicate: Callable[[], bool], timeout: float = 15) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail("timed out waiting")

    def calls(self) -> list[str]:
        if not self.calls_file.exists():
            return []
        return self.calls_file.read_text(encoding="utf-8").split()

    def on_disk(self, run_id: str) -> dict[str, Any]:
        # The manager replaces run.json from its own thread, and on Windows a
        # read that meets the replace gets PermissionError: wait it out.
        path = settings.RUNS_DIR / run_id / "run.json"
        deadline = time.monotonic() + 2
        while True:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)

    def arm_log(self, run_id: str, arm: str) -> str:
        return (settings.RUNS_DIR / run_id / f"{arm}.log").read_text(encoding="utf-8")


# Single instance ---------------------------------------------------------------


class InstanceLockTests(RunsTestCase):
    def test_second_bench_in_another_process_is_refused_before_any_cleanup(self) -> None:
        self.manager()
        # A run the second process would take for dead and sweep.
        run_id = "20261029-010101-pilot"
        live = {"id": run_id, "mode": "pilot", "created_at": "2026-10-29T08:01:01+00:00", "budget_usd": 5.0,
                "tasks": ["task-a"], "status": "running",
                "arms": [{"name": "a", "status": "running", "cleanup_pending": True}]}
        (settings.RUNS_DIR / run_id).mkdir(parents=True)
        (settings.RUNS_DIR / run_id / "run.json").write_text(json.dumps(live), encoding="utf-8")
        (settings.JOBS_DIR / run_id / "a" / "task-a__abc").mkdir(parents=True)
        env = dict(os.environ)
        env.update({"BENCH_DATA_DIR": str(self.root), "BENCH_LEDGER": str(self.work / "ledger.sqlite"),
                    "PYTHONPATH": str(settings.ROOT)})
        child = subprocess.run([sys.executable, "-c", CHILD_SECOND_BENCH], cwd=settings.ROOT, env=env,
                               capture_output=True, text=True, encoding="utf-8", timeout=120)
        self.assertIn("refused: The bench is already running for this profile.", child.stdout, child.stderr)
        self.assertIn("cleanups: 0", child.stdout)
        self.assertEqual(json.loads((settings.RUNS_DIR / run_id / "run.json").read_text(encoding="utf-8")), live)

    def test_lock_is_taken_before_history_is_loaded(self) -> None:
        order: list[str] = []
        real_lock = procs.acquire_instance_lock
        with mock.patch.object(procs, "acquire_instance_lock",
                               lambda d: order.append("lock") or real_lock(d)), \
                mock.patch.object(runs.RunManager, "_load_history", lambda self: order.append("history")):
            self.manager()
        self.assertEqual(order, ["lock", "history"])

    def test_cleanup_containers_is_reexported(self) -> None:
        self.assertTrue(hasattr(runs, "cleanup_containers"))


# Lifecycle ---------------------------------------------------------------------


class StartTests(RunsTestCase):
    def test_config_start_records_source_spec_and_resolved_arms(self) -> None:
        cfg = self.config()
        manager = self.manager()
        started = manager.start(cfg, "pilot", ["a", "m"])
        run_id = started["id"]
        self.assertRegex(run_id, r"^\d{8}-\d{6}-pilot$")
        expected = spec.spec_from_config(cfg, "pilot", ["a", "m"])
        final = self.finish(manager)
        for record in (started, final, self.on_disk(run_id)):
            self.assertEqual(record["source"], "config")
            self.assertEqual(record["mode"], "pilot")
            self.assertEqual(record["profile"], "default")
            self.assertEqual(record["spec"], expected.spec)
            self.assertEqual(record["spec_hash"], expected.spec_hash)
            self.assertIsNone(record["design_ref"])
            self.assertEqual(record["harbor_version"], importlib.metadata.version("harbor"))
            self.assertEqual(record["plan"]["mode"], "pilot")
            self.assertEqual(record["plan"]["spec_hash"], expected.spec_hash)
            resolved = {a["name"]: a for a in record["resolved_arms"]}
            self.assertEqual(list(resolved), ["a", "m"])
            a, m = resolved["a"], resolved["m"]
            self.assertEqual((a["family"], a["kind"], a["harbor_model"]),
                             ("terminus-2", "name", "anthropic/claude-opus-5-5"))
            self.assertEqual((m["family"], m["kind"], m["agent"]),
                             ("mini-swe-agent", "import_path", "bench.agents.mini_swe_agent:MiniSweAgentWithMcp"))
            self.assertEqual(a["effort"], {"setting": "inherit", "value": "medium", "label": "medium"})
            self.assertEqual(a["kwargs"], [{"name": "reasoning_effort", "value": "medium", "source": "effort"},
                                           {"name": "max_turns", "value": 50, "source": "run cap"}])
            self.assertRegex(a["options_fingerprint"], r"^[0-9a-f]{12}$")
            self.assertEqual((a["tasks"], a["trials"]), (["task-a"], 1))
            self.assertIn(" --agent-import-path bench.agents.mini_swe_agent:MiniSweAgentWithMcp ", m["command"])
            self.assertTrue(m["command"].startswith("harbor --% run "))
            self.assertIn(run_id, m["command"])
            self.assertEqual([x["family"] for x in record["arms"]], ["terminus-2", "mini-swe-agent"])
        # The commands harbor would have been given.
        real_m = [c for c in self.commands if "--dry-run" not in c and "m" == c[c.index("--job-name") + 1]][0]
        self.assertEqual(real_m[real_m.index("--agent-import-path") + 1],
                         "bench.agents.mini_swe_agent:MiniSweAgentWithMcp")
        self.assertEqual(final["status"], "done")
        self.assertEqual([x["status"] for x in final["arms"]], ["done", "done"])
        self.assertEqual(final["spend_usd"], 0.5)
        self.assertEqual(final["reserve_usd"], 0.0)
        self.assertEqual([x["cleanup_pending"] for x in final["arms"]], [False, False])
        self.assertEqual(self.cleanups, [], "harbor exiting 0 needs no sweep")
        self.assertTrue(all(isinstance(x["pid"], int) for x in final["arms"]))

    def test_start_spec_runs_a_design(self) -> None:
        cfg = self.config()
        manager = self.manager()
        normalized = spec.normalized_from_config(cfg, "pilot", ["b"])
        ref = {"design_id": "d-cheap-glm-check-and-more", "version": 3, "author": "user",
               "conversation_id": None, "queue_item": "q-0007"}
        started = manager.start_spec(cfg, spec.spec_from_design(normalized, ref))
        self.assertRegex(started["id"], r"^\d{8}-\d{6}-cheap-glm-check-v3$")
        self.assertLessEqual(len(started["id"]), 44)
        final = self.finish(manager)
        self.assertEqual((final["source"], final["mode"], final["design_ref"]), ("design", "design", ref))
        self.assertEqual(final["status"], "done")

    def test_start_refuses_plan_problems_and_a_second_run(self) -> None:
        cfg = self.config()
        manager = self.manager()
        normalized = spec.normalized_from_config(cfg, "pilot", ["b"])
        normalized["arms"][0]["model"] = "claude-unpriced-9"
        with self.assertRaisesRegex(settings.ConfigError, "no price for 'claude-unpriced-9'"):
            manager.start_spec(cfg, spec.spec_from_design(normalized, None))
        self.assertTrue(manager.is_idle())
        self.scenarios["a"] = "hang"
        run_id = manager.start(cfg, "pilot", ["a"])["id"]
        self.assertFalse(manager.is_idle())
        with self.assertRaisesRegex(RuntimeError, "A run is already in progress"):
            manager.start(cfg, "pilot", ["b"])
        with self.assertRaisesRegex(RuntimeError, "A run is already in progress"):
            manager.start_spec(cfg, spec.spec_from_config(cfg, "pilot", ["b"]))
        manager.cancel(run_id)
        self.finish(manager)
        self.assertTrue(manager.is_idle())

    def test_run_ids_never_collide(self) -> None:
        class Fixed(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
                return datetime(2026, 10, 29, 1, 2, 3, tzinfo=tz)

        cfg = self.config()
        manager = self.manager()
        run_spec = spec.spec_from_config(cfg, "pilot", ["a"])
        with mock.patch.object(runs, "datetime", Fixed):
            first = manager._new_run_id(run_spec)
            (settings.RUNS_DIR / first).mkdir(parents=True)
            second = manager._new_run_id(run_spec)
            (settings.JOBS_DIR / second).mkdir(parents=True)
            third = manager._new_run_id(run_spec)
        self.assertEqual((first, second, third),
                         ("20261029-010203-pilot", "20261029-010203-pilot-2", "20261029-010203-pilot-3"))

    def test_listeners_are_called_once_after_the_final_save(self) -> None:
        cfg = self.config()
        manager = self.manager()
        seen: list[tuple[dict[str, Any], dict[str, Any], bool]] = []

        def broken(run: dict[str, Any]) -> None:
            raise RuntimeError("listener bug")

        def good(run: dict[str, Any]) -> None:
            seen.append((run, self.on_disk(run["id"]), manager.is_idle()))

        manager.add_listener(broken)
        manager.add_listener(good)
        with self.assertLogs("bench.runs", level="ERROR") as logs:
            manager.start(cfg, "pilot", ["a"])
            final = self.finish(manager)
        self.assertEqual(len(seen), 1)
        run, disk, idle = seen[0]
        self.assertEqual((run["status"], disk["status"]), ("done", "done"))
        self.assertIsNotNone(disk["finished_at"])
        self.assertTrue(idle)
        self.assertEqual(final["status"], "done")
        self.assertIn("A run listener failed", logs.output[0])


class FinalStatusTests(RunsTestCase):
    def test_every_arm_failing_fails_the_run(self) -> None:
        cfg = self.config()
        manager = self.manager()
        self.scenarios.update(a="fail-dry", b="fail-dry")
        manager.start(cfg, "pilot", ["a", "b"])
        final = self.finish(manager)
        self.assertEqual((final["status"], final["message"]), ("failed", "Every arm failed. See the arm logs."))
        self.assertEqual({a["status"] for a in final["arms"]}, {"failed"})
        self.assertEqual(self.calls(), ["a", "dry", "b", "dry"])

    def test_early_exception_fails_the_arms_instead_of_leaving_them_queued(self) -> None:
        cfg = self.config()
        manager = self.manager()

        def broken(dataset: str, log: Any = None, cancel: Any = None) -> Path:
            raise RuntimeError("Downloading terminal-bench@2.0 timed out")

        with mock.patch.object(datasets, "ensure", broken):
            run_id = manager.start(cfg, "pilot", ["a", "b"])["id"]
            final = self.finish(manager)
        self.assertEqual((final["status"], final["message"]),
                         ("failed", "RuntimeError: Downloading terminal-bench@2.0 timed out"))
        for record in (final, self.on_disk(run_id)):
            for arm in record["arms"]:
                self.assertEqual(arm["status"], "failed")
                self.assertEqual(arm["message"], "Run failed: RuntimeError: Downloading terminal-bench@2.0 timed out")
                self.assertIsNotNone(arm["finished_at"])
        self.assertEqual(self.calls(), [])

    def test_cancel_during_the_dataset_download(self) -> None:
        cfg = self.config()
        manager = self.manager()
        entered = threading.Event()
        given: list[Any] = []

        def slow(dataset: str, log: Any = None, cancel: threading.Event | None = None) -> Path:
            given.append(cancel)
            entered.set()
            if cancel is not None and cancel.wait(10):
                raise datasets.DownloadCancelled(dataset)
            raise AssertionError("cancel never reached the download")

        with mock.patch.object(datasets, "ensure", slow):
            run_id = manager.start(cfg, "pilot", ["a", "b"])["id"]
            self.assertTrue(entered.wait(5))
            started = time.monotonic()
            manager.cancel(run_id)
            final = self.finish(manager, timeout=5)
        self.assertLess(time.monotonic() - started, 5)
        self.assertIs(given[0], manager._cancel)
        self.assertEqual((final["status"], final["message"]),
                         ("cancelled", "Cancelled during the dataset download"))
        self.assertEqual([a["status"] for a in final["arms"]], ["cancelled", "cancelled"])
        self.assertTrue(manager.is_idle())
        self.assertEqual(self.calls(), [])


# Budget --------------------------------------------------------------------------


class BudgetTests(RunsTestCase):
    def test_budget_used_during_preflight_skips_the_real_job(self) -> None:
        cfg = self.config(budget=10, parallel=2)
        manager = self.manager()
        self.scenarios.update(a="spend", b="slow-dry")
        manager.start(cfg, "pilot", ["a", "b"])
        final = self.finish(manager)
        a, b = final["arms"]
        self.assertEqual(a["status"], "budget_stopped")
        self.assertEqual((b["status"], b["message"]), ("skipped", "Run budget was used up before this arm started"))
        self.assertEqual(b["returncode"], 0)
        self.assertEqual(sorted(self.calls()[i:i + 2] for i in range(0, len(self.calls()), 2)),
                         [["a", "dry"], ["a", "real"], ["b", "dry"]])
        self.assertEqual(final["status"], "budget_stopped")

    def test_cancel_as_preflight_exits_skips_the_real_job(self) -> None:
        cfg = self.config()
        manager = self.manager()
        self.scenarios["a"] = "cancel-after-dry"
        manager.start(cfg, "pilot", ["a", "b"])
        final = self.finish(manager)
        self.assertEqual([(a["status"], a["message"]) for a in final["arms"]],
                         [("cancelled", "Cancelled from the dashboard"), ("cancelled", "")])
        self.assertEqual(self.calls(), ["a", "dry"])
        self.assertEqual(final["status"], "cancelled")

    def test_reserve_for_running_trials_stops_before_finished_spend_does(self) -> None:
        cfg = self.config(budget=6)
        manager = self.manager()
        self.scenarios["a"] = "reserve"  # one trial finished at $4, one still running
        run_id = manager.start(cfg, "pilot", ["a"])["id"]
        final = self.finish(manager)
        arm = final["arms"][0]
        self.assertEqual(arm["status"], "budget_stopped")
        self.assertLess(final["spend_usd"], final["budget_usd"], "finished spend alone would not have stopped it")
        self.assertEqual(final["spend_usd"], 4.0)
        self.assertEqual(arm["trials_killed"], 1)
        self.assertEqual(arm["message"],
                         "Run spend $4.00 plus $4.00 held back for running trials reached the $6.00 budget. "
                         "1 trial(s) were killed mid-run; their spend is not in the total, "
                         "so recorded spend is a lower bound.")
        self.assertEqual((arm["trials_running"], arm["reserve_usd"], final["reserve_usd"]), (0, 0.0, 0.0))
        self.assertEqual(final["status"], "budget_stopped")
        self.assertEqual(self.cleanups, [settings.JOBS_DIR / run_id / "a"])
        self.assertFalse(arm["cleanup_pending"])

    def test_trial_estimate_prefers_the_arm_then_the_run_and_respects_the_cap(self) -> None:
        manager = self.manager()
        run = runs.Run(id="r", mode="pilot", created_at="", budget_usd=10, tasks=[], arms=[])
        self.assertEqual(manager._trial_estimate(run, "a"), 0.0)
        manager._largest_trial[("r", "b")] = 3.0
        manager._largest_trial[("other", "a")] = 99.0
        self.assertEqual(manager._trial_estimate(run, "a"), 3.0)
        manager._largest_trial[("r", "a")] = 2.0
        self.assertEqual(manager._trial_estimate(run, "a"), 2.0)
        manager._trial_caps[("r", "a")] = 1.5
        self.assertEqual(manager._trial_estimate(run, "a"), 1.5)

    def test_a_trial_with_no_usage_recorded_is_named_as_missing_spend(self) -> None:
        cfg = self.config()
        manager = self.manager()
        self.scenarios["a"] = "unrecorded"  # one trial at $0.25, one timed out with nothing recorded
        manager.start(cfg, "pilot", ["a"])
        final = self.finish(manager)
        arm = final["arms"][0]
        self.assertEqual((arm["status"], arm["trials_done"], arm["trials_unmeasured"]), ("done_with_errors", 2, 1))
        self.assertEqual(final["spend_usd"], 0.25)
        self.assertEqual(arm["message"],
                         "1 of 2 trials errored before finishing. 1 finished trial(s) ran but recorded no usage; "
                         "their spend is not in the total, so recorded spend is a lower bound.")

    def test_a_trial_with_no_usage_recorded_never_sets_the_reserve(self) -> None:
        # Its $0 is not what it spent, so the arm falls back to the run-wide estimate.
        manager = self.manager()
        run = runs.Run(id="r", mode="pilot", created_at="", budget_usd=10, tasks=[], arms=[])
        manager._largest_trial[("r", "other")] = 3.0
        arm_run = runs.ArmRun(name="a")
        with mock.patch.object(runs.results, "load_job", lambda job_dir, cfg, arm: [{"usage_status": "unmeasured"}]), \
                mock.patch.object(runs.results, "spend", lambda row: 0.0):
            manager._refresh(None, None, self.work, arm_run, run, live=False)
        self.assertEqual((arm_run.trials_done, arm_run.trials_unmeasured, arm_run.spend_usd), (1, 1, 0.0))
        self.assertNotIn(("r", "a"), manager._largest_trial)
        self.assertEqual(manager._trial_estimate(run, "a"), 3.0)

    def test_trial_estimate_survives_another_arm_recording_its_first_trial(self) -> None:
        # An arm's first finished trial adds a key while another arm scans the
        # run-wide estimate; an unguarded scan raised "dictionary changed size
        # during iteration" and got the scanning (healthy) arm killed.
        manager = self.manager()
        run = runs.Run(id="r", mode="pilot", created_at="", budget_usd=10, tasks=[], arms=[])
        for i in range(3000):
            manager._largest_trial[(f"other-{i}", "a")] = 1.0
        errors: list[BaseException] = []
        inserts = [0]
        stop = threading.Event()

        def scan() -> None:
            try:
                while not stop.is_set():
                    manager._trial_estimate(run, "not-started")
            except BaseException as exc:  # noqa: BLE001 - the failure under test
                errors.append(exc)

        def record() -> None:
            i = 0
            while not stop.is_set() and not errors:
                manager._refresh(None, None, self.work, runs.ArmRun(name=f"w{i}"), run, live=False)
                i += 1
            inserts[0] = i

        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        self.addCleanup(sys.setswitchinterval, old_interval)
        with mock.patch.object(runs.results, "load_job", lambda job_dir, cfg, arm: [{"x": 1}]), \
                mock.patch.object(runs.results, "spend", lambda row: 0.5):
            threads = [threading.Thread(target=scan), threading.Thread(target=record)]
            for t in threads:
                t.start()
            time.sleep(0.6)
            stop.set()
            for t in threads:
                t.join(10)
        sys.setswitchinterval(old_interval)
        self.assertEqual(errors, [])
        self.assertGreater(inserts[0], 0)
        self.assertEqual(manager._trial_estimate(run, "not-started"), 0.5)


# Containers -----------------------------------------------------------------------


class CleanupTests(RunsTestCase):
    def test_failed_cleanup_keeps_cleanup_pending_and_the_next_start_sweeps(self) -> None:
        cfg = self.config()
        manager = self.manager()
        self.scenarios["a"] = "hang"
        run_id = manager.start(cfg, "pilot", ["a"])["id"]
        job_dir = settings.JOBS_DIR / run_id / "a"
        # Recorded before harbor starts, so a crash from here on leaves the mark.
        self.wait_for(lambda: (job_dir / "t1" / "config.json").exists()
                      and self.on_disk(run_id)["arms"][0]["cleanup_pending"])
        self.cleanup_mode = "fail"
        manager.cancel(run_id)
        final = self.finish(manager)
        arm = final["arms"][0]
        self.assertEqual(arm["status"], "cancelled", "a cleanup failure must not turn a cancel into a failure")
        self.assertTrue(arm["cleanup_pending"])
        self.assertTrue(self.on_disk(run_id)["arms"][0]["cleanup_pending"])
        self.assertEqual(arm["trials_killed"], 1)
        self.assertIn("1 trial(s) were killed mid-run", arm["message"])
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual(self.cleanups, [job_dir] * 3)
        self.assertIn("container cleanup FAILED: docker container ls failed", self.arm_log(run_id, "a"))
        self.assertIn(arm["pid"], self.killed)

        self.cleanup_mode = "ok"
        self.cleanups.clear()
        later = self.manager()
        self.assertEqual(self.cleanups, [job_dir])
        self.assertFalse(later.get(run_id)["arms"][0]["cleanup_pending"])
        self.assertFalse(self.on_disk(run_id)["arms"][0]["cleanup_pending"])
        self.assertEqual(later.get(run_id)["status"], "cancelled")
        self.assertIn("startup sweep: removed 0 container(s)", self.arm_log(run_id, "a"))

    def test_harbor_exiting_with_an_error_is_swept(self) -> None:
        cfg = self.config()
        manager = self.manager()
        self.scenarios["a"] = "exit3"
        run_id = manager.start(cfg, "pilot", ["a"])["id"]
        final = self.finish(manager)
        arm = final["arms"][0]
        self.assertEqual((arm["status"], arm["message"]), ("failed", "harbor exited with code 3. See the log."))
        self.assertEqual(self.cleanups, [settings.JOBS_DIR / run_id / "a"])
        self.assertFalse(arm["cleanup_pending"])
        self.assertIn("harbor exited with code 3; removed 0 container(s)", self.arm_log(run_id, "a"))

    def test_failed_sweep_after_harbor_error_stays_pending(self) -> None:
        cfg = self.config()
        manager = self.manager()
        self.scenarios["a"] = "exit3"
        self.cleanup_mode = "fail"
        manager.start(cfg, "pilot", ["a"])
        arm = self.finish(manager)["arms"][0]
        self.assertEqual(arm["status"], "failed")
        self.assertTrue(arm["cleanup_pending"])

    def test_emergency_stop_is_quick_and_hands_cleanup_to_a_helper(self) -> None:
        cfg = self.config()
        manager = self.manager()
        self.scenarios["a"] = "hang"
        run_id = manager.start(cfg, "pilot", ["a"])["id"]
        job_dir = settings.JOBS_DIR / run_id / "a"
        self.wait_for(lambda: (job_dir / "t1" / "config.json").exists() and manager._procs)
        pid = manager.get(run_id)["arms"][0]["pid"]
        self.cleanup_mode = "block"  # hold the in-process cleanup so the record can be read
        started = time.monotonic()
        manager.emergency_stop()
        self.assertLess(time.monotonic() - started, 4.0)
        self.assertEqual(self.spawned, [[job_dir]])
        self.assertIn(pid, self.killed)
        self.assertTrue(manager._cancel.is_set())
        self.assertTrue(self.on_disk(run_id)["arms"][0]["cleanup_pending"])
        self.release.set()
        final = self.finish(manager)
        self.assertEqual(final["status"], "cancelled")

    def test_emergency_stop_when_idle_does_nothing_and_never_raises(self) -> None:
        manager = self.manager()
        with mock.patch.object(procs, "kill_tree", side_effect=OSError("boom")):
            manager.emergency_stop()
        self.assertEqual(self.spawned, [])

    def test_kill_on_close_failure_is_logged_on_windows(self) -> None:
        cfg = self.config()
        manager = self.manager()
        with mock.patch.object(procs, "assign_kill_on_close", lambda proc: False), \
                mock.patch.object(runs, "_WINDOWS", True):
            run_id = manager.start(cfg, "pilot", ["a"])["id"]
            self.finish(manager)
        self.assertIn("could not tie harbor", self.arm_log(run_id, "a"))


# Shutdown --------------------------------------------------------------------------


class ShutdownTests(RunsTestCase):
    def test_shutdown_when_idle_prints_nothing(self) -> None:
        manager = self.manager()
        idle_out = io.StringIO()
        with contextlib.redirect_stdout(idle_out):
            manager.shutdown()
            manager.shutdown()
        self.assertEqual(idle_out.getvalue(), "")
        self.assertTrue(manager.is_idle())

    def test_no_run_starts_after_shutdown(self) -> None:
        # A start racing process exit would run arms nothing can stop: the
        # executor's workers keep the interpreter alive until harbor finishes.
        cfg = self.config()
        manager = self.manager()
        manager.shutdown()
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            manager.start(cfg, "pilot", ["a"])
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            manager.start_spec(cfg, spec.spec_from_config(cfg, "pilot", ["a"]))
        self.assertIsNone(manager._thread)
        self.assertEqual(manager.list(), [])
        self.assertEqual(self.commands, [])

    def test_no_run_starts_after_emergency_stop(self) -> None:
        cfg = self.config()
        manager = self.manager()
        manager.emergency_stop()
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            manager.start(cfg, "pilot", ["a"])
        self.assertIsNone(manager._thread)
        self.assertTrue(manager._cancel.is_set(), "a refused start must not clear emergency_stop's cancel")

    def test_emergency_stop_during_a_start_wins(self) -> None:
        # The console handler takes no lock, so it can land while a start is planning.
        cfg = self.config()
        manager = self.manager()
        real_plan = runs.plan_spec
        planning = threading.Event()

        def slow_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
            planning.set()
            time.sleep(0.5)
            return real_plan(*args, **kwargs)

        stopper = threading.Thread(target=lambda: (planning.wait(5), manager.emergency_stop()))
        stopper.start()
        with mock.patch.object(runs, "plan_spec", slow_plan):
            with self.assertRaisesRegex(RuntimeError, "shutting down"):
                manager.start(cfg, "pilot", ["a"])
        stopper.join(10)
        self.assertTrue(manager._cancel.is_set())
        self.assertIsNone(manager._thread)
        self.assertEqual(manager.list(), [])

    def test_shutdown_announces_once_and_stops_the_run(self) -> None:
        cfg = self.config()
        manager = self.manager()
        self.scenarios["a"] = "hang"
        run_id = manager.start(cfg, "pilot", ["a"])["id"]
        self.wait_for(lambda: manager.get(run_id)["arms"][0]["status"] == "running")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            manager.shutdown(timeout=20)
            manager.shutdown(timeout=1)
        self.assertEqual(out.getvalue(), f"Stopping run {run_id}: killing harbor and removing containers...\n")
        self.assertEqual(manager.get(run_id)["status"], "cancelled")
        self.assertTrue(manager.is_idle())


# History ---------------------------------------------------------------------------


class HistoryTests(RunsTestCase):
    def test_legacy_run_json_with_unknown_keys_loads(self) -> None:
        legacy = sorted(DEV_RUNS.glob("*/run.json"))
        if not legacy:
            self.skipTest("tests/fixtures/ledger/dev/runs is missing")
        originals = {}
        for path in legacy:
            data = json.loads(path.read_text(encoding="utf-8"))
            originals[data["id"]] = data
            data = dict(data, from_a_future_version={"x": 1})
            data["arms"] = [dict(a, also_unknown=True) for a in data["arms"]]
            target = settings.RUNS_DIR / data["id"] / "run.json"
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(data), encoding="utf-8")
        nan_run = settings.RUNS_DIR / "20261029-000000-pilot" / "run.json"
        nan_run.parent.mkdir(parents=True)
        nan_run.write_text('{"id": "20261029-000000-pilot", "mode": "pilot", "created_at": "x", '
                           '"budget_usd": Infinity, "tasks": [], "arms": []}', encoding="utf-8")
        before = {p: p.read_bytes() for p in settings.RUNS_DIR.glob("*/run.json")}

        manager = self.manager()
        ids = [r["id"] for r in manager.list()]
        self.assertEqual(sorted(ids), sorted(originals))
        for run_id, original in originals.items():
            loaded = manager.get(run_id)
            self.assertNotIn("from_a_future_version", loaded)
            self.assertNotIn("also_unknown", loaded["arms"][0])
            self.assertEqual(loaded["status"], original["status"])
            self.assertEqual(loaded["spend_usd"], round(original["spend_usd"], 4))
            self.assertEqual([a["status"] for a in loaded["arms"]], [a["status"] for a in original["arms"]])
            self.assertEqual(loaded["source"], "config")
            self.assertEqual(loaded["profile"], "default")
        # Finished legacy runs with nothing pending are neither swept nor rewritten.
        self.assertEqual(self.cleanups, [])
        self.assertEqual({p: p.read_bytes() for p in settings.RUNS_DIR.glob("*/run.json")}, before)

    def test_running_run_is_interrupted_and_swept(self) -> None:
        run_id = "20261029-020202-sample"
        data = {"id": run_id, "mode": "sample", "created_at": "2026-10-29T09:02:02+00:00", "budget_usd": 5.0,
                "tasks": ["task-a"], "status": "running",
                "arms": [{"name": "a", "status": "running"}, {"name": "b", "status": "queued"},
                         {"name": "c", "status": "done"}]}
        path = settings.RUNS_DIR / run_id / "run.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        (settings.JOBS_DIR / run_id / "a" / "task-a__x").mkdir(parents=True)
        self.cleanup_mode = "fail"
        manager = self.manager()
        loaded = manager.get(run_id)
        self.assertEqual(loaded["status"], "interrupted")
        self.assertEqual([a["status"] for a in loaded["arms"]], ["interrupted", "interrupted", "done"])
        self.assertEqual(self.cleanups, [settings.JOBS_DIR / run_id / "a", settings.JOBS_DIR / run_id / "b"])
        self.assertEqual([a["cleanup_pending"] for a in self.on_disk(run_id)["arms"]], [True, True, False])
        self.assertIn("startup sweep: container cleanup FAILED", self.arm_log(run_id, "a"))

    def test_finished_run_with_live_arms_is_swept_once(self) -> None:
        # What older versions left behind when a run failed before its arms ran.
        run_id = "20261029-030303-pilot"
        data = {"id": run_id, "mode": "pilot", "created_at": "2026-10-29T10:03:03+00:00", "budget_usd": 5.0,
                "tasks": ["task-a"], "status": "failed", "message": "RuntimeError: boom",
                "arms": [{"name": "a", "status": "running"}, {"name": "b", "status": "queued"}]}
        path = settings.RUNS_DIR / run_id / "run.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        (settings.JOBS_DIR / run_id / "a" / "task-a__x").mkdir(parents=True)
        first = self.manager()
        self.assertEqual(self.cleanups, [settings.JOBS_DIR / run_id / "a", settings.JOBS_DIR / run_id / "b"])
        loaded = first.get(run_id)
        self.assertEqual((loaded["status"], [a["status"] for a in loaded["arms"]]),
                         ("failed", ["interrupted", "interrupted"]))
        self.assertEqual([a["cleanup_pending"] for a in self.on_disk(run_id)["arms"]], [False, False])
        self.cleanups.clear()
        self.manager()
        self.assertEqual(self.cleanups, [])


# Ledger ------------------------------------------------------------------------------


class FailingLedger:
    available = True

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.errors: list[tuple[str, str]] = []

    def upsert_bench_run(self, run: dict[str, Any]) -> None:
        self.calls.append(("upsert", run["status"]))
        raise RuntimeError("database is locked")

    def ingest_bench_job(self, job_dir: Path, run: dict[str, Any], arm: str, *, final: bool, via: str) -> dict:
        self.calls.append((via, final))
        raise OSError("disk I/O error")

    def set_error(self, where: str, exc: BaseException) -> None:
        self.errors.append((where, str(exc)))


class RecordingLedger(FailingLedger):
    def upsert_bench_run(self, run: dict[str, Any]) -> None:
        self.calls.append(("upsert", run["status"]))

    def ingest_bench_job(self, job_dir: Path, run: dict[str, Any], arm: str, *, final: bool, via: str) -> dict:
        self.calls.append((via, final, arm, Path(job_dir).name, bool(run.get("resolved_arms"))))
        return {}


class LedgerHookTests(RunsTestCase):
    def test_a_failing_ledger_writes_one_line_and_changes_nothing(self) -> None:
        cfg = self.config()
        ledger = FailingLedger()
        manager = self.manager(ledger=ledger)
        self.scenarios["a"] = "slow-ok"
        run_id = manager.start(cfg, "pilot", ["a"])["id"]
        final = self.finish(manager)
        self.assertEqual((final["status"], final["arms"][0]["status"]), ("done", "done"))
        self.assertEqual(final["spend_usd"], 0.25)
        log_text = self.arm_log(run_id, "a")
        self.assertEqual(log_text.count("[ledger]"), 1, log_text)
        self.assertIn("[ledger] OSError: disk I/O error; the next rescan will catch up", log_text)
        lives = [c for c in ledger.calls if c == ("live", False)]
        self.assertGreater(len(lives), 1)
        self.assertIn(("finalize", True), ledger.calls)
        self.assertEqual([c for c in ledger.calls if c[0] == "upsert"], [("upsert", "running"), ("upsert", "done")])
        self.assertGreaterEqual(len(ledger.errors), len(lives) + 3)

    def test_ledger_hooks_get_the_run_and_its_arms(self) -> None:
        cfg = self.config()
        ledger = RecordingLedger()
        manager = self.manager(ledger=ledger)
        manager.start(cfg, "pilot", ["a"])
        self.finish(manager)
        self.assertEqual(ledger.calls[0], ("upsert", "running"))
        self.assertEqual(ledger.calls[-1], ("upsert", "done"))
        self.assertIn(("finalize", True, "a", "a", True), ledger.calls)
        self.assertEqual(ledger.errors, [])

    def test_an_unavailable_ledger_is_never_called(self) -> None:
        cfg = self.config()
        ledger = RecordingLedger()
        ledger.available = False
        manager = self.manager(ledger=ledger)
        manager.start(cfg, "pilot", ["a"])
        self.assertEqual(self.finish(manager)["status"], "done")
        self.assertEqual(ledger.calls, [])

