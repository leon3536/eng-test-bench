"""Dataset snapshots: a download that can be cancelled and dies with the
bench, that never deletes a snapshot another bench may be using, that waits
out Windows' brief file holds, and per-task facts for planning.

harbor never runs: a fake lays out the downloaded tasks, or starts a small
python child where a real process is needed to cancel or kill.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import tomllib
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from tests.support import BenchTestCase, completed, program_name

from bench import datasets, procs, settings

DATASET = "terminal-bench@2.0"

TOML_HARD = """\
version = "1.0"

[metadata]
difficulty = "hard"
category = "machine-learning"
expert_time_estimate_min = 120.0

[agent]
timeout_sec = 1800.0

[environment]
cpus = 2
memory = "4G"
storage = "10G"
"""

TOML_SPARSE = """\
[agent]
timeout_sec = 900

[environment]
memory_mb = 512
"""

SAMPLE = """\
# Task sample for terminal-bench@2.0, drawn 2026-10-28 by `python -m bench.sample_tasks`.
# Two blocks of 10, each in proportion to difficulty. The pilot uses line 1.

# Block 1
adaptive-rejection-sampler               # medium, 900s
protein-assembly                         # hard, 1800s
# Block 2
overfull-hbox                            # easy
no-comment-task
"""

HOLD_LOCK_THEN_FINISH = """
import json, sys, time
from pathlib import Path
from bench import procs
lock, root = Path(sys.argv[1]), Path(sys.argv[2])
with procs.file_lock(lock, timeout=30):
    print("locked", flush=True)
    time.sleep(1.0)
    task = root / "terminal-bench" / "task-a"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('[metadata]\\ndifficulty = "easy"\\n', encoding="utf-8")
    (root / "snapshot.json").write_text(json.dumps({
        "dataset": "terminal-bench@2.0", "tasks_dir": "terminal-bench", "tasks": 1,
        "downloaded_at": "2026-10-29T05:00:00+00:00", "harbor": "0.25.0"}), encoding="utf-8")
"""

HOLD_LOCK = """
import sys
from bench import procs
with procs.file_lock(sys.argv[1], timeout=30):
    print("locked", flush=True)
    sys.stdin.read()
"""


def lay_out_tasks(tasks_dir: Path, tasks: dict[str, str | None]) -> None:
    for name, toml in tasks.items():
        (tasks_dir / name).mkdir(parents=True, exist_ok=True)
        if toml is not None:
            (tasks_dir / name / "task.toml").write_text(toml, encoding="utf-8")


def write_snapshot(root: Path, tasks: dict[str, str | None], downloaded_at: str = "2026-10-29T05:29:58+00:00",
                   sentinel: bool = False) -> Path:
    lay_out_tasks(root / "terminal-bench", tasks)
    (root / "snapshot.json").write_text(json.dumps({
        "dataset": DATASET, "tasks_dir": "terminal-bench", "tasks": len(tasks),
        "downloaded_at": downloaded_at, "harbor": "0.25.0"}), encoding="utf-8")
    if sentinel:
        (root / "terminal-bench" / "in-use.txt").write_text("an arm is reading this", encoding="utf-8")
    return root / "terminal-bench"


class ChildProxy:
    """A real child handed back from a fake. Popen fakes must not return a
    Popen instance: the guard's Popen would run __init__ on it a second time."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc

    def __getattr__(self, name: str) -> Any:
        return getattr(self._proc, name)


class DoneProc:
    """What the fake harbor returns when the download finishes at once."""

    pid = 2_000_000_000

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


def python_child(code: str, *args: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", textwrap.dedent(code), *args], cwd=settings.ROOT,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8")


def reap(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(30)
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None:
            stream.close()


class DatasetTestCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = self.tmpdir()
        self.datasets_dir = self.root / "datasets"
        self.target = self.datasets_dir / DATASET
        for name, value in (("ROOT", self.root), ("DATASETS_DIR", self.datasets_dir),
                            ("TASKS_FILE", self.root / "tasks.txt")):
            patcher = mock.patch.object(datasets, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(datasets._meta_cache, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.harbor_calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_harbor(self, behave: Any) -> None:
        def harbor(argv: list[str], **kw: Any) -> Any:
            self.harbor_calls.append((argv, kw))
            return behave(Path(argv[argv.index("-o") + 1]), kw)

        self.allow_subprocess(lambda argv: program_name(argv) == "harbor", harbor)

    def fake_download(self, tasks: dict[str, str | None] | None = None) -> None:
        def download(staging: Path, kw: dict[str, Any]) -> DoneProc:
            lay_out_tasks(staging / "terminal-bench", tasks or {"task-a": TOML_HARD, "task-b": TOML_SPARSE})
            return DoneProc()

        self.fake_harbor(download)

    def real_child(self, code: str, kw: dict[str, Any], stdout: Any = subprocess.DEVNULL) -> Any:
        child = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.DEVNULL, stdout=stdout,
                                 stderr=subprocess.STDOUT, creationflags=kw.get("creationflags", 0))
        self.addCleanup(lambda: (child.poll() is None and child.kill(), child.wait(30)))
        return ChildProxy(child)

    def allow_taskkill(self, children: list[Any]) -> list[tuple[list[str], dict[str, Any]]]:
        calls: list[tuple[list[str], dict[str, Any]]] = []

        def taskkill(argv: list[str], **kw: Any) -> subprocess.CompletedProcess:
            calls.append((argv, kw))
            for child in children:
                if str(child.pid) == argv[2]:
                    child.kill()
            return completed(argv)

        self.allow_subprocess(lambda argv: program_name(argv) == "taskkill", taskkill)
        return calls

    def staging_dirs(self) -> list[Path]:
        return list(self.datasets_dir.glob(".download-*")) if self.datasets_dir.exists() else []


class EnsureTests(DatasetTestCase):
    def test_downloads_once_and_records_the_snapshot(self) -> None:
        self.fake_download()
        tasks_dir = datasets.ensure(DATASET)
        self.assertEqual(tasks_dir, self.target / "terminal-bench")
        manifest = json.loads((self.target / "snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["tasks"], 2)
        self.assertEqual(manifest["tasks_dir"], "terminal-bench")
        self.assertEqual(datasets.ensure(DATASET), tasks_dir)
        self.assertEqual(len(self.harbor_calls), 1)
        argv, kw = self.harbor_calls[0]
        self.assertEqual(argv[1:4], ["datasets", "download", DATASET])
        self.assertEqual(kw["creationflags"], procs.NO_CTRL_C)
        self.assertNotIn("ANTHROPIC_API_KEY", kw["env"])
        self.assertEqual(self.staging_dirs(), [])

    def test_cancel_kills_the_download_within_seconds(self) -> None:
        children: list[Any] = []

        def slow(staging: Path, kw: dict[str, Any]) -> Any:
            children.append(self.real_child("import time; time.sleep(120)", kw))
            return children[-1]

        self.fake_harbor(slow)
        taskkills = self.allow_taskkill(children)
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        start = time.monotonic()
        with self.assertRaises(datasets.DownloadCancelled) as caught:
            datasets.ensure(DATASET, cancel=cancel)
        self.assertLess(time.monotonic() - start, 3)
        self.assertEqual(caught.exception.dataset, DATASET)
        self.assertIsNotNone(children[0].poll(), "the download is still running")
        if procs._WINDOWS:
            self.assertEqual(len(taskkills), 1)
            argv, kw = taskkills[0]
            self.assertEqual(argv, ["taskkill", "/PID", str(children[0].pid), "/T", "/F"])
            self.assertEqual(kw["creationflags"], procs.NO_CTRL_C)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.staging_dirs(), [])

    def test_a_download_past_the_deadline_is_killed(self) -> None:
        children: list[Any] = []

        def stalled(staging: Path, kw: dict[str, Any]) -> Any:
            children.append(self.real_child("import time; time.sleep(120)", kw))
            return children[-1]

        self.fake_harbor(stalled)
        self.allow_taskkill(children)
        with mock.patch.object(datasets, "DOWNLOAD_TIMEOUT_SEC", 0.3):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                datasets.ensure(DATASET)
        self.assertIsNotNone(children[0].poll())

    def test_a_failed_download_reports_its_output_without_a_log(self) -> None:
        def failing(staging: Path, kw: dict[str, Any]) -> Any:
            child = self.real_child("print('git clone failed: network unreachable'); raise SystemExit(3)",
                                    kw, stdout=kw["stdout"])
            child.wait(30)
            return child

        self.fake_harbor(failing)
        with self.assertRaises(RuntimeError) as caught:
            datasets.ensure(DATASET)
        self.assertIn("exit 3", str(caught.exception))
        self.assertIn("network unreachable", str(caught.exception))
        self.assertEqual(self.staging_dirs(), [])

    def test_output_goes_to_the_arm_log_after_the_bench_lines(self) -> None:
        def chatty(staging: Path, kw: dict[str, Any]) -> Any:
            lay_out_tasks(staging / "terminal-bench", {"task-a": TOML_HARD})
            child = self.real_child("print('downloading tasks')", kw, stdout=kw["stdout"])
            child.wait(30)
            return child

        self.fake_harbor(chatty)
        log_path = self.tmpdir() / "arm.log"
        with open(log_path, "a", encoding="utf-8") as log:
            log.write("[bench] preparing dataset\n")
            datasets.ensure(DATASET, log)
        text = log_path.read_text(encoding="utf-8")
        self.assertLess(text.index("[bench] preparing dataset"), text.index("downloading tasks"))

    def test_a_snapshot_finished_elsewhere_is_kept_not_replaced(self) -> None:
        def raced(staging: Path, kw: dict[str, Any]) -> DoneProc:
            lay_out_tasks(staging / "terminal-bench", {"task-a": TOML_HARD})
            # A process that took no lock finished first, and its arms use it.
            write_snapshot(self.target, {"task-a": TOML_HARD}, downloaded_at="2026-10-29T01:00:00+00:00",
                           sentinel=True)
            return DoneProc()

        self.fake_harbor(raced)
        removed: list[Path] = []
        real_rmtree = shutil.rmtree

        def watch_rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
            removed.append(Path(path))
            real_rmtree(path, *args, **kwargs)

        with mock.patch.object(datasets.shutil, "rmtree", watch_rmtree):
            tasks_dir = datasets.ensure(DATASET)
        self.assertEqual(tasks_dir, self.target / "terminal-bench")
        self.assertTrue((tasks_dir / "in-use.txt").exists())
        self.assertNotIn(self.target, removed)
        manifest = json.loads((self.target / "snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["downloaded_at"], "2026-10-29T01:00:00+00:00")
        self.assertEqual(self.staging_dirs(), [])

    def test_an_incomplete_earlier_attempt_is_replaced(self) -> None:
        (self.target / "terminal-bench" / "half-copied").mkdir(parents=True)  # no snapshot.json
        self.fake_download({"task-a": TOML_HARD})
        tasks_dir = datasets.ensure(DATASET)
        self.assertFalse((tasks_dir / "half-copied").exists())
        self.assertTrue((tasks_dir / "task-a" / "task.toml").exists())

    def test_stale_staging_folders_are_cleared_under_the_lock(self) -> None:
        stale = self.datasets_dir / ".download-dead1234"
        (stale / "terminal-bench").mkdir(parents=True)
        self.fake_download()
        datasets.ensure(DATASET)
        self.assertFalse(stale.exists())

    def test_rename_is_retried_while_a_file_is_held_open(self) -> None:
        self.fake_download()
        real_replace = pathlib.Path.replace
        attempts: list[Path] = []

        def held_twice(self_path: Path, target: Any) -> Any:
            if self_path.name.startswith(".download-"):
                attempts.append(self_path)
                if len(attempts) <= 2:
                    raise PermissionError(5, "Access is denied", str(self_path))
            return real_replace(self_path, target)

        with mock.patch.object(datasets, "_RETRY_BACKOFF", (0, 0, 0, 0)), \
                mock.patch.object(pathlib.Path, "replace", held_twice):
            tasks_dir = datasets.ensure(DATASET)
        self.assertEqual(len(attempts), 3)
        self.assertTrue((tasks_dir / "task-a" / "task.toml").exists())

    def test_rename_gives_up_after_five_tries_and_cleans_up(self) -> None:
        self.fake_download()
        attempts: list[Path] = []

        def always_held(self_path: Path, target: Any) -> Any:
            attempts.append(self_path)
            raise PermissionError(5, "Access is denied", str(self_path))

        with mock.patch.object(datasets, "_RETRY_BACKOFF", (0, 0, 0, 0)), \
                mock.patch.object(pathlib.Path, "replace", always_held):
            with self.assertRaisesRegex(RuntimeError, "Could not move the download"):
                datasets.ensure(DATASET)
        self.assertEqual(len(attempts), 5)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.staging_dirs(), [])

    def test_waits_for_another_process_and_uses_its_snapshot(self) -> None:
        # harbor is not allowed here: a download attempt would fail the test.
        self.datasets_dir.mkdir(parents=True)
        other = python_child(HOLD_LOCK_THEN_FINISH, str(self.datasets_dir / ".lock"), str(self.target))
        self.addCleanup(reap, other)
        self.assertEqual(other.stdout.readline().strip(), "locked")
        start = time.monotonic()
        tasks_dir = datasets.ensure(DATASET)
        self.assertGreater(time.monotonic() - start, 0.3)
        self.assertEqual(tasks_dir, self.target / "terminal-bench")
        self.assertEqual(self.harbor_calls, [])

    def test_cancel_while_waiting_for_another_process(self) -> None:
        self.datasets_dir.mkdir(parents=True)
        other = python_child(HOLD_LOCK, str(self.datasets_dir / ".lock"))
        self.addCleanup(reap, other)
        self.assertEqual(other.stdout.readline().strip(), "locked")
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        start = time.monotonic()
        with self.assertRaises(datasets.DownloadCancelled):
            datasets.ensure(DATASET, cancel=cancel)
        self.assertLess(time.monotonic() - start, 2)

    def test_an_already_cancelled_run_starts_no_download(self) -> None:
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(datasets.DownloadCancelled):
            datasets.ensure(DATASET, cancel=cancel)
        self.assertEqual(self.harbor_calls, [])

    def test_an_existing_snapshot_needs_no_lock_or_download(self) -> None:
        tasks_dir = write_snapshot(self.target, {"task-a": TOML_HARD})
        self.assertEqual(datasets.ensure(DATASET), tasks_dir)
        self.assertFalse((self.datasets_dir / ".lock").exists())


class TaskMetaTests(DatasetTestCase):
    def test_reads_every_task_toml_in_the_snapshot(self) -> None:
        write_snapshot(self.target, {"task-hard": TOML_HARD, "task-sparse": TOML_SPARSE, "not-a-task": None})
        meta = datasets.task_meta(DATASET)
        self.assertEqual(sorted(meta), ["task-hard", "task-sparse"])
        self.assertEqual(meta["task-hard"], {
            "difficulty": "hard", "category": "machine-learning", "agent_timeout_sec": 1800.0,
            "memory_gb": 4.0, "cpus": 2, "expert_time_estimate_min": 120.0,
        })
        self.assertEqual(meta["task-sparse"], {
            "difficulty": None, "category": None, "agent_timeout_sec": 900.0,
            "memory_gb": 0.5, "cpus": None, "expert_time_estimate_min": None,
        })

    def test_cached_until_the_snapshot_is_replaced(self) -> None:
        write_snapshot(self.target, {"task-hard": TOML_HARD})
        with mock.patch.object(datasets.tomllib, "loads", wraps=tomllib.loads) as loads:
            first = datasets.task_meta(DATASET)
            first["task-hard"]["difficulty"] = "changed by a caller"
            second = datasets.task_meta(DATASET)
            self.assertEqual(loads.call_count, 1)
            self.assertEqual(second["task-hard"]["difficulty"], "hard")

            write_snapshot(self.target, {"task-hard": TOML_HARD.replace('"hard"', '"medium"')},
                           downloaded_at="2026-11-01T00:00:00+00:00")
            third = datasets.task_meta(DATASET)
            self.assertEqual(loads.call_count, 2)
        self.assertEqual(third["task-hard"]["difficulty"], "medium")

    def test_memory_sizes(self) -> None:
        for env, expected in (({"memory": "2G"}, 2.0), ({"memory": "512M"}, 0.5), ({"memory": "8g"}, 8.0),
                              ({"memory_mb": 4096}, 4.0), ({"memory": "lots"}, None), ({"memory": 2}, None),
                              ({}, None)):
            self.assertEqual(datasets._memory_gb(env), expected, env)

    def test_a_broken_task_toml_gives_unknowns(self) -> None:
        write_snapshot(self.target, {"broken": "this is = = not toml", "ok": TOML_HARD})
        meta = datasets.task_meta(DATASET)
        self.assertEqual(set(meta["broken"].values()), {None})
        self.assertEqual(meta["ok"]["difficulty"], "hard")

    def test_falls_back_to_the_task_sample_comments(self) -> None:
        (self.root / "tasks.txt").write_text(SAMPLE, encoding="utf-8")
        meta = datasets.task_meta(DATASET)
        self.assertEqual(list(meta), ["adaptive-rejection-sampler", "protein-assembly", "overfull-hbox",
                                      "no-comment-task"])
        self.assertEqual(meta["adaptive-rejection-sampler"], {
            "difficulty": "medium", "category": None, "agent_timeout_sec": 900.0,
            "memory_gb": None, "cpus": None, "expert_time_estimate_min": None,
        })
        self.assertEqual(meta["protein-assembly"]["agent_timeout_sec"], 1800.0)
        self.assertEqual(meta["overfull-hbox"]["difficulty"], "easy")
        self.assertIsNone(meta["overfull-hbox"]["agent_timeout_sec"])
        self.assertIsNone(meta["no-comment-task"]["difficulty"])

    def test_a_sample_from_another_dataset_says_nothing(self) -> None:
        (self.root / "tasks.txt").write_text(SAMPLE, encoding="utf-8")
        self.assertEqual(datasets.task_meta("swe-bench@1.0"), {})

    def test_never_raises(self) -> None:
        self.assertEqual(datasets.task_meta(DATASET), {})  # no snapshot, no sample
        self.target.mkdir(parents=True)
        (self.target / "snapshot.json").write_text("{not json", encoding="utf-8")
        (self.root / "tasks.txt").write_text(SAMPLE, encoding="utf-8")
        self.assertIn("protein-assembly", datasets.task_meta(DATASET))


if __name__ == "__main__":
    unittest.main()


class DownloadCancelledTests(unittest.TestCase):
    def test_the_dataset_argument_is_optional_and_pickles(self) -> None:
        import pickle

        self.assertEqual(str(datasets.DownloadCancelled()), "Cancelled during the dataset download")
        exc = pickle.loads(pickle.dumps(datasets.DownloadCancelled("terminal-bench@2.0")))
        self.assertEqual((str(exc), exc.dataset), ("Cancelled while downloading terminal-bench@2.0",
                                                    "terminal-bench@2.0"))
