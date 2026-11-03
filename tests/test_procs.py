"""Process safety: the per-profile instance lock, the kill-on-close job object,
console-close handling, and container cleanup that reports failure instead of
hiding it.

Docker, harbor and taskkill never run for real: fakes stand in for them
through allow_subprocess. The lock and job-object tests start only small
python children, and kill only those.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from tests.support import BenchTestCase, completed, program_name

from bench import procs, settings

WINDOWS = os.name == "nt"

HOLD_INSTANCE_LOCK = """
import sys
from bench import procs
procs.acquire_instance_lock(sys.argv[1])
print("held", flush=True)
sys.stdin.read()
"""

TRY_INSTANCE_LOCK = """
import sys
from bench import procs
try:
    procs.acquire_instance_lock(sys.argv[1])
except procs.AlreadyRunning as exc:
    print("refused:", exc)
    sys.exit(3)
print("acquired")
"""

HOLD_FILE_LOCK = """
import sys
from bench import procs
with procs.file_lock(sys.argv[1], timeout=30):
    print("locked", flush=True)
    sys.stdin.read()
"""

# The grandparent in the job-object probe: starts a sleeping python child,
# puts it in the kill-on-close job, reports both of its pids, then waits to be killed.
JOB_OWNER = """
import subprocess, sys, time
from pathlib import Path
from bench import procs
out = Path(sys.argv[1])
code = "import os, sys, time; open(sys.argv[1], 'w', encoding='utf-8').write(str(os.getpid())); time.sleep(120)"
child = subprocess.Popen([sys.executable, "-c", code, str(out)], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=procs.NO_CTRL_C)
assigned = procs.assign_kill_on_close(child)
deadline = time.monotonic() + 30
while not (out.exists() and out.read_text(encoding='utf-8')) and time.monotonic() < deadline:
    time.sleep(0.05)
print(assigned, child.pid, out.read_text(encoding='utf-8'), flush=True)
sys.stdin.read()
"""


def python_child(code: str, *args: str) -> subprocess.Popen:
    """A python child that can import bench; it inherits the test's isolated env."""
    return subprocess.Popen([sys.executable, "-c", textwrap.dedent(code), *args], cwd=settings.ROOT,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8")


def run_python(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code), *args], cwd=settings.ROOT,
                          capture_output=True, text=True, encoding="utf-8", timeout=60)


def reap(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(30)
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None:
            stream.close()


class InstanceLockTests(BenchTestCase):
    def test_second_process_is_refused_until_the_holder_dies(self) -> None:
        d = self.tmpdir()
        holder = python_child(HOLD_INSTANCE_LOCK, str(d))
        self.addCleanup(reap, holder)
        self.assertEqual(holder.stdout.readline().strip(), "held", holder.stderr.read() if holder.poll() else "")

        second = run_python(TRY_INSTANCE_LOCK, str(d))
        self.assertEqual(second.returncode, 3, second.stderr)
        self.assertIn(procs.ALREADY_RUNNING, second.stdout)

        holder.kill()  # a hard kill releases nothing by hand: the OS drops the lock
        holder.wait(30)
        deadline = time.monotonic() + 10
        while True:
            third = run_python(TRY_INSTANCE_LOCK, str(d))
            if third.returncode == 0 or time.monotonic() > deadline:
                break
            time.sleep(0.3)  # the venv launcher exits a moment before its interpreter
        self.assertEqual(third.returncode, 0, third.stdout + third.stderr)
        self.assertEqual(third.stdout.strip(), "acquired")

    def test_held_in_this_process_refuses_others_and_is_reentrant(self) -> None:
        d = self.tmpdir()
        fd = procs.acquire_instance_lock(d)
        self.addCleanup(procs.release_instance_lock, fd)
        self.assertEqual(procs.acquire_instance_lock(d), fd)
        self.assertTrue((d / ".bench.lock").exists())

        other = run_python(TRY_INSTANCE_LOCK, str(d))
        self.assertEqual(other.returncode, 3, other.stderr)

        procs.release_instance_lock(fd)
        again = run_python(TRY_INSTANCE_LOCK, str(d))
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)

    def test_already_running_is_a_config_error_with_the_plain_message(self) -> None:
        exc = procs.AlreadyRunning()
        self.assertIsInstance(exc, settings.ConfigError)
        self.assertEqual(str(exc), "The bench is already running for this profile. "
                                   "Use the dashboard URL printed by that process.")


class FileLockTests(BenchTestCase):
    def test_waits_times_out_cancels_and_then_acquires(self) -> None:
        lock = self.tmpdir() / "datasets" / ".lock"
        holder = python_child(HOLD_FILE_LOCK, str(lock))
        self.addCleanup(reap, holder)
        self.assertEqual(holder.stdout.readline().strip(), "locked")

        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            with procs.file_lock(lock, timeout=1):
                self.fail("acquired a lock another process holds")
        self.assertLess(time.monotonic() - start, 3)

        cancel = threading.Event()
        threading.Timer(0.2, cancel.set).start()
        start = time.monotonic()
        with self.assertRaises(procs.LockCancelled):
            with procs.file_lock(lock, timeout=30, cancel=cancel):
                self.fail("acquired a lock another process holds")
        self.assertLess(time.monotonic() - start, 2)

        holder.stdin.close()  # the holder leaves its with-block and exits
        holder.wait(30)
        entered = False
        with procs.file_lock(lock, timeout=10):
            entered = True
        self.assertTrue(entered)

    def test_serialises_threads_in_one_process(self) -> None:
        lock = self.tmpdir() / ".lock"
        inside: list[int] = []
        overlaps: list[int] = []

        def work() -> None:
            with procs.file_lock(lock, timeout=30):
                inside.append(1)
                if len(inside) > 1:
                    overlaps.append(1)
                time.sleep(0.2)
                inside.pop()

        threads = [threading.Thread(target=work) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(overlaps, [])


@unittest.skipUnless(WINDOWS, "job objects are Windows-only")
class KillOnCloseTests(BenchTestCase):
    def test_child_dies_with_the_process_that_owns_the_job(self) -> None:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.WaitForSingleObject.restype = wintypes.DWORD
        k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.TerminateProcess.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        synchronize_and_terminate = 0x00100000 | 0x0001 | 0x1000
        wait_object_0, wait_timeout = 0, 0x102

        def open_process(pid: int) -> int:
            handle = k32.OpenProcess(synchronize_and_terminate, False, pid)
            self.assertTrue(handle, f"could not open pid {pid}")

            def close() -> None:
                if k32.WaitForSingleObject(handle, 0) == wait_timeout:
                    k32.TerminateProcess(handle, 1)  # only if the test failed
                k32.CloseHandle(handle)

            self.addCleanup(close)
            return handle

        out = self.tmpdir() / "grandchild.pid"
        owner = python_child(JOB_OWNER, str(out))
        self.addCleanup(reap, owner)
        line = owner.stdout.readline().split()
        self.assertEqual(len(line), 3, owner.stderr.read() if owner.poll() is not None else line)
        assigned, launcher_pid, interpreter_pid = line[0], int(line[1]), int(line[2])
        self.assertEqual(assigned, "True")
        handles = [open_process(launcher_pid), open_process(interpreter_pid)]
        for handle in handles:
            self.assertEqual(k32.WaitForSingleObject(handle, 0), wait_timeout, "child not running")

        owner.kill()  # the owner dies uncleanly, as the bench does in Task Manager
        owner.wait(30)
        for handle in handles:
            self.assertEqual(k32.WaitForSingleObject(handle, 15000), wait_object_0,
                             "the child outlived the process that owned its job")


class AssignAndKillTests(BenchTestCase):
    def test_assign_returns_false_for_something_that_is_not_a_process(self) -> None:
        self.assertFalse(procs.assign_kill_on_close(mock.Mock(spec=[])))

    @unittest.skipIf(WINDOWS, "POSIX only")
    def test_assign_is_a_no_op_off_windows(self) -> None:
        self.assertFalse(procs.assign_kill_on_close(mock.Mock()))

    @unittest.skipUnless(WINDOWS, "taskkill is Windows-only")
    def test_kill_tree_uses_taskkill_out_of_ctrl_c_reach_and_ignores_dead_pids(self) -> None:
        calls: list[tuple[list[str], dict[str, Any]]] = []

        def taskkill(argv: list[str], **kw: Any) -> subprocess.CompletedProcess:
            calls.append((argv, kw))
            return completed(argv, stderr='ERROR: The process "4321" not found.', returncode=128)

        self.allow_subprocess(lambda argv: program_name(argv) == "taskkill", taskkill)
        procs.kill_tree(4321)
        self.assertEqual(len(calls), 1)
        argv, kw = calls[0]
        self.assertEqual(argv, ["taskkill", "/PID", "4321", "/T", "/F"])
        self.assertEqual(kw["creationflags"], procs.NO_CTRL_C)
        self.assertTrue(kw["capture_output"])
        self.assertEqual(kw["timeout"], 30)

    @unittest.skipUnless(WINDOWS, "taskkill is Windows-only")
    def test_kill_tree_falls_back_to_terminating_the_root_and_never_raises(self) -> None:
        def broken(argv: list[str], **kw: Any) -> Any:
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))

        self.allow_subprocess(lambda argv: program_name(argv) == "taskkill", broken)
        with mock.patch.object(procs.os, "kill", side_effect=OSError(87, "gone")) as kill:
            procs.kill_tree(4321, timeout=5)
        kill.assert_called_once()
        self.assertEqual(kill.call_args.args[0], 4321)


class FakeDocker:
    """The docker CLI over an in-memory list of containers and networks, each
    with its compose project label."""

    def __init__(self, containers: dict[str, str], networks: dict[str, str]) -> None:
        self.stores = {"container": dict(containers), "network": dict(networks)}
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.failing_ls: set[str] = set()
        self.stubborn: set[str] = set()     # rm cannot remove these
        self.rm_returncode: int | None = None

    def __call__(self, argv: list[str], **kw: Any) -> subprocess.CompletedProcess:
        self.calls.append((argv, kw))
        kind, verb = argv[1], argv[2]
        store = self.stores[kind]
        if verb == "ls":
            if kind in self.failing_ls:
                return completed(argv, stderr="failed to connect to the docker API at npipe:////./pipe/x",
                                 returncode=1)
            return completed(argv, stdout="".join(f"{i}\t{p}\n" for i, p in store.items()))
        if verb == "rm":
            ids = [a for a in argv[3:] if not a.startswith("-")]
            for i in ids:
                if i not in self.stubborn:
                    store.pop(i, None)
            stuck = [i for i in ids if i in self.stubborn]
            rc = self.rm_returncode if self.rm_returncode is not None else (1 if stuck else 0)
            return completed(argv, stderr=f"Error response from daemon: cannot remove {stuck}" if rc else "",
                             returncode=rc)
        raise AssertionError(f"unexpected docker call {argv}")

    def verbs(self) -> list[str]:
        return [f"{argv[1]} {argv[2]}" for argv, _ in self.calls]


class CleanupContainersTests(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.job_dir = self.tmpdir() / "jobs" / "20261029-101500-sample" / "claude-code"
        for trial in ("fix-git__AbC1dEf", "cobol-modernization__x9Y8z7w"):
            (self.job_dir / trial).mkdir(parents=True)
        (self.job_dir / "result.json").write_text("{}", encoding="utf-8")
        self.docker = FakeDocker(
            containers={"c1": "fix-git__abc1def__env", "c2": "cobol-modernization__x9y8z7w__env",
                        "c3": "another-job__zzzzzzz__env", "c4": ""},
            networks={"n1": "fix-git__abc1def__env", "n2": "bridge-of-something-else"},
        )
        self.allow_subprocess(lambda argv: program_name(argv) == "docker", self.docker)

    def test_removes_only_this_jobs_containers_and_networks_and_counts_them(self) -> None:
        self.assertEqual(procs.cleanup_containers(self.job_dir), 2)
        self.assertEqual(sorted(self.docker.stores["container"]), ["c3", "c4"])
        self.assertEqual(sorted(self.docker.stores["network"]), ["n2"])
        self.assertEqual(self.docker.verbs(), ["container ls", "container rm", "container ls",
                                               "network ls", "network rm", "network ls"])
        rm_argv = self.docker.calls[1][0]
        self.assertEqual(rm_argv, ["docker", "container", "rm", "-f", "c1", "c2"])

    def test_every_docker_call_is_out_of_ctrl_c_reach_and_bounded(self) -> None:
        procs.cleanup_containers(self.job_dir)
        self.assertTrue(self.docker.calls)
        for argv, kw in self.docker.calls:
            self.assertEqual(kw.get("creationflags"), procs.NO_CTRL_C, argv)
            self.assertIsNotNone(kw.get("timeout"), argv)
            self.assertFalse(kw.get("shell", False), argv)
        if WINDOWS:
            self.assertEqual(procs.NO_CTRL_C, subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)

    def test_a_failed_listing_raises_instead_of_reporting_nothing_to_remove(self) -> None:
        self.docker.failing_ls.add("container")
        with self.assertRaises(subprocess.SubprocessError) as caught:
            procs.cleanup_containers(self.job_dir)
        self.assertIn("docker container ls failed (exit 1)", str(caught.exception))
        self.assertIn("failed to connect", str(caught.exception))
        self.assertEqual(self.docker.verbs(), ["container ls"])

    def test_survivors_after_rm_raise_and_networks_are_left_for_the_retry(self) -> None:
        self.docker.stubborn.add("c2")
        with self.assertRaises(subprocess.SubprocessError) as caught:
            procs.cleanup_containers(self.job_dir)
        self.assertIn("1 container(s) still present after removing 1", str(caught.exception))
        self.assertIn("cannot remove", str(caught.exception))
        self.assertNotIn("network rm", self.docker.verbs())

    def test_a_container_that_appears_during_cleanup_counts_as_a_survivor(self) -> None:
        real_call = self.docker.__call__

        def late_start(argv: list[str], **kw: Any) -> subprocess.CompletedProcess:
            result = real_call(argv, **kw)
            if argv[1:3] == ["container", "rm"]:
                self.docker.stores["container"]["c9"] = "fix-git__abc1def__env"
            return result

        self.allow_subprocess(lambda argv: program_name(argv) == "docker", late_start)
        with self.assertRaises(subprocess.SubprocessError) as caught:
            procs.cleanup_containers(self.job_dir)
        self.assertIn("1 container(s) still present after removing 2", str(caught.exception))

    def test_rm_errors_for_containers_that_vanished_by_themselves_are_not_failures(self) -> None:
        self.docker.rm_returncode = 1  # "No such container": harbor removed one first
        self.assertEqual(procs.cleanup_containers(self.job_dir), 2)

    def test_a_network_that_cannot_be_removed_raises(self) -> None:
        self.docker.stubborn.add("n1")
        with self.assertRaises(subprocess.SubprocessError) as caught:
            procs.cleanup_containers(self.job_dir)
        self.assertIn("1 network(s) still present", str(caught.exception))

    def test_missing_or_empty_job_dirs_need_no_docker(self) -> None:
        self.assertEqual(procs.cleanup_containers(self.tmpdir() / "never-created"), 0)
        self.assertEqual(procs.cleanup_containers(self.tmpdir()), 0)
        self.assertEqual(self.docker.calls, [])

    def test_docker_that_cannot_start_raises_subprocess_error(self) -> None:
        def missing(argv: list[str], **kw: Any) -> Any:
            raise FileNotFoundError(2, "The system cannot find the file specified")

        self.allow_subprocess(lambda argv: program_name(argv) == "docker", missing)
        with self.assertRaises(subprocess.SubprocessError) as caught:
            procs.cleanup_containers(self.job_dir)
        self.assertIn("docker could not be started", str(caught.exception))

    def test_helper_env_fallback_drops_credentials(self) -> None:
        with mock.patch.dict(procs.os.environ, {"BENCH_FAKE_API_KEY": "not-a-real-key-123",
                                               "FAKE_BASE_URL": "http://x", "BENCH_HARMLESS": "1"}), \
                mock.patch("bench.keys.child_env", side_effect=RuntimeError("boom")):
            env = procs._helper_env()
        self.assertNotIn("BENCH_FAKE_API_KEY", env)
        self.assertNotIn("FAKE_BASE_URL", env)
        self.assertEqual(env.get("BENCH_HARMLESS"), "1")


class CleanupWithRetryTests(BenchTestCase):
    def test_retries_then_reports_the_count(self) -> None:
        job_dir = self.tmpdir()
        with mock.patch.object(procs, "cleanup_containers",
                               side_effect=[subprocess.SubprocessError("daemon hiccup"),
                                            OSError("pipe busy"), 2]) as cleanup:
            ok, message = procs.cleanup_with_retry(job_dir, tries=3, delay=0)
        self.assertTrue(ok)
        self.assertEqual(message, "removed 2 container(s)")
        self.assertEqual(cleanup.call_count, 3)

    def test_reports_failed_after_the_last_try(self) -> None:
        sleeps: list[float] = []
        failure = subprocess.SubprocessError("docker container ls failed (exit 1)")
        with mock.patch.object(procs, "cleanup_containers", side_effect=failure) as cleanup, \
                mock.patch.object(procs.time, "sleep", side_effect=sleeps.append):
            ok, message = procs.cleanup_with_retry(self.tmpdir(), tries=3, delay=2.0)
        self.assertFalse(ok)
        self.assertTrue(message.startswith("container cleanup FAILED: docker container ls failed (exit 1)"), message)
        self.assertEqual(cleanup.call_count, 3)
        self.assertEqual(sleeps, [2.0, 2.0])

    def test_does_not_swallow_programming_errors(self) -> None:
        with mock.patch.object(procs, "cleanup_containers", side_effect=TypeError("bug")):
            with self.assertRaises(TypeError):
                procs.cleanup_with_retry(self.tmpdir(), delay=0)


class ComposeProjectTests(BenchTestCase):
    def harbor_sanitizer(self) -> Any:
        """Harbor's own function, compiled from its source without importing harbor."""
        spec = importlib.util.find_spec("harbor")
        if spec is None or not spec.submodule_search_locations:
            self.skipTest("harbor is not installed")
        source = Path(list(spec.submodule_search_locations)[0]) / "environments" / "docker" / "docker.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        func = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "_sanitize_docker_compose_project_name"), None)
        if func is None:
            self.fail("Harbor no longer has _sanitize_docker_compose_project_name; recheck compose_project")
        namespace: dict[str, Any] = {"re": re}
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(source), "exec"), namespace)
        return namespace[func.name]

    def test_matches_harbors_sanitizer(self) -> None:
        sanitize = self.harbor_sanitizer()
        for name in ("fix-git__AbC1dEf", "Fix_Git__x", "_leading", "-dash", ".dot", "has space",
                     "unicode-é__1", "UPPER.case:colon", "0ok", "a" * 80):
            self.assertEqual(procs.compose_project(name), sanitize(name), name)
            self.assertTrue(sanitize(f"{name}__env").startswith(procs.compose_project(name)), name)


class ConsoleCloseTests(BenchTestCase):
    def test_closing_events_run_the_callback_and_others_pass_through(self) -> None:
        calls: list[int] = []
        for event in (procs.CTRL_CLOSE_EVENT, procs.CTRL_LOGOFF_EVENT, procs.CTRL_SHUTDOWN_EVENT):
            self.assertTrue(procs._on_console_event(event, lambda: calls.append(1)))
        for event in (0, 1):  # Ctrl+C, Ctrl+Break
            self.assertFalse(procs._on_console_event(event, lambda: calls.append(1)))
        self.assertEqual(len(calls), 3)

    def test_a_failing_callback_never_escapes(self) -> None:
        def boom() -> None:
            raise RuntimeError("emergency stop failed")

        with mock.patch("sys.stderr"):
            self.assertTrue(procs._on_console_event(procs.CTRL_CLOSE_EVENT, boom))

    @unittest.skipUnless(WINDOWS, "console control handlers are Windows-only")
    def test_installed_handler_is_kept_alive_and_routes_events(self) -> None:
        calls: list[int] = []
        self.addCleanup(procs._remove_console_close_handlers)
        self.assertTrue(procs.install_console_close_handler(lambda: calls.append(1)))
        handler = procs._console_handlers[-1]
        self.assertFalse(handler(0))
        self.assertTrue(handler(procs.CTRL_CLOSE_EVENT))
        self.assertEqual(calls, [1])

    @unittest.skipIf(WINDOWS, "POSIX only")
    def test_install_is_a_no_op_off_windows(self) -> None:
        self.assertFalse(procs.install_console_close_handler(lambda: None))


class DetachedCleanupTests(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.refuse_breakaway = False
        self.refuse_all = False

        def helper(argv: list[str], **kw: Any) -> Any:
            self.calls.append((argv, kw))
            flags = kw.get("creationflags", 0)
            if self.refuse_all or (self.refuse_breakaway and WINDOWS
                                   and flags & subprocess.CREATE_BREAKAWAY_FROM_JOB):
                raise PermissionError(5, "Access is denied")
            return mock.Mock(pid=4242)

        self.allow_subprocess(lambda argv: "bench.procs" in argv, helper)

    def test_builds_the_helper_command(self) -> None:
        os.environ["BENCH_PROFILE"] = "dev"
        dirs = [self.tmpdir() / "a", self.tmpdir() / "b"]
        self.assertTrue(procs.spawn_detached_cleanup(dirs))
        self.assertEqual(len(self.calls), 1)
        argv, kw = self.calls[0]
        self.assertEqual(argv[:4], [sys.executable, "-m", "bench.procs", "cleanup"])
        self.assertEqual(argv[4:], [os.fspath(d.absolute()) for d in dirs])
        self.assertEqual(Path(kw["cwd"]), settings.ROOT)
        env = kw["env"]
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(settings.ROOT))
        self.assertEqual(env["BENCH_PROFILE"], "dev")
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("OPENROUTER_API_KEY", env)
        for stream in ("stdin", "stdout", "stderr"):
            self.assertEqual(kw[stream], subprocess.DEVNULL)
        if WINDOWS:
            flags = kw["creationflags"]
            for flag in (subprocess.CREATE_NO_WINDOW, subprocess.CREATE_NEW_PROCESS_GROUP,
                         subprocess.CREATE_BREAKAWAY_FROM_JOB):
                self.assertTrue(flags & flag, hex(flag))
            self.assertFalse(flags & subprocess.DETACHED_PROCESS)

    @unittest.skipUnless(WINDOWS, "job breakaway is Windows-only")
    def test_retries_without_breakaway_when_the_job_forbids_it(self) -> None:
        self.refuse_breakaway = True
        self.assertTrue(procs.spawn_detached_cleanup([self.tmpdir()]))
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(self.calls[1][1]["creationflags"] & subprocess.CREATE_BREAKAWAY_FROM_JOB)

    def test_reports_false_when_the_helper_cannot_start(self) -> None:
        self.refuse_all = True
        self.assertFalse(procs.spawn_detached_cleanup([self.tmpdir()]))

    def test_nothing_to_clean_starts_nothing(self) -> None:
        self.assertTrue(procs.spawn_detached_cleanup([]))
        self.assertEqual(self.calls, [])


class MainTests(BenchTestCase):
    def test_exit_code_reflects_every_job(self) -> None:
        results = {"a": (True, "removed 1 container(s)"), "b": (False, "container cleanup FAILED: x")}
        with mock.patch.object(procs, "cleanup_with_retry", side_effect=lambda d: results[Path(d).name]), \
                mock.patch("sys.stderr"):
            self.assertEqual(procs.main(["cleanup", "a"]), 0)
            self.assertEqual(procs.main(["cleanup", "a", "b"]), 1)

    def test_bad_usage_fails_without_touching_docker(self) -> None:
        with mock.patch.object(procs, "cleanup_with_retry") as cleanup, mock.patch("sys.stderr"):
            self.assertEqual(procs.main([]), 1)
            self.assertEqual(procs.main(["sweep", "x"]), 1)
        cleanup.assert_not_called()

    def test_import_stays_light(self) -> None:
        heavy = ("harbor", "dotenv", "anthropic", "fastapi", "uvicorn", "httpx",
                 "bench.keys", "bench.runs", "bench.datasets", "bench.server", "bench.schemas")
        probe = run_python(f"""
            import sys
            import bench.procs
            heavy = {heavy!r}
            print(sorted(m for m in sys.modules if m in heavy or m.split(".")[0] in heavy))
        """)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertEqual(probe.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
