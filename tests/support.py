"""Shared scaffolding for bench tests.

BenchTestCase keeps every test away from what costs money or touches real
data: non-loopback network connections are refused, harbor, docker and
taskkill cannot be started unless a test supplies a fake, harness schemas
come from tests/fixtures/schemas instead of a Harbor subprocess, and config
is loaded from temp copies. It depends only on bench.settings, so it keeps
working while other modules change.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from unittest import mock

from tests import TEST_ROOT

from bench import settings

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SCHEMA_DIR = FIXTURES / "schemas"

# Programs no test may start for real: they run agents, drive containers or
# kill process trees.
BLOCKED_PROGRAMS = frozenset({"harbor", "docker", "docker-compose", "taskkill"})
# Shells that would run a blocked program named anywhere in their arguments.
SHELL_WRAPPERS = frozenset({"cmd", "powershell", "pwsh", "wsl", "bash", "sh"})
# Python modules that run blocked programs themselves: the detached container
# cleanup helper (procs.spawn_detached_cleanup) calls docker.
BLOCKED_MODULES = frozenset({"bench.procs"})

# A small valid config for tests that only need something to load.
MINIMAL_YAML = """\
dataset: terminal-bench@2.0
providers:
  anthropic: {env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}
  openrouter: {env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}
modes:
  pilot: {n_tasks: 1}
run:
  budget_usd: 10
arms:
  - {name: claude-code, agent: claude-code, provider: anthropic, model: claude-opus-5-5}
"""

_DATA_PATHS = ("DATA_DIR", "JOBS_DIR", "RUNS_DIR", "DESIGNS_DIR", "ASSISTANT_DIR", "QUEUE_FILE")
_PASS = object()
_schema_cache: dict[str, dict[str, Any]] = {}


class NetworkBlocked(ConnectionRefusedError):
    pass


def fixture_schema(agent: str) -> dict[str, Any]:
    """Stand-in for schemas.agent_schema: the schema Harbor 0.25.0 printed."""
    if agent not in _schema_cache:
        path = SCHEMA_DIR / (agent.replace(":", "_") + ".json")
        if not path.exists():
            raise RuntimeError(f"harbor agents schema {agent} failed: no test fixture {path.name}")
        _schema_cache[agent] = json.loads(path.read_text(encoding="utf-8"))
    return copy.deepcopy(_schema_cache[agent])


def argv_of(args: Any) -> list[str]:
    if isinstance(args, (str, bytes)):
        text = args.decode() if isinstance(args, bytes) else args
        return text.split()
    if isinstance(args, os.PathLike):
        return [os.fspath(args)]
    return [os.fspath(a) if isinstance(a, os.PathLike) else str(a) for a in args or []]


def program_name(args: Any) -> str:
    """Lower-case program name without folder or Windows extension."""
    argv = argv_of(args)
    if not argv:
        return ""
    name = os.path.basename(argv[0].strip('"')).lower()
    for ext in (".exe", ".cmd", ".bat"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def blocked_reason(argv: Sequence[str], shell: bool = False) -> str | None:
    """Why a test may not start argv for real, or None if it may. With
    shell (subprocess's shell=True) every word of the command line counts."""
    name = program_name(argv)
    if name in BLOCKED_PROGRAMS:
        return name
    if shell or name in SHELL_WRAPPERS:
        # cmd /c docker ps, powershell -Command "& docker ps; ...": any word
        # in the command line counts, since the shell decides what runs.
        for token in (argv if shell else argv[1:]):
            for word in re.split(r"[\s;&|()]+", token):
                inner = program_name([word.strip("'\"")]) if word else ""
                if inner in BLOCKED_PROGRAMS:
                    return f"{inner} through {'the shell' if shell else name}"
    for i, token in enumerate(argv[:-1]):
        if token == "-m" and argv[i + 1] in BLOCKED_MODULES:
            return f"python -m {argv[i + 1]}"
    for token in argv:
        if token.startswith("-m") and token[2:] in BLOCKED_MODULES:
            return f"python -m {token[2:]}"
    return None


def is_loopback(host: Any) -> bool:
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    if not isinstance(host, str):
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        return True
    except ValueError:
        return False


def rmtree(path: Path, patience: float = 3.0) -> None:
    """shutil.rmtree that waits out Windows' brief holds on just-closed files."""
    deadline = time.monotonic() + patience
    while True:
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                shutil.rmtree(path, ignore_errors=True)
                return
            time.sleep(0.1)


def completed(argv: Sequence[str] | None = None, stdout: str = "", stderr: str = "",
              returncode: int = 0) -> subprocess.CompletedProcess:
    """A CompletedProcess for fakes passed to allow_subprocess."""
    return subprocess.CompletedProcess(list(argv or []), returncode, stdout, stderr)


class BenchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._allowed: list[tuple[Callable[[list[str]], bool], Callable[..., Any]]] = []
        saved_env = dict(os.environ)
        self.addCleanup(_restore_env, saved_env)
        self._guard_sockets()
        self._guard_subprocesses()
        self._use_schema_fixtures()

    # Guards ----------------------------------------------------------------

    def allow_subprocess(self, predicate: Callable[[list[str]], bool], fake: Callable[..., Any]) -> None:
        """Route matching subprocess.run/Popen calls to fake(argv, **kwargs).

        The fake's return value is returned to the caller in place of a
        CompletedProcess (run) or a Popen object (Popen). Later registrations win.
        A fake must not return a real Popen (an instance of the guarded class):
        Python would run __init__ on it a second time. Wrap a real child in a
        proxy object instead.
        """
        self._allowed.append((predicate, fake))

    def _route(self, args: Any, kwargs: dict[str, Any]) -> Any:
        argv = argv_of(args)
        for predicate, fake in reversed(self._allowed):
            if predicate(argv):
                return fake(argv, **kwargs)
        reason = blocked_reason(argv, shell=bool(kwargs.get("shell")))
        if reason:
            raise AssertionError(f"Test tried to start {reason} ({' '.join(argv[:4])}); "
                                 "use allow_subprocess with a fake")
        return _PASS

    def _guard_subprocesses(self) -> None:
        case = self
        real_run = subprocess.run
        real_popen = subprocess.Popen

        class GuardedPopen(real_popen):  # type: ignore[misc, valid-type]
            def __new__(cls, args: Any = None, *a: Any, **kw: Any) -> Any:
                fake = case._route(args if args is not None else kw.get("args"), kw)
                if fake is not _PASS:
                    return fake  # not a GuardedPopen, so __init__ is skipped
                return super().__new__(cls)

        def guarded_run(*popenargs: Any, **kw: Any) -> Any:
            fake = case._route(popenargs[0] if popenargs else kw.get("args"), kw)
            if fake is not _PASS:
                return fake
            return real_run(*popenargs, **kw)

        # subprocess.run, call and check_output all look up Popen and run in
        # the module at call time, so these two patches cover them.
        for name, value in (("Popen", GuardedPopen), ("run", guarded_run)):
            patcher = mock.patch.object(subprocess, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        if sys.platform == "win32":
            # asyncio's Windows subprocesses use a Popen subclass bound at
            # import time, which the patch above does not reach.
            import asyncio.windows_utils as windows_utils

            real_async_popen = windows_utils.Popen

            class GuardedAsyncPopen(real_async_popen):  # type: ignore[misc, valid-type]
                def __new__(cls, args: Any = None, *a: Any, **kw: Any) -> Any:
                    fake = case._route(args if args is not None else kw.get("args"), kw)
                    if fake is not _PASS:
                        return fake
                    return super().__new__(cls)

            patcher = mock.patch.object(windows_utils, "Popen", GuardedAsyncPopen)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _guard_sockets(self) -> None:
        real_connect = socket.socket.connect
        real_connect_ex = socket.socket.connect_ex
        real_getaddrinfo = socket.getaddrinfo

        def check(address: Any) -> None:
            if isinstance(address, tuple) and address and not is_loopback(address[0]):
                raise NetworkBlocked(f"Network access is blocked in tests: {address[0]}")

        def connect(sock: socket.socket, address: Any) -> None:
            check(address)
            return real_connect(sock, address)

        def connect_ex(sock: socket.socket, address: Any) -> int:
            check(address)
            return real_connect_ex(sock, address)

        def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> list[Any]:
            name = host.decode(errors="replace") if isinstance(host, bytes) else host
            # Refuse name lookups for remote hosts, so a stray request fails at
            # once instead of after a DNS round trip. IP literals need no lookup.
            if name and not is_loopback(name) and not _is_ip_literal(name):
                raise socket.gaierror(socket.EAI_NONAME, f"Network access is blocked in tests: {name}")
            return real_getaddrinfo(host, *args, **kwargs)

        for target, name, value in ((socket.socket, "connect", connect),
                                    (socket.socket, "connect_ex", connect_ex),
                                    (socket, "getaddrinfo", getaddrinfo)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _use_schema_fixtures(self) -> None:
        try:
            from bench import schemas
        except Exception:  # another package may be mid-edit; tests that need it fail on their own
            return
        if hasattr(schemas, "agent_schema"):
            patcher = mock.patch.object(schemas, "agent_schema", fixture_schema)
            patcher.start()
            self.addCleanup(patcher.stop)

    # Files and config ------------------------------------------------------

    def tmpdir(self) -> Path:
        path = Path(tempfile.mkdtemp(prefix="t-", dir=TEST_ROOT))
        self.addCleanup(rmtree, path)
        return path

    def _config_files(self, yaml_text: str, overrides: str | None, pricing: str | None,
                      reference: str | None, tasks: Sequence[str] | None) -> dict[str, Path]:
        d = self.tmpdir()
        text = textwrap.dedent(yaml_text)
        if tasks is not None:
            tasks_path = d / "tasks.txt"
            tasks_path.write_text("".join(f"{t}\n" for t in tasks), encoding="utf-8", newline="\n")
            if "tasks_file:" in text:
                raise ValueError("Pass tasks or set tasks_file in the YAML, not both")
            text += f"\ntasks_file: {json.dumps(tasks_path.as_posix())}\n"
        paths = {"base": d / "bench.yaml", "overrides": d / "overrides.yaml",
                 "pricing": d / "pricing.yaml", "reference": d / "reference.yaml"}
        paths["base"].write_text(text, encoding="utf-8", newline="\n")
        if overrides is not None:
            paths["overrides"].write_text(textwrap.dedent(overrides), encoding="utf-8", newline="\n")
        for key, given, real in (("pricing", pricing, settings.PRICING),
                                 ("reference", reference, settings.REFERENCE)):
            content = textwrap.dedent(given) if given is not None else real.read_text(encoding="utf-8")
            paths[key].write_text(content, encoding="utf-8", newline="\n")
        return paths

    def load_test_config(self, yaml_text: str, overrides: str = "", *, pricing: str | None = None,
                         reference: str | None = None, tasks: Sequence[str] | None = None) -> settings.Config:
        """Load yaml_text (and overrides) as a profile, with temp copies of
        pricing.yaml and reference.yaml unless replacements are given."""
        paths = self._config_files(yaml_text, overrides, pricing, reference, tasks)
        with mock.patch.object(settings, "PRICING", paths["pricing"]), \
                mock.patch.object(settings, "REFERENCE", paths["reference"]):
            return settings.load_config(paths["base"], paths["overrides"])

    def load_shipped_config(self, name: str = "bench") -> settings.Config:
        """config/<name>.yaml as shipped, without any local overrides file."""
        empty = self.tmpdir() / "overrides.yaml"
        return settings.load_config(settings.CONFIG_DIR / f"{name}.yaml", empty)

    def use_config_files(self, yaml_text: str, overrides: str | None = None, *,
                         pricing: str | None = None, reference: str | None = None,
                         tasks: Sequence[str] | None = None) -> tuple[Path, Path]:
        """Point settings at temp config files for the rest of the test, so
        load_config(), base_config() and write_overrides() use them.
        Returns (base, overrides); the overrides file exists only if given."""
        paths = self._config_files(yaml_text, overrides, pricing, reference, tasks)
        for attr, key in (("BASE_CONFIG", "base"), ("OVERRIDES", "overrides"),
                          ("PRICING", "pricing"), ("REFERENCE", "reference")):
            patcher = mock.patch.object(settings, attr, paths[key])
            patcher.start()
            self.addCleanup(patcher.stop)
        return paths["base"], paths["overrides"]

    def isolate_data_dir(self) -> Path:
        """A fresh DATA_DIR for this test. Also patches bench modules that
        imported JOBS_DIR, RUNS_DIR and the rest by name."""
        root = self.tmpdir() / "data"
        new = {"DATA_DIR": root, "JOBS_DIR": root / "jobs", "RUNS_DIR": root / "runs",
               "DESIGNS_DIR": root / "designs", "ASSISTANT_DIR": root / "assistant",
               "QUEUE_FILE": root / "queue.json"}
        old = {name: getattr(settings, name) for name in _DATA_PATHS}
        modules = [m for n, m in list(sys.modules.items())
                   if m is not None and (n == "bench" or n.startswith("bench."))]
        for module in modules:
            for name in _DATA_PATHS:
                if module is settings or getattr(module, name, None) == old[name]:
                    patcher = mock.patch.object(module, name, new[name])
                    patcher.start()
                    self.addCleanup(patcher.stop)
        return root

    def isolate_ledger(self) -> Path:
        path = self.tmpdir() / "ledger" / "bench.sqlite"
        patcher = mock.patch.object(settings, "LEDGER_DB", path)
        patcher.start()
        self.addCleanup(patcher.stop)
        return path


def _restore_env(saved: dict[str, str]) -> None:
    for name in [n for n in os.environ if n not in saved]:
        del os.environ[name]
    for name, value in saved.items():
        if os.environ.get(name) != value:
            os.environ[name] = value
