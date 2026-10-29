"""Readiness checks: Docker capacity parsing and caching, and the
assistant's readiness row. Docker is always a fake here."""

from __future__ import annotations

import os
import subprocess
from unittest import mock

from tests.support import MINIMAL_YAML, BenchTestCase, completed, program_name

from bench import checks

CAPACITY_ARGV = ["docker", "info", "--format", "{{.NCPU}} {{.MemTotal}}"]

OPENROUTER_ONLY = """\
dataset: terminal-bench@2.0
providers:
  openrouter: {env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}
modes:
  pilot: {n_tasks: 1}
run:
  budget_usd: 5
arms:
  - {name: pi-glm, agent: pi, provider: openrouter, model: z-ai/glm-5.3}
"""


def is_docker(argv: list[str]) -> bool:
    return program_name(argv) == "docker"


class ChecksTestCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.dict(checks._cache, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.calls: list[list[str]] = []

    def docker(self, stdout: str = "", returncode: int = 0, raises: BaseException | None = None):
        def fake(argv: list[str], **kwargs):
            self.calls.append(argv)
            if raises is not None:
                raise raises
            return completed(argv, stdout=stdout, returncode=returncode)
        self.allow_subprocess(is_docker, fake)


class DockerCapacityTests(ChecksTestCase):
    def test_parses_cpus_and_memory(self) -> None:
        self.docker("16 33554432000\n")
        self.assertEqual(checks.docker_capacity(), {"ncpu": 16, "mem_gb": 31.25})
        self.assertEqual(self.calls, [CAPACITY_ARGV])

    def test_cached_for_a_minute(self) -> None:
        self.docker("8 17179869184\n")
        first = checks.docker_capacity()
        second = checks.docker_capacity()
        self.assertEqual(first, {"ncpu": 8, "mem_gb": 16.0})
        self.assertEqual(second, first)
        self.assertEqual(len(self.calls), 1)

    def test_none_when_docker_fails(self) -> None:
        cases = [
            dict(stdout="", returncode=1),
            dict(stdout="Cannot connect to the Docker daemon"),
            dict(stdout="0 0\n"),
            dict(stdout="12\n"),
            dict(raises=FileNotFoundError("docker")),
            dict(raises=subprocess.TimeoutExpired(CAPACITY_ARGV, 20)),
        ]
        for case in cases:
            with self.subTest(case=case):
                checks._cache.clear()
                self.docker(**case)
                self.assertIsNone(checks.docker_capacity())


class AssistantRowTests(ChecksTestCase):
    def readiness(self, yaml_text: str) -> dict:
        cfg = self.load_test_config(yaml_text)
        with mock.patch.object(checks, "_docker", return_value={"ok": True, "detail": "Docker Engine 29"}), \
                mock.patch.object(checks, "_git_fix", return_value={"ok": True, "detail": "fixed"}), \
                mock.patch.object(checks, "_snapshot_detail", return_value="snapshot"):
            rows = checks.readiness(cfg)
        by_name = {r["name"]: r for r in rows}
        self.assertEqual(rows[-1]["name"], "Assistant")
        return by_name["Assistant"]

    def test_ready_with_key(self) -> None:
        row = self.readiness(OPENROUTER_ONLY)
        self.assertEqual(row, {"name": "Assistant", "ok": True, "detail": "Claude Opus 5.5 · effort medium"})

    def test_effort_from_config(self) -> None:
        row = self.readiness(MINIMAL_YAML + "assistant: {effort: high}\n")
        self.assertEqual(row["detail"], "Claude Opus 5.5 · effort high")

    def test_missing_key(self) -> None:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        row = self.readiness(OPENROUTER_ONLY)
        self.assertFalse(row["ok"])
        self.assertEqual(row["detail"], "Set ANTHROPIC_API_KEY on Setup")

    def test_disabled(self) -> None:
        row = self.readiness(OPENROUTER_ONLY + "assistant: {enabled: false}\n")
        self.assertFalse(row["ok"])
        self.assertEqual(row["detail"], "Disabled in config")


if __name__ == "__main__":
    import unittest

    unittest.main()
