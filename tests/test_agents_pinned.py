"""The harness subclasses that make an arm's `version` install exactly that release."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from tests.support import BenchTestCase

from bench.prebake.adapters import harbor_agents

harbor_agents()

from bench.agents.cline import PinnedClineCli  # noqa: E402
from bench.agents.goose import PinnedGoose, release_tag  # noqa: E402

MODEL = "openrouter/deepseek/deepseek-v4.1-flash"


class Execs:
    """Records what install() runs; answers every probe as on a bare apt image."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: Any = None) -> Any:
        from harbor.environments.base import ExecResult

        self.calls.append((command, dict(env or {})))
        rc, out = (0, "apt-get") if command.startswith("for manager") else (1 if "command -v" in command else 0, "")
        return ExecResult(stdout=out, stderr="", return_code=rc)


class PinnedHarnesses(BenchTestCase):
    def goose_install(self, **kwargs: Any) -> list[tuple[str, dict[str, str]]]:
        agent = PinnedGoose(logs_dir=self.tmpdir(), model_name=MODEL, **kwargs)
        env = Execs()
        asyncio.run(agent.install(env))  # type: ignore[arg-type]
        return [c for c in env.calls if "download_cli.sh" in c[0]]

    def test_goose_fetches_the_pinned_tag_and_tells_the_script_to_install_it(self) -> None:
        ((command, env),) = self.goose_install(version="1.57.0")
        self.assertIn("/releases/download/v1.57.0/download_cli.sh", command)
        self.assertEqual(env["GOOSE_VERSION"], "v1.57.0")
        agent = PinnedGoose(logs_dir=self.tmpdir(), model_name=MODEL, version="1.57.0")
        asyncio.run(agent.install(Execs()))  # type: ignore[arg-type]
        self.assertEqual(agent.version(), "1.57.0")  # reported as pinned, after the install too

    def test_unpinned_goose_is_harbors_install(self) -> None:
        ((command, env),) = self.goose_install()
        self.assertIn("/releases/download/stable/download_cli.sh", command)
        self.assertNotIn("GOOSE_VERSION", env)
        with self.assertRaises(ValueError):
            self.goose_install(version="1.57")
        self.assertEqual((release_tag("1.57.0"), release_tag("v1.57.0")), ("v1.57.0", "v1.57.0"))

    def test_cline_version_sets_the_npm_version(self) -> None:
        agent = PinnedClineCli(logs_dir=self.tmpdir(), model_name=MODEL, version="3.0.74")
        self.assertEqual((agent.version(), agent._cline_version), ("3.0.74", "3.0.74"))
        same = PinnedClineCli(logs_dir=self.tmpdir(), model_name=MODEL, version="3.0.74", cline_version="3.0.74")
        self.assertEqual(same._cline_version, "3.0.74")
        with self.assertRaises(ValueError):
            PinnedClineCli(logs_dir=self.tmpdir(), model_name=MODEL, version="3.0.74", cline_version="3.0.73")
        self.assertIsNone(PinnedClineCli(logs_dir=Path(self.tmpdir()), model_name=MODEL)._cline_version)
