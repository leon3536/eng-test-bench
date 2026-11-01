"""Pre-bake adapters for Cline CLI, OpenCode, Kilo Code and Goose, and the Dockerfiles
their recordings render.

Their goldens (tests/fixtures/prebake/golden/<family>__bare.Dockerfile) are
rendered from the bare-image recording, the full stock install, because the
canned probes of the three task images (probes.json) do not answer the
`command -v bzip2` and `command -v stdbuf` checks these installs add, and the
task images are not probed again for tests. No image has been built from them.
"""

from __future__ import annotations

import json
from typing import Any

from tests.support import FIXTURES, BenchTestCase

from bench.prebake import adapters, dockerfile, recorder
from bench.prebake.recorder import Recording, Step

GOLDEN = FIXTURES / "prebake" / "golden"
MODEL = "anthropic/claude-opus-5-5"
# The arms as the bench.yaml snippets resolve them.
ARMS: dict[str, tuple[str, dict[str, Any]]] = {
    "cline-cli": ("bench.agents.cline:PinnedClineCli", {"reasoning_effort": "medium", "version": "3.0.74"}),
    "opencode": ("bench.agents.opencode:OpenCodeBunEpipe", {"variant": "medium", "version": "1.19.5"}),
    "kilo-code": ("bench.agents.kilo:KiloCode", {"variant": "medium", "version": "7.8.18"}),
    "goose": ("bench.agents.goose:PinnedGoose", {"reasoning_effort": "medium", "max_turns": 300,
                                                 "version": "1.57.0"}),
}


def adapter(family: str, agent: str | None = None, **changes: Any) -> adapters.Adapter | None:
    path, kwargs = ARMS[family]
    return adapters.adapter_for(agent or path, family, {**kwargs, **changes}, MODEL)


def bare_dockerfile(family: str) -> str:
    a = adapter(family)
    assert a is not None
    recipe = recorder.arm_recipe(a)
    return dockerfile.render(Recording(a.family, a.version, a.variant, None, recipe.bare))[0]


def install_commands(steps: tuple[Any, ...]) -> list[str]:
    return [s.command for s in steps if isinstance(s, Step) and not s.probe]


class NewHarnessAdapters(BenchTestCase):
    def test_rendered_dockerfiles_equal_the_golden_copies(self) -> None:
        for family in ARMS:
            with self.subTest(family=family):
                text = bare_dockerfile(family)
                self.assertEqual(text, (GOLDEN / f"{family}__bare.Dockerfile").read_text(encoding="utf-8"))
                self.assertEqual(dockerfile.check(text), [])

    def test_each_has_its_variant_and_a_smoke_check(self) -> None:
        expected = {"cline-cli": "cline-warm", "opencode": "opencode-warm", "kilo-code": "kilo-warm",
                    "goose": "goose-skip"}
        for family, variant in expected.items():
            with self.subTest(family=family):
                a = adapter(family)
                assert a is not None
                self.assertEqual((a.variant, a.version), (variant, ARMS[family][1]["version"]))
                self.assertTrue(a.smoke)
        self.assertIn("! command -v cline", adapter("cline-cli").smoke)  # type: ignore[union-attr]
        self.assertTrue(adapter("kilo-code").smoke.endswith(  # type: ignore[union-attr]
            "&& ! command -v kilo && ! command -v kilocode && command -v stdbuf >/dev/null"))
        self.assertIn(r'goose --version | grep -q "1\.57\.0"', adapter("goose").smoke)  # type: ignore[union-attr]

    def test_the_npm_harnesses_unbake_their_global_package_in_the_same_run(self) -> None:
        for family, package, removed, binaries in (("cline-cli", "cline", "cline", ("cline",)),
                                                   ("opencode", "opencode-ai", "opencode-ai", ("opencode",)),
                                                   ("kilo-code", "@kilocode/cli", "@kilocode", ("kilo", "kilocode"))):
            with self.subTest(family=family):
                a = adapter(family)
                assert a is not None
                recipe = recorder.arm_recipe(a)
                (step,) = [c for c in install_commands(recipe.bare) if f" {package}@" in c]
                self.assertTrue(step.endswith(" && ".join(f'test ! -e "$P/bin/{b}"' for b in binaries)))
                links = " ".join(f'"$P/bin/{b}"' for b in binaries)
                self.assertIn(f'rm -rf "$P/lib/node_modules/{removed}" {links} "$HOME/.npm/_logs"', step)
                # A trial on the image runs Harbor's own install unchanged, and skips nothing.
                (baked,) = [c for c in install_commands(recipe.baked) if f" {package}@" in c]
                self.assertNotIn(f'test ! -e "$P/bin/{binaries[0]}"', baked)
                self.assertEqual(recipe.skip_execs, ())

    def test_kilos_estimates_allow_for_its_larger_install_than_opencodes(self) -> None:
        # Kilo's two native packages are ~1.5x OpenCode's and hold 419 files each against 2, so its unmeasured
        # sizes and timings must not be OpenCode's copied ones; the budget stays within Harbor's 360 s setup limit.
        kilo, opencode = adapters.KiloAdapter, adapters.OpenCodeAdapter
        for name in ("setup_budget_sec", "expected_gb", "expected_build_min", "stock_setup_sec"):
            with self.subTest(name=name):
                self.assertGreater(getattr(kilo, name), getattr(opencode, name))
        self.assertLess(kilo.setup_budget_sec, 360.0)

    def test_kilo_and_opencode_each_take_only_their_own_agent(self) -> None:
        # KiloCode subclasses OpenCode, so the OpenCode adapter must not claim it.
        self.assertIsNone(adapter("opencode", agent="bench.agents.kilo:KiloCode"))
        self.assertIsNone(adapter("kilo-code", agent="bench.agents.opencode:OpenCodeBunEpipe"))
        self.assertIsInstance(adapter("kilo-code"), adapters.KiloAdapter)
        self.assertIs(adapters.ADAPTERS["kilo-code"], adapters.KiloAdapter)

    def test_goose_skips_only_its_always_apt_step_and_the_pinned_download(self) -> None:
        a = adapter("goose")
        assert a is not None
        recipe = recorder.arm_recipe(a)
        skipped = [s for s in recipe.baked if isinstance(s, Step) and s.signature in recipe.skip_execs]
        self.assertEqual(len(recipe.skip_execs), 2)
        self.assertEqual(skipped[0].command, "set -o pipefail; apt-get update && apt-get install -y libxcb1 libgomp1")
        self.assertIn("releases/download/v1.57.0/download_cli.sh | bash", skipped[1].command)
        self.assertEqual(skipped[1].env["GOOSE_VERSION"], "v1.57.0")

    def test_unpinnable_or_unpinned_arms_are_unsupported(self) -> None:
        # Harbor's own Goose installs the latest stable release whatever its version says.
        self.assertIsNone(adapter("goose", agent="goose").version)  # type: ignore[union-attr]
        # Harbor's own cline-cli installs cline_version (else nightly), not version.
        self.assertIsNone(adapter("cline-cli", agent="cline-cli").version)  # type: ignore[union-attr]
        self.assertEqual(adapter("cline-cli", agent="cline-cli", cline_version="3.0.74").version,  # type: ignore
                         "3.0.74")
        self.assertIsNone(adapter("cline-cli", cline_version="3.0.73").version)  # type: ignore[union-attr]
        self.assertIsNone(adapter("cline-cli", github_user="me").version)  # type: ignore[union-attr]
        self.assertIsNone(adapter("opencode", version=None).version)  # type: ignore[union-attr]
        self.assertIsNone(adapter("kilo-code", version="latest").version)  # type: ignore[union-attr]

    def test_clines_github_token_never_reaches_the_recipe(self) -> None:
        with self.patch_env(GITHUB_TOKEN="ghp_" + "a" * 36, GH_TOKEN="gho_" + "b" * 36):
            a = adapter("cline-cli")
            assert a is not None
            recipe = recorder.arm_recipe(a)
        text = json.dumps(recipe.document)
        self.assertNotIn("GITHUB_TOKEN", text)
        self.assertNotIn("GH_TOKEN", text)
        self.assertEqual(bare_dockerfile("cline-cli"), (GOLDEN / "cline-cli__bare.Dockerfile").read_text("utf-8"))

    def test_clines_record_also_runs_on_a_live_container(self) -> None:
        # The equivalence check runs the stock install through ExecShim, which keeps no recording of steps.
        import asyncio
        import tempfile
        from pathlib import Path

        a = adapter("cline-cli")
        assert a is not None
        with tempfile.TemporaryDirectory() as tmp:
            env = recorder.BareEnv(a, Path(tmp) / "stage")

            class Live:  # everything BareEnv answers, except its steps
                def __getattr__(self, name: str) -> Any:
                    if name == "steps":
                        raise AttributeError(name)
                    return getattr(env, name)

            asyncio.run(a.record(Live(), Path(tmp) / "logs"))
            self.assertTrue(install_commands(tuple(env.steps)))

    def patch_env(self, **values: str) -> Any:
        from unittest import mock
        import os

        recorder._recipes.clear()
        self.addCleanup(recorder._recipes.clear)
        return mock.patch.dict(os.environ, values)
