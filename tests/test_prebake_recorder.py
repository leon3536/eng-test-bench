"""Pre-bake recipes recorded from each harness's own Harbor setup(), and the
Dockerfiles that replay them.

Probe answers are canned from real `docker run --network none` probes of three
local task images (tests/fixtures/prebake/probes.json): fix-git (Debian 12),
regex-log (Ubuntu 24.04) and multi-source-data-merger (Debian 13). So no test
needs Docker. tests/fixtures/prebake/golden holds the Dockerfiles rendered
from them. For fix-git, each prototype Dockerfile the research probes built
and proved (tests/fixtures/prebake/prototypes) is reproduced instruction for
instruction, except for the differences documented in DIFFERENCES.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import shutil
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from tests.support import FIXTURES, BenchTestCase, completed

from bench import extract, prebake, settings
from bench.environments import prebaked
from bench.prebake import adapters, dockerfile, recorder
from bench.prebake.recorder import RecipeError, Recording, Step, Upload

PREBAKE = FIXTURES / "prebake"
MODEL = "anthropic/claude-opus-5-5"
TASKS = ("fix-git", "regex-log", "multi-source-data-merger")
# The five arms as config/bench.yaml resolves them (the 2026-10-29 pilot's kwargs).
ARMS: dict[str, tuple[str, dict[str, Any]]] = {
    "claude-code": ("claude-code", {"reasoning_effort": "medium", "max_budget_usd": "15", "max_turns": 300,
                                    "version": "2.1.312"}),
    "terminus-2": ("terminus-2", {"reasoning_effort": "medium", "max_turns": 300}),
    "mini-swe-agent": ("bench.agents.mini_swe_agent:MiniSweAgentWithMcp",
                       {"reasoning_effort": "medium", "cost_limit": "15", "version": "2.4.6"}),
    "pi": ("pi", {"reasoning_effort": "medium", "max_turns": 300, "version": "1.4.0"}),
    "strands": ("strands", {"factory": "bench_strands_factory:create_agent",
                            "factory_kwargs": {"model_spec": MODEL, "effort": "medium"},
                            "factory_path": "harnesses/strands_factory", "version": "1.57.2"}),
}
UV_SIGNATURE = "5174ef23451a966e03570c57857eb7b78d28bdfbbbdc53ce17411d42fd9c2889"
CRLF_FIX = ("find /installed-agent/strands/factory -type f \\( -name '*.sh' -o -name '*.py' -o -name '*.ps1' "
            "-o -name '*.cmd' -o -name '*.bat' \\) -exec sed -i 's/\\r$//' {} \\;")

# Where a rendered Dockerfile may differ from the prototype it reproduces:
# (prototype instructions it drops, rendered instructions it adds, why). Every
# prefix must match exactly one instruction; the rest must be equal, in order.
_MKDIR = 'RUN ["bash", "-c", "[ -d /installed-agent ] || mkdir -p /installed-agent"]'
_APT = 'RUN ["env", "DEBIAN_FRONTEND=noninteractive", "bash", "-c", "set -o pipefail; apt-get update && '
DIFFERENCES: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    "claude-code": (
        ("LABEL ", 'RUN ["bash", "-c", "set -o pipefail; pkgs=(); for pair in curl:curl'),
        (_MKDIR, _APT + 'apt-get install -y curl nodejs npm procps"]'),
        "Labels go on the build (--label), not in the Dockerfile. The prototype's in-image `command -v` loop "
        "becomes Harbor's own apt command for the packages this image lacks, found by probing it, with "
        "DEBIAN_FRONTEND passed per exec as Harbor does. /installed-agent is baked too, as for every harness."),
    "pi": (
        ('RUN ["bash", "-c", "command -v curl >/dev/null 2>&1 || {',),
        (_APT + 'apt-get install -y curl"]',),
        "The in-image curl check becomes Harbor's apt command, since fix-git lacks curl."),
    "strands": (
        ('RUN ["bash", "-c", "set -o pipefail; missing=(); for dep in curl git;',),
        (_APT + 'apt-get install -y curl"]',),
        "The in-image check becomes Harbor's apt command for curl (fix-git has git)."),
    "terminus-2": (
        ('RUN ["bash", "-c", "need=\'\'; tmux -V',),
        ('RUN ["bash", "-c", "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive '
         'apt-get install -y tmux asciinema"]',),
        "The in-image check becomes the exact install command TmuxSession issues on this image."),
    "mini-swe-agent": ((), (), "None: the prototype was generated the same way."),
}


def arm_adapter(family: str, **changes: Any) -> adapters.Adapter:
    agent, kwargs = ARMS[family]
    adapter = adapters.adapter_for(agent, family, {**kwargs, **changes}, MODEL)
    assert adapter is not None, family
    return adapter


class CannedProbes:
    """recorder.ProbeRunner over the answers recorded from the real images."""

    def __init__(self, **image_changes: Any) -> None:
        self.data = json.loads((PREBAKE / "probes.json").read_text(encoding="utf-8"))
        self.image_changes = image_changes
        self.asked: list[str] = []

    def image(self, image: str) -> dict[str, str]:
        if image not in self.data:
            raise RecipeError(f"{image} is not local (recording never pulls)")
        return {**self.data[image]["image"], **self.image_changes}

    def run(self, image: str, command: str) -> tuple[int, str]:
        self.asked.append(command)
        answers = self.data[image]["answers"]
        if command not in answers:
            raise AssertionError(f"a probe the fixture does not have: {command}")
        rc, out = answers[command]
        return int(rc), str(out)


def instructions(text: str) -> list[str]:
    """Dockerfile instructions with continuation lines joined, comments dropped."""
    out: list[str] = []
    pending = ""
    for line in text.splitlines():
        if not pending and (not line.strip() or line.lstrip().startswith("#")):
            continue
        if line.rstrip().endswith("\\") and not line.rstrip().endswith("]"):
            pending += line.rstrip()[:-1].strip() + " "
            continue
        out.append((pending + line.strip()).strip())
        pending = ""
    return out


class RecorderTestCase(BenchTestCase):
    def record(self, family: str, task: str, probes: CannedProbes | None = None, **changes: Any) -> Recording:
        return recorder.record_for_image(arm_adapter(family, **changes), f"alexgshaw/{task}:20251031",
                                         self.tmpdir(), probes or CannedProbes())


class GoldenDockerfiles(RecorderTestCase):
    def test_rendered_dockerfiles_equal_the_golden_copies(self) -> None:
        for family in ARMS:
            for task in TASKS:
                with self.subTest(family=family, task=task):
                    text, _ = dockerfile.render(self.record(family, task))
                    golden = (PREBAKE / "golden" / f"{family}__{task}.Dockerfile").read_text(encoding="utf-8")
                    self.assertEqual(text, golden)

    def test_fix_git_reproduces_each_prototype_except_documented_differences(self) -> None:
        for family, (dropped, added, why) in DIFFERENCES.items():
            with self.subTest(family=family):
                self.assertTrue(why)
                proto = instructions((PREBAKE / "prototypes" / f"{family}.Dockerfile").read_text(encoding="utf-8"))
                rendered = instructions(dockerfile.render(self.record(family, "fix-git"))[0])
                for prefixes, lines in ((dropped, proto), (added, rendered)):
                    for prefix in prefixes:
                        self.assertEqual(len([x for x in lines if x.startswith(prefix)]), 1, prefix)
                self.assertEqual([x for x in proto if not x.startswith(dropped or ("\0",))],
                                 [x for x in rendered if not x.startswith(added or ("\0",))])

    def test_run_strings_replay_harbors_execs_unchanged(self) -> None:
        for family in ARMS:
            rec = self.record(family, "fix-git")
            commands = [s.command for s in rec.install_steps if isinstance(s, Step)]
            text, _ = dockerfile.render(rec)
            runs = [json.loads(x[len("RUN "):]) for x in instructions(text) if x.startswith("RUN ")]
            with self.subTest(family=family):
                self.assertEqual([argv[-1] for argv in runs], commands)
                for argv in runs:
                    self.assertEqual(argv[-2:-1], ["-c"])
                    self.assertIn(argv[0], ("bash", "env"))
        # The first exec is Harbor's own, from BaseInstalledAgent.setup().
        from harbor.agents.installed.base import BaseInstalledAgent

        self.assertIn('"[ -d /installed-agent ] || mkdir -p /installed-agent"',
                      inspect.getsource(BaseInstalledAgent.setup))
        # pi-warm adds its clean-up inside the same RUN, after Harbor's command.
        pi = next(s for s in self.record("pi", "fix-git").install_steps if "npm install -g" in s.command)
        self.assertTrue(pi.command.endswith('test ! -e "$P/bin/pi"'))
        self.assertIn(" && pi --version && P=\"$(npm config get prefix)\" && rm -rf "
                      "\"$P/lib/node_modules/@earendil-works\" \"$P/bin/pi\"", pi.command)

    def test_no_forbidden_instruction_is_emitted(self) -> None:
        for path in sorted((PREBAKE / "golden").glob("*.Dockerfile")):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertEqual(dockerfile.check(text), [])
                for word in ("--no-install-recommends", "apt-get clean", "rm -rf /var/lib/apt/lists", "LABEL ",
                             "ENV ", "USER ", "WORKDIR ", "SHELL ", "CMD ", "ENTRYPOINT "):
                    self.assertNotIn(word, text)
                self.assertEqual(extract.redact(text), text)
        self.assertEqual(len(dockerfile.check('# syntax=docker/dockerfile:1\nFROM x\nENV A=1\nUSER app\n'
                                              'LABEL a=b\nWORKDIR /x\nRUN true\n')), 5)

    def test_strands_stages_the_runner_and_factory_as_build_context(self) -> None:
        rec = self.record("strands", "fix-git")
        uploads = [s for s in rec.install_steps if isinstance(s, Upload)]
        self.assertEqual([(u.target, u.is_dir) for u in uploads],
                         [("/installed-agent/strands/strands_runner.py", False),
                          ("/installed-agent/strands/factory", True)])
        _, context = dockerfile.render(rec)
        self.assertEqual(sorted(context), ["source", "strands_runner.py"])
        self.assertTrue(context["strands_runner.py"].is_file())
        self.assertTrue((context["source"] / "pyproject.toml").is_file())
        self.assertFalse(any(p.name in (".env", ".venv", "__pycache__") for p in context["source"].rglob("*")))
        if sys.platform == "win32":
            # Harbor writes the runner in text mode, so CRLF on Windows; the bake uploads those bytes.
            self.assertIn(b"\r\n", context["strands_runner.py"].read_bytes())
            self.assertIn(CRLF_FIX, [s.command for s in rec.install_steps if isinstance(s, Step)])

    def test_crlf_fix_is_harbors_own_string(self) -> None:
        from harbor.environments.docker.docker_unix import UnixOps

        source = inspect.getsource(UnixOps.upload_dir)
        for piece in ("-type f \\\\( -name '*.sh' -o -name '*.py' ", "-o -name '*.ps1' -o -name '*.cmd' "
                      "-o -name '*.bat' \\\\) ", "-exec sed -i 's/\\\\r$//' {} \\\\;"):
            self.assertIn(piece, source)

    def test_terminus_2_is_a_noop_where_tmux_and_asciinema_exist(self) -> None:
        self.assertTrue(self.record("terminus-2", "multi-source-data-merger").noop)
        self.assertFalse(self.record("terminus-2", "fix-git").noop)
        text, context = dockerfile.render(self.record("terminus-2", "multi-source-data-merger"))
        self.assertEqual((instructions(text), context), (["ARG BASE", "FROM ${BASE}"], {}))

    def test_what_each_image_needs_comes_from_probing_it(self) -> None:
        def apt(family: str, task: str) -> list[str]:
            return [s.command for s in self.record(family, task).install_steps
                    if isinstance(s, Step) and "apt-get install" in s.command]

        # Ubuntu 24.04 has procps; mini-swe-agent's fix-git has git.
        self.assertEqual(apt("claude-code", "regex-log"),
                         ["set -o pipefail; apt-get update && apt-get install -y curl nodejs npm"])
        self.assertEqual(apt("mini-swe-agent", "fix-git"),
                         ["set -o pipefail; apt-get update && apt-get install -y curl build-essential"])
        self.assertEqual(apt("mini-swe-agent", "regex-log"),
                         ["set -o pipefail; apt-get update && apt-get install -y curl build-essential git"])
        rec = self.record("terminus-2", "fix-git", record_terminal_session=False)
        self.assertEqual([s.command for s in rec.install_steps],
                         ["DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive "
                          "apt-get install -y tmux"])


class Recipes(RecorderTestCase):
    def recipe(self, family: str, **changes: Any) -> recorder.Recipe:
        return recorder.arm_recipe(arm_adapter(family, **changes))

    def test_recipes_need_no_docker_and_are_stable(self) -> None:
        for family in ARMS:
            with self.subTest(family=family):
                first = self.recipe(family)
                self.assertRegex(first.sha256, r"^[0-9a-f]{64}$")
                recorder._recipes.clear()
                self.assertEqual(self.recipe(family).sha256, first.sha256)
                self.assertEqual(first.document["host_platform"], sys.platform)
                self.assertEqual(first.harbor_version, "0.25.0")

    def test_options_install_never_reads_keep_the_recipe(self) -> None:
        cases = {
            "claude-code": [{"reasoning_effort": "high"}, {"max_budget_usd": "2"}, {"max_turns": 50}],
            "mini-swe-agent": [{"cost_limit": "1"}, {"reasoning_effort": "low"}],
            "pi": [{"max_turns": 20}],
            "terminus-2": [{"max_turns": 10}, {"temperature": 0.5}],
            "strands": [{"factory_kwargs": {"model_spec": MODEL, "effort": "high"}}],
        }
        for family, changes in cases.items():
            base = self.recipe(family).sha256
            for change in changes:
                with self.subTest(family=family, change=change):
                    self.assertEqual(self.recipe(family, **change).sha256, base)
        claude = arm_adapter("claude-code")
        other_model = adapters.adapter_for("claude-code", "claude-code", claude.kwargs, "anthropic/claude-haiku-5-5")
        self.assertEqual(recorder.arm_recipe(other_model).sha256, self.recipe("claude-code").sha256)  # type: ignore

    def test_what_install_reads_changes_the_recipe(self) -> None:
        self.assertNotEqual(self.recipe("claude-code", version="2.1.313").sha256, self.recipe("claude-code").sha256)
        self.assertNotEqual(self.recipe("pi", version="1.4.1").sha256, self.recipe("pi").sha256)
        self.assertNotEqual(self.recipe("terminus-2", record_terminal_session=False).sha256,
                            self.recipe("terminus-2").sha256)
        factory = self.tmpdir() / "factory"
        shutil.copytree(settings.ROOT / "harnesses" / "strands_factory", factory,
                        ignore=shutil.ignore_patterns("__pycache__", ".venv", "*.egg-info"))
        before = self.recipe("strands", factory_path=str(factory)).sha256
        target = next(p for p in sorted(factory.rglob("*.py")))
        target.write_text(target.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
        self.assertNotEqual(self.recipe("strands", factory_path=str(factory)).sha256, before)
        with mock.patch.object(recorder, "harbor_version", lambda: "0.26.0"):
            upgraded = self.recipe("claude-code").sha256
        self.assertNotEqual(upgraded, self.recipe("claude-code").sha256)

    def test_only_strands_skips_and_it_skips_the_uv_step(self) -> None:
        for family in ARMS:
            with self.subTest(family=family):
                skips = self.recipe(family).skip_execs
                self.assertEqual(skips, (UV_SIGNATURE,) if family == "strands" else ())
        strands = self.recipe("strands")
        skipped = [s for s in strands.baked if isinstance(s, Step) and s.signature == UV_SIGNATURE]
        self.assertEqual(len(skipped), 1)
        self.assertIn("uv venv /installed-agent/strands/venv --clear", skipped[0].command)
        self.assertEqual(strands.skip_commands, (prebaked.command_digest(skipped[0].command),))

    def test_claude_code_short_circuits_on_the_image(self) -> None:
        baked = self.recipe("claude-code").baked
        self.assertEqual([(s.command, s.probe) for s in baked], [
            ("[ -d /installed-agent ] || mkdir -p /installed-agent", None),
            ('export PATH="$HOME/.local/bin:$PATH"; claude --version', "claude-version")])
        bare = [s.command for s in self.recipe("claude-code").bare if not s.probe]  # type: ignore[union-attr]
        self.assertIn("set -o pipefail; apt-get update && apt-get install -y curl bash nodejs npm procps", bare)

    def test_the_recipe_holds_no_secret(self) -> None:
        for family in ARMS:
            text = json.dumps(self.recipe(family).document)
            self.assertEqual(extract.redact(text), text)


class Adapters(RecorderTestCase):
    def test_each_shipped_harness_has_its_adapter(self) -> None:
        expected = {"claude-code": ("plain", 10.0), "terminus-2": ("plain", 15.0), "mini-swe-agent": ("plain", 20.0),
                    "pi": ("pi-warm", 60.0), "strands": ("strands-skip", 15.0)}
        for family, (variant, budget) in expected.items():
            with self.subTest(family=family):
                adapter = arm_adapter(family)
                self.assertEqual((adapter.variant, adapter.setup_budget_sec), (variant, budget))
                self.assertTrue(adapter.smoke)
                self.assertGreater(adapter.expected_gb, 0)
        self.assertEqual(arm_adapter("terminus-2").version, "0.25.0")
        self.assertEqual(arm_adapter("mini-swe-agent").version, "2.4.6")

    def test_unknown_or_mismatched_harnesses_have_none(self) -> None:
        self.assertIsNone(adapters.adapter_for("aider", "aider", {"version": "1"}, MODEL))
        self.assertIsNone(adapters.adapter_for("pi", "claude-code", {"version": "1"}, MODEL))
        self.assertIsNone(adapters.adapter_for("not.a.module:Agent", "mini-swe-agent", {}, MODEL))
        self.assertIsNone(adapters.adapter_for("claude-code", "claude-code", {}, MODEL).version)  # type: ignore

    def test_resolve_explains_an_arm_that_cannot_be_prebaked(self) -> None:
        cfg = self.load_shipped_config("bench")
        arm = cfg.arm("claude-code")
        unpinned = prebake.resolve(cfg, cfg.run, arm, {"reasoning_effort": "medium"})
        assert unpinned is not None
        self.assertEqual((unpinned.supported, unpinned.env_kwargs()), (False, {}))
        self.assertIn("no pinned version", unpinned.reason)
        ok = prebake.resolve(cfg, cfg.run, arm, ARMS["claude-code"][1])
        assert ok is not None
        self.assertEqual(ok.env_kwargs(), {"prebake_harness": "claude-code", "prebake_version": "2.1.312",
                                           "prebake_recipe": self.recipe_of("claude-code"),
                                           "prebake_prefix": "benchbake", "prebake_required": False})
        broken = dict(ARMS["strands"][1], factory_path="no/such/folder")
        bad = prebake.resolve(cfg, cfg.run, cfg.arm("strands"), broken)
        assert bad is not None
        self.assertFalse(bad.supported)
        self.assertIn("could not be recorded", bad.reason)
        # An import path the config does not allow is never imported into the bench.
        foreign = dataclasses.replace(cfg.arm("mini-swe-agent"), agent="bench.agents.other:Agent")
        with mock.patch.object(adapters, "adapter_for") as adapter_for:
            unknown = prebake.resolve(cfg, cfg.run, foreign, {"version": "1"})
        adapter_for.assert_not_called()
        assert unknown is not None
        self.assertIn("is not an allowed harness import path", unknown.reason)
        self.assertIsNone(prebake.resolve(cfg, dataclasses.replace(cfg.run, prebake="off"), arm, {}))

    def recipe_of(self, family: str) -> str:
        return recorder.arm_recipe(arm_adapter(family)).sha256


class RecipeErrors(RecorderTestCase):
    def test_non_root_or_non_linux_images_are_refused(self) -> None:
        for change, message in (({"User": "app"}, "runs as user 'app'"), ({"Os": "windows"}, "windows image")):
            with self.subTest(change=change), self.assertRaisesRegex(RecipeError, message):
                self.record("claude-code", "fix-git", CannedProbes(**change))
        with self.assertRaisesRegex(RecipeError, "is not local"):
            recorder.record_for_image(arm_adapter("pi"), "alexgshaw/elsewhere:1", self.tmpdir(), CannedProbes())

    def test_a_non_apt_image_is_refused(self) -> None:
        probes = CannedProbes()
        image = "alexgshaw/fix-git:20251031"
        loop = next(c for c in probes.data[image]["answers"] if c.startswith("for manager in"))
        probes.data[image]["answers"][loop] = [0, "apk"]
        with self.assertRaisesRegex(RecipeError, "not apt-based"):
            self.record("pi", "fix-git", probes)

    def test_a_non_root_install_step_or_unknown_probe_raises(self) -> None:
        class NonRoot(adapters.Adapter):
            family = "x"

            async def record(self, env: Any, logs_dir: Path) -> None:
                await env.exec("make install", user="builder")

        class Unanswered(adapters.Adapter):
            family = "y"

            def bare_answer(self, name: str, command: str) -> tuple[int, str] | None:
                return None

            async def record(self, env: Any, logs_dir: Path) -> None:
                await env.exec("command -v curl >/dev/null 2>&1", user="root")

        with self.assertRaisesRegex(RecipeError, "runs as user 'builder'"):
            recorder.arm_recipe(NonRoot("x", {"version": "1"}, MODEL))
        with self.assertRaisesRegex(RecipeError, "unknown probe 'command-v'"):
            recorder.arm_recipe(Unanswered("y", {"version": "1"}, MODEL))

    def test_a_credential_is_never_written_to_a_dockerfile(self) -> None:
        for step in (Step("apt-get install -y x", env={"ANTHROPIC_API_KEY": "sk-ant-api03-abcdefghijklmnop"}),
                     Step("curl -H 'Authorization: Bearer abcdefghijklmnop' https://x"),
                     Step("make", cwd="/src"), Step("make", user="app")):
            with self.subTest(step=step.command):
                with self.assertRaises(RecipeError):
                    dockerfile.render(Recording("x", "1", "plain", "img:1", (step,)))


class DockerProbeRunner(BenchTestCase):
    def test_probes_run_offline_without_pulling_or_credentials(self) -> None:
        calls: list[tuple[list[str], dict[str, str]]] = []
        inspect_json = json.dumps([{"Id": "sha256:abc", "Os": "linux", "Config": {"Env": []}}])

        def docker(argv: list[str], **kw: Any) -> Any:
            calls.append((argv, kw["env"]))
            if argv[1:3] == ["image", "inspect"]:
                return completed(argv, stdout=inspect_json)
            return completed(argv, returncode=0 if argv[-1] in ("true", "command -v bash") else 127)

        self.allow_subprocess(lambda argv: argv[:1] == ["docker"], docker)
        probes = recorder.DockerProbes()
        self.assertEqual(probes.image("alexgshaw/x:1"), {"Id": "sha256:abc", "Os": "linux", "User": ""})
        self.assertEqual(probes.run("alexgshaw/x:1", "tmux -V"), (127, ""))
        self.assertEqual(probes.run("alexgshaw/x:1", "tmux -V"), (127, ""))  # cached
        runs = [argv for argv, _ in calls if argv[1] == "run"]
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[1][1:12], ["run", "--rm", "--pull", "never", "--network", "none", "-u", "root",
                                         "--entrypoint", "bash", "alexgshaw/x:1"])
        for _, env in calls:
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("OPENROUTER_API_KEY", env)

    def test_a_missing_image_or_a_docker_error_is_a_recipe_error(self) -> None:
        self.allow_subprocess(lambda argv: argv[:1] == ["docker"],
                              lambda argv, **kw: completed(argv, returncode=1 if "inspect" in argv else 125,
                                                           stderr="Error: No such image"))
        with self.assertRaisesRegex(RecipeError, "is not local"):
            recorder.DockerProbes().image("alexgshaw/x:1")
        with self.assertRaisesRegex(RecipeError, "could not run a probe"):
            recorder.DockerProbes().run("alexgshaw/x:1", "tmux -V")


if __name__ == "__main__":
    unittest.main()
