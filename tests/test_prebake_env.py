"""The Harbor environment class that runs a trial on a pre-baked image.

No Docker: image inspection is faked per test, and Harbor's own start and
exec are replaced by recorders, so only the class's decisions run.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from tests.support import BenchTestCase

from bench import classify, extract, settings
from bench.environments import prebaked
from bench.environments.prebaked import (
    LABEL_NS,
    PREBAKED_ENV,
    Decision,
    ImageInfo,
    Want,
    bake_labels,
    check_candidate,
    exec_signature,
    prebaked_image_name,
)

RECIPE = "419cd264886607a31cb532a4518020dff22a4e1639b42c2a626fd03f6ce18b2b"
TASK_IMAGE = "alexgshaw/fix-git:20251031"
# strands' uv venv/pip step as Harbor 0.25.0 issues it, and the signature the
# research prototype put in its image's skip-execs label for it.
UV_STEP = ("set -o pipefail; set -euo pipefail; curl -LsSf https://astral.sh/uv/install.sh | sh; "
           "if [ -f \"$HOME/.local/bin/env\" ]; then . \"$HOME/.local/bin/env\"; fi; uv python install 3.12; "
           "uv venv /installed-agent/strands/venv --clear --python 3.12; uv pip install --python "
           "/installed-agent/strands/venv/bin/python 'strands-agents[all]==1.57.2' -e /installed-agent/strands/factory")
UV_SIGNATURE = "5174ef23451a966e03570c57857eb7b78d28bdfbbbdc53ce17411d42fd9c2889"


def want(**changes: Any) -> Want:
    values = dict(harness="claude-code", version="2.1.312", recipe=RECIPE, task_image=TASK_IMAGE,
                  harbor_version="0.25.0")
    values.update(changes)
    return Want(**values)  # type: ignore[arg-type]


BASE = ImageInfo(id="sha256:base", os="linux", labels={"com.docker.compose.project": "harbor-build-fix-git"},
                 layers=("sha256:l1", "sha256:l2"))


def baked(w: Want | None = None, *, layers: tuple[str, ...] = ("sha256:l1", "sha256:l2", "sha256:l3"),
          os_name: str = "linux", noop: bool = False, skips: tuple[str, ...] = (),
          skip_commands: tuple[str, ...] = (), **label_changes: str) -> ImageInfo:
    labels = {**BASE.labels, **bake_labels(w or want(), base_image_id=BASE.id, baked_at="2026-10-29T22:51:09Z",
                                           dockerfile_sha256="d" * 64, skip_execs=skips,
                                           skip_commands=skip_commands, noop=noop)}
    labels.update({LABEL_NS + k: v for k, v in label_changes.items()})
    return ImageInfo(id="sha256:baked", os=os_name, labels=labels, layers=layers)


class Preflight(unittest.TestCase):
    """Harbor's Docker check with a longer limit and retries (a busy daemon is not a stopped one)."""

    def run_preflight(self, outcomes: list[Any]) -> tuple[Any, list[float]]:
        import subprocess

        calls: list[dict[str, Any]] = []
        sleeps: list[float] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> Any:
            calls.append({"cmd": cmd, **kwargs})
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        with mock.patch("shutil.which", return_value="docker"), mock.patch.object(subprocess, "run", fake_run), \
                mock.patch("time.sleep", sleeps.append):
            try:
                prebaked.PrebakedDockerEnvironment.preflight()
                result: Any = None
            except SystemExit as exc:
                result = exc
        self.assertTrue(all(c["timeout"] == prebaked.PREFLIGHT_TIMEOUT_SEC and c["cmd"][-1] == "info" for c in calls))
        return result, sleeps

    def test_a_slow_docker_info_is_tried_again(self) -> None:
        import subprocess

        result, sleeps = self.run_preflight([subprocess.TimeoutExpired("docker info", 60), mock.Mock()])
        self.assertIsNone(result)
        self.assertEqual(sleeps, [0, 15])

    def test_docker_that_never_answers_still_fails_with_harbors_message(self) -> None:
        import subprocess

        result, sleeps = self.run_preflight([subprocess.CalledProcessError(1, "docker info")] * 3)
        self.assertIn("Docker daemon is not running", str(result))
        self.assertEqual(sleeps, [0, 15, 30])


class ImageNames(unittest.TestCase):
    def test_name_carries_family_version_recipe_and_task_image(self) -> None:
        self.assertEqual(prebaked_image_name("benchbake", "claude-code", "2.1.312", RECIPE, TASK_IMAGE),
                         "benchbake/claude-code-2.1.312-r419cd26488/alexgshaw/fix-git:20251031")
        self.assertEqual(want().image, "benchbake/claude-code-2.1.312-r419cd26488/alexgshaw/fix-git:20251031")
        self.assertEqual(prebaked_image_name("benchbake-test/", "terminus-2", "0.25.0", RECIPE, "alexgshaw/x"),
                         "benchbake-test/terminus-2-0.25.0-r419cd26488/alexgshaw/x")

    def test_names_outside_dockers_grammar_raise(self) -> None:
        for args in (("benchbake", "pi", "1.4.0-RC1", RECIPE, TASK_IMAGE),            # uppercase
                     ("benchbake", "pi", "1.4.0", RECIPE, "alexgshaw/x@sha256:" + "a" * 64),
                     ("benchbake", "pi", "1.4.0", RECIPE, "localhost:5000/x:1"),
                     ("benchbake", "pi", "1.4.0", "not-a-sha", TASK_IMAGE),
                     ("bench bake", "pi", "1.4.0", RECIPE, TASK_IMAGE)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                prebaked_image_name(*args)


class Signatures(unittest.TestCase):
    def test_the_strands_uv_step_matches_the_prototypes_label(self) -> None:
        self.assertEqual(exec_signature(UV_STEP, None, None, None), UV_SIGNATURE)
        self.assertEqual(exec_signature(UV_STEP, None, None, {}), UV_SIGNATURE)

    def test_user_cwd_and_env_are_part_of_the_signature(self) -> None:
        sigs = {exec_signature(UV_STEP, None, None, None), exec_signature(UV_STEP, "root", None, None),
                exec_signature(UV_STEP, None, "/app", None), exec_signature(UV_STEP, None, None, {"X": "1"})}
        self.assertEqual(len(sigs), 4)

    def test_credential_values_never_reach_a_signature(self) -> None:
        a = exec_signature("cmd", None, None, {"ANTHROPIC_API_KEY": "sk-ant-aaaaaaaaaaaaaaaa", "X": "1"})
        b = exec_signature("cmd", None, None, {"ANTHROPIC_API_KEY": "sk-ant-bbbbbbbbbbbbbbbb", "X": "1"})
        c = exec_signature("cmd", None, None, {"PLAIN": "sk-ant-aaaaaaaaaaaaaaaa"})
        d = exec_signature("cmd", None, None, {"PLAIN": "sk-ant-cccccccccccccccc"})
        self.assertEqual(a, b)
        self.assertEqual(c, d)
        self.assertEqual(prebaked.redacted_env({"GH_TOKEN": "x", "MODE": "y"}), {"GH_TOKEN": "<set>", "MODE": "y"})


class CheckCandidate(unittest.TestCase):
    def test_a_matching_image_is_used(self) -> None:
        d = check_candidate(baked(skips=(UV_SIGNATURE,)), BASE, want())
        self.assertTrue(d.used)
        self.assertEqual((d.status, d.image, d.image_id, d.task_image_id, d.recipe, d.added_layers, d.noop),
                         (prebaked.READY, want().image, "sha256:baked", "sha256:base", RECIPE, 1, False))
        self.assertEqual(d.skip_execs, (UV_SIGNATURE,))
        self.assertEqual(d.baked_at, "2026-10-29T22:51:09Z")

    def test_every_reason_not_to_use_it(self) -> None:
        cases = {
            "missing": (None, BASE, want(), prebaked.MISSING, "no pre-baked image"),
            "harness version": (baked(**{"harness-version": "2.1.311"}), BASE, want(), prebaked.STALE,
                                "harness-version"),
            "recipe": (baked(**{"recipe-sha256": "f" * 64}), BASE, want(), prebaked.STALE, "recipe-sha256"),
            "harbor upgrade": (baked(), BASE, want(harbor_version="0.26.0"), prebaked.STALE, "harbor-version"),
            "bake format": (baked(**{"bake-format": "0"}), BASE, want(), prebaked.STALE, "bake-format"),
            "base not local": (baked(), None, want(), prebaked.UNVERIFIABLE, "is not local"),
            "base re-pulled": (baked(**{"base-image-id": "sha256:old"}), BASE, want(), prebaked.STALE, "re-pulled"),
            "wrong base layers": (baked(layers=("sha256:x1", "sha256:l2", "sha256:l3")), BASE, want(),
                                  prebaked.STALE, "layer mismatch"),
            "nothing added": (baked(layers=BASE.layers), BASE, want(), prebaked.STALE, "layer mismatch"),
            "os": (baked(os_name="windows"), BASE, want(), prebaked.STALE, "OS windows"),
            "skip label": (baked(**{"skip-execs": "not json"}), BASE, want(), prebaked.STALE, "unreadable"),
            "skip digests": (baked(**{"skip-execs": '["abc"]'}), BASE, want(), prebaked.STALE, "unreadable"),
        }
        for case, (image, base, w, status, reason) in cases.items():
            with self.subTest(case=case):
                d = check_candidate(image, base, w)
                self.assertFalse(d.used)
                self.assertEqual(d.status, status)
                self.assertIn(reason, d.reason)

    def test_a_noop_image_has_exactly_the_task_images_layers(self) -> None:
        w = want(harness="terminus-2", version="0.25.0")
        d = check_candidate(baked(w, layers=BASE.layers, noop=True), BASE, w)
        self.assertEqual((d.used, d.noop, d.added_layers), (True, True, 0))
        d = check_candidate(baked(w, noop=True), BASE, w)
        self.assertEqual((d.used, d.status), (False, prebaked.STALE))

    def test_labels_round_trip_through_docker_inspect_json(self) -> None:
        image = baked(skips=(UV_SIGNATURE,))
        data = {"Id": image.id, "Os": "linux", "Config": {"Labels": dict(image.labels), "Env": ["PATH=/usr/bin"]},
                "RootFS": {"Type": "layers", "Layers": list(image.layers)}}
        self.assertEqual(ImageInfo.from_inspect(data), image)
        self.assertEqual(ImageInfo.from_inspect({"Id": "x", "Config": {"Labels": None}}).labels, {})

    def test_labels_hold_no_secret(self) -> None:
        labels = bake_labels(want(), base_image_id=BASE.id, baked_at="t", dockerfile_sha256="d" * 64,
                             skip_execs=(UV_SIGNATURE,))
        text = json.dumps(labels)
        self.assertEqual(extract.redact(text), text)
        self.assertTrue(all(k.startswith(LABEL_NS) for k in labels))


def _env_class_for(tmp: Path, session: str = "fix-git__abc__env", dockerfile: bool = False,
                   **kwargs: Any) -> prebaked.PrebakedDockerEnvironment:
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.task.config import EnvironmentConfig as TaskEnvironment
    from harbor.models.trial.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    env_dir = tmp / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    if dockerfile:
        (env_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    paths = TrialPaths(trial_dir=tmp / "trial")
    paths.mkdir()
    ek = {"prebake_harness": "claude-code", "prebake_version": "2.1.312", "prebake_recipe": RECIPE,
          "prebake_prefix": "benchbake", "prebake_required": False}
    ek.update(kwargs)
    config = EnvironmentConfig(import_path=PREBAKED_ENV, kwargs={k: v for k, v in ek.items() if v is not None})
    return EnvironmentFactory.create_environment_from_config(  # type: ignore[return-value]
        config=config, environment_dir=env_dir, environment_name="fix-git", session_id=session, trial_paths=paths,
        task_env_config=TaskEnvironment(docker_image=TASK_IMAGE), logger=logging.getLogger("test.prebake"))


class EnvironmentClass(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.images: dict[str, ImageInfo] = {}
        self.inspected: list[str] = []
        self.started: list[str] = []
        self.ran: list[str] = []
        case = self

        async def inspect_image(self_: Any, ref: str) -> ImageInfo | None:
            case.inspected.append(ref)
            return case.images.get(ref)

        async def start(self_: Any, force_build: bool) -> None:
            case.started.append(self_._env_vars.prebuilt_image_name)

        async def run(self_: Any, command: str, cwd: Any = None, env: Any = None, timeout_sec: Any = None,
                      user: Any = None) -> Any:
            from harbor.environments.base import ExecResult

            case.ran.append(command)
            return ExecResult(stdout="ran", stderr="", return_code=0)

        from harbor.environments.docker.docker import DockerEnvironment
        for target, name, value in ((prebaked.PrebakedDockerEnvironment, "_inspect", inspect_image),
                                    (DockerEnvironment, "start", start),
                                    (DockerEnvironment, "exec", run)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def env(self, **kwargs: Any) -> prebaked.PrebakedDockerEnvironment:
        return _env_class_for(self.tmpdir(), **kwargs)

    def record(self, env: prebaked.PrebakedDockerEnvironment) -> dict[str, Any]:
        return json.loads((env.trial_paths.trial_dir / prebaked.RECORD_NAME).read_text(encoding="utf-8"))

    def test_a_verified_image_replaces_the_task_image(self) -> None:
        self.images = {want().image: baked(), TASK_IMAGE: BASE}
        env = self.env()
        asyncio.run(env.start(force_build=False))
        self.assertEqual(self.started, [want().image])
        self.assertEqual(sorted(self.inspected), sorted([want().image, TASK_IMAGE]))
        record = self.record(env)
        self.assertEqual((record["used"], record["status"], record["ran_image"], record["image_id"]),
                         (True, "ready", want().image, "sha256:baked"))
        self.assertEqual((record["harness"], record["version"], record["wanted_recipe"], record["required"]),
                         ("claude-code", "2.1.312", RECIPE, False))
        self.assertEqual(record["task_image"], TASK_IMAGE)
        text = json.dumps(record)
        self.assertEqual(extract.redact(text), text)

    def test_a_missing_image_falls_back_to_stock_with_its_reason(self) -> None:
        self.images = {TASK_IMAGE: BASE}
        env = self.env()
        with self.assertLogs("test.prebake", "WARNING") as logs:
            asyncio.run(env.start(force_build=False))
        self.assertEqual(self.started, [TASK_IMAGE])
        record = self.record(env)
        self.assertEqual((record["used"], record["status"], record["ran_image"]), (False, "missing", TASK_IMAGE))
        self.assertIn("no pre-baked image", record["reason"])
        self.assertIn("stock setup", logs.output[0])

    def test_required_fails_before_any_container_starts(self) -> None:
        from harbor.environments.base import SandboxBuildFailedError

        self.images = {TASK_IMAGE: BASE}
        env = self.env(prebake_required=True)
        with self.assertRaisesRegex(SandboxBuildFailedError, "prebake required but not usable: no pre-baked image"):
            asyncio.run(env.start(force_build=False))
        self.assertEqual(self.started, [])
        self.assertEqual((self.record(env)["used"], self.record(env)["required"]), (False, True))
        # The ledger counts it as infra, not as the harness failing.
        self.assertEqual(classify.harbor_cause("SandboxBuildFailedError"), "sandbox_failed")

    def test_required_arrives_as_json_or_text(self) -> None:
        for value, expected in ((True, True), ("true", True), (False, False), ("false", False)):
            with self.subTest(value=value):
                self.assertIs(self.env(prebake_required=value)._prebake_required, expected)

    def test_only_the_agent_environment_decides(self) -> None:
        self.images = {want().image: baked(), TASK_IMAGE: BASE}
        env = self.env(session="fix-git__abc__verifier__main")
        asyncio.run(env.start(force_build=False))
        self.assertEqual((self.inspected, self.started), ([], [TASK_IMAGE]))
        self.assertFalse((env.trial_paths.trial_dir / prebaked.RECORD_NAME).exists())

    def test_force_build_and_missing_arguments_use_the_task_image(self) -> None:
        self.images = {want().image: baked(), TASK_IMAGE: BASE}
        forced = self.env(dockerfile=True)
        bare = self.env(prebake_recipe=None)
        with self.assertLogs("test.prebake", "WARNING"):
            asyncio.run(forced.start(force_build=True))
            asyncio.run(bare.start(force_build=False))
        self.assertIn("force_build", self.record(forced)["reason"])
        self.assertEqual(bare._decision.status, prebaked.NOT_APPLICABLE)  # type: ignore[union-attr]
        unasked = self.env(prebake_harness=None)
        asyncio.run(unasked.start(force_build=False))
        self.assertFalse((unasked.trial_paths.trial_dir / prebaked.RECORD_NAME).exists())
        self.assertEqual(self.inspected, [])
        self.assertEqual(self.started, [TASK_IMAGE] * 3)

    def test_a_skippable_exec_is_answered_once_then_runs(self) -> None:
        self.images = {want().image: baked(skips=(UV_SIGNATURE,), skip_commands=(prebaked.command_digest(UV_STEP),)),
                       TASK_IMAGE: BASE}
        env = self.env()
        asyncio.run(env.start(force_build=False))
        first = asyncio.run(env.exec(UV_STEP))
        self.assertEqual((first.return_code, first.stdout, self.ran), (0, "", []))
        record = self.record(env)
        self.assertEqual([s["signature"] for s in record["skipped_execs"]], [UV_SIGNATURE])
        second = asyncio.run(env.exec(UV_STEP))
        self.assertEqual((second.stdout, self.ran), ("ran", [UV_STEP]))
        self.assertEqual(asyncio.run(env.exec("tmux -V")).stdout, "ran")

    def test_a_near_miss_runs_and_is_logged(self) -> None:
        self.images = {want().image: baked(skips=(UV_SIGNATURE,), skip_commands=(prebaked.command_digest(UV_STEP),)),
                       TASK_IMAGE: BASE}
        env = self.env()
        asyncio.run(env.start(force_build=False))
        with self.assertLogs("test.prebake", "WARNING") as logs:
            asyncio.run(env.exec(UV_STEP, user="agent"))
        self.assertEqual(self.ran, [UV_STEP])
        self.assertIn("matches a skipped command", logs.output[0])


class HarborInternals(unittest.TestCase):
    """What the class relies on in Harbor 0.25.0. A failure here means a
    Harbor upgrade moved it: re-check the class before changing the pin."""

    def test_harbor_version_is_pinned(self) -> None:
        self.assertEqual(prebaked.harbor_version(), "0.25.0")
        requirements = (settings.ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("harbor==0.25.0", requirements)

    def test_docker_environment_still_has_what_the_class_uses(self) -> None:
        from harbor.environments.base import BaseEnvironment
        from harbor.environments.docker.docker import DockerEnvironment
        from harbor.environments.docker.compose_env import ComposeInfraEnvVars

        for name in ("_engine_cmd", "_dockerfile_path", "start", "exec"):
            self.assertTrue(hasattr(DockerEnvironment, name), name)
        for name in ("_resolve_user", "_merge_env"):
            self.assertTrue(hasattr(BaseEnvironment, name), name)
        self.assertIn("prebuilt_image_name", ComposeInfraEnvVars.model_fields)
        self.assertIn("prebuilt_image_name=task_env_config.docker_image", inspect.getsource(DockerEnvironment))
        compose = Path(inspect.getfile(DockerEnvironment)).with_name("docker-compose-prebuilt.yaml")
        self.assertIn("${PREBUILT_IMAGE_NAME}", compose.read_text(encoding="utf-8"))

    def test_trial_names_the_agent_and_verifier_environments_as_expected(self) -> None:
        from harbor.trial import trial

        source = inspect.getsource(trial)
        self.assertIn('session_id=f"{self.config.trial_name}__env"', source)
        self.assertIn('f"{self.config.trial_name}__verifier__{key}"', source)

    def test_the_class_imports_only_harbor_and_the_standard_library(self) -> None:
        source = inspect.getsource(prebaked)
        imports = [line.split()[1] for line in source.splitlines() if line.startswith(("import ", "from "))]
        self.assertEqual([m for m in imports if m.startswith(("bench", "."))], [])
        allowed = ("__future__", "asyncio", "collections", "dataclasses", "datetime", "functools", "hashlib",
                   "importlib", "json", "re", "typing", "harbor")
        self.assertTrue(all(m.split(".")[0] in allowed for m in imports), imports)

    def test_harbors_cli_parsers_turn_the_flags_into_this_class(self) -> None:
        from harbor.cli.utils import parse_kwargs, resolve_environment_spec

        env_type, import_path = resolve_environment_spec(PREBAKED_ENV)
        self.assertEqual((env_type, import_path), (None, PREBAKED_ENV))
        kwargs = parse_kwargs(['prebake_harness="claude-code"', 'prebake_version="2.1.312"',
                               f'prebake_recipe="{RECIPE}"', 'prebake_prefix="benchbake"', "prebake_required=false"])
        self.assertEqual(kwargs, {"prebake_harness": "claude-code", "prebake_version": "2.1.312",
                                  "prebake_recipe": RECIPE, "prebake_prefix": "benchbake", "prebake_required": False})

    def test_decision_dict_is_plain_json(self) -> None:
        d = Decision(prebaked.READY, "ok", image="i", skip_execs=("a",))
        self.assertEqual(json.loads(json.dumps(d.to_dict()))["skip_execs"], ["a"])
        self.assertTrue(d.to_dict()["used"])


if __name__ == "__main__":
    unittest.main()
