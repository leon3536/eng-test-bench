"""The checks a pre-baked image passes before it is named: static, smoke,
fingerprint, Harbor's install_only setup check and the stock-against-baked
equivalence check.

No Docker and no Harbor trial: FakeDocker answers docker, the child process
is faked at Popen, and Harbor's Trial is replaced where its lifecycle would
start containers. Harbor's own setup() code still runs where the equivalence
check drives it through ExecShim.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from tests.fakes_docker import FakeDocker, FakePopen
from tests.support import BenchTestCase, completed

from bench import procs, settings
from bench.environments.prebaked import PREBAKED_ENV, Want, bake_labels, exec_signature
from bench.prebake import inventory, verify

TASK = "alexgshaw/fix-git:20251031"
RECIPE = "a" * 64
SITE = "/installed-agent/strands/venv/lib/python3.12/site-packages"
MODEL = "anthropic/claude-opus-5-5"


def manifest(meta: list[str], md5: list[str]) -> str:
    return "### META\n" + "\n".join(meta) + "\n### MD5\n" + "\n".join(md5) + "\n### END\n"


class VerifyTestCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.docker = FakeDocker()
        self.allow_subprocess(self.docker.matches, self.docker)
        inventory.forget_images()
        self.addCleanup(inventory.forget_images)
        self.base = self.docker.add_image(TASK)
        self.want = Want("claude-code", "2.1.312", RECIPE, TASK, prefix="benchbake-staging/b1-1")

    def baked(self, config: dict[str, Any] | None = None, layers: tuple[str, ...] | None = None) -> str:
        base = self.docker.find(TASK)
        labels = bake_labels(self.want, base_image_id=self.base, baked_at="2026-10-29T00:00:00Z",
                             dockerfile_sha256="d" * 64)
        self.docker.add_image(self.want.image, layers=layers or (*base["RootFS"]["Layers"], "sha256:x"),
                              labels=labels, config={**{k: v for k, v in base["Config"].items() if k != "Labels"},
                                                     **(config or {})})
        inventory.forget_images()
        return self.want.image


class ImageChecks(VerifyTestCase):
    def test_static_check_is_the_trials_check_plus_an_unchanged_config(self) -> None:
        ref = self.baked()
        result = verify.static_check(ref, self.want)
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["status"], result["base_id"], result["added_layers"]), ("ready", self.base, 1))

    def test_a_changed_config_or_a_wrong_base_fails(self) -> None:
        ref = self.baked(config={"WorkingDir": "/root", "Env": ["PATH=/x"]})
        result = verify.static_check(ref, self.want)
        self.assertFalse(result["ok"])
        self.assertEqual([d.split(":")[0] for d in result["config_diff"]], ["Env", "WorkingDir"])
        self.docker.images.clear()
        self.docker.add_image(TASK)
        self.baked(layers=("sha256:other", "sha256:x", "sha256:y"))
        self.assertEqual(verify.static_check(self.want.image, self.want)["status"], "stale")
        self.assertEqual(verify.static_check("absent/x:1", Want("pi", "1", RECIPE, "absent/y:1"))["status"],
                         "missing")

    def test_smoke_runs_offline_and_reports_the_exit_code(self) -> None:
        ref = self.baked()
        self.assertTrue(verify.smoke(ref, "tmux -V")["ok"])
        run = self.docker.ran("run")[-1]
        for flag in ("--rm", "--network", "--pull"):
            self.assertIn(flag, run)
        self.assertEqual(run[run.index("--network") + 1], "none")
        self.assertEqual(run[run.index("--pull") + 1], "never")
        self.docker.run_handlers.append((lambda a, s: True, lambda a, s: (1, "", "no tmux")))
        result = verify.smoke(ref, "tmux -V")
        self.assertEqual((result["ok"], result["rc"]), (False, 1))
        self.assertIn("no tmux", result["output"])
        self.allow_subprocess(self.docker.matches, mock.Mock(side_effect=OSError("gone")))
        self.assertFalse(verify.smoke(ref, "tmux -V")["ok"])

    def test_fingerprint_is_saved_under_the_image_id(self) -> None:
        ref = self.baked()
        root = self.tmpdir()
        result = verify.fingerprint(ref, "sha256:abc", root)
        self.assertTrue(result["ok"])
        saved = Path(result["path"])
        self.assertEqual(saved, root / "fingerprints" / "sha256_abc.txt")
        self.assertIn("## end", saved.read_text(encoding="utf-8"))
        self.docker.run_handlers.append((lambda a, s: True, lambda a, s: (1, "## os\n", "boom")))
        self.assertFalse(verify.fingerprint(ref, "sha256:abc", root)["ok"])

    def test_check_image_stops_at_the_first_failure(self) -> None:
        ref = self.baked(config={"User": "nobody"})
        result = verify.check_image(ref, self.want, "true", self.tmpdir())
        self.assertFalse(result["ok"])
        self.assertNotIn("smoke", result)
        self.assertEqual(self.docker.ran("run"), [])


class Manifests(unittest.TestCase):
    def test_only_volatile_paths_may_differ(self) -> None:
        stock = manifest(["d\t755\troot:root\t4096\t\t/usr", "f\t644\troot:root\t10\t\t/usr/bin/tmux",
                          "f\t644\troot:root\t5\t\t/var/log/dpkg.log", "d\t755\troot:root\t4096\t\t/tmp/cc-socks",
                          "f\t644\troot:root\t1\t\t/etc/hostname"],
                         ["0" * 32 + "  /usr/bin/tmux", "1" * 32 + "  /var/log/dpkg.log",
                          "2" * 32 + "  /etc/hostname"])
        baked = manifest(["d\t755\troot:root\t12288\t\t/usr", "f\t644\troot:root\t10\t\t/usr/bin/tmux",
                          "f\t644\troot:root\t7\t\t/var/log/dpkg.log", "f\t644\troot:root\t1\t\t/etc/hostname"],
                         ["0" * 32 + "  /usr/bin/tmux", "3" * 32 + "  /var/log/dpkg.log",
                          "4" * 32 + "  /etc/hostname"])
        result = verify.diff_manifests(stock, baked)
        self.assertEqual(result["violations"], [])
        self.assertEqual(result["allowed"], {"log timestamps": 1, "temporary files": 1,
                                             "per-container network files": 1})
        changed = baked.replace("0" * 32 + "  /usr/bin/tmux", "9" * 32 + "  /usr/bin/tmux")
        self.assertEqual(verify.diff_manifests(stock, changed)["violations"], ["/usr/bin/tmux: content"])
        extra = baked.replace("### MD5", "f\t755\troot:root\t3\t\t/root/.local/bin/claude\n### MD5")
        self.assertEqual(verify.diff_manifests(stock, extra)["violations"],
                         ["/root/.local/bin/claude: only in the pre-baked container"])
        mode = baked.replace("f\t644\troot:root\t10\t\t/usr/bin/tmux", "f\t755\troot:root\t10\t\t/usr/bin/tmux")
        self.assertEqual(verify.diff_manifests(stock, mode)["violations"], ["/usr/bin/tmux: mode 644 -> 755"])

    def test_the_allowlist_is_narrow(self) -> None:
        for path in ("/root/.npm/_logs/2026-debug-0.log", "/root/.cache/uv/archive-v0/abc/x.py",
                     "/usr/lib/python3/__pycache__/a.cpython-312.pyc", "/root/.claude.json",
                     "/var/lib/dbus/machine-id", "/root/.config/asciinema/install-id",
                     "/root/.local/state/goose/logs/cli/2026-10-31/20261031_042456.log",
                     f"{SITE}/bench_strands_factory-0.1.0.dist-info/uv_cache.json",
                     f"{SITE}/bench_strands_factory-0.1.0.dist-info/RECORD",
                     "/root/.nvm/.git/index", "/root/.nvm/.git/objects/pack/pack-05cc8096.pack"):
            self.assertIsNotNone(verify.allowed(path), path)
        for path in ("/usr/bin/node", "/root/.local/bin/claude", "/var/lib/dpkg/status", "/root/.bashrc",
                     "/installed-agent/strands/venv/bin/python", "/etc/apt/sources.list",
                     f"{SITE}/bench_strands_factory-0.1.0.dist-info/METADATA",
                     f"{SITE}/bench_strands_factory-0.1.0.dist-info/direct_url.json",
                     f"{SITE}/__editable___bench_strands_factory_0_1_0_finder.py",
                     "/installed-agent/strands/factory/bench_strands_factory/__init__.py",
                     "/root/.nvm/.git/HEAD", "/root/.nvm/.git/packed-refs", "/root/.nvm/nvm.sh"):
            self.assertIsNone(verify.allowed(path), path)

    def test_fingerprints_must_match_section_by_section(self) -> None:
        a = "## dpkg\ntmux=3.3a ii\n## which\ntmux /usr/bin/tmux\n## end\n"
        self.assertEqual(verify.diff_fingerprints(a, a), [])
        b = a.replace("3.3a", "3.4")
        self.assertEqual(verify.diff_fingerprints(a, b), ["dpkg: stock ['tmux=3.3a ii'] vs pre-baked ['tmux=3.4 ii']"])


class SetupCheck(VerifyTestCase):
    def test_judging_an_install_only_trial(self) -> None:
        good = {"agent_setup_sec": 0.5, "exception": None,
                "prebake": {"used": True, "skipped_execs": [{"signature": "s"}]}}
        self.assertTrue(verify.judge_setup(good, 15, 1)["ok"])
        for change, words in (({"prebake": {"used": False, "reason": "no pre-baked image x"}}, "not used"),
                              ({"agent_setup_sec": 503.0}, "over this harness's 15 s budget"),
                              ({"exception": "AgentSetupTimeoutError"}, "AgentSetupTimeoutError"),
                              ({"agent_setup_sec": None}, "did not finish")):
            result = verify.judge_setup({**good, **change}, 15, 1)
            self.assertFalse(result["ok"])
            self.assertIn(words, "; ".join(result["problems"]))
        self.assertIn("0 install step(s) were skipped, 1 expected",
                      verify.judge_setup({**good, "prebake": {"used": True}}, 15, 1)["problems"][0])
        self.assertFalse(verify.judge_setup({"ok": False, "error": "cancelled"}, 15, 0)["ok"])

    def test_the_trial_runs_in_a_child_without_credentials_through_the_env_class(self) -> None:
        seen: dict[str, Any] = {}

        def child(argv: list[str], **kw: Any) -> Any:
            seen.update(argv=argv, env=kw["env"], cwd=kw["cwd"])
            spec = json.loads(Path(argv[-2]).read_text(encoding="utf-8"))
            seen["spec"] = spec
            Path(argv[-1]).write_text(json.dumps({"agent_setup_sec": 2.5, "exception": None,
                                                  "prebake": {"used": True, "skipped_execs": []}}),
                                      encoding="utf-8")
            return FakePopen(0)

        self.allow_subprocess(lambda argv: "bench.prebake.verify" in argv, child)
        task_dir = self.tmpdir() / "fix-git"
        result = verify.setup_check(prefix="benchbake-staging/b1-1", family="terminus-2", version="0.25.0",
                                    recipe=RECIPE, agent="terminus-2", kwargs={"max_turns": 300}, model_name=MODEL,
                                    task_dir=task_dir, budget_sec=15, expected_skips=0, work=self.tmpdir())
        self.assertTrue(result["ok"], result)
        self.assertEqual(seen["argv"][1:4], ["-m", "bench.prebake.verify", "setup-trial"])
        self.assertEqual(Path(seen["cwd"]), settings.ROOT)
        self.assertNotIn("ANTHROPIC_API_KEY", seen["env"])
        self.assertIn("ANTHROPIC_API_KEY", os.environ)  # the test's fake key, kept from the child
        self.assertEqual(seen["spec"]["env_kwargs"], {"prebake_harness": "terminus-2", "prebake_version": "0.25.0",
                                                      "prebake_recipe": RECIPE,
                                                      "prebake_prefix": "benchbake-staging/b1-1",
                                                      "prebake_required": True})

    def test_only_a_setup_over_budget_gets_a_second_trial(self) -> None:
        answers = [61.0, 13.4]
        runs: list[str] = []

        def child(argv: list[str], **kw: Any) -> Any:
            spec = json.loads(Path(argv[-2]).read_text(encoding="utf-8"))
            runs.append(Path(spec["trials_dir"]).name)
            Path(argv[-1]).write_text(json.dumps({"agent_setup_sec": answers[len(runs) - 1], "exception": None,
                                                  "prebake": {"used": True, "skipped_execs": []}}),
                                      encoding="utf-8")
            return FakePopen(0)

        self.allow_subprocess(lambda argv: "bench.prebake.verify" in argv, child)
        kw = dict(prefix="p", family="terminus-2", version="0.25.0", recipe=RECIPE, agent="terminus-2", kwargs={},
                  model_name=MODEL, task_dir=self.tmpdir(), budget_sec=15, expected_skips=0)
        result = verify.setup_check(work=self.tmpdir(), **kw)
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["attempts"], result["earlier"][0]["agent_setup_sec"]), (2, 61.0))
        self.assertEqual(runs, ["trials", "trials-2"])
        answers[:] = [503.0, 590.0]
        runs.clear()
        result = verify.setup_check(work=self.tmpdir(), **kw)
        self.assertEqual((result["ok"], result["attempts"]), (False, 2))
        self.assertIn("590 s", result["problems"][0])
        answers[:] = [1.0]
        runs.clear()
        result = verify.setup_check(work=self.tmpdir(), **{**kw, "expected_skips": 1})
        self.assertEqual((result["ok"], result["attempts"], len(runs)), (False, 1, 1))

    def test_a_cancelled_check_kills_the_child_and_removes_its_containers(self) -> None:
        release = threading.Event()
        self.allow_subprocess(lambda argv: "bench.prebake.verify" in argv,
                              lambda argv, **kw: FakePopen(0, release=release))
        killed: list[int] = []
        cleaned: list[Path] = []
        cancel = threading.Event()
        cancel.set()
        with mock.patch.object(procs, "kill_tree", lambda pid, timeout=30: (killed.append(pid), release.set())), \
                mock.patch.object(procs, "cleanup_with_retry", lambda d, tries=3: cleaned.append(d) or (True, "")):
            result = verify.setup_check(prefix="p", family="pi", version="1", recipe=RECIPE, agent="pi", kwargs={},
                                        model_name=MODEL, task_dir=self.tmpdir(), budget_sec=60, expected_skips=0,
                                        work=self.tmpdir(), cancel=cancel)
        self.assertEqual((result["ok"], result["error"]), (False, "cancelled"))
        self.assertEqual(killed, [0])
        self.assertEqual(len(cleaned), 1)

    def test_the_child_builds_the_trial_from_the_bench_flags(self) -> None:
        captured: dict[str, Any] = {}

        class FakeTrial:
            def __init__(self, config: Any) -> None:
                captured["config"] = config
                self.paths = mock.Mock(trial_dir=Path(config.trials_dir) / config.trial_name)

            @classmethod
            async def create(cls, config: Any) -> FakeTrial:
                return cls(config)

            async def run(self) -> None:
                d = self.paths.trial_dir
                d.mkdir(parents=True)
                (d / "result.json").write_text(json.dumps({
                    "environment_setup": {"started_at": "2026-10-29T00:00:00Z", "finished_at": "2026-10-29T00:00:06Z"},
                    "agent_setup": {"started_at": "2026-10-29T00:00:06Z", "finished_at": "2026-10-29T00:00:08.5Z"},
                    "exception_info": None}), encoding="utf-8")
                (d / "prebake.json").write_text(json.dumps({"used": True, "skipped_execs": []}), encoding="utf-8")

        import harbor.trial.trial as trial_module

        spec = {"agent": "bench.agents.mini_swe_agent:MiniSweAgentWithMcp", "kwargs": {"version": "2.4.6"},
                "model_name": MODEL, "task_dir": str(self.tmpdir()), "trials_dir": str(self.tmpdir()),
                "trial_name": "fix-git__mini-setup",
                "env_kwargs": {"prebake_harness": "mini-swe-agent", "prebake_version": "2.4.6",
                               "prebake_recipe": RECIPE, "prebake_prefix": "benchbake", "prebake_required": True}}
        with mock.patch.object(trial_module, "Trial", FakeTrial):
            result = asyncio.run(verify._setup_trial(spec))
        config = captured["config"]
        self.assertTrue(config.install_only)
        self.assertEqual(config.environment.import_path, PREBAKED_ENV)
        self.assertIs(config.environment.kwargs["prebake_required"], True)
        self.assertEqual(config.environment.kwargs["prebake_recipe"], RECIPE)
        self.assertEqual(config.agent.import_path, spec["agent"])
        self.assertEqual((result["env_setup_sec"], result["agent_setup_sec"]), (6.0, 2.5))
        self.assertTrue(result["prebake"]["used"])

    def test_the_child_reports_its_own_failure_as_a_result(self) -> None:
        work = self.tmpdir()
        (work / "s.json").write_text(json.dumps({"agent": "x"}), encoding="utf-8")
        self.assertEqual(verify.main(["setup-trial", str(work / "s.json"), str(work / "r.json")]), 0)
        self.assertFalse(json.loads((work / "r.json").read_text(encoding="utf-8"))["ok"])
        with mock.patch("sys.stderr", mock.Mock()):
            self.assertEqual(verify.main(["nope"]), 2)


class ExecShimTests(BenchTestCase):
    def shim(self, skip: tuple[str, ...] = ()) -> tuple[verify.ExecShim, list[list[str]]]:
        calls: list[list[str]] = []
        shim = verify.ExecShim("ctr", self.tmpdir(), skip)

        async def fake(args: list[str], timeout_sec: float | None) -> tuple[int, str, str]:
            calls.append(args)
            return 0, "out", ""

        shim._docker = fake  # type: ignore[method-assign]
        return shim, calls

    def test_exec_mirrors_harbors_flags_and_skips_a_signature_once(self) -> None:
        command = "uv pip install x"
        env = {"DEBIAN_FRONTEND": "noninteractive", "API_KEY": "sk-secret-1234567890"}
        skip = exec_signature(command, None, "/w", env)
        shim, calls = self.shim((skip,))
        first = asyncio.run(shim.exec(command, cwd="/w", env=env))
        self.assertEqual((first.return_code, first.stdout, calls), (0, "", []))
        asyncio.run(shim.exec(command, cwd="/w", env=env, user="root"))
        asyncio.run(shim.exec(command, cwd="/w", env=env))
        self.assertEqual(calls[0], ["exec", "-w", "/w", "-e", "DEBIAN_FRONTEND=noninteractive", "-e",
                                    "API_KEY=sk-secret-1234567890", "-u", "root", "ctr", "bash", "-c", command])
        self.assertEqual(len(calls), 2)
        self.assertEqual(shim.skipped, [skip])

    def test_uploads_go_through_harbors_own_ops(self) -> None:
        shim, calls = self.shim()
        source = self.tmpdir()
        (source / "run.py").write_text("print(1)\r\n", encoding="utf-8")
        asyncio.run(shim.upload_dir(source, "/installed-agent/x"))
        self.assertEqual(calls[0][0], "cp")
        self.assertTrue(calls[0][2].startswith("ctr:/installed-agent/x"))
        if sys.platform == "win32":
            self.assertEqual(calls[1][:4], ["exec", "ctr", "bash", "-c"])
            self.assertIn("sed -i 's/\\r$//'", calls[1][4])


class Equivalence(VerifyTestCase):
    def test_stock_and_baked_setups_are_compared(self) -> None:
        """terminus-2's real install against two fake containers: the stock one
        installs tmux, the baked one already has it."""
        self.baked()
        work = self.tmpdir()
        stock_manifest = manifest(["f\t755\troot:root\t9\t\t/usr/bin/tmux"], ["0" * 32 + "  /usr/bin/tmux"])
        commands: dict[str, list[str]] = {"stock": [], "baked": []}

        async def fake_exec(shim: verify.ExecShim, args: list[str], timeout_sec: float | None) -> tuple[int, str, str]:
            side = "stock" if shim.container.endswith("stock") else "baked"
            command = args[-1]
            commands[side].append(command)
            if command in ("tmux -V", "asciinema --version"):
                return (127, "", "") if side == "stock" and not any("apt-get install" in c for c in commands[side]) \
                    else (0, "tmux 3.3a\n", "")
            if command.startswith("cat /etc/os-release"):
                return 0, 'ID=debian\nVERSION_ID="12"\n', ""
            if command.startswith("which "):
                return (0, "", "") if "apt-get" in command else (1, "", "")
            return 0, "", ""

        def fake_docker(args: list[str], timeout: float = 60, input: str | None = None) -> Any:
            if args[:2] == ["run", "-d"]:
                return completed(args, "id\n")
            if args[:2] == ["exec", "-i"]:
                out = stock_manifest if input == verify.MANIFEST_SCRIPT else "## os\nID=debian\n## end\n"
                return completed(args, out)
            return completed(args, "")

        spec = {"image": self.want.image, "task_image": TASK, "family": "terminus-2", "agent": "terminus-2",
                "kwargs": {}, "model_name": MODEL, "token": "t1", "work": str(work)}
        with mock.patch.object(verify.ExecShim, "_docker", fake_exec), mock.patch.object(verify, "docker", fake_docker):
            result = verify._equivalence(spec)
        self.assertTrue(result["ok"], result)
        self.assertTrue(any("apt-get install -y tmux asciinema" in c for c in commands["stock"]))
        self.assertFalse(any("apt-get" in c for c in commands["baked"]))
        self.assertEqual(result["skipped"], {"stock": 0, "baked": 0})
        self.assertTrue((work / "manifest-stock.txt").exists())

    def test_the_containers_are_found_by_their_label(self) -> None:
        calls: list[list[str]] = []

        def fake(args: list[str], timeout: float = 60, input: str | None = None) -> Any:
            calls.append(args)
            return completed(args, "c1\nc2\n" if args[0] == "ps" else "")

        with mock.patch.object(verify, "docker", fake):
            verify.remove_containers("t1")
        self.assertEqual(calls, [["ps", "-aq", "--filter", f"label={verify.VERIFY_LABEL}=t1"],
                                 ["rm", "-f", "c1", "c2"]])


if __name__ == "__main__":
    unittest.main()
