"""Dashboard server: sign-in, config saves, run starts and shutdown.

The State is built over temp config files, a temp data folder and a temp
ledger, with a fake run manager, so nothing here starts harbor or Docker or
makes a paid request. FastAPI's TestClient calls the app in process with the
port-scoped session cookie; no socket is opened.

ServerCase is the base the other server test modules build on.
"""

from __future__ import annotations

import copy
import os
import textwrap
from collections.abc import Callable
from typing import Any
from unittest import mock

import yaml
from fastapi.testclient import TestClient

from tests.support import BenchTestCase

from bench import checks, fsutil, server, settings
from bench.ledger import Ledger
from bench.server import BUSY, QUEUE_ACTIVE, SIGN_IN, State, create_app

PORT = 8797
BASE_URL = f"http://127.0.0.1:{PORT}"
IMPORT_PATH = "bench.agents.mini_swe_agent:MiniSweAgentWithMcp"
TASKS = ["task-a", "task-b", "task-c", "task-d"]

SERVER_YAML = f"""\
dataset: bench-test@1
providers:
  anthropic: {{env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}}
  openrouter: {{env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}}
modes:
  pilot: {{n_tasks: 1}}
  sample: {{n_tasks: 3}}
run:
  budget_usd: 10
baseline_arm: claude-code
arms:
  - name: claude-code
    agent: claude-code
    provider: anthropic
    model: claude-opus-5-5
    options: {{max_budget_usd: 5, reasoning_effort: high}}
  - {{name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}}
  - name: mini
    agent: {IMPORT_PATH}
    family: mini-swe-agent
    provider: openrouter
    model: z-ai/glm-5.3
designs:
  max_budget_usd: 20
"""


class FakeManager:
    """The RunManager surface the server uses. start records the call and
    returns a run; finish() ends it and calls the listeners, as the real
    manager does after its final save."""

    def __init__(self, runs: list[dict[str, Any]] | None = None) -> None:
        self.runs: dict[str, dict[str, Any]] = {r["id"]: copy.deepcopy(r) for r in runs or []}
        self.active_id: str | None = None
        self.started: list[tuple[str, list[str]]] = []
        self.started_specs: list[Any] = []
        self.listeners: list[Callable[[dict[str, Any]], None]] = []
        self.calls: list[str] = []

    def list(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(r) for r in self.runs.values()]

    def get(self, run_id: str) -> dict[str, Any]:
        if run_id not in self.runs:
            raise KeyError(run_id)
        return copy.deepcopy(self.runs[run_id])

    def is_idle(self) -> bool:
        return self.active_id is None

    def add_listener(self, fn: Callable[[dict[str, Any]], None]) -> None:
        self.listeners.append(fn)

    def _new(self, label: str, budget: float, extra: dict[str, Any]) -> dict[str, Any]:
        if self.active_id is not None:
            raise RuntimeError("A run is already in progress")
        run_id = f"20261029-1200{len(self.runs):02d}-{label}"
        self.runs[run_id] = {"id": run_id, "created_at": "2026-10-29T12:00:00+00:00", "status": "running",
                             "spend_usd": 0.0, "budget_usd": budget, "arms": [], **extra}
        self.active_id = run_id
        return copy.deepcopy(self.runs[run_id])

    def start(self, cfg: settings.Config, mode_name: str, arm_names: list[str]) -> dict[str, Any]:
        self.started.append((mode_name, list(arm_names)))
        return self._new(mode_name, cfg.run.budget_usd, {"source": "config", "mode": mode_name})

    def start_spec(self, cfg: settings.Config, spec: Any) -> dict[str, Any]:
        self.started_specs.append(spec)
        return self._new(spec.label, spec.run.budget_usd, {"source": spec.source, "spec": spec.spec,
                                                           "spec_hash": spec.spec_hash,
                                                           "design_ref": spec.design_ref})

    def finish(self, run_id: str, status: str = "done", spend: float = 0.5) -> None:
        self.runs[run_id].update(status=status, spend_usd=spend, finished_at="2026-10-29T12:30:00+00:00")
        self.active_id = None
        for fn in self.listeners:
            fn(copy.deepcopy(self.runs[run_id]))

    def cancel(self, run_id: str) -> None:
        if run_id != self.active_id:
            raise RuntimeError("That run is not in progress")

    def log_path(self, run_id: str, arm: str) -> Any:
        raise KeyError(arm)

    def shutdown(self, timeout: float = 90) -> None:
        self.calls.append("shutdown")

    def emergency_stop(self) -> None:
        self.calls.append("emergency_stop")


class ServerCase(BenchTestCase):
    """A State over temp files and a fake run manager, and a signed-in client."""

    YAML = SERVER_YAML
    OVERRIDES: str | None = None
    TASKS = TASKS

    def setUp(self) -> None:
        super().setUp()
        self.data = self.isolate_data_dir()
        self.ledger_path = self.isolate_ledger()
        self.base_file, self.overrides_file = self.use_config_files(self.YAML, self.OVERRIDES, tasks=self.TASKS)
        self.docker = True
        self.patch(checks, "docker_ok", lambda: self.docker)
        self.patch(checks, "docker_capacity", lambda: None)
        self.patch(checks, "readiness", lambda cfg: [{"name": "Docker", "ok": self.docker, "detail": "test"}])
        self.manager = self.make_manager()
        self.state = self.make_state()
        self.client = self.make_client(self.state)

    # Building -------------------------------------------------------------------

    def patch(self, target: Any, name: str, value: Any) -> None:
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_manager(self) -> Any:
        return FakeManager()

    def make_ledger(self) -> Ledger:
        ledger = Ledger(self.ledger_path, roots=[], root=self.tmpdir())
        ledger.open()
        return ledger

    def client_factory(self) -> Any:
        raise AssertionError("This test makes no assistant request")

    def make_state(self, port: int = PORT, manager: Any = None, ledger: Ledger | None = None) -> State:
        state = State(port, ledger=ledger or self.make_ledger(), manager=manager or self.manager,
                      client_factory=lambda: self.client_factory())
        self.addCleanup(state.shutdown)
        return state

    def make_client(self, state: State, signed_in: bool = True, x_bench: bool = True) -> TestClient:
        client = TestClient(create_app(state), base_url=f"http://127.0.0.1:{state.port}")
        self.addCleanup(client.close)
        if signed_in:
            client.cookies.set(f"bench_session_{state.port}", state.token)
        if x_bench:
            client.headers["X-Bench"] = "1"
        return client

    # Reading --------------------------------------------------------------------

    def overrides(self) -> dict[str, Any]:
        if not self.overrides_file.exists():
            return {}
        return yaml.safe_load(self.overrides_file.read_text(encoding="utf-8")) or {}

    def edit_base(self, old: str, new: str) -> None:
        text = self.base_file.read_text(encoding="utf-8")
        self.assertIn(old, text)
        self.base_file.write_text(text.replace(old, new), encoding="utf-8", newline="\n")

    def ok(self, response: Any, status: int = 200) -> Any:
        self.assertEqual(response.status_code, status, response.text)
        return response.json()


# Sign-in -------------------------------------------------------------------------------


class SignIn(ServerCase):
    def test_the_session_cookie_is_named_for_the_port(self) -> None:
        anonymous = self.make_client(self.state, signed_in=False)
        response = anonymous.get(f"/?token={self.state.token}", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertTrue(response.headers["set-cookie"].startswith(f"bench_session_{PORT}="),
                        response.headers["set-cookie"])
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        self.assertEqual(anonymous.get("/api/state").status_code, 200)

        # The unscoped name of older versions signs nobody in.
        old_style = self.make_client(self.state, signed_in=False)
        old_style.cookies.set("bench_session", self.state.token)
        self.assertEqual(old_style.get("/api/state").status_code, 401)

    def test_two_benches_on_one_host_keep_their_own_sessions(self) -> None:
        other_ledger = Ledger(self.tmpdir() / "other.sqlite", roots=[], root=self.tmpdir())
        other_ledger.open()
        other = self.make_state(port=8795, manager=FakeManager(), ledger=other_ledger)
        mine = self.make_client(self.state, signed_in=False)
        theirs = self.make_client(other, signed_in=False)
        set_mine = mine.get(f"/?token={self.state.token}", follow_redirects=False).headers["set-cookie"]
        set_theirs = theirs.get(f"/?token={other.token}", follow_redirects=False).headers["set-cookie"]
        self.assertNotEqual(set_mine.split("=", 1)[0], set_theirs.split("=", 1)[0])
        # A browser holds both cookies for 127.0.0.1 and sends both to each port.
        both = {f"bench_session_{PORT}": self.state.token, "bench_session_8795": other.token}
        for state in (self.state, other):
            client = self.make_client(state, signed_in=False)
            for name, value in both.items():
                client.cookies.set(name, value)
            self.assertEqual(client.get("/api/state").status_code, 200)

    def test_api_needs_the_cookie_and_writes_need_x_bench(self) -> None:
        anonymous = self.make_client(self.state, signed_in=False)
        self.assertEqual(anonymous.get("/api/state").status_code, 401)
        self.assertEqual(anonymous.put("/api/config/run", json={}).status_code, 401)
        no_header = self.make_client(self.state, x_bench=False)
        self.assertEqual(no_header.get("/api/state").status_code, 200)
        self.assertEqual(no_header.put("/api/config/run", json={"effort": "high"}).status_code, 403)
        self.assertEqual(self.overrides(), {})
        foreign = TestClient(create_app(self.state), base_url="http://evil.example:8797")
        self.addCleanup(foreign.close)
        foreign.cookies.set(f"bench_session_{PORT}", self.state.token)
        self.assertEqual(foreign.get("/api/state").status_code, 403)

    def test_a_wrong_token_and_the_sign_in_page(self) -> None:
        anonymous = self.make_client(self.state, signed_in=False)
        response = anonymous.get("/?token=nope", follow_redirects=False)
        self.assertEqual(response.status_code, 401)
        self.assertNotIn("set-cookie", response.headers)
        page = anonymous.get("/")
        self.assertEqual(page.status_code, 401)
        self.assertIn("valid until the bench is stopped", page.text)
        self.assertNotIn("one-time", SIGN_IN)

    def test_the_page_and_its_files_are_never_served_stale(self) -> None:
        client = self.make_client(self.state)
        self.assertEqual(client.get("/").headers["cache-control"], "no-store")
        for name in ("app.js", "report.js", "style.css"):
            response = client.get(f"/static/{name}")
            self.assertEqual(response.status_code, 200, name)
            self.assertEqual(response.headers["cache-control"], "no-cache", name)


# Config saves ----------------------------------------------------------------------------


class ConfigSaves(ServerCase):
    def run_form(self, **changes: Any) -> dict[str, Any]:
        """Every run field, as the dashboard's form sends them."""
        form = {k: v for k, v in self.ok(self.client.get("/api/state"))["config"]["run"].items()}
        form.update(changes)
        return form

    def test_a_save_after_bench_yaml_changed_on_disk_is_refused(self) -> None:
        self.edit_base("budget_usd: 10", "budget_usd: 40")
        self.edit_base("- {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}",
                       "- {name: pi, agent: pi, provider: anthropic, model: claude-sonnet-5-5}")
        stale_form = self.run_form(effort="high")  # still shows budget 10 from the loaded config
        response = self.client.put("/api/config/run", json=stale_form)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("Reload config", response.json()["detail"])
        arm = self.client.put("/api/config/arms/pi", json={"enabled": False, "model": "claude-opus-5-5"})
        self.assertEqual(arm.status_code, 409)
        mode = self.client.put("/api/config/modes/pilot", json={"n_tasks": 1, "n_attempts": 2})
        self.assertEqual(mode.status_code, 409)
        self.assertFalse(self.overrides_file.exists())

        view = self.ok(self.client.post("/api/config/reload"))
        self.assertEqual(view["run"]["budget_usd"], 40)
        self.ok(self.client.put("/api/config/run", json=self.run_form(effort="high")))
        self.assertEqual(self.overrides(), {"run": {"effort": "high"}})
        self.assertEqual(self.state.cfg.run.budget_usd, 40)
        self.assertEqual(self.state.cfg.arm("pi").model, "claude-sonnet-5-5")

    def test_run_and_mode_saves_do_not_pin_code_defaults(self) -> None:
        # bench.yaml sets only budget_usd; the form shows every field at its default.
        self.ok(self.client.put("/api/config/run", json=self.run_form()))
        self.assertEqual(self.overrides(), {})
        self.ok(self.client.put("/api/config/run", json=self.run_form(effort="high", n_concurrent=3)))
        self.assertEqual(self.overrides(), {"run": {"effort": "high", "n_concurrent": 3}})
        view = self.ok(self.client.put("/api/config/modes/pilot", json={"n_tasks": 1, "n_attempts": 1}))
        self.assertEqual(self.overrides(), {"run": {"effort": "high", "n_concurrent": 3}})
        self.assertEqual(view["modes_overridden"], [])
        self.ok(self.client.put("/api/config/modes/pilot", json={"n_tasks": 1, "n_attempts": 2}))
        self.assertEqual(self.overrides()["modes"], {"pilot": {"n_attempts": 2}})

    def test_harness_setup_run_settings_save_and_validate(self) -> None:
        self.assertEqual({k: self.run_form()[k] for k in ("prebake", "setup_timeout_multiplier",
                                                          "env_build_timeout_multiplier")},
                         {"prebake": "prefer", "setup_timeout_multiplier": "auto", "env_build_timeout_multiplier": 1.0})
        self.ok(self.client.put("/api/config/run", json={"prebake": "require", "setup_timeout_multiplier": 1.5}))
        self.assertEqual(self.overrides(), {"run": {"prebake": "require", "setup_timeout_multiplier": 1.5}})
        self.assertEqual((self.state.cfg.run.prebake, self.state.cfg.run.setup_timeout_multiplier), ("require", 1.5))
        for body in ({"prebake": "sometimes"}, {"setup_timeout_multiplier": "fast"},
                     {"env_build_timeout_multiplier": 0}):
            with self.subTest(body=body):
                self.assertEqual(self.client.put("/api/config/run", json=body).status_code, 400)
        self.assertEqual(self.overrides(), {"run": {"prebake": "require", "setup_timeout_multiplier": 1.5}})

    def test_arm_save_keeps_type_differing_and_hidden_options(self) -> None:
        # bench.yaml: max_budget_usd: 5 (an int; Harbor types it as a string, so the
        # editor shows '5') and reasoning_effort: high (never shown by the editor).
        form = {"enabled": False, "provider": "anthropic", "model": "claude-opus-5-5", "effort": "inherit",
                "family": "claude-code", "options": {"max_budget_usd": "5", "max_turns": ""}}
        view = self.ok(self.client.put("/api/config/arms/claude-code", json=form))
        self.assertEqual(self.overrides(), {"arms": {"claude-code": {"enabled": False}}})
        arm = next(a for a in view["arms"] if a["name"] == "claude-code")
        self.assertEqual(arm["options"], {"max_budget_usd": 5, "reasoning_effort": "high"})
        self.assertTrue(arm["overridden"])

    def test_arm_save_removes_an_editable_option_left_empty_and_pins_real_changes(self) -> None:
        form = {"enabled": True, "options": {"max_budget_usd": "", "max_turns": 80}}
        self.ok(self.client.put("/api/config/arms/claude-code", json=form))
        self.assertEqual(self.overrides(), {"arms": {"claude-code": {"options": {"max_budget_usd": None,
                                                                                 "max_turns": 80}}}})
        self.assertEqual(self.state.cfg.arm("claude-code").options, {"reasoning_effort": "high", "max_turns": 80})

    def test_arm_save_keeps_overrides_of_fields_it_does_not_send(self) -> None:
        self.ok(self.client.put("/api/config/arms/pi", json={"model": "claude-sonnet-5-5"}))
        self.ok(self.client.put("/api/config/arms/pi", json={"enabled": False}))
        self.assertEqual(self.overrides(), {"arms": {"pi": {"model": "claude-sonnet-5-5", "enabled": False}}})

    def test_family_is_editable_and_its_default_is_not_pinned(self) -> None:
        self.ok(self.client.put("/api/config/arms/pi", json={"family": "pi", "enabled": True}))
        self.ok(self.client.put("/api/config/arms/mini", json={"family": "mini-swe-agent"}))
        self.assertEqual(self.overrides(), {})
        view = self.ok(self.client.put("/api/config/arms/pi", json={"family": "pi-fork"}))
        self.assertEqual(self.overrides(), {"arms": {"pi": {"family": "pi-fork"}}})
        arms = {a["name"]: a for a in view["arms"]}
        self.assertEqual((arms["pi"]["family"], arms["pi"]["kind"]), ("pi-fork", "name"))
        self.assertEqual((arms["mini"]["family"], arms["mini"]["kind"]), ("mini-swe-agent", "import_path"))
        self.assertEqual(arms["mini"]["agent"], IMPORT_PATH)

    def test_orphaned_overrides_are_listed(self) -> None:
        self.overrides_file.write_text(textwrap.dedent("""\
            arms:
              strands: {enabled: false}
              pi: {enabled: false}
            modes:
              block1: {n_tasks: 2}
            """), encoding="utf-8", newline="\n")
        view = self.ok(self.client.post("/api/config/reload"))
        self.assertEqual(view["orphaned_overrides"], {"arms": ["strands"], "modes": ["block1"]})
        self.assertNotIn("block1", view["modes"])
        state = self.ok(self.client.get("/api/state"))
        self.assertEqual(state["config"]["orphaned_overrides"]["arms"], ["strands"])
        # The task scope the REPORT tab compares on, and where it comes from.
        self.assertEqual(state["config"]["tasks"], TASKS)
        self.assertTrue(state["config"]["tasks_file"].endswith("tasks.txt"), state["config"]["tasks_file"])

    def test_a_busy_overrides_file_is_503_not_500(self) -> None:
        with mock.patch.object(fsutil, "atomic_write_text", lambda *a, **k: False):
            response = self.client.put("/api/config/run", json={"effort": "high"})
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json(), {"detail": BUSY})
        self.assertEqual(self.state.cfg.run.effort, "omit")

    def test_bad_bodies_are_400(self) -> None:
        response = self.client.put("/api/config/run", content=b"{not json", headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.put("/api/config/run", json=["effort"]).status_code, 400)
        self.assertEqual(self.client.put("/api/config/run", json={"colour": "red"}).status_code, 400)
        self.assertEqual(self.client.put("/api/config/run", json={"budget_usd": -1}).status_code, 400)
        self.assertEqual(self.client.put("/api/config/arms/ghost", json={"enabled": False}).status_code, 400)
        self.assertEqual(self.client.put("/api/config/arms/pi", json={"model": None}).status_code, 400)
        self.assertEqual(self.client.put("/api/config/arms/pi", json={"model": "  "}).status_code, 400)
        self.assertEqual(self.overrides(), {})

    def test_a_key_save_needs_a_text_value(self) -> None:
        before = os.environ.get("OPENROUTER_API_KEY")
        for body in ({"value": None}, {"value": 5}, {"value": "k-123456789", "x": 1}, {}):
            with self.subTest(body=sorted(body)):
                response = self.client.put("/api/keys/OPENROUTER_API_KEY", json=body)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertTrue(os.environ.get("OPENROUTER_API_KEY") == before)

    def test_resets_remove_overrides(self) -> None:
        self.ok(self.client.put("/api/config/run", json={"effort": "high"}))
        self.ok(self.client.put("/api/config/arms/pi", json={"enabled": False}))
        self.ok(self.client.delete("/api/config/arms/pi"))
        self.assertEqual(self.overrides(), {"run": {"effort": "high"}})
        view = self.ok(self.client.delete("/api/config/overrides"))
        self.assertFalse(view["has_overrides"])


# Schemas, plan and run starts ------------------------------------------------------------------


class Runs(ServerCase):
    def test_the_schema_route_accepts_an_allowed_import_path(self) -> None:
        described = self.ok(self.client.get(f"/api/schemas/{IMPORT_PATH}"))
        self.assertEqual(described["agent"], IMPORT_PATH)
        self.assertIn("cost_limit", {f["name"] for f in described["fields"]})
        encoded = self.ok(self.client.get("/api/schemas/" + IMPORT_PATH.replace(":", "%3A")))
        self.assertEqual(encoded["agent"], IMPORT_PATH)
        self.assertEqual(self.client.get("/api/schemas/some.module:Other").status_code, 404)
        self.assertEqual(self.client.get("/api/schemas/terminus-2").status_code, 404)

    def test_post_runs_is_refused_while_the_queue_is_active(self) -> None:
        with mock.patch.object(self.state.queue, "is_active", lambda: True):
            response = self.client.post("/api/runs", json={"mode": "pilot"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], QUEUE_ACTIVE)
        self.assertEqual(self.manager.started, [])

    def test_post_runs_respects_docker_and_the_global_cap(self) -> None:
        self.docker = False
        self.assertEqual(self.client.post("/api/runs", json={"mode": "pilot"}).status_code, 409)
        self.docker = True
        self.state.queue.update_settings(global_cap_usd=5.0)
        capped = self.client.post("/api/runs", json={"mode": "pilot"})
        self.assertEqual(capped.status_code, 409)
        self.assertIn("global cap", capped.json()["detail"])
        self.assertEqual(self.manager.started, [])

        self.state.queue.update_settings(global_cap_usd=None)
        run = self.ok(self.client.post("/api/runs", json={"mode": "pilot"}))
        self.assertEqual(self.manager.started, [("pilot", ["claude-code", "pi", "mini"])])
        self.assertEqual(run["source"], "config")
        picked = self.client.post("/api/runs", json={"mode": "pilot", "arms": ["pi"]})
        self.assertEqual(picked.status_code, 409)  # the fake's run is still active

    def test_unknown_runs_are_404(self) -> None:
        self.assertEqual(self.client.get("/api/runs/nope/results").status_code, 404)
        self.assertEqual(self.client.get("/api/runs/nope/log/pi").status_code, 404)
        self.assertEqual(self.client.post("/api/runs/nope/cancel").status_code, 409)


# Shutdown -----------------------------------------------------------------------------------


class Shutdown(ServerCase):
    def test_shutdown_runs_each_step_once_in_order(self) -> None:
        order: list[str] = []
        self.patch(self.state.queue, "shutdown", lambda timeout=5: order.append("queue"))
        self.patch(self.state.assistant, "shutdown", lambda timeout=10: order.append("assistant"))
        self.patch(self.manager, "shutdown", lambda timeout=90: order.append("runs"))
        real_close = self.state.ledger.close
        self.patch(self.state.ledger, "close", lambda: (order.append("ledger"), real_close()))
        self.state.shutdown()
        self.state.shutdown()
        self.assertEqual(order, ["queue", "assistant", "runs", "ledger"])
        self.assertFalse(self.state.ledger.available)

    def test_a_step_interrupted_by_ctrl_c_runs_again_and_a_failing_one_does_not_block(self) -> None:
        calls: list[str] = []

        def interrupted(timeout: float = 90) -> None:
            calls.append("runs")
            if calls.count("runs") == 1:
                raise KeyboardInterrupt

        def broken(timeout: float = 5) -> None:
            calls.append("queue")
            raise OSError("queue.json is busy")

        notes: list[str] = []
        self.patch(self.state.queue, "shutdown", broken)
        self.patch(self.manager, "shutdown", interrupted)
        self.patch(server, "_note", notes.append)
        with self.assertRaises(KeyboardInterrupt):
            self.state.shutdown()
        self.state.shutdown()
        self.assertEqual(calls, ["queue", "runs", "runs"])
        self.assertEqual(len(notes), 1)
        self.assertIn("queue.json is busy", notes[0])

    def test_main_finishes_a_shutdown_another_ctrl_c_interrupted(self) -> None:
        # The forced-exit path: uvicorn skipped the lifespan, so main()'s finally
        # does all the stopping, and a further press lands in the queue step.
        import contextlib
        import io

        import uvicorn

        from bench import __main__ as entry
        from bench import procs, schemas

        queue_calls: list[str] = []

        def queue_shutdown(timeout: float = 5) -> None:
            queue_calls.append("queue")
            if len(queue_calls) == 1:
                raise KeyboardInterrupt

        def serve(*args: Any, **kwargs: Any) -> None:
            raise KeyboardInterrupt

        self.patch(self.state.queue, "shutdown", queue_shutdown)
        self.patch(self.state.queue, "start_worker", lambda: None)
        self.patch(self.state.ledger, "start_rescan_thread", lambda: None)
        self.patch(server, "State", lambda port: self.state)
        self.patch(server, "create_app", lambda state: "app")
        self.patch(procs, "install_console_close_handler", lambda callback: True)
        self.patch(schemas, "warm", lambda cfg: None)
        self.patch(uvicorn, "run", serve)
        with contextlib.redirect_stdout(io.StringIO()):
            entry.main(["--no-browser", "--port", str(PORT)])
        self.assertEqual(queue_calls, ["queue", "queue"])
        self.assertIn("shutdown", self.manager.calls)
        self.assertFalse(self.state.ledger.available)

    def test_the_lifespan_shuts_the_state_down(self) -> None:
        with TestClient(create_app(self.state), base_url=BASE_URL) as client:
            client.cookies.set(f"bench_session_{PORT}", self.state.token)
            self.assertEqual(client.get("/api/state").status_code, 200)
        self.assertIn("shutdown", self.manager.calls)
        self.assertFalse(self.state.ledger.available)


if __name__ == "__main__":
    import unittest
    unittest.main()
