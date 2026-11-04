"""python -m bench: the process entry point.

uvicorn, the State and the console handler are stand-ins, so main() runs to
the end without serving anything, starting a run or registering a real
Windows handler. What matters is how the process stops: state.shutdown()
must run however uvicorn returns (a second Ctrl+C makes uvicorn skip the
app's own cleanup) and must finish even when another Ctrl+C interrupts it,
closing the console must reach the run manager's
emergency stop, and a second bench for a profile must exit with the plain
lock message instead of a traceback.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any
from unittest import mock

from tests.support import BenchTestCase

from bench import __main__ as entry
from bench import procs, schemas, server, settings


class FakeState:
    """What main() uses of State, recording calls in order."""

    error: BaseException | None = None
    interrupted_shutdowns = 0  # how many shutdown() calls a further Ctrl+C interrupts

    def __init__(self, port: int) -> None:
        if FakeState.error is not None:
            raise FakeState.error
        self.port = port
        self.token = "test-token"
        self.cfg = object()
        self.events: list[str] = []
        self.manager = mock.Mock(name="manager")
        self.ledger = mock.Mock(name="ledger")
        self.queue = mock.Mock(name="queue")
        self.ledger.start_rescan_thread.side_effect = lambda: self.events.append("rescan")
        self.queue.start_worker.side_effect = lambda: self.events.append("queue worker")

    def shutdown(self) -> None:
        if FakeState.interrupted_shutdowns > 0:
            FakeState.interrupted_shutdowns -= 1
            self.events.append("shutdown interrupted")
            raise KeyboardInterrupt
        self.events.append("shutdown")


class Main(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        FakeState.error = None
        FakeState.interrupted_shutdowns = 0
        self.addCleanup(setattr, FakeState, "error", None)
        self.addCleanup(setattr, FakeState, "interrupted_shutdowns", 0)
        self.states: list[FakeState] = []
        self.handlers: list[Any] = []
        self.uvicorn_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.uvicorn_effect: BaseException | None = None

        def make_state(port: int) -> FakeState:
            state = FakeState(port)
            self.states.append(state)
            return state

        def install(callback: Any) -> bool:
            self.handlers.append(callback)
            if self.states:
                self.states[-1].events.append("console handler")
            return True

        def run(*args: Any, **kwargs: Any) -> None:
            self.uvicorn_calls.append((args, kwargs))
            self.states[-1].events.append("serve")
            if self.uvicorn_effect is not None:
                raise self.uvicorn_effect

        import uvicorn
        for target, name, value in ((server, "State", make_state), (server, "create_app", lambda state: "app"),
                                    (procs, "install_console_close_handler", install), (uvicorn, "run", run),
                                    (schemas, "warm", lambda cfg: None)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def main(self) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            entry.main(["--no-browser", "--port", "8797"])
        return out.getvalue()

    def test_ctrl_c_in_uvicorn_still_shuts_the_state_down(self) -> None:
        self.uvicorn_effect = KeyboardInterrupt()
        self.main()
        self.assertEqual(self.states[0].events[-2:], ["serve", "shutdown"])

    def test_another_ctrl_c_during_shutdown_does_not_abandon_it(self) -> None:
        # uvicorn force-exits on a second press; a third one lands in the stopping.
        self.uvicorn_effect = KeyboardInterrupt()
        FakeState.interrupted_shutdowns = 1
        printed = self.main()  # returns: the KeyboardInterrupt does not escape
        events = self.states[0].events
        self.assertEqual(events[-3:], ["serve", "shutdown interrupted", "shutdown"])
        self.assertEqual(sum(e.startswith("shutdown") for e in events), 2)
        self.assertIn("Still stopping", printed)

    def test_a_normal_return_and_a_crash_both_shut_down(self) -> None:
        self.main()
        self.assertEqual(self.states[-1].events.count("shutdown"), 1)
        self.uvicorn_effect = RuntimeError("port in use")
        with self.assertRaises(RuntimeError):
            self.main()
        self.assertEqual(self.states[-1].events[-1], "shutdown")

    def test_uvicorn_bounds_its_graceful_shutdown(self) -> None:
        self.main()
        args, kwargs = self.uvicorn_calls[0]
        self.assertEqual(args, ("app",))
        self.assertEqual(kwargs, {"host": "127.0.0.1", "port": 8797, "log_level": "warning",
                                  "timeout_graceful_shutdown": 5})

    def test_console_close_reaches_the_emergency_stop_before_serving(self) -> None:
        printed = self.main()
        state = self.states[0]
        self.assertEqual(len(self.handlers), 1)
        self.assertIs(self.handlers[0], state.manager.emergency_stop)
        self.assertEqual(state.events, ["console handler", "rescan", "queue worker", "serve", "shutdown"])
        self.assertIn("http://127.0.0.1:8797/?token=test-token", printed)
        self.assertIn("Ctrl+C stops the bench (one press is enough).", printed)

    def test_a_second_bench_exits_with_the_lock_message(self) -> None:
        FakeState.error = procs.AlreadyRunning()
        with self.assertRaises(SystemExit) as caught:
            self.main()
        self.assertEqual(caught.exception.code, procs.ALREADY_RUNNING)
        self.assertEqual((self.uvicorn_calls, self.handlers), ([], []))

    def test_a_config_error_names_both_files(self) -> None:
        FakeState.error = settings.ConfigError("run.budget_usd must be a number")
        with self.assertRaises(SystemExit) as caught:
            self.main()
        self.assertEqual(caught.exception.code, f"Config error ({settings.BASE_CONFIG.name} + "
                                                f"{settings.OVERRIDES.name}): run.budget_usd must be a number")
        self.assertEqual(self.uvicorn_calls, [])


if __name__ == "__main__":
    import unittest
    unittest.main()
