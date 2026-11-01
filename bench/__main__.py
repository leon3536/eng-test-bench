"""Start the bench: load .env, apply the git fixes, open the dashboard.

    python -m bench [--profile dev] [--port 8765] [--no-browser]

Everything runs from this one process. Runs started in the dashboard or the
queue are supervised here, and stopping it stops a run in progress and
removes its containers, however it stops:
- Ctrl+C (once is enough): uvicorn's own shutdown is not relied on, because
  a second press makes it skip the app's cleanup. State.shutdown() runs in a
  finally around the server instead, and is carried on if yet another press
  interrupts it.
- Closing the console window, logging off or shutting Windows down: Windows
  ends the process about 5 seconds later and no Python cleanup runs, so a
  console handler kills harbor at once and hands container removal to a
  helper that outlives the bench.
"""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bench")
    parser.add_argument("--profile", help="use config/<profile>.yaml, e.g. dev")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if args.profile:
        os.environ["BENCH_PROFILE"] = args.profile  # read when settings is imported

    import uvicorn

    from . import keys, prebake, procs, schemas, settings
    from .server import State, create_app

    settings.prepare_process_env()
    keys.load_into_process()
    try:
        state = State(args.port)
    except procs.AlreadyRunning as exc:  # a ConfigError too, but not a config problem
        raise SystemExit(str(exc)) from None
    except settings.ConfigError as exc:
        raise SystemExit(f"Config error ({settings.BASE_CONFIG.name} + {settings.OVERRIDES.name}): {exc}") from None
    procs.install_console_close_handler(state.manager.emergency_stop)
    threading.Thread(target=schemas.warm, args=(state.cfg,), name="schemas", daemon=True).start()
    # Records each arm's install once (Harbor's agent code takes 5-13 s to load), so LAUNCH's first plan and the
    # 'Pre-baked images' readiness row do not wait for it.
    threading.Thread(target=prebake.warm, args=(state.cfg,), name="prebake", daemon=True).start()
    state.ledger.start_rescan_thread()
    state.queue.start_worker()  # the queue always loads paused; nothing starts until a person presses Start

    url = f"http://127.0.0.1:{args.port}/?token={state.token}"
    print(f"Engineering Test Bench ({settings.PROFILE_NAME} profile, {settings.BASE_CONFIG.name})")
    print(f"  Dashboard: {url}")
    print("  Ctrl+C stops the bench (one press is enough). A run in progress is stopped and its containers removed.")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    try:
        # timeout_graceful_shutdown: an open assistant stream must not hold up the cleanup.
        uvicorn.run(create_app(state), host="127.0.0.1", port=args.port, log_level="warning",
                    timeout_graceful_shutdown=5)
    except KeyboardInterrupt:
        pass  # the finally below does the stopping; a traceback would only alarm
    finally:
        # Another Ctrl+C during the stopping must not abandon it: the run's
        # worker threads would keep the process alive and the run would go on.
        while True:
            try:
                state.shutdown()
                break
            except KeyboardInterrupt:
                # State.shutdown repeats only the step that was interrupted.
                print("  Still stopping: a run in progress is being stopped and its containers removed...",
                      flush=True)


if __name__ == "__main__":
    main()
