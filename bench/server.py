"""Dashboard: the bench's single control plane, served on 127.0.0.1 only.

Access needs the dashboard URL printed at startup. Its token is new on each
start, stays valid until the bench stops, and sets a session cookie. The
cookie is named for the port, because browsers scope cookies by host and
not by port: two benches on one machine (say the default and dev profiles)
would otherwise sign each other out. Requests from other hosts are refused,
and every write must carry an X-Bench header, which a page on another site
cannot add without a CORS preflight this server never grants.

Besides the config, keys and runs of the LAUNCH tab, it serves the trial
ledger and the pooled report, run designs, the run queue and the run-design
assistant. The queue is the only way a design becomes a billed run, and it
is reachable only through these routes, behind the session cookie. The
assistant is handed read-only snapshots and a draft writer from here, never
the run manager, the queue or the keys.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import secrets
import sqlite3
import sys
import threading
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import pydantic
from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles

from . import checks, designs, estimates, extract, keys, report, results, schemas, settings
from .assistant.types import SAVES_PER_TURN, ToolContext
from .coverage import ReadOnlyLedger
from .designs import DesignNotFound, DesignStore, DraftWriter
from .ledger import EXPORT_COLUMNS, Ledger, LedgerUnavailable
from .runqueue import RunQueue
from .runs import RunManager, plan
from .settings import Config, ConfigError
from .spec import structure_problems

STATIC = Path(__file__).parent / "static"
COOKIE = "bench_session"  # the cookie is COOKIE + "_" + port
RUN_FIELDS = {"effort", "n_concurrent", "parallel_arms", "budget_usd", "per_trial_cap_usd",
              "max_turns", "timeout_multiplier"}
MODE_FIELDS = {"n_tasks", "n_attempts"}
ARM_FIELDS = {"enabled", "provider", "model", "effort", "family"}

BUSY = "A config file is busy (another program has it open). Try again."
LEDGER_BUSY = "The ledger is busy or locked by another program; try again"
QUEUE_ACTIVE = "The queue is active; pause it or add this run to the queue"
DESIGN_FILTERS = {"all": None, "draft": ("draft",), "approved": ("approved",),
                  "queued": ("queued", "running"), "ran": ("ran",), "archived": None}
# TrialRow as the trials list returns it; a single trial adds usage and the raw JSON.
TRIAL_ROW_KEYS = tuple(k for k in EXPORT_COLUMNS if k != "usage")
VERSION_KEYS = ("design_id", "version", "parent_version", "author", "conversation_id", "created_at", "title",
                "purpose", "rationale", "change_note", "spec", "spec_hash", "status", "approved_at", "archived",
                "latest_version")
META_FIELDS = ("title", "purpose", "rationale", "change_note")

SIGN_IN = """<!doctype html><meta charset="utf-8"><title>Test Bench</title>
<body style="font:15px system-ui;margin:3rem;max-width:36rem">
<h1 style="font-size:20px">Not signed in</h1>
<p>Open the dashboard with the URL printed in the console when the bench started.
It contains a token that is valid until the bench is stopped.</p></body>"""


def _same(a: str | None, b: str) -> bool:
    return secrets.compare_digest((a or "").encode(), b.encode())


def _note(message: str) -> None:
    try:
        print(f"[server] {message}", file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass  # a closed console must not turn a note into an error


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validation_lines(exc: pydantic.ValidationError | RequestValidationError) -> list[str]:
    lines = []
    for error in exc.errors():
        loc = ".".join(str(p) for p in error.get("loc", ()) if p not in ("body", "query", "path"))
        lines.append(f"{loc}: {error.get('msg')}" if loc else str(error.get("msg")))
    return lines


# State --------------------------------------------------------------------------


class State:
    """Everything the dashboard serves, for one profile and one port.

    The parts can be passed in (tests do); otherwise each is built over this
    profile's files. The run manager takes the profile's instance lock, so a
    second bench for the same profile fails here, before anything touches
    containers.
    """

    def __init__(self, port: int, *, ledger: Ledger | None = None, manager: Any = None,
                 client_factory: Callable[[], Any] | None = None) -> None:
        self.port = port
        self.token = secrets.token_urlsafe(24)
        self.lock = threading.RLock()          # config reads and writes, one at a time
        self.cfg: Config = settings.load_config()
        self.config_error: str | None = None
        self._stop_lock = threading.RLock()
        self._stopped: set[str] = set()
        if ledger is None:
            ledger = Ledger(settings.LEDGER_DB, roots=extract.default_roots(self.cfg.ledger.extra_scan_roots))
            ledger.open()  # never raises; an unavailable ledger only disables the REPORT tab
        self.ledger = ledger
        try:
            self.manager = manager if manager is not None else RunManager(ledger=ledger)
        except BaseException:
            ledger.close()  # Windows keeps an open database file locked
            raise
        self.reader = ReadOnlyLedger(self.ledger, lambda: self.cfg)
        self.designs = DesignStore(settings.DESIGNS_DIR)
        # Lambdas, not the functions themselves, so a config reload and test
        # patches reach the queue too.
        self.queue = RunQueue(settings.QUEUE_FILE, self.manager, self.designs, lambda: self.cfg,
                              lambda: checks.docker_ok(), estimate_fn=self._queue_estimate,
                              run_spend=lambda: self.manager.list())
        # The queue registers its own listener; this one keeps a design's run
        # list right for any run that carries a design_ref.
        self.manager.add_listener(self._record_design_run)
        self.spend: Any = None
        self.assistant: Any = None
        self.assistant_error: str | None = None
        self._load_assistant(client_factory)

    def _load_assistant(self, client_factory: Callable[[], Any] | None) -> None:
        """The assistant is optional: if the Anthropic SDK or its modules fail
        to load, the dashboard still serves everything else and says why."""
        try:
            from .assistant import tools
            from .assistant.conversation import ConversationStore
            from .assistant.costs import SpendBook
            from .assistant.engine import AssistantEngine
        except Exception as exc:
            self.assistant_error = f"The assistant could not load: {type(exc).__name__}: {exc}"
            _note(self.assistant_error)
            return
        self.spend = SpendBook(settings.ASSISTANT_DIR / "usage.jsonl")
        extra = {"client_factory": client_factory} if client_factory is not None else {}
        self.assistant = AssistantEngine(ConversationStore(settings.ASSISTANT_DIR / "conversations"), self.spend,
                                         self.tool_context, lambda: self.cfg.assistant, lambda: self.cfg,
                                         tools.TOOL_DEFS, tools.execute, **extra)

    def engine(self) -> Any:
        if self.assistant is None:
            raise RuntimeError(self.assistant_error or "The assistant is not available")
        return self.assistant

    # Config ---------------------------------------------------------------------

    def reload(self) -> Config:
        with self.lock:  # never read overrides.yaml while a save is replacing it
            try:
                self.cfg = settings.load_config()
                self.config_error = None
            except ConfigError as exc:
                self.config_error = str(exc)
                raise
            return self.cfg

    def change_overrides(self, change: Callable[[Config], dict[str, Any]], fresh: bool = True) -> Config:
        """Read, change and write the overrides as one step. With fresh, a
        save is refused when the files on disk no longer match what the
        dashboard loaded: its form values would otherwise be diffed against
        the new bench.yaml and pinned as overrides."""
        with self.lock:
            if fresh:
                _require_fresh(self.cfg)
            overrides = change(self.cfg)
            self.cfg = settings.write_overrides(overrides)
            self.config_error = None
            return self.cfg

    # Listeners and adapters ---------------------------------------------------------

    def _record_design_run(self, run: dict[str, Any]) -> None:
        ref = run.get("design_ref") if isinstance(run, Mapping) else None
        if not isinstance(ref, Mapping):
            return
        design_id, version, run_id = ref.get("design_id"), ref.get("version"), run.get("id")
        if not (isinstance(design_id, str) and isinstance(version, int) and isinstance(run_id, str)):
            return
        try:
            self.designs.record_run(design_id, version, run_id)
        except Exception as exc:  # a record, never a reason to fail the run's other listeners
            _note(f"Could not record run {run_id} on {design_id} v{version}: {exc}")

    def _queue_estimate(self, cfg: Config, spec: dict[str, Any]) -> dict[str, Any]:
        return estimates.queue_estimate(cfg, spec, self.reader.rows)

    def tool_context(self, cid: str) -> ToolContext:
        """What one assistant turn may see: snapshots and a draft writer. The
        engine replaces spend and saves_left with the turn's own."""
        cfg = self.cfg
        manager, queue = self.manager, self.queue

        def active_run() -> dict[str, Any] | None:
            run_id = manager.active_id
            if not run_id:
                return None
            try:
                return manager.get(run_id)
            except KeyError:
                return None

        return ToolContext(
            cfg=cfg,
            readiness=lambda: checks.readiness(cfg),
            key_flags=lambda: {name: keys.is_set(name) for name in keys.allowed_names(cfg)},
            runs_snapshot=lambda: manager.list(),
            active_run=active_run,
            queue_snapshot=lambda: queue.view(),
            ledger=self.reader,
            drafts=DraftWriter(self.designs, lambda: self.cfg, cid, queue_status=queue.version_statuses),
            conversation_id=cid,
            spend=lambda: {},
            saves_left=[SAVES_PER_TURN],
        )

    # Designs ------------------------------------------------------------------------

    def ledger_fingerprint(self) -> str | None:
        try:
            return self.reader.fingerprint()
        except Exception:  # an unavailable ledger: staleness then follows the config alone
            return None

    def check_spec(self, cfg: Config, spec: Any) -> dict[str, Any]:
        """CheckResult: validation and plan, and the estimate when the ledger
        can be read. Writes nothing."""
        validation = designs.validate(cfg, spec)
        out: dict[str, Any] = {
            "validation": {k: validation[k] for k in ("status", "problems", "warnings")},
            "plan": validation.get("plan"),
            "estimate": None,
            "estimate_note": None,
            "checked_at": _now(),
            "stale": False,
            "config_fingerprint": designs.config_fingerprint(cfg),
            "ledger_fingerprint": None,
        }
        if structure_problems(spec):
            out["estimate_note"] = "No estimate: the spec is not complete."
            return out
        try:
            rows = self.reader.rows()
            out["ledger_fingerprint"] = self.reader.fingerprint()
            out["estimate"] = estimates.estimate(cfg, spec, rows, validation.get("plan"))
        except Exception as exc:  # validation stands; only the figures need the ledger
            out["estimate_note"] = f"No estimate: {type(exc).__name__}: {exc}"
        return out

    def record_check(self, cfg: Config, design_id: str, version: int, spec: Any) -> dict[str, Any]:
        check = self.check_spec(cfg, spec)
        try:
            self.designs.record_check(design_id, version, check)
        except Exception as exc:  # a cache: the check is still returned
            _note(f"Could not save the check of {design_id} v{version}: {exc}")
        return check

    def _stale(self, check: Any, cfg_fp: str, ledger_fp: str | None) -> bool:
        if not isinstance(check, Mapping) or not check.get("checked_at"):
            return True
        if check.get("config_fingerprint") != cfg_fp:
            return True
        return ledger_fp is not None and check.get("ledger_fingerprint") != ledger_fp

    def version_detail(self, design_id: str, version: int | None = None) -> dict[str, Any]:
        """VersionDetail, with a check that matches today's config and ledger:
        a missing or stale cached check is redone and saved."""
        cfg = self.cfg
        doc = self.designs.version(design_id, version, self.queue.version_statuses())
        check = doc.get("check")
        if self._stale(check, designs.config_fingerprint(cfg), self.ledger_fingerprint()):
            check = self.record_check(cfg, doc["design_id"], doc["version"], doc["spec"])
        else:
            check = {**check, "stale": False}
        out = {k: doc.get(k) for k in VERSION_KEYS}
        out["runs"] = [self._run_brief(run_id) for run_id in doc.get("runs") or []]
        out["check"] = check
        out["diff_from_parent"] = doc.get("diff_from_parent") or []
        return out

    def _run_brief(self, run_id: str) -> dict[str, Any]:
        try:
            run = self.manager.get(run_id)
        except KeyError:
            return {"run_id": run_id, "status": None, "spend_usd": None, "budget_usd": None}
        return {"run_id": run_id, "status": run.get("status"), "spend_usd": run.get("spend_usd"),
                "budget_usd": run.get("budget_usd")}

    def design_summaries(self, which: str) -> list[dict[str, Any]]:
        if which not in DESIGN_FILTERS:
            raise ValueError(f"filter must be one of {', '.join(DESIGN_FILTERS)}")
        listed = self.designs.list(include_archived=which == "archived",
                                   queue_status=self.queue.version_statuses())
        cfg_fp, ledger_fp = designs.config_fingerprint(self.cfg), self.ledger_fingerprint()
        wanted = DESIGN_FILTERS[which]
        out = []
        for summary in listed:
            if which == "archived" and not summary.get("archived"):
                continue
            if wanted is not None and summary.get("status") not in wanted:
                continue
            summary = dict(summary)
            summary["check_stale"] = summary.get("status") == "unreadable" or self._stale(
                {"checked_at": summary.get("checked_at"), "config_fingerprint": summary.get("config_fingerprint"),
                 "ledger_fingerprint": summary.get("ledger_fingerprint")}, cfg_fp, ledger_fp)
            out.append(summary)
        return out

    # Lifetime -----------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop in order: the queue admits nothing more, the assistant's turn
        stops, the run manager stops a run in progress and removes its
        containers, then the ledger closes. Safe to call again: a step that
        finished is not repeated, and one a second Ctrl+C interrupted is."""
        steps: list[tuple[str, Callable[[], Any]]] = [
            ("queue", lambda: self.queue.shutdown()),
            ("assistant", lambda: self.assistant.shutdown() if self.assistant is not None else None),
            ("runs", lambda: self.manager.shutdown()),
            ("ledger", lambda: self.ledger.close()),
        ]
        with self._stop_lock:
            for name, step in steps:
                if name in self._stopped:
                    continue
                try:
                    step()
                except Exception as exc:  # the remaining steps still run
                    _note(f"Stopping {name} failed: {type(exc).__name__}: {exc}")
                self._stopped.add(name)


# Config views and saves ------------------------------------------------------------------


def config_view(cfg: Config) -> dict[str, Any]:
    arm_over = cfg.overrides.get("arms") or {}
    try:
        orphans = settings.orphaned_overrides(cfg.overrides)
    except (ConfigError, OSError):  # a broken bench.yaml is reported elsewhere; the view must load
        orphans = {"arms": [], "modes": []}
    return {
        "profile": settings.PROFILE_NAME,
        "dataset": cfg.dataset,
        "tasks": cfg.tasks,
        "providers": [asdict(p) for p in cfg.providers.values()],
        "modes": {k: asdict(m) for k, m in cfg.modes.items()},
        "run": asdict(cfg.run),
        "run_overridden": sorted((cfg.overrides.get("run") or {}).keys()),
        "modes_overridden": sorted((cfg.overrides.get("modes") or {}).keys()),
        "baseline_arm": cfg.baseline_arm,
        "arms": [
            {**asdict(a), "kind": a.kind, "overridden": a.name in arm_over,
             "model_warnings": schemas.model_warnings(a)}
            for a in cfg.arms
        ],
        "has_overrides": bool(cfg.overrides),
        # Overrides for names bench.yaml no longer has: they would reattach
        # silently if the name came back, so they are shown for a reset.
        "orphaned_overrides": orphans,
    }


def _require_fresh(cfg: Config) -> None:
    try:
        disk = settings.load_config()
        fresh = disk.effective == cfg.effective and disk.overrides == cfg.overrides
    except ConfigError:
        fresh = False
    if not fresh:
        raise RuntimeError(f"{settings.BASE_CONFIG.name} or {settings.OVERRIDES.name} changed on disk since the "
                           "dashboard loaded them. Press Reload config, then save again.")


def _base_arm(name: str) -> dict[str, Any]:
    for arm in settings.base_config().get("arms") or []:
        if isinstance(arm, dict) and arm.get("name") == name:
            return arm
    raise ConfigError(f"No arm named '{name}' in {settings.BASE_CONFIG.name}")


def _diff(base: dict[str, Any], new: dict[str, Any], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    defaults = defaults or {}
    return {k: v for k, v in new.items() if base.get(k, defaults.get(k)) != v}


def _default_family(cfg: Config, agent: str) -> str:
    """The family load_config gives an arm whose YAML names none: the one
    designs.import_paths lists for an import path, else the agent itself."""
    paths = (cfg.effective.get("designs") or {}).get("import_paths") if isinstance(cfg.effective, dict) else None
    value = paths.get(agent) if isinstance(paths, dict) else None
    if isinstance(value, dict):
        value = value.get("family")
    return value.strip() if isinstance(value, str) and value.strip() else agent


def _arm_overrides(cfg: Config, name: str, body: dict[str, Any]) -> dict[str, Any]:
    """The overrides one arm save produces, for the fields the body carries.

    Options are compared with bench.yaml's values coerced by the same schema,
    so a YAML `max_budget_usd: 5` shown as the text '5' is not an override;
    and only options the editor shows can be removed by leaving them out,
    so a hand-written `reasoning_effort` survives every save."""
    base = _base_arm(name)
    current = cfg.arm(name)
    fields = {k: body[k] for k in ARM_FIELDS if k in body}
    if "model" in fields:
        fields["model"] = _text(fields["model"], "model")
    if "family" in fields:
        family = fields["family"]
        if family is not None and not isinstance(family, str):
            raise ValueError("family must be text")
        fields["family"] = (family or "").strip() or _default_family(cfg, current.agent)
    over = _diff(base, fields, {**settings.ARM_DEFAULTS, "family": _default_family(cfg, current.agent)})

    if "options" in body:
        given = body["options"] or {}
        if not isinstance(given, dict):
            raise ValueError("options must be an object")
        schema = schemas.agent_schema(current.agent)
        options = {}
        for key, value in given.items():
            if value is None or value == "":
                continue  # an empty field means "not set"
            options[key] = schemas.coerce(schema, key, value)
        base_opts = base.get("options") or {}

        def unchanged(key: str, value: Any) -> bool:
            if key not in base_opts:
                return False
            try:
                return schemas.coerce(schema, key, base_opts[key]) == value
            except (ValueError, TypeError):
                return False  # a bad base value counts as changed rather than failing the save

        editable = {f["name"] for f in schemas.describe(current.agent)["fields"]}
        opt_over: dict[str, Any] = {k: v for k, v in options.items() if not unchanged(k, v)}
        opt_over.update({k: None for k in base_opts if k not in options and k in editable})  # None removes
        if opt_over:
            over["options"] = opt_over
    return over


# Request bodies ----------------------------------------------------------------------


async def _json_body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        raise ValueError("The request body is not valid JSON") from None
    if not isinstance(data, dict):
        raise ValueError("The request body must be a JSON object")
    return data


Body = Annotated[dict[str, Any], Depends(_json_body)]


def _only(body: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise ValueError(f"Unknown field(s): {', '.join(unknown)}")


def _whole(value: Any, name: str, low: int | None = None, high: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a whole number")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _spec_input(raw: Any) -> Any:
    """Accept a stored NormalizedSpec where input is expected, so an editor
    can send back the version it loaded; normalize re-derives what it holds."""
    if isinstance(raw, Mapping) and "dataset" in raw and not structure_problems(raw):
        return designs.to_input(raw)
    return raw


def _meta_in(body: Mapping[str, Any], **defaults: str) -> designs.DesignMetaIn:
    values = {**defaults, **{k: body[k] for k in META_FIELDS if k in body}}
    return designs.DesignMetaIn.model_validate(values)


def _report_args(state: State, body: Mapping[str, Any], extra: tuple[str, ...] = ()) -> tuple[Any, Any]:
    _only(body, {"filters", "options"} | set(extra))
    return report.parse_filters(body.get("filters")), report.parse_options(body.get("options"), state.cfg)


def _csv_text(rows: Any) -> str:
    """CSV for Excel: a BOM, so it reads UTF-8, and CRLF line endings."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")
    for row in rows:
        writer.writerow(row)
    return "\ufeff" + buf.getvalue()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resume_from(after: int, last_event_id: str | None) -> int:
    if last_event_id:
        try:
            return max(0, int(last_event_id.strip()))
        except ValueError:
            pass  # not one of ours: fall back to ?after
    return max(0, after)


# The guard -----------------------------------------------------------------------------


class _Guard:
    """Host check, session cookie and the X-Bench header, as plain ASGI so
    streamed responses (exports, the assistant's events) pass through as
    they are produced."""

    def __init__(self, app: Any, state: State, cookie: str) -> None:
        self.app = app
        self.state = state
        self.cookie = cookie
        self.hosts = {f"127.0.0.1:{state.port}", f"localhost:{state.port}"}

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        refused: Response | None = None
        if request.headers.get("host") not in self.hosts:
            refused = PlainTextResponse("Forbidden host", status_code=403)
        elif request.url.path.startswith("/api/"):
            if not _same(request.cookies.get(self.cookie), self.state.token):
                refused = JSONResponse({"detail": "Not signed in. Open the URL printed in the console."}, 401)
            elif request.method not in ("GET", "HEAD") and request.headers.get("x-bench") != "1":
                refused = JSONResponse({"detail": "Missing X-Bench header"}, 403)
        if refused is not None:
            await refused(scope, receive, send)
            return
        await self.app(scope, receive, send)


# The app --------------------------------------------------------------------------------


class _Static(StaticFiles):
    """The front end's files, revalidated on every load (a 304 when unchanged), so a
    browser never mixes an updated file with cached copies of the others."""

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(state: State) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        state.shutdown()  # __main__ calls it again in a finally, for when uvicorn skips this

    app = FastAPI(title="Engineering Test Bench", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    cookie = f"{COOKIE}_{state.port}"
    app.add_middleware(_Guard, state=state, cookie=cookie)
    _add_error_handlers(app)

    @app.get("/")
    def index(request: Request, token: str | None = None):
        if token is not None:
            if not _same(token, state.token):
                return HTMLResponse(SIGN_IN, 401)
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(cookie, state.token, httponly=True, samesite="strict")
            return response
        if not _same(request.cookies.get(cookie), state.token):
            return HTMLResponse(SIGN_IN, 401)
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/static", _Static(directory=STATIC), name="static")

    _state_routes(app, state)
    _config_routes(app, state)
    _run_routes(app, state)
    _ledger_routes(app, state)
    _report_routes(app, state)
    _design_routes(app, state)
    _queue_routes(app, state)
    _assistant_routes(app, state)
    return app


def _add_error_handlers(app: FastAPI) -> None:
    async def bad_request(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, 400)

    async def invalid(_request: Request, exc: Exception) -> JSONResponse:
        lines = _validation_lines(exc)  # type: ignore[arg-type]
        return JSONResponse({"detail": "; ".join(lines) or "Invalid request"}, 400)

    async def conflict(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, 409)

    async def not_found(_request: Request, exc: Exception) -> JSONResponse:
        arg = exc.args[0] if exc.args else ""
        if isinstance(exc, DesignNotFound) or (isinstance(arg, str) and " " in arg):
            detail = str(exc) if isinstance(exc, DesignNotFound) else arg
        else:
            detail = f"Not found: {arg}"
        return JSONResponse({"detail": detail}, 404)

    async def busy(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": BUSY}, 503)

    async def ledger_unavailable(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc) or "The ledger is not available"}, 503)

    async def ledger_busy(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": LEDGER_BUSY}, 503)

    # Starlette picks the handler of the nearest class in the exception's MRO,
    # so DesignNotFound (a KeyError) and VersionConflict (a RuntimeError) need none of their own.
    app.add_exception_handler(ConfigError, bad_request)
    app.add_exception_handler(ValueError, bad_request)
    app.add_exception_handler(pydantic.ValidationError, invalid)
    app.add_exception_handler(RequestValidationError, invalid)
    app.add_exception_handler(RuntimeError, conflict)
    app.add_exception_handler(KeyError, not_found)
    app.add_exception_handler(PermissionError, busy)
    app.add_exception_handler(LedgerUnavailable, ledger_unavailable)
    app.add_exception_handler(sqlite3.OperationalError, ledger_busy)


def _safely(fn: Callable[[], Any], what: str) -> Any:
    """One part of /api/state that fails must not blank the whole dashboard."""
    try:
        return fn()
    except Exception as exc:
        return {"error": f"{what}: {type(exc).__name__}: {exc}"}


def _ledger_block(state: State) -> dict[str, Any]:
    status = state.ledger.status()
    return {"available": status["available"], "error": status["error"], "trials": status["trials"],
            "last_ingest_at": status["last_ingest_at"], "rescan_running": status["rescan"]["running"]}


def _assistant_block(state: State) -> dict[str, Any]:
    if state.assistant is None:
        return {"enabled": state.cfg.assistant.enabled, "available": False, "reason": state.assistant_error,
                "today_usd": None, "daily_cap_usd": state.cfg.assistant.daily_cap_usd, "active_turn": None}
    status = state.assistant.status()
    return {k: status.get(k) for k in ("enabled", "available", "reason", "today_usd", "daily_cap_usd",
                                       "active_turn")}


def _state_routes(app: FastAPI, state: State) -> None:
    @app.get("/api/state")
    def get_state():
        cfg = state.cfg
        return {
            "readiness": checks.readiness(cfg),
            "keys": keys.status(cfg),
            "config": config_view(cfg),
            "config_error": state.config_error,
            "runs": state.manager.list(),
            "active_run": state.manager.active_id,
            "ledger": _safely(lambda: _ledger_block(state), "ledger"),
            "queue": _safely(state.queue.summary, "queue"),
            "assistant": _safely(lambda: _assistant_block(state), "assistant"),
        }

    @app.get("/api/schemas/{agent:path}")
    def get_schema(agent: str):
        if agent not in {h["agent"] for h in designs.allowed_harnesses(state.cfg)}:
            raise KeyError(agent)
        return schemas.describe(agent)


def _config_routes(app: FastAPI, state: State) -> None:
    def provider_name(env_key: str) -> str:
        cfg = state.cfg
        for p in cfg.providers.values():
            if p.env_key == env_key:
                return p.name
        if env_key == keys.ASSISTANT_KEY and env_key in keys.allowed_names(cfg):
            return "anthropic"  # the assistant's key, when no provider uses it (dev profile)
        raise KeyError(env_key)

    @app.put("/api/keys/{env_key}")
    def put_key(env_key: str, body: Body):
        _only(body, {"value"})
        value = body.get("value")
        # str(None) would save the text "None" over a working key.
        if not isinstance(value, str):
            raise ValueError("value must be text")
        keys.set_secret(state.cfg, env_key, value)
        return keys.status(state.cfg)

    @app.delete("/api/keys/{env_key}")
    def delete_key(env_key: str):
        keys.clear_secret(state.cfg, env_key)
        return keys.status(state.cfg)

    @app.post("/api/keys/{env_key}/test")
    def test_key(env_key: str):
        name = provider_name(env_key)
        value = keys.get_secret(env_key)
        if not value:
            raise ValueError(f"{env_key} is not set")
        return checks.test_key(state.cfg, name, value)

    @app.put("/api/config/run")
    def put_run(body: Body):
        unknown = set(body) - RUN_FIELDS
        if unknown:
            raise ValueError(f"Unknown run settings: {', '.join(sorted(unknown))}")

        def change(cfg: Config) -> dict[str, Any]:
            over = copy.deepcopy(cfg.overrides)
            base_run = settings.base_config().get("run") or {}
            merged = {**(over.get("run") or {}), **body}
            over["run"] = _diff(base_run, {k: v for k, v in merged.items() if v != ""}, settings.RUN_DEFAULTS)
            return over

        return config_view(state.change_overrides(change))

    @app.delete("/api/config/run")
    def reset_run():
        def change(cfg: Config) -> dict[str, Any]:
            over = copy.deepcopy(cfg.overrides)
            over.pop("run", None)
            return over

        return config_view(state.change_overrides(change, fresh=False))

    @app.delete("/api/config/modes")
    def reset_modes():
        def change(cfg: Config) -> dict[str, Any]:
            over = copy.deepcopy(cfg.overrides)
            over.pop("modes", None)
            return over

        return config_view(state.change_overrides(change, fresh=False))

    @app.put("/api/config/modes/{mode}")
    def put_mode(mode: str, body: Body):
        if mode not in state.cfg.modes or set(body) - MODE_FIELDS:
            raise ValueError("Unknown mode or field")

        def change(cfg: Config) -> dict[str, Any]:
            over = copy.deepcopy(cfg.overrides)
            base_mode = (settings.base_config().get("modes") or {}).get(mode) or {}
            modes = over.setdefault("modes", {})
            modes[mode] = _diff(base_mode, {**(modes.get(mode) or {}), **body}, settings.MODE_DEFAULTS)
            return over

        return config_view(state.change_overrides(change))

    @app.put("/api/config/arms/{name}")
    def put_arm(name: str, body: Body):
        def change(cfg: Config) -> dict[str, Any]:
            over = copy.deepcopy(cfg.overrides)
            arms = over.setdefault("arms", {})
            # Fields the body leaves out keep their overrides.
            covered = (ARM_FIELDS | {"options"}) & set(body)
            kept = {k: v for k, v in (arms.get(name) or {}).items() if k not in covered}
            arms[name] = {**kept, **_arm_overrides(cfg, name, body)}
            return over

        return config_view(state.change_overrides(change))

    @app.delete("/api/config/arms/{name}")
    def reset_arm(name: str):
        def change(cfg: Config) -> dict[str, Any]:
            over = copy.deepcopy(cfg.overrides)
            (over.get("arms") or {}).pop(name, None)
            return over

        return config_view(state.change_overrides(change, fresh=False))

    @app.delete("/api/config/overrides")
    def reset_all():
        return config_view(state.change_overrides(lambda _cfg: {}, fresh=False))

    @app.post("/api/config/reload")
    def reload_config():
        return config_view(state.reload())


def _run_routes(app: FastAPI, state: State) -> None:
    def arm_names(raw: str | None) -> list[str]:
        if raw:
            return [n for n in raw.split(",") if n]
        return [a.name for a in state.cfg.arms if a.enabled]

    @app.get("/api/plan")
    def get_plan(mode: str, arms: str | None = None):
        return plan(state.cfg, mode, arm_names(arms))

    @app.post("/api/runs")
    def start_run(body: Body):
        cfg = state.cfg
        if state.queue.is_active():
            raise RuntimeError(QUEUE_ACTIVE)
        if not checks.docker_ok():
            raise RuntimeError("Docker is not running")
        cap = state.queue.view().get("global_cap_usd")
        if cap is not None:
            spent = state.queue.cap_spent()
            if spent + cfg.run.budget_usd > cap:
                raise RuntimeError(f"The global cap of ${cap:.2f} has ${max(0.0, cap - spent):.2f} left; this run's "
                                   f"budget is ${cfg.run.budget_usd:.2f}. Raise or reset the cap in the queue.")
        names = body.get("arms") or arm_names(None)
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError("arms must be a list of arm names")
        return state.manager.start(cfg, str(body.get("mode", "")), list(names))

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        state.manager.cancel(run_id)
        return {"ok": True}

    @app.get("/api/runs/{run_id}/results")
    def run_results(run_id: str):
        return results.run_results(state.cfg, state.manager.get(run_id))

    @app.get("/api/runs/{run_id}/log/{arm}")
    def run_log(run_id: str, arm: str, lines: int = 200):
        path = state.manager.log_path(run_id, arm)
        if not path.exists():
            return PlainTextResponse("")
        with open(path, encoding="utf-8", errors="replace") as f:
            tail = deque(f, maxlen=max(1, min(lines, 2000)))
        return PlainTextResponse("".join(tail))


def _ledger_routes(app: FastAPI, state: State) -> None:
    @app.get("/api/ledger/status")
    def ledger_status():
        return state.ledger.status()

    @app.post("/api/ledger/rescan")
    def ledger_rescan(body: Body):
        _only(body, {"reparse"})
        return state.ledger.rescan(reparse=_flag(body.get("reparse", False), "reparse"))

    @app.get("/api/ledger/facets")
    def ledger_facets():
        return state.ledger.facets(state.cfg.tasks)

    @app.get("/api/ledger/runs")
    def ledger_runs():
        return state.ledger.runs()

    @app.post("/api/ledger/trials")
    def ledger_trials(body: Body):
        _only(body, {"filters", "outcomes", "limit", "offset", "order"})
        outcomes = body.get("outcomes")
        if outcomes is not None and (not isinstance(outcomes, list) or not all(isinstance(o, str) for o in outcomes)):
            raise ValueError("outcomes must be a list of outcome names")
        order = body.get("order") or "started_at DESC"
        if not isinstance(order, str):
            raise ValueError("order must be text such as 'started_at DESC'")
        limit = body.get("limit")
        total, rows = state.ledger.query(report.parse_filters(body.get("filters")), outcomes=outcomes,
                                         limit=100 if limit is None else limit, offset=body.get("offset", 0),
                                         order=order)
        return {"total": total, "rows": [{k: r.get(k) for k in TRIAL_ROW_KEYS} for r in rows]}

    @app.get("/api/ledger/trials/{trial_id}")
    def ledger_trial(trial_id: str):
        return state.ledger.trial(trial_id)

    @app.post("/api/ledger/exclusions")
    def add_exclusion(body: Body):
        _only(body, {"target_type", "target_id", "action", "reason"})
        row = state.ledger.add_exclusion(body.get("target_type"), body.get("target_id"),
                                         body.get("action", "exclude"), body.get("reason"))
        return JSONResponse(row, 201)

    @app.get("/api/ledger/exclusions")
    def list_exclusions():
        return {"exclusions": state.ledger.exclusions()}

    @app.post("/api/ledger/export")
    def ledger_export(body: Body):
        _only(body, {"filters", "format"})
        fmt = body.get("format", "csv")
        # Validated and queried here, before the response starts.
        chunks = state.ledger.export_rows(report.parse_filters(body.get("filters")), fmt)
        if fmt == "jsonl":
            return StreamingResponse(chunks, media_type="application/x-ndjson",
                                     headers={"Content-Disposition": f'attachment; filename="ledger-{_stamp()}.jsonl"'})
        return StreamingResponse(chunks, media_type="text/csv; charset=utf-8",
                                 headers={"Content-Disposition": f'attachment; filename="ledger-{_stamp()}.csv"'})


def _report_routes(app: FastAPI, state: State) -> None:
    # POST only to carry nested JSON: none of these changes anything.
    @app.post("/api/report")
    def get_report(body: Body):
        f, o = _report_args(state, body)
        return report.build_report(state.ledger, state.cfg, f, o)

    @app.post("/api/report/coverage")
    def get_coverage(body: Body):
        f, o = _report_args(state, body)
        return report.build_coverage(state.ledger, state.cfg, f, o)

    @app.post("/api/report/pairs")
    def get_pairs(body: Body):
        f, o = _report_args(state, body, ("comparison",))
        return report.build_pairs(state.ledger, state.cfg, f, o, _text(body.get("comparison"), "comparison"))

    @app.post("/api/report/export")
    def export_report(body: Body):
        f, o = _report_args(state, body, ("level",))
        level = body.get("level", "comparisons")
        rows = list(report.export_rows(state.ledger, state.cfg, f, o, level))
        fingerprint = report.build_report(state.ledger, state.cfg, f, o)["provenance"]["fingerprint"]
        short = str(fingerprint).split(":", 1)[-1][:8]
        return Response(_csv_text(rows), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="bench-report-{short}-{level}.csv"'})

    @app.post("/api/report/snapshots")
    def save_snapshot(body: Body):
        f, o = _report_args(state, body, ("name",))
        name = body.get("name")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError("Give the snapshot a name of 1 to 80 characters")
        response = report.build_report(state.ledger, state.cfg, f, o)
        saved = state.ledger.save_snapshot(name, copy.deepcopy(response["spec"]), response)
        return JSONResponse(saved, 201)

    @app.get("/api/report/snapshots")
    def list_snapshots():
        return {"snapshots": state.ledger.snapshots()}

    @app.get("/api/report/snapshots/{snapshot_id}")
    def get_snapshot(snapshot_id: str):
        return state.ledger.snapshot(snapshot_id)


def _design_routes(app: FastAPI, state: State) -> None:
    def created(detail: dict[str, Any]) -> JSONResponse:
        return JSONResponse(detail, 201)

    def save_new(meta_in: designs.DesignMetaIn, spec: Any) -> JSONResponse:
        cfg = state.cfg
        doc = state.designs.create(cfg, meta_in, spec, "user")
        state.record_check(cfg, doc["design_id"], doc["version"], doc["spec"])
        return created(state.version_detail(doc["design_id"], doc["version"]))

    @app.get("/api/designs")
    def list_designs(filter_: Annotated[str, Query(alias="filter")] = "all"):
        return {"designs": state.design_summaries(filter_)}

    @app.post("/api/designs")
    def create_design(body: Body):
        _only(body, {*META_FIELDS, "spec"})
        meta_in = _meta_in(body)
        return save_new(meta_in, designs.normalize(state.cfg, _spec_input(body.get("spec"))))

    @app.post("/api/designs/check")
    def check_design(body: Body):
        _only(body, {"spec"})
        cfg = state.cfg
        try:
            spec = designs.normalize(cfg, _spec_input(body.get("spec")))
        except pydantic.ValidationError as exc:
            problems = _validation_lines(exc)
        except ValueError as exc:
            problems = [str(exc)]
        else:
            return state.check_spec(cfg, spec)
        # A live check of a half-filled form: say what is wrong, not 400.
        return {"validation": {"status": "invalid", "problems": problems, "warnings": []}, "plan": None,
                "estimate": None, "estimate_note": "No estimate: the spec is not complete.", "checked_at": _now(),
                "stale": False, "config_fingerprint": designs.config_fingerprint(cfg), "ledger_fingerprint": None}

    @app.post("/api/designs/from-config")
    def design_from_config(body: Body):
        _only(body, {"mode", "arms", "title"})
        mode = _text(body.get("mode"), "mode")
        arms = body.get("arms")
        if arms is not None and (not isinstance(arms, list) or not all(isinstance(a, str) for a in arms)):
            raise ValueError("arms must be a list of arm names, or null for the enabled arms")
        spec = designs.design_from_config(state.cfg, mode, arms)
        title = body.get("title") or f"{mode} from config"
        return save_new(designs.DesignMetaIn(title=title), spec)

    @app.post("/api/designs/from-run/{run_id}")
    def design_from_run(run_id: str, body: Body):
        _only(body, {"title"})
        run = state.manager.get(run_id)
        title = body.get("title") or f"Run again: {run_id}"
        return save_new(designs.DesignMetaIn(title=title, purpose=f"Repeats run {run_id}."),
                        designs.design_from_run(run))

    @app.get("/api/designs/{design_id}")
    def get_design(design_id: str):
        meta = state.designs.meta(design_id, state.queue.version_statuses())
        return {"meta": meta, "current": state.version_detail(design_id)}

    @app.get("/api/designs/{design_id}/versions/{version}")
    def get_version(design_id: str, version: int):
        return state.version_detail(design_id, version)

    @app.post("/api/designs/{design_id}/versions")
    def add_version(design_id: str, body: Body):
        _only(body, {*META_FIELDS, "spec", "base_version"})
        base = _whole(body.get("base_version"), "base_version", 1)
        cfg = state.cfg
        spec = designs.normalize(cfg, _spec_input(body.get("spec")))
        doc = state.designs.add_version(cfg, design_id, base, _meta_in(body), spec, "user")
        state.record_check(cfg, doc["design_id"], doc["version"], doc["spec"])
        return created(state.version_detail(design_id, doc["version"]))

    @app.post("/api/designs/{design_id}/archive")
    def archive_design(design_id: str):
        pending = [v for (d, v), _s in state.queue.version_statuses().items() if d == design_id]
        if pending:
            raise RuntimeError(f"{design_id} has version(s) {', '.join(map(str, sorted(pending)))} in the queue. "
                               "Remove them from the queue first.")
        state.designs.archive(design_id)
        return {"meta": state.designs.meta(design_id, state.queue.version_statuses())}

    @app.post("/api/designs/{design_id}/unarchive")
    def unarchive_design(design_id: str):
        state.designs.unarchive(design_id)
        return {"meta": state.designs.meta(design_id, state.queue.version_statuses())}


def _queue_routes(app: FastAPI, state: State) -> None:
    @app.get("/api/queue")
    def get_queue():
        return state.queue.view()

    @app.post("/api/queue/items")
    def add_item(body: Body):
        _only(body, {"design_id", "version", "spec_hash", "position", "start"})
        design_id = _text(body.get("design_id"), "design_id")
        version = _whole(body.get("version"), "version", 1)
        start = _flag(body.get("start", False), "start")
        if start and not checks.docker_ok():
            raise RuntimeError("Docker Desktop is not running. Start it, then add and start the design.")
        item = state.queue.add(design_id, version, body.get("spec_hash"), body.get("position", "end"))
        if start:
            try:
                state.queue.start()
            except RuntimeError as exc:
                raise RuntimeError(f"Added as {item['id']}, but the queue did not start: {exc}") from None
        return JSONResponse({"item": item, "queue": state.queue.view()}, 201)

    @app.delete("/api/queue/items/{item_id}")
    def remove_item(item_id: str):
        return state.queue.remove(item_id)

    @app.post("/api/queue/items/{item_id}/move")
    def move_item(item_id: str, body: Body):
        _only(body, {"to_index"})
        return state.queue.move(item_id, _whole(body.get("to_index"), "to_index"))

    @app.post("/api/queue/items/{item_id}/retry")
    def retry_item(item_id: str):
        return state.queue.retry(item_id)

    @app.post("/api/queue/start")
    def start_queue():
        return state.queue.start()

    @app.post("/api/queue/pause")
    def pause_queue():
        return state.queue.pause()

    @app.put("/api/queue/settings")
    def queue_settings(body: Body):
        _only(body, {"global_cap_usd", "reset_cap", "on_failure"})
        return state.queue.update_settings(**body)


def _assistant_routes(app: FastAPI, state: State) -> None:
    def require_conversation(cid: str) -> None:
        # A dependency, because a streamed route's body runs only after the
        # 200 has been sent, too late for a 404.
        state.engine().require(cid)

    @app.get("/api/assistant/status")
    def assistant_status():
        if state.assistant is None:
            s = state.cfg.assistant
            return {"enabled": s.enabled, "available": False, "reason": state.assistant_error,
                    "model": None, "effort_default": s.effort, "fallbacks": s.fallbacks,
                    "key_set": keys.is_set(keys.ASSISTANT_KEY),
                    "spend": {"today_usd": None, "daily_cap_usd": s.daily_cap_usd,
                              "per_turn_cap_usd": s.per_turn_cap_usd},
                    "active_turn": None}
        return state.assistant.status()

    @app.get("/api/assistant/conversations")
    def list_conversations():
        return {"conversations": state.engine().conversations()}

    @app.post("/api/assistant/conversations")
    def create_conversation(body: Body):
        _only(body, {"title", "effort", "design"})
        meta = state.engine().create_conversation(body.get("title"), body.get("effort"), body.get("design"))
        return JSONResponse(meta, 201)

    @app.get("/api/assistant/conversations/{cid}")
    def get_conversation(cid: str, after: int = 0):
        return state.engine().conversation(cid, max(0, after))

    @app.get("/api/assistant/conversations/{cid}/stream", response_class=EventSourceResponse,
             dependencies=[Depends(require_conversation)])
    async def stream_conversation(cid: str, after: int = 0,
                                  last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
                                  ) -> AsyncIterator[ServerSentEvent]:
        async for event in state.engine().stream(cid, _resume_from(after, last_event_id)):
            yield event

    @app.post("/api/assistant/conversations/{cid}/messages")
    def send_message(cid: str, body: Body):
        _only(body, {"text", "attach_design"})
        accepted = state.engine().start_turn(cid, body.get("text"), body.get("attach_design"))
        return JSONResponse(accepted, 202)

    @app.post("/api/assistant/conversations/{cid}/stop")
    def stop_turn(cid: str):
        engine = state.engine()
        engine.require(cid)
        engine.stop(cid)
        return {"ok": True}

    @app.post("/api/assistant/conversations/{cid}/retry")
    def retry_turn(cid: str):
        return JSONResponse(state.engine().retry_turn(cid), 202)

    @app.post("/api/assistant/conversations/{cid}/continue")
    def continue_conversation(cid: str):
        return JSONResponse(state.engine().continue_conversation(cid), 201)

    @app.post("/api/assistant/conversations/{cid}/archive")
    def archive_conversation(cid: str):
        state.engine().archive(cid)
        return {"ok": True}

    @app.get("/api/assistant/usage")
    def assistant_usage(days: int = 30):
        return state.engine().usage(_whole(days, "days", 1, 366))
