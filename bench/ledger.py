"""The trial ledger: every trial of every run, in one SQLite file.

Reports pool many runs ("run, run, run, where are we?"), so trials cannot
live only in per-run folders. The ledger keeps one row per Harbor trial from
any profile, from dashboard runs and plain `harbor run` jobs alike, including
trials killed before they finished. Rows hold facts (tokens per model, the
exception, timings, reward, task digest, config identity, the harness's own
cost and the cost at ingest) plus a redacted raw copy of result.json, so
judgements such as the outcome class are re-derived when the rules change.

Disk stays the truth. Ingestion is idempotent on Harbor's (job_id,
trial_name), runs from three places (live refresh, arm finalize, rescan) and
never overwrites a finished row or deletes anything. Runs never depend on it:
every call from the run manager is guarded, and budgets use results.spend.

SQLite because it is stdlib, transactional and shareable between the default
and dev bench processes. Connections are short-lived (Windows cannot replace
or delete an open file), writes take one process-wide lock plus BEGIN
IMMEDIATE, and WAL lets reports read while arms write.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import io
import json
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import IO, Any

from . import classify, extract, pricing, settings
from .extract import RunContext, ScanRoot, TrialFacts

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 15000
# A live ingest runs inside the run manager's 3-second refresh, which also
# enforces budgets and cancels, so it gives up quickly; the next refresh or
# rescan catches up.
LIVE_BUSY_MS = 1000
BACKUP_KEEP = 7
MAX_LIMIT = 1000
FINAL_READ_RETRIES = 3
FINAL_READ_DELAY = 0.2
REPLACE_PATIENCE = 2.0

# Exclusions the ledger records on first open, once per target. A later
# 'include' sticks, because seeding only happens when no row ever existed.
SEED_EXCLUSIONS: list[tuple[str, str, str]] = [
    ("run", "dev/20261028-222401-pilot", "Pilot 1 ran on git tasks damaged by CRLF line endings"),
]

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
# One writer at a time in this process; SQLite serialises across processes.
_WRITE_LOCK = threading.RLock()


class LedgerUnavailable(Exception):
    """The ledger could not be opened. Deliberately not a RuntimeError, which
    the server maps to 409; this one maps to 503."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime(_TIME_FORMAT)


# Schema ----------------------------------------------------------------------

_SCHEMA_V1 = (
    """CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)""",
    """CREATE TABLE runs (
        run_key TEXT PRIMARY KEY,
        profile TEXT NOT NULL,
        run_id TEXT NOT NULL,
        source TEXT NOT NULL CHECK (source IN ('bench','external')),
        mode TEXT, status TEXT, created_at TEXT, finished_at TEXT, budget_usd REAL, baseline_arm TEXT,
        tasks_json TEXT, plan_json TEXT, config_json TEXT, dataset_json TEXT,
        first_seen_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    """CREATE TABLE trials (
        id INTEGER PRIMARY KEY,
        trial_key TEXT NOT NULL UNIQUE,
        job_id TEXT NOT NULL, trial_name TEXT NOT NULL,
        trial_uid TEXT,
        run_key TEXT NOT NULL REFERENCES runs(run_key),
        profile TEXT NOT NULL, run_id TEXT NOT NULL,
        source TEXT NOT NULL CHECK (source IN ('bench','external')),
        arm TEXT NOT NULL,
        trial_dir TEXT NOT NULL,
        family TEXT NOT NULL,
        implementation TEXT NOT NULL,
        harness_version TEXT,
        harness_version_pinned TEXT,
        harbor_version TEXT,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        harbor_model TEXT NOT NULL,
        effort TEXT,
        effort_label TEXT,
        options_json TEXT NOT NULL,
        cautions_json TEXT NOT NULL DEFAULT '[]',
        config_hash TEXT NOT NULL,
        config_json TEXT NOT NULL,
        config_label TEXT NOT NULL,
        n_concurrent INTEGER,
        dataset TEXT,
        dataset_snapshot TEXT,
        task TEXT NOT NULL,
        task_digest TEXT,
        task_source_type TEXT,
        reward REAL, rewards_json TEXT,
        passed INTEGER,
        exception_type TEXT,
        exception_head TEXT,
        harbor_cause TEXT,
        cap_marker TEXT,
        setup_marker TEXT,
        agent_started INTEGER NOT NULL DEFAULT 0,
        n_input INTEGER, n_cache INTEGER, n_output INTEGER,
        usage_json TEXT NOT NULL DEFAULT '{}',
        primary_share REAL,
        reported_cost_usd REAL,
        cost_at_ingest_usd REAL, pricing_sha_at_ingest TEXT,
        started_at TEXT, finished_at TEXT,
        env_setup_sec REAL, agent_setup_sec REAL, agent_exec_sec REAL, verifier_sec REAL, total_sec REAL,
        outcome TEXT NOT NULL,
        outcome_reason TEXT NOT NULL,
        parser_version INTEGER NOT NULL, classifier_version INTEGER NOT NULL,
        raw_result_z BLOB,
        raw_lock_z BLOB,
        ingest_via TEXT NOT NULL CHECK (ingest_via IN ('live','finalize','rescan')),
        ingested_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    "CREATE INDEX trials_model ON trials(provider, model, family)",
    "CREATE INDEX trials_task ON trials(task, task_digest)",
    "CREATE INDEX trials_cfg ON trials(config_hash)",
    "CREATE INDEX trials_run ON trials(run_key, arm)",
    "CREATE INDEX trials_time ON trials(started_at)",
    "CREATE INDEX trials_dir ON trials(trial_dir)",
    """CREATE VIEW trials_v AS
        SELECT t.*, ROW_NUMBER() OVER (PARTITION BY t.run_key, t.arm, COALESCE(t.task_digest, t.task)
                                       ORDER BY t.started_at, t.trial_name) AS attempt
        FROM trials t""",
    """CREATE TABLE exclusions (
        id INTEGER PRIMARY KEY,
        target_type TEXT NOT NULL CHECK (target_type IN ('trial','run')),
        target_id TEXT NOT NULL,
        action TEXT NOT NULL CHECK (action IN ('exclude','include')),
        reason TEXT NOT NULL CHECK (length(reason) BETWEEN 1 AND 500),
        created_at TEXT NOT NULL)""",
    """CREATE TABLE report_snapshots (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL, created_at TEXT NOT NULL,
        request_json TEXT NOT NULL, response_json TEXT NOT NULL,
        fingerprint TEXT NOT NULL, pricing_sha TEXT NOT NULL, stats_version INTEGER NOT NULL)""",
)


def _migrate_v1(conn: sqlite3.Connection) -> None:
    for statement in _SCHEMA_V1:
        conn.execute(statement)


# Ordered and additive only: MIGRATIONS[n] takes user_version n to n + 1.
MIGRATIONS = [_migrate_v1]

_FACT_COLUMNS = [f.name for f in dataclasses.fields(TrialFacts)]
_INSERT_COLUMNS = _FACT_COLUMNS + ["outcome", "outcome_reason", "classifier_version",
                                   "ingest_via", "ingested_at", "updated_at"]
# What rederive() never touches: identity, provenance, the raw facts, and the
# figures that only the job folder or the moment of ingest knew.
_KEEP_ON_REDERIVE = frozenset({"trial_key", "job_id", "trial_name", "run_key", "profile", "run_id", "source",
                               "arm", "trial_dir", "ingest_via", "ingested_at", "raw_result_z", "raw_lock_z",
                               "harbor_version", "n_concurrent", "cost_at_ingest_usd", "pricing_sha_at_ingest"})
_ROW_COLUMNS = ["id"] + [c for c in _FACT_COLUMNS if c not in ("raw_result_z", "raw_lock_z")] + [
    "outcome", "outcome_reason", "classifier_version", "ingest_via", "ingested_at", "updated_at", "attempt"]
# A killed trial's usage comes from its folder (the harness's own log), not a
# stored blob, so a rescan refreshes these when the folder says more.
_RECOVERED_COLUMNS = ("n_input", "n_cache", "n_output", "usage_json", "primary_share", "reported_cost_usd",
                      "cost_at_ingest_usd", "pricing_sha_at_ingest", "agent_started", "parser_version")

# TrialRow, in export order. Objects are written as JSON text in CSV.
EXPORT_COLUMNS = (
    "id", "trial_key", "run_key", "profile", "run_id", "source", "arm", "trial_name", "attempt", "family",
    "implementation", "harness_version", "harbor_version", "provider", "model", "effort", "effort_label",
    "config_hash", "config_label", "options", "cautions", "dataset", "dataset_snapshot", "task",
    "task_digest_short", "outcome", "outcome_reason", "harbor_cause", "exception_type", "exception_head",
    "reward", "passed", "n_input", "n_cache", "n_output", "usage_status", "tokens_total", "cost_usd",
    "cost_at_ingest_usd", "reported_cost_usd", "unpriced_models", "primary_share", "started_at", "finished_at",
    "phases", "total_sec", "usage", "exclusion", "log_url",
)

# None: sorted in Python, nulls last, because the value depends on today's
# prices (cost_usd) or on whether the usage was recorded (tokens_total).
_ORDERS = {"started_at": "started_at", "task": "task", "family": "family", "tokens_total": None, "cost_usd": None}
# What classify.usage_status reads from a stored row.
_USAGE_FACTS = ("outcome", "n_input", "n_output", "exception_type", "agent_started", "agent_exec_sec")


# Filters ---------------------------------------------------------------------

@dataclass(frozen=True)
class Filters:
    date_from: str | None = None
    date_to: str | None = None
    profiles: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    models: tuple[str, ...] = ()            # 'provider:model'
    families: tuple[str, ...] = ()
    implementations: tuple[str, ...] = ()
    configs: tuple[str, ...] = ()
    exclude_configs: tuple[str, ...] = ()
    harness_versions: tuple[str, ...] = ()
    harbor_versions: tuple[str, ...] = ()
    efforts: tuple[str | None, ...] = ()    # None matches trials with no effort set
    datasets: tuple[str, ...] = ()
    snapshots: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    runs: tuple[str, ...] = ()
    exclude_runs: tuple[str, ...] = ()
    include_cautioned: bool = False


_LIST_FILTERS = tuple(f.name for f in dataclasses.fields(Filters)
                      if f.name not in ("date_from", "date_to", "include_cautioned"))
_MAX_ITEMS = 500
_MAX_ITEM_CHARS = 300


def _date_bound(value: str, *, end: bool) -> tuple[str, str]:
    """(operator, fixed-width UTC text). A bare date_to covers that whole UTC day."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        day = date.fromisoformat(value)
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        if end:
            return "<", (start + timedelta(days=1)).strftime(_TIME_FORMAT)
        return ">=", start.strftime(_TIME_FORMAT)
    dt = datetime.fromisoformat(value)
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return ("<=" if end else ">="), dt.strftime(_TIME_FORMAT)


def parse_filters(body: Mapping | None) -> Filters:
    """Filters from a request body. ValueError on anything unexpected."""
    if body is None:
        return Filters()
    if not isinstance(body, Mapping):
        raise ValueError("filters must be an object")
    known = {f.name for f in dataclasses.fields(Filters)}
    unknown = sorted(str(k) for k in body if k not in known)
    if unknown:
        raise ValueError(f"Unknown filter: {', '.join(unknown)}")
    values: dict[str, Any] = {}
    for key in ("date_from", "date_to"):
        value = body.get(key)
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a date such as 2026-10-29")
        try:
            _date_bound(value, end=key == "date_to")
        except ValueError:
            raise ValueError(f"{key} is not a date: {value!r}") from None
        values[key] = value
    for key in _LIST_FILTERS:
        value = body.get(key)
        if value is None:
            continue
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{key} must be a list")
        if len(value) > _MAX_ITEMS:
            raise ValueError(f"{key} has more than {_MAX_ITEMS} items")
        items = []
        for item in value:
            if item is None and key == "efforts":
                items.append(None)
                continue
            if not isinstance(item, str):
                raise ValueError(f"{key} items must be strings")
            if len(item) > _MAX_ITEM_CHARS:
                raise ValueError(f"{key} has an item longer than {_MAX_ITEM_CHARS} characters")
            if key == "sources" and item not in ("bench", "external"):
                raise ValueError("sources items must be bench or external")
            if key == "models" and ":" not in item:
                raise ValueError("models items look like provider:model, e.g. openrouter:z-ai/glm-5.3")
            items.append(item)
        values[key] = tuple(items)
    if "include_cautioned" in body and body["include_cautioned"] is not None:
        if not isinstance(body["include_cautioned"], bool):
            raise ValueError("include_cautioned must be true or false")
        values["include_cautioned"] = body["include_cautioned"]
    return Filters(**values)


def _where(f: Filters, outcomes: Sequence[str] | None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    def among(column: str, values: Sequence[Any], negate: bool = False) -> None:
        if values:
            clauses.append(f"{column} {'NOT IN' if negate else 'IN'} ({','.join('?' * len(values))})")
            params.extend(values)

    # A killed trial has no start time of its own; its run's start places it.
    when = "COALESCE(started_at, (SELECT r.created_at FROM runs r WHERE r.run_key = trials_v.run_key))"
    for key, end in (("date_from", False), ("date_to", True)):
        value = getattr(f, key)
        if value:
            op, text = _date_bound(value, end=end)
            clauses.append(f"{when} {op} ?")
            params.append(text)
    among("profile", f.profiles)
    among("source", f.sources)
    if f.models:
        pairs = [m.split(":", 1) for m in f.models]
        clauses.append("(" + " OR ".join("(provider = ? AND model = ?)" for _ in pairs) + ")")
        for provider, model in pairs:
            params += [provider, model]
    among("family", f.families)
    among("implementation", f.implementations)
    among("config_hash", f.configs)
    among("config_hash", f.exclude_configs, negate=True)
    among("harness_version", f.harness_versions)
    among("harbor_version", f.harbor_versions)
    if f.efforts:
        named = [e for e in f.efforts if e is not None]
        parts = []
        if named:
            parts.append(f"effort IN ({','.join('?' * len(named))})")
            params.extend(named)
        if any(e is None for e in f.efforts):
            parts.append("effort IS NULL")
        clauses.append("(" + " OR ".join(parts) + ")")
    among("dataset", f.datasets)
    among("dataset_snapshot", f.snapshots)
    among("task", f.tasks)
    among("run_key", f.runs)
    among("run_key", f.exclude_runs, negate=True)
    if not f.include_cautioned:
        clauses.append("cautions_json = '[]'")
    if outcomes:
        bad = [o for o in outcomes if o not in classify.OUTCOMES]
        if bad:
            raise ValueError(f"Unknown outcome: {', '.join(map(str, bad))}")
        among("outcome", list(outcomes))
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


# Paths and locks -----------------------------------------------------------------

def _needs_delete_journal(path: Path) -> bool:
    """WAL needs shared memory on a local disk: not on UNC shares, and not in
    a OneDrive folder whose sync client may lock or upload the -wal file."""
    text = os.path.abspath(path)
    if text.startswith(("\\\\", "//")):
        return True
    return any("onedrive" in part.lower() for part in Path(text).parts)


def _holder_path(db: Path) -> Path:
    return db.with_name(db.name + ".inuse")


def held_elsewhere(db: Path) -> bool:
    """True while any open Ledger (in any process) uses this database file.

    Each open Ledger keeps <db>.inuse open. Windows refuses to delete a file
    another handle has open; elsewhere a shared flock marks it.
    """
    holder = _holder_path(Path(db))
    if not holder.exists():
        return False
    if os.name == "nt":
        try:
            holder.unlink()
        except FileNotFoundError:
            return False
        except PermissionError:
            return True
        return False
    import fcntl
    with open(holder, "a", encoding="utf-8") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return False


def _replace(src: Path, dst: Path, patience: float | None = None) -> None:
    """os.replace that waits out a brief antivirus or indexer hold on Windows,
    like fsutil's atomic writes."""
    deadline = time.monotonic() + (REPLACE_PATIENCE if patience is None else patience)
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def _stats_version() -> int:
    try:
        from .stats import STATS_VERSION
        return int(STATS_VERSION)
    except Exception:  # the report package may not be installed yet
        return 1


def _loads(text: str | None, default: Any = None) -> Any:
    if text is None:
        return default
    try:
        return json.loads(text)
    except ValueError:
        return default


# The ledger -----------------------------------------------------------------------

class Ledger:
    def __init__(self, path: Path | None = None, *, roots: Sequence[ScanRoot] | None = None,
                 root: Path | None = None) -> None:
        self.path = Path(path or settings.LEDGER_DB)
        self.roots: list[ScanRoot] = list(roots) if roots is not None else extract.default_roots()
        self.root = Path(root or settings.ROOT)   # trial folders are stored relative to this
        self.journal_mode = "wal"
        self.warnings: list[str] = []
        self._available = False
        self._open_error: str | None = None
        self._last_error: dict[str, str] | None = None
        self._finished: set[str] = set()
        self._known_runs: set[str] = set()
        self._prefix_cache: dict[str, str] | None = None
        self._rescan_lock = threading.Lock()
        self._holder: IO[str] | None = None

    # Lifecycle ----------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    @property
    def error(self) -> str | None:
        if self._open_error:
            return self._open_error
        return self._last_error["message"] if self._last_error else None

    @property
    def rescan_running(self) -> bool:
        return self._rescan_lock.locked()

    def open(self) -> None:
        """Never raises. On any failure the ledger stays disabled with `error`
        set, and the bench runs normally without it."""
        if self._available:
            return
        self._open_error = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if _needs_delete_journal(self.path):
                self.journal_mode = "delete"
                self.warnings.append("The ledger is on a network or OneDrive folder, so it uses the slower "
                                     "DELETE journal; move it with BENCH_LEDGER for better concurrency.")
            with self._conn() as conn:
                conn.execute(f"PRAGMA journal_mode={self.journal_mode.upper()}")
                check = conn.execute("PRAGMA quick_check").fetchone()
                if check is None or check[0] != "ok":
                    raise sqlite3.DatabaseError(
                        f"integrity check failed ({check[0] if check else 'no result'}). With the bench stopped, "
                        "run: python -m bench.ledger rebuild")
            self._migrate()
            self._seed_exclusions()
            self._load_finished()
            self._holder = self._hold()
            self._available = True
        except Exception as exc:
            self._open_error = f"The ledger could not be opened: {type(exc).__name__}: {exc}"
            self._release()
            return
        # Neither of these may disable a ledger that opened.
        try:
            if self._stale_rows():
                self.rederive()
        except Exception as exc:
            self.set_error("rederive", exc)
        try:
            self.backup()
        except Exception as exc:
            self.set_error("backup", exc)

    def close(self) -> None:
        if self._available:
            try:
                with self._conn() as conn:
                    # One attempt, no waiting: a TRUNCATE checkpoint that waits for
                    # readers also blocks every writer, including another bench
                    # process's running arms.
                    conn.execute("PRAGMA busy_timeout=0")
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass  # a reader still holds the WAL; a later checkpoint gets it
        self._available = False
        self._release()

    def _hold(self) -> IO[str]:
        holder = open(_holder_path(self.path), "a", encoding="utf-8")
        if os.name != "nt":
            import fcntl
            fcntl.flock(holder.fileno(), fcntl.LOCK_SH)
        return holder

    def _release(self) -> None:
        if self._holder is not None:
            with contextlib.suppress(OSError):
                self._holder.close()
            self._holder = None

    def _require(self) -> None:
        if not self._available:
            raise LedgerUnavailable(self._open_error or "The ledger is not open")

    # Connections ----------------------------------------------------------------

    def _connect(self, busy_ms: int | None = None) -> sqlite3.Connection:
        wait = BUSY_TIMEOUT_MS if busy_ms is None else max(int(busy_ms), 0)
        conn = sqlite3.connect(str(self.path), timeout=wait / 1000, isolation_level=None,
                               check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout={int(wait)}")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
        except BaseException:
            conn.close()
            raise
        return conn

    @contextlib.contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextlib.contextmanager
    def _write(self, busy_ms: int | None = None) -> Iterator[sqlite3.Connection]:
        """One write transaction. busy_ms bounds both waits, for this
        process's other writers and for SQLite's lock, so a caller on the run
        manager's refresh path is never held for the full busy timeout."""
        wait = BUSY_TIMEOUT_MS if busy_ms is None else max(int(busy_ms), 0)
        acquired = _WRITE_LOCK.acquire(timeout=wait / 1000) if wait else _WRITE_LOCK.acquire(blocking=False)
        if not acquired:
            raise sqlite3.OperationalError("database is busy (another ledger write in this process)")
        try:
            conn = self._connect(wait)
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    yield conn
                except BaseException:
                    with contextlib.suppress(sqlite3.Error):
                        conn.execute("ROLLBACK")
                    raise
                conn.execute("COMMIT")
            finally:
                conn.close()
        finally:
            _WRITE_LOCK.release()

    # Open-time steps ---------------------------------------------------------------

    def _migrate(self) -> None:
        with self._write() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"it was written by a newer bench (schema {version}, this bench reads "
                                   f"up to {SCHEMA_VERSION})")
            for v in range(version, SCHEMA_VERSION):
                MIGRATIONS[v](conn)
                conn.execute(f"PRAGMA user_version={v + 1}")

    def _seed_exclusions(self) -> None:
        with self._write() as conn:
            for target_type, target_id, reason in SEED_EXCLUSIONS:
                seen = conn.execute("SELECT 1 FROM exclusions WHERE target_type = ? AND target_id = ? LIMIT 1",
                                    (target_type, target_id)).fetchone()
                if seen is None:
                    conn.execute("INSERT INTO exclusions (target_type, target_id, action, reason, created_at) "
                                 "VALUES (?, ?, 'exclude', ?, ?)", (target_type, target_id, reason, _now()))

    def _load_finished(self) -> None:
        with self._conn() as conn:
            rows = conn.execute("SELECT trial_dir FROM trials WHERE outcome != 'incomplete'").fetchall()
        self._finished = {os.path.normcase(r[0]) for r in rows}

    def _stale_rows(self) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT 1 FROM trials WHERE parser_version < ? OR classifier_version < ? LIMIT 1",
                               (extract.PARSER_VERSION, classify.CLASSIFIER_VERSION)).fetchone()
        return row is not None

    def _prefixes(self) -> dict[str, str]:
        if self._prefix_cache is None:
            self._prefix_cache = extract.provider_prefixes()
        return self._prefix_cache

    def _set_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def set_error(self, where: str, exc: BaseException) -> None:
        """Record a failure outside open() (an ingest from a run, a backup).
        The ledger stays available; the next rescan catches up."""
        message = extract.redact(f"{type(exc).__name__}: {exc}")[:500]
        self._last_error = {"at": _now(), "where": str(where), "message": message}
        if self._available:
            try:
                # No waiting: this is often called because the database is busy,
                # from the run manager's refresh.
                with self._write(busy_ms=0) as conn:
                    self._set_meta(conn, "last_error", json.dumps(self._last_error))
            except Exception:
                pass  # the in-memory copy still shows in status()

    # Writing ----------------------------------------------------------------------

    def upsert_run(self, run_key: str, profile: str, run: Mapping[str, Any] | None, *, source: str = "bench",
                   run_id: str | None = None, busy_ms: int | None = None) -> None:
        """Mirror run.json (or what an external job supplies). Fields the
        caller does not know never erase ones already recorded, and
        updated_at moves only when a mirrored field changes, because it is
        part of version(), the report cache key."""
        self._require()
        run = run or {}
        config = run.get("config") if isinstance(run.get("config"), Mapping) else {}
        spec = run.get("spec") if isinstance(run.get("spec"), Mapping) else {}
        baseline = spec.get("baseline_arm") or config.get("baseline_arm") or run.get("baseline_arm")

        def dump(key: str) -> str | None:
            # Redact the values, not the JSON text: a pattern such as
            # 'Bearer \S+' would swallow the closing quote and break the JSON.
            if run.get(key) is None:
                return None
            return json.dumps(extract.redact_data(run[key]), ensure_ascii=False)

        budget = run.get("budget_usd")
        now = _now()
        values = (run_key, profile, str(run_id or run.get("id") or run_key.rsplit("/", 1)[-1]), source,
                  run.get("mode"), run.get("status"), extract.utc_text(run.get("created_at")),
                  extract.utc_text(run.get("finished_at")),
                  float(budget) if isinstance(budget, (int, float)) and not isinstance(budget, bool) else None,
                  baseline, dump("tasks"), dump("plan"), dump("config"), dump("dataset"), now, now)
        mirrored = ("mode", "status", "created_at", "finished_at", "budget_usd", "baseline_arm", "tasks_json",
                    "plan_json", "config_json", "dataset_json")
        with self._write(busy_ms) as conn:
            conn.execute(
                "INSERT INTO runs (run_key, profile, run_id, source, mode, status, created_at, finished_at, "
                "budget_usd, baseline_arm, tasks_json, plan_json, config_json, dataset_json, first_seen_at, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_key) DO UPDATE SET "
                + ", ".join(f"{c} = COALESCE(excluded.{c}, runs.{c})" for c in mirrored)
                + ", updated_at = excluded.updated_at WHERE "
                + " OR ".join(f"runs.{c} IS NOT COALESCE(excluded.{c}, runs.{c})" for c in mirrored), values)
        self._known_runs.add(run_key)

    def _store(self, facts: TrialFacts, via: str, *, incomplete: bool = False, busy_ms: int | None = None) -> str:
        """Insert a new trial, or upgrade an incomplete one. Never overwrites
        a finished row: a second result.json for the same trial is ignored.
        An incomplete row seen incomplete again takes the usage its folder
        now gives (a newer parser recovers more of the harness's log)."""
        row = {name: getattr(facts, name) for name in _FACT_COLUMNS}
        outcome, reason = classify.classify(row)
        now = _now()
        row.update(outcome=outcome, outcome_reason=reason, classifier_version=classify.CLASSIFIER_VERSION,
                   ingest_via=via, ingested_at=now, updated_at=now)
        with self._write(busy_ms) as conn:
            conn.execute("INSERT OR IGNORE INTO runs (run_key, profile, run_id, source, first_seen_at, updated_at) "
                         "VALUES (?, ?, ?, ?, ?, ?)", (facts.run_key, facts.profile, facts.run_id, facts.source,
                                                       now, now))
            existing = conn.execute(f"SELECT id, outcome, {', '.join(_RECOVERED_COLUMNS)} FROM trials "
                                    "WHERE trial_key = ?", (facts.trial_key,)).fetchone()
            if existing is None:
                conn.execute(f"INSERT INTO trials ({', '.join(_INSERT_COLUMNS)}) "
                             f"VALUES ({', '.join(':' + c for c in _INSERT_COLUMNS)})", row)
                status = "inserted"
            elif existing["outcome"] == "incomplete" and not incomplete:
                columns = [c for c in _INSERT_COLUMNS if c not in ("trial_key", "ingested_at", "raw_lock_z")]
                conn.execute(f"UPDATE trials SET {', '.join(f'{c} = :{c}' for c in columns)}, "
                             "raw_lock_z = COALESCE(raw_lock_z, :raw_lock_z) WHERE id = :row_id",
                             {**row, "row_id": existing["id"]})
                status = "upgraded"
            elif existing["outcome"] == "incomplete" and any(
                    existing[c] != row[c] for c in ("n_input", "n_cache", "n_output", "usage_json", "agent_started")):
                columns = [*_RECOVERED_COLUMNS, "outcome", "outcome_reason", "classifier_version", "updated_at"]
                conn.execute(f"UPDATE trials SET {', '.join(f'{c} = :{c}' for c in columns)} WHERE id = :row_id",
                             {**{c: row[c] for c in columns}, "row_id": existing["id"]})
                status = "upgraded"
            else:
                status = "unchanged"
            if status != "unchanged":
                self._set_meta(conn, "last_ingest_at", now)
        return status

    def ingest_job(self, job_dir: Path, ctx: RunContext, *, final: bool, via: str, reparse: bool = False,
                   skipped: list[dict[str, str]] | None = None) -> dict[str, int]:
        """Ingest every trial folder of one Harbor job. Each trial is its own
        transaction. With final=True (the job can no longer finish), trial
        folders without a result are recorded as incomplete. A live ingest
        waits at most LIVE_BUSY_MS for the database and then raises
        sqlite3.OperationalError."""
        self._require()
        if via not in ("live", "finalize", "rescan"):
            raise ValueError(f"via must be live, finalize or rescan (got {via!r})")
        busy_ms = LIVE_BUSY_MS if via == "live" else None
        counts = {"inserted": 0, "upgraded": 0, "unchanged": 0, "incomplete": 0, "skipped": 0, "scanned": 0}
        job_dir = Path(job_dir)
        try:
            trial_dirs = extract.list_dirs(job_dir)
        except FileNotFoundError:
            return counts
        except OSError as exc:
            counts["skipped"] += 1
            if skipped is not None:
                skipped.append({"path": str(job_dir), "reason": f"unreadable folder: {exc}"})
            return counts
        prefixes = self._prefixes()
        job_lock: dict | None = None
        for trial_dir in trial_dirs:
            has_result = (trial_dir / "result.json").exists()
            if not has_result and not (trial_dir / "config.json").exists():
                continue  # not a trial folder
            counts["scanned"] += 1
            key = os.path.normcase(extract.relative_dir(trial_dir, self.root))
            if key in self._finished and not reparse:
                counts["unchanged"] += 1
                continue
            if job_lock is None:
                lock = extract.read_json(job_dir / "lock.json")
                job_lock = lock if isinstance(lock, dict) else {}
            facts = extract.parse_trial(trial_dir, ctx, prefixes, root=self.root, job_lock=job_lock)
            if facts is None and final and has_result:
                # Probably held for a moment by an indexer or antivirus. After
                # finalize no live pass comes back, so a miss here would leave a
                # finished trial 'incomplete' until the next rescan.
                for _ in range(FINAL_READ_RETRIES):
                    time.sleep(FINAL_READ_DELAY)
                    facts = extract.parse_trial(trial_dir, ctx, prefixes, root=self.root, job_lock=job_lock)
                    if facts is not None:
                        break
            if facts is not None:
                counts[self._store(facts, via, busy_ms=busy_ms)] += 1
                self._finished.add(key)
                continue
            if not final:
                if has_result:  # being written, or briefly locked: the next pass retries
                    counts["skipped"] += 1
                continue
            facts = extract.parse_incomplete(trial_dir, ctx, prefixes, root=self.root, job_lock=job_lock)
            if facts is None:
                counts["skipped"] += 1
                if skipped is not None:
                    skipped.append({"path": str(trial_dir), "reason": "no readable result.json or config.json"})
                continue
            stored = self._store(facts, via, incomplete=True, busy_ms=busy_ms)
            counts["incomplete" if stored == "inserted" else stored] += 1
        return counts

    # LedgerHook (called by RunManager through its guard) -------------------------

    def upsert_bench_run(self, run: dict) -> None:
        """While the run is running this is the start-time mirror, on the
        launch request's path, so it waits only briefly; the final mirror
        waits the full busy timeout because nothing in the run waits on it.
        A rescan re-mirrors every run.json either way."""
        profile = settings.PROFILE_NAME
        busy_ms = LIVE_BUSY_MS if run.get("status") == "running" else None
        self.upsert_run(f"{profile}/{run['id']}", profile, run, busy_ms=busy_ms)

    def ingest_bench_job(self, job_dir: Path, run: dict, arm: str, *, final: bool, via: str) -> dict:
        ctx = extract.context_for_bench_arm(run, arm, settings.PROFILE_NAME)
        if ctx.run_key not in self._known_runs:
            self.upsert_run(ctx.run_key, ctx.profile, run, busy_ms=LIVE_BUSY_MS if via == "live" else None)
        return self.ingest_job(job_dir, ctx, final=final, via=via)

    # Rescan and rederive -------------------------------------------------------------

    def rescan(self, *, reparse: bool = False) -> dict:
        """Walk every root and ingest whatever is on disk. Disk is the truth:
        this is what guarantees completeness after a crash or a failed live
        ingest. One at a time; a second call raises RuntimeError."""
        self._require()
        if not self._rescan_lock.acquire(blocking=False):
            raise RuntimeError("A rescan is already running")
        try:
            started, at = time.monotonic(), _now()
            self._prefix_cache = None  # pick up providers added since the last scan
            skipped: list[dict[str, str]] = []
            totals: Counter[str] = Counter()
            upserted: set[str] = set()
            prefixes = self._prefixes()
            for job_dir, ctx, run, terminal in extract.discover(self.roots, skipped, root=self.root):
                try:
                    if ctx.run_key not in upserted:
                        upserted.add(ctx.run_key)
                        if run is not None:
                            self.upsert_run(ctx.run_key, ctx.profile, run, run_id=ctx.run_id)
                        else:
                            self.upsert_run(ctx.run_key, "external", extract.external_run_info(job_dir),
                                            source="external", run_id=ctx.run_id)
                    counts = self.ingest_job(job_dir, ctx, final=terminal, via="rescan", reparse=reparse,
                                             skipped=skipped)
                except sqlite3.OperationalError:
                    raise  # locked beyond busy_timeout: report it rather than half-scan quietly
                except Exception as exc:
                    skipped.append({"path": str(job_dir), "reason": f"{type(exc).__name__}: {exc}"})
                    continue
                totals.update(counts)
                if ctx.source == "external" and counts["inserted"] + counts["upgraded"]:
                    self._note_guessed_provider(job_dir, ctx.run_key, prefixes, skipped)
            if reparse:
                self.rederive(force=True)
            summary = {"at": at, "scanned": totals["scanned"], "inserted": totals["inserted"],
                       "upgraded": totals["upgraded"], "unchanged": totals["unchanged"],
                       "incomplete": totals["incomplete"], "skipped": skipped,
                       "seconds": round(time.monotonic() - started, 3)}
            with self._write() as conn:
                self._set_meta(conn, "last_rescan", json.dumps({**summary, "skipped": skipped[:200]},
                                                               ensure_ascii=False))
            self._last_error = None
            return summary
        finally:
            self._rescan_lock.release()

    def _note_guessed_provider(self, job_dir: Path, run_key: str, prefixes: Mapping[str, str],
                               skipped: list[dict[str, str]]) -> None:
        """A plain Harbor job whose model prefix no bench config names gets
        its provider from Harbor's model_info or the first path segment. The
        trials are kept; the rescan summary says the name is a guess."""
        known = set(prefixes.values())
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT provider, harbor_model FROM trials WHERE run_key = ?",
                                (run_key,)).fetchall()
        for r in rows:
            if r["provider"] not in known and not any(r["harbor_model"].startswith(p) for p in prefixes):
                skipped.append({"path": str(job_dir), "reason": f"provider '{r['provider']}' guessed from model "
                                f"'{r['harbor_model']}'; add its model_prefix to a config to name it"})

    def start_rescan_thread(self) -> bool:
        """Rescan in a daemon thread so it never delays the dashboard."""
        if not self._available or self._rescan_lock.locked():
            return False

        def work() -> None:
            try:
                self.rescan()
            except RuntimeError:
                pass  # another rescan got there first
            except Exception as exc:
                self.set_error("rescan", exc)

        threading.Thread(target=work, name="ledger-rescan", daemon=True).start()
        return True

    def rederive(self, *, force: bool = False) -> int:
        """Recompute derived columns from the stored raw JSON for rows written
        by an older parser or classifier. The raw blobs never change."""
        self._require()
        with self._conn() as conn:
            ids = [r[0] for r in conn.execute(
                "SELECT id FROM trials WHERE ? OR parser_version < ? OR classifier_version < ? ORDER BY id",
                (int(force), extract.PARSER_VERSION, classify.CLASSIFIER_VERSION))]
        prefixes = self._prefixes()
        done = 0
        for row_id in ids:
            with self._write() as conn:
                row = conn.execute("SELECT * FROM trials WHERE id = ?", (row_id,)).fetchone()
                if row is None:
                    continue
                values = self._rederived(dict(row), prefixes)
                columns = sorted(values)
                conn.execute(f"UPDATE trials SET {', '.join(f'{c} = :{c}' for c in columns)} WHERE id = :row_id",
                             {**values, "row_id": row_id})
            done += 1
        return done

    def _rederived(self, row: dict[str, Any], prefixes: Mapping[str, str]) -> dict[str, Any]:
        merged = dict(row)
        if row["raw_result_z"] is not None:
            hm, model = row["harbor_model"] or "", row["model"] or ""
            bench = row["source"] == "bench"
            # The run supplied these at ingest; the raw JSON cannot recreate them.
            ctx = RunContext(
                run_key=row["run_key"], profile=row["profile"], run_id=row["run_id"], source=row["source"],
                arm=row["arm"], provider=row["provider"] if bench else None,
                model_prefix=hm[: len(hm) - len(model)] if bench and model and hm.endswith(model) else None,
                effort_label=row["effort_label"], dataset=row["dataset"] if bench else None,
                local_snapshot=row["dataset_snapshot"] if (row["dataset_snapshot"] or "").startswith("local:")
                else None,
                family=row["family"] if bench else None)
            job_lock = {"harbor": {"version": row["harbor_version"]}, "n_concurrent_trials": row["n_concurrent"]}
            facts = extract.facts_from_result(extract.inflate(row["raw_result_z"]),
                                              extract.inflate(row["raw_lock_z"]), job_lock,
                                              trial_dir=row["trial_dir"], ctx=ctx, prefixes=prefixes)
            fresh = {c: getattr(facts, c) for c in _FACT_COLUMNS if c not in _KEEP_ON_REDERIVE}
            merged.update(fresh)
            values = dict(fresh)
        else:
            # Incomplete: no stored JSON to re-parse; the next rescan re-reads its folder.
            values = {"parser_version": extract.PARSER_VERSION}
            merged.update(values)
        outcome, reason = classify.classify(merged)
        values.update(outcome=outcome, outcome_reason=reason, classifier_version=classify.CLASSIFIER_VERSION,
                      updated_at=_now())
        return values

    # Reading -------------------------------------------------------------------------

    def _exclusion_maps(self, conn: sqlite3.Connection) -> dict[tuple[str, str], dict[str, Any]]:
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for r in conn.execute("SELECT * FROM exclusions ORDER BY id"):
            latest[(r["target_type"], r["target_id"])] = dict(r)
        return latest

    def _trial_row(self, r: Mapping[str, Any], table: Mapping[str, Any],
                   latest: Mapping[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
        usage = _loads(r["usage_json"], {}) or {}
        cost, unpriced = pricing.price_usage(usage, r["provider"], table)
        n_input, n_cache, n_output = r["n_input"], r["n_cache"], r["n_output"]
        status = classify.usage_status({k: r[k] for k in _USAGE_FACTS})
        # Harbor's 0/0/0 for a run that recorded nothing stays in n_input..n_output
        # as the raw fact; the totals and the costs say "not measured".
        measured = status != "unmeasured" and (n_input is not None or n_output is not None)
        if status == "unmeasured":
            cost = None
        digest = r["task_digest"]
        excl = latest.get(("trial", r["trial_key"])) or latest.get(("run", r["run_key"]))
        bench_here = r["source"] == "bench" and r["profile"] == settings.PROFILE_NAME
        return {
            "id": r["id"], "trial_key": r["trial_key"], "run_key": r["run_key"], "profile": r["profile"],
            "run_id": r["run_id"], "source": r["source"], "arm": r["arm"], "trial_name": r["trial_name"],
            "attempt": r["attempt"], "family": r["family"], "implementation": r["implementation"],
            "harness_version": r["harness_version"], "harbor_version": r["harbor_version"],
            "provider": r["provider"], "model": r["model"], "effort": r["effort"], "effort_label": r["effort_label"],
            "config_hash": r["config_hash"], "config_label": r["config_label"],
            "options": _loads(r["options_json"], {}), "cautions": _loads(r["cautions_json"], []),
            "dataset": r["dataset"], "dataset_snapshot": r["dataset_snapshot"], "task": r["task"],
            "task_digest_short": digest.split(":", 1)[-1][:12] if digest else None,
            "outcome": r["outcome"], "outcome_reason": r["outcome_reason"], "harbor_cause": r["harbor_cause"],
            "exception_type": r["exception_type"],
            "exception_head": r["exception_head"][:300] if r["exception_head"] else None,
            "reward": r["reward"], "passed": None if r["passed"] is None else bool(r["passed"]),
            "n_input": n_input, "n_cache": n_cache, "n_output": n_output, "usage_status": status,
            "tokens_total": (n_input or 0) + (n_output or 0) if measured else None,
            "cost_usd": cost, "cost_at_ingest_usd": None if status == "unmeasured" else r["cost_at_ingest_usd"],
            "reported_cost_usd": r["reported_cost_usd"], "unpriced_models": unpriced,
            "primary_share": r["primary_share"], "started_at": r["started_at"], "finished_at": r["finished_at"],
            "phases": {"env_setup_sec": r["env_setup_sec"], "agent_setup_sec": r["agent_setup_sec"],
                       "agent_exec_sec": r["agent_exec_sec"], "verifier_sec": r["verifier_sec"]},
            "total_sec": r["total_sec"],
            "exclusion": ({"target_type": excl["target_type"], "action": excl["action"], "reason": excl["reason"],
                           "created_at": excl["created_at"]} if excl else None),
            "log_url": f"/api/runs/{r['run_id']}/log/{r['arm']}" if bench_here else None,
            # Beyond TrialRow, for the report and estimates.
            "task_digest": digest, "task_source_type": r["task_source_type"], "usage": usage,
            "tokens_uncached": (max((n_input or 0) - (n_cache or 0), 0) + (n_output or 0)) if measured else None,
            "harbor_model": r["harbor_model"], "harness_version_pinned": r["harness_version_pinned"],
            "n_concurrent": r["n_concurrent"], "cap_marker": r["cap_marker"], "setup_marker": r["setup_marker"],
            "agent_started": bool(r["agent_started"]), "rewards": _loads(r["rewards_json"]),
            "trial_uid": r["trial_uid"], "trial_dir": r["trial_dir"], "ingest_via": r["ingest_via"],
            "ingested_at": r["ingested_at"], "updated_at": r["updated_at"],
            "parser_version": r["parser_version"], "classifier_version": r["classifier_version"],
        }

    def query(self, f: Filters, *, outcomes: Sequence[str] | None = None, limit: int | None = None,
              offset: int = 0, order: str = "started_at DESC") -> tuple[int, list[dict]]:
        """(total matching, TrialRows). Bound parameters only; the order comes
        from a whitelist. Exclusions are shown on each row, not applied."""
        self._require()
        m = re.fullmatch(r"(\w+) (ASC|DESC)", order or "")
        if not m or m.group(1) not in _ORDERS:
            raise ValueError("order must be one of started_at, task, family, cost_usd, tokens_total, "
                             "followed by ASC or DESC")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT):
            raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be zero or more")
        column, direction = _ORDERS[m.group(1)], m.group(2)
        where, params = _where(f, outcomes)
        table = pricing.load_table()
        select = f"SELECT {', '.join(_ROW_COLUMNS)} FROM trials_v{where}"
        with self._conn() as conn:
            conn.execute("BEGIN")
            try:
                total = conn.execute(f"SELECT COUNT(*) FROM trials_v{where}", params).fetchone()[0]
                if column is None:
                    rows = conn.execute(select, params).fetchall()
                else:
                    sql = f"{select} ORDER BY {column} IS NULL, {column} {direction}, id {direction}"
                    if limit is not None:
                        sql += " LIMIT ? OFFSET ?"
                        rows = conn.execute(sql, [*params, limit, offset]).fetchall()
                    else:
                        rows = conn.execute(sql + " LIMIT -1 OFFSET ?", [*params, offset]).fetchall()
                latest = self._exclusion_maps(conn)
            finally:
                conn.execute("COMMIT")
        out = [self._trial_row(r, table, latest) for r in rows]
        if column is None:
            name = m.group(1)
            known = sorted((r for r in out if r[name] is not None), key=lambda r: (r[name], r["id"]),
                           reverse=direction == "DESC")
            out = known + [r for r in out if r[name] is None]
            out = out[offset: offset + limit] if limit is not None else out[offset:]
        return total, out

    def trial(self, trial_id: int) -> dict:
        """One TrialRow with its usage, config identity, the full redacted
        exception message and the stored raw JSON. KeyError if unknown."""
        self._require()
        try:
            trial_id = int(trial_id)
        except (TypeError, ValueError):
            raise KeyError(f"No trial {trial_id!r}") from None
        with self._conn() as conn:
            row = conn.execute(f"SELECT {', '.join(_ROW_COLUMNS)}, raw_result_z, raw_lock_z, config_json "
                               "FROM trials_v WHERE id = ?", (trial_id,)).fetchone()
            if row is None:
                raise KeyError(f"No trial {trial_id}")
            latest = self._exclusion_maps(conn)
        out = self._trial_row(row, pricing.load_table(), latest)
        raw = extract.inflate(row["raw_result_z"])
        trial_dir = Path(row["trial_dir"])
        absolute = trial_dir if trial_dir.is_absolute() else self.root / trial_dir
        out.update(
            config=_loads(row["config_json"], {}),
            exception_message=extract.exception_message(raw) if isinstance(raw, Mapping) else None,
            raw=raw, lock=extract.inflate(row["raw_lock_z"]),
            trial_dir=row["trial_dir"], trial_dir_exists=absolute.exists(),
        )
        return out

    def runs(self) -> list[dict]:
        """Every run in the ledger, newest first, with trials by outcome."""
        self._require()
        with self._conn() as conn:
            runs = [dict(r) for r in conn.execute("SELECT * FROM runs ORDER BY created_at DESC, run_key")]
            trials = conn.execute("SELECT run_key, outcome, cost_at_ingest_usd, reported_cost_usd, family, provider, "
                                  "model, dataset_snapshot, n_input, n_output, exception_type, agent_started, "
                                  "agent_exec_sec FROM trials").fetchall()
            latest = self._exclusion_maps(conn)
        by_run: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for t in trials:
            by_run[t["run_key"]].append(t)
        out = []
        for r in runs:
            rows = by_run.get(r["run_key"], [])
            # Finished trials only, as ArmRun counts them: killed ones are trials['incomplete'].
            usage = Counter(classify.usage_status({k: t[k] for k in _USAGE_FACTS}) for t in rows
                            if t["outcome"] != "incomplete")
            snapshots = Counter(t["dataset_snapshot"] for t in rows if t["dataset_snapshot"])
            excl = latest.get(("run", r["run_key"]))
            out.append({
                "run_key": r["run_key"], "profile": r["profile"], "run_id": r["run_id"], "source": r["source"],
                "mode": r["mode"], "status": r["status"], "created_at": r["created_at"],
                "finished_at": r["finished_at"], "budget_usd": r["budget_usd"],
                "trials": dict(Counter(t["outcome"] for t in rows)),
                # The cost at ingest, or the harness's own figure for unpriced trials.
                "spend_usd": round(sum(t["cost_at_ingest_usd"] if t["cost_at_ingest_usd"] is not None
                                       else (t["reported_cost_usd"] or 0.0) for t in rows), 6),
                # spend_usd is a lower bound while any of these, or trials['incomplete'], is above 0:
                # finished trials that ran but recorded no usage, and ones cut short whose usage
                # covers completed calls only.
                "trials_unmeasured": usage["unmeasured"], "trials_lower_bound": usage["lower_bound"],
                "families": sorted({t["family"] for t in rows}),
                "models": sorted({f"{t['provider']}:{t['model']}" for t in rows}),
                "dataset_snapshot": snapshots.most_common(1)[0][0] if snapshots else None,
                "exclusion": ({"action": excl["action"], "reason": excl["reason"], "created_at": excl["created_at"]}
                              if excl else None),
            })
        return out

    def facets(self, task_order: Sequence[str]) -> dict:
        """Every value each filter can take, with trial counts."""
        self._require()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT profile, source, provider, model, family, implementation, harness_version, harbor_version, "
                "effort, dataset, dataset_snapshot, task, task_digest, run_key, outcome, config_hash, config_label, "
                "options_json, cautions_json, started_at FROM trials").fetchall()
            runs = {r["run_key"]: dict(r) for r in conn.execute(
                "SELECT run_key, profile, run_id, source, created_at, status FROM runs")}

        def counted(values: Sequence[Any]) -> list[dict[str, Any]]:
            return [{"value": v, "trials": n}
                    for v, n in sorted(Counter(values).items(), key=lambda kv: (kv[0] is None, str(kv[0])))]

        models: Counter[tuple[str, str]] = Counter((r["provider"], r["model"]) for r in rows)
        families: dict[str, dict[str, Any]] = {}
        configs: dict[str, dict[str, Any]] = {}
        harness_versions: Counter[tuple[str, str]] = Counter()
        datasets: dict[Any, Counter[str]] = defaultdict(Counter)
        tasks: dict[str, dict[str, Any]] = {}
        run_trials: Counter[str] = Counter()
        for r in rows:
            fam = families.setdefault(r["family"], {"value": r["family"], "trials": 0, "implementations": set(),
                                                    "versions": set()})
            fam["trials"] += 1
            fam["implementations"].add(r["implementation"])
            if r["harness_version"]:
                fam["versions"].add(r["harness_version"])
                harness_versions[(r["family"], r["harness_version"])] += 1
            c = configs.setdefault(r["config_hash"], {
                "config_hash": r["config_hash"], "label": r["config_label"], "family": r["family"],
                "implementation": r["implementation"], "provider": r["provider"], "model": r["model"],
                "effort": r["effort"], "options": _loads(r["options_json"], {}),
                "cautions": _loads(r["cautions_json"], []), "trials": 0, "first_seen": None, "last_seen": None})
            c["trials"] += 1
            if r["started_at"]:
                c["first_seen"] = min(filter(None, (c["first_seen"], r["started_at"])))
                c["last_seen"] = max(filter(None, (c["last_seen"], r["started_at"])))
            datasets[r["dataset"]][r["dataset_snapshot"]] += 1
            t = tasks.setdefault(r["task"], {"value": r["task"], "trials": 0, "digests": set()})
            t["trials"] += 1
            if r["task_digest"]:
                t["digests"].add(r["task_digest"])
            run_trials[r["run_key"]] += 1
        order = {name: i for i, name in enumerate(task_order)}
        starts = [r["started_at"] for r in rows if r["started_at"]]
        outcome_counts = Counter(r["outcome"] for r in rows)
        return {
            "profiles": counted([r["profile"] for r in rows]),
            "sources": counted([r["source"] for r in rows]),
            "models": [{"key": f"{p}:{m}", "provider": p, "model": m, "trials": n}
                       for (p, m), n in sorted(models.items())],
            "families": [{**f, "implementations": sorted(f["implementations"]), "versions": sorted(f["versions"])}
                         for _, f in sorted(families.items())],
            "configs": sorted(configs.values(), key=lambda c: (c["family"], c["provider"], c["model"], c["label"])),
            "harness_versions": [{"family": fam, "value": v, "trials": n}
                                 for (fam, v), n in sorted(harness_versions.items())],
            "harbor_versions": counted([r["harbor_version"] for r in rows if r["harbor_version"]]),
            "efforts": counted([r["effort"] for r in rows]),
            "datasets": [{"value": ds, "snapshots": [{"id": s, "trials": n} for s, n in
                                                     sorted(snaps.items(), key=lambda kv: str(kv[0]))]}
                         for ds, snaps in sorted(datasets.items(), key=lambda kv: (kv[0] is None, str(kv[0])))],
            "tasks": [{"value": t["value"], "trials": t["trials"], "digests": len(t["digests"]),
                       "order": order.get(t["value"])}
                      for t in sorted(tasks.values(), key=lambda t: (order.get(t["value"], len(order)), t["value"]))],
            "runs": [{"run_key": k, "profile": (runs.get(k) or {}).get("profile"),
                      "run_id": (runs.get(k) or {}).get("run_id"), "source": (runs.get(k) or {}).get("source"),
                      "created_at": (runs.get(k) or {}).get("created_at"), "status": (runs.get(k) or {}).get("status"),
                      "trials": n}
                     for k, n in sorted(run_trials.items(), key=lambda kv: str((runs.get(kv[0]) or {}).get(
                         "created_at") or ""), reverse=True)],
            "outcomes": [{"value": o, "trials": outcome_counts.get(o, 0)} for o in classify.OUTCOMES],
            "date_range": {"min": min(starts) if starts else None, "max": max(starts) if starts else None},
            "task_order": list(task_order),
        }

    # Exclusions ----------------------------------------------------------------------

    def add_exclusion(self, target_type: str, target_id: str, action: str, reason: str) -> dict:
        """Append an exclusion or re-inclusion. The latest row per target wins;
        nothing is ever deleted."""
        self._require()
        if target_type not in ("trial", "run"):
            raise ValueError("target_type must be trial or run")
        if action not in ("exclude", "include"):
            raise ValueError("action must be exclude or include")
        reason = (reason or "").strip() if isinstance(reason, str) else ""
        if not 1 <= len(reason) <= 500:
            raise ValueError("Give a reason of 1 to 500 characters")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("target_id must be a trial key or run key")
        table, column = ("trials", "trial_key") if target_type == "trial" else ("runs", "run_key")
        created = _now()
        with self._write() as conn:
            if conn.execute(f"SELECT 1 FROM {table} WHERE {column} = ?", (target_id,)).fetchone() is None:
                raise KeyError(f"No {target_type} '{target_id}' in the ledger")
            cur = conn.execute("INSERT INTO exclusions (target_type, target_id, action, reason, created_at) "
                               "VALUES (?, ?, ?, ?, ?)", (target_type, target_id, action, reason, created))
            new_id = cur.lastrowid
        return {"id": new_id, "target_type": target_type, "target_id": target_id, "action": action,
                "reason": reason, "created_at": created}

    def exclusions(self) -> list[dict]:
        self._require()
        with self._conn() as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM exclusions ORDER BY id")]
        latest = {(r["target_type"], r["target_id"]): r["id"] for r in rows}
        for r in rows:
            r["effective"] = latest[(r["target_type"], r["target_id"])] == r["id"]
        return rows

    def effective_exclusions(self) -> tuple[frozenset[str], frozenset[str], tuple[int, ...]]:
        """(excluded trial keys, excluded run keys, ids of the exclude rows in force)."""
        self._require()
        with self._conn() as conn:
            latest = self._exclusion_maps(conn)
        trials, runs, ids = set(), set(), []
        for (target_type, target_id), r in latest.items():
            if r["action"] != "exclude":
                continue
            (trials if target_type == "trial" else runs).add(target_id)
            ids.append(r["id"])
        return frozenset(trials), frozenset(runs), tuple(sorted(ids))

    # Snapshots ---------------------------------------------------------------------

    def save_snapshot(self, name: str, request: dict, response: dict) -> dict:
        self._require()
        name = name.strip() if isinstance(name, str) else ""
        if not 1 <= len(name) <= 80:
            raise ValueError("Give the snapshot a name of 1 to 80 characters")
        provenance = (response or {}).get("provenance") or {}
        created = _now()
        request_json = json.dumps(request, ensure_ascii=False, default=str)
        response_json = json.dumps(response, ensure_ascii=False, default=str)
        fingerprint = str(provenance.get("fingerprint") or "")
        pricing_sha = str(provenance.get("pricing_sha") or pricing.table_sha())
        stats_version = int(provenance.get("stats_version") or _stats_version())
        for _ in range(5):
            sid = f"rs-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
            try:
                with self._write() as conn:
                    conn.execute("INSERT INTO report_snapshots (id, name, created_at, request_json, response_json, "
                                 "fingerprint, pricing_sha, stats_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                 (sid, name, created, request_json, response_json, fingerprint, pricing_sha,
                                  stats_version))
            except sqlite3.IntegrityError:
                continue  # same second and same 4 hex: draw again
            return {"id": sid, "name": name, "created_at": created, "fingerprint": fingerprint,
                    "trials_used": provenance.get("trials_used")}
        raise RuntimeError("Could not allocate a snapshot id; try again")

    def snapshots(self) -> list[dict]:
        self._require()
        with self._conn() as conn:
            rows = conn.execute("SELECT id, name, created_at, fingerprint, "
                                "json_extract(response_json, '$.provenance.trials_used') AS trials_used "
                                "FROM report_snapshots ORDER BY created_at DESC, id DESC").fetchall()
        return [dict(r) for r in rows]

    def snapshot(self, sid: str) -> dict:
        self._require()
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM report_snapshots WHERE id = ?", (sid,)).fetchone()
        if row is None:
            raise KeyError(f"No snapshot '{sid}'")
        return {"id": row["id"], "name": row["name"], "created_at": row["created_at"],
                "request": _loads(row["request_json"], {}), "response": _loads(row["response_json"], {})}

    # Status ---------------------------------------------------------------------------

    def _display_path(self) -> str:
        return extract.relative_dir(self.path, settings.ROOT)

    def status(self) -> dict:
        """Works whether or not the ledger is open."""
        try:
            size = self.path.stat().st_size
        except OSError:
            size = None
        out: dict[str, Any] = {
            "available": self._available, "error": self.error, "path": self._display_path(), "size_bytes": size,
            "schema_version": SCHEMA_VERSION, "parser_version": extract.PARSER_VERSION,
            "classifier_version": classify.CLASSIFIER_VERSION, "stats_version": _stats_version(),
            "trials": 0, "runs": 0, "by_profile": {}, "by_outcome": {}, "last_ingest_at": None,
            "rescan": {"running": self.rescan_running, "last": None}, "last_error": self._last_error,
            "last_backup": None, "pricing_sha": pricing.table_sha(), "seeded_exclusions": [],
            "journal_mode": self.journal_mode, "warnings": list(self.warnings),
        }
        if not self._available:
            return out
        try:
            with self._conn() as conn:
                out["trials"] = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
                out["runs"] = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
                out["by_profile"] = {r[0]: r[1] for r in conn.execute(
                    "SELECT profile, COUNT(*) FROM trials GROUP BY profile ORDER BY profile")}
                out["by_outcome"] = {r[0]: r[1] for r in conn.execute(
                    "SELECT outcome, COUNT(*) FROM trials GROUP BY outcome ORDER BY outcome")}
                meta = {r[0]: r[1] for r in conn.execute("SELECT key, value FROM meta")}
                latest = self._exclusion_maps(conn)
        except sqlite3.Error as exc:
            out["error"] = f"The ledger is busy or locked by another program: {exc}"
            return out
        out["last_ingest_at"] = meta.get("last_ingest_at")
        out["rescan"]["last"] = _loads(meta.get("last_rescan"))
        out["last_error"] = self._last_error or _loads(meta.get("last_error"))
        out["last_backup"] = meta.get("last_backup")
        out["seeded_exclusions"] = [
            {"target_type": t, "target_id": i, "reason": reason,
             "effective_action": (latest.get((t, i)) or {}).get("action")}
            for t, i, reason in SEED_EXCLUSIONS]
        return out

    def version(self) -> tuple[int, str, int]:
        """(max trial id, latest updated_at of trials and runs, max exclusion
        id): changes whenever anything a report reads changes, including a
        run's status, so it is the report cache key."""
        self._require()
        with self._conn() as conn:
            trial_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM trials").fetchone()[0]
            updated = conn.execute("SELECT COALESCE(MAX(x), '') FROM (SELECT MAX(updated_at) AS x FROM trials "
                                   "UNION ALL SELECT MAX(updated_at) FROM runs)").fetchone()[0]
            excl = conn.execute("SELECT COALESCE(MAX(id), 0) FROM exclusions").fetchone()[0]
        return int(trial_id), str(updated), int(excl)

    # Backups and export -----------------------------------------------------------------

    def backup(self, keep: int = BACKUP_KEEP) -> Path | None:
        """At most one copy a day in <ledger>/backups, keeping the newest
        `keep`. The ledger outlives deleted job folders, so this is the only
        way back for them."""
        self._require()
        folder = self.path.parent / "backups"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{self.path.stem}-{datetime.now():%Y%m%d}.sqlite"
        if target.exists():
            return None
        tmp = target.with_name(target.name + ".tmp")
        try:
            src = self._connect()
            try:
                dst = sqlite3.connect(str(tmp))
                try:
                    src.backup(dst)
                finally:
                    dst.close()
            finally:
                src.close()
            _replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        with self._write() as conn:
            self._set_meta(conn, "last_backup", target.name)
        for old in sorted(folder.glob(f"{self.path.stem}-[0-9]*.sqlite"))[:-keep or None]:
            with contextlib.suppress(OSError):
                old.unlink()
        return target

    def export_rows(self, f: Filters, fmt: str = "csv") -> Iterator[str]:
        """TrialRows as CSV (UTF-8 BOM, CRLF, objects as JSON text) or JSON
        lines. Raw JSON is never exported. Validation and the query run on
        the call, not on the first iteration, so a bad format or a busy
        ledger fails before a streamed response has sent its headers."""
        if fmt not in ("csv", "jsonl"):
            raise ValueError("format must be csv or jsonl")
        self._require()
        _, rows = self.query(f, order="started_at ASC")
        return self._export_chunks(rows, fmt)

    @staticmethod
    def _export_chunks(rows: list[dict], fmt: str) -> Iterator[str]:
        if fmt == "jsonl":
            for r in rows:
                yield json.dumps({k: r[k] for k in EXPORT_COLUMNS}, ensure_ascii=False) + "\n"
            return
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(EXPORT_COLUMNS)
        yield "\ufeff" + buf.getvalue()  # the BOM makes Excel read UTF-8
        for r in rows:
            buf.seek(0)
            buf.truncate()
            writer.writerow([_csv_value(r[k]) for k in EXPORT_COLUMNS])
            yield buf.getvalue()


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return "" if value is None else value


# CLI ------------------------------------------------------------------------------

def _cli_roots(args: argparse.Namespace) -> list[ScanRoot]:
    """--root takes a data folder (with jobs/ and runs/, named after its
    profile) or a folder of plain Harbor jobs."""
    given = []
    for raw in args.root or []:
        p = Path(raw)
        if (p / "jobs").is_dir() and (p / "runs").is_dir():
            given.append(ScanRoot(p / "jobs", p.name, p / "runs"))
        else:
            given.append(ScanRoot(p, "external", None))
    if args.only:
        return given
    extra: tuple[Path, ...] = ()
    try:
        extra = settings.load_config().ledger.extra_scan_roots
    except Exception:
        pass  # a broken config must not block a rescan
    return extract.default_roots(extra) + given


def _rebuild(path: Path, roots: list[ScanRoot]) -> int:
    if held_elsewhere(path):
        print("The ledger is in use by a running bench (or another `python -m bench.ledger`). "
              "Stop it, then run rebuild again.")
        return 1
    if path.exists():
        try:
            conn = sqlite3.connect(str(path), timeout=1)
            try:
                conn.execute("BEGIN EXCLUSIVE")
                conn.execute("ROLLBACK")
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc) or "busy" in str(exc):
                print(f"Another program holds a lock on {path.name} ({exc}). Close it and try again.")
                return 1
        except sqlite3.DatabaseError:
            pass  # corrupt: exactly what rebuild is for
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        moved = path.with_name(f"{path.name}.corrupt-{stamp}")
        done: list[tuple[Path, Path]] = []
        try:
            # -wal and -shm first: a WAL left behind without its database would
            # be replayed into the new ledger created under the same name.
            for suffix in ("-wal", "-shm", ""):
                src = path.with_name(path.name + suffix)
                if src.exists():
                    dst = moved.with_name(moved.name + suffix)
                    _replace(src, dst)
                    done.append((src, dst))
        except OSError as exc:
            for src, dst in reversed(done):
                with contextlib.suppress(OSError):
                    _replace(dst, src)
            print(f"Could not move {path.name} aside ({exc.strerror or exc}). "
                  "Close any program that has it open, then run rebuild again.")
            return 1
        print(f"Moved the old ledger to {moved.name}")
    ledger = Ledger(path, roots=roots)
    ledger.open()
    try:
        if not ledger.available:
            print(ledger.error)
            return 1
        print(json.dumps(ledger.rescan(), indent=2, ensure_ascii=False))
        return 0
    finally:
        ledger.close()


def main(argv: list[str] | None = None) -> int:
    with contextlib.suppress(AttributeError, ValueError):
        # Exception text with box-drawing characters crashes under cp1252.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="python -m bench.ledger", description="The bench's trial ledger.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show what the ledger holds")
    for name, text in (("rescan", "ingest every job on disk"),
                       ("rebuild", "with the bench stopped: set the old file aside and rescan")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--root", action="append", metavar="PATH",
                       help="also scan this data folder (jobs/ and runs/) or folder of Harbor jobs")
        p.add_argument("--only", action="store_true", help="scan only the --root folders")
        if name == "rescan":
            p.add_argument("--reparse", action="store_true", help="re-read every trial and re-derive every row")
    export = sub.add_parser("export", help="write trials to a CSV or JSON lines file")
    export.add_argument("--out", required=True, metavar="FILE")
    export.add_argument("--format", choices=("csv", "jsonl"), default="csv")
    args = parser.parse_args(argv)

    path = Path(settings.LEDGER_DB)
    if args.command == "rebuild":
        return _rebuild(path, _cli_roots(args))
    ledger = Ledger(path, roots=_cli_roots(args) if args.command == "rescan" else None)
    ledger.open()
    try:
        if not ledger.available:
            print(ledger.error)
            return 1
        if args.command == "status":
            print(json.dumps(ledger.status(), indent=2, ensure_ascii=False))
        elif args.command == "rescan":
            summary = ledger.rescan(reparse=args.reparse)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            print(json.dumps({"by_outcome": ledger.status()["by_outcome"]}, ensure_ascii=False))
        elif args.command == "export":
            with open(args.out, "w", encoding="utf-8", newline="") as out:
                for chunk in ledger.export_rows(Filters(include_cautioned=True), args.format):
                    out.write(chunk)
            print(f"Wrote {args.out}")
        return 0
    finally:
        ledger.close()


if __name__ == "__main__":
    sys.exit(main())
