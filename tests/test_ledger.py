"""The SQLite ledger: idempotent ingestion, append-only rules, exclusions,
locking, backups, the run manager's hook and the CLI."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest import mock

from tests import ledger_fixtures as lf
from tests.support import BenchTestCase

from bench import classify, extract, ledger, settings
from bench.extract import ScanRoot
from bench.ledger import Filters, Ledger

ROOT = settings.ROOT
ARM = {"name": "pi-glm", "agent": "pi", "provider": "openrouter", "model": "z-ai/glm-5.3"}


class LedgerCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tmp = self.tmpdir()
        self.db = self.tmp / "ledger" / "bench.sqlite"

    def open_ledger(self, roots: list[ScanRoot] | None = None, path: Path | None = None) -> Ledger:
        led = Ledger(path or self.db, roots=roots if roots is not None else [], root=self.tmp)
        led.open()
        self.addCleanup(led.close)
        self.assertTrue(led.available, led.error)
        return led

    def bench_job(self, run_id: str = "r1", status: str = "done", arms: list[dict] | None = None) -> tuple[Path, dict]:
        data = self.tmp / "data"
        run = lf.make_bench_run(data, run_id, arms or [ARM], status=status, profile="dev")
        return data, run

    def ctx(self, run: dict, arm: str = "pi-glm", profile: str = "dev") -> extract.RunContext:
        return extract.context_for_bench_arm(run, arm, profile)

    def rows(self, led: Ledger) -> list[dict]:
        return led.query(Filters(include_cautioned=True), order="started_at ASC")[1]

    def raw_rows(self) -> list[sqlite3.Row]:
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute("SELECT * FROM trials ORDER BY id").fetchall()
        finally:
            conn.close()


class Opening(LedgerCase):
    def test_new_file_gets_schema_version_1(self) -> None:
        self.open_ledger()
        conn = sqlite3.connect(self.db)
        try:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], ledger.SCHEMA_VERSION)
            self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        finally:
            conn.close()

    def test_unopenable_path_leaves_it_disabled(self) -> None:
        folder = self.tmpdir()  # a folder where the file should be
        led = Ledger(folder, roots=[])
        led.open()
        self.addCleanup(led.close)
        self.assertFalse(led.available)
        self.assertIn("could not be opened", led.error or "")
        status = led.status()
        self.assertFalse(status["available"])
        self.assertEqual(status["trials"], 0)
        with self.assertRaises(ledger.LedgerUnavailable):
            led.query(Filters())
        self.assertFalse(led.start_rescan_thread())

    def test_not_a_database_is_disabled_not_replaced(self) -> None:
        self.db.parent.mkdir(parents=True)
        self.db.write_bytes(b"this is not sqlite" * 100)
        led = Ledger(self.db, roots=[])
        led.open()
        self.addCleanup(led.close)
        self.assertFalse(led.available)
        self.assertEqual(self.db.read_bytes()[:18], b"this is not sqlite")

    def test_newer_schema_is_refused(self) -> None:
        self.db.parent.mkdir(parents=True)
        conn = sqlite3.connect(self.db)
        conn.execute("PRAGMA user_version=99")
        conn.close()
        led = Ledger(self.db, roots=[])
        led.open()
        self.addCleanup(led.close)
        self.assertFalse(led.available)
        self.assertIn("newer bench", led.error or "")


class Ingestion(LedgerCase):
    def test_same_job_twice_is_one_row(self) -> None:
        data, run = self.bench_job()
        job = data / "jobs" / "r1" / "pi-glm"
        lf.make_trial(job / "t1")
        led = self.open_ledger()
        first = led.ingest_job(job, self.ctx(run), final=False, via="live")
        second = led.ingest_job(job, self.ctx(run), final=False, via="live")
        self.assertEqual((first["inserted"], second["inserted"], second["unchanged"]), (1, 0, 1))
        # A second ledger on the same file (another bench process) adds nothing either.
        other = self.open_ledger(roots=[ScanRoot(data / "jobs", "dev", data / "runs")])
        summary = other.rescan()
        self.assertEqual((summary["inserted"], summary["unchanged"]), (0, 1))
        self.assertEqual(len(self.raw_rows()), 1)
        self.assertEqual(self.raw_rows()[0]["ingest_via"], "live")

    def test_incomplete_then_upgraded(self) -> None:
        data, run = self.bench_job()
        job = data / "jobs" / "r1" / "pi-glm"
        lf.make_trial(job / "t1", finished=False)
        led = self.open_ledger()
        live = led.ingest_job(job, self.ctx(run), final=False, via="live")
        self.assertEqual(live["incomplete"], 0)  # still running: nothing recorded yet
        self.assertEqual(self.raw_rows(), [])
        final = led.ingest_job(job, self.ctx(run), final=True, via="finalize")
        self.assertEqual(final["incomplete"], 1)
        (row,) = self.raw_rows()
        self.assertEqual(row["outcome"], "incomplete")
        lock_blob = row["raw_lock_z"]
        lf.make_trial(job / "t1")  # the result shows up after all
        again = led.ingest_job(job, self.ctx(run), final=True, via="rescan")
        self.assertEqual(again["upgraded"], 1)
        (row,) = self.raw_rows()
        self.assertEqual(row["outcome"], "passed")
        self.assertIsNotNone(row["raw_result_z"])
        self.assertEqual(row["raw_lock_z"], lock_blob)

    def test_no_incomplete_row_while_the_run_is_running(self) -> None:
        data, _ = self.bench_job(status="running")
        lf.make_trial(data / "jobs" / "r1" / "pi-glm" / "t1", finished=False)
        led = self.open_ledger(roots=[ScanRoot(data / "jobs", "dev", data / "runs")])
        self.assertEqual(led.rescan()["incomplete"], 0)
        self.assertEqual(self.raw_rows(), [])
        run_json = data / "runs" / "r1" / "run.json"
        run = json.loads(run_json.read_text(encoding="utf-8"))
        run["status"] = "cancelled"
        lf.write_json(run_json, run)
        self.assertEqual(led.rescan()["incomplete"], 1)

    def test_a_second_result_never_overwrites(self) -> None:
        data, run = self.bench_job()
        job = data / "jobs" / "r1" / "pi-glm"
        lf.make_trial(job / "t1", reward=1.0)
        led = self.open_ledger()
        led.ingest_job(job, self.ctx(run), final=True, via="live")
        before = self.raw_rows()[0]
        # Same job_id and trial_name in a copied folder, with a different result.
        copy = data / "jobs" / "r1-copy" / "pi-glm"
        lf.make_job(copy)
        lf.make_trial(copy / "t1", reward=0.0, tokens=(5, 0, 1))
        counts = led.ingest_job(copy, self.ctx(run), final=True, via="rescan", reparse=True)
        self.assertEqual(counts["unchanged"], 1)
        (after,) = self.raw_rows()
        self.assertEqual(dict(after), dict(before))

    def test_finalize_retries_a_briefly_locked_result(self) -> None:
        data, run = self.bench_job()
        job = data / "jobs" / "r1" / "pi-glm"
        lf.make_trial(job / "t1")
        led = self.open_ledger()
        real = Path.read_bytes
        misses = [2]

        def held(path: Path) -> bytes:
            if path.name == "result.json" and misses[0] > 0:
                misses[0] -= 1
                raise PermissionError(13, "The process cannot access the file")
            return real(path)

        with mock.patch.object(Path, "read_bytes", held), mock.patch.object(ledger, "FINAL_READ_DELAY", 0.01):
            counts = led.ingest_job(job, self.ctx(run), final=True, via="finalize")
        self.assertEqual((counts["inserted"], counts["incomplete"]), (1, 0))
        self.assertEqual(self.raw_rows()[0]["outcome"], "passed")

    def test_attempts_follow_start_time(self) -> None:
        data, run = self.bench_job()
        job = data / "jobs" / "r1" / "pi-glm"
        later = datetime(2026, 10, 29, 7, 0, tzinfo=lf._T0.tzinfo)
        lf.make_trial(job / "a-second", started_at=later)
        lf.make_trial(job / "b-first")
        lf.make_trial(job / "c-other-task", task="task-b", lock_digest="sha256:" + "e" * 64)
        led = self.open_ledger()
        led.ingest_job(job, self.ctx(run), final=True, via="rescan")
        attempts = {r["trial_name"]: r["attempt"] for r in self.rows(led)}
        self.assertEqual(attempts, {"b-first": 1, "a-second": 2, "c-other-task": 1})

    def test_rederive_updates_columns_but_never_the_blobs(self) -> None:
        led = self.open_ledger(roots=lf.fixture_roots())
        led.rescan()
        before = {r["trial_key"]: (r["raw_result_z"], r["raw_lock_z"], r["ingested_at"]) for r in self.raw_rows()}
        with mock.patch.object(classify, "CLASSIFIER_VERSION", classify.CLASSIFIER_VERSION + 1), \
                mock.patch.object(classify, "classify", lambda f: ("infra", "rules changed")):
            self.assertEqual(led.rederive(), 16)
            self.assertEqual(led.rederive(), 0)  # nothing is stale any more
        rows = self.raw_rows()
        self.assertEqual({r["outcome"] for r in rows}, {"infra"})
        self.assertEqual({r["classifier_version"] for r in rows}, {classify.CLASSIFIER_VERSION + 1})
        after = {r["trial_key"]: (r["raw_result_z"], r["raw_lock_z"], r["ingested_at"]) for r in rows}
        self.assertEqual(after, before)

    def test_forced_rederive_reproduces_the_same_columns(self) -> None:
        led = self.open_ledger(roots=lf.fixture_roots())
        led.rescan()
        keep = ("outcome", "outcome_reason", "config_hash", "config_label", "family", "provider", "model",
                "task_digest", "dataset_snapshot", "n_input", "usage_json", "cap_marker", "started_at")
        before = [tuple(r[c] for c in keep) for r in self.raw_rows()]
        self.assertEqual(led.rederive(force=True), 16)
        self.assertEqual([tuple(r[c] for c in keep) for r in self.raw_rows()], before)


class UnrecordedUsage(LedgerCase):
    """A trial whose usage was not recorded is never a measured $0 trial, and
    ledgers written before that rule upgrade on open, without a re-run."""

    def setUp(self) -> None:
        super().setUp()
        prices = self.tmp / "pricing.yaml"
        prices.write_text("openrouter:\n  glm-5.3: {input: 0.07, output: 1.40, cache_read: 0.069}\n",
                          encoding="utf-8", newline="\n")
        patcher = mock.patch.object(settings, "PRICING", prices)
        patcher.start()
        self.addCleanup(patcher.stop)

    def timed_out(self, job: Path, name: str = "t1", tokens: tuple[int, int, int] | None = (0, 0, 0)) -> Path:
        # Pilot 3's pi-glm: Harbor's 0/0/0 after a 900 s AgentTimeoutError.
        return lf.make_trial(job / name, tokens=tokens, exception_type="AgentTimeoutError", exec_sec=900.07,
                             reward=0.0, message="Agent execution timed out after 900.0 seconds")

    def test_rows_written_by_the_old_rules_upgrade_on_open(self) -> None:
        data, run = self.bench_job()
        job = data / "jobs" / "r1" / "pi-glm"
        self.timed_out(job)
        led = self.open_ledger()
        led.ingest_job(job, self.ctx(run), final=True, via="finalize")
        led.close()
        conn = sqlite3.connect(self.db)
        try:  # as version 1 of the parser and the classifier left it
            conn.execute("UPDATE trials SET parser_version = 1, classifier_version = 1, "
                         "outcome_reason = 'AgentTimeoutError: the agent ran out of its budget'")
            conn.commit()
        finally:
            conn.close()
        reopened = self.open_ledger()  # open() re-derives stale rows from the stored JSON
        (raw,) = self.raw_rows()
        self.assertEqual((raw["parser_version"], raw["classifier_version"]),
                         (extract.PARSER_VERSION, classify.CLASSIFIER_VERSION))
        self.assertEqual((raw["outcome"], raw["outcome_reason"]),
                         ("agent_limit", "AgentTimeoutError: the agent ran out of its budget; no usage recorded"))
        (row,) = self.rows(reopened)
        self.assertEqual((row["n_input"], row["n_cache"], row["n_output"]), (0, 0, 0))  # Harbor's raw fact
        self.assertEqual(row["usage_status"], "unmeasured")
        self.assertIsNone(row["tokens_total"])
        self.assertIsNone(row["tokens_uncached"])
        self.assertIsNone(row["cost_usd"])
        self.assertEqual(reopened.runs()[0]["trials_unmeasured"], 1)

    def test_a_null_token_timeout_is_unmeasured_too(self) -> None:
        data, run = self.bench_job()
        job = data / "jobs" / "r1" / "pi-glm"
        self.timed_out(job, tokens=None)
        led = self.open_ledger()
        led.ingest_job(job, self.ctx(run), final=True, via="finalize")
        (row,) = self.rows(led)
        self.assertEqual((row["outcome"], row["usage_status"], row["tokens_total"], row["cost_usd"]),
                         ("agent_limit", "unmeasured", None, None))

    def test_a_timeout_with_usage_is_a_lower_bound(self) -> None:
        data, run = self.bench_job()
        job = data / "jobs" / "r1" / "pi-glm"
        self.timed_out(job, tokens=(353896, 320384, 82917))
        led = self.open_ledger()
        led.ingest_job(job, self.ctx(run), final=True, via="finalize")
        (row,) = self.rows(led)
        self.assertEqual((row["usage_status"], row["tokens_total"]), ("lower_bound", 353896 + 82917))
        self.assertGreater(row["cost_usd"], 0)
        self.assertEqual((led.runs()[0]["trials_unmeasured"], led.runs()[0]["trials_lower_bound"]), (0, 1))

    def test_an_unmeasured_row_has_no_costs_and_sorts_with_the_nulls(self) -> None:
        data, run = self.bench_job()
        job = data / "jobs" / "r1" / "pi-glm"
        self.timed_out(job, "t0")
        lf.make_trial(job / "t1", tokens=(1000, 0, 100), task="task-b")
        lf.make_trial(job / "t2", tokens=(5000, 0, 100), task="task-c")
        led = self.open_ledger()
        led.ingest_job(job, self.ctx(run), final=True, via="finalize")
        lost = next(r for r in self.rows(led) if r["usage_status"] == "unmeasured")
        stored = next(r for r in self.raw_rows() if r["id"] == lost["id"])
        self.assertEqual(stored["cost_at_ingest_usd"], 0.0)  # the stored figure priced Harbor's zeros
        self.assertEqual((lost["usage_status"], lost["cost_usd"], lost["cost_at_ingest_usd"]),
                         ("unmeasured", None, None))
        for column in ("tokens_total", "cost_usd"):
            for direction in ("ASC", "DESC"):
                with self.subTest(order=f"{column} {direction}"):
                    rows = led.query(Filters(include_cautioned=True), order=f"{column} {direction}")[1]
                    self.assertEqual([r[column] is None for r in rows], [False, False, True])
                    known = [r[column] for r in rows[:2]]
                    self.assertEqual(known, sorted(known, reverse=direction == "DESC"))
                    paged = led.query(Filters(include_cautioned=True), order=f"{column} {direction}", limit=1,
                                      offset=2)[1]
                    self.assertEqual([r["id"] for r in paged], [rows[2]["id"]])

    def test_a_killed_trial_takes_the_usage_its_harness_logged(self) -> None:
        data, run = self.bench_job(status="cancelled")
        job = data / "jobs" / "r1" / "pi-glm"
        trial = lf.make_trial(job / "t1", finished=False)
        led = self.open_ledger(roots=[ScanRoot(data / "jobs", "dev", data / "runs")])
        self.assertEqual(led.rescan()["incomplete"], 1)
        (row,) = self.rows(led)
        self.assertEqual((row["outcome"], row["usage_status"], row["n_input"], row["cost_usd"]),
                         ("incomplete", "unmeasured", None, None))
        # Killed trials are counted as trials['incomplete'] only, as ArmRun's
        # trials_killed counts them, so a page showing both counts each once.
        (killed,) = led.runs()
        self.assertEqual((killed["trials"], killed["trials_unmeasured"], killed["trials_lower_bound"]),
                         ({"incomplete": 1}, 0, 0))
        # The log is read on the next rescan (here it appears late; in a ledger
        # written before parser version 2 it was there all along).
        lf.write_agent_log(trial, "pi", ((1000, 600, 100), (3000, 2400, 250)))
        before = led.version()
        summary = led.rescan()
        self.assertEqual((summary["upgraded"], summary["unchanged"]), (1, 0))
        self.assertNotEqual(led.version(), before)
        (row,) = self.rows(led)
        self.assertEqual((row["outcome"], row["usage_status"]), ("incomplete", "lower_bound"))
        self.assertEqual((row["n_input"], row["n_cache"], row["n_output"]), (4000, 3000, 350))
        self.assertAlmostEqual(row["cost_usd"], (1000 * 0.07 + 3000 * 0.069 + 350 * 1.40) / 1e6)
        self.assertAlmostEqual(row["reported_cost_usd"], 0.002)
        self.assertEqual(row["usage"]["z-ai/glm-5.3"]["recovered_from"], "agent/pi/sessions")
        (spent,) = led.runs()
        self.assertEqual((spent["trials_unmeasured"], spent["trials_lower_bound"]), (0, 0))
        self.assertAlmostEqual(spent["spend_usd"], round((1000 * 0.07 + 3000 * 0.069 + 350 * 1.40) / 1e6, 6))
        self.assertEqual(led.rescan()["unchanged"], 1)  # nothing new: no write, no version change
        # A result after all still upgrades it as before.
        lf.make_trial(trial, reward=1.0)
        self.assertEqual(led.rescan()["upgraded"], 1)
        (row,) = self.rows(led)
        self.assertEqual((row["outcome"], row["usage_status"], row["n_input"]), ("passed", "full", 100000))


class Golden(LedgerCase):
    def test_rescan_of_the_pilots(self) -> None:
        led = self.open_ledger(roots=lf.fixture_roots())
        summary = led.rescan()
        self.assertEqual((summary["scanned"], summary["inserted"], summary["incomplete"]), (16, 15, 1))
        self.assertEqual(summary["skipped"], [])
        status = led.status()
        self.assertEqual(status["by_outcome"], lf.GOLDEN_COUNTS)
        self.assertEqual((status["trials"], status["runs"], status["by_profile"]), (16, 2, {"dev": 16}))
        self.assertEqual(status["seeded_exclusions"], [
            {"target_type": "run", "target_id": f"dev/{lf.PILOT_1}",
             "reason": "Pilot 1 ran on git tasks damaged by CRLF line endings", "effective_action": "exclude"}])
        got = {(r["run_id"], r["arm"]): r["outcome"] for r in self.rows(led)}
        self.assertEqual(got, lf.GOLDEN)
        runs = {r["run_key"]: r for r in led.runs()}
        self.assertEqual(runs[f"dev/{lf.PILOT_1}"]["exclusion"]["action"], "exclude")
        self.assertEqual(runs[f"dev/{lf.PILOT_2}"]["trials"]["setup_failure"], 4)
        self.assertEqual(runs[f"dev/{lf.PILOT_2}"]["status"], "failed")

    def test_trial_detail_and_rows(self) -> None:
        led = self.open_ledger(roots=lf.fixture_roots())
        led.rescan()
        (cc,) = led.query(Filters(models=("openrouter:deepseek/deepseek-v4.1-flash",), families=("claude-code",),
                                  runs=(f"dev/{lf.PILOT_2}",)))[1]
        self.assertEqual(cc["outcome"], "misconfigured_cap")
        self.assertAlmostEqual(cc["reported_cost_usd"], 1.438498)
        self.assertLess(cc["cost_usd"], 0.05)  # the harness priced DeepSeek at Claude rates
        self.assertEqual(cc["task_digest_short"], "b4c60177f24d")
        self.assertIsNone(cc["log_url"])  # dev trials, default-profile server
        detail = led.trial(cc["id"])
        self.assertIn("error_max_budget_usd", detail["exception_message"])
        self.assertEqual(detail["config"]["options"], {"max_budget_usd": 1, "max_turns": 150})
        self.assertNotIn("rollout_details", json.dumps(detail["raw"]))
        self.assertTrue(detail["trial_dir_exists"])
        with self.assertRaises(KeyError):
            led.trial(999999)

    def test_filters(self) -> None:
        led = self.open_ledger(roots=lf.fixture_roots())
        led.rescan()
        count = lambda **kw: led.query(Filters(**kw))[0]  # noqa: E731
        self.assertEqual(count(), 16)
        self.assertEqual(count(efforts=(None,)), 3)  # strands passes no effort
        self.assertEqual(count(efforts=("medium", None)), 16)
        self.assertEqual(count(snapshots=("git:69671fbaac6d",)), 7)
        self.assertEqual(count(exclude_runs=(f"dev/{lf.PILOT_1}",)), 9)
        self.assertEqual(count(date_from="2026-10-29", date_to="2026-10-29"), 16)
        self.assertEqual(count(date_to="2026-10-28"), 0)
        self.assertEqual(count(date_from="2026-10-29T05:30:00Z"), 9)
        total, rows = led.query(Filters(), outcomes=["setup_failure"], limit=3, offset=0, order="tokens_total DESC")
        self.assertEqual((total, len(rows)), (10, 3))
        by_cost = led.query(Filters(), order="cost_usd DESC")[1]
        self.assertEqual(by_cost[0]["arm"], "pi-glm")
        with self.assertRaises(ValueError):
            led.query(Filters(), order="id; DROP TABLE trials")
        with self.assertRaises(ValueError):
            led.query(Filters(), limit=5000)
        with self.assertRaises(ValueError):
            led.query(Filters(), outcomes=["nope"])

    def test_parse_filters(self) -> None:
        f = ledger.parse_filters({"models": ["openrouter:z-ai/glm-5.3"], "efforts": ["medium", None],
                                  "date_to": "2026-10-29", "include_cautioned": True})
        self.assertEqual(f.models, ("openrouter:z-ai/glm-5.3",))
        self.assertEqual(f.efforts, ("medium", None))
        self.assertEqual(ledger.parse_filters(None), Filters())
        for bad in ({"nope": []}, {"date_from": "yesterday"}, {"tasks": "a"}, {"tasks": ["x" * 301]},
                    {"tasks": ["t"] * 501}, {"sources": ["somewhere"]}, {"models": ["glm"]},
                    {"include_cautioned": "yes"}, {"tasks": [1]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ledger.parse_filters(bad)

    def test_facets_and_version(self) -> None:
        led = self.open_ledger(roots=lf.fixture_roots())
        v0 = led.version()
        led.rescan()
        v1 = led.version()
        self.assertNotEqual(v0, v1)
        facets = led.facets(["adaptive-rejection-sampler", "protein-assembly"])
        self.assertEqual(facets["tasks"], [{"value": "adaptive-rejection-sampler", "trials": 16, "digests": 2,
                                            "order": 0}])
        self.assertEqual({m["key"] for m in facets["models"]},
                         {"openrouter:deepseek/deepseek-v4.1-flash", "openrouter:z-ai/glm-5.3"})
        self.assertEqual(sum(o["trials"] for o in facets["outcomes"]), 16)
        self.assertEqual({d["value"] for d in facets["datasets"]}, {"terminal-bench@2.0"})
        led.add_exclusion("trial", self.rows(led)[0]["trial_key"], "exclude", "test")
        self.assertNotEqual(led.version(), v1)

    def test_version_follows_a_run_status_change_and_only_a_change(self) -> None:
        data, run = self.bench_job(status="running")
        job = data / "jobs" / "r1" / "pi-glm"
        lf.make_trial(job / "t1")
        led = self.open_ledger()
        led.upsert_bench_run(run)
        led.ingest_bench_job(job, run, "pi-glm", final=False, via="live")
        v1 = led.version()
        led.upsert_bench_run(run)  # the same state again, as every rescan does
        led.ingest_bench_job(job, run, "pi-glm", final=True, via="finalize")  # nothing new on disk
        self.assertEqual(led.version(), v1)
        time.sleep(0.02)  # coarse clocks: make the next timestamp differ
        led.upsert_bench_run({**run, "status": "done", "finished_at": "2026-10-29T06:00:00+00:00"})
        v2 = led.version()
        self.assertNotEqual(v2, v1)
        self.assertEqual((v2[0], v2[2]), (v1[0], v1[2]))  # no new trial, no new exclusion


class Rescans(LedgerCase):
    def test_one_rescan_at_a_time(self) -> None:
        led = self.open_ledger(roots=lf.fixture_roots())
        with led._rescan_lock:
            self.assertTrue(led.rescan_running)
            with self.assertRaisesRegex(RuntimeError, "already running"):
                led.rescan()
            self.assertFalse(led.start_rescan_thread())
        self.assertTrue(led.start_rescan_thread())
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and (led.rescan_running or led.status()["trials"] < 16):
            time.sleep(0.05)
        status = led.status()
        self.assertEqual(status["trials"], 16)
        self.assertEqual(status["rescan"]["last"]["inserted"], 15)

    def test_external_jobs(self) -> None:
        ext = self.tmp / "harbor-jobs"
        job = lf.make_external_job(ext, "2026-10-29__09-00-00")
        lf.make_trial(job / "t1", model_name="fireworks_ai/accounts/fireworks/models/glm-5p3", family="terminus-2")
        other = lf.make_external_job(ext, "2026-10-29__10-00-00")
        lf.make_trial(other / "t1", model_name="acme/m1", family="terminus-2",
                      job_id="33333333-3333-4333-8333-333333333333")
        prefixes = {"fireworks_ai/accounts/fireworks/models/": "fireworks", "openrouter/": "openrouter"}
        with mock.patch.object(extract, "provider_prefixes", lambda config_dir=None: dict(prefixes)):
            led = self.open_ledger(roots=[ScanRoot(ext, "external", None)])
            summary = led.rescan()
        rows = {r["run_id"]: r for r in self.rows(led)}
        first = rows["2026-10-29__09-00-00"]
        self.assertEqual((first["source"], first["profile"], first["arm"]), ("external", "external",
                                                                            "2026-10-29__09-00-00"))
        self.assertEqual((first["provider"], first["model"]), ("fireworks", "glm-5p3"))
        self.assertEqual(first["run_key"], "external/harbor-jobs/2026-10-29__09-00-00")
        self.assertEqual(first["dataset"], "bench-tests")
        self.assertEqual(rows["2026-10-29__10-00-00"]["provider"], "acme")
        notes = [s["reason"] for s in summary["skipped"]]
        self.assertEqual(len(notes), 1)
        self.assertIn("guessed", notes[0])
        runs = {r["run_key"]: r for r in led.runs()}
        self.assertEqual(runs["external/harbor-jobs/2026-10-29__09-00-00"]["status"], "finished")

    def test_journal_mode_for_network_and_onedrive_folders(self) -> None:
        self.assertTrue(ledger._needs_delete_journal(Path(r"\\server\share\ledger\bench.sqlite")))
        self.assertTrue(ledger._needs_delete_journal(Path(r"C:\Users\me\OneDrive\bench\ledger\bench.sqlite")))
        self.assertFalse(ledger._needs_delete_journal(Path(r"C:\Git\EngineeringTestBench\ledger\bench.sqlite")))


class Exclusions(LedgerCase):
    def test_append_only_latest_wins(self) -> None:
        led = self.open_ledger(roots=lf.fixture_roots())
        led.rescan()
        key = self.rows(led)[-1]["trial_key"]
        first = led.add_exclusion("trial", key, "exclude", "looked wrong")
        led.add_exclusion("trial", key, "include", "it was fine")
        rows = [r for r in led.exclusions() if r["target_id"] == key]
        self.assertEqual([(r["action"], r["effective"]) for r in rows], [("exclude", False), ("include", True)])
        trials, runs, ids = led.effective_exclusions()
        self.assertNotIn(key, trials)
        self.assertIn(f"dev/{lf.PILOT_1}", runs)
        self.assertNotIn(first["id"], ids)
        with self.assertRaises(KeyError):
            led.add_exclusion("trial", "no/such-trial", "exclude", "x")
        with self.assertRaises(ValueError):
            led.add_exclusion("run", f"dev/{lf.PILOT_2}", "exclude", "  ")
        with self.assertRaises(ValueError):
            led.add_exclusion("run", f"dev/{lf.PILOT_2}", "exclude", "x" * 501)
        with self.assertRaises(ValueError):
            led.add_exclusion("task", "x", "exclude", "x")

    def test_seeded_once_and_an_include_sticks(self) -> None:
        led = self.open_ledger(roots=lf.fixture_roots())
        led.rescan()
        target = f"dev/{lf.PILOT_1}"
        seeded = [r for r in led.exclusions() if r["target_id"] == target]
        self.assertEqual(len(seeded), 1)
        led.add_exclusion("run", target, "include", "checked by hand")
        led.close()
        again = self.open_ledger(roots=lf.fixture_roots())
        rows = [r for r in again.exclusions() if r["target_id"] == target]
        self.assertEqual([r["action"] for r in rows], ["exclude", "include"])
        self.assertNotIn(target, again.effective_exclusions()[1])
        self.assertEqual(again.status()["seeded_exclusions"][0]["effective_action"], "include")


class Snapshots(LedgerCase):
    def test_round_trip(self) -> None:
        led = self.open_ledger()
        saved = led.save_snapshot("week 1", {"filters": {}}, {"provenance": {"fingerprint": "sha256:abc",
                                                                             "trials_used": 12,
                                                                             "pricing_sha": "p", "stats_version": 1}})
        self.assertRegex(saved["id"], r"^rs-\d{8}-\d{6}-[0-9a-f]{4}$")
        self.assertEqual(led.snapshots(), [{k: saved[k] for k in ("id", "name", "created_at", "fingerprint",
                                                                  "trials_used")}])
        self.assertEqual(led.snapshot(saved["id"])["response"]["provenance"]["trials_used"], 12)
        with self.assertRaises(KeyError):
            led.snapshot("rs-nope")
        with self.assertRaises(ValueError):
            led.save_snapshot("", {}, {})


class Hook(LedgerCase):
    def test_ledger_hook_with_a_run_manager_run(self) -> None:
        msa = "bench.agents.mini_swe_agent:MiniSweAgentWithMcp"
        arms = [{"name": "msa", "agent": msa, "family": "mini-swe-agent", "provider": "anthropic",
                 "model": "claude-opus-5-5"}]
        data, run = self.bench_job(run_id="20261029-101500-my-design-v2", arms=arms)
        job = data / "jobs" / run["id"] / "msa"
        lf.make_trial(job / "t1", family="mini-swe-agent", model_name="anthropic/claude-opus-5-5", import_path=msa,
                      tokens=(1_000_000, 900_000, 10_000))
        lf.make_trial(job / "t2", finished=False, family="mini-swe-agent", model_name="anthropic/claude-opus-5-5",
                      import_path=msa)
        led = self.open_ledger()
        led.upsert_bench_run(run)
        live = led.ingest_bench_job(job, run, "msa", final=False, via="live")
        self.assertEqual((live["inserted"], live["incomplete"]), (1, 0))
        final = led.ingest_bench_job(job, run, "msa", final=True, via="finalize")
        self.assertEqual((final["unchanged"], final["incomplete"]), (1, 1))
        run_key = f"{settings.PROFILE_NAME}/{run['id']}"
        rows = self.rows(led)
        done = next(r for r in rows if r["outcome"] != "incomplete")
        self.assertEqual((done["run_key"], done["family"], done["implementation"], done["provider"], done["model"]),
                         (run_key, "mini-swe-agent", "bench.agents.mini_swe_agent:MiniSweAgentWithMcp", "anthropic",
                          "claude-opus-5-5"))
        self.assertEqual(done["effort_label"], "medium")
        self.assertEqual(done["dataset_snapshot"], lf.PILOT_2_SNAPSHOT)
        self.assertAlmostEqual(done["cost_usd"], (100_000 * 4 + 900_000 * 0.2 + 10_000 * 20) / 1e6)
        self.assertEqual(done["log_url"], f"/api/runs/{run['id']}/log/msa")
        (mirrored,) = [r for r in led.runs() if r["run_key"] == run_key]
        self.assertEqual((mirrored["status"], mirrored["budget_usd"]), ("done", 5.0))

    def test_run_json_is_stored_redacted_and_still_parses(self) -> None:
        _, run = self.bench_job()
        run["config"]["providers"]["openrouter"]["base_url"] = "https://proxy.example/v1?key=sk-abcdefghijklmnop"
        run["plan"]["arms"][0]["command"] = "harbor run --ae Authorization='Bearer tok.en.value'"
        run["config"]["providers"]["openrouter"]["env"] = {"OPENROUTER_API_KEY": "not-for-the-ledger"}
        led = self.open_ledger()
        led.upsert_bench_run(run)
        conn = sqlite3.connect(self.db)
        try:
            stored = conn.execute("SELECT plan_json, config_json FROM runs").fetchone()
        finally:
            conn.close()
        for text in stored:
            json.loads(text)  # still valid JSON after redaction
            for secret in ("sk-abcdefghijklmnop", "tok.en.value", "not-for-the-ledger"):
                self.assertNotIn(secret, text)

    def test_failures_surface_but_ingest_is_never_fatal_to_the_ledger(self) -> None:
        led = self.open_ledger()
        led.set_error("live", RuntimeError("disk full"))
        self.assertTrue(led.available)
        self.assertIn("disk full", led.error or "")
        self.assertEqual(led.status()["last_error"]["where"], "live")
        with self.assertRaises(ValueError):
            led.ingest_job(self.tmp, extract.context_for_external(self.tmp), final=False, via="sideways")


class Locking(LedgerCase):
    def test_busy_beyond_the_timeout_raises_operational_error(self) -> None:
        led = self.open_ledger(roots=lf.fixture_roots())
        led.rescan()
        holder = sqlite3.connect(self.db, timeout=0.1, isolation_level=None)
        self.addCleanup(holder.close)
        holder.execute("BEGIN EXCLUSIVE")
        try:
            with mock.patch.object(ledger, "BUSY_TIMEOUT_MS", 200):
                with self.assertRaises(sqlite3.OperationalError):
                    led.add_exclusion("run", f"dev/{lf.PILOT_2}", "exclude", "busy test")
        finally:
            holder.execute("ROLLBACK")

    def test_a_busy_database_holds_the_live_refresh_only_briefly(self) -> None:
        data, run = self.bench_job(status="running")
        job = data / "jobs" / "r1" / "pi-glm"
        lf.make_trial(job / "t1")
        led = self.open_ledger()
        holder = sqlite3.connect(self.db, timeout=0.1, isolation_level=None)
        self.addCleanup(holder.close)
        holder.execute("BEGIN IMMEDIATE")  # another program mid-write
        try:
            started = time.monotonic()
            with self.assertRaises(sqlite3.OperationalError) as caught:
                led.ingest_bench_job(job, run, "pi-glm", final=False, via="live")
            led.set_error("live", caught.exception)  # what the run manager's guard does next
            self.assertLess(time.monotonic() - started, 3.0)
        finally:
            holder.execute("ROLLBACK")
        self.assertTrue(led.available)
        self.assertIn("locked", led.error or "")
        # The next refresh catches up once the other program lets go.
        self.assertEqual(led.ingest_bench_job(job, run, "pi-glm", final=False, via="live")["inserted"], 1)

    def test_a_long_write_in_this_process_never_blocks_the_live_refresh(self) -> None:
        data, run = self.bench_job(status="running")
        job = data / "jobs" / "r1" / "pi-glm"
        lf.make_trial(job / "t1")
        led = self.open_ledger()
        held, release = threading.Event(), threading.Event()

        def hold() -> None:
            with ledger._WRITE_LOCK:
                held.set()
                release.wait(30)

        thread = threading.Thread(target=hold)
        thread.start()
        self.addCleanup(thread.join, 30)
        self.addCleanup(release.set)
        self.assertTrue(held.wait(5))
        started = time.monotonic()
        led.set_error("live", RuntimeError("disk full"))
        self.assertLess(time.monotonic() - started, 0.5)
        with mock.patch.object(ledger, "LIVE_BUSY_MS", 200), self.assertRaises(sqlite3.OperationalError):
            led.ingest_bench_job(job, run, "pi-glm", final=False, via="live")
        self.assertLess(time.monotonic() - started, 2.0)

    def test_concurrent_ingest_and_a_subprocess_rescan(self) -> None:
        data = self.tmp / "dev"
        shutil.copytree(lf.FIXTURE_DEV, data)
        led = self.open_ledger(roots=[ScanRoot(data / "jobs", "dev", data / "runs")])
        jobs = [(job, json.loads((data / "runs" / job.parent.name / "run.json").read_text(encoding="utf-8")))
                for job in sorted((data / "jobs").glob("*/*")) if job.is_dir()]
        reader = sqlite3.connect(self.db, isolation_level=None)
        self.addCleanup(reader.close)
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM trials").fetchone()  # holds a read snapshot throughout
        env = {**os.environ, "BENCH_LEDGER": str(self.db), "PYTHONPATH": str(ROOT)}
        proc = subprocess.Popen([sys.executable, "-m", "bench.ledger", "rescan", "--only", "--root", str(data)],
                                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8")
        errors: list[BaseException] = []

        passes = [0] * 4

        def work(i: int) -> None:
            # Keep writing until the subprocess is done, so the two really overlap;
            # reparse makes every trial a write transaction.
            try:
                while passes[i] < 2 or (proc.poll() is None and passes[i] < 200):
                    for job, run in jobs:
                        led.ingest_job(job, extract.context_for_bench_arm(run, job.name, "dev"), final=True,
                                       via="live", reparse=True)
                    passes[i] += 1
            except BaseException as exc:  # reported below
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        out, err = proc.communicate(timeout=120)
        reader.execute("COMMIT")
        self.assertEqual(errors, [])
        self.assertEqual(proc.returncode, 0, err[-2000:])
        rows = self.raw_rows()
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({r["trial_key"] for r in rows}), 16)
        self.assertIn('"scanned": 16', out)

    def test_rebuild_refuses_while_in_use(self) -> None:
        led = self.open_ledger()
        self.assertTrue(ledger.held_elsewhere(self.db))
        with mock.patch.object(settings, "LEDGER_DB", self.db), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(ledger.main(["rebuild", "--only"]), 1)
        self.assertIn("in use", out.getvalue())
        self.assertTrue(self.db.exists())
        led.close()
        self.assertFalse(ledger.held_elsewhere(self.db))

    def test_rebuild_sets_the_old_file_aside(self) -> None:
        led = self.open_ledger(roots=lf.fixture_roots())
        led.rescan()
        led.close()
        roots = ["--only", "--root", str(lf.FIXTURE_DEV)]
        with mock.patch.object(settings, "LEDGER_DB", self.db), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(ledger.main(["rebuild", *roots]), 0)
        self.assertIn('"inserted": 15', out.getvalue())
        set_aside = [p for p in self.db.parent.glob("bench.sqlite.corrupt-*")
                     if not p.name.endswith(("-wal", "-shm"))]
        self.assertEqual(len(set_aside), 1)
        again = self.open_ledger()
        self.assertEqual(again.status()["trials"], 16)

    def test_rebuild_leaves_the_ledger_in_place_when_it_cannot_move_it(self) -> None:
        led = self.open_ledger(roots=lf.fixture_roots())
        led.rescan()
        led.close()
        real = ledger._replace

        def held(src: Path, dst: Path, patience: float | None = None) -> None:
            if src.name == self.db.name:
                raise PermissionError(13, "The process cannot access the file")
            real(src, dst, patience)

        with mock.patch.object(ledger, "_replace", held), mock.patch.object(settings, "LEDGER_DB", self.db), \
                redirect_stdout(io.StringIO()) as out:
            self.assertEqual(ledger.main(["rebuild", "--only"]), 1)
        self.assertIn("Could not move bench.sqlite aside", out.getvalue())
        self.assertEqual(list(self.db.parent.glob("bench.sqlite.corrupt-*")), [])
        again = self.open_ledger()
        self.assertEqual(again.status()["trials"], 16)


class Backups(LedgerCase):
    def test_one_a_day_keeping_seven(self) -> None:
        led = self.open_ledger()
        folder = self.db.parent / "backups"
        today = folder / f"bench-{datetime.now():%Y%m%d}.sqlite"
        self.assertTrue(today.exists())  # open() takes the day's backup
        self.assertIsNone(led.backup())
        today.unlink()
        for day in range(1, 10):
            (folder / f"bench-202601{day:02d}.sqlite").write_bytes(b"old")
        self.assertEqual(led.backup(), today)
        kept = sorted(p.name for p in folder.glob("bench-*.sqlite"))
        self.assertEqual(len(kept), 7)
        self.assertEqual(kept[-1], today.name)
        self.assertNotIn("bench-20260101.sqlite", kept)
        conn = sqlite3.connect(today)
        try:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 1)
        finally:
            conn.close()

    def test_a_held_backup_file_is_waited_out_and_cleaned_up_on_give_up(self) -> None:
        led = self.open_ledger()
        folder = self.db.parent / "backups"
        today = folder / f"bench-{datetime.now():%Y%m%d}.sqlite"
        today.unlink()
        real = os.replace
        calls = [0]

        def held_once(src: object, dst: object) -> None:
            calls[0] += 1
            if calls[0] == 1:
                raise PermissionError(13, "The process cannot access the file")
            real(src, dst)

        with mock.patch.object(ledger.os, "replace", held_once):
            self.assertEqual(led.backup(), today)
        self.assertEqual(calls[0], 2)
        today.unlink()

        def always_held(src: object, dst: object) -> None:
            raise PermissionError(13, "The process cannot access the file")

        with mock.patch.object(ledger.os, "replace", always_held), mock.patch.object(ledger, "REPLACE_PATIENCE", 0.2):
            with self.assertRaises(PermissionError):
                led.backup()
        self.assertFalse(today.exists())
        self.assertEqual(list(folder.glob("*.tmp")), [])


class CommandLine(LedgerCase):
    def test_status_rescan_and_export(self) -> None:
        out_csv = self.tmp / "trials.csv"
        with mock.patch.object(settings, "LEDGER_DB", self.db):
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(ledger.main(["rescan", "--only", "--root", str(lf.FIXTURE_DEV)]), 0)
                self.assertEqual(ledger.main(["status"]), 0)
                self.assertEqual(ledger.main(["export", "--out", str(out_csv)]), 0)
        text = buf.getvalue()
        self.assertIn('"misconfigured_cap": 2', text)
        self.assertIn("dev/20261028-222401-pilot", text)
        data = out_csv.read_bytes()
        self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
        self.assertIn(b"\r\n", data)
        table = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))
        self.assertEqual(table[0][:3], ["id", "trial_key", "run_key"])
        self.assertEqual(len(table), 17)
        self.assertNotIn("raw_result", data.decode("utf-8-sig"))

    def test_export_fails_on_the_call_not_mid_stream(self) -> None:
        led = self.open_ledger(roots=lf.fixture_roots())
        led.rescan()
        with self.assertRaises(ValueError):
            led.export_rows(Filters(), "xml")  # raised by the call itself, before any iteration
        lines = list(led.export_rows(Filters(include_cautioned=True), "jsonl"))
        self.assertEqual(len(lines), 16)
        self.assertEqual(json.loads(lines[0])["run_key"], f"dev/{lf.PILOT_1}")
        led.close()
        with self.assertRaises(ledger.LedgerUnavailable):
            led.export_rows(Filters(), "csv")


if __name__ == "__main__":
    unittest.main()
