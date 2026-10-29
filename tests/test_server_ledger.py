"""Ledger and report endpoints over a temp ledger of the two dev pilots.

The ledger is built from the read-only fixture copies in tests/fixtures/ledger
(the seeded exclusion leaves pilot 1 out of reports), so the numbers are the
golden ones the ledger and report tests use. The per-run results endpoint,
which the LIVE view polls, is checked against results.run_results directly.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from typing import Any
from unittest import mock

from tests import ledger_fixtures as lf
from tests.test_server import ServerCase

from bench import procs, report, results
from bench.ledger import EXPORT_COLUMNS, Ledger
from bench.runs import RunManager
from bench.server import LEDGER_BUSY, TRIAL_ROW_KEYS

LEDGER_YAML = """\
dataset: terminal-bench@2.0
providers:
  openrouter: {env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}
modes:
  pilot: {n_tasks: 1}
run: {budget_usd: 5}
baseline_arm: claude-code-deepseek
arms:
  - {name: claude-code-deepseek, agent: claude-code, provider: openrouter, model: deepseek/deepseek-v4.1-flash}
  - {name: pi-deepseek, agent: pi, provider: openrouter, model: deepseek/deepseek-v4.1-flash}
"""
SAMPLE = ["adaptive-rejection-sampler", "protein-assembly", "path-tracing"]

STATUS_KEYS = {"available", "error", "path", "size_bytes", "schema_version", "parser_version", "classifier_version",
               "stats_version", "trials", "runs", "by_profile", "by_outcome", "last_ingest_at", "rescan", "last_error",
               "last_backup", "pricing_sha"}
RESCAN_KEYS = {"at", "scanned", "inserted", "upgraded", "unchanged", "incomplete", "skipped", "seconds"}
FACET_KEYS = {"profiles", "sources", "models", "families", "configs", "harness_versions", "harbor_versions",
              "efforts", "datasets", "tasks", "runs", "outcomes", "date_range", "task_order"}
RUN_KEYS = {"run_key", "profile", "run_id", "source", "mode", "status", "created_at", "finished_at", "budget_usd",
            "trials", "spend_usd", "trials_unmeasured", "trials_lower_bound", "families", "models", "dataset_snapshot",
            "exclusion"}
TRIAL_ROW = {"id", "trial_key", "run_key", "profile", "run_id", "source", "arm", "trial_name", "attempt", "family",
             "implementation", "harness_version", "harbor_version", "provider", "model", "effort", "effort_label",
             "config_hash", "config_label", "options", "cautions", "dataset", "dataset_snapshot", "task",
             "task_digest_short", "outcome", "outcome_reason", "harbor_cause", "exception_type", "exception_head",
             "reward", "passed", "n_input", "n_cache", "n_output", "usage_status", "tokens_total", "cost_usd",
             "cost_at_ingest_usd", "reported_cost_usd", "unpriced_models", "primary_share", "started_at",
             "finished_at", "phases", "total_sec", "exclusion", "log_url"}
DETAIL_KEYS = TRIAL_ROW | {"usage", "config", "exception_message", "raw", "lock", "trial_dir", "trial_dir_exists"}
EXCLUSION_KEYS = {"id", "target_type", "target_id", "action", "reason", "created_at"}
REPORT_KEYS = {"spec", "provenance", "totals", "groups", "comparisons", "coverage", "warnings"}
COVERAGE_KEYS = {"provenance", "task_order", "untouched_tasks", "comparisons"}
SNAPSHOT_KEYS = {"id", "name", "created_at", "fingerprint", "trials_used"}


class LedgerServerCase(ServerCase):
    YAML = LEDGER_YAML
    TASKS = SAMPLE

    def setUp(self) -> None:
        report.clear_cache()
        self.addCleanup(report.clear_cache)
        super().setUp()

    def make_ledger(self) -> Ledger:
        root = self.tmpdir()
        ledger = Ledger(root / "ledger" / "bench.sqlite", roots=lf.fixture_roots(), root=root)
        ledger.open()
        self.assertTrue(ledger.available, ledger.error)
        ledger.rescan()
        return ledger

    def comparison_key(self) -> str:
        body = self.ok(self.client.post("/api/report", json={}))
        self.assertTrue(body["comparisons"])
        return body["comparisons"][0]["key"]


class LedgerEndpoints(LedgerServerCase):
    def test_the_guard_covers_the_ledger_and_report_routes(self) -> None:
        anonymous = self.make_client(self.state, signed_in=False)
        self.assertEqual(anonymous.get("/api/ledger/status").status_code, 401)
        self.assertEqual(anonymous.post("/api/report", json={}).status_code, 401)
        no_header = self.make_client(self.state, x_bench=False)
        self.assertEqual(no_header.get("/api/ledger/status").status_code, 200)
        for path in ("/api/report", "/api/ledger/trials", "/api/ledger/rescan", "/api/ledger/export",
                     "/api/ledger/exclusions", "/api/report/snapshots"):
            with self.subTest(path):
                self.assertEqual(no_header.post(path, json={}).status_code, 403)

    def test_status_rescan_facets_and_runs(self) -> None:
        status = self.ok(self.client.get("/api/ledger/status"))
        self.assertLessEqual(STATUS_KEYS, set(status))
        self.assertEqual((status["available"], status["trials"]), (True, 16))
        rescan = self.ok(self.client.post("/api/ledger/rescan", json={}))
        self.assertEqual(set(rescan), RESCAN_KEYS)
        self.assertEqual(rescan["inserted"], 0)
        facets = self.ok(self.client.get("/api/ledger/facets"))
        self.assertEqual(set(facets), FACET_KEYS)
        self.assertEqual(facets["task_order"], SAMPLE)
        runs = self.ok(self.client.get("/api/ledger/runs"))
        self.assertEqual({r["run_id"] for r in runs}, {lf.PILOT_1, lf.PILOT_2})
        for run in runs:
            self.assertLessEqual(RUN_KEYS, set(run))
        excluded = next(r for r in runs if r["run_id"] == lf.PILOT_1)
        self.assertEqual(excluded["exclusion"]["action"], "exclude")

    def test_trials_list_and_one_trial(self) -> None:
        listed = self.ok(self.client.post("/api/ledger/trials", json={"limit": 5}))
        self.assertEqual(set(listed), {"total", "rows"})
        self.assertEqual((listed["total"], len(listed["rows"])), (16, 5))
        self.assertEqual(set(listed["rows"][0]), TRIAL_ROW)
        self.assertEqual(set(TRIAL_ROW_KEYS), TRIAL_ROW)
        passed = self.ok(self.client.post("/api/ledger/trials", json={"outcomes": ["passed"],
                                                                       "filters": {"profiles": ["dev"]}}))
        self.assertEqual(passed["total"], 1)
        one = self.ok(self.client.get(f"/api/ledger/trials/{passed['rows'][0]['id']}"))
        self.assertLessEqual(DETAIL_KEYS, set(one))
        self.assertEqual(one["outcome"], "passed")
        self.assertIsNone(one["log_url"])  # a dev trial, and this server is the default profile

    def test_exclusions_add_and_list(self) -> None:
        trial = self.ok(self.client.post("/api/ledger/trials", json={"limit": 1}))["rows"][0]
        empty = self.client.post("/api/ledger/exclusions", json={"target_type": "trial",
                                                                  "target_id": trial["trial_key"],
                                                                  "action": "exclude", "reason": "  "})
        self.assertEqual(empty.status_code, 400)
        added = self.client.post("/api/ledger/exclusions", json={"target_type": "trial",
                                                                  "target_id": trial["trial_key"],
                                                                  "action": "exclude", "reason": "Docker died"})
        self.assertEqual(added.status_code, 201, added.text)
        self.assertEqual(set(added.json()), EXCLUSION_KEYS)
        unknown = self.client.post("/api/ledger/exclusions", json={"target_type": "run", "target_id": "dev/nope",
                                                                    "action": "exclude", "reason": "x"})
        self.assertEqual(unknown.status_code, 404)
        listed = self.ok(self.client.get("/api/ledger/exclusions"))["exclusions"]
        self.assertEqual([e["target_id"] for e in listed], ["dev/" + lf.PILOT_1, trial["trial_key"]])
        self.assertTrue(all(e["effective"] for e in listed))

    def test_trial_export_csv_and_jsonl(self) -> None:
        response = self.client.post("/api/ledger/export", json={"format": "csv"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.headers["content-type"].startswith("text/csv"))
        self.assertRegex(response.headers["content-disposition"], r'attachment; filename="ledger-\d{8}T\d{6}Z\.csv"')
        raw = response.content
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        text = raw.decode("utf-8-sig")
        lines = text.split("\r\n")
        self.assertEqual(lines[0], ",".join(EXPORT_COLUMNS))
        self.assertEqual(len([line for line in lines if line]), 17)
        self.assertNotIn("raw_result", text)

        ndjson = self.client.post("/api/ledger/export", json={"format": "jsonl", "filters": {"profiles": ["dev"]}})
        self.assertEqual(ndjson.headers["content-type"], "application/x-ndjson")
        rows = [json.loads(line) for line in ndjson.text.splitlines() if line]
        self.assertEqual(len(rows), 16)
        self.assertEqual(self.client.post("/api/ledger/export", json={"format": "xlsx"}).status_code, 400)

    def test_report_coverage_pairs_and_export(self) -> None:
        body = self.ok(self.client.post("/api/report", json={"filters": {}, "options": {}}))
        self.assertEqual(set(body), REPORT_KEYS)
        self.assertEqual(body["totals"]["trials"], 9)  # pilot 1 is excluded by the seeded exclusion
        coverage = self.ok(self.client.post("/api/report/coverage", json={}))
        self.assertEqual(set(coverage), COVERAGE_KEYS)
        key = body["comparisons"][0]["key"]
        pairs = self.ok(self.client.post("/api/report/pairs", json={"comparison": key}))
        self.assertEqual(set(pairs), {"comparison", "pairs", "excluded_trials"})
        missing = self.client.post("/api/report/pairs", json={"comparison": "fam:x|y|z~fam:x|y|w"})
        self.assertEqual(missing.status_code, 404)

        fingerprint = body["provenance"]["fingerprint"].split(":", 1)[1][:8]
        for level in ("trials", "pairs", "comparisons"):
            with self.subTest(level):
                exported = self.client.post("/api/report/export", json={"level": level})
                self.assertEqual(exported.status_code, 200, exported.text)
                self.assertIn(f'filename="bench-report-{fingerprint}-{level}.csv"',
                              exported.headers["content-disposition"])
                self.assertTrue(exported.content.startswith(b"\xef\xbb\xbf"))
                self.assertIn(b"\r\n", exported.content)
        self.assertEqual(self.client.post("/api/report/export", json={"level": "cells"}).status_code, 400)

    def test_snapshots_round_trip(self) -> None:
        saved = self.client.post("/api/report/snapshots", json={"name": "Pilots", "options": {"ci_level": 0.9}})
        self.assertEqual(saved.status_code, 201, saved.text)
        snap = saved.json()
        self.assertEqual(set(snap), SNAPSHOT_KEYS)
        self.assertEqual(snap["trials_used"], 9)
        listed = self.ok(self.client.get("/api/report/snapshots"))["snapshots"]
        self.assertEqual([s["id"] for s in listed], [snap["id"]])
        one = self.ok(self.client.get(f"/api/report/snapshots/{snap['id']}"))
        self.assertEqual(set(one), {"id", "name", "created_at", "request", "response"})
        self.assertEqual(one["response"]["provenance"]["fingerprint"], snap["fingerprint"])
        self.assertEqual(one["request"]["options"]["ci_level"], 0.9)
        self.assertEqual(self.client.get("/api/report/snapshots/rs-nope").status_code, 404)
        self.assertEqual(self.client.post("/api/report/snapshots", json={"name": ""}).status_code, 400)

    def test_bad_requests_are_400_and_unknown_ids_404(self) -> None:
        self.assertEqual(self.client.post("/api/report", json={"filters": {"colour": ["red"]}}).status_code, 400)
        self.assertEqual(self.client.post("/api/report", json={"options": {"ci_level": 0.5}}).status_code, 400)
        self.assertEqual(self.client.post("/api/report", json={"filter": {}}).status_code, 400)
        self.assertEqual(self.client.post("/api/ledger/trials", json={"limit": 5000}).status_code, 400)
        self.assertEqual(self.client.post("/api/ledger/trials", json={"order": "id; DROP"}).status_code, 400)
        self.assertEqual(self.client.post("/api/ledger/trials", json={"order": 5}).status_code, 400)
        self.assertEqual(self.client.get("/api/ledger/trials/999999").status_code, 404)
        self.assertEqual(self.client.get("/api/ledger/trials/abc").status_code, 404)

    def test_a_second_rescan_is_409(self) -> None:
        lock = self.state.ledger._rescan_lock
        self.assertTrue(lock.acquire(blocking=False))
        try:
            response = self.client.post("/api/ledger/rescan", json={"reparse": False})
        finally:
            lock.release()
        self.assertEqual(response.status_code, 409)
        self.assertIn("already running", response.json()["detail"])
        self.assertEqual(self.client.post("/api/ledger/rescan", json={"reparse": "yes"}).status_code, 400)

    def test_a_locked_database_is_503(self) -> None:
        def locked(*args: Any, **kwargs: Any) -> Any:
            raise sqlite3.OperationalError("database is locked")

        with mock.patch.object(self.state.ledger, "query", locked):
            response = self.client.post("/api/ledger/trials", json={})
            report_response = self.client.post("/api/report", json={"filters": {"tasks": ["path-tracing"]}})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": LEDGER_BUSY})
        self.assertEqual(report_response.status_code, 503)

    def test_state_has_a_ledger_block(self) -> None:
        block = self.ok(self.client.get("/api/state"))["ledger"]
        self.assertEqual(set(block), {"available", "error", "trials", "last_ingest_at", "rescan_running"})
        self.assertEqual((block["available"], block["error"], block["trials"], block["rescan_running"]),
                         (True, None, 16, False))
        self.assertIsNotNone(block["last_ingest_at"])


class UnavailableLedger(LedgerServerCase):
    def make_ledger(self) -> Ledger:
        return Ledger(self.tmpdir() / "never-opened.sqlite", roots=[], root=self.tmpdir())

    def test_reads_are_503_and_status_still_answers(self) -> None:
        status = self.ok(self.client.get("/api/ledger/status"))
        self.assertFalse(status["available"])
        for method, path in (("post", "/api/report"), ("post", "/api/ledger/trials"), ("get", "/api/ledger/facets"),
                             ("post", "/api/ledger/rescan"), ("get", "/api/report/snapshots")):
            with self.subTest(path):
                response = getattr(self.client, method)(path, **({"json": {}} if method == "post" else {}))
                self.assertEqual(response.status_code, 503, response.text)
        state = self.ok(self.client.get("/api/state"))
        self.assertFalse(state["ledger"]["available"])
        self.assertEqual(state["ledger"]["trials"], 0)


class RunResultsRegression(LedgerServerCase):
    """GET /api/runs/{id}/results is what the LIVE view polls: unchanged."""

    def make_manager(self) -> Any:
        shutil.copytree(lf.FIXTURE_DEV / "runs", self.data / "runs")
        shutil.copytree(lf.FIXTURE_DEV / "jobs", self.data / "jobs")
        manager = RunManager()
        self.addCleanup(procs.release_instance_lock, manager._lock_fd)
        return manager

    def test_results_match_results_run_results(self) -> None:
        for run_id in (lf.PILOT_1, lf.PILOT_2):
            with self.subTest(run_id):
                served = self.ok(self.client.get(f"/api/runs/{run_id}/results"))
                expected = results.run_results(self.state.cfg, self.manager.get(run_id))
                self.assertEqual(served, json.loads(json.dumps(expected)))
                self.assertEqual(set(served), {"run_id", "baseline", "arms", "comparison", "trials"})
                self.assertEqual(len(served["arms"]), 10)
                self.assertTrue(served["trials"], "the copied job folders were not read")


if __name__ == "__main__":
    import unittest
    unittest.main()
