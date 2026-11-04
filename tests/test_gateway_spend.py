"""Correcting a gateway arm's cost from the gateway's spend log (bench.gateway_spend).

Synthetic trials only: one run with a Claude Code arm on GPT-6.1 Terra through
the gateway (no cached tokens recorded), a gateway arm whose caching was
reported (Kimi) and a direct Anthropic arm, all running at the same time. A
synthetic spend-log CSV is matched to the Terra trials by their agent
execution windows. Nothing here reads a real export or the real ledger.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

from tests import ledger_fixtures as lf
from tests.support import BenchTestCase

from bench import extract, gateway_spend, ledger, pricing, report, settings
from bench.extract import RunContext
from bench.ledger import Filters, Ledger

URL = "http://192.0.2.10:4000"
RUN = "20261104-200000-gateway"
RUN_KEY = f"default/{RUN}"
TERRA, KIMI, DIRECT = "claude-code-gw-terra", "claude-code-gw-kimi", "claude-code"
PRICING_YAML = """\
anthropic:
  claude-opus-5-5: {input: 4.00, output: 20.00, cache_read: 0.20, cache_write_5m: 5.00, cache_write_1h: 8.00}
litellm:
  gpt-6.1-terra:     {input: 1.00, output: 5.00, cache_read: 0.05, cache_write_5m: 1.25, cache_write_1h: 2.50}
  kimi-k3-fireworks: {input: 3.00, output: 15.00, cache_read: 0.30}
"""
T0 = lf._T0
PREFIXES = {"litellm_proxy/": "litellm", "anthropic/": "anthropic"}
FIELDS = list(gateway_spend.COLUMNS)


def _z(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def spend_row(start: float, prompt: int, cached: int, out: int, cost: float, rid: str, **extra: Any) -> dict:
    return {"start_utc": _z(at(start)), "end_utc": _z(at(start + 4)), "prompt_tokens": prompt,
            "cached_prompt_tokens": cached, "completion_tokens": out, "cost_usd": cost, "request_id": rid, **extra}


# make_trial puts the agent window at started_at + 60 s for exec_sec: t1 holds 60-360 s, t2 363-663 s
# (3 s after t1, so their 5 s skews overlap) and t3 2060-2360 s.
TERRA_TRIALS = {"t1": (0, (100_000, 0, 2_000)), "t2": (303, (60_000, 0, 1_000)), "t3": (2_000, (40_000, 0, 500))}
BASE_ROWS = [
    spend_row(57, 30_000, 28_000, 600, 0.0075, "r1"),     # 3 s before t1's window: inside the skew
    spend_row(120, 35_000, 33_000, 700, 0.0085, "r2"),
    spend_row(359, 35_000, 33_500, 700, 0.0083, "r3"),    # in t1, and in t2's skew: t1 holds it unpadded
    spend_row(400, 30_000, 28_200, 500, 0.0062, "r4"),
    spend_row(600, 30_000, 28_800, 500, 0.0051, "r5"),
    spend_row(5_000, 9_000, 0, 100, 0.0095, "r6"),        # after every trial: unmatched
    spend_row(120, 35_000, 33_000, 700, 0.0085, "r2"),    # the same call exported twice
]
T1_ROWS = BASE_ROWS[:3]
T2_ROWS = BASE_ROWS[3:5]


def write_csv(path: Path, rows: list[dict], *, bom: bool = False, fields: list[str] | None = None) -> Path:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields or FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(("﻿" if bom else "") + buf.getvalue(), encoding="utf-8", newline="")
    return path


class SpendCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tmp = self.tmpdir()
        self.db = self.tmp / "ledger" / "bench.sqlite"
        prices = self.tmp / "pricing.yaml"
        prices.write_text(PRICING_YAML, encoding="utf-8", newline="\n")
        patcher = mock.patch.object(settings, "PRICING", prices)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.led = Ledger(self.db, roots=[], root=self.tmp)
        self.led.open()
        self.addCleanup(self.led.close)
        self.assertTrue(self.led.available, self.led.error)
        jobs = self.tmp / "jobs" / RUN
        for name, (start, tokens) in TERRA_TRIALS.items():
            self.trial(jobs / TERRA, name, "gpt-6.1-terra", start, tokens)
        self.trial(jobs / KIMI, "k1", "kimi-k3-fireworks", 0, (80_000, 70_000, 1_500))
        self.trial(jobs / DIRECT, "d1", "claude-opus-5-5", 0, (90_000, 75_000, 1_800), provider="anthropic")

    def trial(self, job: Path, name: str, model: str, start: float, tokens: tuple[int, int, int],
              provider: str = "litellm") -> None:
        prefix = "litellm_proxy/" if provider == "litellm" else "anthropic/"
        kwargs: dict[str, Any] = {"reasoning_effort": "medium", "max_turns": 300}
        if provider == "litellm":
            kwargs["base_url"] = URL
        lf.make_job(job)
        lf.make_trial(job / name, family="claude-code", model_name=prefix + model, kwargs=kwargs, tokens=tokens,
                      cost=0.5, started_at=at(start), task=f"task-{name}", trial_name=name,
                      job_id=f"job-{job.name}")
        ctx = RunContext(run_key=RUN_KEY, profile="default", run_id=RUN, source="bench", arm=job.name,
                         provider=provider, model_prefix=prefix, effort_label="medium",
                         dataset="terminal-bench@2.0", local_snapshot=None, family="claude-code",
                         base_url="${LITELLM_URL}" if provider == "litellm" else None)
        with mock.patch.object(self.led, "_prefixes", return_value=PREFIXES):
            self.led.ingest_job(job, ctx, final=True, via="rescan")

    def by_name(self) -> dict[str, dict]:
        return {r["trial_name"]: r for r in self.led.query(Filters(include_cautioned=True))[1]}

    def corrections(self) -> list[sqlite3.Row]:
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute("SELECT * FROM cost_corrections ORDER BY id").fetchall()
        finally:
            conn.close()

    def csv(self, rows: list[dict] | None = None, name: str = "spend.csv", **kw: Any) -> Path:
        return write_csv(self.tmp / name, BASE_ROWS if rows is None else rows, **kw)

    def run_cli(self, *args: str) -> tuple[int, str]:
        with mock.patch.object(settings, "LEDGER_DB", self.db), mock.patch.object(
                Ledger, "_prefixes", return_value=PREFIXES), redirect_stdout(io.StringIO()) as out:
            code = ledger.main(["gateway-spend", *args])
        return code, out.getvalue()

    def import_csv(self, path: Path | None = None, *extra: str) -> tuple[int, str]:
        return self.run_cli(str(path or self.csv()), "--run", RUN, "--arm", TERRA, *extra)


class ReadingTests(SpendCase):
    def test_rows_parse_with_a_bom_and_a_repeated_request_id_is_read_once(self) -> None:
        log = gateway_spend.read_log(self.csv(bom=True))
        self.assertEqual((len(log.rows), log.duplicates), (6, 1))
        first = log.rows[0]
        self.assertEqual((first.line, first.start, first.prompt, first.cached, first.completion, first.cost),
                         (2, at(57), 30_000, 28_000, 600, 0.0075))
        self.assertEqual(log.name, "spend.csv")
        self.assertEqual(len(log.sha), 12)

    def test_a_time_without_a_zone_is_utc_and_an_offset_is_converted(self) -> None:
        rows = [dict(BASE_ROWS[0], start_utc="2026-10-29 05:31:00"),
                dict(BASE_ROWS[1], start_utc="2026-10-29T07:31:00+02:00")]
        log = gateway_spend.read_log(self.csv(rows))
        self.assertEqual([r.start for r in log.rows], [datetime(2026, 10, 29, 5, 31, tzinfo=timezone.utc)] * 2)

    def test_a_missing_column_or_a_bad_value_names_the_problem(self) -> None:
        with self.assertRaisesRegex(ValueError, "no cost_usd column"):
            gateway_spend.read_log(self.csv([{k: v for k, v in BASE_ROWS[0].items() if k != "cost_usd"}],
                                            fields=[f for f in FIELDS if f != "cost_usd"]))
        with self.assertRaisesRegex(ValueError, "line 3: cached_prompt_tokens .* exceeds prompt_tokens"):
            gateway_spend.read_log(self.csv([BASE_ROWS[0], dict(BASE_ROWS[1], cached_prompt_tokens=99_999)]))
        with self.assertRaisesRegex(ValueError, "line 2: start_utc is not a time"):
            gateway_spend.read_log(self.csv([dict(BASE_ROWS[0], start_utc="yesterday")]))


class MatchingTests(SpendCase):
    def trials(self) -> list[dict]:
        return self.led.arm_trials(RUN_KEY, TERRA)

    def test_rows_go_to_the_trial_whose_agent_window_holds_their_start(self) -> None:
        m = gateway_spend.match(gateway_spend.read_log(self.csv()).rows, self.trials())
        names = {t["trial_key"]: t["trial_name"] for t in self.trials()}
        got = {names[k]: [r.request_id for r in rows] for k, rows in m.by_trial.items()}
        # r1 is 3 s early (the skew); r3 is in t1 and in t2's skew, and t1 holds it without it.
        self.assertEqual(got, {"t1": ["r1", "r2", "r3"], "t2": ["r4", "r5"]})
        self.assertEqual([r.request_id for r in m.unmatched], ["r6"])
        self.assertEqual(m.ambiguous, [])

    def test_without_skew_an_early_row_matches_nothing(self) -> None:
        m = gateway_spend.match(gateway_spend.read_log(self.csv()).rows, self.trials(), skew_sec=0)
        self.assertEqual(sorted(r.request_id for r in m.unmatched), ["r1", "r6"])

    def test_a_row_between_two_windows_within_both_skews_is_ambiguous(self) -> None:
        rows = gateway_spend.read_log(self.csv([spend_row(361.5, 1000, 900, 10, 0.001, "gap")])).rows
        m = gateway_spend.match(rows, self.trials())
        self.assertEqual([(r.request_id, sorted(names)) for r, names in m.ambiguous], [("gap", ["t1", "t2"])])
        self.assertEqual(m.by_trial, {})

    def test_a_model_column_keeps_only_the_arms_rows(self) -> None:
        rows = [dict(BASE_ROWS[0], model="gpt-6.1-terra"), dict(BASE_ROWS[1], model="openai/gpt-6.1-terra"),
                dict(BASE_ROWS[2], model="kimi-k3-fireworks")]
        log = gateway_spend.read_log(self.csv(rows, fields=[*FIELDS, "model"]))
        m = gateway_spend.match(log.rows, self.trials(), model="gpt-6.1-terra")
        self.assertEqual([r.request_id for r in m.other_models], ["r3"])
        self.assertEqual(sum(len(v) for v in m.by_trial.values()), 2)


class DryRunTests(SpendCase):
    def test_the_dry_run_prints_the_matching_and_writes_nothing(self) -> None:
        before = self.led.version()
        code, out = self.import_csv(None, "--dry-run")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.corrections(), [])
        self.assertEqual(self.led.version(), before)
        self.assertIn("Dry run: nothing written (2 correction(s) would be).", out)
        self.assertIn("Matched 5 rows", out)
        self.assertIn("Unmatched 1, ambiguous 0", out)
        self.assertIn("unmatched line 7", out)
        self.assertIn("repeat a request id", out)
        lines = {line.split()[0]: line for line in out.splitlines() if line.startswith(("t1 ", "t2 ", "t3 "))}
        # t1: before is 100k fresh input + 2k output at Terra's list price; after is the gateway's logged cost.
        self.assertIn("$0.1100", lines["t1"])
        self.assertIn("$0.0243", lines["t1"])
        self.assertTrue(lines["t1"].rstrip().endswith("new"))
        self.assertTrue(lines["t3"].rstrip().endswith("no rows: upper bound"))

    def test_the_plan_gives_totals_before_and_after(self) -> None:
        s = gateway_spend.plan(self.led, self.csv(), RUN, TERRA)
        self.assertEqual(s["run_key"], RUN_KEY)
        self.assertEqual((s["file"]["rows"], s["file"]["duplicates"]), (6, 1))
        self.assertEqual(s["matched"], gateway_spend.totals(gateway_spend.read_log(self.csv()).rows[:5]))
        t1 = next(t for t in s["trials"] if t["trial_name"] == "t1")
        self.assertEqual((t1["after"]["n_input"], t1["after"]["n_cache"], t1["after"]["n_output"]),
                         (100_000, 94_500, 2_000))
        self.assertAlmostEqual(t1["after_usd"], 0.0243)
        # The corrected split at list price: 5,500 fresh + 94,500 cached + 2,000 output.
        self.assertAlmostEqual(t1["list_price_usd"], (5_500 * 1.0 + 94_500 * 0.05 + 2_000 * 5.0) / 1e6)
        self.assertAlmostEqual(s["before_usd"], 0.110 + 0.065 + 0.0425)
        self.assertAlmostEqual(s["after_usd"], 0.0243 + 0.0113 + 0.0425)

    def test_the_skew_is_an_option(self) -> None:
        code, out = self.import_csv(None, "--dry-run", "--skew", "0")
        self.assertEqual(code, 0, out)
        self.assertIn("Unmatched 2", out)

    def test_an_arm_that_did_not_run_through_a_gateway_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "did not run through a gateway"):
            gateway_spend.plan(self.led, self.csv(), RUN, DIRECT)
        code, out = self.run_cli(str(self.csv()), "--run", RUN, "--arm", "no-such-arm", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("No trials of arm 'no-such-arm'", out)
        code, out = self.run_cli(str(self.csv()), "--run", "no-such-run", "--arm", TERRA)
        self.assertEqual((code, "No run 'no-such-run'" in out), (1, True))


class ImportTests(SpendCase):
    def test_import_corrects_the_matched_trials_and_flags_the_rest_of_the_arm(self) -> None:
        untouched = {n: self.by_name()[n] for n in ("k1", "d1")}
        code, out = self.import_csv()
        self.assertEqual(code, 0, out)
        self.assertIn("Wrote 2 correction(s)", out)
        rows = self.by_name()
        t1 = rows["t1"]
        self.assertEqual((t1["n_input"], t1["n_cache"], t1["n_output"]), (100_000, 94_500, 2_000))
        self.assertAlmostEqual(t1["cost_usd"], 0.0243)
        self.assertAlmostEqual(t1["cost_at_ingest_usd"], 0.0243)
        self.assertEqual(t1["tokens_uncached"], 5_500 + 2_000)
        flag = t1["flags"][gateway_spend.CORRECTED_FLAG]
        self.assertEqual((flag["calls"], flag["file"], flag["recorded"]["n_cache"]), (3, "spend.csv", 0))
        self.assertAlmostEqual(flag["recorded"]["cost_usd"], 0.110)
        note = json.loads(self.corrections()[0]["note_json"])
        self.assertEqual((note["file"], note["rows"], note["file_rows"], note["file_totals"]["calls"]),
                         ("spend.csv", 3, 6, 6))
        # t3 had no rows: its cost stays as computed, flagged an upper bound.
        self.assertIn(gateway_spend.UPPER_BOUND_FLAG, rows["t3"]["flags"])
        self.assertAlmostEqual(rows["t3"]["cost_usd"], 0.0425)
        # Other arms, on the gateway and off it, are exactly as before.
        for name, before in untouched.items():
            self.assertEqual({k: v for k, v in rows[name].items() if k != "updated_at"},
                             {k: v for k, v in before.items() if k != "updated_at"})
        # The stored facts are the harness's, untouched; Harbor's files too.
        conn = sqlite3.connect(self.db)
        try:
            self.assertEqual(conn.execute("SELECT n_cache FROM trials WHERE trial_name = 't1'").fetchone()[0], 0)
        finally:
            conn.close()
        result = json.loads((self.tmp / "jobs" / RUN / TERRA / "t1" / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["agent_result"]["n_cache_tokens"], 0)

    def test_the_same_file_twice_does_not_double_count(self) -> None:
        self.assertEqual(self.import_csv()[0], 0)
        version, first, spend = self.led.version(), self.by_name()["t1"], self.led.runs()[0]["spend_usd"]
        for again in (self.csv(), self.csv(name="spend-again.csv")):  # the same file, then its rows renamed
            code, out = self.import_csv(again)
            self.assertEqual(code, 0, out)
            self.assertIn("Wrote 0 correction(s); 2 already in force", out)
        self.assertEqual(len(self.corrections()), 2)
        self.assertEqual(self.led.version(), version)
        self.assertEqual(self.by_name()["t1"]["cost_usd"], first["cost_usd"])
        self.assertEqual(self.led.runs()[0]["spend_usd"], spend)

    def test_a_new_export_replaces_a_correction_rather_than_adding_to_it(self) -> None:
        self.assertEqual(self.import_csv()[0], 0)
        more = [*BASE_ROWS, spend_row(200, 10_000, 9_000, 100, 0.0021, "r7")]
        code, out = self.import_csv(self.csv(more, name="spend-2.csv"))
        self.assertEqual(code, 0, out)
        self.assertIn("Wrote 1 correction(s); 1 already in force", out)
        t1 = self.by_name()["t1"]
        self.assertEqual((t1["n_input"], t1["flags"][gateway_spend.CORRECTED_FLAG]["calls"]), (110_000, 4))
        self.assertAlmostEqual(t1["cost_usd"], 0.0243 + 0.0021)

    def test_ambiguous_rows_stop_the_import(self) -> None:
        code, out = self.import_csv(self.csv([*BASE_ROWS, spend_row(361.5, 1000, 900, 10, 0.001, "gap")]))
        self.assertEqual(code, 1)
        self.assertIn("ambiguous line 9", out)   # line 8 is the repeated r2
        self.assertIn("Nothing written", out)
        self.assertEqual(self.corrections(), [])

    def test_corrections_survive_a_rederive_and_a_reparse(self) -> None:
        self.assertEqual(self.import_csv()[0], 0)
        # A parser bump re-derives every row, as does a reparse (rescan --reparse ends in rederive(force=True)).
        with mock.patch.object(extract, "PARSER_VERSION", extract.PARSER_VERSION + 1), \
                mock.patch.object(self.led, "_prefixes", return_value=PREFIXES):
            self.assertEqual(self.led.rederive(), 5)
        with mock.patch.object(self.led, "_prefixes", return_value=PREFIXES):
            self.assertEqual(self.led.rederive(force=True), 5)
        self.led.apply_lookup_verdicts({})
        t1 = self.by_name()["t1"]
        self.assertEqual((t1["n_cache"], len(self.corrections())), (94_500, 2))
        self.assertAlmostEqual(t1["cost_usd"], 0.0243)

    def test_clear_takes_the_corrections_back(self) -> None:
        self.assertEqual(self.import_csv()[0], 0)
        code, out = self.run_cli("--run", RUN, "--arm", TERRA, "--clear")
        self.assertEqual(code, 0, out)
        self.assertIn("Wrote 2 correction(s)", out)
        rows = self.by_name()
        self.assertEqual(rows["t1"]["n_cache"], 0)
        self.assertAlmostEqual(rows["t1"]["cost_usd"], 0.110)
        # No correction in force for the arm any more, so nothing marks it.
        self.assertFalse(any(f in r["flags"] for r in rows.values()
                             for f in (gateway_spend.CORRECTED_FLAG, gateway_spend.UPPER_BOUND_FLAG)))
        code, out = self.run_cli("--run", RUN, "--arm", TERRA, "--clear")
        self.assertIn("no corrections in force", out)

    def test_status_counts_the_corrections_in_force(self) -> None:
        self.assertEqual(self.led.status()["cost_corrections"], 0)
        self.assertEqual(self.import_csv()[0], 0)
        self.assertEqual(self.led.status()["cost_corrections"], 2)


class SpendFigureTests(SpendCase):
    def test_run_spend_uses_the_logged_cost_and_counts_both_marks(self) -> None:
        before = self.led.runs()[0]
        self.assertEqual(self.import_csv()[0], 0)
        after = self.led.runs()[0]
        self.assertAlmostEqual(before["spend_usd"] - after["spend_usd"], (0.110 + 0.065) - (0.0243 + 0.0113))
        self.assertEqual((before["trials_cost_corrected"], before["trials_cost_upper_bound"]), (0, 0))
        self.assertEqual((after["trials_cost_corrected"], after["trials_cost_upper_bound"]), (2, 1))

    def test_the_report_takes_the_logged_cost_under_every_pricing(self) -> None:
        self.assertEqual(self.import_csv()[0], 0)
        raw = self.led.query(Filters(include_cautioned=True))[1]
        table = pricing.load_table()
        for mode in ("current", "as_ingested", "reported"):
            o = report.parse_options({"pricing": mode}, None)
            rows = {r["trial_name"]: r for r in report.prepare_rows(raw, o, None, table)}
            with self.subTest(pricing=mode):
                self.assertAlmostEqual(rows["t1"]["cost"], 0.0243)
                self.assertAlmostEqual(rows["t2"]["cost"], 0.0113)
                self.assertAlmostEqual(rows["t3"]["cost"], 0.0425)
                self.assertEqual(rows["t1"]["tokens_uncached"], 7_500)
        o = report.parse_options(None, None)
        rows = report.prepare_rows(raw, o, None, table)
        terra = [r for r in rows if r["arm"] == TERRA]
        caveats = report.flag_caveats([({"name": "Claude Code on Terra via the gateway"}, terra),
                                       ({"name": "Claude Code"}, [r for r in rows if r["arm"] == DIRECT])])
        gateway = [c for c in caveats if "spend-log" in c or "spend log" in c]
        self.assertEqual(len(gateway), 2, caveats)
        self.assertIn("2 Claude Code on Terra via the gateway trials take tokens, cache reads and cost from the "
                      "gateway's spend log", gateway[0])
        self.assertIn("(cost from gateway spend log)", gateway[0])
        self.assertIn("1 Claude Code on Terra via the gateway trial has no spend-log correction", gateway[1])
        self.assertIn("upper bound", gateway[1])
        # The checks on the harness's own figures no longer apply to the two corrected trials; t3 keeps its.
        unknown = [c for c in caveats if c.startswith("1 Claude Code on Terra") and "cache writes" in c]
        self.assertEqual(len(unknown), 1, caveats)
        t1 = next(r for r in terra if r["trial_name"] == "t1")
        self.assertEqual(t1["flags"][gateway_spend.CORRECTED_FLAG]["superseded"], ["cache_writes_unknown"])
        self.assertNotIn("cache_writes_unknown", t1["flags"])

    def test_the_report_cache_key_moves_on_import(self) -> None:
        before = self.led.version()
        self.assertEqual(self.import_csv()[0], 0)
        self.assertNotEqual(self.led.version(), before)

    def test_a_trial_drawer_shows_the_correction(self) -> None:
        self.assertEqual(self.import_csv()[0], 0)
        t1 = self.led.trial(self.by_name()["t1"]["id"])
        self.assertAlmostEqual(t1["cost_usd"], 0.0243)
        self.assertIn(gateway_spend.CORRECTED_FLAG, t1["flags"])
