"""Price lookups, the cost formula, and re-pricing at report time."""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from unittest import mock

from tests import ledger_fixtures as lf
from tests.support import BenchTestCase

from bench import extract, ledger, pricing, results, settings
from bench.settings import Price

TABLE = {
    "anthropic": {"claude-opus-5-5": Price(4.0, 20.0, 0.2), "claude-opus-5": Price(5.0, 25.0, 0.5)},
    "openrouter": {"glm-5.3": Price(0.07, 1.40, 0.069), "deepseek-v4.1-flash": Price(0.045, 0.50, 0.006)},
}


class Lookup(unittest.TestCase):
    def test_exact_and_with_provider_path(self) -> None:
        self.assertEqual(pricing.lookup(TABLE, "openrouter", "glm-5.3"), TABLE["openrouter"]["glm-5.3"])
        self.assertEqual(pricing.lookup(TABLE, "openrouter", "z-ai/glm-5.3"), TABLE["openrouter"]["glm-5.3"])

    def test_date_suffix_matches(self) -> None:
        self.assertEqual(pricing.lookup(TABLE, "anthropic", "claude-opus-5-5-20261001"),
                         TABLE["anthropic"]["claude-opus-5-5"])

    def test_never_a_prefix_match(self) -> None:
        self.assertIsNone(pricing.lookup(TABLE, "openrouter", "z-ai/glm-5.3-flash"))
        self.assertIsNone(pricing.lookup(TABLE, "anthropic", "claude-opus-5-5-fast"))
        # The longer id is its own model, not a variant of the shorter one.
        self.assertEqual(pricing.lookup(TABLE, "anthropic", "claude-opus-5-5"), TABLE["anthropic"]["claude-opus-5-5"])
        self.assertIsNone(pricing.lookup(TABLE, "fireworks", "glm-5.3"))


class Cost(unittest.TestCase):
    def test_unit_cost_matches_results_on_the_pi_deepseek_pilot(self) -> None:
        trial = next(p for p in (lf.FIXTURE_DEV / "jobs" / lf.PILOT_2 / "pi-deepseek").iterdir() if p.is_dir())
        usage = json.loads((trial / "result.json").read_text(encoding="utf-8"))["agent_result"]["model_usage"]
        (u,) = usage.values()
        price = TABLE["openrouter"]["deepseek-v4.1-flash"]
        args = (price, u["n_input_tokens"], u["n_cache_tokens"], u["n_output_tokens"])
        self.assertEqual(pricing.unit_cost(*args), results._cost(*args))
        self.assertAlmostEqual(pricing.unit_cost(*args), 0.0419, places=4)

    def test_cache_read_null_prices_cache_at_input(self) -> None:
        self.assertAlmostEqual(pricing.unit_cost(Price(1.0, 2.0, None), 1_000_000, 400_000, 0), 1.0)

    def test_unpriced_model_with_tokens_has_no_cost(self) -> None:
        usage = {"z-ai/glm-5.3": {"n_input": 100, "n_cache": 0, "n_output": 10},
                 "z-ai/glm-5.3-flash": {"n_input": 5, "n_cache": 0, "n_output": 1}}
        self.assertEqual(pricing.price_usage(usage, "openrouter", TABLE), (None, ["z-ai/glm-5.3-flash"]))

    def test_zero_token_entries_are_ignored(self) -> None:
        usage = {"z-ai/glm-5.3": {"n_input": 1_000_000, "n_cache": 0, "n_output": 0},
                 "<synthetic>": {"n_input": 0, "n_cache": 0, "n_output": 0}}
        cost, unpriced = pricing.price_usage(usage, "openrouter", TABLE)
        self.assertAlmostEqual(cost or 0, 0.07)
        self.assertEqual(unpriced, [])

    def test_nothing_reported_is_not_zero(self) -> None:
        self.assertEqual(pricing.price_usage({}, "openrouter", TABLE), (None, []))
        self.assertEqual(pricing.price_usage({"m": {"n_input": None, "n_output": None}}, "openrouter", TABLE),
                         (None, []))
        self.assertEqual(pricing.price_usage({"z-ai/glm-5.3": {"n_input": 0, "n_cache": 0, "n_output": 0}},
                                             "openrouter", TABLE), (0.0, []))


class TableFile(BenchTestCase):
    def _write(self, path: Path, glm_input: float) -> None:
        path.write_text(f"openrouter:\n  glm-5.3: {{input: {glm_input}, output: 1.40, cache_read: 0.069}}\n"
                        "  deepseek-v4.1-flash: {input: 0.045, output: 0.50, cache_read: 0.006}\n",
                        encoding="utf-8", newline="\n")
        # Coarse file clocks could leave the mtime unchanged between quick edits.
        stamp = time.time() + glm_input
        os.utime(path, (stamp, stamp))

    def test_load_table_follows_edits(self) -> None:
        path = self.tmpdir() / "pricing.yaml"
        self._write(path, 0.07)
        first, sha1 = pricing.load_table(path), pricing.table_sha(path)
        self._write(path, 0.70)
        second, sha2 = pricing.load_table(path), pricing.table_sha(path)
        self.assertEqual(first["openrouter"]["glm-5.3"].input, 0.07)
        self.assertEqual(second["openrouter"]["glm-5.3"].input, 0.70)
        self.assertNotEqual(sha1, sha2)
        self.assertEqual(len(sha1), 12)
        self.assertEqual(pricing.load_table(self.tmpdir() / "missing.yaml"), {})

    def test_editing_prices_changes_report_cost_not_tokens_or_cost_at_ingest(self) -> None:
        path = self.tmpdir() / "pricing.yaml"
        self._write(path, 0.07)
        patcher = mock.patch.object(settings, "PRICING", path)
        patcher.start()
        self.addCleanup(patcher.stop)
        tmp = self.tmpdir()
        trial = next(p for p in (lf.FIXTURE_DEV / "jobs" / lf.PILOT_2 / "pi-glm").iterdir() if p.is_dir())
        led = ledger.Ledger(tmp / "bench.sqlite", roots=lf.fixture_roots())
        led.open()
        self.addCleanup(led.close)
        run = json.loads((lf.FIXTURE_DEV / "runs" / lf.PILOT_2 / "run.json").read_text(encoding="utf-8"))
        ctx = extract.context_for_bench_arm(run, "pi-glm", "dev")
        led.ingest_job(trial.parent, ctx, final=True, via="rescan")
        (before,) = led.query(ledger.Filters())[1]
        self._write(path, 0.70)
        (after,) = led.query(ledger.Filters())[1]
        self.assertGreater(after["cost_usd"], before["cost_usd"])
        self.assertEqual(after["cost_at_ingest_usd"], before["cost_at_ingest_usd"])
        self.assertEqual(after["cost_at_ingest_usd"], before["cost_usd"])
        self.assertEqual((after["n_input"], after["n_output"], after["tokens_total"]),
                         (before["n_input"], before["n_output"], before["tokens_total"]))


if __name__ == "__main__":
    unittest.main()
