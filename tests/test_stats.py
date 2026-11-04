"""bench.stats: quantiles, intervals, sample sizes and schedules, checked
against published tables and hand computation. Pure: no ledger, no files."""

from __future__ import annotations

import math
import os
import random
import statistics
import subprocess
import sys
import unittest

from tests.support import BenchTestCase

from bench import settings, stats

# Two-sided t quantiles from any textbook table.
T95 = {1: 12.706, 2: 4.303, 4: 2.776, 9: 2.262, 30: 2.042, 120: 1.980}
# The run-design spec's table of two-sided 90% quantiles (its "T95").
T90 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
       11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729,
       20: 1.725, 21: 1.721, 22: 1.717, 23: 1.714, 24: 1.711, 25: 1.708, 26: 1.706, 27: 1.703, 28: 1.701,
       29: 1.699, 30: 1.697}


class Distributions(BenchTestCase):
    def test_t_quantile_matches_the_95_table(self) -> None:
        for df, expected in T95.items():
            with self.subTest(df=df):
                self.assertAlmostEqual(stats.t_quantile(0.95, df), expected, delta=1e-3)
        self.assertAlmostEqual(stats.t_quantile(0.90, 9), 1.833, delta=1e-3)

    def test_t_crit_matches_the_planner_table_at_90(self) -> None:
        for df in (1, 4, 9, 30):
            with self.subTest(df=df):
                self.assertAlmostEqual(stats.t_crit(0.90, df), T90[df], delta=1e-3)
        for df, expected in T90.items():
            self.assertAlmostEqual(stats.t_quantile(0.90, df), expected, delta=1.5e-3, msg=f"df {df}")
        self.assertEqual(stats.t_crit(0.90, 12), stats.t_quantile(0.90, 12))

    def test_t_cdf(self) -> None:
        self.assertEqual(stats.t_cdf(0.0, 5), 0.5)
        for x, df in ((0.7, 3), (2.5, 9), (1.1, 40)):
            self.assertAlmostEqual(stats.t_cdf(x, df) + stats.t_cdf(-x, df), 1.0, places=12)
        self.assertAlmostEqual(stats.t_cdf(1.959964, 1e7), 0.975, places=5)
        self.assertAlmostEqual(stats.t_cdf(1.0, 1), 0.75, places=12)  # Cauchy: 1/2 + atan(1)/pi
        self.assertEqual(stats.t_cdf(math.inf, 3), 1.0)
        self.assertEqual(stats.t_cdf(-math.inf, 3), 0.0)

    def test_z_quantile(self) -> None:
        self.assertAlmostEqual(stats.z_quantile(0.95), 1.959964, places=5)
        self.assertAlmostEqual(stats.z_quantile(0.90), 1.644854, places=5)

    def test_bad_arguments(self) -> None:
        for call in (lambda: stats.t_quantile(1.0, 5), lambda: stats.t_quantile(0.0, 5),
                     lambda: stats.t_quantile(0.9, 0), lambda: stats.z_quantile(True),
                     lambda: stats.t_cdf(1.0, 0)):
            with self.assertRaises(ValueError):
                call()


class Intervals(BenchTestCase):
    def test_wilson_known_values(self) -> None:
        for (k, n), (lo, hi) in {(0, 10): (0.0, 0.2775), (5, 10): (0.2366, 0.7634),
                                 (10, 10): (0.7225, 1.0)}.items():
            got = stats.wilson(k, n, 0.95)
            self.assertAlmostEqual(got[0], lo, places=4)
            self.assertAlmostEqual(got[1], hi, places=4)
        self.assertEqual(stats.wilson(0, 10, 0.95)[0], 0.0)
        self.assertEqual(stats.wilson(10, 10, 0.95)[1], 1.0)
        self.assertIsNone(stats.wilson(0, 0, 0.9))
        lo, hi = stats.wilson(2.5, 5, 0.9)  # task-weighted successes may be fractional
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_mcnemar_exact(self) -> None:
        self.assertEqual(stats.mcnemar_exact(0, 3), 0.25)
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(stats.mcnemar_exact(2, 1), 1.0)
        self.assertEqual(stats.mcnemar_exact(1, 7), stats.mcnemar_exact(7, 1))
        self.assertAlmostEqual(stats.mcnemar_exact(1, 7), 2 * 9 / 256)

    def test_newcombe_paired(self) -> None:
        cases = [(4, 2, 1, 3), (10, 0, 0, 10), (0, 5, 0, 5), (3, 0, 4, 1), (20, 6, 2, 12), (5, 0, 0, 5),
                 (10, 0, 0, 0), (0, 0, 0, 7)]
        for a, b, c, d in cases:
            with self.subTest(table=(a, b, c, d)):
                lo, hi = stats.newcombe_paired(a, b, c, d, 0.95)
                n = a + b + c + d
                self.assertGreaterEqual(lo, -1.0)
                self.assertLessEqual(hi, 1.0)
                self.assertLessEqual(lo, (b - c) / n)
                self.assertGreaterEqual(hi, (b - c) / n)
                mlo, mhi = stats.newcombe_paired(a, c, b, d, 0.95)
                self.assertAlmostEqual(mlo, -hi, places=12)  # swapping b and c mirrors it
                self.assertAlmostEqual(mhi, -lo, places=12)
        lo, hi = stats.newcombe_paired(5, 0, 0, 5, 0.9)  # no discordant tasks, yet not a point
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)
        small = stats.newcombe_paired(4, 2, 1, 3, 0.9)
        large = stats.newcombe_paired(40, 20, 10, 30, 0.9)
        self.assertLess(large[1] - large[0], small[1] - small[0])
        with self.assertRaises(ValueError):
            stats.newcombe_paired(0, 0, 0, 0, 0.9)

    def test_geometric_mean_is_symmetric(self) -> None:
        ratios = [2.0, 8.0]
        forward = stats.log_ratio_summary([math.log(r) for r in ratios], 0.9)
        self.assertAlmostEqual(forward["estimate"], 4.0)
        backward = stats.log_ratio_summary([-math.log(r) for r in ratios], 0.9)
        self.assertAlmostEqual(backward["estimate"], 1 / forward["estimate"])
        self.assertAlmostEqual(backward["lo"], 1 / forward["hi"])
        self.assertAlmostEqual(backward["hi"], 1 / forward["lo"])
        one = stats.log_ratio_summary([math.log(3.0)], 0.9)
        self.assertEqual((one["lo"], one["hi"], one["sd_log"], one["n"]), (None, None, None, 1))
        with self.assertRaises(ValueError):
            stats.log_ratio_summary([], 0.9)

    def test_log_ratio_interval_by_hand(self) -> None:
        logs = [0.2, 0.5, 0.9, 0.4, 0.7]
        got = stats.log_ratio_summary(logs, 0.95)
        mean, sd = statistics.fmean(logs), statistics.stdev(logs)
        half = 2.776 * sd / math.sqrt(5)
        self.assertAlmostEqual(got["lo"], math.exp(mean - half), places=3)
        self.assertAlmostEqual(got["hi"], math.exp(mean + half), places=3)
        self.assertAlmostEqual(got["gsd"], math.exp(sd))

    def test_bootstrap_is_seeded(self) -> None:
        rng = random.Random(3)
        values = [rng.lognormvariate(0, 1) for _ in range(12)]
        first = stats.bootstrap_ci(values, statistics.fmean, resamples=500, seed=7, level=0.9)
        self.assertEqual(first, stats.bootstrap_ci(values, statistics.fmean, resamples=500, seed=7, level=0.9))
        self.assertNotEqual(first, stats.bootstrap_ci(values, statistics.fmean, resamples=500, seed=8, level=0.9))
        self.assertLess(first[0], statistics.fmean(values))
        self.assertGreater(first[1], statistics.fmean(values))
        self.assertIsNone(stats.bootstrap_ci(values[:4], statistics.fmean, resamples=500, seed=1, level=0.9))

    def test_stable_seed_is_the_same_in_another_process(self) -> None:
        here = stats.stable_seed(20261028, "fam:a~fam:b", "cost")
        code = ("import sys; sys.path.insert(0, sys.argv[1]); from bench.stats import stable_seed; "
                "print(stable_seed(20261028, 'fam:a~fam:b', 'cost'))")
        env = dict(os.environ, PYTHONHASHSEED="12345")
        out = subprocess.run([sys.executable, "-c", code, str(settings.ROOT)], capture_output=True, text=True,
                             encoding="utf-8", env=env, timeout=60, check=True)
        self.assertEqual(int(out.stdout.strip()), here)
        self.assertNotEqual(stats.stable_seed(1, "x"), stats.stable_seed(2, "x"))

    def test_leave_one_out(self) -> None:
        self.assertEqual(stats.leave_one_out([1.0, 2.0, 6.0], statistics.fmean), (1.5, 4.0))
        self.assertIsNone(stats.leave_one_out([1.0], statistics.fmean))


class SampleSizes(BenchTestCase):
    def test_required_pairs(self) -> None:
        self.assertEqual(stats.required_pairs(0.6, 0.25), 22)
        self.assertEqual(stats.required_pairs(0.4, 0.25), 11)
        self.assertEqual(stats.required_pairs(0.6, 0.25, 0.90), stats.n_needed_ratio(0.6, 1.25, 0.90))

    def test_n_needed_ratio_is_the_smallest_n_that_meets_the_target(self) -> None:
        for sd in (0.2, 0.5, 0.9, 1.4):
            n = stats.n_needed_ratio(sd, 1.3, 0.9)
            target = math.log(1.3)
            self.assertLessEqual(stats.t_quantile(0.9, n - 1) * sd / math.sqrt(n), target)
            if n > 2:
                self.assertGreater(stats.t_quantile(0.9, n - 2) * sd / math.sqrt(n - 1), target)

    def test_n_needed_ratio_is_monotone(self) -> None:
        by_sd = [stats.n_needed_ratio(sd / 10, 1.25, 0.9) for sd in range(1, 15)]
        self.assertEqual(by_sd, sorted(by_sd))
        by_target = [stats.n_needed_ratio(0.6, 1 + h / 100, 0.9) for h in range(10, 60, 5)]
        self.assertEqual(by_target, sorted(by_target, reverse=True))
        self.assertEqual(stats.n_needed_ratio(0.0, 1.25, 0.9), 2)
        self.assertEqual(stats.n_needed_ratio(5.0, 1.01, 0.99, n_max=50), 50)
        with self.assertRaises(ValueError):
            stats.n_needed_ratio(0.6, 1.0, 0.9)

    def test_pass_rate_sizes(self) -> None:
        self.assertEqual(stats.n_needed_pass(0.5, 5, 0.95), 385)  # about 384: why 20 tasks cannot see 5 points
        self.assertEqual(stats.n_needed_pass(0.5, 5, 0.90), 271)
        self.assertAlmostEqual(stats.mde_pp(0.5, 10, 0.95), 100 * 1.959964 * 0.5 / math.sqrt(10), places=4)
        with self.assertRaises(ValueError):
            stats.mde_pp(0.5, 0, 0.9)

    def test_quartiles(self) -> None:
        self.assertEqual(stats.quartiles([4.0, 1.0, 3.0, 2.0, 5.0]),
                         {"min": 1.0, "q1": 2.0, "median": 3.0, "q3": 4.0, "max": 5.0})
        self.assertEqual(stats.quartiles([2.0])["q3"], 2.0)
        self.assertIsNone(stats.quartiles([]))

    def test_pooled_sd(self) -> None:
        self.assertAlmostEqual(stats.pooled_sd([(0.4, 5), (0.8, 3)]), math.sqrt((4 * 0.16 + 2 * 0.64) / 6))
        self.assertAlmostEqual(stats.pooled_sd([(0.5, 4), (3.0, 2), (None, 9)]), 0.5)  # n < 3 is too noisy
        self.assertIsNone(stats.pooled_sd([(0.5, 2)]))
        self.assertIsNone(stats.pooled_sd([]))


class Verdicts(BenchTestCase):
    def test_band_verdict_outcomes(self) -> None:
        consensus = {"low": 2.0, "high": 3.0}
        self.assertEqual(stats.band_verdict((2.2, 2.8), consensus), "in line")
        self.assertEqual(stats.band_verdict((1.1, 1.8), consensus), "lower")
        self.assertEqual(stats.band_verdict((3.2, 5.0), consensus), "higher")
        self.assertEqual(stats.band_verdict((1.5, 2.5), consensus), "undecided")
        pi_tokens = {"value": 3.1}  # band [2.07, 4.65]
        self.assertEqual(stats.band_verdict((2.1, 4.6), pi_tokens), "in line")
        self.assertEqual(stats.band_verdict((2.0, 4.6), pi_tokens), "undecided")
        self.assertEqual(stats.band_verdict((2.0, 4.6), {"value": 3.1, "tolerance": 2.0}), "in line")
        self.assertIsNone(stats.band_verdict(None, consensus))
        self.assertIsNone(stats.band_verdict((1.0, 2.0), None))
        self.assertIsNone(stats.band_verdict((1.0, 2.0), {"note": "no figure"}))
        band = stats.reference_band({"value": 3.1})
        self.assertAlmostEqual(band[0], 2.0667, places=4)
        self.assertAlmostEqual(band[1], 4.65, places=4)


class PublishedReference(BenchTestCase):
    """reference.yaml read as harness / baseline, whichever way it was written."""

    OLD = {"defaults": {"direction": "baseline_over_harness", "tolerance": 1.5},
           "consensus": {"cost_per_trial": {"low": 2.0, "high": 3.0, "source": "S1"},
                         "pass_rate_delta_pp": {"low": -5, "high": 5}},
           "arms": {"pi": {"tokens_per_trial": {"value": 3.1}, "note": "n"},
                    "strands": {"cost_per_trial": {"value": 2.1, "tolerance": 2.0}}, "terminus-2": {"note": "t"}}}
    NEW = {"defaults": {"direction": "harness_over_baseline", "tolerance": 1.5},
           "consensus": {"cost_per_trial": {"low": "1/3", "high": "1/2", "source": "S1"},
                         "pass_rate_delta_pp": {"low": -5, "high": 5}},
           "arms": {"pi": {"tokens_per_trial": {"value": "1/3.1"}, "note": "n"},
                    "strands": {"cost_per_trial": {"value": "1/2.1", "tolerance": 2.0}}, "terminus-2": {"note": "t"}}}

    def test_both_directions_read_the_same(self) -> None:
        old, new = stats.published_reference(self.OLD), stats.published_reference(self.NEW)
        self.assertEqual(old, new)
        self.assertEqual(new["defaults"]["direction"], stats.DIRECTION)
        self.assertEqual(new["consensus"]["cost_per_trial"], {"low": 1 / 3, "high": 0.5, "source": "S1"})
        self.assertEqual(new["consensus"]["pass_rate_delta_pp"], {"low": -5, "high": 5})  # points, never inverted
        self.assertEqual(new["arms"]["pi"]["tokens_per_trial"]["value"], 1 / 3.1)
        self.assertEqual(new["arms"]["strands"]["cost_per_trial"], {"value": 1 / 2.1, "tolerance": 2.0})
        self.assertEqual(new["arms"]["terminus-2"], {"note": "t"})
        self.assertEqual(self.OLD["consensus"]["cost_per_trial"]["low"], 2.0)  # the input is left alone

    def test_a_file_without_a_direction_is_read_the_old_way(self) -> None:
        # Before defaults.direction existed every figure was baseline / harness.
        undirected = {k: v for k, v in self.OLD.items() if k != "defaults"}
        self.assertEqual(stats.reference_direction(undirected), stats.LEGACY_DIRECTION)
        self.assertEqual(stats.published_reference(undirected)["consensus"],
                         stats.published_reference(self.NEW)["consensus"])
        self.assertEqual(stats.published_reference(undirected)["arms"], stats.published_reference(self.NEW)["arms"])
        no_key = {**self.OLD, "defaults": {"tolerance": 1.5}}
        self.assertEqual(stats.published_reference(no_key)["arms"]["strands"]["cost_per_trial"]["value"], 1 / 2.1)

    def test_an_unknown_direction_is_refused(self) -> None:
        for bad in ("baseline/harness", "harness_over_baseline ", 1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                stats.published_reference({**self.NEW, "defaults": {"direction": bad}})

    def test_a_single_value_band_inverts_with_its_tolerance(self) -> None:
        strands = stats.published_reference(self.OLD)["arms"]["strands"]["cost_per_trial"]
        old_band = stats.reference_band(self.OLD["arms"]["strands"]["cost_per_trial"], 1.5)
        band = stats.reference_band(strands, 1.5)
        self.assertAlmostEqual(band[0], 1 / old_band[1])
        self.assertAlmostEqual(band[1], 1 / old_band[0])
        self.assertAlmostEqual(band[0], 1 / (2.1 * 1.5))

    def test_figure_numbers(self) -> None:
        self.assertEqual(stats.figure_number("1/3.1"), 1 / 3.1)
        self.assertEqual(stats.figure_number(" 1 / 4 "), 0.25)
        self.assertEqual(stats.figure_number(0.5), 0.5)
        self.assertEqual(stats.figure_number("0.5"), 0.5)
        for bad in ("1/0", "a/b", "", None, True, "inf", float("nan")):
            with self.subTest(bad=bad):
                self.assertIsNone(stats.figure_number(bad))
        broken = stats.published_reference({"defaults": {"direction": "baseline_over_harness"},
                                            "consensus": {"cost_per_trial": {"low": 0, "high": "x"}}})
        self.assertEqual(broken["consensus"]["cost_per_trial"], {"low": None, "high": None})
        self.assertIsNone(stats.reference_band(broken["consensus"]["cost_per_trial"]))
        self.assertEqual(stats.published_reference(None), {})


class Schedules(BenchTestCase):
    def test_lpt_makespan(self) -> None:
        self.assertEqual(stats.lpt_makespan([5, 4, 3, 3, 3], 2), 10.0)  # LPT's textbook miss (optimum 9)
        self.assertEqual(stats.lpt_makespan([3, 1, 4, 1, 5], 2), 7.0)
        self.assertEqual(stats.lpt_makespan([2, 2, 2], 5), 2.0)
        self.assertEqual(stats.lpt_makespan([1, 2, 3], 1), 6.0)
        self.assertEqual(stats.lpt_makespan([], 3), 0.0)
        self.assertEqual(stats.lpt_makespan([4, 4], 0), 8.0)  # at least one slot

    def test_list_schedule_keeps_the_order(self) -> None:
        self.assertEqual(stats.list_schedule([1, 1, 4], 2), 5.0)
        self.assertEqual(stats.lpt_makespan([1, 1, 4], 2), 4.0)
        self.assertEqual(stats.list_schedule([10, 2, 2, 2], 2), 10.0)
        self.assertEqual(stats.list_schedule([], 2), 0.0)


class PairedLogRatio(BenchTestCase):
    def test_hand_computed_example(self) -> None:
        base = {"a": 4.0, "b": 9.0, "c": 2.0, "d": 5.0, "e": 1.0}
        arm = {"a": 2.0, "b": 3.0, "c": 1.0, "d": 5.0, "x": 7.0}
        logs = [math.log(1 / 2), math.log(1 / 3), math.log(1 / 2), 0.0]  # arm / baseline on a, b, c, d
        mean = statistics.fmean(logs)
        pooled = stats.paired_log_ratio(base, arm, pooled_sd=0.5)
        self.assertEqual((pooled.n, pooled.sd_source, pooled.sd_log), (4, "pooled", 0.5))
        self.assertAlmostEqual(pooled.multiple, math.exp(mean))
        self.assertLess(pooled.multiple, 1.0)  # the arm spends less: under 1x
        half = 2.353 * 0.5 / 2
        self.assertAlmostEqual(pooled.ci[0], math.exp(mean - half), places=3)
        self.assertAlmostEqual(pooled.ci[1], math.exp(mean + half), places=3)
        prior = stats.paired_log_ratio(base, arm)
        self.assertEqual((prior.sd_source, prior.sd_log), ("prior", 0.6))
        observed = stats.paired_log_ratio(base, arm, min_observed=4)
        self.assertEqual(observed.sd_source, "observed")
        self.assertAlmostEqual(observed.sd_log, statistics.stdev(logs))
        self.assertIsNone(stats.paired_log_ratio({"a": 1.0}, {"b": 1.0}))
        self.assertIsNone(stats.paired_log_ratio({"a": 0.0}, {"a": 1.0}))


@unittest.skipUnless(os.environ.get("BENCH_SLOW") == "1", "coverage simulations run with BENCH_SLOW=1")
class Coverage(BenchTestCase):
    def test_t_interval_on_log_ratios_covers(self) -> None:
        rng = random.Random(1)
        hits = 0
        for _ in range(2000):
            logs = [rng.gauss(0.8, 0.6) for _ in range(10)]
            s = stats.log_ratio_summary(logs, 0.95)
            hits += s["lo"] <= math.exp(0.8) <= s["hi"]
        self.assertTrue(0.93 <= hits / 2000 <= 0.97, hits / 2000)

    def test_ratio_of_means_bootstrap_covers_reasonably(self) -> None:
        rng = random.Random(2)
        hits = 0
        reps = 400
        for i in range(reps):
            pairs = []
            for _ in range(10):
                size = rng.lognormvariate(0, 0.8)
                pairs.append((size * 2.5 * rng.lognormvariate(0, 0.3), size * rng.lognormvariate(0, 0.3)))
            ci = stats.bootstrap_ci(pairs, lambda p: sum(b for b, _ in p) / sum(h for _, h in p), resamples=1000,
                                    seed=i, level=0.95)
            hits += ci[0] <= 2.5 <= ci[1]
        self.assertGreaterEqual(hits / reps, 0.85)


if __name__ == "__main__":
    unittest.main()
