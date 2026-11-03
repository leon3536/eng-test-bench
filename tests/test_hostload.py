"""Host CPU load: the busy-share maths, the sampler, and each trial's
contention over its agent execution, as extract stores it in flags_json."""

from __future__ import annotations

import dataclasses
import json
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests import ledger_fixtures as lf
from tests.support import BenchTestCase

from bench import classify, extract, hostload
from bench.extract import RunContext

PREFIXES = {"openrouter/": "openrouter"}
# make_trial's agent execution: 300 s from _T0 + 60 s.
AGENT_START = lf._T0 + timedelta(seconds=60)


def ctx(run_id: str = "r1", source: str = "bench") -> RunContext:
    return RunContext(run_key=f"dev/{run_id}", profile="dev", run_id=run_id, source=source, arm="pi",  # type: ignore[arg-type]
                      provider="openrouter", model_prefix="openrouter/", effort_label="medium",
                      dataset="terminal-bench@2.0", local_snapshot=None)


def write_load(path: Path, start: datetime, cpus: list[float], step: float = hostload.SAMPLE_SEC) -> Path:
    """One hostload line per value, the first a full sample after start."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"t": (start + timedelta(seconds=step * (i + 1))).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                         "cpu": cpu}) for i, cpu in enumerate(cpus)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class BusyFraction(BenchTestCase):
    def test_share_of_cpu_time_spent_busy(self) -> None:
        self.assertAlmostEqual(hostload.busy_fraction((100, 1000), (400, 2000)), 0.3)
        self.assertEqual(hostload.busy_fraction((0, 0), (500, 500)), 1.0)
        self.assertEqual(hostload.busy_fraction((0, 0), (0, 800)), 0.0)

    def test_no_time_passed_is_unknown(self) -> None:
        self.assertIsNone(hostload.busy_fraction((5, 10), (5, 10)))
        self.assertIsNone(hostload.busy_fraction((5, 10), (4, 9)))  # a counter that went back

    def test_windows_counters_take_idle_out_of_kernel_time(self) -> None:
        # GetSystemTimes' kernel time includes idle: busy = kernel + user - idle.
        if hostload.os.name != "nt":
            self.skipTest("Windows only")
        busy, total = hostload.cpu_counters()
        self.assertGreater(total, 0)
        self.assertLessEqual(busy, total)

    def test_proc_stat(self) -> None:
        stat = self.tmpdir() / "stat"
        # user nice system idle iowait irq softirq steal
        stat.write_text("cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 1 2 3 4\n", encoding="ascii")
        self.assertEqual(hostload._proc_stat_counters(str(stat)), (150, 1000))


class Sampling(BenchTestCase):
    def test_sampler_appends_lines_and_stops(self) -> None:
        path = self.tmpdir() / "runs" / "r1" / hostload.HOSTLOAD_FILE
        readings = iter([(0, 100), (25, 200), (125, 300), (125, 400)] + [(125, 400)] * 1000)
        three = threading.Event()
        written: list[int] = []

        def counters() -> tuple[int, int]:
            value = next(readings)
            written.append(1)
            if len(written) >= 4:
                three.set()
            return value

        sampler = hostload.Sampler(path, interval=0.01, counters=counters).start()
        self.assertTrue(three.wait(20))
        sampler.stop()
        self.assertFalse(sampler._thread.is_alive())
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertGreaterEqual(len(rows), 2)
        # 25 of 100, 100 of 100; then 0 of 100 for every later reading.
        self.assertEqual([r["cpu"] for r in rows[:3]], [25.0, 100.0, 0.0][:len(rows[:3])])
        self.assertTrue(all(r["t"].endswith("Z") for r in rows))
        size = path.stat().st_size
        threading.Event().wait(0.05)
        self.assertEqual(path.stat().st_size, size)  # stopped: nothing more written

    def test_a_failing_counter_never_raises(self) -> None:
        path = self.tmpdir() / "hostload.jsonl"
        calls = threading.Event()

        def broken() -> tuple[int, int]:
            calls.set()
            raise OSError("no counters")

        with self.assertLogs("bench.hostload", "WARNING"):
            sampler = hostload.Sampler(path, interval=0.01, counters=broken).start()
            self.assertTrue(calls.wait(20))
            threading.Event().wait(0.05)
            sampler.stop()
        self.assertFalse(path.exists())


class Contention(unittest.TestCase):
    T0 = datetime(2026, 11, 1, 1, 0, tzinfo=timezone.utc)

    def samples(self, cpus: list[float]) -> list[hostload.Sample]:
        return [(self.T0 + timedelta(seconds=5 * (i + 1)), c) for i, c in enumerate(cpus)]

    def test_busy_for_a_quarter_of_the_agent_time_is_contended(self) -> None:
        window = [(self.T0, self.T0 + timedelta(seconds=40))]
        flags = hostload.contention(self.samples([95, 99, 40, 30, 20, 20, 10, 10]), window)
        self.assertEqual(flags, {"host_cpu_mean": 40.5, "host_cpu_hot_share": 0.25, "host_cpu_samples": 8,
                                 "cpu_contended": True})

    def test_less_than_a_quarter_or_too_few_samples_is_not(self) -> None:
        window = [(self.T0, self.T0 + timedelta(seconds=40))]
        self.assertFalse(hostload.contention(self.samples([95, 40, 40, 30, 20, 20, 10, 10]), window)["cpu_contended"])
        short = [(self.T0, self.T0 + timedelta(seconds=10))]
        flags = hostload.contention(self.samples([99, 99, 99, 99]), short)
        self.assertEqual((flags["host_cpu_samples"], flags["cpu_contended"]), (2, False))

    def test_samples_outside_the_window_do_not_count(self) -> None:
        window = [(self.T0 + timedelta(seconds=20), self.T0 + timedelta(seconds=40))]
        flags = hostload.contention(self.samples([99, 99, 99, 10, 10, 10, 10, 10, 99, 99]), window)
        self.assertEqual((flags["host_cpu_samples"], flags["host_cpu_hot_share"]), (4, 0.0))

    def test_nothing_known_sets_no_key(self) -> None:
        window = [(self.T0, self.T0 + timedelta(seconds=40))]
        self.assertEqual(hostload.contention(None, window), {})
        self.assertEqual(hostload.contention([], window), {})
        self.assertEqual(hostload.contention(self.samples([99] * 3), [(self.T0 + timedelta(hours=1),
                                                                       self.T0 + timedelta(hours=2))]), {})

    def test_torn_lines_are_skipped(self) -> None:
        got = hostload.parse_lines(['{"t": "2026-11-01T01:00:05.000000Z", "cpu": 50}', '{"t": "2026-11-0',
                                    '{"t": "x", "cpu": 1}', '[1]', '{"t": "2026-11-01T01:00:10.000000Z", "cpu": true}'])
        self.assertEqual(got, [(self.T0 + timedelta(seconds=5), 50.0)])


class TrialFlags(BenchTestCase):
    """extract reads runs/<run_id>/hostload.jsonl beside jobs/<run_id>/<arm>/<job>/<trial>."""

    def trial(self, cpus: list[float] | None, **kw: object) -> extract.TrialFacts:
        data = self.tmpdir()
        trial = lf.make_trial(data / "jobs" / "r1" / "pi" / "job" / "t", **kw)  # type: ignore[arg-type]
        if cpus is not None:
            write_load(data / "runs" / "r1" / hostload.HOSTLOAD_FILE, AGENT_START, cpus)
        facts = extract.parse_trial(trial, ctx(), PREFIXES)
        assert facts is not None
        return facts

    def test_a_busy_host_marks_the_trial_contended(self) -> None:
        facts = self.trial([99.0] * 30 + [10.0] * 30)  # 300 s of agent time: half of it hot
        flags = json.loads(facts.flags_json)
        self.assertEqual((flags["host_cpu_samples"], flags["host_cpu_hot_share"], flags["cpu_contended"]),
                         (60, 0.5, True))
        self.assertEqual(flags["host_cpu_mean"], 54.5)

    def test_a_quiet_host_is_not_contended(self) -> None:
        flags = json.loads(self.trial([30.0] * 60).flags_json)
        self.assertIs(flags["cpu_contended"], False)

    def test_no_hostload_file_is_unknown(self) -> None:
        flags = json.loads(self.trial(None).flags_json)
        self.assertFalse(any(k in flags for k in hostload.HOST_FLAGS))

    def test_hostload_path_needs_a_bench_run_folder(self) -> None:
        trial = Path("C:/data/jobs/r1/pi/job/t")
        self.assertEqual(extract.hostload_path(trial, ctx()), Path("C:/data/runs/r1/hostload.jsonl"))
        self.assertIsNone(extract.hostload_path(trial, ctx(source="external")))
        self.assertIsNone(extract.hostload_path(Path("C:/data/elsewhere/pi/t"), ctx()))

    def test_contended_timeout_classifies_as_infra(self) -> None:
        facts = self.trial([99.0] * 60, exception_type="AgentTimeoutError", reward=0.0)
        outcome, reason = classify.classify(dataclasses.asdict(facts))
        self.assertEqual(outcome, "infra")
        self.assertEqual(reason, "AgentTimeoutError while the host CPU was busy (100% of the agent's time at >=90%)")


if __name__ == "__main__":
    unittest.main()
