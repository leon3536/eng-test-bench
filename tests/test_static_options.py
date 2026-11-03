"""The REPORT tab's option controls against the server's report options.

report.js sends option paths (OPTION_DEFAULTS, the Compare switch's
COMPARE_CHOICES) that the server must accept and echo back in spec.options,
since the controls show the echo; a renamed option would otherwise fail only
in the browser. demo-report.js answers ?demo=1 with the same option shape.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path
from typing import Any

from bench import classify, report

STATIC = Path(__file__).resolve().parents[1] / "bench" / "static"


def _block(text: str, name: str) -> str:
    """The body of `const NAME = {...};` or `const NAME = [...];` in a script."""
    match = re.search(rf"const {name} = ([{{\[])(.*?)\n  [}}\]];", text, re.S)
    if not match:
        raise AssertionError(f"const {name} not found")
    return match.group(2)


def _option_defaults() -> dict[str, Any]:
    """report.js OPTION_DEFAULTS as {path: value}; JS literals read as Python."""
    body = _block((STATIC / "report.js").read_text(encoding="utf-8"), "OPTION_DEFAULTS")
    out: dict[str, Any] = {}
    for key, raw in re.findall(r"'?([\w.]+)'?: ('[^']*'|true|false|null|[\d.]+)", body):
        out[key] = _literal(raw)
    return out


def _literal(raw: str) -> Any:
    if raw in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[raw]
    return raw.strip("'") if raw.startswith("'") else float(raw)


def _nested(path: str, value: Any) -> dict[str, Any]:
    keys = path.split(".")
    out: dict[str, Any] = {keys[-1]: value}
    for k in reversed(keys[:-1]):
        out = {k: out}
    return out


class ReportOutcomes(unittest.TestCase):
    def test_report_labels_every_ledger_outcome_in_order(self):
        body = _block((STATIC / "report.js").read_text(encoding="utf-8"), "OUTCOMES")
        self.assertEqual(tuple(re.findall(r"^\s+(\w+): \{glyph", body, re.M)), classify.OUTCOMES)


class ReportOptionControls(unittest.TestCase):
    def test_every_default_path_is_a_server_option(self):
        defaults = _option_defaults()
        self.assertIn("compare", defaults)
        self.assertIn("baseline.model", defaults)
        self.assertEqual(defaults["baseline.effort"], report.DEFAULT_BASELINE_EFFORT)
        echo = report.options_dict(report.parse_options({}, None))
        for path, value in defaults.items():
            with self.subTest(path=path):
                found = echo
                for k in path.split("."):
                    self.assertIn(k, found, f"spec.options has no {path}")
                    found = found[k]
                if value is not None:
                    parsed = report.options_dict(report.parse_options(_nested(path, value), None))
                    self.assertEqual(_dig(parsed, path), value)

    def test_compare_choices_are_the_server_axes(self):
        body = _block((STATIC / "report.js").read_text(encoding="utf-8"), "SIDES")
        self.assertEqual(sorted(re.findall(r"^    (\w+): \{", body, re.M)), sorted(report.COMPARE_AXES))
        text = (STATIC / "report.js").read_text(encoding="utf-8")
        choices = re.search(r"const COMPARE_CHOICES = \[(.*?)\];", text).group(1)
        self.assertEqual(sorted(re.findall(r"\['(\w+)'", choices)), sorted(report.COMPARE_AXES))
        for axis in report.COMPARE_AXES:
            self.assertEqual(report.parse_options({"compare": axis}, None).compare, axis)

    def test_baseline_model_option_round_trips(self):
        o = report.parse_options({"compare": "models", "baseline": {"model": "anthropic:claude-sonnet-5-5"}}, None)
        self.assertEqual(report.options_dict(o)["baseline"]["model"], "anthropic:claude-sonnet-5-5")

    def test_demo_answers_with_the_same_option_shape(self):
        body = _block((STATIC / "demo-report.js").read_text(encoding="utf-8"), "DEFAULT_OPTIONS")
        top = set(re.findall(r"(?:^|[{,]\s*)(\w+): ", body.replace("\n", " ")))
        self.assertLessEqual({"compare", "baseline", "setup", "pricing"}, top)
        self.assertIn("model: null", body)
        self.assertIn("MODELS_NO_VERDICT = '" + report.MODELS_NO_VERDICT + "'",
                      (STATIC / "demo-report.js").read_text(encoding="utf-8"))


class ModelProse(unittest.TestCase):
    """report.js shows the server's model-comparison headline and caveats
    with the model labels its cards use ('Sonnet 5.5', not the id)."""

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_model_ids_in_server_prose_read_as_labels(self):
        app = (STATIC / "app.js").read_text(encoding="utf-8")
        page = (STATIC / "report.js").read_text(encoding="utf-8")
        helpers = [re.search(r"^const titleCase = .*?;$", app, re.M).group(0),
                   re.search(r"^const modelLabel = .*?^};$", app, re.S | re.M).group(0)]
        prose = re.search(r"^  function modelProse\(.*?^  }$", page, re.S | re.M).group(0)
        cases = [
            ({"compare": "models", "arm_name": "claude-sonnet-5-5", "baseline_name": "claude-opus-5-5"},
             "claude-sonnet-5-5 vs claude-opus-5-5 with claude-code: claude-sonnet-5-5 used 0.38x the cost of "
             "claude-opus-5-5; models other than claude-opus-5-5-20261001 answered"),
            ({"compare": "models", "arm_name": "openrouter:claude-opus-5-5",
              "baseline_name": "anthropic:claude-opus-5-5"},
             "openrouter:claude-opus-5-5 vs anthropic:claude-opus-5-5 with pi"),
            ({"compare": "harnesses", "arm_name": "pi", "baseline_name": "claude-code"}, "pi on claude-opus-5-5"),
            # Efforts are part of a model comparison's names, and of their labels.
            ({"compare": "models", "arm_name": "claude-opus-5-5 low", "baseline_name": "claude-opus-5-5 medium"},
             "claude-opus-5-5 low vs claude-opus-5-5 medium with pi: claude-opus-5-5 low used 0.67x the cost of "
             "claude-opus-5-5 medium"),
        ]
        script = "\n".join([*helpers, prose, f"const cases = {json.dumps(cases)};",
                            "console.log(JSON.stringify(cases.map(([c, t]) => modelProse(c, t))));"])
        out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60, check=True).stdout
        self.assertEqual(json.loads(out), [
            "Sonnet 5.5 vs Opus 5.5 with claude-code: Sonnet 5.5 used 0.38x the cost of Opus 5.5; models other than "
            "claude-opus-5-5-20261001 answered",
            "openrouter:Opus 5.5 vs Opus 5.5 with pi",
            "pi on claude-opus-5-5",  # comparing harnesses, the text is the server's
            "Opus 5.5 low vs Opus 5.5 medium with pi: Opus 5.5 low used 0.67x the cost of Opus 5.5 medium",
        ])


class DemoAgentTime(unittest.TestCase):
    """demo-report.js gives every comparison the server's time block
    (report.compare_time), and report.js draws it as a third plot and column."""

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_demo_comparisons_carry_the_time_multiple(self):
        script = r"""
const fs = require('fs');
const handlers = [];
const demo = {store: {}, on: (m, p, fn) => handlers.push([m, p, fn]), runs: () => [], json: (x) => x,
  error: (c, m) => ({c, m}), profile: () => 'default', tasks: () => Array.from({length: 30}, (_, i) => `task-${i}`)};
global.window = {benchDemo: new Proxy(demo, {get: (t, k) => (k in t ? t[k] : () => undefined)})};
global.location = {search: ''};
eval(fs.readFileSync(process.argv[1], 'utf8'));
const post = handlers.find(([m, p]) => m === 'POST' && p === '/api/report')[2];
(async () => {
  const out = [];
  for (const compare of ['harnesses', 'models']) {
    const r = await post({body: {options: {compare}}});
    out.push(...r.comparisons.map((c) => c.time));
  }
  console.log(JSON.stringify(out));
})();
"""
        out = subprocess.run(["node", "-e", script, str(STATIC / "demo-report.js")], capture_output=True, text=True,
                             timeout=120, check=True).stdout
        times = json.loads(out)
        self.assertTrue(times)
        keys = {"status", "n", "need", "geomean", "ratio_of_means", "per_task", "dropped", "host_busy_tasks",
                "contended_trials", "basis"}
        for t in times:
            self.assertEqual(set(t), keys)
            self.assertEqual(t["basis"], "agent_exec_sec")
        self.assertTrue(any(t["host_busy_tasks"] > 0 for t in times))  # the demo shows tasks left out
        self.assertTrue(any(t["geomean"] for t in times))

    def test_report_draws_time_next_to_tokens_and_cost(self):
        page = (STATIC / "report.js").read_text(encoding="utf-8")
        self.assertIn("forest(comps, 'time', `Time vs ${baseName}`", page)
        self.assertIn("`Time vs ${baseName}`)", page)  # the comparison table's column
        self.assertIn("timeCell(c.time)", page)
        self.assertIn("left out: host busy", page)
        # The pairs drawer: a time row in the per-task plot and a time column naming tasks the busy host dropped.
        self.assertIn("['Agent time', 'ratio_time', c.time]", page)
        self.assertIn("tdNum(timeRatioText(p))", page)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_demo_pairs_carry_each_sides_time(self):
        script = r"""
const fs = require('fs');
const handlers = [];
const demo = {store: {}, on: (m, p, fn) => handlers.push([m, p, fn]), runs: () => [], json: (x) => x,
  error: (c, m) => ({c, m}), profile: () => 'default', tasks: () => Array.from({length: 30}, (_, i) => `task-${i}`)};
global.window = {benchDemo: new Proxy(demo, {get: (t, k) => (k in t ? t[k] : () => undefined)})};
global.location = {search: ''};
eval(fs.readFileSync(process.argv[1], 'utf8'));
const find = (p) => handlers.find(([m, q]) => m === 'POST' && q === p)[2];
(async () => {
  const out = [];
  for (const compare of ['harnesses', 'models']) {
    const body = {options: {compare}};
    const r = await find('/api/report')({body});
    for (const c of r.comparisons) {
      const pairs = await find('/api/report/pairs')({body: {...body, comparison: c.key}});
      out.push({time: c.time, pairs: pairs.pairs});
    }
  }
  console.log(JSON.stringify(out));
})();
"""
        out = subprocess.run(["node", "-e", script, str(STATIC / "demo-report.js")], capture_output=True, text=True,
                             timeout=120, check=True).stdout
        found = json.loads(out)
        self.assertTrue(found)
        for c in found:
            busy = 0
            for p in c["pairs"]:
                for side in ("arm", "baseline"):
                    self.assertIn("time_sec", p[side])
                    self.assertIn("time_contended", p[side])
                busy += any(p[s]["time_sec"] is None and p[s]["time_contended"] > 0 for s in ("arm", "baseline"))
            self.assertEqual(busy, c["time"]["host_busy_tasks"])  # the drawer's "host busy" rows add up



def _report_parts(*names: str) -> str:
    """report.js consts and functions by name (`const X = {...};`, a one-statement `const x = ...;`, or
    `function x(...) {...}`), as script text node can run."""
    page = (STATIC / "report.js").read_text(encoding="utf-8")
    out = []
    for name in names:
        match = (re.search(rf"^  const {name} = [{{\[]\n.*?^  [}}\]];$", page, re.S | re.M)
                 or re.search(rf"^  const {name} = .*?;$", page, re.S | re.M)
                 or re.search(rf"^  function {name}\(.*?^  }}$", page, re.S | re.M))
        if not match:
            raise AssertionError(f"{name} not found in report.js")
        out.append(match.group(0))
    return "\n".join(out)


def _node(script: str) -> Any:
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=60,
                         check=True).stdout
    return json.loads(out)


_IS_NUM = "const isNum = (v) => typeof v === 'number' && Number.isFinite(v);"


def _gm(lo: float, hi: float, **extra: Any) -> dict[str, Any]:
    return {"status": "ok", "n": 12, "need": 0, "geomean": {"estimate": (lo + hi) / 2, "lo": lo, "hi": hi}, **extra}


class BaselineVerdict(unittest.TestCase):
    """The REPORT tab's verdict against the baseline: read off the server's
    interval (harness / baseline, so under 1x is better; pass rate in points),
    only once a comparison has status 'ok'."""

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_the_interval_against_1x_decides(self):
        cases = [
            (_gm(0.4, 0.8), "cost"),
            (_gm(1.2, 1.9), "cost"),
            (_gm(0.7, 1.3), "tokens_total"),
            (_gm(0.6, 0.9, sensitivity={"unrecorded_arm_high": 1.1, "crashes_excluded": 0.7}), "cost"),
            (_gm(0.6, 0.9, sensitivity={"unrecorded_arm_high": 0.8, "crashes_excluded": 1.4}), "cost"),
            ({**_gm(0.4, 0.8), "status": "low_n", "need": 3}, "cost"),
            ({"status": "insufficient", "n": 2, "need": 6, "geomean": None}, "time"),
            ({"status": "not_measured", "n": 0, "need": 0}, "cost"),
            (_gm(1.3, 2.0), "time"),
            ({"status": "ok", "n": 20, "need": 0, "delta_pp": {"estimate": 12, "lo": 3, "hi": 21}}, "pass"),
            ({"status": "ok", "n": 20, "need": 0, "delta_pp": {"estimate": -12, "lo": -21, "hi": -3}}, "pass"),
            ({"status": "ok", "n": 20, "need": 0, "delta_pp": {"estimate": 0, "lo": None, "hi": None}}, "pass"),
            (None, "cost"),
        ]
        script = "\n".join([_IS_NUM, _report_parts("BETTER_MEANS", "WORSE_MEANS", "baselineVerdict"),
                            f"const cases = {json.dumps(cases)};",
                            "console.log(JSON.stringify(cases.map(([m, k]) => baselineVerdict(m, k))));"])
        found = _node(script)
        self.assertEqual([v and v[0] for v in found], [
            "better", "worse", "same",
            "same",    # trials cut short, costed high, put it at 1x or over
            "better",  # only the unrecorded checks can turn it, not the crash policy
            "few", "few", None, "worse", "better", "worse", "same", None])
        self.assertEqual(found[0][1], "Cheaper than the baseline: the whole interval is below 1×.")
        self.assertEqual(found[5][1], "3 more paired tasks for a verdict")
        # Plain words: no interval jargon in what the person reads.
        for v in filter(None, found):
            self.assertNotRegex(v[1], r"\b(CI|overlaps?|confidence)\b")

    def test_every_comparison_row_shows_it(self):
        page = (STATIC / "report.js").read_text(encoding="utf-8")
        words = re.findall(r"^\s+(\w+): \['[^']+', '([^']+)'", _block(page, "VS_BASELINE"), re.M)
        self.assertEqual(words, [("better", "better"), ("worse", "worse"), ("same", "possibly the same"),
                                 ("few", "not enough data")])
        self.assertIn("h('td', null, baselineRows(c))", page)
        self.assertIn("`Verdict vs ${baseName}`", page)
        self.assertIn("colspan: models ? 8 : 10", page)  # the headline row spans the added column


class HarnessHung(unittest.TestCase):
    """A timed-out trial with zero usage recorded reads "harness hung"."""

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_only_a_timeout_with_zero_usage_is_a_hang(self):
        hung = {"outcome": "agent_limit", "exception_type": "AgentTimeoutError", "agent_started": True,
                "n_input": 0, "n_output": 0}
        cases = [
            hung,
            {**hung, "n_input": None},                       # one side recorded a zero
            {**hung, "n_input": 1200, "n_output": 40},       # it got answers, then ran out of time
            {**hung, "n_input": None, "n_output": None},     # the harness writes usage only on exit
            {**hung, "outcome": "infra"},                    # a busy host's timeout is the host's
            {**hung, "exception_type": "NonZeroAgentExitCodeError"},
            {**hung, "agent_started": False},
            None,
        ]
        script = "\n".join([_report_parts("isHung"), f"const cases = {json.dumps(cases)};",
                            "console.log(JSON.stringify(cases.map(isHung)));"])
        self.assertEqual(_node(script), [True, True, False, False, False, False, False, False])

    def test_the_drawer_and_tables_pass_the_trial(self):
        page = (STATIC / "report.js").read_text(encoding="utf-8")
        self.assertEqual(page.count("outcomeTag(t.outcome, t)"), 3)  # the trial drawer, the ledger, the pairs trials
        self.assertIn("outcomeTag(e.outcome, row)", page)  # the pairs' "Left out" table
        self.assertNotIn("outcomeTag(e.outcome)", page)
        self.assertIn("'harness hung'", page)


class ReportPrint(unittest.TestCase):
    def test_print_button_uses_the_print_stylesheet(self):
        page = (STATIC / "report.js").read_text(encoding="utf-8")
        css = (STATIC / "report.css").read_text(encoding="utf-8")
        self.assertIn("onclick: printReport", page)
        self.assertIn("window.print()", page)
        self.assertIn("document.body.classList.add('rp-print')", page)
        block = css[css.index("@media print"):]
        for hidden in ("header.bar", ".rp-rail", ".rp-tabs", ".rp-side", ".rp-strip-actions"):
            self.assertIn(hidden, block)


_PROFILE_SCRIPT = r"""
const unique = (list) => [...new Set(list)];
const store = {};
const prefs = {get: (k, d) => (k in store ? store[k] : d), set: (k, v) => { store[k] = v; }};
let rp;
let ui;
%PARTS%
const out = [];
for (const [saved, handed] of [[null, null], [{filters: {profiles: ['dev'], models: ['m']}}, null],
                               [null, {profiles: ['dev']}]]) {
  for (const k of Object.keys(store)) delete store[k];
  if (saved) store['report.rail'] = saved;
  rp = {filters: {}, options: {}, handed};
  ui = {state: {config: {profile: 'default'}}};
  loadPrefs();
  out.push(rp.filters);
}
console.log(JSON.stringify(out));
"""


class ReportProfile(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_the_report_opens_on_the_active_profile(self):
        parts = _report_parts("TABS", "LIST_KEYS", "NULLABLE", "cleanFilters", "loadPrefs", "savePrefs")
        self.assertEqual(_node(_PROFILE_SCRIPT.replace("%PARTS%", parts)), [
            {"profiles": ["default"]},
            {"profiles": ["default"], "models": ["m"]},  # other saved filters stay
            {"profiles": ["dev"]},                       # filters handed over from another view win
        ])

def _dig(obj: dict[str, Any], path: str) -> Any:
    for k in path.split("."):
        obj = obj[k]
    return obj


if __name__ == "__main__":
    unittest.main()
