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

from bench import report

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


class ReportOptionControls(unittest.TestCase):
    def test_every_default_path_is_a_server_option(self):
        defaults = _option_defaults()
        self.assertIn("compare", defaults)
        self.assertIn("baseline.model", defaults)
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
        ]
        script = "\n".join([*helpers, prose, f"const cases = {json.dumps(cases)};",
                            "console.log(JSON.stringify(cases.map(([c, t]) => modelProse(c, t))));"])
        out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60, check=True).stdout
        self.assertEqual(json.loads(out), [
            "Sonnet 5.5 vs Opus 5.5 with claude-code: Sonnet 5.5 used 0.38x the cost of Opus 5.5; models other than "
            "claude-opus-5-5-20261001 answered",
            "openrouter:Opus 5.5 vs Opus 5.5 with pi",
            "pi on claude-opus-5-5",  # comparing harnesses, the text is the server's
        ])


def _dig(obj: dict[str, Any], path: str) -> Any:
    for k in path.split("."):
        obj = obj[k]
    return obj


if __name__ == "__main__":
    unittest.main()
