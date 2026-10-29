"""The assistant's tools cannot start, queue, approve or cancel anything, and
cannot leak a key.

The guarantees are structural, so they are tested structurally: exactly nine
tools; a ToolContext with nothing that can start or enqueue; an AST scan of
every bench/assistant module for the imports and names that would reach the
run manager, the queue, the server or the key store; hallucinated tools that
end as errors while a patched RunManager.start_spec and RunQueue.add would
fail the test if touched; strict inputs (a draft cannot smuggle a status);
and redaction of a planted key wherever it could surface. The tools run over
a real temp ledger and design store; nothing calls a model.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest import mock

from tests import FAKE_KEYS
from tests.support import BenchTestCase
from tests.test_coverage_suggest import GLM, MINI, SAMPLE, PlannerWorld

from bench import designs, runqueue, runs, schemas, settings
from bench.assistant import tools
from bench.assistant.types import ToolContext, ToolOutcome
from bench.designs import DesignStore, DraftWriter

ASSISTANT_DIR = Path(tools.__file__).resolve().parent
CID = "c-20261029-010101-ab12"
PLANTED = "plantedvalue-QXZ-9876543210"
EXPECTED_TOOLS = ["get_bench_context", "describe_harness", "list_tasks", "get_coverage", "get_cost_stats",
                  "list_designs", "estimate_cost", "suggest_runs", "save_draft_design"]
# Names that would reach billed work, approval or keys (safety section 3).
BANNED = frozenset({"start", "start_spec", "start_design", "cancel", "add_listener", "set_secret", "clear_secret",
                    "add_exclusion", "rescan", "save_snapshot", "upsert_bench_run", "ingest_bench_job"})
BANNED_MODULES = ("bench.runqueue", "bench.server")
ONLY_IN = {"get_secret": "client", "secret_values": "redact"}

SPEC = {"tasks": SAMPLE[:3], "run": {"budget_usd": 2.0},
        "arms": [{"name": "cc", "agent": "claude-code", "provider": "openrouter", "model": "z-ai/glm-5.3"},
                 {"name": "pi", "agent": "pi", "provider": "openrouter", "model": "z-ai/glm-5.3"}]}
SAVE = {"title": "Block 1 pairs on GLM", "purpose": "Does pi use a third of Claude Code's cost on GLM?",
        "rationale": "get_coverage: 4 pairs; estimate_cost p90 within budget.", "change_note": "first draft",
        "spec": SPEC}


# The import boundary --------------------------------------------------------------------------------

def _resolve(node: ast.ImportFrom) -> str:
    """The absolute module of `from <...> import`, for files in bench/assistant."""
    if not node.level:
        return node.module or ""
    package = ["bench", "assistant"][: max(0, 2 - (node.level - 1))]
    return ".".join(package + ([node.module] if node.module else []))


def boundary_violations(source: str, module: str) -> list[str]:
    """Imports and names a bench/assistant module must not use. Thread
    objects may still .start(): only a receiver named *thread is allowed."""
    out: list[str] = []
    for node in ast.walk(ast.parse(source)):
        line = getattr(node, "lineno", 0)
        names: list[str] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(BANNED_MODULES):
                    out.append(f"{module}:{line} imports {alias.name}")
                names.append(alias.name.rsplit(".", 1)[-1])
        elif isinstance(node, ast.ImportFrom):
            source_module = _resolve(node)
            if source_module.startswith(BANNED_MODULES):
                out.append(f"{module}:{line} imports from {source_module}")
            for alias in node.names:
                if f"{source_module}.{alias.name}".startswith(BANNED_MODULES):
                    out.append(f"{module}:{line} imports {source_module}.{alias.name}")
                names.append(alias.name)
        elif isinstance(node, ast.Attribute):
            if node.attr == "start" and ast.unparse(node.value).lower().endswith("thread"):
                continue
            names.append(node.attr)
        elif isinstance(node, ast.Name):
            if node.id != "start":  # a local variable; reaching a start method needs an attribute or import
                names.append(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
        for name in names:
            if name == "RunManager" or name in BANNED:
                out.append(f"{module}:{line} uses {name}")
            if name in ONLY_IN and module != ONLY_IN[name]:
                out.append(f"{module}:{line} uses {name} (only {ONLY_IN[name]}.py may)")
    return out


class ImportBoundary(BenchTestCase):
    def test_no_assistant_module_reaches_runs_the_queue_the_server_or_keys(self) -> None:
        files = sorted(ASSISTANT_DIR.glob("*.py"))
        self.assertIn("tools.py", [f.name for f in files])
        violations = []
        for path in files:
            violations += boundary_violations(path.read_text(encoding="utf-8"), path.stem)
        self.assertEqual(violations, [])

    def test_the_scan_catches_what_it_forbids(self) -> None:
        bad = (
            "from ..runqueue import RunQueue\n"
            "from .. import server\n"
            "import bench.server\n"
            "from ..runs import RunManager\n"
            "from .. import keys\n"
            "def cancel(): pass\n"
            "def go(manager, ledger):\n"
            "    manager.start_spec(None, None)\n"
            "    manager.start(None, 'pilot', [])\n"
            "    ledger.add_exclusion('run', 'x', 'exclude', 'r')\n"
            "    keys.get_secret('ANTHROPIC_API_KEY')\n"
            "    keys.secret_values()\n"
        )
        found = "\n".join(boundary_violations(bad, "tools"))
        for expected in ("bench.runqueue", "bench.server", "RunManager", "uses cancel", "uses start_spec",
                         "uses start", "uses add_exclusion", "get_secret", "secret_values"):
            self.assertIn(expected, found)
        threads = ("import threading\nturn_thread = threading.Thread()\nturn_thread.start()\n"
                   "class T:\n    def go(self, t):\n        self._thread.start()\n        t.thread.start()\n")
        self.assertEqual(boundary_violations(threads, "engine"), [])
        self.assertEqual(boundary_violations("worker.start()\n", "engine"), ["engine:1 uses start"])
        self.assertEqual(boundary_violations("from .. import keys\nkeys.get_secret('X')\n", "client"), [])


# Registry and context ----------------------------------------------------------------------------------

class Registry(BenchTestCase):
    def test_exactly_nine_tools_in_a_fixed_order(self) -> None:
        self.assertEqual([d["name"] for d in tools.TOOL_DEFS], EXPECTED_TOOLS)
        self.assertEqual(tools.ALLOWED_TOOL_NAMES, frozenset(EXPECTED_TOOLS))
        self.assertEqual(set(tools.INPUT_MODELS), set(EXPECTED_TOOLS))
        for d in tools.TOOL_DEFS:
            with self.subTest(d["name"]):
                self.assertIs(d["eager_input_streaming"], True)
                self.assertNotIn("strict", d)
                self.assertEqual(set(d), {"name", "description", "input_schema", "eager_input_streaming"})
                self.assertEqual(d["input_schema"]["type"], "object")
                self.assertIs(d["input_schema"]["additionalProperties"], False)
                self.assertNotIn("harnesses", d["input_schema"]["properties"])  # inputs say families
        self.assertTrue(tools.TOOLS_HASH.startswith("sha256:"))

    def test_input_models_match_the_schemas_the_model_sees(self) -> None:
        for d in tools.TOOL_DEFS:
            with self.subTest(d["name"]):
                model = tools.INPUT_MODELS[d["name"]]
                schema = d["input_schema"]
                self.assertEqual(set(model.model_fields), set(schema["properties"]))
                required = {n for n, f in model.model_fields.items() if f.is_required()}
                self.assertEqual(required, set(schema.get("required", [])))
        spec = next(d for d in tools.TOOL_DEFS if d["name"] == "save_draft_design")["input_schema"]["properties"]
        self.assertEqual(spec["spec"], designs.DESIGN_SPEC_SCHEMA)

    def test_tool_context_holds_nothing_that_starts_cancels_or_enqueues(self) -> None:
        fields = {f.name for f in dataclasses.fields(ToolContext)}
        self.assertEqual(fields, {"cfg", "readiness", "key_flags", "runs_snapshot", "active_run", "queue_snapshot",
                                  "ledger", "drafts", "conversation_id", "spend", "saves_left"})
        for name in fields:
            for word in ("start", "cancel", "enqueue", "approve", "manager", "runqueue", "secret", "keys"):
                self.assertNotIn(word, name)
        drafts = {n for n in dir(DraftWriter) if not n.startswith("_")}
        self.assertEqual(drafts, {"conversation_id", "create_draft", "add_draft_version", "list", "meta", "version"})


# Tools over a temp ledger and design store -------------------------------------------------------------

class ToolsCase(BenchTestCase):
    def setUp(self) -> None:
        super().setUp()
        os.environ["PLANTED_SERVICE_TOKEN"] = PLANTED  # restored by BenchTestCase
        self.world = PlannerWorld(self)
        self.world.add("pi", [6], infra=True, exception_type=f"Err{PLANTED}",
                       message=f"token {PLANTED} and sk-ant-api03-{'x' * 40}")
        self.reader = self.world.gaps_world()
        self.cfg = self.world.cfg
        self.store_root = self.world.tmp / "designs"
        self.store = DesignStore(self.store_root)

    def ctx(self, **over: Any) -> ToolContext:
        values: dict[str, Any] = dict(
            cfg=self.cfg, readiness=lambda: [{"name": "Docker", "ok": True, "detail": "Docker Engine 29.1"}],
            key_flags=lambda: {"OPENROUTER_API_KEY": True, "ANTHROPIC_API_KEY": False},
            runs_snapshot=lambda: [], active_run=lambda: None,
            queue_snapshot=lambda: {"state": "paused", "pause_reason": "Paused after restart", "waiting": []},
            ledger=self.reader, drafts=DraftWriter(self.store, lambda: self.cfg, CID), conversation_id=CID,
            spend=lambda: {"turn_usd": 0.01, "today_usd": 0.2, "per_turn_cap_usd": 1.0, "daily_cap_usd": 5.0},
            saves_left=[5])
        values.update(over)
        return ToolContext(**values)

    def call(self, name: str, raw: Any, ctx: ToolContext | None = None) -> tuple[ToolOutcome, dict[str, Any]]:
        out = tools.execute(ctx or self.ctx(), name, raw)
        self.assertIsInstance(out, ToolOutcome)
        self.assertLessEqual(len(out.content), tools.MAX_RESULT_CHARS)
        return out, json.loads(out.content)

    def session(self, ctx: ToolContext) -> list[ToolOutcome]:
        """Every tool once, as a model would call them."""
        calls = [
            ("get_bench_context", {}),
            ("describe_harness", {"agent": "claude-code", "provider": "openrouter", "model": "z-ai/glm-5.3"}),
            ("describe_harness", {"agent": MINI}),
            ("list_tasks", {"provider": "openrouter", "model": "z-ai/glm-5.3", "limit": 5}),
            ("get_coverage", {"provider": "openrouter", "model": "z-ai/glm-5.3", "target_pairs": 8}),
            ("get_cost_stats", {"family": "pi", "per_task": True}),
            ("estimate_cost", {"spec": SPEC}),
            ("suggest_runs", {"goal": "reach_target_pairs", "provider": "openrouter", "model": "z-ai/glm-5.3",
                              "target_pairs": 6}),
            ("suggest_runs", {"goal": "pilot_new_arms", "provider": "openrouter", "model": "z-ai/glm-5.3",
                              "families": ["pi", "mini-swe-agent"]}),
            ("save_draft_design", SAVE),
            ("list_designs", {}),
        ]
        outcomes = [tools.execute(ctx, name, raw) for name, raw in calls]
        saved = next(o for o in outcomes if o.draft)
        outcomes.append(tools.execute(ctx, "list_designs", {"design_id": saved.draft["design_id"]}))
        outcomes.append(tools.execute(ctx, "estimate_cost", {"design_id": saved.draft["design_id"], "version": 1}))
        return outcomes


class NothingStarts(ToolsCase):
    def test_hallucinated_tools_are_errors_and_reach_nothing(self) -> None:
        with mock.patch.object(runs.RunManager, "start_spec", side_effect=AssertionError("start_spec")) as start_spec, \
                mock.patch.object(runs.RunManager, "start", side_effect=AssertionError("start")) as start, \
                mock.patch.object(runqueue.RunQueue, "add", side_effect=AssertionError("queue add")) as add, \
                mock.patch.object(runqueue.RunQueue, "start", side_effect=AssertionError("queue start")) as qstart:
            for name in ("start_run", "approve_design", "queue_design", "cancel_run", "set_secret", ""):
                with self.subTest(name):
                    out, body = self.call(name, {"design_id": "d-x-0000", "version": 1})
                    self.assertFalse(out.ok)
                    self.assertEqual(body["error"], "unknown_tool")
            ctx = self.ctx()
            outcomes = self.session(ctx)
        self.assertTrue(all(o.ok for o in outcomes), [o.content[:300] for o in outcomes if not o.ok])
        for patched in (start_spec, start, add, qstart):
            patched.assert_not_called()

    def test_a_draft_cannot_carry_a_status_and_the_handler_never_runs(self) -> None:
        handler = mock.Mock(side_effect=AssertionError("handler ran"))
        ctx = self.ctx()
        with mock.patch.dict(tools._HANDLERS, {"save_draft_design": handler}):
            for extra in ({"status": "approved"}, {"approved": True}, {"author": "user"}):
                out, body = self.call("save_draft_design", {**SAVE, **extra}, ctx)
                self.assertFalse(out.ok)
                self.assertIn("INVALID_JSON", body)
                self.assertTrue(any("Extra inputs" in e["msg"] for e in body["errors"]))
            spec_extra = {**SAVE, "spec": {**SPEC, "status": "approved"}}
            out, body = self.call("save_draft_design", spec_extra, ctx)
            self.assertIn("INVALID_JSON", body)
        handler.assert_not_called()
        self.assertEqual(self.store.list(), [])
        self.assertEqual(ctx.saves_left, [5])

    def test_strings_never_pass_for_numbers(self) -> None:
        handler = mock.Mock(side_effect=AssertionError("handler ran"))
        with mock.patch.dict(tools._HANDLERS, {"estimate_cost": handler}):
            bad = {"spec": {**SPEC, "run": {"budget_usd": "2"}}}
            out, body = self.call("estimate_cost", bad)
            self.assertEqual(body["errors"][0]["loc"], "spec.run.budget_usd")
            out, body = self.call("estimate_cost", {"spec": SPEC, "design_id": "d-x-0000"})
            self.assertIn("INVALID_JSON", body)  # exactly one source
            out, body = self.call("estimate_cost", "not an object")
            self.assertIn("INVALID_JSON", body)
        handler.assert_not_called()

    def test_a_deeply_nested_input_is_refused_not_raised(self) -> None:
        nested: Any = []
        for _ in range(3000):
            nested = [nested]
        out = tools.execute(self.ctx(), "list_tasks", {"extra": nested})
        self.assertFalse(out.ok)
        self.assertIn("INVALID_JSON", out.content)

    def test_list_tasks_limit_stops_where_a_page_still_fits(self) -> None:
        out, body = self.call("list_tasks", {"scope": "all", "limit": 81})
        self.assertFalse(out.ok)
        self.assertIn("INVALID_JSON", body)
        self.assertIn("max 80", json.dumps(tools.TOOL_DEFS))

    def test_drafts_are_always_by_the_assistant(self) -> None:
        ctx = self.ctx()
        out, body = self.call("save_draft_design", SAVE, ctx)
        self.assertTrue(out.ok, out.content)
        self.assertEqual(body["status"], "draft")
        self.assertIn("A person must queue or start it", body["note"])
        doc = self.store.version(body["design_id"], 1)
        self.assertEqual((doc["author"], doc["conversation_id"], doc["status"]), ("assistant", CID, "draft"))
        self.assertIsNone(doc["approved_at"])
        self.assertEqual(out.draft["design_id"], body["design_id"])
        self.assertEqual(ctx.saves_left, [4])
        again = {**SAVE, "design_id": body["design_id"], "base_version": 1, "change_note": "three more tasks",
                 "spec": {**SPEC, "tasks": SAMPLE[:6]}}
        out, body = self.call("save_draft_design", again, ctx)
        self.assertTrue(out.ok, out.content)
        self.assertEqual((body["version"], self.store.version(body["design_id"], 2)["author"]), (2, "assistant"))
        stale = {**again, "spec": {**SPEC, "tasks": SAMPLE[:5]}}  # still based on v1
        out, body = self.call("save_draft_design", stale, ctx)
        self.assertEqual(body["error"], "version_conflict")

    def test_a_draft_over_the_budget_limit_is_saved_invalid(self) -> None:
        out, body = self.call("save_draft_design", {**SAVE, "spec": {**SPEC, "run": {"budget_usd": 1000.0}}})
        self.assertTrue(out.ok, out.content)
        self.assertEqual(body["validation"]["status"], "invalid")
        self.assertTrue(any("max_budget_usd" in p for p in body["validation"]["problems"]))
        self.assertEqual(out.draft["problems"], len(body["validation"]["problems"]))
        self.assertEqual(self.store.version(body["design_id"])["status"], "draft")

    def test_at_most_five_saves_per_turn(self) -> None:
        ctx = self.ctx(saves_left=[1])
        first, _ = self.call("save_draft_design", SAVE, ctx)
        second, body = self.call("save_draft_design", {**SAVE, "title": "Another"}, ctx)
        self.assertTrue(first.ok)
        self.assertFalse(second.ok)
        self.assertEqual(body["error"], "save_limit")
        self.assertEqual(len(self.store.list()), 1)


class ImportPaths(ToolsCase):
    def test_an_import_path_that_left_the_config_is_never_looked_up(self) -> None:
        # Harbor imports an agent's code to print its schema, so a stored design
        # must not reach a path the profile no longer allows.
        spec = {**SPEC, "arms": [dict(SPEC["arms"][1], name="mini", agent=MINI)]}
        out, saved = self.call("save_draft_design", {**SAVE, "spec": spec})
        self.assertTrue(out.ok, out.content)
        narrowed = dataclasses.replace(self.cfg, arms=[a for a in self.cfg.arms if a.agent != MINI],
                                       designs=dataclasses.replace(self.cfg.designs, import_paths={}))
        self.assertNotIn(MINI, {h["agent"] for h in designs.allowed_harnesses(narrowed)})
        calls: list[str] = []
        real = schemas.agent_schema

        def lookup(agent: str) -> dict[str, Any]:
            calls.append(agent)
            if agent == MINI:
                raise AssertionError("looked up an import path the config no longer allows")
            return real(agent)

        ctx = self.ctx(cfg=narrowed, drafts=DraftWriter(self.store, lambda: narrowed, CID))
        with mock.patch.object(schemas, "agent_schema", lookup):
            out, body = self.call("estimate_cost", {"design_id": saved["design_id"]}, ctx)
            again, _ = self.call("estimate_cost", {"design_id": saved["design_id"], "version": 1}, ctx)
        self.assertTrue(out.ok and again.ok, out.content)
        self.assertEqual(body["validation"]["status"], "invalid")
        self.assertTrue(any("not a harness designs may use" in p for p in body["validation"]["problems"]))
        self.assertIsNone(body["estimate"]["arms"][0]["cell"]["config_hash"])
        self.assertIsNotNone(body["estimate"]["run"]["total"]["p50"])
        self.assertNotIn(MINI, calls)


class Secrets(ToolsCase):
    def all_text(self, outcomes: list[ToolOutcome]) -> str:
        return "\n".join(o.content + "\n" + o.summary + "\n" + json.dumps(o.draft) for o in outcomes)

    def test_a_planted_key_never_surfaces_in_a_session(self) -> None:
        text = self.all_text(self.session(self.ctx()))
        self.assertNotIn(PLANTED, text)
        self.assertNotIn("sk-ant-api03", text)
        for value in FAKE_KEYS.values():
            self.assertNotIn(value, text)
        for path in self.store_root.rglob("*"):
            if path.is_file():
                self.assertNotIn(PLANTED, path.read_text(encoding="utf-8"), path.name)

    def test_the_infra_gap_names_a_redacted_exception_type_and_never_the_message(self) -> None:
        out, body = self.call("get_coverage", {"provider": "openrouter", "model": "z-ai/glm-5.3"})
        self.assertTrue(out.ok)
        self.assertNotIn(PLANTED, out.content)
        self.assertNotIn("token ", out.content)
        pi = next(c for c in body["cells"] if c["family"] == "pi")
        self.assertEqual(pi["infra"], 1)

    def test_handler_errors_are_redacted_and_short(self) -> None:
        class BrokenLedger:
            def rows(self) -> list[dict[str, Any]]:
                raise RuntimeError(f"cannot open C:/data/{PLANTED}/bench.sqlite " + "x" * 2000)

            fingerprint = summary = rows

            def coverage(self, *args: Any) -> dict[str, Any]:
                return self.rows()  # type: ignore[return-value]

        ctx = self.ctx(ledger=BrokenLedger())
        for name, raw in (("get_coverage", {"provider": "openrouter", "model": "z-ai/glm-5.3"}),
                          ("get_cost_stats", {}), ("estimate_cost", {"spec": SPEC}),
                          ("list_tasks", {"provider": "openrouter", "model": "z-ai/glm-5.3"}),
                          ("suggest_runs", {"goal": "complete_half_pairs", "provider": "openrouter",
                                            "model": "z-ai/glm-5.3"})):
            with self.subTest(name):
                out, body = self.call(name, raw, ctx)
                self.assertFalse(out.ok)
                self.assertNotIn(PLANTED, out.content)
                self.assertIn("[redacted]", out.content)
                self.assertLessEqual(len(body["detail"]), tools.MAX_ERROR_DETAIL)
        out, body = self.call("estimate_cost", {"spec": SPEC}, ctx)
        self.assertEqual(body["error"], "ledger_unavailable")
        self.assertIn("The checks ran (valid", body["detail"])  # validation still works without the ledger
        out, body = self.call("get_bench_context", {}, ctx)
        self.assertTrue(out.ok)
        self.assertIn("error", body["ledger"])
        self.assertNotIn(PLANTED, out.content)

    def test_invalid_input_is_echoed_redacted(self) -> None:
        out, body = self.call("list_tasks", {"provider": PLANTED, "model": "m", "bogus": True})
        self.assertFalse(out.ok)
        self.assertIn("INVALID_JSON", body)
        self.assertNotIn(PLANTED, out.content)

    def test_a_key_cut_by_the_echo_limit_is_still_redacted(self) -> None:
        # The cut lands inside the planted value: redacting after it would leave a fragment.
        start = len('{"provider": "')
        raw = {"provider": "x" * (tools._INVALID_ECHO_CHARS - start - 10) + PLANTED, "bogus": True}
        out, body = self.call("list_tasks", raw)
        self.assertIn("INVALID_JSON", body)
        self.assertNotIn(PLANTED[:10], out.content)

    def test_bench_context_shows_key_set_booleans_only(self) -> None:
        ctx = self.ctx(key_flags=lambda: {"OPENROUTER_API_KEY": "sk-test-0000000000000000abcd",
                                          "ANTHROPIC_API_KEY": 0})
        out, body = self.call("get_bench_context", {}, ctx)
        self.assertTrue(out.ok)
        providers = {p["name"]: p for p in body["providers"]}
        self.assertIs(providers["openrouter"]["key_set"], True)
        self.assertIs(providers["anthropic"]["key_set"], False)
        self.assertEqual(set(providers["openrouter"]), {"name", "enabled", "key_set", "model_prefix"})
        for value in FAKE_KEYS.values():
            self.assertNotIn(value, out.content)
        self.assertNotIn("masked", out.content)
        self.assertEqual(body["docker"], {"ok": True, "ncpu": 8, "mem_gb": 16.0})
        self.assertEqual(body["sample"]["blocks"], [{"block": 1, "from": 1, "to": 10}, {"block": 2, "from": 11, "to": 20}])
        self.assertEqual(body["baseline_family"], "claude-code")
        self.assertEqual(body["ledger"]["trials"], len(self.world.trials))


class Results(ToolsCase):
    def test_oversized_results_are_cut_to_valid_json(self) -> None:
        big = mock.Mock(return_value=tools._Result({"blob": "é\"" * 30000}, "big"))
        with mock.patch.dict(tools._HANDLERS, {"list_designs": big}):
            out, body = self.call("list_designs", {})
        self.assertTrue(out.ok)
        self.assertIs(body["truncated"], True)
        self.assertIn("narrow the filter", body["hint"])
        self.assertTrue(body["head"].startswith('{"blob"'))

    def test_results_that_cannot_be_rendered_are_errors_not_crashes(self) -> None:
        deep: list[Any] = []
        for _ in range(5000):
            deep = [deep]
        for data, code in (({"mixed": {1, "a"}}, "TypeError"), ({"deep": deep}, "RecursionError")):
            with self.subTest(code):
                handler = mock.Mock(return_value=tools._Result(data, "rendered"))
                with mock.patch.dict(tools._HANDLERS, {"list_designs": handler}):
                    out, body = self.call("list_designs", {})
                self.assertFalse(out.ok)
                self.assertEqual(body["error"], code)
                self.assertIsNone(out.draft)

    def test_results_are_sorted_compact_json(self) -> None:
        out, body = self.call("get_cost_stats", {"family": "pi", "match": "family"})
        self.assertTrue(out.ok)
        self.assertEqual(out.content, json.dumps(body, sort_keys=True, ensure_ascii=False))
        self.assertEqual(body["ledger_fingerprint"], self.reader.fingerprint())
        (cell,) = [c for c in body["cells"] if c["model"] == GLM[1]]
        self.assertEqual(cell["family"], "pi")

    def test_refusals_explain_themselves(self) -> None:
        out, body = self.call("describe_harness", {"agent": "evil.module:Agent"})
        self.assertEqual(body["error"], "not_allowed")
        out, body = self.call("list_designs", {"design_id": "../../config"})
        self.assertEqual(body["error"], "not_found")
        out, body = self.call("list_tasks", {"provider": "openrouter"})
        self.assertEqual(body["error"], "bad_request")
        out, body = self.call("estimate_cost", {"spec": {**SPEC, "arms": [dict(SPEC["arms"][0], agent="evil:Agent")]}})
        self.assertEqual(body["error"], "invalid_design")

    def test_describe_harness_flags_a_mispriced_cap(self) -> None:
        out, body = self.call("describe_harness", {"agent": "claude-code", "provider": "openrouter",
                                                   "model": "z-ai/glm-5.3"})
        self.assertEqual(body["caps"], {"dollar": ["max_budget_usd"], "turn": ["max_turns"],
                                        "dollar_cap_misprices": True})
        self.assertTrue(any("max_budget_usd" in w for w in body["warnings"]))
        out, body = self.call("describe_harness", {"agent": MINI, "provider": "openrouter"})
        self.assertEqual((body["family"], body["kind"]), ("mini-swe-agent", "import_path"))
        self.assertFalse(body["caps"]["dollar_cap_misprices"])

    def test_list_tasks_counts_cost_valid_trials_per_family(self) -> None:
        out, body = self.call("list_tasks", {"provider": "openrouter", "model": "z-ai/glm-5.3", "limit": 3})
        first = body["tasks"][0]
        self.assertEqual((first["position"], first["block"], first["name"]), (1, 1, "task-00"))
        self.assertEqual(first["valid_trials_by_family"], {"claude-code": 1, "pi": 2, "terminus-2": 1})
        self.assertEqual(body["total"], len(SAMPLE))
        out, body = self.call("list_tasks", {"difficulty": "hard"})
        self.assertTrue(all(t["difficulty"] == "hard" for t in body["tasks"]))
        self.assertIsNone(body["tasks"][0]["valid_trials_by_family"])

    def test_saved_drafts_and_estimates_agree(self) -> None:
        ctx = self.ctx()
        saved, body = self.call("save_draft_design", SAVE, ctx)
        _, est = self.call("estimate_cost", {"design_id": body["design_id"]}, ctx)
        self.assertEqual(est["estimate"]["spec_hash"], body["spec_hash"])
        self.assertAlmostEqual(est["estimate"]["run"]["total"]["p50"], body["estimate"]["p50_usd"], places=4)
        _, listed = self.call("list_designs", {"design_id": body["design_id"]}, ctx)
        self.assertEqual(listed["version"]["author"], "assistant")
        self.assertEqual([v["version"] for v in listed["versions"]], [1])
