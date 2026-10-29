"""Designs, queue and assistant endpoints, and the human gate between them.

The State runs over temp files: designs and the queue are the real stores,
the run manager is a fake, Docker is a flag and the assistant talks to a
scripted FakeAnthropic, so nothing here can bill anyone. The checks that
matter most for money: a design only reaches the queue by its exact
spec_hash, one with problems (such as an assistant draft over the budget
limit) is refused there, and a LAUNCH-tab run cannot start beside an
active queue.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from typing import Any
from unittest import mock

from tests.fakes_anthropic import FakeAnthropic, Reply, text, tool_use
from tests.test_server import ServerCase

from bench import checks, keys, server
from bench.assistant import engine as engine_mod
from bench.assistant import tools
from bench.server import QUEUE_ACTIVE

DESIGN_YAML = """\
dataset: bench-test@1
providers:
  anthropic: {env_key: ANTHROPIC_API_KEY, model_prefix: anthropic/}
  openrouter: {env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}
modes:
  pilot: {n_tasks: 1}
run:
  budget_usd: 5
  n_concurrent: 1
baseline_arm: cc
arms:
  - {name: cc, agent: claude-code, provider: anthropic, model: claude-opus-5-5}
  - {name: pi, agent: pi, provider: anthropic, model: claude-opus-5-5}
designs:
  max_budget_usd: 20
"""
# The dev profile's shape: no provider uses the assistant's key.
NO_ANTHROPIC_YAML = """\
dataset: bench-test@1
providers:
  openrouter: {env_key: OPENROUTER_API_KEY, model_prefix: openrouter/}
modes:
  pilot: {n_tasks: 1}
run: {budget_usd: 5}
arms:
  - {name: pi-glm, agent: pi, provider: openrouter, model: z-ai/glm-5.3}
"""
SPEC: dict[str, Any] = {
    "tasks": ["task-a", "task-b"],
    "baseline_arm": "cc",
    "run": {"budget_usd": 4.0},
    "arms": [{"name": "cc", "agent": "claude-code", "provider": "anthropic", "model": "claude-opus-5-5"},
             {"name": "pi", "agent": "pi", "provider": "anthropic", "model": "claude-opus-5-5"}],
}
CID = "c-20261029-101500-ab12"

VERSION_KEYS = {"design_id", "version", "parent_version", "author", "conversation_id", "created_at", "title",
                "purpose", "rationale", "change_note", "spec", "spec_hash", "status", "approved_at", "runs", "check",
                "diff_from_parent"}
CHECK_KEYS = {"validation", "plan", "estimate", "checked_at", "stale"}
SUMMARY_KEYS = {"id", "title", "latest_version", "author", "status", "archived", "updated_at", "arms", "trials",
                "problems", "warnings", "expected_usd", "p90_usd", "check_stale", "runs", "conversation_id"}
VIEW_KEYS = {"state", "pause_reason", "on_failure", "global_cap_usd", "cap_since", "cap_spent_usd",
             "cap_remaining_usd", "committed_usd", "active", "waiting", "held", "history"}
STATUS_KEYS = {"enabled", "available", "reason", "model", "effort_default", "fallbacks", "key_set", "spend",
               "active_turn"}


def spec(**changes: Any) -> dict[str, Any]:
    out = copy.deepcopy(SPEC)
    for key, value in changes.items():
        if key in ("budget_usd", "parallel_arms", "n_concurrent", "effort", "per_trial_cap_usd"):
            out["run"][key] = value
        else:
            out[key] = value
    return out


def sse_blocks(body: str) -> list[dict[str, Any]]:
    """The fields of each server-sent event in a response body."""
    out = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        fields: dict[str, Any] = {}
        for line in block.split("\n"):
            if not line:
                continue
            if line.startswith(":"):
                fields.setdefault("comment", []).append(line[1:].strip())
                continue
            name, _, value = line.partition(":")
            fields[name] = value[1:] if value.startswith(" ") else value
        if fields:
            out.append(fields)
    return out


class DesignCase(ServerCase):
    YAML = DESIGN_YAML
    TASKS = ["task-a", "task-b", "task-c"]

    def setUp(self) -> None:
        self.fake = FakeAnthropic()
        super().setUp()

    def client_factory(self) -> Any:
        return self.fake

    def create(self, body_spec: dict[str, Any] | None = None, title: str = "Block 1 on Opus") -> dict[str, Any]:
        return self.ok(self.client.post("/api/designs", json={"title": title, "purpose": "Price pi",
                                                              "spec": body_spec or spec()}), 201)

    def queue_body(self, detail: dict[str, Any], **changes: Any) -> dict[str, Any]:
        return {"design_id": detail["design_id"], "version": detail["version"], "spec_hash": detail["spec_hash"],
                "position": "end", "start": False, **changes}


# Designs ----------------------------------------------------------------------------------


class Designs(DesignCase):
    def test_every_new_write_route_needs_the_cookie_and_x_bench(self) -> None:
        routes = [("post", "/api/designs"), ("post", "/api/designs/check"), ("post", "/api/designs/from-config"),
                  ("post", "/api/designs/from-run/r1"), ("post", "/api/designs/d-x-0000/versions"),
                  ("post", "/api/designs/d-x-0000/archive"), ("post", "/api/designs/d-x-0000/unarchive"),
                  ("post", "/api/queue/items"), ("delete", "/api/queue/items/q-0001"),
                  ("post", "/api/queue/items/q-0001/move"), ("post", "/api/queue/items/q-0001/retry"),
                  ("post", "/api/queue/start"), ("post", "/api/queue/pause"), ("put", "/api/queue/settings"),
                  ("post", "/api/assistant/conversations"), ("post", f"/api/assistant/conversations/{CID}/messages"),
                  ("post", f"/api/assistant/conversations/{CID}/stop"),
                  ("post", f"/api/assistant/conversations/{CID}/retry"),
                  ("post", f"/api/assistant/conversations/{CID}/continue"),
                  ("post", f"/api/assistant/conversations/{CID}/archive")]
        anonymous = self.make_client(self.state, signed_in=False)
        no_header = self.make_client(self.state, x_bench=False)
        for method, path in routes:
            with self.subTest(path=path, method=method):
                self.assertEqual(getattr(anonymous, method)(path).status_code, 401)
                self.assertEqual(getattr(no_header, method)(path).status_code, 403)
        for path in ("/api/designs", "/api/queue", "/api/assistant/status", "/api/assistant/conversations"):
            self.assertEqual(anonymous.get(path).status_code, 401)
        self.assertEqual(self.ok(self.client.get("/api/designs")), {"designs": []})

    def test_create_read_and_list(self) -> None:
        detail = self.create()
        self.assertLessEqual(VERSION_KEYS, set(detail))
        self.assertEqual((detail["version"], detail["author"], detail["status"]), (1, "user", "draft"))
        self.assertTrue(detail["spec_hash"].startswith("sha256:"))
        self.assertEqual(detail["spec"]["dataset"], "bench-test@1")
        self.assertEqual([a["family"] for a in detail["spec"]["arms"]], ["claude-code", "pi"])
        check = detail["check"]
        self.assertLessEqual(CHECK_KEYS, set(check))
        self.assertEqual(check["validation"]["status"], "valid", check["validation"])
        self.assertFalse(check["stale"])
        self.assertIsNotNone(check["estimate"], check.get("estimate_note"))
        self.assertEqual(check["plan"]["trials"], 4)

        design_id = detail["design_id"]
        listed = self.ok(self.client.get("/api/designs"))["designs"]
        self.assertEqual(len(listed), 1)
        self.assertLessEqual(SUMMARY_KEYS, set(listed[0]))
        self.assertEqual((listed[0]["id"], listed[0]["trials"], listed[0]["problems"]), (design_id, 4, 0))
        self.assertFalse(listed[0]["check_stale"])
        self.assertEqual(self.ok(self.client.get("/api/designs?filter=draft"))["designs"][0]["id"], design_id)
        self.assertEqual(self.ok(self.client.get("/api/designs?filter=approved"))["designs"], [])
        self.assertEqual(self.client.get("/api/designs?filter=mine").status_code, 400)

        whole = self.ok(self.client.get(f"/api/designs/{design_id}"))
        self.assertEqual(set(whole), {"meta", "current"})
        self.assertEqual(whole["current"]["spec_hash"], detail["spec_hash"])
        self.assertEqual(self.ok(self.client.get(f"/api/designs/{design_id}/versions/1"))["version"], 1)
        self.assertEqual(self.client.get(f"/api/designs/{design_id}/versions/7").status_code, 404)
        self.assertEqual(self.client.get("/api/designs/d-nothing-here-0000").status_code, 404)

    def test_new_versions_and_conflicts(self) -> None:
        v1 = self.create()
        design_id = v1["design_id"]
        body = {"base_version": 1, "title": v1["title"], "purpose": v1["purpose"], "rationale": "",
                "change_note": "Bigger budget", "spec": spec(budget_usd=6.0)}
        v2 = self.ok(self.client.post(f"/api/designs/{design_id}/versions", json=body), 201)
        self.assertEqual((v2["version"], v2["parent_version"]), (2, 1))
        self.assertIn({"path": "run.budget_usd", "from": 4.0, "to": 6.0}, v2["diff_from_parent"])
        stale = self.client.post(f"/api/designs/{design_id}/versions", json=body)
        self.assertEqual(stale.status_code, 409)
        self.assertIn("since you read version 1", stale.json()["detail"])
        same = self.client.post(f"/api/designs/{design_id}/versions", json={**body, "base_version": 2})
        self.assertEqual(same.status_code, 409)
        self.assertIn("Nothing changed", same.json()["detail"])
        # The version the editor loaded can be sent back as it is.
        loaded = {**body, "base_version": 2, "spec": {**v2["spec"], "attempts": 2}}
        v3 = self.ok(self.client.post(f"/api/designs/{design_id}/versions", json=loaded), 201)
        self.assertEqual(v3["spec"]["attempts"], 2)

    def test_check_writes_nothing_and_explains_a_half_filled_form(self) -> None:
        checked = self.ok(self.client.post("/api/designs/check", json={"spec": spec()}))
        self.assertLessEqual(CHECK_KEYS, set(checked))
        self.assertEqual(checked["validation"]["status"], "valid")
        self.assertIsNotNone(checked["estimate"])
        self.assertFalse(self.data.joinpath("designs").exists() and any(self.data.joinpath("designs").iterdir()))
        half = self.ok(self.client.post("/api/designs/check", json={"spec": {"tasks": ["task-a"], "arms": []}}))
        self.assertEqual(half["validation"]["status"], "invalid")
        self.assertTrue(any("run" in p for p in half["validation"]["problems"]), half["validation"]["problems"])
        over = self.ok(self.client.post("/api/designs/check", json={"spec": spec(budget_usd=50.0)}))
        self.assertEqual(over["validation"]["status"], "invalid")
        self.assertEqual(self.ok(self.client.get("/api/designs")), {"designs": []})

    def test_from_config_and_from_run(self) -> None:
        detail = self.ok(self.client.post("/api/designs/from-config",
                                          json={"mode": "pilot", "arms": None, "title": "Pilot"}), 201)
        self.assertEqual([a["name"] for a in detail["spec"]["arms"]], ["cc", "pi"])
        self.assertEqual(detail["spec"]["tasks"], ["task-a"])
        self.assertEqual(self.client.post("/api/designs/from-config", json={"mode": "nope"}).status_code, 400)

        self.manager.runs["r1"] = {"id": "r1", "status": "done", "spend_usd": 1.0, "budget_usd": 4.0,
                                   "spec": copy.deepcopy(detail["spec"]), "arms": []}
        again = self.ok(self.client.post("/api/designs/from-run/r1", json={"title": None}), 201)
        self.assertEqual(again["spec_hash"], detail["spec_hash"])
        self.assertEqual(again["title"], "Run again: r1")
        self.assertEqual(self.client.post("/api/designs/from-run/nope", json={}).status_code, 404)

    def test_archive_and_unarchive(self) -> None:
        design_id = self.create()["design_id"]
        meta = self.ok(self.client.post(f"/api/designs/{design_id}/archive"))["meta"]
        self.assertTrue(meta["archived"])
        self.assertEqual(self.ok(self.client.get("/api/designs"))["designs"], [])
        self.assertEqual(self.ok(self.client.get("/api/designs?filter=archived"))["designs"][0]["id"], design_id)
        self.assertFalse(self.ok(self.client.post(f"/api/designs/{design_id}/unarchive"))["meta"]["archived"])

    def test_a_design_with_a_run_lists_the_run(self) -> None:
        detail = self.create()
        self.manager.runs["r9"] = {"id": "r9", "status": "done", "spend_usd": 1.25, "budget_usd": 4.0,
                                   "design_ref": {"design_id": detail["design_id"], "version": 1}}
        self.manager.finish("r9", "done", 1.25)
        current = self.ok(self.client.get(f"/api/designs/{detail['design_id']}"))["current"]
        self.assertEqual(current["status"], "ran")
        self.assertEqual(current["runs"], [{"run_id": "r9", "status": "done", "spend_usd": 1.25,
                                            "budget_usd": 4.0}])


# Queue ------------------------------------------------------------------------------------


class Queue(DesignCase):
    def test_admission_by_hash_problems_and_docker(self) -> None:
        detail = self.create()
        wrong = self.client.post("/api/queue/items", json=self.queue_body(detail, spec_hash="sha256:" + "0" * 64))
        self.assertEqual(wrong.status_code, 409)
        self.docker = False
        down = self.client.post("/api/queue/items", json=self.queue_body(detail, start=True))
        self.assertEqual(down.status_code, 409)
        self.assertEqual(self.ok(self.client.get("/api/queue"))["waiting"], [])
        self.docker = True

        added = self.ok(self.client.post("/api/queue/items", json=self.queue_body(detail)), 201)
        self.assertEqual(set(added), {"item", "queue"})
        self.assertEqual(set(added["queue"]), VIEW_KEYS)
        self.assertEqual((added["item"]["status"], added["queue"]["state"]), ("waiting", "paused"))
        self.assertEqual(self.client.post("/api/queue/items", json=self.queue_body(detail)).status_code, 409)
        listed = self.ok(self.client.get("/api/designs"))["designs"][0]
        self.assertEqual(listed["status"], "queued")
        self.assertEqual(self.manager.started_specs, [])  # the worker is not running; nothing starts

    def test_a_version_with_problems_is_refused(self) -> None:
        detail = self.create(spec(tasks=["task-a", "task-zz"]))
        self.assertEqual(detail["check"]["validation"]["status"], "invalid")
        refused = self.client.post("/api/queue/items", json=self.queue_body(detail))
        self.assertEqual(refused.status_code, 400)
        self.assertIn("task-zz", refused.json()["detail"])
        self.assertEqual(self.ok(self.client.get("/api/queue"))["waiting"], [])

    def test_an_assistant_draft_over_the_budget_limit_is_saved_invalid_and_cannot_be_queued(self) -> None:
        ctx = self.state.tool_context(CID)
        outcome = tools.execute(ctx, "save_draft_design", {
            "title": "Too big", "purpose": "Everything", "rationale": "More is better", "change_note": "First",
            "spec": spec(budget_usd=50.0)})
        self.assertTrue(outcome.ok, outcome.content)
        draft = outcome.draft
        detail = self.ok(self.client.get(f"/api/designs/{draft['design_id']}"))["current"]
        self.assertEqual((detail["author"], detail["status"], detail["conversation_id"]), ("assistant", "draft", CID))
        self.assertEqual(detail["check"]["validation"]["status"], "invalid")
        self.assertTrue(any("max_budget_usd" in p or "budget" in p for p in detail["check"]["validation"]["problems"]))
        refused = self.client.post("/api/queue/items", json=self.queue_body(detail, start=True))
        self.assertEqual(refused.status_code, 400, refused.text)
        view = self.ok(self.client.get("/api/queue"))
        self.assertEqual((view["waiting"], view["state"]), ([], "paused"))
        self.assertEqual(self.manager.started_specs, [])

    def test_item_controls_and_settings(self) -> None:
        first = self.create(title="First")
        second = self.create(spec(tasks=["task-c"]), title="Second")
        a = self.ok(self.client.post("/api/queue/items", json=self.queue_body(first)), 201)["item"]
        b = self.ok(self.client.post("/api/queue/items", json=self.queue_body(second)), 201)["item"]
        moved = self.ok(self.client.post(f"/api/queue/items/{b['id']}/move", json={"to_index": 0}))
        self.assertEqual([i["id"] for i in moved["waiting"]], [b["id"], a["id"]])
        self.assertEqual(self.client.post(f"/api/queue/items/{b['id']}/move", json={"to_index": "0"}).status_code,
                         400)
        self.assertEqual(self.client.post(f"/api/queue/items/{a['id']}/retry").status_code, 409)  # not held
        self.assertEqual(self.client.delete("/api/queue/items/q-9999").status_code, 404)
        removed = self.ok(self.client.delete(f"/api/queue/items/{a['id']}"))
        self.assertEqual([i["id"] for i in removed["waiting"]], [b["id"]])

        settings_view = self.ok(self.client.put("/api/queue/settings",
                                                json={"global_cap_usd": 30, "on_failure": "skip"}))
        self.assertEqual((settings_view["global_cap_usd"], settings_view["on_failure"]), (30.0, "skip"))
        self.assertEqual(self.client.put("/api/queue/settings", json={"global_cap_usd": -1}).status_code, 400)
        self.assertEqual(self.client.put("/api/queue/settings", json={"colour": 1}).status_code, 400)
        cleared = self.ok(self.client.put("/api/queue/settings", json={"global_cap_usd": None, "reset_cap": True}))
        self.assertIsNone(cleared["global_cap_usd"])

        self.docker = False
        self.assertEqual(self.client.post("/api/queue/start").status_code, 409)
        self.docker = True
        self.assertEqual(self.ok(self.client.post("/api/queue/start"))["state"], "running")
        paused = self.ok(self.client.post("/api/queue/pause"))
        self.assertEqual((paused["state"], paused["pause_reason"]), ("paused", "Paused by you"))

    def test_post_runs_is_refused_while_the_queue_runs(self) -> None:
        detail = self.create()
        self.ok(self.client.post("/api/queue/items", json=self.queue_body(detail, start=True)), 201)
        self.assertEqual(self.ok(self.client.get("/api/queue"))["state"], "running")
        refused = self.client.post("/api/runs", json={"mode": "pilot"})
        self.assertEqual(refused.status_code, 409)
        self.assertEqual(refused.json()["detail"], QUEUE_ACTIVE)
        self.assertEqual(self.manager.started, [])
        self.ok(self.client.post("/api/queue/pause"))
        self.ok(self.client.post("/api/runs", json={"mode": "pilot"}))
        self.assertEqual(self.manager.started, [("pilot", ["cc", "pi"])])

    def test_state_carries_the_queue_and_assistant_summaries(self) -> None:
        body = self.ok(self.client.get("/api/state"))
        self.assertEqual(set(body["queue"]), {"state", "pause_reason", "active_item", "waiting", "held",
                                              "cap_remaining_usd"})
        self.assertEqual(set(body["assistant"]), {"enabled", "available", "reason", "today_usd", "daily_cap_usd",
                                                  "active_turn"})
        self.assertEqual((body["queue"]["state"], body["assistant"]["available"]), ("paused", True))
        self.assertEqual(body["assistant"]["daily_cap_usd"], 5.0)


# Assistant ------------------------------------------------------------------------------------


class Assistant(DesignCase):
    def setUp(self) -> None:
        super().setUp()
        self.patch(engine_mod, "STREAM_SECONDS", 0.4)

    def conversation(self) -> str:
        meta = self.ok(self.client.post("/api/assistant/conversations", json={"title": "Plan"}), 201)
        self.assertEqual(meta["effort"], "medium")
        return meta["id"]

    def finish_turn(self) -> None:
        self.assertTrue(self.state.assistant.wait(60), "the turn did not finish")

    def test_status_and_conversation_list(self) -> None:
        status = self.ok(self.client.get("/api/assistant/status"))
        self.assertLessEqual(STATUS_KEYS, set(status))
        self.assertEqual((status["available"], status["model"], status["key_set"]), (True, "claude-opus-5-5", True))
        self.assertNotIn(os.environ["ANTHROPIC_API_KEY"], str(status))
        cid = self.conversation()
        listed = self.ok(self.client.get("/api/assistant/conversations"))["conversations"]
        self.assertEqual([c["id"] for c in listed], [cid])
        self.assertEqual(self.client.post("/api/assistant/conversations", json={"effort": "huge"}).status_code, 400)
        usage = self.ok(self.client.get("/api/assistant/usage?days=7"))
        self.assertEqual((len(usage["days"]), usage["total_usd"]), (7, 0))
        self.assertEqual(self.client.get("/api/assistant/usage?days=0").status_code, 400)
        self.assertEqual(self.client.get("/api/assistant/conversations/c-20990101-000000-0000").status_code, 404)

    def test_a_message_streams_and_resumes_without_duplicates(self) -> None:
        self.fake.script.append(Reply([text("Pi costs about a third of Claude Code.")]))
        cid = self.conversation()
        accepted = self.client.post(f"/api/assistant/conversations/{cid}/messages",
                                    json={"text": "What does pi cost?", "attach_design": None})
        self.assertEqual(accepted.status_code, 202, accepted.text)
        self.assertEqual(set(accepted.json()), {"turn", "accepted_seq"})
        self.finish_turn()

        stream = self.client.get(f"/api/assistant/conversations/{cid}/stream?after=0")
        self.assertEqual(stream.status_code, 200)
        self.assertTrue(stream.headers["content-type"].startswith("text/event-stream"))
        blocks = sse_blocks(stream.text)
        batches = [b for b in blocks if b.get("event") == "batch"]
        self.assertTrue(batches)
        events = [e for b in batches for e in json.loads(b["data"])]
        seqs = [e["seq"] for e in events]
        self.assertEqual(seqs, sorted(set(seqs)))
        self.assertEqual([int(b["id"]) for b in batches], sorted(int(b["id"]) for b in batches))
        self.assertEqual(int(batches[-1]["id"]), seqs[-1])
        kinds = {e["type"] for e in events}
        self.assertLessEqual({"user_message", "turn_started", "assistant_text", "turn_finished"}, kinds)
        self.assertIn("bye", blocks[-1].get("comment", []))
        self.assertEqual(blocks[-1].get("retry"), "1000")

        resumed = self.client.get(f"/api/assistant/conversations/{cid}/stream?after=0",
                                  headers={"Last-Event-ID": batches[-1]["id"]})
        self.assertEqual([b for b in sse_blocks(resumed.text) if b.get("event") == "batch"], [])

        polled = self.ok(self.client.get(f"/api/assistant/conversations/{cid}?after=0"))
        self.assertEqual(set(polled), {"conversation", "events", "next_seq", "active_turn"})
        durable = [e["seq"] for e in polled["events"]]
        self.assertLessEqual(set(durable), set(seqs))
        later = self.ok(self.client.get(f"/api/assistant/conversations/{cid}?after={seqs[0]}"))["events"]
        self.assertNotIn(seqs[0], [e["seq"] for e in later])
        self.assertEqual(self.client.get("/api/assistant/conversations/c-20990101-000000-0000/stream").status_code,
                         404)

    def test_a_second_message_during_a_turn_is_409(self) -> None:
        gate = threading.Event()
        self.fake.script.append(Reply([text("Thinking it over.")], on_event=lambda i: gate.wait(10) if i == 0 else None))
        cid = self.conversation()
        try:
            first = self.client.post(f"/api/assistant/conversations/{cid}/messages", json={"text": "Plan block 1"})
            self.assertEqual(first.status_code, 202)
            second = self.client.post(f"/api/assistant/conversations/{cid}/messages", json={"text": "And block 2"})
            self.assertEqual(second.status_code, 409)
            self.assertIn("already answering", second.json()["detail"])
            active = self.ok(self.client.get("/api/state"))["assistant"]["active_turn"]
            self.assertEqual(active["conversation_id"], cid)
            self.ok(self.client.post(f"/api/assistant/conversations/{cid}/stop"))
        finally:
            gate.set()
        self.finish_turn()
        self.assertIsNone(self.ok(self.client.get("/api/assistant/status"))["active_turn"])

    def test_key_like_text_is_refused_and_nothing_is_stored(self) -> None:
        cid = self.conversation()
        leaked = self.client.post(f"/api/assistant/conversations/{cid}/messages",
                                  json={"text": f"my key is {os.environ['ANTHROPIC_API_KEY']}"})
        self.assertEqual(leaked.status_code, 400)
        self.assertIn("API key", leaked.json()["detail"])
        self.assertEqual(self.ok(self.client.get(f"/api/assistant/conversations/{cid}"))["events"], [])
        self.assertEqual(self.fake.calls, [])

    def test_without_a_key_the_assistant_is_unavailable(self) -> None:
        cid = self.conversation()
        os.environ.pop("ANTHROPIC_API_KEY", None)
        status = self.ok(self.client.get("/api/assistant/status"))
        self.assertFalse(status["available"])
        self.assertFalse(status["key_set"])
        self.assertIn("ANTHROPIC_API_KEY", status["reason"])
        refused = self.client.post(f"/api/assistant/conversations/{cid}/messages", json={"text": "Hello"})
        self.assertEqual(refused.status_code, 409)
        self.assertFalse(self.ok(self.client.get("/api/state"))["assistant"]["available"])
        self.assertEqual(self.fake.calls, [])

    def test_a_tool_round_runs_on_the_servers_read_only_context(self) -> None:
        self.fake.script.extend([
            Reply([tool_use("toolu_1", "get_bench_context", {})], stop_reason="tool_use"),
            Reply([text("You have two arms on Opus 5.5.")]),
        ])
        cid = self.conversation()
        self.ok(self.client.post(f"/api/assistant/conversations/{cid}/messages", json={"text": "What is set up?"}),
                202)
        self.finish_turn()
        events = self.ok(self.client.get(f"/api/assistant/conversations/{cid}"))["events"]
        result = next(e for e in events if e["type"] == "tool_result")
        self.assertTrue(result["ok"], result)
        sent = self.fake.calls[1]["messages"][-1]["content"][0]
        self.assertEqual(sent["tool_use_id"], "toolu_1")
        self.assertNotIn("is_error", sent)
        self.assertIn('"key_set": true', sent["content"])
        self.assertNotIn(os.environ["ANTHROPIC_API_KEY"], sent["content"])
        self.assertNotIn(os.environ["OPENROUTER_API_KEY"], sent["content"])

    def test_continue_and_archive(self) -> None:
        cid = self.conversation()
        fresh = self.ok(self.client.post(f"/api/assistant/conversations/{cid}/continue"), 201)
        self.assertNotEqual(fresh["id"], cid)
        self.assertEqual(self.ok(self.client.post(f"/api/assistant/conversations/{cid}/archive")), {"ok": True})
        retry = self.client.post(f"/api/assistant/conversations/{fresh['id']}/retry")
        self.assertEqual(retry.status_code, 409)
        self.assertEqual(self.fake.calls, [])


class AssistantKeyWithoutProvider(DesignCase):
    YAML = NO_ANTHROPIC_YAML

    def test_setup_lists_the_assistant_key_and_tests_it_with_the_anthropic_tester(self) -> None:
        rows = self.ok(self.client.get("/api/state"))["keys"]
        row = next(r for r in rows if r["env_key"] == "ANTHROPIC_API_KEY")
        self.assertEqual(row["provider"], "assistant")
        calls: list[tuple[str, str]] = []

        def fake_test(cfg: Any, provider: str, value: str) -> dict[str, Any]:
            calls.append((provider, value))
            return {"ok": True, "detail": "Key accepted"}

        with mock.patch.object(checks, "test_key", fake_test):
            result = self.ok(self.client.post("/api/keys/ANTHROPIC_API_KEY/test"))
        self.assertEqual(result, {"ok": True, "detail": "Key accepted"})
        self.assertEqual(calls, [("anthropic", os.environ["ANTHROPIC_API_KEY"])])
        self.assertEqual(self.client.post("/api/keys/FIREWORKS_API_KEY/test").status_code, 404)
        self.assertEqual(keys.ASSISTANT_KEY, "ANTHROPIC_API_KEY")


class AssistantUnloadable(DesignCase):
    """The dashboard serves everything else when the assistant cannot load."""

    def make_state(self, *args: Any, **kwargs: Any) -> Any:
        self.notes: list[str] = []
        # Every assistant module is imported already (above), so restoring
        # sys.modules afterwards drops nothing.
        with (mock.patch.dict("sys.modules", {"bench.assistant.engine": None}),
              mock.patch.object(server, "_note", self.notes.append)):
            return super().make_state(*args, **kwargs)

    def test_routes_answer_409_and_state_still_loads(self) -> None:
        self.assertIsNone(self.state.assistant)
        self.assertEqual(len(self.notes), 1)
        status = self.ok(self.client.get("/api/assistant/status"))
        self.assertFalse(status["available"])
        self.assertIn("could not load", status["reason"])
        self.assertEqual(self.client.post("/api/assistant/conversations", json={}).status_code, 409)
        body = self.ok(self.client.get("/api/state"))
        self.assertFalse(body["assistant"]["available"])
        self.assertEqual(self.ok(self.client.get("/api/designs")), {"designs": []})


if __name__ == "__main__":
    import unittest
    unittest.main()
