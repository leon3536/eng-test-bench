"""Builders for ledger tests: synthetic Harbor trials and job layouts.

The pilot fixtures under tests/fixtures/ledger/dev are trimmed copies of the
two real dev pilots (run.json, job-level and trial-level config, lock and
result JSON; no agent logs, artifacts, rollout_details or env maps). Everything
else is synthesised here in the shape Harbor 0.25.0 writes, so tests can
cover cases the pilots never hit without touching real data.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tests import support  # noqa: F401  (isolates the environment before bench is imported)

from bench.extract import ScanRoot

FIXTURE_DEV = Path(__file__).resolve().parent / "fixtures" / "ledger" / "dev"
PILOT_1 = "20261028-222401-pilot"
PILOT_2 = "20261028-222952-pilot"

GOLDEN_COUNTS = {"setup_failure": 10, "misconfigured_cap": 2, "passed": 1, "failed": 2, "incomplete": 1}
GOLDEN = {
    (PILOT_1, "claude-code-deepseek"): "setup_failure",
    (PILOT_1, "claude-code-glm"): "setup_failure",
    (PILOT_1, "mini-swe-agent-deepseek"): "setup_failure",
    (PILOT_1, "mini-swe-agent-glm"): "setup_failure",
    (PILOT_1, "terminus-2-glm"): "setup_failure",
    (PILOT_1, "strands-glm"): "setup_failure",
    (PILOT_1, "pi-glm"): "incomplete",
    (PILOT_2, "claude-code-deepseek"): "misconfigured_cap",
    (PILOT_2, "claude-code-glm"): "misconfigured_cap",
    (PILOT_2, "mini-swe-agent-deepseek"): "setup_failure",
    (PILOT_2, "mini-swe-agent-glm"): "setup_failure",
    (PILOT_2, "strands-deepseek"): "setup_failure",
    (PILOT_2, "strands-glm"): "setup_failure",
    (PILOT_2, "pi-deepseek"): "passed",
    (PILOT_2, "pi-glm"): "failed",
    (PILOT_2, "terminus-2-deepseek"): "failed",
}
PILOT_2_SNAPSHOT = "local:terminal-bench@2.0@2026-10-29T05:29:58+00:00"

_T0 = datetime(2026, 10, 29, 5, 30, tzinfo=timezone.utc)


def fixture_roots() -> list[ScanRoot]:
    return [ScanRoot(FIXTURE_DEV / "jobs", "dev", FIXTURE_DEV / "runs")]


def fixture_trial_dirs() -> list[Path]:
    """Every trial folder of both pilots, sorted."""
    return sorted(p for p in (FIXTURE_DEV / "jobs").glob("*/*/*") if p.is_dir())


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8", newline="\n")
    return path


def _z(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _timing(start: datetime, seconds: float) -> dict[str, str]:
    return {"started_at": _z(start), "finished_at": _z(start + timedelta(seconds=seconds))}


def make_trial(trial_dir: Path, *, family: str = "pi", model_name: str = "openrouter/z-ai/glm-5.3",
               kwargs: Mapping[str, Any] | None = None, tokens: tuple[int, int, int] | None = (100000, 80000, 5000),
               usage: Mapping[str, Mapping[str, Any]] | None = None, cost: float | None = None,
               exception_type: str | None = None, message: str = "", agent_started: bool = True,
               exec_sec: float = 300.0, reward: float | None = 1.0, lock_digest: str | None = "sha256:" + "a" * 64,
               task_type: str = "local", import_path: str | None = None, steps: Sequence[Mapping] | None = None,
               job_id: str = "11111111-1111-4111-8111-111111111111", trial_name: str | None = None,
               task: str = "task-a", started_at: datetime | None = None, version: str = "1.0.0",
               finished: bool = True, agent_name: str | None = None, env: Mapping[str, str] | None = None,
               task_checksum: str = "c" * 64, environment: Mapping[str, Any] | None = None,
               prebake: Mapping[str, Any] | None = None, setup_sec: float = 55.0) -> Path:
    """Write config.json, lock.json and (when finished) result.json for one
    trial in the shape Harbor 0.25.0 writes them. environment is the
    trial's environment config (--env/--ek); prebake, the prebake.json the
    pre-baked environment class writes at environment start."""
    trial_dir = Path(trial_dir)
    trial_name = trial_name or trial_dir.name
    kwargs = dict(kwargs if kwargs is not None else {"reasoning_effort": "medium", "max_turns": 150})
    agent = {"name": None if import_path else (agent_name or family), "import_path": import_path,
             "model_name": model_name, "kwargs": kwargs}
    if env:
        agent["env"] = dict(env)
    if task_type == "git":
        task_cfg = {"path": task, "git_url": "https://example.invalid/tasks.git",
                    "git_commit_id": "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c", "source": "bench-tests"}
    elif task_type == "package":
        task_cfg = {"name": f"org/{task}", "ref": "v1", "source": "bench-tests"}
    else:
        task_cfg = {"path": f"C:\\data\\snapshot\\{task}", "source": "bench-tests"}
    config = {"task": task_cfg, "trial_name": trial_name, "trials_dir": str(trial_dir.parent),
              "agent": {k: v for k, v in agent.items() if v is not None}, "job_id": job_id}
    if environment:
        config["environment"] = dict(environment)
    write_json(trial_dir / "config.json", config)
    if prebake is not None:
        write_json(trial_dir / "prebake.json", prebake)
    if lock_digest is not None:
        lock_task = {"name": task_cfg.get("name") or task, "type": task_type, "digest": lock_digest,
                     "source": "bench-tests"}
        if task_type == "git":
            lock_task.update(git_url=task_cfg["git_url"], git_commit_id=task_cfg["git_commit_id"], path=task)
        elif task_type == "local":
            lock_task["path"] = task_cfg["path"]
        write_json(trial_dir / "lock.json", {"schema_version": 2, "task": lock_task, "agent": config["agent"],
                                             "environment": {"type": "docker", **(environment or {})}})
    if not finished:
        return trial_dir

    start = started_at or _T0
    agent_result: dict[str, Any] | None
    if tokens is None and usage is None:
        agent_result = {"metadata": {"output": ""}}
    else:
        n_in, n_cache, n_out = tokens if tokens is not None else (None, None, None)
        agent_result = {"n_input_tokens": n_in, "n_cache_tokens": n_cache, "n_output_tokens": n_out,
                        "cost_usd": cost, "model_usage": dict(usage) if usage is not None else None,
                        "rollout_details": [{"prompt_token_ids": [[1, 2, 3]]}], "metadata": None}
    if steps is not None:
        agent_result = None
    exception = None
    if exception_type:
        exception = {"exception_type": exception_type, "exception_message": message,
                     "exception_traceback": "Traceback (most recent call last):\n" + "x" * 6000,
                     "occurred_at": _z(start + timedelta(seconds=60 + exec_sec))}
    result = {
        "id": str(uuid.uuid4()), "task_name": task, "trial_name": trial_name,
        "trial_uri": (trial_dir.as_uri() if trial_dir.is_absolute() else f"file:///{trial_dir.as_posix()}"),
        "task_id": {"path": task_cfg.get("path") or task}, "source": "bench-tests",
        "task_checksum": task_checksum,
        "config": {**config, "timeout_multiplier": 1.0, "agent_timeout_multiplier": None,
                   "agent": {**agent, "override_timeout_sec": None, "skills": [], "mcp_servers": []},
                   "environment": {"type": "docker", "override_cpus": None, **(environment or {})},
                   "extra_instructions": []},
        "agent_info": {"name": family, "version": version,
                       "model_info": {"name": model_name.split("/", 1)[-1], "provider": model_name.split("/")[0]}},
        "agent_result": agent_result,
        "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
        "exception_info": exception,
        "started_at": _z(start),
        "finished_at": _z(start + timedelta(seconds=90 + exec_sec)),
        "environment_setup": _timing(start, 5),
        "agent_setup": _timing(start + timedelta(seconds=5), setup_sec),
        "agent_execution": _timing(start + timedelta(seconds=60), exec_sec) if agent_started else {},
        "verifier": _timing(start + timedelta(seconds=61 + exec_sec), 20),
        "step_results": [dict(s) for s in steps] if steps is not None else None,
    }
    write_json(trial_dir / "result.json", result)
    return trial_dir


def make_job(job_dir: Path, *, harbor: str = "0.25.0", n_concurrent: int = 2) -> Path:
    """Job-level config.json and lock.json, as Harbor writes before any trial."""
    write_json(job_dir / "config.json", {"job_name": job_dir.name, "jobs_dir": str(job_dir.parent),
                                         "n_concurrent_trials": n_concurrent})
    write_json(job_dir / "lock.json", {"schema_version": 3, "created_at": _z(_T0),
                                       "harbor": {"version": harbor, "is_editable": False},
                                       "n_concurrent_trials": n_concurrent, "trials": []})
    return job_dir


def run_dict(run_id: str, arms: Sequence[Mapping[str, Any]], *, status: str = "done", resolved: bool = True,
             dataset: str = "terminal-bench@2.0", downloaded_at: str | None = "2026-10-29T05:29:58+00:00",
             profile: str = "default") -> dict[str, Any]:
    """A run.json shaped like the run manager's (contract C7): spec, design_ref
    and resolved_arms, plus the older plan and config blocks."""
    prefixes = {"openrouter": "openrouter/", "anthropic": "anthropic/"}
    resolved_arms = []
    for a in arms:
        harbor_model = prefixes[a["provider"]] + a["model"]
        resolved_arms.append({
            "name": a["name"], "agent": a["agent"], "family": a.get("family") or a["agent"],
            "kind": "import_path" if ":" in a["agent"] else "name", "provider": a["provider"],
            "model": a["model"], "harbor_model": harbor_model,
            "effort": {"setting": "inherit", "value": "medium", "label": "medium"},
            "kwargs": [{"name": "reasoning_effort", "value": "medium", "source": "effort"}],
            "options_fingerprint": "0" * 16, "tasks": ["task-a"], "trials": 1, "command": "harbor run ...",
        })
    run: dict[str, Any] = {
        "id": run_id, "profile": profile, "source": "design", "mode": None,
        "created_at": "2026-10-29T05:29:52+00:00", "budget_usd": 5.0, "tasks": ["task-a"], "status": status,
        "spec": {"dataset": dataset, "baseline_arm": arms[0]["name"] if arms else None},
        "spec_hash": "sha256:" + "0" * 64,
        "design_ref": {"design_id": "d1", "version": 1, "author": "user", "conversation_id": None,
                       "queue_item": None},
        "arms": [{"name": a["name"], "status": "done" if status != "running" else "running"} for a in arms],
        "config": {"dataset": dataset, "providers": {p: {"env_key": f"{p.upper()}_API_KEY", "model_prefix": v}
                                                     for p, v in prefixes.items()},
                   "baseline_arm": arms[0]["name"] if arms else None,
                   "arms": [{k: a[k] for k in ("name", "agent", "provider", "model")} for a in arms]},
        "plan": {"arms": [{"name": a["name"], "agent": a["agent"], "provider": a["provider"],
                           "model": prefixes[a["provider"]] + a["model"], "effort": "medium"} for a in arms]},
        "harbor_version": "0.25.0",
    }
    if resolved:
        run["resolved_arms"] = resolved_arms
    if downloaded_at:
        run["dataset"] = {"dataset": dataset, "downloaded_at": downloaded_at, "harbor": "0.25.0"}
    return run


def make_bench_run(data_root: Path, run_id: str, arms: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """A bench data folder: runs/<run_id>/run.json and jobs/<run_id>/<arm>/
    with job-level files. Trials are added with make_trial."""
    run = run_dict(run_id, arms, **kwargs)
    write_json(data_root / "runs" / run_id / "run.json", run)
    for a in arms:
        make_job(data_root / "jobs" / run_id / a["name"])
    return run


def make_external_job(root: Path, job_name: str, *, finished: bool = True) -> Path:
    """A plain `harbor run` job: config.json with job_name, no run.json."""
    job_dir = make_job(root / job_name)
    write_json(job_dir / "result.json", {"id": str(uuid.uuid4()), "started_at": "2026-10-28T22:30:18.375875",
                                         "finished_at": "2026-10-28T22:37:48.280730" if finished else None})
    return job_dir


def _jsonl(path: Path, events: Sequence[Mapping[str, Any]], cut: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(e) + "\n" for e in events)
    if cut:
        text += '{"type": "message", "message": {"role": "assistant", "usage": {"inp'  # the kill landed mid-write
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def write_agent_log(trial_dir: Path, harness: str, calls: Sequence[tuple[int, int, int]] = ((1000, 600, 100),),
                    *, cut: bool = True) -> Path:
    """The usage log a harness keeps in agent/ as it runs, in the shape the
    Harbor 0.25.0 adapters read it, as a kill leaves it (cut: a last line
    written halfway). calls are (prompt tokens with cache reads, cache
    reads, output) per completed model call; each costs $0.001 where the
    harness logs a cost."""
    agent = Path(trial_dir) / "agent"
    if harness == "pi":
        events: list[dict[str, Any]] = [
            {"type": "session", "version": 3, "id": "s1"},
            {"type": "model_change", "id": "m1", "provider": "openrouter", "modelId": "z-ai/glm-5.3"},
            {"type": "message", "id": "e1", "message": {"role": "system", "content": "..."}},
            {"type": "message", "id": "e2", "message": {"role": "user", "content": "the task"}}]
        for i, (prompt, cache, out) in enumerate(calls):
            usage = {"input": prompt - cache, "output": out, "cacheRead": cache, "cacheWrite": 0,
                     "totalTokens": prompt + out, "cost": {"total": 0.001}}
            events.append({"type": "message", "id": f"a{i}", "message": {"role": "assistant", "usage": usage}})
            events.append({"type": "message", "id": f"t{i}", "message": {"role": "toolResult", "content": "ok"}})
        return _jsonl(agent / "pi" / "sessions" / "2026-10-29T13-14-34-471Z_s1.jsonl", events, cut)
    if harness == "claude-code":
        events = [{"type": "user", "uuid": "u0", "message": {"role": "user", "content": "the task"}}]
        for i, (prompt, cache, out) in enumerate(calls):
            usage = {"input_tokens": prompt - cache, "cache_read_input_tokens": cache,
                     "cache_creation_input_tokens": 0, "output_tokens": out}
            message = {"id": f"msg_{i}", "model": "z-ai/glm-5.3", "role": "assistant", "usage": usage}
            # One API response is logged once per content block, with the same id.
            events += [{"type": "assistant", "uuid": f"a{i}-{block}", "message": message} for block in (1, 2)]
        events.append(dict(events[-1]))  # a duplicated event uuid
        return _jsonl(agent / "sessions" / "projects" / "-app" / "s1.jsonl", events, cut)
    if harness == "strands":
        events = [{"kind": "message", "sequence": 0, "message": {"role": "user", "content": []}}]
        for i, (prompt, cache, out) in enumerate(calls):
            usage = {"inputTokens": prompt, "outputTokens": out, "cacheReadInputTokens": cache,
                     "totalTokens": prompt + out}  # cache inside inputTokens, as on OpenRouter
            events.append({"kind": "message", "sequence": i + 1,
                           "message": {"role": "assistant", "metadata": {"usage": usage}}})
        return _jsonl(agent / "strands-events.jsonl", events, cut)
    if harness == "mini-swe-agent":
        messages: list[dict[str, Any]] = [{"role": "system", "content": "..."}, {"role": "user", "content": "task"}]
        for prompt, cache, out in calls:
            usage = {"prompt_tokens": prompt, "completion_tokens": out,
                     "prompt_tokens_details": {"cached_tokens": cache}, "cost": 0.001}
            messages.append({"role": "assistant", "content": "...", "extra": {"response": {"usage": usage}}})
        return write_json(agent / "mini-swe-agent.trajectory.json",
                          {"messages": messages, "info": {"model_stats": {"instance_cost": 0.0,
                                                                          "api_calls": len(calls)}}})
    if harness == "terminus-2":
        # Rewritten after every episode, from the host: the main chat's totals
        # so far, which its agent steps' metrics add up to.
        totals = {"total_prompt_tokens": sum(c[0] for c in calls), "total_cached_tokens": sum(c[1] for c in calls),
                  "total_completion_tokens": sum(c[2] for c in calls), "total_cost_usd": 0.001 * len(calls)}
        steps: list[dict[str, Any]] = [{"step_id": 1, "source": "user", "message": "the task"}]
        for i, (prompt, cache, out) in enumerate(calls):
            steps.append({"step_id": i + 2, "source": "agent", "message": "...", "metrics": {
                "prompt_tokens": prompt, "completion_tokens": out, "cached_tokens": cache, "cost_usd": 0.001}})
        return write_json(agent / "trajectory.json", {"schema_version": "ATIF-v1.6", "steps": steps,
                                                       "final_metrics": totals})
    raise ValueError(f"no agent log recipe for {harness}")


def make_format_error_trial(trial_dir: Path, calls: Sequence[tuple[int, int, int]] = (
        (1000, 0, 100), (3000, 2400, 250), (3500, 3000, 50)), *, format_error: int = 1,
        instance_cost: float = 0.04) -> Path:
    """A finished mini-swe-agent trial on Anthropic whose model once replied
    without a tool call, as Harbor 0.25.0 records it (the Opus pilot's
    adaptive-rejection-sampler__saBMoJj, message 18). mini-swe-agent logs that
    reply on the 'user' message that answers it, with interrupt_type
    FormatError. Harbor's totals count every message's usage; model_usage
    counts the assistant steps only, since the ATIF conversion files that
    message as an observation, each step's cost a share of instance_cost by
    completion tokens. calls are as in write_agent_log."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": "..."}, {"role": "user", "content": "task"}]
    for i, (prompt, cache, out) in enumerate(calls):
        usage = {"prompt_tokens": prompt, "completion_tokens": out, "prompt_tokens_details": {"cached_tokens": cache},
                 "cache_read_input_tokens": cache, "cache_creation_input_tokens": prompt - cache}
        if i == format_error:
            messages.append({"role": "user", "content": "Tool call error: No tool calls found in the response.",
                             "extra": {"interrupt_type": "FormatError", "response": {"usage": usage}}})
        else:
            messages += [{"role": "assistant", "content": "...", "extra": {"response": {"usage": usage}}},
                         {"role": "tool", "content": "ok", "extra": {"returncode": 0}}]
    messages.append({"role": "exit", "content": "", "extra": {"exit_status": "Submitted"}})
    write_json(Path(trial_dir) / "agent" / "mini-swe-agent.trajectory.json",
               {"messages": messages, "info": {"model_stats": {"instance_cost": instance_cost,
                                                               "api_calls": len(calls)}}})
    steps = [c for i, c in enumerate(calls) if i != format_error]
    total_out = sum(c[2] for c in calls)
    model_usage = {"n_input_tokens": sum(c[0] for c in steps), "n_cache_tokens": sum(c[1] for c in steps),
                   "n_output_tokens": sum(c[2] for c in steps),
                   "cost_usd": sum(instance_cost * c[2] / total_out for c in steps)}
    return make_trial(trial_dir, family="mini-swe-agent", model_name="anthropic/claude-opus-5-5",
                      tokens=(sum(c[0] for c in calls), sum(c[1] for c in calls), total_out),
                      usage={"anthropic/claude-opus-5-5": model_usage}, cost=instance_cost)


# Anthropic's Opus 5.5 rates, $ per million tokens, as LiteLLM, Claude Code and pi price them.
OPUS_RATES = {"input": 4.0, "read": 0.2, "write_5m": 5.0, "write_1h": 8.0, "output": 20.0}
OPUS_PRICING_YAML = ("anthropic:\n"
                     "  claude-opus-5-5: {input: 4.00, output: 20.00, cache_read: 0.20, cache_write_5m: 5.00, "
                     "cache_write_1h: 8.00}\n"
                     "  claude-opus-5: {input: 5.00, output: 25.00, cache_read: 0.50, cache_write_5m: 6.25, "
                     "cache_write_1h: 10.00}\n"
                     "openrouter:\n  glm-5.3: {input: 0.07, output: 1.40, cache_read: 0.069}\n")


def call_cost(call: Sequence[int], rates: Mapping[str, float] = OPUS_RATES) -> float:
    """What Anthropic bills one call of write_cache_log's shape."""
    fresh, read, w5, w1, out = call
    return (fresh * rates["input"] + read * rates["read"] + w5 * rates["write_5m"] + w1 * rates["write_1h"]
            + out * rates["output"]) / 1e6


def write_cache_log(trial_dir: Path, harness: str, calls: Sequence[Sequence[int]], *,
                    model: str = "claude-opus-5-5", rates: Mapping[str, float] = OPUS_RATES,
                    ) -> tuple[int, int, int, float]:
    """A finished trial's own usage log, Anthropic-style, with its cache writes
    where the harness keeps them: calls are (fresh input, cache reads,
    5-minute writes, 1-hour writes, output). Returns Harbor's totals for it
    (n_input with reads and writes folded in, n_cache, n_output, cost)."""
    agent = Path(trial_dir) / "agent"
    costs = [call_cost(c, rates) for c in calls]
    if harness == "claude-code":
        events: list[dict[str, Any]] = []
        for i, (fresh, read, w5, w1, out) in enumerate(calls):
            usage = {"input_tokens": fresh, "cache_read_input_tokens": read, "cache_creation_input_tokens": w5 + w1,
                     "cache_creation": {"ephemeral_5m_input_tokens": w5, "ephemeral_1h_input_tokens": w1},
                     "output_tokens": out}
            message = {"id": f"msg_{i}", "model": model, "role": "assistant", "usage": usage}
            events += [{"type": "assistant", "uuid": f"a{i}-{block}", "message": message} for block in (1, 2)]
        _jsonl(agent / "sessions" / "projects" / "-app" / "s1.jsonl", events, cut=False)
    elif harness == "pi":
        events = [{"type": "session", "version": 3, "id": "s1"}]
        for i, ((fresh, read, w5, w1, out), cost) in enumerate(zip(calls, costs)):
            usage = {"input": fresh, "output": out, "cacheRead": read, "cacheWrite": w5 + w1, "cacheWrite1h": w1,
                     "totalTokens": fresh + read + w5 + w1 + out, "cost": {"total": cost}}
            events.append({"type": "message", "id": f"a{i}", "message": {"role": "assistant", "model": model,
                                                                           "usage": usage}})
        _jsonl(agent / "pi" / "sessions" / "2026-10-29T13-14-34-471Z_s1.jsonl", events, cut=False)
    elif harness == "strands":
        events = []
        for i, (fresh, read, w5, w1, out) in enumerate(calls):
            usage = {"inputTokens": fresh, "outputTokens": out, "cacheReadInputTokens": read,
                     "cacheWriteInputTokens": w5 + w1, "totalTokens": fresh + out + read + w5 + w1}
            events.append({"kind": "message", "sequence": i, "message": {"role": "assistant",
                                                                         "metadata": {"usage": usage}}})
        _jsonl(agent / "strands-events.jsonl", events, cut=False)
    elif harness == "mini-swe-agent":
        messages: list[dict[str, Any]] = [{"role": "system", "content": "..."}, {"role": "user", "content": "task"}]
        for (fresh, read, w5, w1, out), cost in zip(calls, costs):
            details = {"cached_tokens": read, "cache_creation_tokens": w5 + w1,
                       "cache_creation_token_details": {"ephemeral_5m_input_tokens": w5,
                                                        "ephemeral_1h_input_tokens": w1}}
            usage = {"prompt_tokens": fresh + read + w5 + w1, "completion_tokens": out,
                     "prompt_tokens_details": details, "cache_read_input_tokens": read,
                     "cache_creation_input_tokens": w5 + w1}
            messages.append({"role": "assistant", "content": "...",
                             "extra": {"response": {"usage": usage}, "cost": cost}})
        write_json(agent / "mini-swe-agent.trajectory.json", {"messages": messages})
    elif harness == "terminus-2":
        steps: list[dict[str, Any]] = [{"step_id": 1, "source": "user", "message": "the task"}]
        for i, ((fresh, read, w5, w1, out), cost) in enumerate(zip(calls, costs)):
            steps.append({"step_id": i + 2, "source": "agent", "model_name": model, "metrics": {
                "prompt_tokens": fresh + read + w5 + w1, "completion_tokens": out, "cached_tokens": read,
                "cost_usd": cost}})
        totals = {"total_prompt_tokens": sum(c[0] + c[1] + c[2] + c[3] for c in calls),
                  "total_cached_tokens": sum(c[1] for c in calls), "total_completion_tokens": sum(c[4] for c in calls),
                  "total_cost_usd": sum(costs)}
        write_json(agent / "trajectory.json", {"schema_version": "ATIF-v1.6", "steps": steps, "final_metrics": totals})
    else:
        raise ValueError(f"no cache log recipe for {harness}")
    return (sum(c[0] + c[1] + c[2] + c[3] for c in calls), sum(c[1] for c in calls), sum(c[4] for c in calls),
            sum(costs))


def make_cache_trial(trial_dir: Path, harness: str, calls: Sequence[Sequence[int]], *,
                     model: str = "claude-opus-5-5", reported: float | None = None, **kwargs: Any) -> Path:
    """A finished Opus trial whose Harbor totals and model_usage are its log's
    (write_cache_log); reported replaces the harness's own cost."""
    n_in, n_cache, n_out, cost = write_cache_log(trial_dir, harness, calls, model=model)
    cost = cost if reported is None else reported
    usage = {f"anthropic/{model}": {"n_input_tokens": n_in, "n_cache_tokens": n_cache, "n_output_tokens": n_out,
                                    "cost_usd": cost}}
    return make_trial(trial_dir, family=harness, model_name=f"anthropic/{model}", tokens=(n_in, n_cache, n_out),
                      usage=usage, cost=cost, **kwargs)
