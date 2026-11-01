"""The per-model designs' arms: every harness on one Fireworks, OpenAI or xAI model.

One design per model runs the eight harnesses on the 27-task IT/SaaS scope,
with Claude Code on Opus 5.5 as the baseline arm. That arm's trials are all
reused from the ledger (reuse measured), so a design runs only the eight.
tests/test_agents_providers.py builds each arm's Harbor adapter from these
arms, as a run would, and checks what it sends. The design files are made
from design_input(), so what is tested is what is queued.

Effort (decided 2026-10-31): GPT-6.1 Terra and Grok 4.7 at medium, the run's
effort. DeepSeek V4.1 Flash, GLM 5.3 and Kimi K3 at high: they offer only
low, high and max, Fireworks turns medium into high, and a harness that sent
no effort would leave them at their default, max. So their arms set high
explicitly, while run.effort stays medium for the reused Opus baseline,
whose trials ran at medium.
"""

from __future__ import annotations

import copy
from typing import Any, NamedTuple

# The 27-task IT/SaaS scope, in the seeded-sample order the earlier designs use.
TASKS = ["openssl-selfsigned-cert", "cobol-modernization", "build-cython-ext", "mteb-retrieve",
         "sparql-university", "sqlite-with-gcov", "sqlite-db-truncate", "hf-model-inference", "db-wal-recovery",
         "git-multibranch", "log-summary-date-ranges", "nginx-request-logging", "multi-source-data-merger",
         "reshard-c4-data", "sanitize-git-repo", "kv-store-grpc", "fix-git", "mailman", "cancel-async-tasks",
         "constraints-scheduling", "financial-document-processor", "fix-code-vulnerability", "git-leak-recovery",
         "headless-terminal", "modernize-scientific-stack", "pypi-server", "regex-log"]

RUN_EFFORT = "medium"
VERSIONS = {"claude-code": "2.1.312", "mini-swe-agent": "2.4.6", "pi": "1.4.0", "strands": "1.57.2",
            "cline": "3.0.74", "goose": "1.57.0", "opencode": "1.19.5", "kilo": "7.8.18"}
STRANDS_FACTORY = {"factory": "bench_strands_factory:create_agent", "factory_path": "harnesses/strands_factory"}


class Model(NamedTuple):
    provider: str              # the bench provider (config/bench.yaml)
    model: str                 # without the provider's model prefix
    title: str
    effort: str                # the level every harness arm runs at
    session_header: str | None  # the header LiteLLM harnesses put the trial's session id in
    harbor: str                # Harbor's -m value


MODELS: dict[str, Model] = {
    "deepseek": Model("fireworks", "deepseek-v4p1-flash-1017", "DeepSeek V4.1 Flash", "high", "x-session-affinity",
                      "fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash-1017"),
    "glm": Model("fireworks", "glm-5p3", "GLM 5.3", "high", "x-session-affinity",
                 "fireworks_ai/accounts/fireworks/models/glm-5p3"),
    "kimi": Model("fireworks", "kimi-k3", "Kimi K3", "high", "x-session-affinity",
                  "fireworks_ai/accounts/fireworks/models/kimi-k3"),
    "terra": Model("openai", "gpt-6.1-terra", "GPT-6.1 Terra", "medium", None, "openai/gpt-6.1-terra"),
    "grok": Model("xai", "grok-4.7", "Grok 4.7", "medium", "x-grok-conv-id", "xai/grok-4.7"),
}

HARNESSES = ("terminus-2", "mini-swe-agent", "pi", "strands", "cline", "goose", "opencode", "kilo")

def effort_note(harness: str, m: Model) -> str:
    """How one arm sends the model its effort; the design's rationale carries these."""
    responses = m.provider == "openai"
    return {
        "terminus-2": "reasoning_effort to LiteLLM",
        "mini-swe-agent": ("reasoning.effort on LiteLLM's Responses client" if responses
                           else "reasoning_effort to LiteLLM"),
        "pi": "--thinking",
        "strands": ("factory effort (strands-harness's OpenAI Responses model)" if responses
                    else "model_kwargs.params.reasoning_effort on the model Harbor builds"),
        "cline": "--thinking",
        "goose": ("GOOSE_THINKING_EFFORT" if responses
                  else "reasoning_effort in the request_params of the custom provider bench/agents/goose.py writes"),
        "opencode": "the model variant" if m.provider != "fireworks" else "the model variant, declared in its entry",
        "kilo": "the model variant" if m.provider != "fireworks" else "the model variant, declared in its entry",
    }[harness]


def _arm_effort(m: Model) -> str:
    """The arm's effort field: inherit the run's medium, or the model's own level."""
    return "inherit" if m.effort == RUN_EFFORT else m.effort


def arms(key: str) -> list[dict[str, Any]]:
    """The design's arms (ArmIn form) for one model: the reused Opus baseline, then the eight."""
    m = MODELS[key]
    effort = _arm_effort(m)
    headers = {"session_id_headers": [m.session_header]} if m.session_header else {}

    def arm(name: str, agent: str, options: dict[str, Any], arm_effort: str = effort) -> dict[str, Any]:
        return {"name": name, "agent": agent, "provider": m.provider, "model": m.model, "effort": arm_effort,
                "options": options, "tasks": None}

    if m.provider == "openai":
        strands = {**STRANDS_FACTORY, "version": VERSIONS["strands"],
                   "factory_kwargs": {"model_spec": m.harbor, "effort": m.effort}}
    else:
        # strands-harness has no Fireworks or xAI provider: the model Harbor builds, effort in its params.
        strands = {**STRANDS_FACTORY, "version": VERSIONS["strands"],
                   "model_kwargs": {"params": {"reasoning_effort": m.effort}}}
    return [
        {"name": "claude-code", "agent": "claude-code", "provider": "anthropic", "model": "claude-opus-5-5",
         "effort": "inherit", "options": {"version": VERSIONS["claude-code"]}, "tasks": None},
        arm("terminus-2", "terminus-2", copy.deepcopy(headers)),
        arm("mini-swe-agent", "bench.agents.mini_swe_agent:MiniSweAgentWithMcp",
            {"version": VERSIONS["mini-swe-agent"], **copy.deepcopy(headers)}),
        arm("pi", "bench.agents.pi:PiProviders", {"version": VERSIONS["pi"]}),
        arm("strands", "strands", strands, "omit"),
        arm("cline", "bench.agents.cline:PinnedClineCli", {"version": VERSIONS["cline"]}),
        arm("goose", "bench.agents.goose:PinnedGoose", {"version": VERSIONS["goose"]}),
        arm("opencode", "bench.agents.opencode:OpenCodeBunEpipe",
            {"version": VERSIONS["opencode"], "variant": m.effort}, "omit"),
        arm("kilo", "bench.agents.kilo:KiloCode", {"version": VERSIONS["kilo"], "variant": m.effort}, "omit"),
    ]


def design_input(key: str, budget_usd: float) -> dict[str, Any]:
    """The design as POST /api/designs takes it: meta fields and a DesignSpecIn."""
    m = MODELS[key]
    notes = "; ".join(f"{h}: {effort_note(h, m)}" for h in HARNESSES)
    level = ("high, set on each arm (low/high/max only; Fireworks turns medium into high; never their "
             "default max)" if m.effort != RUN_EFFORT else "medium, the run's effort")
    return {
        "title": f"{m.title}: 8 harnesses, 27 tasks",
        "purpose": (f"Every harness on {m.title} ({m.provider}) over the 27-task IT/SaaS scope, against Claude "
                    "Code on Opus 5.5 as the baseline, whose trials are all reused from the ledger. Pairs form "
                    "only within one provider and model, so this run's report pairs no arm with that baseline: "
                    "the comparison with Opus is a ledger report with compare models and baseline_model "
                    "anthropic:claude-opus-5-5, each harness against itself on Opus."),
        "rationale": f"Effort {level}. How each arm sends it: {notes}.",
        "change_note": "",
        "spec": {
            "tasks": list(TASKS),
            "attempts": 1,
            "baseline_arm": "claude-code",
            "run": {"budget_usd": float(budget_usd), "parallel_arms": 1, "n_concurrent": 2, "effort": RUN_EFFORT,
                    "per_trial_cap_usd": 15.0, "max_turns": 300, "timeout_multiplier": 1.0, "prebake": "prefer",
                    "schedule": "tasks"},
            "arms": arms(key),
            "reuse": "measured",
        },
    }
