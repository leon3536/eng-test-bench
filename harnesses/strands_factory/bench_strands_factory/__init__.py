"""Builds the Strands harness agent for Harbor's built-in `strands` agent.

Harbor's strands adapter installs strands-agents in the task container and
runs a factory that the user supplies; it ships none of its own. The Strands
team's published cost figures (S8, "We ran 94,000 trials to measure the cost
of harness choice") come from the `strands-harness` package's
create_harness(), run through this same adapter. So the bench's strands arms
build that agent here instead of a hand-written one. SSA
(simple-strands-agent) is a different harness and was ruled out.

Harbor's runner (harbor/agents/installed/strands_runner.py) calls
factory(sandbox, **factory_kwargs). The sandbox runs commands in the task
container. Unless factory_kwargs has "model", the runner also passes model=,
a strands Model it built from Harbor's -m value. strands-harness cannot apply
effort or prompt caching to a prebuilt Model, so an arm that wants both passes
model_spec ("provider/name") and the harness builds the model itself.
model_spec must name the arm's own model (the runner's HARBOR_MODEL): Harbor,
the ledger and the run budget price the trial at that model, so a stale
model_spec copied from another arm would be billed at the wrong rates.

This file runs inside the task container under Harbor's Python, never in the
bench's own venv, which has no strands packages.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from strands_harness import create_harness

if TYPE_CHECKING:
    from strands import Agent
    from strands.models import Model
    from strands.sandbox import Sandbox

__all__ = ["STATE_DIR", "create_agent"]

logger = logging.getLogger(__name__)

# strands-harness keeps session snapshots, offloaded tool results and
# long-term memory under ./.agent by default. The runner's working directory
# is the task's own, which the verifier inspects, so the same features keep
# their files here instead.
STATE_DIR = "/tmp/bench-strands-harness"


def create_agent(
    sandbox: Sandbox,
    model: Model | str | None = None,
    *,
    model_spec: str | None = None,
    effort: str | bool | None = "auto",
    **harness_kwargs: Any,
) -> Agent:
    """Return strands-harness's agent for one trial, its tools routed through `sandbox`.

    model_spec: "provider/name", e.g. "anthropic/claude-opus-5-5". strands-harness
        builds the model, applies `effort` and sets up prompt caching. Without it,
        `model` (the Model Harbor built) is used as it is. Raises ValueError
        when it differs from the arm's model (HARBOR_MODEL), before any model call.
    effort: a strands-harness level: "auto", "off", "minimal", "low", "medium",
        "high", "xhigh" or "max". "auto" is the provider's recommended level,
        which is "high" on Anthropic.
    harness_kwargs: any other create_harness() argument, such as builtin_tools.
    """
    # YAML 1.1 reads a bare `off` as false; JSON null means "not set".
    if effort is False:
        effort = "off"
    elif effort is None:
        effort = "auto"
    harness_kwargs.setdefault("session", {"dir": f"{STATE_DIR}/sessions"})
    harness_kwargs.setdefault("memory", {"dir": f"{STATE_DIR}/memory"})
    if model_spec is None and isinstance(model, str):
        model_spec = model
    harbor_model = os.environ.get("HARBOR_MODEL", "").strip()
    if model_spec and harbor_model and model_spec.strip().lower() != harbor_model.lower():
        raise ValueError(
            f"factory_kwargs.model_spec {model_spec!r} does not match the arm's model "
            f"{harbor_model!r}. Set model_spec to {harbor_model!r}, or remove it to use "
            "the model Harbor built."
        )
    if model_spec:
        return create_harness(model=model_spec, effort=effort, sandbox=sandbox, **harness_kwargs)
    if effort != "auto":
        logger.warning(
            "bench_strands_factory: effort=%r is not applied to the model Harbor built, "
            "so it runs at that model's default. Set factory_kwargs.model_spec to apply it.",
            effort,
        )
    return create_harness(model=model, effort="auto", sandbox=sandbox, **harness_kwargs)
