"""What each trial's outcome says about the harness under test.

Harbor records why a trial ended (its FailureCause), but not whether the
harness or the bench caused it. A trial that never reached the model says
nothing about a harness's token use, so it must be counted and shown, never
averaged in as a zero; a crash after real model use is a genuine failure
whose tokens and cost count. A trial whose agent ran but recorded no usage
(a timeout before any model call completed records 0/0/0) is neither: its
reward counts, its tokens and cost are unknown. The rules are heuristics, so
they are versioned: a change re-derives every stored trial from its raw JSON
without a re-run.

Pure: no I/O, no database.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from harbor.models.trial.error_policy import policy_for

from .harnesses import cap_misprices_for
from .hostload import HOT_CPU

# 4: a provider error the verifier still scored keeps its score (passed/failed). Harbor names provider errors
#    from phrases anywhere in the output, which also match the agent's own tool output; and a model whose
#    provider stalls has failed the task like any other failure (2026-11-01).
# 5: an agent timeout while the host CPU was busy for much of the agent's time (flags cpu_contended, from
#    the run's hostload.jsonl) is infra: the host, not the model, ran out the clock (2026-11-02).
# 6: an agent that fetched the benchmark's own answer files and used them (flags answer_lookup, or a verdict
#    by hand in config/answer_lookups.yaml; see lookups.py) is answer_lookup, a fail whatever the verifier
#    said (2026-11-02).
# 7: a provider-type error before the agent started (no model call: e.g. the harness install could not
#    resolve github.com) is setup_failure, ours to rerun, not provider_failure (2026-11-03).
CLASSIFIER_VERSION = 7

OUTCOMES = ("passed", "failed", "answer_lookup", "agent_limit", "agent_crash", "misconfigured_cap",
            "verifier_failure", "provider_failure", "setup_failure", "infra", "incomplete")

# A harness that ran this briefly and reported no tokens never got going.
SETUP_MAX_EXEC_SEC = 60.0

_INFRA = ("sandbox_failed", "harbor_failed", "cancelled")
_LIMIT = ("agent_exhausted_budget", "agent_refused_task")

# How much of what a trial spent its recorded usage covers (usage_status).
USAGE_STATUSES = ("full", "lower_bound", "unmeasured")
# Stopped from outside mid-run, so the harness's own log holds completed calls
# only: Harbor's timeout, and a cancel, kill or sandbox failure. Harbor never
# signals the process in the container, which may go on calling the model
# until the verifier ends.
_CUT_SHORT_OUTCOMES = ("incomplete", "infra")
_CUT_SHORT_EXCEPTIONS = ("AgentTimeoutError",)
# Reached only once the agent phase ran (a reward, a timeout, the harness's
# own stop or crash), whatever the timing fields say: an external job or an
# older Harbor may not record agent_execution.
_AGENT_RAN = ("passed", "failed", "answer_lookup", "verifier_failure", "agent_limit", "misconfigured_cap",
              "agent_crash")


@dataclass(frozen=True)
class Validity:
    tokens: bool    # tokens and cost measure the harness
    reward: bool    # the reward measures the harness
    included: bool  # the trial enters the default analysis


def harbor_cause(exception_type: str | None) -> str | None:
    """Harbor's own FailureCause for an exception type name, so the bench
    follows the pinned Harbor table instead of keeping a copy of it."""
    if not exception_type:
        return None
    return policy_for(exception_type).cause.value


def cap_misprices(family: str, provider: str, model: str, option: str) -> bool:
    """The harness priced its own cap at another provider's rates."""
    return cap_misprices_for(family, provider, option, model)


def _options(f: Mapping[str, Any]) -> Mapping[str, Any]:
    options = f.get("options")
    if isinstance(options, Mapping):
        return options
    raw = f.get("options_json")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def usage_recorded(f: Mapping[str, Any]) -> bool:
    """The trial recorded some model usage. Harbor's adapters write 0/0/0
    as readily as null when no call completed, so neither counts."""
    return (f.get("n_input") or 0) + (f.get("n_output") or 0) > 0


def usage_status(f: Mapping[str, Any]) -> str:
    """'full' when the recorded usage is all the trial spent (nothing, for a
    trial that never reached the model: a setup failure, an agent that never
    started, a provider failure within SETUP_MAX_EXEC_SEC); 'lower_bound' when it covers
    completed calls only, because the harness was stopped from outside;
    'unmeasured' when the agent ran but no usage was recorded, so its tokens
    and cost are unknown rather than zero. Uses a stored outcome if f has one."""
    outcome = f.get("outcome") or classify(f)[0]
    if usage_recorded(f):
        if outcome in _CUT_SHORT_OUTCOMES or f.get("exception_type") in _CUT_SHORT_EXCEPTIONS:
            return "lower_bound"
        return "full"
    if outcome in ("incomplete", *_AGENT_RAN):
        return "unmeasured"  # killed, or the agent ran: whether it spent anything is unknown
    if outcome == "setup_failure" or not f.get("agent_started"):
        return "full"  # the model never answered, so nothing was spent
    if outcome == "provider_failure":
        # Decided by the exception type alone, so a harness may have worked
        # for many minutes before the provider failed it.
        exec_sec = f.get("agent_exec_sec")
        return "full" if exec_sec is not None and exec_sec < SETUP_MAX_EXEC_SEC else "unmeasured"
    return "unmeasured"


def _unrecorded(f: Mapping[str, Any], outcome: str) -> str:
    """The reason's tail for a trial whose usage is unknown, naming what
    Harbor wrote: null (not reported) or zeros (nothing recorded)."""
    if usage_status({**f, "outcome": outcome}) != "unmeasured":
        return ""
    null = f.get("n_input") is None and f.get("n_output") is None
    return "; tokens not reported" if null else "; no usage recorded"


def _cap_option(f: Mapping[str, Any]) -> str | None:
    from .extract import CAP_SIGNATURES  # extract imports this module
    marker = f.get("cap_marker")
    if not marker:
        return None
    return (CAP_SIGNATURES.get(f.get("family") or "") or {}).get(marker)


def _flags(f: Mapping[str, Any]) -> Mapping[str, Any]:
    """The trial's flags (extract.trial_flags and the host load), as a
    mapping or flags_json text; {} when the facts carry none."""
    flags = f.get("flags")
    if isinstance(flags, Mapping):
        return flags
    raw = f.get("flags_json")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def answer_lookup(f: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The trial's answer lookup when it counts: a verdict by hand
    (answer_lookup_verdict) wins over the scan's own (answer_lookup.used).
    None when there was none, it was allowed, or it was not used."""
    flags = _flags(f)
    found = flags.get("answer_lookup") if isinstance(flags.get("answer_lookup"), Mapping) else {}
    by_hand = flags.get("answer_lookup_verdict") if isinstance(flags.get("answer_lookup_verdict"), Mapping) else {}
    if by_hand.get("verdict") == "allowed":
        return None
    if by_hand.get("verdict") == "lookup" or found.get("used") is True:
        return {**found, **by_hand}
    return None


def _lookup_reason(lookup: Mapping[str, Any]) -> str:
    urls = [u for u in (lookup.get("urls") or []) if isinstance(u, str)]
    where = urls[0] if urls else (lookup.get("note") or "")
    return "looked up the benchmark's answer" + (f" ({where})" if where else "")


def classify(f: Mapping[str, Any]) -> tuple[str, str]:
    """(outcome, reason) for one trial's facts. First matching rule wins.
    A finished trial whose usage is unknown says so at the end of its reason."""
    if not f.get("finished_at"):
        return "incomplete", "killed before finishing"
    outcome, reason = _classify(f)
    lookup = answer_lookup(f)
    if lookup is not None:
        # Policy, not a heuristic about the run: whatever else happened, it is a fail that counts.
        return "answer_lookup", (f"{_lookup_reason(lookup)}; otherwise {outcome}: {reason}"
                                 + _unrecorded(f, "answer_lookup"))
    return outcome, reason + _unrecorded(f, outcome)


def _classify(f: Mapping[str, Any]) -> tuple[str, str]:
    exception_type = f.get("exception_type")
    reward = f.get("reward")
    if not exception_type:
        if reward is None:
            return "verifier_failure", "no reward recorded"
        if reward >= 1.0:
            return "passed", f"reward {reward:g}"
        return "failed", f"reward {reward:g}"

    flags = _flags(f)
    if exception_type in _CUT_SHORT_EXCEPTIONS and flags.get("cpu_contended") is True:
        # Other work on the host slowed the agent: its timeout is the host's, so the trial is run again.
        hot = flags.get("host_cpu_hot_share")
        share = f"{100 * hot:.0f}% of the agent's time" if isinstance(hot, (int, float)) else "much of the agent's time"
        return "infra", f"{exception_type} while the host CPU was busy ({share} at >={HOT_CPU:g}%)"
    cause = f.get("harbor_cause") or harbor_cause(exception_type)
    if cause in _INFRA:
        return "infra", f"{exception_type} ({cause})"
    if cause == "model_provider_failed":
        if reward is not None:  # the verifier scored the work regardless
            return ("passed" if reward >= 1.0 else "failed"), f"reward {reward:g} after {exception_type}"
        if not f.get("agent_started"):  # no model call yet: the network failed during setup
            return "setup_failure", f"{exception_type} before the agent started"
        return "provider_failure", f"{exception_type}: the model provider failed"
    if cause == "verifier_failed":
        return "verifier_failure", f"{exception_type}: the verifier failed"
    if cause in _LIMIT:
        what = "ran out of its budget" if cause == "agent_exhausted_budget" else "refused the task"
        return "agent_limit", f"{exception_type}: the agent {what}"

    # Unclassified: the harness exited on its own, or an unknown error type.
    n_input, n_output = f.get("n_input"), f.get("n_output")
    tokens_null = n_input is None and n_output is None
    if not f.get("agent_started"):
        return "setup_failure", "agent never started"
    if not tokens_null and (n_input or 0) + (n_output or 0) == 0:
        return "setup_failure", "no model tokens"
    exec_sec = f.get("agent_exec_sec")
    setup_marker = f.get("setup_marker")
    if tokens_null and ((exec_sec is not None and exec_sec < SETUP_MAX_EXEC_SEC) or setup_marker):
        if setup_marker:
            return "setup_failure", f"setup error ('{setup_marker}'), no tokens reported"
        return "setup_failure", f"no tokens reported after {exec_sec:.1f} s of execution"

    option = _cap_option(f)
    if option and option in _options(f):
        if cap_misprices(f.get("family") or "", f.get("provider") or "", f.get("model") or "", option):
            return "misconfigured_cap", f"stopped by its own {option}, which it prices at another provider's rates"
        return "agent_limit", f"stopped by its own {option}"

    return "agent_crash", f"{exception_type} after using the model"  # '; tokens not reported' when null


def validity(outcome: str, *, crashes: str = "include", misconfigured_cap: str = "exclude",
             usage: str = "full") -> Validity:
    """How a trial with this outcome may be used, under the report's policy.
    usage is the trial's usage_status: unmeasured usage never counts as
    tokens or cost, while the reward and the trial itself still do."""
    if outcome not in OUTCOMES:
        raise ValueError(f"Unknown outcome '{outcome}'")
    if crashes not in ("include", "exclude"):
        raise ValueError("crashes must be include or exclude")
    if misconfigured_cap not in ("include", "exclude"):
        raise ValueError("misconfigured_cap must be include or exclude")
    if usage not in USAGE_STATUSES:
        raise ValueError(f"usage must be one of {', '.join(USAGE_STATUSES)}")
    if outcome in ("passed", "failed", "answer_lookup", "agent_limit"):
        v = Validity(True, True, True)  # answer_lookup is a fail: never passed (Ledger._trial_row)
    elif outcome == "agent_crash":
        ok = crashes == "include"
        v = Validity(ok, ok, ok)
    elif outcome == "misconfigured_cap":
        ok = misconfigured_cap == "include"  # then it counts as agent_limit
        v = Validity(ok, ok, ok)
    elif outcome == "verifier_failure":
        v = Validity(True, False, True)
    else:
        v = Validity(False, False, False)
    if usage == "unmeasured":
        return Validity(False, v.reward, v.included)
    return v
