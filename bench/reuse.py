"""Reusing measured trials: a run that only fills the ledger's gaps.

A design or LAUNCH run with reuse "measured" skips, per arm, every task for
which the ledger already holds enough usable trials of the same cell, so
trials already paid for are not bought twice. What counts is decided as the
REPORT tab decides it, from report.analysis_rows:

- Same cell: the configuration hash the arm's trials will get (family,
  implementation, provider, model, effort as resolved, the arm's options and
  the run caps its harness takes, and the timeout multiplier), on the same
  dataset, with the task matched the way the report pairs it (content digest,
  unless report.match_on is name). The harness version must be known to
  match: pinned to an exact release in options.version (so in the hash), or
  bundled with Harbor (its schema has no version option, and it is not code
  of this bench's own). Either way the trial must have run on the Harbor
  installed now: Harbor ships a bundled harness, and its adapter for an
  installed one turns the harness's trajectory into the usage and cost the
  ledger records, so another Harbor can measure the same release
  differently. An unpinned harness, or one pinned to a tag or range,
  installs whatever release is newest in every trial, so its earlier trials
  are never reused. How a trial got its harness (a stock install or a
  pre-baked image) is provenance, not part of the cell.
- Usable trial: one the report counts for pass rate and for tokens under the
  profile's report policy: a valid outcome (agent_crash only while
  report.crashes includes crashes) and usage recorded. Stricter than the
  report, because a reused task is never run again for that cell: a trial
  excluded by hand (the latest exclusion wins) never counts, even under
  report.exclusions false, and neither does one that spent more than 5% of
  its tokens on another model (a fallback after a refusal measured that
  model; the report only adds a caveat). A trial cut short counts, its usage
  a lower bound, as in the report. Trials of any profile count: the report
  pools every profile, and the cell pins everything a profile could change.
  Two arms of one cell in a design never reuse the same trial.

A task is covered when it has spec.attempts usable trials. resolve()
materialises this into the spec: covered tasks leave the arm's task list and
are listed with the trials they reuse under spec.reused, so the hash a
person approves is what runs. An arm left with no task stays in the spec and
is skipped. summary() is the check: what is reused, the cost avoided, why an
arm reuses nothing, and whether the ledger would now resolve it differently;
drift() is the part of that a queued version is held for.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import datasets, designs, estimates, report, schemas
from .settings import Config
from .spec import ArmSpec, spec_from_design, unresolved

MODES = ("off", "measured")
# An exact release (2.1.312, 1.0.0rc1, 1.2.3-beta.1). A tag (latest), a range
# (^2.1, 2.x) or a bare major (2) installs whichever release is newest when a
# trial starts, and Harbor records the option as given, not what it installed.
_EXACT_VERSION = re.compile(r"\d+\.\d+\.\d+(?:[-+.]?[0-9A-Za-z][0-9A-Za-z.+-]*)?")


def check_mode(value: Any) -> str:
    if value not in MODES:
        raise ValueError(f"reuse must be one of {', '.join(MODES)}")
    return value


def harbor_version() -> str:
    """The Harbor installed now, which is what runs a bundled harness."""
    try:
        return importlib.metadata.version("harbor")
    except importlib.metadata.PackageNotFoundError:
        return ""


def snapshot_id(dataset: str) -> str | None:
    """The dataset_snapshot the ledger gives trials run from this dataset's
    local snapshot (extract.context_for); None before it is downloaded."""
    try:
        info = datasets.info(dataset)
    except Exception:  # an unreadable manifest: as if there were no snapshot
        return None
    if not info or not info.get("downloaded_at"):
        return None
    return f"local:{info.get('dataset') or dataset}@{info['downloaded_at']}"


def usable(row: Mapping[str, Any]) -> bool:
    """The report counts this trial for both pass rate and tokens, no person
    excluded it, and it spent its tokens on the arm's model."""
    return bool(row.get("valid_reward") and row.get("valid_tokens") and not row.get("excluded_by_hand")
                and not _fallback(row))


def _fallback(row: Mapping[str, Any]) -> bool:
    """The harness spent more than 5% of its tokens on another model (or its
    usage says nothing about which model)."""
    share = row.get("primary_share")
    return share is None or share < report.PRIMARY_SHARE_MIN


@dataclass
class Coverage:
    """What the ledger holds for one arm's cell."""
    arm: str
    family: str
    config_hash: str | None = None
    version: str = "unknown"        # pinned | harbor | unpinned | unknown: how the harness version is fixed
    reason: str | None = None       # why this arm can reuse nothing
    covered: dict[str, list[Mapping[str, Any]]] = field(default_factory=dict)  # task -> the trials reused
    short: dict[str, int] = field(default_factory=dict)     # task -> usable trials, fewer than attempts
    cell_trials: int = 0            # usable trials of this cell on any task
    fallback: int = 0               # trials of this cell the report counts, set aside for another model's tokens
    other_harbor: int = 0           # trials of this cell set aside because another Harbor ran them


def coverage(cfg: Config, spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[Coverage]:
    """Per arm of spec, in order, the tasks it asks for that the ledger covers."""
    asked = unresolved(spec)
    run_spec = spec_from_design(asked, None)
    run, attempts = run_spec.run, run_spec.n_attempts
    arm_specs = {a.name: a for a in (*run_spec.arms, *run_spec.skipped_arms)}
    task_of, why_not = _task_matcher(cfg, asked["dataset"], rows)
    current = harbor_version()
    # Two arms of one cell in a design measure twice without reuse, so they never share a reused trial.
    claimed: dict[str, set[str]] = defaultdict(set)
    out = []
    for entry in asked["arms"]:
        arm = arm_specs[entry["name"]]
        cov = Coverage(arm=arm.name, family=arm.family)
        out.append(cov)
        kwargs = estimates.planned_kwargs(cfg, arm, run)
        if kwargs is None:
            cov.reason = ("its configuration could not be worked out: the harness is not one designs may use, or "
                          "its options did not resolve")
            continue
        cov.config_hash = estimates.config_hash_for(arm, kwargs, run.timeout_multiplier)
        cov.version, cov.reason = _version(arm, kwargs, current)
        cov.reason = cov.reason or why_not  # the arm's own reason outlasts the run's
        if cov.reason:
            continue
        wanted = set(arm.tasks)
        found: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for r in rows:
            if not (r.get("config_hash") == cov.config_hash and r.get("family") == arm.family
                    and r.get("provider") == arm.provider and r.get("model") == arm.model
                    and r.get("dataset") == asked["dataset"]):
                continue
            if r.get("harbor_version") != current:
                if usable(r):
                    cov.other_harbor += 1
                continue
            if not usable(r):
                if r.get("valid_reward") and r.get("valid_tokens") and not r.get("excluded_by_hand"):
                    cov.fallback += 1
                continue
            cov.cell_trials += 1
            task = task_of(r)
            if task in wanted and r.get("trial_key") not in claimed[cov.config_hash]:
                found[task].append(r)
        for task in dict.fromkeys(arm.tasks):
            # Full usage first, then the oldest: a later trial changes what a spec reuses only by
            # replacing one that was cut short.
            trials = sorted(found.get(task, []), key=lambda r: (r.get("usage_status") != "full",
                                                                str(r.get("started_at") or "~"),
                                                                str(r.get("trial_key"))))
            if len(trials) >= attempts:
                cov.covered[task] = trials[:attempts]
                claimed[cov.config_hash].update(str(r.get("trial_key")) for r in cov.covered[task])
            elif trials:
                cov.short[task] = len(trials)
    return out


def _version(arm: ArmSpec, kwargs: Mapping[str, Any], current: str) -> tuple[str, str | None]:
    pinned = kwargs.get("version")
    if pinned not in (None, ""):
        if not _EXACT_VERSION.fullmatch(str(pinned)):
            return "unpinned", (f"its options.version {str(pinned)!r} is a tag or range, not an exact release: "
                                "Harbor installs whichever release matches when a trial starts, so earlier trials "
                                "may have run another version")
        if not current:
            return "pinned", ("the installed Harbor version could not be read, and Harbor's adapter for the "
                              "harness is what reads its usage")
        return "pinned", None  # an option, so already part of the configuration hash
    try:
        options = schemas.agent_schema(arm.agent).get("properties") or {}
    except Exception:
        return "unknown", "its harness options could not be read, so whether its version is fixed is unknown"
    if "version" in options:
        return "unpinned", ("its harness version is not pinned (options.version): Harbor installs the latest "
                            "release in every trial, so earlier trials may have run another version")
    if arm.kind == "import_path":
        return "unknown", ("its harness is code of this bench (an import path), which neither Harbor's version "
                           "nor the configuration fixes, so which code earlier trials ran is unknown")
    if not current:
        return "unknown", "the installed Harbor version could not be read"
    return "harbor", None  # ships inside Harbor, so Harbor's version fixes it


def _task_matcher(cfg: Config, dataset: str, rows: Sequence[Mapping[str, Any]]
                  ) -> tuple[Callable[[Mapping[str, Any]], str | None], str | None]:
    """(the task a trial counts for in this run, or None; why no trial can
    count), pairing tasks as the report does."""
    o = estimates.pair_options(cfg)
    if o.same_run_only:
        return (lambda r: None), "report.same_run_only is on, so trials of earlier runs never pair with this run's"
    if o.match_on == "name":
        return (lambda r: str(r.get("task"))), None
    snapshot = snapshot_id(dataset)
    if snapshot is None:
        return (lambda r: None), (f"there is no local snapshot of {dataset} yet, so the task versions this run will "
                                  "use are unknown")
    # The snapshot is fixed once downloaded, so a trial run from it shows the
    # digest this run's trials will get: the latest one's, should its files
    # have been changed in place since. A task none ran from it cannot be matched.
    latest: dict[str, tuple[str, str]] = {}
    for r in rows:
        if r.get("dataset_snapshot") == snapshot and r.get("task_digest"):
            task, started = str(r.get("task")), str(r.get("started_at") or "")
            if task not in latest or started >= latest[task][0]:
                latest[task] = (started, str(r["task_digest"]))
    if not latest:
        return (lambda r: None), (f"no trial has run from the current snapshot of {dataset} yet, so the task "
                                  "versions this run will use are unknown")
    digests = {task: digest for task, (_, digest) in latest.items()}

    def task_of(r: Mapping[str, Any]) -> str | None:
        task = str(r.get("task"))
        return task if r.get("task_digest") == digests.get(task) else None

    return task_of, None


def resolve(cfg: Config, spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
            mode: str | None = None) -> dict[str, Any]:
    """spec with its trials to reuse resolved against rows
    (report.analysis_rows), from the tasks each arm asked for; mode, when
    given, replaces the spec's own setting. Off gives the spec as asked for."""
    asked = designs.with_reuse(spec, check_mode(mode if mode is not None else spec.get("reuse") or "off"))
    if not asked.get("reuse"):
        return asked
    return _materialized(asked, coverage(cfg, asked, rows))


def _materialized(asked: Mapping[str, Any], found: Sequence[Coverage]) -> dict[str, Any]:
    out = copy.deepcopy(dict(asked))
    covered = {c.arm: c.covered for c in found}
    reused: dict[str, list[dict[str, Any]]] = {}
    for arm in out["arms"]:
        tasks = covered.get(arm["name"]) or {}
        if not tasks:
            continue
        asked_tasks = list(out["tasks"] if arm["tasks"] is None else arm["tasks"])
        arm["tasks"] = [t for t in asked_tasks if t not in tasks]
        reused[arm["name"]] = [
            {"task": t, "trials": [str(r["trial_key"]) for r in tasks[t]],
             "runs": list(dict.fromkeys(str(r.get("run_key")) for r in tasks[t]))}
            for t in dict.fromkeys(asked_tasks) if t in tasks
        ]
    out["reused"] = reused
    return out


def materialize(cfg: Config, spec: Mapping[str, Any],
                rows: Callable[[], Sequence[Mapping[str, Any]]], mode: str | None = None) -> dict[str, Any]:
    """resolve(), reading the ledger only when reuse is on."""
    if (mode if mode is not None else spec.get("reuse") or "off") == "off":
        return designs.with_reuse(spec, "off") if spec.get("reuse") else copy.deepcopy(dict(spec))
    return resolve(cfg, spec, rows(), mode)


def summary(cfg: Config, spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """What a reuse spec takes from the ledger, its measured cost, why an arm
    reuses nothing, and how the ledger would resolve it now (changes is
    empty when saving it again would give the same spec). None when off."""
    if not spec.get("reuse"):
        return None
    asked = unresolved(spec)
    found = coverage(cfg, asked, rows)
    fresh = _materialized(asked, found)
    figures = estimates.reused_figures(spec, rows) or {}
    costs = {(a["arm"], t["task"]): t["cost_avoided_usd"] for a in figures.get("arms") or [] for t in a["per_task"]}
    by_key = {r.get("trial_key"): r for r in rows}
    attempts = spec["attempts"]
    changes = _changes(spec, fresh)
    arms_out, cut_short, fallback, other_harbor = [], 0, 0, 0
    for arm, cov in zip(spec["arms"], found):
        name = arm["name"]
        runs = list(spec["tasks"] if arm["tasks"] is None else arm["tasks"])
        fallback += cov.fallback
        other_harbor += cov.other_harbor
        reused = []
        for entry in (spec.get("reused") or {}).get(name) or []:
            lower = sum(1 for k in entry["trials"] if (by_key.get(k) or {}).get("usage_status") == "lower_bound")
            cut_short += lower
            reused.append({**copy.deepcopy(entry), "cost_avoided_usd": costs.get((name, entry["task"])),
                           "lower_bound": lower})
        arm_figures = next((a for a in figures.get("arms") or [] if a["arm"] == name), {})
        arms_out.append({
            "arm": name, "family": cov.family, "config_hash": cov.config_hash, "harness_version": cov.version,
            "reason": cov.reason, "tasks_asked": len(runs) + len(reused), "tasks_reused": len(reused),
            "tasks_to_run": len(runs), "trials_reused": len(reused) * attempts,
            "cost_avoided_usd": arm_figures.get("cost_avoided_usd", 0.0), "skipped": bool(reused) and not runs,
            "cell_trials": cov.cell_trials, "reused": reused,
            "short": [{"task": t, "usable": n} for t, n in cov.short.items()],
        })
    notes = []
    if cut_short:
        notes.append(f"{cut_short} reused trial{'s were' if cut_short != 1 else ' was'} cut short, so "
                     f"{'their' if cut_short != 1 else 'its'} usage is a lower bound, as the report counts it.")
    short = sum(len(a["short"]) for a in arms_out)
    if short:
        notes.append(f"{short} task{'s have' if short != 1 else ' has'} some usable trials but fewer than "
                     f"{attempts} (attempts); {'they run' if short != 1 else 'it runs'} in full.")
    if figures.get("unpriced_tasks"):
        notes.append(f"{figures['unpriced_tasks']} reused task(s) have no measured cost, so the cost avoided "
                     "leaves them out.")
    if fallback:
        many = fallback != 1
        notes.append(f"{fallback} trial{'s' if many else ''} of these configurations spent more than "
                     f"{100 * (1 - report.PRIMARY_SHARE_MIN):.0f}% of {'their' if many else 'its'} tokens on another "
                     f"model (a fallback), so {'they' if many else 'it'} measured that model and "
                     f"{'are' if many else 'is'} not reused.")
    if other_harbor:
        many = other_harbor != 1
        notes.append(f"{other_harbor} trial{'s' if many else ''} of these configurations ran on another Harbor than "
                     f"the {harbor_version() or 'unknown'} installed now, which ships Terminus-2 and reads every "
                     "installed harness's usage, so " + ("they are" if many else "it is") + " not reused.")
    trials_to_run = designs.trial_count(spec)
    return {
        "mode": spec["reuse"],
        "trials_asked": trials_to_run + figures.get("trials", 0),
        "trials_reused": figures.get("trials", 0),
        "trials_to_run": trials_to_run,
        "cost_avoided_usd": figures.get("cost_avoided_usd"),
        "skipped_arms": [a["arm"] for a in arms_out if a["skipped"]],
        "arms": arms_out,
        "current": not changes,
        "changes": changes,
        "notes": notes,
    }


def fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    """What resolving reads from the ledger: each trial, whether it is usable
    and its usage status, and the Harbor installed now. A cached check of a
    reuse spec keys on this, because report.ledger_fingerprint hashes trial
    keys only, so an exclusion or a trial finishing under its key leaves it
    unchanged."""
    keys = sorted(f"{r.get('trial_key')}|{int(usable(r))}|{r.get('usage_status')}" for r in rows)
    text = "\n".join(keys) + "\nharbor:" + harbor_version()
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def drift(cfg: Config, spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """What a saved reuse spec would run differently if it were resolved now:
    tasks the ledger now covers, and reused tasks it no longer covers. Empty
    when off or unchanged; other trials now covering a reused task change
    nothing that runs, so they are left out."""
    if not spec.get("reuse"):
        return []
    asked = unresolved(spec)
    return _changes(spec, _materialized(asked, coverage(cfg, asked, rows)), swaps=False)


def _changes(spec: Mapping[str, Any], fresh: Mapping[str, Any], swaps: bool = True) -> list[str]:
    changes = []
    for arm, now in zip(spec["arms"], fresh["arms"]):
        name = arm["name"]
        runs = list(spec["tasks"] if arm["tasks"] is None else arm["tasks"])
        runs_now = list(spec["tasks"] if now["tasks"] is None else now["tasks"])
        gained = [t for t in runs if t not in runs_now]
        lost = [t for t in runs_now if t not in runs]
        if gained:
            changes.append(f"Arm '{name}': {_short(gained)} now ha{'s' if len(gained) == 1 else 've'} enough "
                           "usable trials; save a new version to reuse them")
        if lost:
            changes.append(f"Arm '{name}': {_short(lost)} no longer ha{'s' if len(lost) == 1 else 've'} "
                           "enough usable trials (excluded since, or the report policy or installed Harbor "
                           "changed); save a new version to run them")
        stored = {e["task"]: e["trials"] for e in (spec.get("reused") or {}).get(name) or []}
        moved = [e["task"] for e in fresh["reused"].get(name) or []
                 if e["task"] in stored and e["trials"] != stored[e["task"]]]
        if moved and swaps:
            changes.append(f"Arm '{name}': other trials now cover {_short(moved)}; save a new version to record "
                           "them")
    return changes


def _short(names: list[str], limit: int = 8) -> str:
    more = f" and {len(names) - limit} more" if len(names) > limit else ""
    return ", ".join(names[:limit]) + more
