"""A gateway arm's tokens and cost corrected from the gateway's own spend log.

Claude Code on a LiteLLM arm that the gateway translates to OpenAI's
Responses API (claude-code-gw-terra: GPT-6.1 Terra through LiteLLM 1.83.7)
never hears about cached tokens: its usage shows cache_read_input_tokens 0
and input_tokens with the cached ones inside. The bench then prices every
prompt token as fresh input, so the arm's cost is an upper bound, though
upstream caching works. The gateway's spend log (LiteLLM_SpendLogs) has the
real split and the cost the gateway logged, one row per model call.

    python -m bench.ledger gateway-spend FILE.csv --run RUN_ID --arm ARM --dry-run
    python -m bench.ledger gateway-spend FILE.csv --run RUN_ID --arm ARM

The CSV has a header row with start_utc, end_utc, prompt_tokens (cached
ones included), cached_prompt_tokens, completion_tokens, cost_usd and
request_id; a model column, if there is one, keeps only the arm's model's
rows. That column (and any filter on the export) is LiteLLM's model_group,
the gateway alias the arm names: the spend log's own model field holds the
deployment's upstream id, which is accepted too unless the provider's
upstream_models gives it to another of its models as well (then it cannot
tell their calls apart, and those rows are left out). Each row goes to the
trial of that arm in that run whose agent execution window holds the row's
start time, give or take --skew seconds for the gateway's clock against this
host's. Matching by time only works while the arm ran one trial at a time
(a task race with n_attempts 1): the command refuses a run where two of the
arm's trials ran at once (n_concurrent or n_attempts above 1). A row near
two back-to-back windows is first given to the one that holds it without
the skew; if that still leaves two, the row is ambiguous and nothing is
written. Rows no trial holds are listed: calls outside the run, or traffic
that is not this arm's.

Each matched trial gets one row in the ledger's cost_corrections table: the
summed tokens, cache reads and output, the logged cost, the number of calls
and where they came from (the file's name, digest, rows and totals). It is a
fact entered by hand, like an exclusion: appended, the latest per trial in
force, never touched by a rescan or rederive, and applied when the ledger
reads its rows, so the report, the ledger table and each run's spend use it
(flagged cost_from_gateway_log). Harbor's trial files are never changed.
Re-importing the same file writes nothing; a new export replaces a trial's
correction rather than adding to it. --clear takes an arm's corrections in
a run back. Trials of a corrected arm with none of their own and no cached
tokens recorded are flagged cost_upper_bound. Other arms and providers are
untouched: the command refuses an arm whose trials did not run through a
gateway.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import pricing, settings
from .fsutil import canonical_json

CORRECTED_FLAG = "cost_from_gateway_log"
UPPER_BOUND_FLAG = "cost_upper_bound"
SOURCE = "gateway_spend_log"
DEFAULT_SKEW_SEC = 5.0
COLUMNS = ("start_utc", "end_utc", "prompt_tokens", "cached_prompt_tokens", "completion_tokens", "cost_usd",
           "request_id")
# A trial whose logged prompt tokens differ from what Claude Code recorded by more than this share is named:
# the export may hold another arm's calls, or miss some.
INPUT_TOLERANCE = 0.25
# Ingest checks (extract.trial_flags) on the harness's own usage and cost. The gateway logs every call it
# billed, so on a corrected trial they question a figure no longer shown, and the ledger sets them aside.
SUPERSEDED_FLAGS = frozenset({"cache_writes_unknown", "cache_writes_exceed_prompt", "reported_cost_mismatch",
                              "unsaved_calls", "unfinished_calls", "declined_attempts"})
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


# Reading the export --------------------------------------------------------------

@dataclass(frozen=True)
class SpendRow:
    line: int               # in the file, the header being line 1
    start: datetime         # UTC
    end: datetime | None
    prompt: int             # cached ones included
    cached: int
    completion: int
    cost: float
    request_id: str
    model: str | None


@dataclass(frozen=True)
class SpendLog:
    name: str
    sha: str                # first 12 hex of sha256 over the file bytes
    rows: list[SpendRow]
    duplicates: int         # rows that repeat a request id read before, left out


def _utc(text: str, line: int, column: str) -> datetime:
    try:
        dt = datetime.fromisoformat(text.strip())
    except ValueError:
        raise ValueError(f"line {line}: {column} is not a time: {text!r}") from None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _count(text: str, line: int, column: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"line {line}: {column} is not a number: {text!r}") from None
    if value < 0 or value != int(value):
        raise ValueError(f"line {line}: {column} must be a whole number of tokens (got {text!r})")
    return int(value)


def _money(text: str, line: int) -> float:
    text = (text or "").strip()
    if not text:
        return 0.0
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"line {line}: cost_usd is not a number: {text!r}") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"line {line}: cost_usd must be zero or more (got {text!r})")
    return value


def read_log(path: Path) -> SpendLog:
    """The export's rows, in file order. ValueError names the line of
    anything it cannot read; a time without a zone is taken as UTC."""
    path = Path(path)
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    header = {(name or "").strip().lower(): name for name in reader.fieldnames or []}
    missing = [c for c in COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{path.name} has no {', '.join(missing)} column; expected {', '.join(COLUMNS)}")
    rows: list[SpendRow] = []
    seen: set[str] = set()
    duplicates = 0
    for i, raw in enumerate(reader, start=2):
        cell = {c: (raw.get(name) or "") for c, name in header.items()}
        if not any(v.strip() for v in cell.values()):
            continue
        if not cell["start_utc"].strip():
            raise ValueError(f"line {i}: start_utc is empty")
        prompt, cached = _count(cell["prompt_tokens"], i, "prompt_tokens"), _count(
            cell["cached_prompt_tokens"], i, "cached_prompt_tokens")
        if cached > prompt:
            raise ValueError(f"line {i}: cached_prompt_tokens ({cached}) exceeds prompt_tokens ({prompt}); "
                             "prompt_tokens must include the cached ones")
        request_id = cell["request_id"].strip()
        if request_id and request_id in seen:
            duplicates += 1
            continue
        seen.add(request_id)
        rows.append(SpendRow(
            line=i, start=_utc(cell["start_utc"], i, "start_utc"),
            end=_utc(cell["end_utc"], i, "end_utc") if cell["end_utc"].strip() else None,
            prompt=prompt, cached=cached, completion=_count(cell["completion_tokens"], i, "completion_tokens"),
            cost=_money(cell["cost_usd"], i), request_id=request_id,
            model=(cell["model"].strip() or None) if "model" in cell else None))
    return SpendLog(path.name, hashlib.sha256(data).hexdigest()[:12], rows, duplicates)


# Matching ------------------------------------------------------------------------

@dataclass
class Matching:
    by_trial: dict[str, list[SpendRow]] = field(default_factory=dict)
    unmatched: list[SpendRow] = field(default_factory=list)
    ambiguous: list[tuple[SpendRow, list[str]]] = field(default_factory=list)   # with the trials it fits
    other_models: list[SpendRow] = field(default_factory=list)
    shared_upstream: list[SpendRow] = field(default_factory=list)   # an upstream id other models stream as too


def _same_model(reported: str, model: str) -> bool:
    name = reported.split("/")[-1]
    return name == model or settings.is_variant(name, model)


def _streams_as(reported: str, upstream: str) -> bool:
    """reported is the upstream id, perhaps behind a LiteLLM provider prefix
    (fireworks_ai/accounts/fireworks/models/...)."""
    parts = reported.split("/")
    return any(settings.streamed_as("/".join(parts[i:]), upstream) for i in range(len(parts)))


def upstream_ids(provider: str, model: str) -> tuple[str | None, tuple[str, ...]]:
    """The upstream id the gateway model streams back as (providers.<p>.
    upstream_models), and the provider's other models that stream as it too."""
    try:
        p = settings.load_config().providers.get(provider)
    except Exception:
        return None, ()   # a broken config only loses the upstream-id match
    upstream = p.upstream_models.get(model) if p is not None else None
    if not upstream:
        return None, ()
    return upstream, tuple(sorted(k for k, v in p.upstream_models.items() if v == upstream and k != model))


def overlapping(trials: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """Pairs of trials whose agent windows overlap in time: a match by time
    cannot split their calls."""
    spans = sorted((s, e, t["trial_name"]) for t in trials for s, e in t["windows"])
    pairs: set[tuple[str, str]] = set()
    for i, (_, end, name) in enumerate(spans):
        for start, _, other in spans[i + 1:]:
            if start >= end:
                break
            if other != name:
                pairs.add((min(name, other), max(name, other)))
    return sorted(pairs)


def match(rows: Sequence[SpendRow], trials: Sequence[Mapping[str, Any]], skew_sec: float = DEFAULT_SKEW_SEC,
          model: str | None = None, upstream: str | None = None, shared: Sequence[str] = ()) -> Matching:
    """Each row to the one trial whose agent window (trials[i]['windows'],
    UTC pairs) holds its start time, within skew_sec either side. A row
    whose model is the upstream id (upstream) counts as the arm's, unless
    other gateway models stream as it too (shared)."""
    pad = timedelta(seconds=max(float(skew_sec), 0.0))
    out = Matching()
    for row in rows:
        if model and row.model and not _same_model(row.model, model):
            if not (upstream and _streams_as(row.model, upstream)):
                out.other_models.append(row)
                continue
            if shared:
                out.shared_upstream.append(row)
                continue
        near = [t for t in trials if any(s - pad <= row.start <= e + pad for s, e in t["windows"])]
        if len(near) > 1:
            # Back-to-back trials whose padded windows touch: the one that holds it unpadded.
            inside = [t for t in near if any(s <= row.start <= e for s, e in t["windows"])]
            if len(inside) == 1:
                near = inside
        if not near:
            out.unmatched.append(row)
        elif len(near) > 1:
            out.ambiguous.append((row, [t["trial_name"] for t in near]))
        else:
            out.by_trial.setdefault(near[0]["trial_key"], []).append(row)
    return out


def totals(rows: Sequence[SpendRow]) -> dict[str, Any]:
    return {"calls": len(rows), "n_input": sum(r.prompt for r in rows), "n_cache": sum(r.cached for r in rows),
            "n_output": sum(r.completion for r in rows), "cost_usd": round(math.fsum(r.cost for r in rows), 9)}


def rows_sha(rows: Sequence[SpendRow]) -> str:
    """Which spend-log rows a correction sums, whatever file they came in."""
    keys = sorted([r.request_id, r.start.strftime(_TIME_FORMAT), r.prompt, r.cached, r.completion, r.cost]
                  for r in rows)
    return hashlib.sha256(canonical_json(keys).encode("utf-8")).hexdigest()[:16]


# What the ledger applies -----------------------------------------------------------

def same_correction(held: Mapping[str, Any] | None, entry: Mapping[str, Any]) -> bool:
    """True when writing entry would change nothing: the correction in force
    (or none, for a 'clear') already says it."""
    if entry["action"] == "clear":
        return held is None
    if held is None:
        return False
    try:
        note = json.loads(held.get("note_json") or "{}")
    except ValueError:
        note = {}
    return ((held["n_input"], held["n_cache"], held["n_output"], held["calls"])
            == (entry["n_input"], entry["n_cache"], entry["n_output"], entry["calls"])
            and math.isclose(held["cost_usd"] or 0.0, entry["cost_usd"] or 0.0, rel_tol=0, abs_tol=1e-9)
            and note.get("rows_sha") == (entry.get("note") or {}).get("rows_sha"))


def corrected_flag(fix: Mapping[str, Any], recorded: tuple[Any, Any, Any], recorded_cost: float | None,
                   price: settings.Price | None, superseded: Sequence[str] = ()) -> dict[str, Any]:
    """TrialRow.flags[CORRECTED_FLAG]: what the trial's figures now come from,
    what the harness had recorded, the corrected tokens at list price as a
    check on the gateway's own prices, and the ingest checks on the
    harness's figures that the correction set aside."""
    try:
        note = json.loads(fix.get("note_json") or "{}")
    except ValueError:
        note = {}
    n_input, n_cache, n_output = int(fix["n_input"] or 0), int(fix["n_cache"] or 0), int(fix["n_output"] or 0)
    return {"source": fix["source"], "calls": fix["calls"], "cost_usd": fix["cost_usd"],
            "list_price_usd": (round(pricing.unit_cost(price, n_input, n_cache, n_output), 6)
                               if price is not None else None),
            "recorded": {"n_input": recorded[0], "n_cache": recorded[1], "n_output": recorded[2],
                         "cost_usd": recorded_cost},
            "file": note.get("file"), "imported_at": fix["created_at"], "superseded": list(superseded)}


def is_upper_bound(row: Mapping[str, Any], arms: frozenset[tuple[str, str, str]]) -> bool:
    """An uncorrected trial of an arm (same provider and model) that has
    corrections elsewhere, whose harness recorded prompt tokens but no cache
    reads: it was priced as all fresh input."""
    return ((row["arm"], row["provider"], row["model"]) in arms and not row["n_cache"]
            and (row["n_input"] or 0) > 0)


def upper_bound_flag() -> dict[str, Any]:
    return {"reason": "the gateway reported no cached tokens and its spend log has not been imported for this trial"}


# Planning and writing --------------------------------------------------------------

def _price(provider: str, model: str) -> settings.Price | None:
    return pricing.lookup(pricing.load_table(), provider, model)


def plan(ledger: Any, path: Path, run: str, arm: str, skew_sec: float = DEFAULT_SKEW_SEC) -> dict[str, Any]:
    """What an import would do, without writing: every trial of the arm with
    its rows, its cost before and after, and the corrections to write
    (summary['entries']). Raises KeyError for an unknown run or arm and
    ValueError for an arm that did not run through a gateway or a file it
    cannot read."""
    log = read_log(path)
    run_key = ledger.resolve_run(run)
    trials = ledger.arm_trials(run_key, arm)
    if not trials:
        raise KeyError(f"No trials of arm '{arm}' in run {run_key}")
    direct = sorted({t["provider"] for t in trials if not t["gateway"]})
    if direct:
        raise ValueError(f"Arm '{arm}' did not run through a gateway in {run_key} (provider "
                         f"{', '.join(direct)}, no ${{VAR}} base_url); its cost is not corrected from a spend log")
    together = overlapping(trials)
    if together:
        shown = ", ".join(f"{a} and {b}" for a, b in together[:5]) + (
            f" (and {len(together) - 5} more pairs)" if len(together) > 5 else "")
        raise ValueError(f"Trials of arm '{arm}' ran at the same time in {run_key} ({shown}; n_concurrent or "
                         "n_attempts above 1): matching by time cannot split their calls. Only a run that ran one "
                         "trial of the arm at a time, such as a task race with n_attempts 1, can be corrected")
    upstream, shared = upstream_ids(trials[0]["provider"], trials[0]["model"])
    m = match(log.rows, trials, skew_sec, model=trials[0]["model"], upstream=upstream, shared=shared)
    file_totals = totals(log.rows)
    price = _price(trials[0]["provider"], trials[0]["model"])
    per_trial, entries, warnings = [], [], []
    for t in trials:
        rows = m.by_trial.get(t["trial_key"], [])
        held = t["correction"]
        before = held["cost_usd"] if held is not None else t["cost_usd"]
        item: dict[str, Any] = {
            "trial_key": t["trial_key"], "trial_name": t["trial_name"], "task": t["task"], "outcome": t["outcome"],
            "rows": len(rows), "recorded": {k: t[k] for k in ("n_input", "n_cache", "n_output", "cost_usd")},
            "before_usd": before, "after_usd": before, "after": None, "list_price_usd": None,
            "windows": len(t["windows"])}
        if not rows:
            # A correction from an earlier import stays in force; without one, the trial is flagged.
            item["status"] = ("kept" if held is not None else "no rows: upper bound"
                              if not t["n_cache"] and (t["n_input"] or 0) > 0 else "no rows")
            per_trial.append(item)
            continue
        fig = totals(rows)
        entry = {"trial_key": t["trial_key"], "action": "correct", "source": SOURCE, **fig,
                 "note": {"file": log.name, "file_sha256": log.sha, "file_rows": len(log.rows), "rows": len(rows),
                          "rows_sha": rows_sha(rows), "file_totals": file_totals, "run_key": run_key, "arm": arm,
                          "skew_sec": skew_sec}}
        unchanged = same_correction(held, entry)
        entries.append(entry)   # the ledger skips an unchanged one too, so a plan gone stale cannot double up
        item.update(after=fig, after_usd=fig["cost_usd"],
                    list_price_usd=(round(pricing.unit_cost(price, fig["n_input"], fig["n_cache"], fig["n_output"]),
                                          6) if price is not None else None),
                    status="unchanged" if unchanged else "replaces" if held is not None else "new")
        recorded_in = t["n_input"] or 0
        if recorded_in and abs(fig["n_input"] / recorded_in - 1) > INPUT_TOLERANCE:
            warnings.append(f"{t['trial_name']}: the gateway logged {fig['n_input']:,} prompt tokens, Claude Code "
                            f"recorded {recorded_in:,}. Check that the export holds this arm's calls only, and all "
                            "of them.")
        per_trial.append(item)
    no_window = [t["trial_name"] for t in trials if not t["windows"]]
    if no_window:
        warnings.append(f"{len(no_window)} trial(s) have no agent execution window recorded (killed before their "
                        f"result was written), so no row can be placed in them: {', '.join(no_window)}.")
    if m.unmatched:
        warnings.append(f"{len(m.unmatched)} row(s) start in no trial's agent window (with {skew_sec:g} s either "
                        "side): calls outside the run, or another arm's traffic. They are not counted.")
    if m.other_models:
        names = sorted({r.model or "" for r in m.other_models})
        warnings.append(f"{len(m.other_models)} row(s) are for other models ({', '.join(names)}) and were left out. "
                        "The model column should be LiteLLM's model_group, the gateway alias the arm names.")
    if m.shared_upstream:
        warnings.append(f"{len(m.shared_upstream)} row(s) carry the upstream id {upstream}, which "
                        f"{', '.join([trials[0]['model'], *shared])} all stream as, so that column cannot tell "
                        "their calls apart; they were left out. Export LiteLLM's model_group as the model column.")
    if log.duplicates:
        warnings.append(f"{log.duplicates} row(s) repeat a request id already read and were left out.")
    if m.ambiguous:
        warnings.append(f"{len(m.ambiguous)} row(s) fit more than one trial's window; nothing is written until "
                        "each row fits one (a smaller --skew, or check the gateway's clock).")
    before_total = math.fsum(i["before_usd"] or 0.0 for i in per_trial)
    after_total = math.fsum(i["after_usd"] or 0.0 for i in per_trial)
    return {
        "file": {"name": log.name, "sha256": log.sha, "rows": len(log.rows), "duplicates": log.duplicates,
                 **{k: file_totals[k] for k in ("n_input", "n_cache", "n_output", "cost_usd")}},
        "run_key": run_key, "arm": arm, "provider": trials[0]["provider"], "model": trials[0]["model"],
        "skew_sec": skew_sec, "trials": per_trial,
        "matched": totals([r for rows in m.by_trial.values() for r in rows]),
        "unmatched": [_row_text(r) for r in m.unmatched],
        "ambiguous": [{**_row_text(r), "trials": names} for r, names in m.ambiguous],
        "other_models": len(m.other_models), "shared_upstream": len(m.shared_upstream),
        "before_usd": round(before_total, 6), "after_usd": round(after_total, 6),
        "entries": entries, "writable": not m.ambiguous, "warnings": warnings,
    }


def plan_clear(ledger: Any, run: str, arm: str) -> dict[str, Any]:
    """The 'clear' rows that take back every correction of the arm in the run."""
    run_key = ledger.resolve_run(run)
    trials = ledger.arm_trials(run_key, arm)
    if not trials:
        raise KeyError(f"No trials of arm '{arm}' in run {run_key}")
    held = [t for t in trials if t["correction"] is not None]
    return {"run_key": run_key, "arm": arm, "writable": True, "warnings": [],
            "entries": [{"trial_key": t["trial_key"], "action": "clear", "source": SOURCE,
                         "note": {"run_key": run_key, "arm": arm}} for t in held],
            "cleared": [t["trial_name"] for t in held]}


def _row_text(r: SpendRow) -> dict[str, Any]:
    return {"line": r.line, "start_utc": r.start.strftime(_TIME_FORMAT), "request_id": r.request_id,
            "prompt_tokens": r.prompt, "cost_usd": r.cost}


def _usd(value: float | None) -> str:
    return "-" if value is None else f"${value:.4f}"


def _n(value: Any) -> str:
    return "-" if value is None else f"{int(value):,}"


def format_summary(s: Mapping[str, Any]) -> str:
    """The plan as plain text, one line per trial."""
    if "cleared" in s:
        names = s["cleared"]
        return (f"Run {s['run_key']}, arm {s['arm']}: "
                + (f"{len(names)} correction(s) to take back: {', '.join(names)}" if names
                   else "no corrections in force"))
    f = s["file"]
    lines = [f"Spend log {f['name']} (sha256 {f['sha256']}): {f['rows']} rows, {_n(f['n_input'])} prompt tokens "
             f"({_n(f['n_cache'])} cached), {_n(f['n_output'])} output, {_usd(f['cost_usd'])}",
             f"Run {s['run_key']}, arm {s['arm']} ({s['provider']}:{s['model']}): {len(s['trials'])} trials, "
             f"rows matched within {s['skew_sec']:g} s of each agent window", ""]
    header = ("trial", "rows", "input", "cached", "output", "before", "after", "list price", "")
    table = [header]
    for t in s["trials"]:
        a = t["after"] or {}
        table.append((t["trial_name"], str(t["rows"]), _n(a.get("n_input")), _n(a.get("n_cache")),
                      _n(a.get("n_output")), _usd(t["before_usd"]), _usd(t["after_usd"]),
                      _usd(t["list_price_usd"]), t["status"]))
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for row in table:
        lines.append("  ".join(cell.ljust(w) if i == 0 or i == len(row) - 1 else cell.rjust(w)
                               for i, (cell, w) in enumerate(zip(row, widths))).rstrip())
    m = s["matched"]
    lines += ["", f"Matched {m['calls']} rows: {_n(m['n_input'])} prompt ({_n(m['n_cache'])} cached), "
                  f"{_n(m['n_output'])} output, {_usd(m['cost_usd'])}. Unmatched {len(s['unmatched'])}, "
                  f"ambiguous {len(s['ambiguous'])}, other models {s['other_models']}.",
              f"Arm cost in this run: {_usd(s['before_usd'])} before, {_usd(s['after_usd'])} after."]
    for r in s["unmatched"][:20]:
        lines.append(f"  unmatched line {r['line']}: {r['start_utc']} {r['request_id'] or '(no id)'} "
                     f"{_n(r['prompt_tokens'])} prompt, {_usd(r['cost_usd'])}")
    if len(s["unmatched"]) > 20:
        lines.append(f"  ... and {len(s['unmatched']) - 20} more")
    for r in s["ambiguous"]:
        lines.append(f"  ambiguous line {r['line']}: {r['start_utc']} {r['request_id'] or '(no id)'} fits "
                     f"{', '.join(r['trials'])}")
    lines += [f"Note: {w}" for w in s["warnings"]]
    return "\n".join(lines)


def cli(ledger: Any, args: argparse.Namespace) -> int:
    """python -m bench.ledger gateway-spend (bench.ledger.main parses the arguments)."""
    try:
        if args.clear:
            summary = plan_clear(ledger, args.run, args.arm)
        elif not args.csv:
            print("Give the spend log's CSV file, or --clear to take the arm's corrections back")
            return 1
        else:
            summary = plan(ledger, Path(args.csv), args.run, args.arm, args.skew)
    except (OSError, UnicodeDecodeError, ValueError, KeyError) as exc:
        print(f"gateway-spend: {exc.args[0] if isinstance(exc, KeyError) and exc.args else exc}")
        return 1
    entries = summary.pop("entries")
    if "cleared" in summary:
        changes = len(entries)
    else:
        changes = sum(1 for t in summary["trials"] if t["status"] in ("new", "replaces"))
    if args.json:
        print(json.dumps({**summary, "corrections_to_write": changes}, indent=2, ensure_ascii=False))
    else:
        print(format_summary(summary))
    if args.dry_run:
        print(f"Dry run: nothing written ({changes} correction(s) would be).")
        return 0
    if not summary["writable"]:
        print("Nothing written: some rows fit more than one trial.")
        return 1
    counts = ledger.add_cost_corrections(entries)
    print(f"Wrote {counts['written']} correction(s); {counts['unchanged']} already in force.")
    return 0
