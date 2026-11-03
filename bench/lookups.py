"""Whether an agent looked up the benchmark's own answer.

Agents cannot see the task's tests in the container, but they can reach the
internet, where Terminal-Bench publishes every task with its reference
solution and its tests: on GitHub (laude-institute/terminal-bench,
harbor-framework/terminal-bench*) and on Hugging Face
(harborframework/terminal-bench*). A trial whose agent fetched those answer
files and then used them counts as a fail (policy of 2026-11-02): it stays in
the ledger, marked, never excluded or run again. Fetching the real-world
upstream fix a task is built on (the bottle commit behind
fix-code-vulnerability) is allowed, and so is a README.

The scan reads the agent's tool calls in order, with what each returned,
from Harbor's ATIF trajectory (agent/trajectory.json), which every installed
harness gets, so one reader serves them all. A call is an answer fetch when
it names a benchmark source and an answer file (solution.sh, solve.sh,
solution/, tests/, test_outputs.py, task.yaml, run-tests.sh), or, after a
bulk download of a source (a git clone, an archive, a dataset snapshot),
when it reads answer files from that local copy. The fetch was used when:
  - something came back: not empty, not only a 404 or an error, and
  - the agent changed something afterwards: a write or edit tool, a shell
    command that writes outside /tmp (a redirect, cp, mv, sed -i, a build,
    a script that opens a file for writing), or one that runs a script
    (bash solve.sh, curl ... | bash). The fetch call itself counts when it
    goes on to run or write what it fetched. A fetch after the agent's last
    change cannot have changed its answer.
These are heuristics, so a person's verdict per trial in
config/answer_lookups.yaml wins over them (with_verdict). A trial without an
ATIF trajectory gets a plain text search of its harness's files; a hit there
cannot be judged, so it is recorded with used null for a person to decide.

Pure apart from reading files.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import settings

VERDICTS = ("lookup", "allowed")
# flags_json keys: what the scan found, and a person's verdict.
FLAG = "answer_lookup"
VERDICT_FLAG = "answer_lookup_verdict"
MAX_URLS = 3
URL_CHARS = 160
NOTE_CHARS = 300
# The plain-text fallback reads no file larger than this, and looks this far
# either side of a source for an answer file.
FALLBACK_MAX_BYTES = 20_000_000
FALLBACK_WINDOW = 2000
# Output this short that reads as an error is a fetch that brought nothing back.
ERROR_ONLY_CHARS = 400

log = logging.getLogger(__name__)

# The benchmark's own repositories and datasets, by any host or client.
SOURCE = re.compile(r"(?:laude-institute|harbor-framework|harborframework)/terminal-bench[\w.-]*", re.I)
ANSWER = re.compile(r"(?<![\w-])(?:solution\.sh|solve\.sh|solution/|tests/|test_outputs\.py|task\.yaml|"
                    r"run-tests\.sh)", re.I)
BULK = re.compile(r"\bgit\s+(?:clone|fetch|pull)\b|\bgh\s+repo\s+clone\b|codeload\.github|/archive/|"
                  r"/(?:zip|tar)ball\b|snapshot_download|load_dataset|\b(?:huggingface-cli|hf)\s+download\b", re.I)
# A path into a local copy of the benchmark.
LOCAL = re.compile(r"terminal-bench[\w.-]*/|original-tasks/", re.I)
# Every Terminal-Bench task, solution and test file carries the canary.
CANARY = re.compile(r"terminal-bench-canary|BENCHMARK DATA SHOULD NEVER APPEAR|Terminal-Bench Canary String", re.I)
_URL = re.compile(r"https?://[^\s'\"`<>()\\]+")
_ERROR = re.compile(r"404|not found|no such file|could not|failed|error|exception|traceback|timed? ?out|denied|"
                    r"refused|rate limit", re.I)

# What changes the task's state. Paths under /tmp, /dev and /proc are scratch:
# they are blanked first, so a redirect or a copy into them is not a change.
_SCRATCH = re.compile(r"(?:/tmp|/dev|/proc)/[^\s'\";|&)]*|/dev/null")
_MUTATES = re.compile(r"""
    (?<![<0-9&>-])>>?[ \t]*(?!\d+\b)['"]?[\w~$./-]       # a redirect into a file (not 2>, not a number)
  | \b(?:cp|mv|rm|tee|touch|install|ln|patch|truncate|rsync|unzip)[ \t]+(?:-\S+[ \t]+)*[^\s;&|-]
  | \bsed[ \t]+-i | \bperl[ \t]+-\w*i
  | \b(?:make|cmake|gcc|g\+\+|cargo|go[ \t]+build|npm[ \t]+(?:i|install|run)|pip3?[ \t]+install|
       apt(?:-get)?[ \t]+install|git[ \t]+(?:apply|commit|checkout|am))\b
  | open\([^)\n]*,[ \t]*(?:mode[ \t]*=[ \t]*)?\\?['"][wax] | \.write_(?:text|bytes)\( | writeFileSync | \.to_csv\(
  | json\.dump\( | \.save\(
""", re.X)
# Running a script or piping into a shell changes the task's state too, wherever
# the script lives, so this is matched before scratch paths are blanked.
_RUNS_SCRIPT = re.compile(r"""
    \|[ \t]*(?:sudo[ \t]+)?(?:ba|z|da)?sh\b                          # curl ... | bash
  | (?<![\w./-])(?:ba|z|da)?sh[ \t]+(?:-[a-bd-z]\w*[ \t]+)*[\w~$./]  # bash solve.sh (not bash -c '...')
  | (?:^|[;&|(][ \t]*)\.?/[\w~$./-]+\.sh\b                          # ./solve.sh
  | \bsource[ \t]+[\w~$./]
""", re.X | re.M)
# Tools that write files themselves, by name (todo lists are notes, not files).
_WRITE_TOOL = re.compile(r"write|edit|patch|str_replace|create|insert", re.I)
# Argument fields that describe a call rather than run it.
_PROSE_KEYS = frozenset({"prompt", "description", "explanation", "reason", "title", "summary", "url"})


@dataclass(frozen=True)
class ToolCall:
    """One tool call, in order: its name, all its argument text, the part
    that runs (prose fields left out), and what it returned (None when the
    log kept no output)."""
    name: str
    text: str
    runs: str
    output: str | None


def _strings(value: Any, skip: frozenset[str] = frozenset()) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            if str(k).lower() not in skip:
                yield from _strings(v, skip)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for v in value:
            yield from _strings(v, skip)


def _content(value: Any) -> str:
    return value if isinstance(value, str) else "\n".join(_strings(value))


def atif_calls(data: Any) -> list[ToolCall] | None:
    """The tool calls of an ATIF trajectory with their outputs. An output is
    matched by source_call_id, else by position within its step (Harbor's
    mini-swe-agent and terminus-2 trajectories leave the id out); a step
    whose outputs cannot be matched gives each of its calls them all."""
    if not isinstance(data, Mapping) or not isinstance(data.get("steps"), list):
        return None
    calls: list[ToolCall] = []
    for step in data["steps"]:
        if not isinstance(step, Mapping):
            continue
        tools = [t for t in (step.get("tool_calls") or []) if isinstance(t, Mapping)]
        if not tools:
            continue
        observation = step.get("observation") if isinstance(step.get("observation"), Mapping) else {}
        results = [r for r in (observation.get("results") or []) if isinstance(r, Mapping)]
        by_id = {r.get("source_call_id"): _content(r.get("content")) for r in results if r.get("source_call_id")}
        joined = "\n".join(_content(r.get("content")) for r in results) if results else None
        for i, tool in enumerate(tools):
            args = tool.get("arguments")
            if tool.get("tool_call_id") in by_id:
                output: str | None = by_id[tool["tool_call_id"]]
            elif results and not by_id and len(results) == len(tools):
                output = _content(results[i].get("content"))
            else:
                output = joined
            calls.append(ToolCall(name=str(tool.get("function_name") or ""), text="\n".join(_strings(args)),
                                  runs="\n".join(_strings(args, _PROSE_KEYS)), output=output))
    return calls


def _received(output: str | None) -> bool | None:
    """Whether a fetch brought something back; None when nothing was logged."""
    if output is None:
        return None
    text = output.strip()
    if not text or text == "(no output)":
        return False
    if CANARY.search(text):
        return True
    return not (len(text) < ERROR_ONLY_CHARS and _ERROR.search(text))


def mutates(call: ToolCall) -> bool:
    """The call changed something outside scratch space (see _MUTATES)."""
    name = call.name.lower()
    if "todo" not in name and _WRITE_TOOL.search(name):
        return True
    return bool(_RUNS_SCRIPT.search(call.runs) or _MUTATES.search(_SCRATCH.sub(" ", call.runs)))


def _applies(call: ToolCall) -> bool:
    """The fetch call itself goes on to change something after naming the
    answer files (curl ... | bash, or curl -o /app/solve.sh && bash ...)."""
    m = SOURCE.search(call.runs) or ANSWER.search(call.runs)
    return mutates(ToolCall("", "", call.runs[m.start():], None)) if m else False


def _urls(text: str) -> list[str]:
    urls = [u.rstrip(".,;:'\"") for u in _URL.findall(text) if SOURCE.search(u)]
    if not urls:  # a client call (hf_hub_download, gh repo clone) names the source without a URL
        urls = [m.group(0) for m in SOURCE.finditer(text)]
    return urls


def _found(urls: Sequence[str], used: bool | None, reason: str, source: str) -> dict[str, Any]:
    seen: list[str] = []
    for u in sorted(urls, key=lambda u: not ANSWER.search(u)):  # the answer files' own URLs first
        u = u[:URL_CHARS]
        if u not in seen:
            seen.append(u)
    return {"urls": seen[:MAX_URLS], "used": used, "reason": reason, "source": source}


def judge(calls: Sequence[ToolCall], source: str = "agent/trajectory.json") -> dict[str, Any] | None:
    """The answer_lookup flag for a trial's tool calls, or None when it
    fetched no answer file. Used when any fetch both brought something
    back and was followed by a change (see the module docstring)."""
    hits: list[tuple[int, list[str]]] = []
    bulk: list[str] | None = None  # the source URLs of the first bulk download so far
    for i, call in enumerate(calls):
        named = SOURCE.search(call.text)
        if named and BULK.search(call.runs) and bulk is None:
            bulk = _urls(call.text)
        if named and ANSWER.search(call.text):
            hits.append((i, _urls(call.text)))
        elif bulk is not None and LOCAL.search(call.runs) and (
                ANSWER.search(call.runs)
                or (call.output and ANSWER.search(call.output) and CANARY.search(call.output))):
            hits.append((i, bulk))
    if not hits:
        return None
    urls = [u for _i, found in hits for u in found]
    came_back = [i for i, _u in hits if _received(calls[i].output) is not False]
    if not came_back:
        return _found(urls, False, "nothing came back (empty, a 404 or an error)", source)
    first = came_back[0]
    last_change = max((j for j, call in enumerate(calls) if mutates(call)), default=-1)
    if last_change <= first:
        if _applies(calls[first]):
            return _found(urls, True, "the answer files came back and the same call ran or wrote them", source)
        return _found(urls, False, "fetched after the agent's last change, so it did not shape the answer", source)
    return _found(urls, True, "the answer files came back and the agent changed files afterwards", source)


def _fallback(agent: Path) -> dict[str, Any] | None:
    """A plain text search of the harness's own files, for a trial without
    an ATIF trajectory: a hit cannot be judged, so used is None."""
    for path in sorted(agent.rglob("*")):
        rel = path.relative_to(agent)
        if "snapshot" in rel.parts or not path.is_file() or path.suffix in (".pack", ".idx"):
            continue
        try:
            if path.stat().st_size > FALLBACK_MAX_BYTES:
                continue
            text = path.read_bytes().decode("utf-8", "ignore")
        except OSError:
            continue
        for m in SOURCE.finditer(text):
            window = text[max(m.start() - FALLBACK_WINDOW, 0): m.end() + FALLBACK_WINDOW]
            if ANSWER.search(window):
                return _found(_urls(window), None, "found in the harness's log, which has no step-by-step "
                              "record to judge use; check by hand", "agent/" + rel.as_posix())
    return None


def scan(agent: Path) -> dict[str, Any] | None:
    """The answer_lookup flag for a trial's agent folder, or None."""
    trajectory = Path(agent) / "trajectory.json"
    calls = None
    if trajectory.is_file():
        try:
            calls = atif_calls(json.loads(trajectory.read_bytes()))
        except (OSError, ValueError):
            calls = None
    if calls is not None:
        return judge(calls)
    return _fallback(Path(agent))


# Verdicts by hand -----------------------------------------------------------------

_cache: dict[str, Any] = {"key": None, "verdicts": {}}


def load_verdicts(path: Path | None = None) -> dict[str, dict[str, str]]:
    """trial_key -> {"verdict": lookup|allowed, "note"} from
    config/answer_lookups.yaml; {} when the file is missing. An entry with
    another verdict is skipped with a warning, and so is an unreadable file:
    neither may stop ingestion."""
    path = Path(path or settings.ANSWER_LOOKUPS)
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if _cache["key"] == key:
        return _cache["verdicts"]
    out: dict[str, dict[str, str]] = {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("%s could not be read: %s", path, exc)
        data = {}
    trials = data.get("trials") if isinstance(data, Mapping) else None
    if trials is not None and not isinstance(trials, Mapping):
        log.warning("%s: trials is a %s, not a mapping of trial_key to verdict", path, type(trials).__name__)
        trials = None
    for trial_key, entry in (trials or {}).items():
        verdict = entry.get("verdict") if isinstance(entry, Mapping) else entry
        if verdict not in VERDICTS:
            log.warning("%s: %s has verdict %r, not one of %s", path, trial_key, verdict, ", ".join(VERDICTS))
            continue
        note = entry.get("note") if isinstance(entry, Mapping) else None
        out[str(trial_key)] = {"verdict": verdict, "note": str(note or "")[:NOTE_CHARS]}
    _cache.update(key=key, verdicts=out)
    return out


def with_verdict(flags: Mapping[str, Any], trial_key: str,
                 verdicts: Mapping[str, Mapping[str, str]] | None = None) -> dict[str, Any]:
    """The flags with this trial's verdict by hand set, or cleared when the
    file no longer names it."""
    out = {k: v for k, v in flags.items() if k != VERDICT_FLAG}
    entry = (load_verdicts() if verdicts is None else verdicts).get(trial_key)
    if entry:
        out[VERDICT_FLAG] = {"verdict": entry["verdict"], "note": entry.get("note") or ""}
    return out
