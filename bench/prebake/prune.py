"""Removing pre-baked images that nothing needs any more.

A candidate is an image whose repository starts with '<prefix>/' or
'benchbake-staging/' AND that carries the org.etb.prebake.bake-format label,
and that is
- a staging tag (a build that never finished; never while a build runs),
- built for another Harbor than the one installed (every trial would
  refuse it), or
- for a (harness, version, recipe) no enabled arm of any profile wants, and
  baked more than prebake.keep_days ago (or any age with everything=True).
Task images, the research prototypes' benchprobe/ images and every other
image are never candidates; benchprobe/ only with include_probes, and the
CLI and the dashboard list them first. Removal is `docker rmi <tag>`, never
forced, so an image a container still uses stays; then `docker image prune
--filter label=org.etb.prebake.bake-format` removes only this builder's own
dangling layers. A dry run (the default everywhere) removes nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..environments.prebaked import LABEL_NS, harbor_version
from ..settings import Config
from . import builder, inventory
from .inventory import DockerError, docker, tail

BAKE_LABEL = LABEL_NS + "bake-format"
PROBE_PREFIX = "benchprobe/"


@dataclass(frozen=True)
class Candidate:
    ref: str
    id: str
    kind: str            # bake | staging | probe
    reason: str
    baked_at: str | None = None
    size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _list(filters: Iterable[str] = ()) -> dict[str, str]:
    args = ["image", "ls", "--no-trunc", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}"]
    for f in filters:
        args += ["--filter", f]
    r = docker(args)
    if r.returncode != 0:
        raise DockerError(f"docker image ls failed: {tail(r.stderr) or 'no output'}")
    out = {}
    for line in r.stdout.splitlines():
        ref, tab, image_id = line.strip().partition("\t")
        if tab and "<none>" not in ref:
            out[ref] = image_id.strip()
    return out


def _age_days(stamp: str | None) -> float | None:
    try:
        when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).total_seconds() / 86400


def candidates(cfg: Config, *, keep_days: float | None = None, include_probes: bool = False,
               everything: bool = False, wanted: set[tuple[str, str, str]] | None = None
               ) -> tuple[list[Candidate], list[str]]:
    """(what prune would remove, notes). wanted defaults to every enabled
    arm's (family, version, recipe) in every profile."""
    keep = cfg.prebake.keep_days if keep_days is None else keep_days
    notes: list[str] = []
    if wanted is None and not everything:
        wanted, notes = inventory.wanted_recipes(cfg)
    prefix = cfg.prebake.prefix.strip("/") + "/"
    labelled = _list([f"label={BAKE_LABEL}"])
    ours = [ref for ref in labelled if ref.startswith((prefix, builder.STAGING + "/"))]
    raw = inventory.inspect_raw(ours, refresh=True)
    installed = harbor_version()
    building = builder.build_active()
    out: list[Candidate] = []
    for ref in ours:
        data = raw.get(ref)
        labels = inventory.labels_of(data)
        if BAKE_LABEL not in labels:
            continue  # untagged since the listing, or not one of the builder's images
        baked_at = labels.get(LABEL_NS + "baked-at")
        common = dict(ref=ref, id=labelled[ref], baked_at=baked_at, size=(data or {}).get("Size"))
        if ref.startswith(builder.STAGING + "/"):
            if building:
                notes.append(f"{ref}: kept, a build is running and may still be verifying it")
            else:
                out.append(Candidate(kind="staging", reason="a staging tag left by a build that did not finish",
                                     **common))
            continue
        built_for = labels.get(LABEL_NS + "harbor-version")
        if built_for != installed:
            out.append(Candidate(kind="bake", reason=f"built for Harbor {built_for}; {installed} is installed, "
                                                     "so every trial would refuse it", **common))
            continue
        if everything:
            out.append(Candidate(kind="bake", reason="every image under the prefix was asked for", **common))
            continue
        key = (labels.get(LABEL_NS + "harness", ""), labels.get(LABEL_NS + "harness-version", ""),
               labels.get(LABEL_NS + "recipe-sha256", ""))
        if key in (wanted or set()):
            continue
        age = _age_days(baked_at)
        if age is None or age > keep:
            out.append(Candidate(kind="bake", reason=f"no enabled arm of any profile runs {key[0]} {key[1]} with "
                                                     f"recipe {key[2][:10]}, and it was baked "
                                                     f"{'at an unknown time' if age is None else f'{age:.0f} days ago'}"
                                                     f" (keep_days {keep:g})", **common))
    if include_probes:
        for ref, image_id in _list().items():
            if ref.startswith(PROBE_PREFIX):
                out.append(Candidate(ref=ref, id=image_id, kind="probe",
                                     reason="a research prototype image (--include-probes)"))
    return out, notes


def prune(found: list[Candidate], *, dry_run: bool = True, run_active: Callable[[], bool] | None = None
          ) -> dict[str, Any]:
    """Removes the candidates' tags (nothing on a dry run), then this
    builder's own dangling layers. Holds the build lock: never during a build
    or a run."""
    result: dict[str, Any] = {"dry_run": dry_run, "candidates": [c.to_dict() for c in found], "removed": [],
                              "failed": {}, "dangling": ""}
    if dry_run or not found:
        return result
    with builder.build_lock(run_active):
        for c in found:
            if not c.ref.startswith(PROBE_PREFIX) and not inventory.bake_label(
                    inventory.inspect_raw([c.ref], refresh=True).get(c.ref), "bake-format"):
                result["failed"][c.ref] = "no longer carries the bake-format label; left alone"
                continue
            r = docker(["rmi", c.ref], timeout=300)
            if r.returncode == 0:
                result["removed"].append(c.ref)
            else:
                result["failed"][c.ref] = tail(r.stderr) or f"docker rmi exited with {r.returncode}"
        r = docker(["image", "prune", "-f", "--filter", f"label={BAKE_LABEL}"], timeout=600)
        result["dangling"] = tail(r.stdout or r.stderr, 300)
    inventory.forget_images()
    return result


# Build cache -------------------------------------------------------------------------
# BuildKit keeps a cache record for every step of every build, --no-cache or not,
# and those records pin the layers on disk after the image itself is removed. A
# task race builds and removes a task's images every task, so each build removes
# the records it created (BuildManager with prune_cache): the ids listed after it
# that were not there before it. Records are never pruned by age or in bulk, so
# other projects' cache stays; a record another build created during the same
# minutes would be the only one at risk, and only its cache, never an image.
#
# Both listings must come from the same command (buildx prints full ids, system df
# cuts them to 12 characters), ids are compared on their first 12 characters, and a
# result that does not look like one build's records (none of the old records left,
# or more new ones than a build of that many images makes) prunes nothing. BuildKit
# keeps a record a child still uses, so children go first, and the prune lists the
# cache again after each pass and repeats for what is left; 'pruned' counts the
# records that are gone, never the commands that succeeded.

CACHE_LIST_TIMEOUT_SEC = 120
CACHE_PRUNE_TIMEOUT_SEC = 120
CACHE_KEY_LEN = 12             # system df -v prints this much of an id
CACHE_MAX_NEW_PER_IMAGE = 300  # a build of one image makes a few dozen records
CACHE_PRUNE_PASSES = 12        # a pass removes at least the newest record of each chain left


@dataclass(frozen=True)
class CacheListing:
    """One listing of the build cache: which command made it, the record ids,
    and each record's parents (buildx only; empty from system df)."""

    source: str
    ids: frozenset[str]
    parents: dict[str, frozenset[str]]
    created: dict[str, datetime] = field(default_factory=dict)   # buildx only

    def keys(self) -> set[str]:
        return {cache_key(i) for i in self.ids}


def cache_key(record: str) -> str:
    """The part of an id both listings print."""
    return record[:CACHE_KEY_LEN]


def _parse_created(value: str) -> datetime | None:
    """A 'Created at' value as buildx prints it ('2026-10-31 11:53:01.123456789 +0000 UTC'), in UTC."""
    parts = value.split()
    if len(parts) < 2:
        return None
    clock = parts[1].partition(".")[0]
    offset = parts[2] if len(parts) > 2 and parts[2][:1] in "+-" else "+0000"
    try:
        return datetime.strptime(f"{parts[0]} {clock} {offset}", "%Y-%m-%d %H:%M:%S %z").astimezone(timezone.utc)
    except ValueError:
        return None


def _cache_records_buildx(text: str) -> dict[str, dict[str, Any]]:
    """Records from `docker buildx du --verbose`: an 'ID: <id>' line opens each
    one, 'Parent: <id>' or 'Parents:' with ' - <id>' lines name its parents, and
    'Created at:' says when it was made."""
    records: dict[str, dict[str, Any]] = {}
    current, in_parents = None, False
    for line in text.splitlines():
        stripped = line.strip()
        if in_parents and current is not None and stripped.startswith("- "):
            records[current]["parents"].add(stripped[2:].strip().rstrip("*"))
            continue
        key, sep, value = line.partition(":")
        key, value = key.strip(), value.strip().rstrip("*")
        in_parents = False
        if sep and key == "ID" and value:
            current = value
            records.setdefault(current, {"parents": set(), "created": None})
        elif sep and current is not None and key in ("Parent", "Parents"):
            in_parents = True
            records[current]["parents"].update(v.strip() for v in value.split(",") if v.strip())
        elif sep and current is not None and key == "Created at":
            records[current]["created"] = _parse_created(value)
    return records


def _cache_ids_buildx(text: str) -> set[str]:
    """IDs from `docker buildx du --verbose`."""
    return set(_cache_records_buildx(text))


def _cache_ids_df(text: str) -> set[str]:
    """IDs from the 'Build cache usage' table of `docker system df -v` (12 characters, a prefix of the full id)."""
    out, table = set(), False
    for line in text.splitlines():
        if line.startswith("CACHE ID"):
            table = True
            continue
        if table:
            if not line.strip():
                break
            out.add(line.split()[0].rstrip("*"))
    return out


def build_cache_listing() -> CacheListing:
    """The build cache records Docker holds now. Raises DockerError when they
    cannot be listed, so a prune never runs on a guess."""
    r = docker(["buildx", "du", "--verbose"], timeout=CACHE_LIST_TIMEOUT_SEC)
    if r.returncode == 0:
        records = _cache_records_buildx(r.stdout)
        return CacheListing("buildx", frozenset(records), {k: frozenset(v["parents"]) for k, v in records.items()},
                            {k: v["created"] for k, v in records.items() if v["created"] is not None})
    r = docker(["system", "df", "-v"], timeout=CACHE_LIST_TIMEOUT_SEC)
    if r.returncode != 0:
        raise DockerError(f"the build cache could not be listed: {tail(r.stderr) or 'no output'}")
    return CacheListing("df", frozenset(_cache_ids_df(r.stdout)), {})


def build_cache_ids() -> set[str]:
    """The build cache record ids Docker holds now (see build_cache_listing)."""
    return set(build_cache_listing().ids)


def prune_order(ids: Iterable[str], parents: dict[str, frozenset[str]]) -> list[str]:
    """The ids with every child before its parents (BuildKit keeps a record a
    child still uses); ids without known links in sorted order."""
    todo = sorted(dict.fromkeys(ids))
    users = {cache_key(i): 0 for i in todo}   # how many records still to prune use this one
    for i in todo:
        for p in parents.get(i, ()):
            if cache_key(p) in users:
                users[cache_key(p)] += 1
    out: list[str] = []
    while todo:
        free = [i for i in todo if users[cache_key(i)] == 0] or todo[:1]   # a cycle: take one anyway
        for i in free:
            todo.remove(i)
            out.append(i)
            for p in parents.get(i, ()):
                if cache_key(p) in users:
                    users[cache_key(p)] -= 1
    return out


def prune_build_cache(ids: Iterable[str]) -> dict[str, Any]:
    """Runs one `docker builder prune --filter id=<id>` per record, in this
    order (Docker takes one id per filter). Never a prune without an id: that
    would remove every project's cache. 'ok' counts the commands that exited
    0, which says nothing about what they removed."""
    ok, failed, error = 0, 0, None
    for record in dict.fromkeys(ids):
        if not record or not record.isalnum():
            continue  # an id Docker would read as a pattern
        r = docker(["builder", "prune", "-f", "--all", "--filter", f"id={record}"], timeout=CACHE_PRUNE_TIMEOUT_SEC)
        if r.returncode == 0:
            ok += 1
        else:
            failed += 1
            error = tail(r.stderr) or f"docker builder prune exited with {r.returncode}"
    return {"ok": ok, "failed": failed, "error": error}


def prune_new_cache(before: CacheListing, images: int,
                    window: tuple[datetime, datetime] | None = None) -> dict[str, Any]:
    """Removes the build cache records listed now that were not in before (the
    listing taken when a build of this many images started). Returns the
    figures for the build's cache record: before, source, created, pruned
    (gone afterwards), left, passes, failed and error. Raises DockerError when
    the cache cannot be listed after the build. window (UTC): only records
    created between these times count as new, for a build that ended long
    before this prune (the console closed during it); that needs buildx."""
    out: dict[str, Any] = {"before": len(before.ids), "source": before.source}
    after = build_cache_listing()
    if window is not None and after.source != "buildx":
        return {**out, "pruned": 0, "error": "not pruned: only buildx lists when each record was made"}
    if after.source != before.source:
        return {**out, "pruned": 0, "error": f"not pruned: the build cache was listed by {before.source} before "
                                            f"the build and by {after.source} after it, so the ids cannot be "
                                            "compared"}
    old = before.keys()
    if old and not old & after.keys():
        return {**out, "pruned": 0, "error": f"not pruned: none of the {len(old)} records listed before the build "
                                            "is listed after it"}
    new = sorted(i for i in after.ids if cache_key(i) not in old)
    if window is not None:
        start, end = window
        new = [i for i in new if i in after.created and start <= after.created[i] <= end]
    limit = CACHE_MAX_NEW_PER_IMAGE * max(1, images)
    if len(new) > limit:
        return {**out, "created": len(new), "pruned": 0,
                "error": f"not pruned: {len(new)} new records is more than a build of {max(1, images)} image(s) "
                         f"makes (at most {limit})"}
    target = {cache_key(i) for i in new}
    left, parents = new, after.parents
    passes, failed, error = 0, 0, None
    while left and passes < CACHE_PRUNE_PASSES:
        passes += 1
        result = prune_build_cache(prune_order(left, parents))
        failed += result["failed"]
        error = result["error"] or error
        try:
            now = build_cache_listing()
        except DockerError as exc:
            error = f"the build cache could not be listed after pruning: {exc}"
            break
        if now.source != before.source:
            error = f"the build cache was listed by {now.source} after pruning, so what is left is not known"
            break
        still = sorted(i for i in now.ids if cache_key(i) in target)
        progress = len(still) < len(left)
        left, parents = still, now.parents
        if not progress:
            break
    if left and error is None:
        error = f"{len(left)} of the {len(new)} records this build made are still there"
    return {**out, "created": len(new), "pruned": len(new) - len(left), "left": len(left), "passes": passes,
            "failed": failed, "error": error}
