"""Run queue: human-approved design versions, run one after another.

The queue is the only way a design becomes a billed run. A person admits one
version by the spec_hash they saw (add), and the worker starts it only if the
version file still hashes to that value and still passes every check under
today's config. So neither an edited file nor a changed profile can run
something nobody approved, and the assistant, which can only save drafts, has
no path in here at all.

Designs run one at a time, through the same RunManager as the LAUNCH tab, and
only while that manager is idle. Parallelism lives inside a run
(parallel_arms x n_concurrent), so Docker Desktop never carries two runs'
containers at once.

The queue is built to stop rather than spend:
- it always loads paused, so a restart never starts anything unattended;
- it pauses when a run fails or is interrupted (queue.on_failure: pause),
  whenever a run is cancelled, when Docker is down, and when the optional
  global cap could be passed, counting an allowance for trials still in
  flight when a run reaches its budget;
- it pauses when it runs dry, so a design added later waits for a person.

queue.json mirrors the in-memory state, which is authoritative: when an
antivirus scan or an editor holds the file, the write is retried on the next
change or worker pass instead of failing the request. The module is named
runqueue because a bench/queue.py would shadow the standard library's queue
module if bench/ were ever on sys.path.
"""

from __future__ import annotations

import copy
import json
import math
import re
import shutil
import sys
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from . import designs, fsutil
from .designs import DesignStore, VersionConflict
from .settings import PREBAKE_OFF, PREBAKE_REQUIRE, Config, ConfigError, QueueSettings
from .spec import RunSpec, spec_from_design
from .spec import spec_hash as hash_of

SCHEMA = 1
ON_FAILURE = ("pause", "skip")
OPEN = ("waiting", "held")          # still to run; the only items a person may remove or move
ACTIVE = ("starting", "running")
FINISHED = ("done", "budget_stopped", "failed", "cancelled", "interrupted")
ITEM_STATUSES = OPEN + ACTIVE + FINISHED

WAIT_SECONDS = 5.0
SAVE_PATIENCE = 2.0
HISTORY_SHOWN = 50
HISTORY_KEPT = 1000                 # finished items kept in queue.json

PAUSED_AFTER_RESTART = "Paused after restart. Press Start queue."
NOT_STARTED = "Not started yet. Press Start queue."
PAUSED_BY_YOU = "Paused by you"
DOCKER_DOWN = "Docker Desktop is not running"
STOPPED = "Paused: the bench stopped."

_ITEM_ID = re.compile(r"q-(\d{4,})")
_EPSILON = 1e-9  # money compared after float sums

# Run status -> (item status, what the queue does next). "policy" follows
# on_failure. A cancel is a person deciding to stop, so it always pauses.
_OUTCOMES: dict[str, tuple[str, str]] = {
    "done": ("done", "continue"),
    "done_with_errors": ("done", "continue"),
    "budget_stopped": ("budget_stopped", "continue"),
    "failed": ("failed", "policy"),
    "interrupted": ("interrupted", "policy"),
    "cancelled": ("cancelled", "pause"),
}
_ITEM_MESSAGES = {
    "done": "",
    "budget_stopped": "Stopped at its budget",
    "failed": "The run failed",
    "interrupted": "The run was interrupted",
    "cancelled": "Cancelled",
}
_PAUSE_VERBS = {"failed": "failed", "interrupted": "was interrupted", "cancelled": "was cancelled"}


class _Unset:
    def __repr__(self) -> str:
        return "UNSET"


UNSET: Any = _Unset()  # update_settings: "leave the global cap as it is"


class Manager(Protocol):
    """What the queue needs from runs.RunManager."""

    def start_spec(self, cfg: Config, spec: RunSpec) -> dict[str, Any]: ...

    def is_idle(self) -> bool: ...

    def add_listener(self, fn: Callable[[dict[str, Any]], None]) -> None: ...

    def list(self) -> list[dict[str, Any]]: ...


class ImageHook(Protocol):
    """What the queue needs from the pre-baked image builder (prebake.builder.QueueImages)."""

    def check(self, cfg: Config, spec: RunSpec) -> dict[str, Any]:
        """{todo: images to build, blocking: [why the run may not start under require]}"""
        ...

    def build(self, cfg: Config, spec: RunSpec) -> str:
        """Starts a build of the spec's missing images, or joins the one running; returns its id."""
        ...

    def status(self, build_id: str) -> dict[str, Any] | None: ...


class RunQueue:
    """The persisted queue and its worker thread.

    The worker never holds the queue lock while it calls the RunManager, the
    design store, Docker or the estimator: the manager calls on_run_finished
    from its own thread, and a lock held across those calls could deadlock
    against it or stall the dashboard for the length of a schema check.
    """

    def __init__(self, path: Path, manager: Manager, store: DesignStore, cfg_getter: Callable[[], Config],
                 docker_ok: Callable[[], bool],
                 estimate_fn: Callable[[Config, dict[str, Any]], dict[str, Any]] | None = None,
                 run_spend: Callable[[], list[dict[str, Any]]] | None = None,
                 reuse_drift: Callable[[Config, dict[str, Any]], list[str]] | None = None,
                 images: ImageHook | None = None) -> None:
        self.path = Path(path)
        self._manager = manager
        self._store = store
        self._cfg_getter = cfg_getter
        self._docker_ok = docker_ok
        self._estimate_fn = estimate_fn
        # The pre-baked image gate (_images_ready): None runs every item without looking at its images.
        self._images = images
        self._run_spend = run_spend if run_spend is not None else manager.list
        # reuse.drift over the ledger: how a version that reuses trials would now run differently. An item is
        # held for it once; a retry runs it as approved unless the ledger moves on again (kept in memory only).
        self._reuse_drift = reuse_drift
        self._drift_held: dict[str, tuple[str, ...]] = {}
        self._drift_accepted: dict[str, tuple[str, ...]] = {}
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._wake = False
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._dirty = False
        self._save_failed = False
        # Set when queue.json could not be read and no copy of it could be
        # kept: until a copy succeeds no save may replace the original unseen.
        self._original_unread = False

        defaults = self._defaults()
        self._state = "paused"
        self._pause_reason = NOT_STARTED
        self._next_seq = 1
        self._global_cap: float | None = defaults.global_cap_usd
        self._cap_since = _now()
        self._on_failure = defaults.on_failure
        self._items: list[dict[str, Any]] = []
        with self._lock:
            self._load()
        # Registered here so a finished run can never leave its item running;
        # a second registration elsewhere is harmless (finishes are matched once).
        add_listener = getattr(manager, "add_listener", None)
        if not callable(add_listener):
            raise TypeError("manager must provide add_listener")
        add_listener(self.on_run_finished)

    # Reading ------------------------------------------------------------------

    def view(self) -> dict[str, Any]:
        """QueueView for GET /api/queue."""
        spent = self._spent_or_none()
        with self._lock:
            return self._view_locked(spent)

    def summary(self) -> dict[str, Any]:
        """The queue line of /api/state."""
        spent = self._spent_or_none()
        with self._lock:
            active = self._active_locked()
            return {
                "state": self._state,
                "pause_reason": self._pause_reason,
                "active_item": copy.deepcopy(active),
                "waiting": sum(1 for i in self._items if i["status"] == "waiting"),
                "held": sum(1 for i in self._items if i["status"] == "held"),
                "cap_remaining_usd": self._remaining(spent),
            }

    def version_statuses(self) -> dict[tuple[str, int], str]:
        """{(design_id, version): status} of items not finished yet, for the
        design store's derived statuses (queued, running)."""
        out: dict[tuple[str, int], str] = {}
        with self._lock:
            for item in self._items:
                key = (item["design_id"], item["version"])
                if item["status"] in ACTIVE:
                    out[key] = item["status"]
                elif item["status"] in OPEN:
                    out.setdefault(key, item["status"])
        return out

    def cap_spent(self) -> float:
        """Spend of every run, queued or manual, created since cap_since. A
        run whose start time cannot be read is counted: the cap errs on the
        side of stopping."""
        with self._lock:
            since = _parse_time(self._cap_since)
        total = 0.0
        for run in self._run_spend() or []:
            if not isinstance(run, Mapping):
                continue
            created = _parse_time(run.get("created_at"))
            if since is not None and created is not None and created < since:
                continue
            total += _money_or_none(run.get("spend_usd")) or 0.0
        return total

    def committed(self) -> float:
        """Budget plus in-flight allowance of every waiting item."""
        with self._lock:
            return self._committed_locked()

    def is_active(self) -> bool:
        with self._lock:
            return self._state == "running" or self._active_locked() is not None

    # Admission ------------------------------------------------------------------

    def add(self, design_id: str, version: int, spec_hash: str, position: str = "end") -> dict[str, Any]:
        """A person approves one version, by the hash they saw. This is the
        only way a design is approved. Returns the new item."""
        if position not in ("end", "front"):
            raise ValueError("position must be 'end' or 'front'")
        if not isinstance(spec_hash, str) or not spec_hash.startswith("sha256:"):
            raise ValueError("spec_hash must be the 'sha256:...' hash of the version you approved")
        with self._lock:
            self._refuse_duplicate_locked(design_id, version)
        cfg = self._cfg_getter()
        doc, figures = self._admit(cfg, design_id, version, spec_hash)
        self._store.mark_approved(design_id, version)
        with self._lock:
            # Again: another add for this version may have finished meanwhile.
            self._refuse_duplicate_locked(design_id, version)
            item = {
                "id": f"q-{self._next_seq:04d}",
                "design_id": design_id,
                "version": version,
                "spec_hash": spec_hash,
                "title": str(doc.get("title") or design_id),
                **figures,
                "status": "waiting",
                "message": "",
                "added_at": _now(),
                "started_at": None,
                "finished_at": None,
                "run_id": None,
                "spend_usd": 0.0,
            }
            self._next_seq += 1
            if position == "front":
                first_open = next((n for n, i in enumerate(self._items) if i["status"] in OPEN), len(self._items))
                self._items.insert(first_open, item)
            else:
                self._items.append(item)
            self._changed_locked()
            return copy.deepcopy(item)

    def remove(self, item_id: str) -> dict[str, Any]:
        with self._lock:
            item = self._find_locked(item_id)
            if item["status"] not in OPEN:
                raise RuntimeError(f"Only waiting or held items can be removed; {item_id} is {item['status']}")
            self._items.remove(item)
            self._changed_locked()
        return self.view()

    def move(self, item_id: str, to_index: int) -> dict[str, Any]:
        """to_index is the item's new place among the waiting and held items,
        counted from 0 as each item's "position" in the view shows it."""
        if not isinstance(to_index, int) or isinstance(to_index, bool):
            raise ValueError("to_index must be a whole number")
        with self._lock:
            item = self._find_locked(item_id)
            if item["status"] not in OPEN:
                raise RuntimeError(f"Only waiting or held items can be moved; {item_id} is {item['status']}")
            others = [i for i in self._items if i["status"] in OPEN and i is not item]
            at = min(max(to_index, 0), len(others))
            # Only the order of open items means anything; history is sorted by time.
            rest = [i for i in self._items if i["status"] not in OPEN]
            self._items = rest + others[:at] + [item] + others[at:]
            self._changed_locked()
        return self.view()

    def retry(self, item_id: str) -> dict[str, Any]:
        """A held item goes back to waiting once its version passes the same
        checks as at add. Nothing starts until the queue runs."""
        with self._lock:
            item = self._find_locked(item_id)
            if item["status"] != "held":
                raise RuntimeError(f"Only held items can be retried; {item_id} is {item['status']}")
            design_id, version, approved = item["design_id"], item["version"], item["spec_hash"]
        try:
            cfg = self._cfg_getter()
            _, figures = self._admit(cfg, design_id, version, approved)
        except Exception as exc:
            with self._lock:
                held = self._get_locked(item_id)
                if held is not None and held["status"] == "held":
                    held["message"] = _one_line(exc)
                    self._changed_locked()
            raise
        with self._lock:
            item = self._find_locked(item_id)
            if item["status"] != "held":
                raise RuntimeError(f"Only held items can be retried; {item_id} is {item['status']}")
            if item_id in self._drift_held:  # held because the ledger moved on: the person runs it as approved
                self._drift_accepted[item_id] = self._drift_held.pop(item_id)
            item.update(figures, status="waiting", message="")
            self._changed_locked()
        return self.view()

    # Control --------------------------------------------------------------------

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._stopping:
                raise RuntimeError("The bench is shutting down")
            self._require_work_locked()
        if not self._docker_up():
            raise RuntimeError(f"{DOCKER_DOWN}. Start it, then start the queue.")
        with self._lock:
            if self._stopping:
                raise RuntimeError("The bench is shutting down")
            self._require_work_locked()
            self._state = "running"
            self._pause_reason = ""
            self._changed_locked()
        return self.view()

    def pause(self, reason: str = PAUSED_BY_YOU) -> dict[str, Any]:
        """Nothing new starts; a run in progress carries on."""
        reason = str(reason or "").strip() or PAUSED_BY_YOU
        with self._lock:
            self._state = "paused"
            self._pause_reason = reason
            self._changed_locked()
        return self.view()

    def update_settings(self, global_cap_usd: Any = UNSET, reset_cap: bool = False,
                        on_failure: str | None = None) -> dict[str, Any]:
        cap = UNSET
        if global_cap_usd is not UNSET:
            if global_cap_usd is not None and not _positive(global_cap_usd):
                raise ValueError("global_cap_usd must be a number above zero, or null for no cap")
            cap = None if global_cap_usd is None else float(global_cap_usd)
        if on_failure is not None and on_failure not in ON_FAILURE:
            raise ValueError("on_failure must be 'pause' or 'skip'")
        if not isinstance(reset_cap, bool):
            raise ValueError("reset_cap must be true or false")
        with self._lock:
            if cap is not UNSET:
                self._global_cap = cap
            if reset_cap:
                self._cap_since = _now()
            if on_failure is not None:
                self._on_failure = on_failure
            self._changed_locked()
        return self.view()

    # Worker ---------------------------------------------------------------------

    def start_worker(self) -> None:
        with self._lock:
            if self._stopping or (self._thread is not None and self._thread.is_alive()):
                return
            self._thread = threading.Thread(target=self._loop, name="run-queue", daemon=True)
            self._thread.start()

    def notify(self) -> None:
        with self._cond:
            self._wake = True
            self._cond.notify_all()

    def shutdown(self, timeout: float = 5) -> None:
        """Stop admitting and save the queue paused. A run in progress is the
        RunManager's to stop; its outcome is still recorded when it ends."""
        with self._cond:
            self._stopping = True
            if self._state == "running":
                self._state = "paused"
                self._pause_reason = STOPPED
            self._persist_locked()
            self._wake = True
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def on_run_finished(self, run: dict[str, Any]) -> None:
        """RunManager listener: record the outcome of the active item's run
        and decide whether the queue carries on. Other runs only wake the
        worker, since the manager may now be idle."""
        if not isinstance(run, Mapping):
            return
        run_id = run.get("id") if isinstance(run.get("id"), str) else None
        ref = run.get("design_ref") if isinstance(run.get("design_ref"), Mapping) else {}
        queue_item = ref.get("queue_item")
        with self._cond:
            item = next((i for i in self._items if i["status"] in ACTIVE
                         and ((queue_item and i["id"] == queue_item) or (run_id and i["run_id"] == run_id))), None)
            if item is None:
                self._wake = True
                self._cond.notify_all()
                return
            status = str(run.get("status") or "")
            # An unknown status is never read as success.
            item_status, then = _OUTCOMES.get(status, ("failed", "pause"))
            message = run.get("message") if isinstance(run.get("message"), str) else ""
            finished_at = run.get("finished_at")
            item.update(
                status=item_status,
                run_id=item["run_id"] or run_id,
                started_at=item["started_at"] or (run.get("created_at") if isinstance(run.get("created_at"), str)
                                                  else None),
                finished_at=finished_at if isinstance(finished_at, str) else _now(),
                spend_usd=_money(_money_or_none(run.get("spend_usd")) or 0.0),
                message=(message.strip() or _ITEM_MESSAGES.get(item_status, f"The run ended as '{status}'"))[:500],
            )
            if then == "pause" or (then == "policy" and self._on_failure == "pause"):
                verb = _PAUSE_VERBS.get(item_status, f"ended as '{status}'")
                self._state = "paused"
                self._pause_reason = f"Paused: {_label(item)} {verb} ({item['run_id'] or 'no run id'})."
            self._trim_history_locked()
            self._changed_locked()

    def _loop(self) -> None:
        while True:
            try:
                again = self._step()
            except Exception as exc:  # the worker must outlive any one bad pass
                _note(f"{type(exc).__name__}: {exc}")
                again = False
            with self._cond:
                if self._stopping:
                    return
                if not again and not self._wake:
                    self._cond.wait(WAIT_SECONDS)
                self._wake = False

    def _step(self) -> bool:
        """One worker pass. Returns True when it tried to start an item, so
        the loop looks again at once (a skipped item lets the next one go)."""
        self._reconcile()
        with self._lock:
            if self._dirty:
                self._persist_locked()
            if self._stopping or self._state != "running" or self._active_locked() is not None:
                return False
            candidate = self._next_waiting_locked()
            if candidate is None:
                held = sum(1 for i in self._items if i["status"] == "held")
                self._state = "paused"
                self._pause_reason = ("Queue finished: nothing is waiting"
                                      + (f" ({held} held)" if held else "")
                                      + ". Add a design and press Start queue.")
                self._changed_locked()
                return False
            item_id = candidate["id"]
        if not self._manager_idle():
            return False
        with self._lock:
            candidate = self._next_waiting_locked()
            if (self._stopping or self._state != "running" or self._active_locked() is not None
                    or candidate is None or candidate["id"] != item_id):
                return True
            candidate["status"] = "starting"
            candidate["message"] = ""
            self._changed_locked()
        try:
            self._try_start(item_id)
        except Exception as exc:
            # Never left 'starting': an active item would block the queue for good.
            self._hold(item_id, f"The run could not be started: {type(exc).__name__}: {_one_line(exc)}")
        return True

    def _reconcile(self) -> None:
        """A lost finish notification must not leave an item running for
        good (blocking manual runs and every queue action): once the manager
        is idle, the active item's run is looked up and finished from its
        record. on_run_finished matches each finish once, so a late real
        notification does no harm."""
        with self._lock:
            active = self._active_locked()
            run_id = active["run_id"] if active is not None and active["status"] == "running" else None
        if not run_id or not self._manager_idle():
            return
        try:
            run = next((r for r in self._run_spend() if isinstance(r, Mapping) and r.get("id") == run_id), None)
        except Exception:
            return
        if run is not None and str(run.get("status") or "") in _OUTCOMES:
            self.on_run_finished(run)

    def _try_start(self, item_id: str) -> None:
        """Start a 'starting' item, after checking again everything add
        checked: the exact approved spec, its validity under today's config,
        Docker and the global cap."""
        with self._lock:
            item = self._get_locked(item_id)
            if item is None or item["status"] != "starting":
                return
            design_id, version, approved = item["design_id"], item["version"], item["spec_hash"]
            allowance = item["inflight_allowance_usd"]
            label = _label(item)
        try:
            cfg = self._cfg_getter()
        except Exception as exc:
            self._hold(item_id, f"The profile's config could not be loaded: {_one_line(exc)}")
            return
        try:
            doc = self._store.version(design_id, version)
        except Exception as exc:
            self._hold(item_id, f"Could not read the design: {_one_line(exc)}")
            return
        problem = _version_problem(doc, approved)
        if problem:
            self._hold(item_id, problem)
            return
        check = designs.validate(cfg, doc["spec"])
        if check.get("problems"):
            self._hold(item_id, "It no longer passes its checks: " + "; ".join(map(str, check["problems"])))
            return
        drift = self._drift(cfg, doc["spec"], label)
        with self._lock:
            accepted = self._drift_accepted.get(item_id)
            if drift and drift != accepted:
                self._drift_held[item_id] = drift
        if drift and drift != accepted:
            self._hold(item_id, "The ledger changed since this version was saved, so resolving it again would run "
                                "other tasks: " + "; ".join(drift) + ". Save a new version in DESIGNS and queue "
                                "that, or retry this item to run it as approved.")
            return
        if not self._docker_up():
            self._hold(item_id, f"{DOCKER_DOWN}. Start it, then retry this item and start the queue.",
                       pause=True, reason=DOCKER_DOWN)
            return
        with self._lock:
            cap = self._global_cap
        if cap is not None:
            budget = float(doc["spec"]["run"]["budget_usd"])
            if allowance is None:
                allowance = _fallback_allowance(doc["spec"], check.get("plan"))
            try:
                spent = self.cap_spent()
            except Exception as exc:
                self._hold(item_id, f"Could not add up spend for the global cap: {_one_line(exc)}", pause=True)
                return
            need = budget + allowance
            if spent + need > cap + _EPSILON:
                left = max(0.0, cap - spent)
                self._hold(item_id,
                           f"The global cap of ${cap:.2f} has ${left:.2f} left; this design needs ${need:.2f} "
                           f"(budget ${budget:.2f} plus ${allowance:.2f} in-flight allowance). "
                           "Raise or reset the cap, then retry this item.",
                           pause=True, reason=f"Global cap: {label} needs ${need:.2f} and ${left:.2f} is left")
                return
        ref = {"design_id": design_id, "version": version, "author": doc.get("author"),
               "conversation_id": doc.get("conversation_id"), "queue_item": item_id}
        try:
            run_spec = spec_from_design(doc["spec"], ref)
        except Exception as exc:
            self._hold(item_id, f"The run could not be built: {_one_line(exc)}")
            return
        if not self._images_ready(item_id, cfg, run_spec, f"{design_id}@v{version}"):
            return
        with self._lock:
            item = self._get_locked(item_id)
            if item is None or item["status"] != "starting":
                return
            if self._stopping or self._state != "running":
                # Paused while it was being checked: it keeps its place.
                item["status"] = "waiting"
                self._changed_locked()
                return
        try:
            run = self._manager.start_spec(cfg, run_spec)
        except Exception as exc:
            if self._manager_closing():
                # The bench is exiting: the item was never started and keeps its place.
                with self._lock:
                    item = self._get_locked(item_id)
                    if item is not None and item["status"] == "starting":
                        item["status"] = "waiting"
                        self._changed_locked()
                return
            self._hold(item_id, f"The run could not start: {_one_line(exc)}")
            return
        run_id = run.get("id") if isinstance(run, Mapping) and isinstance(run.get("id"), str) else None
        with self._lock:
            item = self._get_locked(item_id)
            if item is not None:
                item["run_id"] = item["run_id"] or run_id
                item["started_at"] = item["started_at"] or _now()
                # A run that ended before start_spec returned has been recorded already.
                if item["status"] == "starting":
                    item["status"] = "running"
                    item["message"] = ""  # what the gate said while it built images
                self._changed_locked()
        if run_id:
            try:
                self._store.record_run(design_id, version, run_id)
            except Exception as exc:  # the run is under way; the design's run list is a record, not a gate
                _note(f"Could not record run {run_id} on {design_id} v{version}: {exc}")

    def waiting_specs(self) -> list[RunSpec]:
        """The runs the waiting and held items would start, as approved. An
        item whose design cannot be read is left out (it cannot start as it is)."""
        with self._lock:
            items = [(i["design_id"], i["version"]) for i in self._items if i["status"] in OPEN]
        out = []
        for design_id, version in items:
            try:
                out.append(spec_from_design(self._store.version(design_id, version)["spec"], None))
            except Exception:
                continue
        return out

    def _drift(self, cfg: Config, spec: dict[str, Any], label: str) -> tuple[str, ...]:
        """How a version that reuses trials would now run differently; empty
        when it reuses none, or when the ledger cannot say (it then runs as
        approved, which is still exactly what was approved)."""
        if self._reuse_drift is None or not spec.get("reuse"):
            return ()
        try:
            return tuple(self._reuse_drift(cfg, copy.deepcopy(spec)))
        except Exception as exc:
            _note(f"Could not check the trials {label} reuses against the ledger: {type(exc).__name__}: {exc}")
            return ()

    def _images_ready(self, item_id: str, cfg: Config, spec: RunSpec, ref: str) -> bool:
        """The pre-baked image gate, run last before a start. With
        prebake.auto_build the images the run is missing are built first, the
        item 'starting' meanwhile. Under run.prebake require an item whose
        images are still not ready is held with the reason; under prefer it
        starts, and those trials install their harness themselves. False when
        the item was held or went back to waiting."""
        if self._images is None or spec.run.prebake == PREBAKE_OFF:
            return True
        check = self._check_images(cfg, spec)
        for _ in range(2):  # a second round when the build it joined was someone else's
            if not (check.get("todo") and cfg.prebake.auto_build):
                break
            outcome, detail = self._build_first(item_id, cfg, spec)
            if outcome == "interrupted":
                return False
            if outcome == "refused":
                if spec.run.prebake == PREBAKE_REQUIRE:
                    self._hold(item_id, f"Its pre-baked images could not be built: {detail}. Free the cause, then "
                                        "retry this item.")
                    return False
                _note(f"{ref}: its missing pre-baked images were not built ({detail}); those trials run stock setup")
                break
            check = self._check_images(cfg, spec)
        blocking = check.get("blocking") or []
        if blocking and check.get("race"):
            # A task race builds its own images; what blocks is an arm that can never have one.
            self._hold(item_id, "run.prebake is require, but some of its arms can never use a pre-baked image: "
                       + "; ".join(blocking) + ". Pin their harness version or set run.prebake to prefer in a new "
                       "version; the race builds every other image just before its task.")
            return False
        if blocking:
            self._hold(item_id, "run.prebake is require and its pre-baked images are not ready: " + "; ".join(blocking)
                       + f". Build them (Build missing on LAUNCH, or python -m bench.prebake build --design {ref}), "
                       "then retry this item.")
            return False
        return True

    def _check_images(self, cfg: Config, spec: RunSpec) -> dict[str, Any]:
        try:
            return dict(self._images.check(cfg, spec))  # type: ignore[union-attr]
        except Exception as exc:
            why = f"they could not be checked: {_one_line(exc)}"
            return {"todo": 0, "blocking": [why] if spec.run.prebake == PREBAKE_REQUIRE else []}

    def _build_first(self, item_id: str, cfg: Config, spec: RunSpec) -> tuple[str, str]:
        """Builds the run's missing images and waits for the build: ('built',
        its status), ('refused', why it could not start) or ('interrupted', '')
        when the queue was paused or is stopping; the item then goes back to
        waiting and the build carries on (a later start joins it)."""
        hook = self._images
        try:
            build_id = hook.build(cfg, spec)  # type: ignore[union-attr]
        except Exception as exc:
            return "refused", _one_line(exc)
        while True:
            try:
                build = hook.status(build_id) or {}  # type: ignore[union-attr]
            except Exception as exc:
                return "refused", f"the build's status could not be read: {_one_line(exc)}"
            if not build or (build.get("status") != "running" and build.get("finished_at")):  # its lock is free
                return "built", str(build.get("status") or "")
            counts = build.get("counts") or {}
            done = sum(counts.get(s, 0) for s in ("promoted", "failed", "cancelled"))
            message = (f"Building its pre-baked images first: {done} of {len(build.get('items') or [])} finished "
                       f"(build {build_id})")
            with self._cond:
                item = self._get_locked(item_id)
                if item is None or item["status"] != "starting":
                    return "interrupted", ""
                if self._stopping or self._state != "running":
                    item["status"] = "waiting"
                    item["message"] = f"Paused while its pre-baked images were being built; build {build_id} goes on"
                    self._changed_locked()
                    return "interrupted", ""
                if item["message"] != message:
                    item["message"] = message
                    self._changed_locked()
                self._cond.wait(WAIT_SECONDS)

    def _hold(self, item_id: str, message: str, pause: bool | None = None, reason: str | None = None) -> None:
        """An admission failure. pause=None follows on_failure: under skip the
        worker moves on to the next item."""
        with self._lock:
            item = self._get_locked(item_id)
            if item is None or item["status"] != "starting":
                return
            item["status"] = "held"
            item["message"] = message[:1000]
            if pause is None:
                pause = self._on_failure == "pause"
            if pause:
                self._state = "paused"
                self._pause_reason = reason or f"Paused: {_label(item)} is held. {message}"[:300]
            self._changed_locked()

    # Internals ------------------------------------------------------------------

    def _admit(self, cfg: Config, design_id: str, version: int,
               approved: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """The checks add and retry share: the version is the one approved and
        is valid now. Returns the version and the item's money figures."""
        doc = self._store.version(design_id, version)
        problem = _version_problem(doc, approved)
        if problem:
            raise VersionConflict(problem)
        check = designs.validate(cfg, doc["spec"])
        if check.get("problems"):
            raise ConfigError("This version cannot be queued: " + "; ".join(map(str, check["problems"])))
        return doc, self._figures(cfg, doc["spec"], check.get("plan"))

    def _figures(self, cfg: Config, spec: dict[str, Any], plan: Any) -> dict[str, Any]:
        est: Any = None
        if self._estimate_fn is not None:
            try:
                est = self._estimate_fn(cfg, copy.deepcopy(spec))
            except Exception as exc:  # an estimate is advice; the caps below still bound the run
                _note(f"The cost estimate failed: {type(exc).__name__}: {exc}")
        est = est if isinstance(est, Mapping) else {}
        allowance = _money_or_none(est.get("inflight_allowance_usd"))
        if allowance is None:
            allowance = _fallback_allowance(spec, plan)
        return {
            "budget_usd": _money(float(spec["run"]["budget_usd"])),
            "inflight_allowance_usd": allowance,
            "expected_usd": _money_or_none(est.get("expected_usd")),
            "p90_usd": _money_or_none(est.get("p90_usd")),
        }

    def _refuse_duplicate_locked(self, design_id: str, version: Any) -> None:
        for item in self._items:
            if item["design_id"] == design_id and item["version"] == version and item["status"] in OPEN:
                raise RuntimeError(f"{design_id} v{version} is already in the queue ({item['id']}, {item['status']})")

    def _require_work_locked(self) -> None:
        if self._active_locked() is not None or self._next_waiting_locked() is not None:
            return
        held = sum(1 for i in self._items if i["status"] == "held")
        if held:
            raise RuntimeError(f"Nothing is waiting. Retry or remove the {held} held item(s) first.")
        raise RuntimeError("Nothing is waiting in the queue. Add a design first.")

    def _find_locked(self, item_id: str) -> dict[str, Any]:
        item = self._get_locked(item_id)
        if item is None:
            raise KeyError(f"No queue item '{str(item_id)[:20]}'")
        return item

    def _get_locked(self, item_id: Any) -> dict[str, Any] | None:
        return next((i for i in self._items if i["id"] == item_id), None)

    def _active_locked(self) -> dict[str, Any] | None:
        return next((i for i in self._items if i["status"] in ACTIVE), None)

    def _next_waiting_locked(self) -> dict[str, Any] | None:
        return next((i for i in self._items if i["status"] == "waiting"), None)

    def _committed_locked(self) -> float:
        return sum(i["budget_usd"] + (i["inflight_allowance_usd"] or 0.0)
                   for i in self._items if i["status"] == "waiting")

    def _remaining(self, spent: float | None) -> float | None:
        if self._global_cap is None or spent is None:
            return None
        return _money(max(0.0, self._global_cap - spent))

    def _spent_or_none(self) -> float | None:
        try:
            return self.cap_spent()
        except Exception:  # a view must render even if the run list cannot be read
            return None

    def _view_locked(self, spent: float | None) -> dict[str, Any]:
        open_items = [i for i in self._items if i["status"] in OPEN]

        def shown(item: dict[str, Any], position: int | None = None) -> dict[str, Any]:
            out = copy.deepcopy(item)
            if position is not None:
                out["position"] = position
            return out

        order = {id(i): n for n, i in enumerate(self._items)}
        finished = sorted((i for i in self._items if i["status"] in FINISHED),
                          key=lambda i: (str(i.get("finished_at") or i.get("added_at") or ""), order[id(i)]),
                          reverse=True)
        active = self._active_locked()
        return {
            "state": self._state,
            "pause_reason": self._pause_reason,
            "on_failure": self._on_failure,
            "global_cap_usd": self._global_cap,
            "cap_since": self._cap_since,
            "cap_spent_usd": None if spent is None else _money(spent),
            "cap_remaining_usd": self._remaining(spent),
            "committed_usd": _money(self._committed_locked()),
            "active": None if active is None else shown(active),
            "waiting": [shown(i, n) for n, i in enumerate(open_items) if i["status"] == "waiting"],
            "held": [shown(i, n) for n, i in enumerate(open_items) if i["status"] == "held"],
            "history": [shown(i) for i in finished[:HISTORY_SHOWN]],
        }

    def _trim_history_locked(self) -> None:
        finished = [i for i in self._items if i["status"] in FINISHED]
        if len(finished) <= HISTORY_KEPT:
            return
        oldest = sorted(finished, key=lambda i: str(i.get("finished_at") or ""))[: len(finished) - HISTORY_KEPT]
        drop = {id(i) for i in oldest}
        self._items = [i for i in self._items if id(i) not in drop]

    def _manager_closing(self) -> bool:
        return getattr(self._manager, "_closing", False) is True or self._stopping

    def _manager_idle(self) -> bool:
        try:
            return bool(self._manager.is_idle())
        except Exception:
            return False

    def _docker_up(self) -> bool:
        try:
            return bool(self._docker_ok())
        except Exception:
            return False

    def _defaults(self) -> QueueSettings:
        try:
            queue = self._cfg_getter().queue
        except Exception:  # a broken config must not stop the dashboard from showing the queue
            return QueueSettings()
        cap = queue.global_cap_usd if queue.global_cap_usd is None or _positive(queue.global_cap_usd) else None
        on_failure = queue.on_failure if queue.on_failure in ON_FAILURE else "pause"
        return QueueSettings(on_failure=on_failure, global_cap_usd=None if cap is None else float(cap))

    def _changed_locked(self) -> None:
        self._persist_locked()
        self._wake = True
        self._cond.notify_all()

    # Persistence ----------------------------------------------------------------

    def _record_locked(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "state": self._state,
            "pause_reason": self._pause_reason,
            "next_seq": self._next_seq,
            "global_cap_usd": self._global_cap,
            "cap_since": self._cap_since,
            "on_failure": self._on_failure,
            "items": copy.deepcopy(self._items),
        }

    def _persist_locked(self) -> bool:
        detail = "another program has it open"
        if self._original_unread and not self._copy_unread_original():
            self._dirty = True
            return False
        try:
            ok = fsutil.atomic_write_json(self.path, self._record_locked(), patience=SAVE_PATIENCE)
        except OSError as exc:
            ok, detail = False, str(exc)
        self._dirty = not ok
        if ok:
            self._save_failed = False
        elif not self._save_failed:
            # Once per failure streak: the worker retries every pass.
            _note(f"Could not save {self.path.name} ({detail}). The queue keeps working and saves again on "
                  "its next change.")
            self._save_failed = True
        return ok

    def _load(self) -> None:
        """Read queue.json, always paused, with interrupted runs marked. A file
        that cannot be read is kept aside and the queue starts empty: losing
        approvals only means nothing runs until a person queues again."""
        text = self._read_text()
        if text is None:
            return
        try:
            raw = json.loads(text)
        except ValueError:
            raw = None
        if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
            self._keep_corrupt_copy("it is not a queue record")
            self._dirty = True
            self._persist_locked()
            return
        items: list[dict[str, Any]] = []
        bad = 0
        for entry in raw["items"]:
            item = _item_from(entry)
            if item is None or any(i["id"] == item["id"] for i in items):
                bad += 1
                continue
            items.append(item)
        if bad:
            self._keep_corrupt_copy(f"{bad} item(s) could not be read")
        now = _now()
        for item in items:
            if item["status"] in ACTIVE:
                item["message"] = ("The bench stopped while this run was in progress" if item["run_id"]
                                   else "The bench stopped before this run started")
                item["status"] = "interrupted"
                item["finished_at"] = item["finished_at"] or now
        self._items = items
        # Ids are never reused, so a stale reference can never name a new item.
        seqs = [int(i["id"][2:]) + 1 for i in items]
        stored_seq = raw.get("next_seq")
        self._next_seq = max([self._next_seq, stored_seq if _is_int(stored_seq) and stored_seq > 0 else 1, *seqs])
        if "global_cap_usd" in raw:
            cap = raw["global_cap_usd"]
            if cap is None or _positive(cap):
                self._global_cap = None if cap is None else float(cap)
            else:
                _note(f"{self.path.name} has an invalid global_cap_usd; using the profile's ({self._global_cap}).")
        if raw.get("on_failure") in ON_FAILURE:
            self._on_failure = raw["on_failure"]
        if isinstance(raw.get("cap_since"), str) and _parse_time(raw["cap_since"]) is not None:
            self._cap_since = raw["cap_since"]
        self._state = "paused"
        self._pause_reason = PAUSED_AFTER_RESTART
        self._persist_locked()

    def _read_text(self) -> str | None:
        deadline = time.monotonic() + SAVE_PATIENCE
        while True:
            try:
                return self.path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return None
            except PermissionError:
                if time.monotonic() >= deadline:
                    # Probably valid and only locked for now: copy it, never move it.
                    if not self._keep_corrupt_copy("it stayed locked by another program", move=False):
                        self._original_unread = True
                    return None
                time.sleep(0.1)
            except (OSError, UnicodeDecodeError):
                self._keep_corrupt_copy("it could not be read")
                return None

    def _keep_corrupt_copy(self, why: str, move: bool = True) -> bool:
        """Keep the unreadable file aside. move=False only copies it (a
        locked file is probably valid). Returns False when no copy could be
        kept and the original is still there."""
        self._seed_from_copies()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
        n = 2
        while target.exists():
            target = self.path.with_name(f"{self.path.name}.corrupt-{stamp}-{n}")
            n += 1
        moved = False
        if move:
            try:
                self.path.replace(target)
                moved = True
            except FileNotFoundError:
                return True
            except OSError:
                pass  # locked by a scanner or an editor: a copy keeps the evidence
        if not moved:
            try:
                shutil.copyfile(self.path, target)
            except FileNotFoundError:
                return True
            except OSError as exc:
                _note(f"{self.path.name}: {why}, and no copy could be kept ({exc}). Starting empty; "
                      "the file is not overwritten until a copy can be kept.")
                return False
        self._seed_from_copies()
        _note(f"{self.path.name}: {why}. Kept it as {target.name}.")
        return True

    def _copy_unread_original(self) -> bool:
        """Retry keeping the original that the load could not read. True
        once it is kept (or gone), so saves may go ahead."""
        if self._keep_corrupt_copy("it could not be read at startup", move=False):
            self._original_unread = False
            return True
        return False

    def _seed_from_copies(self) -> None:
        """Ids are never reused, even after a load that kept nothing: the
        highest id in queue.json or any kept copy moves next_seq past it."""
        highest = 0
        try:
            candidates = [self.path, *self.path.parent.glob(f"{self.path.name}.corrupt-*")]
        except OSError:
            return
        for candidate in candidates:
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in _ID_IN_TEXT.finditer(text):
                highest = max(highest, int(match.group(1)))
            seq = _SEQ_IN_TEXT.search(text)
            if seq:
                highest = max(highest, int(seq.group(1)) - 1)
        self._next_seq = max(self._next_seq, highest + 1)


# Helpers -------------------------------------------------------------------------

_ID_IN_TEXT = re.compile(r'"q-(\d{1,9})"')
_SEQ_IN_TEXT = re.compile(r'"next_seq"\s*:\s*(\d{1,9})')


def _note(text: str) -> None:
    try:
        print(f"[queue] {text}", file=sys.stderr)
    except Exception:
        pass  # a stderr that cannot take the text (a cp1252 pipe, a closed console) must not stop the worker


def _version_problem(doc: Mapping[str, Any], approved: str) -> str | None:
    """Why this version file is not the one a person approved, if it is not."""
    design_id, number = doc.get("design_id"), doc.get("version")
    if doc.get("archived"):
        return f"Design {design_id} is archived. Unarchive it before queueing it."
    stored = doc.get("spec_hash")
    if stored != approved:
        return (f"Version {number} of {design_id} has spec hash {_short_hash(stored)}, not the "
                f"{_short_hash(approved)} that was approved. Reload it and check it again.")
    try:
        actual = hash_of(doc["spec"])
    except Exception:
        actual = None
    if actual != stored:
        return (f"The file of version {number} of {design_id} was edited: its spec no longer matches its hash. "
                "Save the change as a new version and approve that.")
    return None


def _fallback_allowance(spec: Mapping[str, Any], plan: Any) -> float:
    """Without an estimate: every container that can be busy when the budget
    stops the run may still finish a trial at the per-trial cap."""
    peak = plan.get("peak_containers") if isinstance(plan, Mapping) else None
    if not _is_int(peak) or peak < 0:
        peak = designs.peak_containers(spec)
    cap = spec["run"].get("per_trial_cap_usd") or 0.0
    return _money(min(peak, designs.trial_count(spec)) * float(cap))


def _item_from(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    item_id, design_id, version = raw.get("id"), raw.get("design_id"), raw.get("version")
    approved, status = raw.get("spec_hash"), raw.get("status")
    if not (isinstance(item_id, str) and _ITEM_ID.fullmatch(item_id)):
        return None
    if not designs.valid_design_id(design_id) or not _is_int(version) or version < 1:
        return None
    if not (isinstance(approved, str) and approved.startswith("sha256:")) or status not in ITEM_STATUSES:
        return None
    return {
        "id": item_id,
        "design_id": design_id,
        "version": version,
        "spec_hash": approved,
        "title": _text(raw.get("title")) or design_id,
        "budget_usd": _money_or_none(raw.get("budget_usd")) or 0.0,
        "inflight_allowance_usd": _money_or_none(raw.get("inflight_allowance_usd")),
        "expected_usd": _money_or_none(raw.get("expected_usd")),
        "p90_usd": _money_or_none(raw.get("p90_usd")),
        "status": status,
        "message": _text(raw.get("message")),
        "added_at": _text(raw.get("added_at")) or None,
        "started_at": _text(raw.get("started_at")) or None,
        "finished_at": _text(raw.get("finished_at")) or None,
        "run_id": _text(raw.get("run_id")) or None,
        "spend_usd": _money_or_none(raw.get("spend_usd")) or 0.0,
    }


def _label(item: Mapping[str, Any]) -> str:
    return f"{item['title']} v{item['version']}"


def _short_hash(value: Any) -> str:
    text = str(value or "none")
    return text[7:19] if text.startswith("sha256:") else text[:12]


def _one_line(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:500] or type(exc).__name__


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and value > 0)


def _money(value: float) -> float:
    return round(float(value), 6)


def _money_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return _money(value)
    return None


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
