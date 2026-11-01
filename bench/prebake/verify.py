"""Checks a pre-baked image must pass before it gets its name.

On every image (check_image):
- static: check_candidate() must call it ready against the local task image
  (labels, recipe, Harbor version, base id, layer prefix, OS), and its Config
  (Env, Cmd, Entrypoint, WorkingDir, User, Shell, Volumes, ExposedPorts)
  must be the task image's, since the bake never sets any of them;
- smoke: the adapter's offline check, `docker run --rm --network none`;
- fingerprint: packages, tool versions, PATH and dotfile hashes, saved as
  prebake/fingerprints/<image id>.txt with its sha256 in builds.jsonl.

On demand, or on the first image of each harness and OS family:
- setup_check: Harbor's real Trial lifecycle with install_only=True (start,
  healthcheck, the harness's unchanged setup(), stop; no agent run, no
  verifier, no model call) through PrebakedDockerEnvironment with
  prebake_required. The image must be used, setup must finish within the
  harness's budget, and exactly the recipe's skippable steps must be skipped.
  This is what catches the pi and strands re-run traps.
- equivalence: Harbor's setup() for real, through ExecShim (docker exec), in
  a fresh task-image container (the stock install) and in a container of the
  image (what a trial runs there); the two root filesystems and fingerprints
  must match except for the volatile paths in ALLOWED.

The two Harbor-driven checks run in a child process (python -m
bench.prebake.verify) with no credentials in its environment.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any

from .. import keys, procs
from ..environments.prebaked import (
    PREBAKED_ENV,
    READY,
    RECORD_NAME,
    ImageInfo,
    Want,
    check_candidate,
    exec_signature,
)
from ..settings import ROOT
from . import PREBAKE_DIR
from .inventory import DockerError, docker, inspect_raw, tail

SMOKE_TIMEOUT_SEC = 300
FINGERPRINT_TIMEOUT_SEC = 600
SETUP_TIMEOUT_SEC = 1800
SETUP_ATTEMPTS = 2            # a setup over budget and nothing else gets one more trial
EQUIVALENCE_TIMEOUT_SEC = 3600
CONFIG_KEYS = ("Env", "Cmd", "Entrypoint", "WorkingDir", "User", "Shell", "Volumes", "ExposedPorts")
VERIFY_LABEL = "org.etb.prebake.verify"
_RUN_LIMITS = ("--cpus", "1", "--memory", "2g")

# Packages, tools and versions an agent can see, plus hashes of the files
# that configure them. Deterministic: no times, no host names, no ids.
FINGERPRINT_SCRIPT = r"""
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
sec(){ echo "## $1"; }
sec os; grep -E '^(PRETTY_NAME|ID|VERSION_ID)=' /etc/os-release 2>/dev/null
sec id; id
sec path; echo "$PATH"
sec path_login; bash -lc 'echo "$PATH"' 2>/dev/null
sec which; for c in bash sh curl wget git gcc make python3 pip3 node npm npx pi claude tmux asciinema uv uvx pgrep; do
  printf '%s ' "$c"; command -v "$c" || echo -; done
sec which_local; for c in claude uv uvx mini-swe-agent; do printf '%s ' "$c"
  [ -e "$HOME/.local/bin/$c" ] && readlink -f "$HOME/.local/bin/$c" || echo -; done
sec dpkg; dpkg-query -W -f='${Package}=${Version} ${db:Status-Abbrev}\n' 2>/dev/null | sort
sec pip_system; python3 -m pip list --format=freeze 2>/dev/null | sort
sec npm_global; command -v npm >/dev/null 2>&1 && npm ls -g --depth=0 2>/dev/null
sec nvm; if [ -s "$HOME/.nvm/nvm.sh" ]; then . "$HOME/.nvm/nvm.sh" >/dev/null 2>&1
  node -v; npm -v; npm config get prefix; npm ls -g --depth=0 2>/dev/null; fi
sec claude; [ -x "$HOME/.local/bin/claude" ] && "$HOME/.local/bin/claude" --version 2>/dev/null
sec uv_tools; [ -x "$HOME/.local/bin/uv" ] &&
  "$HOME/.local/bin/uv" tool list --show-version-specifiers 2>/dev/null
sec venvs; for v in /installed-agent/*/venv; do [ -x "$v/bin/python" ] || continue; echo "-- $v"
  "$v/bin/python" -c 'import importlib.metadata as m
for d in sorted({(d.metadata["Name"] or "").lower() + "==" + d.version for d in m.distributions()}): print(d)' \
  2>/dev/null; done
sec dotfiles; for f in /root/.bashrc /root/.profile /root/.bash_profile /root/.local/bin/env \
  /root/.config/uv/uv-receipt.json; do [ -e "$f" ] && echo "$f $(md5sum < "$f" | cut -c1-32)"; done
sec installed_agent; find /installed-agent -maxdepth 2 -printf '%y %m %p\n' 2>/dev/null | sort
sec end
"""

# Every entry of the container's root filesystem (type, mode, owner, size of
# regular files, link target) and the md5 of every regular file. No times.
MANIFEST_SCRIPT = r"""
export LC_ALL=C
skip() { find / -xdev \( -path /proc -o -path /sys -o -path /dev -o -path /logs \) -prune -o "$@" 2>/dev/null; }
echo "### META"
skip -printf '%y\t%m\t%u:%g\t%s\t%l\t%p\n' | sort -t"$(printf '\t')" -k6
echo "### MD5"
skip -type f -print0 | xargs -0 -r md5sum 2>/dev/null | sort -k2
echo "### END"
"""

# What may differ between a stock container and a pre-baked one after the
# same setup(): the volatile categories the research comparisons found (log
# timestamps, host names, cache ids, .pyc headers, the npm cache index,
# random ids) and nothing else. Directory sizes are never compared.
ALLOWED: tuple[tuple[re.Pattern[str], str], ...] = tuple((re.compile(p), why) for p, why in (
    (r"^/etc/(hostname|hosts|resolv\.conf)$", "per-container network files"),
    (r"^/etc/machine-id$|^/var/lib/dbus/machine-id$", "random machine id"),
    (r"^/var/log(/|$)", "log timestamps"),
    (r"^/var/cache/ldconfig/aux-cache$", "ldconfig cache"),
    (r"^/var/cache/debconf/[^/]+-old$|^/var/lib/dpkg/[^/]+-old$", "package manager backups"),
    (r"^/tmp(/|$)", "temporary files"),
    (r"^/root/\.npm/_logs(/|$)", "npm logs"),
    (r"^/root/\.npm/_cacache(/|$)", "npm cache index"),
    (r"^/root/\.npm/_update-notifier-last-checked$", "npm update check time"),
    (r"^/root/\.cache/uv(/|$)", "uv cache ids"),
    # uv stamps a local package's dist-info with its source folder's mtime, which is the upload time; RECORD hashes
    # that stamp. Every file RECORD lists is compared on its own, so a changed file still fails.
    (r"\.dist-info/(uv_cache\.json|RECORD)$", "uv's source timestamp for a local package"),
    # A git clone's index holds file times, and the server packs the same objects differently on each fetch.
    # HEAD, refs and the checked-out files are still compared, so a different nvm version still fails.
    (r"^/root/\.nvm/\.git/(index|objects/pack/pack-[0-9a-f]+\.(idx|pack|rev))$", "nvm clone's git index and packs"),
    (r"(^|/)__pycache__(/|$)|\.pyc$", ".pyc headers"),
    (r"^/root/\.claude\.json$|^/root/\.claude/backups(/|$)", "Claude Code's per-install ids"),
    (r"^/root/\.config/asciinema/install-id$", "asciinema install id"),
    (r"^/root/\.local/state/goose/logs(/|$)", "goose's own run logs (named by date and time)"),
))


def allowed(path: str) -> str | None:
    """Why a difference at path is expected; None when it is not."""
    for pattern, why in ALLOWED:
        if pattern.search(path):
            return why
    return None


# Static ---------------------------------------------------------------------------


def config_diff(baked: Mapping[str, Any], base: Mapping[str, Any]) -> list[str]:
    """Config fields the bake changed; it must change none."""
    a, b = baked.get("Config") or {}, base.get("Config") or {}
    return [f"{k}: {json.dumps(b.get(k))} -> {json.dumps(a.get(k))}" for k in CONFIG_KEYS
            if (a.get(k) or None) != (b.get(k) or None)]


def static_check(ref: str, want: Want) -> dict[str, Any]:
    """The image under ref against the local task image, as a trial checks it."""
    raw = inspect_raw([ref, want.task_image], refresh=True)
    baked, base = raw.get(ref), raw.get(want.task_image)
    decision = check_candidate(None if baked is None else ImageInfo.from_inspect(baked),
                               None if base is None else ImageInfo.from_inspect(base), want)
    out: dict[str, Any] = {"status": decision.status, "reason": decision.reason, "noop": decision.noop,
                           "added_layers": decision.added_layers, "config_diff": [],
                           "image_id": None if baked is None else baked.get("Id"),
                           "size": None if baked is None else baked.get("Size"),
                           "base_id": None if base is None else base.get("Id")}
    if baked is not None and base is not None:
        out["config_diff"] = config_diff(baked, base)
    out["ok"] = decision.status == READY and not out["config_diff"]
    return out


def _run_in(ref: str, script: str, timeout: float, network: bool = False) -> subprocess.CompletedProcess:
    """bash reading script on stdin in a throwaway container of ref, offline by default."""
    args = ["run", "--rm", "-i", "--pull", "never", *_RUN_LIMITS, "--label", f"{VERIFY_LABEL}=run"]
    if not network:
        args += ["--network", "none"]
    return docker([*args, "--entrypoint", "bash", ref, "-s"], timeout=timeout, input=script)


def smoke(ref: str, script: str) -> dict[str, Any]:
    """The adapter's smoke script, offline in a throwaway container."""
    started = time.monotonic()
    try:
        r = _run_in(ref, f"set -o pipefail\n{script}\n", SMOKE_TIMEOUT_SEC)
    except DockerError as exc:
        return {"ok": False, "rc": None, "output": str(exc), "seconds": round(time.monotonic() - started, 1)}
    return {"ok": r.returncode == 0, "rc": r.returncode, "output": tail(r.stdout + r.stderr, 600),
            "seconds": round(time.monotonic() - started, 1)}


def fingerprint(ref: str, image_id: str, root: Path | None = None) -> dict[str, Any]:
    """FINGERPRINT_SCRIPT's output for the image, saved under its id."""
    try:
        r = _run_in(ref, FINGERPRINT_SCRIPT, FINGERPRINT_TIMEOUT_SEC)
    except DockerError as exc:
        return {"ok": False, "error": str(exc)}
    text = r.stdout
    if "## end" not in text:
        return {"ok": False, "error": f"the fingerprint script failed (exit {r.returncode}): {tail(r.stderr)}"}
    folder = (root or PREBAKE_DIR) / "fingerprints"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{image_id.replace(':', '_')}.txt"
    path.write_text(text, encoding="utf-8", newline="\n")
    return {"ok": True, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "path": str(path)}


def os_family(image: str) -> str:
    """'<ID>-<VERSION_ID>' of the image's /etc/os-release, e.g. debian-12."""
    from .recorder import PROBES, RecipeError

    try:
        rc, out = PROBES.run(image, '. /etc/os-release 2>/dev/null; echo "${ID:-unknown}-${VERSION_ID:-}"')
    except (RecipeError, OSError, subprocess.SubprocessError):
        return "unknown"
    return (out.strip() or "unknown") if rc == 0 else "unknown"


def check_image(ref: str, want: Want, smoke_script: str, root: Path | None = None) -> dict[str, Any]:
    """static, smoke and fingerprint; ok only when all three pass."""
    out: dict[str, Any] = {"static": static_check(ref, want)}
    if not out["static"]["ok"]:
        out["ok"] = False
        return out
    out["smoke"] = smoke(ref, smoke_script)
    out["fingerprint"] = fingerprint(ref, str(out["static"]["image_id"]), root) if out["smoke"]["ok"] else None
    out["ok"] = bool(out["smoke"]["ok"] and out["fingerprint"] and out["fingerprint"]["ok"])
    return out


# Child processes ------------------------------------------------------------------


def _child(kind: str, spec: Mapping[str, Any], work: Path, log: IO[str] | None, cancel: threading.Event | None,
           timeout: float, cleanup: Callable[[], None] | None = None) -> dict[str, Any]:
    """Runs `python -m bench.prebake.verify <kind>` with no credentials and
    returns the result it wrote; {"ok": False, "error": ...} when it did not."""
    work.mkdir(parents=True, exist_ok=True)
    spec_path, result_path = work / f"{kind}.spec.json", work / f"{kind}.result.json"
    spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    result_path.unlink(missing_ok=True)
    env = keys.child_env(None)
    env["HARBOR_TELEMETRY"] = "0"
    argv = [sys.executable, "-m", "bench.prebake.verify", kind, str(spec_path), str(result_path)]
    out = log if log is not None else subprocess.DEVNULL
    if log is not None:
        log.flush()  # the child appends to the same file
    try:
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, creationflags=procs.NO_CTRL_C)
    except OSError as exc:
        return {"ok": False, "error": f"the {kind} check could not start: {exc}"}
    procs.assign_kill_on_close(proc)
    started = time.monotonic()
    error = None
    while proc.poll() is None:
        if cancel is not None and cancel.wait(1.0):
            error = "cancelled"
        elif cancel is None:
            time.sleep(1.0)
        if error is None and time.monotonic() - started > timeout:
            error = f"did not finish within {timeout:g} s"
        if error:
            procs.kill_tree(proc.pid)
            proc.wait()
            if cleanup is not None:
                cleanup()
            return {"ok": False, "error": error}
    if not result_path.exists():
        return {"ok": False, "error": f"the {kind} check exited with {proc.returncode} and wrote no result"}
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return {"ok": False, "error": f"the {kind} check's result is unreadable: {exc}"}


def setup_check(*, prefix: str, family: str, version: str, recipe: str, agent: str, kwargs: Mapping[str, Any],
                model_name: str, task_dir: Path, budget_sec: float, expected_skips: int, work: Path,
                log: IO[str] | None = None, cancel: threading.Event | None = None) -> dict[str, Any]:
    """An install_only Trial on the image named by prefix (the staging name
    during a build), through the env class with prebake_required. A trial
    whose only fault is going over the budget runs once more: a saturated
    host (other builds, other work) slowed one terminus-2 setup from 6.6 s
    to 61 s, while a real re-run trap takes minutes both times."""
    earlier: list[dict[str, Any]] = []
    attempt = 0
    while True:
        attempt += 1
        trials = work / ("trials" if attempt == 1 else f"trials-{attempt}")
        spec = {"agent": agent, "kwargs": dict(kwargs), "model_name": model_name, "task_dir": str(task_dir),
                "trials_dir": str(trials), "trial_name": f"{task_dir.name}__{family}-setup"[:60],
                "env_kwargs": {"prebake_harness": family, "prebake_version": version, "prebake_recipe": recipe,
                               "prebake_prefix": prefix, "prebake_required": True}}
        result = judge_setup(_child("setup-trial", spec, work, log, cancel, SETUP_TIMEOUT_SEC,
                                    cleanup=lambda d=trials: procs.cleanup_with_retry(d, tries=2)),
                             budget_sec, expected_skips)
        if result["ok"] or not result.get("over_budget_only") or attempt >= SETUP_ATTEMPTS:
            return {**result, "attempts": attempt, "earlier": earlier}
        earlier.append({k: result.get(k) for k in ("agent_setup_sec", "env_setup_sec", "problems")})


def judge_setup(result: dict[str, Any], budget_sec: float, expected_skips: int) -> dict[str, Any]:
    """Whether an install_only trial shows a working image: used, no error,
    within the harness's setup budget, and exactly the recipe's skips."""
    if result.get("error") and "agent_setup_sec" not in result:
        return {**result, "ok": False}
    record = result.get("prebake") or {}
    problems = []
    if not record.get("used"):
        problems.append(f"the image was not used: {record.get('reason') or 'no prebake.json'}")
    if result.get("exception"):
        problems.append(f"the trial raised {result['exception']}: {result.get('exception_message') or ''}".strip())
    seconds = result.get("agent_setup_sec")
    over = seconds is not None and seconds > budget_sec
    if seconds is None:
        problems.append("agent setup did not finish")
    elif over:
        problems.append(f"agent setup took {seconds:g} s, over this harness's {budget_sec:g} s budget "
                        "(Harbor's install re-ran on the image, or the host is saturated)")
    skipped = len(record.get("skipped_execs") or [])
    if skipped != expected_skips:
        problems.append(f"{skipped} install step(s) were skipped, {expected_skips} expected")
    return {**result, "ok": not problems, "problems": problems, "budget_sec": budget_sec,
            "expected_skips": expected_skips, "over_budget_only": over and len(problems) == 1}


def equivalence(*, image: str, task_image: str, family: str, agent: str, kwargs: Mapping[str, Any],
                model_name: str, work: Path, log: IO[str] | None = None,
                cancel: threading.Event | None = None) -> dict[str, Any]:
    """Harbor's setup() in a stock task-image container and on the image, compared."""
    token = secrets.token_hex(4)
    spec = {"image": image, "task_image": task_image, "family": family, "agent": agent, "kwargs": dict(kwargs),
            "model_name": model_name, "token": token, "work": str(work)}
    return _child("equivalence", spec, work, log, cancel, EQUIVALENCE_TIMEOUT_SEC,
                  cleanup=lambda: remove_containers(token))


def remove_containers(token: str) -> None:
    """Removes the equivalence check's containers (labelled with its token)."""
    try:
        r = docker(["ps", "-aq", "--filter", f"label={VERIFY_LABEL}={token}"])
        ids = r.stdout.split()
        if ids:
            docker(["rm", "-f", *ids], timeout=120)
    except DockerError:
        pass


# Manifests --------------------------------------------------------------------------


def parse_manifest(text: str) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """({path: (type, mode, owner, size of a file, link)}, {path: md5})."""
    meta: dict[str, tuple[str, ...]] = {}
    md5: dict[str, str] = {}
    section = None
    for line in text.splitlines():
        if line.startswith("### "):
            section = line[4:].strip()
            continue
        if section == "META":
            parts = line.split("\t")
            if len(parts) == 6:
                kind, mode, owner, size, link, path = parts
                meta[path] = (kind, mode, owner, size if kind == "f" else "-", link)
        elif section == "MD5" and len(line) > 34:
            md5[line[34:]] = line[:32]
    return meta, md5


def diff_manifests(stock: str, baked: str) -> dict[str, Any]:
    """Every path whose type, mode, owner, file size, link target or content
    differs, or that only one side has; split into allowed and violations."""
    ma, ha = parse_manifest(stock)
    mb, hb = parse_manifest(baked)
    fields = ("type", "mode", "owner", "size", "link")
    found: list[tuple[str, str]] = []
    for path in sorted(set(ma) | set(mb)):
        if path not in mb:
            found.append((path, "only in stock"))
        elif path not in ma:
            found.append((path, "only in the pre-baked container"))
        else:
            what = [f"{n} {x} -> {y}" for n, x, y in zip(fields, ma[path], mb[path]) if x != y]
            if ma[path][0] == "f" and ha.get(path) != hb.get(path):
                what.append("content")
            if what:
                found.append((path, "; ".join(what)))
    allowed_counts: Counter[str] = Counter()
    violations = []
    for path, what in found:
        why = allowed(path)
        if why:
            allowed_counts[why] += 1
        else:
            violations.append(f"{path}: {what}")
    return {"entries": [len(ma), len(mb)], "files_hashed": [len(ha), len(hb)], "allowed": dict(allowed_counts),
            "violations": violations}


def sections(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            out[current] = []
        elif current is not None:
            out[current].append(line)
    return out


def diff_fingerprints(stock: str, baked: str) -> list[str]:
    """Fingerprint sections that differ; every one must match."""
    a, b = sections(stock), sections(baked)
    out = []
    for name in sorted(set(a) | set(b)):
        if a.get(name) != b.get(name):
            only_a = [x for x in a.get(name, []) if x not in set(b.get(name, []))][:5]
            only_b = [x for x in b.get(name, []) if x not in set(a.get(name, []))][:5]
            out.append(f"{name}: stock {only_a} vs pre-baked {only_b}")
    return out


# The docker-exec environment --------------------------------------------------------


class ExecShim:
    """Enough of Harbor's DockerEnvironment for a harness's setup() against a
    plain container: exec is `docker exec` with the same user, cwd and env
    flags, and uploads go through Harbor's own UnixOps (so the Windows CRLF
    fix runs as in a trial). Execs whose signature is in skip are answered
    with exit code 0 once, as PrebakedDockerEnvironment answers them."""

    def __init__(self, container: str, stage_dir: Path, skip: Sequence[str] = (), log: IO[str] | None = None
                 ) -> None:
        from harbor.environments.docker.docker_unix import UnixOps

        self.container = container
        self.default_user = None
        self.session_id = f"{container}__env"
        self.task_env_config = SimpleNamespace(workdir=None, docker_image=None)
        self.trial_paths = SimpleNamespace(trial_dir=stage_dir, agent_dir=stage_dir / "agent")
        self.logger = logging.getLogger(__name__)
        self.skip = set(skip)
        self.skipped: list[str] = []
        self.execs: list[dict[str, Any]] = []
        self._log = log
        self._ops = UnixOps(self)  # type: ignore[arg-type]

    async def _docker(self, args: list[str], timeout_sec: float | None) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec("docker", *args, stdin=asyncio.subprocess.DEVNULL,
                                                    stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE, env=keys.child_env(None))
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout_sec)
        except asyncio.TimeoutError:
            proc.kill()
            raise
        return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: str | int | None = None) -> Any:
        from harbor.environments.base import ExecResult

        user = user if user is not None else self.default_user
        signature = exec_signature(command, user, cwd, env or None)
        if signature in self.skip:
            self.skip.discard(signature)
            self.skipped.append(signature)
            self._note(0, 0.0, command, skipped=True)
            return ExecResult(stdout="", stderr="", return_code=0)
        args = ["exec"]
        if cwd:
            args += ["-w", cwd]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        if user is not None:
            args += ["-u", str(user)]
        started = time.monotonic()
        rc, out, err = await self._docker([*args, self.container, "bash", "-c", command], timeout_sec)
        self._note(rc, time.monotonic() - started, command)
        return ExecResult(stdout=out, stderr=err, return_code=rc)

    def _note(self, rc: int, seconds: float, command: str, skipped: bool = False) -> None:
        self.execs.append({"rc": rc, "seconds": round(seconds, 2), "skipped": skipped, "command": command[:200]})
        if self._log is not None:
            self._log.write(f"[{self.container}] {'skipped' if skipped else f'rc={rc}'} {seconds:6.1f}s "
                            f"{command[:160]}\n")
            self._log.flush()

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._ops.upload_file(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._ops.upload_dir(source_dir, target_dir)

    async def _run_docker_compose_command(self, command: list[str], check: bool = True,
                                          timeout_sec: int | None = None, **_: Any) -> Any:
        """UnixOps' compose calls, against the plain container."""
        from harbor.constants import MAIN_SERVICE_NAME
        from harbor.environments.base import ExecResult

        if len(command) == 3 and command[0] == "cp" and command[2].startswith(f"{MAIN_SERVICE_NAME}:"):
            args = ["cp", command[1], f"{self.container}:{command[2].split(':', 1)[1]}"]
        elif command[:2] == ["exec", MAIN_SERVICE_NAME]:
            args = ["exec", self.container, *command[2:]]
        else:
            raise RuntimeError(f"unexpected docker compose call: {' '.join(command[:3])}")
        rc, out, err = await self._docker(args, timeout_sec)
        if check and rc != 0:
            raise RuntimeError(f"docker {args[0]} failed: {tail(err)}")
        return ExecResult(stdout=out, stderr=err, return_code=rc)


# Child entry points -----------------------------------------------------------------


def _span(timing: Mapping[str, Any] | None) -> float | None:
    if not timing or not timing.get("started_at") or not timing.get("finished_at"):
        return None
    parse = datetime.fromisoformat
    return round((parse(str(timing["finished_at"]).replace("Z", "+00:00"))
                  - parse(str(timing["started_at"]).replace("Z", "+00:00"))).total_seconds(), 2)


async def _setup_trial(spec: Mapping[str, Any]) -> dict[str, Any]:
    from harbor.cli.utils import parse_kwargs, resolve_environment_spec
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
    from harbor.trial.trial import Trial

    # What `harbor run --env ... --ek ...` makes of runs.prebake_flags, through Harbor's own parsers.
    env_type, import_path = resolve_environment_spec(PREBAKED_ENV)
    env_kwargs = parse_kwargs([f"{k}={json.dumps(v)}" for k, v in spec["env_kwargs"].items()])
    agent = str(spec["agent"])
    who = {"import_path": agent} if ":" in agent else {"name": agent}
    config = TrialConfig(
        task=TaskConfig(path=Path(spec["task_dir"])), trial_name=spec["trial_name"],
        trials_dir=Path(spec["trials_dir"]), install_only=True, job_id=uuid.uuid4(),
        agent=AgentConfig(model_name=spec["model_name"], kwargs=dict(spec["kwargs"]), **who),
        environment=EnvironmentConfig(type=env_type, import_path=import_path, kwargs=env_kwargs))
    started = time.monotonic()
    trial = await Trial.create(config)
    await trial.run()
    trial_dir = trial.paths.trial_dir
    result = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    record_path = trial_dir / RECORD_NAME
    exception = result.get("exception_info") or {}
    return {
        "trial_dir": str(trial_dir), "wall_sec": round(time.monotonic() - started, 2),
        "env_setup_sec": _span(result.get("environment_setup")), "agent_setup_sec": _span(result.get("agent_setup")),
        "exception": exception.get("exception_type"), "exception_message": tail(exception.get("exception_message"),
                                                                                300),
        "prebake": json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else None,
    }


def _equivalence(spec: Mapping[str, Any]) -> dict[str, Any]:
    from .adapters import adapter_for
    from .recorder import arm_recipe, run_sync

    adapter = adapter_for(spec["agent"], spec["family"], spec["kwargs"], spec["model_name"])
    if adapter is None:
        return {"ok": False, "error": f"no pre-bake adapter for {spec['family']}"}
    recipe = arm_recipe(adapter)
    work = Path(spec["work"])
    token = str(spec["token"])
    sides: dict[str, dict[str, Any]] = {}
    try:
        for side, image, skip in (("stock", spec["task_image"], ()), ("baked", spec["image"], recipe.skip_execs)):
            name = f"etb-prebake-eq-{token}-{side}"
            r = docker(["run", "-d", "--rm", "--pull", "never", "--name", name, "--label", f"{VERIFY_LABEL}={token}",
                        *_RUN_LIMITS, image, "sh", "-c", "sleep infinity"], timeout=300)
            if r.returncode != 0:
                return {"ok": False, "error": f"the {side} container did not start: {tail(r.stderr)}"}
            with open(work / f"equivalence-{side}.log", "w", encoding="utf-8") as log:
                shim = ExecShim(name, work / side, skip, log)
                started = time.monotonic()
                run_sync(adapter.record(shim, work / side / "logs"))
                seconds = round(time.monotonic() - started, 1)
            manifest = docker(["exec", "-i", name, "bash", "-s"], timeout=1200, input=MANIFEST_SCRIPT)
            fp = docker(["exec", "-i", name, "bash", "-s"], timeout=FINGERPRINT_TIMEOUT_SEC, input=FINGERPRINT_SCRIPT)
            if "### END" not in manifest.stdout or "## end" not in fp.stdout:
                return {"ok": False, "error": f"the {side} container could not be listed: {tail(manifest.stderr)}"}
            (work / f"manifest-{side}.txt").write_text(manifest.stdout, encoding="utf-8", newline="\n")
            (work / f"fingerprint-{side}.txt").write_text(fp.stdout, encoding="utf-8", newline="\n")
            sides[side] = {"setup_sec": seconds, "skipped": len(shim.skipped), "manifest": manifest.stdout,
                           "fingerprint": fp.stdout, "nonzero_execs": [e for e in shim.execs if e["rc"] != 0]}
    finally:
        remove_containers(token)
    files = diff_manifests(sides["stock"]["manifest"], sides["baked"]["manifest"])
    fingerprints = diff_fingerprints(sides["stock"]["fingerprint"], sides["baked"]["fingerprint"])
    return {"ok": not files["violations"] and not fingerprints, "files": files, "fingerprints": fingerprints,
            "setup_sec": {k: v["setup_sec"] for k, v in sides.items()},
            "skipped": {k: v["skipped"] for k, v in sides.items()},
            "nonzero_execs": {k: v["nonzero_execs"] for k, v in sides.items()}}


def main(argv: list[str] | None = None) -> int:
    """python -m bench.prebake.verify setup-trial|equivalence <spec.json> <result.json>"""
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 3 or args[0] not in ("setup-trial", "equivalence"):
        print("usage: python -m bench.prebake.verify setup-trial|equivalence <spec.json> <result.json>",
              file=sys.stderr)
        return 2
    os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")
    spec = json.loads(Path(args[1]).read_text(encoding="utf-8"))
    try:
        result = asyncio.run(_setup_trial(spec)) if args[0] == "setup-trial" else _equivalence(spec)
    except Exception as exc:  # reported to the parent as the check's failure
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
    Path(args[2]).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
