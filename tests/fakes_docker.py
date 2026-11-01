"""An in-memory Docker CLI for the pre-bake builder's tests.

FakeDocker answers the docker commands bench/prebake issues (image ls and
inspect, tag, rmi, image prune, build, run, pull, ps, rm) from a dict of
images, through BenchTestCase.allow_subprocess, so the production code runs
down to the argv it would give the real docker. A build makes a new image
from its BASE build arg: the base's config and layers, one layer per RUN or
COPY in its Dockerfile, and the --label values. CannedProbes answers the
recorder's probes from tests/fixtures/prebake/probes.json (real answers of
three local task images).
"""

from __future__ import annotations

import copy
import hashlib
import io
import itertools
import json
import re
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tests.support import FIXTURES, completed

from bench.prebake.recorder import RecipeError

PROBES_FILE = FIXTURES / "prebake" / "probes.json"
_ids = itertools.count(1)
OS_FAMILY_PROBE = '. /etc/os-release 2>/dev/null; echo "${ID:-unknown}-${VERSION_ID:-}"'
# What the three probed images' /etc/os-release say.
OS_FAMILIES = {"alexgshaw/fix-git:20251031": "debian-12", "alexgshaw/regex-log:20251031": "ubuntu-24.04",
               "alexgshaw/multi-source-data-merger:20251031": "debian-13"}
FINGERPRINT_OUTPUT = "## os\nID=debian\n## dpkg\ntmux=3.3a-3 ii\n## end\n"


def new_id() -> str:
    return "sha256:" + f"{next(_ids):064x}"


def go_time(when: datetime) -> str:
    """A time as Docker's Go code prints it: '2026-10-31 11:53:01.123456789 +0000 UTC'."""
    when = when.astimezone(timezone.utc)
    return f"{when:%Y-%m-%d %H:%M:%S}.{when.microsecond:06d}123 +0000 UTC"


def cache_id() -> str:
    """A build cache record id as BuildKit prints them: 25 lowercase letters and digits, unique in 12."""
    return hashlib.sha256(str(next(_ids)).encode("ascii")).hexdigest()[:25]


class FakePopen:
    """What subprocess.Popen returns for a docker build or pull: done at once,
    unless release is an Event the test sets later."""

    def __init__(self, rc: int, output: str = "", stdout: Any = None, release: threading.Event | None = None,
                 on_release: Callable[[], int] | None = None) -> None:
        self.pid = 0
        self._rc = rc
        self.returncode: int | None = None if release else rc
        self._release = release
        self._on_release = on_release
        self.killed = False
        if hasattr(stdout, "write"):
            stdout.write(output)
            self.stdout = None
        else:
            self.stdout = io.StringIO(output)

    def poll(self) -> int | None:
        if self.returncode is None and self._release is not None and self._release.is_set():
            self.returncode = self._on_release() if self._on_release else self._rc
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None and self._release is not None:
            self._release.wait(timeout)
        return self.poll() if self.poll() is not None else -1


class FakeDocker:
    def __init__(self) -> None:
        self.images: dict[str, dict[str, Any]] = {}   # id -> `docker image inspect` object
        self.calls: list[list[str]] = []
        self.lock = threading.Lock()
        self.run_handlers: list[tuple[Callable[[list[str], str | None], bool],
                                      Callable[[list[str], str | None], tuple[int, str, str]]]] = []
        self.smoke_fail: set[str] = set()
        self.build_rc = 0
        self.build_release: threading.Event | None = None
        self.vm_free_kb = 900 * 1024 ** 2
        self.pullable: dict[str, dict[str, Any]] = {}   # ref -> image added by `docker pull`
        self.pull_rc = 0
        self.pull_release: threading.Event | None = None
        # BuildKit's build cache: record id -> what made it. Each build adds records; `buildx du --verbose`
        # lists them and `builder prune --filter id=<id>` removes one, except one a remaining record has as its
        # parent (BuildKit keeps it, and the prune still exits 0). cache_parents: record -> its parent;
        # cache_created: record -> its 'Created at' as buildx prints it; cache_in_use: records a running build
        # holds, which a prune keeps too.
        self.cache: dict[str, str] = {}
        self.cache_parents: dict[str, str] = {}
        self.cache_created: dict[str, str] = {}
        self.cache_in_use: set[str] = set()
        self.buildx = True   # False: `docker buildx` is missing, so the listing falls back to `system df -v`

    # Images ------------------------------------------------------------------------

    def add_image(self, ref: str | None, *, layers: tuple[str, ...] = ("sha256:l1", "sha256:l2"),
                  labels: dict[str, str] | None = None, os_name: str = "linux",
                  config: dict[str, Any] | None = None, image_id: str | None = None) -> str:
        image_id = image_id or new_id()
        cfg = {"Env": ["PATH=/usr/local/bin:/usr/bin:/bin"], "Cmd": ["bash"], "WorkingDir": "/app", "User": "",
               "Labels": dict(labels or {}), **(config or {})}
        self.images[image_id] = {"Id": image_id, "RepoTags": [], "Os": os_name, "Size": 1000,
                                 "Config": cfg, "RootFS": {"Type": "layers", "Layers": list(layers)}}
        if ref:
            self.tag(image_id, ref)
        return image_id

    def find(self, ref: str) -> dict[str, Any] | None:
        if ref in self.images:
            return self.images[ref]
        full = ref if ":" in ref.rsplit("/", 1)[-1] else f"{ref}:latest"
        for obj in self.images.values():
            if full in obj["RepoTags"]:
                return obj
        return None

    def tag(self, src: str, dst: str) -> bool:
        obj = self.find(src)
        if obj is None:
            return False
        for other in self.images.values():
            if dst in other["RepoTags"]:
                other["RepoTags"].remove(dst)
        obj["RepoTags"].append(dst)
        return True

    def refs(self) -> set[str]:
        return {t for obj in self.images.values() for t in obj["RepoTags"]}

    # The CLI --------------------------------------------------------------------------

    def matches(self, argv: list[str]) -> bool:
        return bool(argv) and Path(argv[0]).name.lower() in ("docker", "docker.exe")

    def __call__(self, argv: list[str], **kwargs: Any) -> Any:
        with self.lock:
            self.calls.append(list(argv))
        args = argv[1:]
        popen = "capture_output" not in kwargs
        if args[:1] == ["build"]:
            return self._build(args, kwargs)
        if args[:1] == ["pull"]:
            return self._pull(args, kwargs)
        stdin = kwargs.get("input")
        with self.lock:
            rc, out, err = self._answer(args, stdin.decode("utf-8") if isinstance(stdin, bytes) else stdin)
        if popen:
            return FakePopen(rc, out, kwargs.get("stdout"))
        return completed(argv, out, err, rc)

    def _answer(self, args: list[str], stdin: str | None) -> tuple[int, str, str]:
        if args[:2] == ["image", "ls"]:
            label = next((a.split("=", 1)[1] for a in args if a.startswith("label=")), None)
            lines = [f"{tag}\t{obj['Id']}" for obj in self.images.values() for tag in obj["RepoTags"]
                     if label is None or label in obj["Config"]["Labels"]]
            return 0, "\n".join(lines) + ("\n" if lines else ""), ""
        if args[:2] == ["image", "inspect"]:
            found = [copy.deepcopy(self.find(r)) for r in args[2:] if self.find(r) is not None]
            missing = [r for r in args[2:] if self.find(r) is None]
            return (1 if missing else 0), json.dumps(found), "".join(f"Error: No such image: {r}\n" for r in missing)
        if args[:1] == ["tag"]:
            return (0, "", "") if self.tag(args[1], args[2]) else (1, "", f"No such image: {args[1]}")
        if args[:1] == ["rmi"]:
            for ref in args[1:]:
                obj = self.find(ref)
                if obj is None or ref not in obj["RepoTags"] and ref != obj["Id"]:
                    return 1, "", f"Error: No such image: {ref}"
                if ref in obj["RepoTags"]:
                    obj["RepoTags"].remove(ref)
                if not obj["RepoTags"]:
                    del self.images[obj["Id"]]
            return 0, f"Untagged: {' '.join(args[1:])}\n", ""
        if args[:2] == ["image", "prune"]:
            label = next((a.split("=", 1)[1] for a in args if a.startswith("label=")), None)
            gone = [i for i, o in self.images.items() if not o["RepoTags"] and label in o["Config"]["Labels"]]
            for i in gone:
                del self.images[i]
            return 0, "Total reclaimed space: 0B\n", ""
        if args[:1] == ["run"]:
            for predicate, handler in reversed(self.run_handlers):
                if predicate(args, stdin):
                    return handler(args, stdin)
            return self._run(args, stdin)
        if args[:2] == ["buildx", "du"]:
            if not self.buildx:
                return 1, "", "docker: 'buildx' is not a docker command."
            return 0, "".join(self._cache_record(i, what) for i, what in self.cache.items()), ""
        if args[:2] == ["system", "df"]:
            rows = "".join(f"{i[:12]}   regular   1MB   1 hour ago   1 hour ago   1   false\n" for i in self.cache)
            return 0, ("Images space usage:\n\nBuild cache usage: 1MB\n\n"
                       "CACHE ID       CACHE TYPE   SIZE   CREATED   LAST USED   USAGE   SHARED\n" + rows), ""
        if args[:2] == ["builder", "prune"]:
            ids = [a.split("=", 1)[1] for i, a in enumerate(args) if args[i - 1] == "--filter" and a.startswith("id=")]
            if len(ids) != 1:
                return 1, "", "fake docker: a prune here needs exactly one id filter"
            used = {p for c, p in self.cache_parents.items() if c in self.cache} | self.cache_in_use
            gone = [i for i in self.cache if ids[0] in i and i not in used]
            for i in gone:
                del self.cache[i]
                self.cache_parents.pop(i, None)
                self.cache_created.pop(i, None)
            return 0, f"Total: {len(gone)}\n", ""
        if args[:1] in (["ps"], ["rm"]):
            return 0, "", ""
        return 1, "", f"fake docker: unsupported {' '.join(args[:3])}"

    def _cache_record(self, record: str, what: str) -> str:
        """One record as `docker buildx du --verbose` prints it."""
        lines = [f"ID:\t\t{record}"]
        if record in self.cache_parents:
            lines.append(f"Parent:\t\t{self.cache_parents[record]}")
        if record in self.cache_created:
            lines.append(f"Created at:\t{self.cache_created[record]}")
        lines.append(f"Description:\t{what}")
        return "\n".join(lines) + "\n\n"

    def _run(self, args: list[str], stdin: str | None) -> tuple[int, str, str]:
        if "--entrypoint" in args and args[args.index("--entrypoint") + 1] == "df":
            image = args[args.index("--entrypoint") + 2]
            if self.find(image) is None:
                return 125, "", f"Unable to find image '{image}' locally"
            return 0, ("Filesystem 1024-blocks Used Available Capacity Mounted on\n"
                       f"overlay 1000000000 1000 {self.vm_free_kb} 1% /\n"), ""
        image = args[args.index("--entrypoint") + 2] if "--entrypoint" in args else ""
        if self.find(image) is None:
            return 125, "", f"Unable to find image '{image}' locally"
        if stdin is not None and "sec end" in stdin:
            return 0, FINGERPRINT_OUTPUT, ""
        if stdin is not None:
            return (1, "", "tmux: not found") if image in self.smoke_fail else (0, "tmux 3.3a\n", "")
        return 0, "", ""

    def _build(self, args: list[str], kwargs: dict[str, Any]) -> Any:
        base = next(a.split("=", 1)[1] for a in args if a.startswith("BASE="))
        labels = dict(args[i + 1].split("=", 1) for i, a in enumerate(args) if a == "--label")
        tag = args[args.index("-t") + 1]
        dockerfile = Path(args[args.index("-f") + 1]).read_text(encoding="utf-8")
        steps = sum(1 for line in dockerfile.splitlines() if re.match(r"(RUN|COPY) ", line))

        def finish() -> int:
            with self.lock:
                parent = self.find(base)
                if self.build_rc != 0 or parent is None:
                    return self.build_rc or 1
                new = copy.deepcopy(parent)
                new["Id"] = new_id()
                new["RepoTags"] = []
                new["Config"]["Labels"] = {**parent["Config"]["Labels"], **labels}
                new["RootFS"]["Layers"] = parent["RootFS"]["Layers"] + [new_id() for _ in range(steps)]
                self.images[new["Id"]] = new
                self.tag(new["Id"], tag)
                record = None
                for n in range(steps + 2):  # one per step, plus the context and the Dockerfile; one chain
                    child = cache_id()
                    self.cache[child] = f"build of {tag} ({n})"
                    self.cache_created[child] = go_time(datetime.now(timezone.utc))
                    if record is not None:
                        self.cache_parents[child] = record
                    record = child
            return 0

        if self.build_release is not None:
            return FakePopen(1, "#1 building\n", kwargs.get("stdout"), release=self.build_release, on_release=finish)
        rc = finish()
        return FakePopen(rc, f"#1 building {tag}\n#2 DONE\n", kwargs.get("stdout"))

    def _pull(self, args: list[str], kwargs: dict[str, Any]) -> Any:
        ref = args[1]

        def finish() -> int:
            if self.pull_rc != 0 or ref not in self.pullable:
                return self.pull_rc or 1
            with self.lock:
                self.images[self.pullable[ref]["Id"]] = copy.deepcopy(self.pullable[ref])
            return 0

        if self.pull_release is not None:
            return FakePopen(1, "", kwargs.get("stdout"), release=self.pull_release, on_release=finish)
        rc = finish()
        return FakePopen(rc, f"pulling {ref}\nStatus: Downloaded newer image for {ref}\n", kwargs.get("stdout"))

    def ran(self, *words: str) -> list[list[str]]:
        """The docker calls that start with these words."""
        return [c[1:] for c in self.calls if c[1:1 + len(words)] == list(words)]


class CannedProbes:
    """recorder.ProbeRunner over the answers recorded from the real images,
    plus the OS-family probe verify.os_family issues."""

    def __init__(self) -> None:
        self.data = json.loads(PROBES_FILE.read_text(encoding="utf-8"))
        self.asked: list[tuple[str, str]] = []

    def image(self, image: str) -> dict[str, str]:
        if image not in self.data:
            raise RecipeError(f"{image} is not local (recording never pulls)")
        return dict(self.data[image]["image"])

    def run(self, image: str, command: str) -> tuple[int, str]:
        self.asked.append((image, command))
        answers = self.data[image]["answers"]
        if command == OS_FAMILY_PROBE:
            return 0, OS_FAMILIES.get(image, "debian-12") + "\n"
        if command not in answers:
            raise AssertionError(f"a probe the fixture does not have: {command}")
        rc, out = answers[command]
        return int(rc), str(out)
