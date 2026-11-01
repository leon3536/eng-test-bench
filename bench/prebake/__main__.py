"""python -m bench.prebake: pre-baked harness images from the command line.

    python -m bench.prebake status     [selection] [--refresh] [--json]
    python -m bench.prebake pull       [selection] [--dry-run]
    python -m bench.prebake build      [selection] [--jobs N] [--rebuild] [--setup | --no-setup]
                                       [--equivalence | --no-equivalence] [--sample per-os|all] [--dry-run] [--yes]
    python -m bench.prebake verify     [selection] [--setup] [--equivalence] [--sample per-os|all]
    python -m bench.prebake prune      [--keep-days N] [--all] [--include-probes] [--yes]
    python -m bench.prebake dockerfile [selection] [--out DIR]

A selection is one of --mode MODE [--arms a,b] (default: the config's largest
mode with every enabled arm), --design ID[@vN] or --run RUN_ID (what is left
of that run), narrowed with --task NAME. --reuse measured drops the tasks
the ledger already covers (a run's default; a design follows its own
setting). The profile is BENCH_PROFILE's, as for python -m bench;
--config/--overrides load other config files instead.

build and prune change nothing without --yes (build asks when run in a
terminal); prune is a dry run until then. Builds refuse during a run, and
when the drive holding Docker's disk image or Docker's VM would be left with
less than prebake.host_reserve_gb / vm_reserve_gb free.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .. import settings
from ..settings import Config, ConfigError
from . import builder, inventory, prune, pull
from .builder import BuildPlan, BuildRefused, Selection
from .inventory import DockerError


def _selection_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("selection")
    g.add_argument("--mode", help="a config mode (default: the largest one)")
    g.add_argument("--arms", help="comma-separated arm names (default: every enabled arm)")
    g.add_argument("--design", metavar="ID[@vN]", help="a design version (default: its latest)")
    g.add_argument("--run", metavar="RUN_ID", help="what remains of this run")
    g.add_argument("--reuse", choices=("off", "measured"),
                   help="drop tasks the ledger covers (a run's default; a design follows its own)")
    g.add_argument("--task", action="append", default=[], metavar="NAME", help="only this task (repeatable)")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m bench.prebake", description="Pre-baked harness images.",
                                epilog="The profile is BENCH_PROFILE's. See the module docstring for the selection.")
    p.add_argument("--config", type=Path, help="a bench config file instead of the profile's")
    p.add_argument("--overrides", type=Path, help="an overrides file instead of the profile's")
    sub = p.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="which images a selection's arms would find")
    _selection_args(status)
    status.add_argument("--refresh", action="store_true", help="list Docker's images again now")
    status.add_argument("--json", action="store_true")
    pull_p = sub.add_parser("pull", help="pull the selection's task images that are not local")
    _selection_args(pull_p)
    pull_p.add_argument("--dry-run", action="store_true")
    build = sub.add_parser("build", help="build, verify and promote the images a selection needs")
    _selection_args(build)
    build.add_argument("--jobs", type=int, help="builds at once (default prebake.build_jobs)")
    build.add_argument("--rebuild", action="store_true", help="also images that are ready")
    setup = build.add_mutually_exclusive_group()
    setup.add_argument("--setup", dest="setup", action="store_const", const="all",
                       help="the install_only setup check on every image (default: the first per harness and OS)")
    setup.add_argument("--no-setup", dest="setup", action="store_const", const="none",
                       help="no setup check (images are promoted on the static, smoke and fingerprint checks)")
    eq = build.add_mutually_exclusive_group()
    eq.add_argument("--equivalence", dest="equivalence", action="store_const", const="always",
                    help="the equivalence check even where this recipe passed it before")
    eq.add_argument("--no-equivalence", dest="equivalence", action="store_const", const="none",
                    help="no equivalence check, even for a new recipe")
    build.add_argument("--sample", choices=builder.SAMPLES, default="per-os",
                       help="equivalence on one image per harness and OS family, or on all")
    build.add_argument("--dry-run", action="store_true", help="show the plan and the disk check only")
    build.add_argument("--yes", action="store_true", help="build without asking")
    build.set_defaults(setup="auto", equivalence="auto")
    ver = sub.add_parser("verify", help="check the selection's ready images")
    _selection_args(ver)
    ver.add_argument("--setup", action="store_true", help="also Harbor's install_only setup check")
    ver.add_argument("--equivalence", action="store_true", help="also the stock-against-baked comparison")
    ver.add_argument("--sample", choices=builder.SAMPLES, default="per-os")
    ver.add_argument("--json", action="store_true")
    pr = sub.add_parser("prune", help="remove pre-baked images nothing needs (a dry run without --yes)")
    pr.add_argument("--keep-days", type=float, help="keep unwanted images this young (default prebake.keep_days)")
    pr.add_argument("--all", action="store_true", help="every image under the prefix, wanted or not")
    pr.add_argument("--include-probes", action="store_true", help="also the research prototypes' benchprobe/ images")
    pr.add_argument("--yes", action="store_true", help="remove them")
    pr.add_argument("--json", action="store_true")
    df = sub.add_parser("dockerfile", help="print (or write) the Dockerfiles a build would use")
    _selection_args(df)
    df.add_argument("--rebuild", action="store_true", help="also for images that are ready")
    df.add_argument("--out", type=Path, help="write <out>/<image slug>/Dockerfile and its context")
    return p


def load(args: argparse.Namespace) -> Config:
    return settings.load_config(args.config, args.overrides)


def selection(cfg: Config, args: argparse.Namespace) -> Selection:
    arms = tuple(a.strip() for a in (args.arms or "").split(",") if a.strip())
    mode = args.mode
    if mode is None and args.design is None and args.run is None:
        if not cfg.modes:
            raise ConfigError("The config has no modes; pick --design or --run")
        mode = max(cfg.modes, key=lambda m: cfg.modes[m].n_tasks)
    return Selection(mode=mode, arms=arms, design=args.design, run=args.run, reuse=args.reuse,
                     tasks=tuple(dict.fromkeys(args.task)))


def make_plan(cfg: Config, args: argparse.Namespace, rebuild: bool = False) -> BuildPlan:
    sel = selection(cfg, args)
    spec = builder.selection_spec(cfg, sel)
    refresh = bool(getattr(args, "refresh", True))  # status may use the 30 s listing; the rest list afresh
    return builder.plan_builds(cfg, spec, label=sel.label(), rebuild=rebuild, refresh=refresh)


def _gb(value: float | None) -> str:
    return "?" if value is None else f"{value:.1f} GB"


def print_status(plan: BuildPlan, cfg: Config) -> None:
    print(f"Selection: {plan.label}")
    print(f"Prefix {cfg.prebake.prefix}/, max age {cfg.prebake.max_age_days} days\n")
    header = f"{'arm':16} {'mode':8} {'harness':28} {'ready':>7} {'missing':>7} {'stale':>5} {'not local':>9} " \
             f"{'no image':>8}  build now"
    print(header)
    print("-" * len(header))
    for r in plan.readiness.values():
        harness = f"{r.family} {r.version or ''}".strip() + (f" ({r.variant})" if r.variant else "")
        if not r.supported:
            print(f"{r.arm:16} {r.mode:8} {harness:28} stock setup: {r.reason}")
            continue
        build = f"~{r.build_min:g} min, ~{r.build_gb:g} GB" if r.not_ready else "-"
        print(f"{r.arm:16} {r.mode:8} {harness[:28]:28} {len(r.ready):>3}/{r.tasks:<3} {len(r.missing):>7} "
              f"{len(r.stale):>5} {len(r.unverifiable):>9} {len(r.unknown):>8}  {build}")
    print()
    errors = {r.docker_error for r in plan.readiness.values() if r.docker_error}
    for error in sorted(errors):
        print(f"  Docker did not answer, so no image could be checked: {error}")
    for r in plan.readiness.values():
        for task, why in list(r.stale.items())[:5]:
            print(f"  {r.arm} {task}: {why}")
        if r.unverifiable and not r.docker_error:
            print(f"  {r.arm}: task image not local for {', '.join(r.unverifiable[:8])}"
                  f"{' ...' if len(r.unverifiable) > 8 else ''} (python -m bench.prebake pull)")
        if r.noop:
            print(f"  {r.arm}: label-only no-op image for {', '.join(r.noop)}")
        if r.over_age:
            print(f"  {r.arm}: baked more than {cfg.prebake.max_age_days} days ago: {', '.join(r.over_age[:8])}")
    for note in plan.notes:
        print(f"  note: {note}")


def print_plan(plan: BuildPlan, jobs: int) -> None:
    print(f"Selection: {plan.label}")
    print(f"{len(plan.ready)} (arm, task) pair(s) already have a ready image; "
          f"{len(plan.items)} image(s) to build, ~{plan.minutes:g} min one at a time, "
          f"~{plan.wall_minutes(jobs):g} min with {jobs} at once, ~{plan.need_gb:g} GB (expected size x "
          f"{builder.DISK_FACTOR:g}).")
    by_family: dict[str, list[builder.BuildItem]] = {}
    for item in plan.items:
        by_family.setdefault(item.family, []).append(item)
    for family, items in by_family.items():
        names = [t for i in items for t in i.tasks]
        print(f"  {family} {items[0].version} ({len(items)}, ~{sum(i.expected_min for i in items):g} min, "
              f"~{sum(i.expected_gb for i in items):g} GB): {', '.join(names[:12])}{' ...' if len(names) > 12 else ''}")
        whys: dict[str, int] = {}
        for i in items:
            whys[i.why] = whys.get(i.why, 0) + 1
        for why, n in sorted(whys.items(), key=lambda kv: -kv[1]):
            print(f"      {n} x {why[:150]}")
    for u in plan.unsupported:
        print(f"  {u['arm']}: not pre-baked, runs stock setup: {u['reason']}")
    for u in plan.unknown:
        print(f"  {u['arm']} {u['task']}: {u['reason']}")
    for note in plan.notes:
        print(f"  note: {note}")


def print_disk(check: builder.DiskCheck) -> None:
    host, vm = check.host, check.vm
    print(f"Disk check: {'OK' if check.ok else 'REFUSED'} (needs ~{check.need_gb:g} GB)")
    print(f"  host: {host.get('path') or host.get('drive')} ({host.get('source')}): {_gb(host.get('free_gb'))} free, "
          f"reserve {host.get('reserve_gb'):g} GB -> {'ok' if host.get('ok') else 'too little'}")
    print("  VM: " + (f"{_gb(vm.get('free_gb'))} free, reserve {vm.get('reserve_gb'):g} GB -> "
                       f"{'ok' if vm.get('ok') else 'too little'}" if vm.get("checked") else vm.get("detail", "")))
    for reason in check.reasons:
        print(f"  {reason}")


def _confirm(question: str, yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        print("Not confirmed: pass --yes to go ahead.")
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    plan = make_plan(cfg, args)
    if args.json:
        print(json.dumps(plan.to_dict(cfg.prebake.build_jobs), indent=1, default=str))
        return 0
    print_status(plan, cfg)
    return 0


def cmd_pull(cfg: Config, args: argparse.Namespace) -> int:
    spec = builder.selection_spec(cfg, selection(cfg, args))
    images = inventory.task_images(spec.dataset, (t for a in spec.arms for t in a.tasks))
    unknown = sorted(t for t, i in images.items() if not i)
    wanted = [i for i in images.values() if i]
    todo = pull.missing(wanted)
    print(f"{len(wanted)} task image(s), {len(todo)} not local" + (f"; no image name for {', '.join(unknown)}"
                                                                     if unknown else ""))
    for image in todo:
        print(f"  {image}")
    if args.dry_run or not todo:
        return 0
    cancel = threading.Event()
    try:
        result = pull.ensure_local(todo, sys.stdout, cancel)
    except KeyboardInterrupt:
        cancel.set()
        return 130
    for image, error in (result.get("failed") or {}).items():  # type: ignore[union-attr]
        print(f"FAILED {image}: {error}")
    return 1 if result.get("failed") else 0


def cmd_build(cfg: Config, args: argparse.Namespace) -> int:
    jobs = max(1, min(int(args.jobs or cfg.prebake.build_jobs), 16))
    plan = make_plan(cfg, args, rebuild=args.rebuild)
    print_plan(plan, jobs)
    if not plan.items:
        return 0
    check = builder.disk_check(cfg, plan.items, builder.vm_probe(plan.items))
    print_disk(check)
    if args.dry_run:
        print(f"Dry run: nothing built. Busy: build lock {'held' if builder.build_active() else 'free'}; "
              f"runs in progress: {', '.join(builder.active_runs()) or 'none'}.")
        return 0 if check.ok else 1
    if not check.ok:
        return 1
    if not _confirm(f"Build {len(plan.items)} image(s)?", args.yes):
        return 1
    manager = builder.BuildManager()
    try:
        build = manager.start(cfg, plan, jobs=jobs, setup=args.setup, equivalence=args.equivalence,
                              sample=args.sample)
    except BuildRefused as exc:
        print(f"Refused: {exc}")
        return 1
    print(f"Build {build['id']} started; logs in {manager.log_path(build['id']).parent}")
    return _follow(manager, build["id"])


def _follow(manager: builder.BuildManager, build_id: str) -> int:
    seen: dict[int, str] = {}
    try:
        while not manager.wait(2.0):
            _print_changes(manager.status(build_id) or {}, seen)
    except KeyboardInterrupt:
        print("Cancelling: the running builds are stopped and their staging tags removed...")
        with contextlib.suppress(RuntimeError):
            manager.cancel(build_id)
        manager.wait()
    final = manager.status(build_id) or {}
    _print_changes(final, seen)
    print(f"Build {build_id}: {final.get('status')}. {final.get('message') or ''}".rstrip())
    return 0 if final.get("status") == "done" else 1


def _print_changes(build: dict[str, Any], seen: dict[int, str]) -> None:
    for item in build.get("items") or []:
        line = f"{item['status']}{': ' + item['message'] if item.get('message') else ''}"
        if seen.get(item["n"]) != line:
            seen[item["n"]] = line
            print(f"[{time.strftime('%H:%M:%S')}] #{item['n']} {item['family']} {item['task_image']}: {line}")


def cmd_verify(cfg: Config, args: argparse.Namespace) -> int:
    plan = make_plan(cfg, args)
    if not plan.ready_items:
        print("No ready image in this selection to verify.")
        return 1
    cancel = threading.Event()
    try:
        with builder.build_lock():
            results = builder.verify_ready(cfg, plan, setup=args.setup, equivalence=args.equivalence,
                                           sample=args.sample, log=sys.stdout, cancel=cancel)
    except KeyboardInterrupt:
        cancel.set()
        return 130
    if args.json:
        print(json.dumps(results, indent=1, default=str))
    for r in results:
        print(f"{'OK  ' if r.get('ok') else 'FAIL'} {r['image']}" + "".join(
            f"  {k}={(r.get(k) or {}).get('ok')}" for k in ("static", "smoke", "fingerprint", "setup", "equivalence")
            if k in r))
    return 0 if all(r.get("ok") for r in results) else 1


def cmd_prune(cfg: Config, args: argparse.Namespace) -> int:
    found, notes = prune.candidates(cfg, keep_days=args.keep_days, include_probes=args.include_probes,
                                    everything=args.all)
    for note in notes:
        print(f"note: {note}")
    if not found:
        print(f"Nothing to prune under {cfg.prebake.prefix}/ or {builder.STAGING}/.")
        return 0
    print(f"{len(found)} image tag(s) {'to remove' if args.yes else 'that would be removed (dry run)'}:")
    for c in found:
        print(f"  [{c.kind}] {c.ref}: {c.reason}")
    if args.include_probes and any(c.kind == "probe" for c in found) and not args.yes:
        print("The benchprobe/ images above are the research prototypes; pass --yes to remove them too.")
    if not args.yes:
        return 0
    try:
        result = prune.prune(found, dry_run=False)
    except BuildRefused as exc:
        print(f"Refused: {exc}")
        return 1
    if args.json:
        print(json.dumps(result, indent=1))
    print(f"Removed {len(result['removed'])}; failed {len(result['failed'])}. {result['dangling']}".rstrip())
    for ref, error in result["failed"].items():
        print(f"  {ref}: {error}")
    return 1 if result["failed"] else 0


def cmd_dockerfile(cfg: Config, args: argparse.Namespace) -> int:
    from . import dockerfile, recorder

    plan = make_plan(cfg, args, rebuild=args.rebuild)
    if not plan.items:
        print("Every image this selection needs is ready (--rebuild shows them anyway).")
        return 0
    status = 0
    for item in plan.items:
        adapter = item.adapter()
        stage = (args.out or builder.PREBAKE_DIR / "ctx") / item.slug[:60]
        try:
            recording = recorder.record_for_image(adapter, item.task_image, stage / "stage")
            text, context = dockerfile.render(recording)
        except (recorder.RecipeError, DockerError) as exc:
            print(f"# {item.image}: cannot be pre-baked: {exc}")
            status = 1
            continue
        if args.out:
            ctx = stage / "context"
            ctx.mkdir(parents=True, exist_ok=True)
            for name, source in context.items():
                (shutil.copytree if source.is_dir() else shutil.copy2)(source, ctx / name)
            (stage / "Dockerfile").write_text(text, encoding="utf-8", newline="\n")
            print(f"{item.image}: {stage / 'Dockerfile'}")
        else:
            print(f"# ---- {item.image}  (build context: {', '.join(context) or 'none'})")
            print(text)
    return status


COMMANDS = {"status": cmd_status, "pull": cmd_pull, "build": cmd_build, "verify": cmd_verify, "prune": cmd_prune,
            "dockerfile": cmd_dockerfile}


def main(argv: list[str] | None = None) -> int:
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    args = parser().parse_args(argv)
    try:
        cfg = load(args)
        return COMMANDS[args.command](cfg, args)
    except (ConfigError, BuildRefused, DockerError, KeyError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
