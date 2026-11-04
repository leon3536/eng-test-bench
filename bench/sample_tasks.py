"""Draw the fixed task sample in config/tasks.txt.

    python -m bench.sample_tasks

Uses the dataset snapshot in datasets/ (downloading it first if needed), drops tasks
whose agent time limit is over an hour, then draws two blocks of ten in
proportion to difficulty with a fixed seed. A 10-task run uses the first
block; a 20-task run uses both, and each block is balanced on its own.
"""

from __future__ import annotations

import math
import random
import sys
import tomllib
from collections import defaultdict
from datetime import date
from pathlib import Path

from . import datasets, fsutil
from .settings import CONFIG_DIR, ROOT, load_config, prepare_process_env

SEED = 20261028
BLOCK = 10
BLOCKS = 2
MAX_AGENT_TIMEOUT_SEC = 3600
PILOT_TIMEOUT_SEC = 900


def read_tasks(dataset_dir: Path) -> list[dict]:
    tasks = []
    for task_dir in sorted(p for p in dataset_dir.iterdir() if (p / "task.toml").exists()):
        meta = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        tasks.append({
            "name": task_dir.name,
            "difficulty": (meta.get("metadata") or {}).get("difficulty", "unknown"),
            "timeout": float((meta.get("agent") or {}).get("timeout_sec") or 0),
        })
    return tasks


def proportional(counts: dict[str, int], size: int) -> dict[str, int]:
    """Largest-remainder apportionment."""
    total = sum(counts.values())
    exact = {k: size * n / total for k, n in counts.items()}
    out = {k: math.floor(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - out[k], reverse=True)[: size - sum(out.values())]:
        out[k] += 1
    return out


def draw(tasks: list[dict], rng: random.Random) -> list[dict]:
    pool = defaultdict(list)
    for t in tasks:
        pool[t["difficulty"]].append(t)
    picked = []
    for _ in range(BLOCKS):
        quota = proportional({k: len(v) for k, v in pool.items() if v}, BLOCK)
        block = []
        for difficulty in sorted(quota):
            chosen = rng.sample(pool[difficulty], quota[difficulty])
            block += chosen
            pool[difficulty] = [t for t in pool[difficulty] if t not in chosen]
        rng.shuffle(block)
        picked.append(block)
    # The pilot runs the first task: make it a medium one with the standard time limit.
    first = picked[0]
    for i, t in enumerate(first):
        if t["difficulty"] == "medium" and t["timeout"] <= PILOT_TIMEOUT_SEC:
            first.insert(0, first.pop(i))
            break
    return [t for block in picked for t in block]


def main() -> None:
    prepare_process_env()
    cfg = load_config()
    tasks = read_tasks(datasets.ensure(cfg.dataset))
    eligible = [t for t in tasks if t["timeout"] <= MAX_AGENT_TIMEOUT_SEC]
    excluded = sorted(t["name"] for t in tasks if t not in eligible)
    sample = draw(eligible, random.Random(SEED))

    lines = [
        f"# Task sample for {cfg.dataset}, drawn {date.today().isoformat()} by `python -m bench.sample_tasks`.",
        f"# Seed {SEED}. {len(tasks)} tasks; excluded {len(excluded)} with an agent time limit over "
        f"{MAX_AGENT_TIMEOUT_SEC // 60} minutes: {', '.join(excluded) or 'none'}.",
        f"# Two blocks of {BLOCK}, each in proportion to difficulty. The pilot uses line 1.",
        "",
    ]
    for i, t in enumerate(sample):
        if i % BLOCK == 0:
            lines.append(f"# Block {i // BLOCK + 1}")
        lines.append(f"{t['name']:<40} # {t['difficulty']}, {int(t['timeout'])}s")
    out = CONFIG_DIR / "tasks.txt"
    if not fsutil.atomic_write_text(out, "\n".join(lines) + "\n"):
        raise SystemExit(f"{out.relative_to(ROOT)} is open in another program. Close it and run this again.")
    print(f"Wrote {len(sample)} tasks to {out.relative_to(ROOT)}; excluded {excluded}", file=sys.stderr)


if __name__ == "__main__":
    main()
