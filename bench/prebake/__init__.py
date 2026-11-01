"""Pre-baked harness images: a task image plus one harness's exact Harbor
install, baked once, so a trial's own unchanged setup() short-circuits or
finishes in seconds instead of installing the harness in every container.

- recorder.py records the install commands from each harness's real Harbor
  setup() code; adapters.py holds what differs per harness; dockerfile.py
  turns a recording into a Dockerfile that replays them byte for byte.
- bench/environments/prebaked.py is the Harbor environment class a run passes
  with --env; per trial it uses a verified image or falls back to stock setup.

- The builder (python -m bench.prebake: inventory.py, pull.py, builder.py,
  verify.py, prune.py) works out which images a selection needs, builds them
  under staging tags, verifies them and only then gives them their names.

This module stays light (the recorder imports Harbor's agents only when an
arm is resolved): runs.py imports it for every plan.
"""

from __future__ import annotations

import dataclasses
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ..harnesses import is_import_path
from ..settings import DATA_DIR, PREBAKE_OFF, PREBAKE_REQUIRE, ROOT, Config, RunSettings

DEFAULT_PREFIX = "benchbake"
# The builder's own files: build contexts, logs, builds.jsonl, fingerprints,
# setup-check trials and the build and run locks. Docker's images are shared
# by every profile, so this folder is too, and it is outside every ledger scan
# root. BENCH_DATA_DIR (tests, a second checkout) moves it with the run data.
PREBAKE_DIR = DATA_DIR / "prebake" if os.environ.get("BENCH_DATA_DIR") else ROOT / "prebake"
# The setup limit an arm gets from setup_timeout_multiplier 'auto' when some of
# its trials will run Harbor's stock install: 2 x Harbor's 360 s.
AUTO_SETUP_MULTIPLIER = 2.0
# Set once warm() has recorded every enabled arm's install, which loads
# Harbor's agent code (5-13 s, mostly LiteLLM). Status checks wait for it
# rather than stall a dashboard poll on an import another thread holds.
_loaded = threading.Event()


def recipes_ready() -> bool:
    return _loaded.is_set()


def warm(cfg: Config) -> None:
    """Resolves every enabled arm once, as prefer would, so Harbor's agent
    code is loaded before the first plan asks. Run in the background at start."""
    try:
        from .inventory import wanted_recipes

        wanted_recipes(cfg, all_profiles=False)  # resolves under prefer, whatever cfg's run.prebake says
    except Exception:  # surfaced by the plan that needs it
        pass
    finally:
        _loaded.set()


class _Arm(Protocol):
    name: str
    agent: str
    family: str


@dataclass(frozen=True)
class ArmPrebake:
    """How one arm uses pre-baked images under its run's prebake mode."""
    mode: str                     # prefer | require (off resolves to no ArmPrebake)
    family: str
    supported: bool
    reason: str = ""              # why the arm is not supported, so it runs stock setup
    version: str | None = None    # the harness version in the image name
    recipe: str | None = None     # sha256 of the arm's install recipe (recorder.arm_recipe)
    variant: str | None = None    # plain | pi-warm | strands-skip
    prefix: str = DEFAULT_PREFIX
    # Whether some trial is expected to run Harbor's stock install: always
    # true until an image inventory says every task's image is ready.
    expect_stock: bool = True

    @property
    def required(self) -> bool:
        return self.mode == PREBAKE_REQUIRE

    def env_kwargs(self) -> dict[str, Any]:
        """The --ek values for PrebakedDockerEnvironment; {} when unsupported."""
        if not self.supported:
            return {}
        return {"prebake_harness": self.family, "prebake_version": self.version, "prebake_recipe": self.recipe,
                "prebake_prefix": self.prefix, "prebake_required": self.required}

    def summary(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def resolve(cfg: Config, run: RunSettings, arm: _Arm, kwargs: Mapping[str, Any]) -> ArmPrebake | None:
    """The arm's prebake under run.prebake; None when the mode is off. Never
    raises: an arm whose install cannot be recorded is unsupported, with the
    reason, and runs stock setup."""
    if run.prebake == PREBAKE_OFF:
        return None
    family = arm.family or arm.agent
    base = {"mode": run.prebake, "family": family, "prefix": cfg.prebake.prefix}
    if is_import_path(arm.agent) and arm.agent not in cfg.designs.import_paths:
        # Recording imports the agent's code into this process: only a path the config allows.
        return ArmPrebake(**base, supported=False, reason=f"{arm.agent} is not an allowed harness import path")
    try:
        # Imported here: the recorder imports Harbor's agents, which only a prebake mode needs.
        from ..environments.prebaked import prebaked_image_name
        from .adapters import adapter_for
        from .recorder import arm_recipe
        adapter = adapter_for(arm.agent, family, kwargs, cfg.harbor_model(arm))  # type: ignore[arg-type]
        if adapter is None:
            return ArmPrebake(**base, supported=False, reason=f"no pre-bake adapter for {family}")
        if not adapter.version:
            return ArmPrebake(**base, supported=False, reason="no pinned version, so Harbor installs the latest in "
                                                              "every trial and an image would go stale")
        recipe = arm_recipe(adapter)
        # The task part of the name is checked per trial; this checks the arm's part.
        prebaked_image_name(cfg.prebake.prefix, family, adapter.version, recipe.sha256, "task:tag")
    except Exception as exc:  # a planning aid: the arm still runs, with stock setup
        return ArmPrebake(**base, supported=False, reason=f"its install could not be recorded: "
                                                          f"{type(exc).__name__}: {exc}")
    return ArmPrebake(**base, supported=True, version=adapter.version, recipe=recipe.sha256,
                      variant=adapter.variant)


def setup_timeout_multiplier(run: RunSettings, prebake: ArmPrebake | None) -> float | None:
    """The --agent-setup-timeout-multiplier for an arm, None for none. A number
    applies to every arm. 'auto' doubles the limit only for an arm whose
    trials may run Harbor's stock install under a prebake mode: unsupported, or
    falling back. Mode off ('auto' gives nothing) keeps today's command, and
    require never runs stock setups (a missing image fails fast instead)."""
    value = run.setup_timeout_multiplier
    if value != "auto":
        return None if float(value) == 1 else float(value)
    if prebake is None:
        return None
    if not prebake.supported or (prebake.expect_stock and not prebake.required):
        return AUTO_SETUP_MULTIPLIER
    return None
