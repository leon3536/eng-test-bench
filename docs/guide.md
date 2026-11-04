# Engineering Test Bench: guide

The full guide to running and extending the bench. For the results and a short overview, see the [README](../README.md).

This bench screens agent harnesses (Claude Code, Terminus 2, mini-swe-agent,
Pi, Strands, Cline, Goose, OpenCode, Kilo and DeepSeek Harness) on fixed models
with [Harbor](https://harborframework.com).
It compares their token use and cost per trial with published figures. The
basis is "Agent harness evidence, ranked" (28 Oct 2026): changing harness
moves cost 2x to 3x and accuracy by under 5 points. The bench checks whether
your own sample shows roughly the same thing.

Every trial from every run goes into one ledger. Runs add up over time, and
the **REPORT** tab tells you where the comparison stands and what to run next.

It runs natively on Windows 11 with Docker Desktop.


## Start

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1               # default profile: Opus 5.5 on all five harnesses
powershell -ExecutionPolicy Bypass -File .\start.ps1 -Profile dev  # cheap OpenRouter models (DeepSeek, GLM)
```

`start.cmd` does the same thing and takes the same arguments
(`start.cmd -Profile dev`). You can also double-click it. If the start fails,
the window stays open with the error until you press a key. On a machine whose
policy already allows local scripts (RemoteSigned or Unrestricted) you can run
`.\start.ps1` directly. If you downloaded the bench as a zip, run
`Get-ChildItem -Recurse | Unblock-File` once.

**Install.** The first start creates `.venv` with `py -3.13` and installs
`requirements.txt`. It writes the marker `.venv\.bench-ready` only after pip
succeeds. So an install that failed or was interrupted is retried on the next
start, and a changed `requirements.txt` is installed again.

**Dashboard.** The console prints a dashboard URL and opens it. The URL's
token is valid until the bench stops, so treat it like a password. It also
lands in your browser history. Restarting the bench makes it invalid.
Everything runs from that one process.

**Stopping.**
- **Ctrl+C:** one press is enough. If a run is active, the bench prints a line
  saying it is stopping that run. Then it kills Harbor's process tree and
  removes the run's containers before it exits. A second press does not leave
  the run going.
- **Closing the console window** also stops the run. Every Harbor process sits
  in a Windows job object that ends with the bench, and a detached helper
  removes the run's containers after the bench has gone.
- **If the bench is killed outright** (for example from Task Manager), the job
  object still ends Harbor. The agents inside the trial containers keep
  running, and keep calling the model with no budget, until those containers
  are removed, so run the cleanup command straight away:
  `.venv\Scripts\python -m bench.procs cleanup <job folder>`.

**One bench per profile.** A lock file stops a second start of the same
profile. The second process prints "The bench is already running for this
profile. Use the dashboard URL printed by that process." and touches nothing.
The default and dev profiles can run side by side on different ports
(`-Port`). Each port keeps its own sign-in.

## Keys

Keys live in `.env`, which git ignores. Set them in the dashboard's
**Systems and keys** panel (click KEYS in the header), or edit the file and
restart:

```
ANTHROPIC_API_KEY=
FIREWORKS_AI_API_KEY=
OPENROUTER_API_KEY=
```

- **Key tests are free.** They call read-only endpoints and use no tokens.
- **Each Harbor process gets only its own provider's key.** Before the bench
  starts Harbor, it removes three kinds of variable from the environment:
  - every credential-like variable: any name containing `KEY`, `SECRET`,
    `TOKEN` (but not `TOKENS`), `PASSWORD`, `CREDENTIAL` or `AUTH`, the same
    rule Harbor uses. So `MAX_TOKENS` is kept. The `GIT_CONFIG_*` settings
    for the line-ending fix are kept too.
  - every endpoint override ending in `_BASE_URL` or `_API_BASE`, so a gateway
    set in your shell cannot silently redirect an arm.
  - the switches that send Claude through Bedrock or Vertex: every `AWS_*`
    variable, `CLAUDE_CODE_USE_BEDROCK`, `CLAUDE_CODE_USE_VERTEX`,
    `CLAUDE_FORCE_OAUTH`, `ANTHROPIC_VERTEX_*` and `CLOUD_ML_REGION`.

  Then it puts back only the arm's provider key.
- **`.env` is not read again inside Harbor.** LiteLLM and Harbor call
  `load_dotenv()`, which would find the project's `.env` and restore the other
  keys. The bench sets `PYTHON_DOTENV_DISABLED=1` (and
  `LITELLM_MODE=PRODUCTION`) for its Harbor processes to prevent that.
  `harbor agents schema` runs with the same cleaned environment and no key.
- **Assistant key.** The run-design assistant uses `ANTHROPIC_API_KEY`. The
  dev profile has no Anthropic arm, so there the key panel shows an extra
  **Assistant** row for it.
- **Provider defaults.** Fireworks is the default for open models, because it
  charges one fixed, published price per model, so computed cost matches the
  bill (decided 2026-10-29). OpenRouter is off in the default profile: it
  routes each request to one of many providers at different prices, and in
  pilot 3 the bill ran 1.3–3.1× the list price. The dev profile uses it.

## How a run works

1. **Plan.** On the **LAUNCH** tab, pick a mode (`pilot` = 1 task,
   `sample` = 10) and the arms, then preview. The preview shows the exact
   Harbor commands, the options each harness gets, the trial count, the
   worst-case spend, warnings and the peak number of containers, and how
   each arm's trials get their harness (see *Pre-baked harness images*).
   Runs can also come from a design (see below). Both kinds take the same
   path and record the same full spec.
2. **Dataset snapshot.** The first run downloads the dataset once into
   `datasets/`, and every arm runs from that fixed copy (see below). Task
   images not on this machine are then pulled (`prebake.prepull`), so a
   first pull of a 15-20 GB image does not count against Harbor's 600 s
   environment limit inside a trial. A run does not start while a pre-baked
   image build runs, and each arm checks its images again right before it
   launches (written to its log).
3. **Preflight.** Each arm first runs `harbor run --dry-run`, which validates
   its options and spends nothing. Harbor's dry run also checks that a model
   key is set (it never calls the model), so an arm without its provider key
   fails here; the plan already flags a missing key.
4. **Run.** By default a run is a series of task races (`run.schedule:
   tasks`, see *Task races* below): one task at a time, every arm on it at
   once. With `run.schedule: arms` each arm is one Harbor job over all its
   tasks, and `parallel_arms` controls how many run at once. Either way each
   Harbor job is supervised by its own worker thread. Every few
   seconds the bench adds up the spend of finished trials, plus an in-flight
   reserve for each trial still running. The reserve is the most expensive
   finished trial so far in that arm (or else in the run), capped by the
   arm's dollar cap where one is enforced. Before any trial has finished, the
   reserve is zero. The run stops when spend plus reserve reaches the budget.
   - Harbor writes a trial's cost only when the trial finishes, so the trials
     in flight can still take a run past its budget. That is at most
     `n_concurrent` x `parallel_arms` trials.
     In a task race that is at most `run.max_containers` trials.
   - Trials killed by a stop write no result, so their spend is missing from
     the total. A stopped run's recorded spend is therefore a lower bound, and
     the run record says how many trials were killed.
   - A trial can also finish with no usage recorded: Harbor writes 0 tokens
     for a timeout that came before any model call completed. Its spend is
     unknown, not $0, so the arm record counts it (`trials_unmeasured`) and
     says the recorded spend is a lower bound. It is not used as the
     reserve's largest trial. The same holds for a provider failure that
     came after the first minute of work with nothing recorded.
5. **Results.** The **ANALYSIS** tab shows tokens, cost and pass rate per arm
   for one run, and each arm's multiple of the baseline (arm ÷ baseline)
   next to the published figure. **REPORT** pools every run (see below).
   - Read the multiples. Pass rates on 10 tasks are noise.

### Task races

**Why.** Building every pre-baked image of a campaign up front filled
Docker's disk: about 240 images, about 460 GB. Images share no layers across
tasks, so the bench now keeps one task's images at a time.

**How** (`run.schedule: tasks`, the default for the profiles and for new
designs). For each task, in the run's order (after reuse has taken out what
the ledger already has, per arm):
1. **Build.** The pre-baked image every arm that runs the task needs is built
   (the IMAGES panel's build manager, up to 6 at once, one per arm image).
   Arms that share a harness, version and install recipe share one image. An
   image that is ready already is used as it is. The disk check runs for this
   task's images only. The verification is the usual one; Harbor's
   install-only setup check runs once per harness, recipe and OS family and is
   remembered across builds (`builds.jsonl`), and so is the file-by-file
   equivalence check (`verify.jsonl`).
2. **Race.** Every arm that runs the task starts its own Harbor job on it at
   once (`--include-task-name <task>`, with its usual flags, pre-baked image
   environment and preflight retry), at most `run.max_containers` (6, the
   host's real CPU cores) trial containers in all. With more arms the task
   races as 6 and then the rest, in arm order, as slots free up.
   `parallel_arms` does not apply.
3. **Remove.** The task's pre-baked images are removed (`docker rmi`, never
   forced, only names under `prebake.prefix` that carry the builder's
   `bake-format` label, never a task image), except an image a later task of
   the run needs (two tasks on one task image) or that a waiting or held queue
   item with `schedule: arms` will look for (a queued task race builds its
   own). `prebake.keep_images: true` keeps them all.

Builds and races never overlap: the run holds the run lock throughout, and its
builds pass the lock as the run's own (`owner_run`), while every other build
and run is still refused. An arm whose image could not be built (or whose
build did not start, e.g. the disk check) follows `run.prebake`: under
`prefer` its trial on that task runs Harbor's stock install with the 720 s
setup limit; under `require` its trial fails at once as an infrastructure
error (the environment class refuses before any container starts), and the
other arms race as usual. Budget, cancel, the reserve for running trials and
the queue work as before; a budget stop or a cancel stops every lane of the
current task, its images are still removed, and the tasks after it are
recorded as skipped (budget) or cancelled.

**Build cache.** BuildKit keeps a cache record for every step of every build,
`--no-cache` or not, and those records hold the layers on disk after the image
is removed. So each race build lists the build cache before it starts
(`docker buildx du --verbose`, or `docker system df -v` without buildx) and,
when it ends, removes exactly the records that were not there before, one
`docker builder prune -f --all --filter id=<id>` each. Nothing is pruned by
age or in bulk, so other projects' images, containers and cache are never
touched; the only record at risk is one another project's build creates in
the same minutes, and then only that cache record. The safeguards:
- Nothing is pruned when the cache cannot be listed before the build, when the
  two listings come from different commands (buildx prints full ids, `system
  df` cuts them to 12 characters, so they cannot be compared), when none of
  the records listed before is listed after, or when there are more new
  records than a build of that many images makes (300 per image). Ids are
  compared on their first 12 characters.
- BuildKit keeps a record that a child still uses, so children are pruned
  before their parents (buildx lists the parents), and the cache is listed
  again after each pass and the rest pruned again, up to 12 passes.
  `pruned` counts the records that are gone; `left` the ones still there.
- If the console closes during a build, the build's docker processes are
  killed and what its prune needs is written to `prebake/cache-pending.json`.
  The next build that prunes its cache finishes it first, and only for the
  records buildx says were created while that build ran (2 minutes slack
  either side); without buildx it prunes nothing and drops the file.

The build's record, the Task race panel and the run's `images.log` show how
many records were created, pruned and left.

**Records.** `run.json` has `schedule` and a `race` block: per task its status
(`waiting`, `building`, `racing`, `done`, `budget_stopped`, `skipped`,
`cancelled`, `interrupted`), its lanes (one arm record per arm on that task),
each arm's image (`ready`, `building`, `built`, `failed`, `not built`, `stock
setup`), the build (id, status, cache) and the images removed or kept with the
reason. The arm records add up their lanes. Job folders are one per (arm,
task): `jobs/<run>/<arm>/<task>/<trial>/`, with `race.json` in the arm folder
marking the layout; the ledger, the results view and the container sweep read
both layouts. `images.log` in the run folder logs each task's build and
removal; each arm's log has a `--- task race: <task> ---` line per task.

**Planning.** The plan (and LAUNCH's preflight) shows the schedule as tasks x
arms, the images to build, the longest task's build minutes and the peak disk
(the largest task's images). A missing image is neither a warning nor, under
`require`, a problem: the race builds it. Build missing is not offered on
LAUNCH then. The estimate's wall time adds up, per task, the build minutes and
the race (each arm's job: 1 minute of preflight and its attempts on
`n_concurrent` slots, `run.max_containers` at once). The live view shows a
**Task race** panel: progress by task, the current task, and each arm's lane
(its image, building, running, done, and the trial's result).

`run.schedule: arms` keeps the earlier behaviour exactly: the same commands and
job folders, all images built beforehand (Build missing, or the queue with
`prebake.auto_build`). Its records are not byte for byte the earlier ones:
`run.json` carries `"schedule": "arms"` and an empty `race`, and the plan has
`schedule` and `race: null`.

**Spec hash and schedule.** `schedule` and `max_containers` are written to a
spec only when they differ from the defaults (`tasks`, 6), so every design
version stored before task races keeps its spec hash. That hash therefore
does not say which schedule a version ran with: a stored version without
`schedule` runs as a task race from now on, also when it ran as arms before
under the same hash. Trials are the same either way (reuse and the ledger
match them on their arm and task, never on the schedule); only the wall time
and the disk differ. A cached design check whose estimate predates task races
(no `schedule` in it) counts as stale and is redone. To run a stored version
as arms, save a new version with `schedule: arms`.

### Cost: computed and billed

- **Computed cost** is tokens times the rates in `config/pricing.yaml`, the
  same rule for every harness. Multiples use it by default. Cache writes are
  priced at their own rates (`cache_write_5m`, `cache_write_1h`: 1.25x and 2x
  input on Anthropic). Harbor folds them into the input count, so the ledger
  reads each trial's write counts from the harness's own log in the trial
  folder.
- **Reported cost** is what the harness says it spent. On OpenRouter it can be
  the truest figure, because OpenRouter returns the billed cost of each
  request. For example, Pi reported $0.013 where list prices gave $0.042.
- **The ledger keeps both.** REPORT can price by `reported` as an alternative
  view. A task's pair uses the reported figures only when both sides have
  them; otherwise both sides use computed cost.
- **Checks at ingest.** Each trial carries flags (`flags` on a trial row) for
  what only its log shows: model attempts declined before a fallback model
  answered (Claude Code), calls that finished but were never saved
  (mini-swe-agent) or never finished (pi), cache writes no log gave, and a
  reported cost more than 0.5% from the computed one. REPORT turns them into
  caveats, with an upper bound on the missing cost where one is known; the
  Ledger table marks such a trial's cost with ⚠ (the checks in its tooltip),
  and the trial drawer lists them under **Cost checks**, with each model's
  cache writes (5-minute / 1-hour, or unknown).
- **Budgets stay on the safe side.** A trial's spend counts as the larger of
  the two, so a lower billed figure never loosens a budget. The one exception
  is a harness known to price a provider wrongly: Claude Code prices every
  model at Claude rates, so off Anthropic its computed cost is used.

### Limits per trial

- **Dollar cap.** Claude Code (`max_budget_usd`) and mini-swe-agent
  (`cost_limit`) take a dollar cap per trial. Claude Code prices every model
  at Claude rates, so its cap would stop trials early on other providers.
  There the bench does not pass it the run's cap, and the plan says so. If an
  arm sets such a cap in its own options anyway, the report leaves the trials
  it stopped out by default.
- **Turn cap.** Terminus 2, Pi and Claude Code take a turn cap.
- **Strands takes neither.** It runs the Strands team's `strands-harness`
  0.3.2 through Harbor's strands adapter (see *Harnesses* below). A strands
  trial is bounded only by its task timeout. The run budget sees its cost
  when it finishes, so trials in flight can take a run past its budget (see
  How a run works). Its subagent and web_fetch calls are not in Harbor's
  token counts, so its recorded spend is a lower bound when it uses them.

## Pre-baked harness images

**Why.** Harbor installs the harness inside every trial container before the
agent starts. For Claude Code that is an apt install of Node.js and npm,
334-550 s per trial, and Harbor's setup limit is 360 s: under parallel load 13
of 23 Claude Code trials failed before they began. A pre-baked image is the
task image with one harness already installed, by the same commands Harbor
issues. A trial on it runs Harbor's unchanged setup, which then finds the
harness and finishes in seconds:

| Harness | Stock setup | Pre-baked |
|---|---|---|
| claude-code | 120-550 s | about 0.5 s |
| terminus-2 | 14-30 s | 2.5-4 s |
| mini-swe-agent | 50-149 s | 3-4 s |
| pi | 37-51 s | 11-20 s (npm still resolves at trial time) |
| strands | 23-78 s | 1.5 s |

**What the agent sees.** The same packages, versions, binaries, PATH and
config as after a stock setup (checked file by file when a recipe is new).
What differs: file times and the apt and dpkg logs show the bake time, Claude
Code's `~/.claude.json` user and machine ids and `/var/lib/dbus/machine-id`
are the same in every trial instead of fresh, and unpinned dependencies (apt
packages, uv, npm and pip trees) are frozen at bake time where stock resolves
them per trial. Per-trial state (the tmux session, `CLAUDE_CONFIG_DIR`, the
agent's run) is never baked. Some harnesses still go online during setup, as
stock does (mini-swe-agent downloads uv; pi runs `npm install -g`).

**How "the same" is checked.** For a new recipe, the build runs Harbor's own
setup twice, stock in a fresh task-image container and on the pre-baked image,
and compares a fingerprint (packages, tool versions, `PATH`, dotfile hashes)
and every file of the two root filesystems. A difference fails the build
unless its path is on a short allowlist of files that differ between any two
stock setups too (`ALLOWED` in `bench/prebake/verify.py`):
- per-container files: `/etc/hostname`, `hosts`, `resolv.conf`, machine ids,
  `/tmp`;
- logs and timestamps: `/var/log`, the ldconfig cache, package manager `-old`
  backups, npm's logs and update-check time;
- caches with random ids: npm's cache index (`~/.npm/_cacache`), uv's cache
  (`~/.cache/uv`), `.pyc` files and `__pycache__`;
- uv's source timestamp for a local package: the `uv_cache.json` and `RECORD`
  of a `.dist-info` (uv stamps the upload time into them). Every file `RECORD`
  lists is still compared on its own;
- nvm's git index and packs (`~/.nvm/.git/index`, `objects/pack/*`): a clone's
  index holds file times and the server packs the same objects differently on
  each fetch. `HEAD`, the refs and the checked-out files are still compared,
  so a different nvm version fails;
- per-install ids: Claude Code's `~/.claude.json` and its backups,
  asciinema's `install-id`.

**Trials stay comparable.** The image is passed as a Harbor environment class
(`--env bench.environments.prebaked:PrebakedDockerEnvironment --ek ...`), not
through agent options, timeouts or task files, so a pre-baked trial has the
same config hash, task digest, pairing and reuse cell as a stock one. The
setup limits (`agent_setup_timeout_multiplier`, `override_setup_timeout_sec`)
are left out of the config hash too: a setup limit cannot change a trial that
got past setup. How each trial got its harness is provenance only: the ledger's
`setup_mode` (`stock`, `prebaked`, `fallback`), REPORT's Setup filter, and
`<trial>/prebake.json`.

**Modes** (`run.prebake`, in Run settings and in a design's run settings):
- `prefer` (the default): a trial uses its pre-baked image when one is ready
  and otherwise falls back to stock setup. The plan warns with counts, and
  `setup_timeout_multiplier: auto` doubles the setup limit (720 s) for arms
  that may fall back.
- `require`: launching waits until every image is ready (a plan problem; the
  queue holds such a design, or builds first with `prebake.auto_build`). An
  image that vanishes mid-run fails its trial at once as an infrastructure
  error instead of a 330-550 s stock install.
- `off`: exactly the stock command, for stock-against-pre-baked comparisons.

Per trial, the environment class checks the image's labels (harness, version,
Harbor version, recipe), that its layers start with the local task image's
layers, and its OS. It never pulls. It writes `prebake.json` in the trial
folder (outside every mount, so the agent never sees it), and the arm log and
message count the trials that ran stock setup.

**Commands.** A selection is `--mode M [--arms a,b]`, `--design ID[@vN]` or
`--run RUN_ID` (what is left of a run), narrowed with `--task`; `--reuse
measured` drops what the ledger already covers.

```powershell
.venv\Scripts\python -m bench.prebake status --design <id>@v1     # which images are ready, missing or stale
.venv\Scripts\python -m bench.prebake pull   --design <id>@v1     # task images not on this machine
.venv\Scripts\python -m bench.prebake build  --design <id>@v1 --dry-run   # what it would build, and the disk check
.venv\Scripts\python -m bench.prebake build  --design <id>@v1 --yes
.venv\Scripts\python -m bench.prebake verify --design <id>@v1 --setup
.venv\Scripts\python -m bench.prebake prune                       # a dry run; --yes removes
```

The dashboard does the same: each LAUNCH module has a **Setup** cell (images
ready of its tasks), the preflight has **Pull** and **Build missing** buttons,
and the **IMAGES** panel (the Images readiness chip, or IMAGES on LAUNCH)
shows each build's progress with a log per image, the disk, and prune with a
dry run first. A build verifies each image before it gets its final name:
labels, layers and a smoke test always; Harbor's own setup in an install-only
trial for the first image of each harness and OS family; and for a new recipe
a file-by-file comparison with a stock setup.
Builds and runs exclude each other: a build refuses while a run or the queue
is active, and a run refuses while a build runs.

**Staleness.** An image's name and labels carry the harness version and a
recipe hash of Harbor's install commands, so a new harness version, a changed
strands factory or a Harbor upgrade makes it stale and trials fall back (or,
under `require`, wait for a build). An image older than `prebake.max_age_days`
(14) still runs, with a warning: its package lists go stale, so an agent's
`apt-get install` without an update can fail where stock would not. Rebake
before each campaign (`build --rebuild`).

**Disk.** Images are per task and harness and share no layers across tasks:
about 1.9 GB per Claude Code image, 1.7 mini-swe-agent, 1.25 strands, 0.6 pi
and 0.2 terminus-2, so 27 tasks x 5 harnesses is about 135 GB built up front
(far more with the build cache). A task race (the default schedule, see *Task
races*) keeps only one task's images, about 6 GB for five harnesses. They live in
Docker Desktop's disk image (`docker_data.vhdx`), which grows on the drive
that holds it and does not shrink by itself after a prune. A build refuses
unless that drive keeps `prebake.host_reserve_gb` (25 GB) and Docker's own
disk keeps `prebake.vm_reserve_gb` (50 GB) free; the **Docker disk** readiness
row shows both. If the disk image is on a small system drive, move it in
Docker Desktop: Settings > Resources > Advanced > Disk image location.

**When a trial ran stock setup** (`prebake.json`'s reason, the arm message,
the ledger's `fallback`):

| Reason | What to do |
|---|---|
| `no pre-baked image ...` | Build it (Build missing, or `build`). |
| `stale: labels differ: harness-version=...` | The arm's version changed: build again. |
| `stale: labels differ: harbor-version=...` | Harbor was upgraded: build again. |
| `stale: labels differ: recipe-sha256=...` | Harbor's install commands changed (options, factory): build again. |
| `stale: baked from ..., but the local task image is ...` | The task image was pulled again: build again. |
| `the task image ... is not local` | Pull it (Pull, or `pull`). |
| `not built FROM the local task image (layer mismatch)` | Build again. |

## Harnesses: names, import paths and families

- **Agent.** An arm's `agent` is either a Harbor agent name, such as
  `claude-code`, or an import path, `module.path:Class`. An import path runs
  with `--agent-import-path`. Harbor 0.25 prints a deprecation warning for that
  flag in the arm log, which is harmless. Harbor and its schema subprocesses
  get the project root on `PYTHONPATH`, so `bench.*` import paths resolve.
- **Family.** An arm's optional `family` is the harness label used for
  reference figures, the cap-mispricing rule, pooling and reports. It defaults
  to the agent string, so give one to every import-path arm.
- **mini-swe-agent arms** use `bench.agents.mini_swe_agent:MiniSweAgentWithMcp`
  with `family: mini-swe-agent`. Harbor installs mini-swe-agent without
  LiteLLM's proxy extra, but LiteLLM 1.106 imports `mcp` from code that
  mini-swe-agent reaches. So every trial failed with
  `ModuleNotFoundError: No module named 'mcp'`. The subclass adds just `mcp`.
- **Cline, Goose and OpenCode arms** use `bench.agents.cline:PinnedClineCli`
  (`family: cline-cli`), `bench.agents.goose:PinnedGoose` (`family: goose`) and
  `bench.agents.opencode:OpenCodeBunEpipe` (`family: opencode`). Harbor's
  cline-cli installs `cline@nightly` unless `cline_version` is set, and Harbor's
  goose installs the latest stable release whatever `version` says; the two
  subclasses make `version` install exactly that release. OpenCode 1.19.4+
  (Bun 1.4.4) can exit 1 on a late `EPIPE` after a finished run; the OpenCode
  subclass keeps such a run, and nothing else, as finished. OpenCode has no
  effort option, so its arms set `effort: omit` and `variant: medium`.
  Harbor's Cline and OpenCode adapters leave reasoning tokens out of the
  output (and OpenCode's, cache writes out of the prompt), so the ledger takes
  these three harnesses' usage from their own logs (bench/extract.py).
- **Kilo Code arms** use `bench.agents.kilo:KiloCode` (`family: kilo-code`).
  Harbor has no Kilo adapter; Kilo is an OpenCode fork with the same JSON
  stream, so the subclass is the OpenCode one (late `EPIPE` included) with
  Kilo's npm package (`@kilocode/cli`, pinned by `version`), binary (`kilo`),
  config (`~/.config/kilo/kilo.json`), `--auto` for OpenCode's
  `--dangerously-skip-permissions` (Kilo still rejects skill-shell and
  sandbox-escalation requests), and its log as `agent/kilo.txt`. Only
  three Kilo defaults change: telemetry (`KILO_TELEMETRY_LEVEL=off`), Claude
  Code's config and skills (`KILO_DISABLE_CLAUDE_CODE=1`) and the update check
  (`KILO_DISABLE_AUTOUPDATE=1`), so it stays comparable with OpenCode. Like
  OpenCode it takes `effort: omit` and `variant: medium`. Its usage comes from
  kilo.txt's `step_finish` events, read as OpenCode's are; the cost Kilo
  reports there is from its own catalog (or OpenRouter's bill) and may be 0,
  which is fine: the bench prices every call itself. ACP mode is refused.
- **Strands arms** use Harbor's built-in `strands` agent with the factory in
  `harnesses/strands_factory`. The factory builds the agent with
  `strands-harness==0.3.2`, which is what the Strands team's published figures
  (S8) measured.
  - The arm must set the Harbor option `version: "1.57.2"` (strands-agents).
    Harbor's default, 1.54.0, cannot be installed together with
    strands-harness.
  - The default profile passes
    `factory_kwargs: {model_spec: anthropic/claude-opus-5-5, effort: medium}`,
    so strands-harness applies effort and prompt caching itself.
    `model_spec` must be the arm's own model. If you change the arm's model,
    change `model_spec` too: a trial whose `model_spec` differs fails at the
    start, before any model call, so it is never billed as the wrong model.
  - Dev arms use the model Harbor builds, so strands-harness applies neither.
  - The factory's README lists what was verified and the known gaps. For
    example, subagent and web-fetch calls are not in Harbor's token counts.
- **Designs** may use only Harbor agent names from the config arms (plus
  `designs.extra_harnesses`) and import paths listed in
  `designs.import_paths`. The config arms' own import paths are added
  automatically. An import path is code that runs on your machine.

```yaml
- name: mini-swe-agent
  agent: bench.agents.mini_swe_agent:MiniSweAgentWithMcp
  family: mini-swe-agent
  provider: anthropic
  model: claude-opus-5-5
```

## The ledger

Every trial from every run, in every profile, goes into one SQLite file,
`ledger/bench.sqlite` (git-ignored).
- **What it covers:** bench runs and plain `harbor run` jobs found under
  `jobs/` and `profiles/*/jobs/`, plus any folders in
  `ledger.extra_scan_roots`. Trials killed before they finished are recorded
  as incomplete, with the usage of the calls that completed, read from the
  harness's own log in the trial's `agent/` folder where it left one. A run's
  recorded spend is a lower bound while it has killed trials
  (`trials.incomplete`), finished trials that recorded no usage
  (`trials_unmeasured`) or finished trials cut short (`trials_lower_bound`).
- **When it updates:** live while a run is going, again when an arm finishes,
  and on a rescan. A rescan runs in the background at startup, from the
  **Rescan** button on REPORT, or from the command line. Ingesting is
  idempotent, and the files on disk are the truth.
- **Facts are kept; judgements are made at report time.**
  - Cost is recomputed from the current `pricing.yaml`; the cost at ingest is
    kept for audit. A re-derive that changes a trial's usage (a new parser
    reading its cache writes, say) prices it again at the table of that
    moment, so `as_ingested` then means "as last re-derived" for that trial.
  - Recovered cache writes (terminus-2) and the cost checks use the rates at
    ingest. After editing `pricing.yaml`, run `rescan --reparse` to redo them.
  - Each trial's outcome is classified from Harbor's failure cause: passed,
    failed, agent limit, crash, mispriced cap, setup, provider, infra,
    verifier or incomplete.
  - How each trial got its harness is recorded as provenance (`setup_mode`,
    the trial row's `setup`): `stock` (Harbor installed it in the trial),
    `prebaked` (the trial ran on a pre-baked image, from the trial's
    `prebake.json`), `fallback` (a pre-baked image was asked for but not
    usable, so Harbor installed it) or empty (killed before its environment
    started). The image, its id, recipe, bake time, the reason and the setup
    time limit are kept with it. It is never part of a trial's configuration
    or task identity, so stock and pre-baked trials of one arm pool and pair.
- **Exclusions** apply to a trial or a whole run and need a reason. They are
  append-only, and an "include" undoes an earlier exclude.
  - One exclusion is seeded: run `dev/20261028-222401-pilot`, because its git
    tasks were damaged by CRLF line endings.
  - It is inserted only once, so if you include that run again, your choice
    sticks.
- **A ledger problem never affects a run.** The arm log gets one
  `[ledger] ...` line, and the next rescan catches up. If the ledger cannot
  open, the dashboard keeps working and REPORT shows the error.
- **Backups:** a daily copy goes to `ledger/backups/`, keeping 7.

```powershell
.venv\Scripts\python -m bench.ledger status
.venv\Scripts\python -m bench.ledger rescan [--reparse] [--root PATH ...]
.venv\Scripts\python -m bench.ledger export --out FILE [--format csv|jsonl]
.venv\Scripts\python -m bench.ledger rebuild   # bench stopped: renames the DB to bench.sqlite.corrupt-<time> and rescans
```

## REPORT

REPORT is a live query over the ledger. You can filter it by date, profile,
model, family, configuration, run and task.

**Task scope.** REPORT opens on the tasks in the profile's task file
(`config/tasks.txt`, or the profile's `tasks_file`), so every harness is
compared with the baseline on the same tasks, and trials on tasks outside
it (an early sample, a pilot) never come in by accident. The strip at the
top says so ("27 tasks from config/tasks.txt"); **All tasks** switches the
scope off in one click until the page is left or reloaded (it is not saved),
and ticking tasks under Filters replaces it. The report says which applied
(`scope`), snapshots keep it, and the report and ledger exports name it in
the file name and give the filters in every row. A design's PAIRS lines
count pairs on the same tasks, so they quote the REPORT's multiple.

**Pairing.**
- Each harness is paired with the baseline (the Claude Code family) on the
  same model and identical task content, pooling every matching run.
- Pairing across runs is on. "Same run only" is an option.
- Lines group by harness family per model, with a breakdown by configuration.

**Multiples.**
- A multiple is harness ÷ baseline on the same tasks and model, so 0.5 means
  the harness used half of what Claude Code used, and under 1 means less.
  The published "2x to 3x" (Claude Code costs 2 to 3 times as much) is read
  the same way, as harness ÷ baseline between 0.33 and 0.5;
  `config/reference.yaml` restates each published figure so (1/3.1 for a
  published 3.1x).
- Two estimators are shown:
  - **Headline:** the geometric mean of the per-task ratios (the typical
    task), with a t interval on the log ratios.
  - **Ratio of means:** the multiple of the average bills, with a task
    bootstrap. This matches the published per-task averages, so verdicts use
    it.
- Every interval is 90% (`stats.confidence`).

**Thresholds.**
- Under 3 paired tasks: no number, only how many more pairs are needed.
- 3 to 7: the estimate is shown greyed, with no verdict.
- 8 or more: a verdict against the published band: within, overlaps, above or
  below. Below means the harness used less, relative to Claude Code, than the
  published figure says; above, more. The verdict never says "confirmed".

**Verdicts only for Claude models.** The published figures were measured on
Claude, so other models get "n/a" plus where the estimate falls
(`reference.yaml` `defaults.applies_to: ['anthropic:claude-*']`). Dev-profile
results are not comparable with the published figures.

**Which trials count.**
- A harness crash after it used the model counts as a failure, with its tokens
  and cost.
- Setup, provider and infrastructure failures are counted and shown, never
  averaged in as zeros.
- A trial whose agent ran but recorded no usage counts for pass rate, but is
  left out of tokens and cost. An example is a timeout before any model call
  completed, which Harbor records as 0 tokens. A reward, a timeout or a
  verifier run means the agent ran even when Harbor wrote no agent timing; a
  provider failure counts as having run once it is past the first minute.
  - Each group counts these trials (`usage_unmeasured`).
  - A comparison gets a caveat whenever either side has any. With one trial
    a side per task, each one costs its task a value on one side, so the
    caveat says how many shared tasks that left without one.
- A timeout that did record usage counts, as a lower bound. The call in
  flight is lost, so the group counts these trials too (`usage_lower_bound`).
  One that recorded fewer tokens than every uncut trial of the same harness
  on the same task is set aside like a trial with no usage: it ran longer
  than they did, so its record is missing part of what it spent.
- A verdict must survive these trials. Each side is scored again with its
  timeouts and unrecorded trials at 2x its usual spend on their task (from
  its uncut trials there, else from the other side's value and the ratio on
  tasks nothing was cut short on). This brings back tasks a side lost. If
  that changes the verdict, the verdict is withheld and says why, and the
  two figures appear under `sensitivity` (`unrecorded_arm_high`,
  `unrecorded_baseline_high`), so `robust` reflects them.
- Trials stopped by a mispriced cap are left out by default.

**Pass rate.** It is shown as a paired difference with its interval and the
smallest difference the sample could detect. Pass rates on 10 to 20 tasks
cannot confirm a 5-point claim.

**Caveats.** Each comparison lists its caveats. For example, a trial that
spent more than 5% of its tokens on a fallback model counts for pass rate
only, and a trial whose own cost differs from the computed one is named.

**Comparing models.** With `compare: models` the harness is held fixed and
each model it ran is compared with the baseline model (`baseline_model` in the
profile, `anthropic:claude-opus-5-5` by default; the report's
`baseline.model` option overrides it), task by task on identical task
content. Multiples read model ÷ baseline model, pass rates are the model's
minus the baseline model's in points, and there are no verdicts: the
published figures compare harnesses. On the dashboard, the **Compare** switch
above the multiples (or Compare in the rail's Advanced section) turns it on;
the cards are then one per harness, headed **Cost vs Opus 5.5** and so on, and
**Baseline model**, next to Baseline family, picks the model the others are
divided by.

**Coverage: what to run next.**
- For each comparison, coverage shows:
  - the pairs in hand and the current precision;
  - how many more paired tasks reach the target of ±25% at 90%;
  - which tasks to run next, and roughly what that costs.
- With no data yet, the target takes about 22 paired tasks. Coverage never
  goes beyond the 20-task sample in `config/tasks.txt`, so in that case it
  says how wide the interval will remain.

**Options and output.**
- Compare: `harnesses` (the default) or `models`.
- Pricing: `current` (the default), `as_ingested` or `reported`.
- Setup: `any` (the default), `stock` (a fallback included) or `prebaked`,
  to compare the trials of one kind only. Each group counts its trials by
  setup (`setup_modes`) and gives its minutes without the harness setup
  (`per_trial.minutes_ex_setup`), since a pre-baked image cuts that setup
  from minutes to seconds. Estimates plan each arm's minutes with the setup
  it will have.
- Snapshots save a report with its request, and CSV export lets you recompute
  everything. A snapshot saved before multiples were harness ÷ baseline is
  shown converted (every multiple, interval and band inverted, above and
  below swapped) and says so.

## Designs and the run queue

A design is a run spec that does not depend on the config arms.
- **Contents:** arms (each with a harness, provider, model, effort, options
  and an optional task subset), an explicit task list, attempts, run
  settings, a budget and a baseline arm.
- **Storage:** immutable, numbered versions under `designs/<id>/` in the
  profile's data folder. Designs are tracked by git, as small records of what
  was run and why.

**Making one.** On the **DESIGNS** tab, start a new design, build one from
config arms, or start from a past run ("Run again"). The assistant can also
save drafts. Each version is:
- validated by the same planner as LAUNCH: harness schemas, keys, enabled
  providers, exact prices, effort, options, tasks, containers and limits;
- estimated: expected and p90 cost, the chance of staying within budget, and
  the worst case under caps.

**Limits** (`designs:`): a budget of at most $150 ($10 in dev), up to 12 arms
and 5 attempts. A warning appears above 8 containers at once, and a design
cannot run above 16.

**Running one.**
- **Add to queue** or **Start now** confirm one specific version by its spec
  hash, and only that exact version runs.
- The queue runs one design at a time, each with its own budget. Parallelism
  comes from the arms within a run (`parallel_arms` x `n_concurrent`).
- **The queue is always paused after a restart.** Press **Start queue** to
  resume.
- It pauses when a run fails or is interrupted (`queue.on_failure: pause`, or
  `skip` to move on). It always pauses when you cancel a run, when Docker is
  down, and when the global cap would be passed.
- **Optional global cap** (`queue.global_cap_usd`, editable in the queue
  panel). An item starts only if the spend since the cap was set, plus the
  item's budget and an in-flight allowance, fits under the cap.
- **Direct LAUNCH runs** still work. They record the same full spec, and they
  are refused while the queue is running.

### Reusing measured trials

With **reuse** set to `measured` (off by default), each arm skips every task
the ledger already holds enough usable trials of, so a run only fills gaps.
It works on designs (`spec.reuse`), on **Add to queue** from LAUNCH, on a
direct LAUNCH run and in the assistant's drafts.
- **Same cell.** A trial counts for an arm only if it has the configuration
  hash the arm's trials will get: family and implementation, provider and
  model, effort as resolved, the arm's own options, the run caps its harness
  takes, and the timeout multiplier. It must also be on the same dataset,
  with the task matched the way REPORT pairs it (by content digest, so it
  must be the version in today's local snapshot, as its latest trial shows
  it; after a fresh download nothing is reused until one trial has run from
  it). The harness version must be known to match. It is pinned to an exact
  release in `options.version` (`2.1.312`, not a tag or range such as
  `latest` or `^2.1`), or the harness ships inside Harbor (Terminus-2).
  Either way the trial must have run on the Harbor installed now: Harbor
  ships Terminus-2, and its adapter for an installed harness reads the
  trajectory and usage the ledger records, so another Harbor can measure the
  same release differently (the check counts the trials this sets aside). An
  unpinned harness is never reused, because Harbor installs its latest
  release in every trial, and neither is unpinned harness code of this
  bench's own (an import path). Whether a trial's harness was installed in
  the trial or came pre-baked in its image is not part of the cell.
- **Usable trial.** It counts in REPORT for pass rate and for tokens under
  the profile's report policy. That means passed, failed or agent limit, and
  a crash while `report.crashes` is `include`, and its usage was recorded. A
  trial cut short counts with its usage as a lower bound, as in REPORT, and
  the check says so. Reuse is stricter than REPORT in two ways, because a
  reused task is never run again for that cell: a trial excluded by hand
  (the latest exclusion wins) never counts, even with `report.exclusions:
  false`, and neither does one that spent more than 5% of its tokens on
  another model (Claude Code falling back after a refusal measured that
  model). Trials from any profile count, because REPORT pools every profile
  and the cell already pins what a profile could change.
- **Covered** means at least `attempts` usable trials. Covered tasks leave the
  arm's task list, and `spec.reused` lists the trials and runs each one
  reuses, so the approved spec hash is exactly what runs. Two arms of one
  cell never share a reused trial. An arm with nothing left is skipped. A run
  with nothing left cannot be queued ("Nothing to run").
- **Checks.** The check shows the trials reused and the cost avoided (from
  those trials' own measured cost), and the estimate covers only the trials
  that run. A saved version's check is redone after any new trial, outcome
  or exclusion, resolving it again against the ledger, and warns if saving
  it again would reuse something different.
- **Queue.** A queued version runs exactly as saved; it is not resolved
  again. If the ledger has since changed which tasks it would run (another
  run measured them, or a reused trial was excluded), the queue holds it
  when its turn comes: save a new version and queue that, or retry the item
  to run it as approved.
- **LAUNCH.** `GET /api/plan?...&reuse=measured` previews the reduced run.
  `POST /api/runs` with `reuse` must send that plan's `spec_hash`, and it is
  refused if the ledger has changed what would be reused since then.

## The assistant

The assistant drawer is available on every view. Claude Opus 5.5
(`claude-opus-5-5`) helps you design runs.
- **What it can do.** It reads the bench's context, the harness descriptions,
  the task list, ledger coverage and cost statistics. It estimates costs,
  suggests runs, lists designs and saves draft designs. Every number it quotes
  comes from the bench's own statistics over the ledger, not from the model.
- **Drafts only.** It cannot start, queue, approve or cancel runs, change
  configuration, or see keys. You review its draft on DESIGNS and press
  **Add to queue** or **Start now** yourself.
- **Settings** (`assistant:`):
  - effort `medium`, fixed per conversation;
  - server-side fallbacks on: a declined request is answered by Anthropic's
    fallback model, and the panel shows which model answered;
  - at most 12 requests per message;
  - $1.00 per turn and $5.00 per local day for each profile, checked before
    every request.
- **Spend** is recorded in `assistant/usage.jsonl`.
- **It needs `ANTHROPIC_API_KEY`.** Without it, the drawer says why and spends
  nothing. A message that looks like a key is refused before anything is
  stored or sent.
- **Conversations** are kept under `assistant/` in the profile's data folder
  (git-ignored).

## Configuration

| File | What it holds |
|---|---|
| `config/bench.yaml` | Default profile: dataset, providers, modes, run settings, the five arms, and the report, stats, assistant, designs, queue and ledger settings |
| `config/dev.yaml` | Dev profile: the same harnesses on DeepSeek and GLM through OpenRouter |
| `config/<profile>.overrides.yaml`, `config/overrides.yaml` | Changes made in the dashboard. The YAML files above are never rewritten. |
| `config/tasks.txt` | The task scope: the tasks runs take, in order, and the tasks REPORT compares on unless told otherwise |
| `config/pricing.yaml` | USD per million tokens, by provider and model |
| `config/reference.yaml` | The published multiples the results are compared with, as harness ÷ Claude Code, and how they apply (direction, estimator, tolerance, which models) |

Model prices match exactly, or as a dated, `-latest` or tagged variant of a
listed id. So `glm-5.3-flash` is not priced as `glm-5.3`.

The `prebake:` section of a profile names and keeps the pre-baked images:
`prefix` (image names start with it), `build_jobs`, `max_age_days`,
`keep_days` (prune keeps an image no arm wants this long), `auto_build` (the
queue builds a design's missing images before it starts it), `prepull`, and
the disk reserves `host_reserve_gb` and `vm_reserve_gb`, and `keep_images`
(a task race keeps each task's images instead of removing them). The run
settings `prebake`, `setup_timeout_multiplier`, `env_build_timeout_multiplier`,
`schedule` (`tasks` or `arms`) and `max_containers` (a task race's containers
at once, 6) are in Run settings with the others.

### Settings per arm

Each arm's settings panel is built from `harbor agents schema <agent>`, so it
offers only options that harness accepts:
- Effort, where the harness supports it.
- Thinking display and thinking budget (Claude Code).
- Interleaved thinking (Terminus 2).
- Turn and cost limits, and everything else Harbor exposes.

An arm's effort is one of three values:
- `inherit`: follows the run setting.
- `omit`: uses the harness default.
- A specific level. YAML reads a bare `off` as false, so quote it: `effort: 'off'`.

Harbor's strands adapter has no effort option. Strands arms set effort in
`factory_kwargs` instead. The default profile passes `medium` to match
`run.effort`. If you change `run.effort`, change the strands arm's
`factory_kwargs.effort` too; nothing keeps them in step automatically. Dev
strands arms run at the model default.

### Task sample

`python -m bench.sample_tasks` redraws `config/tasks.txt`:
- The seed is fixed (`20261028`).
- Tasks with an agent time limit over an hour are left out. For Terminal-Bench
  2.0 these are `build-pov-ray` and `sam-cell-seg`.
- Tasks are drawn in proportion to difficulty, in two blocks of 10. A 10-task
  run uses block 1, and a 20-task run uses both.

## Tests

```powershell
.venv\Scripts\python -m unittest discover -s tests -t .
```

The tests use only the standard library's unittest. They need no network,
Docker, Harbor run or key. Docker is answered by `tests/fakes_docker.py`, and
the pre-bake recorder's probes by real answers canned in
`tests/fixtures/prebake/probes.json`. `tests/__init__.py` points `BENCH_ENV_FILE` at a
temporary file so `.env` is never read, and the Anthropic client is faked.

## Windows notes

- **Line endings.** Harbor fetches tasks with git. With Git for Windows'
  default `core.autocrlf=true`, 833 of Terminal-Bench 2.0's 857 files arrive
  with CRLF endings, including every Dockerfile and shell script, and fail
  inside the Linux containers. The bench sets `core.autocrlf=false` and
  `core.longpaths=true` for its own processes only. Your global git config is
  untouched.
- **Dataset snapshot.** The registry pins Terminal-Bench 2.0 to a branch, not
  a commit, so every `harbor run --dataset` re-downloads its tasks into a
  shared cache and deletes the old copy first.
  - With several Harbor processes running, one deletes files that another's
    containers are using. On Windows that crashed processes and killed
    containers in the first dev pilot.
  - Arms started hours apart could also get different task versions.

  The bench downloads each dataset once into `datasets/` and runs every arm
  with `harbor run --path`. Each run records which snapshot it used.
- **Stopping Harbor.** Harbor cleans up on SIGTERM, which Windows can't
  deliver. When the bench cancels a run or hits the budget, it kills Harbor's
  process tree and removes the trial containers itself.
  - Removal is retried, and a failure is written to the arm log.
  - The job object and the console-close handler cover the cases where the
    bench itself goes away (see *Start*).
- **Docker's disk image.** Docker Desktop keeps images in `docker_data.vhdx`,
  by default under `%LOCALAPPDATA%\Docker\wsl\disk` on C:. Pre-baked images
  grow it by about 135 GB for 27 tasks x 5 harnesses, and it does not shrink
  after a prune without a compaction. Move it to a roomy drive in Docker
  Desktop (Settings > Resources > Advanced > Disk image location); the bench
  finds a moved one through Docker Desktop's settings store.
- **Busy files.** Antivirus scanners, indexers and editors briefly lock files.
  So the bench writes its records and config files to a temporary file, then
  swaps it in, and retries for a moment if the file is busy. If a config file
  stays locked, the dashboard says so and asks you to try again. A
  `queue.json` that cannot be read at startup is copied aside as
  `queue.json.corrupt-<time>` (git-ignored) and the queue starts empty and
  paused; a locked one is never overwritten until that copy has been kept.

## Layout

```
bench/             the bench: settings, keys, schemas, datasets, runs, results, checks, server,
                   plus ledger, report, stats, designs, runqueue, estimates, coverage and procs
bench/agents/      harness subclasses run by import path (mini-swe-agent with mcp; pinned cline and
                   goose; opencode past Bun's late EPIPE; kilo code on the same base)
bench/prebake/     pre-baked harness images: recipes, builds, verify, prune (python -m bench.prebake)
bench/environments/ the Harbor environment class a run passes with --env (pre-baked image, or stock fallback)
bench/assistant/   the run-design assistant
bench/static/      the dashboard page
harnesses/         add-ons Harbor installs into task containers (strands_factory)
config/            profiles, task sample, pricing, published reference figures
datasets/          dataset snapshots (git-ignored)
ledger/            the trial ledger for all profiles, and its backups (git-ignored)
prebake/           the image builder's contexts, logs, build records and locks, shared by every profile (git-ignored)
designs/           default-profile run designs (tracked)
jobs/, runs/       default-profile Harbor output and run records (git-ignored)
assistant/, queue.json   default-profile assistant conversations and spend, and the run queue (git-ignored)
profiles/<name>/   the same for other profiles: designs/ tracked, the rest git-ignored
tests/             unittest suite
briefs/            task briefs for parallel agents
```
