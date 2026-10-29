# Engineering Test Bench

This bench screens agent harnesses (Claude Code, Terminus 2, mini-swe-agent,
Pi and Strands) on fixed models with [Harbor](https://harborframework.com).
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
   worst-case spend, warnings and the peak number of containers. Runs can
   also come from a design (see below). Both kinds take the same path and
   record the same full spec.
2. **Dataset snapshot.** The first run downloads the dataset once into
   `datasets/`, and every arm runs from that fixed copy (see below).
3. **Preflight.** Each arm first runs `harbor run --dry-run`, which validates
   its options and spends nothing. Harbor's dry run also checks that a model
   key is set (it never calls the model), so an arm without its provider key
   fails here; the plan already flags a missing key.
4. **Run.** Each arm is one Harbor job. The `parallel_arms` setting controls
   how many run at once, each supervised by its own worker thread. Every few
   seconds the bench adds up the spend of finished trials, plus an in-flight
   reserve for each trial still running. The reserve is the most expensive
   finished trial so far in that arm (or else in the run), capped by the
   arm's dollar cap where one is enforced. Before any trial has finished, the
   reserve is zero. The run stops when spend plus reserve reaches the budget.
   - Harbor writes a trial's cost only when the trial finishes, so the trials
     in flight can still take a run past its budget. That is at most
     `n_concurrent` x `parallel_arms` trials.
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
   for one run, and each arm's multiple of the baseline next to the published
   figure. **REPORT** pools every run (see below).
   - Read the multiples. Pass rates on 10 tasks are noise.

### Cost: computed and billed

- **Computed cost** is tokens times the rates in `config/pricing.yaml`, the
  same rule for every harness. Multiples use it by default.
- **Reported cost** is what the harness says it spent. On OpenRouter it can be
  the truest figure, because OpenRouter returns the billed cost of each
  request. For example, Pi reported $0.013 where list prices gave $0.042.
- **The ledger keeps both.** REPORT can price by `reported` as an alternative
  view.
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
    kept for audit.
  - Each trial's outcome is classified from Harbor's failure cause: passed,
    failed, agent limit, crash, mispriced cap, setup, provider, infra,
    verifier or incomplete.
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

**Pairing.**
- Each harness is paired with the baseline (the Claude Code family) on the
  same model and identical task content, pooling every matching run.
- Pairing across runs is on. "Same run only" is an option.
- Lines group by harness family per model, with a breakdown by configuration.

**Multiples.**
- A multiple is baseline ÷ harness on the same tasks and model, so 2.0 means
  the harness used half. The published "2x to 3x" is read the same way, as
  baseline ÷ harness between 2 and 3.
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
  below. The verdict never says "confirmed".

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

**Caveats.** Each comparison lists its caveats. For example, Harbor reports no
cache-write tokens on Anthropic, which pulls Claude cost multiples toward 1.

**Coverage: what to run next.**
- For each comparison, coverage shows:
  - the pairs in hand and the current precision;
  - how many more paired tasks reach the target of ±25% at 90%;
  - which tasks to run next, and roughly what that costs.
- With no data yet, the target takes about 22 paired tasks. Coverage never
  goes beyond the 20-task sample in `config/tasks.txt`, so in that case it
  says how wide the interval will remain.

**Options and output.**
- Pricing: `current` (the default), `as_ingested` or `reported`.
- Snapshots save a report with its request, and CSV export lets you recompute
  everything.

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
| `config/tasks.txt` | The fixed 20-task sample, in two balanced blocks of 10 |
| `config/pricing.yaml` | USD per million tokens, by provider and model |
| `config/reference.yaml` | The published multiples the results are compared with, and how they apply (direction, estimator, tolerance, which models) |

Model prices match exactly, or as a dated, `-latest` or tagged variant of a
listed id. So `glm-5.3-flash` is not priced as `glm-5.3`.

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
Docker, Harbor run or key. `tests/__init__.py` points `BENCH_ENV_FILE` at a
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
bench/agents/      harness subclasses run by import path (mini-swe-agent with mcp)
bench/assistant/   the run-design assistant
bench/static/      the dashboard page
harnesses/         add-ons Harbor installs into task containers (strands_factory)
config/            profiles, task sample, pricing, published reference figures
datasets/          dataset snapshots (git-ignored)
ledger/            the trial ledger for all profiles, and its backups (git-ignored)
designs/           default-profile run designs (tracked)
jobs/, runs/       default-profile Harbor output and run records (git-ignored)
assistant/, queue.json   default-profile assistant conversations and spend, and the run queue (git-ignored)
profiles/<name>/   the same for other profiles: designs/ tracked, the rest git-ignored
tests/             unittest suite
briefs/            task briefs for parallel agents
```
