# Brief v2: Report, Designs, Queue and Assistant in the dashboard (`bench/static/*`)

This brief stands on its own. Read it in full before you start. The v1 brief
(`briefs/dashboard-frontend.md`) still describes the parts of the dashboard
that have not changed. Every shape quoted here was taken from the backend as
it is built (`bench/server.py` and the modules it calls). Each one was checked
by calling the real app in process over temp data.

## What changed and why

Until now the bench only had per-run results. You could see one run's
multiples, but you could not ask "where are we across everything we have
run?", and every run had to be configured by hand on LAUNCH. The backend now
records every trial from every run and profile in a SQLite **ledger**. A
**pooled report** over that ledger pairs each harness with the baseline
(Claude Code) on identical tasks and the same model. It gives multiples with
confidence intervals, pass-rate differences with intervals, verdicts against
the published figures, exclusions and coverage ("what to run next").

Runs can now be described as **designs**: immutable, numbered versions of a
run spec, each identified by a `spec_hash`. A design becomes a billed run only
through the **run queue**. A person admits one exact version by its hash, and
the queue runs designs one at a time and pauses on any failure.

A Claude Opus 5.5 **assistant** reads the ledger, estimates costs and saves
*draft* designs. It cannot queue, approve, start or cancel anything. Its
output streams to the browser over server-sent events (SSE).

Your job is the front end for all four, plus the small changes the existing
views need.

## Glossary (you will see these names everywhere)

| Term | Meaning |
|---|---|
| **agent** | What Harbor runs. Either a registered name (`claude-code`, `pi`, `strands`, `terminus-2`) or an import path `module.path:Class` (for example `bench.agents.mini_swe_agent:MiniSweAgentWithMcp`). |
| **kind** | `"name"` or `"import_path"`, derived from the agent. |
| **family** | The harness label that results pool and compare under, for example `mini-swe-agent` for the import-path agent above. It defaults to the agent. Always show the family; show the agent as secondary text. |
| **multiple** | **baseline ÷ harness** on the same tasks and model. 2.0× means the harness used half of what Claude Code used. |
| **trial_key** | `<harbor job id>/<trial name>`, one trial in the ledger. |
| **run_key** | `<profile>/<run id>`, for example `dev/20261028-222952-pilot`. |
| **group key** | One pooled line in the report, for example `fam:openrouter\|z-ai/glm-5.3\|pi`. |
| **comparison key** | `<arm group key>~<baseline group key>`. |
| **config_hash** | 16 hex characters identifying one exact harness configuration (family, provider, model, effort, options). |
| **design id** | `d-<slug>-<4 hex>`, for example `d-block-1-on-deepseek-b7a8`. Versions are 1, 2, 3, ... and never change once written. |
| **spec_hash** | `sha256:<64 hex>` of a version's normalized spec. Show the first 12 hex characters after `sha256:`. |
| **queue item** | `q-0001` and so on: one admitted design version in the queue. |
| **conversation id** | `c-YYYYMMDD-HHMMSS-<4 hex>`. |

## Your scope

- You own `bench/static/*`: `app.js`, `index.html`, `style.css`, `demo.js`.
- You may add more plain scripts under `bench/static/` (for example
  `report.js`, `designs.js`, `queue.js` and `assistant.js`), loaded with
  `<script src>` after `app.js`. They are served from `/static/<name>`.
- The v1 rules for these files still apply:
  - plain JavaScript, with no build step, no libraries and no CDNs;
  - it must work offline;
  - keep the existing element ids and classes.
- Reuse the helpers that are already in `app.js`: `h`, `fill`, `api`, `busy`,
  `toast`, `confirmAction`, `tip`, `panel`, `field`, `money`, `times`,
  `tokens`, `renderMarkdown` and `changed`.
- Do **not** edit `bench/*.py`, `config/*`, `tests/*`, `harnesses/*` or
  anything else outside `bench/static/`. If you need an API change or find a
  backend bug, put it in your report with the endpoint and a short
  reproduction.
- Do not commit.

## Hard rules (v1's rules, extended for the new billed paths)

1. **Never read, print, create or edit `.env` or any `.env.*` file.** Point
   the test server at a throwaway file with `BENCH_ENV_FILE` (see Testing).
2. **Never call an endpoint that starts a billed run, enqueues a design, or
   makes a model or provider request.** Docker Desktop **is running** on this
   machine now, so the queue would really start containers and bill
   accounts. Wire these endpoints to the UI, but exercise them only in demo
   mode (see Testing), never against a server:
   - `POST /api/runs`
   - `POST /api/queue/items` (any body)
   - `POST /api/queue/start`
   - `POST /api/queue/items/{id}/retry`
   - `POST /api/assistant/conversations/{cid}/messages`
   - `POST /api/assistant/conversations/{cid}/retry`
   - `POST /api/keys/{ENV_KEY}/test`. It calls the provider's API. It is
     unchanged since v1, so do not run it again.
3. **Leave config and data as you found them.**
   - Run your test server with `BENCH_DATA_DIR` and `BENCH_LEDGER` set to
     temp folders (see Testing). Without them, a design you save is written
     to `designs/` in the repository, which git tracks, and exclusions and
     snapshots go into the shared `ledger/bench.sqlite`, which the dev server
     holds open.
   - `config/overrides.yaml` exists now. It holds only a header comment and
     `{}`. Saving forms writes to it. When you finish, call
     `DELETE /api/config/overrides`, which rewrites exactly that content, and
     confirm that `/api/state` → `config.has_overrides` is `false`.
4. Key values may appear only in the password input of `#key-dialog`, as in
   v1. Never render a key anywhere else. The API returns masked values only.
5. Ports and processes:
   - Your test server uses port **8799**, as in v1.
   - A dev-profile bench runs on **8798** with real data under
     `profiles/dev/`. Leave that port and those files alone.
   - Stop only processes you started.

## API basics (unchanged from v1, plus three additions)

- **Sign-in:** open `/?token=<token>`. The token is printed at startup. It
  sets an HttpOnly session cookie and redirects to `/`. The cookie is now
  named `bench_session_<port>`, so two benches on one machine no longer sign
  each other out. The browser handles this for you.
- **Header:** every non-GET request must send `X-Bench: 1`. That includes
  the read-only report POSTs below.
- **Errors** are `{"detail": "..."}`. Show `detail` in the toast.

  | Status | Meaning |
  |---|---|
  | 400 | Invalid input, including validation errors joined with `; `. |
  | 401 | Not signed in. |
  | 403 | Wrong host, or the `X-Bench` header is missing. |
  | 404 | Not found. |
  | 409 | Conflict: stale version, hash mismatch, queue active, Docker down, a turn already running. |
  | **503, new** | A config file or the ledger is busy or locked: `"A config file is busy (another program has it open). Try again."` or `"The ledger is busy or locked by another program; try again"`. An unavailable ledger also answers 503 on every ledger and report route except `GET /api/ledger/status`. Treat 503 as "try again in a moment", not as a disconnect. |

- **Times** are ISO-8601 UTC. Ledger times end in `Z` with microseconds;
  design, queue and assistant times end in `+00:00`. `new Date()` parses
  both. Money is USD. `null` means "not measured". Never show a null as 0.
- **File downloads come from POST routes** (the CSV and JSONL exports).
  1. `fetch` with the `X-Bench` header.
  2. Take `res.blob()`.
  3. Read the file name from `Content-Disposition`.
  4. Make an object URL and click a temporary `<a download>`.
  5. Revoke the object URL.

---

## Changed endpoints and shapes

### `GET /api/state`: three new blocks, and changes inside `config` and `runs`

```
{ readiness: [...], keys: [...], config: {...}, config_error, runs: [Run], active_run,
  ledger:    { available: bool, error: str|null, trials: int, last_ingest_at: iso|null, rescan_running: bool },
  queue:     { state: "paused"|"running", pause_reason: str, active_item: QueueItem|null,
               waiting: int, held: int, cap_remaining_usd: number|null },
  assistant: { enabled: bool, available: bool, reason: str|null, today_usd: number|null,
               daily_cap_usd: number, active_turn: {conversation_id, turn, started_at}|null } }
```

- If one of the three new blocks fails, it arrives as `{"error": "<what>: <type>: <message>"}`.
  Handle that, because the rest of the state still loads.
- **`readiness` has a sixth row, `"Assistant"`.** Examples:
  `{ok: false, detail: "Set ANTHROPIC_API_KEY on Setup"}`,
  `{ok: true, detail: "Claude Opus 5.5 · effort medium"}` or `"Disabled in config"`.
  Add it to `SYS_LABEL`, for example as `'Assistant': 'AI'`.
- **`keys`** can include a row with `provider: "assistant"` and `env_key: "ANTHROPIC_API_KEY"`.
  It appears in profiles where no provider uses that key (the dev profile).
  Label it "Assistant (Claude Opus 5.5)". Set, Replace and Clear work as for
  any key.
- **`config.arms[]`** gains `family: str` and `kind: "name"|"import_path"`.
  Show the family as the arm's harness label. Tag import-path agents as
  "custom" and show the import path in small mono text.
- **`config.orphaned_overrides`** is new: `{arms: [name], modes: [name]}`.
  These are overrides for arms or modes that `bench.yaml` no longer has. They
  would reattach silently if the name came back.
  - Show a warn banner in the settings drawer listing them.
  - Give each orphaned arm a Reset button (`DELETE /api/config/arms/{name}`).
  - Give the modes one Reset button (`DELETE /api/config/modes`).
- **`runs[]`** (Run) gains new fields. `mode` is the config mode, or `"design"` for queued runs.

  | Field | Value |
  |---|---|
  | `profile` | Profile name. |
  | `source` | `"config"` (LAUNCH) or `"design"` (queue). |
  | `spec` | The normalized spec it ran (shape under Designs). |
  | `spec_hash` | Hash of that spec. |
  | `design_ref` | `{design_id, version, author: "user"\|"assistant", conversation_id\|null, queue_item: "q-0001"\|null}`, or `null` for LAUNCH runs. |
  | `harbor_version` | Harbor version string. |
  | `reserve_usd` | Budget held back for trials still running. |
  | `resolved_arms` | `[{name, agent, family, kind, provider, model, harbor_model, effort: {setting, value, label}, kwargs: [{name, value, source}], options_fingerprint, tasks: [str], trials, command}]` |

  - Run ids: LAUNCH runs look like `20261029-120000-pilot`. Design runs look
    like `20261029-120000-block-1-on-deeps-v2` (16-character slug plus
    version).
  - **ArmRun** gains:
    - `family`;
    - `trials_running`;
    - `reserve_usd`: held back from the budget while trials run;
    - `trials_killed`: stopped mid-run, so their spend is not in `spend_usd`;
    - `cleanup_pending`: containers may still exist, and the next start retries;
    - `pid`.
  - Old runs may lack any of these new fields. Treat a missing field as unknown.

### Config saves

- `PUT /api/config/run`, `PUT /api/config/modes/{mode}` and
  `PUT /api/config/arms/{name}` now answer **409** with:
  `"bench.yaml or overrides.yaml changed on disk since the dashboard loaded them. Press Reload config, then save again."`
  Show it in the toast. Then offer the Reload button, but keep the user's
  form values in place.
- `PUT /api/config/arms/{name}` accepts **`family`** (text). An empty string
  resets it to the default.
  - Add a Family text input to the arm form, with the default agent as its
    placeholder.
  - Options the editor does not show are kept server-side, so sending the
    options you show is still right.

### `GET /api/schemas/{agent}`

- It now accepts any harness a design may use, including import paths.
- `encodeURIComponent(agent)` works as it is (`:` becomes `%3A`).
- The shape is unchanged.
- An agent that is not allowed answers 404.

### `GET /api/plan?mode=&arms=`

The shape is the same as before, with these additions:

```
top level: source: "config", label: "<mode>", spec_hash: "sha256:...", warnings: [str], peak_containers: int
per arm:   family, kind, tasks: int (task count), options_fingerprint: "12 hex", reserve_note: str
```

Notes:

- `model` per arm is the Harbor model, for example
  `openrouter/deepseek/deepseek-v4.1-flash`.
- An unpriced model is now listed in `problems`.
- Commands start `harbor --% run ...`. `--%` is PowerShell's stop-parsing
  token, so a command pastes into PowerShell as it is. Keep them in `pre.cmd`.
- Show in the preflight:
  - `warnings`;
  - "Peak N containers";
  - each arm's `reserve_note` as small text (it explains how much budget a
    running trial holds back).

### `POST /api/runs` (do not call)

- The body is unchanged: `{mode, arms}`. Extra fields are ignored. Stop
  sending `launch_code`: launch codes are dropped (decided 2026-10-29, see
  LAUNCH below).
- New 409s:
  - `"The queue is active; pause it or add this run to the queue"`;
  - `"Docker is not running"`;
  - a global-cap message from the queue.

### `GET /api/runs/{id}/results`

The shape is the same, with additive fields only:

| Where | New fields |
|---|---|
| `arms[]` | `family`, `provider`, `measured` (trials that count for tokens and cost), `excluded` (setup failures with no tokens, left out of the means), `cost_partial` |
| `comparison[]` | `family`, `verdict_note` (why no verdict, for example a non-Claude model), `excluded`, `baseline_excluded` |
| `trials[]` | `family`, `failure_cause`, `reward_valid`, `reported_mispriced`, `cost_partial`, `unpriced_floor_cost` |

Add a small "N setup failures left out" note per arm when `excluded > 0`.

---

## New endpoints

### Ledger

| Method and path | Body | Returns |
|---|---|---|
| `GET /api/ledger/status` | | LedgerStatus (below). It answers even when the ledger is unavailable (`available: false`, `error` set). |
| `POST /api/ledger/rescan` | `{reparse?: bool}` | `{at, scanned, inserted, upgraded, unchanged, incomplete, skipped: [{path, reason}], seconds}`. 409 `"... already running"` while another rescan runs, including the one at startup. |
| `GET /api/ledger/facets` | | Facets (below), for the filter rail |
| `GET /api/ledger/runs` | | `[LedgerRun]` across all profiles |
| `POST /api/ledger/trials` | `{filters?: Filters, outcomes?: [outcome], limit?: 1..1000 (default 100), offset?: int, order?: "<col> ASC\|DESC"}` | `{total: int, rows: [TrialRow]}`. `col` is one of `started_at`, `task`, `family`, `tokens_total`, `cost_usd`. The default order is `started_at DESC`. |
| `GET /api/ledger/trials/{id}` | | TrialRow plus detail (below). `id` is the numeric `id`, not the trial_key. 404 if unknown. |
| `POST /api/ledger/exclusions` | `{target_type: "trial"\|"run", target_id: trial_key\|run_key, action: "exclude"\|"include", reason: str (1-500)}` | **201** `{id, target_type, target_id, action, reason, created_at}`. 400 for a missing reason, 404 for an unknown target. |
| `GET /api/ledger/exclusions` | | `{exclusions: [{id, target_type, target_id, action, reason, created_at, effective: bool}]}`. The latest row for a target wins. `effective` marks the one in force. |
| `POST /api/ledger/export` | `{filters?: Filters, format: "csv"\|"jsonl"}` | A file: `ledger-<UTC stamp>.csv` (BOM, CRLF) or `.jsonl`. |

LedgerStatus (real values from the current data):

```json
{ "available": true, "error": null, "path": "C:/.../ledger/bench.sqlite", "size_bytes": 172032,
  "schema_version": 1, "parser_version": 1, "classifier_version": 1, "stats_version": 1,
  "trials": 16, "runs": 2, "by_profile": {"dev": 16},
  "by_outcome": {"failed": 2, "incomplete": 1, "misconfigured_cap": 2, "passed": 1, "setup_failure": 10},
  "last_ingest_at": "2026-10-29T12:41:18.169698Z",
  "rescan": {"running": false, "last": {"at": "...Z", "scanned": 16, "inserted": 15, "upgraded": 0, "unchanged": 0,
                                         "incomplete": 1, "skipped": [], "seconds": 0.347}},
  "last_error": null, "last_backup": "bench-20261029.sqlite", "pricing_sha": "2304790bb213",
  "seeded_exclusions": [{"target_type": "run", "target_id": "dev/20261028-222401-pilot",
                         "reason": "Pilot 1 ran on git tasks damaged by CRLF line endings",
                         "effective_action": "exclude"}],
  "journal_mode": "wal", "warnings": [] }
```

`last_error` is `{at, where, message}` or `null`.

**Outcomes**, in this order:

`passed`, `failed`, `agent_limit`, `agent_crash`, `misconfigured_cap`,
`verifier_failure`, `provider_failure`, `setup_failure`, `infra`, `incomplete`

Use a glyph as well as colour so colour is never the only cue:

| Outcome | Glyph |
|---|---|
| passed | ● |
| failed | ○ |
| agent_limit | ▲ |
| agent_crash | × |
| misconfigured_cap | $ |
| verifier_failure | ? |
| provider_failure | ≈ |
| setup_failure | ◦ |
| infra | ! |
| incomplete | … |

Plain labels:

| Outcome | Label |
|---|---|
| agent_limit | "hit its own limit" |
| agent_crash | "harness crashed" |
| misconfigured_cap | "stopped by a mispriced cap" |
| setup_failure | "never got going" |
| provider_failure | "provider error" |
| infra | "infrastructure" |
| incomplete | "killed before finishing" |

Facets. All lists carry trial counts:

```
{ profiles: [{value, trials}], sources: [{value: "bench"|"external", trials}],
  models: [{key: "openrouter:z-ai/glm-5.3", provider, model, trials}],
  families: [{value, trials, implementations: ["builtin"|import path], versions: [str]}],
  configs: [{config_hash, label, family, implementation, provider, model, effort, options: {}, cautions: [str],
             trials, first_seen, last_seen}],
  harness_versions: [{family, value, trials}], harbor_versions: [{value, trials}],
  efforts: [{value: str|null, trials}],                  // null = harness default
  datasets: [{value, snapshots: [{id, trials}]}],        // ids like "git:69671fbaac6d", "local:terminal-bench@2.0@2026-10-29T05:29:58+00:00"
  tasks: [{value, trials, digests: int, order: int|null}],   // order = position in config/tasks.txt, 0-based
  runs: [{run_key, profile, run_id, source, created_at, status, trials}],
  outcomes: [{value, trials}], date_range: {min, max}, task_order: [str] }   // task_order = this profile's tasks.txt
```

LedgerRun:

```
{ run_key, profile, run_id, source, mode, status, created_at, finished_at, budget_usd,
  trials: {<outcome>: n}, spend_usd, families: [str], models: ["provider:model"], dataset_snapshot,
  exclusion: {action, reason, created_at}|null }
```

TrialRow, exactly these keys:

```
id, trial_key, run_key, profile, run_id, source, arm, trial_name, attempt, family, implementation,
harness_version, harbor_version, provider, model, effort, effort_label, config_hash, config_label,
options, cautions, dataset, dataset_snapshot, task, task_digest_short, outcome, outcome_reason,
harbor_cause, exception_type, exception_head, reward, passed, n_input, n_cache, n_output, tokens_total,
cost_usd, cost_at_ingest_usd, reported_cost_usd, unpriced_models, primary_share, started_at, finished_at,
phases: {env_setup_sec, agent_setup_sec, agent_exec_sec, verifier_sec}, total_sec,
exclusion: {action, reason, created_at}|null, log_url: "/api/runs/<run_id>/log/<arm>"|null
```

Field notes:

- `cost_usd` is priced with today's `pricing.yaml`.
- `reported_cost_usd` is what the harness or OpenRouter billed. On
  OpenRouter it may be the truer figure.
- `log_url` is set only for this profile's own bench runs.

The single trial (`GET /api/ledger/trials/{id}`) adds:

- `usage: {model: {n_input, n_cache, n_output, reported_cost_usd, primary}}`
- `config: {effort, extras, family, implementation, model, options, provider, timeouts}`
- `exception_message` (full, redacted)
- `raw` (trimmed Harbor `result.json`) and `lock` (`lock.json`)
- `trial_dir` and `trial_dir_exists`
- bookkeeping: `task_digest`, `task_source_type`, `tokens_uncached`,
  `harbor_model`, `harness_version_pinned`, `n_concurrent`, `cap_marker`,
  `setup_marker`, `agent_started`, `rewards`, `trial_uid`, `ingest_via`,
  `ingested_at`, `updated_at`, `parser_version` and `classifier_version`

Show `raw` and `lock` only inside a collapsed `<details>` as preformatted JSON.

### Report

All four routes are **POST only to carry nested JSON. They change nothing.**

| Method and path | Body | Returns |
|---|---|---|
| `POST /api/report` | `{filters?: Filters, options?: Options}` | Report |
| `POST /api/report/coverage` | `{filters?, options?}` | `{provenance, task_order: [str], untouched_tasks: [str], comparisons: [{key, provider, model, group, baseline_group, headline, coverage: Coverage}]}` |
| `POST /api/report/pairs` | `{filters?, options?, comparison: key}` | `{comparison, pairs: [PairRow], excluded_trials: [{trial_key, side: "arm"\|"baseline", task, outcome, reason}]}`. 404 for an unknown key. |
| `POST /api/report/export` | `{filters?, options?, level: "trials"\|"pairs"\|"comparisons"}` | CSV file `bench-report-<8 hex>-<level>.csv` |
| `POST /api/report/snapshots` | `{name: 1-80 chars, filters?, options?}` | **201** `{id: "rs-...", name, created_at, fingerprint, trials_used}` |
| `GET /api/report/snapshots` | | `{snapshots: [{id, name, created_at, fingerprint, trials_used}]}`, newest first |
| `GET /api/report/snapshots/{id}` | | `{id, name, created_at, request: {filters, options}, response: Report}`. 404 if unknown. |

**Filters.** Every key is optional, and an empty list means no constraint.
An unknown key or a bad value answers 400. Each list holds at most 500 items.

```
{ date_from: "YYYY-MM-DD"|ISO, date_to: "YYYY-MM-DD"|ISO, profiles: [str], sources: ["bench"|"external"],
  models: ["provider:model"], families: [str], implementations: ["builtin"|import path],
  configs: [config_hash], exclude_configs: [config_hash], harness_versions: [str], harbor_versions: [str],
  efforts: [str|null], datasets: [str], snapshots: [snapshot id], tasks: [str],
  runs: [run_key], exclude_runs: [run_key], include_cautioned: bool (default false) }
```

**Options.** Every key is optional. The defaults come from the profile's
`report:` and `stats:` config. The response echoes every value in
`spec.options`, so build your controls from that.

```
{ group_by: "family"|"implementation"|"config",            // default family
  split_harness_versions: bool,                            // false
  baseline: {family: "claude-code", config: config_hash|null},
  pairing: {match_on: "digest"|"name", same_run_only: bool},   // digest, false
  policy: {crashes: "include"|"exclude", misconfigured_cap: "exclude"|"include", exclusions: bool},  // include, exclude, true
  pricing: "current"|"as_ingested"|"reported",             // current
  ci_level: 0.80..0.99,                                    // 0.9
  bootstrap: {resamples: 500..10000, seed: int},           // 2000, 20261028
  min_pairs: {estimate: int, verdict: int},                // 3, 8
  coverage_target: {ratio_halfwidth: 1.01..10, pass_delta_pp: number, prior_sd_log: number},  // 1.25, 5, 0.6
  include_pairs: bool }                                    // false; true adds pairs_table to each comparison
```

The pricing modes:

| Mode | Meaning |
|---|---|
| `"current"` | Today's `pricing.yaml`. |
| `"as_ingested"` | The prices recorded when each trial was ingested. |
| `"reported"` | The harness's or OpenRouter's own billed cost. |

**Report:**

```
{ spec: {filters (normalised), options (normalised)},
  provenance: {generated_at, fingerprint: "sha256:<16 hex>", trials_used, exclusions_applied, pricing, pricing_sha,
               parser_versions: [int], classifier_version, stats_version, harbor_versions: [str], includes_active_run: bool},
  totals: {trials, runs, profiles: [str], by_outcome: {}, excluded_manual: int, spend_usd},
  groups: [Group], comparisons: [Comparison],
  coverage: {task_order: [str], groups: [group key], cells: {<group key>: {<task>: {valid, passed, by_outcome: {}}}}},
  warnings: [str] }
```

**Group** (descriptive and unpaired; label it "unpaired; task mix may differ"):

```
{ key, label: "claude-code · openrouter:deepseek/deepseek-v4.1-flash", name, provider, model, family,
  implementation: str|null, config_hash: str|null, harness_version: str|null, is_baseline: bool,
  configs: [{config_hash, label, trials}],          // the per-config breakdown of what this group pools
  config_diff: {<option>: [values]},                // options that differ between pooled configs; {} = one config
  harness_versions: {<version>: n}, runs: int, tasks: int,
  trials: {total, by_outcome: {}, valid_tokens, valid_cost, valid_reward, included},
  pass_rate: {estimate, lo, hi, tasks, method: "wilson_tasks"}|null,        // fractions 0..1
  setup_failure_rate: {estimate, lo, hi, trials, method: "wilson_trials"},
  per_trial: {tokens_total: {mean, median, n}, tokens_uncached: {mean, median, n}, cost: {mean, median, n},
              minutes: {median, n}},
  reported_vs_computed_cost: {median_ratio: number|null, n} }
```

**Comparison:**

```
{ key, provider, model, group, baseline_group, arm_name, baseline_name,
  headline: str,      // a ready-made sentence; show it as the row's summary
  pairs: {n_tasks, same_run_fraction: number|null, arm_only_tasks: [str], baseline_only_tasks: [str],
          dropped: {no_arm_valid, no_baseline_valid, tokens_not_reported, unpriced, zero_value}},
  metrics: {tokens_total: Metric, tokens_uncached: Metric, cost: Metric},
  pass: Pass, coverage: Coverage, caveats: [str],
  pairs_table: [PairRow] }   // only when options.include_pairs
```

**Metric:**

```
{ status: "ok"|"low_n"|"insufficient"|"not_measured", n: int, need: int,   // need = pairs still missing for the next status
  geomean: {estimate, lo, hi, sd_log, gsd, method: "t_on_log_ratios"}|null,   // the headline multiple
  ratio_of_means: {estimate, lo: number|null, hi: number|null, method: "task_bootstrap"}|null,  // used for verdicts
  per_task: {min, q1, median, q3, max}|null, leave_one_out: [min, max]|null,
  sensitivity: {crashes_excluded, misconfigured_cap_included, same_run_only},   // each a geomean or null
  robust: bool|null,
  reference: { published: {value?|low?, high?, source}, band: [lo, hi], estimator: "ratio_of_means"|"geomean",
               applies: bool, position: "inside"|"above"|"below"|null,
               verdict: "within"|"overlaps"|"above"|"below"|"insufficient"|"not_applicable" }|null,
  dropped: {...same keys as pairs.dropped} }
```

- `tokens_uncached` never has a reference.
- `not_measured` means the harness reports no tokens. Strands does this today.

**Pass** (harness minus baseline, in percentage points, on paired tasks):

```
{ status, n, need, arm_rate: 0..1|null, baseline_rate: 0..1|null,
  delta_pp: {estimate, lo, hi, method: "newcombe_paired"|"task_bootstrap"|null},
  discordant: {arm_only, baseline_only}|null, mcnemar_p: number|null,
  mde_pp: number|null,          // the smallest difference this many tasks can detect
  reference: {published: {low: -5, high: 5, source}, band: [-5, 5], applies: bool, verdict}|null }
```

**Coverage** ("where are we, what to run next"):

```
{ metric: "cost"|"tokens_total"|"pass", pairs, need_for_estimate, need_for_verdict,
  halfwidth_factor_now: number|null,   // CI is estimate x/÷ this now
  target_factor: 1.25, sd_log_used, sd_source: "observed"|"prior",
  est_pairs_needed, est_more_pairs, pass_tasks_needed,
  next: [{task, side: "arm"|"baseline"|"both", reason: "complete pair"|"new task"}],
  est_cost_usd: number|null, per_trial: {arm: {mean_cost_usd, median_minutes, setup_failure_rate}, baseline: {...}},
  capped_at_sample: bool, note: str|null }
```

**PairRow:**

```
{ task, digest_short, run_key: str|null, same_run: bool,
  arm: {trials, valid, tokens_total, cost, pass, trial_keys: [str]}, baseline: {same},
  ratio_tokens: number|null, ratio_cost: number|null }
```

**What today's data looks like.** The ledger holds 16 trials from the two
dev pilots on OpenRouter. Pilot 1 (`dev/20261028-222401-pilot`) is excluded
by a seeded exclusion, because its git tasks were damaged by CRLF line
endings. The two `misconfigured_cap` trials are excluded by policy. So a
default report uses 9 trials:

- `totals.excluded_manual` is 7;
- all 7 comparisons are `insufficient` (0 paired tasks);
- every reference verdict is `not_applicable`, because the published figures
  apply only to `anthropic:claude-*`.

You must still render the populated states. Here is an illustrative Metric
with `status: "ok"` (the values are made up; the shape is exact). Use it in
demo mode:

```json
{ "status": "ok", "n": 12, "need": 0,
  "geomean": {"estimate": 2.41, "lo": 1.88, "hi": 3.09, "sd_log": 0.46, "gsd": 1.58, "method": "t_on_log_ratios"},
  "ratio_of_means": {"estimate": 2.62, "lo": 2.05, "hi": 3.30, "method": "task_bootstrap"},
  "per_task": {"min": 1.1, "q1": 1.9, "median": 2.4, "q3": 3.0, "max": 5.2}, "leave_one_out": [2.28, 2.60],
  "sensitivity": {"crashes_excluded": 2.47, "misconfigured_cap_included": null, "same_run_only": 2.35}, "robust": true,
  "reference": {"published": {"low": 2.0, "high": 3.0, "source": "S1-S5: changing harness moves cost 2x to 3x"},
                "band": [2.0, 3.0], "estimator": "ratio_of_means", "applies": true, "position": "inside", "verdict": "overlaps"},
  "dropped": {"no_arm_valid": 1, "no_baseline_valid": 0, "tokens_not_reported": 0, "unpriced": 0, "zero_value": 0} }
```

### Designs

**DesignSpecIn** is what an editor sends. It is strict: an unknown key, or a
string where a number belongs, answers 400.

```
{ tasks: [str],                 // 1-100, unique; seeded-sample order recommended
  attempts?: int (default 1),   // 1..designs.max_attempts (5)
  baseline_arm?: str|null,      // one of the arm names, or null
  run: { budget_usd: number > 0,                // required; at most designs.max_budget_usd (150 default, 10 dev)
         parallel_arms?: int 1-12, n_concurrent?: int 1-8,
         effort?: "low"|"medium"|"high"|"xhigh"|"max"|"omit",
         per_trial_cap_usd?: number|null, max_turns?: int|null,   // null = no cap
         timeout_multiplier?: number (0, 4] },
  arms: [{ name: str,           // [a-z][a-z0-9-]{0,31}, unique
           agent: str,          // an allowed harness (see "Allowed harnesses" below)
           provider: str, model: str,   // model without the provider prefix, e.g. "deepseek/deepseek-v4.1-flash"
           effort?: "inherit" (default)|"omit"|a level from the schema (any text when effort_levels is []),
           options?: {name: typed JSON value},   // numbers as numbers, booleans as true/false, json kinds as objects
           tasks?: [str]|null }] }               // subset of the design's tasks, in their order; null = all
```

- **Leave a run field out** to use the profile default. Sending `null` for
  `parallel_arms`, `n_concurrent`, `effort` or `timeout_multiplier` is a 400.
  For the two caps, `null` means "no cap".
- The server fills every omitted field and stores a **NormalizedSpec**. It
  is the same shape with every run field explicit, plus `dataset` and each
  arm's `family`. Arms always carry `effort`, `options` and `tasks`.
- **You may send a stored NormalizedSpec back as `spec` unchanged** (with
  `dataset` and `family`). The server converts it. So an editor can load a
  version, change fields and send it back.

| Method and path | Body | Returns |
|---|---|---|
| `GET /api/designs?filter=all\|draft\|approved\|queued\|ran\|archived` | | `{designs: [DesignSummary]}`. `all` leaves out archived designs. `queued` includes running. Any other filter is a 400. |
| `POST /api/designs` | `{title (1-80), purpose? (≤1000), rationale? (≤1500), change_note? (≤200), spec: DesignSpecIn}` | **201** VersionDetail (author `"user"`) |
| `POST /api/designs/check` | `{spec}` | CheckResult. **Writes nothing.** A half-filled spec answers 200 with `validation.status: "invalid"` and the reasons. Use it as the live check. |
| `POST /api/designs/from-config` | `{mode, arms: [name]\|null, title?}` | **201** VersionDetail. `arms: null` means the enabled arms. |
| `POST /api/designs/from-run/{run_id}` | `{title?}` | **201** VersionDetail ("Run again"). The title defaults to `"Run again: <run_id>"`. 404 for an unknown run. |
| `GET /api/designs/{id}` | | `{meta: DesignMeta, current: VersionDetail}` (current = latest version) |
| `GET /api/designs/{id}/versions/{n}` | | VersionDetail. 404 if missing. |
| `POST /api/designs/{id}/versions` | `{base_version: int, title (required), purpose?, rationale?, change_note?, spec}` | **201** VersionDetail. See the 409s below. |
| `POST /api/designs/{id}/archive` | | `{meta}`. 409 while any version is in the queue: `"... Remove them from the queue first."` |
| `POST /api/designs/{id}/unarchive` | | `{meta}` |

The 409s from `POST /api/designs/{id}/versions`:

- `"Version 2 of d-... was saved since you read version 1. Load the latest version and apply your change again."`
  Offer "Reload latest", and keep the user's edits so they can reapply them.
- `"Nothing changed ..."`

**DesignSummary** (real example):

```json
{ "id": "d-block-1-on-deepseek-b7a8", "title": "Block 1 on DeepSeek", "created_at": "...+00:00", "updated_at": "...",
  "latest_version": 2, "archived": false, "author": "user", "conversation_id": null,
  "status": "draft", "approved_at": null, "runs": [], "spec_hash": "sha256:1379c169f78d...",
  "arms": 2, "trials": 3, "problems": 0, "warnings": 2, "expected_usd": 0.107933, "p90_usd": 0.230486,
  "checked_at": "...", "config_fingerprint": "sha256:...", "ledger_fingerprint": "sha256:...", "check_stale": false }
```

Notes on DesignSummary:

- `author` is `"user"` or `"assistant"`.
- `conversation_id` is set for assistant drafts.
- `status` is one of `draft`, `approved`, `queued`, `running`, `ran`,
  `archived` or `unreadable`.
  - It is derived and never stored.
  - `approved` means admitted to the queue at least once.
- `expected_usd` is the p50 estimate.
- `problems`, `warnings`, `expected_usd` and `p90_usd` come from the cached
  check and may be `null`.
- An unreadable design lists as
  `{id, title: null, status: "unreadable", archived: false, updated_at: null, error, check_stale: true}`.

**DesignMeta:**

```
{ schema: 1, id, title, created_at, updated_at, latest_version, archived, status,
  versions: {"1": {author, conversation_id, created_at, spec_hash, change_note, approved_at, runs: [run_id], status}, "2": {...}} }
```

Build the version selector from `versions`.

**VersionDetail:**

```
{ design_id, version, parent_version: int|null, author, conversation_id, created_at, title, purpose, rationale,
  change_note, spec: NormalizedSpec, spec_hash, status, approved_at, archived, latest_version,
  runs: [{run_id, status, spend_usd, budget_usd}],
  check: CheckResult,
  diff_from_parent: [{path: "run.budget_usd" | "arms[1].options.x" | "title" ..., from, to}] }
```

`check` is always fresh: a stale cached check is redone on read.

NormalizedSpec (real example):

```json
{ "dataset": "terminal-bench@2.0", "tasks": ["adaptive-rejection-sampler", "protein-assembly"], "attempts": 1,
  "baseline_arm": "cc",
  "run": {"budget_usd": 4.0, "parallel_arms": 5, "n_concurrent": 1, "effort": "medium", "per_trial_cap_usd": 1.0,
          "max_turns": 150, "timeout_multiplier": 1.0},
  "arms": [{"name": "cc", "agent": "claude-code", "family": "claude-code", "provider": "openrouter",
            "model": "deepseek/deepseek-v4.1-flash", "effort": "inherit", "options": {}, "tasks": null},
           {"name": "pi", "agent": "pi", "family": "pi", "provider": "openrouter",
            "model": "deepseek/deepseek-v4.1-flash", "effort": "high", "options": {}, "tasks": ["adaptive-rejection-sampler"]}] }
```

**CheckResult:**

```
{ validation: {status: "valid"|"invalid", problems: [str], warnings: [str]},
  plan: PlanResult|null, estimate: Estimate|null, estimate_note: str|null,
  checked_at, stale: false, config_fingerprint, ledger_fingerprint: str|null }
```

- **PlanResult** is the `GET /api/plan` shape described above, without
  `mode`. It has `source: "design"`, `label: "design"` and `spec_hash`.
- Problems block queueing.
- A per-arm plan warning also appears in `validation.warnings`, prefixed with
  the arm name.
- Without keys, each arm on a provider with no key gets a problem such as
  `"cc: ANTHROPIC_API_KEY is not set"`. The plan and estimate are still
  returned.

**Estimate** (real shape):

```
{ spec_hash, ledger_fingerprint, seed, replicates: 2000,
  arms: [{ arm, cell: {family, provider, model, effort, config_hash, options_fingerprint}, trials,
           basis: "measured"|"measured_any_options"|"borrowed"|"prior", basis_detail: str,
           confidence: "high"|"medium"|"low",
           evidence: {trials, tasks, design_tasks_observed, runs: [run_key]},
           per_trial: {mean, p50, p90}, total: {p10, p50, p90}, cap_usd: number|null,
           minutes_per_trial: {p50, p90}, arm_minutes: {p50, p90}, p_infra, unpriced_models: [str], notes: [str] }],
  run: { total: {p10, p50, p90}, budget_usd, p_within_budget: 0..1, suggested_budget_usd,
         worst_case_capped_usd, uncapped_arms: [name], inflight_allowance_usd,
         wall_minutes: {p50, p90}, peak_containers },
  pairing: [{ arm, baseline_cell: {family, provider, model, arm}, metric, pairs_now, pairs_after,
              multiple_now: number|null, ci_factor_now: number|null, ci_factor_after: number|null,
              sd_log, sd_source: "observed"|"prior" }],
  warnings: [str], assumptions: [str] }
```

**Allowed harnesses.** No endpoint lists the harnesses a design may use.
Derive them from `config.arms`: take each distinct `{agent, family, kind}`.
This matches the backend's list while `designs.extra_harnesses` and
`designs.import_paths` are empty, which they are in both shipped profiles.
The backend rejects anything else with a clear problem message.

### Queue

The queue is the only way a design becomes a billed run.

**QueueView** (`GET /api/queue`; every queue mutation returns it too):

```
{ state: "paused"|"running", pause_reason: str, on_failure: "pause"|"skip",
  global_cap_usd: number|null, cap_since: iso, cap_spent_usd: number|null, cap_remaining_usd: number|null,
  committed_usd: number,        // budgets plus in-flight allowances of waiting items
  active: QueueItem|null, waiting: [QueueItem], held: [QueueItem], history: [QueueItem] }  // history: last 50, newest first
```

QueueItem (real example). `waiting` and `held` items carry `position`, their
0-based place among the open items:

```json
{ "id": "q-0001", "design_id": "d-block-1-on-deepseek-b7a8", "version": 2,
  "spec_hash": "sha256:1379c169f78d...", "title": "Block 1 on DeepSeek",
  "budget_usd": 6.0, "inflight_allowance_usd": 0.355728, "expected_usd": 0.107933, "p90_usd": 0.230486,
  "status": "waiting", "message": "", "added_at": "...", "started_at": null, "finished_at": null,
  "run_id": null, "spend_usd": 0.0, "position": 0 }
```

Item status is one of:

| Status | Meaning |
|---|---|
| `waiting` | In line to run. |
| `held` | Could not start. `message` says why; Retry it after fixing the cause. |
| `starting` | Transient, while the queue checks the item again. |
| `running` | Its run is in progress. |
| `done` | Finished. |
| `budget_stopped` | Stopped at its budget. |
| `failed` | The run failed. |
| `cancelled` | Cancelled. |
| `interrupted` | The run was interrupted. |

Pause reasons are plain sentences. Show them as they are. Examples:

- `"Not started yet. Press Start queue."`
- `"Paused after restart. Press Start queue."`
- `"Paused by you"`
- `"Docker Desktop is not running"`
- `"Queue finished: nothing is waiting. Add a design and press Start queue."`
- `"Paused: <title> vN failed (<run id>)."`
- `"Global cap: ..."`

| Method and path | Body | Returns |
|---|---|---|
| `POST /api/queue/items` (**billed path; do not call**) | `{design_id, version, spec_hash, position: "end"\|"front", start: bool}` | **201** `{item: QueueItem, queue: QueueView}`. See the errors below. |
| `DELETE /api/queue/items/{id}` | | QueueView. Only waiting or held items, otherwise 409. 404 for an unknown id. |
| `POST /api/queue/items/{id}/move` | `{to_index: int}` | QueueView. `to_index` counts among the waiting and held items, as `position` shows. |
| `POST /api/queue/items/{id}/retry` (**billed path; do not call**) | | QueueView. Held back to waiting after checking again. 409 if not held. |
| `POST /api/queue/start` (**billed; do not call**) | | QueueView. 409 when Docker is down, or nothing is waiting (`"Nothing is waiting in the queue. Add a design first."`). |
| `POST /api/queue/pause` | | QueueView. The current run carries on; nothing new starts. |
| `PUT /api/queue/settings` | `{global_cap_usd?: number > 0 \| null, reset_cap?: bool, on_failure?: "pause"\|"skip"}` | QueueView. `reset_cap` restarts the cap's spend count from now. |

Errors from `POST /api/queue/items`:

- 409 when the hash no longer matches the version:
  `"Version 1 of d-... has spec hash 44391139a752, not the 000000000000 that was approved. Reload it and check it again."`
- 409 when that version is already in the queue:
  `"d-... v2 is already in the queue (q-0001, waiting)"`
- 400 when the version has problems: `"This version cannot be queued: ..."`
- 409 with `start: true` while Docker is down.
- `"Added as q-0001, but the queue did not start: ..."` when the item was
  added but the start failed.

Also:

- "Start now" in the UI is `position: "front", start: true`.
- To cancel the running item, use the existing `POST /api/runs/{run_id}/cancel`
  (`queue.active.run_id`). The queue then pauses.

### Assistant

| Method and path | Body | Returns |
|---|---|---|
| `GET /api/assistant/status` | | `{enabled, available, reason: str\|null, model: "claude-opus-5-5", effort_default: "medium", fallbacks: "default", key_set: bool, today_usd, daily_cap_usd, spend: {today_usd, daily_cap_usd, per_turn_cap_usd}, active_turn: {conversation_id, turn, started_at}\|null}` |
| `GET /api/assistant/conversations` | | `{conversations: [ConversationMeta]}`, newest first. Archived ones are included. |
| `POST /api/assistant/conversations` | `{title?, effort?: "low"\|"medium"\|"high"\|"xhigh"\|"max", design?: {id, version}}` | **201** ConversationMeta. No model call. 400 for a bad effort. 409 if the assistant is off or could not load. |
| `GET /api/assistant/conversations/{cid}?after=<seq>` | | `{conversation: ConversationMeta, events: [Event], next_seq: int, active_turn}`. Events have `seq > after`. |
| `GET /api/assistant/conversations/{cid}/stream?after=<seq>` | | SSE (below). 404 for an unknown cid. |
| `POST /api/assistant/conversations/{cid}/messages` (**calls the model; do not call**) | `{text: 1..8000 chars, attach_design?: {id, version}\|null}` | **202** `{turn, accepted_seq}`. See the errors below. |
| `POST /api/assistant/conversations/{cid}/stop` | | `{ok: true}`. A running tool finishes first. |
| `POST /api/assistant/conversations/{cid}/retry` (**calls the model; do not call**) | | **202** `{turn, accepted_seq}`. Only right after a turn that ended in an error, otherwise 409. |
| `POST /api/assistant/conversations/{cid}/continue` | | **201** ConversationMeta of a new conversation. Its first event is a `user_message` with `recap: true`. **No model call.** |
| `POST /api/assistant/conversations/{cid}/archive` | | `{ok: true}`. 409 while its turn runs. |
| `GET /api/assistant/usage?days=1..366` | | `{days: [{date: "YYYY-MM-DD", usd, requests}], total_usd}` |

The 409s from `POST /api/assistant/conversations/{cid}/messages` (nothing is
stored or sent):

- the assistant is off;
- no key: `"Set ANTHROPIC_API_KEY on Setup to use the assistant."`;
- the conversation is archived or locked;
- another turn is active anywhere: `"The assistant is already answering a message. Wait for it, or press Stop."`

The 400s:

- empty or too long;
- key-like text: `"This looks like an API key. Set keys on Setup; never paste them into chat."`
  Keep the text in the box when this happens;
- daily cap: `"Assistant daily cap $5.00 reached: ..."`

**ConversationMeta:**

```
{ id, title, created_at, updated_at, model, effort, fallbacks, prompt_hash, tools_hash,
  status: "idle"|"busy"|"locked"|"archived", locked_reason: str|null,
  turns, requests, spend_usd, last_context_tokens,
  next_seq,                         // a stored hint only; use the top-level next_seq from GET /conversations/{cid}
  designs: [{id, version}],         // designs attached or drafted in this conversation
  continued_from: cid|null,
  last_turn: {turn, status, error_kind: str|null, retryable: bool}|null,
  active: bool }
```

`status: "locked"` means the conversation can take no more messages. That
happens when it grew too long, or the model, prompt or tools changed.
`locked_reason` says which. Offer **Continue in a new conversation**.

**Every event has** `{seq: int, type, at: iso, turn: int}` plus the fields
below. Durable events are stored and come back on reload. Ephemeral events
are live only; they are dropped when the next turn starts and are lost on a
bench restart.

| type | durable | fields |
|---|---|---|
| `user_message` | yes | `text`, `attach_design: {id, version}\|null`, `recap?: true` |
| `turn_started` | yes | `retry?: true` |
| `status` | no | `text`, for example "Thinking…", "Writing…", "Preparing estimate_cost…" or "Calling get_coverage…" |
| `thinking_delta` | no | `block_key`, `text` (append) |
| `text_delta` | no | `block_key`, `text` (append) |
| `thinking_summary` | yes | `block_key`, `text`. Replaces the streamed thinking with the same `block_key`. |
| `assistant_text` | yes | `block_key`, `text` (Markdown). Replaces the streamed text with the same `block_key`. |
| `tool_call` | yes | `tool_use_id`, `name`, `input` (object) |
| `tool_result` | yes | `tool_use_id`, `name`, `ok`, `summary` (a human line, for example "Checked coverage · openrouter:z-ai/glm-5.3 · 3 gaps"), `draft: {design_id, version, spec_hash, title, expected_usd, p90_usd, problems: int}\|null` |
| `usage` | yes | `request_usd`, `turn_usd`, `conversation_usd`, `today_usd`, `served_model`, `fallback_served: bool`, `partial: bool` |
| `notice` | yes | `level: "info"\|"warn"`, `code`, `message`, and for `fallback` also `from_model` and `to_model`; for `refusal` also `category` |
| `turn_error` | yes | `kind`, `message`, `retryable: bool`, `retry_after_s: number\|null` |
| `turn_finished` | yes | `status: "done"\|"stopped"\|"refused"\|"truncated"\|"cap_reached"\|"max_steps"\|"error"`, `cost_usd`, `requests` |

- The notice codes are `fallback`, `retry`, `refusal`, `truncated`, `cap`,
  `locked`, `stopped`, `max_steps` and `empty`.
- The `turn_error` kinds are `auth`, `permission`, `model_not_found`,
  `bad_request`, `rate_limited`, `server`, `timeout`, `connection`,
  `invalid_json` and `internal`.

How the stream behaves:

- **`block_key`** is `"<turn>:<step>:<index>"`.
- **Durable blocks arrive after their deltas, and not in content order.**
  Within one step you see this order:
  1. the thinking and text deltas;
  2. `usage`;
  3. `status` "Calling …";
  4. the durable `thinking_summary`, `assistant_text` and `tool_call`;
  5. `tool_result`.

  So place blocks by `block_key`, and tool cards by `tool_use_id`, never by
  arrival order.
- **Streamed blocks that never get a durable event were discarded by the
  server.** This happens with a refused reply, a reply cut off with a tool
  call, or what a declined model wrote before a server-side fallback. When
  `turn_finished` arrives, remove any streamed block of that turn that has no
  durable twin, or show it struck through as "not kept".
- **The model that answered** is `usage.served_model`. When
  `fallback_served` is true, a `notice` with `code: "fallback"` says
  `"<from_model> declined this request (<category>); <to_model> answered instead."`

**SSE wire format** (real capture). The route is a plain GET, so the session
cookie is enough and no `X-Bench` header is needed:

```
content-type: text/event-stream; charset=utf-8

event: batch
data: [{"seq": 1, "type": "user_message", "at": "...", "turn": 0, "text": "...", "attach_design": null, "recap": true}]
id: 1

```

- A `batch` event comes at most every 100 ms when there are new events.
  - `data` is a JSON array of events.
  - `id` is the last seq in the batch.
- Idle time carries keepalive comments.
- After 300 s the server sends a `: bye` comment with `retry: 1000` and
  closes. `EventSource` reconnects by itself 1 s later and sends
  `Last-Event-ID`. The server prefers that header over `?after`, so you get
  no duplicates.
- The default `onmessage` never fires, because every event is named `batch`.
  Listen with `es.addEventListener('batch', e => ...)`.

---

## The UI to build

Follow the existing look:

- the dark instrument panel, panel codes and LEDs;
- monospaced numbers, `.tag ok|warn|bad|run`, `.chip` and `.card`;
- `confirmAction({code, title, body, ok, tone})`;
- the drawer.

Use plain, short labels and `–` for missing values. Never re-render a form,
filter rail, editor or composer while it holds unsent input.

### Navigation and header

- **Tabs:** LIVE · LAUNCH · DESIGNS · QUEUE · REPORT · ANALYSIS.
  - Add `<section id="view-designs">`, `<section id="view-queue">` and
    `<section id="view-report">` with matching nav buttons.
  - Extend `VIEWS`.
  - Keep the hash routing (`#report` and so on).
- **Header queue chip** next to `#status-pill`, from `state.queue`:
  - "Q 2" for two waiting, with " · 1 held" in warn when there are held items;
  - a `run` tag while `state` is `running`;
  - "paused" otherwise.
  - Clicking it opens QUEUE.
- **Assistant button** `#open-assistant`: add an LED.
  - grey: `assistant.available` is false (show `reason` in a tooltip);
  - green: ready;
  - amber, pulsing: `active_turn` is set.
- **Ledger LED** in `#systems`:
  - green: available;
  - amber: `rescan_running`;
  - red: `error` set, with the error in the tooltip.
- **Auto-switching to LIVE.** Today the page jumps to LIVE when a new
  `active_run` appears. The queue now starts runs without a click. Switch
  only when none of these is true:
  - a design editor has unsaved edits;
  - the report filter rail is being edited;
  - the assistant composer holds text.

  Otherwise show a toast "Run <id> started from the queue · Open LIVE".

### REPORT tab: pooled statistics across every run

Data:

- `GET /api/ledger/facets` and `GET /api/ledger/status` when the tab opens.
- `POST /api/report` with the current filters and options.
- Debounce filter changes by 400 ms.
- Discard stale responses with a sequence counter, as `loadAnalysis` does.
- Keep the previous result visible but dimmed while loading.

The tab is built from these parts:

1. **Status strip.** One line, for example
   "9 of 16 trials used · 1 run · dev · last ingest 05:47 · fingerprint ce89ff13".
   - Below it, a stacked outcome bar (`totals.by_outcome`) with glyphs and counts.
   - A tag "includes a running run" when `provenance.includes_active_run` is set.
   - A tag "7 excluded by hand" from `totals.excluded_manual`.
   - **Buttons:**
     - Rescan: `POST /api/ledger/rescan`. Disabled while a rescan runs. The
       toast shows the inserted, upgraded and incomplete counts.
     - Save snapshot: a name dialog, then `POST /api/report/snapshots`.
     - Export: a menu of trials, pairs or comparisons as CSV, plus the ledger
       as CSV or JSONL.
2. **Filter rail.** A left column at 1100 px and wider; a collapsible panel
   above the content when narrower.
   - Fill it from facets, with trial counts:
     - date range with 7d, 30d and All presets;
     - profiles;
     - models grouped by provider;
     - families;
     - implementations, showing the import path for custom ones;
     - configurations: label, short hash, trial count and first/last dates;
     - harness versions;
     - dataset snapshots;
     - runs: include or exclude;
     - tasks, with "Block 1" and "Block 2" shortcuts. Block 1 is
       `task_order[0..9]`, Block 2 is `task_order[10..19]`.
   - Configurations with `cautions` get a hazard tag. They stay out unless
     "Include cautioned configs" is ticked.
   - **Advanced:**
     - Group by: Harness / Implementation / Exact config.
     - Split by harness version.
     - Baseline family, optionally pinned to one config.
     - Pair on "task content (default)" or "task name".
     - Same run only.
     - "Count harness crashes as failures" (`policy.crashes`).
     - "Include trials stopped by a mispriced cap" (`policy.misconfigured_cap`).
     - "Apply manual exclusions" (`policy.exclusions`).
     - Pricing: current table / as ingested / harness-reported.
     - CI level (0.80–0.99).
   - Add a "Reset to defaults" control.
   - Keep the filter and option state in `localStorage`, wrapped in try/catch,
     under a profile-specific key. Do not put it in the hash; the hash is the
     view.
3. **Headline: multiples vs baseline.** One card per `provider:model`.
   - **Card header:** the model, the baseline label, the paired-task count
     and a caveat count.
   - **Two small forest plots, Tokens and Cost.** Use inline SVG.
     - The x-axis is log scale from 0.25× to 8×, with a line at 1×.
     - Shade the published `reference.band`. Its `source` goes in a tooltip.
     - Per harness row:
       - a filled dot with a whisker for `geomean` (estimate, lo, hi);
       - a hollow diamond with a thin whisker for `ratio_of_means`.
     - `low_n` rows are greyed and tagged "low n".
     - `insufficient` rows have no mark and read "needs N more paired tasks"
       (from `need`).
     - `not_measured` rows read "tokens not reported by this harness".
   - **Accuracy strip:** Δ pass in points on −50…+50. Shade the band from
     `pass.reference.band` and add "can detect ±N pts" from `mde_pp`.
   - **A table under the plots,** one row per comparison:
     - pairs n;
     - Tokens × as "2.41× [1.88–3.09]", with the ratio of means in small text;
     - Cost × in the same layout;
     - published value or band, with the source in a tooltip;
     - verdict chip:

       | `verdict` | Chip text |
       |---|---|
       | `within` | "within published range" |
       | `overlaps` | "overlaps" |
       | `above` | "above" |
       | `below` | "below" |
       | `insufficient` | "too few pairs" |
       | `not_applicable` | "n/a for this model", plus where the estimate falls (`position`) |

     - pass harness vs baseline with Δ pp [lo, hi];
     - setup failure rate from the arm's Group;
     - a caveat icon listing `caveats`.

     Use `headline` as each row's sentence. Every number must appear as text
     as well as in a plot.
   - **Fixed explainer:** "Multiple = baseline ÷ harness on the same tasks and
     model; 2.0× means the harness used half. Intervals are 90%. Pass rates on
     fewer than about 100 tasks cannot confirm a 5-point claim."
   - Wording never says "confirmed".
4. **Group summary,** labelled "unpaired; task mix may differ". Per Group:
   - outcome mini-bar;
   - pass rate with its Wilson range;
   - mean and median tokens (total and uncached);
   - mean cost next to the harness-reported cost ratio (`reported_vs_computed_cost`);
   - median minutes;
   - harness versions;
   - **"N configs pooled"** with a tooltip listing `configs` (label, hash,
     trials) and `config_diff`.

   This is the per-config breakdown. With Group by = Exact config, each
   config becomes its own group.
5. **Coverage: what exists, what to run next.** Per model, a matrix from
   `report.coverage`.
   - **Rows** are tasks in `task_order`, with Block 1 and Block 2 dividers,
     then any other tasks seen.
   - **Columns** are groups, baseline first.
   - **A cell** shows the count of valid trials, shaded by `passed / valid`.
     Corner glyphs mark crash, setup, infra and incomplete from `by_outcome`.
     A dashed outline marks the missing side of a pair.
   - **Under each comparison,** for example: "Cost CI now ×/÷1.62. A ×/÷1.25
     target needs about 22 paired tasks (14 more), about $3.10", built from
     `coverage`. Add `coverage.note` when set, and "±5 pts on pass rate needs
     about N tasks" from `pass_tasks_needed`.
   - **"Copy next-run list"** copies `coverage.next` as text.
6. **Drill-down drawer.** Click a comparison row to load
   `POST /api/report/pairs`.
   - A pairs table: task, short digest, same-run tick, trials and valid per
     side, tokens and cost per side, the ratios, pass per side.
   - A strip of per-task log ratios.
   - Sensitivity: primary, crashes excluded, mispriced cap included,
     same-run only, the leave-one-out range, and "robust yes/no".
   - `excluded_trials`, with outcome and reason.
   - "Exclude trial…" and "Exclude run…" open a dialog with a required reason,
     then `POST /api/ledger/exclusions`. Excluded rows show a struck-through
     badge with the reason. "Include again" posts `action: "include"`.
7. **Sub-tabs inside REPORT:**
   - **Ledger:** a paged trial table (`POST /api/ledger/trials`, 100 per
     page, with the rail's filters).
     - Outcome badges.
     - Clicking a row opens `GET /api/ledger/trials/{id}`: outcome reason,
       exception message, phases, per-model usage, computed vs reported cost,
       options, a copyable `trial_dir`, and "View log" when `log_url` is set.
   - **Runs:** `GET /api/ledger/runs` across profiles: status, outcome counts,
     spend, and Exclude/Include with a reason.
   - **Exclusions:** `GET /api/ledger/exclusions`, plus `status.seeded_exclusions`.
     The seeded one ("Pilot 1 ran on git tasks damaged by CRLF line endings")
     shows with an "Include again" button.
   - **Snapshots:** name, date, fingerprint and trials used. Opening one
     renders its stored `response` read-only, with "Compare to now": per
     comparison, the change in estimate and CI width.
8. **Empty and error states:**
   - No trials: "Nothing in the ledger yet. Rescan, or start a run."
   - No baseline trials on a model: "No usable Claude Code trials on this
     model yet. Multiples need the baseline on the same tasks."
   - 503 or `ledger.available: false`: a red banner with the error. The rest
     of the dashboard keeps working.
9. **Refresh:**
   - Re-request when `state.ledger.last_ingest_at` changes. Debounce this to
     10 s while a run is active.
   - From ANALYSIS, an **"Open in Report"** button presets
     `filters.runs = ["<config.profile>/<run id>"]`.

### DESIGNS tab: list, view, validate, edit, enqueue

Use three columns on wide screens and stack them on narrow ones. Poll
`GET /api/designs?filter=...` when the tab opens and every 10 s while it is
open.

1. **List (left).**
   - Filter chips: All, Drafts, Approved, Queued, Ran, Archived.
   - Each card shows:
     - the title, id, latest version and an author badge, "AI" or "You";
     - a status tag;
     - "$0.11 p50 · $0.23 p90", with a "stale" tag when `check_stale`;
     - the problem count in bad-text and the warning count in warn-text;
     - the last run id.
   - Unreadable designs show their `error`.
   - **Buttons:**
     - "New design": an empty editor prefilled with the profile's run
       defaults as placeholders.
     - "From config arms…": mode plus arm checkboxes, then
       `POST /api/designs/from-config`.
     - "From past run…": pick from `state.runs`, then
       `POST /api/designs/from-run/{id}`.
2. **Viewer and editor (centre).**
   - **Header:**
     - title and purpose;
     - a version selector from `meta.versions`, showing author, change note
       and created time;
     - the status tag;
     - the spec hash, short.
   - **For an assistant-authored version,** show a banner "Drafted by the
     assistant" with a link that opens its conversation (`conversation_id`)
     in the Assistant panel, plus the rationale.
   - **"Changes from v<parent>"** renders `diff_from_parent` as `path: from → to`.
   - **Tasks:**
     - The seeded-sample strip, positions 1–20 from `config.tasks`, with a
       Block 1/2 divider. Click to toggle a task.
     - A text box for other task names.
     - Count and attempts (1–5).
     - The server warns when tasks depart from seeded order. Show that warning.
   - **Run settings:**
     - budget (required), parallel arms, trials per arm (`n_concurrent`),
       effort, per-trial cap (empty means none, sent as `null`), max turns
       (empty means none) and timeout multiplier;
     - placeholders show `config.run`;
     - leave a field out of the body to use the profile default;
     - show "Peak N containers" from `check.plan.peak_containers`;
     - a "Suggested budget: $X" link fills the budget from
       `check.estimate.run.suggested_budget_usd`.
   - **Arms table:**
     - name;
     - harness: a select of the allowed harnesses (see above), labelled by
       family, with import paths tagged "custom";
     - provider: from `config.providers`, with disabled ones marked "(off)";
     - model: text, with a datalist of the models in `config.arms`;
     - effort: "inherit", "omit", then the schema's levels. Use a text input
       when `effort_levels` is `[]`, and only inherit or omit when it is
       `null`;
     - option chips, plus "Edit options", which reuses the LAUNCH arm-option
       form from `GET /api/schemas/{agent}` and sends typed values;
     - tasks: "all" or "3 of 10", with a subset editor that sends `null` for all;
     - a baseline radio, sent as `baseline_arm`;
     - remove;
     - an "Add arm" button.
   - **Check panel.**
     - It is live: debounce 800 ms through `POST /api/designs/check`.
     - A saved, unedited version shows its `check` from VersionDetail.
     - It never re-renders the editor itself.
     - Show problems first, in bad-text; they block queueing. Then warnings.
     - **Estimate block:**
       - totals p10/p50/p90;
       - "N% chance within $budget" from `p_within_budget`;
       - the worst case under caps;
       - uncapped arms, named, with the note "run budget still applies after
         every trial";
       - the in-flight allowance;
       - wall time p50/p90.
     - **A per-arm table:** trials, p50/p90 per trial, total, minutes, and a
       basis tag:
       - "measured n=6 · high";
       - "any options · medium";
       - "borrowed · low";
       - "prior · low".

       Put `basis_detail` in a tooltip.
     - **Pairing rows,** for example: "pi vs claude-code: 0 → 1 paired tasks;
       interval ×/÷– → ×/÷–", from `estimate.pairing`.
     - `estimate_note` when there is no estimate.
     - Harbor commands in a `<details>`, as `pre.cmd` blocks.
     - `estimate.assumptions` in a collapsed list.
   - **Buttons:**
     - **Save as new version:** `POST /api/designs/{id}/versions` with
       `base_version` set to the version you loaded. A new design uses
       `POST /api/designs`.
       - On the stale-version 409, offer "Reload latest" and keep the user's
         edits.
       - A title is required on every save.
     - **Duplicate:** a new design from this spec.
     - **Archive / Unarchive.**
     - **Ask assistant:** opens the panel, creating a conversation with
       `design: {id, version}`, or attaching it to the next message.
     - **Add to queue** and **Start now:** the armed primary buttons.
       - Enable them only when the shown version is saved, unedited, valid,
         and not already queued or running.
       - Both open `confirmAction({code: 'ARM', tone: 'armed', ...})`. The
         body shows:
         - title, version and the first 12 hex of `spec_hash`;
         - arms × trials;
         - budget;
         - p50/p90;
         - worst case under caps;
         - in-flight allowance;
         - the global cap remaining (`state.queue.cap_remaining_usd`);
         - "This bills <the providers of its arms>."
       - Start now also says "Starts immediately (front of the queue)."
       - Both send `{design_id, version, spec_hash}` from the version shown.
         Add to queue sends `position: "end", start: false`; Start now sends
         `position: "front", start: true`.
       - **Do not call either against a server while testing.**
3. **On narrow screens,** the queue column moves to the QUEUE tab. On wide
   screens, show a compact queue summary here that links to QUEUE.

### QUEUE tab: order, status, pause/resume, remove

Poll `GET /api/queue` every 3 s while QUEUE or DESIGNS is open.

- **State pill:** "RUNNING", or "PAUSED: <pause_reason>".
- **Start queue / Pause buttons.**
  - Start queue is billed. It confirms with code ARM: "Run <k> designs in
    order? Budgets $S plus in-flight allowance $A (`committed_usd`). Global
    cap $C ($R left). The queue pauses if a run fails or is cancelled."
  - Pause needs no confirm.
- **Global cap meter:** "$3.10 spent since 28 Oct of $20.00 · $7.50 committed".
  - **Edit cap:** empty means no cap. Send `PUT /api/queue/settings`
    `{global_cap_usd: number|null}`.
  - **Reset count** (`reset_cap: true`, after a confirm).
  - **On failure** select: pause / skip.
- **Active item card:**
  - the design title and version, linked to DESIGNS;
  - the run id;
  - spend of budget, live from the matching run in `state.runs`;
  - a trials progress bar from that run's arms;
  - "Open in LIVE";
  - **Cancel:** confirm, then `POST /api/runs/{run_id}/cancel`, then show
    that the queue paused.
- **Waiting and held lists,** in `position` order. Each row shows:
  - position;
  - title vN;
  - budget;
  - p50/p90;
  - status tag;
  - up/down arrows (`POST .../move` with `to_index`);
  - Remove (confirm, then `DELETE`).
- **Held items** show `message` in warn-text, plus **Retry**.
  - Retry can start a run at once when the queue is running, so it confirms
    with code ARM.
  - Do not call it while testing.
- **History:** the last 50 items with status, spend of budget, finish time and
  a link to ANALYSIS for `run_id`.
- **Empty state:** "Nothing in the queue. Add a design from DESIGNS."

### LAUNCH, LIVE, ANALYSIS and Setup: small changes

- **LAUNCH:**
  - Show plan `warnings`, "Peak N containers", each arm's family and kind,
    and `reserve_note`.
  - When `state.queue.state === "running"`, the server refuses
    `POST /api/runs`. Replace "Start run" with "Add to queue":
    1. `POST /api/designs/from-config {mode, arms}`;
    2. the ARM confirm for the resulting version;
    3. `POST /api/queue/items`.
  - **Remove the launch-code lock** (decided by the user, 2026-10-29). Delete
    the `launch_code_set` check, the "Locked: ..." messages, the
    `#launch-code` field in index.html and the `launch_code` in the request
    body. LAUNCH gets the same two-step ARM confirm the queue uses: the
    first press arms it, and the armed state shows the plan's `budget_usd`,
    `worst_case_capped_usd` and `uncapped_arms` (by name, as a warning) before
    the second press sends `POST /api/runs {mode, arms}`. One gate for every
    billed action, direct or queued.
- **LIVE:**
  - For a run with `design_ref`, show the design title and version (from the
    queue's active item, or `GET /api/designs/{id}`), an author badge, and
    "Queue item q-0003".
  - Per arm:
    - "N running" from `trials_running`;
    - "$X held back" from `reserve_usd`;
    - a warn tag "containers may remain" when `cleanup_pending`;
    - "N killed (spend not counted)" when `trials_killed`.
- **ANALYSIS:**
  - Add "Open in Report", "Run again as design" (`POST /api/designs/from-run/{id}`,
    then open DESIGNS on it) and "Ask the assistant" (opens the panel with a
    prefilled question about this run).
  - Show the new `excluded` and `verdict_note` fields.
- **History table:** add a source column (LAUNCH, or a "design" tag linking
  to the design).
- **Setup / Configure:** the Assistant readiness row, the assistant key row,
  the Family field and kind tag on arm forms, the orphaned-overrides banner,
  and the stale-save 409 handling, all described above.

### Assistant panel (`#assistant`, already in `index.html`)

The current panel talks to `/api/assistant` and `/api/assistant/chat`. Those
routes do not exist. Replace that code. Conversations now live on the server,
so drop the `sessionStorage` chat history.

- **Opening it:**
  - `#open-assistant` toggles a docked right panel, about 420 px, that
    narrows the main area and stays open across views.
  - Load `GET /api/assistant/status` and `GET /api/assistant/conversations`.
- **Header:**
  - A conversation select (newest first) and **New**. New asks for effort:
    Low, Medium (the default, `effort_default`) or High, with xhigh and max
    under "More". Then `POST /api/assistant/conversations`.
  - Once a conversation exists, show its effort read-only.
  - Use `#assistant-model` for the model. Show `status.model`, then the last
    `usage.served_model` of the conversation, plus "(fallback)" when
    `fallback_served`.
  - **Spend line:** "This turn $0.06 · Conversation $0.42 · Today $1.10 of $5.00".
    It comes from the latest `usage` event. Before any usage event, use
    `state.assistant.today_usd` and `daily_cap_usd`, and conversation
    `spend_usd`.
  - Archive, and **Continue** for a locked conversation.
- **Permanent note:** "The assistant reads bench data and saves draft designs.
  Only you can queue or start runs."
- **Unavailable** (`available: false`): show the `reason` and a matching
  action. For example "Set ANTHROPIC_API_KEY on Setup" (open the key dialog
  for that key), or the daily-cap text. Disable the composer.
- **Message list:**
  - User bubbles. A recap message (`recap: true`) is shown collapsed as "Recap of <previous conversation>".
  - Assistant text as safe Markdown. Reuse `renderMarkdown`: escaped, no raw
    HTML, no images, no clickable links (URLs shown as text).
  - Thinking in a collapsed "Reasoning summary" disclosure.
  - **Tool chips:** the `tool_result.summary`, or "Calling <name>…" until the
    result arrives. Expanding a chip shows the `tool_call.input` JSON.
  - **Draft cards** for every `tool_result` with a `draft`:
    "Draft saved: <title> v<version> · $0.22 p50 ($0.44 p90) · 0 problems · Open".
    - Open navigates to DESIGNS on `draft.design_id` and `draft.version`.
      Always use these structured fields, never text from the model.
    - Mark problems > 0 in bad-text.
  - **Notices** by code:
    - fallback: "Answered by <to_model> after <from_model> declined";
    - retry;
    - refusal, with its category;
    - truncated, cap, stopped, max_steps, empty;
    - locked, as a banner with a **Continue** button
      (`POST .../continue`, then switch to the new conversation).
  - **Per-turn footer** from `turn_finished`: "$0.04 · 2 requests", and the
    status when it is not `done`.
  - **`turn_error`:** a bad-text block with `message`. Show **Retry** only when
    `retryable`; when `retry_after_s` is set, enable it after that delay.
- **Composer** (`#assistant-text`, `#assistant-send`):
  - Enter sends; Shift+Enter adds a newline.
  - "Attach open design" when DESIGNS has a version open. It sends
    `attach_design: {id, version}`.
  - Send becomes **Stop** while a turn runs: `active_turn` is set, or between
    `turn_started` and `turn_finished`.
  - The status line (`#assistant-status`, and a visible copy) shows the
    latest ephemeral `status` text.
  - On a 400 or 409, show `detail` in the toast and keep the text in the box.
- **Starter chips** for an empty conversation:
  - "Where are the coverage gaps?"
  - "Cheapest runs to get 10 paired tasks per harness on DeepSeek"
  - "Price block 1 on Opus 5.5 for all five harnesses"
  - "Why did mini-swe-agent fail?"
- **Streaming:**
  1. On open: `GET /api/assistant/conversations/{cid}` renders `events`.
     Remember the top-level `next_seq`.
  2. Then open one
     `new EventSource('/api/assistant/conversations/' + cid + '/stream?after=' + (next_seq - 1))`.
     Apply each `batch` in seq order, skipping any `seq` you have already
     applied.
  3. Let `EventSource` reconnect by itself.
  4. After three consecutive `error` events with no batch between them, close
     it. Fall back to polling `GET ...?after=<last seq>` every 1 s while a
     turn is active, and every 10 s otherwise.
  5. Close the EventSource when the panel closes or the conversation changes.
- Mirror `state.assistant.active_turn` and `today_usd` from the regular poll,
  so the LED and spend stay right while the panel is closed.

---

## Polling

| What | When |
|---|---|
| `GET /api/state` | As in v1: 3 s while a run is active or LIVE is open, 10 s otherwise. It now also carries the ledger, queue and assistant summaries. |
| `GET /api/queue` | Every 3 s while QUEUE or DESIGNS is open. |
| `GET /api/designs` | When DESIGNS opens, and every 10 s while it is open. |
| `POST /api/report` | On REPORT open, on filter change, and when `ledger.last_ingest_at` changes. |
| Assistant | Through SSE, with the polling fallback above. |

Re-render read-only parts only when their data changed. Comparing JSON
strings is fine.

## Testing

From the project root in Git Bash. `node` and headless Edge are installed, as
in v1.

```bash
T="$TEMP/bench-ui-v2"; mkdir -p "$T" && : > "$T/test.env"
env -u ANTHROPIC_API_KEY -u OPENROUTER_API_KEY -u FIREWORKS_AI_API_KEY \
  BENCH_ENV_FILE="$T/test.env" BENCH_DATA_DIR="$T/data" BENCH_LEDGER="$T/ledger/bench.sqlite" \
  .venv/Scripts/python -u -m bench --no-browser --port 8799
# Run it in the background. It prints "Dashboard: http://127.0.0.1:8799/?token=..."
```

What each part of that command does:

- `env -u ...` makes sure no real key from your shell reaches the bench.
  Without a key, the assistant is unavailable and refuses messages before
  any network call.
- `BENCH_DATA_DIR` keeps designs, conversations, `queue.json` and runs in the
  temp folder.
- `BENCH_LEDGER` keeps exclusions and snapshots out of the shared ledger.
- At startup the ledger rescans `profiles/dev/jobs`, read-only. So REPORT
  shows the 16 real pilot trials (9 used by default), while RUNS and History
  start empty.

Exercise these against the server:

- **Syntax:** `node --check` on every script you touch.
- **Rendering:** headless Edge with `--dump-dom`, as in v1. All six tabs and
  the assistant panel must fill in without errors.
- **Ledger and report:**
  - status, facets, runs, trials list and one trial;
  - a report with a few filter and option changes (`ci_level` 0.95,
    `group_by` config, and `policy.misconfigured_cap: "include"`, which makes
    the two claude-code `misconfigured_cap` trials count in their groups);
  - coverage and pairs;
  - a rescan;
  - an exclusion and its "include again";
  - a snapshot saved and opened;
  - each export downloaded.
- **Designs:**
  - create one with `POST /api/designs`;
  - live-check a half-filled spec with `/api/designs/check`;
  - save a second version, and trigger the stale 409 by saving from v1 again;
  - "From config arms…";
  - archive and unarchive.

  With no keys, Anthropic arms show one problem each
  (`"<arm>: ANTHROPIC_API_KEY is not set"`). That is expected, and it keeps
  Add to queue disabled. The plan and estimate still render.
- **Queue:**
  - `GET /api/queue`;
  - `PUT /api/queue/settings` (set a cap, reset it, set it back to `null`,
    switch on_failure);
  - `POST /api/queue/pause`.
- **Assistant, without a model:**
  - `GET /api/assistant/status` (unavailable, with its reason);
  - create a conversation;
  - **Continue** it. The new conversation gets one durable `user_message`
    event with `recap: true` and makes no model call. It exercises the
    initial GET, the EventSource batch, `id` and `Last-Event-ID`;
  - archive it.

Exercise **only in demo mode** (`?demo`). Extend `demo.js` so these are
answered in the browser and never sent:

- queue items, start, retry, move and remove on a fake in-memory queue;
- populated report comparisons (use the Metric example above);
- an available assistant with a scripted turn. Fake `EventSource` too, or
  have the panel use its polling path in demo mode. Script thinking deltas,
  text deltas, a `save_draft_design` `tool_call` and `tool_result` with a
  `draft`, `usage` with a fallback notice, and `turn_finished`;
- the ARM confirm dialogs for Add to queue, Start now, Start queue and Retry.

Today `demo.js` refuses `POST /api/runs`. Make it refuse every billed path
listed in Hard rule 2 in the same way.

At the end:

- `DELETE /api/config/overrides`, then confirm that `config/overrides.yaml`
  again holds only the header and `{}`, and that `has_overrides` is false.
- Stop your server.
  - Killing the Git Bash job can leave `python.exe` listening.
  - Check with `netstat -ano | grep ':8799 '`.
  - Stop only that PID: `taskkill //PID <pid> //F`.
- Remove `$TEMP/bench-ui-v2`.
- Confirm that the repository has no new `designs/`, `assistant/` or
  `queue.json`. `ledger/` existed before you started; leave it alone.

## Done means

- Every script passes `node --check`. All six tabs and the assistant panel
  render without console errors in headless Edge, against the server and in
  demo mode.
- Every endpoint in this brief is wired to the UI. Every non-billed one has
  been exercised once against the test server. Every billed one has been
  exercised in demo mode only.
- The report shows, with today's data:
  - the status strip and outcome bar;
  - insufficient comparisons that say how many pairs they need;
  - `not_applicable` verdicts with the "measured on Claude models" caveat;
  - the seeded exclusion with its reason;
  - per-config breakdowns;
  - the coverage matrix and next-run list.

  In demo mode, the populated plots, CIs and verdict chips render.
- Designs can be listed, viewed, live-checked, edited into a new version,
  archived and unarchived. Add to queue and Start now confirm with the ARM
  dialog and send the exact `spec_hash` shown.
- The queue shows order, status, pause/resume, cap, remove and move. Every
  billed action confirms first.
- The assistant panel:
  - shows the conversation;
  - streams, with the polling fallback;
  - places blocks by `block_key`;
  - renders draft cards that open DESIGNS;
  - shows spend today against the cap and the model that answered.
- Forms, the filter rail, the editor and the composer keep what the user is
  typing across polls.
- The repository is unchanged apart from `bench/static/*`. The overrides file
  is back to `{}`.

## Known gaps (work around them, and list them in your report)

1. **Launch codes are dropped (resolved 2026-10-29).** The backend will not
   add them. Remove the lock and use the ARM confirm on LAUNCH, as described
   under "LAUNCH, LIVE, ANALYSIS and Setup".
2. **No endpoint lists the harnesses a design may use, or the design limits**
   (`max_budget_usd`, `max_arms`, `max_attempts`, container limits). Derive
   the harnesses from `config.arms`, and rely on the check's problem
   messages for the limits.
3. **No endpoint lists the dataset snapshot's 89 tasks, or task difficulty.**
   Use `config.tasks` (the 20-task seeded sample) plus free text.
4. **Groups have no input/cache/output token split,** only total and uncached.
5. **`ConversationMeta.next_seq` is only updated when a turn ends.** Use the
   top-level `next_seq` from `GET /conversations/{cid}`.
6. **Ephemeral events are lost on a bench restart.** A reloaded transcript
   shows only durable events.

## Report back

- What you built, and how the files are split.
- Anything you changed outside `app.js`.
- Backend bugs or API gaps you found, with the endpoint and a short
  reproduction. Include any of the known gaps above that blocked you.

## Addendum (2026-10-29, revised the same day): trials whose usage was not recorded

A trial can end with its agent running but no usage recorded. Pilot 3's
pi-glm timed out after 900 s before its first model call completed, and
Harbor wrote 0/0/0 tokens. Such a trial used to count as a measured $0
trial. It now still counts for pass rate, but its tokens and cost are
unknown. A timeout that did record usage is kept, as a lower bound (the call
in flight is lost). All changes are additive except where a value is now
`null`.

The revision (versions 3) changed: which trials are `unmeasured` (an
outcome that means the agent ran, whatever its timing; a long provider
failure), the reason's tail, LedgerRun's counts, the verdict check behind
`withheld`, the caveat rule, two new `sensitivity` keys, a new `dropped`
key, the report setting aside floors below a harness's uncut trials, and
`cost_at_ingest_usd` being `null` where `cost_usd` is.

- **`usage_status`**: `"full"` | `"lower_bound"` | `"unmeasured"`, on every
  TrialRow (and the ledger CSV/JSONL export), every analysis row (the
  report's trials CSV), and every `GET /api/runs/{id}/results` trial.
  - `unmeasured`: the agent ran and nothing was recorded, or the trial was
    killed and its harness's log held no completed call. A reward, a
    timeout or a verifier run means the agent ran even without Harbor's
    agent timing, and a provider failure counts once it is past its first
    60 s. On a TrialRow, `tokens_total`, `tokens_uncached`, `cost_usd` and
    `cost_at_ingest_usd` are `null`, while `n_input`/`n_cache`/`n_output`
    keep Harbor's raw values (often 0). Show "not recorded", never 0 or $0.
  - `lower_bound`: completed calls only. This covers a Harbor timeout with
    usage, or a killed (`incomplete`) trial whose usage was recovered from
    its harness's own log. Show the figure with a "≥" or a "lower bound"
    tag. A recovered trial's `usage` entries carry `recovered_from` (for
    example `"agent/pi/sessions"`).
  - **Analysis rows only** (not TrialRows) may show `unmeasured` for a
    trial the ledger calls `lower_bound`: a timeout that recorded fewer
    tokens than every uncut trial of its group on the same task is set
    aside from tokens and cost, like one that recorded nothing.
- **`outcome_reason`** may end with `"; no usage recorded"` (Harbor wrote
  zeros) or `"; tokens not reported"` (Harbor wrote null), on any finished
  outcome whose usage is `unmeasured`.
- **LedgerStatus**: `parser_version`, `classifier_version` and
  `stats_version` are now 3. Existing rows re-derive on the next open or
  rescan, with no re-run.
- **LedgerRun** gains two counts of finished trials (killed ones stay in
  `trials.incomplete`, as ArmRun keeps them in `trials_killed`, so no trial
  is counted twice):
  - `trials_unmeasured`: ran but recorded no usage;
  - `trials_lower_bound`: cut short, so their usage covers completed calls.
  `spend_usd` is a lower bound while either of these or
  `trials.incomplete` is above 0.
- **Group.trials** gains:
  - `usage_unmeasured`: counted trials left out of tokens and cost
    (including floors set aside, see above);
  - `usage_lower_bound`: trials in the means whose usage is a floor.
  `per_trial.*.n` can now be smaller than `valid_reward`. Show both counts
  next to the group's cost, for example "1 trial without usage, left out".
- **Report `totals`** gains `usage_unmeasured`. While it is above 0,
  `totals.spend_usd` is a lower bound.
- **Metric.reference** gains `withheld: str|null`. Each side is scored
  again with its timeouts and unrecorded trials at 2x its usual spend on
  their task. If that changes the verdict, `verdict` is `"insufficient"`
  and `withheld` says why, for example "3 paired tasks rest on pi trials
  cut short or with no usage recorded; had those cost 2x pi's usual spend
  on their task, the verdict would be 'below', not 'overlaps'". Show the
  text in place of a verdict. The headline then ends in
  `"; verdict withheld: …"`.
- **Metric.sensitivity** gains `unrecorded_arm_high` and
  `unrecorded_baseline_high`: the geomean with that side's timeouts and
  unrecorded trials scored as above, or `null` when it has none. They feed
  `robust` like the other keys.
- **Metric.dropped** (and `pairs.dropped`) gains `usage_not_recorded`:
  shared tasks where every counted trial of one side ran but recorded no
  usage, whether Harbor wrote zeros or null (each trial's
  `excluded_trials[].reason` says which). Such tasks were counted under
  `tokens_not_reported` before the revision, which is now rarely above 0.
  `not_measured` also covers a side whose every trial lost its usage.
- **Caveats** may include:
  - "No usage was recorded for k of n A trials and j of m B trials; they
    count for pass rate but are left out of tokens and cost[, which leaves
    N shared tasks without a value on one side]. …" whenever either side
    has any;
  - "N X trial(s) cut short by a time limit recorded fewer tokens than
    every uncut X trial on the same task, …";
  - "No cost verdict: …" / "No token verdict: …";
  - "N X trial(s) hit the time limit; … lower bounds."
- **`excluded_trials[].reason`** can be `"no usage recorded"`,
  `"tokens not reported"` or
  `"cut short below every uncut trial on its task"`.
- **Coverage** `per_trial.*.mean_cost_usd` is a planning figure: trials
  without usage count at the group's most expensive trial, so it can exceed
  the group's `per_trial.cost.mean`.
- **`GET /api/runs/{id}/results`**: `arms[]` gains two counts.
  - `unmeasured` is the part of `excluded` that ran but recorded no usage,
    so its spend is not in `total_cost`.
  - `lower_bound` counts measured timeouts.
  Replace "N setup failures left out" with two notes: "N setup failures
  left out" for `excluded - unmeasured`, and "N without usage (spend not
  counted)" for `unmeasured`.
- **ArmRun** gains `trials_unmeasured` (finished trials only). The arm's
  `message` may end with "N finished trial(s) ran but recorded no usage; …
  recorded spend is a lower bound." Show "N without usage (spend not
  counted)" next to "N killed (spend not counted)".
