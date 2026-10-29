# Brief: build the dashboard front end (`bench/static/app.js`)

## What this project is

`c:\Git\EngineeringTestBench` is a local test bench that runs agent harnesses
(Claude Code, Terminus 2, mini-swe-agent, Pi, Strands) on a fixed model using
Harbor, and compares their token use and cost against published figures. It
runs natively on Windows 11. One Python process (`python -m bench`) loads keys
from `.env`, then serves a dashboard on 127.0.0.1. The dashboard is the
control plane. It manages keys, edits configuration, previews and starts runs,
and shows results.

The backend is written and has been tested. The page markup
(`bench/static/index.html`) and styling (`bench/static/style.css`) exist.
**What's missing is `bench/static/app.js`**, which holds all the page logic.

## Your scope

- Write `bench/static/app.js`. Use plain JavaScript with no build step, no
  libraries and no CDNs. It must work offline.
- You may make small additions to `index.html` and `style.css` if needed. Keep
  the existing element ids and classes.
- Do **not** edit `bench/*.py`, `config/*` or anything else. Another agent is
  working on the backend. If you need an API change or find a backend bug,
  describe it in your final report instead of editing.
- Do not commit.

## Hard rules

1. **Never read, print, create or edit `.env`.** The user's permission
   settings deny it. For testing, point the bench at a throwaway file with
   `BENCH_ENV_FILE` (see Testing) and use fake keys such as
   `sk-test-0000000000000000abcd`.
2. **Never call `POST /api/runs`.** It starts real runs that bill real
   accounts. `GET /api/plan` is free and is how you test the Run tab.
3. **Leave the config as you found it.** Saving forms writes
   `config/overrides.yaml` in the real project. When you finish, call
   `DELETE /api/config/overrides` or delete that file, and confirm it doesn't
   exist. It doesn't exist now.
4. Key values may appear only in the password input of the existing
   `<dialog id="key-dialog">`. Clear it after saving. Never render a key
   anywhere else; the API only returns masked values.

## API contract

`bench/server.py` is the source of truth; read it. To sign in, open
`/?token=<token>`, which sets an HttpOnly session cookie and redirects to `/`.
The token is printed at startup. Every non-GET request must send the header
`X-Bench: 1`. Errors come back as JSON `{"detail": "..."}` with status 400
(invalid input), 401 (not signed in), 403 (wrong host or missing header),
404 or 409 (conflict, for example a run already in progress). Show `detail`
in the toast.

| Method and path | Body | Returns |
|---|---|---|
| `GET /api/state` | | Everything the page polls (shape below) |
| `GET /api/schemas/{agent}` | | Harness option schema (below). The first call per agent takes a few seconds. |
| `PUT /api/keys/{ENV_KEY}` | `{"value": "..."}` | Key status list |
| `DELETE /api/keys/{ENV_KEY}` | | Key status list |
| `POST /api/keys/{ENV_KEY}/test` | | `{"ok": true\|false\|null, "detail": "..."}` (read-only provider call, no tokens used) |
| `PUT /api/config/run` | any of `effort, n_concurrent, budget_usd, per_trial_cap_usd, max_turns, timeout_multiplier` | config view. `""` resets a field to its default. `null` means "no cap" (caps only). |
| `DELETE /api/config/run` | | config view (resets all run settings) |
| `PUT /api/config/modes/{pilot\|sample}` | `{"n_tasks"?, "n_attempts"?}` | config view |
| `DELETE /api/config/modes` | | config view |
| `PUT /api/config/arms/{name}` | `{"enabled", "provider", "model", "effort", "options": {...}}` | config view. Send the **full desired set** of `options`; leave out unset fields. |
| `DELETE /api/config/arms/{name}` | | config view (resets that arm) |
| `DELETE /api/config/overrides` | | config view (resets everything) |
| `POST /api/config/reload` | | config view (rereads the YAML files) |
| `GET /api/plan?mode=pilot\|sample&arms=a,b` | | Plan (below). Leave out `arms` to use the enabled arms. |
| `POST /api/runs` | `{"mode", "arms": [...]}` | **Do not call** (bills money) |
| `POST /api/runs/{id}/cancel` | | `{"ok": true}` |
| `GET /api/runs/{id}/results` | | Results (below) |
| `GET /api/runs/{id}/log/{arm}?lines=200` | | `text/plain` log tail |

### Shapes

`GET /api/state`:
```
{ readiness: [{name, ok, detail}],
  keys: [{provider, env_key, enabled, set, masked, source: ".env"|"shell"|null}],
  config: { dataset, tasks: [names], providers: [{name, env_key, model_prefix, enabled}],
            modes: {pilot: {n_tasks, n_attempts}, sample: {...}},
            run: {effort, n_concurrent, budget_usd, per_trial_cap_usd, max_turns, timeout_multiplier},
            run_overridden: [field names], modes_overridden: [mode names], baseline_arm,
            arms: [{name, agent, provider, model, enabled, effort, options: {}, overridden,
                    model_warnings: {option_name: message}}],
            has_overrides },
  config_error: string|null, runs: [Run], active_run: run id|null }
```
- Arm `effort` is `"inherit"` (follows `run.effort`), `"omit"` (harness default) or a level.
- Run: `{id, mode, created_at, finished_at, budget_usd, spend_usd, tasks, status, message, arms: [ArmRun], plan, config}`.
  Run `status`: `running, done, done_with_errors, budget_stopped, cancelled, failed, interrupted`.
- ArmRun: `{name, status, message, trials_total, trials_done, spend_usd, started_at, finished_at, returncode}`.
  ArmRun `status`: `queued, preflight, running, done, failed, cancelled, skipped, budget_stopped, interrupted`.

`GET /api/schemas/{agent}`:
```
{ agent, effort_levels: null | [] | ["low", ...],
  fields: [{name, kind: "enum"|"boolean"|"integer"|"number"|"string"|"json",
            enum, default, description, group: "reasoning"|"limits"|"other", caution}] }
```
`effort_levels: null` means the harness takes no effort setting (Strands).
`[]` means any text is accepted (mini-swe-agent).

`GET /api/plan`:
```
{ mode, tasks, n_attempts, trials, budget_usd, worst_case_capped_usd, uncapped_arms: [names],
  arms: [{name, agent, provider, model, trials, effort, kwargs: [{name, value, source}],
          command, problems: [..], warnings: [..]}],
  problems: [..] }
```

`GET /api/runs/{id}/results`:
```
{ run_id, baseline,
  arms: [{arm, agent, model, trials, scored, passed, errors, pass_rate, mean_input, mean_cache,
          mean_output, mean_tokens, cache_share, mean_cost, total_cost, reported_cost_total,
          cost_per_success, other_model_share, models, unpriced_models, mean_minutes}],
  comparison: [{arm, tokens_multiple, cost_multiple,
                published_tokens: {value, source}|null,
                published_cost: {value, source}|{low, high, source}|null,
                tokens_verdict, cost_verdict: "in line"|"lower"|"higher"|null, note}],
  trials: [{arm, task, trial, reward, passed, error, error_message, n_input, n_cache, n_output,
            computed_cost, reported_cost, minutes, models}] }
```
An arm with `trials: 0` has only `arm, agent, model, trials`. Multiples are
**baseline ÷ arm**: 2.0 means that harness used half of what the baseline used.

## What each tab does

Use the ids already in `index.html`. Plain, short labels. Use `–` for missing values.

**Header.** `#status-pill` shows "Idle", or "Running <id>: $spend of $budget"
(class `running`), or "Disconnected" (class `bad`) if polling fails.

**Setup.**
- `#readiness`: one row per check, with the name, a ✓ or ✗ (`ok-text` / `bad-text`) and the detail.
- `#keys`: a table with Provider, Variable, Status and Actions.
  - Status reads "Set …abcd (.env)" or "Not set". A provider with `enabled: false` gets a tag "off in config" (OpenRouter is off by default).
  - Actions: **Set** / **Replace** opens the dialog. **Test** (when set) shows the result inline and keeps it across re-renders. **Clear** (when set, after a confirm).

**Configure.** Render the forms when the tab opens, after a save and after a
reload. **Never re-render while the user is editing.** Show `config_error` in
`#config-error`. `#reload-config` and `#reset-all` (with a confirm) call the
matching endpoints.
- `#run-settings`: effort (select: low, medium, high, xhigh, max, plus
  "omit, each harness's default"), parallel trials, run budget (USD),
  per-trial cap (USD, empty = none), max turns (empty = none) and timeout
  multiplier.
  - Put a `.dot` beside overridden fields (`run_overridden`).
  - Add a **Save** button, and a **Reset** button that calls `DELETE /api/config/run`.
- `#modes`: n_tasks (1 to `tasks.length`) and n_attempts for pilot and sample, with Save and Reset.
- `#arms`: one `.card` per arm, built from `GET /api/schemas/{agent}`:
  - **Header:** the name, the agent, an "edited" tag if `overridden`, and an
    **enabled** checkbox.
  - **Provider** (select; label disabled providers "(off)") and **Model** (text).
  - **Effort** select: "inherit (run: <run.effort>)", "omit (harness
    default)", then the schema's levels.
    - If `effort_levels` is `null`, disable it and show "not supported by this harness".
    - If it is `[]`, use a text input instead.
  - Groups with a `.group-title`:
    - **Reasoning and thinking**: fields in group `reasoning`.
    - **Limits**: fields in group `limits`. Placeholders show the run caps that
      apply when the field is empty: `max_budget_usd` and `cost_limit` use
      `run.per_trial_cap_usd`; `max_turns` uses `run.max_turns`.
    - **Other**: inside a collapsed `<details>`.
  - Inputs by kind:
    - `enum`: a select with an empty "default" option.
    - `boolean`: a three-way select: default, true, false.
    - `integer` / `number`: a number input.
    - `string`: a text input.
    - `json`: a textarea. Check that it parses before sending.

    Put the field's description in `title`. Show `caution` and
    `model_warnings[name]` under the field as `small.warn-text`.
  - **Save** sends the full option set (unset fields left out). **Reset**
    appears only if the arm is overridden.

**Run.**
- `#run-form`:
  - A mode select ("Pilot: 1 task × 1 attempt" and so on, from `config.modes`).
  - One checkbox per arm, checked for enabled arms.
  - **Preview plan** (calls `GET /api/plan`) and **Start run**.
- `#plan`:
  - A summary line: trials, arms, run budget, worst case for capped arms.
    Name the uncapped arms, and say that the run budget still applies to
    them and is checked after each trial.
  - A table with arm, model, effort, the options passed (`.chip` showing
    `name=value` plus the source in `.src`), trials, problems (`bad-text`)
    and warnings (`warn-text`).
  - A `<details>` holding each command in `pre.cmd`.
- **Start run** stays disabled until a preview shows no problems. If no run is
  active, it asks: "Start N trials? This bills your provider accounts. The run
  stops at $budget." (You won't call it, but wire it up.)
- `#active-card` (hidden when nothing is running):
  - One row per arm: status tag, trials done of total, spend, message.
  - **Cancel** (after a confirm).
  - An arm select and a `pre.log` showing that arm's log tail, refreshed while
    the tab is open.
- `#history`: id, mode, status, arms, spend of budget, start time and a
  **Results** button that switches to the Results tab for that run.

**Results.**
- `#results-run` lists the runs, defaulting to the newest run that has trials.
- `#results` shows, in order:
  1. A note: "Ten tasks can't separate pass rates; compare the token and cost multiples."
  2. A per-arm table with the baseline row in class `baseline`. Columns:
     trials, passed of scored, errors, mean tokens, input / cache / output,
     cache share, mean cost, cost per success, share from other models, mean
     minutes. Mark `unpriced_models` if present.
  3. A comparison table: arm, tokens × against the published value, and cost ×
     against the published value or the low–high band. Verdicts are tags:
     `ok` for "in line", `warn` otherwise. The source or note goes in small text.
  4. A `<details>` with the trials table.
- Refresh while the selected run is active.

**Polling.** Poll `GET /api/state` every 3 s while a run is active or the Run
tab is open, and every 10 s otherwise. Re-render read-only sections only when
their data changed; comparing JSON strings is fine.

## Testing

From the project root in Git Bash. `node` is installed.

```bash
mkdir -p "$TEMP/bench-ui-test" && : > "$TEMP/bench-ui-test/test.env"
BENCH_ENV_FILE="$TEMP/bench-ui-test/test.env" .venv/Scripts/python -u -m bench --no-browser --port 8799
# run that in the background; it prints "Dashboard: http://127.0.0.1:8799/?token=..."
```

- Syntax: `node --check bench/static/app.js`.
- API: sign in with `curl -c jar "http://127.0.0.1:8799/?token=..."`, then
  `curl -b jar -H "X-Bench: 1" ...`.
- Rendering: headless Edge runs the page and dumps the DOM after the scripts finish:
  `"/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless=new --disable-gpu --virtual-time-budget=10000 --dump-dom "http://127.0.0.1:8799/?token=..."`.
  Check that every tab's containers fill in. Tabs are hidden, not removed, so all of them render.
- Exercise:
  - Setting and clearing a fake key.
  - Saving a run setting and an arm option (for example `thinking_display`
    on claude-code), and confirming it in `/api/state`.
  - Previewing the pilot and sample plans.
- Docker Desktop is not running, so readiness shows Docker ✗. That's expected.
  No runs exist, so the Results and History states must handle "no runs yet".
- At the end: stop the server, remove the temp folder, and confirm
  `config/overrides.yaml` doesn't exist.

## Done means

- `app.js` passes `node --check` and renders all four tabs without errors in headless Edge.
- Every endpoint above (except `POST /api/runs`) is wired to the UI and has been exercised once.
- Forms keep what the user is typing across polls.
- The repository is unchanged apart from `bench/static/*`, with no `config/overrides.yaml`.

## Addendum: backend changes after this brief was written

Restart your test server to pick these up.

1. **New run setting `parallel_arms`** (integer ≥ 1): how many arms run at
   once. All of them share the run budget. It appears in `config.run`, is
   accepted by `PUT /api/config/run`, and can appear in `run_overridden`. Add
   it to the Run settings form next to parallel trials.
2. **New ArmRun status `done_with_errors`**: Harbor exited cleanly, but some
   trials errored. `message` reads "N of M trials errored before finishing".
   Show it as a `warn` tag.
3. **`config.profile`** is `"default"` or a profile name such as `"dev"`.
   Show it in `#profile-tag` when it isn't `"default"`.
4. **`GET /api/plan` returns two new fields:**
   - `dataset`, for example `"terminal-bench@2.0"`.
   - `dataset_snapshot`: `null` until the dataset is downloaded, then
     `{dataset, tasks_dir, tasks, downloaded_at, harbor, path}`.

   If `dataset_snapshot` is `null`, add a note to the plan summary that the
   first run downloads the dataset once. Commands now use
   `--path datasets\terminal-bench@2.0\...` instead of `--dataset`.
5. **Run dicts have a new `dataset` field**: the snapshot info the run used.
   Show it in the Results header if present.
6. **`/api/state` readiness has a fifth row**, "Dataset snapshot".

A separate dev-profile server runs on port 8798, with real (cheap) runs under
`profiles/dev/`. Leave that port and those files alone. Your test server on
8799 and the default profile are unaffected.

## Report back

- What you built.
- Anything you changed outside `app.js`.
- Any backend bugs or API gaps you found, with the endpoint and a short reproduction.
