# Engineering Test Bench (`eng-test-bench`) — Architecture & Pipeline Breakdown

**Repository Commit:** `2f67cf6e760581f3c67c09f6eb6b73bd32490084`  
**Stack:** Python 3.13, FastAPI + Uvicorn, Harbor `0.25.0`, Docker, SQLite, Anthropic SDK `2.3.0` (`claude-opus-5-5`), Vanilla JS SPA (no build step)

---

## 1. End-to-End Pipeline Overview

`eng-test-bench` benchmarks AI coding agent harnesses on fixed models and tasks (such as `terminal-bench@2.0`) under controlled, paired execution conditions. Every experiment progresses through eight deterministic pipeline stages:

```text
+-----------------------------------------------------------------------------+
| 1. Config, Keys & Normalized Run Specs                                      |
|    config/*.yaml, config/tasks*.txt, .env -> settings.py, keys.py, spec.py  |
+--------------------------------------+--------------------------------------+
                                       v
+-----------------------------------------------------------------------------+
| 2. Versioned Designs, Preflight Checks, Reuse & Estimates                   |
|    designs/<id>/vNNNN.json -> checks.py, reuse.py, estimates.py             |
+--------------------------------------+--------------------------------------+
                                       v
+-----------------------------------------------------------------------------+
| 3. Single-Concurrency Run Queue & Task Races via Harbor                     |
|    runqueue.py (q-NNNN) -> runs.py (RunManager) -> harbor run               |
|    Custom harnesses: bench/agents/* (--agent-import-path)                   |
+-----------------------+------------------------------+----------------------+
                        |                              | (if prebake enabled)
                        v                              v
+-------------------------------+    +----------------------------------------+
| Host Safety & Load Sampling   |    | 4. Pre-Baked Docker Image Pipeline     |
| hostload.py, procs.py         |    | recorder.py -> dockerfile.py ->        |
| (NO_CTRL_C, kill_tree)        |    | pull.py -> builder.py -> verify.py ->  |
+-----------------------+-------+    | inventory.py -> environments/prebaked  |
                        |            +-----------------+----------------------+
                        v                              v
+-----------------------------------------------------------------------------+
| 5. Trial Extraction, Costing, Gateway Reconciliation & Answer-Lookup Audit  |
|    extract.py + pricing.py + gateway_spend.py + classify.py + lookups.py    |
+--------------------------------------+--------------------------------------+
                                       v
+-----------------------------------------------------------------------------+
| 6. SQLite Ledger Persistence                                                |
|    ledger.py (bench.sqlite), results.py, fsutil.py (atomic Windows-safe I/O)|
+--------------------------------------+--------------------------------------+
                                       v
+-----------------------------------------------------------------------------+
| 7. Paired Bootstrap Statistics, Pooled Report & Coverage Planner            |
|    stats.py -> report.py (harness / baseline multiples) -> coverage.py      |
+--------------------------------------+--------------------------------------+
                                       v
+-----------------------------------------------------------------------------+
| 8. FastAPI SSE Control Room & Read-Only Opus 5.5 Assistant                  |
|    server.py (create_app) <-> bench/static/* (?demo) + bench/assistant/*    |
+-----------------------------------------------------------------------------+
```

### Cross-Cutting Safety & Cross-Platform Invariants

- **Windows-safe atomic filesystem primitives (`bench/fsutil.py`):** Handles concurrent readers (antivirus, search indexers, background dashboard pollers) that cause `PermissionError` on `Path.replace()` via temporary files + `fsync` + retry (`atomic_write_text`, `atomic_write_json`), write-once exclusive JSON records (`create_exclusive_json` using `open("x")`), binary LF-only append (`append_jsonl`), and canonical JSON hashing (`canonical_json`, `sha256_hex`).
- **Process tree containment (`bench/procs.py`):** Enforces single-instance locking (`acquire_instance_lock`), launches subprocesses without console Ctrl+C propagation (`NO_CTRL_C`), assigns Windows kill-on-close Job Objects (`assign_kill_on_close`), performs recursive process-tree teardown (`kill_tree`), and cleans up labeled Compose projects (`cleanup_containers`, `cleanup_with_retry`).
- **Credential scrubbing (`bench/keys.py`):** Loads `.env` secrets safely, exposes only masked key statuses to UI/APIs (`status`, `mask`), never logs secret values, injects `PYTHONPATH` for custom Harbor agents, and strips `ANTHROPIC_BASE_URL` and sibling ambient gateway variables from child environments (`child_env`, `hide_gateway_urls`).

---

## 2. Stage-by-Stage Pipeline Breakdown

### Stage 1: Configuration, Profiles, Harness Registry & Specs
- **Entry & Launchers:** `start.cmd` delegates to `start.ps1` (bootstrapping/activating `.venv`, installing `requirements.txt`, opening the browser), which runs `python -m bench` (`bench/__main__.py`). Supports profile isolation (e.g., default production vs `-Profile dev` via `config/dev.yaml`).
- **Configuration Loading (`bench/settings.py`):** Merges YAML defaults (`config/bench.yaml`, `config/dev.yaml`, `config/pricing.yaml`, `config/reference.yaml`, `config/answer_lookups.yaml`, plus sparse `config/overrides.yaml`) into typed dataclasses (`Config`, `Provider`, `Arm`, `Mode`, `RunSettings`, `Price`, `AssistantSettings`, `DesignLimits`, `StatsSettings`, `QueueSettings`, `LedgerSettings`, `PrebakeSettings`).
- **Task Selection (`bench/datasets.py`, `bench/sample_tasks.py`):** `datasets.ensure` materializes local dataset snapshots (`terminal-bench@2.0`), parses per-task `task.toml` metadata (`timeout_sec`, difficulty), and enumerates task Docker images. `sample_tasks.py` drew the difficulty-proportional `20`-task sample (`SEED = 20261028`, two `10`-task blocks, dropping `> 3600s` tasks) into `config/tasks.txt`, while full benchmark evaluations run `27` tasks (`docs/results.md`, `docs/report.html`).
- **Harnesses & Option Schemas (`bench/harnesses.py`, `bench/schemas.py`):** Maps canonical harness names vs `--agent-import-path` subclasses onto unified `family` names (so custom wrappers pool under their canonical harness family), normalizes effort settings (`effective_effort`, `options_without_effort`), and caches Harbor CLI option schemas via credential-free subprocess inspection (`agent_schema`, `warm`).
- **Deterministic Run Specs (`bench/spec.py`):** Normalizes run configs, designs, or UI launch requests into immutable `ArmSpec` and `RunSpec` structures, computing `race_lane_slots` / `race_peak` concurrency and canonical `sha256:` `spec_hash` digests.

### Stage 2: Designs, Preflight Checks, Trial Reuse & Cost/Time Estimation
- **Versioned Immutable Designs (`bench/designs.py`):** Stores named runs under `designs/<id>/` (or `profiles/dev/designs/<id>/`) with IDs `d-<slug>-<4hex>`. `DesignStore` persists append-only numbered snapshots (`v0001.json`, `v0001.check.json`, `v0002.json`, ...) and an atomic `meta.json` index tracking `latest_version`, `spec_hash`, approvals, and linked run IDs. Saved versions never mutate; edits create new versions referencing `parent_version`. `DraftWriter` lets the assistant save `author: "assistant"` drafts safely.
- **Preflight Readiness Checks (`bench/checks.py`):** TTL-cached probes (`_cached`) verify Docker availability (`_docker`), CPU/container capacity (`docker_capacity`), roomiest host drive & free disk (`_roomiest_drive`, `_docker_disk`), dataset snapshot availability, git fix status, provider API keys (`test_key`), and pre-bake image readiness (`_prebake_images`).
- **Trial Reuse (`bench/reuse.py`):** Matches normalized arms and tasks in a design spec against existing valid trials in the SQLite ledger (`resolve`, `coverage`, `drift`) so unchanged baseline/arm tasks can be marked `reused_tasks` rather than re-billed.
- **Forecasting (`bench/estimates.py`):** Builds empirical per-cell models (`CellModel`, `cell_stats`) from historical ledger medians combined with configurable three-point priors to estimate projected USD spend and wall-clock runtime (`spend_of`, `_ArmSim`).

### Stage 3: Single-Concurrency Run Queue & Task Races via Harbor
- **Run Queue (`bench/runqueue.py`):** `RunQueue` admits approved design versions (`q-0001`, ...) bound to an exact `spec_hash`. Only **one run** executes at a time (`Manager`), enforcing global spend caps and automatically pausing the queue on failure, cancellation, or cap exhaustion so unattended batches cannot burn runaway spend.
- **Task-Race Execution (`bench/runs.py`):** Instead of running one harness for all 27 tasks sequentially and another later (which biases results by time-of-day load and API latency), `RunManager` executes **task races**:
  - For each task in the schedule, all enabled arms (`ArmRun`) race concurrently inside Harbor (`harbor run`) within `max_containers` slot budgets (`_Slots`, `race_build_jobs`).
  - Tracks per-arm `reserve_usd`, `trials_running`, `trials_killed`, `trials_unmeasured`, `cleanup_pending`, and live log streaming.
  - Passes custom agent subclasses via `--agent-import-path` (`bench/agents/*`):
    - `providers.py`: Shared multi-provider env routing & Responses API detection (`Limits`, `needs_responses_api`).
    - `opencode.py` (`OpenCodeBunEpipe`): Mitigates Bun EPIPE crashes on exit (`finished_despite_epipe`) and patches multi-provider routing.
    - `kilo.py` (`KiloCode`): Inherits `OpenCodeBunEpipe` workarounds for Kilo CLI.
    - `cline.py` (`PinnedClineCli`), `goose.py` (`PinnedGoose`), `mini_swe_agent.py` (`MiniSweAgentWithMcp`), `pi.py` (`PiProviders`), `dsh.py` (`DeepSeekHarness`), `droid.py` (`Droid`, plus diagnostic log recovery).
- **Host Contention Monitoring (`bench/hostload.py`):** `Sampler` samples host CPU counters (`busy_fraction`) during runs so trials executed under `>= 90%` host saturation for `>= 25%` of agent runtime (`contention`, `is_contended`) are excluded from paired latency comparisons without discarding validity for pass/cost.

### Stage 4: Pre-Baked Docker Image Pipeline (`bench/prebake/*` & `bench/environments/`)
Installing agents inside containers during every live trial added 1–5 minutes per trial and risked transient install failures or Harbor build timeouts (e.g., large `hf-model-inference` 15 GB / `mteb-retrieve` 21.6 GB pulls competing with Harbor's 600s setup limit). The prebake subsystem eliminates setup overhead while guaranteeing container equivalence:
1. **Probe & Record (`bench/prebake/recorder.py`, `adapters.py`):** `ProbeRunner` captures OS/tooling answers (`DockerProbes`) on local task images, and `_RecordingEnv` records exact Harbor `install()` exec commands (`Step`) and host file uploads (`Upload`) into a credential-free `Recording`.
2. **Strict Dockerfile Rendering (`bench/prebake/dockerfile.py`):** `render()` emits only `ARG BASE`, `FROM ${BASE}`, `COPY --chmod=755 ...`, and JSON-exec `RUN [...]` lines byte-identical to Harbor's own setup calls, forbidding `ENV`, `USER`, `WORKDIR`, `SHELL`, `CMD`, `ENTRYPOINT`, `LABEL`, or syntax directives (`FORBIDDEN_INSTRUCTIONS`) and rejecting any secret-bearing step.
3. **Pull, Build, Verify & Inventory (`pull.py`, `builder.py`, `verify.py`, `inventory.py`, `prune.py`):**
   - `pull.ensure_local()` pre-pulls missing task images concurrently.
   - `builder.BuildManager` builds under `benchbake-staging/` with disk guardrails (`DiskCheck`) and attaches `org.etb.prebake.*` metadata via Docker `--label` flags.
   - `verify.py` runs `static_check`, `config_diff`, `smoke`, and `equivalence` checks before promoting images to `<prefix>/`.
   - `inventory.py` runs batched credential-free `docker image ls` / `docker image inspect` (`readiness`) using the exact same pure `check_candidate()` function as live trials so plans and trials never disagree.
   - `prune.py` removes only labeled stale/staging/wrong-Harbor bakes (`org.etb.prebake.bake-format`) and never deletes task images.
4. **Runtime Verification (`bench/environments/prebaked.py`):** `PrebakedDockerEnvironment` (loaded inside Harbor using only Harbor + stdlib) runs `check_candidate()` against `Want` labels (Harbor version, bake format, harness version, recipe digest, base task image ID) before booting the pre-baked image.

### Stage 5: Trial Extraction, Costing, Gateway Spend & Answer-Lookup Audit
- **Artifact Extraction (`bench/extract.py`):** Parses raw Harbor trial directories (`RunContext`, `TrialFacts`, `HarnessLog`) across all 11 harness formats (Claude Code, Pi, Goose, OpenCode, Kilo, Cline, Terminus-2, mini-swe-agent, Strands, Droid, DeepSeek Harness), extracting durations, token counts, unmeasured flags (`usage_status`), stop causes, and secret-redacted transcripts.
- **Pricing & Gateway Reconciliation (`bench/pricing.py`, `bench/gateway_spend.py`):** Computes USD spend from `config/pricing.yaml` (`price_usage`, handling cached prompt reads and `1.25x` 5-minute cache writes). When arms report mispriced or translation-inflated token counts (such as LiteLLM gateway routing), `gateway_spend.py` matches upstream request IDs against actual gateway `SpendLog` rows (`SpendRow`, `Matching`) and emits corrected/upper-bound costs (`corrected_flag`, `upper_bound_flag`).
- **Outcome Classification (`bench/classify.py`):** Classifies each trial (`classify`, `Validity`) into `passed`, `failed`, `answer_lookup`, `agent_limit`, `agent_crash`, `verifier_failure`, `provider_failure`, `setup_failure`, `infra`, or `incomplete`, separating agent/model failures from infrastructure noise and host contention.
- **Answer-Lookup Detection (`bench/lookups.py`):** Parses ATIF/agent trajectories (`atif_calls`, `dsh_calls`, `droid_calls`) and evaluates bash/tool calls against `config/answer_lookups.yaml` patterns (`judge`, `scan`). Trials that downloaded or read benchmark answer keys/tests online (`13` total across the documented benchmark runs) are automatically overridden to `answer_lookup` failures so internet-enabled agents cannot inflate pass rates.

### Stage 6: SQLite Ledger (`bench/ledger.py`, `bench/results.py`)
- **`bench/results.py`:** Summarizes individual runs (`run_results`, `summarize`, `compare`) with per-arm pass counts, median cost/time, and exclusion flags.
- **`bench/ledger.py`:** Maintains a unified SQLite ledger (`bench.sqlite`, `Ledger`) across all configured profiles (`default`, `dev`, etc.), ingesting runs and trials keyed by `trial_key` (`<harbor job id>/<trial name>`), persisting manual exclusions (`excluded_manual`), answer-lookup verdicts, gateway spend corrections, and named report snapshots, and exposing filtered query helpers (`Filters`, `parse_filters`).

### Stage 7: Paired Bootstrap Statistics, Pooled Report & Coverage (`bench/stats.py`, `bench/report.py`, `bench/coverage.py`)
- **Paired Statistics (`bench/stats.py`):** Avoids unpaired selection bias by evaluating task intersections:
  - `log_ratio_summary` & `bootstrap_ci`: Nonparametric paired confidence intervals for cost (`harness ÷ baseline`) and agent run-time multiples (`RatioEstimate`), excluding `is_contended` trials from time ratios while keeping minimum overlap thresholds (`>= 8` pairs).
  - `newcombe_paired`, `wilson`, `mcnemar_exact`: Pass-rate differences and confidence intervals.
  - `leave_one_out`, `n_needed_ratio`: Sensitivity and sample-size power checks.
- **Pooled Cross-Run Report (`bench/report.py`):** Computes pooled group summaries (`<family>|<model>|<effort>`) and comparison pairs against the selected baseline harness (e.g., Claude Code on Claude models, Goose on non-Claude models) or baseline model (`Opus 5.5 medium`). Evaluates verdicts (`<`, `=`, `>`, overlaps) against `config/reference.yaml`.
- **Coverage & Next-Run Planner (`bench/coverage.py`):** `ReadOnlyLedger` and `coverage()` compute the `harness × model` completeness matrix, surface missing comparisons or thin baselines (`_gaps`), and greedily suggest highest-value next runs (`suggest_runs`).

### Stage 8: FastAPI Control-Room Dashboard, SPA Frontend & Read-Only Assistant
- **FastAPI Backend (`bench/server.py`):** `create_app` mounts modular route groups (`_state_routes`, `_config_routes`, `_run_routes`, `_ledger_routes`, `_report_routes`, `_design_routes`, `_queue_routes`, plus prebake & assistant SSE endpoints), holding in-memory authoritative `State` synced with disk/ledger locks.
- **Vanilla JS SPA (`bench/static/*`):**
  - `index.html` + `style.css` render a dark industrial 6-tab control room (**LIVE**, **LAUNCH**, **DESIGNS**, **QUEUE**, **REPORT**, **ANALYSIS**) plus the **IMAGES** modal and **Assistant** drawer with zero build tooling or external CDNs.
  - Modular IIFEs (`app.js`, `designs.js`, `queue.js`, `report.js`, `assistant.js`) coordinate via shared UI primitives (`h`, `api`, `formDialog`, `armConfirm` requiring typed `ARM` confirmation for billed operations) and dirty-input `hold()` locks so background `/api/state` polls never clobber user edits.
  - `?demo` mode (`demo.js`, `demo-designs.js`, `demo-queue.js`, `demo-report.js`, `demo-images.js`, `demo-assistant.js`) intercepts `fetch` and `EventSource` in-browser while blocking all billed server POST routes with `409`.
- **Run-Design Assistant (`bench/assistant/*`):**
  - `client.py`: Builds per-turn `anthropic.Anthropic` clients pinned to `https://api.anthropic.com` using `claude-opus-5-5`, adaptive summarized thinking, ephemeral prompt caching, and optional `server-side-fallback-2026-07-01` (`claude-opus-5`, `claude-opus-4-8`).
  - `tools.py` & `types.py`: Isolates the LLM behind `ToolContext` — strictly read-only ledger/coverage/design/estimate/run inspection plus `save_draft_design` (capped at `SAVES_PER_TURN = 5`), with zero ability to read keys, mutate configs, enqueue items, or start billed runs.
  - `engine.py`, `conversation.py`, `costs.py`, `redact.py`: Streams assistant thinking/text/tool cards over SSE (`transcript.jsonl`), preserves verbatim `messages.jsonl` thinking-block replay integrity, locks conversations cleanly above `LOCK_CONTEXT_TOKENS = 150,000` (continuing via plain-text `recap()`), enforces per-turn and daily USD spend caps (`SpendBook`), and redacts credentials across all logs.

---

## 3. Mapping the Android Benchmark Version onto `eng-test-bench`

An Android test-bench implementation preserves `eng-test-bench`'s core experimental guarantees—task-paired races, pre-baked reproducibility, strict secret isolation, cheat detection, SQLite ledgering, and paired bootstrap comparison—while replacing Docker/Linux-container primitives in Stages 1–5 with Android emulator/device equivalents:

| Pipeline Stage | Current `eng-test-bench` (Harbor + Docker) | Android Version Equivalent Mapping |
|---|---|---|
| **1. Config, Datasets & Specs** (`settings.py`, `datasets.py`, `spec.py`) | `terminal-bench@2.0` task folders (`task.toml`, task Docker image) + Harbor agent names / `bench.agents.*`. | Android task suite manifests (`task.toml` / scenario JSON declaring target API level, ABI, base AVD profile, starter APK(s), verifier Instrumentation/UIAutomator/ADB script, timeout) + Android harness adapters (mobile-use / ADB / GUI / coding-on-device agents). `spec_hash` incorporates AVD system-image digest, APK SHA-256, and harness version. |
| **2. Designs, Checks & Reuse** (`designs.py`, `checks.py`, `reuse.py`) | Checks Docker daemon, container limit, roomiest drive free space, dataset download, and pre-baked images. | Checks `adb` daemon availability, host KVM/hardware-acceleration readiness, host RAM/disk budget per concurrent AVD, Android SDK/emulator binaries, base AVD snapshots ready, and connected physical devices (if any). Reuses immutable `DesignStore` (`vNNNN.json` / `.check.json`) and `reuse.py` trial matching on identical `spec_hash` + Android task IDs. |
| **3. Queue & Task Races** (`runqueue.py`, `runs.py`, `procs.py`, `hostload.py`) | Single-concurrency `RunQueue`; `RunManager` runs concurrent task races bounded by `max_containers`; `kill_tree` tears down Harbor/Compose trees; `hostload.py` flags CPU contention `>= 90%`. | Keep `RunQueue` single-concurrency cap intact. Replace container slots with an **AVD/Device Pool** (`max_emulators` / allocated ADB serials `emulator-5554`, `emulator-5556`, ...), booting all compared harness arms simultaneously per Android scenario on isolated ephemeral AVD instances from identical snapshots. `procs.py` tracks emulator/QEMU/ADB child PIDs with `kill_tree` orphan cleanup; `hostload.py` samples host CPU **plus** ADB/guest frame-drop or UI-idle stalls. |
| **4. Pre-Baking Pipeline** (`prebake/*`, `environments/prebaked.py`) | Records Harbor `install()` shell steps -> strict exec-form Dockerfiles -> `benchbake/` images verified via `org.etb.prebake.*` labels. | **Snapshot & APK Pre-Baking:** Records deterministic provisioning (installing harness companion APKs, ADB helper binaries, accessibility services, test APKs, permissions/settings flags) onto a base AVD without embedding API keys (`keys.child_env(None)`), saves labeled read-only AVD Quickboot snapshots, verifies booted snapshot health & equivalence (`verify.py` analog), and launches each trial with `-snapshot <name> -read-only` / `-no-snapshot-save` so trials start cleanly in seconds and never contaminate state. |
| **5. Extraction, Costing & Anti-Cheat** (`extract.py`, `classify.py`, `lookups.py`) | Extracts agent logs/tokens (`pricing.py`), Gateway logs, classifies `agent_limit` / `infra`, checks trajectories against `config/answer_lookups.yaml`. | `extract.py` pulls Harbor/agent JSON logs **plus** `adb logcat`, Instrumentation/UIAutomator test XML/results, accessibility/UI trees, and step screenshots/screen recordings. `lookups.py` audits shell commands, intent invocations, and network/proxy logs so agents cannot `adb pull`/inspect verifier APKs, hard-coded test assertions on `/data/local/tmp`, or fetch task solutions online. |
| **6. SQLite Ledger** (`ledger.py`, `results.py`) | Unified `bench.sqlite` keyed by `trial_key` (`job/trial`), storing outcomes, tokens, USD, duration, host load, exclusions. | Identical schema & ingestion model (adding optional device/API-level/AVD metadata columns or `run_context` attributes), preserving cross-profile pooling and manual exclusions. |
| **7. Bootstrap Stats & Coverage** (`stats.py`, `report.py`, `coverage.py`) | Task-paired `harness ÷ baseline` cost and latency CIs, Wilson/Newcombe pass intervals, `coverage()` matrix gaps. | Reused verbatim at the statistical layer: Android scenarios act as pairing keys alongside `(model, effort, api_level)`, computing paired pass deltas, token cost multiples, and latency multiples across mobile agent harnesses. |
| **8. Dashboard UI & Assistant** (`server.py`, `bench/static/*`, `bench/assistant/*`) | 6-tab SPA (`LIVE`, `LAUNCH`, `DESIGNS`, `QUEUE`, `REPORT`, `ANALYSIS`), `IMAGES` prebake modal, SSE Opus 5.5 assistant with `DraftWriter`. | Identical backend SPA/SSE/assistant architecture: rename/adapt the `IMAGES` drawer to **EMULATORS / SNAPSHOTS** (showing AVD snapshot readiness, disk footprint, pull/bake/prune actions), extend `LIVE` and `REPORT` trial drawers with device serial telemetry, `logcat` tails, and screenshot filmstrips while keeping the Opus 5.5 assistant strictly read-only + `save_draft_design`. |
