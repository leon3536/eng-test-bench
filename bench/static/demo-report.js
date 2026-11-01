/* Demo mode for the REPORT area. Loaded after demo.js only with ?demo.

   Owned by the REPORT builder, with report.js and report.css. Brief:
   "Testing" asks demo mode for populated report comparisons (built here from
   the brief's Metric example), so the plots, intervals and verdict chips
   render. The fake ledger covers every state the tab draws:
   - anthropic:claude-opus-5-5: verdicts within, overlaps (the brief's Metric
     example), above and below, and a harness that reports no tokens. Every
     multiple is harness ÷ baseline, as the server gives it;
   - anthropic:claude-sonnet-5-5: a verdict withheld for unrecorded usage, and
     too few pairs for a verdict (low n);
   - openrouter:moonshotai/kimi-k3: published figures do not apply (n/a, with
     the estimate's position), low n, and too few pairs for any estimate;
   - fireworks:deepseek-v4.1-pro: a harness with no baseline trials at all.
   With compare 'models' the same fake trials are compared the other way:
   each harness held fixed, every model it ran ÷ its trials on the baseline
   model (options.baseline.model, else the profile's baseline_model), paired
   by task, with no verdicts. Some fake trials carry the ingest cost checks
   (flags), so the ledger's mark, the trial drawer and the caveats show them.

   Answered here: POST /api/report, /coverage, /pairs and /export, the report
   snapshots, and the ledger trials, trial detail, exclusions and runs that
   belong to the fake models. The ledger status, facets (with the fake models
   added), real trials and real exclusions still come from the server. The CI
   level and the model and harness filters change the fake report; the other
   options are echoed. Nothing here is billed. */

(() => {
  'use strict';

  if (!window.benchDemo) return;  // demo.js did not load
  const demo = window.benchDemo;
  const store = demo.store.report = {exclusions: [], snapshots: [], nextExclusion: 90001, nextSnapshot: 2};

  // config/reference.yaml as the server reads it: harness ÷ Claude Code, so a published "2x" is 1/2.
  const CONSENSUS = {low: 1 / 3, high: 1 / 2, source: 'S1-S5: changing harness moves cost 2x to 3x'};
  const PI_TOKENS = {value: 1 / 3.1, source: 'S10 via S9: 742k vs 236k tokens per task, same model (secondhand)'};
  const STRANDS_COST = {value: 1 / 2.1,
    source: 'S8: $0.74 to $1.57 per trial on Anthropic models (estimated, vendor study)'};
  const PASS_REF = {low: -5, high: 5, source: 'S1-S5: changing harness moves accuracy by under 5 points'};
  const NOT_APPLICABLE = 'Published figures were measured on Claude models; this is shown for position only.';
  const MODELS_NO_VERDICT = 'Published figures compare harnesses, not models, so model comparisons get no verdict.';
  const BASELINE = 'claude-code';
  const MIN_ESTIMATE = 3;
  const MIN_VERDICT = 8;
  const SD_LOG = 0.46;
  const Z90 = 1.6449;

  // The fake ledger. g / rom: [estimate, lo, hi] at 90%, harness ÷ baseline. pass: [harness rate, baseline rate,
  // Δ pp, lo, hi, detectable pp].
  const MODELS = [
    {provider: 'anthropic', model: 'claude-opus-5-5', applies: true, base: {tokens: 1.9e6, cost: 1.42, pass: 0.68, minutes: 16.5}, baseTasks: 16, arms: [
      {family: 'terminus-2', n: 12, setup: 0.04,
        tokens: {g: [0.48, 0.38, 0.59], rom: [0.45, 0.37, 0.56]},
        // The brief's illustrative Metric, as given.
        cost: {g: [0.41, 0.32, 0.53], rom: [0.38, 0.30, 0.49], ref: CONSENSUS, perTask: {min: 0.19, q1: 0.33, median: 0.42, q3: 0.53, max: 0.91},
          loo: [0.38, 0.44], sens: {crashes_excluded: 0.40, misconfigured_cap_included: null, same_run_only: 0.43, unrecorded_arm_high: null, unrecorded_baseline_high: null}, droppedArm: 1},
        pass: [0.58, 0.67, -8.3, -29.0, 12.4, 38]},
      {family: 'mini-swe-agent', n: 14, setup: 0.07, implementation: null, configs: 2,
        tokens: {g: [0.39, 0.34, 0.45], rom: [0.40, 0.35, 0.47]},
        cost: {g: [0.42, 0.37, 0.47], rom: [0.41, 0.37, 0.46], ref: CONSENSUS},
        pass: [0.62, 0.66, -3.6, -18.0, 11.0, 34]},
      {family: 'pi', n: 10, setup: 0.0, lowerBound: 1,
        tokens: {g: [0.22, 0.18, 0.26], rom: [0.20, 0.17, 0.24], ref: PI_TOKENS},
        cost: {g: [0.23, 0.20, 0.27], rom: [0.23, 0.20, 0.28], ref: CONSENSUS},
        pass: [0.70, 0.68, 2.0, -14.0, 18.0, 41]},
      {family: 'strands', n: 9, setup: 0.11,
        tokens: {notMeasured: true},
        cost: {g: [0.87, 0.74, 1.02], rom: [0.85, 0.74, 0.98], ref: STRANDS_COST},
        pass: [0.56, 0.67, -11.1, -35.0, 13.0, 44]},
    ]},
    {provider: 'anthropic', model: 'claude-sonnet-5-5', applies: true, base: {tokens: 1.4e6, cost: 0.61, pass: 0.61, minutes: 13.2}, baseTasks: 10, arms: [
      {family: 'strands', n: 9, setup: 0.09, unmeasured: 3,
        tokens: {g: [0.52, 0.38, 0.71], rom: [0.51, 0.37, 0.67]},
        cost: {g: [0.45, 0.33, 0.62], rom: [0.43, 0.32, 0.59], ref: CONSENSUS,
          withheld: "3 paired tasks rest on strands trials cut short or with no usage recorded; had those cost 2x strands's usual spend on their task, the verdict would be 'overlaps', not 'within'"},
        pass: [0.52, 0.61, -9.0, -33.0, 15.0, 44]},
      {family: 'pi', n: 5, setup: 0.0,
        tokens: {g: [0.30, 0.19, 0.48], rom: [0.29, 0.18, 0.45], ref: PI_TOKENS},
        cost: {g: [0.34, 0.21, 0.56], rom: [0.32, 0.20, 0.53], ref: CONSENSUS},
        pass: [0.60, 0.60, 0.0, -38.0, 38.0, 62]},
    ]},
    {provider: 'openrouter', model: 'moonshotai/kimi-k3', applies: false, base: {tokens: 2.3e6, cost: 0.31, pass: 0.52, minutes: 18.1}, baseTasks: 12, arms: [
      {family: 'pi', n: 9, setup: 0.0,
        tokens: {g: [0.29, 0.23, 0.38], rom: [0.28, 0.21, 0.36], ref: PI_TOKENS},
        cost: {g: [0.26, 0.20, 0.33], rom: [0.24, 0.19, 0.31], ref: CONSENSUS},
        pass: [0.55, 0.52, 3.3, -21.0, 27.0, 46]},
      {family: 'terminus-2', n: 5, setup: 0.12,
        tokens: {g: [0.63, 0.36, 1.11], rom: [0.59, 0.32, 1.25]},
        cost: {g: [0.43, 0.23, 0.83], rom: [0.40, 0.20, 0.91], ref: CONSENSUS},
        pass: [0.40, 0.60, -20.0, -55.0, 15.0, 62]},
      {family: 'mini-swe-agent', n: 1, setup: 0.5,
        tokens: {g: null}, cost: {g: null, ref: CONSENSUS}, pass: null},
    ]},
    {provider: 'fireworks', model: 'deepseek-v4.1-pro', applies: false, base: null, baseTasks: 0, arms: [
      {family: 'pi', n: 0, setup: 0.0, noBaseline: true, armTasks: 4, tokens: null, cost: null, pass: null},
    ]},
  ];
  const DEMO_MODELS = new Set(MODELS.map((m) => `${m.provider}:${m.model}`));

  // Helpers -------------------------------------------------------------------

  function rng(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function hash(text) {
    let h1 = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) h1 = Math.imul(h1 ^ text.charCodeAt(i), 16777619) >>> 0;
    return h1;
  }

  const hex = (seed, n) => {
    const r = rng(seed);
    let out = '';
    while (out.length < n) out += Math.floor(r() * 16).toString(16);
    return out;
  };
  const isoAgo = (ms) => new Date(Date.now() - ms).toISOString().replace('Z', '000Z');
  const round = (v, d = 4) => (v === null || v === undefined ? null : Math.round(v * 10 ** d) / 10 ** d);
  const clone = (x) => JSON.parse(JSON.stringify(x));
  const profile = () => (demo.config() && demo.config().profile) || 'default';
  const modelKey = (m) => `${m.provider}:${m.model}`;
  const groupKey = (m, family) => `fam:${m.provider}|${m.model}|${family}`;

  /** The inverse normal CDF (Acklam), for scaling the 90% intervals to another CI level. */
  function probit(p) {
    const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924];
    const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857];
    const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878];
    const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
    const lo = 0.02425;
    if (p < lo) {
      const q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p > 1 - lo) return -probit(1 - p);
    const q = p - 0.5;
    const r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  function tasks() {
    const cfg = demo.config();
    const list = (cfg && cfg.tasks) || [];
    return list.length >= 20 ? list.slice(0, 20) : [...list, ...Array.from({length: 20 - list.length}, (_, i) => `demo-task-${i + 1}`)];
  }

  // The world: groups, comparisons, pairs and trials, built once -----------------------

  let world = null;

  function getWorld() {
    if (!world) world = buildWorld();
    return world;
  }

  /** How a fake trial got its harness: most recent ones from a pre-baked image, older ones stock, a few fallbacks. */
  function setupOf(family, task, i, run) {
    const version = family === BASELINE ? '2.1.312' : family === 'pi' ? '1.4.0' : '2.0.0';
    const recipe = hex(hash(`${family}${version}recipe`), 64);
    const image = `benchbake/${family}-${version}-r${recipe.slice(0, 10)}/alexgshaw/${task}:20251031`;
    const mode = run === 0 ? (hash(`${family}${task}`) % 9 === 0 ? 'fallback' : 'prebaked') : 'stock';
    const stockSec = family === BASELINE ? 334.2 : 38.4;
    const seconds = mode === 'prebaked' ? (family === BASELINE ? 0.5 : family === 'pi' ? 19.6 : 2.8) : stockSec;
    const row = {mode, limit_sec: mode === 'prebaked' ? 360 : 720, image: mode === 'stock' ? null : image,
      image_id: mode === 'prebaked' ? `sha256:${hex(hash(image), 64)}` : null, recipe: mode === 'stock' ? null : recipe,
      baked_at: mode === 'prebaked' ? isoAgo(2 * 86400e3) : null,
      reason: mode === 'fallback' ? `no pre-baked image ${image}`
        : mode === 'prebaked' ? 'pre-baked image verified' : null};
    return {row, seconds};
  }

  /** Claude Code's own cost ÷ the computed cost: equal on Opus, low on Sonnet (as its cache-read price is). */
  const reportedFactor = (m) => (m.model === 'claude-sonnet-5-5' ? 0.93 : 1.002);

  /** The ingest cost checks (TrialRow.flags) a few fake trials carry, one of each kind the dashboard shows. */
  function flagsOf(row, m, i) {
    const flags = {};
    if (row.usage_status === 'unmeasured' || row.cost_usd === null) return flags;
    const opus = m.model === 'claude-opus-5-5';
    const nth = hash(`${row.family}${row.task}${m.model}${i}`) % 7;
    if (row.reported_cost_usd !== null && Math.abs(row.reported_cost_usd / row.cost_usd - 1) > 0.005) {
      flags.reported_cost_mismatch = {reported_usd: row.reported_cost_usd, computed_usd: row.cost_usd,
        diff_pct: round((row.reported_cost_usd / row.cost_usd - 1) * 100, 2)};
    }
    if (opus && row.family === BASELINE && nth === 0) {
      flags.declined_attempts = {attempts: 1, cost_usd_max: 0.02603, models: {[m.model]: {n_input: 17336,
        n_cache: 12635, n_output: 0, n_cache_write_5m: 4699, n_cache_write_1h: 0}}};
    }
    if (opus && row.family === 'mini-swe-agent' && nth === 1) {
      flags.unsaved_calls = {calls: 1, cost_usd_max: 0.0096, source: 'agent/mini-swe-agent.txt',
        bound: 'printed running cost'};
    }
    if (opus && row.family === 'pi' && nth === 2) {
      flags.unfinished_calls = {calls: 1, cost_usd_max: 0.0412, source: 'agent/pi/sessions',
        bound: 'costliest completed call'};
    }
    if (opus && row.family === 'terminus-2' && nth === 3) {
      flags.cache_writes_unknown = {calls: 3, models: [], reason: 'not in the harness log'};
    }
    return flags;
  }

  /** report.flag_caveats for one side: the flagged trials of a group, in the server's words. */
  function flagCaveats(name, rows) {
    const out = [];
    const hit = (flag) => rows.filter((r) => r.flags && r.flags[flag]);
    const lead = (list, one, many) => `${list.length} ${name} trial${list.length === 1 ? '' : 's'} `
      + `${list.length === 1 ? one : many}`;
    const usd = (v) => (v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`);
    const upTo = (list, flag) => ` (up to ${usd(list.reduce((t, r) => t + r.flags[flag].cost_usd_max, 0))} in all)`;
    const declined = hit('declined_attempts');
    if (declined.length) {
      out.push(`${lead(declined, 'has', 'have')} model attempts declined before another model answered (a refusal `
        + 'fallback); only the session log lists them and whether they are billed is unknown, so the cost leaves '
        + `them out${upTo(declined, 'declined_attempts')}.`);
    }
    const lost = [
      ['unsaved_calls', 'lost', 'model calls that completed but were never saved (a timeout during a command)'],
      ['unfinished_calls', 'started', 'model calls that never finished (a kill mid-call)'],
    ];
    for (const [flag, verb, what] of lost) {
      const list = hit(flag);
      if (list.length) {
        const calls = list.reduce((t, r) => t + r.flags[flag].calls, 0);
        out.push(`${lead(list, verb, verb)} ${what}: ${calls} call${calls === 1 ? '' : 's'} missing from the cost`
          + `${upTo(list, flag)}.`);
      }
    }
    const mismatch = hit('reported_cost_mismatch');
    if (mismatch.length) {
      const worst = mismatch.map((r) => r.flags.reported_cost_mismatch.diff_pct)
        .reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a), 0);
      out.push(`${lead(mismatch, 'reports', 'report')} a cost of its own more than 0.5% from the cost computed from `
        + `its tokens (largest ${worst > 0 ? '+' : ''}${worst.toFixed(1)}%): a call the log does not show, or a price `
        + 'that differs.');
    }
    const unknown = hit('cache_writes_unknown');
    if (unknown.length) {
      out.push(`${lead(unknown, 'has', 'have')} cache writes no harness log gave; they are priced as fresh input, `
        + 'so the cost is low by the write premium.');
    }
    return out;
  }

  function buildWorld() {
    const order = tasks();
    const prof = profile();
    const simulated = demo.runs().filter((r) => r.status !== 'running');
    const runIds = simulated.length ? simulated.slice(0, 3).map((r) => r.id) : ['20261027-101500-demo', '20261028-091200-demo'];
    // A fake trial borrows the simulated run's arm of its own harness, so its folder and View log match its family.
    const runArms = new Map(simulated.map((r) => [r.id, r.arms || []]));
    const armFor = (runId, family) => ((runArms.get(runId) || []).find((a) => (a.family || a.name) === family) || {}).name || `${family}-demo`;
    const trials = new Map();       // trial_key -> TrialRow
    const cells = {};               // group key -> task -> cell
    const pairsBy = {};             // comparison key -> {pairs, excluded}
    const groups = [];
    const comparisons = [];
    let nextId = 900001;

    const addTrial = (m, family, task, i, {outcome, tokens, cost, usage = 'full', run = 0}) => {
      const job = `${hex(hash(`${modelKey(m)}|${family}|${run}`), 8)}-${hex(hash(`${family}${run}`), 4)}-4${hex(hash(`${m.model}${run}`), 3)}-a${hex(hash(task), 3)}-${hex(hash(`${modelKey(m)}${family}`), 12)}`;
      const name = `${task}__${hex(hash(`${modelKey(m)}${family}${task}${i}`), 7)}`;
      const runId = runIds[run % runIds.length];
      const passed = outcome === 'passed';
      const total = usage === 'unmeasured' ? null : Math.round(tokens);
      const setup = setupOf(family, task, i, run);
      const row = {
        id: nextId++, trial_key: `${job}/${name}`, run_key: `${prof}/${runId}`, profile: prof, run_id: runId, source: 'bench',
        arm: armFor(runId, family), trial_name: name, attempt: i + 1, family, implementation: 'builtin',
        harness_version: family === BASELINE ? '2.1.312' : family === 'pi' ? '1.4.0' : '2.0.0', harbor_version: '0.25.0',
        provider: m.provider, model: m.model, effort: 'medium', effort_label: 'medium',
        config_hash: hex(hash(`${modelKey(m)}${family}cfg`), 16), config_label: `${family} · ${modelKey(m)} · effort=medium`,
        options: {max_turns: 150}, cautions: [], dataset: 'terminal-bench@2.0', dataset_snapshot: 'local:terminal-bench@2.0@2026-10-29T05:29:58+00:00',
        task, task_digest_short: hex(hash(`${task}digest`), 12), outcome,
        outcome_reason: passed ? 'reward 1' : outcome === 'failed' ? 'reward 0'
          : outcome === 'setup_failure' ? 'agent never started' : outcome === 'agent_limit' ? `AgentTimeoutError: the agent ran out of its budget${usage === 'unmeasured' ? '; no usage recorded' : ''}` : 'killed before finishing',
        harbor_cause: outcome === 'agent_limit' ? 'agent_exhausted_budget' : null,
        exception_type: outcome === 'agent_limit' ? 'AgentTimeoutError' : null,
        exception_head: outcome === 'agent_limit' ? 'Agent execution timed out after 900.0 seconds' : null,
        reward: outcome === 'setup_failure' ? null : passed ? 1 : 0, passed: outcome === 'setup_failure' ? null : passed,
        n_input: total === null ? 0 : Math.round(total * 0.97), n_cache: total === null ? 0 : Math.round(total * 0.82), n_output: total === null ? 0 : Math.round(total * 0.03),
        usage_status: usage, tokens_total: total, cost_usd: usage === 'unmeasured' ? null : round(cost),
        cost_at_ingest_usd: usage === 'unmeasured' ? null : round(cost),
        reported_cost_usd: family === BASELINE && usage !== 'unmeasured' ? round(cost * reportedFactor(m)) : null,
        unpriced_models: [], primary_share: 1.0, started_at: isoAgo((40 - i) * 3600e3), finished_at: isoAgo((40 - i) * 3600e3 - 14 * 60e3),
        phases: {env_setup_sec: 3.1, agent_setup_sec: setup.seconds,
          agent_exec_sec: outcome === 'agent_limit' ? 900.2 : 640.5, verifier_sec: 22.7},
        setup: setup.row, setup_mode: setup.row.mode, flags: {},
        total_sec: outcome === 'agent_limit' ? 964.4 : 704.7, exclusion: null,
        log_url: runArms.has(runId) ? `/api/runs/${runId}/log/${armFor(runId, family)}` : null,
      };
      row.flags = flagsOf(row, m, i);
      trials.set(row.trial_key, row);
      return row;
    };

    const addCell = (gk, task, rows) => {
      const by = {};
      for (const r of rows) by[r.outcome] = (by[r.outcome] || 0) + 1;
      const valid = rows.filter((r) => r.outcome === 'passed' || r.outcome === 'failed' || r.outcome === 'agent_limit').length;
      (cells[gk] = cells[gk] || {})[task] = {valid, passed: rows.filter((r) => r.outcome === 'passed').length, by_outcome: by};
    };

    MODELS.forEach((m, mi) => {
      const r = rng(1000 + mi);
      const baseKey = groupKey(m, BASELINE);
      const armSpecs = m.arms;
      // Baseline trials: one or two per task over its tasks; their tokens set each task's scale.
      const scale = order.map(() => 0.5 + r() * 1.1);
      const baseRows = {};
      if (m.base) {
        for (let ti = 0; ti < m.baseTasks; ti++) {
          const task = order[ti];
          const n = ti % 4 === 0 ? 2 : 1;
          baseRows[task] = Array.from({length: n}, (_, i) => addTrial(m, BASELINE, task, i, {
            outcome: ti === 4 && i === 1 ? 'misconfigured_cap' : r() < m.base.pass ? 'passed' : ti === 3 && i === 0 ? 'agent_limit' : 'failed',
            tokens: m.base.tokens * scale[ti], cost: m.base.cost * scale[ti], run: ti % 3,
          }));
          addCell(baseKey, task, baseRows[task]);
        }
        groups.push(group(m, BASELINE, true, Object.values(baseRows).flat(), m.base, {configs: 2}));
      }
      armSpecs.forEach((spec, ai) => {
        const gk = groupKey(m, spec.family);
        const ratioT = spec.tokens && spec.tokens.g ? spec.tokens.g[0] : 0.5;  // harness ÷ baseline
        const ratioC = spec.cost && spec.cost.g ? spec.cost.g[0] : 0.5;
        const armRows = {};
        const pairs = [];
        const excluded = [];
        const nTasks = spec.noBaseline ? spec.armTasks : spec.n;
        for (let ti = 0; ti < nTasks; ti++) {
          const task = order[ti];
          const jitter = () => Math.exp(SD_LOG * (r() * 2 - 1) * 1.2);
          const rows = [addTrial(m, spec.family, task, 0, {
            outcome: r() < (spec.pass ? spec.pass[0] : 0.5) ? 'passed' : 'failed',
            tokens: (m.base ? m.base.tokens : 1.1e6) * scale[ti] * ratioT * jitter(),
            cost: (m.base ? m.base.cost : 0.2) * scale[ti] * ratioC * jitter(), run: (ti + ai) % 3,
            usage: spec.lowerBound && ti === 2 ? 'lower_bound' : 'full',
          })];
          if (spec.unmeasured && ti < spec.unmeasured) {
            const lost = addTrial(m, spec.family, task, 1, {outcome: 'agent_limit', tokens: 0, cost: 0, usage: 'unmeasured', run: 1});
            rows.push(lost);
            excluded.push({trial_key: lost.trial_key, side: 'arm', task, outcome: 'agent_limit', reason: 'no usage recorded'});
          }
          armRows[task] = rows;
          addCell(gk, task, rows);
          const base = (baseRows[task] || []).filter((x) => x.outcome !== 'misconfigured_cap');  // left out by the default policy
          if (base.length && !spec.noBaseline) {
            const capped = baseRows[task].find((x) => x.outcome === 'misconfigured_cap');
            if (capped) {
              excluded.push({trial_key: capped.trial_key, side: 'baseline', task, outcome: 'misconfigured_cap',
                reason: "misconfigured_cap excluded by the report policy (stopped by its own max_budget_usd, which it prices at another provider's rates)"});
            }
            const mean = (list, k) => {
              const vals = list.map((x) => x[k]).filter((v) => typeof v === 'number');
              return vals.length ? vals.reduce((t, v) => t + v, 0) / vals.length : null;
            };
            const counted = rows.filter((x) => x.usage_status !== 'unmeasured');
            const at = mean(counted, 'tokens_total');
            const ac = mean(counted, 'cost_usd');
            const bt = mean(base, 'tokens_total');
            const bc = mean(base, 'cost_usd');
            pairs.push({
              task, digest_short: rows[0].task_digest_short, run_key: rows[0].run_key === base[0].run_key ? rows[0].run_key : null,
              same_run: rows[0].run_key === base[0].run_key,
              arm: {trials: rows.length, valid: rows.filter((x) => x.outcome !== 'setup_failure').length,
                tokens_total: spec.tokens && spec.tokens.notMeasured ? null : round(at, 0), cost: round(ac),
                pass: rows.filter((x) => x.passed).length / rows.length, trial_keys: rows.map((x) => x.trial_key)},
              baseline: {trials: base.length, valid: base.length, tokens_total: round(bt, 0), cost: round(bc),
                pass: base.filter((x) => x.passed).length / base.length, trial_keys: base.map((x) => x.trial_key)},
              ratio_tokens: spec.tokens && spec.tokens.notMeasured ? null : round(at / bt, 3), ratio_cost: round(ac / bc, 3),
            });
          }
        }
        // A setup failure on the next task: the harness side of that pair is missing.
        if (m.base && spec.n < m.baseTasks && spec.setup > 0) {
          const task = order[spec.n];
          const lost = addTrial(m, spec.family, task, 0, {outcome: 'setup_failure', tokens: 0, cost: 0, run: 2});
          lost.tokens_total = null;
          lost.cost_usd = null;
          armRows[task] = [lost];
          addCell(gk, task, [lost]);
          excluded.push({trial_key: lost.trial_key, side: 'arm', task, outcome: 'setup_failure', reason: 'setup_failure: agent never started'});
        }
        // pi on the first model ran one task the baseline never did.
        if (mi === 0 && spec.family === 'pi') {
          const task = order[m.baseTasks];
          armRows[task] = [addTrial(m, spec.family, task, 0, {outcome: 'passed', tokens: 4e5, cost: 0.3, run: 0})];
          addCell(gk, task, armRows[task]);
        }
        const rowsAll = Object.values(armRows).flat();
        groups.push(group(m, spec.family, false, rowsAll, m.base ? {
          tokens: m.base.tokens * ratioT, cost: m.base.cost * ratioC, pass: spec.pass ? spec.pass[0] : 0.4, minutes: m.base.minutes * 0.8,
        } : {tokens: 1.1e6, cost: 0.18, pass: 0.45, minutes: 12}, spec));
        if (!spec.noBaseline) {
          const key = `${gk}~${baseKey}`;
          pairsBy[key] = {pairs, excluded};
          comparisons.push({m, spec, key, group: gk, baseline_group: baseKey, armOnly: mi === 0 && spec.family === 'pi' ? [order[m.baseTasks]] : []});
        }
      });
    });
    return {order, trials, cells, pairsBy, groups, comparisons, runIds};
  }

  function wilson(k, n, z = Z90) {
    if (!n) return {estimate: null, lo: null, hi: null};
    const p = k / n;
    const d = 1 + z * z / n;
    const c = (p + z * z / (2 * n)) / d;
    const w = (z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d;
    return {estimate: round(p), lo: round(Math.max(0, c - w)), hi: round(Math.min(1, c + w))};
  }

  function group(m, family, isBaseline, rows, base, spec = {}) {
    const by = {};
    for (const r of rows) by[r.outcome] = (by[r.outcome] || 0) + 1;
    const counted = rows.filter((r) => r.outcome !== 'setup_failure');
    const measured = counted.filter((r) => r.usage_status !== 'unmeasured');
    const taskSet = [...new Set(counted.map((r) => r.task))];
    const passedTasks = taskSet.filter((t) => counted.some((r) => r.task === t && r.passed)).length;
    const setups = rows.filter((r) => r.outcome === 'setup_failure').length;
    const nConfigs = spec.configs || 1;
    const configs = Array.from({length: nConfigs}, (_, i) => ({
      config_hash: hex(hash(`${modelKey(m)}${family}cfg${i ? i : ''}`), 16),
      label: `${family} · ${modelKey(m)} · effort=medium${i ? ' · max_budget_usd=1' : ''}${family === 'mini-swe-agent' && i ? ' · implementation=bench.agents.mini_swe_agent:MiniSweAgentWithMcp' : ''}`,
      trials: i ? Math.max(1, Math.floor(rows.length / 3)) : rows.length - (nConfigs > 1 ? Math.max(1, Math.floor(rows.length / 3)) : 0),
    }));
    return {
      key: groupKey(m, family), label: `${family} · ${modelKey(m)}`, name: family, provider: m.provider, model: m.model, family,
      implementation: spec.implementation === null ? null : 'builtin', config_hash: null,
      harness_version: family === BASELINE ? '2.1.312' : null, is_baseline: isBaseline,
      configs, config_diff: nConfigs > 1 ? (family === 'mini-swe-agent' ? {implementation: ['builtin', 'bench.agents.mini_swe_agent:MiniSweAgentWithMcp']} : {max_budget_usd: [null, '1']}) : {},
      harness_versions: family === BASELINE ? {'2.1.312': rows.length} : family === 'pi' ? {'1.4.0': rows.length} : {'2.0.0': rows.length},
      runs: new Set(rows.map((r) => r.run_key)).size, tasks: taskSet.length,
      trials: {total: rows.length, by_outcome: by, valid_tokens: measured.length, valid_cost: measured.length, valid_reward: counted.length,
        included: counted.length, usage_unmeasured: counted.length - measured.length, usage_lower_bound: rows.filter((r) => r.usage_status === 'lower_bound').length},
      pass_rate: taskSet.length ? {...wilson(passedTasks, taskSet.length), tasks: taskSet.length, method: 'wilson_tasks'} : null,
      setup_failure_rate: {...wilson(setups, rows.length), trials: rows.length, method: 'wilson_trials'},
      per_trial: {
        tokens_total: {mean: round(base.tokens, 0), median: round(base.tokens * 0.93, 0), n: measured.length},
        tokens_uncached: {mean: round(base.tokens * 0.16, 0), median: round(base.tokens * 0.14, 0), n: measured.length},
        cost: {mean: round(base.cost), median: round(base.cost * 0.91), n: measured.length},
        minutes: {median: round(base.minutes, 2), n: counted.length},
      },
      reported_vs_computed_cost: family === BASELINE ? {median_ratio: reportedFactor(m), n: measured.length}
        : {median_ratio: null, n: 0},
    };
  }

  // The report for one request ---------------------------------------------------

  const DEFAULT_OPTIONS = {
    group_by: 'family', split_harness_versions: false, baseline: {family: BASELINE, config: null, model: null},
    compare: 'harnesses',
    pairing: {match_on: 'digest', same_run_only: false}, policy: {crashes: 'include', misconfigured_cap: 'exclude', exclusions: true},
    pricing: 'current', ci_level: 0.9, bootstrap: {resamples: 2000, seed: 20261028}, min_pairs: {estimate: MIN_ESTIMATE, verdict: MIN_VERDICT},
    coverage_target: {ratio_halfwidth: 1.25, pass_delta_pp: 5.0, prior_sd_log: 0.6}, include_pairs: false, setup: 'any',
  };
  const FILTER_KEYS = ['profiles', 'sources', 'models', 'families', 'implementations', 'configs', 'exclude_configs', 'harness_versions',
    'harbor_versions', 'efforts', 'datasets', 'snapshots', 'tasks', 'runs', 'exclude_runs', 'setup_modes'];

  function normalise(body) {
    const f = (body && body.filters) || {};
    const filters = {date_from: f.date_from || null, date_to: f.date_to || null};
    for (const k of FILTER_KEYS) filters[k] = Array.isArray(f[k]) ? f[k] : [];
    filters.include_cautioned = f.include_cautioned === true;
    const options = clone(DEFAULT_OPTIONS);
    const o = (body && body.options) || {};
    for (const [k, v] of Object.entries(o)) {
      if (v === null || v === undefined) continue;
      if (typeof v === 'object' && !Array.isArray(v) && options[k] && typeof options[k] === 'object') {
        for (const [k2, v2] of Object.entries(v)) if (v2 !== null && v2 !== undefined) options[k][k2] = v2;
      } else {
        options[k] = v;
      }
    }
    if (!options.baseline.model) {
      options.baseline.model = (demo.config() && demo.config().baseline_model) || 'anthropic:claude-opus-5-5';
    }
    return {filters, options};
  }

  const effective = () => {
    const latest = new Map();
    for (const e of store.exclusions) latest.set(`${e.target_type}|${e.target_id}`, e);
    return latest;
  };

  /** The exclusion in force for a fake trial (its own, else its run's), as TrialRow.exclusion shows it. */
  function exclusionOf(row) {
    const latest = effective();
    const own = latest.get(`trial|${row.trial_key}`);
    const run = latest.get(`run|${row.run_key}`);
    const e = own || run;
    return e ? {target_type: e.target_type, action: e.action, reason: e.reason, created_at: e.created_at} : null;
  }

  const manual = (row) => {
    const e = exclusionOf(row);
    return Boolean(e && e.action === 'exclude');
  };

  function verdictFor(ref, rom, status, applies) {
    const band = ref.value ? [round(ref.value / 1.5, 3), round(ref.value * 1.5, 3)] : [ref.low, ref.high];
    const est = rom ? rom[0] : null;
    const position = est === null ? null : est < band[0] ? 'below' : est > band[1] ? 'above' : 'inside';
    let verdict = 'insufficient';
    if (!applies) verdict = 'not_applicable';
    else if (status === 'ok' && rom) {
      const [, lo, hi] = rom;
      verdict = lo >= band[0] && hi <= band[1] ? 'within' : hi < band[0] ? 'below' : lo > band[1] ? 'above' : 'overlaps';
    }
    return {published: clone(ref), band, estimator: 'ratio_of_means', applies, position: applies || est !== null ? position : null, verdict, withheld: null};
  }

  function metric(spec, n, applies, scale, kind) {
    const dropped = {no_arm_valid: 0, no_baseline_valid: 0, usage_not_recorded: 0, tokens_not_reported: 0, unpriced: 0, zero_value: 0};
    if (spec && spec.notMeasured) {
      return {status: 'not_measured', n: 0, need: 0, geomean: null, ratio_of_means: null, per_task: null, leave_one_out: null,
        sensitivity: {crashes_excluded: null, misconfigured_cap_included: null, same_run_only: null, unrecorded_arm_high: null, unrecorded_baseline_high: null},
        robust: null, reference: null,
        dropped: {...dropped, tokens_not_reported: n}};
    }
    const status = n < MIN_ESTIMATE || !spec || !spec.g ? 'insufficient' : n < MIN_VERDICT ? 'low_n' : 'ok';
    const need = status === 'insufficient' ? MIN_ESTIMATE - n : status === 'low_n' ? MIN_VERDICT - n : 0;
    const widen = ([e, lo, hi]) => [e, round(e * (lo / e) ** scale, 3), round(e * (hi / e) ** scale, 3)];
    const g = status === 'insufficient' ? null : widen(spec.g);
    const rom = status === 'insufficient' ? null : widen(spec.rom);
    let reference = spec && spec.ref ? verdictFor(spec.ref, rom, status, applies) : null;
    if (reference && spec.withheld && applies) reference = {...reference, verdict: 'insufficient', withheld: spec.withheld};
    if (kind === 'tokens_uncached') reference = null;
    return {
      status, n, need,
      geomean: g && {estimate: g[0], lo: g[1], hi: g[2], sd_log: SD_LOG, gsd: round(Math.exp(SD_LOG), 3), method: 't_on_log_ratios'},
      ratio_of_means: rom && {estimate: rom[0], lo: rom[1], hi: rom[2], method: 'task_bootstrap'},
      per_task: g && (spec.perTask || {min: round(g[0] / 2.2, 2), q1: round(g[0] / 1.3, 2), median: g[0], q3: round(g[0] * 1.25, 2), max: round(g[0] * 2.1, 2)}),
      leave_one_out: g && (spec.loo || [round(g[0] * 0.95, 2), round(g[0] * 1.06, 2)]),
      sensitivity: g ? (spec.sens || {crashes_excluded: round(g[0] * 1.02, 2), misconfigured_cap_included: null, same_run_only: round(g[0] * 0.97, 2),
        unrecorded_arm_high: spec.withheld ? round(g[0] / 0.78, 2) : null, unrecorded_baseline_high: null})
        : {crashes_excluded: null, misconfigured_cap_included: null, same_run_only: null, unrecorded_arm_high: null, unrecorded_baseline_high: null},
      robust: status === 'ok' ? true : status === 'low_n' ? false : null,
      reference,
      dropped: {...dropped, no_arm_valid: spec && spec.droppedArm ? spec.droppedArm : 0},
    };
  }

  function passBlock(spec, n, applies, scale) {
    const ref = {published: clone(PASS_REF), band: [-5, 5], applies, verdict: applies ? 'insufficient' : 'not_applicable'};
    if (!spec || n < MIN_ESTIMATE) {
      return {status: 'insufficient', n, need: MIN_ESTIMATE - n, arm_rate: null, baseline_rate: null,
        delta_pp: {estimate: null, lo: null, hi: null, method: null}, discordant: null, mcnemar_p: null, mde_pp: 82.2, reference: ref};
    }
    const [arm, base, d, lo, hi, mde] = spec;
    const status = n < MIN_VERDICT ? 'low_n' : 'ok';
    const l = round(d - (d - lo) * scale, 1);
    const h2 = round(d + (hi - d) * scale, 1);
    if (applies && status === 'ok') ref.verdict = l >= -5 && h2 <= 5 ? 'within' : h2 < -5 ? 'below' : l > 5 ? 'above' : 'overlaps';
    return {status, n, need: status === 'low_n' ? MIN_VERDICT - n : 0, arm_rate: arm, baseline_rate: base,
      delta_pp: {estimate: d, lo: l, hi: h2, method: 'newcombe_paired'},
      discordant: {arm_only: Math.max(0, Math.round(n * 0.2 + d / 10)), baseline_only: Math.max(0, Math.round(n * 0.2 - d / 10))},
      mcnemar_p: round(Math.min(1, 0.4 + Math.abs(10 / (d || 1)) / 10), 3), mde_pp: mde, reference: ref};
  }

  function coverageFor(c, n, w, armGroup, baseGroup, target) {
    const z = Z90;
    const needed = Math.ceil((z * SD_LOG / Math.log(target)) ** 2);
    const pairedTasks = new Set((w.pairsBy[c.key] || {pairs: []}).pairs.map((p) => p.task));
    const more = Math.max(0, needed - n);
    const next = w.order.filter((t) => !pairedTasks.has(t)).slice(0, Math.min(more, w.order.length)).map((task) => {
      const armHas = Boolean((w.cells[c.group] || {})[task] && w.cells[c.group][task].valid);
      const baseHas = Boolean((w.cells[c.baseline_group] || {})[task] && w.cells[c.baseline_group][task].valid);
      return {task, side: armHas ? 'baseline' : baseHas ? 'arm' : 'both', reason: armHas || baseHas ? 'complete pair' : 'new task'};
    });
    const armCost = armGroup.per_trial.cost.mean;
    const baseCost = baseGroup.per_trial.cost.mean;
    const capped = needed > w.order.length;
    return {
      metric: 'cost', pairs: n, need_for_estimate: Math.max(0, MIN_ESTIMATE - n), need_for_verdict: Math.max(0, MIN_VERDICT - n),
      halfwidth_factor_now: n >= MIN_ESTIMATE ? round(Math.exp(z * SD_LOG / Math.sqrt(n)), 3) : null,
      target_factor: target, sd_log_used: SD_LOG, sd_source: n >= 3 ? 'observed' : 'prior',
      est_pairs_needed: needed, est_more_pairs: more, pass_tasks_needed: 271,
      next, est_cost_usd: round(more * (armCost + baseCost), 2),
      per_trial: {arm: {mean_cost_usd: armCost, median_minutes: armGroup.per_trial.minutes.median, setup_failure_rate: armGroup.setup_failure_rate.estimate},
        baseline: {mean_cost_usd: baseCost, median_minutes: baseGroup.per_trial.minutes.median, setup_failure_rate: baseGroup.setup_failure_rate.estimate}},
      capped_at_sample: capped,
      note: capped ? `Reaching x/÷${target} needs about ${needed} paired tasks, more than the ${w.order.length} in the sample.` : null,
    };
  }

  function sameRun(key, n) {
    const pairs = ((getWorld().pairsBy[key] || {}).pairs || []).slice(0, n);
    return pairs.length ? round(pairs.filter((p) => p.same_run).length / pairs.length, 3) : null;
  }

  /** variant 'then' is the seeded snapshot: fewer pairs, wider intervals, a little lower. */
  function buildReport(body, {variant = 'now'} = {}) {
    const w = getWorld();
    const spec = normalise(body);
    if (spec.options.compare === 'models') return buildModelsReport(spec, variant);
    const ci = Math.min(0.99, Math.max(0.8, Number(spec.options.ci_level) || 0.9));
    const scale = (probit((1 + ci) / 2) / Z90) * (variant === 'then' ? 1.35 : 1);
    const f = spec.filters;
    const keepModel = (m) => !f.models.length || f.models.includes(modelKey(m));
    const keepFamily = (family) => !f.families.length || family === BASELINE || f.families.includes(family);
    const groups = clone(w.groups.filter((g) => keepModel(g) && keepFamily(g.family)));
    const byKey = new Map(groups.map((g) => [g.key, g]));
    const target = spec.options.coverage_target.ratio_halfwidth || 1.25;
    const comparisons = w.comparisons.filter((c) => keepModel(c.m) && keepFamily(c.spec.family)).map((c) => {
      const m = c.m;
      const s = c.spec;
      const n = variant === 'then' ? Math.max(1, s.n - 4) : s.n;
      const shift = (block) => (variant === 'then' && block && block.g ? {...block, g: block.g.map((v) => round(v * 0.93, 2)), rom: block.rom.map((v) => round(v * 0.93, 2))} : block);
      const metrics = {
        tokens_total: metric(shift(s.tokens), n, m.applies, scale, 'tokens_total'),
        tokens_uncached: metric(s.tokens && s.tokens.notMeasured ? s.tokens : shift(s.tokens && s.tokens.g ? {g: s.tokens.g.map((v) => round(v * 1.2, 2)), rom: s.tokens.rom.map((v) => round(v * 1.2, 2))} : null), n, m.applies, scale, 'tokens_uncached'),
        cost: metric(shift(s.cost), n, m.applies, scale, 'cost'),
      };
      const pass = passBlock(s.pass, n, m.applies, scale);
      const caveats = [];
      if (s.implementation === null) caveats.push(`${s.family} pools configurations that differ in implementation.`);
      caveats.push(`${BASELINE} pools configurations that differ in max_budget_usd.`);
      if (s.unmeasured) caveats.push(`No usage was recorded for ${s.unmeasured} of ${n + s.unmeasured} ${s.family} trials and 0 of ${n} ${BASELINE} trials; they count for pass rate but are left out of tokens and cost. Such trials often ran to a time limit, so the means may understate ${s.family}.`);
      if (s.lowerBound) caveats.push(`${s.lowerBound} ${s.family} trial hit the time limit; their usage covers completed model calls only, so their tokens and cost are lower bounds.`);
      if (metrics.cost.reference && metrics.cost.reference.withheld) caveats.push(`No cost verdict: ${metrics.cost.reference.withheld}.`);
      if (s.tokens && s.tokens.notMeasured) caveats.push(`${s.family} reports no token counts, so only its cost can be compared.`);
      if (!m.applies) caveats.push(NOT_APPLICABLE);
      caveats.push(...flagCaveats(`${s.family} on ${m.model}`, rowsOf(w, c.group)),
        ...flagCaveats(`${BASELINE} on ${m.model}`, rowsOf(w, c.baseline_group)));
      const armGroup = byKey.get(c.group);
      const baseGroup = byKey.get(c.baseline_group);
      const cov = armGroup && baseGroup ? coverageFor(c, n, w, armGroup, baseGroup, target) : null;
      if (cov && cov.note) caveats.push(cov.note);
      const gm = metrics.cost.geomean;
      const head = gm
        ? `${s.family} used ${gm.estimate.toFixed(2)}x the cost of ${BASELINE} `
          + `(${Math.round(ci * 100)}% CI ${gm.lo.toFixed(2)}-${gm.hi.toFixed(2)}x)`
          + `${metrics.tokens_total.geomean ? `, ${metrics.tokens_total.geomean.estimate.toFixed(2)}x the tokens` : ''}`
        : `${MIN_ESTIMATE - n} more for an estimate`;
      const withheld = metrics.cost.reference && metrics.cost.reference.withheld;
      return {
        key: c.key, provider: m.provider, model: m.model, group: c.group, baseline_group: c.baseline_group, arm_name: s.family, baseline_name: BASELINE,
        headline: `${s.family} vs ${BASELINE} on ${modelKey(m)}: ${n} paired task${n === 1 ? '' : 's'}; ${head}${metrics.cost.status === 'low_n' ? ' (low n)' : ''}${withheld ? `; verdict withheld: ${withheld}` : ''}`,
        pairs: {n_tasks: n, same_run_fraction: sameRun(c.key, n), arm_only_tasks: c.armOnly, baseline_only_tasks: [],
          dropped: {no_arm_valid: s.setup > 0 ? 1 : 0, no_baseline_valid: 0, usage_not_recorded: 0, tokens_not_reported: s.tokens && s.tokens.notMeasured ? n : 0, unpriced: 0, zero_value: 0}},
        metrics, pass, coverage: cov, caveats,
      };
    });
    // The setup option: stock counts fallbacks too; a trial with no recorded setup only under 'any'.
    const setup = spec.options.setup;
    const keepSetup = (r) => setup === 'any'
      || (setup === 'prebaked' ? r.setup_mode === 'prebaked' : ['stock', 'fallback'].includes(r.setup_mode));
    const keepModes = (r) => !f.setup_modes.length || f.setup_modes.includes(r.setup_mode);
    const trials = [...w.trials.values()]
      .filter((r) => keepModel(r) && keepFamily(r.family) && keepSetup(r) && keepModes(r));
    for (const g of groups) {
      g.setup_modes = {};
      for (const r of trials) {
        if (r.family === g.family && `${r.provider}:${r.model}` === `${g.provider}:${g.model}`) {
          g.setup_modes[r.setup_mode || 'unknown'] = (g.setup_modes[r.setup_mode || 'unknown'] || 0) + 1;
        }
      }
      const pt = g.per_trial || {};
      if (pt.minutes) {
        const median = round(pt.minutes.median - (g.family === BASELINE ? 1.2 : 0.4), 2);
        pt.minutes_ex_setup = {median, n: pt.minutes.n};
      }
    }
    const manualOut = spec.options.policy.exclusions ? trials.filter(manual).length : 0;
    const used = trials.length - manualOut;
    const byOutcome = {};
    for (const r of trials) if (!manual(r) || !spec.options.policy.exclusions) byOutcome[r.outcome] = (byOutcome[r.outcome] || 0) + 1;
    const fp = `sha256:${hex(hash(JSON.stringify(spec) + variant + store.exclusions.length), 16)}`;
    return {
      spec,
      provenance: {generated_at: new Date().toISOString().replace(/\.\d+Z$/, 'Z'), fingerprint: fp, trials_used: used,
        exclusions_applied: effective().size, pricing: spec.options.pricing, pricing_sha: '2437ed4f51bf', parser_versions: [2],
        classifier_version: 2, stats_version: 4, harbor_versions: ['0.25.0'],
        includes_active_run: variant === 'now' && Boolean(demo.runs().some((r) => r.status === 'running')),
        direction: 'harness_over_baseline'},
      scope: scopeOf(f),
      totals: {trials: used, runs: w.runIds.length, profiles: [profile()], by_outcome: byOutcome, excluded_manual: manualOut,
        spend_usd: round(trials.reduce((t, r) => t + (r.cost_usd || 0), 0), 4), usage_unmeasured: trials.filter((r) => r.usage_status === 'unmeasured').length},
      groups, comparisons,
      coverage: {task_order: w.order, groups: groups.map((g) => g.key), cells: Object.fromEntries(groups.map((g) => [g.key, clone(w.cells[g.key] || {})]))},
      warnings: [],
    };
  }

  // compare 'models' ------------------------------------------------------------------

  const modelPairsBy = {};   // comparison key -> {pairs, excluded}, for /api/report/pairs

  const counts = (r) => ['passed', 'failed', 'agent_limit'].includes(r.outcome);
  const mean = (list) => (list.length ? list.reduce((t, v) => t + v, 0) / list.length : null);
  const rowsOf = (w, gk) => [...w.trials.values()].filter((r) => groupKey(r, r.family) === gk);

  /** The paired tasks of two fake groups: per task, the means of each side's counted trials. */
  function pairGroups(w, armKey, baseKey) {
    const byTask = (gk) => {
      const out = new Map();
      for (const r of rowsOf(w, gk).filter(counts)) {
        if (!out.has(r.task)) out.set(r.task, []);
        out.get(r.task).push(r);
      }
      return out;
    };
    const arm = byTask(armKey);
    const base = byTask(baseKey);
    const side = (rows) => {
      const measured = rows.filter((r) => r.usage_status !== 'unmeasured');
      return {trials: rows.length, valid: rows.length,
        tokens_total: round(mean(measured.map((r) => r.tokens_total)), 0),
        cost: round(mean(measured.map((r) => r.cost_usd))), pass: rows.filter((r) => r.passed).length / rows.length,
        trial_keys: rows.map((r) => r.trial_key)};
    };
    return w.order.filter((t) => arm.has(t) && base.has(t)).map((task) => {
      const a = side(arm.get(task));
      const b = side(base.get(task));
      const first = arm.get(task)[0];
      const same = first.run_key === base.get(task)[0].run_key;
      return {task, digest_short: first.task_digest_short, run_key: same ? first.run_key : null, same_run: same,
        arm: a, baseline: b,
        ratio_tokens: a.tokens_total && b.tokens_total ? round(a.tokens_total / b.tokens_total, 3) : null,
        ratio_cost: a.cost && b.cost ? round(a.cost / b.cost, 3) : null};
    });
  }

  /** A multiple from per-task ratios, in metric()'s shape, with no published reference. */
  function pairedMetric(pairs, key, armKey, baseKey, scale, kind) {
    const ok = pairs.filter((p) => isFinite(p[key]) && p[key] > 0);
    if (ok.length < MIN_ESTIMATE) return metric(null, ok.length, false, scale, kind);
    const logs = ok.map((p) => Math.log(p[key]));
    const mu = mean(logs);
    const sd = Math.sqrt(logs.reduce((t, v) => t + (v - mu) ** 2, 0) / Math.max(1, logs.length - 1)) || SD_LOG;
    const half = Z90 * sd / Math.sqrt(ok.length);
    const g = [round(Math.exp(mu), 3), round(Math.exp(mu - half), 3), round(Math.exp(mu + half), 3)];
    const ratio = ok.reduce((t, p) => t + p.arm[armKey], 0) / ok.reduce((t, p) => t + p.baseline[baseKey], 0);
    const rom = [round(ratio, 3), round(ratio * Math.exp(-half * 1.1), 3), round(ratio * Math.exp(half * 1.1), 3)];
    return metric({g, rom}, ok.length, false, scale, kind);
  }

  /** Pass rate on the paired tasks, model − baseline model, in passBlock()'s shape, with no reference. */
  function pairedPass(pairs, scale) {
    const n = pairs.length;
    if (!n) return {...passBlock(null, 0, false, scale), reference: null};
    const armRate = mean(pairs.map((p) => p.arm.pass));
    const baseRate = mean(pairs.map((p) => p.baseline.pass));
    const d = round((armRate - baseRate) * 100, 1);
    const half = Math.min(60, 100 * Z90 * Math.sqrt(0.5 / n));
    const spec = [round(armRate, 3), round(baseRate, 3), d, d - half, d + half, round(2.8 * half / Z90, 0)];
    return {...passBlock(spec, n, false, scale), reference: null};
  }

  /** One model comparison: the group g against its harness's group on the baseline model. */
  function modelComparison(w, g, base, spec, ci, scale) {
    const key = `${g.key}~${base.key}`;
    const pairs = pairGroups(w, g.key, base.key);
    modelPairsBy[key] = {pairs, excluded: []};
    const tokens = (kind) => pairedMetric(pairs, 'ratio_tokens', 'tokens_total', 'tokens_total', scale, kind);
    const metrics = {tokens_total: tokens('tokens_total'), tokens_uncached: tokens('tokens_uncached'),
      cost: pairedMetric(pairs, 'ratio_cost', 'cost', 'cost', scale, 'cost')};
    const n = pairs.length;
    const who = `${g.model} vs ${base.model} with ${g.family}`;
    const gm = metrics.cost.geomean;
    const settle = metrics.cost.status === 'low_n' ? `; ${metrics.cost.need} more for a settled figure` : '';
    const headline = gm
      ? `${who}: ${g.model} used ${gm.estimate.toFixed(2)}x the cost of ${base.model} (${Math.round(ci * 100)}% CI `
        + `${gm.lo.toFixed(2)}-${gm.hi.toFixed(2)}x, ${n} tasks)${settle}`
      : `${who}: ${n} paired task${n === 1 ? '' : 's'}; ${MIN_ESTIMATE - n} more for an estimate`;
    const target = spec.options.coverage_target.ratio_halfwidth || 1.25;
    return {
      key, compare: 'models', provider: g.provider, model: g.model, baseline_provider: base.provider,
      baseline_model: base.model, harness: g.family, group: g.key, baseline_group: base.key, arm_name: g.model,
      baseline_name: base.model, headline,
      pairs: {n_tasks: n, same_run_fraction: n ? round(pairs.filter((x) => x.same_run).length / n, 3) : null,
        arm_only_tasks: [], baseline_only_tasks: [], dropped: {no_arm_valid: 0, no_baseline_valid: 0,
          usage_not_recorded: 0, tokens_not_reported: 0, unpriced: 0, zero_value: 0}},
      metrics, pass: pairedPass(pairs, scale),
      coverage: coverageFor({key, group: g.key, baseline_group: base.key}, n, {...w, pairsBy: modelPairsBy}, g, base,
        target),
      caveats: [...flagCaveats(`${g.family} on ${g.model}`, rowsOf(w, g.key)),
        ...flagCaveats(`${g.family} on ${base.model}`, rowsOf(w, base.key)), MODELS_NO_VERDICT],
    };
  }

  function buildModelsReport(spec, variant) {
    const w = getWorld();
    const ci = Math.min(0.99, Math.max(0.8, Number(spec.options.ci_level) || 0.9));
    const scale = probit((1 + ci) / 2) / Z90;
    const f = spec.filters;
    const baseModel = spec.options.baseline.model;
    const keepModel = (g) => !f.models.length || f.models.includes(modelKey(g)) || modelKey(g) === baseModel;
    const keepFamily = (family) => !f.families.length || f.families.includes(family);
    const groups = clone(w.groups.filter((g) => keepModel(g) && keepFamily(g.family)));
    for (const g of groups) {
      g.harness_key = `fam:${g.family}`;
      g.harness_name = g.family;
      g.is_baseline = modelKey(g) === baseModel;
    }
    const warnings = [];
    const comparisons = [];
    for (const family of [...new Set(groups.map((g) => g.family))].sort()) {
      const mine = groups.filter((g) => g.family === family);
      const base = mine.find((g) => g.is_baseline);
      const others = mine.filter((g) => !g.is_baseline);
      if (!base && others.length) {
        const one = others.length === 1;
        warnings.push(`No trials on ${baseModel} with ${family}, so ${others.length} group${one ? '' : 's'} there `
          + `${one ? 'has' : 'have'} no comparison.`);
      }
      if (base) comparisons.push(...others.map((g) => modelComparison(w, g, base, spec, ci, scale)));
    }
    const trials = [...w.trials.values()].filter((r) => keepModel(r) && keepFamily(r.family));
    const byOutcome = {};
    for (const r of trials) byOutcome[r.outcome] = (byOutcome[r.outcome] || 0) + 1;
    for (const g of groups) {
      g.setup_modes = {};
      for (const r of trials.filter((x) => groupKey(x, x.family) === g.key)) {
        g.setup_modes[r.setup_mode] = (g.setup_modes[r.setup_mode] || 0) + 1;
      }
    }
    return {
      spec,
      provenance: {generated_at: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
        fingerprint: `sha256:${hex(hash(JSON.stringify(spec) + variant), 16)}`, trials_used: trials.length,
        exclusions_applied: effective().size, pricing: spec.options.pricing, pricing_sha: '2437ed4f51bf',
        parser_versions: [6], classifier_version: 2, stats_version: 4, harbor_versions: ['0.25.0'],
        includes_active_run: false, direction: 'harness_over_baseline'},
      scope: scopeOf(f),
      totals: {trials: trials.length, runs: w.runIds.length, profiles: [profile()], by_outcome: byOutcome,
        excluded_manual: 0, spend_usd: round(trials.reduce((t, r) => t + (r.cost_usd || 0), 0), 4),
        usage_unmeasured: trials.filter((r) => r.usage_status === 'unmeasured').length},
      groups, comparisons,
      coverage: {task_order: w.order, groups: groups.map((g) => g.key),
        cells: Object.fromEntries(groups.map((g) => [g.key, clone(w.cells[g.key] || {})]))},
      warnings,
    };
  }

  /** report.task_scope: the request's tasks against the profile's task file (the fake comparisons ignore them). */
  function scopeOf(f) {
    const cfg = demo.config();
    const scope = new Set((cfg && cfg.tasks) || []);
    const chosen = new Set(f.tasks);
    const same = chosen.size === scope.size && [...chosen].every((t) => scope.has(t));
    return {kind: !chosen.size ? 'all' : same ? 'scope' : 'chosen', tasks: chosen.size, scope_tasks: scope.size,
      tasks_file: (cfg && cfg.tasks_file) || 'config/tasks.txt'};
  }

  function coverageOf(report) {
    const touched = new Set(Object.values(report.coverage.cells).flatMap((c) => Object.keys(c)));
    return {
      provenance: report.provenance, task_order: report.coverage.task_order,
      untouched_tasks: report.coverage.task_order.filter((t) => !touched.has(t)),
      comparisons: report.comparisons.map((c) => ({key: c.key, provider: c.provider, model: c.model, group: c.group,
        baseline_group: c.baseline_group, headline: c.headline, coverage: c.coverage})),
    };
  }

  function pairsOf(body) {
    const w = getWorld();
    const report = buildReport(body);
    if (!report.comparisons.some((c) => c.key === body.comparison)) return null;
    const found = w.pairsBy[body.comparison] || modelPairsBy[body.comparison];
    const excluded = clone(found.excluded);
    if (report.spec.options.policy.exclusions) {
      for (const p of found.pairs) {
        for (const [side, keys] of [['arm', p.arm.trial_keys], ['baseline', p.baseline.trial_keys]]) {
          for (const key of keys) {
            const row = w.trials.get(key);
            const e = row && exclusionOf(row);
            if (e && e.action === 'exclude') excluded.push({trial_key: key, side, task: row.task, outcome: row.outcome, reason: `excluded by hand: ${e.reason}`});
          }
        }
      }
    }
    return {comparison: body.comparison, pairs: clone(found.pairs), excluded_trials: excluded};
  }

  function csv(rows) {
    const cell = (v) => {
      const s = v === null || v === undefined ? '' : typeof v === 'object' ? JSON.stringify(v) : String(v);
      return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    return `﻿${rows.map((r) => r.map(cell).join(',')).join('\r\n')}\r\n`;
  }

  function exportCsv(body) {
    const report = buildReport(body);
    const level = body.level || 'comparisons';
    // The filters set, as report.export_rows writes them.
    const used = JSON.stringify(Object.fromEntries(Object.entries(report.spec.filters)
      .filter(([, v]) => v !== null && v !== false && !(Array.isArray(v) && !v.length))));
    let rows;
    if (level === 'pairs') {
      rows = [['comparison', 'task', 'digest_short', 'same_run', 'arm_tokens_total', 'baseline_tokens_total',
        'arm_cost', 'baseline_cost', 'ratio_tokens', 'ratio_cost', 'direction', 'filters']];
      for (const c of report.comparisons) {
        for (const p of (getWorld().pairsBy[c.key] || {pairs: []}).pairs) {
          rows.push([c.key, p.task, p.digest_short, p.same_run, p.arm.tokens_total, p.baseline.tokens_total, p.arm.cost,
            p.baseline.cost, p.ratio_tokens, p.ratio_cost, report.provenance.direction, used]);
        }
      }
    } else if (level === 'trials') {
      rows = [['trial_key', 'run_key', 'family', 'provider', 'model', 'task', 'outcome', 'usage_status', 'tokens_total',
        'cost_usd', 'filters']];
      for (const r of getWorld().trials.values()) {
        rows.push([r.trial_key, r.run_key, r.family, r.provider, r.model, r.task, r.outcome, r.usage_status,
          r.tokens_total, r.cost_usd, used]);
      }
    } else {
      rows = [['comparison', 'metric', 'status', 'n', 'need', 'geomean', 'geomean_lo', 'geomean_hi', 'verdict',
        'headline', 'direction', 'filters']];
      for (const c of report.comparisons) {
        for (const [name, m] of Object.entries(c.metrics)) {
          rows.push([c.key, name, m.status, m.n, m.need, m.geomean && m.geomean.estimate, m.geomean && m.geomean.lo, m.geomean && m.geomean.hi,
            m.reference && m.reference.verdict, c.headline, report.provenance.direction, used]);
        }
      }
    }
    const scope = report.scope;
    const tasks = scope.kind === 'scope' ? `${scope.tasks}-scope-tasks`
      : scope.kind === 'all' ? 'all-tasks' : `${scope.tasks}-tasks`;
    const name = `bench-report-${report.provenance.fingerprint.slice(7, 15)}-${tasks}-${level}.csv`;
    return new Response(csv(rows), {headers: {'Content-Type': 'text/csv; charset=utf-8', 'Content-Disposition': `attachment; filename="${name}"`}});
  }

  // Ledger rows for the fake models ----------------------------------------------

  const isDemoTarget = (type, id) => (type === 'run'
    ? getWorld().runIds.some((r) => `${profile()}/${r}` === id)
    : getWorld().trials.has(id));

  function trialRow(row) {
    return {...clone(row), exclusion: exclusionOf(row)};
  }

  function trialsQuery(body) {
    const f = (body && body.filters) || {};
    const models = Array.isArray(f.models) ? f.models : [];
    if (!models.some((m) => DEMO_MODELS.has(m))) return undefined;  // the real ledger answers
    const outcomes = Array.isArray(body.outcomes) ? body.outcomes : [];
    const list = (k) => (Array.isArray(f[k]) && f[k].length ? f[k] : null);
    const keep = {families: list('families'), tasks: list('tasks'), runs: list('runs'),
      exclude_runs: list('exclude_runs'), setup_modes: list('setup_modes')};
    let rows = [...getWorld().trials.values()].filter((r) => models.includes(`${r.provider}:${r.model}`)
      && (!keep.families || keep.families.includes(r.family)) && (!keep.tasks || keep.tasks.includes(r.task))
      && (!keep.runs || keep.runs.includes(r.run_key)) && (!keep.exclude_runs || !keep.exclude_runs.includes(r.run_key))
      && (!keep.setup_modes || keep.setup_modes.includes(r.setup_mode))
      && (!outcomes.length || outcomes.includes(r.outcome)));
    const [col, dir] = String(body.order || 'started_at DESC').split(' ');
    const sign = dir === 'ASC' ? 1 : -1;
    rows = rows.sort((a, b) => {
      const x = a[col];
      const y = b[col];
      if (x === y) return 0;
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      return (x < y ? -1 : 1) * sign;
    });
    const offset = Number(body.offset) || 0;
    const limit = Number(body.limit) || 100;
    return {total: rows.length, rows: rows.slice(offset, offset + limit).map(trialRow)};
  }

  function trialDetail(id) {
    const row = [...getWorld().trials.values()].find((r) => r.id === id);
    if (!row) return null;
    const t = trialRow(row);
    const recovered = t.usage_status === 'lower_bound' ? {recovered_from: 'agent/pi/sessions'} : {};
    return {
      ...t,
      usage: t.usage_status === 'unmeasured' ? {} : {[t.model]: {n_input: t.n_input, n_cache: t.n_cache, n_output: t.n_output,
        reported_cost_usd: t.reported_cost_usd, primary: true, reported_as: null, synthesized: false, ...recovered,
        ...(t.provider === 'anthropic' && !t.flags.cache_writes_unknown
          ? {n_cache_write_5m: Math.round(t.n_cache * 0.06), n_cache_write_1h: 0} : {})}},
      config: {effort: t.effort, extras: {}, family: t.family, implementation: t.implementation, model: t.model, options: t.options, provider: t.provider,
        timeouts: {agent_sec: 900}},
      exception_message: t.exception_head, task_digest: `sha256:${t.task_digest_short}${hex(hash(t.task), 52)}`, task_source_type: 'local',
      tokens_uncached: t.tokens_total === null ? null : Math.round(t.tokens_total * 0.16), harbor_model: `${t.provider}/${t.model}`,
      harness_version_pinned: t.harness_version, n_concurrent: 2, cap_marker: null, setup_marker: t.outcome === 'setup_failure' ? 'agent never started' : null,
      agent_started: t.outcome !== 'setup_failure', rewards: t.reward === null ? null : {reward: t.reward}, trial_uid: hex(hash(t.trial_key), 32),
      ingest_via: 'run', ingested_at: t.finished_at, updated_at: t.finished_at, parser_version: 2, classifier_version: 2,
      raw: {agent_info: {name: t.family, version: t.harness_version, model_info: {name: t.model, provider: t.provider}},
        verifier_result: {rewards: t.reward === null ? {} : {reward: t.reward}}, exception_info: t.exception_type ? {exception_type: t.exception_type, exception_message: t.exception_head} : null},
      lock: {agent: {name: t.family, model_name: `${t.provider}/${t.model}`, kwargs: t.options}, task: {name: t.task}},
      trial_dir: `profiles/${t.profile}/jobs/${t.run_id}/${t.arm}/${t.trial_name}`, trial_dir_exists: false,
    };
  }

  function demoRuns() {
    const w = getWorld();
    const rows = [...w.trials.values()];
    return w.runIds.map((id) => {
      const mine = rows.filter((r) => r.run_id === id);
      const trials = {};
      for (const r of mine) trials[r.outcome] = (trials[r.outcome] || 0) + 1;
      const key = `${profile()}/${id}`;
      const e = effective().get(`run|${key}`);
      return {run_key: key, profile: profile(), run_id: id, source: 'bench', mode: 'demo', status: 'done', created_at: isoAgo(3 * 86400e3),
        finished_at: isoAgo(3 * 86400e3 - 3600e3), budget_usd: 30, trials, spend_usd: round(mine.reduce((t, r) => t + (r.cost_usd || 0), 0), 4),
        trials_unmeasured: mine.filter((r) => r.usage_status === 'unmeasured').length,
        trials_lower_bound: mine.filter((r) => r.usage_status === 'lower_bound').length,
        families: [...new Set(mine.map((r) => r.family))], models: [...new Set(mine.map((r) => `${r.provider}:${r.model}`))],
        dataset_snapshot: 'local:terminal-bench@2.0@2026-10-29T05:29:58+00:00', exclusion: e ? {action: e.action, reason: e.reason, created_at: e.created_at} : null};
    });
  }

  async function realJson(path, fallback) {
    try {
      const res = await demo.server(path, {credentials: 'same-origin', cache: 'no-store'});
      return res.ok ? await res.json() : fallback;
    } catch {
      return fallback;
    }
  }

  function seededSnapshot() {
    if (store.snapshots.length) return;
    const body = {filters: {}, options: {}};
    const response = buildReport(body, {variant: 'then'});
    store.snapshots.push({id: 'rs-demo-0001', name: 'Before the fourth pilot (demo)', created_at: isoAgo(6 * 86400e3),
      fingerprint: response.provenance.fingerprint, trials_used: response.provenance.trials_used, request: clone(response.spec), response});
  }

  // Routes ------------------------------------------------------------------------

  /** False when the filters name runs (ANALYSIS's Open in Report, or Report on it): the fake comparisons are built
   *  per model and harness and cannot be narrowed to a run, so the real ledger answers, which never holds a
   *  simulated run. The Ledger sub-tab does narrow the fake trials by run. */
  function fakeApplies(body) {
    const runs = body && body.filters && Array.isArray(body.filters.runs) ? body.filters.runs : [];
    return !runs.length;
  }

  demo.on('POST', '/api/report', ({body}) => (fakeApplies(body) ? buildReport(body || {}) : undefined));
  demo.on('POST', '/api/report/coverage', ({body}) => (fakeApplies(body) ? coverageOf(buildReport(body || {})) : undefined));
  demo.on('POST', '/api/report/pairs', ({body}) => {
    if (!fakeApplies(body)) return undefined;
    const out = pairsOf(body || {});
    return out || demo.error(404, `No comparison '${(body || {}).comparison}' in this report`);
  });
  demo.on('POST', '/api/report/export', ({body}) => exportCsv(body || {}));

  demo.on('GET', '/api/report/snapshots', async () => {
    seededSnapshot();
    const real = await realJson('/api/report/snapshots', {snapshots: []});
    const mine = [...store.snapshots].reverse().map(({id, name, created_at, fingerprint, trials_used}) => ({
      id, name, created_at, fingerprint, trials_used, direction: 'harness_over_baseline'}));
    return {snapshots: [...mine, ...(real.snapshots || [])]};
  });
  demo.on('GET', /^\/api\/report\/snapshots\/([^/]+)$/, ({match}) => {
    seededSnapshot();
    const s = store.snapshots.find((x) => x.id === decodeURIComponent(match[1]));
    return s ? {id: s.id, name: s.name, created_at: s.created_at, request: s.request, response: s.response} : undefined;
  });
  demo.on('POST', '/api/report/snapshots', ({body}) => {
    const name = body && typeof body.name === 'string' ? body.name.trim() : '';
    if (!name || name.length > 80) return demo.error(400, 'Give the snapshot a name of 1 to 80 characters');
    seededSnapshot();
    const response = buildReport(body);
    const s = {id: `rs-demo-${String(store.nextSnapshot++).padStart(4, '0')}`, name, created_at: new Date().toISOString().replace('Z', '+00:00'),
      fingerprint: response.provenance.fingerprint, trials_used: response.provenance.trials_used, request: clone(response.spec), response};
    store.snapshots.push(s);
    return demo.json({id: s.id, name, created_at: s.created_at, fingerprint: s.fingerprint, trials_used: s.trials_used}, 201);
  });

  demo.on('GET', '/api/ledger/facets', async () => {
    const facets = await realJson('/api/ledger/facets', null);
    if (!facets) return undefined;
    const rows = [...getWorld().trials.values()];
    for (const m of MODELS) {
      const key = modelKey(m);
      if (!facets.models.some((x) => x.key === key)) {
        facets.models.push({key, provider: m.provider, model: m.model, trials: rows.filter((r) => `${r.provider}:${r.model}` === key).length});
      }
    }
    // The harness and setup counts include the fake trials too, as the model counts do.
    facets.setup_modes = facets.setup_modes || [];
    for (const r of rows) {
      let mode = facets.setup_modes.find((x) => x.value === r.setup_mode);
      if (!mode) {
        mode = {value: r.setup_mode, trials: 0};
        facets.setup_modes.push(mode);
      }
      mode.trials += 1;
    }
    for (const r of rows) {
      let fam = facets.families.find((x) => x.value === r.family);
      if (!fam) {
        fam = {value: r.family, trials: 0, implementations: [], versions: []};
        facets.families.push(fam);
      }
      fam.trials += 1;
      if (!fam.implementations.includes(r.implementation)) fam.implementations.push(r.implementation);
      if (r.harness_version && !fam.versions.includes(r.harness_version)) fam.versions.push(r.harness_version);
    }
    return facets;
  });
  // The status counts the fake trials and runs too, so the strip's "N of M trials used" adds up.
  demo.on('GET', '/api/ledger/status', async () => {
    const status = await realJson('/api/ledger/status', null);
    if (!status || !status.available) return status || undefined;
    const rows = [...getWorld().trials.values()];
    status.trials += rows.length;
    status.runs += getWorld().runIds.length;
    status.by_profile = {...status.by_profile, [profile()]: (status.by_profile[profile()] || 0) + rows.length};
    for (const r of rows) status.by_outcome[r.outcome] = (status.by_outcome[r.outcome] || 0) + 1;
    return status;
  });
  demo.on('GET', '/api/ledger/runs', async () => {
    const real = await realJson('/api/ledger/runs', null);
    return real ? [...demoRuns(), ...real] : undefined;
  });
  demo.on('POST', '/api/ledger/trials', ({body}) => trialsQuery(body || {}));
  demo.on('GET', /^\/api\/ledger\/trials\/(\d+)$/, ({match}) => {
    const id = Number(match[1]);
    if (id < 900000) return undefined;
    return trialDetail(id) || demo.error(404, `No trial ${id}`);
  });
  demo.on('POST', '/api/ledger/exclusions', ({body}) => {
    const b = body || {};
    if (!isDemoTarget(b.target_type, b.target_id)) return undefined;  // a real trial or run: the real ledger records it
    const reason = typeof b.reason === 'string' ? b.reason.trim() : '';
    if (!reason || reason.length > 500) return demo.error(400, 'Give a reason of 1 to 500 characters');
    if (!['exclude', 'include'].includes(b.action)) return demo.error(400, 'action must be exclude or include');
    const row = {id: store.nextExclusion++, target_type: b.target_type, target_id: b.target_id, action: b.action, reason,
      created_at: new Date().toISOString().replace('Z', '000Z')};
    store.exclusions.push(row);
    return demo.json(row, 201);
  });
  demo.on('GET', '/api/ledger/exclusions', async () => {
    const real = await realJson('/api/ledger/exclusions', null);
    if (!real) return undefined;
    const latest = effective();
    const mine = store.exclusions.map((e) => ({...e, effective: latest.get(`${e.target_type}|${e.target_id}`) === e}));
    return {exclusions: [...(real.exclusions || []), ...mine]};
  });
})();
