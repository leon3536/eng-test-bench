/* Demo mode: simulated runs, so the live monitor and the analysis views can be
   previewed without spending anything. Loaded only when the page URL has ?demo,
   before app.js starts, followed by demo-report.js, demo-designs.js,
   demo-queue.js and demo-assistant.js (DEMO_SCRIPTS in app.js).

   It wraps fetch. /api/state gets simulated runs added to the real state; a
   simulated run's results, log and cancel are answered here. Everything else,
   including keys, config, schemas and plans, goes to the real server, except
   what an area's demo file answers through window.benchDemo (below).

   The simulated records carry the v2 fields: ArmRun family, trials_running,
   reserve_usd, trials_killed, trials_unmeasured, cleanup_pending and pid; the
   run's reserve_usd, source and design_ref (one past run came from the queue,
   on a non-Claude model, so its comparisons carry verdict notes); and the
   addendum's usage_status per trial with measured/excluded/unmeasured/
   lower_bound per arm. "Run again as design" on a simulated run is refused.

   Billed paths are never sent: POST /api/runs, /api/queue/items, /api/queue/start,
   /api/queue/items/{id}/retry, /api/assistant/conversations/{cid}/messages and
   /retry, and /api/keys/{ENV_KEY}/test. A demo file may answer one in the
   browser; if none does, it is refused here with a 409. */

(() => {
  'use strict';

  const realFetch = window.fetch.bind(window);
  const PRICE = {input: 4, cache: 0.2, output: 20};  // claude-opus-5-5, USD per million tokens
  const PROFILES = {
    'claude-code': {tokens: 1.9e6, cache: 0.86, out: 0.022, pass: 0.72, err: 0.03, min: [9, 22], reports: true, helper: 0.015},
    'terminus-2': {tokens: 0.95e6, cache: 0.62, out: 0.03, pass: 0.64, err: 0.05, min: [8, 20]},
    'mini-swe-agent': {tokens: 0.62e6, cache: 0.55, out: 0.035, pass: 0.56, err: 0.04, min: [6, 16]},
    pi: {tokens: 0.6e6, cache: 0.7, out: 0.03, pass: 0.6, err: 0.04, min: [6, 15]},
    strands: {tokens: 0.8e6, cache: 0.5, out: 0.03, pass: 0.5, err: 0.08, min: [7, 18]},
  };
  const FALLBACK = {tokens: 0.9e6, cache: 0.6, out: 0.03, pass: 0.55, err: 0.05, min: [8, 18]};
  // [type, message, failure_cause]. A timeout either kept its completed calls (usage is a lower bound) or recorded
  // nothing (unmeasured); a setup failure never reached the model, so it has no tokens and is left out of the means.
  const ERRORS = [
    ['AgentTimeoutError', 'Agent execution timed out after 1800 seconds', 'agent_limit'],
    ['RateLimitError', 'Provider returned 429 after 6 retries', 'provider_failure'],
    ['DockerComposeError', 'Container exited before the verifier ran', 'infra'],
    ['EnvironmentStartTimeoutError', 'The task environment did not start within 600 seconds', 'setup_failure'],
  ];
  const DOLLAR_CAPPED = new Set(['claude-code', 'mini-swe-agent']);  // harnesses whose per-trial dollar cap the plan counts
  const KEY_PROBLEM = /\b[A-Z][A-Z0-9_]*_API_KEY is not set\b/;  // a plan problem the demo treats as solved
  const TRIAL_CAP = 15;
  const REFERENCE = {
    consensus: {low: 2.0, high: 3.0, source: 'S1-S5: changing harness moves cost 2x to 3x'},
    arms: {
      pi: {tokens: {value: 3.1, source: 'S10 via S9: 742k vs 236k tokens per task, same model (secondhand)'}},
      strands: {cost: {value: 2.1, source: 'S8: $0.74 to $1.57 per trial on Anthropic models (estimated, vendor study)'}},
      'mini-swe-agent': {note: 'S2: 829 vs 16,581 tokens of preamble per step (20x); no per-task total published'},
      'terminus-2': {note: "S3: no cost figures; on Opus 4.6 it scored 63.8% against Claude Code's 70.1%"},
    },
  };

  const pad2 = (n) => String(n).padStart(2, '0');
  const stamp = (t) => {
    const d = new Date(t);
    return `${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}-${pad2(d.getHours())}${pad2(d.getMinutes())}${pad2(d.getSeconds())}`;
  };
  const iso = (t) => new Date(t).toISOString().replace(/\.\d+Z$/, '+00:00');
  const clock = (t) => {
    const d = new Date(t);
    return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
  };
  const sum = (list, fn) => list.reduce((total, x) => total + (fn(x) || 0), 0);
  const mean = (list, fn) => (list.length ? sum(list, fn) / list.length : null);

  function rng(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Lays out a whole run up front: when each arm and trial starts and ends, and how each trial turns out. */
  function simulate(spec) {
    const r = rng(spec.seed);
    const ms = spec.msPerMinute;
    const tasks = spec.tasks;
    const difficulty = tasks.map(() => 0.45 + r() * 0.9);
    const slots = Array(spec.parallelArms).fill(0);
    const arms = spec.arms.map((arm) => {
      const p = PROFILES[arm.agent] || FALLBACK;
      const slot = slots.indexOf(Math.min(...slots));
      const start = slots[slot];
      const preflightEnd = start + 0.5;
      const sim = {...arm, start, preflightEnd, trials: [], fail: spec.failArms && spec.failArms.includes(arm.name)};
      if (sim.fail) {
        sim.end = preflightEnd + 0.2;
        slots[slot] = sim.end;
        return sim;
      }
      const lanes = Array(spec.nConc).fill(preflightEnd);
      tasks.forEach((task, ti) => {
        for (let attempt = 0; attempt < spec.attempts; attempt++) {
          const lane = lanes.indexOf(Math.min(...lanes));
          const begin = lanes[lane];
          const minutes = p.min[0] + r() * (p.min[1] - p.min[0]) * difficulty[ti];
          const errored = r() < p.err;
          const passed = errored ? null : r() < Math.min(0.95, p.pass / difficulty[ti] * 0.85);
          const err = errored ? ERRORS[Math.floor(r() * ERRORS.length)] : null;
          const cause = err ? err[2] : null;
          const usage = cause === 'setup_failure' ? 'none' : cause === 'agent_limit' ? (r() < 0.5 ? 'unmeasured' : 'lower_bound') : 'full';
          const recorded = usage === 'full' || usage === 'lower_bound';
          const size = recorded ? p.tokens * (0.55 + r() * 1.05) * difficulty[ti] * (errored ? 0.4 : 1) : 0;
          const nOutput = Math.round(size * p.out);
          const nInput = Math.round(size - nOutput);
          const nCache = Math.round(nInput * Math.min(0.97, p.cache * (0.9 + r() * 0.15)));
          const cost = (((nInput - nCache) * PRICE.input + nCache * PRICE.cache + nOutput * PRICE.output) / 1e6) * (spec.priceScale || 1);
          const id = Math.floor(r() * 2 ** 31).toString(36).slice(0, 7);
          sim.trials.push({
            begin, end: begin + minutes, row: {
              arm: arm.name, family: arm.family || arm.agent, task, trial: `${task}__${id}`,
              reward: passed === null ? null : passed ? 1.0 : 0.0, reward_valid: passed !== null, passed,
              error: err ? err[0] : null, failure_cause: cause, error_message: err ? err[1] : '',
              n_input: nInput, n_cache: nCache, n_output: nOutput,
              computed_cost: cost, reported_cost: p.reports && recorded ? cost * (1.03 + r() * 0.06) : null,
              cost_partial: false, unpriced_floor_cost: 0, reported_mispriced: false,
              unpriced_models: [], models: !recorded ? [] : p.helper ? ['claude-haiku-5-5', arm.model] : [arm.model],
              other_model_tokens: p.helper && recorded ? Math.round(size * p.helper) : 0,
              minutes,
              usage_status: usage === 'none' ? 'full' : usage,  // a trial that never reached the model recorded all it spent: nothing
            },
          });
          lanes[lane] = begin + minutes;
        }
      });
      sim.end = Math.max(...lanes) + 0.2;
      slots[slot] = sim.end;
      return sim;
    });

    // Budget: walk trials in finishing order; once spend passes the budget, the run stops there.
    const finishing = arms.flatMap((a) => a.trials.map((t) => ({a, t}))).sort((x, y) => x.t.end - y.t.end);
    let spent = 0;
    let budgetStop = null;
    for (const {t} of finishing) {
      spent += t.row.computed_cost;
      if (spent > spec.budget) {
        budgetStop = t.end;
        break;
      }
    }
    const end = Math.max(...arms.map((a) => a.end));
    return {...spec, arms, budgetStop, endMin: budgetStop !== null ? budgetStop + 0.1 : end, cancelAt: null, ms};
  }

  /** The run record /api/state would hold for a simulation at time now. */
  function snapshot(run, now) {
    const at = (minute) => run.start + minute * run.ms;
    const stopMin = run.cancelAt !== null ? (run.cancelAt - run.start) / run.ms : run.budgetStop;
    const stopReason = run.cancelAt !== null ? 'cancelled' : run.budgetStop !== null ? 'budget_stopped' : null;
    const nowMin = (now - run.start) / run.ms;
    const spendOf = (t) => Math.max(t.row.computed_cost, t.row.reported_cost || 0);
    const arms = run.arms.map((a, index) => {
      const total = run.tasks.length * run.attempts;
      const finished = a.trials.filter((t) => t.end <= nowMin && (stopMin === null || t.end <= stopMin));
      const unmeasured = finished.filter((t) => t.row.usage_status === 'unmeasured').length;
      const out = {
        name: a.name, family: a.family || a.agent, status: 'queued', message: '', trials_total: total, trials_done: finished.length,
        spend_usd: Math.round(sum(finished, spendOf) * 1e4) / 1e4,
        started_at: null, finished_at: null, returncode: null,
        // v2 counters, as runs.ArmRun reports them.
        trials_running: 0, reserve_usd: 0, trials_killed: 0, trials_unmeasured: unmeasured, cleanup_pending: false, pid: null,
      };
      const lowerBound = unmeasured
        ? ` ${unmeasured} finished trial(s) ran but recorded no usage; their spend is unknown, so this arm's recorded spend is a lower bound.` : '';
      if (nowMin >= a.start) out.started_at = iso(at(a.start));
      if (stopMin !== null && stopMin <= nowMin && a.end > stopMin) {
        if (a.start >= stopMin) {
          out.status = stopReason === 'cancelled' ? 'cancelled' : 'skipped';
          out.message = stopReason === 'cancelled' ? '' : 'Run budget was used up before this arm started';
          out.started_at = null;
        } else {
          out.status = stopReason;
          out.message = (stopReason === 'cancelled' ? 'Cancelled from the dashboard'
            : `Run spend passed the $${run.budget.toFixed(2)} budget`) + lowerBound;
          out.finished_at = iso(at(stopMin));
          out.returncode = 1;
          out.trials_killed = a.trials.filter((t) => t.begin <= stopMin && t.end > stopMin).length;
          out.cleanup_pending = Boolean(run.cleanupPending);
        }
        return out;
      }
      if (nowMin < a.start) return out;
      if (nowMin < a.preflightEnd) {
        out.status = 'preflight';
        out.pid = 4100 + index;
        return out;
      }
      if (a.fail) {
        out.status = 'failed';
        out.message = 'Preflight failed, so nothing was spent. See the log.';
        out.finished_at = iso(at(a.preflightEnd));
        out.returncode = 2;
        return out;
      }
      if (nowMin < a.end) {
        // Each running trial holds back the largest finished trial's spend, at most the arm's dollar cap.
        const largest = Math.max(0, ...finished.map(spendOf));
        out.status = 'running';
        out.pid = 4100 + index;
        out.trials_running = a.trials.filter((t) => t.begin <= nowMin && t.end > nowMin).length;
        out.reserve_usd = Math.round(out.trials_running * (DOLLAR_CAPPED.has(a.agent) ? Math.min(TRIAL_CAP, largest) : largest) * 1e4) / 1e4;
        out.message = lowerBound.trim();
        return out;
      }
      out.status = 'done';
      out.message = lowerBound.trim();
      out.finished_at = iso(at(a.end));
      out.returncode = 0;
      return out;
    });
    const over = (stopMin !== null && stopMin <= nowMin) || nowMin >= run.endMin;
    const statuses = new Set(arms.map((a) => a.status));
    const status = !over ? 'running'
      : stopReason === 'cancelled' ? 'cancelled'
        : statuses.has('budget_stopped') || statuses.has('skipped') ? 'budget_stopped'
          : [...statuses].every((s) => s === 'done') ? 'done' : 'done_with_errors';
    const plan = {
      n_attempts: run.attempts,
      worst_case_capped_usd: 0,
      uncapped_arms: [],
      peak_containers: Math.min(run.parallelArms, run.arms.length) * run.nConc,
      warnings: [],
      arms: run.arms.map((a) => ({
        name: a.name, agent: a.agent, family: a.family || a.agent, kind: a.kind || 'name', provider: a.provider || 'anthropic',
        model: `${a.provider || 'anthropic'}/${a.model}`, effort: a.agent === 'strands' ? 'harness default' : 'medium',
      })),
    };
    for (const a of run.arms) {
      if (DOLLAR_CAPPED.has(a.agent)) plan.worst_case_capped_usd += TRIAL_CAP * run.tasks.length * run.attempts;
      else plan.uncapped_arms.push(a.name);
    }
    return {
      id: run.id, mode: run.mode, created_at: iso(run.start),
      finished_at: over ? iso(at(stopMin !== null && stopMin <= nowMin ? stopMin : run.endMin)) : null,
      budget_usd: run.budget, spend_usd: Math.round(sum(arms, (a) => a.spend_usd) * 1e4) / 1e4,
      reserve_usd: Math.round(sum(arms, (a) => a.reserve_usd) * 1e4) / 1e4,
      tasks: run.tasks, status, message: '', arms, plan,
      config: {baseline_arm: run.baseline, run: {n_concurrent: run.nConc, parallel_arms: run.parallelArms}},
      profile: demoConfig ? demoConfig.profile : 'default', source: 'config', design_ref: null, spec_hash: '', harbor_version: '0.25.0',
      // runs.Run.to_dict always sends these three; nothing in the page reads them yet.
      dataset: {name: 'terminal-bench@2.0'}, spec: {}, resolved_arms: [],
      ...run.extra,
    };
  }

  /** results.summarize: means over measured trials; setup failures and trials without usage are left out of them. */
  function summarize(arm, rows) {
    const out = {
      arm: arm.name, agent: arm.agent, family: arm.family || arm.agent, provider: arm.provider || 'anthropic', model: arm.model,
      trials: rows.length, measured: 0, excluded: 0, unmeasured: 0, lower_bound: 0,
    };
    if (!rows.length) return out;
    const scored = rows.filter((x) => x.passed !== null);
    const passed = scored.filter((x) => x.passed).length;
    const cost = (x) => (x.computed_cost !== null ? x.computed_cost : x.reported_cost || 0);
    const measured = rows.filter((x) => x.n_input || x.n_output || cost(x));
    const totalIn = sum(rows, (x) => x.n_input);
    const totalTokens = totalIn + sum(rows, (x) => x.n_output);
    return Object.assign(out, {
      measured: measured.length, excluded: rows.length - measured.length,
      unmeasured: rows.filter((x) => x.usage_status === 'unmeasured' && !cost(x)).length,
      lower_bound: measured.filter((x) => x.usage_status === 'lower_bound').length,
      scored: scored.length, passed, errors: rows.filter((x) => x.error).length,
      pass_rate: scored.length ? passed / scored.length : null,
      mean_input: mean(measured, (x) => x.n_input), mean_cache: mean(measured, (x) => x.n_cache), mean_output: mean(measured, (x) => x.n_output),
      mean_tokens: measured.length ? sum(measured, (x) => x.n_input + x.n_output) / measured.length : null,
      cache_share: totalIn ? sum(rows, (x) => x.n_cache) / totalIn : null,
      mean_cost: mean(measured, cost), total_cost: sum(rows, cost), cost_partial: false,
      reported_cost_total: rows.every((x) => x.reported_cost !== null) ? sum(rows, (x) => x.reported_cost) : null,
      cost_per_success: passed ? sum(rows, cost) / passed : null,
      other_model_share: totalTokens ? sum(rows, (x) => x.other_model_tokens) / totalTokens : 0,
      models: [...new Set(rows.flatMap((x) => x.models))].sort(),
      unpriced_models: [], mean_minutes: mean(rows, (x) => x.minutes),
    });
  }

  function verdict(measured, ref) {
    if (measured === null || !ref) return null;
    const [low, high] = 'value' in ref ? [ref.value / 1.5, ref.value * 1.5] : [ref.low, ref.high];
    return measured < low ? 'lower' : measured > high ? 'higher' : 'in line';
  }

  function compare(summaries, baseline) {
    const base = summaries.find((s) => s.arm === baseline);
    if (!base || !base.trials) return [];
    const ratio = (a, b) => (a && b ? a / b : null);
    // Published figures were measured on Claude models, so other models get no verdict (results.compare's applies_to).
    const claude = (s) => /^claude-/.test(String(s.model || '').split('/').pop());
    return summaries.filter((s) => s.arm !== baseline && s.trials).map((s) => {
      const ref = REFERENCE.arms[s.family] || REFERENCE.arms[s.agent] || {};
      const tokensX = ratio(base.mean_tokens, s.mean_tokens);
      const costX = ratio(base.mean_cost, s.mean_cost);
      const publishedCost = ref.cost || REFERENCE.consensus;
      const applies = claude(s) && claude(base);
      return {
        arm: s.arm, family: s.family, tokens_multiple: tokensX, cost_multiple: costX,
        published_tokens: ref.tokens || null, published_cost: publishedCost,
        tokens_verdict: applies ? verdict(tokensX, ref.tokens) : null, cost_verdict: applies ? verdict(costX, publishedCost) : null,
        verdict_note: applies ? null
          : `No verdict: the published figures were measured on Claude models, and this comparison runs ${s.model} on ${s.provider}.`,
        note: ref.note || null, excluded: s.excluded, baseline_excluded: base.excluded,
      };
    });
  }

  function results(run, now) {
    const nowMin = (now - run.start) / run.ms;
    const stopMin = run.cancelAt !== null ? (run.cancelAt - run.start) / run.ms : run.budgetStop;
    const summaries = [];
    const trials = [];
    for (const a of run.arms) {
      const rows = a.trials
        .filter((t) => t.end <= nowMin && (stopMin === null || t.end <= stopMin))
        .map((t) => t.row);
      trials.push(...rows);
      summaries.push(summarize(a, rows));
    }
    return {run_id: run.id, baseline: run.baseline, arms: summaries, comparison: compare(summaries, run.baseline), trials};
  }

  function log(run, armName, now) {
    const a = run.arms.find((x) => x.name === armName);
    if (!a) return null;
    const at = (minute) => run.start + minute * run.ms;
    const nowMin = (now - run.start) / run.ms;
    const stopMin = run.cancelAt !== null ? (run.cancelAt - run.start) / run.ms : run.budgetStop;
    if (nowMin < a.start || (stopMin !== null && a.start >= stopMin)) return '';
    const base = `harbor run --dataset terminal-bench@2.0 --agent ${a.agent} --model ${a.provider || 'anthropic'}/${a.model} --n-attempts ${run.attempts} `
      + `--n-concurrent ${run.nConc} --jobs-dir jobs\\${run.id} --job-name ${a.name} --max-retries 0 --quiet`;
    const lines = [`[${clock(at(a.start))}] $ ${base} --dry-run`];
    if (nowMin < a.preflightEnd) return `${lines.join('\n')}\n`;
    if (a.fail) {
      lines.push(`[${clock(at(a.preflightEnd))}] Error: invalid agent kwarg for ${a.agent}: 'max_thinking_tokens' rejected by model`);
      lines.push(`[${clock(at(a.preflightEnd))}] exit code 2`);
      return `${lines.join('\n')}\n`;
    }
    lines.push(`[${clock(at(a.preflightEnd))}] Validated ${run.tasks.length * run.attempts} trial configs. Nothing was run.`,
      `[${clock(at(a.preflightEnd))}] exit code 0`, `[${clock(at(a.preflightEnd))}] $ ${base}`);
    const events = [];
    for (const t of a.trials) {
      if (t.begin <= nowMin && (stopMin === null || t.begin <= stopMin)) events.push([t.begin, `▶ ${t.row.trial} started`]);
      if (t.end <= nowMin && (stopMin === null || t.end <= stopMin)) {
        const x = t.row;
        const what = x.error ? `! ${x.trial} ${x.error}: ${x.error_message}`
          : `${x.passed ? '✓' : '✗'} ${x.trial} reward=${x.reward.toFixed(1)}`;
        events.push([t.end, `${what}  tokens=${((x.n_input + x.n_output) / 1e6).toFixed(2)}M  cost=$${x.computed_cost.toFixed(2)}  ${x.minutes.toFixed(1)}m`]);
      }
    }
    events.sort((x, y) => x[0] - y[0]);
    for (const [minute, text] of events) lines.push(`[${clock(at(minute))}] ${text}`);
    if (stopMin !== null && stopMin <= nowMin && a.end > stopMin) {
      lines.push(`[${clock(at(stopMin))}] stopping harbor (pid 4${a.name.length}12) and its containers`, `[${clock(at(stopMin))}] removed 2 container(s)`);
    } else if (nowMin >= a.end) {
      lines.push(`[${clock(at(a.end))}] exit code 0`);
    }
    return `${lines.join('\n')}\n`;
  }

  // The simulated runs, built from the real config on the first state request.
  let runs = null;
  let demoConfig = null;
  const pageLoad = Date.now();

  function build(state) {
    const cfg = state.config;
    demoConfig = cfg;
    const arms = cfg.arms.map((a) => ({name: a.name, agent: a.agent, family: a.family || a.agent, kind: a.kind || 'name', provider: a.provider, model: a.model}));
    const tasks = cfg.tasks.length ? cfg.tasks : Array.from({length: 10}, (_, i) => `task-${i + 1}`);
    const day = 24 * 3600e3;
    const make = (offset, spec) => {
      const start = pageLoad - offset;
      return {...simulate({baseline: cfg.baseline_arm, arms, nConc: 2, parallelArms: 2, attempts: 1, budget: 150, msPerMinute: 60000, ...spec}),
        id: `${stamp(start)}-${spec.slug || spec.mode}`, start, extra: spec.extra, cleanupPending: Boolean(spec.cleanupPending)};
    };
    // One past run came from the queue: an assistant-drafted design on a non-Claude model, so its comparisons carry
    // verdict notes instead of verdicts. Its design id is made up, so DESIGNS answers 404 for it unless demo-designs.js knows it.
    const deepseek = arms.map((a) => ({...a, provider: 'openrouter', model: 'deepseek/deepseek-v4.1-flash'}));
    const designRun = {
      source: 'design', mode: 'design', spec_hash: `sha256:5e1d0c3a9b7f${'2468ace0'.repeat(6)}6a4d`,
      design_ref: {design_id: 'd-sample-on-deepseek-3f2a', version: 2, author: 'assistant', conversation_id: 'c-20261027-101500-ab12', queue_item: 'q-0002'},
    };
    runs = [
      make(100e3, {mode: 'sample', tasks: tasks.slice(0, 10), seed: 7, msPerMinute: 1100}),
      make(2 * day, {mode: 'design', slug: 'sample-on-deepse-v2', tasks: tasks.slice(0, 10), seed: 42, arms: deepseek, priceScale: 0.08,
        budget: 20, extra: designRun}),
      make(3 * day + 5 * 3600e3, {mode: 'pilot', tasks: tasks.slice(0, 1), seed: 3, failArms: [arms[1] && arms[1].name]}),
      make(5 * day, {mode: 'sample', tasks: tasks.slice(0, 10), seed: 11, budget: 30, cleanupPending: true}),
    ];
  }

  const json = (data, status = 200) => Promise.resolve(new Response(JSON.stringify(data), {
    status, headers: {'Content-Type': 'application/json'},
  }));

  // Billed paths (brief, Hard rule 2): answered by a demo file or refused, never sent.
  const BILLED = [
    ['POST', /^\/api\/runs$/, 'launching is disabled. Open the dashboard without ?demo to start real runs'],
    ['POST', /^\/api\/queue\/items$/, 'adding to the queue is simulated only'],
    ['POST', /^\/api\/queue\/start$/, 'starting the queue is simulated only'],
    ['POST', /^\/api\/queue\/items\/[^/]+\/retry$/, 'retrying a queue item is simulated only'],
    ['POST', /^\/api\/assistant\/conversations\/[^/]+\/(messages|retry)$/, 'the assistant model is never called'],
    ['POST', /^\/api\/keys\/[^/]+\/test$/, 'key tests call the provider, so they are off'],
  ];

  function billed(method, path) {
    const hit = BILLED.find(([m, pattern]) => m === method && pattern.test(path));
    return hit ? hit[2] : '';
  }

  const refuse = (why) => json({detail: `Demo mode: ${why}.`}, 409);

  function describe(input, init) {
    const request = input instanceof Request ? input : null;
    const method = String(init.method || (request && request.method) || 'GET').toUpperCase();
    const url = new URL(request ? request.url : String(input), location.href);
    return {method, url, path: url.pathname};
  }

  /** The real server, for everything that bills nothing. A billed path is refused here too. */
  function server(input, init = {}) {
    const {method, path} = describe(input, init);
    const why = billed(method, path);
    return why ? refuse(why) : realFetch(input, init);
  }

  // Registry: demo-*.js files answer their area's requests here.

  const handlers = [];    // {method, pattern, handle}, tried in registration order
  const stateHooks = [];

  /** The first registered handler's answer, or null when none answered. */
  async function answer(input, init) {
    const {method, url, path} = describe(input, init);
    let body;
    if (typeof init.body === 'string') {
      try {
        body = JSON.parse(init.body);
      } catch {
        body = init.body;
      }
    }
    for (const handler of handlers) {
      if (handler.method !== '*' && handler.method !== method) continue;
      const match = typeof handler.pattern === 'string' ? (handler.pattern === path ? [path] : null) : path.match(handler.pattern);
      if (!match) continue;
      let out;
      try {
        out = await handler.handle({method, path, url, query: url.searchParams, body, match, input, init});
      } catch (err) {
        console.error(`Demo handler for ${method} ${path} failed:`, err);
        return json({detail: `Demo handler failed: ${err.message}`}, 500);
      }
      if (out === undefined) continue;
      return out instanceof Response ? out : json(out);
    }
    return null;
  }

  window.benchDemo = {
    /**
     * Answers `method` (or '*') on `pattern`, an exact path or a RegExp tested on the path. handle(ctx) gets
     * {method, path, url, query, body (parsed JSON), match, input, init} and returns a Response, any other value
     * (sent as JSON 200), or undefined to pass the request on. May be async. Not for GET /api/state: use onState.
     */
    on(method, pattern, handle) {
      handlers.push({method: String(method).toUpperCase(), pattern, handle});
    },
    /** fn(state) edits each /api/state answer in place (its own block, e.g. state.queue), after the simulated runs are in. */
    onState(fn) {
      stateHooks.push(fn);
    },
    json,
    /** An error answer the page shows as it would the server's: {"detail": ...}. */
    error: (status, detail) => json({detail}, status),
    /** A request through every demo handler, as the page makes it. */
    fetch: (input, init) => window.fetch(input, init),
    /** A request to the real server, past the demo handlers. Billed paths are refused. */
    server,
    /** The real config from the first /api/state, or null before it. */
    config: () => demoConfig,
    /** The simulated run records as /api/state carries them now, newest first. */
    runs: () => (runs ? runs.map((run) => snapshot(run, Date.now())).sort((a, b) => b.created_at.localeCompare(a.created_at)) : []),
    /** GET /api/runs/{id}/results for a simulated run, or null. */
    results(id) {
      const run = runs && runs.find((r) => r.id === id);
      return run ? results(run, Date.now()) : null;
    },
    /**
     * Starts a simulated run now and returns its id. It shows in /api/state (as active_run while it runs), and its
     * results, log and cancel are answered like the others. opts: {arms: [{name, agent, model}], tasks: [str],
     * mode ('design'), id, budget (10), attempts (1), nConc (2), parallelArms (2), baseline, seed,
     * msPerMinute (1100: a simulated minute takes about a second), failArms: [name],
     * extra: fields copied onto the run record, e.g. {source: 'design', design_ref, spec_hash}}.
     */
    simulateRun(opts) {
      if (!runs) throw new Error('The demo runs are built on the first /api/state; start a run after that.');
      const begin = Date.now();
      const mode = opts.mode || 'design';
      const run = {
        ...simulate({
          baseline: opts.baseline || demoConfig.baseline_arm, arms: opts.arms, tasks: opts.tasks, mode,
          nConc: opts.nConc || 2, parallelArms: opts.parallelArms || 2, attempts: opts.attempts || 1,
          budget: opts.budget || 10, msPerMinute: opts.msPerMinute || 1100, seed: opts.seed || begin % 9973,
          failArms: opts.failArms || [],
        }),
        id: opts.id || `${stamp(begin)}-${mode}`, start: begin, extra: opts.extra || {},
      };
      runs.push(run);
      return run.id;
    },
    /** Shared scratch space: each demo file keeps its fake data under its own key (store.queue, store.designs ...). */
    store: {},
  };

  window.fetch = async (input, init = {}) => {
    const answered = await answer(input, init);
    if (answered) return answered;
    const {method, path} = describe(input, init);
    // ANALYSIS "Run again as design" on a simulated run: the server has never seen it. A demo file may answer first.
    const again = method === 'POST' && path.match(/^\/api\/designs\/from-run\/([^/]+)$/);
    if (again && runs && runs.some((r) => r.id === decodeURIComponent(again[1]))) {
      return refuse(`run ${decodeURIComponent(again[1])} is simulated, so the bench has no spec to run again`);
    }
    // LAUNCH: the real plan, with "X_API_KEY is not set" taken out (as demo-designs.js does for checks), so the
    // demo can show LAUNCH's two-step confirm. The confirm's second press is still refused below.
    if (path === '/api/plan' && method === 'GET') {
      const res = await server(input, init);
      if (!res.ok) return res;
      const plan = await res.json();
      const keep = (list) => (list || []).filter((p) => !KEY_PROBLEM.test(p));
      const before = plan.problems.length;
      plan.problems = keep(plan.problems);
      for (const a of plan.arms || []) a.problems = keep(a.problems);
      if (plan.problems.length < before) plan.warnings = [...(plan.warnings || []), 'Demo: provider keys count as set here, so missing keys are not listed.'];
      return json(plan);
    }
    if (path === '/api/state' && method === 'GET') {
      const res = await server(input, init);
      if (!res.ok) return res;
      const state = await res.json();
      if (!runs) build(state);
      const now = Date.now();
      const records = runs.map((run) => snapshot(run, now)).sort((a, b) => b.created_at.localeCompare(a.created_at));
      state.runs = [...records, ...state.runs];
      const live = records.find((r) => r.status === 'running');
      state.active_run = state.active_run || (live ? live.id : null);
      for (const hook of stateHooks) {
        try {
          hook(state);
        } catch (err) {
          console.error('A demo state hook failed:', err);
        }
      }
      return json(state);
    }
    const m = path.match(/^\/api\/runs\/([^/]+)\/(results|cancel|log)(?:\/([^/]+))?$/);
    const run = m && runs && runs.find((r) => r.id === decodeURIComponent(m[1]));
    if (!run) return server(input, init);
    const now = Date.now();
    if (m[2] === 'results') return json(results(run, now));
    if (m[2] === 'log') {
      const text = log(run, decodeURIComponent(m[3] || ''), now);
      if (text === null) return json({detail: `Not found: ${m[3]}`}, 404);
      return Promise.resolve(new Response(text, {headers: {'Content-Type': 'text/plain'}}));
    }
    if (snapshot(run, now).status !== 'running') return json({detail: 'That run is not in progress'}, 409);
    run.cancelAt = now;
    run.endMin = (now - run.start) / run.ms;
    return json({ok: true});
  };
})();
