/* Demo mode for the QUEUE area: a fake in-memory queue. Loaded after demo.js only with ?demo.

   Owned by the QUEUE builder, with queue.js and queue.css. Brief: "Testing"
   asks demo mode for queue items, start, retry, move and remove on a fake
   in-memory queue, with the matching state.queue block, and for the ARM
   confirms of Add to queue, Start now, Start queue and Retry.

   Every queue route is answered here, so nothing about the queue reaches the
   server, billed or not, and state.queue is replaced with this queue's
   summary. The fake follows bench/runqueue.py: it loads paused; a person
   starts it; it runs one design at a time, through benchDemo.simulateRun and
   only while no other run is in progress; it pauses when a run fails or is
   cancelled, when it runs dry, and when the global cap could be passed. The
   error answers carry the server's wording.

   Items added from DESIGNS or LAUNCH (enqueueDesign) read their version
   through benchDemo.fetch, so demo-designs.js answers for its own designs and
   the real server for the rest. The queue is seeded on the first /api/state:
   three finished items at once, on demo.js's older simulated runs so their
   history links open ANALYSIS (those runs still say LAUNCH there; only runs
   this queue starts carry a design_ref), then two waiting items and one held,
   on the DESIGNS demo's designs (benchDemo.store.designs gives their
   versions; the third, which fails, is an earlier version of one of them),
   then on other listed designs, made up only when none can be read. No route
   waits for that seed: demo-designs.js reads this queue for its statuses
   while it seeds. */

(() => {
  'use strict';

  if (!window.benchDemo) return;  // demo.js did not load
  const demo = window.benchDemo;

  const OPEN = ['waiting', 'held'];
  const ACTIVE = ['starting', 'running'];
  const FINISHED = ['done', 'budget_stopped', 'failed', 'cancelled', 'interrupted'];
  // Run status -> [item status, what the queue does next], as in runqueue.py. 'policy' follows on_failure; a
  // cancel is a person deciding to stop, so it always pauses.
  const OUTCOMES = {
    done: ['done', 'continue'], done_with_errors: ['done', 'continue'], budget_stopped: ['budget_stopped', 'continue'],
    failed: ['failed', 'policy'], interrupted: ['interrupted', 'policy'], cancelled: ['cancelled', 'pause'],
  };
  const MESSAGES = {
    done: '', budget_stopped: 'Stopped at its budget', failed: 'The run failed', interrupted: 'The run was interrupted', cancelled: 'Cancelled',
  };
  const VERBS = {failed: 'failed', interrupted: 'was interrupted', cancelled: 'was cancelled'};
  const DOCKER_DOWN = 'Docker Desktop is not running';
  const MADE_UP = [  // seeded open items when no design can be read
    {title: 'Block 1 on DeepSeek', version: 2, budget: 6, p50: 0.107933, p90: 0.230486, allowance: 0.355728},
    {title: 'Pi effort sweep', version: 1, budget: 4, p50: 0.3141, p90: 0.6219, allowance: 0.42},
    {title: 'Strands vs Claude Code', version: 3, budget: 8, p50: 1.1203, p90: 2.0517, allowance: 0.9},
  ];
  const PAST = ['Opus baseline sample', 'One-task pilot', 'Opus sample, second pass'];  // titles of the seeded history

  const pad2 = (n) => String(n).padStart(2, '0');
  const iso = (t) => new Date(t).toISOString().replace(/\.\d+Z$/, '+00:00');
  const stamp = (t) => {
    const d = new Date(t);
    return `${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}-${pad2(d.getHours())}${pad2(d.getMinutes())}${pad2(d.getSeconds())}`;
  };
  const round = (v) => Math.round(v * 1e6) / 1e6;
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const label = (item) => `${item.title} v${item.version}`;
  const short = (hash) => String(hash || 'none').replace(/^sha256:/, '').slice(0, 12);
  const slug = (text, max = 40) => String(text).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+/, '').slice(0, max).replace(/-+$/, '') || 'design';
  const dollars = (v) => `$${v.toFixed(2)}`;

  /** A stable made-up hex string for made-up designs: `n` hex characters from `text`. */
  function hex(text, n) {
    let out = '';
    let x = 2166136261;
    for (let pass = 0; out.length < n; pass++) {
      for (const ch of `${text}#${pass}`) x = Math.imul(x ^ ch.charCodeAt(0), 16777619) >>> 0;
      out += x.toString(16).padStart(8, '0');
    }
    return out.slice(0, n);
  }

  // The fake queue. Items keep what the demo needs to run them in fields starting with '_', never sent.
  const q = {
    state: 'paused', pause_reason: 'Not started yet. Press Start queue.', on_failure: 'pause',
    global_cap_usd: 75, cap_since: iso(Date.now()), next_seq: 1, items: [],
  };
  demo.store.queue = q;
  let seeded = false;      // set by the first /api/state
  const madeUp = new Set();  // design ids this file made up for its seed: they exist nowhere
  let docker = true;       // the Docker readiness row of the latest /api/state
  let realActive = null;   // a run on the real server, which also keeps the queue waiting

  class DemoError extends Error {
    constructor(status, message) {
      super(message);
      this.status = status;
    }
  }
  const fail = (status, message) => {
    throw new DemoError(status, message);
  };

  /** Seeds the queue once the real config is known (the first /api/state): finished items at once, open items once
   *  the designs they point at are read. No route waits for that: DESIGNS' fake asks this queue for its statuses
   *  while it seeds its own designs. */
  function seedOnce() {
    if (seeded || !demo.config()) return;
    seeded = true;
    seedHistory();
    seedOpen().catch((err) => console.warn('The demo queue could not seed its waiting items:', err));
  }

  /** A route; a DemoError becomes the server's {"detail"} answer. */
  const route = (fn) => async (ctx) => {
    seedOnce();
    try {
      return await fn(ctx);
    } catch (err) {
      if (err instanceof DemoError) return demo.error(err.status, err.message);
      throw err;
    }
  };

  const find = (id) => q.items.find((i) => i.id === id) || fail(404, `No queue item '${String(id).slice(0, 20)}'`);
  const active = () => q.items.find((i) => ACTIVE.includes(i.status)) || null;
  const countOf = (status) => q.items.filter((i) => i.status === status).length;
  const committed = () => q.items.filter((i) => i.status === 'waiting').reduce((t, i) => t + i.budget_usd + (i.inflight_allowance_usd || 0), 0);
  const nextId = () => `q-${String(q.next_seq++).padStart(4, '0')}`;

  function body(raw) {
    if (raw === undefined) return {};
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) fail(400, 'The request body must be a JSON object');
    return raw;
  }
  function only(b, allowed) {
    const unknown = Object.keys(b).filter((k) => !allowed.includes(k)).sort();
    if (unknown.length) fail(400, `Unknown field(s): ${unknown.join(', ')}`);
  }

  /** Spend of every simulated run created since the cap count started. */
  function capSpent() {
    const since = Date.parse(q.cap_since);
    return demo.runs().reduce((total, r) => (Date.parse(r.created_at) < since ? total : total + (r.spend_usd || 0)), 0);
  }
  const remaining = () => (q.global_cap_usd === null ? null : round(Math.max(0, q.global_cap_usd - capSpent())));

  /** An item as the server sends it, with `position` for open items. */
  function shown(item, position) {
    const out = {};
    for (const [key, value] of Object.entries(item)) if (!key.startsWith('_')) out[key] = value;
    if (position !== undefined) out.position = position;
    return JSON.parse(JSON.stringify(out));
  }

  /** GET /api/queue's QueueView. */
  function view() {
    const open = q.items.filter((i) => OPEN.includes(i.status)).map((item, n) => [item, n]);
    const finished = q.items.filter((i) => FINISHED.includes(i.status))
      .sort((a, b) => String(b.finished_at || b.added_at).localeCompare(String(a.finished_at || a.added_at)));
    const now = active();
    return {
      state: q.state, pause_reason: q.pause_reason, on_failure: q.on_failure,
      global_cap_usd: q.global_cap_usd, cap_since: q.cap_since, cap_spent_usd: round(capSpent()), cap_remaining_usd: remaining(),
      committed_usd: round(committed()),
      active: now ? shown(now) : null,
      waiting: open.filter(([i]) => i.status === 'waiting').map(([i, n]) => shown(i, n)),
      held: open.filter(([i]) => i.status === 'held').map(([i, n]) => shown(i, n)),
      history: finished.slice(0, 50).map((i) => shown(i)),
    };
  }

  /** The queue block of /api/state. */
  function summary() {
    const now = active();
    return {
      state: q.state, pause_reason: q.pause_reason, active_item: now ? shown(now) : null,
      waiting: countOf('waiting'), held: countOf('held'), cap_remaining_usd: remaining(),
    };
  }

  // Versions ---------------------------------------------------------------

  /** A design version, through the demo handlers (demo-designs.js) or the real server. Its errors pass on. */
  async function versionOf(id, version) {
    let res;
    try {
      res = await demo.fetch(`/api/designs/${encodeURIComponent(id)}/versions/${version}`, {cache: 'no-store', credentials: 'same-origin'});
    } catch {
      fail(503, `Could not read ${id} v${version}. Try again.`);
    }
    const data = await res.json().catch(() => null);
    if (!res.ok) fail(res.status, (data && typeof data.detail === 'string' && data.detail) || `Could not read ${id} v${version} (HTTP ${res.status})`);
    if (!data || !data.spec || !Array.isArray(data.spec.arms) || !data.spec.run) fail(500, `${id} v${version} has no readable spec`);
    return data;
  }

  /** What add and retry check: the version is the one approved, and has no problems now. */
  function admit(detail, approved) {
    if (detail.spec_hash !== approved) {
      fail(409, `Version ${detail.version} of ${detail.design_id} has spec hash ${short(detail.spec_hash)}, not the ${short(approved)} `
        + 'that was approved. Reload it and check it again.');
    }
    const problems = (detail.check && detail.check.validation && detail.check.validation.problems) || [];
    if (problems.length) fail(400, `This version cannot be queued: ${problems.join('; ')}`);
  }

  /** Without an estimate: every container that can be busy when the budget stops the run may finish a trial at the cap. */
  function fallbackAllowance(spec) {
    const trials = spec.arms.reduce((t, a) => t + (a.tasks || spec.tasks || []).length * (spec.attempts || 1), 0);
    const peak = (spec.run.parallel_arms || 1) * (spec.run.n_concurrent || 1);
    return Math.min(peak, trials) * (spec.run.per_trial_cap_usd || 0);
  }

  /** A waiting item for one version, with its money figures from the version's check. */
  function itemFrom(detail) {
    const spec = detail.spec;
    const est = detail.check && detail.check.estimate ? detail.check.estimate.run : null;
    const total = est && est.total ? est.total : {};
    return {
      id: nextId(), design_id: detail.design_id, version: detail.version, spec_hash: detail.spec_hash,
      title: String(detail.title || detail.design_id),
      budget_usd: round(spec.run.budget_usd),
      inflight_allowance_usd: round(est && isNum(est.inflight_allowance_usd) ? est.inflight_allowance_usd : fallbackAllowance(spec)),
      expected_usd: isNum(total.p50) ? round(total.p50) : null, p90_usd: isNum(total.p90) ? round(total.p90) : null,
      status: 'waiting', message: '', added_at: iso(Date.now()), started_at: null, finished_at: null, run_id: null, spend_usd: 0,
      _run: {arms: spec.arms, tasks: spec.tasks, attempts: spec.attempts || 1, baseline: spec.baseline_arm},
      _author: detail.author || 'user', _conversation: detail.conversation_id || null,
    };
  }

  // Seed -------------------------------------------------------------------

  /** Three finished items on demo.js's older runs, oldest first, so their ids count up in time. */
  function seedHistory() {
    const past = demo.runs().filter((r) => r.status !== 'running').slice(0, 3).reverse();
    past.forEach((run, n) => {
      const title = PAST[n] || `Earlier design ${n + 1}`;
      const status = (OUTCOMES[run.status] || ['failed'])[0];
      const designId = `d-${slug(title)}-${hex(title, 4)}`;
      madeUp.add(designId);
      q.items.push({
        id: nextId(), design_id: designId, version: 1, spec_hash: `sha256:${hex(title, 64)}`, title,
        budget_usd: run.budget_usd, inflight_allowance_usd: round(run.budget_usd * 0.04), expected_usd: null, p90_usd: null,
        status, message: MESSAGES[status] || '', added_at: iso(Date.parse(run.created_at) - 90e3), started_at: run.created_at,
        finished_at: run.finished_at, run_id: run.id, spend_usd: round(run.spend_usd || 0),
      });
    });
  }

  /** Two waiting items and one held, on the latest versions of designs the page can read (demo-designs.js's, or the
   *  server's), made-up ones otherwise. The third always fails, so running the queue shows a failure pausing it. */
  async function seedOpen() {
    const cfg = demo.config();
    const enabled = cfg.arms.filter((a) => a.enabled);
    const pool = enabled.length ? enabled : cfg.arms;
    const base = pool.find((a) => a.name === cfg.baseline_arm) || pool[0];
    const arms = [base, pool.find((a) => a !== base)].filter(Boolean).map((a) => ({name: a.name, agent: a.agent, model: a.model, provider: a.provider}));
    const tasks = (cfg.tasks.length ? cfg.tasks : ['task-1', 'task-2', 'task-3']).slice(0, 3);
    const day = 24 * 3600e3;
    const store = demo.store.designs && typeof demo.store.designs.version === 'function' ? demo.store.designs : null;
    const read = async () => {
      // Listing every design also lets the DESIGNS demo seed its own. Those come first: valid versions with estimates,
      // while a design saved on the real bench may be anything (a draft with a $183 budget, say).
      const res = await demo.fetch('/api/designs?filter=all', {cache: 'no-store', credentials: 'same-origin'});
      const list = res.ok ? ((await res.json()).designs || []) : [];
      const own = new Set(store && typeof store.ids === 'function' ? store.ids() : []);
      const usable = list.filter((d) => d && typeof d.id === 'string' && d.title && !d.archived && Number.isInteger(d.latest_version))
        .sort((a, b) => Number(own.has(b.id)) - Number(own.has(a.id))).slice(0, 2);
      const picks = usable.map((d) => [d.id, d.latest_version]);
      // The third item (the one that fails) is an earlier version of one of them, when one has it, so its chip
      // opens a design that exists and Start queue can name the providers it bills.
      const older = usable.find((d) => d.latest_version > 1);
      if (older) picks.push([older.id, older.latest_version - 1]);
      return (await Promise.all(picks.map(([id, n]) => (store && store.version(id, n))
        || versionOf(id, n).catch(() => null)))).filter((d) => d && d.spec && d.spec_hash);
    };
    const timeout = new Promise((resolve) => setTimeout(() => resolve([]), 8000));
    const details = await Promise.race([read().catch(() => []), timeout]);
    MADE_UP.forEach((made, n) => {
      const detail = details[n];
      const item = detail ? itemFrom(detail) : {
        id: nextId(), design_id: `d-${slug(made.title)}-${hex(made.title, 4)}`, version: made.version,
        spec_hash: `sha256:${hex(`${made.title} v${made.version}`, 64)}`, title: made.title,
        budget_usd: made.budget, inflight_allowance_usd: made.allowance, expected_usd: made.p50, p90_usd: made.p90,
        status: 'waiting', message: '', added_at: '', started_at: null, finished_at: null, run_id: null, spend_usd: 0,
        _run: {arms, tasks, attempts: 1, baseline: base && base.name},
      };
      if (!detail) madeUp.add(item.design_id);
      item._seeded = true;  // admitted before the page loaded: Retry needs no version check
      item.added_at = iso(Date.now() - (3 - n) * 0.1 * day);
      if (n === 1) {
        item.status = 'held';
        item.message = `${DOCKER_DOWN}. Start it, then retry this item and start the queue.`;
      }
      // Every arm of the third design fails its preflight, so running the queue shows a failure pausing it.
      if (n === 2) item._failArms = item._run.arms.map((a) => a.name);
      q.items.push(item);
    });
  }

  // Worker -----------------------------------------------------------------

  /** One pass of the queue's worker: record a finished run, then start the next design if the queue may. */
  function tick() {
    if (!seeded) return;
    reconcile();
    if (q.state !== 'running' || active()) return;
    const next = q.items.find((i) => i.status === 'waiting');
    if (!next) {
      const held = countOf('held');
      q.state = 'paused';
      q.pause_reason = `Queue finished: nothing is waiting${held ? ` (${held} held)` : ''}. Add a design and press Start queue.`;
      return;
    }
    if (realActive || demo.runs().some((r) => r.status === 'running')) return;  // one run at a time
    if (!docker) {
      hold(next, `${DOCKER_DOWN}. Start it, then retry this item and start the queue.`, true, DOCKER_DOWN);
      return;
    }
    if (q.global_cap_usd !== null) {
      const allowance = next.inflight_allowance_usd || 0;
      const need = next.budget_usd + allowance;
      const spent = capSpent();
      if (spent + need > q.global_cap_usd + 1e-9) {
        const left = Math.max(0, q.global_cap_usd - spent);
        hold(next, `The global cap of ${dollars(q.global_cap_usd)} has ${dollars(left)} left; this design needs ${dollars(need)} `
          + `(budget ${dollars(next.budget_usd)} plus ${dollars(allowance)} in-flight allowance). Raise or reset the cap, then retry this item.`,
        true, `Global cap: ${label(next)} needs ${dollars(need)} and ${dollars(left)} is left`);
        return;
      }
    }
    start(next);
  }

  function hold(item, message, pause, reason) {
    item.status = 'held';
    item.message = message;
    if (pause) {
      q.state = 'paused';
      q.pause_reason = reason;
    }
  }

  function start(item) {
    const now = Date.now();
    const run = item._run;
    const taken = new Set(demo.runs().map((r) => r.id));
    let id = `${stamp(now)}-${slug(item.title, 16)}-v${item.version}`;
    for (let n = 2; taken.has(id); n++) id = `${stamp(now)}-${slug(item.title, 16)}-v${item.version}-${n}`;
    // An arm whose every task is reused from the ledger (tasks []) runs nothing.
    const arms = run.arms.filter((a) => !Array.isArray(a.tasks) || a.tasks.length);
    const names = arms.map((a) => a.name);
    demo.simulateRun({
      id, arms: arms.map((a) => ({name: a.name, agent: a.agent, model: a.model})), tasks: run.tasks, attempts: run.attempts,
      budget: item.budget_usd, baseline: names.includes(run.baseline) ? run.baseline : names[0], failArms: item._failArms || [],
      extra: {
        source: 'design', spec_hash: item.spec_hash,
        design_ref: {design_id: item.design_id, version: item.version, author: item._author || 'user',
          conversation_id: item._conversation || null, queue_item: item.id},
      },
    });
    Object.assign(item, {status: 'running', message: '', run_id: id, started_at: iso(now)});
  }

  /** The active item's run has ended: record it, and pause as the outcome and on_failure say. */
  function reconcile() {
    const item = active();
    if (!item || item.status !== 'running') return;
    const run = demo.runs().find((r) => r.id === item.run_id);
    if (!run || run.status === 'running') return;
    // demo.js calls a run whose every arm failed "done_with_errors"; the bench calls it failed.
    const status = run.arms.length && run.arms.every((a) => a.status === 'failed') ? 'failed' : run.status;
    const [itemStatus, then] = OUTCOMES[status] || ['failed', 'pause'];
    Object.assign(item, {
      status: itemStatus, finished_at: run.finished_at || iso(Date.now()), spend_usd: round(run.spend_usd || 0),
      message: (run.message || '').trim() || MESSAGES[itemStatus] || `The run ended as '${status}'`,
    });
    if (then === 'pause' || (then === 'policy' && q.on_failure === 'pause')) {
      q.state = 'paused';
      q.pause_reason = `Paused: ${label(item)} ${VERBS[itemStatus] || `ended as '${status}'`} (${item.run_id}).`;
    }
  }

  function startQueue() {
    if (!active() && !q.items.some((i) => i.status === 'waiting')) {
      const held = countOf('held');
      if (held) fail(409, `Nothing is waiting. Retry or remove the ${held} held item(s) first.`);
      fail(409, 'Nothing is waiting in the queue. Add a design first.');
    }
    if (!docker) fail(409, `${DOCKER_DOWN}. Start it, then start the queue.`);
    q.state = 'running';
    q.pause_reason = '';
  }

  // Routes -----------------------------------------------------------------

  demo.on('GET', '/api/queue', route(() => {
    tick();
    return view();
  }));

  // Add to queue and Start now (enqueueDesign): billed on a server, simulated here.
  demo.on('POST', '/api/queue/items', route(async (ctx) => {
    const b = body(ctx.body);
    only(b, ['design_id', 'version', 'spec_hash', 'position', 'start']);
    if (typeof b.design_id !== 'string' || !b.design_id) fail(400, 'design_id must be text');
    if (!Number.isInteger(b.version)) fail(400, 'version must be a whole number');
    if (b.version < 1) fail(400, 'version must be at least 1');
    const startNow = b.start === undefined ? false : b.start;
    if (typeof startNow !== 'boolean') fail(400, 'start must be true or false');
    if (startNow && !docker) fail(409, 'Docker Desktop is not running. Start it, then add and start the design.');
    const position = b.position === undefined ? 'end' : b.position;
    if (position !== 'end' && position !== 'front') fail(400, "position must be 'end' or 'front'");
    if (typeof b.spec_hash !== 'string' || !b.spec_hash.startsWith('sha256:')) {
      fail(400, "spec_hash must be the 'sha256:...' hash of the version you approved");
    }
    const duplicate = () => {
      const open = q.items.find((i) => i.design_id === b.design_id && i.version === b.version && OPEN.includes(i.status));
      if (open) fail(409, `${b.design_id} v${b.version} is already in the queue (${open.id}, ${open.status})`);
    };
    duplicate();
    const detail = await versionOf(b.design_id, b.version);
    admit(detail, b.spec_hash);
    duplicate();  // again: another add may have finished meanwhile
    const item = itemFrom(detail);
    const first = q.items.findIndex((i) => OPEN.includes(i.status));
    if (position === 'front' && first >= 0) q.items.splice(first, 0, item);
    else q.items.push(item);
    if (startNow) {
      try {
        startQueue();
      } catch (err) {
        if (err instanceof DemoError) fail(409, `Added as ${item.id}, but the queue did not start: ${err.message}`);
        throw err;
      }
    }
    tick();
    return demo.json({item: shown(item), queue: view()}, 201);
  }));

  demo.on('DELETE', /^\/api\/queue\/items\/([^/]+)$/, route(({match}) => {
    const id = decodeURIComponent(match[1]);
    const item = find(id);
    if (!OPEN.includes(item.status)) fail(409, `Only waiting or held items can be removed; ${id} is ${item.status}`);
    q.items.splice(q.items.indexOf(item), 1);
    return view();
  }));

  demo.on('POST', /^\/api\/queue\/items\/([^/]+)\/move$/, route((ctx) => {
    const b = body(ctx.body);
    only(b, ['to_index']);
    if (!Number.isInteger(b.to_index)) fail(400, 'to_index must be a whole number');
    const id = decodeURIComponent(ctx.match[1]);
    const item = find(id);
    if (!OPEN.includes(item.status)) fail(409, `Only waiting or held items can be moved; ${id} is ${item.status}`);
    const others = q.items.filter((i) => OPEN.includes(i.status) && i !== item);
    const at = Math.min(Math.max(b.to_index, 0), others.length);
    // Only the order of open items means anything; history is sorted by time.
    q.items = [...q.items.filter((i) => !OPEN.includes(i.status)), ...others.slice(0, at), item, ...others.slice(at)];
    return view();
  }));

  // Retry: billed on a server (a running queue can start the item at once), simulated here.
  demo.on('POST', /^\/api\/queue\/items\/([^/]+)\/retry$/, route(async ({match}) => {
    const id = decodeURIComponent(match[1]);
    const item = find(id);
    if (item.status !== 'held') fail(409, `Only held items can be retried; ${id} is ${item.status}`);
    if (!item._seeded) {
      try {
        admit(await versionOf(item.design_id, item.version), item.spec_hash);
      } catch (err) {
        if (err instanceof DemoError && item.status === 'held') item.message = err.message;
        throw err;
      }
    }
    if (item.status !== 'held') fail(409, `Only held items can be retried; ${id} is ${item.status}`);
    Object.assign(item, {status: 'waiting', message: ''});
    tick();
    return view();
  }));

  // Start queue: billed on a server, simulated here.
  demo.on('POST', '/api/queue/start', route(() => {
    startQueue();
    tick();
    return view();
  }));

  demo.on('POST', '/api/queue/pause', route(() => {
    q.state = 'paused';
    q.pause_reason = 'Paused by you';
    return view();
  }));

  demo.on('PUT', '/api/queue/settings', route((ctx) => {
    const b = body(ctx.body);
    only(b, ['global_cap_usd', 'reset_cap', 'on_failure']);
    if ('global_cap_usd' in b && b.global_cap_usd !== null && !(isNum(b.global_cap_usd) && b.global_cap_usd > 0)) {
      fail(400, 'global_cap_usd must be a number above zero, or null for no cap');
    }
    if (b.on_failure !== undefined && b.on_failure !== null && b.on_failure !== 'pause' && b.on_failure !== 'skip') {
      fail(400, "on_failure must be 'pause' or 'skip'");
    }
    if ('reset_cap' in b && typeof b.reset_cap !== 'boolean') fail(400, 'reset_cap must be true or false');
    if ('global_cap_usd' in b) q.global_cap_usd = b.global_cap_usd;
    if (b.reset_cap) q.cap_since = iso(Date.now());
    if (b.on_failure) q.on_failure = b.on_failure;
    return view();
  }));

  // A design this file made up is not found, as the server would say, without asking it. demo-designs.js
  // registered first and passes on ids it does not hold.
  demo.on('GET', /^\/api\/designs\/([^/]+)(?:\/versions\/[^/]+)?$/, ({match}) => {
    const id = decodeURIComponent(match[1]);
    return madeUp.has(id) ? demo.error(404, `No design '${id}' (the demo queue made this one up)`) : undefined;
  });

  demo.onState((state) => {
    const check = (state.readiness || []).find((c) => c.name === 'Docker');
    docker = !check || check.ok;
    const simulated = new Set(demo.runs().map((r) => r.id));
    realActive = state.active_run && !simulated.has(state.active_run) ? state.active_run : null;
    seedOnce();
    tick();
    state.queue = summary();
  });
})();
