/* Demo mode for the DESIGNS area. Loaded after demo.js only with ?demo.

   Owned by the DESIGNS builder, with designs.js and designs.css. Brief:
   "Testing" asks demo mode for a valid design version (no problems), so Add
   to queue and Start now can open their ARM confirm.

   Every design write stays in the browser: new designs, new versions,
   from-config, from-run, archive and unarchive. A design saved on the server
   is copied in here the first time the page writes to it, and from then on
   this copy answers for it. Reads of other designs, and every live check, go
   to the real server; the checks come back with the missing-key problems
   taken out, as if the provider keys were set, so a demo version can be valid.
   Three demo designs are added to the list: a user design with two versions,
   an assistant draft, and a design that reuses measured trials. A reuse spec
   is resolved from the real check's reuse summary, so what a demo version
   reuses is what the real ledger holds for it.

   benchDemo.store.designs is for the other demo files (QUEUE shows matching
   titles from it):
     ids()             the design ids held here
     version(id, n)    {design_id, version, title, author, conversation_id, spec, spec_hash, check} or null
     summary(id)       a DesignSummary, or null
   A version's queued, running, approved and ran state comes from GET /api/queue
   (through every demo handler), so a fake queue's items show up here. */

(() => {
  'use strict';

  if (!window.benchDemo) return;  // demo.js did not load
  const demo = window.benchDemo;

  const RUN_KEYS = ['budget_usd', 'parallel_arms', 'n_concurrent', 'effort', 'per_trial_cap_usd', 'max_turns', 'timeout_multiplier'];
  const OPTIONAL_RUN = {prebake: 'prefer', setup_timeout_multiplier: 'auto', env_build_timeout_multiplier: 1};
  const RUN_EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max', 'omit'];
  const KEY_PROBLEM = /\b[A-Z][A-Z0-9_]*_API_KEY is not set\b/;
  const META_LIMITS = {title: 80, purpose: 1000, rationale: 1500, change_note: 200};
  const DAY = 24 * 3600e3;

  const designs = new Map();  // id -> {meta: {id, title, created_at, updated_at, latest_version, archived}, docs: {n: doc}, checks: {n: CheckResult}}
  let seeding = null;

  class Refusal extends Error {
    constructor(status, detail) {
      super(detail);
      this.status = status;
    }
  }

  const iso = (t = Date.now()) => new Date(t).toISOString().replace(/\.\d+Z$/, '+00:00');
  const clone = (v) => (v === undefined ? v : JSON.parse(JSON.stringify(v)));
  const sortKeys = (v) => (Array.isArray(v) ? v.map(sortKeys)
    : v && typeof v === 'object' ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])])) : v);
  const canon = (v) => JSON.stringify(sortKeys(v));
  const slug = (text) => String(text).normalize('NFKD').replace(/[^\x00-\x7f]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  const hex = (n) => Array.from({length: n}, () => Math.floor(Math.random() * 16).toString(16)).join('');

  /** A stand-in "sha256:<64 hex>" when the server could not plan the spec (the real hash comes from the plan). */
  function fakeHash(text) {
    let out = '';
    for (let round = 0; out.length < 64; round++) {
      let h = 0x811c9dc5 ^ round;
      for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193) >>> 0;
      out += h.toString(16).padStart(8, '0');
    }
    return `sha256:${out.slice(0, 64)}`;
  }

  /** Wraps a handler so a Refusal answers like the server would ({"detail": ...} with its status). */
  const guarded = (handle) => async (ctx) => {
    try {
      return await handle(ctx);
    } catch (err) {
      if (err instanceof Refusal) return demo.error(err.status, err.message);
      throw err;
    }
  };

  async function config() {
    if (!demo.config()) await demo.fetch('/api/state', {cache: 'no-store'});
    return demo.config();
  }

  async function serverJson(path, body) {
    const init = body === undefined ? {} : {method: 'POST', headers: {'X-Bench': '1', 'Content-Type': 'application/json'}, body: JSON.stringify(body)};
    const res = await demo.server(path, {credentials: 'same-origin', cache: 'no-store', ...init});
    const data = await res.json().catch(() => null);
    if (!res.ok) throw new Refusal(res.status, (data && data.detail) || `HTTP ${res.status}`);
    return data;
  }

  // Specs ------------------------------------------------------------------

  /** The server's DesignSpecIn -> NormalizedSpec, closely enough for the demo. Reuse is resolved by resolved(). */
  function normalize(cfg, s) {
    const families = Object.fromEntries(cfg.arms.map((a) => [a.agent, a.family || a.agent]));
    const run = {};
    for (const k of RUN_KEYS) run[k] = s.run && k in s.run ? s.run[k] : cfg.run[k];
    // As spec.OPTIONAL_RUN_KEYS: written only when they differ from the defaults, so older specs keep their hash.
    for (const [k, plain] of Object.entries(OPTIONAL_RUN)) {
      const v = s.run && k in s.run ? s.run[k] : cfg.run[k];
      if (v !== undefined && v !== plain) run[k] = v;
    }
    const tasks = [...(s.tasks || [])];
    return {
      dataset: cfg.dataset,
      tasks,
      attempts: s.attempts ?? 1,
      baseline_arm: s.baseline_arm ?? null,
      run,
      arms: (s.arms || []).map((a) => {
        const p = cfg.providers.find((x) => x.name === a.provider);
        const model = p && p.model_prefix && a.model.startsWith(p.model_prefix) ? a.model.slice(p.model_prefix.length) : a.model;
        const subset = Array.isArray(a.tasks) ? [...a.tasks].sort((x, y) => tasks.indexOf(x) - tasks.indexOf(y)) : null;
        return {name: a.name, agent: a.agent, family: families[a.agent] || a.agent, provider: a.provider, model,
          effort: a.effort || 'inherit', options: sortKeys(a.options || {}), tasks: subset};
      }),
      ...(s.reuse === 'measured' ? {reuse: 'measured', reused: {}} : {}),
    };
  }

  /** A reuse spec as the server saves it (reuse.resolve), from the real server's check of it: each arm's reused
      tasks leave its task list and are listed in spec.reused with their trials and runs. */
  function resolved(spec, check) {
    if (!spec.reuse) return spec;
    const out = clone(spec);
    for (const a of (check.reuse && check.reuse.arms) || []) {
      const arm = out.arms.find((x) => x.name === a.arm);
      if (!arm || !(a.reused || []).length) continue;
      const gone = new Set(a.reused.map((e) => e.task));
      arm.tasks = (arm.tasks || out.tasks).filter((t) => !gone.has(t));
      out.reused[a.arm] = a.reused.map((e) => ({task: e.task, trials: [...e.trials], runs: [...e.runs]}));
    }
    return out;
  }

  /** An arm's tasks as asked for, its reused tasks put back in the design's order (the server's asked_tasks). */
  function askedTasks(spec, arm) {
    const reused = ((spec.reused || {})[arm.name] || []).map((e) => e.task);
    if (!reused.length) return arm.tasks === null ? null : [...arm.tasks];
    const wanted = new Set([...(arm.tasks || spec.tasks), ...reused]);
    const asked = spec.tasks.filter((t) => wanted.has(t));
    return asked.length === spec.tasks.length ? null : asked;
  }

  /** A NormalizedSpec as DesignSpecIn, the way the server's to_input does it. */
  function toInput(spec) {
    const run = {...spec.run};
    if (!RUN_EFFORTS.includes(run.effort)) delete run.effort;
    return {
      tasks: [...spec.tasks], attempts: spec.attempts, baseline_arm: spec.baseline_arm, run,
      arms: spec.arms.map(({family, ...arm}) => ({...clone(arm), tasks: askedTasks(spec, arm)})),
      ...(spec.reuse ? {reuse: spec.reuse} : {}),
    };
  }

  /** The real server's check, with "X_API_KEY is not set" taken out: the demo treats every key as set. */
  function keyed(check) {
    const c = clone(check);
    const keep = (list) => (list || []).filter((p) => !KEY_PROBLEM.test(p));
    const before = c.validation.problems.length;
    c.validation.problems = keep(c.validation.problems);
    if (c.plan) {
      c.plan.problems = keep(c.plan.problems);
      for (const a of c.plan.arms || []) a.problems = keep(a.problems);
    }
    if (c.validation.problems.length < before) {
      c.validation.warnings = [...c.validation.warnings, 'Demo: provider keys count as set here, so missing keys are not listed.'];
    }
    c.validation.status = c.validation.problems.length ? 'invalid' : 'valid';
    return c;
  }

  async function realCheck(specIn) {
    try {
      return keyed(await serverJson('/api/designs/check', {spec: specIn}));
    } catch (err) {
      if (err instanceof Refusal && err.status === 400) throw err;
      return {  // the server could not check it: a valid stand-in, so the demo still flows
        validation: {status: 'valid', problems: [], warnings: [`Demo: the server's check was unavailable (${err.message}).`]},
        plan: null, estimate: null, estimate_note: 'No estimate: the server\'s check was unavailable in demo mode.',
        checked_at: iso(), stale: false, config_fingerprint: null, ledger_fingerprint: null,
      };
    }
  }

  /** What the server answers 400 for on a save: the meta limits, and a spec the check calls incomplete. */
  function metaIn(body) {
    const out = {};
    for (const [k, limit] of Object.entries(META_LIMITS)) {
      const v = body[k] === undefined ? '' : body[k];
      if (typeof v !== 'string') throw new Refusal(400, `${k}: Input should be a valid string`);
      if (v.trim().length > limit) throw new Refusal(400, `${k}: String should have at most ${limit} characters`);
      out[k] = v.trim();
    }
    if (!out.title) throw new Refusal(400, 'title: String should have at least 1 character');
    return out;
  }

  async function checkedSpec(specIn) {
    const cfg = await config();
    const check = await realCheck(specIn);
    if (!check.plan && !check.estimate && /not complete/.test(check.estimate_note || '')) {
      throw new Refusal(400, check.validation.problems.join('; ') || 'The spec is not complete.');
    }
    const spec = resolved(normalize(cfg, specIn), check);
    return {spec, check, hash: (check.plan && check.plan.spec_hash) || fakeHash(canon(spec))};
  }

  // Diffs, statuses and shapes ---------------------------------------------

  function diff(a, b, path = '') {
    const isObj = (v) => v && typeof v === 'object' && !Array.isArray(v);
    if (isObj(a) && isObj(b)) {
      const out = [];
      for (const key of [...Object.keys(a), ...Object.keys(b).filter((k) => !(k in a))]) {
        const sub = path ? `${path}.${key}` : key;
        if (!(key in b)) out.push({path: sub, from: a[key], to: null});
        else if (!(key in a)) out.push({path: sub, from: null, to: b[key]});
        else out.push(...diff(a[key], b[key], sub));
      }
      return out;
    }
    if (Array.isArray(a) && Array.isArray(b) && (a.length || b.length) && [...a, ...b].every(isObj)) {
      const out = [];
      for (let i = 0; i < Math.max(a.length, b.length); i++) {
        const sub = `${path}[${i}]`;
        if (i >= a.length) out.push({path: sub, from: null, to: b[i]});
        else if (i >= b.length) out.push({path: sub, from: a[i], to: null});
        else out.push(...diff(a[i], b[i], sub));
      }
      return out;
    }
    return canon(a) === canon(b) ? [] : [{path, from: clone(a), to: clone(b)}];
  }

  /** Per version, what the queue says: its open item's status, when it was first admitted, the runs it made. */
  async function queueInfo() {
    const info = new Map();
    let q = null;
    try {
      const res = await demo.fetch('/api/queue', {credentials: 'same-origin', cache: 'no-store'});
      if (res.ok) q = await res.json();
    } catch { /* no queue: every version reads as never queued */ }
    if (!q) return info;
    const items = [q.active, ...(q.waiting || []), ...(q.held || []), ...(q.history || [])].filter(Boolean);
    for (const item of items) {
      const key = `${item.design_id}#${item.version}`;
      const row = info.get(key) || {open: null, approved_at: null, runs: []};
      if (['waiting', 'held', 'starting', 'running'].includes(item.status) || item === q.active) row.open = row.open || item.status;
      if (item.added_at && (!row.approved_at || item.added_at < row.approved_at)) row.approved_at = item.added_at;
      if (item.run_id && !row.runs.includes(item.run_id)) row.runs.push(item.run_id);
      info.set(key, row);
    }
    return info;
  }

  function versionState(d, n, info) {
    const q = info.get(`${d.meta.id}#${n}`) || {open: null, approved_at: null, runs: []};
    const doc = d.docs[n];
    const runs = [...new Set([...(doc.runs || []), ...q.runs])];
    const approved = doc.approved_at || q.approved_at;
    let status = 'draft';
    if (d.meta.archived) status = 'archived';
    else if (q.open === 'starting' || q.open === 'running') status = 'running';
    else if (q.open === 'waiting' || q.open === 'held') status = 'queued';
    else if (runs.length) status = 'ran';
    else if (approved) status = 'approved';
    return {status, approved_at: approved || null, runs};
  }

  /** A simulated run, or one from the page's last /api/state (ui is app.js's, a global binding but not on window). */
  const findRun = (runId) => demo.runs().find((r) => r.id === runId)
    || (typeof ui !== 'undefined' && ui.state ? ui.state.runs.find((r) => r.id === runId) || null : null);

  function runBrief(runId) {
    const run = findRun(runId);
    return run ? {run_id: runId, status: run.status, spend_usd: run.spend_usd, budget_usd: run.budget_usd}
      : {run_id: runId, status: null, spend_usd: null, budget_usd: null};
  }

  function detailOf(d, n, info) {
    const doc = d.docs[n];
    const s = versionState(d, n, info);
    return {
      design_id: d.meta.id, version: n, parent_version: doc.parent_version, author: doc.author, conversation_id: doc.conversation_id,
      created_at: doc.created_at, title: doc.title, purpose: doc.purpose, rationale: doc.rationale, change_note: doc.change_note,
      spec: clone(doc.spec), spec_hash: doc.spec_hash, status: s.status, approved_at: s.approved_at, archived: d.meta.archived,
      latest_version: d.meta.latest_version, runs: s.runs.map(runBrief), check: {...clone(d.checks[n]), stale: false},
      diff_from_parent: clone(doc.diff),
    };
  }

  function metaOf(d, info) {
    const versions = {};
    for (const n of Object.keys(d.docs).map(Number).sort((x, y) => x - y)) {
      const doc = d.docs[n];
      const s = versionState(d, n, info);
      versions[n] = {author: doc.author, conversation_id: doc.conversation_id, created_at: doc.created_at, spec_hash: doc.spec_hash,
        change_note: doc.change_note, approved_at: s.approved_at, runs: s.runs, status: s.status};
    }
    return {schema: 1, ...clone(d.meta), status: versions[d.meta.latest_version].status, versions};
  }

  function summaryOf(d, info) {
    const n = d.meta.latest_version;
    const doc = d.docs[n];
    const check = d.checks[n] || {};
    const v = check.validation || {};
    const total = check.estimate && check.estimate.run ? check.estimate.run.total : {};
    const s = versionState(d, n, info);
    return {
      id: d.meta.id, title: d.meta.title, created_at: d.meta.created_at, updated_at: d.meta.updated_at, latest_version: n,
      archived: d.meta.archived, author: doc.author, conversation_id: doc.conversation_id, status: s.status, approved_at: s.approved_at,
      runs: s.runs, spec_hash: doc.spec_hash, arms: doc.spec.arms.length,
      trials: doc.spec.arms.reduce((t, a) => t + (a.tasks ? a.tasks.length : doc.spec.tasks.length), 0) * doc.spec.attempts,
      problems: Array.isArray(v.problems) ? v.problems.length : null, warnings: Array.isArray(v.warnings) ? v.warnings.length : null,
      expected_usd: total.p50 ?? null, p90_usd: total.p90 ?? null, checked_at: check.checked_at || null,
      config_fingerprint: check.config_fingerprint || null, ledger_fingerprint: check.ledger_fingerprint || null, check_stale: false,
    };
  }

  const matches = (s, filter) => (filter === 'archived' ? s.archived
    : s.archived ? false
      : filter === 'all' ? true
        : filter === 'queued' ? ['queued', 'running'].includes(s.status) : s.status === filter);

  // Writes -----------------------------------------------------------------

  function versionDoc(id, n, parent, author, conversationId, at, meta, spec, hash, previous) {
    const doc = {design_id: id, version: n, parent_version: parent, author, conversation_id: conversationId, created_at: at,
      title: meta.title, purpose: meta.purpose, rationale: meta.rationale, change_note: meta.change_note, spec, spec_hash: hash, diff: []};
    if (previous) {
      doc.diff = [...['title', 'purpose', 'rationale'].filter((k) => previous[k] !== doc[k]).map((k) => ({path: k, from: previous[k], to: doc[k]})),
        ...diff(previous.spec, spec)];
    }
    return doc;
  }

  /** `tag`: a fixed id suffix, so a seeded design keeps its id across reloads (DESIGNS reopens the one open last). */
  async function create(body, specIn, {author = 'user', conversationId = null, at = Date.now(), tag = null} = {}) {
    const meta = metaIn(body);
    const {spec, check, hash} = await checkedSpec(specIn);
    const id = `d-${slug(meta.title).slice(0, 24).replace(/-+$/, '') || 'design'}-${tag || hex(4)}`;
    const when = iso(at);
    designs.set(id, {
      meta: {id, title: meta.title, created_at: when, updated_at: when, latest_version: 1, archived: false},
      docs: {1: versionDoc(id, 1, null, author, conversationId, when, meta, spec, hash, null)},
      checks: {1: {...check, checked_at: when}},
    });
    return id;
  }

  async function addVersion(id, base, body, specIn, {author = 'user', conversationId = null, at = Date.now()} = {}) {
    const d = designs.get(id);
    if (d.meta.archived) throw new Refusal(409, `Design ${id} is archived. Unarchive it before saving a new version.`);
    if (!Number.isInteger(base) || base < 1) throw new Refusal(400, 'base_version must be a whole number of at least 1');
    const latest = d.meta.latest_version;
    if (base !== latest) {
      throw new Refusal(409, `Version ${latest} of ${id} was saved since you read version ${base}. Load the latest version and apply your change again.`);
    }
    const meta = metaIn(body);
    const {spec, check, hash} = await checkedSpec(specIn);
    const current = d.docs[latest];
    if (hash === current.spec_hash && canon(spec) === canon(current.spec) && ['title', 'purpose', 'rationale'].every((k) => current[k] === meta[k])) {
      throw new Refusal(409, `Nothing changed: the spec, title, purpose and rationale match version ${latest}.`);
    }
    const n = latest + 1;
    const when = iso(at);
    d.docs[n] = versionDoc(id, n, latest, author, conversationId, when, meta, spec, hash, current);
    d.checks[n] = {...check, checked_at: when};
    Object.assign(d.meta, {latest_version: n, title: meta.title, updated_at: when});
    return n;
  }

  /** A design saved on the server, copied in on the page's first write to it. False when the server has none. */
  async function adopt(id) {
    if (designs.has(id)) return true;
    let data;
    try {
      data = await serverJson(`/api/designs/${encodeURIComponent(id)}`);
    } catch (err) {
      if (err instanceof Refusal && err.status === 404) return false;
      throw err;
    }
    const d = {meta: {id, title: data.meta.title, created_at: data.meta.created_at, updated_at: data.meta.updated_at,
      latest_version: data.meta.latest_version, archived: data.meta.archived}, docs: {}, checks: {}};
    for (const n of Object.keys(data.meta.versions).map(Number)) {
      const v = n === data.current.version ? data.current : await serverJson(`/api/designs/${encodeURIComponent(id)}/versions/${n}`);
      const entry = data.meta.versions[n];
      d.docs[n] = {design_id: id, version: n, parent_version: v.parent_version, author: v.author, conversation_id: v.conversation_id,
        created_at: v.created_at, title: v.title, purpose: v.purpose, rationale: v.rationale, change_note: v.change_note,
        spec: v.spec, spec_hash: v.spec_hash, diff: v.diff_from_parent || [], approved_at: entry.approved_at, runs: entry.runs || []};
      d.checks[n] = keyed(v.check);
    }
    designs.set(id, d);
    return true;
  }

  // Three demo designs: a user design with a second version, an assistant draft, and one that reuses measured trials.
  async function seed() {
    const cfg = await config();
    const tasks = cfg.tasks.length ? cfg.tasks : Array.from({length: 10}, (_, i) => `task-${i + 1}`);
    const arm = (name) => cfg.arms.find((a) => a.name === name) || null;
    const baseline = arm(cfg.baseline_arm) || cfg.arms[0];
    // Name-registered harnesses with no host-file options, so the check has nothing to object to.
    const others = cfg.arms.filter((a) => a !== baseline && a.kind !== 'import_path' && !Object.keys(a.options || {}).length).slice(0, 2);
    const armIn = (a, extra = {}) => ({name: a.name, agent: a.agent, provider: a.provider, model: a.model, effort: 'inherit', ...extra});
    const run = {budget_usd: 20, parallel_arms: 2, n_concurrent: 2};
    const first = {tasks: tasks.slice(0, 5), attempts: 1, baseline_arm: baseline.name, run, arms: [armIn(baseline), ...others.map((a) => armIn(a))]};
    const id = await create({title: 'Block 1 head start (demo)', purpose: 'Pair the other harnesses with Claude Code on the first five seeded tasks.'},
      first, {at: Date.now() - 2 * DAY, tag: 'b1a4'});
    const second = clone(first);
    second.run.budget_usd = 30;
    if (second.arms[1]) second.arms[1].effort = 'high';
    await addVersion(id, 1, {title: 'Block 1 head start (demo)', purpose: 'Pair the other harnesses with Claude Code on the first five seeded tasks.',
      change_note: 'Budget up to $30, one arm at high effort'}, second, {at: Date.now() - DAY});
    const partner = others[0] || baseline;
    const half = tasks.slice(5, 8);
    const draft = {
      tasks: half, attempts: 1, baseline_arm: baseline.name, run: {budget_usd: 12, n_concurrent: 2},
      arms: [armIn(baseline), ...(partner === baseline ? [] : [armIn(partner, {tasks: half.slice(0, 2)})])],
    };
    await create({
      title: 'Fill the half-pairs (demo draft)', purpose: 'Complete pairs that have only one side measured.',
      rationale: `Coverage shows **${half.length} tasks** with a baseline trial but no ${partner.name} trial on the same model. `
        + 'Running them pairs every one, which narrows the cost interval for that comparison.\n\n'
        + '- Budget is the p90 estimate plus 15%.\n- One attempt each: more tasks beat more attempts.',
    }, draft, {author: 'assistant', conversationId: 'c-20261029-101500-a1b2', at: Date.now() - 3 * 3600e3, tag: 'c3f7'});
    // Reuse measured trials: the real server resolves it against the ledger, so each arm skips the tasks the ledger
    // already has for its exact configuration (options copied, so a pinned version counts) and may be skipped whole.
    const pinned = cfg.arms.filter((a) => a !== baseline && a.kind !== 'import_path'
      && Object.keys(a.options || {}).every((k) => k === 'version')).slice(0, 2);
    const withOptions = (a) => armIn(a, {options: clone(a.options || {})});
    await create({
      title: 'Fill the gaps on block 1 (demo)',
      purpose: 'Pair every harness with Claude Code on the first four seeded tasks, running only what the ledger lacks.',
    }, {tasks: tasks.slice(0, 4), attempts: 1, baseline_arm: baseline.name, run: {budget_usd: 15, n_concurrent: 2},
      arms: [withOptions(baseline), ...pinned.map(withOptions)], reuse: 'measured'}, {at: Date.now() - 2 * 3600e3, tag: 'e5d1'});
  }

  const ready = () => {
    if (!seeding) seeding = seed().catch((err) => console.warn('Demo designs could not be seeded:', err));
    return seeding;
  };

  // Routes -----------------------------------------------------------------

  const created = async (id) => demo.json(detailOf(designs.get(id), designs.get(id).meta.latest_version, await queueInfo()), 201);
  const idOf = (match) => decodeURIComponent(match[1]);

  demo.on('GET', '/api/designs', guarded(async ({url, query}) => {
    await ready();
    const filter = query.get('filter') || 'all';
    const real = await serverJson(`${url.pathname}${url.search}`);  // also answers a bad filter with its 400
    const info = await queueInfo();
    const mine = [...designs.values()].map((d) => summaryOf(d, info)).filter((s) => matches(s, filter));
    const list = [...mine, ...real.designs.filter((s) => !designs.has(s.id))];
    list.sort((a, b) => String(b.updated_at || '').localeCompare(String(a.updated_at || '')));
    return {designs: list};
  }));

  demo.on('POST', '/api/designs/check', guarded(async ({body}) => keyed(await serverJson('/api/designs/check', body || {}))));

  demo.on('POST', '/api/designs', guarded(async ({body}) => {
    await ready();
    const {spec, ...rest} = body || {};
    return created(await create(rest, spec || {}));
  }));

  demo.on('POST', '/api/designs/from-config', guarded(async ({body}) => {
    await ready();
    const cfg = await config();
    const mode = cfg.modes[body.mode];
    if (!mode) throw new Refusal(400, `Unknown mode '${body.mode}'`);
    const names = body.arms == null ? cfg.arms.filter((a) => a.enabled).map((a) => a.name) : [...new Set(body.arms)];
    if (!names.length) throw new Refusal(400, 'Pick at least one arm');
    const arms = names.map((n) => {
      const a = cfg.arms.find((x) => x.name === n);
      if (!a) throw new Refusal(400, `Unknown arm '${n}'`);
      return {name: a.name, agent: a.agent, provider: a.provider, model: a.model, effort: a.effort, options: clone(a.options || {})};
    });
    if (![undefined, 'off', 'measured'].includes(body.reuse)) throw new Refusal(400, 'reuse must be one of off, measured');
    const run = {...cfg.run};
    if (!RUN_EFFORTS.includes(run.effort)) delete run.effort;
    const spec = {tasks: cfg.tasks.slice(0, mode.n_tasks), attempts: mode.n_attempts,
      baseline_arm: names.includes(cfg.baseline_arm) ? cfg.baseline_arm : null, run, arms,
      ...(body.reuse === 'measured' ? {reuse: 'measured'} : {})};
    return created(await create({title: body.title || `${body.mode} from config`}, spec));
  }));

  demo.on('POST', /^\/api\/designs\/from-run\/([^/]+)$/, guarded(async ({match, body}) => {
    await ready();
    const cfg = await config();
    const runId = idOf(match);
    const run = findRun(runId);
    if (!run) throw new Refusal(404, `Not found: run ${runId}`);
    let spec;
    if (run.spec && run.spec.arms) {
      spec = toInput(run.spec);
    } else {  // a simulated run keeps no spec: rebuild it from the record
      const names = run.arms.map((a) => a.name);
      const fromCfg = (name) => cfg.arms.find((a) => a.name === name) || {};
      const runCfg = (run.config && run.config.run) || {};
      spec = {
        tasks: [...run.tasks], attempts: (run.plan && run.plan.n_attempts) || 1,
        baseline_arm: run.config && names.includes(run.config.baseline_arm) ? run.config.baseline_arm : null,
        run: {budget_usd: run.budget_usd, ...(runCfg.n_concurrent ? {n_concurrent: runCfg.n_concurrent} : {}),
          ...(runCfg.parallel_arms ? {parallel_arms: runCfg.parallel_arms} : {})},
        arms: run.arms.map((a) => ({name: a.name, agent: fromCfg(a.name).agent || a.agent, provider: fromCfg(a.name).provider || 'anthropic',
          model: String(a.model || fromCfg(a.name).model || '').replace(/^anthropic\//, '')})),
      };
    }
    return created(await create({title: (body && body.title) || `Run again: ${runId}`, purpose: `Repeats run ${runId}.`}, spec));
  }));

  demo.on('GET', /^\/api\/designs\/([^/]+)$/, guarded(async ({match}) => {
    await ready();
    const d = designs.get(idOf(match));
    if (!d) return undefined;  // the server's
    const info = await queueInfo();
    return {meta: metaOf(d, info), current: detailOf(d, d.meta.latest_version, info)};
  }));

  demo.on('GET', /^\/api\/designs\/([^/]+)\/versions\/([^/]+)$/, guarded(async ({match}) => {
    await ready();
    const d = designs.get(idOf(match));
    if (!d) return undefined;
    const n = Number(match[2]);
    if (!d.docs[n]) throw new Refusal(404, `Design ${d.meta.id} has no version ${match[2]}`);
    return detailOf(d, n, await queueInfo());
  }));

  demo.on('POST', /^\/api\/designs\/([^/]+)\/versions$/, guarded(async ({match, body}) => {
    await ready();
    const id = idOf(match);
    if (!(await adopt(id))) return undefined;  // unknown to the server too: its 404
    const {spec, base_version: base, ...rest} = body || {};
    const n = await addVersion(id, base, rest, spec || {});
    return demo.json(detailOf(designs.get(id), n, await queueInfo()), 201);
  }));

  demo.on('POST', /^\/api\/designs\/([^/]+)\/(archive|unarchive)$/, guarded(async ({match}) => {
    await ready();
    const id = idOf(match);
    if (!(await adopt(id))) return undefined;
    const d = designs.get(id);
    const info = await queueInfo();
    if (match[2] === 'archive') {
      const open = Object.keys(d.docs).filter((n) => (info.get(`${id}#${n}`) || {}).open);
      if (open.length) throw new Refusal(409, `${id} has version(s) ${open.join(', ')} in the queue. Remove them from the queue first.`);
    }
    d.meta.archived = match[2] === 'archive';
    return {meta: metaOf(d, info)};
  }));

  demo.store.designs = {
    ids: () => [...designs.keys()],
    version(id, n) {
      const d = designs.get(id);
      const doc = d && d.docs[n];
      return doc ? {design_id: id, version: n, title: doc.title, author: doc.author, conversation_id: doc.conversation_id,
        spec: clone(doc.spec), spec_hash: doc.spec_hash, check: clone(d.checks[n])} : null;
    },
    summary: (id) => (designs.has(id) ? summaryOf(designs.get(id), new Map()) : null),
  };
})();
