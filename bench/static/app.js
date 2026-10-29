/* Dashboard logic. Plain JavaScript with no build step, so it works offline.

   Six views. LIVE monitors the active run and opens by itself when a run
   starts. LAUNCH is the configuration mode: quick per-arm choices, a preflight
   that re-plans whenever settings change, and the launch control, which asks
   through the shared ARM confirm (or adds the setup to the queue while the
   queue runs). ANALYSIS breaks down past runs. Detailed settings live in a
   slide-in drawer.
   DESIGNS, QUEUE and REPORT live in designs.js, queue.js and report.js, and the
   assistant panel in assistant.js. Each registers itself with registerView or
   registerPanel (see Areas) and uses the helpers and shared components here.

   The server is the source of truth: this file polls /api/state and sends
   edits back. Drawer forms render on open, after a save and after a reload,
   and never while they hold unsaved edits. Key values only ever pass through
   the password field of the key dialog. */

'use strict';

const DASH = '–';
const EFFORT_LEVELS = ['low', 'medium', 'high', 'xhigh', 'max'];
const CLAUDE_MODELS = ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5', 'claude-fable-5-1'];
const VIEWS = ['live', 'launch', 'designs', 'queue', 'report', 'analysis'];  // tab order
const FAST_POLL_MS = 3000;
const SLOW_POLL_MS = 10000;
const DOLLAR_CAPS = ['max_budget_usd', 'cost_limit'];
const TURN_CAPS = ['max_turns'];
const OPEN_ARM = new Set(['queued', 'preflight', 'running']);
const STATUS_TAG = {
  queued: '', preflight: 'warn', running: 'run', done: 'ok', done_with_errors: 'warn',
  budget_stopped: 'warn', skipped: 'warn', interrupted: 'warn', cancelled: '', failed: 'bad',
};
const STATUS_LED = {
  queued: '', preflight: 'warn live', running: 'ok live', done: 'ok', done_with_errors: 'warn',
  budget_stopped: 'warn', skipped: 'warn', interrupted: 'warn', cancelled: '', failed: 'bad',
};
const SYS_LABEL = {Docker: 'Docker', Harbor: 'Harbor', 'Git line endings': 'Git', 'Task sample': 'Tasks', 'Dataset snapshot': 'Dataset', Assistant: 'AI'};
/** The keys row the assistant gets in profiles where no provider uses its key (provider "assistant"). */
const ASSISTANT_KEY_LABEL = 'Assistant (Claude Opus 5.5)';
const OUTCOME = {
  pass: {label: 'Pass', glyph: '✓', text: 'ok-text'},
  fail: {label: 'Fail', glyph: '✕', text: 'bad-text'},
  error: {label: 'Error', glyph: '!', text: 'warn-text'},
  unscored: {label: 'Unscored', glyph: '–', text: 'muted'},
};
const DEMO = new URLSearchParams(location.search).has('demo');
const NATIVE_FETCH = window.fetch;  // app.js loads first: demo.js replaces window.fetch after this
// Billed paths (brief, Hard rule 2). In demo mode api() refuses them itself unless demo.js's fetch wrapper is in place.
const BILLED_PATHS = [/^\/api\/runs\/?$/, /^\/api\/queue\/items\/?$/, /^\/api\/queue\/start\/?$/, /^\/api\/queue\/items\/[^/]+\/retry\/?$/,
  /^\/api\/assistant\/conversations\/[^/]+\/(messages|retry)\/?$/, /^\/api\/keys\/[^/]+\/test\/?$/];

const ui = {
  view: null,
  state: null,
  connected: true,
  lastSync: 0,
  dirty: new Set(),          // drawer forms with unsaved edits: 'run', 'modes', 'arm'
  keyTests: {},              // env_key -> {masked, pending} or {masked, ok, detail}
  keyTarget: null,
  schemas: new Map(),        // agent -> schema
  schemaLoads: new Map(),    // agent -> pending request
  drawer: null,              // {kind: 'system' | 'settings' | 'arm', arm}
  mode: null,                // mode picked on the Launch view
  plan: null,
  planError: '',
  planFor: '',               // the inputs the plan or error was built from
  planPending: '',
  planRetry: '',             // the inputs whose plan failed as busy or unreachable: planned again after planRetryAt
  planRetryAt: 0,
  planSeq: 0,
  launching: false,          // the LAUNCH ARM confirm (or its Add to queue path) is open
  launchDraft: null,         // {inputs, design_id, version}: the draft Add to queue saved, reused while the inputs hold
  busyNote: '',              // /api/state answered 503 (a config file or the ledger is locked); cleared by the next good poll
  editingModel: new Set(),   // modules whose model field holds unsent typing
  live: null,                // telemetry for the run being watched
  analysisRun: null,
  trialFilter: {arm: '', outcome: ''},
};

// Helpers ----------------------------------------------------------------

const $ = (selector, root = document) => root.querySelector(selector);

/** Child nodes from nested arrays. null, undefined and false are skipped; strings become text, never markup. */
const nodes = (children) => children.flat(Infinity)
  .filter((child) => child !== null && child !== undefined && child !== false)
  .map((child) => (child instanceof Node ? child : document.createTextNode(String(child))));

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  let value;
  for (const [key, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (key === 'class') el.className = v;
    else if (key === 'value') value = v;
    else if (key === 'style') el.style.cssText = v;
    else if (key.startsWith('on')) el.addEventListener(key.slice(2), v);
    else el.setAttribute(key, v === true ? '' : String(v));
  }
  el.append(...nodes(children));
  if (value !== undefined) el.value = value;  // after children, so a select can find its option
  return el;
}

function svg(tag, attrs, ...children) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, v] of Object.entries(attrs || {})) {
    if (v !== null && v !== undefined && v !== false) el.setAttribute(key, String(v));
  }
  el.append(...nodes(children));
  return el;
}

const fill = (el, ...children) => el.replaceChildren(...nodes(children));

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const usd = new Intl.NumberFormat('en-US', {style: 'currency', currency: 'USD'});
const money = (v) => (!isNum(v) ? DASH : v > 0 && v < 0.005 ? '<$0.01' : usd.format(v));
const count = (v) => (isNum(v) ? Math.round(v).toLocaleString('en-US') : DASH);
const minutes = (v) => (isNum(v) ? v.toFixed(1) : DASH);
const multiple = (v) => (isNum(v) ? v.toFixed(v >= 10 ? 1 : 2) : DASH);
const times = (v) => (isNum(v) ? `${multiple(v)}×` : DASH);
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const titleCase = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const statusLabel = (s) => String(s || '').replace(/_/g, ' ');
const shown = (v) => (typeof v === 'string' ? v : JSON.stringify(v));
const pad2 = (n) => String(n).padStart(2, '0');
const sum = (list, fn = (x) => x) => list.reduce((total, item) => total + (fn(item) || 0), 0);
const enc = encodeURIComponent;

function tokens(v) {
  if (!isNum(v)) return DASH;
  const a = Math.abs(v);
  if (a >= 1e6) return `${(v / 1e6).toFixed(a >= 1e7 ? 1 : 2)}M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(a >= 1e4 ? 0 : 1)}k`;
  return String(Math.round(v));
}

function pct(v) {
  if (!isNum(v)) return DASH;
  return `${(v * 100).toFixed(v > 0 && v < 0.1 ? 1 : 0)}%`;
}

function when(iso) {
  if (!iso) return DASH;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, {dateStyle: 'medium', timeStyle: 'short'});
}

function clockTime(t) {
  const d = new Date(t);
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
}

function hms(ms) {
  if (!isNum(ms) || ms < 0) return DASH;
  const s = Math.floor(ms / 1000);
  return `${pad2(Math.floor(s / 3600))}:${pad2(Math.floor(s / 60) % 60)}:${pad2(s % 60)}`;
}

function duration(min) {
  if (!isNum(min)) return DASH;
  if (min < 1) return `${Math.max(1, Math.round(min * 60))}s`;
  if (min < 60) return `${Math.round(min)} min`;
  return `${Math.floor(min / 60)}h ${pad2(Math.round(min % 60))}m`;
}

/** A round axis maximum at or above v. */
function niceCeil(v) {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  const n = v / p;
  return [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find((step) => n <= step + 1e-9) * p;
}

const outcomeOf = (t) => (t.error ? 'error' : t.passed === true ? 'pass' : t.passed === false ? 'fail' : 'unscored');
const trialCost = (t) => (isNum(t.computed_cost) ? t.computed_cost : isNum(t.reported_cost) ? t.reported_cost : null);
const trialKey = (t) => `${t.arm}/${t.trial || t.task}`;

const note = (text) => h('small', null, text);
const warn = (text) => h('small', {class: 'warn-text'}, text);
const led = (cls = '') => h('span', {class: `led ${cls}`.trim()});
const tag = (text, cls = '') => h('span', {class: `tag ${cls}`.trim()}, text);
const statusTag = (status) => tag(statusLabel(status), STATUS_TAG[status] || '');
const eyebrow = (text) => h('div', {class: 'eyebrow'}, text);
const thead = (...cols) => h('thead', null, h('tr', null, cols.map((c) => (typeof c === 'string' ? h('th', null, c) : c))));
const thNum = (label) => h('th', {class: 'num'}, label);
const tdNum = (text, title) => h('td', {class: 'num', title: title || null}, text);
const cell = (outcome, extra = '') => h('span', {class: `cell ${outcome} ${extra}`.trim()});

function panel({code, title, meta = null, cls = '', bodyId = null, flush = false}, ...body) {
  return h('section', {class: `panel ${cls}`.trim()},
    h('div', {class: 'panel-head'},
      code && h('span', {class: 'panel-code'}, code),
      h('h2', {class: 'panel-title'}, title),
      meta !== null && h('div', {class: 'panel-meta'}, meta)),
    h('div', {class: flush ? 'panel-body flush' : 'panel-body', id: bodyId}, body));
}

function field(label, control, {dot = false, title = null, notes = [], raw = false} = {}) {
  return h('label', {class: 'field', title},
    h('span', {class: raw ? 'label raw' : 'label'}, label, dot && h('span', {class: 'dot', title: 'Changed in the dashboard'})),
    control, notes);
}

/** A select over [value, label] pairs. A current value missing from the list is kept as an extra option. */
function select(name, choices, value, attrs = {}) {
  const list = choices.slice();
  if (value !== null && value !== undefined && !list.some(([v]) => v === value)) list.push([value, value]);
  return h('select', {name, ...attrs, value}, list.map(([v, label]) => h('option', {value: v}, label)));
}

function numberInput(name, value, attrs = {}) {
  return h('input', {type: 'number', name, value: value === null || value === undefined ? '' : value, ...attrs});
}

function formActions(onReset) {
  return h('div', {class: 'actions'},
    h('button', {type: 'submit', class: 'btn primary'}, 'Save'),
    onReset && h('button', {type: 'button', class: 'btn ghost', onclick: onReset}, 'Reset'),
    h('span', {class: 'unsaved', hidden: true}, '● Unsaved changes'));
}

/** Last rendered JSON per section, so polls only touch what changed. */
const rendered = new Map();
function changed(key, data) {
  const json = JSON.stringify(data);
  if (rendered.get(key) === json) return false;
  rendered.set(key, json);
  return true;
}

// Server -----------------------------------------------------------------

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }

  /** 503: a config file or the ledger is busy or locked. Try again in a moment; the bench is not offline. */
  get busy() {
    return this.status === 503;
  }

  /** 409 from a config save: the YAML changed on disk since it was loaded. Reload, then save again. */
  get staleConfig() {
    return this.status === 409 && /changed on disk/i.test(this.message);
  }
}

function errorDetail(data, status) {
  const detail = data && data.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
  if (typeof data === 'string' && data.trim()) return data.trim().slice(0, 200);
  if (status === 503) return 'The bench is busy (a config file or the ledger is locked). Try again in a moment.';
  if (status === 409) return 'The bench refused this because something changed or is in use (HTTP 409). Reload and try again.';
  return `Request failed (HTTP ${status})`;
}

/** True once demo.js is loaded and answers every request: its registry exists and its fetch wrapper is installed. */
const demoGuarded = () => Boolean(window.benchDemo) && window.fetch !== NATIVE_FETCH;

async function api(method, path, body) {
  // Demo mode promises that nothing billed is sent. If demo.js failed to load, refuse here rather than reach the server.
  const bare = new URL(path, location.href).pathname;
  if (DEMO && method !== 'GET' && BILLED_PATHS.some((re) => re.test(bare)) && !demoGuarded()) {
    throw new ApiError(409, 'Demo mode: the demo scripts did not load, so nothing that bills is sent.');
  }
  const init = {method, headers: {}, credentials: 'same-origin', cache: 'no-store'};
  if (method !== 'GET') init.headers['X-Bench'] = '1';
  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, 'Cannot reach the bench. Is it still running?');
  }
  const isJson = (res.headers.get('content-type') || '').includes('application/json');
  const data = isJson ? await res.json().catch(() => null) : await res.text();
  if (!res.ok) throw new ApiError(res.status, errorDetail(data, res.status));
  return data;
}

let toastTimer = 0;
/** A short message. `action` ({label, run}) adds a button, for example "Open LIVE", and keeps the toast up longer. */
function toast(message, bad = false, action = null) {
  const el = $('#toast');
  const hide = () => { el.className = bad ? 'bad' : ''; };
  fill(el, message, action && h('button', {type: 'button', class: 'toast-action', onclick: () => {
    hide();
    action.run();
  }}, action.label));
  el.className = ['show', bad && 'bad', action && 'actionable'].filter(Boolean).join(' ');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(hide, action ? 9000 : bad ? 7000 : 3500);
}

/** Runs an action with its button disabled and shows any error in the toast. */
async function busy(button, action) {
  if (button) button.disabled = true;
  try {
    return await action();
  } catch (err) {
    toast(err.message, true);
    return undefined;
  } finally {
    if (button) button.disabled = false;
  }
}

/** A styled confirm. `body` is text or nodes. Resolves true only when the confirm button was pressed. */
function confirmAction({code = 'CONFIRM', title, body, ok = 'Confirm', tone = 'armed'}) {
  const dialog = $('#confirm-dialog');
  $('#confirm-code').textContent = code;
  $('#confirm-title').textContent = title;
  fill($('#confirm-body'), body);
  const okButton = $('#confirm-ok');
  okButton.textContent = ok;
  okButton.className = tone === 'danger' ? 'btn danger' : 'btn primary';
  dialog.className = tone;
  dialog.returnValue = '';
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener('close', () => resolve(dialog.returnValue === 'ok'), {once: true});
  });
}

// Tooltip ----------------------------------------------------------------

const tips = new WeakMap();
let tipTarget = null;

/** Attaches hover and focus details to a mark. Rows are [value, label] pairs or plain strings. */
function tip(el, title, rows = [], focusable = false) {
  tips.set(el, {title, rows});
  el.classList.add('has-tip');
  if (focusable) {
    el.tabIndex = 0;
    // A mark made focusable only for its tooltip (a bar, a matrix cell) has no text of its own: name it from the tooltip.
    if (!el.textContent.trim() && !el.hasAttribute('aria-label')) {
      const said = rows.filter(Boolean).map((r) => (Array.isArray(r) ? `${r[1]}: ${r[0]}` : String(r)));
      el.setAttribute('aria-label', [title, ...said].filter(Boolean).join('; '));
      if (!el.hasAttribute('role')) el.setAttribute('role', 'img');
    }
  }
  return el;
}

function showTip(title, rows, x, y) {
  const box = $('#tooltip');
  fill(box,
    title && h('div', {class: 'tt-title'}, title),
    rows.map((row) => (Array.isArray(row)
      ? h('div', {class: 'tt-row'}, h('b', null, row[0]), h('span', null, row[1]))
      : h('div', {class: 'tt-text'}, row))));
  box.hidden = false;
  const pad = 14;
  const w = box.offsetWidth;
  const ht = box.offsetHeight;
  box.style.left = `${Math.min(x + pad, window.innerWidth - w - 8)}px`;
  box.style.top = `${y + pad + ht > window.innerHeight ? y - ht - pad : y + pad}px`;
}

function hideTip() {
  tipTarget = null;
  $('#tooltip').hidden = true;
}

function setupTooltip() {
  document.addEventListener('pointerover', (e) => {
    const el = e.target.closest && e.target.closest('.has-tip');
    if (!el || !tips.has(el)) return;
    tipTarget = el;
    const {title, rows} = tips.get(el);
    showTip(title, rows, e.clientX, e.clientY);
  });
  document.addEventListener('pointermove', (e) => {
    if (!tipTarget) return;
    if (!tipTarget.isConnected) { hideTip(); return; }
    const {title, rows} = tips.get(tipTarget);
    showTip(title, rows, e.clientX, e.clientY);
  });
  document.addEventListener('pointerout', (e) => {
    if (tipTarget && !tipTarget.contains(e.relatedTarget)) hideTip();
  });
  document.addEventListener('focusin', (e) => {
    const el = e.target.closest && e.target.closest('.has-tip');
    if (!el || !tips.has(el)) return;
    tipTarget = el;
    const r = el.getBoundingClientRect();
    const {title, rows} = tips.get(el);
    showTip(title, rows, r.right, r.bottom);
  });
  document.addEventListener('focusout', hideTip);
}

// Areas ------------------------------------------------------------------
//
// Every tab is a view registered here: LIVE, LAUNCH and ANALYSIS at the end of
// this file, REPORT, DESIGNS and QUEUE by their own files. The assistant panel
// registers as a panel. Each area file is one IIFE that registers itself when
// it loads, so nothing it declares can clash with another file. The hooks are
// called through callArea, so one area's error is logged and never stops the
// poll or another area.

const areas = new Map();     // name -> hooks
const pollers = [];

/**
 * Registers a tab. `name` is one of VIEWS; the view draws into <section id="view-<name>">.
 *   render(state, force)  Draws the view from /api/state: with force=true when the view is shown (or go() reopens
 *                         it), and with force=false after every poll while it stays shown. Use changed() to skip
 *                         work. Even with force, never rebuild a form, rail, editor or composer holding unsent input.
 *   open(params)          go(name, params) hands navigation parameters here first, shown or not.
 *   show() / hide()       The view became visible / another view replaced it. Called after render(state, true).
 *   onState(state, prev)  After every poll, whatever view is shown. prev is null on the first poll.
 *   hold()                While the area holds unsent input: a short reason, otherwise ''. A run starting then
 *                         does not switch the page to LIVE, and leaving the page asks first.
 *   fastPoll()            true asks for the 3 s /api/state poll while this view is shown.
 *   setup()               Once, before the first poll.
 * Other functions in `hooks` are reached through the helpers under Navigation, never called directly.
 */
function registerView(name, hooks) {
  if (!VIEWS.includes(name)) throw new Error(`registerView: no view named ${name}`);
  areas.set(name, hooks);
}

/**
 * Registers a part that is not a tab (the assistant panel). It takes the setup, onState and hold hooks, plus
 * escape(), which returns true when it handled the Escape key (for example by closing itself).
 */
function registerPanel(name, hooks) {
  if (VIEWS.includes(name)) throw new Error(`registerPanel: ${name} is a view`);
  areas.set(name, hooks);
}

/** Calls one hook of one area. Returns undefined when it has none, or when it threw (logged). An async hook's rejection is logged too. */
function callArea(name, hook, ...args) {
  const hooks = areas.get(name);
  if (!hooks || typeof hooks[hook] !== 'function') return undefined;
  try {
    const out = hooks[hook](...args);
    if (out && typeof out.then === 'function') out.then(null, (err) => console.error(`${name}.${hook} failed:`, err));
    return out;
  } catch (err) {
    console.error(`${name}.${hook} failed:`, err);
    return undefined;
  }
}

/** Why the page should not jump to another view by itself: the first area holding unsent input, or ''. */
function holdReason() {
  for (const name of areas.keys()) {
    const reason = callArea(name, 'hold');
    if (reason) return String(reason);
  }
  return '';
}

/**
 * Area polling beside /api/state. run() is awaited and never overlaps itself. It runs at once when when() turns
 * true (its view opening), then every `ms` while when() stays true. The handle's now() runs it on demand, after
 * one in flight if there is one, and resolves when that run is done. run() shows its own errors; a throw is logged.
 */
function poller({ms, when = () => true, run}) {
  const p = {ms, when, run, last: 0, on: false, running: null, again: false};
  pollers.push(p);
  return {now: () => runPoller(p)};
}

function runPoller(p) {
  if (p.running) {
    p.again = true;
    return p.running;
  }
  p.running = (async () => {
    await null;  // so p.running is set before the first run starts
    do {
      p.again = false;
      p.last = Date.now();
      try {
        await p.run();
      } catch (err) {
        console.warn('A poll failed:', err);
      }
    } while (p.again);
    p.running = null;
  })();
  return p.running;
}

/** Every second, and whenever the view changes. */
function checkPollers() {
  const now = Date.now();
  for (const p of pollers) {
    let on = false;
    try {
      on = Boolean(p.when());
    } catch { /* treated as off */ }
    if (on && (!p.on || now - p.last >= p.ms)) runPoller(p);
    p.on = on;
  }
}

// Navigation between areas -----------------------------------------------

/** Shows a view, first handing `params` to its open hook (even when it is already shown). */
function go(name, params = null) {
  if (!VIEWS.includes(name)) return;
  if (params) callArea(name, 'open', params);
  if (ui.view === name) renderView(true);
  else showView(name);
}

/** A run: LIVE while it is the active run, otherwise ANALYSIS on it. */
function openRun(runId) {
  if (ui.state && runId && runId === ui.state.active_run) showView('live');
  else openAnalysis(runId);
}

/** REPORT; with filters, those replace the rail's (options kept), e.g. openReport({runs: [runKey(id)]}). */
const openReport = (filters = null) => go('report', filters ? {filters} : null);
/** DESIGNS on one version; the latest when version is null. */
const openDesign = (id, version = null) => go('designs', {id, version});
/** QUEUE, optionally scrolled to one item. */
const openQueue = (item = null) => go('queue', item ? {item} : null);

/**
 * Opens the assistant panel. opts: {conversationId} shows that conversation; {design: {id, version}} starts or
 * attaches a design; {prompt} puts text in the composer without sending it. Several may be combined.
 */
const openAssistant = (opts = {}) => callArea('assistant', 'open', opts);
const closeAssistant = () => callArea('assistant', 'close');
const assistantOpen = () => callArea('assistant', 'isOpen') === true;

/** The design version DESIGNS shows, {id, version, title, edited}, or null. For "Attach open design". */
const shownDesign = () => callArea('designs', 'current') || null;
/** The latest QueueView QUEUE has loaded (GET /api/queue), or null. */
const queueView = () => callArea('queue', 'data') || null;
/** Reloads the queue now (after a change from another area). Resolves when done. */
const refreshQueue = () => Promise.resolve(callArea('queue', 'refresh'));
/** Reloads the design list now. Resolves when done. */
const refreshDesigns = () => Promise.resolve(callArea('designs', 'refresh'));
/** Fills `el` with QUEUE's compact summary, for the DESIGNS column on wide screens. */
const queueSummary = (el) => callArea('queue', 'summary', el);
/** The providers these queue items bill: {bills: [provider], unknown: [label]} (unknown: could not be loaded in time). */
const queueBills = (items) => Promise.resolve(callArea('queue', 'bills', items)).then((out) => out || {bills: [], unknown: items.map((i) => `${i.title || i.design_id} v${i.version}`)});

// Shared components ------------------------------------------------------
//
// Used by more than one area. Their signatures are a contract between the
// area files: extend them only with new optional arguments.

const STALE = Symbol('stale');

/** "a", "a and b", "a, b and c". */
function listText(items) {
  const list = items.filter(Boolean).map(String);
  return list.length < 2 ? list.join('') : `${list.slice(0, -1).join(', ')} and ${list[list.length - 1]}`;
}

/** The first 12 hex characters of a "sha256:<64 hex>" spec hash. */
const shortHash = (hash) => (typeof hash === 'string' && hash ? hash.replace(/^sha256:/, '').slice(0, 12) : DASH);

/** "$0.11 p50 · $0.23 p90". */
const estimateText = (p50, p90) => (isNum(p50) || isNum(p90) ? `${money(p50)} p50 · ${money(p90)} p90` : DASH);

/** <profile>/<run id>, the ledger's and the report's name for a run of this profile. */
const runKey = (runId) => `${(ui.state && ui.state.config.profile) || 'default'}/${runId}`;

/**
 * True for the shape a block takes when building it failed: {error} and nothing else. A block that works can carry
 * its own `error` field too (the ledger's {available: false, error, trials, …}), and that one is not a failure.
 */
const failedBlock = (b) => Boolean(b && typeof b === 'object' && b.error && Object.keys(b).length === 1);

/** A /api/state block that can fail on its own (ledger, queue, assistant): the block, or null when missing or failed. */
function stateBlock(name) {
  const b = ui.state && ui.state[name];
  return b && typeof b === 'object' && !failedBlock(b) ? b : null;
}

/** Why that block failed ("<what>: <type>: <message>"), or ''. */
function stateError(name) {
  const b = ui.state && ui.state[name];
  if (!ui.state) return '';
  if (!b || typeof b !== 'object') return `${name}: not in /api/state`;
  return failedBlock(b) ? String(b.error) : '';
}

/**
 * "$7.50 left" of the global cap, from state.queue or a QueueView. state.queue has no global_cap_usd, so when its
 * cap_remaining_usd is null QUEUE's last QueueView tells "no cap" from "a cap whose spend could not be read".
 */
function capLeftText(q) {
  if (!q) return DASH;
  if (isNum(q.cap_remaining_usd)) return `${money(q.cap_remaining_usd)} left`;
  const v = 'global_cap_usd' in q ? q : queueView();
  if (!v || !('global_cap_usd' in v)) return 'unknown';
  if (isNum(v.global_cap_usd)) return 'unknown';  // a cap whose spend could not be read
  return 'no global cap';
}

/** A trial's cost or tokens under its usage_status: "not recorded" when unmeasured, "≥ …" when a lower bound. */
function usageValue(value, status, format = money) {
  if (status === 'unmeasured') return 'not recorded';
  if (!isNum(value)) return DASH;
  return status === 'lower_bound' ? `≥ ${format(value)}` : format(value);
}

/** "AI" for an assistant-authored design or version, "You" for one of yours. */
const authorBadge = (author) => (author === 'assistant'
  ? tag('AI', 'ai')
  : tag(author === 'user' ? 'You' : author || DASH));

/**
 * "title vN" that opens that design version in DESIGNS. `ref` is anything with design_id (or id) and version:
 * a QueueItem, a run's design_ref, an assistant draft, a DesignSummary with version set.
 */
function designChip(ref, {author = false, hash = false} = {}) {
  const id = ref.design_id || ref.id;
  const label = `${ref.title || id} v${ref.version}`;
  return tip(h('button', {type: 'button', class: 'design-chip', onclick: (e) => {
    e.stopPropagation();
    openDesign(id, ref.version);
  }},
  author && ref.author && authorBadge(ref.author),
  h('span', {class: 'dc-title'}, label),
  hash && ref.spec_hash && h('span', {class: 'dc-hash'}, shortHash(ref.spec_hash))),
  label, [[id, 'design'], ref.spec_hash ? [shortHash(ref.spec_hash), 'spec hash'] : null, 'Opens it in DESIGNS'].filter(Boolean));
}

/**
 * A modal made on the fly and removed when it closes, for input: a snapshot name, an exclusion reason, a cap,
 * a mode and its arms. `body` is nodes (field() works). On confirm, read(form) runs: it returns the dialog's
 * value, and may be async (send the request there); a thrown Error shows inside the dialog, which stays open.
 * Native validation (required, min, max) runs first. `focus` is a selector inside the dialog, or 'cancel'.
 * Resolves the value (true when read returns undefined), or null when cancelled.
 */
let dialogSeq = 0;
function formDialog({code = 'INPUT', title, body = [], ok = 'Save', tone = '', read = () => true, focus = null}) {
  const id = `dlg-${++dialogSeq}`;
  const error = h('p', {class: 'bad-text dialog-error', role: 'alert', hidden: true});
  const cancel = h('button', {type: 'button', class: 'btn ghost'}, 'Cancel');
  const confirm = h('button', {type: 'submit', class: tone === 'danger' ? 'btn danger' : 'btn primary'}, ok);
  const form = h('form', null,
    h('span', {class: 'panel-code dialog-code', 'aria-hidden': 'true'}, code),
    h('h3', {id: `${id}-title`}, title),
    h('div', {class: 'dialog-body', id: `${id}-body`}, body),
    error,
    h('div', {class: 'actions'}, cancel, confirm));
  // Named and described, so a screen reader landing on Cancel still hears the title and what the dialog says.
  const dialog = h('dialog', {class: ['dyn', tone].filter(Boolean).join(' '), 'aria-labelledby': `${id}-title`, 'aria-describedby': `${id}-body`}, form);
  const opener = document.activeElement !== document.body ? document.activeElement : null;
  let value = null;
  let pending = false;
  // Once a request is sent it may already be acting, so the dialog stays until the answer: Escape is refused,
  // and if the browser closes it anyway (Chrome lets a second Escape through), it opens again.
  const lock = (on) => {
    pending = on;
    if ('closedBy' in dialog) dialog.closedBy = on ? 'none' : 'closerequest';
    confirm.disabled = on;
    cancel.disabled = on;
  };
  document.body.append(dialog);
  return new Promise((resolve) => {
    dialog.addEventListener('close', () => {
      if (pending) {
        dialog.showModal();
        return;
      }
      dialog.remove();
      resolve(value);
      // Back to the control that opened it, once the caller has enabled it again.
      if (opener) setTimeout(() => {
        const lost = !document.activeElement || document.activeElement === document.body;
        if (lost && opener.isConnected && !opener.disabled && opener.getClientRects().length) opener.focus({preventScroll: true});
      }, 0);
    });
    dialog.addEventListener('cancel', (e) => {
      if (pending) e.preventDefault();
    });
    cancel.addEventListener('click', () => dialog.close());
    form.addEventListener('input', () => { error.hidden = true; });
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (pending) return;
      lock(true);
      error.hidden = true;
      try {
        const result = await read(form);
        value = result === undefined ? true : result;
        lock(false);
        dialog.close();
      } catch (err) {
        error.textContent = err && err.message ? err.message : String(err);
        error.hidden = false;
      } finally {
        if (pending) lock(false);
      }
    });
    dialog.showModal();
    const first = focus === 'cancel' ? cancel
      : (focus && dialog.querySelector(focus)) || dialog.querySelector('.dialog-body input, .dialog-body select, .dialog-body textarea') || confirm;
    first.focus();
  });
}

/**
 * The two-step gate for every billed action: LAUNCH, Add to queue, Start now, Start queue and Retry. Step one
 * is the button that called this; step two is the dialog's confirm button. The dialog opens with Cancel focused,
 * so a stray Enter bills nothing.
 *   summary   one sentence at the top
 *   rows      [[label, value]] as a parameter grid; value is text or a node; falsy rows are skipped
 *   warnings  warn-text lines, e.g. the uncapped arms by name; falsy entries are skipped
 *   lines     plain sentences after the grid, e.g. "Starts immediately (front of the queue)."
 *   bills     provider names, shown as "This bills anthropic and openrouter."
 *   send      optional async () => result, run on confirm. The dialog stays open until it settles and shows a
 *             thrown error inside itself (the person can confirm again or cancel). Must resolve a truthy value.
 * Resolves send's result (true without send) once confirmed, or null when cancelled.
 */
function armConfirm({title, summary = '', rows = [], warnings = [], lines = [], bills = [], ok = '▶ Confirm', send = null}) {
  const shownRows = rows.filter(Boolean);
  return formDialog({
    code: 'ARM', title, ok, tone: 'armed', focus: 'cancel',
    body: [
      summary && h('p', null, summary),
      shownRows.length > 0 && h('dl', {class: 'params arm-params'},
        shownRows.map(([label, value]) => h('div', null, h('dt', null, label), h('dd', null, value)))),
      warnings.filter(Boolean).map((w) => h('p', {class: 'warn-text arm-line'}, h('span', null, '!'), h('span', null, w))),
      lines.filter(Boolean).map((line) => h('p', {class: 'arm-line'}, line)),
      bills.length > 0 && h('p', {class: 'arm-bills'}, `This bills ${listText(bills)}.`),
    ],
    read: send ? () => send() : () => true,
  });
}

/**
 * Add to queue (start=false) or Start now (start=true) for one saved design version: the ARM confirm, then
 * POST /api/queue/items with the version's exact spec_hash. `detail` is a VersionDetail: from
 * GET /api/designs/{id}/versions/{n}, or the 201 of a save, of from-config or of from-run.
 * Resolves {item, queue} once added, or null when cancelled. Errors show inside the dialog. Start now also starts
 * the queue, so its confirm lists the waiting designs that run after this one and bills their providers too.
 */
async function enqueueDesign(detail, {start = false} = {}) {
  const check = detail.check || {};
  const plan = check.plan || null;
  const est = check.estimate ? check.estimate.run : null;
  const problems = (check.validation && check.validation.problems) || [];
  const spec = detail.spec || {arms: [], run: {}};
  const uncapped = est ? est.uncapped_arms : plan ? plan.uncapped_arms : [];
  const worst = est ? est.worst_case_capped_usd : plan ? plan.worst_case_capped_usd : null;
  // Start now starts the whole queue: once this design is done, every waiting design runs in order. So the
  // confirm names them, their budgets and their providers too, from a fresh GET /api/queue.
  await Promise.race([refreshQueue().catch(() => null), new Promise((r) => setTimeout(r, 4000))]);
  const v = queueView();
  const q = v || stateBlock('queue');
  const running = Boolean(q && q.state === 'running');
  const same = (i) => i.design_id === detail.design_id && i.version === detail.version;
  const after = start && v ? v.waiting.filter((i) => !same(i)).sort((a, b) => a.position - b.position) : [];
  const more = after.length ? await queueBills(after) : {bills: [], unknown: []};
  const other = ui.state && ui.state.active_run && !(v && v.active && v.active.run_id === ui.state.active_run) ? ui.state.active_run : null;
  const busyNow = other || (v && v.active ? v.active.run_id || v.active.id : null);
  const order = after.slice(0, 6).map((i) => `${i.title || i.design_id} v${i.version}`).join(', ') + (after.length > 6 ? ` and ${after.length - 6} more` : '');
  const startLine = busyNow ? `Goes to the front of the queue and starts once run ${busyNow} ends.` : 'Starts immediately (front of the queue).';
  const addLine = running ? 'Joins the end of the queue. It runs when the queue reaches it.'
    : 'Joins the end of the queue. The queue is paused: nothing starts until you press Start queue.';
  return armConfirm({
    title: start ? 'Start this design now?' : 'Add this design to the queue?',
    rows: [
      ['Design', `${detail.title} v${detail.version}`],
      ['Spec hash', shortHash(detail.spec_hash)],
      ['Arms × trials', plan ? `${plural(plan.arms.length, 'arm')} · ${plural(plan.trials, 'trial')}` : DASH],
      ['Budget', money(spec.run.budget_usd)],
      ['Estimate', est ? estimateText(est.total.p50, est.total.p90) : DASH],
      ['Worst case, capped', money(worst)],
      ['In-flight allowance', est ? money(est.inflight_allowance_usd) : DASH],
      ['Global cap', capLeftText(q)],
    ],
    warnings: [
      ...problems.map((p) => `Problem: ${p}`),
      uncapped && uncapped.length > 0 && `No dollar cap: ${uncapped.join(', ')}. The run budget still applies after every trial.`,
      after.length > 0 && `${running ? 'The queue is running' : 'This also starts the queue'}: after this design, `
        + `${plural(after.length, 'waiting design')} ${after.length === 1 ? 'runs' : 'run in order'} (${order}), with ${after.length === 1 ? 'budget' : 'budgets'} ${money(sum(after, (i) => i.budget_usd))} `
        + `plus in-flight allowance ${money(sum(after, (i) => i.inflight_allowance_usd))}. To run only this one, press Pause on QUEUE once it starts.`,
      start && v && v.held.length > 0 && `${plural(v.held.length, 'held item')} ${v.held.length === 1 ? 'stays' : 'stay'} held until you retry ${v.held.length === 1 ? 'it' : 'them'}.`,
      more.unknown.length > 0 && `Could not load the providers of ${listText(more.unknown)}.`,
      start && !v && 'The queue could not be loaded, so the designs waiting behind this one are not shown.',
    ],
    lines: [start ? startLine : addLine],
    bills: [...new Set([...spec.arms.map((a) => a.provider), ...more.bills].filter(Boolean))],
    ok: start ? '▶ Start now' : '▶ Add to queue',
    send: async () => {
      let out;
      try {
        out = await api('POST', '/api/queue/items', {
          design_id: detail.design_id, version: detail.version, spec_hash: detail.spec_hash,
          position: start ? 'front' : 'end', start,
        });
      } catch (err) {
        if (/^Added as /.test(err.message)) {  // added, but the start failed
          refreshQueue();
          poll();
        }
        throw err;
      }
      // Started, but it can be paused again already: this item held on the global cap, say. Then say why.
      const paused = start && out.queue && out.queue.state !== 'running';
      if (paused) toast(`${out.item.id} added, but the queue paused: ${out.queue.pause_reason || out.item.message || 'see QUEUE'}`, true);
      else toast(start ? `${out.item.id} added and the queue started` : `Added as ${out.item.id}`);
      refreshQueue();
      refreshDesigns();
      poll();
      return out;
    },
  });
}

/** fn once `ms` pass without another call. The returned function has cancel(), flush() and pending(). */
function debounce(fn, ms) {
  let timer = 0;
  let args = [];
  const run = () => {
    timer = 0;
    fn(...args);
  };
  const wrapped = (...a) => {
    args = a;
    clearTimeout(timer);
    timer = setTimeout(run, ms);
  };
  wrapped.cancel = () => {
    clearTimeout(timer);
    timer = 0;
  };
  wrapped.flush = () => {
    if (!timer) return;
    clearTimeout(timer);
    run();
  };
  wrapped.pending = () => Boolean(timer);
  return wrapped;
}

/**
 * Keeps only the newest answer: `const fresh = latest();` then `const data = await fresh(api(...));`.
 * An older request resolves to STALE once a newer one was made, and its errors are dropped.
 */
function latest() {
  let seq = 0;
  return async (promise) => {
    const mine = ++seq;
    try {
      const value = await promise;
      return mine === seq ? value : STALE;
    } catch (err) {
      if (mine !== seq) return STALE;
      throw err;
    }
  };
}

/** Per-profile browser storage for view preferences (filters, chosen tabs). Never throws; reads fall back. */
const prefs = {
  // Demo mode keeps its own: a demo design or conversation must not be reopened on the real bench.
  key: (name) => `bench.${(ui.state && ui.state.config.profile) || 'default'}.${DEMO ? 'demo.' : ''}${name}`,
  get(name, fallback = null) {
    try {
      const raw = localStorage.getItem(prefs.key(name));
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(name, value) {
    try {
      if (value === null || value === undefined) localStorage.removeItem(prefs.key(name));
      else localStorage.setItem(prefs.key(name), JSON.stringify(value));
    } catch { /* storage unavailable: the preference lasts until reload */ }
  },
};

/** Saves a file a POST route returns (the CSV and JSONL exports). Resolves the file name; errors are ApiErrors. */
async function download(path, body = {}) {
  let res;
  try {
    res = await fetch(path, {
      method: 'POST', credentials: 'same-origin', cache: 'no-store',
      headers: {'X-Bench': '1', 'Content-Type': 'application/json'}, body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, 'Cannot reach the bench. Is it still running?');
  }
  if (!res.ok) {
    const isJson = (res.headers.get('content-type') || '').includes('application/json');
    const data = isJson ? await res.json().catch(() => null) : await res.text();
    throw new ApiError(res.status, errorDetail(data, res.status));
  }
  const blob = await res.blob();
  const disposition = res.headers.get('content-disposition') || '';
  const match = disposition.match(/filename\*=UTF-8''([^;]+)/i) || disposition.match(/filename="?([^";]+)"?/i);
  const name = match ? decodeURIComponent(match[1].trim()) : 'download';
  const url = URL.createObjectURL(blob);
  const link = h('a', {href: url, download: name, hidden: true});
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);  // after the click has handed the file to the browser
  return name;
}

/** Copies text and says so in the toast. Resolves false when the browser refused. */
async function copyText(text, done = 'Copied') {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = h('textarea', {style: 'position:fixed;left:-9999px;opacity:0', value: text});
    document.body.append(area);
    area.select();
    const ok = document.execCommand('copy');
    area.remove();
    if (!ok) {
      toast('Could not copy. Select the text and copy it by hand.', true);
      return false;
    }
  }
  toast(done);
  return true;
}

// Views and polling ------------------------------------------------------

function showView(name, {refresh = true} = {}) {
  if (!VIEWS.includes(name)) name = 'launch';
  const previous = ui.view;
  ui.view = name;
  for (const button of document.querySelectorAll('.views button')) {
    const on = button.dataset.view === name;
    button.classList.toggle('active', on);
    button.setAttribute('aria-selected', String(on));
    button.tabIndex = on ? 0 : -1;  // one tab stop for the tab list; the arrow keys move between tabs (init)
  }
  for (const view of VIEWS) $(`#view-${view}`).hidden = view !== name;
  if (location.hash !== `#${name}`) history.replaceState(null, '', `${location.pathname}${location.search}#${name}`);
  hideTip();
  if (previous && previous !== name) callArea(previous, 'hide');
  if (ui.state) renderView(true);
  if (previous !== name) callArea(name, 'show');
  checkPollers();
  if (refresh) poll();
}

function renderView(force = false) {
  if (ui.state) callArea(ui.view, 'render', ui.state, force);
}

let pollTimer = 0;
let pollSeq = 0;

async function poll() {
  clearTimeout(pollTimer);
  const seq = ++pollSeq;
  let state = null;
  let error = null;
  try {
    state = await api('GET', '/api/state');
  } catch (err) {
    error = err;
  }
  if (seq !== pollSeq) return;  // a newer poll took over
  try {
    if (state) applyState(state);
    else if (error.busy && ui.state) setBusy(error);
    else setDisconnected(error);
  } finally {
    pollTimer = setTimeout(poll, pollInterval());  // even if a render threw
  }
}

const pollInterval = () => (ui.view === 'live' || (ui.state && ui.state.active_run) || callArea(ui.view, 'fastPoll') === true
  ? FAST_POLL_MS : SLOW_POLL_MS);

function applyState(state) {
  const previous = ui.state;
  ui.state = state;
  ui.connected = true;
  ui.busyNote = '';
  ui.lastSync = Date.now();
  renderHeader();
  renderConfigError();
  for (const name of areas.keys()) callArea(name, 'onState', state, previous);
  fitSystems();  // QUEUE's header chip, drawn in its onState, changes the room the header has left
  if (!previous) {
    const hash = location.hash.slice(1);
    showView(VIEWS.includes(hash) ? hash : state.active_run ? 'live' : 'launch', {refresh: false});
  } else if (state.active_run && state.active_run !== previous.active_run) {
    const run = activeRun();
    const fromQueue = Boolean(run && run.source === 'design');
    const hold = ui.view === 'live' ? '' : holdReason();
    if (hold) {
      // A queued run starts without a click; never pull the page away from unsent input.
      toast(`Run ${state.active_run} started${fromQueue ? ' from the queue' : ''}`, false, {label: 'Open LIVE', run: () => showView('live')});
      renderView(false);
    } else {
      toast(`Run ${state.active_run} is live${fromQueue ? ' (from the queue)' : ''}`);
      showView('live', {refresh: false});
    }
  } else {
    renderView(false);
  }
  trackLive();
  refreshDrawerReadOnly();
}

function setDisconnected(error) {
  if (ui.connected) toast(error.message, true);
  ui.connected = false;
  renderPill(error.message);
}

/** /api/state answered 503: something is locked for a moment. The page keeps the last state and says so, but stays online. */
function setBusy(error) {
  if (!ui.busyNote) toast(error.message);
  ui.busyNote = error.message;
  renderPill(error.message);
}

// Header -----------------------------------------------------------------

const activeRun = () => (ui.state ? ui.state.runs.find((r) => r.id === ui.state.active_run) || null : null);
const findArm = (name) => ui.state.config.arms.find((a) => a.name === name) || null;
const armedNames = () => ui.state.config.arms.filter((a) => a.enabled).map((a) => a.name);
const dockerOk = () => {
  const check = ui.state.readiness.find((c) => c.name === 'Docker');
  return !check || check.ok;
};

/** Keys the armed modules need, and how many are set. */
function keyCoverage() {
  const s = ui.state;
  const providers = new Set(s.config.arms.filter((a) => a.enabled).map((a) => a.provider));
  const needed = s.keys.filter((k) => providers.has(k.provider));
  return {needed, set: needed.filter((k) => k.set).length};
}

function renderHeader() {
  const s = ui.state;
  const keys = keyCoverage();
  const ledger = [stateBlock('ledger'), stateError('ledger')];
  if (changed('systems', [s.readiness, keys.set, keys.needed.length, ledger])) {
    const keyLed = !keys.needed.length ? '' : keys.set === keys.needed.length ? 'ok' : 'bad';
    const failing = s.readiness.filter((c) => !c.ok);
    fill($('#systems'),
      // When the header has no room for every chip (fitSystems), the readiness chips fold into this one.
      tip(h('button', {type: 'button', class: 'sys sys-sum', onclick: () => openDrawer('system'),
        'aria-label': `Systems: ${failing.length ? `${failing.length} not ready (${failing.map((c) => c.name).join(', ')})` : 'all ready'}. Opens Setup`},
      led(failing.length ? 'bad' : 'ok'), h('span', {class: 'lbl'}, 'SYS'),
      h('span', {class: 'n'}, `${s.readiness.length - failing.length}/${s.readiness.length}`)),
      'Systems', s.readiness.map((c) => [`${c.ok ? '✓' : '✗'} ${c.detail || ''}`.trim(), c.name])),
      s.readiness.map((c) => tip(h('button', {type: 'button', class: 'sys sys-ready', onclick: () => openDrawer('system'),
        'aria-label': `${c.name}: ${c.ok ? 'ready' : 'not ready'}${c.detail ? `, ${c.detail}` : ''}`},
      led(c.ok ? 'ok' : 'bad'), h('span', {class: 'lbl'}, (SYS_LABEL[c.name] || c.name).toUpperCase())), c.name, [c.detail])),
      tip(h('button', {type: 'button', class: 'sys', onclick: () => openDrawer('system'),
        'aria-label': `API keys: ${keys.needed.length ? `${keys.set} of ${keys.needed.length} set` : 'none needed'}`},
      led(keyLed), h('span', {class: 'lbl'}, 'KEYS'), h('span', {class: 'n'}, `${keys.set}/${keys.needed.length}`)),
      'API keys', keys.needed.length
        ? keys.needed.map((k) => [k.set ? `set ${k.masked && k.masked !== 'set' ? k.masked : ''}` : 'not set', k.env_key])
        : ['No armed module needs a key']),
      ledgerChip());
  }
  const profile = s.config.profile || 'default';
  $('#profile-tag').textContent = profile;
  $('#profile-tag').className = profile === 'default' ? 'tag' : 'tag warn';
  $('#live-led').className = s.active_run ? 'led ok live' : 'led';
  $('#runs-count').textContent = s.runs.length ? String(s.runs.length) : '';
  $('#sl-dataset').textContent = s.config.dataset || DASH;
  $('#sl-baseline').textContent = s.config.baseline_arm || DASH;
  renderPill();
}

function renderPill(problem = '') {
  const pill = $('#status-pill');
  const run = ui.connected ? activeRun() : null;
  $('#link-led').className = ui.connected ? 'led ok' : 'led bad';
  $('#sl-link').textContent = ui.connected ? 'ONLINE' : 'OFFLINE';
  $('#sl-poll').textContent = `${pollInterval() / 1000}S`;
  if (!ui.connected) {
    pill.textContent = 'DISCONNECTED';
    pill.className = 'bad';
  } else if (ui.busyNote) {
    pill.textContent = 'BUSY';
    pill.className = 'busy';
  } else if (ui.state && ui.state.active_run) {
    const budget = run && Number.isInteger(run.budget_usd) ? `$${run.budget_usd}` : run && money(run.budget_usd);
    // The budget part goes first when the header is short of room (fitSystems); the title keeps it.
    fill(pill, run ? ['RUNNING ', money(run.spend_usd), h('span', {class: 'pill-budget'}, ` / ${budget}`)] : 'RUNNING');
    pill.className = 'running';
  } else {
    pill.textContent = 'IDLE';
    pill.className = '';
  }
  pill.title = problem || ui.busyNote || (run ? `Run ${run.id}: ${money(run.spend_usd)} spent of ${money(run.budget_usd)}` : '');
  fitSystems();
}

/**
 * The ledger's chip in #systems: green while it is available, amber while a rescan runs, red with the error
 * (or a failed state block) in the tooltip. It opens REPORT, which reads the ledger.
 */
function ledgerChip() {
  const b = stateBlock('ledger');
  const failed = stateError('ledger');
  const cls = !b ? 'bad' : b.error || !b.available ? 'bad' : b.rescan_running ? 'warn live' : 'ok';
  const rows = !b ? [failed || 'The ledger status is missing.']
    : [
      [count(b.trials), 'trials'],
      [b.last_ingest_at ? when(b.last_ingest_at) : DASH, 'last ingest'],
      b.rescan_running ? '◐ Rescanning the job folders…' : null,
      b.error ? `✗ ${b.error}` : !b.available ? '✗ Not available' : null,
      'Opens REPORT',
    ].filter(Boolean);
  const said = !b ? 'status unknown' : b.error || !b.available ? 'unavailable' : b.rescan_running ? 'rescanning' : `${count(b.trials)} trials`;
  return tip(h('button', {type: 'button', class: 'sys', onclick: () => openReport(), 'aria-label': `Ledger: ${said}. Opens REPORT`},
    led(cls), h('span', {class: 'lbl'}, 'LEDGER'), b && isNum(b.trials) && h('span', {class: 'n'}, count(b.trials))), 'Ledger', rows);
}

/**
 * Makes room in the header, shedding the least important text first until every chip shows and nothing spills:
 * the settings label, the chip names, the brand's kicker, the run budget in the pill, then the readiness chips fold
 * into one SYS chip (KEYS and LEDGER stay), and last the queue chip's "paused". Each step is a class; the CSS hides
 * what it names.
 */
const FIT_STEPS = [['bar', 'tight'], ['systems', 'compact'], ['bar', 'brand-tight'], ['bar', 'pill-short'], ['systems', 'summary'],
  ['bar', 'chip-short']];
function fitSystems() {
  const bar = $('.bar');
  const systems = $('#systems');
  const brand = $('.brand');
  const els = {bar, systems};
  for (const [el, cls] of FIT_STEPS) els[el].classList.remove(cls);
  const room = () => getComputedStyle(systems).display === 'none'
    ? bar.scrollWidth <= bar.clientWidth + 1
    : systems.scrollWidth <= systems.clientWidth + 1 && bar.scrollWidth <= bar.clientWidth + 1;
  const brandFits = () => brand.scrollWidth <= brand.clientWidth + 1;
  for (const [el, cls] of FIT_STEPS) {
    if (room() && brandFits()) break;
    els[el].classList.add(cls);
  }
}

function renderConfigError() {
  const s = ui.state;
  const banner = $('#config-error');
  banner.hidden = !s.config_error;
  banner.textContent = s.config_error
    ? `Config files have an error, so the bench is still using the last good config. ${s.config_error}`
    : '';
}

/** Ticks the clocks every second without re-rendering anything else. */
function tick() {
  const now = Date.now();
  $('#clock').textContent = clockTime(now);
  $('#sl-sync').textContent = ui.lastSync ? `${Math.round((now - ui.lastSync) / 1000)}S AGO` : DASH;
  const run = activeRun();
  const elapsed = $('#m-elapsed');
  if (run && elapsed) elapsed.textContent = hms(now - Date.parse(run.created_at));
  if (tipTarget && !tipTarget.isConnected) hideTip();
  checkPollers();
}

// Drawer -----------------------------------------------------------------

const DRAWERS = {
  system: {code: 'SYS', title: 'Systems and keys'},
  settings: {code: 'CFG', title: 'Run settings'},
  arm: {code: 'MOD', title: 'Module'},
};

const confirmDiscard = () => confirmAction({
  code: 'UNSAVED', title: 'Discard changes?', body: 'This panel has edits that have not been saved.', ok: 'Discard', tone: 'danger',
});

async function openDrawer(kind, arm = null) {
  if (ui.drawer && ui.dirty.size && !(await confirmDiscard())) return;
  ui.dirty.clear();
  ui.drawer = {kind, arm};
  // Below this width the two panels would overlap, so the drawer takes over.
  if (assistantOpen() && window.innerWidth < 1240) closeAssistant();
  renderDrawer();
  const drawer = $('#drawer');
  drawer.inert = false;
  drawer.classList.add('open');
  $('#scrim').hidden = false;
  setTimeout(() => {
    const first = $('#drawer-body input:not([type=checkbox]), #drawer-body select, #drawer-body button');
    (first || $('#drawer-close')).focus({preventScroll: true});
  }, 60);
}

async function closeDrawer() {
  if (!ui.drawer) return;
  if (ui.dirty.size && !(await confirmDiscard())) return;
  ui.dirty.clear();
  ui.drawer = null;
  const drawer = $('#drawer');
  drawer.classList.remove('open');
  drawer.inert = true;
  $('#scrim').hidden = true;
}

function renderDrawer() {
  const {kind, arm} = ui.drawer;
  const meta = DRAWERS[kind];
  $('#drawer-code').textContent = meta.code;
  $('#drawer-title').textContent = kind === 'arm' ? `Module · ${arm}` : meta.title;
  const body = $('#drawer-body');
  rendered.delete('d-readiness');
  rendered.delete('keys');
  rendered.delete('d-orphans');
  if (kind === 'system') {
    fill(body,
      panel({code: 'SYS-1', title: 'Readiness', flush: true, bodyId: 'd-readiness'}),
      panel({code: 'SYS-2', title: 'API keys', flush: true},
        h('p', {class: 'muted', style: 'margin:12px 14px 4px;font-size:12.5px'},
          'Keys are saved to the bench\'s key file (.env by default), which git ignores. Only the last four characters are ever shown. ',
          'Each run gets only its own provider\'s key. Tests call read-only endpoints and use no tokens.'),
        h('div', {class: 'scroll'}, h('table', {class: 'grid-t', id: 'd-keys'}))));
    renderReadiness();
    renderKeys();
  } else if (kind === 'settings') {
    const files = configFiles();
    const runForm = h('form', {id: 'run-settings', class: 'panel-body'});
    const modesForm = h('form', {id: 'modes', class: 'panel-body'});
    runForm.addEventListener('submit', saveRunSettings);
    modesForm.addEventListener('submit', saveModes);
    trackEdits(runForm, 'run');
    trackEdits(modesForm, 'modes');
    fill(body,
      h('div', {id: 'd-orphans', hidden: true}),
      h('section', {class: 'panel'},
        h('div', {class: 'panel-head'}, h('span', {class: 'panel-code'}, 'CFG-1'), h('h2', {class: 'panel-title'}, 'Run settings')),
        h('p', {class: 'muted', style: 'margin:12px 14px 0;font-size:12.5px'}, 'These hold for every arm in a run. A green dot marks a setting changed in the dashboard.'),
        runForm),
      h('section', {class: 'panel'},
        h('div', {class: 'panel-head'}, h('span', {class: 'panel-code'}, 'CFG-2'), h('h2', {class: 'panel-title'}, 'Sample sizes')),
        modesForm),
      panel({code: 'CFG-3', title: 'Config files'},
        h('p', {class: 'muted', style: 'margin:0 0 12px;font-size:12.5px'},
          'Dashboard changes go to ', h('code', null, files.overrides), '. ', h('code', null, files.base),
          ' keeps the defaults. Reload after editing the YAML by hand.'),
        h('div', {class: 'actions'},
          h('button', {type: 'button', class: 'btn ghost', onclick: reloadConfig}, 'Reload config files'),
          h('button', {type: 'button', class: 'btn danger', id: 'reset-all', onclick: resetAll,
            disabled: !ui.state.config.has_overrides && !ui.state.config_error}, 'Reset all dashboard changes'))));
    renderOrphans();
    renderRunSettings(runForm, ui.state.config);
    renderModes(modesForm, ui.state.config);
  } else {
    const form = h('form', {id: 'arm-form'});
    form.addEventListener('submit', saveArm);
    form.addEventListener('input', () => updateHarborModel(form));
    form.addEventListener('change', () => updateHarborModel(form));
    trackEdits(form, 'arm');
    fill(body, form);
    const current = findArm(arm);
    if (current) renderArmForm(form, current, ui.state.config);
    else fill(form, h('p', {class: 'bad-text'}, `No arm named ${arm}.`));
  }
}

/** After a save: re-render the drawer's forms that hold no unsaved edits. */
function refreshDrawerForms() {
  if (!ui.drawer) return;
  const cfg = ui.state.config;
  if (ui.drawer.kind === 'settings') {
    if (!ui.dirty.has('run')) renderRunSettings($('#run-settings'), cfg);
    if (!ui.dirty.has('modes')) renderModes($('#modes'), cfg);
    const reset = $('#reset-all');
    if (reset) reset.disabled = !cfg.has_overrides && !ui.state.config_error;
    renderOrphans();
  } else if (ui.drawer.kind === 'arm' && !ui.dirty.has('arm')) {
    const arm = findArm(ui.drawer.arm);
    if (arm) renderArmForm($('#arm-form'), arm, cfg);
  }
}

/** On every poll: only the read-only parts of an open drawer change. */
function refreshDrawerReadOnly() {
  if (ui.drawer && ui.drawer.kind === 'system') {
    renderReadiness();
    renderKeys();
  } else if (ui.drawer && ui.drawer.kind === 'settings') {
    renderOrphans();
  }
}

function trackEdits(form, key) {
  const mark = () => {
    ui.dirty.add(key);
    const unsaved = form.querySelector('.unsaved');
    if (unsaved) unsaved.hidden = false;
  };
  form.addEventListener('input', mark);
  form.addEventListener('change', mark);
}

/** Takes the config view a save returned and refreshes everything built from it. */
function applyConfig(cfg, {key = null, force = false, message = ''} = {}) {
  ui.state.config = cfg;
  ui.state.config_error = null;
  if (force) ui.dirty.clear();
  if (key) ui.dirty.delete(key);
  refreshDrawerForms();
  renderConfigError();
  renderHeader();
  if (ui.view === 'launch') renderLaunch();
  if (message) toast(message);
  poll();
}

/**
 * busy() for config saves. A 409 saying the YAML changed on disk gets a Reload button in the toast; the form is
 * not re-rendered, so what was typed stays for the next Save.
 */
async function saveConfig(button, action) {
  if (button) button.disabled = true;
  try {
    return await action();
  } catch (err) {
    configError(err);
    return undefined;
  } finally {
    if (button) button.disabled = false;
  }
}

function configError(err) {
  if (err.staleConfig) toast(err.message, true, {label: 'Reload config', run: () => reloadConfigFiles()});
  else toast(err.message, true);
}

/** Reloads the YAML files. Forms holding unsaved edits keep them unless `force` (the explicit Reload button). */
async function reloadConfigFiles({force = false, button = null} = {}) {
  await busy(button, async () => {
    try {
      const cfg = await api('POST', '/api/config/reload');
      const kept = !force && ui.dirty.size > 0;
      applyConfig(cfg, {force, message: kept ? 'Config files reloaded. Your edits are still in the form: save again.' : 'Config files reloaded'});
    } finally {
      poll();  // picks up config_error if the files don't load
    }
  });
}

/** Overrides for arms or modes the base YAML no longer has. They would reattach silently if the name came back. */
function orphanNames(cfg) {
  const o = cfg.orphaned_overrides || {};
  return {arms: o.arms || [], modes: o.modes || []};
}

/** The warn banner at the top of the settings drawer, with a Reset per orphaned arm and one for the modes. */
function renderOrphans() {
  const box = $('#d-orphans');
  if (!box) return;
  const {arms, modes} = orphanNames(ui.state.config);
  if (!changed('d-orphans', [arms, modes])) return;
  box.hidden = !arms.length && !modes.length;
  if (box.hidden) {
    fill(box);
    return;
  }
  const files = configFiles();
  fill(box, h('div', {class: 'banner warn orphans', role: 'status'},
    h('span', {'aria-hidden': 'true'}, '!'),
    h('div', {class: 'orphans-body'},
      h('p', null, h('b', null, 'Changes kept for names that are gone. '),
        `${files.overrides} still changes ${listText([...arms.map((a) => `arm ${a}`), ...modes.map((m) => `mode ${m}`)])}, `
        + `which ${files.base} no longer has. They would come back silently if the name returned.`),
      h('div', {class: 'actions'},
        arms.map((name) => h('button', {type: 'button', class: 'btn small ghost', onclick: (e) => resetOrphanArm(name, e.currentTarget)},
          `Reset arm ${name}`)),
        modes.length > 0 && h('button', {type: 'button', class: 'btn small ghost', onclick: (e) => resetOrphanModes(e.currentTarget)},
          'Reset sample sizes')))));
}

async function resetOrphanArm(name, button) {
  await busy(button, async () => {
    applyConfig(await api('DELETE', `/api/config/arms/${enc(name)}`), {message: `Changes for arm ${name} removed`});
  });
}

/** DELETE /api/config/modes resets every mode, so say so when it would also undo changes to modes that still exist. */
async function resetOrphanModes(button) {
  const cfg = ui.state.config;
  const gone = new Set(orphanNames(cfg).modes);
  const live = cfg.modes_overridden.filter((m) => !gone.has(m));
  if (live.length) {
    const ok = await confirmAction({
      code: 'RESET', title: 'Reset all sample sizes?', ok: 'Reset sample sizes', tone: 'danger',
      body: `This also undoes your changes to ${listText(live)}. Every mode returns to its value in ${configFiles().base}.`,
    });
    if (!ok) return;
  }
  await busy(button, async () => {
    applyConfig(await api('DELETE', '/api/config/modes'), {message: 'Sample sizes reset'});
  });
}

/** The YAML files behind the running profile: `python -m bench --profile dev` uses config/dev.yaml. */
function configFiles() {
  const profile = ui.state && ui.state.config.profile;
  const named = profile && profile !== 'default';
  return {
    base: named ? `config/${profile}.yaml` : 'config/bench.yaml',
    overrides: named ? `config/${profile}.overrides.yaml` : 'config/overrides.yaml',
  };
}

// Systems and keys

function renderReadiness() {
  const box = $('#d-readiness');
  if (!box || !changed('d-readiness', ui.state.readiness)) return;
  fill(box, h('ul', {class: 'checklist'}, ui.state.readiness.map((c) => h('li', null,
    led(c.ok ? 'ok' : 'bad'),
    h('span', {class: 'name'}, c.name),
    h('span', {class: 'detail'}, c.ok ? '✓ ' : '✗ ', c.detail)))));
}

function renderKeys() {
  const table = $('#d-keys');
  if (!table || !changed('keys', [ui.state.keys, ui.keyTests])) return;
  fill(table, thead('Provider', 'Variable', 'Status', ''), h('tbody', null, ui.state.keys.map(keyRow)));
}

function keyRow(k) {
  const saved = ui.keyTests[k.env_key];
  const test = saved && saved.masked === k.masked ? saved : null;  // drop results for a replaced key
  const masked = k.masked && k.masked !== 'set' ? `${k.masked} ` : '';
  return h('tr', null,
    h('td', null, h('b', null, k.provider === 'assistant' ? ASSISTANT_KEY_LABEL : k.provider), !k.enabled && [' ', tag('off in config')]),
    h('td', null, h('code', null, k.env_key)),
    h('td', null,
      k.set ? h('span', {class: 'ok-text'}, `● Set ${masked}(${k.source})`) : h('span', {class: 'muted'}, '○ Not set'),
      test && keyTestLine(test)),
    h('td', null, h('div', {class: 'actions', style: 'justify-content:flex-end;flex-wrap:nowrap'},
      h('button', {type: 'button', class: `btn small ${k.set ? 'ghost' : 'primary'}`, onclick: () => openKeyDialog(k)}, k.set ? 'Replace' : 'Set'),
      k.set && h('button', {type: 'button', class: 'btn small ghost', disabled: Boolean(test && test.pending), onclick: () => testKey(k)}, 'Test'),
      k.set && h('button', {type: 'button', class: 'btn small danger', onclick: (e) => clearKey(k, e.currentTarget)}, 'Clear'))));
}

function keyTestLine(test) {
  if (test.pending) return h('span', {class: 'sub'}, 'Testing…');
  if (test.ok === true) return h('span', {class: 'sub ok-text'}, `✓ ${test.detail}`);
  if (test.ok === false) return h('span', {class: 'sub bad-text'}, `✗ ${test.detail}`);
  return h('span', {class: 'sub'}, test.detail);
}

async function testKey(k) {
  ui.keyTests[k.env_key] = {masked: k.masked, pending: true};
  renderKeys();
  let result;
  try {
    result = await api('POST', `/api/keys/${enc(k.env_key)}/test`);
  } catch (err) {
    result = {ok: false, detail: err.message};
  }
  ui.keyTests[k.env_key] = {masked: k.masked, ok: result.ok, detail: result.detail};
  renderKeys();
}

function keysChanged(envKey, keys, message) {
  delete ui.keyTests[envKey];
  ui.state.keys = keys;
  renderKeys();
  renderHeader();
  if (ui.view === 'launch') renderLaunch();
  toast(message);
  poll();
}

async function clearKey(k, button) {
  const where = k.source === 'shell'
    ? 'It comes from the shell that started the bench, so this clears it for this session only.'
    : 'This removes it from the key file.';
  const ok = await confirmAction({code: 'KEY', title: `Clear ${k.env_key}?`, body: where, ok: 'Clear key', tone: 'danger'});
  if (!ok) return;
  await busy(button, async () => {
    const keys = await api('DELETE', `/api/keys/${enc(k.env_key)}`);
    keysChanged(k.env_key, keys, `${k.env_key} cleared`);
  });
}

function openKeyDialog(k) {
  const dialog = $('#key-dialog');
  const input = $('#key-value');
  ui.keyTarget = k.env_key;
  $('#key-title').textContent = `${k.set ? 'Replace' : 'Set'} ${k.env_key}`;
  input.value = '';
  dialog.returnValue = '';  // Escape keeps the previous returnValue otherwise
  dialog.showModal();
  input.focus();
}

function setupKeyDialog() {
  const dialog = $('#key-dialog');
  const input = $('#key-value');
  // Enter would otherwise submit with the form's first button, which is Cancel.
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      dialog.close('save');
    }
  });
  dialog.addEventListener('close', () => {
    const envKey = ui.keyTarget;
    const value = input.value.trim();
    input.value = '';
    ui.keyTarget = null;
    if (dialog.returnValue !== 'save' || !envKey) return;
    if (!value) {
      toast('Enter a key to save.', true);
      return;
    }
    busy(null, async () => {
      const keys = await api('PUT', `/api/keys/${enc(envKey)}`, {value});
      keysChanged(envKey, keys, `${envKey} saved`);
    });
  });
}

// Run settings and sample sizes

function readNumber(input, empty) {
  const label = input.closest('.field').querySelector('.label').textContent;
  if (input.validity.badInput) throw new Error(`${label} must be a number`);
  const raw = input.value.trim();
  return raw === '' ? empty : Number(raw);
}

function renderRunSettings(form, cfg) {
  const run = cfg.run;
  const over = new Set(cfg.run_overridden);
  const efforts = [...EFFORT_LEVELS.map((l) => [l, l]), ['omit', "omit, each harness's default"]];
  fill(form,
    h('div', {class: 'row'},
      field('Effort', select('effort', efforts, run.effort),
        {dot: over.has('effort'), title: 'Used by every arm whose effort is inherit', notes: [note('Arms set to inherit')]}),
      field('Parallel trials', numberInput('n_concurrent', run.n_concurrent, {min: 1, step: 1, placeholder: 'default'}),
        {dot: over.has('n_concurrent'), notes: [note('Per arm')]}),
      'parallel_arms' in run && field('Arms at once', numberInput('parallel_arms', run.parallel_arms, {min: 1, step: 1, placeholder: 'default'}),
        {dot: over.has('parallel_arms'), notes: [note('All share the budget')]}),
      field('Run budget (USD)', numberInput('budget_usd', run.budget_usd, {min: 0, step: 'any', placeholder: 'default'}),
        {dot: over.has('budget_usd'), notes: [note('Checked after each trial')]}),
      field('Per-trial cap (USD)', numberInput('per_trial_cap_usd', run.per_trial_cap_usd, {min: 0, step: 'any', placeholder: 'none'}),
        {dot: over.has('per_trial_cap_usd'), notes: [note('Empty = none')]}),
      field('Max turns', numberInput('max_turns', run.max_turns, {min: 1, step: 1, placeholder: 'none'}),
        {dot: over.has('max_turns'), notes: [note('Empty = none')]}),
      field('Timeout multiplier', numberInput('timeout_multiplier', run.timeout_multiplier, {min: 0, step: 'any', placeholder: 'default'}),
        {dot: over.has('timeout_multiplier')})),
    formActions(resetRunSettings));
  ui.dirty.delete('run');
}

async function saveRunSettings(event) {
  event.preventDefault();
  const els = event.currentTarget.elements;
  await saveConfig(event.submitter, async () => {
    const body = {effort: els.effort.value};
    for (const name of ['n_concurrent', 'parallel_arms', 'budget_usd', 'timeout_multiplier']) {
      if (els[name]) body[name] = readNumber(els[name], '');
    }
    for (const name of ['per_trial_cap_usd', 'max_turns']) body[name] = readNumber(els[name], null);  // empty = no cap
    applyConfig(await api('PUT', '/api/config/run', body), {key: 'run', message: 'Run settings saved'});
  });
}

async function resetRunSettings(event) {
  await busy(event.currentTarget, async () => {
    applyConfig(await api('DELETE', '/api/config/run'), {key: 'run', message: 'Run settings reset'});
  });
}

function renderModes(form, cfg) {
  const over = new Set(cfg.modes_overridden);
  const max = cfg.tasks.length || null;
  fill(form,
    ...Object.entries(cfg.modes).map(([name, mode]) => h('div', {class: 'row'},
      field(`${titleCase(name)}: tasks`, numberInput(`${name}.n_tasks`, mode.n_tasks, {min: 1, max, step: 1, required: true}),
        {dot: over.has(name)}),
      field(`${titleCase(name)}: attempts`, numberInput(`${name}.n_attempts`, mode.n_attempts, {min: 1, step: 1, required: true}),
        {dot: over.has(name)}))),
    h('p', {class: 'muted', style: 'margin:0 0 12px;font-size:12.5px'},
      `Each mode runs the first n tasks of the task sample, which lists ${cfg.tasks.length}.`),
    formActions(resetModes));
  ui.dirty.delete('modes');
}

async function saveModes(event) {
  event.preventDefault();
  const els = event.currentTarget.elements;
  await saveConfig(event.submitter, async () => {
    let cfg = null;
    for (const [name, current] of Object.entries(ui.state.config.modes)) {
      const body = {
        n_tasks: readNumber(els[`${name}.n_tasks`], current.n_tasks),
        n_attempts: readNumber(els[`${name}.n_attempts`], current.n_attempts),
      };
      if (body.n_tasks === current.n_tasks && body.n_attempts === current.n_attempts) continue;
      cfg = await api('PUT', `/api/config/modes/${enc(name)}`, body);
    }
    applyConfig(cfg || ui.state.config, {key: 'modes', message: cfg ? 'Sample sizes saved' : 'No changes to save'});
  });
}

async function resetModes(event) {
  await busy(event.currentTarget, async () => {
    applyConfig(await api('DELETE', '/api/config/modes'), {key: 'modes', message: 'Sample sizes reset'});
  });
}

/** The drawer's Reload button: re-reads the files and redraws every form from them. */
async function reloadConfig(event) {
  await reloadConfigFiles({force: true, button: event.currentTarget});
}

async function resetAll(event) {
  const button = event.currentTarget;
  const ok = await confirmAction({
    code: 'RESET', title: 'Reset all dashboard changes?',
    body: `Every setting returns to its value in ${configFiles().base}.`, ok: 'Reset all', tone: 'danger',
  });
  if (!ok) return;
  await busy(button, async () => {
    applyConfig(await api('DELETE', '/api/config/overrides'), {force: true, message: 'All dashboard changes reset'});
  });
}

// Arm form (drawer)

function loadSchema(agent) {
  if (!ui.schemaLoads.has(agent)) {
    const load = api('GET', `/api/schemas/${enc(agent)}`).then((schema) => {
      ui.schemas.set(agent, schema);
      return schema;
    });
    load.catch(() => ui.schemaLoads.delete(agent));  // let the next render retry
    ui.schemaLoads.set(agent, load);
  }
  return ui.schemaLoads.get(agent);
}

function renderArmForm(form, arm, cfg) {
  const schema = ui.schemas.get(arm.agent);
  if (!schema) {
    fill(form, h('p', {class: 'muted'}, 'Loading harness options…'));
    loadSchema(arm.agent).then(() => {
      const fresh = form.isConnected && findArm(arm.name);
      if (fresh && !ui.dirty.has('arm')) renderArmForm(form, fresh, ui.state.config);
    }, (err) => {
      if (form.isConnected) fill(form, h('p', {class: 'bad-text'}, `Could not load this harness's options: ${err.message}`));
    });
    return;
  }
  const run = cfg.run;
  const warnings = arm.model_warnings || {};
  const groups = {reasoning: [], limits: [], other: []};
  for (const f of schema.fields) (groups[f.group] || groups.other).push(f);
  const otherOpen = Boolean(form.querySelector('details.other-options[open]'));
  const otherSet = groups.other.filter((f) => arm.options[f.name] !== undefined && arm.options[f.name] !== null).length;
  const optionRow = (code, title, fields) => fields.length > 0 && [
    h('div', {class: 'group-title'}, `${code} · ${title.toUpperCase()}`),
    h('div', {class: 'row'}, fields.map((f) => optionField(f, arm, run, warnings))),
  ];

  fill(form,
    h('section', {class: 'panel'},
      h('div', {class: 'panel-head'},
        h('span', {class: 'panel-code'}, 'MOD-1'),
        h('h2', {class: 'panel-title'}, arm.family || arm.agent),
        h('div', {class: 'panel-meta'},
          kindTag(arm),
          arm.name === cfg.baseline_arm && tag('baseline', 'ok'),
          arm.overridden && tag('edited', 'warn'),
          h('label', {class: 'switch'}, h('input', {type: 'checkbox', name: 'enabled', checked: arm.enabled}),
            h('span', {class: 'switch-label'}, 'ARMED')))),
      h('div', {class: 'panel-body'},
        h('p', {class: 'arm-harness'}, 'Harness ', h('code', null, arm.agent),
          arm.kind === 'import_path' ? ' (an import path, run as a custom harness)' : ''),
        h('div', {class: 'row'},
          field('Provider', select('provider', cfg.providers.map((p) => [p.name, p.enabled ? p.name : `${p.name} (off)`]), arm.provider)),
          field('Model', h('input', {type: 'text', class: 'mono', name: 'model', value: arm.model, required: true, list: 'model-options', spellcheck: 'false', autocomplete: 'off'}),
            {notes: [h('small', {class: 'harbor-model'})]}),
          effortField(arm, schema, run),
          field('Family', h('input', {type: 'text', class: 'mono', name: 'family', value: arm.family || '', placeholder: arm.agent,
            maxlength: 64, spellcheck: 'false', autocomplete: 'off'}),
          {title: 'The harness label results pool and compare under', notes: [note('Empty resets it to the default')]})),
        optionRow('A', 'Reasoning and thinking', groups.reasoning),
        optionRow('B', 'Limits', groups.limits),
        groups.other.length > 0 && h('details', {class: 'other-options', open: otherOpen},
          h('summary', null, `C · Other options (${groups.other.length}${otherSet ? `, ${otherSet} set` : ''})`),
          h('div', {class: 'row', style: 'margin-top:10px'}, groups.other.map((f) => optionField(f, arm, run, warnings)))),
        formActions(arm.overridden && resetArm))),
    modelDatalist());
  updateHarborModel(form);
  ui.dirty.delete('arm');
}

/** "custom" for an arm whose agent is an import path (module.path:Class) rather than a registered harness name. */
function kindTag(arm) {
  return Boolean(arm && arm.kind === 'import_path')
    && tip(tag('custom'), 'Custom harness', [arm.agent, 'An import path, not a registered harness name']);
}

/** Model suggestions: current Claude models plus every model an arm already uses. */
function modelDatalist() {
  const models = new Set([...CLAUDE_MODELS, ...ui.state.config.arms.map((a) => a.model)]);
  return h('datalist', {id: 'model-options'}, [...models].map((m) => h('option', {value: m})));
}

/** Shows the model id Harbor will get: the provider's prefix plus the model. */
function updateHarborModel(form) {
  const hint = form.querySelector('.harbor-model');
  const els = form.elements;
  if (!hint || !els.provider || !els.model) return;
  const provider = ui.state.config.providers.find((p) => p.name === els.provider.value);
  hint.textContent = `Harbor gets ${provider ? provider.model_prefix : ''}${els.model.value.trim()}`;
}

function effortField(arm, schema, run) {
  const levels = schema.effort_levels;
  const fixed = [['inherit', `inherit (run: ${run.effort})`], ['omit', 'omit (harness default)']];
  if (levels === null) {
    return field('Effort', select('effort', fixed, arm.effort, {disabled: true}),
      {notes: [note('not supported by this harness')]});
  }
  if (levels.length === 0) {
    const listId = `effort-levels-${arm.name.replace(/[^\w-]/g, '_')}`;
    return field('Effort', [
      h('input', {type: 'text', class: 'mono', name: 'effort', value: arm.effort, list: listId, placeholder: 'inherit', spellcheck: 'false', autocomplete: 'off'}),
      h('datalist', {id: listId}, ['inherit', 'omit', ...EFFORT_LEVELS].map((v) => h('option', {value: v}))),
    ], {notes: [note(`inherit (run: ${run.effort}), omit, or any level`)]});
  }
  const notes = [];
  if (arm.effort === 'inherit' && run.effort !== 'omit' && !levels.includes(run.effort)) {
    notes.push(warn(`This harness doesn't accept the run effort "${run.effort}".`));
  }
  return field('Effort', select('effort', [...fixed, ...levels.map((l) => [l, l])], arm.effort), {notes});
}

function optionField(f, arm, run, warnings) {
  const notes = [];
  if (f.caution) notes.push(warn(f.caution));
  if (warnings[f.name]) notes.push(warn(warnings[f.name]));
  return field(f.name, optionControl(f, arm.options[f.name], run), {title: f.description || null, notes, raw: true});
}

function optionControl(f, value, run) {
  const name = `opt:${f.name}`;
  const unset = value === undefined || value === null;
  const hasDefault = f.default !== undefined && f.default !== null;
  const defaultLabel = hasDefault ? `default (${shown(f.default)})` : 'default';
  switch (f.kind) {
    case 'enum':
      return select(name, [['', defaultLabel], ...(f.enum || []).map((v) => [String(v), String(v)])], unset ? '' : String(value));
    case 'boolean':
      return select(name, [['', defaultLabel], ['true', 'true'], ['false', 'false']], unset ? '' : String(value));
    case 'integer':
    case 'number':
      return numberInput(name, unset ? '' : value, {step: f.kind === 'integer' ? 1 : 'any', placeholder: placeholderFor(f, run)});
    case 'json':
      return h('textarea', {name, rows: 3, spellcheck: 'false', value: unset ? '' : JSON.stringify(value, null, 2),
        placeholder: placeholderFor(f, run) || 'JSON'});
    default:
      return h('input', {type: 'text', class: 'mono', name, value: unset ? '' : String(value), spellcheck: 'false', autocomplete: 'off',
        placeholder: placeholderFor(f, run)});
  }
}

/** Empty limits fall back to the run caps, so the placeholder shows the cap that applies. */
function placeholderFor(f, run) {
  if (DOLLAR_CAPS.includes(f.name)) return isNum(run.per_trial_cap_usd) ? `${run.per_trial_cap_usd} (run cap)` : 'no run cap';
  if (TURN_CAPS.includes(f.name)) return isNum(run.max_turns) ? `${run.max_turns} (run cap)` : 'no run cap';
  if (f.default === undefined || f.default === null) return null;
  return `default: ${f.kind === 'json' ? JSON.stringify(f.default) : shown(f.default)}`;
}

function parseOption(f, raw) {
  switch (f.kind) {
    case 'enum':
      return (f.enum || []).find((v) => String(v) === raw) ?? raw;
    case 'boolean':
      return raw === 'true';
    case 'integer': {
      const n = Number(raw);
      if (!Number.isInteger(n)) throw new Error(`${f.name} must be a whole number`);
      return n;
    }
    case 'number': {
      const n = Number(raw);
      if (!Number.isFinite(n)) throw new Error(`${f.name} must be a number`);
      return n;
    }
    case 'json':
      try {
        return JSON.parse(raw);
      } catch (err) {
        throw new Error(`${f.name} is not valid JSON: ${err.message}`);
      }
    default:
      return raw;
  }
}

/** The full option set as the form shows it. Empty fields are left out. */
function readOptions(form, schema) {
  const options = {};
  for (const f of schema.fields) {
    const el = form.elements[`opt:${f.name}`];
    if (!el) continue;
    if (el.validity && el.validity.badInput) throw new Error(`${f.name} must be a number`);
    const raw = el.value.trim();
    if (raw !== '') options[f.name] = parseOption(f, raw);
  }
  return options;
}

async function saveArm(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const name = ui.drawer && ui.drawer.arm;
  const arm = name && findArm(name);
  const schema = arm && ui.schemas.get(arm.agent);
  if (!schema) return;
  await saveConfig(event.submitter, async () => {
    const els = form.elements;
    const body = {
      enabled: els.enabled.checked,
      provider: els.provider.value,
      model: els.model.value.trim(),
      effort: els.effort.value.trim() || 'inherit',
      family: els.family ? els.family.value.trim() : undefined,  // '' resets it to the default
      options: readOptions(form, schema),
    };
    if (!body.model) throw new Error('Model is required');
    const cfg = await api('PUT', `/api/config/arms/${enc(name)}`, body);
    applyConfig(cfg, {key: 'arm', message: `${name} saved`});
  });
}

async function resetArm(event) {
  const name = ui.drawer && ui.drawer.arm;
  if (!name) return;
  await busy(event.currentTarget, async () => {
    const cfg = await api('DELETE', `/api/config/arms/${enc(name)}`);
    applyConfig(cfg, {key: 'arm', message: `${name} reset to ${configFiles().base}`});
  });
}

// LAUNCH -----------------------------------------------------------------

function renderLaunch(force = false) {
  const root = $('#view-launch');
  const cfg = ui.state.config;
  if (!ui.mode || !(ui.mode in cfg.modes)) ui.mode = Object.keys(cfg.modes)[0] || null;
  if (!root.querySelector('.launch')) {
    fill(root, h('div', {class: 'launch'},
      panel({code: 'L-01', title: 'Loadout', meta: h('span', {id: 'loadout-meta'}), flush: true},
        h('div', {class: 'modules', id: 'modules'})),
      h('aside', {class: 'panel mission hot'},
        h('div', {class: 'panel-head'}, h('span', {class: 'panel-code'}, 'L-02'), h('h2', {class: 'panel-title'}, 'Mission'),
          h('div', {class: 'panel-meta', id: 'mission-meta'})),
        h('section', {id: 'mode-section'}),
        h('section', {id: 'params-section'}),
        h('section', {id: 'preflight'}),
        h('section', {id: 'launch-section'}))),
      h('datalist', {id: 'model-options-launch'}));
    force = true;
  }
  for (const agent of new Set(cfg.arms.map((a) => a.agent))) {
    if (!ui.schemas.has(agent)) loadSchema(agent).then(() => ui.view === 'launch' && renderModules(), () => {});
  }
  renderModules(force);
  renderMission(force);
  schedulePreflight();
  renderPreflight();
}

const moduleCard = (name) => $(`#modules .module[data-arm="${CSS.escape(name)}"]`);

function renderModules(force = false) {
  const cfg = ui.state.config;
  const container = $('#modules');
  if (!container) return;
  const models = new Set([...CLAUDE_MODELS, ...cfg.arms.map((a) => a.model)]);
  if (changed('model-options', [...models])) {
    fill($('#model-options-launch'), [...models].map((m) => h('option', {value: m})));
  }
  const existing = new Map([...container.children].map((card) => [card.dataset.arm, card]));
  const planCurrent = Boolean(ui.plan) && ui.planFor === planInputs();
  const cards = cfg.arms.map((arm) => {
    let card = existing.get(arm.name);
    if (!card) card = h('article', {class: 'module', 'data-arm': arm.name});
    const editing = ui.editingModel.has(arm.name) && card.contains(document.activeElement);
    const planArm = ui.plan ? ui.plan.arms.find((a) => a.name === arm.name) || null : null;
    const signature = JSON.stringify([arm, cfg.providers, cfg.run.effort, cfg.baseline_arm, Boolean(ui.schemas.get(arm.agent)), planArm, planCurrent]);
    if (!editing && (force || card.dataset.sig !== signature)) {
      buildModule(card, arm, cfg, planArm, planCurrent);
      card.dataset.sig = signature;
    }
    return card;
  });
  if (cards.length !== container.children.length || cards.some((c, i) => container.children[i] !== c)) {
    container.replaceChildren(...cards);
  }
  const armed = cfg.arms.filter((a) => a.enabled).length;
  $('#loadout-meta').textContent = `${armed} OF ${cfg.arms.length} ARMED`;
}

function buildModule(card, arm, cfg, planArm, planCurrent) {
  const schema = ui.schemas.get(arm.agent);
  const provider = cfg.providers.find((p) => p.name === arm.provider);
  const optionCount = Object.keys(arm.options || {}).length;
  card.className = arm.enabled ? 'module armed' : 'module';
  const family = arm.family || arm.agent;
  fill(card,
    h('div', {class: 'module-head'},
      h('div', {class: 'module-id'},
        h('div', {class: 'module-name'}, arm.name),
        h('div', {class: 'module-agent'}, `harness ${family}`,
          arm.kind !== 'import_path' && family !== arm.agent && ` · agent ${arm.agent}`),
        arm.kind === 'import_path' && h('div', {class: 'module-path mono', title: arm.agent}, arm.agent),
        h('div', {class: 'module-badges'},
          kindTag(arm),
          arm.name === cfg.baseline_arm && tag('baseline', 'ok'),
          arm.overridden && tag('edited', 'warn'),
          optionCount > 0 && tag(plural(optionCount, 'option')))),
      h('label', {class: 'switch', title: arm.enabled ? 'Included in the next run' : 'Left out of the next run'},
        h('input', {type: 'checkbox', checked: arm.enabled, 'aria-label': `Arm ${arm.name}`,
          onchange: (e) => quickSave(arm.name, {enabled: e.target.checked})}),
        h('span', {class: 'switch-label'}, arm.enabled ? 'ARMED' : 'SAFE'))),
    h('div', {class: 'module-body'},
      h('div', {class: 'two'},
        field('Provider', select('provider', cfg.providers.map((p) => [p.name, p.enabled ? p.name : `${p.name} (off)`]), arm.provider,
          {onchange: (e) => quickSave(arm.name, {provider: e.target.value})})),
        field('Effort', quickEffort(arm, schema, cfg.run))),
      field('Model', h('input', {
        type: 'text', class: 'mono', value: arm.model, list: 'model-options-launch', spellcheck: 'false', autocomplete: 'off',
        'aria-label': `Model for ${arm.name}`,
        oninput: () => ui.editingModel.add(arm.name),
        onkeydown: (e) => {
          if (e.key === 'Escape') {
            e.target.value = arm.model;
            ui.editingModel.delete(arm.name);
            e.target.blur();
          }
        },
        onchange: (e) => {
          ui.editingModel.delete(arm.name);
          const model = e.target.value.trim();
          if (!model) e.target.value = arm.model;
          else if (model !== arm.model) quickSave(arm.name, {model});
        },
      })),
      h('div', {class: 'module-plan', style: planCurrent ? null : 'opacity:.5'}, modulePlan(arm, planArm))),
    h('div', {class: 'module-foot'},
      h('span', {class: 'note mono'}, `${provider ? provider.model_prefix : ''}${arm.model}`),
      h('button', {type: 'button', class: 'btn ghost small', onclick: () => openDrawer('arm', arm.name)},
        'Configure ', h('span', {class: 'arrow'}, '›'))));
}

function quickEffort(arm, schema, run) {
  const onchange = (e) => quickSave(arm.name, {effort: e.target.value.trim() || 'inherit'});
  const fixed = [['inherit', `inherit (${run.effort})`], ['omit', 'omit (default)']];
  if (!schema) return select('effort', fixed, arm.effort, {disabled: true, title: 'Loading harness options…'});
  if (schema.effort_levels === null) {
    return select('effort', [['omit', 'n/a (harness)'], ['inherit', 'n/a (harness)']], arm.effort,
      {disabled: true, title: 'This harness takes no effort setting'});
  }
  if (schema.effort_levels.length === 0) {
    return h('input', {type: 'text', class: 'mono', value: arm.effort, placeholder: 'inherit', list: 'effort-any', spellcheck: 'false',
      title: 'Any level text is accepted', onchange});
  }
  return select('effort', [...fixed, ...schema.effort_levels.map((l) => [l, l])], arm.effort, {onchange});
}

function modulePlan(arm, planArm) {
  if (!arm.enabled) return h('span', {class: 'note mono faint', style: 'font-size:11px'}, 'Safe: not in the next run.');
  if (!planArm) return h('span', {class: 'mono faint', style: 'font-size:11px'}, 'Awaiting preflight…');
  const chip = (k) => tip(h('span', {class: 'chip'}, `${k.name}=${shown(k.value)}`, h('span', {class: 'src'}, k.source)),
    k.name, [[JSON.stringify(k.value), 'value'], [k.source, 'source']]);
  return [
    planArm.kwargs.length ? h('div', {class: 'chips'}, planArm.kwargs.map(chip)) : h('span', {class: 'mono faint', style: 'font-size:11px'}, 'No options passed'),
    planArm.problems.map((p) => h('div', {class: 'issue bad-text'}, `✗ ${p}`)),
    planArm.warnings.map((w) => h('div', {class: 'issue warn-text'}, `! ${w}`)),
  ];
}

async function quickSave(name, patch) {
  const arm = findArm(name);
  if (!arm) return;
  const body = {enabled: arm.enabled, provider: arm.provider, model: arm.model, effort: arm.effort, options: arm.options, ...patch};
  const card = moduleCard(name);
  const noteEl = card && card.querySelector('.module-foot .note');
  if (noteEl) {
    noteEl.className = 'note saving';
    noteEl.textContent = 'SAVING…';
  }
  try {
    const cfg = await api('PUT', `/api/config/arms/${enc(name)}`, body);
    applyConfig(cfg);
    const saved = moduleCard(name);
    const savedNote = saved && saved.querySelector('.module-foot .note');
    if (savedNote) {
      savedNote.className = 'note saved';
      savedNote.textContent = '✓ SAVED';
      setTimeout(() => {
        if (!savedNote.isConnected || savedNote.className !== 'note saved') return;
        const fresh = findArm(name);
        const provider = fresh && ui.state.config.providers.find((p) => p.name === fresh.provider);
        savedNote.className = 'note mono';
        savedNote.textContent = fresh ? `${provider ? provider.model_prefix : ''}${fresh.model}` : '';
      }, 1800);
    }
  } catch (err) {
    configError(err);
    renderModules(true);  // put the controls back to the saved values
  }
}

function renderMission(force = false) {
  const cfg = ui.state.config;
  const run = cfg.run;
  const orphans = orphanNames(cfg);
  const signature = [cfg.modes, run, cfg.run_overridden, ui.mode, orphans];
  if (!force && !changed('mission', signature)) return;
  rendered.set('mission', JSON.stringify(signature));
  const over = new Set(cfg.run_overridden);
  fill($('#mode-section'),
    h('div', {class: 'section-head'}, eyebrow('Mode')),
    h('div', {class: 'segmented', role: 'radiogroup', 'aria-label': 'Mode'},
      Object.entries(cfg.modes).map(([name, m]) => h('button', {
        type: 'button', role: 'radio', 'aria-checked': String(name === ui.mode), class: name === ui.mode ? 'active' : null,
        onclick: () => {
          ui.mode = name;
          renderMission(true);
          schedulePreflight(0);
          renderPreflight();
        },
      }, h('b', null, name.toUpperCase()), h('span', null, `${plural(m.n_tasks, 'task')} × ${m.n_attempts}`)))));
  const param = (label, value, key) => h('div', {title: over.has(key) ? 'Changed in the dashboard' : null},
    h('dt', null, label), h('dd', {class: over.has(key) ? 'changed' : null}, value));
  fill($('#params-section'),
    h('div', {class: 'section-head'}, eyebrow('Parameters'),
      h('button', {type: 'button', class: 'btn ghost small', onclick: () => openDrawer('settings')}, 'Edit ', h('span', {class: 'arrow'}, '›'))),
    h('dl', {class: 'params'},
      param('Effort', run.effort, 'effort'),
      param('Run budget', money(run.budget_usd), 'budget_usd'),
      param('Per-trial cap', isNum(run.per_trial_cap_usd) ? money(run.per_trial_cap_usd) : 'none', 'per_trial_cap_usd'),
      param('Max turns', isNum(run.max_turns) ? String(run.max_turns) : 'none', 'max_turns'),
      param('Parallel trials', String(run.n_concurrent), 'n_concurrent'),
      'parallel_arms' in run ? param('Arms at once', String(run.parallel_arms), 'parallel_arms')
        : param('Timeout', `×${run.timeout_multiplier}`, 'timeout_multiplier')),
    (orphans.arms.length > 0 || orphans.modes.length > 0) && h('p', {class: 'orphan-note warn-text'},
      h('span', null, '! '),
      `Changes are kept for ${listText([...orphans.arms.map((a) => `arm ${a}`), ...orphans.modes.map((m) => `mode ${m}`)])}, which the config no longer has. `,
      h('button', {type: 'button', class: 'btn ghost small', onclick: () => openDrawer('settings')}, 'Review ', h('span', {class: 'arrow'}, '›'))));
}

/** What a plan depends on. A plan built from anything else is out of date. */
function planInputs() {
  const s = ui.state;
  return JSON.stringify([ui.mode, armedNames(), s.config, s.keys.map((k) => [k.env_key, k.set])]);
}

let preflightTimer = 0;
const PREFLIGHT_RETRY_MS = 3000;  // after a busy (503) or unreachable answer
function schedulePreflight(delay = 300) {
  if (!ui.state || !ui.mode) return;
  const inputs = planInputs();
  if (inputs === ui.planPending) return;
  // A plan for these inputs is final, unless it failed because the bench was busy or unreachable for a moment.
  if (inputs === ui.planFor && !(inputs === ui.planRetry && Date.now() >= ui.planRetryAt)) return;
  clearTimeout(preflightTimer);
  preflightTimer = setTimeout(runPreflight, delay);
}

async function runPreflight() {
  const inputs = planInputs();
  const arms = armedNames();
  const seq = ++ui.planSeq;
  let retry = false;
  ui.planPending = inputs;
  if (!arms.length) {
    ui.plan = null;
    ui.planError = 'Arm at least one module to plan a run.';
  } else {
    renderPreflight();
    try {
      const plan = await api('GET', `/api/plan?${new URLSearchParams({mode: ui.mode, arms: arms.join(',')})}`);
      if (seq !== ui.planSeq) return;
      ui.plan = plan;
      ui.planError = '';
    } catch (err) {
      if (seq !== ui.planSeq) return;
      ui.plan = null;
      ui.planError = err.message;
      retry = err.busy || err.status === 0;  // "try again in a moment", not an answer about these inputs
    }
  }
  ui.planFor = inputs;
  ui.planPending = '';
  ui.planRetry = retry ? inputs : '';
  ui.planRetryAt = retry ? Date.now() + PREFLIGHT_RETRY_MS : 0;
  if (retry) {
    clearTimeout(preflightTimer);
    preflightTimer = setTimeout(() => {
      if (ui.view === 'launch') schedulePreflight(0);  // other views plan again when LAUNCH renders
    }, PREFLIGHT_RETRY_MS);
  }
  if (ui.view === 'launch') {
    renderPreflight();
    renderModules();
  }
}

/** One line per distinct issue, naming every arm it applies to. */
function groupByArm(pairs) {
  const byText = new Map();
  for (const [arm, text] of pairs) byText.set(text, [...(byText.get(text) || []), arm]);
  return [...byText].map(([text, arms]) => (arms.length > 1 ? `${text} (${arms.join(', ')})` : `${arms[0]}: ${text}`));
}

/**
 * What the LAUNCH control does now. 'run' starts the run (POST /api/runs). 'queue' while the queue is running,
 * when the server refuses direct runs: the setup is saved as a design and added to the end of the queue.
 */
const launchMode = () => {
  const q = stateBlock('queue');
  return q && q.state === 'running' ? 'queue' : 'run';
};

/** Why the launch control is locked, or '' when it isn't. */
function launchBlock() {
  const s = ui.state;
  const queued = launchMode() === 'queue';
  if (!queued && s.active_run) return 'A run is in progress.';
  if (ui.planPending || ui.planFor !== planInputs()) return 'Running preflight…';
  if (ui.planError) return 'Preflight could not plan this run.';
  if (!ui.plan) return 'Preflight has not run yet.';
  if (ui.plan.problems.length) return 'Clear the problems above first.';
  if (!queued && !dockerOk()) return 'Docker is not running.';
  return '';
}

function renderPreflight() {
  const box = $('#preflight');
  if (!box) return;
  const p = ui.plan;
  const checking = Boolean(ui.planPending) || ui.planFor !== planInputs();
  let status;
  if (checking) status = [led('warn live'), h('span', {class: 'warn-text'}, 'CHECKING')];
  else if (ui.planError) status = [led('bad'), h('span', {class: 'bad-text'}, 'BLOCKED')];
  else if (p && p.problems.length) status = [led('bad'), h('span', {class: 'bad-text'}, `${plural(p.problems.length, 'PROBLEM').toUpperCase()}`)];
  else if (p && !dockerOk()) status = [led('bad'), h('span', {class: 'bad-text'}, 'DOCKER DOWN')];
  else if (p) status = [led('ok'), h('span', {class: 'ok-text'}, 'CLEAR')];
  else status = [led(), h('span', {class: 'muted'}, 'STANDING BY')];

  const issues = [];
  if (ui.planError) issues.push(['bad', ui.planError]);
  if (p) {
    // Problems not tied to an arm (container limits and the like) are listed once, as the server words them.
    const armProblems = new Set(p.arms.flatMap((a) => a.problems.map((x) => `${a.name}: ${x}`)));
    for (const text of p.problems.filter((x) => !armProblems.has(x))) issues.push(['bad', text]);
    for (const text of groupByArm(p.arms.flatMap((a) => a.problems.map((x) => [a.name, x])))) issues.push(['bad', text]);
    for (const text of p.warnings || []) issues.push(['warn', text]);
    for (const text of groupByArm(p.arms.flatMap((a) => a.warnings.map((x) => [a.name, x])))) issues.push(['warn', text]);
  }
  if (!dockerOk()) issues.push(['bad', 'Docker is not running. Start Docker Desktop; the status updates on its own.']);

  const capped = p ? p.arms.length - p.uncapped_arms.length : 0;
  const reserve = p ? groupByArm(p.arms.filter((a) => a.reserve_note).map((a) => [a.name, a.reserve_note])) : [];
  box.classList.toggle('stale', checking && Boolean(p));
  fill(box,
    h('div', {class: 'section-head'}, eyebrow('Preflight'), h('div', {class: 'preflight-status'}, status)),
    p && h('ul', {class: 'preflight-lines'},
      h('li', null, h('span', null, 'Trials'), h('span', null, `${p.trials} = ${plural(p.arms.length, 'arm')} × ${p.tasks.length} × ${p.n_attempts}`)),
      h('li', null, h('span', null, 'Run budget'), h('span', null, `${money(p.budget_usd)} hard stop`)),
      h('li', null, h('span', null, 'Worst case, capped arms'), h('span', null, capped ? money(p.worst_case_capped_usd) : DASH)),
      p.uncapped_arms.length > 0 && h('li', null, h('span', null, 'No dollar cap'), h('span', {class: 'warn-text'}, p.uncapped_arms.join(', '))),
      isNum(p.peak_containers) && h('li', null, h('span', null, 'Docker'), h('span', null, `Peak ${plural(p.peak_containers, 'container')}`))),
    p && p.uncapped_arms.length > 0 && h('p', {class: 'mono faint', style: 'font-size:11px;margin:8px 0 0'},
      'The run budget still applies to uncapped arms and is checked after each trial.'),
    issues.length > 0 && h('ul', {class: 'issues'}, issues.map(([kind, text]) => h('li', {class: kind === 'bad' ? 'bad-text' : 'warn-text'},
      h('span', null, kind === 'bad' ? '✗' : '!'), h('span', null, text)))),
    reserve.length > 0 && h('div', {class: 'reserve-notes'},
      h('div', {class: 'eyebrow'}, 'Held back while trials run'),
      reserve.map((text) => h('small', null, text))),
    p && p.arms.some((a) => a.command) && h('details', {style: 'margin-top:10px'},
      h('summary', null, 'Harbor commands'),
      p.arms.map((a) => [h('div', {class: 'cmd-name'}, a.name), h('pre', {class: 'cmd'}, a.command || DASH)])));

  const queued = launchMode() === 'queue';
  const block = launchBlock();
  $('#mission-meta').textContent = ui.mode ? ui.mode.toUpperCase() : '';
  // Update the control in place: rebuilding it on every poll would drop keyboard focus.
  let button = $('#launch');
  if (!button) {
    button = h('button', {type: 'button', class: 'btn primary launch-btn', id: 'launch', onclick: launch},
      h('span', {class: 'launch-label'}, '▶ LAUNCH'), h('small'));
    fill($('#launch-section'), button, h('div', {class: 'launch-note'}));
  }
  button.disabled = Boolean(block) || ui.launching;
  button.classList.toggle('queue-mode', queued);
  button.querySelector('.launch-label').textContent = queued ? '▶ ADD TO QUEUE' : '▶ LAUNCH';
  button.querySelector('small').textContent = !p ? 'no plan yet'
    : queued ? `${plural(p.trials, 'trial')} · runs when the queue reaches it` : `${plural(p.trials, 'trial')} · stops at ${money(p.budget_usd)}`;
  $('#launch-section .launch-note').textContent = block
    || (DEMO ? 'Demo mode: the confirm opens, but nothing is sent.'
      : queued ? 'The queue is running, so a direct run is refused. This saves the setup as a design and adds it to the queue.'
        : 'Ready. Launch asks once more, then bills your provider accounts.');
}

/**
 * LAUNCH, step one: the ARM confirm shows the plan's budget, worst case under caps and the uncapped arms; its
 * confirm button is step two and sends POST /api/runs {mode, arms}. While the queue runs, Add to queue instead.
 */
async function launch() {
  if (ui.launching || launchBlock()) return;
  if (launchMode() === 'queue') {
    await addLaunchToQueue();
    return;
  }
  const plan = ui.plan;
  const mode = ui.mode;
  const arms = armedNames();
  const inputs = planInputs();
  const capped = plan.arms.length - plan.uncapped_arms.length;
  const queueProblem = stateError('queue');
  ui.launching = true;
  renderPreflight();
  try {
    // The global cap as GET /api/queue has it: state.queue cannot tell "no cap" from "a cap whose spend is unknown".
    await Promise.race([refreshQueue().catch(() => null), new Promise((r) => setTimeout(r, 3000))]);
    const run = await armConfirm({
      title: 'Launch this run?',
      summary: `Start ${plural(plan.trials, 'trial')}? The run stops at ${money(plan.budget_usd)}.`,
      rows: [
        ['Mode', (mode || '').toUpperCase()],
        ['Arms', arms.join(', ')],
        ['Trials', `${plural(plan.arms.length, 'arm')} · ${plural(plan.trials, 'trial')}`],
        ['Budget', money(plan.budget_usd)],
        ['Worst case, capped', capped ? money(plan.worst_case_capped_usd) : DASH],
        isNum(plan.peak_containers) && ['Peak containers', String(plan.peak_containers)],
        ['Global cap', queueProblem ? 'unknown' : capLeftText(queueView() || stateBlock('queue'))],
      ],
      warnings: [
        plan.uncapped_arms.length > 0 && `No dollar cap: ${plan.uncapped_arms.join(', ')}. The run budget still applies after every trial.`,
        ...(plan.warnings || []),
        queueProblem && `Queue status unavailable (${queueProblem}). The server still checks the global cap.`,
      ],
      bills: [...new Set(plan.arms.map((a) => a.provider).filter(Boolean))],
      ok: '▶ Launch run',
      send: async () => {
        // The dialog may have stayed open while settings, keys or the bench changed.
        if (planInputs() !== inputs) throw new Error('Settings changed since this opened. Cancel and launch again.');
        const block = launchMode() === 'queue' ? 'The queue started meanwhile. Cancel, then add this run to the queue.' : launchBlock();
        if (block) throw new Error(block);
        return api('POST', '/api/runs', {mode, arms});
      },
    });
    if (run) {
      toast(`Run ${run.id} launched`);
      await poll();
    }
  } finally {
    ui.launching = false;
    if (ui.view === 'launch') renderPreflight();
    refocusLaunch();
  }
}

/** The LAUNCH control was disabled while its dialog was open, so focus fell to the page: put it back. */
function refocusLaunch() {
  const button = $('#launch');
  const lost = !document.activeElement || document.activeElement === document.body;
  if (lost && button && button.isConnected && !button.disabled) button.focus({preventScroll: true});
}

/**
 * Add to queue from LAUNCH: POST /api/designs/from-config saves this setup as a draft design, then the queue's
 * ARM confirm (enqueueDesign) admits that exact version. Each save is a new draft, so while the inputs are
 * unchanged a draft saved by an earlier press that was not queued is used again.
 */
async function addLaunchToQueue() {
  const button = $('#launch');
  const inputs = planInputs();
  ui.launching = true;
  renderPreflight();
  try {
    const detail = await busy(button, () => launchDraft(inputs));
    if (!detail) return;
    const problems = (detail.check && detail.check.validation && detail.check.validation.problems) || [];
    const open = {label: 'Open design', run: () => openDesign(detail.design_id, detail.version)};
    if (problems.length) {
      toast(`Saved as ${detail.title} v${detail.version}, but it cannot be queued: ${problems[0]}`
        + `${problems.length > 1 ? ` (and ${plural(problems.length - 1, 'more problem')})` : ''}`, true, open);
      return;
    }
    const added = await enqueueDesign(detail);
    if (added) ui.launchDraft = null;  // queued: the next press saves a new design
    else toast(`Not queued. The draft ${detail.title} v${detail.version} stays in DESIGNS.`, false, open);
  } finally {
    ui.launching = false;
    if (ui.view === 'launch') renderPreflight();
    refocusLaunch();
  }
}

/** The VersionDetail to queue: the earlier unqueued draft for these inputs, or a new one from the config. */
async function launchDraft(inputs) {
  const kept = ui.launchDraft;
  if (kept && kept.inputs === inputs) {
    try {
      const detail = await api('GET', `/api/designs/${enc(kept.design_id)}/versions/${kept.version}`);
      if (detail.status === 'draft' && !detail.archived) return detail;
    } catch { /* gone or unreadable: save a new one */ }
  }
  const mode = ui.mode;
  const arms = armedNames();
  const title = `${titleCase(mode)} from LAUNCH: ${arms.join(', ')}`.slice(0, 80);
  const detail = await api('POST', '/api/designs/from-config', {mode, arms, title});
  ui.launchDraft = {inputs, design_id: detail.design_id, version: detail.version};
  refreshDesigns();
  return detail;
}

// Run sources ------------------------------------------------------------
//
// A queued run's record carries design_ref {design_id, version, author, conversation_id, queue_item} but no
// title. The queue's items carry titles; otherwise one GET /api/designs names every design at once.

const designTitles = {map: new Map(), asked: new Set(), loading: null, at: 0};

/** The design's title for a design_ref, or null while unknown (a lookup starts; the views redraw when it lands). */
function designTitle(ref) {
  if (!ref || !ref.design_id) return null;
  const q = stateBlock('queue');
  const view = queueView();
  const items = [q && q.active_item, ...(view ? [view.active, ...(view.waiting || []), ...(view.held || []), ...(view.history || [])] : [])];
  const item = items.find((it) => it && it.design_id === ref.design_id && it.version === ref.version && it.title);
  if (item) return item.title;
  if (designTitles.map.has(ref.design_id)) return designTitles.map.get(ref.design_id);
  designTitles.asked.add(ref.design_id);
  loadDesignTitles();
  return null;
}

function loadDesignTitles() {
  if (designTitles.loading || Date.now() - designTitles.at < 30000) return;
  designTitles.at = Date.now();
  const asked = [...designTitles.asked];
  designTitles.loading = api('GET', '/api/designs?filter=all').then((data) => {
    for (const d of data.designs || []) if (d.id && d.title) designTitles.map.set(d.id, d.title);
    for (const id of asked) if (!designTitles.map.has(id)) designTitles.map.set(id, null);  // archived or gone: the id stands in
  }, () => { /* the ids stand in; asked again after 30 s */ }).finally(() => {
    designTitles.loading = null;
    designTitles.asked.clear();
    if (ui.view === 'live' && activeRun()) renderLive(false);
    else if (ui.view === 'analysis') renderAnalysis(false);
  });
}

/** Where a run came from: "LAUNCH", or the design chip (title vN with its author) and the queue item. */
function runSource(run, {queueItem = true} = {}) {
  const ref = run && run.design_ref;
  if (ref && ref.design_id) {
    return [
      designChip({...ref, title: designTitle(ref)}, {author: true}),
      queueItem && ref.queue_item && h('button', {type: 'button', class: 'link-btn', onclick: (e) => {
        e.stopPropagation();
        openQueue(ref.queue_item);
      }}, `Queue item ${ref.queue_item}`),
    ];
  }
  return run && run.source === 'design' ? tag('design') : h('span', {class: 'src-launch'}, 'LAUNCH');
}

/**
 * The per-arm counters a running arm reports (missing on old runs, so a missing field shows nothing): trials in
 * flight, budget held back for them, trials killed or without usage (their spend is not counted), and containers
 * that may remain after a failed cleanup.
 */
function armFlags(a) {
  const open = a.status === 'running' || a.status === 'preflight';
  const flags = [
    isNum(a.trials_running) && (open || a.trials_running > 0)
      && tip(h('span', {class: 'chip'}, `${a.trials_running} running`), 'Running now', ['Trials Harbor has started and not finished']),
    isNum(a.reserve_usd) && (open || a.reserve_usd > 0)
      && tip(h('span', {class: 'chip'}, `${money(a.reserve_usd)} held back`), 'Held back',
        ['Budget reserved for the trials still running, so the run stops before they could take it past the budget']),
    isNum(a.trials_killed) && a.trials_killed > 0
      && h('span', {class: 'chip warn-text'}, `${a.trials_killed} killed (spend not counted)`),
    isNum(a.trials_unmeasured) && a.trials_unmeasured > 0
      && h('span', {class: 'chip warn-text'}, `${a.trials_unmeasured} without usage (spend not counted)`),
    a.cleanup_pending === true && tip(tag('! containers may remain', 'warn'), 'Cleanup pending',
      ['Some of this arm\'s containers may still exist. The next start retries the cleanup.']),
  ].filter(Boolean);
  return flags.length ? h('div', {class: 'lane-flags'}, flags) : null;
}

// LIVE -------------------------------------------------------------------

function newLive(run) {
  return {
    runId: run.id, ended: false, seeded: false, resultsSeq: 0, results: null,
    samples: [{t: Date.parse(run.created_at), spend: 0}],
    events: [], seen: new Set(), fresh: new Set(), order: {}, armStatus: {},
    logArm: null, logPicked: false, logSeq: 0,
  };
}

function pushEvent(e) {
  const L = ui.live;
  L.events.unshift({t: Date.now(), isNew: true, ...e});
  if (L.events.length > 80) L.events.length = 80;
}

/** Collects telemetry for the active run on every poll, whatever view is open. */
async function trackLive() {
  const s = ui.state;
  const run = activeRun();
  if (!run) {
    const L = ui.live;
    if (L && !L.ended) {
      L.ended = true;
      const final = s.runs.find((r) => r.id === L.runId);
      if (final) {
        pushEvent({glyph: '■', cls: final.status === 'done' ? 'ok-text' : 'warn-text', who: 'RUN', text: statusLabel(final.status).toUpperCase(), value: money(final.spend_usd)});
        toast(`Run ${final.id} ${statusLabel(final.status)}`);
      }
      try {
        L.results = await api('GET', `/api/runs/${enc(L.runId)}/results`);
      } catch { /* the standby view shows what the state already has */ }
      if (ui.view === 'live') renderLive(true);
    }
    return;
  }
  if (!ui.live || ui.live.runId !== run.id) ui.live = newLive(run);
  const L = ui.live;
  const last = L.samples[L.samples.length - 1];
  if (last.spend !== run.spend_usd || Date.now() - last.t > 15000) L.samples.push({t: Date.now(), spend: run.spend_usd});
  if (L.samples.length > 600) L.samples.splice(1, L.samples.length - 600);

  for (const a of run.arms) {
    const before = L.armStatus[a.name];
    if (before !== undefined && before !== a.status) {
      const cls = a.status === 'failed' ? 'bad-text' : a.status === 'done' ? 'ok-text' : OPEN_ARM.has(a.status) ? 'ok-text' : 'warn-text';
      pushEvent({glyph: '▸', cls, who: a.name, text: `→ ${statusLabel(a.status).toUpperCase()}`, value: a.message ? a.message.slice(0, 40) : ''});
    }
    L.armStatus[a.name] = a.status;
  }

  const seq = ++L.resultsSeq;
  try {
    const data = await api('GET', `/api/runs/${enc(run.id)}/results`);
    if (seq !== L.resultsSeq || ui.live !== L) return;
    let added = 0;
    for (const t of data.trials) {
      const key = trialKey(t);
      if (L.seen.has(key)) continue;
      L.seen.add(key);
      (L.order[t.arm] = L.order[t.arm] || []).push(key);
      added++;
      if (!L.seeded) continue;
      L.fresh.add(key);
      const o = outcomeOf(t);
      pushEvent({glyph: OUTCOME[o].glyph, cls: OUTCOME[o].text, who: t.arm, text: `${t.task} ${OUTCOME[o].label.toUpperCase()}`,
        value: `${usageValue(trialCost(t), t.usage_status)} · ${minutes(t.minutes)}m`});
    }
    if (!L.seeded) {
      pushEvent({glyph: '◆', cls: 'ok-text', who: 'LINK', text: `monitoring ${run.id}`, value: `${plural(added, 'trial')} on record`});
      L.seeded = true;
    }
    L.results = data;
  } catch { /* keep the last results; the next poll retries */ }
  if (ui.view === 'live' && activeRun()) {
    renderLive(false);
    refreshLog();
  }
}

function renderLive(force = false) {
  const root = $('#view-live');
  const run = activeRun();
  if (!run) {
    renderStandby(root, force);
    return;
  }
  if (!ui.live || ui.live.runId !== run.id) ui.live = newLive(run);
  if (root.dataset.run !== run.id || !root.querySelector('.run-strip')) {
    buildMonitor(root, run);
    force = true;
  }
  updateMonitor(run, force);
}

function buildMonitor(root, run) {
  root.dataset.run = run.id;
  for (const key of [...rendered.keys()]) if (key.startsWith('m-')) rendered.delete(key);
  const legend = h('div', {class: 'legend'},
    h('span', null, cell('pass'), 'pass'), h('span', null, cell('fail'), 'fail'), h('span', null, cell('error'), 'error'),
    h('span', null, cell('run'), 'in flight'), h('span', null, cell(''), 'queued'));
  fill(root,
    h('div', {class: 'run-strip'},
      h('div', {class: 'rs-id'}, led('ok live'), h('div', null, eyebrow('Run'), h('span', {class: 'rs-value', id: 'm-id'}))),
      h('div', null, eyebrow('Mode'), h('span', {class: 'rs-value', id: 'm-mode'})),
      h('div', {class: 'rs-source'}, eyebrow('Source'), h('span', {class: 'rs-value', id: 'm-source'})),
      h('div', null, eyebrow('Status'), h('span', {class: 'rs-value', id: 'm-status'})),
      h('div', null, eyebrow('Elapsed'), h('span', {class: 'rs-value clock', id: 'm-elapsed'}, '00:00:00')),
      h('div', null, eyebrow('Started'), h('span', {class: 'rs-value', id: 'm-started'})),
      h('div', {class: 'rs-end'}, h('button', {type: 'button', class: 'btn danger', id: 'abort', onclick: abortRun}, '■ Abort run'))),
    h('div', {class: 'kpis', id: 'm-kpis'}),
    h('div', {class: 'grid live-grid'},
      panel({code: 'M-01', title: 'Arm lanes', meta: legend, cls: 'p-lanes hot', flush: true, bodyId: 'm-lanes'}),
      panel({code: 'M-02', title: 'Spend telemetry', meta: h('span', {id: 'm-tele-meta'}), cls: 'p-tele', bodyId: 'm-tele'}),
      panel({code: 'M-03', title: 'Event feed', meta: h('span', {id: 'm-feed-meta'}), cls: 'p-feed', flush: true},
        h('ul', {class: 'feed', id: 'm-feed'})),
      panel({code: 'M-04', title: 'Console', meta: h('span', {id: 'm-console-meta'}), cls: 'p-console', flush: true},
        h('div', {class: 'console-tabs', id: 'm-console-tabs'}),
        h('pre', {class: 'log', id: 'm-log'}, 'Connecting to the log…'))));
}

/** Projection, burn rate and ETA from what the run has finished so far. */
function monitorModel(run, L) {
  const trials = (L.results && L.results.trials) || [];
  const config = run.config && run.config.run ? run.config.run : ui.state.config.run;
  const nConc = config.n_concurrent || 1;
  const byArm = {};
  for (const t of trials) (byArm[t.arm] = byArm[t.arm] || []).push(t);
  const arms = run.arms.map((a) => {
    const ts = byArm[a.name] || [];
    const counts = {pass: 0, fail: 0, error: 0, unscored: 0};
    for (const t of ts) counts[outcomeOf(t)]++;
    const done = Math.max(a.trials_done, ts.length);
    return {...a, ts, counts, done, perTrial: done ? a.spend_usd / done : null, open: OPEN_ARM.has(a.status)};
  });
  const doneTotal = sum(arms, (a) => a.done);
  const spend = run.spend_usd || 0;
  const meanPerTrial = doneTotal ? sum(arms, (a) => (a.done ? a.spend_usd : 0)) / doneTotal : null;
  let projected = 0;
  for (const a of arms) {
    if (!a.open) projected += a.spend_usd;
    else if (isNum(a.perTrial)) projected += a.perTrial * a.trials_total;
    else if (isNum(meanPerTrial)) projected += meanPerTrial * a.trials_total;
    else { projected = null; break; }
  }
  const total = sum(arms, (a) => (a.open ? a.trials_total : a.done));
  const elapsedMin = (Date.now() - Date.parse(run.created_at)) / 60000;
  const remaining = sum(arms, (a) => (a.open ? Math.max(0, a.trials_total - a.done) : 0));
  const rate = elapsedMin > 0 ? doneTotal / elapsedMin : 0;
  const eta = remaining === 0 ? 0 : rate > 0 ? remaining / rate : null;
  const counts = {pass: 0, fail: 0, error: 0, unscored: 0};
  for (const a of arms) for (const k of Object.keys(counts)) counts[k] += a.counts[k];
  const inflight = sum(arms, (a) => (a.status !== 'running' ? 0
    : isNum(a.trials_running) ? a.trials_running : Math.min(nConc, Math.max(0, a.trials_total - a.done))));
  return {
    arms, nConc, spend, projected, meanPerTrial, doneTotal, total, remaining, eta, counts, inflight,
    burn: elapsedMin > 0.25 ? spend / elapsedMin : null,
    budget: run.budget_usd,
    reserve: isNum(run.reserve_usd) ? run.reserve_usd : null,  // held back for trials in flight; missing on old runs
    plan: run.plan || {},
  };
}

function updateMonitor(run, force) {
  const L = ui.live;
  const m = monitorModel(run, L);
  $('#m-id').textContent = run.id;
  $('#m-mode').textContent = (run.mode || '').toUpperCase();
  const title = designTitle(run.design_ref);
  if (force || changed('m-source', [run.source, run.design_ref, title])) fill($('#m-source'), runSource(run));
  fill($('#m-status'), statusTag(run.status));
  $('#m-started').textContent = clockTime(Date.parse(run.created_at));
  $('#m-elapsed').textContent = hms(Date.now() - Date.parse(run.created_at));
  if (force || changed('m-kpis', [m.spend, m.projected, m.burn && m.burn.toFixed(3), m.doneTotal, m.total, m.counts, m.eta && Math.round(m.eta),
    m.inflight, m.reserve])) {
    renderKpis(m);
  }
  if (force || changed('m-lanes', [run.arms, L.results && L.results.trials.length, m.nConc])) renderLanes(run, m);
  if (force || changed('m-tele', [L.samples.length, m.spend, m.projected, m.eta && Math.round(m.eta)])) renderTelemetry(run, m);
  if (force || changed('m-feed', [L.events.length, L.events[0] && L.events[0].t])) renderFeed();
  if (force || changed('m-tabs', [run.arms.map((a) => [a.name, a.status]), L.logArm])) renderConsoleTabs(run);
  L.fresh.clear();
}

function kpi(label, value, sub, {cls = '', extra = null, unit = null} = {}) {
  return h('div', {class: `kpi ${cls}`.trim()},
    eyebrow(label),
    h('div', {class: 'k-value'}, value, unit && h('small', null, unit)),
    extra,
    h('div', {class: 'k-sub', title: typeof sub === 'string' ? sub : null}, sub));
}

function gauge(fraction, {marker = null, tone = ''} = {}) {
  const clamp = (v) => Math.max(0, Math.min(1, v));
  return h('div', {class: 'gauge'},
    h('div', {class: `fill ${tone}`.trim(), style: `width:${(clamp(fraction) * 100).toFixed(1)}%`}),
    isNum(marker) && h('div', {class: marker > 1 ? 'proj over' : 'proj', style: `left:${(clamp(marker) * 100).toFixed(1)}%`}));
}

function renderKpis(m) {
  const spendShare = m.budget ? m.spend / m.budget : 0;
  const projShare = isNum(m.projected) && m.budget ? m.projected / m.budget : null;
  const over = isNum(projShare) && projShare > 1;
  const scored = m.counts.pass + m.counts.fail;
  const finishAt = isNum(m.eta) ? clockTime(Date.now() + m.eta * 60000) : null;
  const held = isNum(m.reserve) && m.reserve > 0 ? ` · ${money(m.reserve)} held back` : '';
  fill($('#m-kpis'),
    kpi('Spend', money(m.spend), `${pct(spendShare)} of ${money(m.budget)} budget${held}`,
      {cls: 'hot', extra: gauge(spendShare, {marker: projShare, tone: spendShare > 0.9 ? 'bad' : spendShare > 0.75 ? 'warn' : ''})}),
    kpi('Projected', isNum(m.projected) ? money(m.projected) : DASH,
      over ? `over budget: stops at ${money(m.budget)}`
        : isNum(m.projected) ? `from ${plural(m.doneTotal, 'finished trial')}` : 'needs a finished trial',
      {cls: over ? 'alert' : ''}),
    kpi('Burn rate', isNum(m.burn) ? money(m.burn) : DASH, isNum(m.meanPerTrial) ? `${money(m.meanPerTrial)} per trial` : 'per trial: –',
      {unit: isNum(m.burn) ? '/min' : null}),
    kpi('Progress', String(m.doneTotal), `${pct(m.total ? m.doneTotal / m.total : 0)} done · ${m.inflight} in flight`,
      {unit: `/ ${m.total}`, extra: gauge(m.total ? m.doneTotal / m.total : 0)}),
    kpi('Outcomes', h('span', {class: 'outcomes'},
      tip(h('span', null, cell('pass'), String(m.counts.pass)), 'Passed', [[String(m.counts.pass), 'trials']]),
      tip(h('span', null, cell('fail'), String(m.counts.fail)), 'Failed', [[String(m.counts.fail), 'trials']]),
      tip(h('span', null, cell('error'), String(m.counts.error)), 'Errors', [[String(m.counts.error), 'trials']])),
    scored ? `pass rate ${pct(m.counts.pass / scored)} of ${scored} scored` : 'no scored trials yet'),
    kpi('ETA', isNum(m.eta) ? (m.eta === 0 ? 'now' : `≈ ${duration(m.eta)}`) : DASH,
      finishAt ? `finishes ≈ ${finishAt.slice(0, 5)}` : 'needs a finished trial'));
}

function renderLanes(run, m) {
  const L = ui.live;
  const planArms = Object.fromEntries(((run.plan && run.plan.arms) || []).map((a) => [a.name, a]));
  fill($('#m-lanes'), m.arms.map((a) => {
    const pa = planArms[a.name] || findArm(a.name) || {};
    const scored = a.counts.pass + a.counts.fail;
    const projection = !a.open ? a.spend_usd : isNum(a.perTrial) ? a.perTrial * a.trials_total : null;
    const progress = a.trials_total ? a.done / a.trials_total : 0;
    const harness = a.family || pa.family || pa.agent;
    return h('div', {class: OPEN_ARM.has(a.status) && a.status !== 'queued' ? 'lane active' : 'lane'},
      h('div', null,
        h('div', {class: 'lane-name'}, led(STATUS_LED[a.status] || ''), a.name),
        h('div', {class: 'lane-meta', title: pa.agent && pa.agent !== harness ? `harness ${pa.agent}` : null},
          [harness, pa.model, pa.effort && `effort ${pa.effort}`].filter(Boolean).join(' · '))),
      h('div', {class: 'lane-status'},
        h('span', {class: 'st'}, statusTag(a.status)),
        h('small', {title: a.message || null}, a.message ? a.message.slice(0, 60)
          : a.finished_at ? `ended ${clockTime(Date.parse(a.finished_at))}`
            : a.started_at ? `since ${clockTime(Date.parse(a.started_at))}` : 'waiting')),
      h('div', null, laneCells(a, L, m.nConc), h('div', {class: 'lane-progress'}, h('i', {style: `width:${(progress * 100).toFixed(1)}%`})),
        armFlags(a)),
      h('div', {class: 'lane-stats'},
        h('span', {class: 'lbl'}, 'TRIALS'), h('span', {class: 'lbl'}, 'PASS · FAIL · ERR'), h('span', {class: 'lbl'}, 'SPEND'),
        h('span', {class: 'big'}, `${a.done}/${a.trials_total}`),
        h('span', {class: 'big'}, h('span', {class: 'ok-text'}, a.counts.pass), ' · ', h('span', {class: 'bad-text'}, a.counts.fail), ' · ',
          h('span', {class: 'warn-text'}, a.counts.error)),
        h('span', {class: 'big'}, money(a.spend_usd)),
        h('span', null, ''),
        h('span', {class: 'muted'}, scored ? pct(a.counts.pass / scored) : DASH),
        h('span', {class: 'muted'}, isNum(projection) ? `≈ ${money(projection)}` : DASH)));
  }));
}

function laneCells(a, L, nConc) {
  const byKey = new Map(a.ts.map((t) => [trialKey(t), t]));
  const keys = (L.order[a.name] || []).filter((k) => byKey.has(k));
  const cells = keys.map((key) => {
    const t = byKey.get(key);
    const o = outcomeOf(t);
    return tip(cell(o, L.fresh.has(key) ? 'fresh' : ''), t.task, [
      [OUTCOME[o].label, 'outcome'],
      [usageValue(trialCost(t), t.usage_status), 'cost'],
      [`${minutes(t.minutes)} min`, 'duration'],
      [usageValue((t.n_input || 0) + (t.n_output || 0), t.usage_status, tokens), 'tokens'],
      t.error ? `${t.error}: ${(t.error_message || '').slice(0, 120)}` : null,
    ].filter(Boolean));
  });
  const pendingDetail = Math.max(0, a.trials_done - keys.length);
  for (let i = 0; i < pendingDetail; i++) cells.push(tip(cell('unscored'), 'Finished', ['Details arrive with the next poll']));
  const remaining = Math.max(0, a.trials_total - keys.length - pendingDetail);
  const running = isNum(a.trials_running) ? a.trials_running : nConc;  // the arm's own count, when the bench reports it
  const inflight = a.status === 'running' ? Math.min(running, remaining) : 0;
  for (let i = 0; i < inflight; i++) cells.push(tip(cell('run'), 'In flight', ['Trial running now']));
  for (let i = inflight; i < remaining; i++) cells.push(cell(a.open ? '' : 'void'));
  return h('div', {class: a.trials_total > 48 ? 'cells dense' : 'cells'}, cells);
}

function renderTelemetry(run, m) {
  const box = $('#m-tele');
  const L = ui.live;
  const W = Math.max(280, box.clientWidth - 28 || 340);
  const H = 190;
  const pad = {l: 50, r: 12, t: 14, b: 24};
  const t0 = Date.parse(run.created_at);
  const now = Date.now();
  const end = isNum(m.eta) && m.eta > 0 ? now + m.eta * 60000 : now;
  const tMax = Math.max(end, t0 + 60000);
  const yMax = niceCeil(Math.max(m.budget || 0, m.projected || 0, m.spend) * 1.05);
  const x = (t) => pad.l + ((t - t0) / (tMax - t0)) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - v / yMax) * (H - pad.t - pad.b);
  const points = [...L.samples.filter((s) => s.t >= t0), {t: now, spend: m.spend}];
  const line = points.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p.spend).toFixed(1)}`).join('');
  const area = `${line}L${x(now).toFixed(1)},${y(0)}L${x(points[0].t).toFixed(1)},${y(0)}Z`;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax);
  const chart = svg('svg', {class: 'chart', viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img',
    'aria-label': `Spend ${money(m.spend)} of ${money(m.budget)} budget`},
    svg('defs', null, svg('linearGradient', {id: 'spendFill', x1: 0, y1: 0, x2: 0, y2: 1},
      svg('stop', {offset: '0%', 'stop-color': '#4cbb17', 'stop-opacity': 0.32}),
      svg('stop', {offset: '100%', 'stop-color': '#4cbb17', 'stop-opacity': 0}))),
    ticks.map((v) => [
      svg('line', {class: 'grid-line', x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v)}),
      svg('text', {x: pad.l - 8, y: y(v) + 3.5, 'text-anchor': 'end'}, v >= 100 ? `$${Math.round(v)}` : `$${v.toFixed(v < 10 ? 2 : 0)}`),
    ]),
    m.budget && svg('line', {x1: pad.l, x2: W - pad.r, y1: y(m.budget), y2: y(m.budget), stroke: '#f0b03a', 'stroke-width': 1}),
    m.budget && svg('text', {x: W - pad.r, y: y(m.budget) - 5, 'text-anchor': 'end', style: 'fill:#f0b03a'}, `BUDGET ${money(m.budget)}`),
    svg('path', {d: area, fill: 'url(#spendFill)'}),
    isNum(m.projected) && end > now && svg('line', {x1: x(now), y1: y(m.spend), x2: x(end), y2: y(m.projected),
      stroke: '#74df3c', 'stroke-width': 1.5, 'stroke-dasharray': '4 4', opacity: 0.7}),
    isNum(m.projected) && end > now && svg('circle', {cx: x(end), cy: y(m.projected), r: 3, fill: 'none', stroke: '#74df3c', 'stroke-width': 1.5}),
    svg('path', {d: line, fill: 'none', stroke: '#4cbb17', 'stroke-width': 2, 'stroke-linejoin': 'round'}),
    svg('circle', {cx: x(now), cy: y(m.spend), r: 6, fill: '#4cbb17', opacity: 0.25}),
    svg('circle', {cx: x(now), cy: y(m.spend), r: 3, fill: '#74df3c'}),
    svg('line', {class: 'axis', x1: pad.l, x2: W - pad.r, y1: y(0), y2: y(0)}),
    svg('text', {x: pad.l, y: H - 6}, clockTime(t0).slice(0, 5)),
    svg('text', {x: x(now), y: H - 6, 'text-anchor': end > now ? 'middle' : 'end'}, 'NOW'),
    end > now && svg('text', {x: W - pad.r, y: H - 6, 'text-anchor': 'end'}, `ETA ${clockTime(end).slice(0, 5)}`));
  // Crosshair: snaps to the nearest sample.
  const cross = svg('line', {x1: 0, x2: 0, y1: pad.t, y2: H - pad.b, stroke: '#b6c2b9', 'stroke-width': 1, opacity: 0});
  const hit = svg('rect', {class: 'hit', x: pad.l, y: pad.t, width: W - pad.l - pad.r, height: H - pad.t - pad.b});
  hit.addEventListener('pointermove', (e) => {
    const r = chart.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    let best = points[0];
    for (const p of points) if (Math.abs(x(p.t) - px) < Math.abs(x(best.t) - px)) best = p;
    cross.setAttribute('x1', x(best.t));
    cross.setAttribute('x2', x(best.t));
    cross.setAttribute('opacity', 0.6);
    showTip(clockTime(best.t), [[money(best.spend), 'spend'], [pct(m.budget ? best.spend / m.budget : 0), 'of budget']], e.clientX, e.clientY);
  });
  hit.addEventListener('pointerleave', () => {
    cross.setAttribute('opacity', 0);
    hideTip();
  });
  chart.append(cross, hit);
  fill(box, chart,
    h('div', {class: 'legend', style: 'margin-top:8px'},
      h('span', null, h('i', {class: 'sw line', style: 'background:#4cbb17'}), 'spend'),
      h('span', null, h('i', {class: 'sw line', style: 'background:repeating-linear-gradient(90deg,#74df3c 0 4px,transparent 4px 8px)'}), 'projection'),
      h('span', null, h('i', {class: 'sw line', style: 'background:#f0b03a'}), 'budget')));
  $('#m-tele-meta').textContent = isNum(m.burn) ? `${money(m.burn)}/MIN` : '';
}

function renderFeed() {
  const L = ui.live;
  fill($('#m-feed'), L.events.length
    ? L.events.map((e) => h('li', {class: e.isNew ? 'new' : null},
      h('span', {class: 't'}, clockTime(e.t)),
      h('span', {class: `g ${e.cls}`}, e.glyph),
      h('span', {class: 'what'}, h('b', null, e.who), ' ', e.text),
      h('span', {class: 'v'}, e.value || '')))
    : h('li', {class: 'empty'}, 'Waiting for the first event…'));
  for (const e of L.events) e.isNew = false;
  $('#m-feed-meta').textContent = `${L.events.length} EVENTS`;
}

function renderConsoleTabs(run) {
  const L = ui.live;
  const current = run.arms.find((a) => a.status === 'running' || a.status === 'preflight');
  if (!L.logPicked && current && current.name !== L.logArm) L.logArm = current.name;
  if (!L.logArm && run.arms.length) L.logArm = run.arms[0].name;
  fill($('#m-console-tabs'), run.arms.map((a) => h('button', {
    type: 'button', class: a.name === L.logArm ? 'active' : null,
    onclick: () => {
      L.logArm = a.name;
      L.logPicked = true;
      renderConsoleTabs(run);
      refreshLog();
    },
  }, led(STATUS_LED[a.status] || ''), a.name)));
  $('#m-console-meta').textContent = L.logPicked ? 'PINNED' : 'FOLLOWING ACTIVE ARM';
}

async function refreshLog() {
  const L = ui.live;
  const run = activeRun();
  if (ui.view !== 'live' || !L || !run || !L.logArm) return;
  const seq = ++L.logSeq;
  let text;
  try {
    text = await api('GET', `/api/runs/${enc(run.id)}/log/${enc(L.logArm)}?lines=200`);
  } catch (err) {
    text = `Could not load the log: ${err.message}`;
  }
  const pre = $('#m-log');
  if (seq !== L.logSeq || !pre) return;
  const content = text || 'No output yet.';
  if (pre.textContent === content) return;
  const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24;
  pre.textContent = content;
  if (atBottom) pre.scrollTop = pre.scrollHeight;
}

async function abortRun(event) {
  const run = activeRun();
  if (!run) return;
  const button = event.currentTarget;
  const fromQueue = run.source === 'design';
  const ok = await confirmAction({
    code: 'ABORT', title: 'Abort run?',
    body: `Cancel run ${run.id}? Trials in progress are stopped and their containers removed.${fromQueue ? ' The queue then pauses.' : ''}`,
    ok: 'Abort run', tone: 'danger',
  });
  if (!ok) return;
  await busy(button, async () => {
    await api('POST', `/api/runs/${enc(run.id)}/cancel`);
    toast('Abort sent. Stopping trials…');
    poll();
  });
}

function renderStandby(root, force) {
  const s = ui.state;
  const L = ui.live;
  const last = (L && L.ended && s.runs.find((r) => r.id === L.runId)) || s.runs[0] || null;
  const keys = keyCoverage();
  const title = last ? designTitle(last.design_ref) : null;
  const signature = [last, s.readiness, keys.set, keys.needed.length, L && L.ended && L.results && L.results.trials.length, title];
  if (!force && root.dataset.run === '' && !changed('standby', signature)) return;
  rendered.set('standby', JSON.stringify(signature));
  root.dataset.run = '';
  let outcomes = null;
  if (L && last && L.runId === last.id && L.results) {
    const counts = {pass: 0, fail: 0, error: 0, unscored: 0};
    for (const t of L.results.trials) counts[outcomeOf(t)]++;
    outcomes = `${counts.pass} ✓ · ${counts.fail} ✕ · ${counts.error} !`;
  }
  const justEnded = L && L.ended && last && last.id === L.runId;
  const ending = !justEnded ? null
    : last.status === 'done' ? ['RUN ', 'COMPLETE', 'The run has finished. Its full breakdown is in Analysis.']
      : last.status === 'done_with_errors' ? ['RUN ', 'DONE', 'The run has finished, but some arms ended with errors. See Analysis and the arms\' logs.']
        : last.status === 'cancelled' ? ['RUN ', 'ABORTED', 'The run was aborted. Trials that finished are in Analysis.']
          : last.status === 'budget_stopped' ? ['BUDGET ', 'STOP', 'The run hit its budget and stopped. Trials that finished are in Analysis.']
            : ['RUN ', 'ENDED', `The run ended: ${statusLabel(last.status)}. See Analysis and the run's logs.`];
  const tally = (field) => (last ? sum(last.arms, (a) => (isNum(a[field]) ? a[field] : 0)) : 0);
  const leftovers = last ? last.arms.filter((a) => a.cleanup_pending === true).map((a) => a.name) : [];
  const caveats = [
    tally('trials_killed') > 0 && `${tally('trials_killed')} killed (spend not counted)`,
    tally('trials_unmeasured') > 0 && `${tally('trials_unmeasured')} without usage (spend not counted)`,
    leftovers.length > 0 && `Containers may remain from ${listText(leftovers)}; the next start retries the cleanup.`,
  ].filter(Boolean);
  fill(root, h('div', {class: 'standby'},
    h('section', {class: 'panel hot'},
      h('div', {class: 'panel-head'}, h('span', {class: 'panel-code'}, 'S-00'), h('h2', {class: 'panel-title'}, 'Live telemetry'),
        h('div', {class: 'panel-meta'}, led(), 'NO SIGNAL')),
      h('div', {class: 'standby-hero'},
        h('div', {class: 'big'}, ending ? [ending[0], h('span', null, ending[1])] : ['STAND', h('span', null, 'BY')]),
        h('p', null, ending ? ending[2]
          : 'No run in progress. Pick harnesses and models in Launch. This view switches to live telemetry the moment a run starts.'),
        last && h('div', null,
          h('div', {class: 'eyebrow', style: 'margin-bottom:8px'}, justEnded ? 'Final' : 'Last run'),
          h('dl', {class: 'last-run'},
            h('div', null, h('dt', null, 'Run'), h('dd', null, last.id)),
            h('div', null, h('dt', null, 'Status'), h('dd', null, statusTag(last.status))),
            h('div', null, h('dt', null, 'Spend'), h('dd', null, `${money(last.spend_usd)} / ${money(last.budget_usd)}`)),
            h('div', null, h('dt', null, outcomes ? 'Outcomes' : 'Trials'),
              h('dd', null, outcomes || `${sum(last.arms, (a) => a.trials_done)} / ${sum(last.arms, (a) => a.trials_total)}`))),
          h('div', {class: 'last-source'}, h('span', {class: 'eyebrow'}, 'Source'), runSource(last)),
          caveats.map((text) => h('p', {class: 'warn-text last-caveat'}, `! ${text}`))),
        h('div', {class: 'actions'},
          h('button', {type: 'button', class: 'btn primary', onclick: () => showView('launch')}, 'Open launch ', h('span', {class: 'arrow'}, '›')),
          last && h('button', {type: 'button', class: 'btn ghost', onclick: () => openAnalysis(last.id)}, 'Analyze ', last.id, h('span', {class: 'arrow'}, ' ›'))))),
    panel({code: 'S-01', title: 'Systems', meta: h('button', {type: 'button', class: 'btn ghost small', onclick: () => openDrawer('system')}, 'Keys ›'), flush: true},
      h('ul', {class: 'checklist'},
        s.readiness.map((c) => h('li', null, led(c.ok ? 'ok' : 'bad'), h('span', {class: 'name'}, c.name), h('span', {class: 'detail'}, c.detail))),
        h('li', null, led(!keys.needed.length ? '' : keys.set === keys.needed.length ? 'ok' : 'bad'), h('span', {class: 'name'}, 'API keys'),
          h('span', {class: 'detail'}, keys.needed.length
            ? keys.needed.map((k) => `${k.provider} ${k.set ? 'set' : 'missing'}`).join(' · ')
            : 'No armed module needs a key'))))));
}

// ANALYSIS ---------------------------------------------------------------

function openAnalysis(runId) {
  ui.analysisRun = runId;
  showView('analysis');
}

function renderAnalysis(force = false) {
  const root = $('#view-analysis');
  const runs = ui.state.runs;
  if (!root.querySelector('.analysis')) {
    fill(root, h('div', {class: 'analysis'},
      h('aside', {class: 'panel runlist'},
        h('div', {class: 'panel-head'}, h('span', {class: 'panel-code'}, 'A-00'), h('h2', {class: 'panel-title'}, 'Run log'),
          h('div', {class: 'panel-meta', id: 'a-count'})),
        h('ol', {id: 'a-runs'})),
      h('div', {id: 'a-detail'})));
    force = true;
  }
  if (!runs.some((r) => r.id === ui.analysisRun)) {
    const withTrials = runs.find((r) => r.arms.some((a) => a.trials_done > 0));
    ui.analysisRun = (withTrials || runs[0] || {}).id || null;
  }
  const listSig = [runs.map((r) => [r.id, r.status, r.spend_usd, sum(r.arms, (a) => a.trials_done), r.source, r.design_ref,
    designTitle(r.design_ref)]), ui.analysisRun];
  if (force || changed('a-runs', listSig)) {
    rendered.set('a-runs', JSON.stringify(listSig));
    $('#a-count').textContent = `${runs.length} ON RECORD`;
    // The source column: LAUNCH, or DESIGN with a chip that opens the design (beside the button, never inside it).
    fill($('#a-runs'), runs.length
      ? runs.map((r) => {
        const design = r.source === 'design';
        return h('li', {class: r.id === ui.analysisRun ? 'active' : null},
          h('button', {
            type: 'button', class: r.id === ui.analysisRun ? 'active' : null,
            onclick: () => {
              ui.analysisRun = r.id;
              renderAnalysis(true);
            },
          },
          led(STATUS_LED[r.status] || ''),
          h('span', {class: 'rid'}, r.id),
          h('span', {class: 'rspend'}, money(r.spend_usd)),
          h('span', {class: 'rmeta'}, [design ? 'DESIGN' : `LAUNCH ${(r.mode || '').toUpperCase()}`, statusLabel(r.status),
            plural(r.arms.length, 'arm'), when(r.created_at)].join(' · '))),
          design && r.design_ref && h('div', {class: 'rsource'}, runSource(r, {queueItem: false})));
      })
      : h('li', {class: 'empty'}, 'No runs on record.'));
  }
  loadAnalysis(force);
}

let analysisSeq = 0;
async function loadAnalysis(force) {
  const box = $('#a-detail');
  const id = ui.analysisRun;
  if (!id) {
    if (force || changed('a-empty', true)) {
      fill(box, h('section', {class: 'panel'}, h('div', {class: 'empty-state'},
        h('div', {class: 'big'}, 'NO RUNS ON RECORD'),
        h('p', null, 'Analysis fills in once a run has finished trials.'),
        h('button', {type: 'button', class: 'btn primary', onclick: () => showView('launch')}, 'Open launch ›'))));
    }
    return;
  }
  rendered.delete('a-empty');
  const run = ui.state.runs.find((r) => r.id === id);
  const active = id === ui.state.active_run;
  const source = JSON.stringify([id, run && run.status, run && run.arms.map((a) => a.trials_done), run && designTitle(run.design_ref)]);
  if (!force && !active && rendered.get('a-source') === source) return;
  rendered.set('a-source', source);
  if (box.dataset.run !== id) {
    box.style.opacity = '.5';
  }
  const seq = ++analysisSeq;
  try {
    const data = await api('GET', `/api/runs/${enc(id)}/results`);
    if (seq !== analysisSeq) return;
    const json = JSON.stringify([data, run]);
    if (force || box.dataset.run !== id || rendered.get('a-data') !== json) {
      rendered.set('a-data', json);
      renderAnalysisDetail(run, data);
    }
    box.dataset.run = id;
  } catch (err) {
    if (seq !== analysisSeq) return;
    rendered.delete('a-source');
    delete box.dataset.run;
    fill(box, h('section', {class: 'panel'}, h('div', {class: 'empty-state'},
      h('div', {class: 'big'}, 'NO DATA'), h('p', {class: 'bad-text'}, `Could not load results: ${err.message}`))));
  }
  box.style.opacity = '';
}

function renderAnalysisDetail(run, data) {
  const box = $('#a-detail');
  const armsWithData = data.arms.filter((a) => a.trials > 0);
  const finished = sum(data.arms, (a) => a.trials);
  const planned = run ? sum(run.arms, (a) => a.trials_total) : finished;
  const scored = sum(armsWithData, (a) => a.scored);
  const passed = sum(armsWithData, (a) => a.passed);
  const totalCost = sum(armsWithData, (a) => a.total_cost);
  // Means cover measured trials only: setup failures and trials without usage are left out of tokens and cost.
  const measuredOf = (a) => (isNum(a.measured) ? a.measured : a.trials);
  const measured = sum(armsWithData, measuredOf);
  const tokensTotal = sum(armsWithData, (a) => (isNum(a.mean_tokens) ? a.mean_tokens * measuredOf(a) : 0));
  const costMeasured = sum(armsWithData, (a) => (isNum(a.mean_cost) ? a.mean_cost * measuredOf(a) : 0));
  const unmeasured = sum(armsWithData, (a) => a.unmeasured || 0);
  const floors = sum(armsWithData, (a) => a.lower_bound || 0);
  const atLeast = (text) => (unmeasured || floors ? `≥ ${text}` : text);
  const errors = sum(armsWithData, (a) => a.errors);
  const live = run && run.id === ui.state.active_run;
  const minutesTaken = run && run.finished_at ? (Date.parse(run.finished_at) - Date.parse(run.created_at)) / 60000 : null;

  fill(box,
    h('div', {class: 'detail-head'},
      led(STATUS_LED[run ? run.status : ''] || ''),
      h('h1', null, data.run_id),
      run && statusTag(run.status),
      live && h('button', {type: 'button', class: 'btn primary small', onclick: () => showView('live')}, 'Watch live ›'),
      h('span', {class: 'meta'}, [
        run && (run.source === 'design' ? 'DESIGN' : `LAUNCH ${(run.mode || '').toUpperCase()}`),
        run && `started ${when(run.created_at)}`,
        isNum(minutesTaken) && `took ${duration(minutesTaken)}`,
        `baseline ${data.baseline}`,
      ].filter(Boolean).join(' · ')),
      run && run.design_ref && h('span', {class: 'detail-source'}, runSource(run))),
    h('div', {class: 'actions detail-actions'},
      h('button', {type: 'button', class: 'btn ghost small', onclick: () => openReport({runs: [runKey(data.run_id)]})},
        'Open in Report ', h('span', {class: 'arrow'}, '›')),
      h('button', {type: 'button', class: 'btn ghost small', onclick: (e) => runAgainAsDesign(data.run_id, e.currentTarget)},
        'Run again as design'),
      h('button', {type: 'button', class: 'btn ghost small', onclick: () => askAboutRun(data.run_id)},
        h('span', {class: 'ai-mark', 'aria-hidden': 'true'}, '◆'), 'Ask the assistant')),
    h('div', {class: 'kpis'},
      kpi('Trials', String(finished), finished < planned ? `${planned - finished} not finished` : 'all finished', {unit: `/ ${planned}`}),
      kpi('Pass rate', scored ? pct(passed / scored) : DASH, `${passed} passed of ${scored} scored`, {cls: 'hot'}),
      kpi('Total cost', atLeast(money(totalCost)), unmeasured ? `lower bound: ${unmeasured} without usage`
        : run ? `${pct(run.budget_usd ? totalCost / run.budget_usd : 0)} of ${money(run.budget_usd)} budget` : ''),
      kpi('Cost / trial', measured ? atLeast(money(costMeasured / measured)) : DASH,
        measured < finished ? `mean of ${measured} measured` : 'like-for-like priced'),
      kpi('Tokens / trial', measured ? atLeast(tokens(tokensTotal / measured)) : DASH,
        measured < finished ? `mean of ${measured} measured` : 'input + output'),
      kpi('Errors', String(errors), errors ? 'harness or infrastructure' : 'none', {cls: errors ? 'alert' : ''})),
    h('div', {class: 'note-strip'}, tag('note', 'warn'), "Ten tasks can't separate pass rates; compare the token and cost multiples."),
    unmeasured > 0 && h('div', {class: 'note-strip'}, tag('usage', 'warn'),
      `${plural(unmeasured, 'trial')} ran without recording usage, so ${unmeasured === 1 ? 'its' : 'their'} spend is not counted `
      + 'and the costs here are lower bounds.'),
    h('div', {class: 'two-up'},
      panel({code: 'A-01', title: 'Tokens vs baseline', meta: `${data.baseline} ÷ arm`}, multiplesChart(data, 'tokens')),
      panel({code: 'A-02', title: 'Cost vs baseline', meta: `${data.baseline} ÷ arm`}, multiplesChart(data, 'cost'))),
    h('div', {class: 'stack'},
      panel({code: 'A-03', title: 'Task matrix', meta: h('div', {class: 'legend'},
        h('span', null, cell('pass'), 'pass'), h('span', null, cell('fail'), 'fail'), h('span', null, cell('error'), 'error'),
        h('span', null, cell('unscored'), 'unscored'), h('span', null, cell(''), 'not run')), flush: true},
      h('div', {class: 'scroll'}, taskMatrix(run, data))),
      h('div', {class: 'two-up', style: 'margin:0'},
        panel({code: 'A-04', title: 'Token composition', meta: 'mean per trial'}, tokenComposition(armsWithData)),
        panel({code: 'A-05', title: 'Cost per trial', meta: 'each dot is one trial'}, costStrip(armsWithData, data.trials))),
      panel({code: 'A-06', title: 'Arm summary', flush: true}, h('div', {class: 'scroll'}, armTable(data))),
      panel({code: 'A-07', title: 'Trials', meta: h('span', {id: 'a-trials-meta'}), flush: true},
        trialFilters(data), h('div', {class: 'table-scroll', id: 'a-trials'}))));
  renderTrialsTable(data);
}

/** POST /api/designs/from-run/{id}: a new draft design repeating this run's spec, opened in DESIGNS. */
async function runAgainAsDesign(runId, button) {
  await busy(button, async () => {
    const detail = await api('POST', `/api/designs/from-run/${enc(runId)}`, {});
    toast(`Saved as ${detail.title} v${detail.version}, a draft. Nothing is queued yet.`);
    refreshDesigns();
    openDesign(detail.design_id, detail.version);
  });
}

/** Opens the assistant with a question about this run in the composer, not sent. */
function askAboutRun(runId) {
  openAssistant({prompt: `Look at run ${runId} (${runKey(runId)}). How do its arms compare with the baseline on tokens and cost, `
    + 'what should I make of its errors and left-out trials, and what would be worth running next?'});
}

function multiplesChart(data, kind) {
  const rows = data.comparison.map((c) => ({
    arm: c.arm, value: c[`${kind}_multiple`], pub: c[`published_${kind}`], verdict: c[`${kind}_verdict`], note: c.note,
    why: c.verdict_note || null,  // why there is no verdict, e.g. a non-Claude model
    left: [c.excluded > 0 && `${c.excluded} of ${c.arm}'s trials left out of the means`,
      c.baseline_excluded > 0 && `${c.baseline_excluded} of ${data.baseline}'s trials left out of the means`].filter(Boolean),
  }));
  const whys = [...new Set(rows.map((r) => r.why).filter(Boolean))];
  if (!rows.length) {
    const base = data.arms.find((a) => a.arm === data.baseline);
    return h('p', {class: 'muted', style: 'margin:0'}, base && base.trials
      ? 'No other arm has finished trials yet.'
      : `No comparison yet: the baseline arm (${data.baseline}) has no finished trials.`);
  }
  const band = (p) => (!p ? null : isNum(p.value) ? [p.value / 1.5, p.value * 1.5, p.value] : [p.low, p.high, null]);
  const peak = Math.max(2, ...rows.map((r) => (isNum(r.value) ? r.value : 0)), ...rows.map((r) => (band(r.pub) || [0, 0])[1]));
  const max = niceCeil(peak * 1.08);
  const x = (v) => `${Math.max(0, Math.min(100, (v / max) * 100)).toFixed(2)}%`;
  const step = max <= 3 ? 0.5 : max <= 6 ? 1 : max <= 12 ? 2 : 5;
  const ticks = [];
  for (let v = 0; v <= max + 1e-9; v += step) ticks.push(v);
  return h('div', {class: 'mchart'},
    rows.map((r) => {
      const b = band(r.pub);
      const verdictCls = r.verdict === 'in line' ? 'ok' : r.verdict ? 'warn' : '';
      const published = !r.pub ? 'none published' : isNum(r.pub.value) ? `${times(r.pub.value)} published` : `${multiple(r.pub.low)}–${times(r.pub.high)} published`;
      return h('div', {class: 'mrow'},
        h('div', {class: 'who', title: r.note || null}, r.arm),
        tip(h('div', {class: 'mtrack'},
          b && h('div', {class: 'band', style: `left:${x(b[0])};width:calc(${x(b[1])} - ${x(b[0])})`}),
          h('div', {class: 'parity', style: `left:${x(1)}`}),
          isNum(r.value) && h('div', {class: 'mbar', style: `width:${x(r.value)}`}),
          b && isNum(b[2]) && h('div', {class: 'pub', style: `left:${x(b[2])}`})),
        r.arm, [[times(r.value), 'measured'], [published.replace(' published', ''), 'published'],
          [r.verdict || (r.why ? 'none' : DASH), 'verdict'], r.why, r.pub && r.pub.source ? r.pub.source : null, r.note, ...r.left].filter(Boolean), true),
        h('div', {class: 'verdict', style: 'text-align:right'},
          h('div', {class: 'mono', style: 'font-size:13px;font-weight:600'}, times(r.value)),
          r.verdict ? tag(r.verdict, verdictCls)
            : r.why ? tip(tag('– no verdict'), 'No verdict', [r.why], true)
              : h('span', {class: 'mono faint', style: 'font-size:10.5px'}, 'no ref')));
    }),
    h('div', {class: 'maxis'}, ticks.map((v) => h('span', {style: `left:${x(v)}`}, v === 1 ? '1× parity' : `${v}×`))),
    whys.map((why) => h('p', {class: 'verdict-note'}, why)),
    h('div', {class: 'legend chart-foot'},
      h('span', null, h('i', {class: 'sw', style: 'background:#4cbb17;height:8px'}), 'measured'),
      h('span', null, h('i', {class: 'sw', style: 'background:repeating-linear-gradient(-45deg,rgba(182,194,185,.35) 0 3px,transparent 3px 6px);border:1px solid #3b4b3f'}), 'published range'),
      h('span', null, h('i', {class: 'sw line', style: 'background:#e3eae4;width:2px;height:11px'}), 'published value'),
      h('span', null, h('i', {class: 'sw line', style: 'background:#f0b03a;width:1px;height:11px'}), 'parity')));
}

function taskMatrix(run, data) {
  const arms = data.arms.map((a) => a.arm);
  const tasks = run && run.tasks && run.tasks.length ? run.tasks : [...new Set(data.trials.map((t) => t.task))];
  const grid = {};
  for (const t of data.trials) ((grid[t.task] = grid[t.task] || {})[t.arm] = grid[t.task][t.arm] || []).push(t);
  const attempts = (run && run.plan && run.plan.n_attempts) || Math.max(1, ...data.trials.map((t) => (grid[t.task][t.arm] || []).length));
  if (!tasks.length) return h('p', {class: 'muted', style: 'padding:14px;margin:0'}, 'No tasks recorded for this run.');
  return h('table', {class: 'matrix'},
    h('thead', null, h('tr', null, h('th', null, 'Task'), arms.map((a) => h('th', null, a === data.baseline ? `${a} ◆` : a)))),
    h('tbody', null, tasks.map((task) => h('tr', null,
      h('td', {class: 'task'}, task),
      arms.map((arm) => {
        const ts = (grid[task] && grid[task][arm]) || [];
        const slots = [];
        for (let i = 0; i < attempts; i++) {
          const t = ts[i];
          slots.push(t
            ? tip(cell(outcomeOf(t)), `${arm} · ${task}`, [
              [OUTCOME[outcomeOf(t)].label, 'outcome'], [isNum(t.reward) ? String(Math.round(t.reward * 100) / 100) : DASH, 'reward'],
              [usageValue(trialCost(t), t.usage_status), 'cost'], [`${minutes(t.minutes)} min`, 'duration'],
              [usageValue((t.n_input || 0) + (t.n_output || 0), t.usage_status, tokens), 'tokens'],
              t.error ? `${t.error}: ${(t.error_message || '').slice(0, 120)}` : null,
            ].filter(Boolean), true)
            : cell(''));
        }
        return h('td', null, h('div', {class: 'cells'}, slots));
      })))),
    h('tfoot', null, h('tr', null, h('td', {class: 'task'}, 'PASSED'),
      data.arms.map((a) => h('td', null, a.trials ? `${a.passed}/${a.scored}` : DASH)))));
}

function tokenComposition(arms) {
  if (!arms.length) return h('p', {class: 'muted', style: 'margin:0'}, 'No finished trials yet.');
  const rows = arms.map((a) => ({
    arm: a.arm, cache: a.mean_cache || 0, fresh: Math.max(0, (a.mean_input || 0) - (a.mean_cache || 0)), out: a.mean_output || 0, total: a.mean_tokens || 0,
  }));
  const max = Math.max(...rows.map((r) => r.total), 1);
  const seg = (v, color, label, r) => v > 0 && tip(h('i', {style: `flex:${v} 1 0;background:${color}`}), `${r.arm} · ${label}`,
    [[tokens(v), 'tokens per trial'], [pct(r.total ? v / r.total : 0), 'of the arm\'s tokens']]);
  return [
    h('div', {class: 'legend', style: 'margin-bottom:12px'},
      h('span', null, h('i', {class: 'sw', style: 'background:var(--series-cache)'}), 'cache read'),
      h('span', null, h('i', {class: 'sw', style: 'background:var(--series-fresh)'}), 'fresh input'),
      h('span', null, h('i', {class: 'sw', style: 'background:var(--series-output)'}), 'output')),
    rows.map((r) => h('div', {class: 'hbar-row'},
      h('div', {class: 'who'}, r.arm),
      h('div', null, h('div', {class: 'hbar', style: `width:${((r.total / max) * 100).toFixed(1)}%`},
        seg(r.cache, 'var(--series-cache)', 'cache read', r),
        seg(r.fresh, 'var(--series-fresh)', 'fresh input', r),
        seg(r.out, 'var(--series-output)', 'output', r))),
      h('div', {class: 'tot'}, tokens(r.total)))),
  ];
}

function costStrip(arms, allTrials) {
  if (!arms.length) return h('p', {class: 'muted', style: 'margin:0'}, 'No finished trials yet.');
  const trials = allTrials.filter((t) => t.usage_status !== 'unmeasured');  // their cost is unknown, not $0
  const byArm = {};
  for (const t of trials) (byArm[t.arm] = byArm[t.arm] || []).push(t);
  const costs = trials.map(trialCost).filter(isNum);
  const max = niceCeil(Math.max(...costs, ...arms.map((a) => a.mean_cost || 0), 0.01) * 1.05);
  const x = (v) => `${Math.max(0, Math.min(100, (v / max) * 100)).toFixed(2)}%`;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * max);
  return [
    h('div', {class: 'legend', style: 'margin-bottom:12px'},
      h('span', null, h('i', {class: 'sw d-pass', style: 'border-radius:50%'}), 'pass'),
      h('span', null, h('i', {class: 'sw d-fail', style: 'border-radius:50%'}), 'fail'),
      h('span', null, h('i', {class: 'sw d-error', style: 'border-radius:50%'}), 'error'),
      h('span', null, h('i', {class: 'sw line', style: 'background:var(--text);width:2px;height:11px'}), 'mean')),
    arms.map((a) => h('div', {class: 'strip-row'},
      h('div', {class: 'who'}, a.arm),
      h('div', {class: 'strip'},
        (byArm[a.arm] || []).filter((t) => isNum(trialCost(t))).map((t) => {
          const o = outcomeOf(t);
          return tip(h('span', {class: 'dotm', style: `left:${x(trialCost(t))}`}, h('i', {class: `d-${o}`})),
            `${a.arm} · ${t.task}`, [[usageValue(trialCost(t), t.usage_status), 'cost'], [OUTCOME[o].label, 'outcome'],
              [`${minutes(t.minutes)} min`, 'duration'], t.usage_status === 'lower_bound' ? 'Timed out: completed calls only' : null].filter(Boolean), true);
        }),
        isNum(a.mean_cost) && h('span', {class: 'mean', style: `left:${x(a.mean_cost)}`})),
      h('div', {class: 'stat'}, `mean ${money(a.mean_cost)}`, h('span', {class: 'sub'}, `per pass ${money(a.cost_per_success)}`)))),
    h('div', {class: 'saxis'}, ticks.map((v) => h('span', {style: `left:${x(v)}`}, money(v)))),
  ];
}

function armTable(data) {
  return h('table', {class: 'grid-t compact'},
    thead('Arm', thNum('Trials'), thNum('Passed'), thNum('Errors'), thNum('Tokens'),
      thNum('In / cache / out'), thNum('Cached'), thNum('Cost'), thNum('Per pass'),
      thNum('Other model'), thNum('Min')),
    h('tbody', null, data.arms.map((a) => {
      const has = a.trials > 0;
      const isBase = a.arm === data.baseline;
      const unpriced = has && a.unpriced_models && a.unpriced_models.length > 0;
      const harness = a.family || a.agent;
      // Addendum: excluded = setup failures + trials that ran without usage (unmeasured); lower_bound = measured timeouts.
      const setupLeft = (a.excluded || 0) - (a.unmeasured || 0);
      const floor = a.lower_bound > 0 ? '≥ ' : '';
      return h('tr', {class: isBase ? 'baseline' : null},
        h('td', null, h('b', null, a.arm), isBase && [' ', tag('baseline', 'ok')],
          unpriced && [' ', h('span', {class: 'tag warn', title: 'Not in config/pricing.yaml, so the harness-reported cost is used where there is one'},
            `unpriced: ${a.unpriced_models.join(', ')}`)],
          a.cost_partial && [' ', tag('cost partial', 'warn')],
          h('span', {class: 'sub mono'}, harness === a.arm ? a.model : `${harness} · ${a.model}`),
          setupLeft > 0 && h('span', {class: 'sub arm-left'}, `${plural(setupLeft, 'setup failure')} left out`),
          a.unmeasured > 0 && h('span', {class: 'sub arm-left warn-text'}, `${a.unmeasured} without usage (spend not counted)`),
          a.lower_bound > 0 && h('span', {class: 'sub arm-left'}, `${a.lower_bound} timed out: usage is a lower bound`)),
        tdNum(a.trials, isNum(a.measured) ? `${a.measured} measured` : null),
        tdNum(has ? `${a.passed} of ${a.scored}` : DASH),
        tdNum(has ? a.errors : DASH),
        tdNum(isNum(a.mean_tokens) ? `${floor}${tokens(a.mean_tokens)}` : DASH, has && isNum(a.mean_tokens) ? count(a.mean_tokens) : null),
        tdNum(has && isNum(a.mean_input) ? `${tokens(a.mean_input)} / ${tokens(a.mean_cache)} / ${tokens(a.mean_output)}` : DASH),
        tdNum(pct(a.cache_share)),
        tdNum(isNum(a.mean_cost) ? `${floor}${money(a.mean_cost)}` : DASH),
        tdNum(money(a.cost_per_success)),
        tdNum(has ? pct(a.other_model_share) : DASH, has && a.models ? a.models.join(', ') : null),
        tdNum(minutes(a.mean_minutes)));
    })));
}

function trialFilters(data) {
  const f = ui.trialFilter;
  if (f.arm && !data.arms.some((a) => a.arm === f.arm)) f.arm = '';
  const rerender = () => {
    renderTrialsTable(data);
    for (const b of document.querySelectorAll('#a-outcome-seg button')) b.classList.toggle('active', b.dataset.value === f.outcome);
  };
  return h('div', {class: 'filters'},
    eyebrow('Filter'),
    select('arm', [['', 'All arms'], ...data.arms.map((a) => [a.arm, a.arm])], f.arm, {
      'aria-label': 'Arm', onchange: (e) => { f.arm = e.target.value; rerender(); },
    }),
    h('div', {class: 'seg', id: 'a-outcome-seg', role: 'group', 'aria-label': 'Outcome'},
      [['', 'ALL'], ['pass', 'PASS'], ['fail', 'FAIL'], ['error', 'ERROR'], ['unscored', 'UNSCORED']].map(([value, label]) => h('button', {
        type: 'button', 'data-value': value, class: f.outcome === value ? 'active' : null,
        onclick: () => { f.outcome = value; rerender(); },
      }, label))));
}

function renderTrialsTable(data) {
  const box = $('#a-trials');
  if (!box) return;
  const f = ui.trialFilter;
  const rows = data.trials.filter((t) => (!f.arm || t.arm === f.arm) && (!f.outcome || outcomeOf(t) === f.outcome));
  $('#a-trials-meta').textContent = `${rows.length} OF ${data.trials.length}`;
  if (!rows.length) {
    fill(box, h('p', {class: 'muted', style: 'padding:14px;margin:0'}, data.trials.length ? 'No trials match the filter.' : 'No finished trials yet.'));
    return;
  }
  const shortModel = (m) => m.split('/').pop().replace(/^claude-/, '');
  fill(box, h('table', {class: 'grid-t compact'},
    thead('Arm', 'Task', 'Outcome', thNum('Reward'), thNum('Input'), thNum('Cache'), thNum('Output'),
      thNum('Cost'), thNum('Reported'), thNum('Min'), 'Models'),
    h('tbody', null, rows.map((t) => {
      const o = outcomeOf(t);
      const status = t.usage_status;
      // Addendum: no usage recorded means the tokens and cost are unknown, never 0; a timeout's usage is a floor.
      const usage = status === 'unmeasured'
        ? h('td', {class: 'num muted', colspan: 3, title: 'The agent ran, but no usage was recorded'}, 'not recorded')
        : [tdNum(usageValue(t.n_input, status, count)), tdNum(usageValue(t.n_cache, status, count)), tdNum(usageValue(t.n_output, status, count))];
      return h('tr', null,
        h('td', null, h('b', null, t.arm)),
        h('td', {class: 'mono', title: t.trial || null}, t.task || DASH),
        h('td', null, h('span', {class: OUTCOME[o].text}, `${OUTCOME[o].glyph} ${OUTCOME[o].label}`),
          t.error && h('span', {class: 'sub clip', title: t.error_message || null}, `${t.error}${t.error_message ? `: ${t.error_message}` : ''}`),
          status === 'unmeasured' && h('span', {class: 'sub warn-text'}, 'no usage recorded'),
          status === 'lower_bound' && h('span', {class: 'sub'}, '≥ lower bound: completed calls only')),
        tdNum(isNum(t.reward) ? String(Math.round(t.reward * 100) / 100) : DASH),
        usage,
        tdNum(usageValue(t.computed_cost, status), t.cost_partial ? 'Some models are unpriced: this is part of the cost' : null),
        tdNum(money(t.reported_cost)), tdNum(minutes(t.minutes)),
        h('td', {class: 'mono', style: 'font-size:11.5px', title: t.models && t.models.length ? t.models.join(', ') : null},
          t.models && t.models.length ? t.models.map(shortModel).join(' + ') : DASH));
    }))));
}

// Markdown ---------------------------------------------------------------

/** A small Markdown subset built as DOM nodes, never as HTML: headings, lists, code, tables, bold. */
function renderMarkdown(src) {
  // Every line break a model may write (CR, CRLF, U+2028, U+2029) splits lines; '.' in the patterns below stops at them.
  const lines = String(src).replace(/\r\n?|[\u2028\u2029]/g, '\n').split('\n');
  const listItem = /^\s*([-*•]|\d+[.)])\s+/;
  const tableRow = /^\s*\|.*\|\s*$/;
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      const code = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
      i++;
      out.push(h('pre', {class: 'md-code'}, code.join('\n')));
    } else if (!line.trim()) {
      i++;
    } else if (/^#{1,4}\s+/.test(line)) {
      out.push(h('div', {class: 'md-h'}, inlineMarkdown(line.replace(/^#{1,4}\s+/, ''))));
      i++;
    } else if (tableRow.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      const rows = [];
      while (i < lines.length && tableRow.test(lines[i])) rows.push(lines[i++]);
      const cells = (row) => row.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
      out.push(h('div', {class: 'md-table'}, h('table', {class: 'grid-t compact'},
        h('thead', null, h('tr', null, cells(rows[0]).map((c) => h('th', null, inlineMarkdown(c))))),
        h('tbody', null, rows.slice(2).map((row) => h('tr', null, cells(row).map((c) => h('td', null, inlineMarkdown(c)))))))));
    } else if (listItem.test(line)) {
      const [list, next] = markdownList(lines, i);
      out.push(list);
      i = next;
    } else {
      const para = [line];
      i++;
      while (i < lines.length && lines[i].trim() && !/^\s*```|^#{1,4}\s/.test(lines[i]) && !listItem.test(lines[i]) && !tableRow.test(lines[i])) {
        para.push(lines[i++]);
      }
      out.push(h('p', null, inlineMarkdown(para.join('\n'))));
    }
  }
  return out;
}

/**
 * A list starting at lines[start]. Deeper-indented items nest under the previous item,
 * a blank line between items of the same kind keeps the list going, and an ordered
 * list keeps its first number. Returns the list element and the index after it.
 */
function markdownList(lines, start) {
  const item = /^(\s*)([-*•]|\d+[.)])\s+(.*)$/;
  const indentOf = (m) => m[1].replace(/\t/g, '    ').length;
  const first = lines[start].match(item);
  if (!first) return [h('p', null, inlineMarkdown(lines[start])), start + 1];  // never throw on odd model text
  const indent = indentOf(first);
  const ordered = /\d/.test(first[2]);
  const items = [];
  let i = start;
  while (i < lines.length) {
    const m = lines[i].match(item);
    if (!m) {
      if (!lines[i].trim()) {
        // A blank line continues the list only if the next item is the same kind at the same depth.
        let j = i + 1;
        while (j < lines.length && !lines[j].trim()) j++;
        const n = j < lines.length && lines[j].match(item);
        if (n && indentOf(n) >= indent && /\d/.test(n[2]) === ordered) {
          i = j;
          continue;
        }
      } else if (/^\s{2,}\S/.test(lines[i]) && items.length) {
        items[items.length - 1].text += `\n${lines[i].trim()}`;  // wrapped continuation of the item
        i++;
        continue;
      }
      break;
    }
    const depth = indentOf(m);
    if (depth < indent) break;
    if (depth > indent) {
      if (!items.length) items.push({text: '', children: []});
      const [child, next] = markdownList(lines, i);
      items[items.length - 1].children.push(child);
      i = next;
      continue;
    }
    if (/\d/.test(m[2]) !== ordered) break;
    items.push({text: m[3], children: [], number: ordered ? parseInt(m[2], 10) : null});
    i++;
  }
  const startAt = ordered && items[0] && items[0].number !== 1 ? items[0].number : null;
  const list = h(ordered ? 'ol' : 'ul', {start: startAt},
    items.map((it) => h('li', null, inlineMarkdown(it.text), it.children)));
  return [list, i];
}

function inlineMarkdown(text) {
  const parts = [];
  const pattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*)/g;
  let last = 0;
  let match;
  while ((match = pattern.exec(text))) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    const token = match[0];
    parts.push(token[0] === '`' ? h('code', null, token.slice(1, -1)) : h('b', null, token.slice(2, -2)));
    last = match.index + token.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

// Start ------------------------------------------------------------------

registerView('live', {render: (state, force) => renderLive(force)});
registerView('launch', {render: (state, force) => renderLaunch(force)});
registerView('analysis', {render: (state, force) => renderAnalysis(force)});

/** Area scripts loaded only in demo mode, after demo.js, each registering its fake API on window.benchDemo. */
const DEMO_SCRIPTS = ['demo.js', 'demo-report.js', 'demo-designs.js', 'demo-queue.js', 'demo-assistant.js'];

function init() {
  for (const button of document.querySelectorAll('.views button')) {
    button.addEventListener('click', () => showView(button.dataset.view));
  }
  // The tab list's keys: Left and Right move to the previous or next tab and open it, Home and End the first and last.
  $('.views').addEventListener('keydown', (e) => {
    const tabs = [...document.querySelectorAll('.views button')];
    const at = tabs.indexOf(document.activeElement);
    if (at < 0) return;
    const to = {ArrowLeft: at - 1, ArrowRight: at + 1, Home: 0, End: tabs.length - 1}[e.key];
    if (to === undefined) return;
    e.preventDefault();
    const tab = tabs[(to + tabs.length) % tabs.length];
    tab.focus();
    showView(tab.dataset.view);
  });
  $('#open-settings').addEventListener('click', () => openDrawer('settings'));
  $('#drawer-close').addEventListener('click', closeDrawer);
  $('#scrim').addEventListener('click', closeDrawer);
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || document.querySelector('dialog[open]')) return;
    if (callArea('assistant', 'escape') === true) return;
    if (ui.drawer) closeDrawer();
  });
  setupKeyDialog();
  setupTooltip();
  for (const name of areas.keys()) callArea(name, 'setup');
  // Shared suggestions for free-text effort fields.
  document.body.append(h('datalist', {id: 'effort-any'}, ['inherit', 'omit', ...EFFORT_LEVELS].map((v) => h('option', {value: v}))));
  if (DEMO) {
    $('#demo-tag').hidden = false;
    $('#demo-banner').hidden = false;
  }
  window.addEventListener('hashchange', () => {
    const view = location.hash.slice(1);
    if (VIEWS.includes(view) && view !== ui.view) showView(view);
  });
  let resizeTimer = 0;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      fitSystems();
      rendered.delete('m-tele');
      if (ui.view === 'live' && activeRun()) renderLive(false);
    }, 150);
  });
  window.addEventListener('beforeunload', (e) => {
    if (ui.dirty.size || holdReason()) {
      e.preventDefault();
      e.returnValue = '';
    }
  });
  setInterval(tick, 1000);
  tick();
  poll();
}

function loadScript(name) {
  return new Promise((resolve) => {
    const script = document.createElement('script');
    script.src = `/static/${name}`;
    script.onload = resolve;
    script.onerror = resolve;  // a missing demo file leaves that area on the real server
    document.head.append(script);
  });
}

/** Runs once every <script> in the page has registered its area. In demo mode the fakes load first. */
async function boot() {
  // Simulated runs and fake area data for previewing without spending anything, loaded in order.
  if (DEMO) for (const name of DEMO_SCRIPTS) await loadScript(name);
  init();
  // Fail closed: without demo.js the page talks to the real bench. api() then refuses every billed path, and the
  // banner says so instead of promising simulated data.
  if (DEMO && !demoGuarded()) {
    const banner = $('#demo-banner');
    banner.className = 'banner bad';
    banner.textContent = 'Demo mode did not load (demo.js failed), so this page shows the real bench. '
      + 'Everything that bills is refused here. Reload, or open the dashboard without ?demo.';
  }
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot, {once: true});
else boot();
