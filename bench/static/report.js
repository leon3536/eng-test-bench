/* REPORT tab: pooled statistics across every run (<section id="view-report">).

   Owned by the REPORT builder, with report.css and demo-report.js. Brief:
   "REPORT tab", the Ledger and Report endpoints, and the addendum on trials
   whose usage was not recorded.

   The tab reads the ledger, which holds every trial of every run and profile.
   A status strip sits above five sub-tabs, and the first two share a filter
   rail:
   - Report: multiples vs the baseline per model (harness ÷ baseline, so
     under 1× is less than the baseline: forest plots, a pass strip and a
     table), the unpaired group summary and the coverage matrix. A
     comparison row opens the pairs drawer, where trials and runs can be
     excluded or included again.
   - Ledger: the paged trial table. A row opens the trial drawer.
   - Runs, Exclusions and Snapshots.

   The rail and every request carry the task scope, the profile's task file
   (config tasks), unless tasks are ticked or the person switches to all
   tasks; the strip says which applies and switches it in one click.

   Rail edits apply 400 ms after the last one. Until then hold() keeps a run
   the queue starts from pulling the page to LIVE. The rail is rebuilt only
   while it holds nothing unsent. Every request here is a read (the report
   routes are POST only to carry JSON) except exclusions, snapshots and
   rescan, and none is billed. */

(() => {
  'use strict';

  // Ledger outcomes in the ledger's order. The glyph makes sure colour is never the only cue.
  const OUTCOMES = {
    passed: {glyph: '●', label: 'passed'},
    failed: {glyph: '○', label: 'failed'},
    agent_limit: {glyph: '▲', label: 'hit its own limit'},
    agent_crash: {glyph: '×', label: 'harness crashed'},
    misconfigured_cap: {glyph: '$', label: 'stopped by a mispriced cap'},
    verifier_failure: {glyph: '?', label: 'verifier failed'},
    provider_failure: {glyph: '≈', label: 'provider error'},
    setup_failure: {glyph: '◦', label: 'never got going'},
    infra: {glyph: '!', label: 'infrastructure'},
    incomplete: {glyph: '…', label: 'killed before finishing'},
  };
  const OUTCOME_ORDER = Object.keys(OUTCOMES);
  const CORNER = ['agent_crash', 'setup_failure', 'infra', 'incomplete'];  // marked in the corner of a coverage cell
  // Reference verdicts: [glyph, chip text, tag tone, what it means]. Never "confirmed". Multiples are
  // harness ÷ baseline, so below the published range means relatively cheaper than published.
  const VERDICTS = {
    within: ['✓', 'within published range', 'ok', 'The whole interval lies inside the published range.'],
    overlaps: ['≈', 'overlaps', '', 'The interval reaches both inside and outside the published range.'],
    above: ['▲', 'above published range', 'warn',
      'This harness used more, relative to the baseline, than the published figure says.'],
    below: ['▼', 'below published range', 'warn',
      'This harness used less, relative to the baseline, than the published figure says.'],
    insufficient: ['…', 'too few pairs', '', 'Too few paired tasks for a verdict.'],
    not_applicable: ['–', 'n/a for this model', '', 'The published figures were measured on Claude models.'],
  };
  const POSITION = {inside: 'estimate inside the band', above: 'estimate above the band', below: 'estimate below the band'};
  const USAGE = {lower_bound: ['≥ lower bound', 'warn'], unmeasured: ['usage not recorded', 'bad']};
  const TABS = [['report', 'Report'], ['ledger', 'Ledger'], ['runs', 'Runs'], ['exclusions', 'Exclusions'], ['snapshots', 'Snapshots']];
  const RAIL_TABS = new Set(['report', 'ledger']);
  const LIST_KEYS = ['profiles', 'sources', 'models', 'families', 'implementations', 'configs', 'exclude_configs',
    'harness_versions', 'harbor_versions', 'efforts', 'datasets', 'snapshots', 'tasks', 'runs', 'exclude_runs',
    'setup_modes'];
  const NULLABLE = new Set(['efforts', 'setup_modes']);  // null is a value: harness default, setup not recorded
  // How a trial got its harness (the ledger's setup_mode): [label, tag tone, what it means].
  const SETUP = {
    prebaked: ['pre-baked', 'ok', 'started from a pre-baked image: its task image with the harness already installed'],
    stock: ['stock', '', 'Harbor installed the harness inside the trial'],
    fallback: ['fallback', 'warn',
      'meant to use a pre-baked image, but none was usable, so Harbor installed the harness'],
  };
  const setupLabel = (mode) => (SETUP[mode] ? SETUP[mode][0] : 'not recorded');
  // Code defaults for the Advanced controls until the server's echo (spec.options) arrives.
  const OPTION_DEFAULTS = {
    group_by: 'family', split_harness_versions: false, 'baseline.family': 'claude-code', 'baseline.config': null,
    'pairing.match_on': 'digest', 'pairing.same_run_only': false, 'policy.crashes': 'include',
    'policy.misconfigured_cap': 'exclude', 'policy.exclusions': true, pricing: 'current', ci_level: 0.9, setup: 'any',
    compare: 'harnesses', 'baseline.model': null,
  };
  // options.compare: harnesses (each model held fixed) or models (each harness held fixed).
  const COMPARE_CHOICES = [['harnesses', 'Harnesses'], ['models', 'Models']];
  // How each way names its two sides, its multiple and what enough pairs earn.
  const SIDES = {
    harnesses: {arm: 'harness', Arm: 'Harness', base: 'baseline', ratio: 'harness ÷ baseline',
      diff: 'harness − baseline', goal: 'a verdict'},
    models: {arm: 'model', Arm: 'Model', base: 'baseline model', ratio: 'model ÷ baseline model',
      diff: 'model − baseline model', goal: 'a settled figure'},
  };
  const TICKS = [0.125, 0.25, 0.5, 1, 2, 4];   // the forest plots' log axis (harness ÷ baseline)
  const tickLabel = (v) => (v < 1 ? `1/${Math.round(1 / v)}×` : `${v}×`);   // 1/8× is shorter than 0.125×
  const LOG_LO = Math.log(0.125);
  const LOG_SPAN = Math.log(4) - LOG_LO;
  const PASS_TICKS = [-50, -25, 0, 25, 50];
  const PAGE = 100;
  const SORTABLE = {started_at: 'DESC', task: 'ASC', family: 'ASC', tokens_total: 'DESC', cost_usd: 'DESC'};  // first direction on click

  const rp = {
    built: false,
    tab: 'report',
    filters: {},            // the rail: what the next request sends
    options: {},            // options the person set; the rest come from the server's echo (spec.options)
    allTasks: false,        // true: no task scope; false: the profile's task file, unless tasks are ticked
    scopeFor: undefined,    // the scope the last render saw, to reload when the task file changes
    handed: null,           // filters from openReport that arrived before the first poll (prefs are per profile)
    railOpen: false,        // the rail below 1100 px, where it collapses above the content
    railStale: false,       // a rebuild waits until the rail holds nothing unsent
    railFor: '',
    secOpen: new Set(['Models', 'Harnesses', 'Advanced']),
    invalid: '',            // a rail value typed but not valid, e.g. CI level 1.5
    facets: null,
    status: null,
    metaError: null,
    report: null,
    coverage: null,
    reportFor: '',
    reportError: null,
    stale: true,            // the ledger changed since the report loaded
    busy: '',               // a 503 for the banner
    rescanning: false,
    ingestAt: undefined,
    ledger: {offset: 0, order: 'started_at DESC', outcomes: [], data: null, for: '', error: null},
    runs: null,
    runsError: null,
    exclusions: null,
    exclError: null,
    snapshots: null,
    snapError: null,
    snapshot: null,         // the opened snapshot (read-only)
    compare: null,          // {id, now: Report} for "Compare to now"
    side: null,             // what the drawer shows: {kind: 'pairs', key} or {kind: 'trial', id, back}
    sideReturn: null,
    showPending: false,
    dialog: null,           // {input, initial} while an exclusion or snapshot dialog is open
  };

  const fresh = {report: latest(), meta: latest(), ledger: latest(), runs: latest(), excl: latest(), snaps: latest(), side: latest(), snap: latest()};
  const apply = debounce(() => reload(), 400);
  const ingestSoon = debounce(() => afterIngest(), 800);
  const ingestLater = debounce(() => afterIngest(), 10000);

  // Small helpers ---------------------------------------------------------

  const shortFp = (fp) => (typeof fp === 'string' && fp ? fp.replace(/^sha256:/, '').slice(0, 8) : DASH);
  const trialName = (key) => String(key || '').split('/').pop() || DASH;
  const unique = (list) => [...new Set(list)];
  const vkey = (v) => (v === null ? '\u0000null' : String(v));
  const glyphOf = (o) => (OUTCOMES[o] || {glyph: '·'}).glyph;
  const labelOf = (o) => (OUTCOMES[o] || {label: statusLabel(o)}).label;
  const isMine = (row) => Boolean(ui.state && row && row.profile === ui.state.config.profile);
  const is503 = (err) => Boolean(err && err.status === 503);
  const at = (p) => `${p.toFixed(2)}%`;
  const xLog = (v) => Math.max(0, Math.min(100, ((Math.log(v) - LOG_LO) / LOG_SPAN) * 100));
  const xPass = (v) => Math.max(0, Math.min(100, v + 50));
  // Signed once rounded, so float noise (a converted snapshot against now) reads 0.00, not −0.00.
  const signed = (v, digits = 1) => {
    if (!isNum(v)) return DASH;
    const r = Number(v.toFixed(digits));
    return `${r > 0 ? '+' : r < 0 ? '−' : ''}${Math.abs(r).toFixed(digits)}`;
  };
  const pts = (v) => (isNum(v) ? `${signed(v)} pts` : DASH);
  const interval = (lo, hi, fmt = multiple) => (isNum(lo) && isNum(hi) ? `[${fmt(lo)}–${fmt(hi)}]` : '');
  const ppInterval = (lo, hi) => (isNum(lo) && isNum(hi) ? `[${signed(lo)}, ${signed(hi)}]` : '');
  const halfWidth = (ci) => (ci && isNum(ci.lo) && isNum(ci.hi) && ci.lo > 0 ? Math.sqrt(ci.hi / ci.lo) : null);
  const dayString = (t) => new Date(t).toISOString().slice(0, 10);
  const secs = (s) => (isNum(s) ? duration(s / 60) : DASH);
  const dig = (obj, path) => path.split('.').reduce((o, k) => (o && typeof o === 'object' ? o[k] : undefined), obj);
  const params = (rows) => h('dl', {class: 'params rp-params'},
    rows.filter(Boolean).map(([k, v]) => h('div', null, h('dt', null, k), h('dd', null, v === null || v === undefined || v === '' ? DASH : v))));

  /** Dims elements while a request runs ("dim, don't blank"). Overlapping loads are counted. */
  async function dimWhile(els, promise) {
    const list = els.filter(Boolean);
    for (const el of list) {
      el.dataset.loads = String((Number(el.dataset.loads) || 0) + 1);
      el.classList.add('rp-loading');
    }
    try {
      return await promise;
    } finally {
      for (const el of list) {
        const n = Math.max(0, (Number(el.dataset.loads) || 0) - 1);
        el.dataset.loads = String(n);
        if (!n) el.classList.remove('rp-loading');
      }
    }
  }

  /** A 503 means the ledger is busy or locked: the red banner says so ("try again"), never "disconnected". */
  function noteError(err) {
    if (is503(err)) {
      rp.busy = err.message;
      renderBanner();
    }
    return err;
  }

  function clearBusy() {
    if (!rp.busy) return;
    rp.busy = '';
    renderBanner();
  }

  function outcomeTag(outcome) {
    return h('span', {class: `rp-otag rp-o-${outcome}`}, h('b', {'aria-hidden': 'true'}, glyphOf(outcome)), labelOf(outcome));
  }

  function outcomeEntries(byOutcome) {
    const by = byOutcome || {};
    const known = OUTCOME_ORDER.filter((o) => by[o] > 0);
    const other = Object.keys(by).filter((o) => !OUTCOMES[o] && by[o] > 0);
    return [...known, ...other].map((o) => [o, by[o]]);
  }

  /** A stacked bar of outcomes with a glyph legend carrying every count as text. */
  function outcomeBar(byOutcome) {
    const entries = outcomeEntries(byOutcome);
    const total = sum(entries, (e) => e[1]);
    if (!total) return h('span', {class: 'faint'}, 'No trials');
    return h('div', {class: 'rp-obar-wrap'},
      h('div', {class: 'rp-obar', role: 'img', 'aria-label': entries.map(([o, n]) => `${n} ${labelOf(o)}`).join(', ')},
        entries.map(([o, n]) => tip(h('i', {class: `rp-o-${o}`, style: `flex:${n} 1 0`}), labelOf(o),
          [[count(n), 'trials'], [pct(n / total), 'of these']]))),
      h('div', {class: 'rp-olegend'}, entries.map(([o, n]) => h('span', {class: `rp-o-${o}`},
        h('b', {'aria-hidden': 'true'}, glyphOf(o)), `${count(n)} ${labelOf(o)}`))));
  }

  /** Glyph counts in one line, for tables. */
  function outcomeCounts(byOutcome) {
    const entries = outcomeEntries(byOutcome);
    if (!entries.length) return h('span', {class: 'faint'}, DASH);
    return h('span', {class: 'rp-ocounts'}, entries.map(([o, n]) => tip(h('span', {class: `rp-o-${o}`},
      h('b', {'aria-hidden': 'true'}, glyphOf(o)), count(n)), labelOf(o), [[count(n), 'trials']], true)));
  }

  function usageTag(status) {
    const u = USAGE[status];
    return u ? tag(u[0], u[1]) : null;
  }

  /** The struck-through "excluded" badge with its reason, or null. */
  const shortRef = (ref) => String(ref || '').replace(/^[^/]+\/[^/]+-r[0-9a-f]{10}\//, '');

  /** A trial's harness setup as a tag: pre-baked, stock or fallback, with the image and setup time in its tooltip. */
  function setupTag(setup, phases) {
    const s = setup || {};
    if (!s.mode) return h('span', {class: 'faint'}, DASH);
    const [label, tone, meaning] = SETUP[s.mode] || [s.mode, '', ''];
    const took = phases && isNum(phases.agent_setup_sec) ? `${phases.agent_setup_sec.toFixed(1)} s` : null;
    const rows = [meaning, took && [took, 'agent setup'], s.image && [shortRef(s.image), 'image'],
      s.baked_at && [when(s.baked_at), 'baked'], s.mode === 'fallback' && s.reason,
      isNum(s.limit_sec) && [`${Math.round(s.limit_sec)} s`, 'setup limit']];
    return tip(tag(label, tone), 'Harness setup', rows.filter(Boolean), true);
  }

  // Why a trial's cache writes are unknown (flags.cache_writes_unknown.reason), as a clause.
  const WRITES_UNKNOWN = {'no harness log': 'no harness log gave them',
    'not in the harness log': 'the harness log has none', 'trial folder gone': 'the trial folder is gone'};

  /**
   * A trial's cost checks (TrialRow.flags, raised at ingest from the harness's own log and the prices) as
   * [label, sentence] rows: spend the cost may leave out, writes priced as input, a harness figure that disagrees.
   */
  function flagRows(flags) {
    const f = flags || {};
    const upTo = (x, how = 'up to') => (x && isNum(x.cost_usd_max) ? `, ${how} ${money(x.cost_usd_max)}` : '');
    const calls = (n) => plural(isNum(n) ? n : 0, 'model call');
    const rows = [];
    if (f.declined_attempts) {
      rows.push(['Declined attempts', `${plural(f.declined_attempts.attempts || 0, 'attempt')} declined before another `
        + 'model answered (a refusal fallback). Whether they are billed is unknown, so the cost leaves them out'
        + `${upTo(f.declined_attempts, 'up to')} at today's prices.`]);
    }
    if (f.unsaved_calls) {
      const how = f.unsaved_calls.bound ? ` (${f.unsaved_calls.bound})` : '';
      rows.push(['Calls never saved', `${calls(f.unsaved_calls.calls)} completed but never saved (a timeout during a `
        + `command), missing from the cost${upTo(f.unsaved_calls)}${how}.`]);
    }
    if (f.unfinished_calls) {
      rows.push(['Calls cut off', `${calls(f.unfinished_calls.calls)} never finished (a kill mid-call), missing from `
        + `the cost${upTo(f.unfinished_calls, 'about')} at most (estimated from the costliest completed call).`]);
    }
    if (f.reported_cost_mismatch) {
      const m = f.reported_cost_mismatch;
      const off = isNum(m.diff_pct) ? `${signed(m.diff_pct)}%` : 'well';
      // Four decimals: the check fires at 0.5%, which cents would round away ($0.05 vs $0.05).
      const exact = (v) => (isNum(v) ? `$${v.toFixed(4)}` : DASH);
      rows.push(['Own cost differs', `The harness reports ${exact(m.reported_usd)}, ${off} from the `
        + `${exact(m.computed_usd)} computed from its tokens: a call the log does not show, or a price that differs.`]);
    }
    if (f.cache_writes_unknown) {
      const u = f.cache_writes_unknown;
      const which = (u.models || []).length ? ` (${u.models.join(', ')})` : '';
      rows.push(['Cache writes unknown', `${isNum(u.calls) ? `${calls(u.calls)}: ` : ''}`
        + `${WRITES_UNKNOWN[u.reason] || u.reason || 'not recorded'}${which}. Priced as fresh input, so the cost is `
        + 'low by the write premium.']);
    }
    if (f.cache_writes_exceed_prompt) {
      rows.push(['Writes over prompt', 'Recorded cache writes exceed the prompt on '
        + `${[].concat(f.cache_writes_exceed_prompt).join(', ')}.`]);
    }
    if (f.trial_folder_missing) {
      rows.push(['Folder gone', 'No trial folder when last re-derived; its cache writes and log checks are the '
        + 'figures stored before.']);
    }
    return rows;
  }

  /** The ledger table's mark beside a cost its checks question, with the checks in its tooltip. */
  function flagMark(flags) {
    const rows = flagRows(flags);
    if (!rows.length) return null;
    return tip(h('span', {class: 'rp-flag', 'aria-label': plural(rows.length, 'cost check')}, '⚠'), 'Cost checks',
      rows.map(([label, text]) => [text, label]), true);
  }

  /** The trial drawer's harness setup: which image it ran (or why it ran stock) and its bake. */
  function setupSection(t) {
    const s = t.setup || {};
    if (!s.mode && !t.prebake) return null;
    const sub = s.mode ? `${setupLabel(s.mode)}${SETUP[s.mode] ? `: ${SETUP[s.mode][2]}` : ''}` : 'not recorded';
    return sideSection('Harness setup', sub, params([
      ['Mode', setupLabel(s.mode)],
      ['Image', s.image ? h('span', {class: 'mono rp-impl'}, s.image) : DASH],
      ['Image id', s.image_id ? h('span', {class: 'mono'}, String(s.image_id).slice(0, 19)) : DASH],
      ['Recipe', s.recipe ? h('span', {class: 'mono'}, String(s.recipe).slice(0, 12)) : DASH],
      ['Baked', s.baked_at ? when(s.baked_at) : DASH],
      ['Setup limit', isNum(s.limit_sec) ? `${Math.round(s.limit_sec)} s` : DASH],
      s.reason && ['Why', s.reason],
    ].filter(Boolean)));
  }

  function exclusionBadge(ex) {
    if (!ex) return null;
    return ex.action === 'exclude'
      ? tip(h('span', {class: 'tag bad rp-struck'}, 'excluded'), 'Excluded by hand', [ex.reason, [when(ex.created_at), 'since']], true)
      : tip(h('span', {class: 'tag ok'}, 'included again'), 'Included again by hand', [ex.reason, [when(ex.created_at), 'since']], true);
  }

  function publishedText(ref) {
    if (!ref) return DASH;
    const p = ref.published || {};
    if (isNum(p.value)) return `${times(p.value)}`;
    if (isNum(p.low) && isNum(p.high)) return `${multiple(p.low)}–${times(p.high)}`;
    if (Array.isArray(ref.band)) return `${multiple(ref.band[0])}–${times(ref.band[1])}`;
    return DASH;
  }

  function passPublishedText(ref) {
    if (!ref || !Array.isArray(ref.band)) return DASH;
    const [lo, hi] = ref.band;
    return lo === -hi ? `±${Math.abs(hi)} pts` : `${signed(lo, 0)} to ${signed(hi, 0)} pts`;
  }

  function verdictText(ref) {
    if (!ref) return DASH;
    if (ref.withheld) return `verdict withheld: ${ref.withheld}`;
    const v = VERDICTS[ref.verdict];
    const text = v ? v[1] : statusLabel(ref.verdict);
    return ref.verdict === 'not_applicable' && POSITION[ref.position] ? `${text} · ${POSITION[ref.position]}` : text;
  }

  /** The verdict chip, its meaning in the title; a withheld verdict shows its reason instead. */
  function verdictChip(ref) {
    if (!ref) return h('span', {class: 'faint'}, 'no published figure');
    if (ref.withheld) return h('span', {class: 'rp-withheld'}, h('b', null, 'Verdict withheld: '), ref.withheld);
    const [glyph, text, tone, meaning] = VERDICTS[ref.verdict] || ['·', statusLabel(ref.verdict), '', ''];
    return h('span', {class: 'rp-verdict', title: meaning || null}, tag(`${glyph} ${text}`, tone),
      ref.verdict === 'not_applicable' && POSITION[ref.position] && h('small', null, POSITION[ref.position]));
  }

  /** not_measured: the harness reports no tokens, or (since the usage addendum) one side's every trial lost its usage. */
  function notMeasuredText(m) {
    const d = m.dropped || {};
    return d.usage_not_recorded > 0 && !(d.tokens_not_reported > 0)
      ? 'no usage recorded on one side' : 'tokens not reported by this harness';
  }

  function metricText(m) {
    if (!m) return DASH;
    if (m.status === 'not_measured') return notMeasuredText(m);
    if (!m.geomean) return `needs ${plural(m.need, 'more paired task')}`;
    return `${times(m.geomean.estimate)} ${interval(m.geomean.lo, m.geomean.hi)}`;
  }

  // Filters, options and preferences --------------------------------------

  /** Known filter keys only, with the right types; an empty list means no constraint and is left out. */
  function cleanFilters(f) {
    const out = {};
    if (!f || typeof f !== 'object') return out;
    if (typeof f.date_from === 'string' && f.date_from) out.date_from = f.date_from;
    if (typeof f.date_to === 'string' && f.date_to) out.date_to = f.date_to;
    for (const k of LIST_KEYS) {
      if (!Array.isArray(f[k])) continue;
      const list = f[k].filter((v) => typeof v === 'string' || (v === null && NULLABLE.has(k)));
      if (list.length) out[k] = unique(list).slice(0, 500);
    }
    if (f.include_cautioned === true) out.include_cautioned = true;
    return out;
  }

  // The task scope: the profile's task file, as the config view lists it.
  const scopeTasks = () => {
    const cfg = ui.state && ui.state.config;
    return cfg && Array.isArray(cfg.tasks) ? cfg.tasks : [];
  };
  const scopeFile = () => (ui.state && ui.state.config && ui.state.config.tasks_file) || 'config/tasks.txt';

  /** What the rail asks for: its filters, and the task scope unless it is off or tasks are ticked. */
  function railFilters() {
    const f = cleanFilters(rp.filters);
    const scope = scopeTasks();
    if (!rp.allTasks && !f.tasks && scope.length) f.tasks = scope.slice(0, 500);
    return f;
  }

  const requestBody = () => ({filters: railFilters(), options: JSON.parse(JSON.stringify(rp.options))});
  const bodyKey = () => JSON.stringify(requestBody());

  /** One sentence on the tasks a report covers (its scope block), e.g. for the explainer and snapshots. */
  function scopeText(scope) {
    if (!scope) return '';
    if (scope.kind === 'scope') return `${plural(scope.tasks, 'task')} from ${scope.tasks_file}`;
    if (scope.kind === 'chosen') return `${plural(scope.tasks, 'chosen task')}`;
    return scope.scope_tasks ? `all tasks, including those outside ${scope.tasks_file}` : 'all tasks';
  }

  /** Switches the task scope off (every task) or on and reloads at once. Not saved: each visit opens on the scope. */
  function setAllTasks(on) {
    rp.allTasks = on;
    apply.cancel();
    savePrefs();
    renderStrip(true);
    renderRail();
    reload();
  }

  function activeCount(f) {
    const c = cleanFilters(f);
    return Object.keys(c).filter((k) => k !== 'date_to' || !c.date_from).length;
  }

  function optValue(path) {
    const mine = dig(rp.options, path);
    if (mine !== undefined) return mine;
    const echo = rp.report ? dig(rp.report.spec.options, path) : undefined;
    if (echo !== undefined) return echo;
    // The profile's baseline_model until the server's echo arrives.
    if (path === 'baseline.model') return (ui.state && ui.state.config && ui.state.config.baseline_model) || null;
    return OPTION_DEFAULTS[path];
  }

  /** Whether the report compares models (each harness held fixed) rather than harnesses (each model held fixed). */
  const byModels = (r = rp.report) => (r ? dig(r.spec, 'options.compare') : optValue('compare')) === 'models';

  function optSet(path, value) {
    const keys = path.split('.');
    let o = rp.options;
    for (const k of keys.slice(0, -1)) o = o[k] = o[k] && typeof o[k] === 'object' ? o[k] : {};
    o[keys[keys.length - 1]] = value;
  }

  const shownOptions = () => Object.keys(OPTION_DEFAULTS).map((p) => optValue(p));

  function loadPrefs() {
    const saved = prefs.get('report.rail', null);
    if (saved && typeof saved === 'object') {
      rp.filters = cleanFilters(saved.filters);
      rp.options = saved.options && typeof saved.options === 'object' && !Array.isArray(saved.options) ? saved.options : {};
    }
    // The REPORT always opens on the task scope; "All tasks" lasts until the page is left or reloaded.
    rp.allTasks = false;
    prefs.set('report.allTasks', null);  // an earlier build saved the switch
    const tab = prefs.get('report.tab', 'report');
    rp.tab = TABS.some(([t]) => t === tab) ? tab : 'report';
    if (rp.handed) {
      rp.filters = rp.handed;
      rp.handed = null;
      rp.tab = 'report';
      savePrefs();
    }
    rp.railOpen = prefs.get('report.railOpen', false) === true;
  }

  const savePrefs = () => prefs.set('report.rail', {filters: cleanFilters(rp.filters), options: rp.options});

  // Skeleton, sub-tabs and the drawer -------------------------------------

  function build(root) {
    fill(root,
      h('div', {id: 'rp-banner', class: 'banner bad rp-banner', role: 'alert', hidden: true}),
      h('section', {class: 'panel rp-strip', id: 'rp-strip'}),
      h('nav', {class: 'rp-tabs', role: 'tablist', 'aria-label': 'Report sections'}, TABS.map(([name, label]) => h('button', {
        type: 'button', role: 'tab', 'data-tab': name, onclick: () => setTab(name),
      }, label, h('span', {class: 'rp-tab-n', id: `rp-tab-n-${name}`})))),
      h('div', {class: 'rp-layout', id: 'rp-layout'},
        h('aside', {class: 'panel rp-rail', id: 'rp-rail', 'aria-label': 'Report filters'}),
        h('div', {class: 'rp-content'}, TABS.map(([name]) => h('div', {class: 'rp-pane', id: `rp-pane-${name}`, role: 'tabpanel', hidden: true})))),
      h('div', {class: 'rp-scrim', id: 'rp-scrim', hidden: true, onclick: closeSide}),
      h('aside', {class: 'rp-side', id: 'rp-side', hidden: true, 'aria-labelledby': 'rp-side-title'},
        h('div', {class: 'rp-side-head'},
          h('span', {class: 'panel-code', id: 'rp-side-code'}, 'R'),
          h('h2', {id: 'rp-side-title'}),
          h('button', {type: 'button', class: 'btn ghost small rp-side-close', onclick: closeSide, 'aria-label': 'Close the drawer'}, 'Close ', h('span', {class: 'arrow'}, '✕'))),
        h('div', {class: 'rp-side-body', id: 'rp-side-body'})));
    rp.built = true;
    showTab();
  }

  function showTab() {
    for (const button of document.querySelectorAll('#view-report .rp-tabs button')) {
      const on = button.dataset.tab === rp.tab;
      button.classList.toggle('active', on);
      button.setAttribute('aria-selected', String(on));
    }
    for (const [name] of TABS) $(`#rp-pane-${name}`).hidden = name !== rp.tab;
    $('#rp-layout').classList.toggle('rp-norail', !RAIL_TABS.has(rp.tab));
    if (!$(`#rp-pane-${rp.tab}`).firstChild) renderPane(rp.tab);  // a placeholder until its data arrives
  }

  function renderPane(name) {
    if (name === 'report') renderReportPane();
    else if (name === 'ledger') renderLedgerPane();
    else if (name === 'runs') renderRunsPane();
    else if (name === 'exclusions') renderExclusionsPane();
    else if (name === 'snapshots') renderSnapshotsPane();
  }

  function setTab(name) {
    if (rp.tab === name) return;
    rp.tab = name;
    prefs.set('report.tab', name);
    showTab();
    loadTab();
  }

  /** The shown sub-tab's data. Lists reload on every visit; the report and ledger only when their request changed. */
  function loadTab() {
    if (rp.tab === 'report') {
      if (rp.stale || !rp.report || rp.reportFor !== bodyKey()) loadReport();
      else if (!rp.coverage) loadReport();
    } else if (rp.tab === 'ledger') {
      loadLedger();
    } else if (rp.tab === 'runs') {
      loadRuns();
    } else if (rp.tab === 'exclusions') {
      loadExclusions();
    } else if (rp.tab === 'snapshots') {
      loadSnapshots();
    }
  }

  function updateTabCounts() {
    const s = rp.status;
    const set = (name, n) => {
      const el = $(`#rp-tab-n-${name}`);
      if (el) el.textContent = isNum(n) ? count(n) : '';
    };
    set('ledger', s ? s.trials : null);
    set('runs', s ? s.runs : null);
    set('exclusions', rp.exclusions ? rp.exclusions.filter((e) => e.effective && e.action === 'exclude').length : null);
    set('snapshots', rp.snapshots ? rp.snapshots.length : null);
  }

  function showSide(code, title, body) {
    const side = $('#rp-side');
    $('#rp-side-code').textContent = code;
    $('#rp-side-title').textContent = title;
    fill($('#rp-side-body'), body);
    $('#rp-side-body').scrollTop = 0;
    if (side.hidden) {
      rp.sideReturn = document.activeElement;
      side.hidden = false;
      $('#rp-scrim').hidden = false;
      // As with the app's drawer: a docked assistant moves left beside it (assistant.css); a floating one would
      // cover it, so it closes.
      document.body.classList.add('rp-side-open');
      if (assistantOpen() && window.innerWidth < 1240) closeAssistant();
      side.querySelector('.rp-side-close').focus();
    }
  }

  function closeSide() {
    rp.side = null;
    const side = $('#rp-side');
    if (!side || side.hidden) return;
    side.hidden = true;
    $('#rp-scrim').hidden = true;
    document.body.classList.remove('rp-side-open');
    fill($('#rp-side-body'));
    hideTip();
    const back = rp.sideReturn;
    rp.sideReturn = null;
    if (back && back.isConnected && typeof back.focus === 'function') back.focus();
  }

  /** A small popup menu of [label, async action] items; an action's error shows in the toast. */
  function menu(label, items) {
    const list = h('div', {class: 'rp-menu-list', role: 'menu', hidden: true},
      items.map(([text, run]) => h('button', {type: 'button', role: 'menuitem', onclick: async (e) => {
        closeMenus();
        await busy(e.currentTarget, run);
      }}, text)));
    const button = h('button', {type: 'button', class: 'btn small', 'aria-haspopup': 'menu', 'aria-expanded': 'false', onclick: () => {
      const open = list.hidden;
      closeMenus();
      list.hidden = !open;
      button.setAttribute('aria-expanded', String(open));
      if (open) list.querySelector('button').focus();
    }}, label, h('span', {class: 'arrow'}, ' ▾'));
    return h('div', {class: 'rp-menu'}, button, list);
  }

  function closeMenus() {
    for (const list of document.querySelectorAll('#view-report .rp-menu-list')) {
      list.hidden = true;
      list.previousElementSibling.setAttribute('aria-expanded', 'false');
    }
  }

  // Banner and status strip -----------------------------------------------

  function renderBanner() {
    const box = $('#rp-banner');
    if (!box) return;
    const L = ui.state ? ui.state.ledger : null;
    const failed = stateError('ledger');
    let text = '';
    if (failed) text = `The ledger is unavailable: ${failed}`;
    else if (L && L.available === false) text = `The ledger is unavailable${L.error ? `: ${L.error}` : ''}. The rest of the dashboard keeps working.`;
    else if (rp.status && rp.status.available === false) text = `The ledger is unavailable${rp.status.error ? `: ${rp.status.error}` : ''}.`;
    else if (rp.busy) text = rp.busy;
    if (!changed('report:banner', text) && box.hidden === !text) return;
    box.hidden = !text;
    fill(box, text && [
      h('span', {class: 'rp-banner-text'}, h('b', null, '! '), text),
      h('button', {type: 'button', class: 'btn ghost small', onclick: () => {
        rp.busy = '';
        renderBanner();
        refreshAll();
      }}, 'Try again'),
    ]);
  }

  function renderStrip(force = false) {
    const box = $('#rp-strip');
    if (!box) return;
    const r = rp.report;
    const s = rp.status;
    const L = stateBlock('ledger');
    const rescanning = rp.rescanning || Boolean(L && L.rescan_running) || Boolean(s && s.rescan && s.rescan.running);
    const ingest = (L && L.last_ingest_at) || (s && s.last_ingest_at) || null;
    const sig = [r && r.provenance, r && r.totals, r && r.scope, rp.allTasks, scopeTasks().length,
      s && [s.trials, s.warnings, s.last_error, s.available], ingest, rescanning];
    if (!changed('report:strip', sig) && !force && box.firstChild) return;
    const p = r ? r.provenance : null;
    const t = r ? r.totals : null;
    const ledgerTrials = s ? s.trials : L ? L.trials : null;
    const fp = p && tip(h('span', {class: 'rp-fp'}, `fingerprint ${shortFp(p.fingerprint)}`), 'Report provenance', [
      [when(p.generated_at), 'generated'], [p.pricing, 'pricing'], [p.pricing_sha || DASH, 'price table'],
      [count(p.exclusions_applied), 'exclusions applied'], [(p.parser_versions || []).join(', ') || DASH, 'parser'],
      [String(p.classifier_version), 'classifier'], [String(p.stats_version), 'stats'],
      [(p.harbor_versions || []).join(', ') || DASH, 'Harbor'], p.fingerprint,
    ], true);
    const line = [
      t ? `${count(p.trials_used)} of ${count(ledgerTrials)} trials used` : (ledgerTrials !== null ? `${count(ledgerTrials)} trials in the ledger` : 'Loading…'),
      t && plural(t.runs, 'run'),
      t && t.profiles.length ? t.profiles.join(', ') : null,
      ingest && `last ingest ${clockTime(ingest).slice(0, 5)}`,
    ].filter(Boolean);
    const lowerBound = t && t.usage_unmeasured > 0;
    const save = h('button', {type: 'button', class: 'btn small', disabled: !r, onclick: saveSnapshot}, 'Save snapshot');
    fill(box,
      h('div', {class: 'panel-head'},
        h('span', {class: 'panel-code'}, 'R-00'),
        h('h2', {class: 'panel-title'}, 'Ledger report'),
        h('div', {class: 'panel-meta rp-strip-actions'},
          h('button', {type: 'button', class: 'btn small', disabled: rescanning, onclick: (e) => doRescan(e.currentTarget)},
            rescanning ? [led('warn live'), 'Rescanning…'] : 'Rescan'),
          save,
          menu('Export', [
            ['Report trials (CSV)', () => exportReport('trials')],
            ['Report pairs (CSV)', () => exportReport('pairs')],
            ['Report comparisons (CSV)', () => exportReport('comparisons')],
            ['Ledger (CSV)', () => exportLedger('csv')],
            ['Ledger (JSONL)', () => exportLedger('jsonl')],
          ]))),
      h('div', {class: 'panel-body rp-strip-body'},
        h('div', {class: 'rp-strip-line'},
          line.map((text, i) => [i > 0 && h('span', {class: 'faint'}, ' · '), h('span', null, text)]),
          fp && [h('span', {class: 'faint'}, ' · '), fp],
          t && [h('span', {class: 'faint'}, ' · '), tip(h('span', null, `spend ${lowerBound ? '≥ ' : ''}${money(t.spend_usd)}`), 'Spend of the trials used',
            [lowerBound ? `${plural(t.usage_unmeasured, 'trial')} recorded no usage, so this is a lower bound.` : 'Every trial used recorded its usage.'], true)]),
        h('div', {class: 'rp-strip-tags'},
          scopeControl(r),
          p && p.includes_active_run && tag('includes a running run', 'run'),
          t && t.excluded_manual > 0 && tag(`${count(t.excluded_manual)} excluded by hand`, 'warn'),
          lowerBound && tag(`${plural(t.usage_unmeasured, 'trial')} without usage`, 'warn'),
          p && p.pricing !== 'current' && tag(`pricing: ${statusLabel(p.pricing)}`),
          s && s.last_error && tip(tag('ledger error', 'bad'), 'Last ledger error', [[when(s.last_error.at), 'at'], [s.last_error.where || DASH, 'where'], s.last_error.message], true),
          s && (s.warnings || []).map((w) => h('span', {class: 'warn-text rp-strip-warn'}, `! ${w}`))),
        t ? outcomeBar(t.by_outcome) : null));
  }

  /** The tasks the report covers, and the one-click switch between the task scope and every task. */
  function scopeControl(r) {
    const n = scopeTasks().length;
    const scope = r && r.scope;
    if (!n && !(scope && scope.kind === 'chosen')) return null;
    if ((rp.filters.tasks || []).length) {
      return h('span', {class: 'rp-scope'}, tag(`tasks: ${plural(rp.filters.tasks.length, 'ticked task')}`),
        h('small', {class: 'faint'}, ' chosen in Filters, instead of the scope'));
    }
    if (rp.allTasks) {
      return h('span', {class: 'rp-scope'},
        tip(tag('all tasks', 'warn'), 'Task scope off',
          [`Trials on tasks outside ${scopeFile()} count too, so comparisons may pair on different tasks.`,
            'Not saved: the next visit opens on the scope again.'], true),
        h('button', {type: 'button', class: 'btn ghost small', onclick: () => setAllTasks(false)},
          `Only the ${n} in ${scopeFile()}`));
    }
    return h('span', {class: 'rp-scope'},
      tip(tag(`${plural(n, 'task')} from ${scopeFile()}`, 'ok'), 'Task scope',
        ["Every comparison is on these tasks, the profile's task file. Trials on other tasks are left out."], true),
      h('button', {type: 'button', class: 'btn ghost small', onclick: () => setAllTasks(true)}, 'All tasks'));
  }

  // Rail --------------------------------------------------------------------

  function railQuiet() {
    const rail = $('#rp-rail');
    const focus = document.activeElement;
    return !apply.pending() && !rp.invalid && !(rail && focus && focus !== rail && rail.contains(focus));
  }

  const railSig = () => JSON.stringify([rp.facets, shownOptions(), cleanFilters(rp.filters), rp.allTasks,
    scopeTasks().length, rp.metaError && rp.metaError.message]);

  /** Rebuilds the rail when its data changed, but never while it holds unsent input (unless forced by a deliberate replace). */
  function renderRail(force = false) {
    const rail = $('#rp-rail');
    if (!rail) return;
    if (!force && !railQuiet()) {
      rp.railStale = true;
      return;
    }
    rp.railStale = false;
    const sig = railSig();
    if (!force && sig === rp.railFor && rail.firstChild) return;
    rp.railFor = sig;
    const old = rail.querySelector('.rp-rail-body');
    const scroll = old ? old.scrollTop : 0;
    rail.classList.toggle('rp-collapsed', !rp.railOpen);
    fill(rail,
      h('div', {class: 'panel-head'},
        h('span', {class: 'panel-code'}, 'R-01'),
        h('h2', {class: 'panel-title'}, 'Filters'),
        h('div', {class: 'panel-meta'},
          h('span', {id: 'rp-rail-n'}),
          h('button', {type: 'button', class: 'btn ghost small rp-rail-toggle', 'aria-expanded': String(rp.railOpen), onclick: toggleRail},
            rp.railOpen ? 'Hide' : 'Show'))),
      h('div', {class: 'rp-rail-body'}, railBody()));
    const body = rail.querySelector('.rp-rail-body');
    if (body) body.scrollTop = scroll;
    updateCounts();
  }

  function toggleRail() {
    rp.railOpen = !rp.railOpen;
    prefs.set('report.railOpen', rp.railOpen);
    const rail = $('#rp-rail');
    rail.classList.toggle('rp-collapsed', !rp.railOpen);
    const button = rail.querySelector('.rp-rail-toggle');
    button.textContent = rp.railOpen ? 'Hide' : 'Show';
    button.setAttribute('aria-expanded', String(rp.railOpen));
  }

  /** After every rail edit: remember it, apply it 400 ms after the last one. */
  function edited() {
    savePrefs();
    rp.railFor = railSig();
    apply();
    updateCounts();
  }

  function updateCounts() {
    const f = rp.filters;
    for (const el of document.querySelectorAll('#rp-rail .rp-sec-n[data-keys]')) {
      const n = el.dataset.keys.split(',').reduce((total, k) => total
        + (k === 'dates' ? (f.date_from || f.date_to ? 1 : 0) : k === 'include_cautioned' ? (f.include_cautioned ? 1 : 0) : (f[k] || []).length), 0);
      el.textContent = n ? `${n} chosen` : '';
    }
    const meta = $('#rp-rail-n');
    if (meta) {
      const n = activeCount(f);
      meta.textContent = `${n ? `${n} active` : 'none active'}${apply.pending() ? ' · applying…' : ''}`;
    }
  }

  function toggleValue(key, value, on) {
    const list = (rp.filters[key] || []).filter((v) => vkey(v) !== vkey(value));
    if (on) list.push(value);
    if (list.length) rp.filters[key] = list;
    else delete rp.filters[key];
  }

  /** Puts the rail's checkboxes, exclude buttons and dates back in line with rp.filters, without rebuilding it. */
  function syncRail() {
    const f = rp.filters;
    for (const box of document.querySelectorAll('#rp-rail input[data-k]')) {
      box.checked = (f[box.dataset.k] || []).some((v) => vkey(v) === box.dataset.v);
    }
    for (const button of document.querySelectorAll('#rp-rail .rp-x[data-k]')) {
      const out = (f[button.dataset.k] || []).some((v) => vkey(v) === button.dataset.v);
      button.setAttribute('aria-pressed', String(out));
      button.closest('.rp-opt').classList.toggle('rp-out', out);
    }
    const from = $('#rp-rail input[name="date_from"]');
    const to = $('#rp-rail input[name="date_to"]');
    if (from) from.value = f.date_from ? f.date_from.slice(0, 10) : '';
    if (to) to.value = f.date_to ? f.date_to.slice(0, 10) : '';
    syncPresets();
    updateCounts();
  }

  function presetDays() {
    const f = rp.filters;
    if (!f.date_from && !f.date_to) return 0;
    if (f.date_to) return null;
    for (const days of [7, 30]) if (f.date_from === dayString(Date.now() - days * 864e5)) return days;
    return null;
  }

  function syncPresets() {
    const active = presetDays();
    for (const button of document.querySelectorAll('#rp-rail .rp-presets button')) {
      const on = Number(button.dataset.days) === active;
      button.classList.toggle('active', on);
      button.setAttribute('aria-pressed', String(on));
    }
  }

  /** One collapsible rail section. `keys` are the filter keys it sets, for its "N chosen" count. */
  function section(title, keys, body, {tools = null} = {}) {
    const det = h('details', {class: 'rp-sec', open: rp.secOpen.has(title)},
      h('summary', null, h('span', {class: 'rp-sec-title'}, title), h('span', {class: 'rp-sec-n', 'data-keys': keys.join(',')})),
      tools && h('div', {class: 'rp-sec-tools'}, tools),
      h('div', {class: 'rp-sec-body'}, body));
    det.addEventListener('toggle', () => {
      if (det.open) rp.secOpen.add(title);
      else rp.secOpen.delete(title);
    });
    return det;
  }

  /** A checkbox row: ticked means "only these". With excludeKey, a ✕ button leaves the value out instead. */
  function optRow(key, item, {excludeKey = null} = {}) {
    const chosen = (rp.filters[key] || []).some((v) => vkey(v) === vkey(item.value));
    const out = Boolean(excludeKey) && (rp.filters[excludeKey] || []).includes(item.value);
    const box = h('input', {type: 'checkbox', checked: chosen, 'data-k': key, 'data-v': vkey(item.value), onchange: () => {
      toggleValue(key, item.value, box.checked);
      if (box.checked && excludeKey) toggleValue(excludeKey, item.value, false);
      syncRail();
      edited();
    }});
    const leaveOut = excludeKey && h('button', {
      type: 'button', class: 'rp-x', 'data-k': excludeKey, 'data-v': vkey(item.value), 'aria-pressed': String(out),
      title: 'Leave these out', 'aria-label': `Leave out ${item.text || item.value}`,
      onclick: () => {
        const now = !(rp.filters[excludeKey] || []).includes(item.value);
        toggleValue(excludeKey, item.value, now);
        if (now) toggleValue(key, item.value, false);
        syncRail();
        edited();
      },
    }, '✕');
    return h('div', {class: `rp-opt${out ? ' rp-out' : ''}${item.missing ? ' rp-missing' : ''}`},
      h('label', null, box, h('span', {class: 'rp-opt-text'},
        h('span', {class: 'rp-opt-label'}, item.label, item.badge || null),
        item.sub && h('small', null, item.sub))),
      h('span', {class: 'rp-opt-n'}, isNum(item.count) ? count(item.count) : ''),
      leaveOut);
  }

  /** Items plus any chosen value the facets no longer list (so it can be unticked), e.g. a run handed over by ANALYSIS. */
  function withChosen(key, items, excludeKey = null) {
    const known = new Set(items.map((i) => vkey(i.value)));
    const extra = [...(rp.filters[key] || []), ...(excludeKey ? rp.filters[excludeKey] || [] : [])]
      .filter((v) => !known.has(vkey(v)));
    return [...items, ...unique(extra).map((v) => ({value: v, label: v === null ? 'harness default' : String(v), count: 0, missing: true, sub: 'not in the ledger'}))];
  }

  function railBody() {
    const f = rp.facets;
    if (!f) {
      return h('p', {class: 'muted rp-pad'}, rp.metaError ? `Could not load the filters: ${rp.metaError.message}` : 'Loading filters…');
    }
    const filters = rp.filters;
    const configs = f.configs || [];
    const implTrials = {};
    for (const c of configs) implTrials[c.implementation] = (implTrials[c.implementation] || 0) + c.trials;
    const impls = unique([...(f.families || []).flatMap((fam) => fam.implementations || []), ...Object.keys(implTrials)]);
    const familiesOf = (impl) => (f.families || []).filter((fam) => (fam.implementations || []).includes(impl)).map((fam) => fam.value);
    const versions = new Map();
    for (const v of f.harness_versions || []) {
      const row = versions.get(v.value) || {value: v.value, families: [], count: 0};
      row.families.push(v.family);
      row.count += v.trials;
      versions.set(v.value, row);
    }
    const order = f.task_order || [];
    const taskFacet = new Map((f.tasks || []).map((t) => [t.value, t]));
    const taskItem = (task) => {
      const t = taskFacet.get(task);
      return {value: task, label: task, count: t ? t.trials : 0, sub: t && t.digests > 1 ? `${t.digests} task versions` : null};
    };
    const otherTasks = (f.tasks || []).filter((t) => !order.includes(t.value)).map((t) => taskItem(t.value));
    const block = (from, to) => order.slice(from, to);
    const setTasks = (list) => {
      const same = (filters.tasks || []).length === list.length && list.every((t) => (filters.tasks || []).includes(t));
      if (same || !list.length) delete filters.tasks;
      else filters.tasks = list.slice();
      syncRail();
      edited();
    };
    const byProvider = new Map();
    for (const m of f.models || []) {
      if (!byProvider.has(m.provider)) byProvider.set(m.provider, []);
      byProvider.get(m.provider).push(m);
    }
    const missingModels = withChosen('models', (f.models || []).map((m) => ({value: m.key}))).filter((i) => i.missing);
    const cautioned = configs.filter((c) => (c.cautions || []).length).length;

    return [
      section('Dates', ['dates'], datesBody()),
      section('Profiles', ['profiles', 'sources'], [
        withChosen('profiles', (f.profiles || []).map((p) => ({value: p.value, label: p.value, count: p.trials}))).map((i) => optRow('profiles', i)),
        (f.sources || []).length > 0 && h('div', {class: 'rp-subhead'}, 'Source'),
        withChosen('sources', (f.sources || []).map((s) => ({value: s.value, label: s.value, count: s.trials}))).map((i) => optRow('sources', i)),
      ]),
      section('Models', ['models'], [
        [...byProvider].map(([provider, list]) => [
          h('div', {class: 'rp-subhead'}, provider),
          list.map((m) => optRow('models', {value: m.key, label: m.model, text: m.key, count: m.trials})),
        ]),
        missingModels.map((i) => optRow('models', i)),
      ]),
      section('Harnesses', ['families'], withChosen('families', (f.families || []).map((fam) => ({
        value: fam.value, label: fam.value, count: fam.trials,
        sub: (fam.versions || []).length ? `versions ${fam.versions.join(', ')}` : null,
      }))).map((i) => optRow('families', i))),
      section('Implementations', ['implementations'], withChosen('implementations', impls.map((impl) => ({
        value: impl, text: impl, count: implTrials[impl],
        label: impl === 'builtin' ? 'built in' : h('span', {class: 'mono rp-impl'}, impl),
        badge: impl === 'builtin' ? null : tag('custom'),
        sub: familiesOf(impl).join(', ') || null,
      }))).map((i) => optRow('implementations', i))),
      section('Configurations', ['configs', 'exclude_configs', 'include_cautioned'], [
        h('label', {class: 'switch rp-switch'},
          h('input', {type: 'checkbox', checked: filters.include_cautioned === true, onchange: (e) => {
            if (e.target.checked) filters.include_cautioned = true;
            else delete filters.include_cautioned;
            edited();
          }}),
          h('span', null, `Include cautioned configs${cautioned ? ` (${cautioned})` : ''}`)),
        withChosen('configs', configs.map((c) => ({
          value: c.config_hash, text: c.label, count: c.trials,
          label: c.label,
          badge: (c.cautions || []).length ? tip(tag('caution', 'hazard'), 'Cautions', c.cautions, true) : null,
          sub: `#${c.config_hash.slice(0, 8)} · ${when(c.first_seen)} – ${when(c.last_seen)}`,
        })), 'exclude_configs').map((i) => optRow('configs', i, {excludeKey: 'exclude_configs'})),
      ]),
      section('Harness versions', ['harness_versions'], withChosen('harness_versions', [...versions.values()].map((v) => ({
        value: v.value, label: v.value, count: v.count, sub: v.families.join(', '),
      }))).map((i) => optRow('harness_versions', i))),
      section('Efforts', ['efforts'], withChosen('efforts', (f.efforts || []).map((e) => ({
        value: e.value, label: e.value === null ? 'harness default' : e.value, count: e.trials,
      }))).map((i) => optRow('efforts', i))),
      section('Harness setup', ['setup_modes'], withChosen('setup_modes', (f.setup_modes || []).map((m) => ({
        value: m.value, label: setupLabel(m.value), count: m.trials,
        sub: SETUP[m.value] ? SETUP[m.value][2] : 'killed before its environment started',
      }))).map((i) => optRow('setup_modes', i))),
      section('Datasets', ['datasets', 'snapshots'], [
        withChosen('datasets', (f.datasets || []).map((d) => ({value: d.value, label: d.value, count: sum(d.snapshots || [], (s) => s.trials)}))).map((i) => optRow('datasets', i)),
        h('div', {class: 'rp-subhead'}, 'Snapshots'),
        withChosen('snapshots', (f.datasets || []).flatMap((d) => (d.snapshots || []).map((s) => ({
          value: s.id, label: h('span', {class: 'mono rp-impl'}, s.id), text: s.id, count: s.trials, sub: d.value,
        })))).map((i) => optRow('snapshots', i)),
      ]),
      section('Runs', ['runs', 'exclude_runs'], withChosen('runs', (f.runs || []).map((r) => ({
        value: r.run_key, text: r.run_key, count: r.trials,
        label: h('span', {class: 'mono'}, r.run_id),
        sub: `${r.profile} · ${statusLabel(r.status)} · ${when(r.created_at)}`,
      })), 'exclude_runs').map((i) => optRow('runs', i, {excludeKey: 'exclude_runs'}))),
      section('Tasks', ['tasks'], [
        scopeTasks().length > 0 && h('label', {class: 'switch rp-switch'},
          h('input', {type: 'checkbox', checked: !rp.allTasks, onchange: (e) => setAllTasks(!e.target.checked)}),
          h('span', null, `Only the ${plural(scopeTasks().length, 'task')} in ${scopeFile()}`)),
        scopeTasks().length > 0 && h('small', {class: 'faint rp-scope-note'},
          'Ticking tasks below replaces the scope.'),
        order.length > 0 && h('div', {class: 'rp-subhead'}, 'Block 1'),
        block(0, 10).map((t) => optRow('tasks', taskItem(t))),
        order.length > 10 && h('div', {class: 'rp-subhead'}, 'Block 2'),
        block(10, 20).map((t) => optRow('tasks', taskItem(t))),
        order.length > 20 && h('div', {class: 'rp-subhead'}, 'Rest of the sample'),
        block(20).map((t) => optRow('tasks', taskItem(t))),
        otherTasks.length > 0 && h('div', {class: 'rp-subhead'}, 'Other tasks'),
        withChosen('tasks', otherTasks).filter((i) => !order.includes(i.value)).map((i) => optRow('tasks', i)),
      ], {tools: order.length ? [
        h('button', {type: 'button', class: 'btn ghost small', onclick: () => setTasks(block(0, 10)), title: 'Tasks 1 to 10 of this profile\'s sample'}, 'Block 1'),
        order.length > 10 && h('button', {type: 'button', class: 'btn ghost small', onclick: () => setTasks(block(10, 20)), title: 'Tasks 11 to 20 of this profile\'s sample'}, 'Block 2'),
        h('button', {type: 'button', class: 'btn ghost small', onclick: () => setTasks([]),
          title: 'Untick every task: the task scope applies, or every task with it off'}, 'Clear'),
      ] : null}),
      section('Harbor versions', ['harbor_versions'], withChosen('harbor_versions', (f.harbor_versions || []).map((v) => ({
        value: v.value, label: v.value, count: v.trials,
      }))).map((i) => optRow('harbor_versions', i))),
      advancedSection(),
      h('div', {class: 'rp-rail-foot'},
        h('button', {type: 'button', class: 'btn ghost small rp-reset', onclick: resetRail}, 'Reset to defaults'),
        h('small', {class: 'faint'}, 'Kept in this browser for this profile.')),
    ];
  }

  function datesBody() {
    const f = rp.filters;
    const onDate = (e) => {
      const input = e.target;
      if (input.value) f[input.name] = input.value;
      else delete f[input.name];
      syncPresets();
      edited();
    };
    const preset = (days) => h('button', {type: 'button', 'data-days': days, onclick: () => {
      delete f.date_to;
      if (days) f.date_from = dayString(Date.now() - days * 864e5);
      else delete f.date_from;
      syncRail();
      edited();
    }}, days ? `${days}d` : 'All');
    const range = rp.facets && rp.facets.date_range;
    return [
      h('div', {class: 'rp-seg rp-presets', role: 'group', 'aria-label': 'Date presets'}, preset(7), preset(30), preset(0)),
      h('div', {class: 'rp-dates'},
        field('From', h('input', {type: 'date', name: 'date_from', value: f.date_from ? f.date_from.slice(0, 10) : '', oninput: onDate})),
        field('To', h('input', {type: 'date', name: 'date_to', value: f.date_to ? f.date_to.slice(0, 10) : '', oninput: onDate}))),
      range && range.min && h('small', {class: 'faint'}, `Trials from ${when(range.min)} to ${when(range.max)} (UTC dates).`),
    ];
  }

  /** The report options. Values shown come from the server's echo (spec.options) unless the person set them. */
  function advancedSection() {
    const f = rp.facets || {};
    const changedOption = () => edited();
    const seg = (path, choices, after = null) => {
      const group = h('div', {class: 'rp-seg', role: 'group'}, choices.map(([v, label]) => h('button', {
        type: 'button', 'data-v': v, class: optValue(path) === v ? 'active' : null, 'aria-pressed': String(optValue(path) === v),
        onclick: () => {
          optSet(path, v);
          for (const b of group.children) {
            b.classList.toggle('active', b.dataset.v === v);
            b.setAttribute('aria-pressed', String(b.dataset.v === v));
          }
          changedOption();
          if (after) after();
        },
      }, label)));
      return group;
    };
    const sw = (path, label, onOff = null) => h('label', {class: 'switch rp-switch'},
      h('input', {type: 'checkbox', checked: onOff ? optValue(path) === onOff[0] : optValue(path) === true, onchange: (e) => {
        optSet(path, onOff ? (e.target.checked ? onOff[0] : onOff[1]) : e.target.checked);
        changedOption();
      }}),
      h('span', null, label));
    const family = optValue('baseline.family');
    const families = (f.families || []).map((fam) => [fam.value, fam.value]);
    const pinChoices = [['', 'Any config'], ...(f.configs || []).filter((c) => c.family === family)
      .map((c) => [c.config_hash, `#${c.config_hash.slice(0, 8)} · ${c.label.split(' · ').slice(1).join(' · ')}`])];
    const models = optValue('compare') === 'models';
    const pin = select('baseline_config', pinChoices, optValue('baseline.config') || '', {disabled: models,
      onchange: (e) => {
        optSet('baseline.config', e.target.value || null);
        changedOption();
      }});
    const baseModel = optValue('baseline.model') || '';
    const modelChoices = (f.models || []).map((m) => [m.key, m.key]);
    if (baseModel && !modelChoices.some(([v]) => v === baseModel)) {
      modelChoices.unshift([baseModel, `${baseModel} (no trials)`]);
    }
    const ciError = h('small', {class: 'warn-text', hidden: !rp.invalid}, rp.invalid);
    const ci = numberInput('ci_level', optValue('ci_level'), {min: 0.8, max: 0.99, step: 0.01, oninput: (e) => {
      const v = Number(e.target.value);
      if (e.target.value !== '' && isNum(v) && v >= 0.8 && v <= 0.99) {
        rp.invalid = '';
        optSet('ci_level', Math.round(v * 100) / 100);
        changedOption();
      } else {
        rp.invalid = 'The CI level must be between 0.80 and 0.99.';
        updateCounts();
      }
      ciError.hidden = !rp.invalid;
      ciError.textContent = rp.invalid;
    }});
    return section('Advanced', [], [
      field('Compare', seg('compare', COMPARE_CHOICES, () => renderRail(true)), {notes: [note(models
        ? 'Each harness held fixed: every model it ran ÷ its baseline model.'
        : 'Each model held fixed: every harness ÷ the baseline family.')]}),
      field('Group by', seg('group_by', [['family', 'Harness'], ['implementation', 'Implementation'], ['config', 'Exact config']])),
      sw('split_harness_versions', 'Split by harness version'),
      field('Baseline family', select('baseline_family', families, family, {disabled: models, onchange: (e) => {
        optSet('baseline.family', e.target.value);
        optSet('baseline.config', null);
        pin.value = '';
        changedOption();
      }}), {notes: models ? [note('Comparing models: each harness is its own baseline.')] : []}),
      field('Baseline model', select('baseline_model', modelChoices, baseModel, {disabled: !models, onchange: (e) => {
        optSet('baseline.model', e.target.value);
        changedOption();
      }}), {notes: [note(models ? 'Every model is compared with this one, harness by harness.'
        : 'Used when comparing models.')]}),
      field('Baseline config', pin, {notes: [note('Pin the baseline to one exact configuration.')]}),
      field('Pair on', select('match_on', [['digest', 'task content (default)'], ['name', 'task name']], optValue('pairing.match_on'), {onchange: (e) => {
        optSet('pairing.match_on', e.target.value);
        changedOption();
      }})),
      sw('pairing.same_run_only', 'Same run only'),
      sw('policy.crashes', 'Count harness crashes as failures', ['include', 'exclude']),
      sw('policy.misconfigured_cap', 'Include trials stopped by a mispriced cap', ['include', 'exclude']),
      sw('policy.exclusions', 'Apply manual exclusions'),
      field('Pricing', select('pricing', [['current', 'current table'], ['as_ingested', 'as ingested'], ['reported', 'harness-reported']], optValue('pricing'), {onchange: (e) => {
        optSet('pricing', e.target.value);
        changedOption();
      }})),
      field('CI level', ci, {notes: [note('0.80 to 0.99'), ciError]}),
      field('Harness setup', seg('setup', [['any', 'Any'], ['stock', 'Stock'], ['prebaked', 'Pre-baked']]),
        {notes: [note('Stock includes fallbacks. Neither changes what a trial measures.')]}),
    ]);
  }

  function resetRail() {
    apply.cancel();
    rp.filters = {};
    rp.options = {};
    rp.invalid = '';
    rp.allTasks = false;
    savePrefs();
    renderRail(true);
    const reset = $('#rp-rail .rp-reset');
    if (reset) reset.focus();
    reload();
  }

  /** The rail's filters were applied: reload what reads them. */
  function reload() {
    updateCounts();
    rp.ledger.offset = 0;
    loadReport().then(refreshSide);
    if (rp.tab === 'ledger') loadLedger();
  }

  function refreshAll() {
    loadMeta();
    loadReport().then(refreshSide);
    if (rp.tab !== 'report') loadTab();
  }

  /**
   * The drawer after the report reloaded. A comparison can leave the report (its last usable trial was excluded,
   * or the filters changed): then the drawer keeps its last pairs, says so, and refreshes only their trial rows,
   * so "Include again" stays in reach.
   */
  function refreshSide() {
    const side = rp.side;
    if (!side) return;
    if (side.kind === 'trial') {
      loadTrial();
    } else if (comparisonOf(side.key) || !rp.report) {
      loadPairs();
    } else if (side.data) {
      loadPairTrials(side);
    } else {
      renderPairs();
    }
  }

  function afterIngest() {
    rp.stale = true;
    if (ui.view === 'report') refreshAll();
  }

  // Loading ---------------------------------------------------------------

  async function loadMeta() {
    const out = await fresh.meta(Promise.allSettled([api('GET', '/api/ledger/facets'), api('GET', '/api/ledger/status')]));
    if (out === STALE) return;
    const [facets, status] = out;
    if (facets.status === 'fulfilled') {
      rp.facets = facets.value;
      rp.metaError = null;
    } else {
      rp.metaError = noteError(facets.reason);
    }
    if (status.status === 'fulfilled') rp.status = status.value;
    else noteError(status.reason);
    if (facets.status === 'fulfilled' && status.status === 'fulfilled') clearBusy();
    renderBanner();
    renderStrip();
    renderRail();
    updateTabCounts();
    if (rp.tab === 'exclusions') renderExclusionsPane();
    if (rp.tab === 'report' && rp.report) renderReportPane();
  }

  async function loadReport() {
    const body = requestBody();
    const key = JSON.stringify(body);
    let out;
    try {
      out = await dimWhile([$('#rp-pane-report'), $('#rp-strip')], fresh.report(Promise.all([
        api('POST', '/api/report', body),
        api('POST', '/api/report/coverage', body).catch((err) => ({error: err})),
      ])));
    } catch (err) {
      rp.reportError = noteError(err);
      renderStrip(true);
      renderReportPane();
      return;
    }
    if (out === STALE) return;
    [rp.report, rp.coverage] = out;
    rp.reportFor = key;
    rp.reportError = null;
    rp.stale = false;
    if (!rp.coverage.error) clearBusy();
    renderStrip();
    renderReportPane();
    renderRail();
    if (rp.side && rp.side.kind === 'pairs' && rp.side.data) renderPairs();
  }

  async function loadLedger() {
    const L = rp.ledger;
    const body = {filters: railFilters(), limit: PAGE, offset: L.offset, order: L.order};
    if (L.outcomes.length) body.outcomes = L.outcomes;
    const key = JSON.stringify(body);
    try {
      const data = await dimWhile([$('#rp-pane-ledger')], fresh.ledger(api('POST', '/api/ledger/trials', body)));
      if (data === STALE) return;
      L.data = data;
      L.for = key;
      L.error = null;
      clearBusy();
    } catch (err) {
      L.error = noteError(err);
    }
    renderLedgerPane();
  }

  async function loadRuns() {
    try {
      const data = await dimWhile([$('#rp-pane-runs')], fresh.runs(api('GET', '/api/ledger/runs')));
      if (data === STALE) return;
      rp.runs = data;
      rp.runsError = null;
    } catch (err) {
      rp.runsError = noteError(err);
    }
    renderRunsPane();
  }

  async function loadExclusions() {
    loadMeta();  // the seeded exclusions come with the status
    try {
      const data = await dimWhile([$('#rp-pane-exclusions')], fresh.excl(api('GET', '/api/ledger/exclusions')));
      if (data === STALE) return;
      rp.exclusions = data.exclusions;
      rp.exclError = null;
    } catch (err) {
      rp.exclError = noteError(err);
    }
    updateTabCounts();
    renderExclusionsPane();
  }

  async function loadSnapshots() {
    try {
      const data = await dimWhile([$('#rp-pane-snapshots')], fresh.snaps(api('GET', '/api/report/snapshots')));
      if (data === STALE) return;
      rp.snapshots = data.snapshots;
      rp.snapError = null;
    } catch (err) {
      rp.snapError = noteError(err);
    }
    updateTabCounts();
    renderSnapshotsPane();
  }

  // Actions -----------------------------------------------------------------

  async function doRescan() {
    rp.rescanning = true;
    renderStrip(true);
    try {
      const out = await api('POST', '/api/ledger/rescan', {});
      const skipped = out.skipped && out.skipped.length ? `, ${count(out.skipped.length)} skipped` : '';
      toast(`Rescan done: ${count(out.inserted)} inserted, ${count(out.upgraded)} upgraded, ${count(out.incomplete)} incomplete${skipped}`);
      clearBusy();
    } catch (err) {
      noteError(err);
      toast(err.message, true);
    } finally {
      rp.rescanning = false;
      renderStrip(true);
    }
    rp.stale = true;
    refreshAll();
    poll();
  }

  async function exportReport(level) {
    const name = await download('/api/report/export', {...requestBody(), level});
    toast(`Saved ${name}`);
  }

  async function exportLedger(format) {
    const name = await download('/api/ledger/export', {filters: railFilters(), format});
    toast(`Saved ${name}`);
  }

  async function saveSnapshot() {
    const body = requestBody();
    const n = activeCount({...rp.filters, tasks: undefined});
    const tasks = scopeText(rp.report && rp.report.scope);
    const input = h('input', {type: 'text', name: 'name', required: true, maxlength: 80, autocomplete: 'off', placeholder: 'e.g. After pilot 3'});
    rp.dialog = {input, initial: ''};
    const saved = await formDialog({
      code: 'SNAP', title: 'Save a snapshot', ok: 'Save snapshot',
      body: [
        h('p', null, 'Stores this report as it is now, with its filters and options, so you can open it later and compare it with the ledger then.'),
        field('Name', input, {notes: [note(`${tasks ? `${tasks} · ` : ''}`
          + `${n ? plural(n, 'other filter') : 'No other filters'} · 1 to 80 characters`)]}),
      ],
      read: async () => {
        const name = input.value.trim();
        if (!name) throw new Error('Give the snapshot a name.');
        return api('POST', '/api/report/snapshots', {name, ...body});
      },
    });
    rp.dialog = null;
    if (!saved) return;
    toast(`Saved snapshot "${saved.name}" (${plural(saved.trials_used, 'trial')})`);
    loadSnapshots();
  }

  /**
   * Exclude or include a trial (trial_key) or a run (run_key) with a required reason, then reload what it touches.
   * The request is sent inside the dialog, so its error shows there.
   */
  async function editExclusion(targetType, targetId, action, {current = null} = {}) {
    const what = targetType === 'run' ? 'run' : 'trial';
    const reason = h('textarea', {name: 'reason', required: true, maxlength: 500, rows: 3,
      placeholder: action === 'exclude' ? 'Why leave it out? For example: task files damaged by CRLF line endings' : 'Why count it again?'});
    rp.dialog = {input: reason, initial: ''};
    const out = await formDialog({
      code: action === 'exclude' ? 'EXCLUDE' : 'INCLUDE',
      title: action === 'exclude' ? `Exclude this ${what}?` : `Include this ${what} again?`,
      tone: action === 'exclude' ? 'danger' : '',
      ok: action === 'exclude' ? `Exclude ${what}` : 'Include again',
      body: [
        params([[what, h('span', {class: 'mono'}, targetId)], current && current.reason && ['Now', `${current.action === 'exclude' ? 'excluded' : 'included'}: ${current.reason}`]]),
        h('p', null, action === 'exclude'
          ? `Reports leave this ${what} out while "Apply manual exclusions" is on. Nothing is deleted, and you can include it again.`
          : `Reports count this ${what} again. The exclusion stays in the history.`),
        field('Reason', reason, {notes: [note('Required, up to 500 characters. Kept in the exclusion history.')]}),
      ],
      read: async () => {
        const text = reason.value.trim();
        if (!text) throw new Error('Give a reason.');
        return api('POST', '/api/ledger/exclusions', {target_type: targetType, target_id: targetId, action, reason: text});
      },
    });
    rp.dialog = null;
    if (!out) return null;
    toast(action === 'exclude' ? `Excluded ${targetType === 'run' ? targetId : trialName(targetId)}` : `${targetType === 'run' ? targetId : trialName(targetId)} is included again`);
    rp.stale = true;
    refreshAll();
    if (rp.tab !== 'runs') rp.runs = null;
    loadExclusions();
    return out;
  }

  /** The keyboard's way into a clickable trial row: a button in one cell (the row itself handles the mouse). */
  function openButton(t, opts, label) {
    return h('button', {type: 'button', class: 'link-btn rp-open', 'aria-label': `Open trial ${t.trial_name || t.id}`, onclick: (e) => {
      e.stopPropagation();
      openTrial(t.id, opts);
    }}, label);
  }

  function exclusionButtons(row, {run = true} = {}) {
    const ex = row.exclusion;
    return h('span', {class: 'rp-acts'},
      ex && ex.action === 'exclude'
        ? h('button', {type: 'button', class: 'btn ghost small', onclick: (e) => {
          e.stopPropagation();
          editExclusion('trial', row.trial_key, 'include', {current: ex});
        }}, 'Include again')
        : h('button', {type: 'button', class: 'btn ghost small', onclick: (e) => {
          e.stopPropagation();
          editExclusion('trial', row.trial_key, 'exclude', {current: ex});
        }}, 'Exclude trial…'),
      run && row.run_key && h('button', {type: 'button', class: 'btn ghost small', onclick: (e) => {
        e.stopPropagation();
        editExclusion('run', row.run_key, 'exclude');
      }}, 'Exclude run…'));
  }

  // REPORT sub-tab ----------------------------------------------------------

  function renderReportPane() {
    const box = $('#rp-pane-report');
    if (!box) return;
    const r = rp.report;
    const err = rp.reportError;
    if (!changed('report:main', [r, rp.coverage && !rp.coverage.error ? rp.coverage : null, err && err.message, rp.status && rp.status.trials])) return;
    if (!r) {
      fill(box, err ? errorPanel(err, () => loadReport()) : panel({code: 'R-02', title: 'Multiples vs baseline'}, h('p', {class: 'muted'}, 'Loading the report…')));
      return;
    }
    fill(box,
      err && h('div', {class: `banner ${is503(err) ? 'warn' : 'bad'}`}, `${err.status === 400 ? 'These filters or options are not valid' : 'The report could not refresh'}: ${err.message}. Showing the last result.`,
        err.status === 400 && h('button', {type: 'button', class: 'btn ghost small', onclick: resetRail}, 'Reset to defaults')),
      (r.warnings || []).map((w) => h('div', {class: 'note-strip'}, tag('note', 'warn'), w)),
      r.totals.trials ? [
        headlinePanel(r),
        groupsPanel(r),
        coveragePanel(r, rp.coverage),
      ] : emptyReport());
  }

  function emptyReport() {
    const none = rp.status && rp.status.trials === 0;
    return h('section', {class: 'panel'}, h('div', {class: 'empty-state'},
      h('div', {class: 'big'}, none ? 'NOTHING YET' : 'NO MATCHING TRIALS'),
      h('p', null, none ? 'Nothing in the ledger yet. Rescan, or start a run.' : 'No trials in the ledger match these filters.'),
      none
        ? h('button', {type: 'button', class: 'btn primary', onclick: (e) => doRescan(e.currentTarget)}, 'Rescan')
        : h('button', {type: 'button', class: 'btn ghost', onclick: resetRail}, 'Reset filters')));
  }

  function errorPanel(err, retry) {
    return h('section', {class: 'panel'}, h('div', {class: 'empty-state'},
      h('div', {class: 'big'}, is503(err) ? 'LEDGER BUSY' : 'NO DATA'),
      h('p', {class: is503(err) ? 'warn-text' : 'bad-text'}, err.message),
      h('div', {class: 'actions'},
        err.status === 400 && h('button', {type: 'button', class: 'btn ghost', onclick: resetRail}, 'Reset filters'),
        retry && h('button', {type: 'button', class: 'btn ghost', onclick: retry}, 'Try again'))));
  }

  /**
   * Groups and comparisons per what the report holds fixed, in its order: per provider:model when harnesses are
   * compared, per harness (the group's harness_key) when models are. title names it.
   */
  function modelsOf(r) {
    const models = new Map();
    if (byModels(r)) {
      const groups = new Map((r.groups || []).map((g) => [g.key, g]));
      const get = (g) => {
        if (!models.has(g.harness_key)) {
          models.set(g.harness_key, {key: g.harness_key, title: g.harness_name || g.name, groups: [], comparisons: []});
        }
        return models.get(g.harness_key);
      };
      for (const g of r.groups || []) get(g).groups.push(g);
      for (const c of r.comparisons || []) if (groups.has(c.group)) get(groups.get(c.group)).comparisons.push(c);
      return [...models.values()].sort((a, b) => a.title.localeCompare(b.title));
    }
    const get = (provider, model) => {
      const key = `${provider}:${model}`;
      if (!models.has(key)) models.set(key, {key, title: key, provider, model, groups: [], comparisons: []});
      return models.get(key);
    };
    for (const g of r.groups || []) get(g.provider, g.model).groups.push(g);
    for (const c of r.comparisons || []) get(c.provider, c.model).comparisons.push(c);
    return [...models.values()];
  }

  /** How the report names its two sides: harness and baseline, or model and baseline model. */
  const sidesOf = (r = rp.report) => SIDES[byModels(r) ? 'models' : 'harnesses'];

  /** The baseline's name in headings: "Opus 5.5" when models are compared, "Claude Code" when harnesses are. */
  function baselineLabel(r) {
    return byModels(r) ? modelLabel(dig(r.spec, 'options.baseline.model'))
      : harnessLabel(dig(r.spec, 'options.baseline.family') || 'claude-code');
  }

  /** The Harnesses | Models switch above the multiples (the same option as Compare in Advanced). */
  function compareSwitch(r, readOnly) {
    const now = byModels(r) ? 'models' : 'harnesses';
    const text = byModels(r) ? `model ÷ ${baselineLabel(r)}, harness by harness`
      : `harness ÷ ${baselineLabel(r)}, model by model`;
    return h('div', {class: 'rp-compare-bar'},
      eyebrow('Compare'),
      readOnly ? tag(now === 'models' ? 'Models' : 'Harnesses')
        : h('div', {class: 'rp-seg', role: 'group', 'aria-label': 'Compare'}, COMPARE_CHOICES.map(([v, label]) => h(
          'button', {type: 'button', class: now === v ? 'active' : null, 'aria-pressed': String(now === v),
            onclick: () => {
              if (optValue('compare') === v) return;
              optSet('compare', v);
              edited();
              renderRail(true);
            }}, label))),
      h('span', {class: 'faint'}, text));
  }

  function headlinePanel(r, {readOnly = false} = {}) {
    const ci = Math.round((dig(r.spec, 'options.ci_level') || 0.9) * 100);
    const models = modelsOf(r);
    const tasks = scopeText(r.scope);
    const lead = byModels(r)
      ? `Multiple = model ÷ baseline model (${baselineLabel(r)}) on the same tasks with the same harness: 0.5× means `
        + 'the model used half as much, under 1× less than the baseline model, over 1× more. Published figures compare '
        + 'harnesses, so model comparisons get no verdict. '
      : 'Multiple = harness ÷ baseline on the same tasks and model: 0.5× means the harness used half as much, under 1× '
        + 'less than the baseline, over 1× more. ';
    return panel({code: 'R-02', title: byModels(r) ? 'Multiples vs baseline model' : 'Multiples vs baseline',
      meta: `${plural((r.comparisons || []).length, 'comparison')} · ${ci}% intervals`},
      compareSwitch(r, readOnly),
      h('p', {class: 'rp-explainer'}, `${lead}Intervals are ${ci}%. ${tasks ? `Tasks: ${tasks}. ` : ''}`
        + 'Pass rates on fewer than about 100 tasks cannot confirm a 5-point claim.'),
      models.map((m) => modelCard(m, r, {readOnly})));
  }

  function modelCard(m, r, {readOnly = false} = {}) {
    const groups = new Map((r.groups || []).map((g) => [g.key, g]));
    const base = m.groups.find((g) => g.is_baseline);
    const models = byModels(r);
    const baseFamily = models ? dig(r.spec, 'options.baseline.model')
      : dig(r.spec, 'options.baseline.family') || 'claude-code';
    const baseName = baselineLabel(r);
    const S = sidesOf(r);
    const comps = m.comparisons;
    const caveats = unique(comps.flatMap(caveatsOf));
    const ns = comps.map((c) => c.pairs.n_tasks);
    const lo = Math.min(...ns);
    const hi = Math.max(...ns);
    const noBaseline = !base || !(base.trials.valid_reward > 0 || base.trials.valid_cost > 0 || base.trials.valid_tokens > 0);
    const head = h('div', {class: 'rp-card-head'},
      h('h3', null, m.title),
      h('span', {class: 'rp-card-meta'},
        base ? `baseline ${models ? baseName : base.name}` : `no ${baseFamily} baseline`),
      comps.length > 0 && h('span', {class: 'rp-card-meta'}, lo === hi ? plural(lo, 'paired task') : `${lo}–${hi} paired tasks`),
      caveats.length > 0 && tag(`⚠ ${plural(caveats.length, 'caveat')}`, 'warn'));
    const noneText = models
      ? `No usable ${baseName} trials with ${m.title} yet. Multiples need the baseline model on the same tasks.`
      : `No usable ${baseName} trials on this model yet. Multiples need the baseline on the same tasks.`;
    if (!comps.length) {
      return h('article', {class: 'rp-card'}, head, h('p', {class: 'rp-card-empty'}, noBaseline ? noneText
        : models ? `Only ${baseName} has trials with ${m.title}, so there is nothing to compare yet.`
          : `Only ${base.name} has trials on this model, so there is nothing to compare yet.`));
    }
    return h('article', {class: 'rp-card'}, head,
      noBaseline && h('p', {class: 'rp-card-empty'}, noneText),
      h('div', {class: 'rp-plots'}, forest(comps, 'tokens_total', `Tokens vs ${baseName}`, S),
        forest(comps, 'cost', `Cost vs ${baseName}`, S)),
      passStrip(comps, S),
      plotLegend(baseName, models),
      h('div', {class: 'scroll'}, compTable(comps, groups, readOnly, baseName, models)),
      caveats.length > 0 && caveatList(comps, caveats));
  }

  function gridLines(ticks, x) {
    return ticks.map((v) => svg('line', {x1: at(x(v)), x2: at(x(v)), y1: 0, y2: 30, class: 'rp-grid'}));
  }

  /** An interval's whisker at height y; ends past the axis are pinned to its edge (the text beside says the real values). */
  function whisker(ci, x, y, cls) {
    if (!ci || !isNum(ci.lo) || !isNum(ci.hi)) return null;
    return svg('line', {x1: at(x(ci.lo)), x2: at(x(ci.hi)), y1: y, y2: y, class: cls});
  }

  const diamond = (p, y, cls) => svg('svg', {x: at(p), y, width: 1, height: 1, overflow: 'visible'},
    svg('path', {d: 'M0 -5L5 0L0 5L-5 0Z', class: cls}));

  function forest(comps, metric, title, S = sidesOf()) {
    return h('div', {class: 'rp-forest'},
      h('div', {class: 'rp-forest-head'}, eyebrow(title),
        h('span', {class: 'faint'}, `${S.ratio}, log scale; left of 1× is less`)),
      comps.map((c) => forestRow(c, metric, S)),
      axisRow(TICKS.map((v) => [xLog(v), tickLabel(v), v === 1])));
  }

  /** A comparison's candidate as rows and plots label it: the harness, or the model ("Sonnet 5.5"). */
  const armLabel = (c) => (c.compare === 'models' ? modelLabel(c.arm_name) : c.arm_name);
  /** The server's headline and caveats name a model comparison's sides by id; show them as the cards do. */
  function modelProse(c, text) {
    if (c.compare !== 'models' || !text) return text;
    return [c.arm_name, c.baseline_name].filter(Boolean).sort((a, b) => b.length - a.length).reduce((out, id) => {
      const literal = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      return out.replace(new RegExp(`(?<![\\w.:/-])${literal}(?![\\w.-])`, 'g'), modelLabel(id));
    }, text);
  }
  const caveatsOf = (c) => (c.caveats || []).map((text) => modelProse(c, text));
  /** "Sonnet 5.5 vs Opus 5.5 with pi", or "terminus-2 vs claude-code". */
  const pairTitle = (c) => (c.compare === 'models'
    ? `${modelLabel(c.arm_name)} vs ${modelLabel(c.baseline_name)} with ${c.harness}`
    : `${c.arm_name} vs ${c.baseline_name}`);

  function forestRow(c, metric, S = sidesOf()) {
    const m = c.metrics ? c.metrics[metric] : null;
    const label = h('div', {class: 'rp-who', title: `${c.arm_name} vs ${c.baseline_name}`},
      h('span', null, armLabel(c)), m && m.status === 'low_n' && tag('low n'));
    if (!m || m.status === 'not_measured') {
      return h('div', {class: 'rp-frow'}, label,
        h('div', {class: 'rp-track-note'}, m ? notMeasuredText(m) : 'tokens not reported by this harness'),
        h('div', {class: 'rp-val faint'}, m ? `n=${m.n}` : DASH));
    }
    if (!m.geomean || m.status === 'insufficient') {
      return h('div', {class: 'rp-frow'}, label,
        h('div', {class: 'rp-track-note'}, `needs ${plural(m.need, 'more paired task')}`),
        h('div', {class: 'rp-val faint'}, `n=${m.n}`));
    }
    const g = m.geomean;
    const rom = m.ratio_of_means;
    const ref = m.reference;
    const band = ref && Array.isArray(ref.band) && ref.band.every(isNum) ? ref.band : null;
    const track = svg('svg', {class: 'rp-track', 'aria-hidden': 'true'},
      band && svg('rect', {x: at(xLog(band[0])), width: at(xLog(band[1]) - xLog(band[0])), y: 2, height: 26, class: 'rp-band'}),
      gridLines(TICKS, xLog),
      svg('line', {x1: at(xLog(1)), x2: at(xLog(1)), y1: 0, y2: 30, class: 'rp-parity'}),
      whisker(rom, xLog, 21, 'rp-rom-ci'),
      whisker(g, xLog, 10, 'rp-gm-ci'),
      rom && isNum(rom.estimate) && diamond(xLog(rom.estimate), 21, 'rp-rom'),
      svg('circle', {cx: at(xLog(g.estimate)), cy: 10, r: 4.5, class: 'rp-gm'}));
    const rows = [
      [`${times(g.estimate)} ${interval(g.lo, g.hi)}`, 'geometric mean'],
      rom && [`${times(rom.estimate)} ${interval(rom.lo, rom.hi)}`.trim(), 'ratio of means'],
      [count(m.n), 'paired tasks'],
      ref && [publishedText(ref), 'published'],
      ref && [verdictText(ref), 'verdict'],
      ref && ref.published && ref.published.source,
      m.status === 'low_n' && `Low n: ${plural(m.need, 'more pair')} for ${S.goal}.`,
    ].filter(Boolean);
    return h('div', {class: `rp-frow${m.status === 'low_n' ? ' rp-lown' : ''}`}, label,
      tip(h('div', {class: 'rp-track-wrap', role: 'img', 'aria-label': `${armLabel(c)}: ${metricText(m)}`}, track),
        `${armLabel(c)} · ${metric === 'cost' ? 'cost' : 'tokens'} vs `
          + `${c.compare === 'models' ? modelLabel(c.baseline_name) : c.baseline_name}`, rows, true),
      h('div', {class: 'rp-val'}, h('b', null, times(g.estimate)), ' ', h('span', {class: 'rp-ci'}, interval(g.lo, g.hi)),
        rom && isNum(rom.estimate) && h('small', null, `mean ratio ${times(rom.estimate)}`)));
  }

  function axisRow(ticks) {
    return h('div', {class: 'rp-frow rp-faxis', 'aria-hidden': 'true'}, h('span'),
      svg('svg', {class: 'rp-axis'}, ticks.map(([p, label, strong], i) => svg('text', {
        x: at(p), y: 11, 'text-anchor': i === 0 ? 'start' : i === ticks.length - 1 ? 'end' : 'middle', class: strong ? 'rp-strong' : null,
      }, label))),
      h('span'));
  }

  function passStrip(comps, S = sidesOf()) {
    return h('div', {class: 'rp-forest rp-pass'},
      h('div', {class: 'rp-forest-head'}, eyebrow('Δ pass rate'),
        h('span', {class: 'faint'}, `${S.diff}, in points, on paired tasks`)),
      comps.map((c) => passRow(c, S)),
      axisRow(PASS_TICKS.map((v) => [xPass(v), v === 0 ? '0' : `${signed(v, 0)}`, v === 0])));
  }

  function passRow(c, S = sidesOf()) {
    const p = c.pass;
    const d = p && p.delta_pp;
    const label = h('div', {class: 'rp-who'}, h('span', null, armLabel(c)), p && p.status === 'low_n' && tag('low n'));
    const mde = p && isNum(p.mde_pp) ? `can detect ±${Math.round(p.mde_pp)} pts` : '';
    if (!p || !d || !isNum(d.estimate)) {
      return h('div', {class: 'rp-frow'}, label,
        h('div', {class: 'rp-track-note'}, p && p.status === 'insufficient' ? `needs ${plural(p.need, 'more paired task')}` : 'no paired pass rates yet'),
        h('div', {class: 'rp-val'}, h('small', null, mde || DASH)));
    }
    const ref = p.reference;
    const band = ref && Array.isArray(ref.band) && ref.band.every(isNum) ? ref.band : null;
    const track = svg('svg', {class: 'rp-track', 'aria-hidden': 'true'},
      band && svg('rect', {x: at(xPass(band[0])), width: at(xPass(band[1]) - xPass(band[0])), y: 2, height: 26, class: 'rp-band'}),
      gridLines(PASS_TICKS, xPass),
      svg('line', {x1: at(xPass(0)), x2: at(xPass(0)), y1: 0, y2: 30, class: 'rp-parity'}),
      whisker(d, xPass, 15, 'rp-gm-ci'),
      svg('circle', {cx: at(xPass(d.estimate)), cy: 15, r: 4.5, class: 'rp-gm'}));
    const rows = [
      [`${pts(d.estimate)} ${ppInterval(d.lo, d.hi)}`, 'Δ pass'],
      [`${pct(p.arm_rate)} vs ${pct(p.baseline_rate)}`, `${S.arm} vs ${S.base}`],
      [count(p.n), 'paired tasks'],
      p.discordant && [`${p.discordant.arm_only} / ${p.discordant.baseline_only}`,
        `only ${S.arm} / only ${S.base} passed`],
      isNum(p.mcnemar_p) && [p.mcnemar_p.toFixed(3), 'McNemar p'],
      mde && [mde, 'resolution'],
      ref && [passPublishedText(ref), 'published'],
      ref && [verdictText(ref), 'verdict'],
      ref && ref.published && ref.published.source,
    ].filter(Boolean);
    return h('div', {class: `rp-frow${p.status === 'low_n' ? ' rp-lown' : ''}`}, label,
      tip(h('div', {class: 'rp-track-wrap', role: 'img', 'aria-label': `${armLabel(c)}: ${pts(d.estimate)}`}, track),
        `${armLabel(c)} · Δ pass`, rows, true),
      h('div', {class: 'rp-val'}, h('b', null, pts(d.estimate)), ' ', h('span', {class: 'rp-ci'}, ppInterval(d.lo, d.hi)), mde && h('small', null, mde)));
  }

  function plotLegend(baseName, models = false) {
    const sw = (...marks) => svg('svg', {class: 'rp-sw', viewBox: '0 0 24 14', width: 24, height: 14, 'aria-hidden': 'true'}, marks);
    return h('div', {class: 'legend rp-legend'},
      h('span', null, sw(svg('line', {x1: 1, x2: 23, y1: 7, y2: 7, class: 'rp-gm-ci'}), svg('circle', {cx: 12, cy: 7, r: 4, class: 'rp-gm'})), 'geometric mean with its interval'),
      h('span', null, sw(svg('line', {x1: 1, x2: 23, y1: 7, y2: 7, class: 'rp-rom-ci'}),
        svg('path', {d: 'M12 2L17 7L12 12L7 7Z', class: 'rp-rom'})),
        models ? 'ratio of means' : 'ratio of means (used for verdicts)'),
      !models && h('span', null, sw(svg('rect', {x: 1, y: 1, width: 22, height: 12, class: 'rp-band'})),
        'published band'),
      h('span', null, sw(svg('line', {x1: 12, x2: 12, y1: 0, y2: 14, class: 'rp-parity'})), `1× = same as ${baseName}`),
      h('span', null, tag('low n'), models ? 'too few pairs for a settled figure' : 'too few pairs for a verdict'));
  }

  function metricCell(m) {
    if (!m) return tdNum(DASH);
    if (m.status === 'not_measured') return h('td', {class: 'num faint'}, 'not reported');
    if (!m.geomean) return h('td', {class: 'num faint'}, `needs ${m.need} more`);
    const g = m.geomean;
    const rom = m.ratio_of_means;
    return h('td', {class: `num${m.status === 'low_n' ? ' rp-lown' : ''}`},
      `${times(g.estimate)} ${interval(g.lo, g.hi)}`,
      rom && isNum(rom.estimate) && h('small', {class: 'sub rp-rom-text'}, h('span', null, `mean ratio ${times(rom.estimate)}`), ' ', h('span', null, interval(rom.lo, rom.hi))),
      m.status === 'low_n' && h('small', {class: 'sub'}, 'low n'));
  }

  function passCell(p) {
    if (!p) return DASH;
    const d = p.delta_pp || {};
    return [
      h('span', {class: 'mono'}, isNum(p.arm_rate) || isNum(p.baseline_rate) ? `${pct(p.arm_rate)} vs ${pct(p.baseline_rate)}` : DASH),
      h('small', {class: 'sub'}, isNum(d.estimate) ? `Δ ${pts(d.estimate)} ${ppInterval(d.lo, d.hi)}` : p.status === 'insufficient' ? `needs ${p.need} more` : DASH),
      p.reference && h('div', {class: 'rp-pass-verdict'}, verdictChip(p.reference)),
    ];
  }

  function compTable(comps, groups, readOnly, baseName, models = false) {
    const S = SIDES[models ? 'models' : 'harnesses'];
    return h('table', {class: 'grid-t compact rp-cmp'},
      thead(S.Arm, thNum('Pairs'), thNum(`Tokens vs ${baseName}`), thNum(`Cost vs ${baseName}`), !models && 'Published',
        !models && 'Verdict', `Pass, ${S.arm} vs ${S.base}`, thNum('Setup failures')),
      comps.map((c) => {
        const open = readOnly ? null : () => openPairs(c.key);
        const arm = groups.get(c.group);
        const sf = arm && arm.setup_failure_rate;
        const refs = [['Tokens', c.metrics.tokens_total], ['Cost', c.metrics.cost]]
          .filter(([, m]) => m && m.reference).map(([name, m]) => [name, m.reference]);
        const both = refs.length > 1;
        return h('tbody', {class: readOnly ? null : 'rp-click', onclick: open},
          h('tr', null,
            h('td', null,
              readOnly
                ? h('b', null, armLabel(c))
                : h('button', {type: 'button', class: 'rp-link', title: 'Open the pairs', onclick: (e) => {
                  e.stopPropagation();
                  open();
                }}, armLabel(c), h('span', {class: 'arrow'}, ' ›')),
              h('span', {class: 'sub mono'}, models ? `${c.provider}:${c.model}` : `vs ${c.baseline_name}`),
              (c.caveats || []).length > 0 && tip(h('span', {class: 'rp-cav'},
                `⚠ ${plural(c.caveats.length, 'caveat')}`), 'Caveats', caveatsOf(c), true)),
            tdNum(count(c.pairs.n_tasks)),
            metricCell(c.metrics.tokens_total),
            metricCell(c.metrics.cost),
            !models && h('td', null, refs.length
              ? refs.map(([name, ref]) => h('div', null, both && h('span', {class: 'faint'}, `${name} `),
                tip(h('span', {class: 'rp-pub'}, publishedText(ref)), `Published (${name.toLowerCase()})`,
                  [(ref.published && ref.published.source) || 'No source given', [ref.estimator ? statusLabel(ref.estimator) : DASH, 'compared with']], true)))
              : h('span', {class: 'faint'}, 'none published')),
            !models && h('td', null, refs.length
              ? refs.map(([name, ref]) => h('div', {class: 'rp-vrow'}, both && h('span', {class: 'faint'}, `${name} `), verdictChip(ref)))
              : DASH),
            h('td', null, passCell(c.pass)),
            tdNum(sf ? `${pct(sf.estimate)} ${interval(sf.lo, sf.hi, pct)}` : DASH, sf ? `${plural(sf.trials, 'trial')}, Wilson interval` : null)),
          h('tr', {class: 'rp-headline'}, h('td', {colspan: models ? 6 : 8}, modelProse(c, c.headline))));
      }));
  }

  function caveatList(comps, caveats) {
    return h('details', {class: 'rp-caveats'},
      h('summary', null, `Caveats (${caveats.length})`),
      h('ul', null, caveats.map((text) => {
        const who = comps.filter((c) => caveatsOf(c).includes(text)).map(armLabel);
        return h('li', null, text, who.length < comps.length && h('span', {class: 'faint'}, ` · ${who.join(', ')}`));
      })));
  }

  function groupsPanel(r) {
    return panel({code: 'R-03', title: 'Groups', meta: 'unpaired; task mix may differ', flush: true},
      h('div', {class: 'scroll'}, groupTable(r)));
  }

  function groupTable(r) {
    const models = modelsOf(r);
    const pair = (s, fmt) => (s ? `${fmt(s.mean)} · ${fmt(s.median)}` : DASH);
    return h('table', {class: 'grid-t compact rp-groups'},
      thead('Group', 'Outcomes', thNum('Pass rate'), thNum('Tokens mean · median'), thNum('Cost mean'),
        thNum('Reported ÷ computed'), thNum('Median min'), thNum('Setup failures'), 'Configs'),
      h('tbody', null, models.map((m) => [
        h('tr', {class: 'rp-block'}, h('th', {colspan: 9, scope: 'colgroup'}, m.title)),
        [...m.groups].sort((a, b) => Number(b.is_baseline) - Number(a.is_baseline)).map((g) => {
          const t = g.trials;
          const pt = g.per_trial || {};
          const rvc = g.reported_vs_computed_cost || {};
          const sf = g.setup_failure_rate;
          const versions = Object.entries(g.harness_versions || {}).map(([v, n]) => `${v} ×${n}`).join(', ');
          return h('tr', {class: g.is_baseline ? 'baseline' : null},
            h('td', null, h('b', null, byModels(r) ? `${g.provider}:${g.model}` : g.name),
              g.is_baseline && [' ', tag('baseline', 'ok')],
              g.implementation && g.implementation !== 'builtin' && h('span', {class: 'sub mono rp-impl'}, g.implementation),
              g.implementation === null && h('span', {class: 'sub'}, 'mixed implementations'),
              versions && h('span', {class: 'sub mono'}, `version ${versions}`)),
            h('td', null, outcomeCounts(t.by_outcome), h('small', {class: 'sub'}, `${count(t.total)} trials · ${count(t.included)} counted`)),
            h('td', {class: 'num'}, g.pass_rate ? [`${pct(g.pass_rate.estimate)} ${interval(g.pass_rate.lo, g.pass_rate.hi, pct)}`, h('small', {class: 'sub'}, plural(g.pass_rate.tasks, 'task'))] : DASH),
            h('td', {class: 'num'}, pair(pt.tokens_total, tokens),
              h('small', {class: 'sub'}, `uncached ${pair(pt.tokens_uncached, tokens)}`),
              pt.tokens_total && h('small', {class: 'sub'}, `n=${count(pt.tokens_total.n)}`)),
            h('td', {class: 'num rp-costcell'},
              pt.cost ? money(pt.cost.mean) : DASH,
              pt.cost && h('small', {class: 'sub'}, `median ${money(pt.cost.median)}`),
              pt.cost && h('small', {class: 'sub'}, `${count(pt.cost.n)} of ${count(t.valid_reward)} measured`),
              t.usage_unmeasured > 0 && h('small', {class: 'sub warn-text'}, `${plural(t.usage_unmeasured, 'trial')} without usage, left out`),
              t.usage_lower_bound > 0 && h('small', {class: 'sub warn-text'}, `${count(t.usage_lower_bound)} lower bound${t.usage_lower_bound === 1 ? '' : 's'} in the mean`)),
            h('td', {class: 'num'}, isNum(rvc.median_ratio) ? [times(rvc.median_ratio), h('small', {class: 'sub'}, `n=${count(rvc.n)}`)] : DASH),
            h('td', {class: 'num'}, pt.minutes ? minutes(pt.minutes.median) : DASH,
              pt.minutes_ex_setup && isNum(pt.minutes_ex_setup.median)
                && h('small', {class: 'sub'}, `${minutes(pt.minutes_ex_setup.median)} ex setup`),
              setupModesText(g.setup_modes) && h('small', {class: 'sub'}, setupModesText(g.setup_modes))),
            tdNum(sf ? `${pct(sf.estimate)} ${interval(sf.lo, sf.hi, pct)}` : DASH, sf ? `${plural(sf.trials, 'trial')}` : null),
            h('td', null, configsCell(g)));
        }),
      ])));
  }

  /** "10 pre-baked · 3 stock": how a group's trials got their harness; nothing when every one ran stock. */
  function setupModesText(modes) {
    const entries = Object.entries(modes || {}).filter(([, n]) => n > 0);
    if (!entries.length || (entries.length === 1 && entries[0][0] === 'stock')) return '';
    return entries.map(([m, n]) => `${n} ${m === 'unknown' ? 'not recorded' : setupLabel(m)}`).join(' · ');
  }

  function configsCell(g) {
    const configs = g.configs || [];
    const diff = Object.entries(g.config_diff || {});
    const diffText = diff.map(([k, vals]) => `${k}: ${vals.map((v) => (v === null ? 'unset' : shown(v))).join(' / ')}`);
    if (configs.length <= 1) {
      const c = configs[0];
      return c ? tip(h('span', {class: 'mono faint'}, `1 config #${c.config_hash.slice(0, 8)}`), 'Configuration', [c.label, [c.config_hash, 'hash'], [count(c.trials), 'trials']], true) : DASH;
    }
    const rows = [...configs.map((c) => [`#${c.config_hash.slice(0, 8)} · ${plural(c.trials, 'trial')}`, c.label.split(' · ').slice(2).join(' · ') || c.label]), ...diffText];
    return h('details', {class: 'rp-pooled'},
      tip(h('summary', null, `${configs.length} configs pooled`), `${g.name}: configs pooled`, rows),
      h('ul', null, configs.map((c) => h('li', null, h('span', {class: 'mono'}, `#${c.config_hash.slice(0, 8)}`), ' ', c.label, ' ', h('span', {class: 'faint'}, plural(c.trials, 'trial'))))),
      diffText.length > 0 && h('p', {class: 'warn-text'}, `Differ in ${diffText.join('; ')}`));
  }

  function coveragePanel(r, cov) {
    const models = modelsOf(r);
    const covOk = cov && !cov.error;
    const covBy = new Map((covOk ? cov.comparisons || [] : []).map((c) => [c.key, c.coverage]));
    const order = (r.coverage && r.coverage.task_order) || [];
    const passDelta = dig(r.spec, 'options.coverage_target.pass_delta_pp') || 5;
    return panel({code: 'R-04', title: 'Coverage', meta: 'what exists, what to run next'},
      h('div', {class: 'legend rp-cov-legend'},
        h('span', null, h('span', {class: 'rp-cell rp-has', style: '--rate:1'}, '2'), 'valid trials, shaded by share passed'),
        h('span', null, h('span', {class: 'rp-cell rp-has', style: '--rate:0'}, '1'), 'none passed'),
        h('span', null, h('span', {class: 'rp-cell rp-gap'}), 'missing side of a pair'),
        CORNER.map((o) => h('span', null, h('b', {class: 'rp-corner-key'}, glyphOf(o)), labelOf(o)))),
      models.map((m) => coverageModel(m, r, order, covBy, passDelta)),
      cov && cov.error && h('p', {class: 'warn-text'}, `The coverage request failed: ${cov.error.message}`),
      covOk && (cov.untouched_tasks || []).length > 0 && h('p', {class: 'muted rp-untouched'},
        `${plural(cov.untouched_tasks.length, 'task')} of the sample ${cov.untouched_tasks.length === 1 ? 'has' : 'have'} no trials under these filters: `,
        h('span', {class: 'mono'}, cov.untouched_tasks.join(', '))));
  }

  function coverageModel(m, r, order, covBy, passDelta) {
    const cols = [...m.groups].sort((a, b) => Number(b.is_baseline) - Number(a.is_baseline));
    const cells = (r.coverage && r.coverage.cells) || {};
    const base = cols.find((g) => g.is_baseline);
    const has = (g, task) => {
      const c = (cells[g.key] || {})[task];
      return Boolean(c && c.valid > 0);
    };
    const seen = unique(cols.flatMap((g) => Object.keys(cells[g.key] || {})));
    const extra = seen.filter((t) => !order.includes(t)).sort();
    const rowsFor = (tasks, label) => (tasks.length ? [
      h('tr', {class: 'rp-block'}, h('th', {colspan: cols.length + 1, scope: 'colgroup'}, label)),
      tasks.map((task) => h('tr', null, h('td', {class: 'task'}, task), cols.map((g) => covCell(g, task, cells, base, cols, has)))),
    ] : []);
    return h('div', {class: 'rp-cov-model'},
      h('h3', {class: 'rp-cov-title'}, m.title),
      h('div', {class: 'scroll'}, h('table', {class: 'matrix rp-cov'},
        h('thead', null, h('tr', null, h('th', null, 'Task'), cols.map((g) => {
          const name = byModels(r) ? modelLabel(`${g.provider}:${g.model}`) : g.name;
          return h('th', {title: g.label}, g.is_baseline ? `${name} ◆` : name);
        }))),
        h('tbody', null,
          rowsFor(order.slice(0, 10), 'Block 1'),
          rowsFor(order.slice(10, 20), 'Block 2'),
          rowsFor(order.slice(20), 'Rest of the sample'),
          rowsFor(extra, 'Other tasks seen')))),
      m.comparisons.map((c) => coverageLine(c, covBy.get(c.key) || c.coverage, m, passDelta)));
  }

  function covCell(g, task, cells, base, cols, has) {
    const c = (cells[g.key] || {})[task];
    const valid = c ? c.valid : 0;
    const counterpart = g.is_baseline ? cols.some((o) => !o.is_baseline && has(o, task)) : Boolean(base && has(base, task));
    const missing = !valid && counterpart;
    const corners = c ? CORNER.filter((o) => c.by_outcome && c.by_outcome[o] > 0) : [];
    const rate = valid ? c.passed / valid : null;
    const box = h('span', {class: `rp-cell${valid ? ' rp-has' : ''}${missing ? ' rp-gap' : ''}${rate !== null && rate > 0.5 ? ' rp-hi' : ''}`, style: valid ? `--rate:${rate.toFixed(3)}` : null},
      valid ? String(valid) : c ? '0' : '',
      corners.length > 0 && h('i', {class: 'rp-corner'}, corners.map(glyphOf).join('')));
    if (!c && !missing) return h('td', null, box);
    const rows = [
      [count(valid), 'valid trials'],
      c && [count(c.passed), 'passed'],
      ...(c ? outcomeEntries(c.by_outcome).map(([o, n]) => [count(n), `${glyphOf(o)} ${labelOf(o)}`]) : []),
      missing && (byModels() ? (g.is_baseline ? 'Another model ran this task; the baseline model has no valid trial.'
        : 'The baseline model ran this task; this model has no valid trial.')
        : g.is_baseline ? 'The harness ran this task; the baseline has no valid trial.'
          : 'The baseline ran this task; this harness has no valid trial.'),
    ].filter(Boolean);
    return h('td', null, tip(box, `${g.name} · ${task}`, rows, true));
  }

  function coverageLine(c, cv, m, passDelta) {
    if (!cv) return null;
    const metric = {cost: 'Cost', tokens_total: 'Tokens', pass: 'Pass rate'}[cv.metric] || statusLabel(cv.metric);
    const parts = [];
    if (cv.metric !== 'pass') {
      parts.push(isNum(cv.halfwidth_factor_now)
        ? `${metric} CI now ×/÷${multiple(cv.halfwidth_factor_now)}.`
        : `${metric} CI: no estimate yet with ${plural(cv.pairs, 'paired task')}.`);
      if (isNum(cv.est_pairs_needed)) {
        parts.push(`A ×/÷${multiple(cv.target_factor)} target needs about ${count(cv.est_pairs_needed)} paired tasks (${count(cv.est_more_pairs)} more)${isNum(cv.est_cost_usd) ? `, about ${money(cv.est_cost_usd)}` : ''}.`);
      }
    }
    if (isNum(cv.pass_tasks_needed)) parts.push(`±${passDelta} pts on pass rate needs about ${count(cv.pass_tasks_needed)} tasks.`);
    const next = cv.next || [];
    const where = c.compare === 'models' ? `with ${m.title}` : `on ${m.key}`;
    const nextList = () => [`# ${c.arm_name} vs ${c.baseline_name} ${where}: next tasks to run (task, side, reason)`,
      ...next.map((n) => `${n.task}\t${n.side}\t${n.reason}`)].join('\n');
    return h('div', {class: 'rp-covline'},
      h('div', {class: 'rp-covhead'}, h('b', null, pairTitle(c)),
        h('span', {class: 'faint'}, `${plural(cv.pairs, 'pair')} · `
          + `${count(cv.need_for_estimate)} more for an estimate · ${count(cv.need_for_verdict)} more for `
          + `${c.compare === 'models' ? 'a settled figure' : 'a verdict'}`
          + `${cv.sd_source === 'prior' ? ' · spread from the prior' : ''}`)),
      h('p', null, parts.join(' ')),
      cv.note && h('p', {class: 'muted'}, cv.note),
      next.length > 0 && h('div', {class: 'rp-next'},
        h('span', {class: 'eyebrow'}, `Next ${next.length}`),
        h('span', {class: 'chips'}, next.slice(0, 10).map((n) => tip(h('span', {class: 'chip'}, n.task, h('span', {class: 'src'}, n.side === 'both' ? 'both' : n.side)),
          n.task, [[n.side === 'both' ? 'both sides' : `${n.side} side`, 'run'], [n.reason, 'why']]))),
        next.length > 10 && h('small', {class: 'faint'}, `and ${next.length - 10} more`),
        h('button', {type: 'button', class: 'btn ghost small', onclick: () => copyText(nextList(), `Copied ${plural(next.length, 'task')}`)}, 'Copy next-run list')));
  }

  // Pairs drawer ------------------------------------------------------------

  const comparisonOf = (key) => (rp.report ? (rp.report.comparisons || []).find((c) => c.key === key) || null : null);
  /** Whether the open pairs drawer's comparison is between models. */
  const comparisonModels = () => Boolean(rp.side && rp.side.comp && rp.side.comp.compare === 'models');

  function openPairs(key) {
    const c = comparisonOf(key);
    rp.side = {kind: 'pairs', key, data: null, error: null};
    showSide('R-05', c ? pairTitle(c) : 'Pairs', [h('p', {class: 'muted'}, 'Loading the pairs…')]);
    loadPairs();
  }

  /** The ledger rows behind a pairs answer, for their outcomes and the exclusion buttons. */
  async function pairTrials(pairs, c) {
    const keys = new Set();
    for (const p of pairs.pairs) for (const k of [...p.arm.trial_keys, ...p.baseline.trial_keys]) keys.add(k);
    for (const e of pairs.excluded_trials) keys.add(e.trial_key);
    const tasks = unique([...pairs.pairs.map((p) => p.task), ...pairs.excluded_trials.map((e) => e.task)]);
    if (!tasks.length || !c) return [];
    const models = unique([`${c.provider}:${c.model}`,
      `${c.baseline_provider || c.provider}:${c.baseline_model || c.model}`]);
    const filters = {...railFilters(), models, tasks: tasks.slice(0, 500)};
    delete filters.runs;          // a trial stays listed after its run is excluded or filtered
    delete filters.exclude_runs;
    const res = await api('POST', '/api/ledger/trials', {filters, limit: 1000});
    return res.rows.filter((row) => keys.has(row.trial_key));
  }

  async function loadPairs() {
    const side = rp.side;
    if (!side || side.kind !== 'pairs') return;
    const body = {...requestBody(), comparison: side.key};
    const c = comparisonOf(side.key);
    try {
      const out = await dimWhile([$('#rp-side-body')], fresh.side((async () => {
        const pairs = await api('POST', '/api/report/pairs', body);
        return {pairs, trials: await pairTrials(pairs, c)};
      })()));
      if (out === STALE || rp.side !== side) return;
      side.data = out;
      side.comp = c;
      side.error = null;
    } catch (err) {
      if (rp.side !== side) return;
      if (err.status !== 404 || !side.data) side.error = noteError(err);
    }
    renderPairs();
  }

  /** Only the trial rows of the pairs already shown (their comparison has left the report). */
  async function loadPairTrials(side) {
    try {
      const trials = await dimWhile([$('#rp-side-body')], fresh.side(pairTrials(side.data.pairs, side.comp)));
      if (trials === STALE || rp.side !== side) return;
      side.data = {...side.data, trials};
    } catch (err) {
      noteError(err);
    }
    renderPairs();
  }

  function renderPairs() {
    const side = rp.side;
    const box = $('#rp-side-body');
    if (!side || side.kind !== 'pairs' || !box) return;
    const current = comparisonOf(side.key);
    const c = current || side.comp;
    if (side.error && !side.data) {
      fill(box, h('p', {class: side.error.status === 404 ? 'muted' : 'bad-text'}, side.error.status === 404
        ? 'This comparison is not in the report under the current filters.' : `Could not load the pairs: ${side.error.message}`));
      return;
    }
    if (!side.data || !c) return;
    const {pairs, trials} = side.data;
    const models = c.compare === 'models';
    const S = SIDES[models ? 'models' : 'harnesses'];
    const sideOf = new Map();
    for (const p of pairs.pairs) {
      for (const k of p.arm.trial_keys) sideOf.set(k, S.arm);
      for (const k of p.baseline.trial_keys) sideOf.set(k, 'baseline');
    }
    for (const e of pairs.excluded_trials) {
      if (!sideOf.has(e.trial_key)) sideOf.set(e.trial_key, e.side === 'arm' ? S.arm : 'baseline');
    }
    const d = c.pairs.dropped || {};
    const dropped = Object.entries(d).filter(([, n]) => n > 0).map(([k, n]) => `${n} ${statusLabel(k)}`);
    $('#rp-side-title').textContent = pairTitle(c);
    fill(box,
      !current && h('div', {class: 'banner warn'}, 'This comparison is no longer in the report under the current filters, for example because its last usable trial was excluded. The pairs below are as they were; the trial rows and their buttons are current.'),
      side.error && h('p', {class: 'bad-text'}, `Could not refresh the pairs: ${side.error.message}`),
      h('p', {class: 'rp-side-lead'}, modelProse(c, c.headline)),
      params([
        models && ['Harness', c.harness || DASH],
        ['Model', h('span', {class: 'mono'}, `${c.provider}:${c.model}`)],
        models && ['Baseline model', h('span', {class: 'mono'}, `${c.baseline_provider}:${c.baseline_model}`)],
        ['Paired tasks', count(c.pairs.n_tasks)],
        ['Same run', isNum(c.pairs.same_run_fraction) ? pct(c.pairs.same_run_fraction) : DASH],
        [`${S.Arm} only`, c.pairs.arm_only_tasks.length ? plural(c.pairs.arm_only_tasks.length, 'task') : 'none'],
        ['Baseline only', c.pairs.baseline_only_tasks.length ? plural(c.pairs.baseline_only_tasks.length, 'task') : 'none'],
        ['Dropped', dropped.length ? dropped.join(', ') : 'none'],
      ].filter(Boolean)),
      sideSection('Per-task ratios', `${S.ratio} on each paired task, log scale; left of 1× is less`,
        ratioStrip(pairs.pairs, c, S)),
      sideSection('Pairs',
        `${plural(pairs.pairs.length, 'task')}${models ? ' · M is the model, B the baseline model' : ''}`,
        pairsTable(pairs.pairs, models ? 'M' : 'H')),
      sideSection('Sensitivity', `the token and cost multiple (${S.ratio}) under other choices`,
        sensitivityTable(c, S)),
      sideSection('Trials', 'every trial behind these pairs; exclude a trial or its run, with a reason', trialsTable(trials, sideOf)),
      sideSection('Left out', 'trials of these two groups the report could not use', excludedTable(pairs.excluded_trials, trials)),
      (c.caveats || []).length > 0 && sideSection('Caveats', '',
        h('ul', {class: 'rp-list'}, caveatsOf(c).map((t) => h('li', null, t)))));
  }

  function sideSection(title, sub, ...body) {
    return h('section', {class: 'rp-side-sec'},
      h('div', {class: 'rp-side-h'}, h('h3', null, title), sub && h('span', {class: 'faint'}, sub)),
      body);
  }

  function ratioStrip(pairs, c, S = SIDES.harnesses) {
    const rows = [['Tokens', 'ratio_tokens', c.metrics.tokens_total], ['Cost', 'ratio_cost', c.metrics.cost]];
    if (!pairs.length) return h('p', {class: 'muted'}, 'No paired tasks yet.');
    return h('div', {class: 'rp-forest rp-ratios'},
      rows.map(([label, key, m]) => {
        const marks = pairs.filter((p) => isNum(p[key]) && p[key] > 0).map((p) => tip(svg('g', {class: 'rp-dot'},
          svg('circle', {cx: at(xLog(p[key])), cy: 15, r: 11, class: 'rp-hit'}),
          svg('circle', {cx: at(xLog(p[key])), cy: 15, r: 4, class: 'rp-gm'})),
        p.task, [[times(p[key]), `${label.toLowerCase()}, ${S.ratio}`],
          [key === 'ratio_tokens' ? tokens(p.baseline.tokens_total) : money(p.baseline.cost), `${S.base} mean`],
          [key === 'ratio_tokens' ? tokens(p.arm.tokens_total) : money(p.arm.cost), `${S.arm} mean`],
          [p.same_run ? 'yes' : 'no', 'same run']], true));
        const g = m && m.geomean;
        return h('div', {class: 'rp-frow'},
          h('div', {class: 'rp-who'}, label),
          h('div', {class: 'rp-track-wrap'}, svg('svg', {class: 'rp-track'},
            gridLines(TICKS, xLog),
            svg('line', {x1: at(xLog(1)), x2: at(xLog(1)), y1: 0, y2: 30, class: 'rp-parity'}),
            g && isNum(g.estimate) && svg('line', {x1: at(xLog(g.estimate)), x2: at(xLog(g.estimate)), y1: 3, y2: 27, class: 'rp-gm-line'}),
            marks)),
          h('div', {class: 'rp-val'}, marks.length ? `${marks.length} tasks` : 'none', g && h('small', null, `geomean ${times(g.estimate)}`)));
      }),
      axisRow(TICKS.map((v) => [xLog(v), tickLabel(v), v === 1])));
  }

  /** The pairs; arm is the candidate side's letter in the heads: H for a harness, M for a model. */
  function pairsTable(pairs, arm = 'H') {
    if (!pairs.length) return h('p', {class: 'muted'}, 'No task has a valid trial on both sides yet.');
    const passText = (s) => (isNum(s.pass) ? pct(s.pass) : DASH);
    return h('div', {class: 'scroll'}, h('table', {class: 'grid-t compact'},
      thead('Task', 'Digest', 'Same run', thNum(`Trials ${arm} / B`), thNum(`Valid ${arm} / B`), thNum(`Tokens ${arm}`),
        thNum('Tokens B'), thNum(`Cost ${arm}`), thNum('Cost B'), thNum(`Tokens ${arm} ÷ B`), thNum(`Cost ${arm} ÷ B`),
        thNum(`Pass ${arm} / B`)),
      h('tbody', null, pairs.map((p) => h('tr', null,
        h('td', {class: 'mono'}, p.task),
        h('td', {class: 'mono faint'}, p.digest_short || DASH),
        h('td', null, p.same_run ? h('span', {class: 'ok-text'}, '✓ yes') : h('span', {class: 'faint'}, '– no')),
        tdNum(`${p.arm.trials} / ${p.baseline.trials}`),
        tdNum(`${p.arm.valid} / ${p.baseline.valid}`),
        tdNum(tokens(p.arm.tokens_total)), tdNum(tokens(p.baseline.tokens_total)),
        tdNum(money(p.arm.cost)), tdNum(money(p.baseline.cost)),
        tdNum(times(p.ratio_tokens)), tdNum(times(p.ratio_cost)),
        tdNum(`${passText(p.arm)} / ${passText(p.baseline)}`))))));
  }

  function sensitivityTable(c, S = SIDES.harnesses) {
    const cols = [['Tokens', c.metrics.tokens_total], ['Cost', c.metrics.cost]];
    const val = (m, fn) => (m && m.status !== 'not_measured' && m.status !== 'insufficient' ? fn(m) : m ? metricText(m) : DASH);
    const row = (label, fn) => h('tr', null, h('td', null, label), cols.map(([, m]) => tdNum(val(m, fn))));
    return h('div', {class: 'scroll'}, h('table', {class: 'grid-t compact'},
      thead('', ...cols.map(([name]) => thNum(name))),
      h('tbody', null,
        row('Primary (geometric mean)', (m) => `${times(m.geomean.estimate)} ${interval(m.geomean.lo, m.geomean.hi)}`),
        row('Ratio of means', (m) => (m.ratio_of_means ? `${times(m.ratio_of_means.estimate)} ${interval(m.ratio_of_means.lo, m.ratio_of_means.hi)}` : DASH)),
        row('Crashes excluded', (m) => times(m.sensitivity && m.sensitivity.crashes_excluded)),
        row('Mispriced cap included', (m) => times(m.sensitivity && m.sensitivity.misconfigured_cap_included)),
        row('Same run only', (m) => times(m.sensitivity && m.sensitivity.same_run_only)),
        row('Leave one out', (m) => (m.leave_one_out ? `${times(m.leave_one_out[0])} – ${times(m.leave_one_out[1])}` : DASH)),
        row('Per task: min · median · max', (m) => (m.per_task ? `${multiple(m.per_task.min)} · ${multiple(m.per_task.median)} · ${times(m.per_task.max)}` : DASH)),
        row('Spread (geometric SD)', (m) => (isNum(m.geomean.gsd) ? `×/÷${multiple(m.geomean.gsd)}` : DASH)),
        row(`${S.Arm}'s unrecorded trials at 2× spend`,
          (m) => times(m.sensitivity && m.sensitivity.unrecorded_arm_high)),
        row(`${titleCase(S.base)}'s unrecorded trials at 2× spend`,
          (m) => times(m.sensitivity && m.sensitivity.unrecorded_baseline_high)),
        row('Robust', (m) => (m.robust === true ? '✓ yes' : m.robust === false ? '✕ no' : DASH)))));
  }

  function trialsTable(trials, sideOf) {
    if (!trials.length) return h('p', {class: 'muted'}, 'No ledger rows match these pairs under the current filters.');
    const order = (t) => `${sideOf.get(t.trial_key) === 'baseline' ? 1 : 0}|${t.task}|${t.trial_name}`;
    return h('div', {class: 'scroll'}, h('table', {class: 'grid-t compact'},
      thead('Side', 'Task', 'Trial', 'Outcome', thNum('Tokens'), thNum('Cost'), 'Run', ''),
      h('tbody', null, [...trials].sort((a, b) => order(a).localeCompare(order(b))).map((t) => {
        const excluded = t.exclusion && t.exclusion.action === 'exclude';
        // The row opens the trial on a click; for the keyboard the trial name is the button, so Enter on the row's
        // Exclude buttons never reaches a row handler.
        return h('tr', {class: 'rp-click', onclick: () => openTrial(t.id, {back: true})},
        h('td', null, sideOf.get(t.trial_key) || DASH),
        h('td', {class: 'mono'}, t.task),
        h('td', {class: 'mono'}, openButton(t, {back: true}, h('span', {class: excluded ? 'rp-struck' : null}, t.trial_name)), ' ', exclusionBadge(t.exclusion)),
        h('td', null, outcomeTag(t.outcome)),
        tdNum(usageValue(t.tokens_total, t.usage_status, tokens)),
        tdNum(usageValue(t.cost_usd, t.usage_status)),
        h('td', {class: 'mono'}, t.run_key),
        h('td', null, exclusionButtons(t)));
      }))));
  }

  function excludedTable(list, trials) {
    if (!list.length) return h('p', {class: 'muted'}, 'Nothing was left out.');
    const rowOf = new Map(trials.map((t) => [t.trial_key, t]));
    return h('div', {class: 'scroll'}, h('table', {class: 'grid-t compact'},
      thead('Side', 'Task', 'Trial', 'Outcome', 'Reason', ''),
      h('tbody', null, list.map((e) => {
        const row = rowOf.get(e.trial_key);
        const manual = /excluded by hand/.test(e.reason);
        return h('tr', null,
          h('td', null, e.side === 'arm' ? SIDES[comparisonModels() ? 'models' : 'harnesses'].arm : e.side),
          h('td', {class: 'mono'}, e.task),
          h('td', {class: 'mono'}, h('span', {class: manual ? 'rp-struck' : null}, trialName(e.trial_key))),
          h('td', null, outcomeTag(e.outcome)),
          h('td', {class: 'rp-wrap'}, e.reason),
          h('td', null, manual && h('button', {type: 'button', class: 'btn ghost small', onclick: () => editExclusion('trial', e.trial_key, 'include', {current: row && row.exclusion})}, 'Include again'),
            row && h('button', {type: 'button', class: 'btn ghost small', onclick: () => openTrial(row.id, {back: true})}, 'Details')));
      }))));
  }

  // Trial drawer -------------------------------------------------------------

  function openTrial(id, {back = false} = {}) {
    const from = back && rp.side && rp.side.kind === 'pairs' ? rp.side : null;
    rp.side = {kind: 'trial', id, back: from};
    showSide('R-06', 'Trial', [h('p', {class: 'muted'}, 'Loading the trial…')]);
    loadTrial();
  }

  async function loadTrial() {
    const side = rp.side;
    if (!side || side.kind !== 'trial') return;
    try {
      const t = await dimWhile([$('#rp-side-body')], fresh.side(api('GET', `/api/ledger/trials/${enc(side.id)}`)));
      if (t === STALE || rp.side !== side) return;
      side.data = t;
      renderTrial(t);
    } catch (err) {
      if (rp.side !== side) return;
      noteError(err);
      fill($('#rp-side-body'), h('p', {class: 'bad-text'}, `Could not load the trial: ${err.message}`));
    }
  }

  function renderTrial(t) {
    const side = rp.side;
    const box = $('#rp-side-body');
    const ph = t.phases || {};
    const usage = Object.entries(t.usage || {});
    // Harbor's totals minus the per-model sums: a shortfall was added in, a surplus kept.
    const gap = (u) => Object.values(u.unattributed || {});
    const logBox = h('pre', {class: 'log rp-log', hidden: true});
    const back = side.back;
    $('#rp-side-title').textContent = t.trial_name || 'Trial';
    fill(box,
      back && h('button', {type: 'button', class: 'btn ghost small rp-back', onclick: () => {
        rp.side = back;
        showSide('R-05', 'Pairs', []);
        renderPairs();
      }}, '‹ Back to the pairs'),
      h('div', {class: 'rp-trial-head'}, outcomeTag(t.outcome), usageTag(t.usage_status), exclusionBadge(t.exclusion),
        t.passed === true ? tag('reward 1', 'ok') : isNum(t.reward) ? tag(`reward ${Math.round(t.reward * 100) / 100}`) : null),
      h('p', {class: 'mono faint rp-key'}, t.trial_key),
      t.outcome_reason && h('p', null, t.outcome_reason),
      t.exception_type && [h('div', {class: 'cmd-name'}, t.exception_type), h('pre', {class: 'cmd'}, t.exception_message || t.exception_head || '')],
      h('div', {class: 'actions rp-trial-acts'},
        exclusionButtons(t),
        isMine(t) && t.run_id && h('button', {type: 'button', class: 'btn ghost small', onclick: () => {
          closeSide();
          openRun(t.run_id);
        }}, 'Open run'),
        t.log_url && h('button', {type: 'button', class: 'btn ghost small', onclick: (e) => busy(e.currentTarget, async () => {
          const text = await api('GET', t.log_url);
          logBox.hidden = false;
          logBox.textContent = typeof text === 'string' && text ? text : 'The log is empty.';
        })}, 'View log')),
      logBox,
      params([
        ['Run', h('span', {class: 'mono'}, t.run_key)],
        ['Harness', `${t.family} · ${t.arm}`],
        ['Implementation', t.implementation === 'builtin' ? 'built in' : h('span', {class: 'mono rp-impl'}, t.implementation)],
        ['Model', h('span', {class: 'mono'}, `${t.provider}:${t.model}`)],
        ['Effort', t.effort_label || t.effort || 'harness default'],
        ['Task', h('span', {class: 'mono'}, `${t.task} · ${t.task_digest_short || DASH}`)],
        ['Started', when(t.started_at)],
        ['Finished', when(t.finished_at)],
        ['Harness version', t.harness_version],
        ['Harbor version', t.harbor_version],
        ['Dataset', h('span', {class: 'mono'}, t.dataset_snapshot || t.dataset)],
        ['Config', h('span', {class: 'mono'}, `#${String(t.config_hash || '').slice(0, 8)}`)],
      ]),
      sideSection('Cost and tokens', t.usage_status === 'unmeasured' ? 'no usage was recorded' : t.usage_status === 'lower_bound' ? 'completed calls only: a lower bound' : '', params([
        ['Cost, current prices', usageValue(t.cost_usd, t.usage_status)],
        ['Cost at ingest', usageValue(t.cost_at_ingest_usd, t.usage_status)],
        ['Reported by the harness', money(t.reported_cost_usd)],
        ['Tokens', usageValue(t.tokens_total, t.usage_status, tokens)],
        ['Uncached tokens', usageValue(t.tokens_uncached, t.usage_status, tokens)],
        ['In / cache / out (raw)', `${tokens(t.n_input)} / ${tokens(t.n_cache)} / ${tokens(t.n_output)}`],
        ['Unpriced models', (t.unpriced_models || []).join(', ') || 'none'],
        ['Primary model share', pct(t.primary_share)],
      ])),
      flagRows(t.flags).length > 0 && sideSection('Cost checks', 'raised at ingest from the harness\'s own log',
        h('dl', {class: 'params rp-params rp-flags'}, flagRows(t.flags).map(([k, v]) => h('div', null, h('dt', null, k),
          h('dd', null, v))))),
      usage.length > 0 && sideSection('Usage per model', '', h('div', {class: 'scroll'}, h('table', {class: 'grid-t compact'},
        thead('Model', thNum('Input'), thNum('Cache'), thNum('Output'), thNum('Writes 5m / 1h'), thNum('Reported'),
          'Notes'),
        h('tbody', null, usage.map(([model, u]) => h('tr', null,
          h('td', {class: 'mono'}, model, u.primary && [' ', tag('primary', 'ok')]),
          tdNum(count(u.n_input)), tdNum(count(u.n_cache)), tdNum(count(u.n_output)),
          isNum(u.n_cache_write_5m) || isNum(u.n_cache_write_1h)
            ? tdNum(`${count(u.n_cache_write_5m || 0)} / ${count(u.n_cache_write_1h || 0)}`)
            : tdNum(h('span', {class: 'faint'}, 'unknown'),
              'No harness log gave the cache writes; priced as fresh input'),
          tdNum(money(u.reported_cost_usd)),
          h('td', {class: 'faint'}, [u.recovered_from && `recovered from ${u.recovered_from}`, u.synthesized && 'synthesized', gap(u).some((v) => v > 0) && 'reconciled with the totals', gap(u).some((v) => v < 0) && 'above the totals', u.reported_as && `reported as ${u.reported_as}`].filter(Boolean).join(' · ') || DASH))))))),
      sideSection('Phases', `${secs(t.total_sec)} in all`, params([
        ['Environment setup', secs(ph.env_setup_sec)], ['Agent setup', secs(ph.agent_setup_sec)],
        ['Agent run', secs(ph.agent_exec_sec)], ['Verifier', secs(ph.verifier_sec)],
      ])),
      setupSection(t),
      sideSection('Options', t.config_label || '', Object.keys(t.options || {}).length
        ? h('div', {class: 'chips'}, Object.entries(t.options).map(([k, v]) => h('span', {class: 'chip'}, h('span', {class: 'src'}, k), shown(v))))
        : h('p', {class: 'muted'}, 'Harness defaults.'),
      (t.cautions || []).length > 0 && h('ul', {class: 'rp-list warn-text'}, t.cautions.map((x) => h('li', null, x)))),
      t.trial_dir && sideSection('Trial folder', t.trial_dir_exists ? 'on disk' : 'not found on disk',
        h('div', {class: 'rp-dir'}, h('code', null, t.trial_dir),
          h('button', {type: 'button', class: 'btn ghost small', onclick: () => copyText(t.trial_dir, 'Folder path copied')}, 'Copy'))),
      h('details', {class: 'rp-raw'}, h('summary', null, 'Harbor result.json (trimmed)'), h('pre', {class: 'cmd'}, JSON.stringify(t.raw, null, 2))),
      h('details', {class: 'rp-raw'}, h('summary', null, 'lock.json'), h('pre', {class: 'cmd'}, JSON.stringify(t.lock, null, 2))),
      h('details', {class: 'rp-raw'}, h('summary', null, 'Bookkeeping'), params([
        ['Harbor model', t.harbor_model], ['Harness version pinned', t.harness_version_pinned], ['Concurrent trials', t.n_concurrent],
        ['Harbor cause', t.harbor_cause], ['Cap marker', t.cap_marker], ['Setup marker', t.setup_marker],
        ['Agent started', t.agent_started === true ? 'yes' : t.agent_started === false ? 'no' : DASH],
        ['Rewards', t.rewards ? JSON.stringify(t.rewards) : DASH], ['Task source', t.task_source_type],
        ['Trial uid', t.trial_uid], ['Ingested via', t.ingest_via], ['Ingested', when(t.ingested_at)], ['Updated', when(t.updated_at)],
        ['Parser / classifier', `${shown(t.parser_version)} / ${shown(t.classifier_version)}`], ['Task digest', t.task_digest],
      ].map(([k, v]) => [k, v === null || v === undefined ? DASH : h('span', {class: 'mono rp-impl'}, String(v))]))));
  }

  // LEDGER sub-tab -----------------------------------------------------------

  function renderLedgerPane() {
    const box = $('#rp-pane-ledger');
    if (!box) return;
    const L = rp.ledger;
    const data = L.data;
    const [col, dir] = L.order.split(' ');
    const counts = new Map(((rp.facets && rp.facets.outcomes) || []).map((o) => [o.value, o.trials]));
    const sortHead = (key, label, num) => {
      const on = col === key;
      const button = h('button', {type: 'button', class: `rp-sort${on ? ' active' : ''}`, 'aria-label': `Sort by ${label}`, onclick: () => {
        L.order = `${key} ${on ? (dir === 'DESC' ? 'ASC' : 'DESC') : SORTABLE[key]}`;
        L.offset = 0;
        loadLedger();
      }}, label, on && h('span', {class: 'arrow'}, dir === 'DESC' ? ' ▼' : ' ▲'));
      return h('th', {class: num ? 'num' : null, 'aria-sort': on ? (dir === 'DESC' ? 'descending' : 'ascending') : null}, button);
    };
    const total = data ? data.total : 0;
    const first = total ? L.offset + 1 : 0;
    const last = data ? L.offset + data.rows.length : 0;
    fill(box, panel({code: 'R-07', title: 'Ledger', meta: data ? `${count(total)} trials match` : null, flush: true},
      h('div', {class: 'filters rp-ofilter'},
        eyebrow('Outcome'),
        h('div', {class: 'rp-seg', role: 'group', 'aria-label': 'Outcome'},
          OUTCOME_ORDER.map((o) => {
            const on = L.outcomes.includes(o);
            return h('button', {type: 'button', class: on ? 'active' : null, 'aria-pressed': String(on), title: labelOf(o), onclick: () => {
              L.outcomes = on ? L.outcomes.filter((x) => x !== o) : [...L.outcomes, o];
              L.offset = 0;
              loadLedger();
            }}, h('b', {class: `rp-o-${o} rp-og`, 'aria-hidden': 'true'}, glyphOf(o)), ` ${labelOf(o)}`, counts.has(o) && h('span', {class: 'faint'}, ` ${count(counts.get(o))}`));
          })),
        L.outcomes.length > 0 && h('button', {type: 'button', class: 'btn ghost small', onclick: () => {
          L.outcomes = [];
          L.offset = 0;
          loadLedger();
        }}, 'All outcomes')),
      L.error && h('p', {class: `${is503(L.error) ? 'warn-text' : 'bad-text'} rp-pad`}, `Could not load the trials: ${L.error.message}`),
      !data && !L.error && h('p', {class: 'muted rp-pad'}, 'Loading the trials…'),
      data && (data.rows.length ? h('div', {class: 'table-scroll'}, h('table', {class: 'grid-t compact rp-ledger'},
        h('thead', null, h('tr', null,
          sortHead('started_at', 'Started'), h('th', null, 'Run'), sortHead('family', 'Harness'), h('th', null, 'Model'), sortHead('task', 'Task'),
          h('th', null, 'Outcome'), h('th', {class: 'num'}, 'Reward'), sortHead('tokens_total', 'Tokens', true), sortHead('cost_usd', 'Cost', true),
          h('th', {class: 'num'}, 'Reported'), h('th', {class: 'num'}, 'Min'), h('th', null, 'Setup'),
          h('th', null, ''))),
        h('tbody', null, data.rows.map((t) => h('tr', {class: 'rp-click', onclick: () => openTrial(t.id)},
        h('td', {class: 'mono'}, openButton(t, {}, when(t.started_at))),
        h('td', {class: 'mono'}, t.run_id, h('span', {class: 'sub'}, t.profile)),
        h('td', null, h('b', null, t.family), h('span', {class: 'sub mono'}, t.arm)),
        h('td', {class: 'mono'}, t.model, h('span', {class: 'sub'}, t.provider)),
        h('td', {class: 'mono'}, t.task),
        h('td', null, outcomeTag(t.outcome), t.outcome_reason && h('span', {class: 'sub clip', title: t.outcome_reason}, t.outcome_reason)),
        tdNum(isNum(t.reward) ? String(Math.round(t.reward * 100) / 100) : DASH),
        tdNum(usageValue(t.tokens_total, t.usage_status, tokens), isNum(t.tokens_total) ? count(t.tokens_total) : null),
        h('td', {class: 'num nowrap'}, usageValue(t.cost_usd, t.usage_status), flagMark(t.flags)),
        tdNum(money(t.reported_cost_usd)),
        tdNum(isNum(t.total_sec) ? minutes(t.total_sec / 60) : DASH),
        h('td', null, setupTag(t.setup, t.phases)),
        h('td', null, exclusionBadge(t.exclusion))))))) : h('p', {class: 'muted rp-pad'}, 'No trials match these filters.')),
      data && total > PAGE && h('div', {class: 'rp-pager'},
        h('button', {type: 'button', class: 'btn ghost small', disabled: L.offset === 0, onclick: () => {
          L.offset = Math.max(0, L.offset - PAGE);
          loadLedger();
        }}, '‹ Previous'),
        h('span', {class: 'mono'}, `${count(first)}–${count(last)} of ${count(total)}`),
        h('button', {type: 'button', class: 'btn ghost small', disabled: last >= total, onclick: () => {
          L.offset += PAGE;
          loadLedger();
        }}, 'Next ›')),
      data && total <= PAGE && total > 0 && h('div', {class: 'rp-pager'}, h('span', {class: 'mono faint'}, `${count(first)}–${count(last)} of ${count(total)}`))));
  }

  // RUNS sub-tab -------------------------------------------------------------

  function renderRunsPane() {
    const box = $('#rp-pane-runs');
    if (!box) return;
    const runs = rp.runs;
    fill(box, panel({code: 'R-08', title: 'Runs', meta: runs ? `${plural(runs.length, 'run')} across profiles` : null, flush: true},
      rp.runsError && h('p', {class: `${is503(rp.runsError) ? 'warn-text' : 'bad-text'} rp-pad`}, `Could not load the runs: ${rp.runsError.message}`),
      !runs && !rp.runsError && h('p', {class: 'muted rp-pad'}, 'Loading the runs…'),
      runs && (runs.length ? h('div', {class: 'scroll'}, h('table', {class: 'grid-t compact rp-runs'},
        thead('Run', 'Status', 'Created', 'Outcomes', thNum('Spend'), 'Harnesses and models', 'Dataset', 'Exclusion'),
        h('tbody', null, runs.map((r) => {
          const excluded = r.exclusion && r.exclusion.action === 'exclude';
          const total = sum(Object.values(r.trials || {}));
          const killed = (r.trials || {}).incomplete || 0;
          // Spend is a lower bound while any trial ran without usage, was cut short, or was killed.
          const floor = r.trials_unmeasured > 0 || r.trials_lower_bound > 0 || killed > 0;
          // The row's actions sit under its id, so they stay in view however wide the other columns get.
          return h('tr', {class: excluded ? 'rp-excluded' : null},
            h('td', {class: 'mono'}, h('span', {class: excluded ? 'rp-struck' : null}, r.run_id),
              h('span', {class: 'sub'}, `${r.profile} · ${r.source}${r.mode ? ` · ${r.mode}` : ''}`),
              h('span', {class: 'rp-acts rp-run-acts'},
                excluded
                  ? h('button', {type: 'button', class: 'btn ghost small', onclick: () => editExclusion('run', r.run_key, 'include', {current: r.exclusion})}, 'Include again')
                  : h('button', {type: 'button', class: 'btn ghost small', onclick: () => editExclusion('run', r.run_key, 'exclude', {current: r.exclusion})}, 'Exclude…'),
                h('button', {type: 'button', class: 'btn ghost small', onclick: () => handOver({runs: [r.run_key]})}, 'Report on it'),
                isMine(r) && h('button', {type: 'button', class: 'btn ghost small', onclick: () => openRun(r.run_id)}, 'Open'))),
            h('td', null, statusTag(r.status)),
            h('td', {class: 'mono'}, when(r.created_at), r.finished_at && h('span', {class: 'sub'}, `finished ${when(r.finished_at)}`)),
            h('td', null, outcomeCounts(r.trials), h('small', {class: 'sub'}, plural(total, 'trial'))),
            h('td', {class: 'num'}, `${floor ? '≥ ' : ''}${money(r.spend_usd)}`,
              isNum(r.budget_usd) && h('small', {class: 'sub'}, `budget ${money(r.budget_usd)}`),
              r.trials_unmeasured > 0 && h('small', {class: 'sub warn-text'}, `${count(r.trials_unmeasured)} without usage (spend not counted)`),
              r.trials_lower_bound > 0 && h('small', {class: 'sub warn-text'}, `${count(r.trials_lower_bound)} cut short (usage is a lower bound)`),
              killed > 0 && h('small', {class: 'sub warn-text'}, `${count(killed)} killed (spend not counted)`)),
            h('td', null, h('span', {class: 'rp-wrap'}, (r.families || []).join(', ')), h('span', {class: 'sub mono rp-wrap'}, (r.models || []).join(', '))),
            h('td', {class: 'mono faint rp-snapid'}, r.dataset_snapshot || DASH),
            h('td', {class: 'rp-wrap'}, r.exclusion ? [exclusionBadge(r.exclusion), h('small', {class: 'sub'}, r.exclusion.reason)] : h('span', {class: 'faint'}, 'counted')));
        })))) : h('p', {class: 'muted rp-pad'}, 'No runs in the ledger yet.'))));
  }

  // EXCLUSIONS sub-tab -------------------------------------------------------

  function renderExclusionsPane() {
    const box = $('#rp-pane-exclusions');
    if (!box) return;
    const seeded = (rp.status && rp.status.seeded_exclusions) || [];
    const list = rp.exclusions;
    const effectiveFor = (type, id) => (list || []).find((e) => e.effective && e.target_type === type && e.target_id === id) || null;
    fill(box,
      panel({code: 'R-09', title: 'Seeded exclusions', meta: 'shipped with the bench', flush: true},
        seeded.length ? h('div', {class: 'scroll'}, h('table', {class: 'grid-t compact'},
          thead('Target', 'Reason', 'In force', ''),
          h('tbody', null, seeded.map((s) => {
            const excluded = s.effective_action === 'exclude';
            return h('tr', null,
              h('td', {class: 'mono'}, h('span', {class: excluded ? 'rp-struck' : null}, s.target_id), h('span', {class: 'sub'}, s.target_type)),
              h('td', {class: 'rp-wrap'}, s.reason),
              h('td', null, excluded ? tag('excluded', 'bad') : tag('included again', 'ok')),
              h('td', null, h('button', {type: 'button', class: 'btn ghost small', onclick: () => editExclusion(s.target_type, s.target_id, excluded ? 'include' : 'exclude', {current: effectiveFor(s.target_type, s.target_id)})},
                excluded ? 'Include again' : 'Exclude again')));
          })))) : h('p', {class: 'muted rp-pad'}, rp.status ? 'No seeded exclusions.' : 'Loading…')),
      panel({code: 'R-10', title: 'Exclusion history', meta: list ? `${count(list.filter((e) => e.effective && e.action === 'exclude').length)} in force · ${count(list.length)} ${list.length === 1 ? 'entry' : 'entries'}` : null, flush: true},
        rp.exclError && h('p', {class: `${is503(rp.exclError) ? 'warn-text' : 'bad-text'} rp-pad`}, `Could not load the exclusions: ${rp.exclError.message}`),
        !list && !rp.exclError && h('p', {class: 'muted rp-pad'}, 'Loading…'),
        list && (list.length ? h('div', {class: 'scroll'}, h('table', {class: 'grid-t compact'},
          thead('When', 'Target', 'Action', 'Reason', 'In force', ''),
          h('tbody', null, [...list].reverse().map((e) => h('tr', {class: e.effective ? null : 'rp-superseded'},
            h('td', {class: 'mono'}, when(e.created_at)),
            h('td', {class: 'mono'}, e.target_type === 'trial' ? trialName(e.target_id) : e.target_id, h('span', {class: 'sub'}, e.target_type)),
            h('td', null, e.action === 'exclude' ? tag('exclude', 'bad') : tag('include', 'ok')),
            h('td', {class: 'rp-wrap'}, e.reason),
            h('td', null, e.effective ? h('span', {class: 'ok-text'}, '● in force') : h('span', {class: 'faint'}, '○ superseded')),
            h('td', null, e.effective && h('button', {type: 'button', class: 'btn ghost small', onclick: () => editExclusion(e.target_type, e.target_id, e.action === 'exclude' ? 'include' : 'exclude', {current: e})},
              e.action === 'exclude' ? 'Include again' : 'Exclude again')))))))
          : h('p', {class: 'muted rp-pad'}, 'Nothing has been excluded by hand.'))));
  }

  // SNAPSHOTS sub-tab --------------------------------------------------------

  function renderSnapshotsPane() {
    const box = $('#rp-pane-snapshots');
    if (!box) return;
    const list = rp.snapshots;
    const s = rp.snapshot;
    fill(box,
      panel({code: 'R-11', title: 'Snapshots', meta: list ? plural(list.length, 'snapshot') : null, flush: true},
        rp.snapError && h('p', {class: `${is503(rp.snapError) ? 'warn-text' : 'bad-text'} rp-pad`}, `Could not load the snapshots: ${rp.snapError.message}`),
        !list && !rp.snapError && h('p', {class: 'muted rp-pad'}, 'Loading…'),
        list && (list.length ? h('div', {class: 'scroll'}, h('table', {class: 'grid-t compact'},
          thead('Name', 'Saved', 'Fingerprint', thNum('Trials used'), ''),
          h('tbody', null, list.map((x) => h('tr', {class: s && s.id === x.id ? 'baseline' : null},
            h('td', null, h('b', null, x.name), h('span', {class: 'sub mono'}, x.id),
              x.direction !== 'harness_over_baseline' && tip(
                h('span', {class: 'sub faint'}, 'saved as baseline ÷ harness; shown converted'),
                'Saved before multiples were harness ÷ baseline',
                ['Opening it inverts every multiple, interval and band, and swaps above and below, so it reads '
                  + 'like a report today.'], true)),
            h('td', {class: 'mono'}, when(x.created_at)),
            h('td', {class: 'mono'}, shortFp(x.fingerprint)),
            tdNum(count(x.trials_used)),
            h('td', null, h('button', {type: 'button', class: 'btn ghost small', onclick: (e) => busy(e.currentTarget, () => openSnapshot(x.id))}, s && s.id === x.id ? 'Shown' : 'Open')))))))
          : h('p', {class: 'muted rp-pad'}, 'No snapshots yet. "Save snapshot" above stores the report as it is now.'))),
      s && snapshotPanel(s));
  }

  async function openSnapshot(id) {
    const s = await fresh.snap(api('GET', `/api/report/snapshots/${enc(id)}`));
    if (s === STALE) return;
    rp.snapshot = s;
    rp.compare = null;
    renderSnapshotsPane();
  }

  function snapshotPanel(s) {
    const r = s.response || {};
    const req = s.request || {};
    const n = activeCount({...(req.filters || {}), tasks: undefined});
    const cmp = rp.compare && rp.compare.id === s.id ? rp.compare.now : null;
    return panel({code: 'R-12', title: `Snapshot: ${s.name}`, cls: 'hot rp-snap', meta: h('span', {class: 'actions'},
      h('button', {type: 'button', class: 'btn small', onclick: (e) => busy(e.currentTarget, () => compareNow(s))}, cmp ? 'Compare again' : 'Compare to now'),
      h('button', {type: 'button', class: 'btn ghost small', onclick: () => useSnapshotFilters(s)}, 'Use its filters'),
      h('button', {type: 'button', class: 'btn ghost small', onclick: () => {
        rp.snapshot = null;
        rp.compare = null;
        renderSnapshotsPane();
      }}, 'Close'))},
    h('div', {class: 'note-strip'}, tag('read-only', 'warn'),
      `Saved ${when(s.created_at)} · ${r.provenance ? `${count(r.provenance.trials_used)} trials used · `
        + `fingerprint ${shortFp(r.provenance.fingerprint)}` : ''} · ${scopeText(r.scope) || 'tasks not recorded'} · `
        + `${n ? plural(n, 'other filter') : 'no other filters'}. These are the numbers as they were then.`),
    r.provenance && r.provenance.converted_from && h('div', {class: 'note-strip'}, tag('converted', 'warn'),
      'Saved when multiples read baseline ÷ harness. Shown converted to harness ÷ baseline: every multiple, '
      + 'interval and published band inverted, and above and below swapped. Per-task quartiles are the saved ones '
      + 'inverted; a fresh report interpolates them between the new ratios, so its median and quartiles can differ.'),
    cmp && compareTable(r, cmp),
    r.comparisons ? [headlinePanel(r, {readOnly: true}), groupsPanel(r)] : h('p', {class: 'muted'}, 'This snapshot holds no report.'));
  }

  async function compareNow(s) {
    const req = s.request || {};
    const now = await api('POST', '/api/report', {filters: req.filters || {}, options: req.options || {}});
    rp.compare = {id: s.id, now};
    renderSnapshotsPane();
  }

  function useSnapshotFilters(s) {
    const req = s.request || {};
    const options = JSON.parse(JSON.stringify(req.options || {}));
    delete options.include_pairs;
    apply.cancel();
    rp.filters = cleanFilters(req.filters || {});
    // Its tasks: the scope itself (kept as the scope), every task (the scope off), or tasks ticked.
    const scope = scopeTasks();
    const ticked = rp.filters.tasks || [];
    rp.allTasks = !ticked.length;
    if (ticked.length === scope.length && scope.every((t) => ticked.includes(t))) delete rp.filters.tasks;
    rp.options = options;
    rp.invalid = '';
    savePrefs();
    renderRail(true);
    setTab('report');
    loadReport();
    toast(`Filters and options of "${s.name}" applied`);
  }

  /** Per comparison: the change in estimate and interval width between the snapshot and now. */
  function compareTable(then, now) {
    const nowBy = new Map((now.comparisons || []).map((c) => [c.key, c]));
    const keys = unique([...(then.comparisons || []).map((c) => c.key), ...(now.comparisons || []).map((c) => c.key)]);
    const thenBy = new Map((then.comparisons || []).map((c) => [c.key, c]));
    const change = (a, b) => {
      const ga = a && a.geomean;
      const gb = b && b.geomean;
      const est = ga && gb ? `${times(ga.estimate)} → ${times(gb.estimate)} (${signed(gb.estimate - ga.estimate, 2)})` : `${metricText(a)} → ${metricText(b)}`;
      const wa = halfWidth(ga);
      const wb = halfWidth(gb);
      const width = isNum(wa) && isNum(wb) ? `×/÷${multiple(wa)} → ×/÷${multiple(wb)}${wb < wa - 0.005 ? ' narrower' : wb > wa + 0.005 ? ' wider' : ''}` : isNum(wb) ? `now ×/÷${multiple(wb)}` : DASH;
      return h('td', {class: 'num'}, est, h('small', {class: 'sub'}, width));
    };
    return h('div', {class: 'rp-compare'},
      h('h3', {class: 'rp-cov-title'}, `Compared with now (${count(now.provenance.trials_used)} trials used, fingerprint ${shortFp(now.provenance.fingerprint)})`),
      h('div', {class: 'scroll'}, h('table', {class: 'grid-t compact'},
        thead('Comparison', thNum('Pairs'), thNum('Tokens: estimate · CI width'), thNum('Cost: estimate · CI width'), thNum('Δ pass')),
        h('tbody', null, keys.map((key) => {
          const a = thenBy.get(key);
          const b = nowBy.get(key);
          const c = b || a;
          const da = a && a.pass && a.pass.delta_pp;
          const db = b && b.pass && b.pass.delta_pp;
          return h('tr', null,
            h('td', null, h('b', null, pairTitle(c)), h('span', {class: 'sub mono'}, `${c.provider}:${c.model}`),
              !a && tag('new'), !b && tag('gone', 'warn')),
            tdNum(`${a ? count(a.pairs.n_tasks) : DASH} → ${b ? count(b.pairs.n_tasks) : DASH}`),
            change(a && a.metrics.tokens_total, b && b.metrics.tokens_total),
            change(a && a.metrics.cost, b && b.metrics.cost),
            tdNum(`${pts(da && da.estimate)} → ${pts(db && db.estimate)}`));
        })))));
  }

  /** Replaces the rail's filters (options kept), as openReport does. */
  function handOver(filters) {
    apply.cancel();
    rp.invalid = '';
    rp.filters = cleanFilters(filters);
    savePrefs();
    renderRail(true);
    if (rp.tab !== 'report') {
      rp.tab = 'report';
      prefs.set('report.tab', 'report');
      showTab();
    }
    rp.stale = true;
    loadReport();
  }

  // Hooks -------------------------------------------------------------------

  function setup() {
    // Escape closes an open menu, then the drawer. Capture phase, so the page's own Escape handling does not also run.
    document.addEventListener('keydown', (e) => {
      if (e.key !== 'Escape' || ui.view !== 'report' || document.querySelector('dialog[open]')) return;
      if (document.querySelector('#view-report .rp-menu-list:not([hidden])')) {
        closeMenus();
        e.stopPropagation();
      } else if (rp.side) {
        closeSide();
        e.stopPropagation();
      }
    }, true);
    document.addEventListener('click', (e) => {
      if (!(e.target.closest && e.target.closest('#view-report .rp-menu'))) closeMenus();
    });
  }

  function render(state, force) {
    if (!rp.built) {
      loadPrefs();
      build($('#view-report'));
      force = true;
    }
    renderBanner();
    renderStrip(force);
    if (force || rp.railStale) renderRail();
    const scopeKey = scopeTasks().join('\n');
    if (rp.scopeFor !== undefined && rp.scopeFor !== scopeKey) {
      renderRail();
      if (ui.view === 'report') reload();
      else rp.stale = true;
    }
    rp.scopeFor = scopeKey;
    if (rp.showPending) {
      rp.showPending = false;
      onShow();
    }
  }

  /** The view became visible: facets and status, then whatever the shown sub-tab needs. */
  function onShow() {
    if (!rp.built) {
      rp.showPending = true;  // shown before the first poll: load once render has built the tab
      return;
    }
    loadMeta();
    if (rp.stale || !rp.report || rp.reportFor !== bodyKey()) loadReport();
    if (rp.tab !== 'report') loadTab();
  }

  function onState(state, prev) {
    const L = stateBlock('ledger');
    const ingest = L ? L.last_ingest_at || null : undefined;
    if (ingest !== undefined) {
      if (prev && rp.ingestAt !== undefined && ingest !== rp.ingestAt) (state.active_run ? ingestLater : ingestSoon)();
      rp.ingestAt = ingest;
    }
    if (rp.built) renderBanner();
  }

  function hold() {
    if (apply.pending()) return 'The report filters have changes not applied yet';
    if (rp.invalid) return rp.invalid;
    const d = rp.dialog;
    if (d && d.input.isConnected && d.input.value.trim() !== d.initial) return 'A report dialog holds typed text';
    return '';
  }

  registerView('report', {
    setup,
    render,
    onState,
    hold,
    show: onShow,
    hide() {
      closeMenus();
      closeSide();
    },
    /** params.filters replaces the rail's filters (keys left out are cleared); options are kept. */
    open(params) {
      if (!params || !params.filters) return;
      if (!rp.built) {
        rp.handed = cleanFilters(params.filters);
        return;
      }
      if (ui.view === 'report') handOver(params.filters);
      else {
        // showView renders and then show() loads with the new filters.
        apply.cancel();
        rp.invalid = '';
        rp.filters = cleanFilters(params.filters);
        savePrefs();
        renderRail(true);
        if (rp.tab !== 'report') {
          rp.tab = 'report';
          prefs.set('report.tab', 'report');
          showTab();
        }
        rp.stale = true;
      }
    },
  });
})();
