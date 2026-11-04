/* DESIGNS tab: list, view, validate, edit and enqueue designs (<section id="view-designs">).

   Owned by the DESIGNS builder, with designs.css and demo-designs.js. Brief:
   "DESIGNS tab", the Designs endpoints, and "Allowed harnesses".

   Three columns on wide screens: the design list (D-01), one version in the
   editor (D-02), and its check with the queue controls (D-03) above a compact
   queue summary. They stack on narrow screens, where the queue summary gives
   way to the QUEUE tab.

   A saved version never changes, so the editor is a draft of the next one. It
   starts from the version shown; Save sends it to
   POST /api/designs/{id}/versions with that version as base_version. While it
   differs from that version, a live check (POST /api/designs/check, 800 ms
   after typing stops) replaces the saved check, hold() keeps the page here,
   and polls never rebuild it. Add to queue and Start now go through
   enqueueDesign (the ARM confirm), and only for a saved, unedited, valid
   version that is not queued or running.

   Other areas reach this one through openDesign, shownDesign and
   refreshDesigns in app.js, which call the open, current and refresh hooks. */

(() => {
  'use strict';

  const FILTERS = [['all', 'All'], ['draft', 'Drafts'], ['approved', 'Approved'], ['queued', 'Queued'], ['ran', 'Ran'], ['archived', 'Archived']];
  // Design statuses are derived by the server. The word is always shown; the colour only adds to it.
  const DESIGN_TAG = {draft: '', approved: 'ok', queued: 'warn', running: 'run', ran: 'ok', archived: '', unreadable: 'bad', unsaved: 'warn'};
  const RUN_EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max', 'omit'];
  const RUN_FIELDS = ['budget_usd', 'parallel_arms', 'n_concurrent', 'effort', 'per_trial_cap_usd', 'max_turns',
    'timeout_multiplier', 'prebake', 'setup_timeout_multiplier', 'env_build_timeout_multiplier', 'schedule',
    'max_containers'];
  const RUN_TEXT = ['effort', 'prebake', 'schedule'];  // choices, not numbers; empty means the profile's
  const CAPS = ['per_trial_cap_usd', 'max_turns'];  // empty means no cap (null); any other empty run field is left out
  const LABELS = {
    attempts: 'Attempts', 'run.budget_usd': 'Budget', 'run.parallel_arms': 'Arms at once', 'run.n_concurrent': 'Trials at once',
    'run.per_trial_cap_usd': 'Per-trial cap', 'run.max_turns': 'Max turns', 'run.timeout_multiplier': 'Timeout multiplier',
    'run.prebake': 'Harness setup', 'run.setup_timeout_multiplier': 'Setup limit',
    'run.env_build_timeout_multiplier': 'Image pull limit', 'run.schedule': 'Schedule',
    'run.max_containers': 'Containers at once',
  };
  const BASIS = {measured: 'measured', measured_any_options: 'any options', borrowed: 'borrowed', prior: 'prior'};
  const CONFIDENCE_TAG = {high: 'ok', medium: '', low: 'warn'};
  const BLOCK = 10;  // the seeded sample comes in blocks of ten tasks, each balanced by difficulty

  const st = {
    filter: null,       // list filter, read from prefs on the first render
    list: null,         // [DesignSummary] for listFor
    listFor: '',
    listError: '',
    ed: null,           // the open design (see setBase), or null
    wanted: null,       // {id, version} from openDesign, taken on the next render
    pending: null,      // {id, version} asked for while the editor held edits
    loading: null,      // {id, version} being loaded
    loadError: '',
    conflict: '',       // the stale-version 409, until Reload latest
    merged: null,       // after Reload latest: {version, both: [field]}
    sig: '',            // the open design's list entry when it was loaded; a change reloads it
    check: null,        // the CheckResult D-03 shows
    live: false,        // st.check is a live check of the edits, not the saved one
    checking: false,
    checkError: '',
    formErrors: [],
  };
  let armSeq = 0;       // screen keys for arms, so a rename keeps its row

  const fetchList = latest();
  const fetchShown = latest();
  const fetchQuiet = latest();
  const fetchCheck = latest();
  const checkLater = debounce(() => runCheck(), 800);

  const listPoll = poller({ms: 10000, when: () => ui.view === 'designs', run: loadList});
  // The queue column and the queue state of the open version follow QUEUE's 3 s poll; this only redraws them.
  poller({ms: 3000, when: () => ui.view === 'designs', run: () => {
    renderActions();
    renderQueueSummary();
  }});

  // Small pieces -----------------------------------------------------------

  const designTag = (status) => tag(status === 'unsaved' ? 'unsaved' : statusLabel(status) || DASH, DESIGN_TAG[status] || '');
  const param = (label, value) => h('div', null, h('dt', null, label), h('dd', null, value));
  const section = (title, meta, ...body) => h('section', {class: 'dz-section'}, h('div', {class: 'section-head'}, eyebrow(title), meta), body);
  const linkButton = (label, onclick, title = null) => h('button', {type: 'button', class: 'dz-link', onclick, title}, label);
  const issue = (kind, text) => h('li', {class: kind === 'bad' ? 'bad-text' : 'warn-text'}, h('span', null, kind === 'bad' ? '✗' : '!'), h('span', null, text));
  const clip = (text, n) => (text.length > n ? `${text.slice(0, n - 1)}…` : text);
  const sortKeys = (obj) => Object.fromEntries(Object.keys(obj).sort().map((k) => [k, obj[k]]));
  const latestVersion = (ed) => (ed.meta ? ed.meta.latest_version : ed.base.latest_version);
  const seededPositions = () => new Map((ui.state.config.tasks || []).map((t, i) => [t, i]));
  const armByKey = (key) => (st.ed ? st.ed.form.arms.find((a) => a.key === key) || null : null);
  const armRowEl = (key) => $(`#dz-editor .dz-arm[data-key="${key}"]`);
  const parseNames = (text) => String(text).split(/[\s,]+/).map((t) => t.trim()).filter(Boolean);

  function valueText(v) {
    if (v === null || v === undefined) return 'none';
    return clip(typeof v === 'string' ? v : JSON.stringify(v), 90);
  }

  /**
   * The diff entries under spec.reused (raw JSON says little there), as one line per arm: the tasks it now reuses,
   * no longer reuses, and reuses from other trials. spec is the version shown, which names the task at a place.
   */
  function reusedChanges(diff, spec) {
    const arms = new Map();
    const of = (name) => arms.get(name) || arms.set(name, {from: [], to: [], moved: new Set(), retasked: new Set()}).get(name);
    const names = (v) => (typeof v === 'string' ? [v] : (Array.isArray(v) ? v : v ? [v] : []).map((e) => e && e.task))
      .filter((t) => typeof t === 'string');
    for (const d of diff) {
      const m = /^reused(?:\.([a-z][a-z0-9-]*)(?:\[(\d+)\](?:\.(\w+))?)?)?$/.exec(d.path);
      if (!m) continue;
      const [, name, at, key] = m;
      if (!name) {  // reuse turned on or off: the whole block
        for (const [n, v] of Object.entries(d.from || {})) of(n).from.push(...names(v));
        for (const [n, v] of Object.entries(d.to || {})) of(n).to.push(...names(v));
      } else if (!key || key === 'task') {
        of(name).from.push(...names(d.from));
        of(name).to.push(...names(d.to));
        if (key) of(name).retasked.add(at);
      } else {
        of(name).moved.add(at);  // the trials or runs of the task at that place
      }
    }
    const current = (spec && spec.reused) || {};
    return [...arms].map(([name, a]) => ({
      name,
      added: a.to.filter((t) => !a.from.includes(t)),
      removed: a.from.filter((t) => !a.to.includes(t)),
      swapped: [...a.moved].filter((at) => !a.retasked.has(at))
        .map((at) => ((current[name] || [])[Number(at)] || {}).task).filter((t) => typeof t === 'string'),
    })).filter((x) => x.added.length || x.removed.length || x.swapped.length);
  }

  function diffItems(diff, spec) {
    const plain = diff.filter((d) => !/^reused(\.|\[|$)/.test(d.path)).map((d) => h('li', null,
      h('span', {class: 'dz-path'}, d.path), ': ',
      h('span', {class: 'dz-from', title: JSON.stringify(d.from)}, valueText(d.from)), ' → ',
      h('span', {class: 'dz-to', title: JSON.stringify(d.to)}, valueText(d.to))));
    const reused = reusedChanges(diff, spec).map((x) => {
      const text = [x.added.length && `+ ${x.added.join(', ')}`, x.removed.length && `− ${x.removed.join(', ')}`,
        x.swapped.length && `other trials for ${x.swapped.join(', ')}`].filter(Boolean).join(' · ');
      return h('li', null, h('span', {class: 'dz-path'}, `reused · ${x.name}`), ': ', h('span', {title: text}, clip(text, 120)));
    });
    return [...plain, ...reused];
  }

  function countText(n, word, cls, glyph) {
    if (!isNum(n)) return h('span', {class: 'faint'}, `${word}s ${DASH}`);
    return h('span', {class: n ? cls : 'faint'}, `${n ? glyph : '·'} ${plural(n, word)}`);
  }

  /** The harnesses a design may use: each distinct {agent, family, kind} of the config's arms (brief, "Allowed harnesses"). */
  function allowedHarnesses() {
    const out = new Map();
    for (const a of ui.state.config.arms) {
      if (!out.has(a.agent)) out.set(a.agent, {agent: a.agent, family: a.family || a.agent, kind: a.kind || (a.agent.includes(':') ? 'import_path' : 'name')});
    }
    return [...out.values()];
  }

  // The editor's values ----------------------------------------------------
  //
  // st.ed = {id, base, meta, form, start, bad}. form holds what the inputs show (numbers as typed), so a rebuild
  // loses nothing; base is the VersionDetail it started from (null for a new design); start is comparable(form)
  // at that moment, so "edited" means "differs from it"; bad holds the number fields the browser could not parse.

  function baselineName(f) {
    const arm = f.arms.find((a) => a.key === f.baselineKey);
    return arm ? arm.name.trim() : null;
  }

  /** The values as the server compares them: trimmed text, arms without their screen keys. */
  function comparable(f) {
    const run = {};
    for (const k of RUN_FIELDS) run[k] = String(f.run[k]).trim();
    return {
      title: f.title.trim(), purpose: f.purpose.trim(), rationale: f.rationale.trim(), change_note: f.change_note.trim(),
      tasks: f.tasks, attempts: f.attempts.trim(), baseline: baselineName(f), run, reuse: f.reuse,
      arms: f.arms.map((a) => ({
        name: a.name.trim(), agent: a.agent, provider: a.provider, model: a.model.trim(), effort: a.effort.trim() || 'inherit',
        options: a.options, tasks: a.tasks,
      })),
    };
  }

  const isEdited = () => Boolean(st.ed) && JSON.stringify(comparable(st.ed.form)) !== st.ed.start;

  function runForm(run) {
    const out = {};
    for (const k of RUN_FIELDS) {
      const v = run ? run[k] : null;
      // An effort the design input cannot name (the profile's own) shows as the profile default and is left out.
      if (k === 'effort') out[k] = RUN_EFFORTS.includes(v) ? v : '';
      else if (k === 'prebake') out[k] = ['off', 'prefer', 'require'].includes(v) ? v : '';
      else if (k === 'schedule') out[k] = ['tasks', 'arms'].includes(v) ? v : '';
      else out[k] = v === 'auto' ? 'auto' : isNum(v) ? String(v) : '';
    }
    return out;
  }

  function formFrom(detail) {
    const spec = detail.spec;
    const seeded = seededPositions();
    const arms = spec.arms.map((a) => ({
      key: ++armSeq, name: a.name, agent: a.agent, provider: a.provider, model: a.model, effort: a.effort || 'inherit',
      options: sortKeys(a.options || {}), tasks: askedTasks(spec, a),
    }));
    const baseline = arms.find((a) => a.name === spec.baseline_arm);
    return {
      title: detail.title || '', purpose: detail.purpose || '', rationale: detail.rationale || '', change_note: '',
      tasks: [...spec.tasks], extra: spec.tasks.filter((t) => !seeded.has(t)).join('\n'),
      attempts: String(spec.attempts ?? 1), baselineKey: baseline ? baseline.key : null,
      run: runForm(spec.run), arms, reuse: spec.reuse || 'off',
    };
  }

  /** The tasks an arm of a saved spec takes from the ledger instead of running (spec.reused). */
  const reusedOf = (spec, name) => ((spec.reused && spec.reused[name]) || []).filter((e) => e && typeof e.task === 'string');

  /** An arm's tasks as asked for (the server's spec.asked_tasks): a version that reuses measured trials has them
      out of the arm's tasks and in spec.reused, so the editor puts them back. Saving resolves them afresh. */
  function askedTasks(spec, arm) {
    const reused = reusedOf(spec, arm.name).map((e) => e.task);
    if (!reused.length) return Array.isArray(arm.tasks) ? [...arm.tasks] : null;
    const kept = Array.isArray(arm.tasks) ? arm.tasks : spec.tasks;
    const wanted = new Set([...kept, ...reused]);
    const asked = [...new Set(spec.tasks)].filter((t) => wanted.has(t));
    asked.push(...kept.filter((t) => !spec.tasks.includes(t)));  // outside the design: kept for the check to report
    return asked.length === spec.tasks.length && asked.every((t, i) => t === spec.tasks[i]) ? null : asked;
  }

  /** A new design: run fields empty, so the profile defaults apply (shown as placeholders). The caps start at the
      profile's values, because an empty cap means no cap. */
  function blankForm() {
    const run = ui.state.config.run;
    const form = {title: '', purpose: '', rationale: '', change_note: '', tasks: [], extra: '', attempts: '1', baselineKey: null,
      run: runForm(null), arms: [], reuse: 'off'};
    for (const k of CAPS) form.run[k] = isNum(run[k]) ? String(run[k]) : '';
    return form;
  }

  /** DesignSpecIn from the editor, plus what the form alone can tell is wrong. */
  function buildSpec(ed) {
    const f = ed.form;
    const errors = [...ed.bad].map((k) => `${LABELS[k] || k.split('.').pop()} is not a number`);
    const number = (raw) => {
      const s = String(raw).trim();
      const n = s === '' ? NaN : Number(s);
      return Number.isFinite(n) ? n : undefined;
    };
    const run = {};
    for (const k of RUN_FIELDS) {
      const raw = String(f.run[k]).trim();
      if (RUN_TEXT.includes(k)) {
        if (raw) run[k] = raw;  // empty: the profile's
        continue;
      }
      if (k === 'setup_timeout_multiplier' && raw.toLowerCase() === 'auto') {
        run[k] = 'auto';
        continue;
      }
      if (k === 'setup_timeout_multiplier' && raw && number(raw) === undefined) {
        errors.push('Setup limit must be auto or a number');
      }
      const n = number(raw);
      if (n !== undefined) run[k] = n;
      else if (CAPS.includes(k) && raw === '' && !ed.bad.has(`run.${k}`)) run[k] = null;  // empty cap: none
    }
    const spec = {
      tasks: [...f.tasks],
      baseline_arm: baselineName(f),
      run,
      arms: f.arms.map((a) => ({
        name: a.name.trim(), agent: a.agent, provider: a.provider, model: a.model.trim(), effort: a.effort.trim() || 'inherit',
        options: a.options, tasks: a.tasks ? [...a.tasks] : null,
      })),
    };
    const attempts = number(f.attempts);
    if (attempts !== undefined) spec.attempts = attempts;
    if (f.reuse === 'measured') spec.reuse = 'measured';  // resolved against the ledger by every check and save
    return {spec, errors};
  }

  /** The run the option form's cap placeholders refer to: the profile's, with this design's caps. */
  function designRun() {
    const run = {...ui.state.config.run};
    for (const k of CAPS) {
      const s = String(st.ed.form.run[k]).trim();
      run[k] = s === '' ? null : Number(s);
    }
    return run;
  }

  // Queue state of the open version ----------------------------------------

  /** The open queue item (waiting, held, starting or running) for one version, from QUEUE's last poll. */
  function queueItemFor(id, version) {
    const q = queueView();
    if (!q) return null;
    return [q.active, ...(q.waiting || []), ...(q.held || [])]
      .find((item) => item && item.design_id === id && item.version === version) || null;
  }

  function liveStatus(ed) {
    if (!ed.base) return 'unsaved';
    const b = ed.base;
    if (b.archived) return 'archived';
    const item = queueItemFor(b.design_id, b.version);
    if (item) return ['starting', 'running'].includes(item.status) ? 'running' : 'queued';
    return b.status;
  }

  /** Why Add to queue and Start now are off, or '' when they are on. */
  function queueBlock() {
    const ed = st.ed;
    if (!ed) return 'Open a design first.';
    if (!ed.base) return 'Save the design first.';
    const b = ed.base;
    if (b.archived) return 'Archived: unarchive it to queue it.';
    if (isEdited()) return 'Save or discard your edits first.';
    const item = queueItemFor(b.design_id, b.version);
    if (item) return `v${b.version} is in the queue as ${item.id} (${statusLabel(item.status)}).`;
    if (b.status === 'queued' || b.status === 'running') return `v${b.version} is ${b.status}.`;
    const v = b.check && b.check.validation;
    if (!v) return 'This version has not been checked.';
    if (v.problems.length) return `Clear the ${plural(v.problems.length, 'problem')} first.`;
    if (v.status !== 'valid') return 'This version did not pass its check.';
    return '';
  }

  // Layout -----------------------------------------------------------------

  function build(root) {
    const button = (label, onclick, title) => h('button', {type: 'button', class: 'btn ghost small', onclick, title}, label);
    fill(root, h('div', {class: 'dz-layout'},
      h('aside', {class: 'panel dz-list-panel', 'aria-label': 'Designs'},
        h('div', {class: 'panel-head'}, h('span', {class: 'panel-code'}, 'D-01'), h('h2', {class: 'panel-title'}, 'Designs'),
          h('div', {class: 'panel-meta', id: 'dz-list-meta'})),
        h('div', {class: 'dz-list-actions'},
          button('+ New design', () => newDesign(), 'An empty design with the profile defaults'),
          button('From config arms…', () => fromConfig(), 'A design from a mode and arms on LAUNCH'),
          button('From past run…', () => fromRun(), 'Run again: the spec a past run executed')),
        h('div', {class: 'dz-filters', role: 'group', 'aria-label': 'Show'},
          FILTERS.map(([value, label]) => h('button', {type: 'button', 'data-f': value, 'aria-pressed': 'false', onclick: () => setFilter(value)}, label))),
        h('div', {class: 'dz-list', id: 'dz-list'})),
      h('div', {class: 'dz-main', id: 'dz-editor', oninput: onEdit, onchange: onEdit}),
      h('div', {class: 'dz-side'},
        h('section', {class: 'panel dz-check-panel', id: 'dz-check'},
          h('div', {class: 'panel-head'}, h('span', {class: 'panel-code'}, 'D-03'), h('h2', {class: 'panel-title'}, 'Check'),
            h('div', {class: 'panel-meta dz-check-status'})),
          h('div', {class: 'dz-armbar'},
            h('div', {class: 'actions'},
              h('button', {type: 'button', class: 'btn primary dz-enqueue', disabled: true, onclick: (e) => enqueue(false, e.currentTarget)}, 'Add to queue'),
              h('button', {type: 'button', class: 'btn primary dz-start', disabled: true, onclick: (e) => enqueue(true, e.currentTarget)}, '▶ Start now')),
            h('p', {class: 'dz-arm-note'})),
          h('div', {class: 'panel-body dz-check-body'})),
        h('div', {class: 'dz-queue-summary', id: 'dz-queue-summary'}))));
    renderEditor();
    renderCheck();
  }

  function markFilter() {
    for (const b of document.querySelectorAll('#view-designs .dz-filters button')) {
      const on = b.dataset.f === st.filter;
      b.classList.toggle('active', on);
      b.setAttribute('aria-pressed', String(on));
    }
  }

  // List -------------------------------------------------------------------

  async function loadList() {
    const filter = st.filter || 'all';
    try {
      const data = await fetchList(api('GET', `/api/designs?filter=${enc(filter)}`));
      if (data === STALE) return;
      st.list = data.designs;
      st.listFor = filter;
      st.listError = '';
    } catch (err) {
      st.listError = err.status === 503 ? `Busy: ${err.message}` : err.message;
    }
    if (ui.view === 'designs') renderList();
    followShown();
  }

  function setFilter(value) {
    if (value === st.filter) return;
    st.filter = value;
    prefs.set('designs.filter', value);
    markFilter();
    renderList();
    listPoll.now();
  }

  function renderList(force = false) {
    const box = $('#dz-list');
    if (!box) return;
    const shownId = st.ed ? st.ed.id : null;
    if (!changed('designs:list', [st.list, st.listFor, st.filter, st.listError, shownId]) && !force) return;
    const list = st.list || [];
    box.classList.toggle('dz-stale', st.listFor !== st.filter);
    $('#dz-list-meta').textContent = st.list ? `${list.length} ${st.listFor.toUpperCase()}` : '';
    const empty = {
      all: 'No designs yet. Start one with New design, or from your config arms or a past run.',
      draft: 'No drafts.', approved: 'No approved designs.', queued: 'Nothing queued or running.', ran: 'No design has run yet.',
      archived: 'Nothing archived.',
    };
    fill(box,
      st.listError && h('p', {class: st.listError.startsWith('Busy') ? 'warn-text dz-list-msg' : 'bad-text dz-list-msg'}, st.listError),
      !st.list && !st.listError && h('p', {class: 'faint dz-list-msg'}, 'Loading designs…'),
      st.list && !list.length && h('p', {class: 'faint dz-list-msg'}, empty[st.listFor] || 'Nothing here.'),
      list.length > 0 && h('ol', {class: 'dz-cards'}, list.map((s) => card(s, s.id === shownId))));
  }

  function card(s, active) {
    if (s.status === 'unreadable') {
      return h('li', {class: 'dz-card unreadable'},
        h('div', {class: 'dz-card-main'},
          h('span', {class: 'dz-card-top'}, h('span', {class: 'dz-card-title mono'}, s.id), designTag('unreadable')),
          h('span', {class: 'bad-text dz-card-error'}, s.error || 'This design could not be read.')));
    }
    const runs = s.runs || [];
    const lastRun = runs[runs.length - 1];
    return h('li', {class: active ? 'dz-card active' : 'dz-card'},
      h('button', {type: 'button', class: 'dz-card-main', 'aria-current': active ? 'true' : null, onclick: () => pick(s.id)},
        h('span', {class: 'dz-card-top'}, authorBadge(s.author), h('span', {class: 'dz-card-title'}, s.title || s.id), designTag(s.status)),
        h('span', {class: 'dz-card-id'}, `${s.id} · v${s.latest_version}`),
        h('span', {class: 'dz-card-est'}, estimateText(s.expected_usd, s.p90_usd), s.check_stale && tag('stale', 'warn')),
        h('span', {class: 'dz-card-counts'},
          countText(s.problems, 'problem', 'bad-text', '✗'), countText(s.warnings, 'warning', 'warn-text', '!'),
          isNum(s.arms) && h('span', {class: 'faint'}, `${plural(s.arms, 'arm')} · ${plural(s.trials, 'trial')}`))),
      lastRun && h('button', {type: 'button', class: 'dz-card-run', title: 'Open this run', onclick: () => openRun(lastRun)},
        `last run ${lastRun}`));
  }

  /** The open design's list entry changed (new version, status, check): reload the version shown. */
  function followShown() {
    const ed = st.ed;
    if (!ed || !ed.id || !st.list || st.listFor !== st.filter) return;
    const s = st.list.find((x) => x.id === ed.id);
    if (!s) return;
    const sig = sigOf(s);
    if (sig === st.sig) return;
    st.sig = sig;
    reloadShown();
  }

  const sigOf = (s) => (s ? JSON.stringify([s.latest_version, s.status, s.checked_at, s.check_stale, s.archived, s.runs]) : '');
  const listEntry = (id) => (st.list && st.listFor === st.filter ? st.list.find((x) => x.id === id) || null : null);

  // Opening and leaving ----------------------------------------------------

  async function fetchVersion(id, version) {
    const data = await api('GET', `/api/designs/${enc(id)}`);
    const detail = version && version !== data.current.version
      ? await api('GET', `/api/designs/${enc(id)}/versions/${enc(version)}`)
      : data.current;
    return {meta: data.meta, detail};
  }

  /**
   * Loads one version into the editor. `dropEdits`: the person already chose to drop the edits it holds.
   * `restored`: the design open last time, reopened by itself; if it is gone, the editor just stays empty.
   */
  async function load(id, version = null, dropEdits = false, restored = false) {
    st.loading = {id, version};
    st.loadError = '';
    st.pending = null;
    // Dropped from now on: only typing after this point can still hold the page or keep the old form.
    if (dropEdits && st.ed) st.ed.start = JSON.stringify(comparable(st.ed.form));
    renderActions();
    const box = $('#dz-editor');
    if (box) box.classList.add('dz-loading');
    if (!st.ed) renderEditor();
    let data;
    try {
      data = await fetchShown(fetchVersion(id, version));
    } catch (err) {
      st.loading = null;
      if (box) box.classList.remove('dz-loading');
      if (err.status === 404 && prefs.get('designs.open') === id) prefs.set('designs.open', null);
      if (isEdited()) {  // edits arrived while it loaded: keep them
        toast(`Could not open ${id}: ${err.message}`, true);
        return;
      }
      st.ed = null;
      st.loadError = restored && err.status === 404 ? '' : `Could not open ${id}${version ? ` v${version}` : ''}: ${err.message}`;
      renderEditor();
      renderCheck();
      renderActions();
      renderList();
      return;
    }
    if (data === STALE) return;
    st.loading = null;
    if (box) box.classList.remove('dz-loading');
    if (isEdited()) {  // typed into the old editor meanwhile: never drop that, offer the design instead
      st.pending = {id, version};
      renderNotices();
      return;
    }
    setBase(data.detail, data.meta);
  }

  /** Shows a saved version: the editor starts from it, and D-03 shows its saved check. */
  function setBase(detail, meta) {
    const form = formFrom(detail);
    st.ed = {id: detail.design_id, base: detail, meta, form, start: JSON.stringify(comparable(form)), bad: new Set()};
    Object.assign(st, {conflict: '', merged: null, pending: null, loadError: '', sig: sigOf(listEntry(detail.design_id))});
    prefs.set('designs.open', detail.design_id);
    renderEditor();
    showSavedCheck();
    renderList();
  }

  /** Re-reads the open version when its list entry changed. */
  async function reloadShown() {
    const ed = st.ed;
    if (!ed || !ed.base || st.loading) return;
    let data;
    try {
      data = await fetchQuiet(fetchVersion(ed.id, ed.base.version));
    } catch (err) {
      if (err.status === 404) toast(`${ed.id} v${ed.base.version} is gone: ${err.message}`, true);
      return;
    }
    if (data === STALE || st.ed !== ed) return;
    if (JSON.stringify([data.detail, data.meta]) === JSON.stringify([ed.base, ed.meta])) return;
    refreshBase(ed, data.detail, data.meta);
  }

  /** A fresh read of the version the editor is based on. Without edits it simply shows it. With edits the form stays:
      the version's content cannot change, only its status, runs, archive state and saved check. */
  function refreshBase(ed, detail, meta) {
    if (st.ed !== ed) return;
    if (!isEdited()) {
      setBase(detail, meta);
      return;
    }
    ed.base = detail;
    ed.meta = meta;
    st.sig = sigOf(listEntry(ed.id));
    renderNotices();
    renderActions();
  }

  /** True when there is nothing to lose, or the person chose to discard their edits. */
  async function leaveEdits() {
    if (!isEdited()) return true;
    return confirmAction({
      code: 'DISCARD', title: 'Discard your edits?', tone: 'danger', ok: 'Discard edits',
      body: `${st.ed.form.title.trim() || 'This design'} has changes that are not saved. They are lost if you go on.`,
    });
  }

  async function pick(id, version = null) {
    if (st.ed && st.ed.id === id && st.ed.base && !version && st.ed.base.version === latestVersion(st.ed) && !st.loading) return;
    if (!(await leaveEdits())) return;
    load(id, version, true);
  }

  function takeWanted() {
    const want = st.wanted;
    st.wanted = null;
    if (isEdited()) {
      st.pending = want;  // never drop typing: the editor says what was asked for
      renderNotices();
      return;
    }
    load(want.id, want.version);
  }

  async function newDesign() {
    if (!(await leaveEdits())) return;
    const form = blankForm();
    st.ed = {id: null, base: null, meta: null, form, start: JSON.stringify(comparable(form)), bad: new Set()};
    Object.assign(st, {conflict: '', merged: null, pending: null, loadError: '', sig: ''});
    prefs.set('designs.open', null);
    renderEditor();
    renderList();
    scheduleCheck({now: true});
    const title = $('#dz-editor [data-k="title"]');
    if (title) title.focus();
  }

  async function discard() {
    const ed = st.ed;
    if (!ed || !(await leaveEdits())) return;
    if (st.ed !== ed) return;
    if (ed.base) {
      setBase(ed.base, ed.meta);
      reloadShown();  // the list may have moved on while it held edits
      return;
    }
    st.ed = null;
    renderEditor();
    renderCheck();
    renderActions();
    renderList();
  }

  /** A version a create route returned: show it and refresh the list. */
  async function opened(detail, message) {
    let meta = null;
    try {
      meta = (await api('GET', `/api/designs/${enc(detail.design_id)}`)).meta;
    } catch { /* the version selector then lists only this version */ }
    setBase(detail, meta || {latest_version: detail.version, versions: {[detail.version]: {
      author: detail.author, created_at: detail.created_at, change_note: detail.change_note, status: detail.status,
    }}});
    listPoll.now();
    toast(message);
  }

  // Editing ----------------------------------------------------------------

  /** Every input, select and textarea in the editor carries data-k, the form value it edits. */
  function onEdit(e) {
    const el = e.target;
    const k = el && el.dataset ? el.dataset.k : null;
    const ed = st.ed;
    if (!k || !ed) return;
    const f = ed.form;
    if (el.validity && el.validity.badInput) ed.bad.add(k);
    else ed.bad.delete(k);
    const [head, a, b] = k.split('.');
    if (head === 'run') {
      f.run[a] = el.value;
      if (a === 'effort') for (const arm of f.arms) refreshEffort(arm.key);
    } else if (head === 'arm') {
      const arm = armByKey(Number(a));
      if (!arm) return;
      const before = arm[b];
      arm[b] = el.value;
      if (b === 'agent' && before !== el.value) {
        refreshEffort(arm.key);
        updateArmNotes(arm);
      } else if (b === 'provider' || b === 'model' || b === 'name') {
        updateArmNotes(arm);
      }
    } else if (head === 'baseline') {
      if (!el.checked) return;
      f.baselineKey = el.value === '' ? null : Number(el.value);
      for (const row of document.querySelectorAll('#dz-editor .dz-arm')) row.classList.toggle('is-base', Number(row.dataset.key) === f.baselineKey);
    } else if (head === 'extra') {
      setExtra(el.value);
    } else if (head === 'attempts') {
      f.attempts = el.value;
      updateTaskCount();
    } else if (head in f) {
      f[head] = el.value;  // title, purpose, rationale, change_note, reuse
    }
    touched();
  }

  function touched() {
    renderActions();
    scheduleCheck();
  }

  /** Seeded tasks keep the sample's order; tasks typed in the box follow them. */
  function toggleTask(task, on = null) {
    const f = st.ed.form;
    const pos = seededPositions();
    const has = f.tasks.includes(task);
    const want = on === null ? !has : on;
    if (want === has) return;
    if (!want) {
      f.tasks = f.tasks.filter((t) => t !== task);
    } else {
      const p = pos.get(task);
      let at = f.tasks.findIndex((t) => pos.has(t) && pos.get(t) > p);
      if (at < 0) at = f.tasks.reduce((last, t, i) => (pos.has(t) ? i + 1 : last), 0);
      f.tasks.splice(at, 0, task);
    }
    const button = $(`#dz-editor .dz-task[data-task="${CSS.escape(task)}"]`);
    if (button) button.setAttribute('aria-pressed', String(want));
  }

  function setTasks(list, on) {
    for (const task of list) toggleTask(task, on);
    tasksChanged();
  }

  function setExtra(text) {
    const f = st.ed.form;
    const pos = seededPositions();
    const names = parseNames(text);
    f.extra = text;
    f.tasks = [...f.tasks.filter((t) => pos.has(t)), ...names.filter((t) => !pos.has(t))];
    const seeded = names.filter((t) => pos.has(t));
    const hint = $('#dz-editor .dz-extra-note');
    if (hint) {
      fill(hint, seeded.length > 0 && h('small', {class: 'warn-text'},
        `In the seeded sample, so pick ${seeded.length === 1 ? 'it' : 'them'} in the strip above: ${listText(seeded)}`));
    }
    updateTaskCount();
    refreshSubsets();
  }

  function tasksChanged() {
    updateTaskCount();
    refreshSubsets();
    touched();
  }

  function updateTaskCount() {
    const el = $('#dz-editor .dz-task-count');
    if (!el || !st.ed) return;
    const f = st.ed.form;
    const n = f.tasks.length;
    const attempts = Number(f.attempts) || 1;
    el.textContent = n ? `${plural(n, 'task')} × ${attempts} = ${plural(n * attempts, 'trial')} per arm` : 'No tasks chosen';
  }

  function addArm() {
    const f = st.ed.form;
    const cfg = ui.state.config;
    const used = new Set(f.arms.map((a) => a.name.trim()));
    const from = cfg.arms.find((a) => !used.has(a.name)) || cfg.arms[0] || null;
    const stem = from ? from.name : 'arm';
    let name = stem;
    for (let i = 2; used.has(name); i++) name = `${stem.slice(0, 28)}-${i}`;
    const harness = from ? from.agent : (allowedHarnesses()[0] || {}).agent || '';
    const provider = from ? from.provider : (cfg.providers.find((p) => p.enabled) || cfg.providers[0] || {}).name || '';
    // A copy of that config arm. Its own options are allowed as they are, host-file ones included (strands needs them).
    const arm = {key: ++armSeq, name, agent: harness, provider, model: from ? from.model : '', effort: from ? from.effort || 'inherit' : 'inherit',
      options: from ? sortKeys(JSON.parse(JSON.stringify(from.options || {}))) : {}, tasks: null};
    f.arms.push(arm);
    if (f.baselineKey === null && name === cfg.baseline_arm) f.baselineKey = arm.key;
    renderArms();
    const input = $(`#dz-editor [data-k="arm.${arm.key}.name"]`);
    if (input) input.focus();
    touched();
  }

  function removeArm(key) {
    const f = st.ed.form;
    f.arms = f.arms.filter((a) => a.key !== key);
    if (f.baselineKey === key) f.baselineKey = null;
    renderArms();
    touched();
  }

  async function editOptions(key, button) {
    const arm = armByKey(key);
    if (!arm) return;
    const schema = await busy(button, () => loadSchema(arm.agent));
    if (!schema || !st.ed || armByKey(key) !== arm) return;
    const run = designRun();
    const known = new Set(schema.fields.map((f) => f.name));
    const kept = Object.fromEntries(Object.entries(arm.options).filter(([name]) => !known.has(name)));
    const groups = {reasoning: [], limits: [], other: []};
    for (const f of schema.fields) (groups[f.group] || groups.other).push(f);
    const current = {options: arm.options};
    const group = (title, fields) => fields.length > 0 && [
      h('div', {class: 'group-title'}, title),
      h('div', {class: 'row'}, fields.map((f) => optionField(f, current, run, {}))),
    ];
    const otherSet = groups.other.filter((f) => arm.options[f.name] !== undefined && arm.options[f.name] !== null).length;
    const family = (allowedHarnesses().find((x) => x.agent === arm.agent) || {}).family || arm.agent;
    const options = await formDialog({
      code: 'OPT', title: `Options · ${arm.name}`, ok: 'Apply',
      body: [
        h('p', {class: 'muted'}, `${family} harness. Empty fields are left out. This changes the design only, not your config arm.`),
        group('A · REASONING AND THINKING', groups.reasoning),
        group('B · LIMITS', groups.limits),
        groups.other.length > 0 && h('details', {open: otherSet > 0},
          h('summary', null, `C · Other options (${groups.other.length}${otherSet ? `, ${otherSet} set` : ''})`),
          h('div', {class: 'row', style: 'margin-top:10px'}, groups.other.map((f) => optionField(f, current, run, {})))),
        Object.keys(kept).length > 0 && h('p', {class: 'warn-text'},
          `Kept as they are, not options of this harness: ${listText(Object.keys(kept))}.`),
      ],
      read: (form) => ({...kept, ...readOptions(form, schema)}),
    });
    if (!options || armByKey(key) !== arm) return;
    arm.options = sortKeys(options);
    updateArmNotes(arm);
    touched();
  }

  async function editSubset(key) {
    const arm = armByKey(key);
    const f = st.ed && st.ed.form;
    if (!arm || !f) return;
    if (!f.tasks.length) {
      toast('Choose the design\'s tasks first.');
      return;
    }
    const chosen = new Set(arm.tasks || f.tasks);
    const scope = (value, label, checked) => h('label', {class: 'dz-check'},
      h('input', {type: 'radio', name: 'scope', value, checked}), h('span', null, label));
    const out = await formDialog({
      code: 'TASKS', title: `Tasks · ${arm.name}`, ok: 'Apply',
      body: [
        h('div', {class: 'dz-checks'},
          scope('all', `Every task of the design (${f.tasks.length})`, arm.tasks === null),
          scope('some', 'Only the tasks ticked below', arm.tasks !== null)),
        h('div', {class: 'dz-checks dz-task-picks', onchange: (e) => {
          if (e.target.name === 'task') e.currentTarget.closest('form').elements.scope.value = 'some';
        }}, f.tasks.map((t, i) => h('label', {class: 'dz-check'},
          h('input', {type: 'checkbox', name: 'task', value: t, checked: chosen.has(t)}),
          h('span', {class: 'mono'}, `${pad2(i + 1)}  ${t}`)))),
        note('Every task is sent as null, so the arm also runs tasks added to the design later. Use a subset to complete half-pairs.'),
      ],
      read: (form) => {
        if (form.elements.scope.value === 'all') return {tasks: null};
        const picked = [...form.querySelectorAll('input[name="task"]:checked')].map((i) => i.value);
        if (!picked.length) throw new Error('Tick at least one task, or choose every task.');
        return {tasks: picked};
      },
    });
    if (!out || armByKey(key) !== arm) return;
    arm.tasks = out.tasks;
    updateArmNotes(arm);
    touched();
  }

  /** Applies a suggested budget from the check. */
  function useBudget(value) {
    if (!st.ed) return;
    st.ed.form.run.budget_usd = String(value);
    st.ed.bad.delete('run.budget_usd');
    const input = $('#dz-editor [data-k="run.budget_usd"]');
    if (input) input.value = String(value);
    touched();
  }

  // Editor -----------------------------------------------------------------

  function renderEditor() {
    const box = $('#dz-editor');
    if (!box) return;
    const ed = st.ed;
    if (!ed) {
      fill(box, panel({code: 'D-02', title: 'Design'},
        st.loading ? h('p', {class: 'muted'}, `Loading ${st.loading.id}…`)
          : h('div', {class: 'empty-state'},
            h('div', {class: 'big'}, st.loadError ? 'NOT FOUND' : 'NO DESIGN OPEN'),
            st.loadError ? h('p', {class: 'bad-text'}, st.loadError) : h('p', null, 'Pick a design from the list, or start one.'),
            h('div', {class: 'actions dz-center'},
              h('button', {type: 'button', class: 'btn small', onclick: () => newDesign()}, '+ New design'),
              h('button', {type: 'button', class: 'btn ghost small', onclick: () => fromConfig()}, 'From config arms…'),
              h('button', {type: 'button', class: 'btn ghost small', onclick: () => fromRun()}, 'From past run…')))));
      return;
    }
    // A rebuild only happens without unsaved input (or from the model, which holds it); keep the caret's field.
    const active = document.activeElement && box.contains(document.activeElement) ? document.activeElement.dataset.k : null;
    fill(box, editorPanel(ed));
    if (active) {
      const el = box.querySelector(`[data-k="${CSS.escape(active)}"]`);
      if (el && el.type !== 'radio') el.focus();
    }
    renderArms();
    renderNotices();
    updateTaskCount();
    updateEditorNotes();
    renderActions();
  }

  function editorPanel(ed) {
    const b = ed.base;
    const f = ed.form;
    return h('section', {class: 'panel dz-editor'},
      h('div', {class: 'panel-head'},
        h('span', {class: 'panel-code'}, 'D-02'),
        h('h2', {class: 'panel-title'}, b ? 'Design' : 'New design'),
        h('div', {class: 'panel-meta'},
          b && h('span', {class: 'dz-id'}, b.design_id),
          h('span', {class: 'dz-status'}),
          b && tip(h('span', {class: 'dz-hash'}, shortHash(b.spec_hash)), 'Spec hash', [b.spec_hash], true))),
      h('div', {class: 'panel-body dz-editor-body'},
        h('div', {class: 'dz-notices'}),
        h('div', {class: 'dz-headline'},
          field('Title', h('input', {type: 'text', class: 'dz-title-input', value: f.title, maxlength: 80, required: true,
            placeholder: 'Required on every save', autocomplete: 'off', 'data-k': 'title'})),
          b && versionPicker(ed)),
        field('Purpose', h('textarea', {rows: 2, value: f.purpose, maxlength: 1000, placeholder: 'What this design is for', 'data-k': 'purpose'})),
        b && h('div', {class: 'dz-vermeta'}, authorBadge(b.author), h('span', null, `v${b.version} saved ${when(b.created_at)}`),
          b.change_note && h('span', {class: 'muted'}, `· ${b.change_note}`)),
        b && b.runs && b.runs.length > 0 && h('div', {class: 'dz-runs'}, eyebrow('Runs of this version'),
          b.runs.map((r) => h('button', {type: 'button', class: 'dz-run-chip', onclick: () => openRun(r.run_id), title: 'Open this run'},
            h('span', {class: 'mono'}, r.run_id), r.status ? statusTag(r.status) : tag('unknown'),
            h('span', {class: 'mono muted'}, `${money(r.spend_usd)} of ${money(r.budget_usd)}`)))),
        b && b.author === 'assistant' && assistantBanner(b),
        b && b.diff_from_parent && b.diff_from_parent.length > 0 && h('details', {class: 'dz-diff', open: true},
          h('summary', null, `Changes from v${b.parent_version}`),
          h('ul', null, diffItems(b.diff_from_parent, b.spec))),
        actionBar(ed),
        tasksSection(f),
        runSection(f),
        armsSection(f),
        h('div', {class: 'dz-notes-row'},
          field('Rationale', h('textarea', {rows: 2, value: f.rationale, maxlength: 1500, placeholder: 'Why this design (optional)', 'data-k': 'rationale'})),
          field('Change note', h('input', {type: 'text', value: f.change_note, maxlength: 200, autocomplete: 'off',
            placeholder: b ? `What v${latestVersion(ed) + 1} changes (optional)` : 'Optional', 'data-k': 'change_note'}))),
        h('datalist', {id: 'dz-model-options'},
          [...new Set([...CLAUDE_MODELS, ...ui.state.config.arms.map((a) => a.model), ...f.arms.map((a) => a.model)])]
            .filter(Boolean).map((m) => h('option', {value: m})))));
  }

  function versionPicker(ed) {
    const versions = ed.meta && ed.meta.versions ? Object.entries(ed.meta.versions) : [];
    const choices = versions.sort((x, y) => Number(y[0]) - Number(x[0])).map(([n, v]) => [n, [
      `v${n}`, v.author === 'assistant' ? 'AI' : 'You', when(v.created_at),
      v.status && v.status !== 'draft' ? statusLabel(v.status) : '', v.change_note ? clip(v.change_note, 48) : '',
    ].filter(Boolean).join(' · ')]);
    return field('Version', select('dz-version', choices, String(ed.base.version), {
      onchange: async (e) => {
        const n = Number(e.target.value);
        if (!(await leaveEdits())) {
          e.target.value = String(st.ed.base.version);
          return;
        }
        load(ed.id, n, true);
      },
    }), {notes: [ed.base.version !== latestVersion(ed) && h('small', {class: 'warn-text'}, `Latest is v${latestVersion(ed)}`)]});
  }

  function assistantBanner(b) {
    return h('div', {class: 'dz-ai'},
      h('div', {class: 'dz-ai-head'}, tag('AI', 'ai'), h('b', null, 'Drafted by the assistant'),
        b.conversation_id && h('button', {type: 'button', class: 'btn ghost small', onclick: () => openAssistant({conversationId: b.conversation_id})},
          'Open its conversation ', h('span', {class: 'arrow'}, '›'))),
      b.rationale ? h('div', {class: 'dz-rationale'}, renderMarkdown(b.rationale)) : h('p', {class: 'muted'}, 'No rationale given.'));
  }

  function actionBar(ed) {
    const b = ed.base;
    return h('div', {class: 'dz-actionbar'},
      h('button', {type: 'button', class: 'btn dz-save', onclick: (e) => save(e.currentTarget)}, b ? 'Save as new version' : 'Save design'),
      h('button', {type: 'button', class: 'btn ghost dz-discard', onclick: () => discard()}, b ? 'Discard edits' : 'Discard'),
      h('span', {class: 'unsaved dz-unsaved', hidden: true}, b ? `● Unsaved · based on v${b.version}` : '● Not saved yet'),
      h('span', {class: 'dz-grow'}),
      b && h('button', {type: 'button', class: 'btn ghost small', onclick: (e) => duplicate(e.currentTarget), title: 'A new design from this spec'}, 'Duplicate…'),
      b && h('button', {type: 'button', class: 'btn ghost small dz-archive',
        onclick: (e) => (st.ed && st.ed.base && st.ed.base.archived ? unarchive : archive)(e.currentTarget)}, b.archived ? 'Unarchive' : 'Archive'),
      b && h('button', {type: 'button', class: 'btn ghost small', onclick: () => openAssistant({design: {id: b.design_id, version: b.version}}),
        title: 'Open the assistant with this version attached'}, h('span', {class: 'dz-ai-mark'}, '◆'), 'Ask assistant'));
  }

  function tasksSection(f) {
    const seeded = ui.state.config.tasks || [];
    const chosen = new Set(f.tasks);
    const blocks = [];
    for (let i = 0; i < seeded.length; i += BLOCK) blocks.push(seeded.slice(i, i + BLOCK).map((t, j) => [t, i + j]));
    return section('Tasks', h('span', {class: 'dz-task-count'}),
      seeded.length
        ? h('div', {class: 'dz-strip'}, blocks.map((list, n) => h('div', {class: 'dz-block'},
          h('div', {class: 'dz-block-head'},
            h('span', null, `BLOCK ${n + 1}`), h('span', {class: 'faint'}, `${list[0][1] + 1}–${list[list.length - 1][1] + 1}`),
            linkButton('all', () => setTasks(list.map(([t]) => t), true), `Add every task of block ${n + 1}`),
            linkButton('none', () => setTasks(list.map(([t]) => t), false), `Remove every task of block ${n + 1}`)),
          h('div', {class: 'dz-tasks'}, list.map(([t, i]) => h('button', {
            type: 'button', class: 'dz-task', 'data-task': t, 'aria-pressed': String(chosen.has(t)), title: t,
            onclick: () => {
              toggleTask(t);
              tasksChanged();
            },
          }, h('span', {class: 'dz-task-n'}, pad2(i + 1)), h('span', {class: 'dz-task-name'}, t)))))))
        : h('p', {class: 'muted'}, 'This profile has no seeded task sample yet. Type task names below.'),
      h('div', {class: 'dz-task-more'},
        field('Other tasks', h('textarea', {rows: 2, class: 'mono', value: f.extra, spellcheck: 'false', 'data-k': 'extra',
          placeholder: 'Task names outside the seeded sample, one per line'}), {notes: [h('span', {class: 'dz-extra-note'})]}),
        field('Attempts', h('input', {type: 'number', min: 1, max: 5, step: 1, value: f.attempts, placeholder: '1', 'data-k': 'attempts'}),
          {notes: [note('1–5 per task and arm')]})),
      h('div', {class: 'dz-order'}));
  }

  function runSection(f) {
    const run = ui.state.config.run;
    const num = (k, label, attrs, notes = []) => field(label,
      h('input', {type: 'number', value: f.run[k], 'data-k': `run.${k}`, ...attrs}), {notes});
    const profile = (v) => (isNum(v) ? `${v} (profile)` : 'profile default');
    return section('Run settings', h('span', {class: 'dz-peak'}),
      h('div', {class: 'row'},
        num('budget_usd', 'Budget (USD)', {min: 0, step: 'any', required: true, placeholder: `required · profile ${run.budget_usd}`},
          [h('span', {class: 'dz-suggest'}), note('A hard stop, checked after every trial')]),
        field('Schedule', select(null, [['', `profile (${run.schedule || 'tasks'})`],
          ['tasks', 'tasks: a race per task'], ['arms', 'arms: one job per arm']], f.run.schedule,
        {'data-k': 'run.schedule'}),
        {notes: [note('tasks: each task\'s images built, raced by every arm, removed')]}),
        num('max_containers', 'Containers at once',
          {min: 1, max: 16, step: 1, placeholder: profile(run.max_containers)},
          [note('In one task race')]),
        num('parallel_arms', 'Arms at once', {min: 1, max: 12, step: 1, placeholder: profile(run.parallel_arms)},
          [note('Schedule arms only')]),
        num('n_concurrent', 'Trials at once, per arm', {min: 1, max: 8, step: 1, placeholder: profile(run.n_concurrent)}),
        field('Effort', select(null, [['', `profile (${run.effort})`], ...RUN_EFFORTS.map((e) => [e, e === 'omit' ? "omit, each harness's default" : e])],
          f.run.effort, {'data-k': 'run.effort'}), {notes: [note('For arms set to inherit')]}),
        num('per_trial_cap_usd', 'Per-trial cap (USD)', {min: 0, step: 'any', placeholder: 'none'},
          [note(`Empty = no cap · profile ${isNum(run.per_trial_cap_usd) ? money(run.per_trial_cap_usd) : 'none'}`)]),
        num('max_turns', 'Max turns', {min: 1, step: 1, placeholder: 'none'},
          [note(`Empty = no cap · profile ${isNum(run.max_turns) ? run.max_turns : 'none'}`)]),
        num('timeout_multiplier', 'Timeout multiplier', {min: 0, max: 4, step: 'any', placeholder: profile(run.timeout_multiplier)}),
        field('Harness setup', select(null, [['', `profile (${run.prebake || 'prefer'})`],
          ['prefer', 'prefer: pre-baked, else stock'], ['require', 'require: pre-baked only'],
          ['off', 'off: stock install']], f.run.prebake, {'data-k': 'run.prebake'}),
        {notes: [note('A pre-baked image per task, or Harbor\'s install in each trial')]}),
        field('Setup limit ×', h('input', {type: 'text', class: 'mono', value: f.run.setup_timeout_multiplier,
          'data-k': 'run.setup_timeout_multiplier', inputmode: 'decimal', spellcheck: 'false', autocomplete: 'off',
          placeholder: `${run.setup_timeout_multiplier ?? 'auto'} (profile)`}),
        {notes: [note('auto, or times Harbor\'s 360 s')]}),
        num('env_build_timeout_multiplier', 'Image pull limit ×',
          {min: 0, max: 4, step: 'any', placeholder: profile(run.env_build_timeout_multiplier)},
          [note('Times Harbor\'s 600 s')]),
        field('Reuse measured trials', select(null, [['off', 'Off'], ['measured', 'On']], f.reuse, {'data-k': 'reuse'}),
          {notes: [note('Skip tasks the ledger already has for this exact configuration')]})),
      h('p', {class: 'dz-hint'}, 'Leave a field empty to use the profile default shown in it.'));
  }

  function armsSection(f) {
    return section('Arms',
      h('label', {class: 'dz-nobase', title: 'Compare with the ledger\'s baseline cell on the same model instead'},
        h('input', {type: 'radio', name: 'dz-baseline', value: '', checked: f.baselineKey === null, 'data-k': 'baseline'}),
        h('span', null, 'No baseline arm')),
      h('div', {class: 'dz-arms'}),
      h('div', {class: 'actions'}, h('button', {type: 'button', class: 'btn ghost small dz-add-arm', onclick: () => addArm()}, '+ Add arm')));
  }

  /** Redraws the arm rows from the model (adding or removing one); nothing typed is lost. */
  function renderArms() {
    const box = $('#dz-editor .dz-arms');
    if (!box || !st.ed) return;
    const f = st.ed.form;
    fill(box, f.arms.length
      ? f.arms.map((a) => armRow(a, f))
      : h('p', {class: 'muted dz-no-arms'}, 'No arms yet. Each arm is one harness on one model; add one below.'));
    const none = $('#dz-editor .dz-nobase input');
    if (none) none.checked = f.baselineKey === null;
    for (const a of f.arms) updateArmNotes(a);
  }

  function armRow(a, f) {
    const cfg = ui.state.config;
    const k = (name) => `arm.${a.key}.${name}`;
    const harnesses = allowedHarnesses().map((x) => [x.agent, x.kind === 'import_path' ? `${x.family} (custom)` : x.family]);
    return h('div', {class: f.baselineKey === a.key ? 'dz-arm is-base' : 'dz-arm', 'data-key': a.key},
      h('label', {class: 'dz-base', title: 'Baseline: the arm the others are compared with'},
        h('input', {type: 'radio', name: 'dz-baseline', value: String(a.key), checked: f.baselineKey === a.key, 'data-k': 'baseline',
          'aria-label': 'Baseline arm'}),
        h('span', null, 'BASE')),
      h('div', {class: 'dz-arm-fields'},
        field('Name', h('input', {type: 'text', class: 'mono', value: a.name, maxlength: 32, spellcheck: 'false', autocomplete: 'off',
          'data-k': k('name'), title: 'Lowercase letters, digits and -, starting with a letter'})),
        field('Harness', select(null, harnesses, a.agent, {'data-k': k('agent')}), {notes: [h('span', {class: 'dz-agent'})]}),
        field('Provider', select(null, cfg.providers.map((p) => [p.name, p.enabled ? p.name : `${p.name} (off)`]), a.provider,
          {'data-k': k('provider')})),
        h('div', {class: 'dz-model'},
          field('Model', h('input', {type: 'text', class: 'mono', value: a.model, list: 'dz-model-options', spellcheck: 'false',
            autocomplete: 'off', 'data-k': k('model')}), {notes: [h('small', {class: 'dz-harbor'})]})),
        field('Effort', h('span', {class: 'dz-effort'}, effortControl(a, f))),
        field('Tasks', h('button', {type: 'button', class: 'btn ghost small dz-subset', onclick: () => editSubset(a.key)}),
          {notes: [h('small', {class: 'dz-arm-reuse'})]})),
      h('button', {type: 'button', class: 'btn ghost small dz-remove', title: 'Remove this arm', 'aria-label': 'Remove this arm',
        onclick: () => removeArm(a.key)}, '✕'),
      h('div', {class: 'dz-arm-options'},
        h('span', {class: 'chips dz-opts'}),
        h('button', {type: 'button', class: 'btn ghost small', onclick: (e) => editOptions(a.key, e.currentTarget)}, 'Edit options')));
  }

  /** The effort control follows the harness schema: levels, any text ([]), or only inherit and omit (null). */
  function effortControl(a, f, error = '') {
    const schema = ui.schemas.get(a.agent);
    const attrs = {'data-k': `arm.${a.key}.effort`};
    const fixed = [['inherit', `inherit (${f.run.effort || ui.state.config.run.effort})`], ['omit', 'omit (harness default)']];
    if (error) {
      return [h('input', {type: 'text', class: 'mono', value: a.effort, list: 'effort-any', placeholder: 'inherit', spellcheck: 'false', autocomplete: 'off', ...attrs}),
        h('small', {class: 'bad-text'}, `Options did not load: ${error}`)];
    }
    if (!schema) {
      const agent = a.agent;
      loadSchema(agent).then(() => refreshEffort(a.key, agent), (err) => refreshEffort(a.key, agent, err.message));
      return select(null, fixed, a.effort, {...attrs, disabled: true, title: 'Loading harness options…'});
    }
    const levels = schema.effort_levels;
    if (levels === null) return select(null, fixed, a.effort, {...attrs, title: 'This harness has no effort levels'});
    if (levels.length === 0) {
      return h('input', {type: 'text', class: 'mono', value: a.effort, list: 'effort-any', placeholder: 'inherit', spellcheck: 'false',
        autocomplete: 'off', title: 'Any level text is accepted', ...attrs});
    }
    return select(null, [...fixed, ...levels.map((l) => [l, l])], a.effort, attrs);
  }

  function refreshEffort(key, agent = null, error = '') {
    const arm = armByKey(key);
    const row = armRowEl(key);
    if (!arm || !row || (agent && arm.agent !== agent)) return;
    const cell = row.querySelector('.dz-effort');
    if (!cell || cell.contains(document.activeElement)) return;
    fill(cell, effortControl(arm, st.ed.form, error));
    updateArmNotes(arm);  // the option chips can now tell known options from others
  }

  /** The small read-only parts of one arm row: the import path, Harbor's model id, the task count, the option chips. */
  function updateArmNotes(a) {
    const row = armRowEl(a.key);
    if (!row || !st.ed) return;
    const cfg = ui.state.config;
    const f = st.ed.form;
    const harness = allowedHarnesses().find((x) => x.agent === a.agent);
    fill(row.querySelector('.dz-agent'), harness && harness.kind === 'import_path'
      ? [tag('custom'), h('small', {class: 'mono dz-path-text'}, a.agent)]
      : !harness && h('small', {class: 'warn-text'}, 'Not a harness of this profile'));
    const provider = cfg.providers.find((p) => p.name === a.provider);
    row.querySelector('.dz-harbor').textContent = `Harbor gets ${provider ? provider.model_prefix : ''}${a.model.trim()}`;
    row.querySelector('.dz-base input').setAttribute('aria-label', `Baseline: ${a.name.trim() || 'this arm'}`);
    row.querySelector('.dz-remove').setAttribute('aria-label', `Remove arm ${a.name.trim()}`);
    const subset = row.querySelector('.dz-subset');
    if (a.tasks === null) {
      subset.textContent = 'all';
    } else {
      const inside = a.tasks.filter((t) => f.tasks.includes(t)).length;
      const outside = a.tasks.length - inside;
      subset.textContent = `${inside} of ${f.tasks.length}${outside ? ` · ${outside} outside` : ''}`;
    }
    subset.title = a.tasks === null ? 'Runs every task of the design' : a.tasks.join(', ');
    updateArmReuse(a);
    const schema = ui.schemas.get(a.agent);
    const known = schema ? new Set(schema.fields.map((x) => x.name)) : null;
    const entries = Object.entries(a.options || {});
    fill(row.querySelector('.dz-opts'), entries.length
      ? entries.map(([name, value]) => {
        const odd = known && !known.has(name);
        return tip(h('span', {class: odd ? 'chip dz-odd' : 'chip'}, clip(`${name}=${shown(value)}`, 48), odd && h('span', {class: 'src'}, 'not an option')),
          name, [[JSON.stringify(value), 'value'], odd ? `Not an option of ${a.agent}` : null].filter(Boolean));
      })
      : h('span', {class: 'faint dz-none'}, 'No options'));
  }

  /** Under an arm's task subset: what the check found it reuses, in a few words (the check panel has the detail). */
  function updateArmReuse(a) {
    const row = armRowEl(a.key);
    const el = row && row.querySelector('.dz-arm-reuse');
    if (!el) return;
    const r = st.check && st.check.reuse;
    const x = r ? r.arms.find((y) => y.arm === a.name.trim()) || null : null;
    el.classList.toggle('ok-text', Boolean(x && x.tasks_reused));
    el.textContent = !x ? ''
      : x.skipped ? `Skipped: all ${x.tasks_reused} reused`
        : x.tasks_reused ? `${x.tasks_reused} reused · runs ${x.tasks_to_run}`
          : x.harness_version === 'unpinned' ? 'Not reused: version not pinned' : 'None reused';
    el.title = x ? reuseArmText(x) : '';
  }

  function refreshSubsets() {
    if (st.ed) for (const a of st.ed.form.arms) updateArmNotes(a);
  }

  function renderNotices() {
    const box = $('#dz-editor .dz-notices');
    const ed = st.ed;
    if (!box || !ed) return;
    const b = ed.base;
    const newest = b ? latestVersion(ed) : null;
    const small = (label, onclick, cls = 'btn ghost small') => h('button', {type: 'button', class: cls, onclick: (e) => onclick(e.currentTarget)}, label);
    fill(box,
      st.pending && h('div', {class: 'banner warn dz-banner'},
        h('span', null, `Asked to open ${st.pending.id}${st.pending.version ? ` v${st.pending.version}` : ''}. Your unsaved edits are still here.`),
        h('span', {class: 'actions'},
          small('Discard and open', () => {
            const want = st.pending;
            st.pending = null;
            load(want.id, want.version, true);
          }),
          small('Keep editing', () => {
            st.pending = null;
            renderNotices();
          }))),
      st.conflict && h('div', {class: 'banner bad dz-banner'},
        h('span', null, st.conflict, h('br'), h('small', null, 'Your edits stay in the form. Reload latest puts them on the newest version.')),
        h('span', {class: 'actions'}, small('Reload latest', reloadLatest, 'btn small'))),
      st.merged && h('div', {class: 'note-strip dz-banner'},
        h('span', null, `Your edits are now on v${st.merged.version}.`,
          st.merged.both.length > 0 && ` v${st.merged.version} also changed ${listText(st.merged.both)}; your values were kept. Check them before saving.`),
        h('span', {class: 'actions'}, small('OK', () => {
          st.merged = null;
          renderNotices();
        }))),
      b && !st.conflict && b.version !== newest && h('div', {class: 'note-strip dz-banner'},
        h('span', null, `This is v${b.version}; v${newest} is the latest. A version saved from here would conflict with it.`),
        h('span', {class: 'actions'},
          small(`Open v${newest}`, () => pick(ed.id, newest)),
          small(`Start v${newest + 1} from these values`, restoreAsNew))),
      b && b.archived && h('div', {class: 'banner warn dz-banner'},
        h('span', null, 'Archived: left out of All, and it cannot be queued or get new versions.'),
        h('span', {class: 'actions'}, small('Unarchive', unarchive))));
  }

  /** In-place parts of the editor that come from the check: suggested budget, peak containers, the task-order warning. */
  function updateEditorNotes() {
    const box = $('#dz-editor');
    if (!box || !st.ed) return;
    const c = st.check;
    const est = c && c.estimate;
    const suggest = box.querySelector('.dz-suggest');
    if (suggest) {
      const v = est ? est.run.suggested_budget_usd : null;
      fill(suggest,
        isNum(v) && linkButton(`Suggested budget: ${money(v)}`, () => useBudget(v), 'About p90 plus a margin, from the estimate'),
        est && isNum(est.run.p_within_budget) && h('small', null, ` ${pct(est.run.p_within_budget)} chance within ${money(est.run.budget_usd)}`));
    }
    const peak = box.querySelector('.dz-peak');
    const n = c && c.plan ? c.plan.peak_containers : est ? est.run.peak_containers : null;
    if (peak) peak.textContent = isNum(n) ? `PEAK ${plural(n, 'CONTAINER')}` : '';
    const order = box.querySelector('.dz-order');
    if (order) {
      const ws = c && c.validation ? c.validation.warnings.filter((w) => /seeded-sample order|outside the seeded sample/.test(w)) : [];
      fill(order, ws.map((w) => h('p', {class: 'warn-text dz-line'}, h('span', null, '!'), h('span', null, w))));
    }
    for (const a of st.ed.form.arms) updateArmReuse(a);
  }

  /** Buttons, the unsaved mark and the status tag, updated in place (never rebuilt, so focus stays). */
  function renderActions() {
    const ed = st.ed;
    const edited = isEdited();
    const box = $('#dz-editor');
    if (box && ed) {
      const save = box.querySelector('.dz-save');
      const archived = Boolean(ed.base && ed.base.archived);
      if (save) {
        save.disabled = !edited || archived;
        save.title = archived ? 'Unarchive it to save a new version' : edited ? '' : 'Nothing to save yet';
        save.classList.toggle('dz-ready', edited && !archived);
      }
      const discardButton = box.querySelector('.dz-discard');
      if (discardButton) discardButton.hidden = !edited && Boolean(ed.base);
      const archiveButton = box.querySelector('.dz-archive');
      if (archiveButton) archiveButton.textContent = archived ? 'Unarchive' : 'Archive';
      const unsaved = box.querySelector('.dz-unsaved');
      if (unsaved) unsaved.hidden = !edited;
      const status = box.querySelector('.dz-status');
      const s = liveStatus(ed);
      if (status && status.dataset.s !== s) {
        fill(status, designTag(s));
        status.dataset.s = s;
      }
    }
    const block = queueBlock();
    for (const sel of ['#dz-check .dz-enqueue', '#dz-check .dz-start']) {
      const button = $(sel);
      if (button && !button.classList.contains('dz-busy')) button.disabled = Boolean(block);
    }
    const noteEl = $('#dz-check .dz-arm-note');
    if (noteEl) {
      noteEl.textContent = block || 'Ready. A queued design bills your provider accounts when the queue runs it.';
      noteEl.className = block ? 'dz-arm-note' : 'dz-arm-note ok-text';
    }
  }

  // Check ------------------------------------------------------------------

  /** After any edit: the saved check while the editor matches its version, otherwise a live check 800 ms after typing stops. */
  function scheduleCheck({now = false} = {}) {
    const ed = st.ed;
    if (!ed) return;
    if (ed.base && !isEdited()) {
      checkLater.cancel();
      fetchCheck(Promise.resolve(null));  // a live check still in flight was for edits since undone
      showSavedCheck();
      return;
    }
    st.checking = true;
    renderCheckStatus();
    checkLater();
    if (now) checkLater.flush();
  }

  function showSavedCheck() {
    Object.assign(st, {check: st.ed && st.ed.base ? st.ed.base.check : null, live: false, checking: false, checkError: '', formErrors: []});
    renderCheck();
  }

  async function runCheck() {
    const ed = st.ed;
    if (!ed || (ed.base && !isEdited())) return;
    const {spec, errors} = buildSpec(ed);
    let result;
    try {
      result = await fetchCheck(api('POST', '/api/designs/check', {spec}));
    } catch (err) {
      if (st.ed !== ed) return;
      st.checking = false;
      st.checkError = err.status === 503 ? `${err.message} Checking again in a moment.` : err.message;
      if (err.status === 503) {
        setTimeout(() => {
          if (st.ed === ed && (isEdited() || !ed.base)) scheduleCheck();
        }, 3000);
      }
      renderCheck();
      return;
    }
    if (result === STALE || st.ed !== ed) return;
    Object.assign(st, {check: result, live: true, checking: false, checkError: '', formErrors: errors});
    renderCheck();
  }

  function renderCheckStatus() {
    const panelEl = $('#dz-check');
    if (!panelEl) return;
    panelEl.classList.toggle('dz-dim', st.checking);
    const c = st.check;
    const v = c && c.validation;
    const problems = (v ? v.problems.length : 0) + st.formErrors.length;
    let status;
    if (!st.ed) status = [led(), h('span', {class: 'muted'}, 'STANDING BY')];
    else if (st.checking) status = [led('warn live'), h('span', {class: 'warn-text'}, 'CHECKING')];
    else if (st.checkError) status = [led('bad'), h('span', {class: 'bad-text'}, 'CHECK FAILED')];
    else if (!v) status = [led(), h('span', {class: 'muted'}, 'NOT CHECKED')];
    else if (problems) status = [led('bad'), h('span', {class: 'bad-text'}, plural(problems, 'PROBLEM').toUpperCase())];
    else status = [led('ok'), h('span', {class: 'ok-text'}, v.warnings.length ? `VALID · ${plural(v.warnings.length, 'WARNING')}` : 'VALID')];
    fill(panelEl.querySelector('.dz-check-status'), status);
  }

  function renderCheck() {
    const panelEl = $('#dz-check');
    if (!panelEl) return;
    renderCheckStatus();
    renderActions();
    updateEditorNotes();
    const body = panelEl.querySelector('.dz-check-body');
    const ed = st.ed;
    if (!ed) {
      fill(body, h('p', {class: 'muted'}, 'Open or start a design to see its check: problems, warnings and the cost estimate.'));
      return;
    }
    const c = st.check;
    const open = new Set([...body.querySelectorAll('details[open][data-d]')].map((d) => d.dataset.d));
    const v = c && c.validation;
    const est = c && c.estimate;
    const plan = c && c.plan;
    const issues = [
      ...st.formErrors.map((text) => ['bad', text]),
      ...(v ? v.problems.map((text) => ['bad', text]) : []),
      ...(v ? v.warnings.map((text) => ['warn', text]) : []),
    ];
    fill(body,
      h('p', {class: 'dz-check-src'},
        st.live ? 'Live check of your edits' : ed.base ? `Saved check of v${ed.base.version}` : 'Not checked yet',
        c && c.checked_at && ` · ${clockTime(c.checked_at)}`,
        v && v.problems.length > 0 && h('span', {class: 'bad-text'}, ' · problems block queueing')),
      st.checkError && h('p', {class: st.checkError.includes('again') ? 'warn-text' : 'bad-text'}, st.checkError),
      issues.length > 0 && h('ul', {class: 'issues dz-issues'}, issues.map(([kind, text]) => issue(kind, text))),
      plan && h('p', {class: 'dz-line muted'}, h('span', null, '·'),
        h('span', null, `${plural(plan.trials, 'trial')} in ${plural(plan.arms.length, 'arm')} · peak ${plural(plan.peak_containers, 'container')}`
          + `${plan.skipped_arms && plan.skipped_arms.length ? ` · ${plural(plan.skipped_arms.length, 'arm')} skipped` : ''}`)),
      reuseBlock(c, ed, open),
      est ? estimateBlock(est, plan) : c && c.estimate_note && h('p', {class: 'muted dz-est-note'}, c.estimate_note),
      plan && plan.arms.some((a) => a.command) && h('details', {'data-d': 'cmd', open: open.has('cmd')},
        h('summary', null, 'Harbor commands'),
        plan.arms.map((a) => [h('div', {class: 'cmd-name'}, a.name), h('pre', {class: 'cmd'}, a.command || DASH),
          a.reserve_note && h('small', {class: 'dz-reserve'}, a.reserve_note)])),
      est && est.assumptions && est.assumptions.length > 0 && h('details', {'data-d': 'assume', open: open.has('assume')},
        h('summary', null, `Assumptions (${est.assumptions.length})`),
        h('ul', {class: 'dz-assume'}, est.assumptions.map((text) => h('li', null, text)))));
  }

  /** An estimate's task races: tasks x arms, the longest task's image build and the peak disk. */
  function raceText(race) {
    const build = race.build_wall_min_max;
    const gb = race.peak_disk_gb;
    return `${plural(race.tasks, 'task')} × ${plural(race.arms, 'arm')}`
      + (isNum(build) && build > 0 ? ` · builds ≤ ${duration(build)} per task` : '')
      + (isNum(gb) ? ` · peak ~${gb < 10 ? gb.toFixed(1) : Math.ceil(gb)} GB` : '');
  }

  function estimateBlock(est, plan) {
    const r = est.run || {};
    const t = r.total || {};
    const uncapped = r.uncapped_arms || [];
    const allUncapped = plan ? uncapped.length >= plan.arms.length : false;
    return h('div', {class: 'dz-estimate'},
      eyebrow('Estimate'),
      h('div', {class: 'dz-totals'}, [['p10', t.p10], ['p50', t.p50], ['p90', t.p90]].map(([k, value]) => h('div', {class: k === 'p50' ? 'main' : null},
        h('span', null, k), h('b', null, money(value))))),
      h('dl', {class: 'params dz-est'},
        param('Within budget', isNum(r.p_within_budget) ? `${pct(r.p_within_budget)} chance within ${money(r.budget_usd)}` : DASH),
        param('Worst case, capped', allUncapped ? DASH : money(r.worst_case_capped_usd)),
        param('In-flight allowance', money(r.inflight_allowance_usd)),
        param('Wall time', r.wall_minutes ? `${duration(r.wall_minutes.p50)} p50 · ${duration(r.wall_minutes.p90)} p90` : DASH),
        param('Peak containers', count(r.peak_containers)),
        // Under schedule tasks: tasks x arms, the longest task's image build and the peak disk (one task's images).
        r.race && param('Task races', raceText(r.race)),
        param('Suggested budget', isNum(r.suggested_budget_usd) ? linkButton(money(r.suggested_budget_usd), () => useBudget(r.suggested_budget_usd),
          'Use it as the budget') : DASH),
        // Reused trials are not in the figures above: they are not run, and their own measured cost is not spent.
        est.reuse && h('div', {class: 'dz-wide'}, h('dt', null, 'Reused, not run'),
          h('dd', null, `${plural(est.reuse.trials, 'trial')} · ${money(est.reuse.cost_avoided_usd)} not spent`
            + `${est.reuse.unpriced_tasks ? ` (${plural(est.reuse.unpriced_tasks, 'task')} without a cost)` : ''}`))),
      uncapped.length > 0 && h('p', {class: 'warn-text dz-line'}, h('span', null, '!'),
        h('span', null, `No dollar cap: ${listText(uncapped)}. The run budget still applies after every trial.`)),
      (est.warnings || []).map((w) => h('p', {class: 'warn-text dz-line'}, h('span', null, '!'), h('span', null, w))),
      est.arms && est.arms.length > 0 && h('p', {class: 'dz-check-src'}, 'Per arm · p50 / p90 · minutes include preflight'),
      est.arms && est.arms.length > 0 && h('div', {class: 'scroll'}, h('table', {class: 'grid-t compact dz-est-table'},
        thead('Arm', thNum('Trials'), thNum('Per trial'), thNum('Total'), thNum('Minutes'), 'Basis'),
        h('tbody', null, est.arms.map((a) => h('tr', null,
          h('td', {class: 'mono'}, a.arm),
          tdNum(count(a.trials)),
          tdNum(a.per_trial ? `${money(a.per_trial.p50)} / ${money(a.per_trial.p90)}` : DASH),
          tdNum(a.total ? `${money(a.total.p50)} / ${money(a.total.p90)}` : DASH),
          tdNum(a.arm_minutes ? `${minutes(a.arm_minutes.p50)} / ${minutes(a.arm_minutes.p90)}` : DASH, 'Arm minutes, preflight included'),
          h('td', null, basisTag(a))))))),
      (est.arms || []).flatMap((a) => [
        ...(a.unpriced_models || []).map((m) => h('p', {class: 'warn-text dz-line'}, h('span', null, '!'), h('span', null, `${a.arm}: ${m} has no price`))),
        ...(a.notes || []).map((text) => h('p', {class: 'dz-line muted'}, h('span', null, '·'), h('span', null, `${a.arm}: ${text}`))),
      ]),
      est.pairing && est.pairing.length > 0
        && [eyebrow(pairsTitle(est.pairing)), h('ul', {class: 'dz-pairs'}, est.pairing.map(pairingLine))]);
  }

  /** "Pairs on 27 tasks from config/tasks.txt", the REPORT's default scope; an older check's says nothing. */
  function pairsTitle(pairing) {
    const scope = (pairing.find((p) => p.scope) || {}).scope;
    if (scope) return `Pairs on ${plural(scope.tasks, 'task')} from ${scope.tasks_file}`;
    return pairing.some((p) => 'scope' in p) ? 'Pairs on every task in the ledger' : 'Pairs';
  }

  function basisTag(a) {
    const what = a.basis === 'measured' ? `measured n=${a.evidence ? a.evidence.trials : '?'}` : BASIS[a.basis] || a.basis || 'unknown';
    const rows = [a.basis_detail || null];
    if (a.evidence) {
      rows.push([count(a.evidence.trials), 'trials seen'], [count(a.evidence.tasks), 'tasks seen'],
        [count(a.evidence.design_tasks_observed), 'of these tasks seen'], [count((a.evidence.runs || []).length), 'runs']);
    }
    rows.push([isNum(a.cap_usd) ? money(a.cap_usd) : 'none', 'dollar cap']);
    return tip(tag(a.confidence ? `${what} · ${a.confidence}` : what, CONFIDENCE_TAG[a.confidence] || ''), 'Basis', rows.filter(Boolean), true);
  }

  const factor = (v) => (isNum(v) ? v.toFixed(2) : DASH);

  function pairingLine(p) {
    const base = p.baseline_cell || {};
    const baseName = base.family || base.arm || 'baseline';
    const metric = p.metric || 'cost';
    const now = p.multiple_now || {};  // {value, ci, n}: arm ÷ baseline, or null below 3 paired tasks
    const ci = Array.isArray(now.ci) && isNum(now.ci[0]) && isNum(now.ci[1]) ? now.ci : null;
    return tip(h('li', null,
      `${p.arm} vs ${baseName}: ${p.pairs_now} → ${plural(p.pairs_after, 'paired task')}; `
        + `interval ×/÷${factor(p.ci_factor_now)} → ×/÷${factor(p.ci_factor_after)}`,
      isNum(now.value) && h('span', {class: 'muted'}, ` · now ${times(now.value)} ${baseName}'s ${metric}`),
      p.scope && p.scope.outside > 0 && h('span', {class: 'muted'},
        ` · ${plural(p.scope.outside, 'task')} outside ${p.scope.tasks_file} not counted`)),
    `${p.arm} · ${metric}`, [
      [base.arm || DASH, 'baseline arm'], [`${base.provider || DASH}:${base.model || DASH}`, 'model'],
      isNum(now.value) && [ci ? `${multiple(ci[0])}–${times(ci[1])}` : DASH,
        `now, ${p.arm} ÷ ${baseName}, ${plural(now.n, 'pair')}`],
      [isNum(p.sd_log) ? p.sd_log.toFixed(2) : DASH, `log spread (${p.sd_source || DASH})`],
    ].filter(Boolean), true);
  }

  // Reused trials ------------------------------------------------------------
  //
  // A design with reuse "measured" is resolved against the ledger by every check and every save: each arm leaves
  // out the tasks the ledger already has `attempts` usable trials of for its exact configuration, and the saved
  // version records them in spec.reused with the trials and runs it reuses. check.reuse is the summary: per arm
  // what it reuses and its cost, or why it reuses nothing, and for a saved check whether saving again would reuse
  // something else (current, changes).

  /** The reuse section of D-03, or null when the design does not reuse. */
  function reuseBlock(c, ed, open) {
    const spec = !st.live && ed.base ? ed.base.spec : null;  // a saved version: exactly what it runs and reuses
    const r = c && (c.reuse || (spec && spec.reuse ? savedReuse(spec) : null));
    if (!r) return null;
    const avoided = isNum(r.cost_avoided_usd) && r.cost_avoided_usd > 0 ? ` · about ${money(r.cost_avoided_usd)} not spent` : '';
    return h('div', {class: 'dz-reuse'},
      h('div', {class: 'section-head'}, eyebrow(spec ? `Reused in v${ed.base.version}` : 'Reused when saved'),
        h('span', {class: 'dz-reuse-total'}, `${count(r.trials_reused)} of ${plural(r.trials_asked, 'trial')}${avoided}`)),
      spec && r.current === false && driftStrip(ed),
      h('ul', {class: 'dz-reuse-arms'}, r.arms.map((a) => reuseArmItem(a, spec, open))),
      (r.notes || []).map((text) => h('p', {class: 'dz-line muted'}, h('span', null, '·'), h('span', null, text))));
  }

  /** check.reuse rebuilt from a saved spec alone, for a check that could not read the ledger: no costs, no reasons. */
  function savedReuse(spec) {
    const attempts = spec.attempts || 1;
    const arms = spec.arms.map((a) => {
      const reused = reusedOf(spec, a.name).map((e) => ({...e, cost_avoided_usd: null, lower_bound: 0}));
      const runs = Array.isArray(a.tasks) ? a.tasks.length : spec.tasks.length;
      return {arm: a.name, family: a.family, config_hash: null, harness_version: null, reason: null, tasks_asked: runs + reused.length,
        tasks_reused: reused.length, tasks_to_run: runs, trials_reused: reused.length * attempts, cost_avoided_usd: null,
        skipped: reused.length > 0 && !runs, cell_trials: null, reused, short: []};
    });
    const reused = sum(arms, (a) => a.trials_reused);
    return {trials_reused: reused, trials_asked: reused + sum(arms, (a) => a.tasks_to_run) * attempts, cost_avoided_usd: null,
      skipped_arms: arms.filter((a) => a.skipped).map((a) => a.arm), arms, current: null, changes: [],
      notes: ['The ledger could not be read, so this lists what the version saved, without costs.']};
  }

  /** The ledger now resolves this saved version differently: it still runs as saved; a new version reuses the latest. */
  function driftStrip(ed) {
    const b = ed.base;
    const newest = b.version === latestVersion(ed) && !b.archived;
    return h('div', {class: 'note-strip dz-banner'},
      h('span', null, 'The ledger has changed since this version was saved. Save a new version to reuse the latest trials.',
        h('br'), h('small', null, `v${b.version} still runs exactly the tasks it saved.`)),
      newest && h('span', {class: 'actions'}, h('button', {type: 'button', class: 'btn small',
        onclick: (e) => save(e.currentTarget, 'Reused trials resolved again against the ledger')}, `Save v${b.version + 1}`)));
  }

  /** The tasks an arm runs: the saved version's own list, or for edits the arm's tasks less what the check reuses. */
  function tasksToRun(a, spec) {
    if (spec) {
      const arm = spec.arms.find((x) => x.name === a.arm);
      return arm ? (Array.isArray(arm.tasks) ? arm.tasks : spec.tasks) : [];
    }
    const f = st.ed.form;
    const arm = f.arms.find((x) => x.name.trim() === a.arm);
    const reused = new Set(a.reused.map((e) => e.task));
    return arm ? (arm.tasks || f.tasks).filter((t) => !reused.has(t)) : [];
  }

  function reuseArmItem(a, spec, open) {
    const toRun = tasksToRun(a, spec);
    const key = `reuse:${a.arm}`;
    return h('li', {class: 'dz-reuse-arm'},
      h('div', {class: 'dz-reuse-head'},
        h('b', {class: 'mono'}, a.arm),
        a.skipped && tag('skipped', 'ok'),
        a.harness_version && tag(`version ${REUSE_VERSION[a.harness_version] || a.harness_version}`, a.harness_version === 'unpinned' ? 'warn' : ''),
        a.config_hash && h('span', {class: 'dz-hash mono', title: 'Configuration hash: the cell whose trials count'}, a.config_hash)),
      reuseArmNote(a, h('p', {class: a.tasks_reused ? 'dz-line ok-text' : 'dz-line muted'}, reuseArmText(a))),
      a.reused.length > 0 && h('details', {'data-d': key, open: open.has(key)},
        h('summary', null, `${plural(a.reused.length, 'reused task')}${isNum(a.cost_avoided_usd) ? ` · ${money(a.cost_avoided_usd)}` : ''}`),
        h('ul', {class: 'dz-reused'}, a.reused.map((e) => h('li', null,
          h('div', {class: 'dz-reused-task'},
            h('span', {class: 'mono'}, e.task),
            e.lower_bound > 0
              ? tip(h('span', {class: 'mono muted'}, `≥ ${money(e.cost_avoided_usd)}`), 'Cut short',
                [`${plural(e.lower_bound, 'reused trial')} stopped early, so the usage is a lower bound.`], true)
              : h('span', {class: 'mono muted'}, money(e.cost_avoided_usd))),
          h('div', {class: 'dz-reused-from'}, e.trials.map(trialRef), e.runs.map(runRef)))))),
      toRun.length > 0 && a.tasks_reused > 0 && h('details', {'data-d': `${key}:run`, open: open.has(`${key}:run`)},
        h('summary', null, `${plural(toRun.length, 'task')} to run`),
        h('p', {class: 'dz-reuse-run mono'}, toRun.join(', '))),
      a.short.length > 0 && h('p', {class: 'dz-line warn-text'}, h('span', null, '!'),
        h('span', null, `Too few usable trials, so ${a.short.length === 1 ? 'it runs' : 'these run'} in full: `
          + a.short.map((s) => `${s.task} (${s.usable} usable)`).join(', '))),
      (a.host_busy || []).length > 0 && h('p', {class: 'dz-line warn-text'}, h('span', null, '!'),
        h('span', null, `All usable trials ran while the host CPU was busy, so the time is unknown and `
          + `${a.host_busy.length === 1 ? 'it runs' : 'these run'} again: `
          + a.host_busy.map((s) => `${s.task} (${s.usable} usable)`).join(', '))));
  }

  /** A reused trial: its trial name, the whole trial key in the tooltip. */
  function trialRef(key) {
    const name = String(key).split('/').pop();
    return tip(h('span', {class: 'chip'}, name), 'Trial', [[key, 'trial key']], true);
  }

  /** A run trials were reused from: it opens when it is one of this profile's runs, otherwise it is only named. */
  function runRef(key) {
    const [profile, runId] = String(key).split('/');
    const own = runId && profile === ((ui.state && ui.state.config.profile) || 'default') && ui.state.runs.some((r) => r.id === runId);
    return own
      ? h('button', {type: 'button', class: 'chip dz-run-ref', title: 'Open this run', onclick: () => openRun(runId)}, key)
      : h('span', {class: 'chip', title: 'Run'}, key);
  }

  // Saving and the other actions ---------------------------------------------

  /** `fallbackNote`: the change note when the form has none, for a save that only resolves the reused trials again. */
  async function save(button, fallbackNote = '') {
    const ed = st.ed;
    if (!ed) return;
    const f = ed.form;
    if (!f.title.trim()) {
      toast('A title is required on every save.', true);
      const title = $('#dz-editor [data-k="title"]');
      if (title) title.focus();
      return;
    }
    const {spec, errors} = buildSpec(ed);
    if (errors.length) {
      toast(errors.join('; '), true);
      return;
    }
    const body = {title: f.title.trim(), purpose: f.purpose.trim(), rationale: f.rationale.trim(), change_note: f.change_note.trim() || fallbackNote, spec};
    await busy(button, async () => {
      let detail;
      try {
        detail = ed.base
          ? await api('POST', `/api/designs/${enc(ed.id)}/versions`, {base_version: ed.base.version, ...body})
          : await api('POST', '/api/designs', body);
      } catch (err) {
        // Someone saved a newer version: keep every edit and offer to put them on it.
        if (err.status === 409 && /was saved (since you read|by someone else)/.test(err.message)) {
          st.conflict = err.message;
          renderNotices();
          return;
        }
        throw err;
      }
      if (st.ed !== ed) return;
      await opened(detail, `Saved ${detail.title} v${detail.version}`);
    });
  }

  /** After the stale-version 409: the latest version with your edits on it. A field you changed keeps your value;
      every other field takes the latest one. Fields both sides changed are listed so you can check them. */
  async function reloadLatest(button) {
    const ed = st.ed;
    if (!ed || !ed.base) return;
    await busy(button, async () => {
      const {meta, detail} = await fetchVersion(ed.id, null);
      if (st.ed !== ed) return;
      const mine = ed.form;
      const old = formFrom(ed.base);
      const out = formFrom(detail);
      const start = JSON.stringify(comparable(out));
      const both = [];
      const take = (label, get, apply) => {
        const m = JSON.stringify(get(comparable(mine)));
        const o = JSON.stringify(get(comparable(old)));
        if (m === o) return;
        const t = JSON.stringify(get(comparable(out)));
        if (t !== o && t !== m) both.push(label);
        apply();
      };
      for (const k of ['title', 'purpose', 'rationale', 'attempts']) take(k, (c) => c[k], () => { out[k] = mine[k]; });
      take('tasks', (c) => c.tasks, () => {
        out.tasks = [...mine.tasks];
        out.extra = mine.extra;
      });
      for (const k of RUN_FIELDS) take(`run.${k}`, (c) => c.run[k], () => { out.run[k] = mine.run[k]; });
      take('reuse', (c) => c.reuse, () => { out.reuse = mine.reuse; });
      const baselineMine = baselineName(mine);
      let baseline = baselineName(out);
      take('arms', (c) => c.arms, () => { out.arms = mine.arms; });
      take('baseline_arm', (c) => c.baseline, () => { baseline = baselineMine; });
      const kept = out.arms.find((a) => a.name.trim() === baseline);
      out.baselineKey = kept ? kept.key : null;
      out.change_note = mine.change_note;
      st.ed = {id: ed.id, base: detail, meta, form: out, start, bad: new Set(ed.bad)};
      Object.assign(st, {conflict: '', merged: {version: detail.version, both}, pending: null, sig: sigOf(listEntry(ed.id))});
      renderEditor();
      scheduleCheck({now: true});
      renderList();
    });
  }

  /** An older version's values as the start of the next one: based on the latest, so it saves without a conflict. */
  async function restoreAsNew(button) {
    const ed = st.ed;
    if (!ed || !ed.base) return;
    await busy(button, async () => {
      const {meta, detail} = await fetchVersion(ed.id, null);
      if (st.ed !== ed) return;
      const form = ed.form;
      if (!form.change_note.trim()) form.change_note = `Back to v${ed.base.version}`;
      st.ed = {id: ed.id, base: detail, meta, form, start: JSON.stringify(comparable(formFrom(detail))), bad: new Set(ed.bad)};
      Object.assign(st, {conflict: '', merged: null, pending: null, sig: sigOf(listEntry(ed.id))});
      renderEditor();
      scheduleCheck({now: true});
    });
  }

  async function duplicate() {
    const ed = st.ed;
    if (!ed || !ed.base) return;
    const {spec, errors} = buildSpec(ed);
    if (errors.length) {
      toast(errors.join('; '), true);
      return;
    }
    const f = ed.form;
    const edited = isEdited();
    const detail = await formDialog({
      code: 'COPY', title: 'Duplicate this design', ok: 'Create copy',
      body: [
        field('Title', h('input', {type: 'text', name: 'title', required: true, maxlength: 80, autocomplete: 'off',
          value: `Copy of ${f.title.trim() || ed.base.title}`.slice(0, 80)})),
        h('p', {class: 'muted'}, edited
          ? `A new design from the spec in the editor, your unsaved edits included. ${ed.base.design_id} stays as saved.`
          : `A new design from v${ed.base.version} of ${ed.base.design_id}.`),
      ],
      read: (form) => api('POST', '/api/designs', {
        title: form.elements.title.value.trim(), purpose: f.purpose.trim(), rationale: f.rationale.trim(),
        change_note: `Copied from ${ed.base.design_id} v${ed.base.version}${edited ? ' with edits' : ''}`, spec,
      }),
    });
    if (detail) await opened(detail, `Created ${detail.design_id}`);
  }

  async function archive(button) {
    const ed = st.ed;
    if (!ed || !ed.base) return;
    const ok = await confirmAction({
      code: 'ARCHIVE', title: `Archive ${ed.base.title}?`, ok: 'Archive', tone: 'danger',
      body: 'It leaves the All list and cannot be queued or get new versions until you unarchive it. Every version is kept.',
    });
    if (!ok) return;
    await busy(button, async () => {
      await api('POST', `/api/designs/${enc(ed.id)}/archive`);
      toast(`Archived ${ed.id}`);
      await afterArchive(ed);
    });
  }

  async function unarchive(button) {
    const ed = st.ed;
    if (!ed || !ed.base) return;
    await busy(button, async () => {
      await api('POST', `/api/designs/${enc(ed.id)}/unarchive`);
      toast(`Unarchived ${ed.id}`);
      await afterArchive(ed);
    });
  }

  async function afterArchive(ed) {
    listPoll.now();
    if (st.ed !== ed) return;
    const data = await fetchVersion(ed.id, ed.base.version);
    refreshBase(ed, data.detail, data.meta);  // edits, if any, stay in the form
  }

  async function enqueue(start, button) {
    const ed = st.ed;
    if (queueBlock() || !ed || !ed.base) return;
    button.classList.add('dz-busy');
    button.disabled = true;
    try {
      await enqueueDesign(ed.base, {start});  // the ARM confirm; errors show inside it
    } finally {
      button.classList.remove('dz-busy');
      renderActions();
      // The button was disabled while the dialog was open, so the browser could not hand focus back to it.
      const lost = !document.activeElement || document.activeElement === document.body;
      if (lost && button.isConnected && !button.disabled) button.focus({preventScroll: true});
    }
  }

  async function fromConfig() {
    if (!(await leaveEdits())) return;
    const cfg = ui.state.config;
    const modes = Object.keys(cfg.modes);
    if (!modes.length || !cfg.arms.length) {
      toast('This profile has no modes or arms to start from.', true);
      return;
    }
    const mode = modes.includes(ui.mode) ? ui.mode : modes[0];
    const title = h('input', {type: 'text', name: 'title', maxlength: 80, autocomplete: 'off', placeholder: `${mode} from config`});
    const detail = await formDialog({
      code: 'CFG', title: 'New design from config arms', ok: 'Create design',
      body: [
        field('Mode', select('mode', modes.map((m) => [m, `${m} · ${plural(cfg.modes[m].n_tasks, 'task')} × ${cfg.modes[m].n_attempts}`]), mode, {
          onchange: (e) => { title.placeholder = `${e.target.value} from config`; },
        })),
        h('div', {class: 'field'}, h('span', {class: 'label'}, 'Arms'),
          h('div', {class: 'dz-checks'}, cfg.arms.map((a) => h('label', {class: 'dz-check'},
            h('input', {type: 'checkbox', name: 'arm', value: a.name, checked: a.enabled}),
            h('span', null, a.name, a.name === cfg.baseline_arm && h('span', {class: 'faint'}, ' · baseline')),
            h('small', {class: 'mono faint'}, `${a.family || a.agent} · ${a.provider}/${a.model}`))))),
        field('Title', title, {notes: [note('Optional')]}),
        h('label', {class: 'dz-check'}, h('input', {type: 'checkbox', name: 'reuse'}),
          h('span', null, 'Reuse measured trials'),
          h('small', {class: 'faint'}, 'Skip tasks the ledger already has for this exact configuration')),
        h('p', {class: 'muted'}, 'Saves version 1 of a new design from the mode\'s tasks and the arms as configured. Nothing runs.'),
      ],
      read: (form) => {
        const arms = [...form.querySelectorAll('input[name="arm"]:checked')].map((i) => i.value);
        if (!arms.length) throw new Error('Tick at least one arm.');
        const body = {mode: form.elements.mode.value, arms};
        if (form.elements.reuse.checked) body.reuse = 'measured';
        const text = form.elements.title.value.trim();
        if (text) body.title = text;
        return api('POST', '/api/designs/from-config', body);
      },
    });
    if (detail) await opened(detail, `Created ${detail.design_id} from config`);
  }

  async function fromRun() {
    if (!(await leaveEdits())) return;
    const runs = ui.state.runs || [];
    if (!runs.length) {
      toast('There are no runs in this profile yet.');
      return;
    }
    const title = h('input', {type: 'text', name: 'title', maxlength: 80, autocomplete: 'off', placeholder: `Run again: ${runs[0].id}`});
    const detail = await formDialog({
      code: 'RUN', title: 'New design from a past run', ok: 'Create design',
      body: [
        field('Run', select('run', runs.map((r) => [r.id,
          `${r.id} · ${statusLabel(r.status)} · ${money(r.spend_usd)} of ${money(r.budget_usd)}${r.source === 'design' ? ' · design' : ''}`]), runs[0].id, {
          onchange: (e) => { title.placeholder = `Run again: ${e.target.value}`; },
        })),
        field('Title', title, {notes: [note('Optional')]}),
        h('p', {class: 'muted'}, 'Saves version 1 of a new design with the spec that run executed. Nothing runs.'),
      ],
      read: (form) => {
        const body = {};
        const text = form.elements.title.value.trim();
        if (text) body.title = text;
        return api('POST', `/api/designs/from-run/${enc(form.elements.run.value)}`, body);
      },
    });
    if (detail) await opened(detail, `Created ${detail.design_id} from the run`);
  }

  function renderQueueSummary(force = false) {
    const el = $('#dz-queue-summary');
    if (el && (changed('designs:queue', [queueView(), stateBlock('queue')]) || force)) queueSummary(el);
  }

  // Registration -----------------------------------------------------------

  function render(state, force) {
    const root = $('#view-designs');
    if (!root.querySelector('.dz-layout')) {
      build(root);
      force = true;
    }
    if (st.filter === null) {
      const saved = prefs.get('designs.filter', 'all');
      st.filter = FILTERS.some(([value]) => value === saved) ? saved : 'all';
      markFilter();
    }
    renderList(force);
    const cfg = state.config;
    const cfgChanged = changed('designs:cfg', [cfg.arms, cfg.providers, cfg.tasks, cfg.run, cfg.modes]);
    if (st.wanted) {
      takeWanted();
    } else if (cfgChanged && !force && st.ed) {
      // Never rebuild edits; the live check reruns on the new config instead.
      if (isEdited() || !st.ed.base) scheduleCheck();
      else renderEditor();
    }
    renderActions();
    renderQueueSummary(force);
  }

  registerView('designs', {
    render,
    show() {
      if (st.ed || st.loading || st.loadError) return;
      const id = prefs.get('designs.open');  // the design open last time
      if (typeof id === 'string' && id) load(id, null, false, true);
    },
    onState(state, prev) {
      // New trials in the ledger change the estimate: recheck edits (a saved check reloads through the list). What
      // edits that reuse trials take also changes with an exclusion or an outcome, which only the ledger version shows.
      const now = stateBlock('ledger');
      const before = prev && prev.ledger && !prev.ledger.error ? prev.ledger : null;
      const changed = now && before && (now.last_ingest_at !== before.last_ingest_at
        || (st.ed && st.ed.form.reuse === 'measured' && now.version !== before.version));
      if (changed && st.ed && (isEdited() || !st.ed.base)) scheduleCheck();
    },
    /** params: {id, version}; a null version means the latest. */
    open(params) {
      if (!params || !params.id) return;
      const want = {id: params.id, version: params.version || null};
      const ed = st.ed;
      if (ed && ed.base && ed.id === want.id && !st.loading && ed.base.version === (want.version || latestVersion(ed))) return;
      st.wanted = want;
    },
    /** The version on screen: {id, version, title, edited}, or null (nothing open, or a design not saved yet). */
    current() {
      const ed = st.ed;
      if (!ed || !ed.base) return null;
      return {id: ed.base.design_id, version: ed.base.version, title: ed.base.title, edited: isEdited()};
    },
    /** Reloads the list now (and the open version, when it changed). */
    refresh: () => listPoll.now(),
    hold() {
      if (!isEdited()) return '';
      const f = st.ed.form;
      return `Unsaved edits to ${f.title.trim() || (st.ed.base ? st.ed.base.title : 'a new design')}`;
    },
  });
})();
