/* QUEUE tab: order, status, pause and resume, cap, remove (<section id="view-queue">).

   Owned by the QUEUE builder, with queue.css and demo-queue.js. Brief:
   "QUEUE tab", the Queue endpoints, and the header queue chip under
   "Navigation and header".

   The queue is the only way a design becomes a billed run. This view draws it
   from GET /api/queue, polled every 3 s while QUEUE or DESIGNS is open: the
   state and its pause reason, the global cap, the item running now (its live
   spend and trials come from the matching run in /api/state), the waiting and
   held items in position order, and the last 50 finished items. Every queue
   mutation answers a QueueView, which replaces the one on screen at once.

   Start queue and Retry can start a billed run, so they confirm through
   armConfirm (code ARM). Add to queue and Start now live in DESIGNS and LAUNCH
   (enqueueDesign in app.js). Remove, Cancel and Reset count confirm through
   confirmAction; Edit cap is a formDialog. Nothing here holds unsent input on
   the page itself: the on-failure select saves as soon as it changes.

   Contract: everything stays inside this IIFE. The view registers with
   registerView('queue', hooks) (see Areas in app.js) and draws only inside
   #view-queue and the header chip #queue-chip. Other areas reach it through
   openQueue(item), queueView(), refreshQueue() and queueSummary(el) in app.js,
   which call the open, data, refresh and summary hooks below. */

(() => {
  'use strict';

  // Queue-item statuses (QueueItem.status): [glyph, tag class]. STATUS_TAG is for runs, and the glyph keeps the
  // status readable without its colour.
  const ITEM_TAG = {
    waiting: ['·', ''], held: ['!', 'warn'], starting: ['▸', 'run'], running: ['▸', 'run'],
    done: ['✓', 'ok'], budget_stopped: ['$', 'warn'], failed: ['✕', 'bad'], cancelled: ['■', ''], interrupted: ['!', 'warn'],
  };
  const ON_FAILURE = [['pause', 'Pause the queue'], ['skip', 'Skip to the next design']];
  const ON_FAILURE_NOTE = {
    pause: 'A failed or interrupted run pauses the queue. A cancel always does.',
    skip: 'A failed or interrupted run is recorded and the next design starts. A cancel still pauses.',
  };
  const LOOKUP_MS = 5000;  // how long a confirm waits for design versions (their providers) before opening without them
  const HISTORY_SHOWN = 50;

  let view = null;        // the latest QueueView
  let error = null;       // the ApiError of the last failed GET /api/queue; the last view stays on screen
  let seq = 0;            // bumped by every request that may replace the view, so an older GET never overwrites a newer answer
  let loads = 0;          // answers applied, so openQueue only calls an item missing once a fresh view arrived
  let wanted = null;      // item id from openQueue, highlighted while QUEUE stays open
  let wantedAt = 0;       // `loads` when it was asked for
  let scrolled = '';      // the wanted id already scrolled into view
  let refocus = null;     // {item, act}: the row button to focus again after a move redraws the list
  let cancelling = null;  // {runId, label}: Cancel was sent for the active item's run
  let notice = null;      // {glyph, cls, text}: a line above the panels until dismissed or the queue starts
  let pausing = false;    // the person pressed Pause, so the pause needs no toast
  let dialogs = 0;        // input dialogs open (Edit cap): a run starting then does not pull the page to LIVE
  let savingPolicy = false;  // the on-failure select's PUT is in flight
  const providers = new Map();  // 'design_id@version' -> Promise<[provider] | null>; a saved version never changes

  const load = poller({
    ms: 3000,
    when: () => ui.view === 'queue' || ui.view === 'designs',
    run: async () => {
      const mine = ++seq;
      let data = null;
      let failure = null;
      try {
        data = await api('GET', '/api/queue');
      } catch (err) {
        failure = err;
      }
      if (mine !== seq) return;  // a mutation answered meanwhile, and its view is newer
      if (data) {
        view = data;
        loads++;
      }
      error = failure;
      if (ui.view === 'queue' && ui.state) renderQueue(ui.state, false);
    },
  });

  /** Shows the QueueView a mutation answered, at once. */
  function setView(data) {
    if (!data || typeof data !== 'object' || !Array.isArray(data.waiting)) return;
    seq++;
    loads++;
    view = data;
    error = null;
    if (ui.view === 'queue' && ui.state) renderQueue(ui.state, false);
  }

  // Helpers ----------------------------------------------------------------

  const itemLabel = (item) => `${item.title || item.design_id} v${item.version}`;
  const openItems = (v) => [...v.waiting, ...v.held].sort((a, b) => a.position - b.position);
  const runOf = (item) => (item && item.run_id && ui.state ? ui.state.runs.find((r) => r.id === item.run_id) || null : null);
  const pausedText = (reason) => (/^paused/i.test(reason || '') ? reason : `Queue paused: ${reason || DASH}`);
  /** The pause reason after the strip's own "PAUSED:" label, without the server's leading "Paused: " repeated. */
  const reasonText = (reason) => (reason ? reason.replace(/^paused:\s*/i, '') : DASH);
  const pair = (label, value) => h('div', null, h('dt', null, label), h('dd', null, value));

  function itemTag(status) {
    const [glyph, cls] = ITEM_TAG[status] || ['?', ''];
    return tag(`${glyph} ${statusLabel(status)}`, cls);
  }

  /** "28 Oct", or "today 15:09" when the cap count started today. */
  function sinceText(iso) {
    const d = new Date(iso);
    if (!iso || Number.isNaN(d.getTime())) return DASH;
    if (d.toDateString() === new Date().toDateString()) return `today ${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
    return d.toLocaleDateString('en-GB', {day: 'numeric', month: 'short'});
  }

  /** A run's spend is a lower bound while any arm has killed trials or trials that recorded no usage. */
  const spendFloor = (run) => Boolean(run && (run.arms || []).some((a) => a.trials_killed > 0 || a.trials_unmeasured > 0));

  /** The promise's value, or `fallback` once `ms` have passed. */
  const within = (promise, ms, fallback) => Promise.race([promise, new Promise((resolve) => setTimeout(() => resolve(fallback), ms))]);

  const versionPath = (item) => `/api/designs/${enc(item.design_id)}/versions/${enc(item.version)}`;
  const providersIn = (detail) => [...new Set(((detail.spec && detail.spec.arms) || []).map((a) => a.provider).filter(Boolean))];

  /** The providers one design version bills, from its spec, or null when it could not be read. Cached per version. */
  function providersOf(item) {
    const key = `${item.design_id}@${item.version}`;
    if (!providers.has(key)) {
      providers.set(key, api('GET', versionPath(item)).then(providersIn).catch(() => {
        providers.delete(key);  // asked again next time
        return null;
      }));
    }
    return providers.get(key);
  }

  /** Every provider these items bill, and the labels of those whose providers could not be loaded in time. */
  async function billsFor(items) {
    const found = await within(Promise.all(items.map(providersOf)), LOOKUP_MS, null);
    if (!found) return {bills: [], unknown: items.map(itemLabel)};
    return {
      bills: [...new Set(found.flat().filter(Boolean))].sort(),
      unknown: items.filter((item, n) => !found[n]).map(itemLabel),
    };
  }

  /** Runs one queue mutation with its button disabled and shows the QueueView it answers. A 404 or 409 reloads the
   *  queue, since it means the queue changed under the page. Resolves true on success. */
  function act(button, send, done = '') {
    return busy(button, async () => {
      try {
        setView(await send());
      } catch (err) {
        if (err.status === 404 || err.status === 409) load.now();
        throw err;
      }
      if (done) toast(done);
      return true;
    });
  }

  // View -------------------------------------------------------------------

  function renderQueue(state, force) {
    const root = $('#view-queue');
    if (!root.querySelector('.qu')) {
      build(root);
      force = true;
    }
    noteCancel();
    renderBanner(force);
    renderNotice(force);
    renderStrip(force);
    renderActive(force);
    renderCap(force);
    renderNext(force);
    renderHistory(force);
    showWanted();
  }

  /** The frame: built once per page, its parts redrawn only when their data changes. */
  function build(root) {
    const onFailure = select('on_failure', ON_FAILURE, view ? view.on_failure : 'pause', {
      id: 'qu-on-failure', onchange: (e) => saveOnFailure(e.currentTarget),
    });
    fill(root, h('div', {class: 'qu'},
      h('div', {class: 'qu-strip', id: 'qu-strip'}),
      h('div', {id: 'qu-banner'}),
      h('div', {id: 'qu-notice'}),
      h('div', {class: 'qu-top'},
        panel({code: 'Q-01', title: 'Running now', meta: h('span', {id: 'qu-active-meta'}), bodyId: 'qu-active'}),
        panel({code: 'Q-02', title: 'Global cap', meta: h('span', {id: 'qu-cap-meta'})},
          h('div', {id: 'qu-cap'}),
          h('div', {class: 'qu-cap-controls'},
            h('div', {class: 'actions'},
              h('button', {type: 'button', class: 'btn ghost small', id: 'qu-edit-cap', onclick: editCap}, 'Edit cap'),
              h('button', {type: 'button', class: 'btn ghost small', id: 'qu-reset-cap', onclick: (e) => resetCap(e.currentTarget)}, 'Reset count')),
            field('On failure', onFailure, {notes: [h('small', {id: 'qu-on-failure-note'})]})))),
      panel({code: 'Q-03', title: 'Up next', meta: h('span', {class: 'qu-meta', id: 'qu-next-meta'}), flush: true, bodyId: 'qu-next'}),
      panel({code: 'Q-04', title: 'History', meta: h('span', {id: 'qu-history-meta'}), flush: true, bodyId: 'qu-history'})));
  }

  /** A failed GET /api/queue (503 is "busy, try again", never offline) or a failed queue block in /api/state. */
  function renderBanner(force) {
    const problem = stateError('queue');
    const text = !error ? ''
      : error.status === 503 ? `The queue is busy: ${error.message} Trying again every 3 s.`
        : error.status === 0 ? error.message
          : `Could not load the queue: ${error.message}`;
    if (!changed('queue:banner', [text, problem, Boolean(view)]) && !force) return;
    const shown = text || (problem && `The queue summary in /api/state failed: ${problem}`);
    fill($('#qu-banner'), shown && h('div', {class: error && error.status === 503 ? 'banner warn' : 'banner bad', role: 'alert'},
      h('span', null, shown, view && error ? ' The last answer stays on screen.' : ''),
      error && h('button', {type: 'button', class: 'btn ghost small qu-retry-load', onclick: (e) => busy(e.currentTarget, () => load.now())},
        'Try again')));
  }

  function renderNotice(force) {
    if (!changed('queue:notice', notice) && !force) return;
    fill($('#qu-notice'), notice && h('div', {class: 'note-strip qu-notice', role: 'status'},
      h('span', {class: notice.cls, 'aria-hidden': 'true'}, notice.glyph),
      h('span', {class: 'qu-notice-text'}, notice.text),
      h('button', {type: 'button', class: 'btn ghost small', onclick: () => {
        notice = null;
        renderNotice(true);
      }}, 'Dismiss')));
  }

  /** "RUNNING" or "PAUSED: <pause_reason>", the counts, and Start queue or Pause. */
  function renderStrip(force) {
    const el = $('#qu-strip');
    const v = view;
    const data = v && [v.state, v.pause_reason, v.waiting.length, v.held.length, v.committed_usd, Boolean(v.active)];
    if (!changed('queue:strip', [data, Boolean(error)]) && !force) return;
    if (!v) {
      el.className = 'qu-strip';
      fill(el, h('div', {class: 'qu-state'}, led(error ? 'bad' : ''), h('b', null, error ? 'QUEUE UNAVAILABLE' : 'LOADING THE QUEUE…')));
      return;
    }
    const running = v.state === 'running';
    const canStart = Boolean(v.active) || v.waiting.length > 0;
    const why = canStart ? null : v.held.length
      ? `Nothing is waiting. Retry or remove the ${plural(v.held.length, 'held item')} first.`
      : 'Nothing is waiting. Add a design from DESIGNS first.';
    el.className = running ? 'qu-strip running' : 'qu-strip';
    fill(el,
      h('div', {class: 'qu-state'},
        led(running ? 'ok live' : v.held.length ? 'warn' : ''),
        h('b', null, running ? 'RUNNING' : 'PAUSED:'),
        !running && h('span', {class: 'qu-reason'}, reasonText(v.pause_reason))),
      h('div', {class: 'qu-counts'},
        h('span', null, h('b', null, String(v.waiting.length)), ' waiting'),
        v.held.length > 0 && h('span', {class: 'warn-text'}, ` · ${v.held.length} held`),
        h('span', null, ' · ', h('b', null, money(v.committed_usd)), ' committed')),
      h('div', {class: 'actions qu-controls'},
        running
          ? h('button', {type: 'button', class: 'btn ghost', id: 'qu-pause', onclick: (e) => pauseQueue(e.currentTarget)}, '❚❚ Pause')
          : h('button', {type: 'button', class: 'btn primary', id: 'qu-start', disabled: !canStart, title: why,
            onclick: (e) => startQueue(e.currentTarget)}, '▶ Start queue'),
        why && h('small', {class: 'muted qu-why'}, why)));
  }

  // Active item ------------------------------------------------------------

  /** The item running now: its frame (buttons) when the item or its state changes, its live figures every poll. */
  function renderActive(force) {
    const body = $('#qu-active');
    const v = view;
    const a = v && v.active;
    const run = runOf(a);
    const live = Boolean(run && ui.state && run.id === ui.state.active_run);
    const other = ui.state ? ui.state.active_run : null;
    const frame = a
      ? [a.id, a.status, a.run_id, a.started_at, live, Boolean(run), Boolean(cancelling && cancelling.runId === a.run_id), a.id === wanted]
      : [Boolean(v), v && v.state, other, v && v.waiting.length];
    if (changed('queue:active', frame) || force || !body.firstChild) buildActive(body, a, run, live, other);
    if (a) updateActive(a, run);
  }

  function buildActive(body, a, run, live, other) {
    const v = view;
    $('#qu-active-meta').textContent = a ? a.id : '';
    if (!a) {
      fill(body, h('div', {class: 'qu-idle'},
        h('p', null, 'No design is running.'),
        h('small', {class: 'muted'}, !v ? DASH
          : v.state === 'running' && other ? `Run ${other} is in progress. The next design starts after it ends.`
            : v.state === 'running' ? 'The queue starts the next waiting design within a few seconds.'
              : v.waiting.length ? `Press Start queue to run the waiting designs in order.${other ? ` They wait for run ${other} to end.` : ''}`
                : 'Add a design from DESIGNS, then press Start queue.'),
        other && h('div', {class: 'actions'},
          h('button', {type: 'button', class: 'btn ghost small', onclick: () => showView('live')}, 'Open LIVE ', h('span', {class: 'arrow'}, '›')))));
      return;
    }
    const stopping = Boolean(cancelling && cancelling.runId === a.run_id);
    fill(body, h('div', {class: a.id === wanted ? 'qu-active qu-wanted' : 'qu-active', 'data-item': a.id},
      h('div', {class: 'qu-active-head'}, designChip(a, {hash: true}), itemTag(a.status)),
      h('dl', {class: 'params'},
        pair('Run', a.run_id
          ? h('button', {type: 'button', class: 'qu-link', title: live ? 'Open in LIVE' : 'Open the run', onclick: () => openRun(a.run_id)}, a.run_id)
          : 'starting…'),
        pair('Started', when(a.started_at)),
        pair('Estimate', estimateText(a.expected_usd, a.p90_usd)),
        pair('In-flight allowance', money(a.inflight_allowance_usd))),
      h('div', {class: 'qu-live', id: 'qu-live'}),
      stopping && h('p', {class: 'warn-text qu-stopping'}, 'Cancel sent. Trials are stopping; the queue pauses when the run ends.'),
      h('div', {class: 'actions'},
        h('button', {type: 'button', class: 'btn primary', disabled: !live, title: live ? null : 'The run is not live yet',
          onclick: () => showView('live')}, 'Open in LIVE ', h('span', {class: 'arrow'}, '›')),
        h('button', {type: 'button', class: 'btn danger', id: 'qu-cancel', disabled: !a.run_id || stopping,
          title: a.run_id ? null : 'The run has not started yet', onclick: (e) => cancelActive(e.currentTarget)}, 'Cancel'))));
  }

  /** Spend of budget and trials, live from the matching run in /api/state. */
  function updateActive(a, run) {
    const box = $('#qu-live');
    if (!box) return;
    const arms = run ? run.arms || [] : [];
    const spend = run && isNum(run.spend_usd) ? run.spend_usd : a.spend_usd;
    const budget = run && isNum(run.budget_usd) ? run.budget_usd : a.budget_usd;
    const floor = spendFloor(run);
    if (!changed('queue:live', [a.id, a.status, run && run.status, spend, budget, arms]) && box.firstChild) return;
    const done = sum(arms, (x) => x.trials_done);
    const total = sum(arms, (x) => x.trials_total);
    const inFlight = arms.some((x) => isNum(x.trials_running)) ? sum(arms, (x) => x.trials_running) : null;
    const share = isNum(budget) && budget > 0 && isNum(spend) ? spend / budget : 0;
    fill(box,
      h('div', {class: 'qu-meter-row'},
        h('span', {class: 'qu-big'}, floor ? usageValue(spend, 'lower_bound') : money(spend)),
        h('span', {class: 'muted'}, `of ${money(budget)} budget`),
        floor && tag('lower bound', 'warn')),
      gauge(share, {tone: share >= 1 ? 'bad' : share > 0.85 ? 'warn' : ''}),
      h('div', {class: 'qu-meter-row'},
        h('span', {class: 'qu-big'}, run ? `${done} / ${total}` : DASH),
        h('span', {class: 'muted'}, 'trials'),
        isNum(inFlight) && inFlight > 0 && h('span', {class: 'muted'}, `· ${inFlight} running`)),
      h('div', {class: 'qu-progress', role: 'progressbar', 'aria-label': 'Trials finished', 'aria-valuemin': 0,
        'aria-valuemax': total, 'aria-valuenow': done},
      h('i', {style: `width:${(total ? Math.min(1, done / total) * 100 : 0).toFixed(1)}%`})),
      arms.length > 0 && h('ul', {class: 'qu-arms'}, arms.map((x) => h('li', {title: x.message || null},
        h('span', {class: 'qu-arm-name'}, led(STATUS_LED[x.status] || ''), x.name),
        statusTag(x.status),
        h('span', {class: 'qu-num'}, `${count(x.trials_done)}/${count(x.trials_total)}`),
        h('span', {class: 'qu-num'}, money(x.spend_usd)),
        x.trials_killed > 0 && h('small', {class: 'warn-text'}, `${x.trials_killed} killed (spend not counted)`),
        x.trials_unmeasured > 0 && h('small', {class: 'warn-text'}, `${x.trials_unmeasured} without usage (spend not counted)`),
        x.cleanup_pending && tag('containers may remain', 'warn')))),
      !run && note(a.status === 'starting'
        ? 'Starting: the queue checks this version once more before its run begins.'
        : 'The run shows here with the next poll.'));
  }

  async function cancelActive(button) {
    const a = view && view.active;
    if (!a || !a.run_id) return;
    const ok = await confirmAction({
      code: 'CANCEL', title: 'Cancel the running design?', ok: 'Cancel run', tone: 'danger',
      body: [
        h('p', null, `Cancel run ${a.run_id} of ${itemLabel(a)} (${a.id})?`),
        h('p', null, 'Trials in progress are stopped and their containers removed. The queue then pauses: nothing new starts until you press Start queue.'),
      ],
    });
    if (!ok) return;
    await busy(button, async () => {
      await api('POST', `/api/runs/${enc(a.run_id)}/cancel`);
      cancelling = {runId: a.run_id, label: `${itemLabel(a)} (${a.id})`};
      toast(`Cancel sent for ${a.run_id}. Stopping trials…`);
      if (ui.view === 'queue') renderActive(false);
      load.now();
      poll();
    });
  }

  /** Once the cancelled run has left the active item, says what the queue did. */
  function noteCancel() {
    const q = view || null;
    if (!cancelling || !q) return;
    if (q.active && q.active.run_id === cancelling.runId) return;
    const after = q.state === 'paused' ? pausedText(q.pause_reason) : 'The queue is still running.';
    notice = {glyph: '■', cls: 'warn-text', text: `Cancelled ${cancelling.label}. ${after}`};
    cancelling = null;
  }

  // Global cap -------------------------------------------------------------

  /** "$3.10 spent since 28 Oct of $20.00 · $7.50 committed", a gauge, and what the cap leaves. */
  function renderCap(force) {
    const v = view;
    const sel = $('#qu-on-failure');
    const data = v && [v.global_cap_usd, v.cap_since, v.cap_spent_usd, v.cap_remaining_usd, v.committed_usd, v.on_failure];
    const dirty = changed('queue:cap', data) || force;
    if (dirty) {
      $('#qu-edit-cap').disabled = !v;
      $('#qu-reset-cap').disabled = !v;
      sel.disabled = !v || savingPolicy;
      $('#qu-on-failure-note').textContent = v ? ON_FAILURE_NOTE[v.on_failure] || '' : '';
    }
    // The select is never rebuilt; it follows the server unless it is being used or saved.
    if (v && document.activeElement !== sel && !savingPolicy) sel.value = v.on_failure;
    if (!dirty) return;
    const box = $('#qu-cap');
    if (!v) {
      $('#qu-cap-meta').textContent = '';
      fill(box, h('p', {class: 'muted qu-meter'}, DASH));
      return;
    }
    const cap = v.global_cap_usd;
    const spent = v.cap_spent_usd;
    const left = v.cap_remaining_usd;
    const hasCap = isNum(cap) && cap > 0;
    const over = hasCap && isNum(left) && v.committed_usd > left + 1e-9;
    $('#qu-cap-meta').textContent = hasCap ? capLeftText(v) : 'no cap';
    fill(box,
      h('p', {class: 'qu-meter'},
        h('b', null, money(spent)), ' spent since ',
        tip(h('span', {class: 'qu-since'}, sinceText(v.cap_since)), 'Count started', [when(v.cap_since), 'Reset count starts it again from now'], true),
        hasCap ? [' of ', h('b', null, money(cap))] : ' · no global cap',
        ' · ', h('b', null, money(v.committed_usd)), ' committed'),
      hasCap && gauge(isNum(spent) ? spent / cap : 0, {
        marker: isNum(spent) ? (spent + v.committed_usd) / cap : null,
        tone: isNum(left) && left <= 0 ? 'bad' : over ? 'warn' : '',
      }),
      h('dl', {class: 'params qu-cap-figs'},
        pair('Left', capLeftText(v)),
        pair('Committed', tip(h('span', null, money(v.committed_usd)), 'Committed',
          ['Budgets plus in-flight allowances of the waiting designs', 'The gauge marker shows spend plus this'], true))),
      hasCap && !isNum(spent) && h('p', {class: 'warn-text qu-warn'}, '! The spend since the count started could not be read.'),
      over && h('p', {class: 'warn-text qu-warn'},
        `! The waiting designs commit ${money(v.committed_usd)}, more than the ${money(left)} left. `
        + 'The first one that does not fit is held, and the queue pauses.'));
  }

  async function editCap() {
    const v = view;
    if (!v) return;
    dialogs++;
    try {
      const out = await formDialog({
        code: 'CAP', title: 'Global cap', ok: 'Save cap',
        body: [
          h('p', null, 'The most that all runs together may spend, queued or launched, counted since the last reset. '
            + 'A design starts only if its budget plus its in-flight allowance fits in what is left.'),
          field('Cap (USD)', numberInput('global_cap_usd', isNum(v.global_cap_usd) ? v.global_cap_usd : '', {
            min: 0.01, step: 'any', placeholder: 'No cap', inputmode: 'decimal', class: 'mono',
          }), {notes: [note(`Empty means no cap. ${money(v.cap_spent_usd)} spent since ${sinceText(v.cap_since)}.`)]}),
        ],
        read: async (form) => {
          const cap = readNumber(form.elements.global_cap_usd, null);
          if (cap !== null && !(isNum(cap) && cap > 0)) throw new Error('Enter an amount above zero, or leave it empty for no cap.');
          return api('PUT', '/api/queue/settings', {global_cap_usd: cap});
        },
      });
      if (!out) return;
      setView(out);
      toast(isNum(out.global_cap_usd) ? `Global cap set to ${money(out.global_cap_usd)}` : 'Global cap removed');
      poll();
    } finally {
      dialogs--;
    }
  }

  async function resetCap(button) {
    const v = view;
    if (!v) return;
    const ok = await confirmAction({
      code: 'RESET', title: 'Reset the cap count?', ok: 'Reset count',
      body: [
        h('p', null, `The global cap counts the spend of every run created since ${when(v.cap_since)}: ${money(v.cap_spent_usd)} so far.`),
        h('p', null, isNum(v.global_cap_usd)
          ? `Resetting starts the count again from now, so the whole ${money(v.global_cap_usd)} is available again.`
          : 'Resetting starts the count again from now. No cap is set, so nothing is held back either way.'),
      ],
    });
    if (!ok) return;
    if (await act(button, () => api('PUT', '/api/queue/settings', {reset_cap: true}), 'Cap count reset')) poll();
  }

  async function saveOnFailure(sel) {
    const value = sel.value;
    savingPolicy = true;
    // Skip lets the queue keep billing after a failed run, and an arrow key on the closed select is enough to pick
    // it, so that direction asks first. Back to pause needs no confirm.
    if (value === 'skip' && (!view || view.on_failure !== 'skip')) {
      const ok = await confirmAction({
        code: 'POLICY', title: 'Keep going after a failure?', ok: 'Skip failures', tone: 'armed',
        body: [
          h('p', null, 'A failed or interrupted run is recorded and the next waiting design starts, so the queue keeps billing after failures.'),
          h('p', null, 'A cancel still pauses the queue.'),
        ],
      });
      if (!ok) {
        savingPolicy = false;
        sel.value = view ? view.on_failure : 'pause';
        sel.focus();
        return;
      }
    }
    sel.disabled = true;
    try {
      setView(await api('PUT', '/api/queue/settings', {on_failure: value}));
      toast(value === 'skip' ? 'On failure: skip to the next design' : 'On failure: pause the queue');
    } catch (err) {
      toast(err.message, true);
    } finally {
      savingPolicy = false;
      sel.disabled = !view;
      if (view) sel.value = view.on_failure;
      $('#qu-on-failure-note').textContent = view ? ON_FAILURE_NOTE[view.on_failure] || '' : '';
    }
  }

  // Up next ----------------------------------------------------------------

  /** Waiting and held items in position order (one list: position and move count across both). */
  function renderNext(force) {
    const body = $('#qu-next');
    const v = view;
    const items = v ? openItems(v) : [];
    if (!changed('queue:next', v && [items, v.state, Boolean(v.active), wanted]) && !force) return;
    fill($('#qu-next-meta'), v && [
      `${v.waiting.length} waiting`,
      v.held.length > 0 && h('span', {class: 'warn-text'}, ` · ${v.held.length} held`),
      h('button', {type: 'button', class: 'btn ghost small', onclick: () => showView('designs')}, 'Add a design ', h('span', {class: 'arrow'}, '›'))]);
    if (!v) {
      fill(body, h('p', {class: 'qu-pad muted'}, DASH));
      return;
    }
    if (!items.length) {
      fill(body, h('div', {class: 'empty-state'},
        h('p', null, v.active ? 'Nothing else is waiting. Add a design from DESIGNS.' : 'Nothing in the queue. Add a design from DESIGNS.'),
        h('button', {type: 'button', class: 'btn ghost', onclick: () => showView('designs')}, 'Open DESIGNS ', h('span', {class: 'arrow'}, '›'))));
      return;
    }
    fill(body, h('ol', {class: 'qu-list', 'aria-label': 'Waiting and held designs, in order'},
      h('li', {class: 'qu-row qu-head', 'aria-hidden': 'true'},
        h('span', {class: 'qu-pos'}, '#'), h('span', null, 'Design'), h('span', {class: 'qu-num'}, 'Budget'),
        h('span', {class: 'qu-num'}, 'Estimate'), h('span', {class: 'qu-num'}, 'Allowance'), h('span', null, 'Status'), h('span', null, '')),
      items.map((item, n) => nextRow(item, n === 0, n === items.length - 1))));
    if (refocus) {
      const row = body.querySelector(`[data-item="${CSS.escape(refocus.item)}"]`);
      const target = row && (row.querySelector(`[data-act="${refocus.act}"]:not(:disabled)`) || row.querySelector('button:not(:disabled)'));
      if (target) target.focus();
      refocus = null;
    }
  }

  function nextRow(item, first, last) {
    const held = item.status === 'held';
    const fig = (label, value, title = null) => h('div', {class: 'qu-fig', title},
      h('span', {class: 'qu-lbl'}, label), h('span', {class: 'qu-num'}, value));
    const iconButton = (glyph, which, label, disabled, to) => h('button', {
      type: 'button', class: 'btn ghost small qu-icon', 'data-act': which, 'aria-label': label, title: label, disabled,
      onclick: (e) => moveItem(item, to, which, e.currentTarget),
    }, glyph);
    return h('li', {class: ['qu-row', held && 'held', item.id === wanted && 'qu-wanted'].filter(Boolean).join(' '), 'data-item': item.id},
      h('span', {class: 'qu-pos', title: `Position ${item.position + 1}`}, String(item.position + 1)),
      h('div', {class: 'qu-what'},
        h('div', {class: 'qu-title'}, designChip(item, {hash: true}), h('span', {class: 'qu-id'}, item.id)),
        held && item.message && h('p', {class: 'warn-text qu-msg'}, h('span', {'aria-hidden': 'true'}, '! '), item.message),
        h('small', {class: 'qu-added'}, `added ${when(item.added_at)}`)),
      h('div', {class: 'qu-figs'},
        fig('Budget', money(item.budget_usd)),
        fig('Estimate', estimateText(item.expected_usd, item.p90_usd)),
        fig('Allowance', money(item.inflight_allowance_usd), 'In-flight allowance: what trials still running may add once the budget is reached')),
      h('span', {class: 'qu-status'}, itemTag(item.status)),
      h('div', {class: 'qu-acts'},
        iconButton('⤒', 'front', `Move ${item.id} to the front`, first, 0),
        iconButton('↑', 'up', `Move ${item.id} up`, first, item.position - 1),
        iconButton('↓', 'down', `Move ${item.id} down`, last, item.position + 1),
        held && h('button', {type: 'button', class: 'btn ghost small', 'data-act': 'retry', onclick: (e) => retryItem(item, e.currentTarget)}, 'Retry'),
        h('button', {type: 'button', class: 'btn danger small', 'data-act': 'remove', onclick: (e) => removeItem(item, e.currentTarget)}, 'Remove')));
  }

  /** Up, down or to the front: to_index counts among the waiting and held items, as position does. */
  async function moveItem(item, to, which, button) {
    refocus = {item: item.id, act: which};  // the list redraws; keyboard focus stays on this row
    const ok = await act(button, () => api('POST', `/api/queue/items/${enc(item.id)}/move`, {to_index: to}));
    if (!ok) refocus = null;
  }

  async function removeItem(item, button) {
    const ok = await confirmAction({
      code: 'REMOVE', title: 'Remove from the queue?', ok: 'Remove', tone: 'danger',
      body: [
        h('p', null, `Remove ${item.id}, ${itemLabel(item)}, from the queue?`),
        h('p', null, 'The version stays approved in DESIGNS; add it again from there to run it.'),
      ],
    });
    if (!ok) return;
    if (!(await act(button, () => api('DELETE', `/api/queue/items/${enc(item.id)}`), `${item.id} removed from the queue`))) return;
    refreshDesigns();
    poll();
  }

  /** Retry a held item: billed, since a running queue can start it at once. Loads the version first, for its
   *  providers and its problems as of now. */
  async function retryItem(item, button) {
    const detail = await busy(button, () => within(
      api('GET', versionPath(item)).catch((err) => ({failed: err.message})), LOOKUP_MS, {failed: 'it took too long to load'}));
    if (!detail) return;
    const loaded = !detail.failed;
    const problems = loaded && detail.check && detail.check.validation ? detail.check.validation.problems || [] : [];
    const bills = loaded ? providersIn(detail) : [];
    if (loaded) providers.set(`${item.design_id}@${item.version}`, Promise.resolve(bills));
    const v = view;
    const out = await armConfirm({
      title: `Retry ${item.id}?`,
      summary: `Check ${itemLabel(item)} again and put it back in line at position ${item.position + 1}?`,
      rows: [
        ['Design', itemLabel(item)],
        ['Spec hash', shortHash(item.spec_hash)],
        ['Budget', money(item.budget_usd)],
        ['Estimate', estimateText(item.expected_usd, item.p90_usd)],
        ['In-flight allowance', money(item.inflight_allowance_usd)],
        ['Global cap', capLeftText(v)],
      ],
      warnings: [
        item.message && `Held: ${item.message}`,
        ...problems.map((p) => `Problem now: ${p}`),
        problems.length > 0 && 'While these remain, the retry fails and the item stays held.',
        !loaded && `Could not load the design version (${detail.failed}), so its providers are not shown.`,
      ],
      lines: [v && v.state === 'running'
        ? 'The queue is running: if this design is next in line, its run can start at once.'
        : 'The queue is paused: nothing starts until you press Start queue.'],
      bills,
      ok: '▶ Retry',
      send: () => api('POST', `/api/queue/items/${enc(item.id)}/retry`),
    });
    if (!out) {
      load.now();  // a failed retry stores its reason on the item
      return;
    }
    setView(out);
    toast(`${item.id} is waiting again`);
    poll();
  }

  // Start and pause --------------------------------------------------------

  /** Start queue: billed. "Run <k> designs in order? Budgets $S plus in-flight allowance $A …" */
  async function startQueue(button) {
    const v = view;
    if (!v) return;
    const waiting = v.waiting.slice().sort((a, b) => a.position - b.position);
    const found = await busy(button, () => billsFor(waiting));
    if (!found) return;
    const k = waiting.length;
    const budgets = sum(waiting, (i) => i.budget_usd);
    const allowance = sum(waiting, (i) => i.inflight_allowance_usd);
    const hasCap = isNum(v.global_cap_usd);
    const capText = hasCap ? `Global cap ${money(v.global_cap_usd)} (${capLeftText(v)}).` : 'No global cap.';
    const failText = v.on_failure === 'skip'
      ? 'A failed run is skipped; the queue pauses if a run is cancelled.'
      : 'The queue pauses if a run fails or is cancelled.';
    const other = ui.state && ui.state.active_run && !(v.active && v.active.run_id === ui.state.active_run) ? ui.state.active_run : null;
    const order = waiting.slice(0, 8).map(itemLabel).join(' → ') + (k > 8 ? ` and ${k - 8} more` : '');
    const out = await armConfirm({
      title: k ? 'Start the queue?' : 'Resume the queue?',
      summary: k
        ? `Run ${plural(k, 'design')} in order? Budgets ${money(budgets)} plus in-flight allowance ${money(allowance)} `
          + `(${money(v.committed_usd)} committed). ${capText} ${failText}`
        : `Nothing new is waiting; ${v.active ? itemLabel(v.active) : 'the design in progress'} carries on. ${failText}`,
      rows: [
        k > 0 && ['In order', order],
        ['Budgets', money(budgets)],
        ['In-flight allowance', money(allowance)],
        ['Committed', money(v.committed_usd)],
        ['Global cap', hasCap ? `${money(v.global_cap_usd)} · ${capLeftText(v)}` : 'none'],
        ['On failure', v.on_failure],
      ],
      warnings: [
        v.held.length > 0 && (v.held.length === 1
          ? '1 held item stays held until you retry it.'
          : `${v.held.length} held items stay held until you retry them.`),
        other && `Run ${other} is in progress. The first design starts after it ends.`,
        hasCap && isNum(v.cap_remaining_usd) && v.committed_usd > v.cap_remaining_usd + 1e-9
          && `The waiting designs commit ${money(v.committed_usd)}, more than the ${money(v.cap_remaining_usd)} the cap has left. `
          + 'The first one that does not fit is held, and the queue pauses.',
        hasCap && !isNum(v.cap_spent_usd) && 'The spend under the global cap could not be read.',
        ui.state && !dockerOk() && 'Docker Desktop is not running, so the queue cannot start.',
        found.unknown.length > 0 && `Could not load the providers of ${listText(found.unknown)}.`,
      ],
      bills: found.bills,
      ok: '▶ Start queue',
      send: () => api('POST', '/api/queue/start'),
    });
    if (!out) {
      refocusStrip();
      return;
    }
    notice = null;
    setView(out);
    // The answer can already be paused again: the first item held on the cap, say. Say so instead of "started".
    if (out.state === 'running') toast(k ? `Queue started: ${plural(k, 'design')} in line` : 'Queue resumed');
    else toast(pausedText(out.pause_reason), true);
    refocusStrip();
    poll();
  }

  /** After a dialog from the strip closes, focus goes back to its control (the button was disabled meanwhile). */
  function refocusStrip() {
    setTimeout(() => {
      if (document.activeElement && document.activeElement !== document.body) return;
      const el = $('#qu-start') || $('#qu-pause');
      if (el && !el.disabled) el.focus({preventScroll: true});
    }, 0);
  }

  async function pauseQueue(button) {
    pausing = true;
    const ok = await act(button, () => api('POST', '/api/queue/pause'), 'Queue paused. A run in progress carries on.');
    if (!ok) pausing = false;
    poll();
  }

  // History ----------------------------------------------------------------

  function renderHistory(force) {
    const body = $('#qu-history');
    const v = view;
    const items = v ? v.history : [];
    const floors = items.map((item) => spendFloor(runOf(item)));
    if (!changed('queue:history', v && [items, floors, wanted]) && !force) return;
    $('#qu-history-meta').textContent = !v || !items.length ? ''
      : `${items.length >= HISTORY_SHOWN ? `last ${HISTORY_SHOWN}` : plural(items.length, 'item')} · newest first`;
    if (!v) {
      fill(body, h('p', {class: 'qu-pad muted'}, DASH));
      return;
    }
    if (!items.length) {
      fill(body, h('p', {class: 'qu-pad muted'}, 'No finished items yet.'));
      return;
    }
    fill(body, h('div', {class: 'scroll'}, h('table', {class: 'grid-t compact qu-history'},
      thead('Item', 'Design', 'Status', thNum('Spend of budget'), 'Finished', 'Run'),
      h('tbody', null, items.map((item, n) => h('tr', {class: item.id === wanted ? 'qu-wanted' : null, 'data-item': item.id},
        h('td', {class: 'mono'}, item.id),
        h('td', null, designChip(item)),
        h('td', null, itemTag(item.status),
          item.message && h('small', {class: 'qu-hmsg', title: item.message}, item.message)),
        tdNum(`${usageValue(item.spend_usd, floors[n] ? 'lower_bound' : 'full')} of ${money(item.budget_usd)}`,
          floors[n] ? 'A lower bound: some trials were killed or recorded no usage' : null),
        h('td', {class: 'mono'}, when(item.finished_at)),
        h('td', null, item.run_id
          ? h('button', {type: 'button', class: 'btn ghost small qu-run', title: 'Open in ANALYSIS', onclick: () => openAnalysis(item.run_id)},
            item.run_id, h('span', {class: 'arrow'}, ' ›'))
          : DASH)))))));
  }

  /** Scrolls to the item openQueue asked for, or says it is not there once a fresh view has loaded. */
  function showWanted() {
    if (!wanted || !view) return;
    const row = $(`#view-queue [data-item="${CSS.escape(wanted)}"]`);
    if (row) {
      if (scrolled !== wanted) {
        scrolled = wanted;
        row.scrollIntoView({block: 'center'});
      }
      return;
    }
    if (loads <= wantedAt) return;
    notice = {glyph: '?', cls: 'muted', text: `${wanted} is not in the queue, nor among its last ${HISTORY_SHOWN} finished items.`};
    wanted = null;
    renderNotice(true);
  }

  // Header chip, summary, hooks -------------------------------------------

  /** Header chip: "Q 2 · 1 held", a run tag while the queue runs, "paused" otherwise. Opens QUEUE. */
  function renderChip(state) {
    const chip = $('#queue-chip');
    const q = state.queue && typeof state.queue === 'object' && !state.queue.error ? state.queue : null;
    chip.hidden = !q;
    if (!q || !changed('queue:chip', q)) return;
    const running = q.state === 'running';
    fill(chip,
      running ? tag('run', 'run') : h('span', {class: 'muted qu-chip-word'}, 'paused'),
      h('b', null, `Q ${q.waiting}`),
      q.held > 0 && h('span', {class: 'warn-text'}, `· ${q.held} held`));
    chip.title = [
      running ? 'The queue is running' : pausedText(q.pause_reason),
      q.active_item && `Now: ${itemLabel(q.active_item)} (${q.active_item.id})`,
      `${q.waiting} waiting, ${q.held} held · global cap: ${capLeftText(q)}`,
    ].filter(Boolean).join('\n');
    chip.setAttribute('aria-label', `Run queue: ${running ? 'running' : 'paused'}, ${q.waiting} waiting${q.held ? `, ${q.held} held` : ''}`);
  }

  /** A pause the person did not ask for (a failure, the cap, Docker, the queue running dry) gets a toast. */
  function notePause(state, prev) {
    const q = state.queue && !state.queue.error ? state.queue : null;
    const p = prev && prev.queue && !prev.queue.error ? prev.queue : null;
    if (!q || !p || p.state !== 'running' || q.state !== 'paused') return;
    if (!pausing) {
      const text = pausedText(q.pause_reason);
      const bad = !/^Queue finished/.test(q.pause_reason || '');
      setTimeout(() => toast(text, bad), 0);  // after app.js's own "Run … ended" toast for this poll
    }
    pausing = false;
  }

  /** The compact queue summary DESIGNS shows on wide screens. */
  function summary(el) {
    const q = view || stateBlock('queue');
    const waiting = view ? view.waiting.length : q ? q.waiting : 0;
    const held = view ? view.held.length : q ? q.held : 0;
    const active = view ? view.active : q ? q.active_item : null;
    fill(el,
      h('button', {type: 'button', class: 'btn ghost small', onclick: () => openQueue()},
        q ? `Queue: ${q.state === 'running' ? 'running' : 'paused'} · ${waiting} waiting${held ? ` · ${held} held` : ''}` : 'Queue',
        h('span', {class: 'arrow'}, ' ›')),
      active && h('div', {class: 'sub'}, 'Now: ', designChip(active)));
  }

  registerView('queue', {
    render: renderQueue,
    setup() {
      $('#queue-chip').addEventListener('click', () => openQueue());
      // One load at start, so the chip's tooltip and other areas' confirms know the global cap (state.queue does
      // not carry it); after that the queue is polled only while QUEUE or DESIGNS is open.
      load.now();
    },
    onState(state, prev) {
      renderChip(state);
      notePause(state, prev);
    },
    /** params: {item}: highlight and scroll to that queue item. */
    open(params) {
      wanted = params.item || null;
      wantedAt = loads;
      scrolled = '';
      if (wanted) load.now();
    },
    hide() {
      wanted = null;
      scrolled = '';
    },
    fastPoll: () => Boolean(view && (view.state === 'running' || view.active)),
    data: () => view,
    refresh: () => load.now(),
    /** {bills, unknown} for these items: the providers their versions bill (app.js queueBills, Start now's confirm). */
    bills: billsFor,
    summary,
    hold: () => (dialogs > 0 ? 'the global cap dialog is open' : ''),
  });
})();
