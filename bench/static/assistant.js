/* Assistant panel (<section id="assistant"> in index.html, opened by #open-assistant).

   Owned by the assistant builder, with assistant.css and demo-assistant.js.
   Brief: "Assistant panel" under "The UI to build", the Assistant endpoints,
   "How the stream behaves" and the SSE wire format.

   Conversations live on the server. The panel lists them (GET
   /api/assistant/conversations), shows one from its stored events (GET
   /api/assistant/conversations/{cid}), then follows it over server-sent
   events: one EventSource on .../stream?after=<last seq>, whose "batch"
   events carry arrays of events in seq order. After three failed connections
   in a row it polls .../{cid}?after=<last seq> instead, every second while a
   turn runs and every 10 s otherwise. Sending a message and Retry call the
   model and bill the Anthropic key; in demo mode demo-assistant.js answers
   both in the browser.

   Events are placed by key, never by arrival order: thinking and text blocks by
   block_key ("<turn>:<step>:<index>"), tool cards by tool_use_id. A durable
   thinking_summary or assistant_text replaces the streamed deltas with the same
   block_key. A streamed block still without its durable twin when its turn
   ends was discarded by the server, so it is folded away as "not kept".

   Contract: everything stays inside this IIFE. app.js and the other areas
   reach the panel only through the hooks registered at the bottom
   (openAssistant, closeAssistant, assistantOpen and holdReason in app.js).
   setup() rebuilds everything inside #assistant, keeping the ids the brief
   names: #assistant-log, #assistant-text, #assistant-send, #assistant-status,
   #assistant-model and #assistant-form. The header button #open-assistant and
   its LED #assistant-btn-led are also this file's. While the panel is open,
   body has the class as-docked, and assistant.css narrows main on wide screens. */

(() => {
  'use strict';

  // ASSISTANT --------------------------------------------------------------

  const STARTERS = [
    'Where are the coverage gaps?',
    'Cheapest runs to get 10 paired tasks per harness on DeepSeek',
    'Price block 1 on Opus 5.5 for all five harnesses',
    'Why did mini-swe-agent fail?',
  ];
  const EFFORT_MAIN = ['low', 'medium', 'high'];
  const EFFORT_MORE = ['xhigh', 'max'];
  const EFFORT_LABEL = {low: 'Low', medium: 'Medium', high: 'High', xhigh: 'Extra high', max: 'Max'};
  const MAX_TEXT = 8000;            // characters in one message (POST .../messages)
  const STOP_GRACE_MS = 400;        // a double-click on Send must not stop the turn it just started
  const STREAM_ERRORS = 3;          // EventSource errors in a row, with no batch between, before polling
  const LONG_STREAM_MS = 20000;     // a connection open this long ended normally (the server's 300 s rotation)
  const POLL_BUSY_MS = 1000;
  const POLL_IDLE_MS = 10000;
  const STALE_TURN_MS = 20000;      // a turn seen starting, then silent this long while /api/state says none runs
  const DOCK_MIN_WIDTH = 1240;      // below this the panel floats over main instead of narrowing it (assistant.css)
  const STEP = 1e6;                 // sort keys: step * STEP + content index
  const TOOL_SLOT = 5e5;            // tool cards sort after every block of their step
  const FINISH_TONE = {stopped: '', refused: 'warn', truncated: 'warn', cap_reached: 'warn', max_steps: 'warn', error: 'bad'};
  const NOTICE_TAG = {
    retry: 'retry', refusal: 'declined', truncated: 'cut off', cap: 'cap', locked: 'locked',
    stopped: 'stopped', max_steps: 'step limit', empty: 'empty',
  };

  const as = {
    built: false,
    open: false,
    status: null,            // GET /api/assistant/status
    statusError: '',
    list: null,              // [ConversationMeta], newest first
    listError: '',
    cid: null,               // the conversation shown
    meta: null,              // its ConversationMeta
    tx: null,                // its transcript (newTranscript)
    loading: false,          // its first GET is in flight
    loadError: '',
    gen: 0,                  // bumped when the conversation changes or the panel closes: older answers are dropped
    fetching: null,          // the GET queue, so meta and events never arrive out of order
    es: null,
    esErrors: 0,
    esOpenedAt: 0,
    polling: false,          // the stream failed: GET ...?after= on a timer instead
    pollTimer: 0,
    linkError: '',           // the last polling or refresh failure, shown in the status line
    attach: null,            // {id, version, title}: sent with the next message as attach_design
    sending: false,          // a POST that starts a turn is in flight
    sentAt: 0,               // performance.now() of the last accepted send, for STOP_GRACE_MS
    sentWall: 0,
    pendingTurn: null,       // {cid, turn} from a 202, until that turn's turn_finished
    stopping: false,
    today: null,             // today's assistant spend: from /api/state, the status, or the latest live usage
    activeTurn: undefined,   // state.assistant.active_turn (any conversation); undefined before the first poll
    retryTimer: 0,
    txCount: 0,
  };

  // Transcript model -------------------------------------------------------
  //
  // One turn per turn number. A turn holds the user message (or a retry mark),
  // keyed slots (blocks, tool cards, notices, errors) and its turn_finished.
  // Slots sort by (key, arrival): blocks by step and index, tool cards after
  // the blocks of the step they belong to, notices and errors right after
  // whatever was placed last when they came.

  function newTranscript() {
    return {
      id: ++as.txCount,
      lastSeq: 0,              // the highest seq applied; events at or below it are skipped
      nextSeq: null,           // the top-level next_seq of the last GET
      turns: new Map(),        // turn number -> turn
      status: '',              // the latest ephemeral status text
      usage: null,             // the latest usage event
      lastEventAt: 0,          // Date.now() of the last applied event
      openKeys: new Set(),     // disclosures the person opened, kept across repaints
      arrival: 0,
      live: false,             // false while the first GET is applied (stored history), true after
    };
  }

  function turnOf(tx, n) {
    const key = Number.isInteger(n) ? n : 0;
    let t = tx.turns.get(key);
    if (!t) {
      t = {
        n: key, user: null, retry: false, started: false, liveOpen: false, finished: null, usage: null,
        slots: new Map(), maxKey: -1, stepSeen: 0, el: null, headEl: null, footEl: null, state: '', dirty: true,
      };
      tx.turns.set(key, t);
    }
    return t;
  }

  function slotFor(tx, t, id, kind, key) {
    let s = t.slots.get(id);
    if (!s) {
      s = {id, kind, key, arrival: tx.arrival++, text: '', durable: false, call: null, result: null, event: null, el: null};
      t.slots.set(id, s);
      t.maxKey = Math.max(t.maxKey, key);
    }
    s.dirty = true;
    return s;
  }

  /** The slot of a thinking or text block, by its block_key "<turn>:<step>:<index>". */
  function blockSlot(tx, t, e, kind) {
    const [, step, index] = String(e.block_key || '').split(':').map(Number);
    const st = Number.isFinite(step) ? step : t.stepSeen;
    t.stepSeen = Math.max(t.stepSeen, st);
    return slotFor(tx, t, `b:${e.block_key}`, kind, st * STEP + (Number.isFinite(index) ? index : 0));
  }

  function applyEvent(tx, e) {
    const t = turnOf(tx, e.turn);
    t.dirty = true;
    switch (e.type) {
      case 'user_message':
        t.user = e;
        t.headEl = null;
        break;
      case 'turn_started':
        t.started = true;
        t.retry = Boolean(e.retry);
        t.headEl = null;
        if (tx.live) t.liveOpen = true;
        break;
      case 'status':
        tx.status = typeof e.text === 'string' ? e.text : '';
        break;
      case 'thinking_delta':
      case 'text_delta': {
        const s = blockSlot(tx, t, e, e.type === 'thinking_delta' ? 'think' : 'text');
        if (!s.durable && typeof e.text === 'string') s.text += e.text;
        break;
      }
      case 'thinking_summary':
      case 'assistant_text': {
        // Replaces whatever streamed under the same block_key.
        const s = blockSlot(tx, t, e, e.type === 'thinking_summary' ? 'think' : 'text');
        s.text = typeof e.text === 'string' ? e.text : '';
        s.durable = true;
        break;
      }
      case 'tool_call':
      case 'tool_result': {
        const id = `t:${e.tool_use_id || e.seq}`;
        const s = t.slots.get(id) || slotFor(tx, t, id, 'tool', t.stepSeen * STEP + TOOL_SLOT);
        s.dirty = true;
        if (e.type === 'tool_call') s.call = e;
        else s.result = e;
        break;
      }
      case 'usage':
        t.usage = e;
        tx.usage = e;
        if (tx.live && isNum(e.today_usd)) as.today = e.today_usd;
        break;
      case 'notice':
      case 'turn_error':
        slotFor(tx, t, `${e.type === 'notice' ? 'n' : 'e'}:${e.seq}`, e.type === 'notice' ? 'notice' : 'error', t.maxKey).event = e;
        break;
      case 'turn_finished':
        t.finished = e;
        t.liveOpen = false;
        tx.status = '';
        break;
      default:
        break;  // a type this panel does not know yet
    }
  }

  /** Applies events in seq order, skipping any already applied. Returns true when a live turn finished. */
  function applyEvents(events, gen) {
    if (gen !== as.gen || !as.tx || !Array.isArray(events) || !events.length) return false;
    const tx = as.tx;
    let any = false;
    let finished = false;
    for (const e of events.slice().sort((a, b) => a.seq - b.seq)) {
      if (!e || !Number.isInteger(e.seq) || e.seq <= tx.lastSeq) continue;
      tx.lastSeq = e.seq;
      applyEvent(tx, e);
      any = true;
      if (e.type === 'turn_finished') {
        if (as.pendingTurn && as.pendingTurn.turn === e.turn) as.pendingTurn = null;
        as.stopping = false;
        if (tx.live) {
          finished = true;
          announce(e.status === 'done' ? 'The assistant finished.' : `The assistant's turn ended: ${statusLabel(e.status)}.`);
        }
      }
      if (e.type === 'status' && tx.live) announce(e.text);
    }
    if (!any) return false;
    tx.lastEventAt = Date.now();
    paint();
    if (finished) afterTurn(gen);
    return finished;
  }

  const latestTurn = (tx) => {
    let last = null;
    if (tx) for (const t of tx.turns.values()) if (!last || t.n > last.n) last = t;
    return last;
  };

  // Server ---------------------------------------------------------------------

  const convPath = (cid, rest = '') => `/api/assistant/conversations/${enc(cid)}${rest}`;

  async function loadStatus() {
    try {
      as.status = await api('GET', '/api/assistant/status');
      as.statusError = '';
      if (isNum(as.status.today_usd)) as.today = as.status.today_usd;
      if (as.status.active_turn !== undefined) as.activeTurn = as.status.active_turn;
    } catch (err) {
      as.statusError = err.message;
    }
    paint();
  }

  async function loadList() {
    try {
      const data = await api('GET', '/api/assistant/conversations');
      as.list = Array.isArray(data.conversations) ? data.conversations : [];
      as.listError = '';
      const mine = as.cid && as.list.find((c) => c.id === as.cid);
      if (mine) as.meta = mine;
      const recap = as.tx && as.tx.turns.get(0);
      if (recap) recap.dirty = true;  // its label names the conversation it continues
    } catch (err) {
      as.listError = err.message;
    }
    paint();
  }

  /** GET .../{cid}?after=<last seq>: fresh meta and the events not applied yet. Queued, never overlapping. */
  function fetchEvents(gen) {
    const run = () => (gen === as.gen && as.cid ? getEvents(gen) : false);
    as.fetching = (as.fetching || Promise.resolve()).then(run, run);
    return as.fetching;
  }

  async function getEvents(gen) {
    const tx = as.tx;
    let data;
    try {
      data = await api('GET', convPath(as.cid, `?after=${tx.lastSeq}`));
    } catch (err) {
      if (gen !== as.gen) return false;
      if (tx.live) as.linkError = err.status === 503 ? `Busy, trying again: ${err.message}` : err.message;
      else as.loadError = err.message;
      paint();
      return false;
    }
    if (gen !== as.gen) return false;
    as.linkError = '';
    if (data.conversation) as.meta = data.conversation;
    if (data.active_turn !== undefined) as.activeTurn = data.active_turn;
    // The stream continues after the last event applied. That is next_seq - 1 (the top-level one: the meta's
    // next_seq is only saved when a turn ends), except that it cannot skip an event added while the GET was answered.
    if (isNum(data.next_seq)) tx.nextSeq = data.next_seq;
    applyEvents(data.events, gen);
    tx.live = true;
    paint();
    return true;
  }

  /** After a live turn_finished: the meta (spend, title, status) and the list (titles, busy marks) changed. */
  function afterTurn(gen) {
    fetchEvents(gen);
    loadList();
  }

  // Stream, with the polling fallback ---------------------------------------

  function connect(gen) {
    if (gen !== as.gen || !as.open || !as.cid || !as.tx) return;
    disconnect();
    if (as.polling) {
      schedulePoll(gen, 0);
      return;
    }
    const Source = window.EventSource;  // read now: demo mode replaces it after this file loads
    let es;
    try {
      es = new Source(convPath(as.cid, `/stream?after=${as.tx.lastSeq}`));
    } catch {
      startPolling(gen);
      return;
    }
    as.es = es;
    as.esErrors = 0;
    as.esOpenedAt = 0;
    es.addEventListener('open', () => {
      if (as.es === es) as.esOpenedAt = Date.now();
    });
    // Every event is named "batch"; the default onmessage never fires.
    es.addEventListener('batch', (e) => {
      if (as.es !== es) return;
      as.esErrors = 0;
      as.linkError = '';
      let events;
      try {
        events = JSON.parse(e.data);
      } catch {
        return;
      }
      applyEvents(events, gen);
    });
    es.addEventListener('error', () => {
      if (as.es !== es) return;
      // The server closes every stream after 300 s and the browser reconnects with Last-Event-ID: not a failure.
      if (as.esOpenedAt && Date.now() - as.esOpenedAt > LONG_STREAM_MS) as.esErrors = 0;
      as.esOpenedAt = 0;
      as.esErrors++;
      // CLOSED: the browser gave up (an HTTP error), so it will not try again by itself.
      if (as.esErrors >= STREAM_ERRORS || es.readyState === 2) startPolling(gen);
    });
  }

  function disconnect() {
    if (as.es) {
      as.es.close();
      as.es = null;
    }
    clearTimeout(as.pollTimer);
  }

  function startPolling(gen) {
    if (gen !== as.gen) return;
    disconnect();
    as.polling = true;
    paint();
    schedulePoll(gen, 0);
  }

  function schedulePoll(gen, ms) {
    clearTimeout(as.pollTimer);
    as.pollTimer = setTimeout(async () => {
      if (gen !== as.gen || !as.open || !as.polling) return;
      await fetchEvents(gen);
      if (gen !== as.gen || !as.open || !as.polling) return;
      schedulePoll(gen, turnRunning() ? POLL_BUSY_MS : POLL_IDLE_MS);
    }, ms);
  }

  /** Reloads what changed while nothing was listening (or the first time), then follows the stream again. */
  async function sync() {
    disconnect();
    const gen = ++as.gen;
    as.polling = false;
    if (!as.cid || !as.tx) {
      paint();
      return;
    }
    as.loading = !as.tx.live;
    as.loadError = '';
    paint();
    const ok = await fetchEvents(gen);
    if (gen !== as.gen) return;
    as.loading = false;
    paint();
    if (ok) connect(gen);
  }

  async function showConversation(cid) {
    as.cid = cid || null;
    as.meta = cid && as.list ? as.list.find((c) => c.id === cid) || null : null;
    as.tx = cid ? newTranscript() : null;
    as.pendingTurn = null;
    as.stopping = false;
    as.linkError = '';
    if (cid) prefs.set('assistant.conversation', cid);
    await sync();
  }

  /** The conversation to show when none is chosen: the one answering now, the last one shown, or the newest open one. */
  function defaultConversation() {
    const list = (as.list || []).filter((c) => c.status !== 'archived' && !c.error);
    const busyCid = as.activeTurn && as.activeTurn.conversation_id;
    if (busyCid && list.some((c) => c.id === busyCid)) return busyCid;
    const saved = prefs.get('assistant.conversation', null);
    if (saved && list.some((c) => c.id === saved)) return saved;
    return list.length ? list[0].id : null;
  }

  // State ----------------------------------------------------------------------

  /** Whether the assistant can take a message: from /api/state, else from the status. */
  function availability() {
    const a = stateBlock('assistant') || as.status;
    if (!a) return {known: false, available: false, reason: ''};
    return {known: true, available: a.available === true, reason: a.reason || ''};
  }

  const keyMissing = () => Boolean((as.status && as.status.key_set === false) || /_API_KEY/.test(availability().reason));

  /** The shown conversation's turn runs: the server says so, this page sent it, or it saw it start and not finish. */
  function turnRunning() {
    if (!as.cid) return false;
    const a = as.activeTurn;
    if (a && a.conversation_id === as.cid) {
      // A poll answered just before turn_finished arrived still names the turn; this page knows better.
      const t = as.tx && as.tx.turns.get(a.turn);
      if (!(t && t.finished)) return true;
    }
    const p = as.pendingTurn;
    if (p && p.cid === as.cid) {
      const t = as.tx && as.tx.turns.get(p.turn);
      if (!(t && t.finished)) return true;
    }
    const t = latestTurn(as.tx);
    return Boolean(t && t.liveOpen && !t.finished);
  }

  /** Another conversation's turn runs: the server takes one turn at a time. */
  const busyElsewhere = () => Boolean(as.activeTurn && as.activeTurn.conversation_id && as.activeTurn.conversation_id !== as.cid);

  const closedReason = (meta) => (!meta ? ''
    : meta.error ? 'This conversation could not be read.'
      : meta.status === 'archived' ? 'Archived: read only.'
        : meta.status === 'locked' ? 'Locked: continue in a new conversation.' : '');

  function canSend() {
    return availability().available && !as.sending && !as.loading && !turnRunning() && !busyElsewhere() && !closedReason(as.meta);
  }

  // Actions --------------------------------------------------------------------

  /** Sends the composer's text. A first message without a conversation starts one at the default effort. */
  async function send() {
    const box = $('#assistant-text');
    const typed = box.value;
    const text = typed.trim();
    if (!text || !canSend()) return;
    if (text.length > MAX_TEXT) {
      toast(`Messages are limited to ${count(MAX_TEXT)} characters; this one has ${count(text.length)}.`, true);
      return;
    }
    as.sending = true;
    paint();
    try {
      let cid = as.cid;
      if (!cid) {
        const body = as.attach ? {design: {id: as.attach.id, version: as.attach.version}} : {};
        const meta = await api('POST', '/api/assistant/conversations', body);
        await loadList();
        await showConversation(meta.id);
        cid = meta.id;
      }
      // Calls the model (billed). Errors leave the text in the box: a 400 for key-like text, a 409 while busy.
      const out = await api('POST', convPath(cid, '/messages'), {
        text, attach_design: as.attach ? {id: as.attach.id, version: as.attach.version} : null,
      });
      if (box.value === typed) {
        box.value = '';
        resizeBox();
      }
      as.attach = null;
      accepted(cid, out);
    } catch (err) {
      toast(err.message, true);
    } finally {
      as.sending = false;
      paint();
    }
  }

  /** A 202 {turn, accepted_seq} from messages or retry: the turn runs; make sure something is listening. */
  function accepted(cid, out) {
    as.pendingTurn = {cid, turn: out && out.turn};
    as.sentAt = performance.now();
    as.sentWall = Date.now();
    if (cid !== as.cid) return;
    if (as.polling) schedulePoll(as.gen, 300);
    else if (!as.es) connect(as.gen);
  }

  async function stopTurn() {
    if (!as.cid || as.stopping || performance.now() - as.sentAt < STOP_GRACE_MS) return;
    as.stopping = true;
    paint();
    try {
      await api('POST', convPath(as.cid, '/stop'));
      announce('Stopping. A running tool finishes first.');
    } catch (err) {
      as.stopping = false;
      toast(err.message, true);
      paint();
    }
  }

  /** Runs the last turn again after an error (billed). Only offered right after a retryable turn_error. */
  async function retryTurn() {
    if (!as.cid || as.sending) return;
    as.sending = true;
    paint();
    try {
      accepted(as.cid, await api('POST', convPath(as.cid, '/retry')));
    } catch (err) {
      toast(err.message, true);
    } finally {
      as.sending = false;
      paint();
    }
  }

  /** A new conversation seeded with a recap of this one (no model call), then shows it. */
  async function continueFrom(button, cid) {
    await busy(button, async () => {
      const meta = await api('POST', convPath(cid, '/continue'));
      toast(`Continued in "${meta.title}"`);
      await loadList();
      await showConversation(meta.id);
    });
  }

  async function archive(button) {
    const meta = as.meta;
    if (!meta) return;
    const ok = await confirmAction({
      code: 'AI', title: 'Archive this conversation?', ok: 'Archive', tone: 'danger',
      body: [
        h('p', null, `"${meta.title}" moves under Archived in the list. You can still read it, but it takes no more messages.`),
        h('p', null, 'Continue starts a new conversation from its recap.'),
      ],
    });
    if (!ok) return;
    await busy(button, async () => {
      await api('POST', convPath(meta.id, '/archive'));
      toast('Conversation archived');
      await loadList();
      if (as.cid === meta.id) await showConversation(defaultConversation());
    });
  }

  /** New: asks for the effort (and an optional title), then POST /api/assistant/conversations. */
  async function newConversation() {
    const fallback = (as.status && as.status.effort_default) || 'medium';
    const radio = (level) => h('label', {class: 'as-effort-opt'},
      h('input', {type: 'radio', name: 'effort', value: level, checked: level === fallback}),
      h('span', null, EFFORT_LABEL[level] || level), level === fallback && h('small', null, 'default'));
    const design = as.attach;
    const meta = await formDialog({
      code: 'AI', title: 'New conversation', ok: 'Start',
      body: [
        h('fieldset', {class: 'as-effort'},
          h('legend', null, 'Effort'),
          h('div', {class: 'as-effort-row'}, EFFORT_MAIN.map(radio)),
          h('details', {class: 'as-effort-more', open: EFFORT_MORE.includes(fallback)},
            h('summary', null, 'More'), h('div', {class: 'as-effort-row'}, EFFORT_MORE.map(radio)))),
        h('p', {class: 'muted'}, 'Higher effort thinks longer and costs more per message. It stays fixed for the conversation.'),
        field('Title', h('input', {name: 'title', maxlength: 80, placeholder: 'Optional: named after your first message'})),
        design && h('label', {class: 'as-check'}, h('input', {type: 'checkbox', name: 'design', checked: true}),
          h('span', null, 'Start with design ', h('b', {class: 'mono'}, `${design.title || design.id} v${design.version}`))),
      ],
      focus: 'input[name="effort"]:checked',
      read: (form) => {
        const picked = form.querySelector('input[name="effort"]:checked');
        const body = {effort: picked ? picked.value : fallback};
        const title = form.elements.title.value.trim();
        if (title) body.title = title;
        if (design && form.elements.design && form.elements.design.checked) body.design = {id: design.id, version: design.version};
        return api('POST', '/api/assistant/conversations', body);
      },
    });
    if (!meta) return;
    await loadList();
    await showConversation(meta.id);
    focusBox();
  }

  /** Puts text in the composer without sending it; text already there stays, the new text goes below it. */
  function putPrompt(text) {
    const box = $('#assistant-text');
    const now = box.value.trim();
    if (!now) box.value = text;
    else if (!now.includes(text.trim())) box.value = `${box.value.replace(/\s+$/, '')}\n\n${text}`;
    resizeBox();
  }

  /** A starter question goes into the box (never over text already typed); Enter sends it. */
  function useStarter(text) {
    const box = $('#assistant-text');
    putPrompt(text);
    box.focus();
    box.setSelectionRange(box.value.length, box.value.length);
  }

  /** Opens a design version in DESIGNS; a panel floating over the page gets out of the way. */
  function toDesign(id, version) {
    openDesign(id, version);
    closeIfFloating();
  }

  function closeIfFloating() {
    if (window.innerWidth < DOCK_MIN_WIDTH) closePanel();
  }

  /** Opens the key dialog for the assistant's key (its keys row is looked up when clicked), or Setup without one. */
  function keyButton() {
    return h('button', {type: 'button', class: 'btn small primary', onclick: () => {
      const row = ui.state && ui.state.keys.find((k) => k.env_key === 'ANTHROPIC_API_KEY');
      if (row) openKeyDialog(row);
      else openDrawer('system');
    }}, 'Set ANTHROPIC_API_KEY on Setup');
  }

  const continueButton = (cid) => h('button', {type: 'button', class: 'btn small primary', onclick: (e) => continueFrom(e.currentTarget, cid)},
    'Continue in a new conversation');

  // Panel ------------------------------------------------------------------------

  function setOpen(open) {
    as.open = open;
    $('#assistant').hidden = !open;
    const button = $('#open-assistant');
    button.setAttribute('aria-expanded', String(open));
    button.classList.toggle('active', open);
    const docked = document.body.classList.contains('as-docked');
    document.body.classList.toggle('as-docked', open);
    // Docking changes main's width; views that size charts to it re-measure on resize.
    if (docked !== open && window.innerWidth >= DOCK_MIN_WIDTH) window.dispatchEvent(new Event('resize'));
  }

  async function openPanel(opts = {}) {
    if (!as.built) return;
    const wasOpen = as.open;
    setOpen(true);
    if (typeof opts.prompt === 'string' && opts.prompt.trim()) putPrompt(opts.prompt);
    const d = opts.design;
    if (d && d.id && Number.isInteger(d.version)) as.attach = {id: d.id, version: d.version, title: d.title || null};
    paint();
    if (!wasOpen || opts.prompt) focusBox();
    const loads = Promise.all([loadStatus(), loadList()]);
    if (opts.conversationId && opts.conversationId !== as.cid) {
      await showConversation(opts.conversationId);
      return;
    }
    await loads;
    if (!as.open) return;
    if (!as.cid) {
      const cid = defaultConversation();
      if (cid) await showConversation(cid);
      else paint();
    } else if (!wasOpen || !as.es) {
      await sync();  // events that came while the panel was closed
    }
  }

  function closePanel() {
    if (!as.open) return;
    setOpen(false);
    disconnect();
    as.gen++;  // answers in flight are dropped; opening again re-syncs from the last seq applied
    as.loading = false;
  }

  function focusBox() {
    const box = $('#assistant-text');
    if (!box.disabled) box.focus();
    else $('#assistant-close').focus();
  }

  function resizeBox() {
    const box = $('#assistant-text');
    box.style.height = 'auto';
    box.style.height = `${Math.min(box.scrollHeight, 140)}px`;
    box.style.overflowY = box.scrollHeight > 140 ? 'auto' : 'hidden';
    paintCount();
  }

  let lastAnnounced = '';
  /** The screen-reader status line; repeats are dropped. */
  function announce(text) {
    if (!text || text === lastAnnounced) return;
    lastAnnounced = text;
    $('#assistant-status').textContent = text;
  }

  // Painting -------------------------------------------------------------------

  let paintQueued = false;
  /** Repaints the open panel at most once a frame. */
  function paint() {
    if (!as.built || paintQueued) return;
    paintQueued = true;
    requestAnimationFrame(() => {
      paintQueued = false;
      paintPanel();
    });
  }

  function paintPanel() {
    paintButtonLed();
    paintPanelLed();
    if (!as.open) return;
    paintHeader();
    paintSpend();
    paintBanners();
    paintLog();
    paintComposer();
  }

  function paintPanelLed() {
    const av = availability();
    const named = as.activeTurn;
    const busyNow = Boolean(named && !(named.conversation_id === as.cid && !turnRunning())) || turnRunning();
    $('#assistant-led').className = busyNow ? 'led warn live' : av.available ? 'led ok' : 'led';  // as the header button's LED
    const u = as.tx && as.tx.usage;
    const requested = (as.status && as.status.model) || (as.meta && as.meta.model) || '';
    const model = $('#assistant-model');
    if (u && u.served_model) {
      model.textContent = `${u.served_model}${u.fallback_served ? ' (fallback)' : ''}`;
      model.title = u.fallback_served ? `${requested || 'The model'} declined; ${u.served_model} answered the last request`
        : `The model that answered the last request`;
    } else {
      model.textContent = requested || DASH;
      model.title = 'The assistant model';
    }
  }

  function paintHeader() {
    const list = as.list || [];
    const sel = $('#as-conv');
    const rows = list.map((c) => [c.id, c.title, c.status, c.created_at, c.active]);
    if (changed('assistant:conversations', [rows, as.cid, Boolean(as.list)])) {
      const label = (c) => `${c.title || c.id}${c.status === 'locked' ? ' · locked' : ''}${c.active ? ' · answering' : ''} · ${shortWhen(c.created_at)}`;
      const open = list.filter((c) => c.status !== 'archived');
      const archived = list.filter((c) => c.status === 'archived');
      const known = list.some((c) => c.id === as.cid);
      fill(sel,
        !list.length && !as.cid && h('option', {value: ''}, as.list ? 'No conversations yet' : 'Loading…'),
        list.length > 0 && !as.cid && h('option', {value: ''}, 'Choose a conversation'),
        as.cid && !known && h('option', {value: as.cid}, (as.meta && as.meta.title) || as.cid),
        open.map((c) => h('option', {value: c.id, title: c.id}, label(c))),
        archived.length > 0 && h('optgroup', {label: 'Archived'}, archived.map((c) => h('option', {value: c.id, title: c.id}, label(c)))));
      sel.value = as.cid || '';
      sel.disabled = !list.length && !as.cid;
    }
    const meta = as.meta;
    const running = turnRunning();
    if (changed('assistant:effort', meta && meta.effort)) {
      fill($('#as-effort'), meta && meta.effort && tip(tag(`effort ${meta.effort}`), 'Effort', ['Set when the conversation started']));
    }
    const archiveButton = $('#as-archive');
    archiveButton.hidden = !meta || meta.status === 'archived';
    archiveButton.disabled = running || Boolean(meta && meta.active);
    archiveButton.title = archiveButton.disabled ? 'Stop the running turn first' : 'Archive this conversation';
    $('#as-continue').hidden = !meta || !(meta.status === 'locked' || meta.status === 'archived');
  }

  function shortWhen(iso) {
    const d = new Date(iso);
    if (!iso || Number.isNaN(d.getTime())) return DASH;
    return d.toLocaleString(undefined, {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'});
  }

  /** "This turn $0.06 of $1.00 · Conversation $0.42 · Today $1.10 of $5.00": the latest usage, else the poll and the meta. */
  function paintSpend() {
    const a = stateBlock('assistant');
    const s = as.status;
    const cap = a && isNum(a.daily_cap_usd) ? a.daily_cap_usd : s ? s.daily_cap_usd : null;
    const turnCap = s && s.spend ? s.spend.per_turn_cap_usd : null;
    const last = latestTurn(as.tx);
    const turnUsd = last && last.n > 0 && last.usage ? last.usage.turn_usd : null;
    const convUsd = as.tx && as.tx.usage ? as.tx.usage.conversation_usd : as.meta ? as.meta.spend_usd : null;
    const today = as.today;
    const designs = (as.meta && Array.isArray(as.meta.designs) && as.meta.designs) || [];
    if (changed('assistant:spend', [turnUsd, turnCap, convUsd, today, cap, Boolean(as.cid)])) {
      const fraction = isNum(today) && isNum(cap) && cap > 0 ? today / cap : null;
      const tone = fraction === null ? '' : fraction >= 1 ? 'bad' : fraction >= 0.8 ? 'warn' : '';
      fill($('#as-spend-line'),
        h('span', null, 'This turn ', h('b', null, money(turnUsd)), isNum(turnCap) && h('span', {class: 'faint'}, ` of ${money(turnCap)}`)),
        h('span', null, 'Conversation ', h('b', null, as.cid ? money(convUsd) : DASH)),
        h('span', {class: tone ? `${tone}-text` : null}, 'Today ', h('b', null, money(today)), ` of ${money(cap)}`,
          tone === 'bad' ? ' ✕ cap reached' : tone === 'warn' ? ' ! near the cap' : ''));
      const g = $('#as-spend-gauge');
      fill(g, fraction !== null && gauge(fraction, {tone}));
      tip(g, 'Today against the daily cap', [[money(today), 'spent today'], [money(cap), 'daily cap'], [money(turnCap), 'per-message cap']]);
    }
    if (changed('assistant:designs', [designs, as.cid, [...draftTitles()]])) {
      const row = $('#as-designs');
      row.hidden = !designs.length;
      const titles = draftTitles();
      fill(row, h('span', {class: 'as-k'}, 'Designs'), designs.map((d) => {
        const chip = designChip({id: d.id, version: d.version, title: titles.get(d.id) || null});
        chip.addEventListener('click', closeIfFloating);
        return chip;
      }));
    }
  }

  /** design id -> title, from the drafts this transcript saved (meta.designs carries ids only). */
  function draftTitles() {
    const titles = new Map();
    if (as.tx) {
      for (const t of as.tx.turns.values()) {
        for (const s of t.slots.values()) {
          const d = s.result && s.result.draft;
          if (d && d.design_id && d.title) titles.set(d.design_id, d.title);
        }
      }
    }
    return titles;
  }

  function paintBanners() {
    const av = availability();
    const meta = as.meta;
    const err = stateError('assistant') || as.statusError;
    const other = busyElsewhere() ? as.activeTurn.conversation_id : null;
    const otherMeta = other && (as.list || []).find((c) => c.id === other);
    const key = [av, err, keyMissing(), meta && [meta.id, meta.status, meta.locked_reason, meta.error], other, otherMeta && otherMeta.title, as.listError];
    if (!changed('assistant:banners', key)) return;
    const banner = (tone, text, ...actions) => h('div', {class: `banner ${tone} as-banner`},
      h('span', {class: 'as-glyph'}, tone === 'bad' ? '✕' : '!'),
      h('div', null, h('div', null, text), actions.filter(Boolean).length > 0 && h('div', {class: 'actions'}, actions)));
    fill($('#as-banners'),
      err && banner('bad', `Assistant status: ${err}`),
      !err && av.known && !av.available && banner('warn', av.reason || 'The assistant is unavailable.', keyMissing() && keyButton()),
      meta && meta.error && banner('bad', `This conversation could not be read: ${meta.error}`),
      meta && !meta.error && meta.status === 'locked' && banner('warn',
        `Locked: ${meta.locked_reason || 'this conversation takes no more messages.'}`, continueButton(meta.id)),
      meta && !meta.error && meta.status === 'archived' && banner('warn', 'Archived: read only.', continueButton(meta.id)),
      other && banner('warn', `The assistant is answering in "${(otherMeta && otherMeta.title) || other}". One message runs at a time.`,
        h('button', {type: 'button', class: 'btn small', onclick: () => showConversation(other)}, 'Show it')),
      as.listError && banner('bad', `Conversations: ${as.listError}`));
  }

  function paintLog() {
    const log = $('#assistant-log');
    const pinned = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    const top = log.scrollTop;
    const tx = as.tx;
    if (!tx || as.loading || as.loadError || !tx.turns.size) {
      const key = [as.cid, as.loading, as.loadError, availability().available, tx && tx.turns.size];
      if (log.dataset.tx !== 'empty' || changed('assistant:empty', key)) {
        changed('assistant:empty', key);
        log.dataset.tx = 'empty';
        fill(log, emptyState());
      }
      return;
    }
    if (log.dataset.tx !== String(tx.id)) {
      log.dataset.tx = String(tx.id);
      fill(log);
    }
    const turns = [...tx.turns.values()].sort((a, b) => a.n - b.n);
    const running = turnRunning();
    const latest = turns[turns.length - 1];
    const gate = retryGate();
    for (const t of turns) paintTurn(t, t === latest, running, gate);
    const want = turns.map((t) => t.el);
    if (!sameChildren(log, want)) log.replaceChildren(...want);
    log.setAttribute('aria-busy', String(running));
    log.scrollTop = pinned ? log.scrollHeight : top;
  }

  const sameChildren = (el, list) => el.children.length === list.length && list.every((node, i) => el.children[i] === node);

  function emptyState() {
    if (as.loadError) {
      return h('div', {class: 'as-empty'},
        h('p', {class: 'bad-text'}, `✕ Could not load this conversation: ${as.loadError}`),
        h('div', {class: 'actions'}, h('button', {type: 'button', class: 'btn small', onclick: () => sync()}, 'Try again')));
    }
    if (as.loading) return h('div', {class: 'as-empty'}, h('p', {class: 'muted'}, 'Loading the conversation…'));
    const ready = availability().available;
    return h('div', {class: 'as-empty'},
      eyebrow(as.cid ? 'New conversation' : 'Ask about this bench'),
      h('p', null, 'The assistant reads the ledger, checks coverage, prices runs and saves draft designs for you to review.'),
      h('div', {class: 'as-starters'}, STARTERS.map((q) => h('button', {
        type: 'button', class: 'as-starter', disabled: !ready, title: 'Puts the question in the box; press Enter to send it',
        onclick: () => useStarter(q),
      }, h('span', {class: 'arrow'}, '›'), q))));
  }

  /** What decides whether the latest turn's Retry is offered; its error slot repaints when this changes. */
  function retryGate() {
    return [availability().available, turnRunning(), busyElsewhere(), as.sending, closedReason(as.meta)].join('|');
  }

  function paintTurn(t, latest, running, gate) {
    // A finished turn stays finished; the latest unfinished one is live while it runs, otherwise it was cut off.
    const state = t.finished ? 'done' : latest && running ? 'live' : t.started ? 'cut' : 'idle';
    const stateChanged = t.state !== state;
    const gateKey = latest ? gate : 'earlier';
    const gateChanged = t.gate !== gateKey;
    if (t.el && !t.dirty && !stateChanged && !gateChanged) return;
    t.state = state;
    t.gate = gateKey;
    t.dirty = false;
    if (!t.el) t.el = h('div', {class: 'as-turn', 'data-turn': String(t.n)});
    const headKey = t.user ? (t.user.recap ? `recap:${recapTitle()}` : `user:${t.user.seq}`) : t.retry ? 'retry' : '';
    if (t.headKey !== headKey) {
      t.headKey = headKey;
      t.headEl = headEl(t);
    }
    const slots = [...t.slots.values()].sort((a, b) => a.key - b.key || a.arrival - b.arrival);
    for (const s of slots) {
      if (!s.el || s.dirty || stateChanged || (s.kind === 'error' && (gateChanged || latest))) {
        const el = slotEl(s, state, latest);
        if (s.el && s.el.parentNode) s.el.replaceWith(el);
        s.el = el;
        s.dirty = false;
      }
    }
    t.footEl = footEl(t, state);
    const want = [t.headEl, ...slots.map((s) => s.el), t.footEl].filter(Boolean);
    if (!sameChildren(t.el, want)) t.el.replaceChildren(...want);
    t.el.classList.toggle('live', state === 'live');
  }

  function headEl(t) {
    if (t.user) return userEl(t.user);
    if (t.retry) return h('div', {class: 'as-retry-mark'}, '↻ Retried after an error');
    return null;
  }

  /** The conversation a recap continues: its title once the list has it, else its id. */
  function recapTitle() {
    const from = as.meta && as.meta.continued_from;
    const prev = from && (as.list || []).find((c) => c.id === from);
    return prev ? `"${prev.title}"` : from || 'the previous conversation';
  }

  function userEl(e) {
    if (e.recap) {
      return disclosure(`recap:${e.seq}`, 'as-recap', `Recap of ${recapTitle()}`, h('div', {class: 'as-recap-body'}, e.text));
    }
    const d = e.attach_design;
    let chip = null;
    if (d && d.id) {
      chip = designChip({id: d.id, version: d.version});
      chip.addEventListener('click', closeIfFloating);
    }
    return h('div', {class: 'as-msg as-user'}, h('div', null, e.text), chip && h('div', {class: 'as-attached'}, 'Attached ', chip));
  }

  /** A <details> whose open state survives repaints. */
  function disclosure(id, cls, summary, body) {
    const keys = as.tx.openKeys;
    const d = h('details', {class: cls, open: keys.has(id)}, h('summary', null, summary), body);
    d.addEventListener('toggle', () => {
      if (d.open) keys.add(id);
      else keys.delete(id);
    });
    return d;
  }

  function slotEl(s, state, latest) {
    if (s.kind === 'tool') return toolEl(s, state);
    if (s.kind === 'notice') return noticeEl(s.event);
    if (s.kind === 'error') return errorEl(s.event, latest);
    // A streamed block without its durable twin once the turn is over was discarded by the server.
    if (!s.durable && state !== 'live') {
      return disclosure(s.id, 'as-dropped', `${s.kind === 'think' ? 'Reasoning' : 'Reply'} not kept`,
        h('div', null,
          h('p', {class: 'faint'}, 'Streamed, then discarded by the bench: a refused reply, a reply cut off with a tool call, or what a declined model wrote before a fallback.'),
          h('div', {class: 'as-dropped-body'}, s.text || DASH)));
    }
    if (s.kind === 'think') {
      return disclosure(s.id, `as-think${s.durable ? '' : ' streaming'}`, s.durable ? 'Reasoning summary' : 'Reasoning…',
        h('div', {class: 'as-think-body'}, s.text));
    }
    return h('div', {class: `as-msg as-bot${s.durable ? '' : ' streaming'}`},
      s.text ? markdownOrText(s.text) : h('p', {class: 'muted'}, '…'));
  }

  /** Model text as Markdown; if that ever fails on odd input, as plain text, so one block never blanks the log. */
  function markdownOrText(text) {
    try {
      return renderMarkdown(text);
    } catch (err) {
      console.warn('assistant: Markdown failed, shown as text:', err);
      return h('p', {class: 'as-plain'}, text);
    }
  }

  function toolEl(s, state) {
    const c = s.call;
    const r = s.result;
    const name = (c && c.name) || (r && r.name) || 'a tool';
    const ok = r ? r.ok !== false : null;
    const label = r ? r.summary || `${name} ${ok ? 'finished' : 'failed'}`
      : state === 'live' ? `Calling ${name}…` : `${name}: no result`;
    const input = c && c.input !== undefined ? JSON.stringify(c.input, null, 2) : null;
    return h('div', {class: 'as-tool-wrap'},
      disclosure(s.id, `as-tool ${!r ? (state === 'live' ? 'run' : 'cut') : ok ? 'ok' : 'bad'}`,
        [h('span', {class: 'as-glyph'}, !r ? '◆' : ok ? '✓' : '✕'), h('span', {class: 'as-tool-label'}, label)],
        h('div', {class: 'as-tool-body'},
          h('div', {class: 'as-tool-name'}, h('code', null, name), r && !ok && h('span', {class: 'bad-text'}, 'failed')),
          input ? h('pre', {class: 'as-json'}, input) : h('p', {class: 'faint'}, 'The call input is not stored with this event.'))),
      r && r.draft && draftCard(r.draft));
  }

  /** "Draft saved: <title> v<version> · $0.22 p50 ($0.44 p90) · 0 problems · Open", from the structured draft only. */
  function draftCard(d) {
    const problems = isNum(d.problems) ? d.problems : null;
    const estimate = isNum(d.expected_usd) || isNum(d.p90_usd) ? `${money(d.expected_usd)} p50 (${money(d.p90_usd)} p90)` : 'no estimate';
    return h('div', {class: 'as-draft'},
      h('div', {class: 'as-draft-title'}, tag('AI', 'ai'),
        h('span', null, 'Draft saved: ', h('b', null, `${d.title || d.design_id} v${d.version}`))),
      h('div', {class: 'as-draft-meta'},
        h('span', null, estimate),
        problems === null ? h('span', {class: 'muted'}, 'problems unknown')
          : h('span', {class: problems > 0 ? 'bad-text' : 'ok-text'}, problems > 0 ? `✕ ${plural(problems, 'problem')}` : '✓ 0 problems'),
        h('span', {class: 'faint', title: d.spec_hash || null}, shortHash(d.spec_hash)),
        h('button', {type: 'button', class: 'btn small primary', onclick: () => toDesign(d.design_id, d.version)}, 'Open ›')));
  }

  function noticeEl(e) {
    const code = e.code || '';
    const warnLevel = e.level === 'warn';
    if (code === 'locked') {
      return h('div', {class: 'banner warn as-banner as-inline'}, h('span', {class: 'as-glyph'}, '!'),
        h('div', null, h('div', null, e.message || 'This conversation is locked.'),
          as.cid && h('div', {class: 'actions'}, continueButton(as.cid))));
    }
    const text = code === 'fallback'
      ? `Answered by ${e.to_model || 'a fallback model'} after ${e.from_model || 'the requested model'} declined`
      : e.message || statusLabel(code) || 'Notice';
    return h('div', {class: `as-notice ${warnLevel ? 'warn' : 'info'}`, title: code === 'fallback' ? e.message || null : null},
      h('span', {class: 'as-glyph'}, code === 'fallback' ? '⇄' : warnLevel ? '!' : 'i'),
      h('span', {class: 'as-notice-text'}, text),
      code === 'refusal' && e.category && tag(e.category, 'warn'),
      NOTICE_TAG[code] && tag(NOTICE_TAG[code], warnLevel ? 'warn' : ''));
  }

  function errorEl(e, latest) {
    const allowed = latest && e.retryable && retryAllowed();
    const readyAt = isNum(e.retry_after_s) ? Date.parse(e.at) + e.retry_after_s * 1000 : 0;
    const wait = readyAt - Date.now();
    if (latest) clearTimeout(as.retryTimer);
    if (allowed && wait > 0) {
      as.retryTimer = setTimeout(() => {
        const t = latestTurn(as.tx);
        if (t) t.gate = '';  // repaint the countdown
        paint();
      }, Math.min(wait, 1000) + 20);
    }
    return h('div', {class: 'as-error'},
      h('div', {class: 'bad-text'}, h('span', {class: 'as-glyph'}, '✕'), ' ', e.message || 'This turn failed.'),
      h('div', {class: 'as-error-meta'},
        tag(statusLabel(e.kind || 'error'), 'bad'),
        !e.retryable && h('span', {class: 'faint'}, 'Not retryable'),
        e.retryable && !latest && h('span', {class: 'faint'}, 'A later message followed'),
        allowed && h('button', {type: 'button', class: 'btn small', disabled: wait > 0, onclick: () => retryTurn()},
          wait > 0 ? `Retry in ${Math.ceil(wait / 1000)} s` : '↻ Retry'),
        latest && e.retryable && !allowed && h('span', {class: 'faint'}, 'Retry is offered once the assistant is free')));
  }

  /** Retry is only for the latest turn, after it finished with an error, while nothing else runs. */
  function retryAllowed() {
    const t = latestTurn(as.tx);
    return Boolean(t && t.finished && t.finished.status === 'error') && availability().available
      && !turnRunning() && !busyElsewhere() && !as.sending && !closedReason(as.meta);
  }

  function footEl(t, state) {
    if (state === 'cut' && t.n > 0) {
      return h('div', {class: 'as-turn-foot'}, tag('interrupted', 'warn'),
        h('span', {class: 'faint'}, 'No finish was recorded; the bench may have stopped during this turn.'));
    }
    const f = t.finished;
    if (!f) return null;
    const u = t.usage;
    return h('div', {class: 'as-turn-foot'},
      h('span', null, `${money(f.cost_usd)} · ${plural(isNum(f.requests) ? f.requests : 0, 'request')}`),
      f.status && f.status !== 'done' && tag(statusLabel(f.status), FINISH_TONE[f.status] || ''),
      u && u.fallback_served && h('span', {class: 'warn-text'}, `⇄ ${u.served_model} (fallback)`));
  }

  function paintComposer() {
    const box = $('#assistant-text');
    const button = $('#assistant-send');
    const av = availability();
    const closed = closedReason(as.meta);
    const running = turnRunning();
    box.disabled = !av.available || Boolean(closed);
    box.placeholder = !av.known ? 'Checking the assistant…' : !av.available ? 'The assistant is unavailable' : closed
      || (busyElsewhere() ? 'The assistant is answering another conversation' : 'Ask about coverage, costs or designs…');
    if (running) {
      button.textContent = as.stopping ? 'Stopping…' : '■ Stop';
      button.className = 'btn danger';
      button.disabled = as.stopping || !as.cid;
      button.dataset.action = 'stop';
      button.title = 'Stop this turn. A running tool finishes first.';
    } else {
      button.textContent = as.sending ? 'Sending…' : 'Send';
      button.className = 'btn primary';
      button.disabled = !canSend();
      button.dataset.action = 'send';
      button.title = 'Send (Enter). Shift+Enter adds a line.';
    }
    paintAttach(box.disabled);
    paintCount();
    const line = $('#as-statusline');
    const text = running ? (as.tx && as.tx.status) || (as.stopping ? 'Stopping…' : 'Working…')
      : as.linkError ? `✕ ${as.linkError}` : '';
    line.hidden = !text && !as.polling;
    $('#as-status-led').className = running ? 'led warn live' : as.linkError ? 'led bad' : 'led';
    $('#as-status-text').textContent = text;
    $('#as-status-mode').textContent = as.polling ? `polling every ${running ? POLL_BUSY_MS / 1000 : POLL_IDLE_MS / 1000} s` : '';
  }

  function paintAttach(disabled) {
    const shown = shownDesign();
    const open = shown && shown.id && Number.isInteger(shown.version) ? shown : null;
    const same = open && as.attach && as.attach.id === open.id && as.attach.version === open.version;
    if (!changed('assistant:attach', [as.attach, open, disabled])) return;
    const row = $('#as-attach');
    let chip = null;
    if (as.attach) {
      chip = designChip(as.attach);
      chip.addEventListener('click', closeIfFloating);
    }
    row.hidden = !as.attach && !open;
    fill(row,
      as.attach && [h('span', {class: 'as-k'}, 'Attached'), chip,
        h('button', {type: 'button', class: 'as-x', 'aria-label': 'Remove the attached design', title: 'Remove', onclick: () => {
          as.attach = null;
          paint();
        }}, '✕')],
      open && !same && h('button', {type: 'button', class: 'btn ghost small', disabled, onclick: () => {
        as.attach = {id: open.id, version: open.version, title: open.title || null};
        paint();
      }}, `+ Attach open design · ${open.title || open.id} v${open.version}`),
      open && !same && open.edited && h('span', {class: 'faint'}, 'saved version; unsaved edits are not sent'));
  }

  function paintCount() {
    const n = $('#assistant-text').value.length;
    const el = $('#as-count');
    el.hidden = n < MAX_TEXT - 1000;
    el.textContent = `${count(n)} / ${count(MAX_TEXT)}`;
    el.className = n > MAX_TEXT ? 'as-count bad-text' : 'as-count faint';
  }

  /** The header LED: grey while unavailable (the reason in the tooltip), green when ready, amber while a turn runs. */
  function paintButtonLed(state = ui.state) {
    if (!state) return;
    const a = state.assistant && !state.assistant.error ? state.assistant : null;
    // A turn this page sent or is streaming counts at once, and one it saw finish counts as done at once:
    // /api/state only catches up at its next poll. A turn in another conversation is as /api/state says.
    const named = a && a.active_turn;
    const seenDone = Boolean(named && named.conversation_id === as.cid && !turnRunning());
    const busyTurn = Boolean(named && !seenDone) || turnRunning();
    $('#assistant-btn-led').className = busyTurn ? 'led warn live' : a && a.available ? 'led ok' : 'led';
    const said = !a ? (state.assistant && state.assistant.error) || 'Assistant status unknown'
      : busyTurn ? 'The assistant is answering' : a.available ? 'Assistant ready' : a.reason || 'Assistant unavailable';
    const button = $('#open-assistant');
    button.title = said;
    button.setAttribute('aria-label', `Assistant: ${said}`);  // the LED's colour, in words
  }

  // Usage ----------------------------------------------------------------------

  async function loadUsage() {
    const body = $('#as-usage-body');
    fill(body, h('p', {class: 'faint'}, 'Loading…'));
    let data;
    try {
      data = await api('GET', '/api/assistant/usage?days=14');
    } catch (err) {
      fill(body, warn(err.message));
      return;
    }
    const days = (Array.isArray(data.days) ? data.days : []).slice().reverse();
    const top = Math.max(0, ...days.map((d) => (isNum(d.usd) ? d.usd : 0)));
    fill(body,
      h('table', {class: 'grid-t compact as-usage-t'},
        thead('Day', thNum('Requests'), thNum('Spend'), h('th', null, h('span', {class: 'sr-only'}, 'Bar'))),
        h('tbody', null, days.map((d) => h('tr', null,
          h('td', {class: 'mono'}, d.date),
          tdNum(count(d.requests)),
          tdNum(money(d.usd)),
          h('td', {class: 'as-ubar'}, h('span', {style: `width:${top > 0 && isNum(d.usd) ? ((d.usd / top) * 100).toFixed(0) : 0}%`})))))),
      h('p', {class: 'faint'}, `${money(data.total_usd)} over ${plural(days.length, 'day')}`));
  }

  // Setup ----------------------------------------------------------------------

  /** Builds the panel inside #assistant once. The composer is never rebuilt, so typing survives every repaint. */
  function build() {
    fill($('#assistant'),
      h('div', {class: 'panel-head'},
        h('span', {class: 'panel-code'}, 'AI'),
        h('h2', {class: 'panel-title'}, 'Assistant'),
        h('div', {class: 'panel-meta'},
          h('span', {class: 'led', id: 'assistant-led'}),
          h('span', {id: 'assistant-model', class: 'as-model'}),
          h('button', {type: 'button', class: 'btn ghost small', id: 'assistant-close', 'aria-label': 'Close the assistant'}, '✕'))),
      h('div', {class: 'as-bar'},
        h('select', {id: 'as-conv', 'aria-label': 'Conversation'}),
        h('span', {id: 'as-effort'}),
        h('button', {type: 'button', class: 'btn ghost small', id: 'assistant-clear', title: 'Start a new conversation'}, '+ New'),
        h('button', {type: 'button', class: 'btn ghost small', id: 'as-continue', hidden: true, title: 'A new conversation that starts from a recap of this one'}, 'Continue'),
        h('button', {type: 'button', class: 'btn ghost small', id: 'as-archive', hidden: true}, 'Archive')),
      h('div', {class: 'as-spend'},
        h('div', {class: 'as-spend-line', id: 'as-spend-line'}),
        h('div', {class: 'as-spend-gauge', id: 'as-spend-gauge'}),
        h('div', {class: 'as-designs', id: 'as-designs', hidden: true}),
        h('details', {class: 'as-usage', id: 'as-usage'}, h('summary', null, 'Spend by day'), h('div', {id: 'as-usage-body'}))),
      h('div', {class: 'as-banners', id: 'as-banners'}),
      h('div', {class: 'assistant-log', id: 'assistant-log', role: 'log', 'aria-label': 'Conversation', 'aria-live': 'off'}),
      h('div', {class: 'as-composer'},
        h('div', {class: 'as-attach', id: 'as-attach', hidden: true}),
        h('form', {class: 'assistant-input', id: 'assistant-form'},
          h('textarea', {id: 'assistant-text', rows: 1, placeholder: 'Ask about coverage, costs or designs…', 'aria-label': 'Message to the assistant', spellcheck: 'true'}),
          h('button', {type: 'submit', class: 'btn primary', id: 'assistant-send'}, 'Send')),
        h('div', {class: 'as-count faint', id: 'as-count', hidden: true}),
        h('div', {class: 'as-statusline', id: 'as-statusline', hidden: true},
          h('span', {class: 'led', id: 'as-status-led'}),
          h('span', {class: 'as-status-text', id: 'as-status-text'}),
          h('span', {class: 'faint', id: 'as-status-mode'}))),
      h('div', {class: 'sr-only', id: 'assistant-status', role: 'status', 'aria-live': 'polite'}),
      h('div', {class: 'assistant-foot'}, 'The assistant reads bench data and saves draft designs. Only you can queue or start runs.'));
  }

  function setup() {
    build();
    as.built = true;
    const box = $('#assistant-text');
    $('#open-assistant').addEventListener('click', () => (as.open ? closePanel() : openPanel()));
    $('#assistant-close').addEventListener('click', () => {
      closePanel();
      $('#open-assistant').focus();
    });
    $('#assistant-clear').addEventListener('click', () => newConversation());
    $('#as-archive').addEventListener('click', (e) => archive(e.currentTarget));
    $('#as-continue').addEventListener('click', (e) => as.cid && continueFrom(e.currentTarget, as.cid));
    $('#as-conv').addEventListener('change', (e) => {
      const cid = e.currentTarget.value;
      if (cid && cid !== as.cid) showConversation(cid);
    });
    $('#as-usage').addEventListener('toggle', (e) => {
      if (e.currentTarget.open) loadUsage();
    });
    box.addEventListener('input', resizeBox);
    box.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' || e.shiftKey || e.isComposing) return;
      e.preventDefault();
      if (!turnRunning()) send();
    });
    $('#assistant-form').addEventListener('submit', (e) => {
      e.preventDefault();
      if ($('#assistant-send').dataset.action === 'stop') stopTurn();
      else send();
    });
    paintPanelLed();
  }

  /** After every /api/state poll: the header LED, today's spend and the running turn, even while the panel is closed. */
  function onState(state, prev) {
    paintButtonLed(state);
    const a = state.assistant && !state.assistant.error ? state.assistant : null;
    const pa = prev && prev.assistant && !prev.assistant.error ? prev.assistant : null;
    if (a) {
      // While a turn streams, a poll answered before the latest usage event must not lower the figure.
      if (isNum(a.today_usd)) as.today = turnRunning() && isNum(as.today) ? Math.max(as.today, a.today_usd) : a.today_usd;
      as.activeTurn = a.active_turn || null;
      expireSilentTurn();
    }
    if (as.open) {
      if (a && (!pa || pa.available !== a.available || pa.reason !== a.reason)) loadStatus();
      if (a && pa && JSON.stringify(a.active_turn) !== JSON.stringify(pa.active_turn)) loadList();
    }
    paint();
  }

  /** A turn this page saw start that went silent while /api/state says nothing runs: the bench stopped mid-turn. */
  function expireSilentTurn() {
    if (as.activeTurn || !as.tx) return;
    const silentFor = Date.now() - Math.max(as.tx.lastEventAt, as.sentWall);
    if (silentFor < STALE_TURN_MS) return;
    const t = latestTurn(as.tx);
    if (t && t.liveOpen && !t.finished) {
      t.liveOpen = false;
      t.dirty = true;
    }
    if (as.pendingTurn) as.pendingTurn = null;
    as.stopping = false;
  }

  // Panel hooks ------------------------------------------------------------------
  //
  // What app.js and the other areas call, through openAssistant, closeAssistant,
  // assistantOpen and holdReason in app.js.

  registerPanel('assistant', {
    setup,
    onState,
    /** opts: {conversationId} shows it; {design: {id, version}} attaches it to the next message (a first message
     *  starts a conversation with it); {prompt} puts text in the composer without sending. They combine. */
    open: (opts = {}) => openPanel(opts || {}),
    close: closePanel,
    isOpen: () => as.open,
    hold() {
      const box = document.getElementById('assistant-text');
      return box && box.value.trim() ? 'The assistant composer holds a message' : '';
    },
    escape() {
      if (!as.open || !$('#assistant').contains(document.activeElement)) return false;
      closePanel();
      $('#open-assistant').focus();
      return true;
    },
  });
})();
