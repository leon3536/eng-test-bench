/* Demo mode for the assistant panel. Loaded after demo.js only with ?demo.

   Owned by the assistant builder, with assistant.js and assistant.css. Brief:
   "Testing" (an available assistant with a scripted turn; fake EventSource, or
   the panel's polling path in demo mode).

   The whole assistant API is answered in the browser from an in-memory store
   (benchDemo.store.assistant), so nothing reaches the real server and no model
   is called: status, conversations (list, create, get, continue, archive),
   messages, stop, retry and usage. window.EventSource is replaced for the
   conversation stream: it sends "batch" events from the same store every
   100 ms, as the server does. Other EventSource URLs get the real one.

   A message plays a scripted turn with thinking and text deltas, two tool
   calls (get_coverage, then save_draft_design with a draft), usage, and a
   final step that a declined model starts and a fallback model answers.
   Messages containing "fail" end in a retryable rate-limit error, "refuse" in
   a refusal, and "lock" in a locked conversation. Set
   benchDemo.store.assistant.breakStream = true to make the stream fail, so
   the panel falls back to polling. */

(() => {
  'use strict';

  if (!window.benchDemo) return;  // demo.js did not load

  const MODEL = 'claude-opus-5-5';
  const FALLBACK_MODEL = 'claude-opus-5';
  const KEY_TEXT = 'This looks like an API key. Set keys on Setup; never paste them into chat.';
  const BUSY = 'The assistant is already answering a message. Wait for it, or press Stop.';
  const STOPPED = Symbol('stopped');
  const DRAFT_CID = 'c-20261029-101500-a1b2';  // the conversation of demo-designs.js's assistant draft

  const pad2 = (n) => String(n).padStart(2, '0');
  const iso = (t = Date.now()) => new Date(t).toISOString().replace(/\.\d+Z$/, '+00:00');
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const round = (v) => Math.round(v * 1e6) / 1e6;
  const hex = (n) => Array.from({length: n}, () => '0123456789abcdef'[Math.floor(Math.random() * 16)]).join('');

  function newCid(t) {
    const d = new Date(t);
    return `c-${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}-${pad2(d.getHours())}${pad2(d.getMinutes())}${pad2(d.getSeconds())}-${hex(4)}`;
  }

  const S = {
    conversations: [],   // {meta, events: [{event, durable}], next, turn}
    todayUsd: 0.37,
    capUsd: 5,
    perTurnCapUsd: 1,
    active: null,        // the running turn: {cid, turn, started_at, usd, requests, stop}
    breakStream: false,
  };
  benchDemo.store.assistant = S;

  // Store -----------------------------------------------------------------------

  const find = (cid) => S.conversations.find((c) => c.meta.id === cid) || null;
  const activeView = () => (S.active ? {conversation_id: S.active.cid, turn: S.active.turn, started_at: S.active.started_at} : null);
  const view = (c) => ({...structuredClone(c.meta), active: Boolean(S.active && S.active.cid === c.meta.id)});
  const since = (c, after) => c.events.filter((x) => x.event.seq > after).map((x) => ({...x.event}));

  function create({id = null, title = null, effort = 'medium', design = null, continuedFrom = null, at = Date.now()} = {}) {
    const c = {
      meta: {
        id: id || newCid(at), title: title || 'New conversation', created_at: iso(at), updated_at: iso(at),
        model: MODEL, effort, fallbacks: 'default', prompt_hash: `sha256:${hex(64)}`, tools_hash: `sha256:${hex(64)}`,
        status: 'idle', locked_reason: null, turns: 0, requests: 0, spend_usd: 0, last_context_tokens: 0, next_seq: 1,
        designs: design ? [design] : [], continued_from: continuedFrom, last_turn: null,
      },
      events: [], next: 1, turn: 0,
    };
    S.conversations.unshift(c);
    S.conversations.sort((a, b) => b.meta.created_at.localeCompare(a.meta.created_at));
    return c;
  }

  function append(c, type, durable, data = {}, at = Date.now()) {
    const event = {seq: c.next++, type, at: iso(at), turn: c.turn, ...data};
    c.events.push({event, durable});
    c.meta.updated_at = iso(at);
    return event;
  }

  /** A new turn drops the previous turn's live-only events, as the server does. */
  function startTurn(c) {
    c.turn = c.meta.turns + 1;
    c.meta.turns = c.turn;
    c.meta.status = 'busy';
    c.events = c.events.filter((x) => x.durable);
  }

  function addUsage(c, T, usd, served = MODEL) {
    T.usd += usd;
    T.requests++;
    S.todayUsd = round(S.todayUsd + usd);
    c.meta.requests++;
    c.meta.spend_usd = round(c.meta.spend_usd + usd);
    c.meta.last_context_tokens += 9000;
    append(c, 'usage', true, {
      request_usd: round(usd), turn_usd: round(T.usd), conversation_usd: c.meta.spend_usd, today_usd: S.todayUsd,
      served_model: served, fallback_served: served !== MODEL, partial: false,
    });
  }

  /** turn_error (and the locked notice, for an error that locks the conversation), then turn_finished. */
  function finish(c, T, status, error = null) {
    const {lock = null, ...fields} = error || {};
    if (error) append(c, 'turn_error', true, {retry_after_s: null, ...fields});
    if (lock) append(c, 'notice', true, {level: 'warn', code: 'locked', message: `${lock} Continue in a new conversation.`});
    append(c, 'turn_finished', true, {status, cost_usd: round(T.usd), requests: T.requests});
    c.meta.status = lock ? 'locked' : 'idle';
    c.meta.locked_reason = lock;
    c.meta.next_seq = c.next;
    c.meta.last_turn = {turn: T.turn, status, error_kind: error ? error.kind : null, retryable: Boolean(error && error.retryable)};
    if (S.active === T) S.active = null;
  }

  // Scripted turns ----------------------------------------------------------------

  async function pause(T, ms) {
    await sleep(ms);
    if (T.stop) throw STOPPED;
  }

  /** Streams text in small pieces as deltas under one block_key. */
  async function stream(c, T, type, key, text, ms = 40) {
    for (const piece of text.match(/\s*\S+/g) || [text]) {
      append(c, type, false, {block_key: key, text: piece});
      await pause(T, ms);
    }
  }

  const status = (c, text) => append(c, 'status', false, {text});

  /**
   * The draft the scripted turn "saves": demo-designs.js's assistant-authored design (store.designs, read only), so
   * Open finds it in DESIGNS. Listing designs through the demo handlers first makes that file seed its designs.
   * Without it, a made-up id that DESIGNS will not find.
   */
  async function draftRef() {
    const store = benchDemo.store.designs;
    try {
      if (store && typeof store.ids === 'function') {
        // Another area's fake must never hold up or break this turn: at most two seconds, then the made-up draft.
        await Promise.race([benchDemo.fetch('/api/designs', {credentials: 'same-origin', cache: 'no-store'}), sleep(2000)]);
        for (const id of store.ids()) {
          const summary = store.summary(id) || {};
          const n = summary.latest_version || 1;
          const v = store.version(id, n);
          if (!v || v.author !== 'assistant') continue;
          const check = v.check || {};
          const total = check.estimate && check.estimate.run ? check.estimate.run.total || {} : {};
          const problems = check.validation && Array.isArray(check.validation.problems) ? check.validation.problems.length : 0;
          return {design_id: id, version: n, title: v.title, spec_hash: v.spec_hash, expected_usd: total.p50 ?? 0.22, p90_usd: total.p90 ?? 0.44, problems};
        }
      }
    } catch { /* the made-up draft below */ }
    return {design_id: 'd-glm-5-3-coverage-gaps-4c1e', version: 1, title: 'GLM 5.3 coverage gaps', spec_hash: `sha256:${hex(64)}`,
      expected_usd: 0.22, p90_usd: 0.44, problems: 0};
  }

  async function fullTurn(c, T) {
    const n = T.turn;
    // Step 1: say what it checks, then get_coverage.
    status(c, 'Thinking…');
    await pause(T, 250);
    const think1 = 'The person wants to know where paired coverage is thin. The main open model in this profile is GLM 5.3 on OpenRouter, so I check coverage there against the claude-code baseline at 10 pairs per family.';
    await stream(c, T, 'thinking_delta', `${n}:1:0`, think1, 30);
    status(c, 'Writing…');
    const text1 = 'I will check paired coverage on GLM 5.3 against Claude Code first.';
    await stream(c, T, 'text_delta', `${n}:1:1`, text1);
    status(c, 'Preparing get_coverage…');
    await pause(T, 300);
    addUsage(c, T, 0.0412);
    status(c, 'Calling get_coverage…');
    await pause(T, 800);
    append(c, 'thinking_summary', true, {block_key: `${n}:1:0`, text: think1});
    append(c, 'assistant_text', true, {block_key: `${n}:1:1`, text: text1});
    const coverageId = `toolu_demo_${hex(8)}`;
    append(c, 'tool_call', true, {
      tool_use_id: coverageId, name: 'get_coverage',
      input: {provider: 'openrouter', model: 'z-ai/glm-5.3', baseline_family: 'claude-code', target_pairs: 10, match: 'family'},
    });
    append(c, 'tool_result', true, {tool_use_id: coverageId, name: 'get_coverage', ok: true, summary: 'Checked coverage · openrouter:z-ai/glm-5.3 · 3 gaps', draft: null});

    // Step 2: save a draft that fills the gaps.
    status(c, 'Thinking…');
    await pause(T, 200);
    const think2 = 'pi, strands and mini-swe-agent are each short of 10 pairs. One design on the six missing tasks, one attempt each, closes all three and reuses the existing baseline trials.';
    await stream(c, T, 'thinking_delta', `${n}:2:0`, think2, 30);
    status(c, 'Writing…');
    const text2 = 'Three families are short. I am saving one draft that fills all three gaps.';
    await stream(c, T, 'text_delta', `${n}:2:1`, text2);
    status(c, 'Preparing save_draft_design…');
    await pause(T, 300);
    addUsage(c, T, 0.0587);
    status(c, 'Calling save_draft_design…');
    await pause(T, 900);
    append(c, 'thinking_summary', true, {block_key: `${n}:2:0`, text: think2});
    append(c, 'assistant_text', true, {block_key: `${n}:2:1`, text: text2});
    const {expected_usd: p50, p90_usd: p90, problems, ...ref} = await draftRef();
    const saveId = `toolu_demo_${hex(8)}`;
    append(c, 'tool_call', true, {
      tool_use_id: saveId, name: 'save_draft_design',
      input: {
        title: ref.title, purpose: 'Reach 10 paired tasks per harness on GLM 5.3.',
        rationale: 'Coverage shows pi, strands and mini-swe-agent short by 2 to 4 pairs on the seeded sample; the baseline already has those tasks.',
        change_note: 'First version.',
        spec: {
          arms: [
            {name: 'pi-glm', agent: 'pi', provider: 'openrouter', model: 'z-ai/glm-5.3'},
            {name: 'strands-glm', agent: 'strands', provider: 'openrouter', model: 'z-ai/glm-5.3'},
            {name: 'mswea-glm', agent: 'bench.agents.mini_swe_agent:MiniSweAgentWithMcp', provider: 'openrouter', model: 'z-ai/glm-5.3'},
          ],
          tasks: ['fix-git', 'regex-log', 'sqlite-with-gcov', 'log-summary', 'csv-to-parquet', 'build-pmars'],
          run: {attempts: 1, budget_usd: 3},
        },
      },
    });
    append(c, 'tool_result', true, {
      tool_use_id: saveId, name: 'save_draft_design', ok: true, summary: `Saved draft ${ref.title} v${ref.version}`,
      draft: {...ref, expected_usd: p50, p90_usd: p90, problems},
    });
    if (!c.meta.designs.some((d) => d.id === ref.design_id && d.version === ref.version)) {
      c.meta.designs.push({id: ref.design_id, version: ref.version});
    }

    // Step 3: the requested model starts, declines, and the fallback model answers. The declined text never
    // gets a durable twin, so the panel folds it away when the turn finishes.
    status(c, 'Writing…');
    await stream(c, T, 'text_delta', `${n}:3:0`, 'Here is the draft and the reasoning behind each');
    append(c, 'notice', true, {
      level: 'info', code: 'fallback', from_model: MODEL, to_model: FALLBACK_MODEL,
      message: `${MODEL} declined this request (uncertain); ${FALLBACK_MODEL} answered instead.`,
    });
    status(c, 'Writing…');
    const text3 = [
      `**Draft saved: ${ref.title} v${ref.version}.** It is waiting in the Designs view; only you can queue or start it.`,
      '',
      '| Family | Pairs now | After | Trials |',
      '|---|---|---|---|',
      '| pi | 7 | 10 | 3 |',
      '| strands | 6 | 10 | 4 |',
      '| mini-swe-agent | 8 | 10 | 2 |',
      '',
      `- **Cost:** $${p50.toFixed(2)} p50, $${p90.toFixed(2)} p90.`,
      '- **Wall time:** about 25 minutes with two arms in parallel.',
      '- **Caveat:** strands failed 2 of 6 GLM trials on setup last time; if it fails again the queue pauses.',
      '',
      'Demo reply, scripted in the browser. No model was called.',
    ].join('\n');
    await stream(c, T, 'text_delta', `${n}:3:2`, text3, 18);
    addUsage(c, T, 0.0371, FALLBACK_MODEL);
    append(c, 'assistant_text', true, {block_key: `${n}:3:2`, text: text3});
    finish(c, T, 'done');
  }

  async function errorTurn(c, T) {
    status(c, 'Thinking…');
    await pause(T, 300);
    await stream(c, T, 'thinking_delta', `${T.turn}:1:0`, 'Checking the failed trials first, then their logs', 40);
    finish(c, T, 'error', {
      kind: 'rate_limited', retryable: true, retry_after_s: 4,
      message: 'Anthropic is rate limiting this key (429). Nothing was run. Retry in a few seconds.',
    });
  }

  async function refusedTurn(c, T) {
    status(c, 'Writing…');
    await pause(T, 300);
    await stream(c, T, 'text_delta', `${T.turn}:1:0`, 'Sure, the quickest way to get around the cap is to');
    addUsage(c, T, 0.0123);
    append(c, 'notice', true, {
      level: 'warn', code: 'refusal', category: 'policy',
      message: 'Claude declined this request (category: policy). Nothing was run or added to the conversation; rephrase and try again.',
    });
    finish(c, T, 'refused');
  }

  async function lockedTurn(c, T) {
    status(c, 'Writing…');
    await pause(T, 300);
    const text = 'This conversation is close to the context limit, so I will stop here.';
    await stream(c, T, 'text_delta', `${T.turn}:1:0`, text);
    addUsage(c, T, 0.0458);
    append(c, 'assistant_text', true, {block_key: `${T.turn}:1:0`, text});
    finish(c, T, 'error', {
      kind: 'bad_request', retryable: false, lock: 'This conversation grew too long for the model.',
      message: 'The request was too large for the model\'s context window.',
    });
  }

  function play(c, kind, retry) {
    const T = {cid: c.meta.id, turn: c.turn, started_at: iso(), usd: 0, requests: 0, stop: false};
    S.active = T;
    const script = kind === 'error' && !retry ? errorTurn : kind === 'refuse' ? refusedTurn : kind === 'lock' ? lockedTurn : fullTurn;
    setTimeout(async () => {
      try {
        await script(c, T);
      } catch (err) {
        if (err !== STOPPED) throw err;
        append(c, 'notice', true, {level: 'info', code: 'stopped', message: 'Stopped. Anything already run is kept.'});
        finish(c, T, 'stopped');
      }
    }, 120);
    return T;
  }

  const scriptFor = (text) => (/\bfail/i.test(text) ? 'error' : /\brefuse/i.test(text) ? 'refuse' : /\block/i.test(text) ? 'lock' : 'full');

  // Seed: one finished conversation, one locked, one archived -----------------------

  function seed() {
    const hour = 3600e3;
    const old = create({title: 'Pricing Sonnet against Opus', effort: 'low', at: Date.now() - 50 * hour});
    old.turn = 1;
    old.meta.turns = 1;
    append(old, 'user_message', true, {text: 'Is Sonnet much cheaper than Opus for our runs?', attach_design: null}, Date.now() - 50 * hour);
    append(old, 'turn_started', true, {}, Date.now() - 50 * hour);
    append(old, 'assistant_text', true, {block_key: '1:1:0', text: 'Sonnet 5.5 costs half of Opus 5.5 per input and output token; cache reads cost the same.'});
    append(old, 'usage', true, {request_usd: 0.021, turn_usd: 0.021, conversation_usd: 0.021, today_usd: 0.021, served_model: MODEL, fallback_served: false, partial: false});
    append(old, 'turn_finished', true, {status: 'done', cost_usd: 0.021, requests: 1});
    Object.assign(old.meta, {status: 'archived', requests: 1, spend_usd: 0.021, next_seq: old.next, last_turn: {turn: 1, status: 'done', error_kind: null, retryable: false}});

    const locked = create({title: 'Pilot 3: why did pi-glm time out?', effort: 'high', at: Date.now() - 26 * hour});
    locked.turn = 1;
    locked.meta.turns = 1;
    append(locked, 'user_message', true, {text: 'Why did pi-glm time out in pilot 3?', attach_design: null});
    append(locked, 'turn_started', true, {});
    append(locked, 'thinking_summary', true, {block_key: '1:1:0', text: 'Read the cost stats for pi on GLM, then explain the timeout.'});
    const statsId = `toolu_demo_${hex(8)}`;
    append(locked, 'assistant_text', true, {block_key: '1:1:1', text: 'Reading the cost stats for pi on GLM 5.3.'});
    append(locked, 'tool_call', true, {tool_use_id: statsId, name: 'get_cost_stats', input: {family: 'pi', provider: 'openrouter', model: 'z-ai/glm-5.3', match: 'family', per_task: true}});
    append(locked, 'tool_result', true, {tool_use_id: statsId, name: 'get_cost_stats', ok: true, summary: 'Read cost stats · 1 families', draft: null});
    append(locked, 'assistant_text', true, {block_key: '1:2:0', text: 'The trial ran 900 s without one completed model call, so **no usage was recorded**. It counts for pass rate, but not for cost.'});
    append(locked, 'usage', true, {request_usd: 0.083, turn_usd: 0.083, conversation_usd: 0.083, today_usd: 0.104, served_model: MODEL, fallback_served: false, partial: false});
    append(locked, 'turn_finished', true, {status: 'done', cost_usd: 0.083, requests: 2});
    Object.assign(locked.meta, {
      status: 'locked', locked_reason: "The assistant's tools changed since this conversation started.",
      requests: 2, spend_usd: 0.083, next_seq: locked.next, last_turn: {turn: 1, status: 'done', error_kind: null, retryable: false},
    });

    const recent = create({title: 'Where are the coverage gaps?', effort: 'medium', at: Date.now() - 2 * hour});
    recent.turn = 1;
    recent.meta.turns = 1;
    append(recent, 'user_message', true, {text: 'Where are the coverage gaps?', attach_design: null});
    append(recent, 'turn_started', true, {});
    // The server keeps the last turn's live events until the next turn starts, so a reload sees both.
    append(recent, 'status', false, {text: 'Writing…'});
    append(recent, 'text_delta', false, {block_key: '1:1:0', text: 'Checking coverage on'});
    const covId = `toolu_demo_${hex(8)}`;
    append(recent, 'assistant_text', true, {block_key: '1:1:0', text: 'Checking coverage on DeepSeek V4 first.'});
    append(recent, 'tool_call', true, {tool_use_id: covId, name: 'get_coverage', input: {provider: 'fireworks', model: 'deepseek-v4', baseline_family: 'claude-code', target_pairs: 10, match: 'family'}});
    append(recent, 'tool_result', true, {tool_use_id: covId, name: 'get_coverage', ok: true, summary: 'Checked coverage · fireworks:deepseek-v4 · 5 gaps', draft: null});
    append(recent, 'assistant_text', true, {block_key: '1:2:0', text: [
      'On DeepSeek V4 every harness is short of 10 pairs:',
      '',
      '| Family | Pairs | Missing |',
      '|---|---|---|',
      '| pi | 4 | 6 |',
      '| strands | 3 | 7 |',
      '| terminus-2 | 5 | 5 |',
      '',
      'Ask me to price a design that fills them.',
    ].join('\n')});
    append(recent, 'usage', true, {request_usd: 0.052, turn_usd: 0.094, conversation_usd: 0.094, today_usd: 0.27, served_model: MODEL, fallback_served: false, partial: false});
    append(recent, 'turn_finished', true, {status: 'done', cost_usd: 0.094, requests: 2});
    Object.assign(recent.meta, {status: 'idle', requests: 2, spend_usd: 0.094, next_seq: recent.next, last_turn: {turn: 1, status: 'done', error_kind: null, retryable: false}});

    // The conversation that drafted DESIGNS' demo draft "Fill the half-pairs (demo draft)": demo-designs.js names it
    // as that version's conversation_id. Its draft card is filled in when the conversation is first read (drafted()).
    const at = Date.now() - 3 * hour - 90e3;
    const draft = create({id: DRAFT_CID, title: 'Fill the half-pairs', effort: 'medium', at});
    draft.turn = 1;
    draft.meta.turns = 1;
    append(draft, 'user_message', true, {text: 'Draft a design that completes the half-measured pairs.', attach_design: null}, at);
    append(draft, 'turn_started', true, {}, at);
    append(draft, 'assistant_text', true, {block_key: '1:1:0', text: 'Three seeded tasks have a baseline trial but no partner trial on the same model. Saving a draft that pairs them.'}, at);
    const saveId = `toolu_demo_${hex(8)}`;
    append(draft, 'tool_call', true, {tool_use_id: saveId, name: 'save_draft_design', input: {title: 'Fill the half-pairs (demo draft)'}}, at);
    append(draft, 'tool_result', true, {tool_use_id: saveId, name: 'save_draft_design', ok: true, summary: 'Saved draft Fill the half-pairs (demo draft) v1', draft: null}, at);
    append(draft, 'assistant_text', true, {block_key: '1:2:0', text: 'Saved it as a draft. Open it in DESIGNS to check it, then add it to the queue if it looks right.'}, at);
    append(draft, 'usage', true, {request_usd: 0.031, turn_usd: 0.058, conversation_usd: 0.058, today_usd: 0.162, served_model: MODEL, fallback_served: false, partial: false}, at);
    append(draft, 'turn_finished', true, {status: 'done', cost_usd: 0.058, requests: 2}, at);
    Object.assign(draft.meta, {requests: 2, spend_usd: 0.058, next_seq: draft.next, last_turn: {turn: 1, status: 'done', error_kind: null, retryable: false}});
  }
  seed();

  /** Fills the seeded draft card of DRAFT_CID from the DESIGNS demo's assistant draft, once that file has seeded it. */
  let draftFilled = null;
  function drafted(c) {
    if (c.meta.id !== DRAFT_CID) return Promise.resolve();
    if (!draftFilled) {
      draftFilled = draftRef().then((ref) => {
        const result = c.events.find((x) => x.event.type === 'tool_result' && x.event.name === 'save_draft_design');
        if (result && !/^d-glm-/.test(ref.design_id)) {
          result.event.draft = {...ref};
          if (!c.meta.designs.length) c.meta.designs = [{id: ref.design_id, version: ref.version}];
        }
      });
    }
    return draftFilled;
  }

  // Routes ---------------------------------------------------------------------------

  const CONV = /^\/api\/assistant\/conversations\/([^/]+)$/;
  const ACTION = /^\/api\/assistant\/conversations\/([^/]+)\/(messages|stop|retry|continue|archive)$/;
  const looksLikeKey = (text) => /\bsk-[A-Za-z0-9_-]{16,}|\b[A-Za-z0-9_-]{40,}\b/.test(text);

  function statusView() {
    const capped = S.todayUsd >= S.capUsd;
    return {
      enabled: true, available: !capped,
      reason: capped ? `Assistant daily cap $${S.capUsd.toFixed(2)} reached; raise assistant.daily_cap_usd in config or wait until tomorrow.` : null,
      model: MODEL, effort_default: 'medium', fallbacks: 'default', key_set: true,
      today_usd: S.todayUsd, daily_cap_usd: S.capUsd,
      spend: {today_usd: S.todayUsd, daily_cap_usd: S.capUsd, per_turn_cap_usd: S.perTurnCapUsd},
      active_turn: activeView(),
    };
  }

  benchDemo.onState((state) => {
    const s = statusView();
    state.assistant = {enabled: s.enabled, available: s.available, reason: s.reason, today_usd: s.today_usd, daily_cap_usd: s.daily_cap_usd, active_turn: s.active_turn};
  });

  benchDemo.on('GET', '/api/assistant/status', () => statusView());
  benchDemo.on('GET', '/api/assistant/conversations', () => ({conversations: S.conversations.map(view)}));

  benchDemo.on('POST', '/api/assistant/conversations', ({body}) => {
    const b = body && typeof body === 'object' ? body : {};
    const effort = b.effort || 'medium';
    if (!['low', 'medium', 'high', 'xhigh', 'max'].includes(effort)) return benchDemo.error(400, 'Effort must be one of low, medium, high, xhigh, max');
    const design = b.design && b.design.id ? {id: b.design.id, version: b.design.version} : null;
    return benchDemo.json(view(create({title: typeof b.title === 'string' ? b.title.trim().slice(0, 80) : null, effort, design})), 201);
  });

  benchDemo.on('GET', CONV, async ({match, query}) => {
    const c = find(decodeURIComponent(match[1]));
    if (!c) return benchDemo.error(404, `No conversation '${match[1]}'`);
    await drafted(c);
    return {conversation: view(c), events: since(c, Number(query.get('after')) || 0), next_seq: c.next, active_turn: activeView()};
  });

  benchDemo.on('POST', ACTION, ({match, body}) => {
    const c = find(decodeURIComponent(match[1]));
    if (!c) return benchDemo.error(404, `No conversation '${match[1]}'`);
    const action = match[2];
    const closed = () => (c.meta.status === 'archived' ? 'This conversation is archived.'
      : c.meta.status === 'locked' ? `${c.meta.locked_reason} Continue in a new conversation.` : '');

    if (action === 'stop') {
      if (S.active && S.active.cid === c.meta.id) S.active.stop = true;
      return {ok: true};
    }
    if (action === 'archive') {
      if (S.active && S.active.cid === c.meta.id) return benchDemo.error(409, 'Stop the running turn before archiving this conversation.');
      c.meta.status = 'archived';
      c.meta.updated_at = iso();
      return {ok: true};
    }
    if (action === 'continue') {
      const said = c.events.filter((x) => x.event.type === 'user_message' || x.event.type === 'assistant_text')
        .map((x) => `${x.event.type === 'user_message' ? 'You' : 'Assistant'}: ${x.event.text}`).slice(-12);
      const recap = [`This conversation continues "${c.meta.title}" (${c.meta.id}). A plain-text recap of where it ended follows; tool results and reasoning are not included, so call the tools again for any figure you need.`,
        '', ...(said.length ? said : ['(The earlier conversation had no messages.)'])].join('\n');
      const next = create({title: `${c.meta.title} (continued)`, effort: c.meta.effort, continuedFrom: c.meta.id});
      next.meta.designs = structuredClone(c.meta.designs);
      append(next, 'user_message', true, {text: recap, attach_design: null, recap: true});
      return benchDemo.json(view(next), 201);
    }

    // messages and retry: what the server would bill.
    const why = closed();
    if (why) return benchDemo.error(409, why);
    if (S.active) return benchDemo.error(409, BUSY);
    if (S.todayUsd >= S.capUsd) return benchDemo.error(400, `Assistant daily cap $${S.capUsd.toFixed(2)} reached: $${S.todayUsd.toFixed(2)} spent today.`);
    if (action === 'retry') {
      if (!c.meta.last_turn || c.meta.last_turn.status !== 'error') return benchDemo.error(409, 'Retry is only possible right after a turn that ended in an error.');
      startTurn(c);
      const started = append(c, 'turn_started', true, {retry: true});
      play(c, 'full', true);
      return benchDemo.json({turn: c.turn, accepted_seq: started.seq}, 202);
    }
    const text = body && typeof body.text === 'string' ? body.text : '';
    if (!text.trim()) return benchDemo.error(400, 'Type a message first.');
    if (text.length > 8000) return benchDemo.error(400, 'Messages are limited to 8000 characters.');
    if (looksLikeKey(text)) return benchDemo.error(400, KEY_TEXT);
    const attach = body.attach_design && body.attach_design.id ? {id: body.attach_design.id, version: body.attach_design.version} : null;
    if (c.meta.turns === 0 && c.meta.title === 'New conversation') c.meta.title = text.trim().split('\n')[0].slice(0, 60);
    if (attach && !c.meta.designs.some((d) => d.id === attach.id && d.version === attach.version)) c.meta.designs.push(attach);
    startTurn(c);
    const accepted = append(c, 'user_message', true, {text, attach_design: attach});
    append(c, 'turn_started', true, {});
    play(c, scriptFor(text), false);
    return benchDemo.json({turn: c.turn, accepted_seq: accepted.seq}, 202);
  });

  benchDemo.on('GET', '/api/assistant/usage', ({query}) => {
    const n = Math.max(1, Math.min(366, Number(query.get('days')) || 30));
    const days = [];
    for (let i = n - 1; i >= 0; i--) {
      const d = new Date(Date.now() - i * 864e5);
      const usd = i === 0 ? S.todayUsd : [0, 0.42, 1.18, 0, 0.09, 2.31, 0.66][i % 7];
      days.push({date: `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`, usd: round(usd), requests: usd ? Math.max(1, Math.round(usd * 14)) : 0});
    }
    return {days, total_usd: round(days.reduce((t, d) => t + d.usd, 0))};
  });

  // Anything else under /api/assistant stays in the browser too.
  benchDemo.on('*', /^\/api\/assistant\//, ({method, path}) => benchDemo.error(404, `Demo mode: no fake for ${method} ${path}`));

  // The stream ---------------------------------------------------------------------

  const RealEventSource = window.EventSource;
  const STREAM = /^\/api\/assistant\/conversations\/([^/]+)\/stream$/;

  /** The conversation stream from the store: a "batch" event every 100 ms when there are new events. */
  class DemoEventSource extends EventTarget {
    constructor(url, init) {
      const u = new URL(String(url), location.href);
      const m = u.pathname.match(STREAM);
      if (!m) return new RealEventSource(url, init);  // not the assistant's stream: the real thing
      super();
      this.url = u.href;
      this.withCredentials = false;
      this.readyState = DemoEventSource.CONNECTING;
      this.onopen = null;
      this.onmessage = null;
      this.onerror = null;
      this.cid = decodeURIComponent(m[1]);
      this.last = Number(u.searchParams.get('after')) || 0;
      this.timer = setTimeout(() => this.connect(), 40);
    }

    emit(event) {
      this.dispatchEvent(event);
      const handler = this[`on${event.type}`];
      if (typeof handler === 'function') handler.call(this, event);
    }

    connect() {
      if (this.readyState === DemoEventSource.CLOSED) return;
      if (!find(this.cid)) {  // the server answers 404, and the browser gives up
        this.readyState = DemoEventSource.CLOSED;
        this.emit(new Event('error'));
        return;
      }
      if (S.breakStream) {  // keeps failing; the browser keeps retrying
        this.readyState = DemoEventSource.CONNECTING;
        this.emit(new Event('error'));
        this.timer = setTimeout(() => this.connect(), 300);
        return;
      }
      this.readyState = DemoEventSource.OPEN;
      this.emit(new Event('open'));
      this.timer = setInterval(() => this.tick(), 100);
    }

    tick() {
      const c = find(this.cid);
      if (!c || S.breakStream) {
        clearInterval(this.timer);
        this.connect();
        return;
      }
      const events = since(c, this.last);
      if (!events.length) return;
      this.last = events[events.length - 1].seq;
      this.emit(new MessageEvent('batch', {data: JSON.stringify(events), lastEventId: String(this.last)}));
    }

    close() {
      this.readyState = DemoEventSource.CLOSED;
      clearTimeout(this.timer);
      clearInterval(this.timer);
    }
  }
  DemoEventSource.CONNECTING = 0;
  DemoEventSource.OPEN = 1;
  DemoEventSource.CLOSED = 2;
  window.EventSource = DemoEventSource;
})();
