/* Demo mode for pre-baked harness images: builds, pulls, prunes and the disk
   measurement are simulated in the browser, so nothing changes Docker. Loaded
   after demo.js, in demo mode only (DEMO_SCRIPTS in app.js).

   GET /api/prebake/status comes from the real server, which only reads
   Docker, with every image a simulated build promoted moved to ready. LAUNCH's
   plan (GET /api/plan, through benchDemo.onPlan) and state.prebake follow the
   same simulated images, so the Setup cells, Build missing and IMAGES agree.
   A simulated build runs its items `jobs` at a time, a few seconds each, and
   one item fails its setup check so the failure path shows too. */

(() => {
  'use strict';

  const demo = window.benchDemo;
  if (!demo) return;

  const ITEM_MS = 5000;        // one simulated image
  const PULL_MS = 4000;        // one simulated pull
  const DISK_FACTOR = 1.15;    // as bench/prebake/builder.py
  const BUSY = 'A build is already running (here or from python -m bench.prebake).';
  const S = demo.store.images = {builds: [], built: new Set(), pulled: false, pruned: new Set(), seq: 0, vm: null};

  const pad2 = (n) => String(n).padStart(2, '0');
  const iso = (t) => new Date(t).toISOString().replace(/\.\d+Z$/, '+00:00');
  const stamp = (t) => {
    const d = new Date(t);
    return `${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}`
      + `-${pad2(d.getHours())}${pad2(d.getMinutes())}${pad2(d.getSeconds())}`;
  };
  const key = (family, task) => `${family}|${task}`;
  const clone = (v) => JSON.parse(JSON.stringify(v));
  const text = (body) => new Response(body, {headers: {'Content-Type': 'text/plain'}});

  // Builds ------------------------------------------------------------------------

  /** A simulated build of a status's items, or a pull of its task images. */
  function begin(kind, label, items, jobs) {
    const start = Date.now();
    const build = {
      id: `b${stamp(start)}-d${(++S.seq).toString(16).padStart(3, '0')}`, kind, label, created_at: iso(start), jobs,
      start, cancelledAt: null,
      options: kind === 'build' ? {setup: 'auto', equivalence: 'auto', sample: 'per-os'} : {images: items},
      plan: kind === 'build' ? items.map((it, i) => ({...it, n: i + 1})) : [],
      fail: kind === 'build' && items.length > 2 ? Math.min(items.length, 3) : null,
    };
    S.builds.unshift(build);
    return build;
  }

  /** An item's state at time t: queued, then recording, building and verifying, then promoted (or failed). */
  function itemAt(build, it, t) {
    const slot = Math.floor((it.n - 1) / build.jobs);
    const from = build.start + slot * ITEM_MS;
    const until = from + ITEM_MS;
    const stop = build.cancelledAt;
    const base = {
      n: it.n, family: it.family, version: it.version, recipe: it.recipe, task_image: it.task_image,
      image: it.image, tasks: it.tasks, arms: it.arms, why: it.why, staging: null, image_id: null, size: null,
      noop: false, os_family: null, verify: {}, started_at: t >= from ? iso(from) : null, finished_at: null,
      seconds: null, log: `prebake/logs/${build.id}/${pad2(it.n)}-${it.family}.log`, message: '',
    };
    if (stop && stop < until) {
      return {...base, status: 'cancelled', message: 'Cancelled', finished_at: iso(Math.max(stop, from))};
    }
    if (t < from) return {...base, status: 'queued'};
    if (t < until) {
      const step = ['recording', 'building', 'verifying'][Math.min(2, Math.floor(((t - from) / ITEM_MS) * 3))];
      return {...base, status: step};
    }
    const done = {
      ...base, finished_at: iso(until), seconds: ITEM_MS / 1000, image_id: `sha256:${'d'.repeat(8)}${pad2(it.n)}`,
      os_family: it.task_image.includes('regex-log') ? 'ubuntu-24.04' : 'debian-12',
    };
    if (it.n === build.fail) {
      return {...done, status: 'failed',
        message: 'setup check: agent setup took 72.4 s, over the harness budget; not promoted'};
    }
    return {...done, status: 'promoted', size: 1.9e9, message: 'verified and promoted'};
  }

  /** A build as GET /api/prebake/builds/{id} answers it, now. Promoted images join S.built. */
  function view(build, t = Date.now()) {
    let items = [];
    let status;
    let finished = null;
    if (build.kind === 'pull') {
      const end = build.start + PULL_MS;
      status = build.cancelledAt && build.cancelledAt < end ? 'cancelled' : t >= end ? 'done' : 'running';
      if (status === 'done') S.pulled = true;
      finished = status === 'running' ? null : iso(Math.min(end, build.cancelledAt || end));
    } else {
      items = build.plan.map((it) => itemAt(build, it, t));
      for (const it of items.filter((i) => i.status === 'promoted')) {
        for (const task of it.tasks) S.built.add(key(it.family, task));
      }
      const open = items.some((i) => !['promoted', 'failed', 'cancelled'].includes(i.status));
      const failed = items.some((i) => i.status === 'failed');
      status = open ? 'running' : build.cancelledAt ? 'cancelled' : failed ? 'done_with_errors' : 'done';
      const last = (m, i) => (i.finished_at && i.finished_at > m ? i.finished_at : m);
      finished = open ? null : items.reduce(last, build.created_at);
    }
    const counts = {};
    for (const i of items) counts[i.status] = (counts[i.status] || 0) + 1;
    return {
      id: build.id, kind: build.kind, label: build.label, created_at: build.created_at, jobs: build.jobs,
      options: build.options, items, status, finished_at: finished, counts, pulled: {},
      message: status === 'done_with_errors' ? '1 image was not promoted: see its log' : '',
      disk: build.kind === 'build' ? {ok: true, need_gb: 0, host: {}, vm: {}, reasons: []} : {},
    };
  }

  const summary = (b) => {
    const v = view(b);
    return {id: v.id, kind: v.kind, label: v.label, created_at: v.created_at, finished_at: v.finished_at,
      status: v.status, message: v.message, jobs: v.jobs, items: v.items.length, counts: v.counts};
  };
  const running = () => S.builds.find((b) => view(b).status === 'running') || null;
  const findBuild = (id) => S.builds.find((b) => b.id === id) || null;

  function logOf(build, n) {
    const v = view(build);
    if (n === 0) {
      const lines = v.items.filter((i) => i.started_at)
        .map((i) => `[${i.started_at}] #${i.n} ${i.family} ${i.task_image}: ${i.status}`);
      const head = `[${v.created_at}] demo: simulated ${v.kind}, ${v.jobs} at once; nothing reaches Docker`;
      return [head, ...lines].join('\n');
    }
    const it = v.items.find((i) => i.n === n);
    if (!it) return null;
    const lines = [`[${it.started_at || v.created_at}] demo: recording ${it.family} ${it.version}'s install on `
      + `${it.task_image}`];
    if (['building', 'verifying', 'promoted', 'failed'].includes(it.status)) {
      lines.push('#1 [internal] load build definition from Dockerfile', `#2 FROM docker.io/${it.task_image}`,
        '#3 RUN ["bash","-c","set -o pipefail; apt-get update && apt-get install -y curl ..."]', '#3 DONE 41.2s',
        '#4 exporting to image');
    }
    if (['verifying', 'promoted', 'failed'].includes(it.status)) {
      lines.push('verify: static OK, smoke OK, fingerprint OK');
    }
    if (it.status === 'promoted') lines.push(`promoted to ${it.image}`);
    if (it.status === 'failed') lines.push(`verify: ${it.message}`, 'removed its staging tag');
    return lines.join('\n');
  }

  // The real answers, with the simulated images ready ---------------------------

  const builtTask = (family, task) => S.built.has(key(family, task));

  /** Moves the tasks a simulated build covered to ready in a readiness or plan images block; returns them. */
  function moveBuilt(block, family) {
    const moved = [...block.missing, ...Object.keys(block.stale), ...block.unverifiable]
      .filter((t) => builtTask(family, t));
    block.ready = [...block.ready, ...moved];
    block.missing = block.missing.filter((t) => !moved.includes(t));
    block.unverifiable = block.unverifiable.filter((t) => !moved.includes(t));
    for (const t of moved) delete block.stale[t];
    if (moved.length && !block.oldest_baked_at) block.oldest_baked_at = iso(Date.now());
    return moved;
  }

  /** Longest first onto `jobs` builders, as BuildPlan.wall_minutes. */
  function wallMinutes(items, jobs) {
    const loads = Array(Math.max(1, jobs || 4)).fill(0);
    for (const it of [...items].sort((a, b) => b.expected_min - a.expected_min)) {
      loads[loads.indexOf(Math.min(...loads))] += it.expected_min;
    }
    return Math.round(Math.max(...loads) * 10) / 10;
  }

  /** GET /api/prebake/status with what simulated builds and pulls changed. */
  function overlayStatus(status) {
    const s = clone(status);
    for (const r of Object.values(s.readiness || {})) {
      if (!r.supported) continue;
      moveBuilt(r, r.family);
      r.all_ready = r.ready.length === r.tasks;
    }
    s.items = (s.items || []).filter((it) => !it.tasks.every((t) => builtTask(it.family, t)));
    s.images = s.items.length;
    s.minutes = Math.round(s.items.reduce((m, it) => m + it.expected_min, 0) * 10) / 10;
    s.need_gb = Math.round(s.items.reduce((m, it) => m + it.expected_gb, 0) * DISK_FACTOR * 100) / 100;
    s.wall_minutes = wallMinutes(s.items, s.jobs);
    if (S.pulled) s.pull = [];
    s.builds = [...S.builds.map(summary), ...(s.builds || [])];
    const live = running();
    s.active = live ? live.id : s.active;
    if (live) s.blocked = BUSY;
    if (S.vm && s.disk) Object.assign(s.disk, S.vm);
    return s;
  }

  /** LAUNCH's plan: arms whose every task is built start pre-baked, without the doubled setup limit. */
  function overlayPlan(plan) {
    if (!plan.prebake) return;
    let left = 0;
    let wall = 0;
    let gb = 0;
    for (const a of plan.arms || []) {
      const pb = a.prebake;
      const im = pb && pb.images;
      if (!pb || !pb.supported || !im || im.error) continue;
      const before = im.tasks - im.ready.length;
      const moved = moveBuilt(im, pb.family);
      const rest = im.tasks - im.ready.length;
      im.all_ready = rest === 0;
      left += rest;
      const share = before ? rest / before : 0;
      im.build_min = Math.round(im.build_min * share * 10) / 10;
      im.build_gb = Math.round(im.build_gb * share * 100) / 100;
      wall = Math.max(wall, im.build_min);
      gb += im.build_gb;
      if (moved.length) recount(plan, a, pb, im, rest);
    }
    plan.prebake.images = Math.min(plan.prebake.images, left);
    if (!plan.prebake.images) {
      plan.prebake.build_wall_min = 0;
      plan.prebake.build_gb = 0;
    } else if (S.built.size) {
      plan.prebake.build_wall_min = Math.max(1, Math.round(wall / Math.max(1, plan.prebake.jobs)));
      plan.prebake.build_gb = Math.round(gb * DISK_FACTOR * 100) / 100;
    }
    if (S.pulled) {
      plan.prebake.pull = [];
      plan.warnings = (plan.warnings || []).filter((w) => !/task image\(s\) are not on this machine yet/.test(w));
    }
  }

  /** The server's texts counted the images before the simulated build: say it again with today's counts. */
  function recount(plan, a, pb, im, rest) {
    a.warnings = a.warnings.filter((w) => !/pre-baked image|installs itself/.test(w));
    a.setup_warnings = [];
    const gone = new Set(a.problems.filter((p) => /pre-baked images are not ready/.test(p))
      .map((p) => `${a.name}: ${p}`));
    a.problems = a.problems.filter((p) => !gone.has(`${a.name}: ${p}`));
    plan.problems = plan.problems.filter((p) => !gone.has(p));
    plan.image_problems = (plan.image_problems || []).filter((p) => !gone.has(p));
    if (rest && pb.mode !== 'require') {
      const warning = `${rest} of ${im.tasks} trials will install the harness themselves (stock setup): no ready `
        + 'pre-baked image. Build missing images first to avoid it';
      a.warnings.push(warning);
      a.setup_warnings.push(warning);
    }
    if (rest && pb.mode === 'require') {
      const problem = `run.prebake is require, but ${rest} of ${im.tasks} pre-baked images are not ready. `
        + 'Build them first: Build missing on LAUNCH';
      a.problems.push(problem);
      plan.problems.push(`${a.name}: ${problem}`);
      plan.image_problems.push(`${a.name}: ${problem}`);
    }
    if (!rest) {
      pb.expect_stock = false;
      pb.setup_timeout_multiplier = null;
      a.command = String(a.command || '').replace(/ --agent-setup-timeout-multiplier \S+/, '');
    }
  }

  // Routes --------------------------------------------------------------------------

  async function realStatus(query) {
    const res = await demo.server(`/api/prebake/status?${query}`);
    if (!res.ok) return {error: res};
    return {status: overlayStatus(await res.json())};
  }

  const selectionQuery = (body) => {
    const b = body || {};
    const query = {...(b.mode ? {mode: b.mode} : {}), ...(b.arms ? {arms: [].concat(b.arms).join(',')} : {}),
      ...(b.reuse ? {reuse: b.reuse} : {})};
    return new URLSearchParams(query).toString();
  };

  const blocked = () => {
    if (running()) return BUSY;
    if (demo.runs().some((r) => r.status === 'running')) {
      return 'A run is in progress. Images are not built during a run: a build slows its trials\' containers.';
    }
    return '';
  };

  demo.on('GET', '/api/prebake/status', async ({url}) => {
    const out = await realStatus(url.searchParams.toString());
    return out.error || out.status;
  });

  demo.on('GET', '/api/prebake/builds', async () => {
    const res = await demo.server('/api/prebake/builds');
    const real = res.ok ? await res.json() : {builds: [], active: null};
    const live = running();
    return {builds: [...S.builds.map(summary), ...real.builds], active: live ? live.id : real.active};
  });

  demo.on('GET', /^\/api\/prebake\/builds\/([^/]+)$/, ({match}) => {
    const b = findBuild(decodeURIComponent(match[1]));
    return b ? view(b) : undefined;
  });

  demo.on('GET', /^\/api\/prebake\/builds\/([^/]+)\/log\/(\d+)$/, ({match}) => {
    const b = findBuild(decodeURIComponent(match[1]));
    if (!b) return undefined;
    const log = logOf(b, Number(match[2]));
    return log === null ? demo.error(404, `Not found: ${match[2]}`) : text(log);
  });

  demo.on('POST', /^\/api\/prebake\/builds\/([^/]+)\/cancel$/, ({match}) => {
    const b = findBuild(decodeURIComponent(match[1]));
    if (!b || view(b).status !== 'running') return demo.error(409, 'That build is not in progress');
    b.cancelledAt = Date.now();
    return {ok: true};
  });

  demo.on('POST', '/api/prebake/build', async ({body}) => {
    const why = blocked();
    if (why) return demo.error(409, why);
    const out = await realStatus(selectionQuery(body));
    if (out.error) return out.error;
    const s = out.status;
    if (!s.items.length) return demo.error(409, 'Nothing to build: every image this selection needs is ready');
    const build = begin('build', `${s.label} (demo)`, s.items, (body && body.jobs) || s.jobs || 4);
    return demo.json(view(build), 201);
  });

  demo.on('POST', '/api/prebake/pull', async ({body}) => {
    const why = blocked();
    if (why) return demo.error(409, why);
    const out = await realStatus(selectionQuery(body));
    if (out.error) return out.error;
    if (!out.status.pull.length) {
      return demo.error(409, 'Every task image this selection needs is on this machine already');
    }
    const build = begin('pull', `pull (demo): ${out.status.label}`, out.status.pull, 2);
    return demo.json(view(build), 201);
  });

  demo.on('POST', '/api/prebake/prune', ({body}) => {
    const candidates = [
      {ref: 'benchbake/claude-code-2.1.300-r0f3c9a1b2c/alexgshaw/fix-git:20251031', id: 'sha256:aaa1', kind: 'bake',
        reason: 'built for Harbor 0.24.0; 0.25.0 is installed, so every trial would refuse it',
        baked_at: iso(Date.now() - 21 * 864e5), size: 1.8e9},
      {ref: 'benchbake-staging/b20261028-101500-d001-3/terminus-2-0.25.0-rd1db4d88b3/alexgshaw/regex-log:20251031',
        id: 'sha256:aaa2', kind: 'staging', reason: 'a staging tag left by a build that did not finish', baked_at: null,
        size: 1.6e8},
    ].filter((c) => !S.pruned.has(c.ref));
    if (!body || !Array.isArray(body.refs)) {
      return {dry_run: true, candidates, notes: ['Demo: these images are made up; nothing is removed.']};
    }
    const why = blocked();
    if (why) return demo.error(409, why);
    const removed = candidates.filter((c) => body.refs.includes(c.ref)).map((c) => c.ref);
    for (const ref of removed) S.pruned.add(ref);
    return {dry_run: false, candidates, removed, failed: {}, dangling: 'Total reclaimed space: 0B', notes: [],
      skipped: body.refs.filter((r) => !removed.includes(r))};
  });

  demo.on('POST', '/api/prebake/disk', async () => {
    const res = await demo.server('/api/prebake/status');
    const disk = res.ok ? (await res.json()).disk || {} : {};
    S.vm = {vm_free_gb: 868.9, vm_checked_at: Date.now() / 1000};
    return {...disk, ...S.vm};
  });

  demo.onPlan(overlayPlan);

  // The dashboard's poll: a simulated build is the active one, and its progress moves the version, so LAUNCH plans
  // again.
  demo.onState((state) => {
    if (!S.builds.length || !state.prebake || state.prebake.error) return;
    const live = running();
    const last = summary(S.builds[0]);
    state.prebake = {...state.prebake, active: live ? summary(live) : state.prebake.active, last, elsewhere: false,
      version: `demo:${last.id}:${last.status}:${S.built.size}:${S.pulled ? 'pulled' : ''}`};
  });
})();
