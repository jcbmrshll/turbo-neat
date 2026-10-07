// turbo-neat monitor: run list, live charts, episode and network viewers, and a
// leaderboard of the population with each member's lineage.
// No framework; every view is plain DOM built with h(), and labels that come
// from a run (metric names, config values) only ever go in as text.

const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"].map((v) => `var(${v})`);
const POLL_MS = 2000;
const RUNS_POLL_MS = 5000;
// a running run that hasn't logged for this long is probably dead
const STALE_S = 180;
// the stat tiles, in order, when the run logs them
const HEADLINE = [
  "max_fitness",
  "mean_fitness",
  "fitness_against_baseline",
  "challenger_fitness",
  "num_species",
  "mean_condensed_hidden_nodes",
  "mean_condensed_connections",
];
// charts that lead the grid; the rest follow in the order they were first logged
const CHART_ORDER = ["fitness", "fitness_against_baseline", "challenger_fitness"];
const BOARD_LIMIT = 50;
// how many generations of parents the pedigree starts with, and steps by
const PEDIGREE_DEPTH = 6;
const PEDIGREE_STEP = 3;

const state = {
  runs: [],
  runId: null,
  run: null,
  rows: [],
  metricsNext: 0,
  media: [],
  mediaNext: 0,
  // media key -> index into that key's entries, or null to follow the latest
  mediaSel: {},
  connected: true,
  // the leaderboard panel
  board: null,
  boardScope: "alive",
  boardSort: "fitness",
  speciesNames: {},
  // the open member tabs, in order (see openTab), and the one shown: a genome id,
  // or null for the run's own tab
  tabs: [],
  activeTab: null,
  // the run tab's charts are drawn to its width, so they wait while it's hidden
  overviewStale: false,
  // step -> that generation's champion ({id, name, birth_species})
  champions: new Map(),
};

// ---------------------------------------------------------------- helpers

function h(tag, attrs = {}, ...children) {
  const svg = ["svg", "path", "line", "circle", "text", "g", "rect"].includes(tag);
  const el = svg
    ? document.createElementNS("http://www.w3.org/2000/svg", tag)
    : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return res.json();
}

function fmt(v) {
  if (v == null || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e4) return Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(v);
  if (Number.isInteger(v)) return v.toLocaleString("en");
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  if (a === 0) return "0";
  return v.toPrecision(3);
}

// axis ticks are already round numbers; just drop float noise and trailing zeros
function fmtTick(v) {
  return Math.abs(v) >= 1e4 ? fmt(v) : String(+v.toPrecision(6));
}

function ago(t) {
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

function duration(s) {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
}

function started(t) {
  return new Date(t * 1000).toLocaleString("en", {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
  });
}

// "41 / 300 gens" (generations done) when the run said how long it will be
function progress(run) {
  if (run.step == null) return "";
  const total = run.num_generations;
  return total ? `${run.step + 1} / ${total} gens` : `gen ${run.step}`;
}

function status(run) {
  if (run.status === "running" && Date.now() / 1000 - run.updated > STALE_S) return "stale";
  return run.status;
}

const tip = document.getElementById("tip");
function showTip(x, y, ...children) {
  tip.replaceChildren(...children);
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  const left = x + 14 + r.width > innerWidth ? x - 14 - r.width : x + 14;
  const top = Math.min(innerHeight - r.height - 4, Math.max(4, y - r.height / 2));
  tip.style.left = `${left}px`;
  tip.style.top = `${top}px`;
}
function hideTip() {
  tip.hidden = true;
}

// ---------------------------------------------------------------- routing

function route() {
  const m = location.pathname.match(/^\/run\/([\w-]+)/);
  return m ? m[1] : null;
}

// the member whose tab is shown: /run/<run id>/genome/<genome id>
function routeGenome() {
  const m = location.pathname.match(/^\/run\/[\w-]+\/genome\/(\d+)/);
  return m ? Number(m[1]) : null;
}

function go(runId) {
  history.pushState(null, "", runId ? `/run/${runId}` : "/");
  select(runId);
}

window.addEventListener("popstate", () => {
  const runId = route();
  if (runId && runId === state.runId) openTab(routeGenome(), { push: false });
  else select(runId);
});

// ---------------------------------------------------------------- sidebar

function renderSide() {
  const side = document.getElementById("side");
  const byProject = new Map();
  for (const run of state.runs) {
    if (!byProject.has(run.project)) byProject.set(run.project, []);
    byProject.get(run.project).push(run);
  }
  const children = [];
  for (const [project, runs] of byProject) {
    children.push(h("div", { class: "side-hdr", text: project }));
    for (const run of runs) {
      children.push(
        h(
          "a",
          {
            class: `run-row${run.id === state.runId ? " active" : ""}`,
            href: `/run/${run.id}`,
            onclick: (e) => {
              if (e.metaKey || e.ctrlKey) return;
              e.preventDefault();
              go(run.id);
            },
          },
          h("span", { class: `dot ${status(run)}`, title: status(run) }),
          h("span", { class: "name", text: run.name }),
          h("span", { class: "step", text: progress(run) }),
          h("span", { class: "when", text: `${started(run.created)} · ${ago(run.updated)}` }),
        ),
      );
    }
  }
  if (!children.length) children.push(h("div", { class: "side-hdr", text: "no runs" }));
  side.replaceChildren(...children);
}

async function pollRuns() {
  try {
    state.runs = await api("/api/runs");
    setConnected(true);
  } catch {
    setConnected(false);
  }
  renderSide();
  if (!state.runId && !route()) renderEmpty();
}

function setConnected(ok) {
  state.connected = ok;
  const el = document.getElementById("conn");
  el.textContent = ok ? "" : "server unreachable";
  el.classList.toggle("down", !ok);
}

// ---------------------------------------------------------------- run view

const main = document.getElementById("main");
let view = null; // the containers of the current run view

function renderEmpty() {
  if (state.runs.length) {
    // nothing selected: open the newest run
    go(state.runs[0].id);
    return;
  }
  view = null;
  document.body.classList.remove("has-board");
  main.replaceChildren(
    h(
      "div",
      { class: "empty" },
      h("h1", { text: "No runs yet" }),
      h("p", { text: "Start a training run with the monitor attached:" }),
      h("p", {}, h("code", { text: "uv run examples/boids.py --monitor" })),
      h("p", { text: "It will show up here as soon as it logs its first generation." }),
    ),
  );
}

async function select(runId) {
  if (!runId) {
    state.runId = null;
    renderSide();
    renderEmpty();
    return;
  }
  Object.assign(state, {
    runId,
    run: null,
    rows: [],
    metricsNext: 0,
    media: [],
    mediaNext: 0,
    mediaSel: {},
    board: null,
    speciesNames: {},
    tabs: [],
    activeTab: null,
    overviewStale: false,
    overviewScroll: 0,
    champions: new Map(),
  });
  renderSide();
  document.body.classList.remove("has-board");
  document.getElementById("board").replaceChildren();
  view = {
    head: h("div", { class: "runhead" }),
    tabbar: h("nav", { class: "tabs" }),
    overview: h("div", { class: "tab-panel" }),
    panels: h("div"),
    tiles: h("div", { class: "tiles" }),
    mediaTitle: h("div", { class: "stitle", text: "Champion" }),
    media: h("div", { class: "media-grid" }),
    chartsTitle: h("div", { class: "stitle", text: "Metrics" }),
    charts: h("div", { class: "chart-grid" }),
    configTitle: h("div", { class: "stitle", text: "Config" }),
    config: h("div", { class: "config-grid" }),
    mediaCards: {},
  };
  view.overview.append(
    view.tiles,
    view.mediaTitle,
    view.media,
    view.chartsTitle,
    view.charts,
    view.configTitle,
    view.config,
  );
  main.replaceChildren(h("div", { class: "page" }, view.head, view.tabbar, view.overview, view.panels));
  main.scrollTop = 0;
  // the member tabs this run had open, then the one in the address
  for (const tab of savedTabs(runId)) addTab(tab.id, tab.name);
  openTab(routeGenome(), { push: false });
  await pollRun();
}

let polling = false;
async function pollRun() {
  const runId = state.runId;
  if (!runId || polling) return;
  polling = true;
  try {
    const [run, metrics, media] = await Promise.all([
      api(`/api/runs/${runId}`),
      api(`/api/runs/${runId}/metrics?since=${state.metricsNext}`),
      api(`/api/runs/${runId}/media?since=${state.mediaNext}`),
    ]);
    if (runId !== state.runId) return;
    setConnected(true);
    const firstLoad = !state.run;
    state.run = run;
    state.metricsNext = metrics.next;
    state.mediaNext = media.next;
    state.rows.push(...metrics.rows);
    state.media.push(...media.rows);

    renderHead();
    if (firstLoad) renderConfig();
    if (firstLoad || metrics.rows.length) {
      renderTiles();
      if (state.activeTab == null) renderCharts();
      else state.overviewStale = true;
    }
    if (firstLoad || media.rows.length) {
      if (state.activeTab == null) renderMedia();
      else state.overviewStale = true;
    }
    // members are logged before the metrics of their generation, so a new
    // generation of metrics means the leaderboard has one too
    if (firstLoad || metrics.rows.length) {
      pollBoard();
      pollChampions();
      const tab = activeTab();
      if (tab) pollMember(tab);
      // the rest catch up when they're shown
      for (const t of state.tabs) if (t !== tab) t.stale = true;
    }
  } catch (e) {
    if (String(e.message).startsWith("404")) {
      main.replaceChildren(h("div", { class: "empty" }, h("h1", { text: "No such run" })));
      document.body.classList.remove("has-board");
      state.runId = null;
    } else {
      setConnected(false);
    }
  } finally {
    polling = false;
  }
}

function renderHead() {
  const run = state.run;
  const st = status(run);
  const end = run.status === "running" ? Date.now() / 1000 : run.updated;
  const sep = () => h("span", { class: "sep", text: "·" });
  view.head.replaceChildren(
    h("div", { class: "crumbs" }, run.project, h("span", { class: "id", text: run.id })),
    h("h1", { text: run.name }),
    h(
      "div",
      { class: "meta" },
      h("span", { class: `dot ${st}` }),
      h("span", { text: st }),
      sep(),
      run.seed != null && h("span", { text: `seed ${run.seed}` }),
      run.seed != null && sep(),
      h("span", { text: `started ${started(run.created)}` }),
      sep(),
      h("span", { text: duration(end - run.created) }),
      run.step != null && sep(),
      run.step != null && h("span", { text: progress(run) }),
    ),
  );
}

function latest(key) {
  for (let i = state.rows.length - 1; i >= 0; i--) {
    const v = state.rows[i][key];
    if (v != null) return v;
  }
  return null;
}

function renderTiles() {
  const tiles = HEADLINE.filter((k) => latest(k) != null).map((k) =>
    h(
      "div",
      { class: "tile" },
      h("div", { class: "lab", text: k.replaceAll("_", " ") }),
      h("div", { class: "val", text: fmt(latest(k)) }),
    ),
  );
  view.tiles.replaceChildren(...tiles);
  view.tiles.hidden = !tiles.length;
}

// ---------------------------------------------------------------- charts

// Group metric keys into charts: "a/b" keys chart together under "a", and
// max_/mean_/min_ variants of one quantity chart together under its name.
function groupMetrics(keys) {
  const groups = new Map();
  const add = (group, series, key) => {
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push({ name: series, key });
  };
  const set = new Set(keys);
  for (const key of keys) {
    const slash = key.lastIndexOf("/");
    const stat = key.match(/^(max|mean|min)_(.+)$/);
    if (slash > 0) {
      add(key.slice(0, slash), key.slice(slash + 1), key);
    } else if (stat && ["max", "mean", "min"].filter((s) => set.has(`${s}_${stat[2]}`)).length > 1) {
      add(stat[2], stat[1], key);
    } else {
      add(key, key, key);
    }
  }
  for (const series of groups.values()) {
    const rank = { max: 0, mean: 1, min: 2 };
    series.sort((a, b) => (rank[a.name] ?? 3) - (rank[b.name] ?? 3));
  }
  const order = [...groups.keys()].sort((a, b) => {
    const ra = CHART_ORDER.indexOf(a), rb = CHART_ORDER.indexOf(b);
    return (ra < 0 ? 99 : ra) - (rb < 0 ? 99 : rb);
  });
  return order.map((name) => ({ name, series: groups.get(name) }));
}

function metricKeys() {
  const keys = [];
  // "generation" is the step itself, already the x axis
  const seen = new Set(["step", "time", "generation"]);
  for (const row of state.rows) {
    for (const k of Object.keys(row)) {
      if (!seen.has(k)) {
        seen.add(k);
        keys.push(k);
      }
    }
  }
  return keys;
}

function renderCharts() {
  const groups = groupMetrics(metricKeys());
  view.chartsTitle.hidden = !groups.length;
  view.charts.replaceChildren(...groups.map(chartCard));
}

function chartCard(group) {
  const series = group.series.map((s, i) => {
    // species_*/s<id> series are named and coloured after their species
    const species = group.name.startsWith("species_") && s.name.match(/^s(\d+)$/);
    return {
      ...s,
      name: species ? state.speciesNames[species[1]] || s.name : s.name,
      color: species ? speciesColor(Number(species[1])) : SERIES[i % SERIES.length],
      points: state.rows.filter((r) => r[s.key] != null).map((r) => [r.step, r[s.key]]),
    };
  });
  const single = series.length === 1;
  return seriesCard(group.name.replaceAll("_", " "), series, single && fmt(latest(series[0].key)));
}

// a card with a line chart of the series ([step, value] points), a legend when
// there's more than one, and `latestText` in the corner if given
function seriesCard(title, series, latestText) {
  const single = series.length === 1;
  const card = h(
    "div",
    { class: "card chart-card" },
    h(
      "div",
      { class: "card-hdr" },
      h("span", { class: "title", text: title }),
      h("span", { class: "spacer" }),
      latestText && h("span", { class: "latest", text: latestText }),
    ),
    !single &&
      h(
        "div",
        { class: "legend" },
        series.map((s) =>
          h("span", { class: "key" }, h("i", { style: `background:${s.color}` }), s.name),
        ),
      ),
  );
  const holder = h("div");
  card.append(holder);
  // draw once the card is laid out, so the chart knows its width
  requestAnimationFrame(() => holder.replaceChildren(lineChart(series, holder.clientWidth)));
  return card;
}

function niceTicks(lo, hi, count) {
  if (lo === hi) {
    const pad = Math.abs(lo) * 0.1 || 1;
    lo -= pad;
    hi += pad;
  }
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const start = Math.floor(lo / step) * step;
  // the last tick must reach hi, or the top of the data runs off the chart
  const ticks = [+start.toPrecision(12)];
  while (ticks[ticks.length - 1] < hi) ticks.push(+(start + ticks.length * step).toPrecision(12));
  return ticks;
}

function lineChart(series, width) {
  const W = Math.max(width, 200), H = 150;
  const all = series.flatMap((s) => s.points);
  const svg = h("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  if (!all.length) return svg;

  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]).filter(Number.isFinite);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const yt = niceTicks(Math.min(...ys), Math.max(...ys), 3);
  // room for the widest y label (9px mono is ~5.5px a character)
  const labelW = Math.max(...yt.map((t) => fmtTick(t).length)) * 5.5;
  const pad = { l: Math.max(28, labelW + 12), r: 10, t: 10, b: 18 };
  const y0 = yt[0], y1 = yt[yt.length - 1];
  const sx = (x) => pad.l + (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const sy = (y) => pad.t + (1 - (y - y0) / (y1 - y0)) * (H - pad.t - pad.b);

  for (const t of yt) {
    svg.append(
      h("line", { class: "grid", x1: pad.l, x2: W - pad.r, y1: sy(t), y2: sy(t) }),
      h("text", { class: "tick", x: pad.l - 6, y: sy(t) + 3, "text-anchor": "end", text: fmtTick(t) }),
    );
  }
  svg.append(h("line", { class: "axis", x1: pad.l, x2: W - pad.r, y1: H - pad.b, y2: H - pad.b }));
  const xt = niceTicks(x0, x1, Math.max(2, Math.floor((W - pad.l - pad.r) / 80)))
    .filter((t) => t >= x0 && t <= x1 && Number.isInteger(t));
  for (const t of xt) {
    svg.append(h("text", { class: "tick", x: sx(t), y: H - 5, "text-anchor": "middle", text: fmtTick(t) }));
  }

  for (const s of series) {
    // break the line wherever a series skips a step (e.g. a species that died out)
    let d = "", prev = null;
    for (const [x, y] of s.points) {
      if (!Number.isFinite(y)) { prev = null; continue; }
      d += `${prev === null ? "M" : "L"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`;
      prev = x;
    }
    svg.append(h("path", { class: "line", d, stroke: s.color }));
  }

  // hover: a crosshair snapped to the nearest logged step, every series in one tooltip
  const steps = [...new Set(xs)].sort((a, b) => a - b);
  const lookup = series.map((s) => new Map(s.points));
  const cross = h("line", { class: "cross", y1: pad.t, y2: H - pad.b, visibility: "hidden" });
  const dots = series.map((s) => h("circle", { class: "hover-dot", r: 4, fill: s.color, visibility: "hidden" }));
  svg.append(cross, ...dots);
  svg.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const target = x0 + ((px - pad.l) / (W - pad.l - pad.r)) * (x1 - x0);
    let best = steps[0];
    for (const st of steps) if (Math.abs(st - target) < Math.abs(best - target)) best = st;
    cross.setAttribute("x1", sx(best));
    cross.setAttribute("x2", sx(best));
    cross.setAttribute("visibility", "visible");
    const rows = [];
    series.forEach((s, i) => {
      const v = lookup[i].get(best);
      const ok = v != null && Number.isFinite(v);
      dots[i].setAttribute("visibility", ok ? "visible" : "hidden");
      if (!ok) return;
      dots[i].setAttribute("cx", sx(best));
      dots[i].setAttribute("cy", sy(v));
      rows.push([v, s]);
    });
    rows.sort((a, b) => b[0] - a[0]);
    showTip(
      e.clientX,
      e.clientY,
      h("div", { class: "head", text: `gen ${best}` }),
      ...rows.map(([v, s]) =>
        h(
          "div",
          { class: "row" },
          h("i", { style: `background:${s.color}` }),
          h("b", { text: fmt(v) }),
          series.length > 1 && h("span", { text: s.name }),
        ),
      ),
    );
  });
  svg.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
    hideTip();
  });
  return svg;
}

// redraw charts at the new width when the window is resized
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (!view) return;
    const tab = activeTab();
    if (tab) {
      if (tab.individual) renderMember(tab);
    } else {
      relayoutOverview();
    }
  }, 150);
});

// redraw what the run tab draws to fit its width
function relayoutOverview() {
  state.overviewStale = false;
  if (state.rows.length) renderCharts();
  // networks are drawn to fit; leave images alone so gifs don't restart
  for (const card of Object.values(view.mediaCards)) {
    if (card.shown?.endsWith(".json")) card.shown = null;
  }
  renderMedia();
}

// ---------------------------------------------------------------- media

const MEDIA_TITLES = { episode: "episode", network: "network" };
const MEDIA_ORDER = ["episode", "network"];
// the side the policy plays in the champion's game against an environment's
// built-in baseline: evojax's single-player slimevolley gives it the right
const BASELINE_SEAT = { slimevolley: 1 };

function mediaByKey() {
  const byKey = new Map();
  for (const m of state.media) {
    if (!byKey.has(m.key)) byKey.set(m.key, []);
    byKey.get(m.key).push(m);
  }
  for (const list of byKey.values()) list.sort((a, b) => a.step - b.step);
  const keys = [...byKey.keys()].sort((a, b) => {
    const ra = MEDIA_ORDER.indexOf(a), rb = MEDIA_ORDER.indexOf(b);
    return (ra < 0 ? 99 : ra) - (rb < 0 ? 99 : rb);
  });
  return keys.map((k) => [k, byKey.get(k)]);
}

function renderMedia() {
  const groups = mediaByKey();
  view.mediaTitle.hidden = !groups.length;
  for (const [key, entries] of groups) {
    let card = view.mediaCards[key];
    if (!card) {
      card = {
        hdr: h("div", { class: "card-hdr" }),
        body: h("div", { class: "media-body" }),
        who: h("div"),
        foot: h("div"),
        shown: null,
      };
      card.el = h("div", { class: "card" }, card.hdr, card.body, card.who, card.foot);
      view.mediaCards[key] = card;
      view.media.append(card.el);
    }
    updateMediaCard(key, entries, card);
  }
}

function updateMediaCard(key, entries, card) {
  const sel = state.mediaSel[key];
  const idx = sel == null ? entries.length - 1 : sel;
  const entry = entries[idx];
  const set = (i) => {
    state.mediaSel[key] = i >= entries.length - 1 ? null : Math.max(0, i);
    updateMediaCard(key, entries, card);
  };
  card.hdr.replaceChildren(
    h("span", { class: "title", text: MEDIA_TITLES[key] || key.replaceAll("_", " ") }),
    h("span", { class: "spacer" }),
    h(
      "span",
      { class: "scrub" },
      h("button", { onclick: () => set(idx - 1), disabled: idx === 0, title: "previous", text: "‹" }),
      h("span", { class: "pos", text: `gen ${entry.step}` }),
      h("button", { onclick: () => set(idx + 1), disabled: idx === entries.length - 1, title: "next", text: "›" }),
      h("button", {
        class: sel == null ? "on" : null,
        onclick: () => set(entries.length - 1),
        title: "follow the newest",
        text: "latest",
      }),
    ),
  );
  card.who.replaceChildren(...championLine(key, entry));
  // swapping the element restarts a gif, so only do it when the entry changes
  const id = entry.file || `error-${entry.step}`;
  if (card.shown === id) return;
  card.shown = id;
  card.foot.replaceChildren();
  if (entry.error) {
    card.body.replaceChildren(h("div", { class: "media-error", text: `couldn't render: ${entry.error}` }));
    return;
  }
  const src = `/media/${state.runId}/${encodeURIComponent(entry.file)}`;
  if (entry.file.endsWith(".json")) {
    card.body.replaceChildren();
    fetch(src)
      .then((r) => r.json())
      .then((data) => {
        if (card.shown !== entry.file) return;
        if (data.type === "network") {
          const { svg, summary } = networkView(data, card.body.clientWidth - 22);
          card.body.replaceChildren(svg);
          card.foot.replaceChildren(h("div", { class: "net-foot", text: summary }));
        } else {
          card.body.replaceChildren(h("pre", { text: JSON.stringify(data, null, 2) }));
        }
      });
  } else if (entry.file.endsWith(".mp4")) {
    card.body.replaceChildren(h("video", { src, autoplay: true, loop: true, muted: true, controls: true }));
  } else {
    card.body.replaceChildren(h("img", { src, alt: `${key} at gen ${entry.step}` }));
  }
}

// the champion's episode and network are that generation's champion's; name it,
// and in a game against a built-in baseline, who played which side
function championLine(key, entry) {
  const champ = state.champions.get(entry.step);
  if (!champ || !MEDIA_ORDER.includes(key)) return [];
  const name = h("b", {}, genomeLink(champ.id, champ.name));
  const seat = key === "episode" ? BASELINE_SEAT[entry.env] : undefined;
  if (seat === undefined) {
    return [h("div", { class: "ep-games" }, h("span", { class: "ep-player" }, h("span", { class: "seat", text: "champion" }), swatch(champ.birth_species), name))];
  }
  const players = [seatPlayer(name, seat), seatPlayer(h("span", { text: "baseline" }), 1 - seat)];
  if (seat === 1) players.reverse();
  return [h("div", { class: "ep-games" }, h("span", { class: "ep-match" }, players[0], h("span", { class: "dim", text: "vs" }), players[1]))];
}

async function pollChampions() {
  const runId = state.runId;
  try {
    const champions = await api(`/api/runs/${runId}/champions`);
    if (runId !== state.runId) return;
    state.champions = new Map(champions.map((c) => [c.step, c]));
    if (state.activeTab == null) renderMedia();
    else state.overviewStale = true;
  } catch {
    // runs from before members were logged have none
  }
}

// ---------------------------------------------------------------- network

function networkView(net, width) {
  const nodes = new Map(net.nodes.map((n) => [n.id, { ...n, depth: n.kind === "input" ? 0 : 1 }]));
  // longest path from the inputs; bounded so a cycle can't hang the page
  for (let pass = 0; pass < nodes.size; pass++) {
    let changed = false;
    for (const e of net.edges) {
      const a = nodes.get(e.from), b = nodes.get(e.to);
      if (a && b && b.kind !== "input" && b.depth < a.depth + 1) {
        b.depth = a.depth + 1;
        changed = true;
      }
    }
    if (!changed) break;
  }
  const hidden = [...nodes.values()].filter((n) => n.kind === "hidden");
  const outDepth = Math.max(1, ...hidden.map((n) => n.depth + 1));
  for (const n of nodes.values()) if (n.kind === "output") n.depth = outDepth;

  const columns = [];
  for (const n of nodes.values()) (columns[n.depth] ||= []).push(n);
  const inputs = columns[0] || [];
  const outputs = columns[outDepth] || [];

  const ROW = 20;
  const rows = Math.max(...columns.map((c) => (c ? c.length : 0)), 3);
  // deep, narrow networks still get room to fan out
  const H = Math.max(rows * ROW + 16, 240);
  const labelW = (list) => Math.max(0, ...list.map((n) => (n.label || "").length)) * 6.2 + 12;
  const padL = Math.min(160, labelW(inputs)), padR = Math.min(160, labelW(outputs));
  const W = Math.max(width, padL + padR + 120);
  const colX = (d) => padL + (d / outDepth) * (W - padL - padR);

  // order each column by where its inputs sit, so edges cross less
  const placeColumn = (col, d) => {
    if (d > 0 && d < outDepth) {
      for (const n of col) {
        const ys = net.edges.filter((e) => e.to === n.id).map((e) => nodes.get(e.from)?.y).filter((y) => y != null);
        n.order = ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : H / 2;
      }
      col.sort((a, b) => a.order - b.order);
    } else {
      col.sort((a, b) => a.id - b.id);
    }
    // spread short columns out, up to 3 rows apart
    const gap = Math.min(ROW * 3, (H - 16) / col.length);
    const top = (H - col.length * gap) / 2 + gap / 2;
    col.forEach((n, i) => {
      n.x = colX(d);
      n.y = top + i * gap;
    });
  };
  columns.forEach((col, d) => col && placeColumn(col, d));

  const svg = h("svg", { class: "net", viewBox: `0 0 ${W} ${H}`, height: H });
  const maxW = Math.max(1e-6, ...net.edges.map((e) => Math.abs(e.weight)));
  const edgeEls = [];
  for (const e of net.edges) {
    const a = nodes.get(e.from), b = nodes.get(e.to);
    if (!a || !b) continue;
    const s = Math.abs(e.weight) / maxW;
    const mx = (a.x + b.x) / 2;
    const el = h("path", {
      class: "edge",
      d: `M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`,
      stroke: e.weight >= 0 ? "var(--s1)" : "var(--s2)",
      "stroke-width": (0.75 + 2.25 * s).toFixed(2),
      opacity: (0.3 + 0.55 * s).toFixed(2),
    });
    el.dataset.from = e.from;
    el.dataset.to = e.to;
    edgeEls.push(el);
    svg.append(el);
  }
  for (const n of nodes.values()) {
    const node = h("circle", { class: `node ${n.kind}`, cx: n.x, cy: n.y, r: n.kind === "hidden" ? 4.5 : 5 });
    node.addEventListener("pointerenter", (ev) => {
      for (const el of edgeEls) {
        el.classList.toggle("dim", el.dataset.from != n.id && el.dataset.to != n.id);
      }
      showTip(
        ev.clientX,
        ev.clientY,
        h("div", { class: "head", text: n.label ? `${n.kind} · ${n.label}` : `${n.kind} · node ${n.id}` }),
        h("div", { class: "row" }, h("b", { text: n.activation })),
        n.kind !== "input" && h("div", { class: "row" }, "bias ", h("b", { text: fmt(n.bias) })),
      );
    });
    node.addEventListener("pointerleave", () => {
      edgeEls.forEach((el) => el.classList.remove("dim"));
      hideTip();
    });
    svg.append(node);
    if (n.kind === "input") {
      svg.append(h("text", { class: "lab", x: n.x - 9, y: n.y + 3.5, "text-anchor": "end", text: n.label }));
    } else if (n.kind === "output") {
      svg.append(h("text", { class: "lab", x: n.x + 9, y: n.y + 3.5, text: n.label }));
    }
  }
  const summary = `${hidden.length} hidden · ${net.edges.length} connections`;
  return { svg, summary };
}

// ---------------------------------------------------------------- leaderboard

function speciesColor(id) {
  return SERIES[((id % SERIES.length) + SERIES.length) % SERIES.length];
}

// "big-red-dog", with the species part set apart
function nameEl(name) {
  const cut = name.lastIndexOf("-");
  return h("span", { class: "gname" }, name.slice(0, cut), h("span", { class: "sp", text: name.slice(cut) }));
}

function swatch(species) {
  return h("i", { class: "swatch", style: `background:${speciesColor(species)}` });
}

// a link that opens a member's tab; `at` ({step, game}) opens it on one of its games
function genomeLink(id, name, at = null) {
  return h(
    "a",
    {
      class: "glink",
      href: `/run/${state.runId}/genome/${id}`,
      onclick: (e) => {
        if (e.metaKey || e.ctrlKey) return;
        e.preventDefault();
        openTab(id, { name, at });
      },
    },
    nameEl(name),
  );
}

async function pollBoard() {
  const runId = state.runId;
  const q = `scope=${state.boardScope}&sort=${state.boardSort}&limit=${BOARD_LIMIT}`;
  try {
    const board = await api(`/api/runs/${runId}/leaderboard?${q}`);
    if (runId !== state.runId) return;
    const renamed = JSON.stringify(board.species_names) !== JSON.stringify(state.speciesNames);
    state.board = board;
    state.speciesNames = board.species_names;
    renderBoard();
    if (renamed && state.rows.length) {
      if (state.activeTab == null) renderCharts();
      else state.overviewStale = true;
    }
  } catch {
    // the run's charts carry on without it
  }
}

const BOARD_SORTS = [
  { sort: "fitness", label: "fitness", title: "fitness in its latest generation" },
  { sort: "best", label: "best", title: "best fitness in any generation" },
  { sort: "age", label: "gens", title: "generations in the population" },
  { sort: "children", label: "kids", title: "children" },
];

// the leaderboard, in the panel on the right; it's there whenever the run logged
// its members
function renderBoard() {
  const board = state.board;
  const has = board && board.step != null;
  document.body.classList.toggle("has-board", Boolean(has));
  const panel = document.getElementById("board");
  if (!has) {
    panel.replaceChildren();
    return;
  }
  const toggle = (on, text, title, onclick) => h("button", { class: on ? "on" : null, text, title, onclick });
  const scopes = h(
    "span",
    { class: "scrub" },
    [["alive", "this gen"], ["all", "all time"]].map(([value, text]) =>
      toggle(state.boardScope === value, text, null, () => {
        state.boardScope = value;
        // all time defaults to the best ever, this generation to the latest
        state.boardSort = value === "all" ? "best" : "fitness";
        pollBoard();
      }),
    ),
  );
  const sorts = h(
    "span",
    { class: "scrub" },
    h("span", { class: "dim", text: "by" }),
    BOARD_SORTS.map((c) =>
      toggle(state.boardSort === c.sort, c.label, c.title, () => {
        state.boardSort = c.sort;
        pollBoard();
      }),
    ),
  );
  const th = (text, cls, sort) =>
    h("th", { class: [cls, sort && sort === state.boardSort && "on"].filter(Boolean).join(" ") || null, text });
  const head = h(
    "tr",
    {},
    th("#", "rank"),
    th("individual"),
    th("fitness", "num", "fitness"),
    th("best", "num", "best"),
    th("gens", "num", "age"),
    th("kids", "num", "children"),
    h("th", { text: "rank over life", title: "its rank by fitness in each generation it was in" }),
  );
  // in this generation, rank lines share one time axis so longevity shows
  const shared = state.boardScope === "alive" && {
    x0: Math.min(...board.rows.map((r) => r.history[0]?.[0] ?? board.step)),
    x1: board.step,
  };
  const open = new Set(state.tabs.map((t) => t.id));
  const rows = board.rows.map((r, i) =>
    h(
      "tr",
      {
        class: ["row", r.id === state.activeTab && "sel", open.has(r.id) && "open", !r.alive && "gone"].filter(Boolean).join(" "),
        title: `${r.name} · ${r.species_name}${r.alive ? `, born gen ${r.born}` : `, gen ${r.born}–${r.last}`}`,
        onclick: () => openTab(r.id, { name: r.name }),
      },
      h("td", { class: "rank", text: i + 1 }),
      h("td", { class: "who" }, swatch(r.birth_species), nameEl(r.name), r.champion && h("span", { class: "tag", text: "champ" })),
      h("td", { class: "num", text: fmt(r.fitness) }),
      h("td", { class: "num", text: fmt(r.best) }),
      h("td", { class: "num", text: r.age }),
      h("td", { class: "num", text: r.children }),
      h(
        "td",
        { title: r.history.length ? `rank ${r.history.at(-1)[2]} of ${board.population} in gen ${r.history.at(-1)[0]}` : null },
        rankline(r.history, board.population, speciesColor(r.birth_species), shared),
      ),
    ),
  );
  // the table keeps its scroll from one generation to the next
  if (!panel.firstChild) panel.append(h("div", { class: "bp-head" }), h("div", { class: "bp-table" }));
  const [top, wrap] = panel.children;
  top.replaceChildren(
    h("div", { class: "stitle", text: "Leaderboard" }),
    h("div", { class: "bp-sub", text: `gen ${board.step} · ${board.population} members · ${board.total.toLocaleString("en")} seen` }),
    h("div", { class: "bp-controls" }, scopes, sorts),
  );
  const scrolled = wrap.scrollTop;
  wrap.replaceChildren(h("table", { class: "board" }, h("thead", {}, head), h("tbody", {}, rows)));
  wrap.scrollTop = scrolled;
}

// a member's rank over its life ([step, fitness, rank] points), first place at the
// top, on one log scale (1 to the population size) for every row so they compare;
// `domain` ({x0, x1}) puts rows on one time axis too
function rankline(points, population, color, domain) {
  const W = 64, H = 16;
  const svg = h("svg", { class: "spark", viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  const pts = points.map(([step, , rank]) => [step, rank]);
  if (!pts.length) return svg;
  const x0 = domain ? domain.x0 : pts[0][0];
  const x1 = domain ? domain.x1 : pts[pts.length - 1][0];
  const sx = (x) => 3 + (x1 === x0 ? 1 : (x - x0) / (x1 - x0)) * (W - 6);
  const sy = (rank) => 3 + (Math.log(rank) / Math.log(Math.max(2, population))) * (H - 6);
  if (pts.length > 1) {
    const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("");
    svg.append(h("path", { d, stroke: color, fill: "none", "stroke-width": 1.25 }));
  }
  const [lx, ly] = pts[pts.length - 1];
  svg.append(h("circle", { cx: sx(lx), cy: sy(ly), r: 2, fill: color }));
  return svg;
}

// ---------------------------------------------------------------- tabs

// The run's own tab is always there; each member opened gets a tab of its own,
// which keeps its place (pedigree depth, the generation and game it shows, how far
// down it's scrolled) while others are shown.

function activeTab() {
  return state.tabs.find((t) => t.id === state.activeTab) || null;
}

// the member tabs a run had open, kept for the browser session
function savedTabs(runId) {
  try {
    return JSON.parse(sessionStorage.getItem(`tabs:${runId}`)) || [];
  } catch {
    return [];
  }
}

function saveTabs() {
  sessionStorage.setItem(`tabs:${state.runId}`, JSON.stringify(state.tabs.map((t) => ({ id: t.id, name: t.name }))));
}

function addTab(id, name) {
  let tab = state.tabs.find((t) => t.id === id);
  if (!tab) {
    tab = {
      id,
      name: name || null,
      individual: null,
      depth: PEDIGREE_DEPTH,
      // the generation of its life the network and episode cards show (pinned to
      // its latest when first shown), and the game shown when it was opened on
      // one (by the game's index), rather than its own
      step: null,
      game: null,
      // network and episode data fetched, by "<kind>@<step>"
      data: {},
      cards: null,
      scroll: 0,
      stale: true,
      el: h("div", { class: "tab-panel", hidden: true }),
    };
    state.tabs.push(tab);
    view.panels.append(tab.el);
    saveTabs();
  }
  return tab;
}

// show a member's tab, opening it if need be; null shows the run's tab. `at`
// ({step, game}) turns it to one of its games, e.g. the one it played with the
// member whose tab it was opened from
function openTab(id, { name = null, push = true, at = null } = {}) {
  if (!view) return;
  if (id != null) {
    const tab = addTab(id, name);
    if (at) {
      tab.step = at.step;
      tab.game = at.game;
    }
  }
  if (push && id !== state.activeTab) {
    history.pushState(null, "", id == null ? `/run/${state.runId}` : `/run/${state.runId}/genome/${id}`);
  }
  // each tab comes back to where it was scrolled
  const leaving = activeTab();
  if (leaving) leaving.scroll = main.scrollTop;
  else state.overviewScroll = main.scrollTop;
  state.activeTab = id;
  view.overview.hidden = id != null;
  for (const t of state.tabs) t.el.hidden = t.id !== id;
  renderTabs();
  if (state.board) renderBoard();

  const tab = activeTab();
  if (tab) {
    renderMember(tab);
    if (tab.stale) pollMember(tab);
    main.scrollTop = tab.scroll;
  } else {
    if (state.overviewStale) relayoutOverview();
    main.scrollTop = state.overviewScroll || 0;
  }
}

function closeTab(id) {
  const i = state.tabs.findIndex((t) => t.id === id);
  if (i < 0) return;
  const [tab] = state.tabs.splice(i, 1);
  tab.el.remove();
  saveTabs();
  if (state.activeTab === id) {
    // the tab to its left, or the run's own
    openTab(state.tabs[i - 1]?.id ?? state.tabs[i]?.id ?? null);
  } else {
    renderTabs();
    if (state.board) renderBoard();
  }
}

function renderTabs() {
  const item = (id, on, label, closable) =>
    h(
      "div",
      {
        class: `tab${on ? " on" : ""}`,
        title: id == null ? "the run" : `#${id}`,
        onclick: () => openTab(id),
        // middle click closes, like a browser tab
        onauxclick: (e) => e.button === 1 && closable && closeTab(id),
      },
      label,
      closable &&
        h("button", {
          class: "x",
          title: "close",
          text: "×",
          onclick: (e) => {
            e.stopPropagation();
            closeTab(id);
          },
        }),
    );
  view.tabbar.replaceChildren(
    item(null, state.activeTab == null, h("span", { text: "run" }), false),
    ...state.tabs.map((t) => {
      const species = t.individual?.birth_species;
      const label = h("span", {}, species != null && swatch(species), t.name ? nameEl(t.name) : `#${t.id}`);
      return item(t.id, t.id === state.activeTab, label, true);
    }),
  );
}

// ---------------------------------------------------------------- member tab

async function pollMember(tab) {
  const runId = state.runId, depth = tab.depth;
  let data;
  try {
    data = await api(`/api/runs/${runId}/individuals/${tab.id}?depth=${depth}`);
  } catch (e) {
    if (!String(e.message).startsWith("404")) return;
    data = { missing: true, id: tab.id };
  }
  if (runId !== state.runId || !state.tabs.includes(tab) || depth !== tab.depth) return;
  tab.stale = false;
  // a new generation changes little about most members; skip redrawing when nothing did
  if (JSON.stringify(data) === JSON.stringify(tab.individual)) return;
  const named = !tab.individual;
  tab.individual = data;
  if (data.name && data.name !== tab.name) {
    tab.name = data.name;
    saveTabs();
  }
  if (named) renderTabs();
  if (tab.id === state.activeTab) renderMember(tab);
}

function renderMember(tab) {
  const ind = tab.individual;
  if (!ind) {
    tab.el.replaceChildren(h("div", { class: "card loading", text: "tracing…" }));
    return;
  }
  if (ind.missing) {
    tab.el.replaceChildren(h("div", { class: "card loading", text: `no genome #${ind.id} in this run` }));
    return;
  }
  const sep = () => h("span", { class: "sep", text: "·" });
  const [parent, mate] = ind.parents;
  const nameOf = (id) => ind.pedigree.nodes.find((n) => n.id === id)?.name;
  const parentage =
    parent < 0
      ? h("span", { text: "founder" })
      : h(
          "span",
          {},
          "child of ",
          genomeLink(parent, nameOf(parent)),
          mate >= 0 && mate !== parent && [" × ", genomeLink(mate, nameOf(mate))],
        );
  const share = ind.population ? ` (${Math.round((100 * ind.descendants) / ind.population)}%)` : "";
  const tile = (lab, val) => h("div", { class: "tile" }, h("div", { class: "lab", text: lab }), h("div", { class: "val", text: val }));

  const head = h(
    "div",
    { class: "card ind" },
    h(
      "div",
      { class: "ind-head" },
      swatch(ind.birth_species),
      h("h2", {}, nameEl(ind.name)),
      h("span", { class: "gid", text: `#${ind.id}` }),
      ind.champion && h("span", { class: "tag", text: "champion" }),
    ),
    h(
      "div",
      { class: "ind-meta" },
      h("span", { text: ind.species_name }),
      ind.species !== ind.birth_species && h("span", { class: "dim", text: `(born ${state.speciesNames[ind.birth_species] ?? ind.birth_species})` }),
      sep(),
      h("span", { text: ind.alive ? `alive, born gen ${ind.born}` : `gen ${ind.born}–${ind.last}` }),
      sep(),
      parentage,
    ),
  );
  const tiles = h(
    "div",
    { class: "tiles" },
    tile("fitness", fmt(ind.fitness)),
    tile("best fitness", fmt(ind.best)),
    tile("generations", ind.age),
    tile("children", ind.children),
    tile("living descendants", `${ind.descendants}${share}`),
    ind.champion_gens > 0 && tile("gens as champion", ind.champion_gens),
  );

  const depthCtl = h(
    "span",
    { class: "scrub" },
    h("button", { title: "fewer generations", text: "‹", disabled: tab.depth <= 1, onclick: () => setDepth(tab, tab.depth - PEDIGREE_STEP) }),
    h("span", { class: "pos", text: `${tab.depth} gens back` }),
    h("button", { title: "more generations", text: "›", disabled: !ind.pedigree.deeper, onclick: () => setDepth(tab, tab.depth + PEDIGREE_STEP) }),
  );
  const pedBody = h("div", { class: "ped-scroll" });
  const pedigree = h(
    "div",
    { class: "card" },
    h("div", { class: "card-hdr" }, h("span", { class: "title", text: "pedigree" }), h("span", { class: "spacer" }), depthCtl),
    pedBody,
    h("div", { class: "net-foot", text: "fitter parent solid, mate dashed, line of descent highlighted · ringed members have been champion · click one to open it" }),
  );

  const popSeries = (key, name, color) => ({
    name,
    color,
    points: state.rows.filter((r) => r[key] != null).map((r) => [r.step, r[key]]),
  });
  const chart = seriesCard("fitness along the line of descent", [
    { name: "line of descent", color: "var(--s1)", points: ind.line_fitness },
    popSeries("max_fitness", "population max", "var(--s2)"),
    popSeries("mean_fitness", "population mean", "var(--s3)"),
  ].filter((s) => s.points.length));

  // pinned when first shown, so the cards don't change under the reader as
  // generations go by
  if (tab.step == null) tab.step = ind.last;
  const cards = memberCards(tab);

  const line = h(
    "div",
    { class: "card" },
    h("div", { class: "card-hdr" }, h("span", { class: "title", text: `line of descent · ${ind.line.length - 1} fitter parents back to a founder` })),
    h(
      "div",
      { class: "board-wrap short" },
      h(
        "table",
        { class: "board" },
        h("thead", {}, h("tr", {}, h("th", { class: "num", text: "born" }), h("th", { text: "individual" }), h("th", { class: "num", text: "gens" }), h("th", { class: "num", text: "best" }), h("th", { text: "mate" }))),
        h(
          "tbody",
          {},
          [...ind.line].reverse().map((a) =>
            h(
              "tr",
              { class: a.id === ind.id ? "sel" : null },
              h("td", { class: "num dim", text: a.born }),
              h("td", {}, swatch(a.birth_species), a.id === ind.id ? nameEl(a.name) : genomeLink(a.id, a.name)),
              h("td", { class: "num", text: a.age }),
              h("td", { class: "num", text: fmt(a.best) }),
              h("td", {}, a.mate ? genomeLink(a.mate.id, a.mate.name) : h("span", { class: "dim", text: a.parents[0] < 0 ? "founder" : "–" })),
            ),
          ),
        ),
      ),
    ),
  );

  const kids = ind.child_list;
  const children =
    kids.length > 0 &&
    h(
      "div",
      { class: "card" },
      h("div", { class: "card-hdr" }, h("span", { class: "title", text: `children · ${ind.children}, fittest first` })),
      h(
        "div",
        { class: "chips" },
        kids.map((c) => h("span", { class: `chip${c.alive ? "" : " gone"}` }, swatch(c.birth_species), genomeLink(c.id, c.name), h("span", { class: "dim", text: fmt(c.best) }))),
        ind.children > kids.length && h("span", { class: "dim", text: `and ${ind.children - kids.length} more` }),
      ),
    );

  const scrolled = main.scrollTop;
  tab.el.replaceChildren(
    ...[
      head,
      tiles,
      h("div", { class: "lineage-grid" }, cards.network.el, cards.episode.el),
      // its family, below what it is and does
      h("div", { class: "stitle", text: "Lineage" }),
      pedigree,
      h("div", { class: "lineage-grid" }, chart, line),
      children,
    ].filter(Boolean),
  );
  if (tab.id === state.activeTab) main.scrollTop = scrolled;
  loadMemberCards(tab);
  requestAnimationFrame(() => {
    pedBody.replaceChildren(pedigreeView(ind, pedBody.clientWidth));
    // the member itself is on the right; start there
    pedBody.scrollLeft = pedBody.scrollWidth;
  });
}

// the member's network and episodes, in the generation tab.step. Made once per
// tab and kept across redraws, so an episode that's playing isn't restarted by
// every new generation; hidden for runs that didn't log them
function memberCards(tab) {
  if (tab.cards) return tab.cards;
  const card = () => {
    const c = { hdr: h("div", { class: "card-hdr" }), body: h("div", { class: "media-body" }), foot: h("div"), shown: null };
    c.el = h("div", { class: "card" }, c.hdr, c.body, c.foot);
    c.el.hidden = true;
    return c;
  };
  tab.cards = { network: card(), episode: card() };
  return tab.cards;
}

function loadMemberCards(tab) {
  loadNetwork(tab, tab.cards.network);
  loadEpisodes(tab, tab.cards.episode);
}

// the member's network or episodes ("network" or "episodes") in tab.step: null
// if there are none, undefined if the reader moved on while fetching
async function memberData(tab, kind) {
  const runId = state.runId, step = tab.step;
  const key = `${kind}@${step}`;
  if (!tab.data[key]) {
    let data = null;
    try {
      data = await api(`/api/runs/${runId}/individuals/${tab.id}/${kind}?step=${step}`);
    } catch {
      // not logged (or not yet)
    }
    if (data) tab.data[key] = data;
    if (runId !== state.runId || !state.tabs.includes(tab) || step !== tab.step) return undefined;
    return data;
  }
  return tab.data[key];
}

// steps through the generations of the member's life; moves both cards together
function stepScrub(tab, steps, step) {
  const last = tab.individual.last;
  if (steps.length < 2 && step === last) return h("span", { class: "latest", text: `gen ${step}` });
  const idx = steps.indexOf(step);
  const go = (s) => {
    tab.step = s;
    loadMemberCards(tab);
  };
  return h(
    "span",
    { class: "scrub" },
    h("button", { onclick: () => go(steps[idx - 1]), disabled: idx <= 0, title: "previous", text: "‹" }),
    h("span", { class: "pos", text: `gen ${step}` }),
    h("button", { onclick: () => go(steps[idx + 1]), disabled: idx >= steps.length - 1, title: "next", text: "›" }),
    h("button", { class: step === last ? "on" : null, onclick: () => go(last), title: "its latest generation", text: "latest" }),
  );
}

async function loadNetwork(tab, card) {
  const net = await memberData(tab, "network");
  if (net === undefined) return;
  card.el.hidden = !net;
  if (!net) return;
  card.hdr.replaceChildren(h("span", { class: "title", text: "network" }), h("span", { class: "spacer" }), stepScrub(tab, net.steps, net.step));
  requestAnimationFrame(() => {
    const { svg, summary } = networkView(net, card.body.clientWidth - 22);
    card.body.replaceChildren(svg);
    card.foot.replaceChildren(h("div", { class: "net-foot", text: summary }));
  });
}

// two-player games: which side the member played, in the renderers' colours
const SEATS = [
  { name: "left", color: "var(--s1)" },
  { name: "right", color: "var(--s2)" },
];

// a player of a two-player game, marked with its side's colour in the picture
function seatPlayer(who, seat) {
  return h(
    "span",
    { class: "ep-player" },
    h("i", { class: "swatch", style: `background:${SEATS[seat].color}` }),
    who,
    h("span", { class: "seat", text: SEATS[seat].name }),
  );
}

async function loadEpisodes(tab, card) {
  const eps = await memberData(tab, "episodes");
  if (eps === undefined) return;
  card.el.hidden = !eps?.episodes.length;
  if (card.el.hidden) return;
  // a two-player task shows each side the same view of the game (slimevolley
  // mirrors observations), so one game is enough: the member's own, the one it's
  // seated first in, unless it was opened on another
  const ep =
    eps.episodes.find((e) => e.index === tab.game) ||
    eps.episodes.find((e) => e.seat === 0) ||
    eps.episodes[0];
  card.hdr.replaceChildren(
    h("span", { class: "title", text: "episode" }),
    h("span", { class: "spacer" }),
    stepScrub(tab, eps.steps, eps.step),
  );
  // two-player games: say who played which side, in the order and colours of the
  // picture; the opponent opens on this same game
  const twoPlayer = ep.players.length === 2;
  const player = (p, seat) =>
    seatPlayer(
      p.id === tab.id
        ? h("b", {}, nameEl(p.name || `#${p.id}`))
        : p.name
          ? genomeLink(p.id, p.name, { step: eps.step, game: ep.index })
          : `#${p.id}`,
      seat,
    );
  const matchup =
    twoPlayer &&
    h("div", { class: "ep-games" }, h("span", { class: "ep-match" }, player(ep.players[0], 0), h("span", { class: "dim", text: "vs" }), player(ep.players[1], 1)));
  card.foot.replaceChildren(...[matchup].filter(Boolean));
  // swapping the image restarts the gif, so only when the episode changes
  const src = `/api/runs/${state.runId}/member_episodes/${eps.step}/${ep.index}`;
  if (card.shown === src) return;
  card.shown = src;
  card.body.replaceChildren(h("div", { class: "loading", text: "rendering…" }));
  const img = h("img", { alt: `episode at gen ${eps.step}` });
  img.addEventListener("load", () => card.shown === src && card.body.replaceChildren(img));
  img.addEventListener("error", () => {
    if (card.shown === src) card.body.replaceChildren(h("div", { class: "media-error", text: "couldn't render this episode" }));
  });
  img.src = src;
}

function setDepth(tab, depth) {
  tab.depth = Math.max(1, Math.min(48, depth));
  pollMember(tab);
}

// ancestors laid out left to right by the generation they were born in, with the
// line of descent straight across the middle
function pedigreeView(ind, width) {
  const nodes = new Map(ind.pedigree.nodes.map((n) => [n.id, { ...n }]));
  const kids = new Map();
  for (const n of nodes.values()) {
    for (const p of new Set(n.parents)) {
      if (!nodes.has(p)) continue;
      if (!kids.has(p)) kids.set(p, []);
      kids.get(p).push(n);
    }
  }
  const steps = [...new Set([...nodes.values()].map((n) => n.born))].sort((a, b) => a - b);
  const COL = 96, GAP = 24, PADX = 72, PADY = 28, AXIS = 16;
  const W = Math.max(width, PADX * 2 + (steps.length - 1) * COL);
  const colX = (i) => (steps.length === 1 ? W / 2 : PADX + (i * (W - 2 * PADX)) / (steps.length - 1));

  // newest first, so each column can line up behind the children it already placed
  for (let i = steps.length - 1; i >= 0; i--) {
    const col = [...nodes.values()].filter((n) => n.born === steps[i]);
    for (const n of col) {
      const ys = (kids.get(n.id) || []).map((k) => k.y);
      n.bary = ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 0;
    }
    col.sort((a, b) => a.bary - b.bary || a.id - b.id);
    const anchor = col.findIndex((n) => n.on_line);
    const mid = col.reduce((a, n) => a + n.bary, 0) / col.length;
    const top = anchor >= 0 ? -anchor * GAP : mid - ((col.length - 1) * GAP) / 2;
    col.forEach((n, j) => {
      n.x = colX(i);
      n.y = top + j * GAP;
      n.col = i;
    });
  }
  const ys = [...nodes.values()].map((n) => n.y);
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const H = hi - lo + 2 * PADY + AXIS;
  for (const n of nodes.values()) n.y += PADY - lo;

  const svg = h("svg", { class: "ped", viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  steps.forEach((st, i) => svg.append(h("text", { class: "gen", x: colX(i), y: H - 4, "text-anchor": "middle", text: `gen ${st}` })));

  const edgeEls = [];
  for (const n of nodes.values()) {
    [...new Set(n.parents)].forEach((pid) => {
      const p = nodes.get(pid);
      if (!p) return;
      const fitter = pid === n.parents[0];
      const cls = fitter && p.on_line && n.on_line ? "line" : fitter ? "primary" : "mate";
      const mx = (p.x + n.x) / 2;
      const el = h("path", { class: `edge ${cls}`, d: `M${p.x},${p.y} C${mx},${p.y} ${mx},${n.y} ${n.x},${n.y}` });
      el.dataset.from = pid;
      el.dataset.to = n.id;
      edgeEls.push(el);
      svg.append(el);
    });
  }
  for (const n of nodes.values()) {
    const subject = n.id === ind.id;
    if (n.champion_gens > 0) svg.append(h("circle", { class: "champ", cx: n.x, cy: n.y, r: subject ? 10.5 : 8.5 }));
    const node = h("circle", {
      class: `node${subject ? " subject" : ""}`,
      cx: n.x,
      cy: n.y,
      r: subject ? 7 : 5,
      fill: speciesColor(n.birth_species),
    });
    node.addEventListener("pointerenter", (ev) => {
      for (const el of edgeEls) el.classList.toggle("dim", el.dataset.from != n.id && el.dataset.to != n.id);
      showTip(
        ev.clientX,
        ev.clientY,
        h("div", { class: "head", text: n.name }),
        h("div", { class: "row" }, `#${n.id} · ${n.species_name} · gen ${n.born}${n.last > n.born ? `–${n.last}` : ""}`),
        h("div", { class: "row" }, "best ", h("b", { text: fmt(n.best) }), ` · ${n.children} children`),
      );
    });
    node.addEventListener("pointerleave", () => {
      edgeEls.forEach((el) => el.classList.remove("dim"));
      hideTip();
    });
    if (!subject) {
      node.addEventListener("click", () => {
        hideTip();
        openTab(n.id, { name: n.name });
      });
    }
    svg.append(node);
    // name the line of descent, alternating above and below so neighbours don't collide
    if (n.on_line) {
      const below = (steps.length - 1 - n.col) % 2 === 0;
      svg.append(
        h("text", {
          class: `lab${subject ? " subject" : ""}`,
          x: n.x,
          y: below ? n.y + (subject ? 19 : 16) : n.y - (subject ? 12 : 10),
          "text-anchor": "middle",
          text: n.name,
        }),
      );
    }
  }
  return svg;
}

// ---------------------------------------------------------------- config

function configValue(v) {
  if (v == null) return "–";
  if (Array.isArray(v) && v.every((x) => typeof x !== "object")) return v.join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v).replace(/<function (\S+) at 0x[0-9a-f]+>/g, "$1");
}

function renderConfig() {
  const config = state.run.config || {};
  const sections = Object.entries(config).filter(([, v]) => v && typeof v === "object" && !Array.isArray(v));
  const flat = Object.entries(config).filter(([, v]) => !(v && typeof v === "object" && !Array.isArray(v)));
  if (flat.length) sections.unshift(["config", Object.fromEntries(flat)]);
  view.configTitle.hidden = !sections.length;
  view.config.replaceChildren(
    ...sections.map(([name, values]) =>
      h(
        "div",
        { class: "card" },
        h("div", { class: "card-hdr" }, h("span", { class: "title", text: name.replaceAll("_", " ") })),
        h(
          "table",
          {},
          Object.entries(values).map(([k, v]) =>
            h("tr", {}, h("td", { text: k }), h("td", { text: configValue(v) })),
          ),
        ),
      ),
    ),
  );
}

// ---------------------------------------------------------------- start

async function start() {
  await pollRuns();
  const runId = route();
  if (runId) select(runId);
  else renderEmpty();
  setInterval(pollRuns, RUNS_POLL_MS);
  setInterval(() => {
    if (document.visibilityState === "visible") pollRun();
  }, POLL_MS);
}

start();
