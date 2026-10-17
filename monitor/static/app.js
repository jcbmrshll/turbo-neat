// turbo-neat monitor: run list, live charts, episode and network viewers, and a
// leaderboard of the population with each member's lineage.
// No framework; every view is plain DOM built with h(), and labels that come
// from a run (metric names, config values) only ever go in as text.

import { descentView } from "./descent.js";
import { h } from "./dom.js";
import { NdArray, loadArrays, episodePlayer, traceNetwork } from "./episodes.js";
import { BLUE, ORANGE, WELL } from "./palette.js";
import { RENDERERS } from "./renderers.js";

const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"].map((v) => `var(${v})`);
const POLL_MS = 2000;
const RUNS_POLL_MS = 5000;
// a running run that hasn't logged for this long is probably dead
const STALE_S = 180;
// the stat tiles, in order, when the run logs them; "max_*" stands for the best
// score of each of the run's tests, e.g. max_fitness_against_baseline, and
// fitness_against_baseline is what runs logged before tests named their metrics
const HEADLINE = [
  "max_fitness",
  "mean_fitness",
  "max_*",
  "fitness_against_baseline",
  "num_species",
  "mean_condensed_hidden_nodes",
  "mean_condensed_connections",
];
// charts that lead the grid; the rest follow in the order they were first logged
const CHART_ORDER = ["fitness", "fitness_against_baseline"];
const BOARD_LIMIT = 50;
// the windows offered over the metric charts, in epochs
const CHART_WINDOWS = [50, 200, 1000];

const state = {
  runs: [],
  runId: null,
  run: null,
  rows: [],
  metricsNext: 0,
  media: [],
  mediaNext: 0,
  // the epoch the run tab and the leaderboard show, or null to follow the latest
  epoch: null,
  connected: true,
  // the leaderboard panel
  board: null,
  boardScope: "alive",
  // the species the leaderboard shows, or null for all of them
  boardSpecies: null,
  speciesNames: {},
  // the open member tabs, in order (see openTab), and the one shown: a genome id,
  // or null for the run's own tab
  tabs: [],
  activeTab: null,
  // the run tab's charts are drawn to its width, so they wait while it's hidden
  overviewStale: false,
  // the metric charts show the last this-many epochs, or all of them if
  // null; kept across runs and reloads
  chartWindow: Number(localStorage.getItem("chartWindow")) || null,
};

// ---------------------------------------------------------------- helpers

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

// "41 / 300 epochs" (epochs done) when the run said how long it will be
function progress(run) {
  if (run.step == null) return "";
  const total = run.num_generations;
  return total ? `${run.step + 1} / ${total} epochs` : `epoch ${run.step}`;
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
  const children = [h("div", { class: "stitle side-title", text: "Runs" })];
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
          h("span", { class: "when", text: [run.mean_fitness != null && `fitness ${fmt(run.mean_fitness)}`, ago(run.updated)].filter(Boolean).join(" · ") }),
        ),
      );
    }
  }
  if (!state.runs.length) children.push(h("div", { class: "side-hdr", text: "no runs" }));
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
      h("p", { text: "It will show up here as soon as it logs its first epoch." }),
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
    epoch: null,
    board: null,
    boardSpecies: null,
    speciesNames: {},
    tabs: [],
    activeTab: null,
    overviewStale: false,
    overviewScroll: 0,
  });
  renderSide();
  document.body.classList.remove("has-board");
  document.getElementById("board").replaceChildren();
  view = {
    head: h("div", { class: "runhead" }),
    tabbar: h("nav", { class: "tabs" }),
    epochBar: epochBar(),
    overview: h("div", { class: "tab-panel" }),
    panels: h("div"),
    tiles: h("div", { class: "tiles" }),
    mediaTitle: h("div", { class: "stitle", text: "Media" }),
    media: h("div", { class: "media-grid" }),
    chartsTitle: h("div", { class: "stitle-row" }),
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
  renderChartsTitle();
  const top = h("div", { class: "topbar" }, view.tabbar, view.epochBar.el);
  main.replaceChildren(h("div", { class: "page" }, view.head, top, view.overview, view.panels));
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
      view.epochBar.update();
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

// the latest value of a metric, as of the epoch shown
function latest(key) {
  for (let i = state.rows.length - 1; i >= 0; i--) {
    if (state.epoch != null && state.rows[i].step > state.epoch) continue;
    const v = state.rows[i][key];
    if (v != null) return v;
  }
  return null;
}

function renderTiles() {
  const tests = metricKeys().filter((k) => k.startsWith("max_") && !HEADLINE.includes(k));
  const keys = HEADLINE.flatMap((k) => (k === "max_*" ? tests : [k]));
  const tiles = keys.filter((k) => latest(k) != null).map((k) =>
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

// ---------------------------------------------------------------- epoch

// each chart's epoch line, to move without redrawing the chart
const chartMarkers = [];

// show the run as it was at an epoch (null: the latest, following new ones)
function setEpoch(epoch) {
  const last = state.rows.at(-1)?.step;
  state.epoch = epoch == null || epoch >= last ? null : epoch;
  view.epochBar.update();
  renderTiles();
  chartMarkers.forEach((place) => place());
  renderMedia();
  clearTimeout(setEpoch.board);
  // dragging the slider asks for the leaderboard once it settles
  setEpoch.board = setTimeout(pollBoard, 120);
}

// the slider across the top that picks the epoch shown, with a box to type one in;
// made once per run, and updated in place so a drag isn't interrupted
function epochBar() {
  const range = h("input", { type: "range", min: 0, max: 0, value: 0, step: 1 });
  const box = h("input", { type: "number", class: "epoch-box", min: 0, step: 1, title: "type an epoch" });
  const of = h("span", { class: "dim" });
  const latest = h("button", { text: "latest", title: "back to the current epoch, following new ones", onclick: () => setEpoch(null) });
  range.addEventListener("input", () => setEpoch(Number(range.value)));
  // typed epochs apply on enter or leaving the box, clamped to the run's
  box.addEventListener("change", () => {
    const first = state.rows[0]?.step ?? 0;
    if (box.value !== "") setEpoch(Math.max(first, Math.round(Number(box.value))));
    else update();
  });
  box.addEventListener("keydown", (e) => e.key === "Enter" && box.blur());
  const el = h("div", { class: "epoch-bar scrub" }, h("span", { class: "dim", text: "epoch" }), range, box, of, latest);
  function update() {
    const first = state.rows[0]?.step ?? 0, last = state.rows.at(-1)?.step ?? 0;
    el.hidden = !state.rows.length;
    range.min = box.min = first;
    range.max = box.max = last;
    range.value = state.epoch ?? last;
    // not while it's being typed in
    if (document.activeElement !== box) box.value = state.epoch ?? last;
    of.textContent = `of ${last}`;
    latest.disabled = state.epoch == null;
    el.classList.toggle("pinned", state.epoch != null);
  }
  return { el, update };
}

// ---------------------------------------------------------------- charts

// Group metric keys into charts: "a/b" keys chart together under "a",
// max_/mean_/min_ variants of one quantity chart together under its name, and a
// condensed quantity (e.g. mean_condensed_connections, only what's on a path from
// the inputs to the outputs) charts with the whole of it (mean_connections).
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
    const whole = key.replace(/^(max|mean|min)_condensed_/, "$1_");
    if (slash > 0) {
      add(key.slice(0, slash), key.slice(slash + 1), key);
    } else if (whole !== key && set.has(whole)) {
      add(whole, "condensed", key);
    } else if (stat && set.has(`${stat[1]}_condensed_${stat[2]}`)) {
      add(key, "all", key);
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

// whether the run has only ever had one species, which leaves the species charts
// and map with nothing to compare
function singleSpecies() {
  const ids = new Set();
  for (const k of metricKeys()) {
    const m = k.match(/^species_[^/]+\/s(\d+)$/);
    if (m) ids.add(m[1]);
  }
  return ids.size <= 1;
}

function renderCharts() {
  chartMarkers.length = 0;
  const oneSpecies = singleSpecies();
  const groups = groupMetrics(metricKeys()).filter((g) => !(oneSpecies && g.name.startsWith("species_")));
  view.chartsTitle.hidden = !groups.length;
  // the window ends at the newest generation logged
  const last = state.rows.at(-1)?.step;
  const from = state.chartWindow && last != null ? last - state.chartWindow + 1 : -Infinity;
  const media = CHART_MEDIA.map((k) => view.mediaCards[k]?.el).filter(Boolean);
  view.charts.replaceChildren(...groups.map((g) => chartCard(g, from)), ...media);
}

function setChartWindow(n) {
  state.chartWindow = n > 0 ? Math.floor(n) : null;
  if (state.chartWindow) localStorage.setItem("chartWindow", state.chartWindow);
  else localStorage.removeItem("chartWindow");
  renderChartsTitle();
  if (state.rows.length) renderCharts();
}

// "Metrics", and the window of epochs the charts show: all, a preset, or any N
function renderChartsTitle() {
  const win = state.chartWindow;
  const toggle = (n, text, title) =>
    h("button", { class: win === n ? "on" : null, text, title, onclick: () => setChartWindow(n) });
  const custom = h("input", {
    type: "number",
    min: 1,
    step: 1,
    placeholder: "N",
    title: "show the last N epochs",
    value: win && !CHART_WINDOWS.includes(win) ? win : null,
    class: win && !CHART_WINDOWS.includes(win) ? "on" : null,
    onchange: (e) => setChartWindow(Number(e.target.value)),
    onkeydown: (e) => e.key === "Enter" && e.target.blur(),
  });
  view.chartsTitle.replaceChildren(
    h("div", { class: "stitle", text: "Metrics" }),
    h("span", { class: "spacer" }),
    h(
      "span",
      { class: "scrub" },
      h("span", { class: "dim", text: "last" }),
      toggle(null, "all", "every epoch"),
      CHART_WINDOWS.map((n) => toggle(n, fmt(n), `the last ${n} epochs`)),
      custom,
      h("span", { class: "dim", text: "epochs" }),
    ),
  );
}

function chartCard(group, from) {
  const series = group.series.map((s, i) => {
    // species_*/s<id> series are named and coloured after their species
    const species = group.name.startsWith("species_") && s.name.match(/^s(\d+)$/);
    return {
      ...s,
      name: species ? state.speciesNames[species[1]] || s.name : s.name,
      color: species ? speciesColor(Number(species[1])) : SERIES[i % SERIES.length],
      points: state.rows.filter((r) => r[s.key] != null && r.step >= from).map((r) => [r.step, r[s.key]]),
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

  // the epoch shown, when it isn't the latest
  const mark = h("line", { class: "epoch-mark", y1: pad.t, y2: H - pad.b });
  const placeMark = () => {
    const on = state.epoch != null && state.epoch >= x0 && state.epoch <= x1;
    mark.setAttribute("visibility", on ? "visible" : "hidden");
    if (on) {
      mark.setAttribute("x1", sx(state.epoch));
      mark.setAttribute("x2", sx(state.epoch));
    }
  };
  placeMark();
  chartMarkers.push(placeMark);
  svg.append(mark);

  // hover: a crosshair snapped to the nearest logged step, every series in one tooltip;
  // a click shows that epoch
  const steps = [...new Set(xs)].sort((a, b) => a - b);
  const lookup = series.map((s) => new Map(s.points));
  const cross = h("line", { class: "cross", y1: pad.t, y2: H - pad.b, visibility: "hidden" });
  const dots = series.map((s) => h("circle", { class: "hover-dot", r: 4, fill: s.color, visibility: "hidden" }));
  svg.append(cross, ...dots);
  const nearest = (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const target = x0 + ((px - pad.l) / (W - pad.l - pad.r)) * (x1 - x0);
    let best = steps[0];
    for (const st of steps) if (Math.abs(st - target) < Math.abs(best - target)) best = st;
    return best;
  };
  svg.addEventListener("click", (e) => setEpoch(nearest(e)));
  svg.addEventListener("pointermove", (e) => {
    const best = nearest(e);
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
      h("div", { class: "head", text: `epoch ${best}` }),
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
  // the species map is drawn to fit; leave images alone so gifs don't restart
  for (const card of Object.values(view.mediaCards)) {
    if (card.shown?.endsWith(".json")) card.shown = null;
  }
  renderMedia();
}

// ---------------------------------------------------------------- media

const MEDIA_TITLES = { episode: "sample episode", species_pca: "species map" };
const MEDIA_ORDER = ["episode"];
// runs from before the test's network was dropped logged it; members' networks are
// on their own tabs
const HIDDEN_MEDIA = new Set(["network"]);
// media drawn as charts, which sit at the end of the chart grid rather than with
// the episode
const CHART_MEDIA = ["species_pca"];
// the side the policy plays in a test game against an environment's built-in
// baseline: evojax's single-player slimevolley gives it the right
const BASELINE_SEAT = { slimevolley: 1 };
// races against a field of bots: the tested car is drawn in the first seat's
// colours wherever on the grid it started, and the bots each in another team's
const BOT_RACES = new Set(["f1tenth"]);
// races between the generation's fittest, in fitness order on the grid
const FIELD_RACES = new Set(["f1tenth_field"]);

function mediaByKey() {
  const byKey = new Map();
  const oneSpecies = singleSpecies();
  for (const m of state.media) {
    if (HIDDEN_MEDIA.has(m.key) || (oneSpecies && m.key === "species_pca")) continue;
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
  view.mediaTitle.hidden = !groups.some(([key]) => !CHART_MEDIA.includes(key));
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
      card.el = h("div", { class: "card media-card" }, card.hdr, card.body, card.who, card.foot);
      view.mediaCards[key] = card;
      if (CHART_MEDIA.includes(key)) view.charts.append(card.el);
      else view.media.append(card.el);
    }
    updateMediaCard(key, entries, card);
  }
}

// a card shows its media from the epoch shown, or the last epoch before it that has
// some (tests log their episode only every so often)
function updateMediaCard(key, entries, card) {
  let entry = entries[0];
  for (const e of entries) if (state.epoch == null || e.step <= state.epoch) entry = e;
  card.hdr.replaceChildren(
    h("span", { class: "title", text: MEDIA_TITLES[key] || key.replaceAll("_", " ") }),
    h("span", { class: "spacer" }),
    h("span", { class: "latest", text: `epoch ${entry.step}` }),
  );
  card.who.replaceChildren(...playerLine(key, entry));
  // swapping the element restarts a gif or a player, so only do it when the entry
  // changes
  const id = entry.file || entry.episode || `error-${entry.step}`;
  if (card.shown === id) return;
  card.shown = id;
  card.foot.replaceChildren();
  if (entry.error) {
    card.body.replaceChildren(h("div", { class: "media-error", text: `couldn't render: ${entry.error}` }));
    return;
  }
  if (entry.episode) {
    showEpisode(card, entry);
    return;
  }
  const src = `/media/${state.runId}/${encodeURIComponent(entry.file)}`;
  if (entry.file.endsWith(".json")) {
    card.body.replaceChildren();
    fetch(src)
      .then((r) => r.json())
      .then((data) => {
        if (card.shown !== entry.file) return;
        if (data.type === "species_pca") {
          const { svg, summary } = speciesMapView(data, card.body.clientWidth - 22);
          card.body.replaceChildren(svg);
          card.foot.replaceChildren(h("div", { class: "net-foot", text: summary }));
        } else {
          card.body.replaceChildren(h("pre", { text: JSON.stringify(data, null, 2) }));
        }
      });
  } else if (entry.file.endsWith(".mp4")) {
    card.body.replaceChildren(h("video", { src, autoplay: true, loop: true, muted: true, controls: true }));
  } else {
    card.body.replaceChildren(h("img", { src, alt: `${key} at epoch ${entry.step}` }));
  }
}

// a test's episode, played beside the network of the agent picked in it (the
// first one with observations, or the one clicked), lit up as it plays
async function showEpisode(card, entry) {
  const render = RENDERERS[entry.env];
  let arrays;
  try {
    if (!render) throw new Error(`no renderer for ${entry.env}`);
    arrays = await loadArrays(`/api/runs/${state.runId}/episodes/${encodeURIComponent(entry.episode)}`);
  } catch (e) {
    if (card.shown === entry.episode) card.body.replaceChildren(h("div", { class: "media-error", text: `couldn't show this episode: ${e.message}` }));
    return;
  }
  if (card.shown !== entry.episode) return;
  const rows = obsRows(arrays, entry.players || []).filter((r) => r.id != null);
  const player = episodePlayer(render(arrays), { seat: rows[0]?.seat ?? null });
  if (!rows.length) {
    card.body.replaceChildren(player.el);
    return;
  }
  const net = { body: h("div", { class: "ep-net" }), foot: h("div") };
  card.body.replaceChildren(h("div", { class: "ep-grid" }, player.el, h("div", {}, net.body, net.foot)));
  const pick = (seat) => {
    const row = rows.find((r) => r.seat === seat);
    if (!row) return false;
    player.setSeat(seat);
    fetchNetwork(row.id, entry.step).then((network) => {
      if (card.shown !== entry.episode || player.seat !== seat) return;
      if (network) drawNetwork(net, network, { arrays, row, player });
      else net.body.replaceChildren(h("div", { class: "media-error", text: "no network logged for it" }));
    });
    return true;
  };
  player.onSeat = pick;
  requestAnimationFrame(() => pick(rows[0].seat));
}

// a genome's network in an epoch, kept once fetched; null if it wasn't logged
async function fetchNetwork(id, step) {
  const key = `${state.runId}/${id}@${step}`;
  if (!(key in networkCache)) {
    try {
      networkCache[key] = await api(`/api/runs/${state.runId}/individuals/${id}/network?step=${step}`);
    } catch {
      return null;
    }
  }
  return networkCache[key];
}
const networkCache = {};

// the agents of an episode whose observations it recorded, so whose networks can
// be rerun: one per row of arrays.obs, {row, seat, id}, in the seat
// arrays.obs_seats gives it (its row by default), played by the genome in that
// seat, or by the one genome playing every row (a flock), from players (in seat
// order, ids or {id})
function obsRows(arrays, players) {
  if (!arrays.obs) return [];
  const ids = players.map((p) => (typeof p === "object" ? p.id : p));
  const count = arrays.obs.shape.length > 2 ? arrays.obs.shape[1] : 1;
  const seats = arrays.obs_seats ? [...arrays.obs_seats.data] : [...Array(count).keys()];
  return seats.map((seat, row) => ({ row, seat, id: ids.length === count ? ids[row] : ids.length === 1 ? ids[0] : null }));
}

// the episode's observations in one row, (frames, inputs)
function obsOf(arrays, row) {
  const obs = arrays.obs;
  if (obs.shape.length === 2) return obs;
  const [frames, , inputs] = obs.shape;
  const data = new Float32Array(frames * inputs);
  for (let t = 0; t < frames; t++) data.set(obs.at(t).at(row).data, t * inputs);
  return new NdArray(data, [frames, inputs]);
}

// the episode shown is of the genomes the run's test picked; name them, and in a
// game against a built-in baseline or bots, who played which side
function playerLine(key, entry) {
  if (!MEDIA_ORDER.includes(key)) return [];
  const players = entry.players || [];
  if (key === "episode" && FIELD_RACES.has(entry.env)) {
    // the field on the grid, fittest first, each linking to the member
    if (players.length) {
      const who = (p) => (p.name ? genomeLink(p.id, p.name, { step: entry.step }) : `#${p.id}`);
      return [matchup(players.map((p, seat) => seatPlayer(who(p), seat, entry.env, p.reward)))];
    }
    const text = "the epoch's fittest racing each other, P1 the fittest";
    return [h("div", { class: "ep-games" }, h("span", { class: "ep-player" }, h("span", { class: "dim", text })))];
  }
  // runs from before tests named their genomes have none
  const p = players[0];
  if (!p) return [];
  const name = h("b", {}, p.name ? genomeLink(p.id, p.name) : `#${p.id}`);
  if (key === "episode" && BOT_RACES.has(entry.env)) {
    const score = p.reward == null ? [] : [h("span", { class: "seat", text: `${p.reward.toFixed(1)}${REWARD_UNITS[entry.env] || ""}` })];
    const policy = h("span", { class: "ep-player" }, h("i", { class: "swatch", style: seatOf(entry.env, 0).style }), name, ...score);
    return [matchup([policy, h("span", { class: "ep-player" }, h("span", { text: "bots" }))])];
  }
  const seat = key === "episode" ? BASELINE_SEAT[entry.env] : undefined;
  if (seat === undefined) {
    return [h("div", { class: "ep-games" }, h("span", { class: "ep-player" }, swatch(p.birth_species), name))];
  }
  const sides = [seatPlayer(name, seat, entry.env), seatPlayer(h("span", { text: "baseline" }), 1 - seat, entry.env)];
  if (seat === 1) sides.reverse();
  return [matchup(sides)];
}

// ---------------------------------------------------------------- network

// net drawn to width; with a trace (episodes.js traceNetwork), also show(frame),
// which lights it up with what it did at that frame: each node coloured by the
// sign of its output, as strong as the output is against its largest over the
// episode, each edge likewise by what it carried (its source's output times its
// weight), and the inputs' and outputs' values beside their labels
function networkView(net, width, trace = null) {
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
  // room for a value (" -0.42") after each label, when there are values
  const valueChars = trace ? 6 : 0;
  const labelW = (list) => (Math.max(0, ...list.map((n) => (n.label || "").length)) + valueChars) * 6.2 + 12;
  const padL = Math.min(160 + valueChars * 6.2, labelW(inputs));
  const padR = Math.min(160 + valueChars * 6.2, labelW(outputs));
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
    el.edge = e;
    edgeEls.push(el);
    svg.append(el);
  }
  // each node's column in the trace, and its range over the episode
  const col = new Map(net.nodes.map((n, i) => [n.id, i]));
  const range = trace && traceRanges(trace);
  let frame = 0;
  const valueAt = (id) => trace.values[frame * trace.nodes + col.get(id)];
  const valueEls = [];
  const nodeEls = [];
  for (const n of nodes.values()) {
    const node = h("circle", { class: `node ${n.kind}${trace ? " lit" : ""}`, cx: n.x, cy: n.y, r: n.kind === "hidden" ? 4.5 : 5 });
    node.id_ = n.id;
    nodeEls.push(node);
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
        trace && h("div", { class: "row" }, "output ", h("b", { text: fmt(valueAt(n.id)) })),
        trace && h("div", { class: "row" }, "over the episode ", h("b", { text: `${fmt(range.lo[col.get(n.id)])} to ${fmt(range.hi[col.get(n.id)])}` })),
      );
    });
    node.addEventListener("pointerleave", () => {
      edgeEls.forEach((el) => el.classList.remove("dim"));
      hideTip();
    });
    svg.append(node);
    if (n.kind === "input" || n.kind === "output") {
      const value = trace ? h("tspan", { class: "val" }) : null;
      const input = n.kind === "input";
      // inputs' labels end at their node, and outputs' start at theirs, so the
      // values go on the side by the node
      const parts = input ? [n.label, value && " ", value] : [value, value && " ", n.label];
      svg.append(h("text", { class: "lab", x: n.x + (input ? -9 : 9), y: n.y + 3.5, "text-anchor": input ? "end" : null }, ...parts));
      if (value) valueEls.push([value, n.id]);
    }
  }
  const summary = `${hidden.length} hidden · ${net.edges.length} connections`;
  if (!trace) return { svg, summary, show: null };

  const show = (t) => {
    frame = Math.min(t, trace.frames - 1);
    for (const node of nodeEls) {
      const i = col.get(node.id_);
      node.style.fill = activityColor(valueAt(node.id_), range.lo[i], range.hi[i]);
    }
    for (const el of edgeEls) {
      const e = el.edge;
      const carried = valueAt(e.from) * e.weight;
      const i = col.get(e.from);
      // an edge carries the most at its source's extreme
      const most = Math.max(Math.abs(range.lo[i]), Math.abs(range.hi[i])) * Math.abs(e.weight);
      el.setAttribute("stroke", carried >= 0 ? "var(--s1)" : "var(--s2)");
      el.style.opacity = most > 0 ? (0.06 + (0.84 * Math.abs(carried)) / most).toFixed(3) : 0.06;
    }
    for (const [el, id] of valueEls) el.textContent = fmtValue(valueAt(id));
  };
  return { svg, summary, show };
}

// each node's least and most output over a trace, and the most each edge carried
function traceRanges(trace) {
  const lo = new Float32Array(trace.nodes).fill(Infinity), hi = new Float32Array(trace.nodes).fill(-Infinity);
  for (let t = 0; t < trace.frames; t++) {
    for (let i = 0; i < trace.nodes; i++) {
      const v = trace.values[t * trace.nodes + i];
      if (v < lo[i]) lo[i] = v;
      if (v > hi[i]) hi[i] = v;
    }
  }
  return { lo, hi };
}

// a node's colour for an output v, of outputs from lo to hi: blue for positive,
// orange for negative, fading to the background at zero
const RGB = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const [POS_RGB, NEG_RGB, ZERO_RGB] = [BLUE, ORANGE, WELL].map(RGB);
function activityColor(v, lo, hi) {
  const most = Math.max(Math.abs(lo), Math.abs(hi));
  const u = most > 0 && Number.isFinite(v) ? Math.max(-1, Math.min(1, v / most)) : 0;
  const to = u >= 0 ? POS_RGB : NEG_RGB;
  const mix = ZERO_RGB.map((z, k) => Math.round(z + (to[k] - z) * Math.abs(u)));
  return `rgb(${mix})`;
}

// a value in a fixed width, so labels beside it hold still as it changes
function fmtValue(v) {
  if (!Number.isFinite(v)) return "  –  ";
  const a = Math.abs(v);
  const text = a >= 999.5 ? fmt(v) : a >= 99.95 ? v.toFixed(0) : a >= 9.995 ? v.toFixed(1) : v.toFixed(2);
  return (v < 0 ? "" : "+") + text;
}

// ---------------------------------------------------------------- species map

// how many of the biggest species get a name beside their dot; the rest on hover
const SPECIES_MAP_LABELS = 8;

// each species' mean parameters on the first two principal components across
// species: a dot per species, sized by its members, both axes to one scale so
// distances on the map are distances between species
function speciesMapView(data, width) {
  const W = Math.max(width, 200), H = 300;
  const pad = { l: 34, r: 14, t: 14, b: 22 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const svg = h("svg", { class: "chart species-map", viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  const species = data.species.filter((s) => s.x != null && s.y != null);
  const [ev1, ev2] = data.explained_variance.map((v) => `${Math.round((v ?? 0) * 100)}%`);
  const summary = `${species.length} species · PC1 ${ev1} · PC2 ${ev2} of variance`;
  if (!species.length) return { svg, summary };

  const maxSize = Math.max(...species.map((s) => s.size));
  const radius = (s) => 4 + 8 * Math.sqrt(s.size / maxSize);
  // the same units per pixel on both axes, with room for the biggest dot
  const xs = species.map((s) => s.x), ys = species.map((s) => s.y);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const span = Math.max((Math.max(...xs) - Math.min(...xs)) / (pw - 28), (Math.max(...ys) - Math.min(...ys)) / (ph - 28)) || 1 / Math.min(pw, ph);
  const x0 = cx - (pw / 2) * span, x1 = cx + (pw / 2) * span;
  const y0 = cy - (ph / 2) * span, y1 = cy + (ph / 2) * span;
  const sx = (x) => pad.l + ((x - x0) / (x1 - x0)) * pw;
  const sy = (y) => pad.t + (1 - (y - y0) / (y1 - y0)) * ph;

  const step = (lo, hi, count) => {
    const t = niceTicks(lo, hi, count);
    return t.filter((v) => v >= lo && v <= hi);
  };
  for (const t of step(y0, y1, Math.max(2, Math.floor(ph / 60)))) {
    svg.append(
      h("line", { class: t === 0 ? "axis" : "grid", x1: pad.l, x2: W - pad.r, y1: sy(t), y2: sy(t) }),
      h("text", { class: "tick", x: pad.l - 6, y: sy(t) + 3, "text-anchor": "end", text: fmtTick(t) }),
    );
  }
  for (const t of step(x0, x1, Math.max(2, Math.floor(pw / 80)))) {
    svg.append(
      h("line", { class: t === 0 ? "axis" : "grid", x1: sx(t), x2: sx(t), y1: pad.t, y2: H - pad.b }),
      h("text", { class: "tick", x: sx(t), y: H - 7, "text-anchor": "middle", text: fmtTick(t) }),
    );
  }
  svg.append(
    h("text", { class: "tick axis-name", x: W - pad.r, y: H - pad.b - 5, "text-anchor": "end", text: `PC1 ${ev1}` }),
    h("text", { class: "tick axis-name", x: pad.l + 5, y: pad.t + 9, text: `PC2 ${ev2}` }),
  );

  const name = (s) => state.speciesNames[s.id] ?? `s${s.id}`;
  const labelled = new Set([...species].sort((a, b) => b.size - a.size).slice(0, SPECIES_MAP_LABELS).map((s) => s.id));
  // the biggest underneath, so no dot hides a smaller one
  for (const s of [...species].sort((a, b) => b.size - a.size)) {
    const r = radius(s);
    const dot = h(
      "g",
      { class: "sp-dot" },
      h("circle", { class: "hit", cx: sx(s.x), cy: sy(s.y), r: r + 6 }),
      h("circle", { class: "dot", cx: sx(s.x), cy: sy(s.y), r, fill: speciesColor(s.id) }),
    );
    dot.addEventListener("pointermove", (e) =>
      showTip(
        e.clientX,
        e.clientY,
        h("div", { class: "head" }, swatch(s.id), ` ${name(s)}`),
        h("div", { class: "row" }, h("b", { text: s.size }), h("span", { text: s.size === 1 ? "member" : "members" })),
        h("div", { class: "row" }, h("b", { text: fmt(s.fitness) }), h("span", { text: "mean fitness" })),
        h("div", { class: "row" }, h("span", { text: `(${fmt(s.x)}, ${fmt(s.y)})` })),
      ),
    );
    dot.addEventListener("pointerleave", hideTip);
    svg.append(dot);
  }
  for (const s of species.filter((s) => labelled.has(s.id))) {
    const right = sx(s.x) < W - 80;
    svg.append(
      h("text", {
        class: "sp-lab",
        x: sx(s.x) + (right ? 1 : -1) * (radius(s) + 4),
        y: sy(s.y) + 3,
        "text-anchor": right ? "start" : "end",
        text: name(s),
      }),
    );
  }
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
  let q = `scope=${state.boardScope}&limit=${BOARD_LIMIT}`;
  if (state.boardSpecies != null) q += `&species=${state.boardSpecies}`;
  if (state.epoch != null) q += `&epoch=${state.epoch}`;
  try {
    const board = await api(`/api/runs/${runId}/leaderboard?${q}`);
    if (runId !== state.runId) return;
    const renamed = JSON.stringify(board.species_names) !== JSON.stringify(state.speciesNames);
    state.board = board;
    state.speciesNames = board.species_names;
    renderBoard();
    // the species charts and the species map are labelled with these names
    if (renamed && state.rows.length) {
      if (state.activeTab == null) relayoutOverview();
      else state.overviewStale = true;
    }
  } catch {
    // the run's charts carry on without it
  }
}

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
  const allTime = state.boardScope === "all";
  const toggle = (on, text, title, onclick) => h("button", { class: on ? "on" : null, text, title, onclick });
  const scopes = h(
    "span",
    { class: "scrub" },
    [["alive", "this epoch"], ["all", "all time"]].map(([value, text]) =>
      toggle(state.boardScope === value, text, null, () => {
        state.boardScope = value;
        pollBoard();
      }),
    ),
  );
  // the species there are, biggest first; one picked that has since died out stays
  // in the list, with no members
  const choices = [...board.species];
  const picked = state.boardSpecies;
  if (picked != null && !choices.some((s) => s.id === picked)) {
    choices.push({ id: picked, name: state.speciesNames[picked] ?? `s${picked}`, members: 0 });
  }
  const filter = h(
    "span",
    { class: "scrub" },
    picked != null && swatch(picked),
    h(
      "select",
      {
        class: picked != null ? "on" : null,
        title: allTime ? "the species each was in when last seen" : "the species each is in",
        onchange: (e) => {
          state.boardSpecies = e.target.value === "" ? null : Number(e.target.value);
          pollBoard();
        },
      },
      h("option", { value: "", text: `all species (${board.species.length})` }),
      choices.map((s) => h("option", { value: s.id, text: `${s.name} · ${s.members}`, selected: s.id === picked })),
    ),
  );
  const head = h(
    "tr",
    {},
    h("th", { class: "rank", text: "#" }),
    h("th", { text: "individual" }),
    h("th", { class: "num", text: "fitness", title: "fitness in its latest epoch" }),
    allTime && h("th", { class: "num", text: "generation", title: "the generation it was born into: the epoch it joined the population" }),
    h("th", { class: "num", text: "epochs", title: "epochs in the population" }),
    h("th", { class: "num", text: "offspring" }),
  );
  const open = new Set(state.tabs.map((t) => t.id));
  const rows = board.rows.map((r, i) =>
    h(
      "tr",
      {
        class: ["row", r.id === state.activeTab && "sel", open.has(r.id) && "open", !r.alive && "gone"].filter(Boolean).join(" "),
        title: `${r.name} · ${r.species_name} · generation ${r.born}, ${r.alive ? `alive since epoch ${r.born}` : `alive epochs ${r.born}–${r.last}`}`,
        onclick: () => openTab(r.id, { name: r.name }),
      },
      h("td", { class: "rank", text: i + 1 }),
      h("td", { class: "who" }, swatch(r.birth_species), nameEl(r.name)),
      h("td", { class: "num", text: fmt(r.fitness) }),
      allTime && h("td", { class: "num", text: r.born }),
      h("td", { class: "num", text: r.age }),
      h("td", { class: "num", text: r.offspring }),
    ),
  );
  if (!rows.length) {
    const who = state.speciesNames[picked] ?? `s${picked}`;
    rows.push(h("tr", {}, h("td", { class: "dim", colspan: 6, text: `no ${who} ${allTime ? "yet" : "in this epoch"}` })));
  }
  // the table keeps its scroll from one generation to the next
  if (!panel.firstChild) panel.append(h("div", { class: "bp-head" }), h("div", { class: "bp-table" }));
  const [top, wrap] = panel.children;
  // rebuilding the species list while it's open would close it
  const choosing = top.contains(document.activeElement) && document.activeElement.tagName === "SELECT";
  top.replaceChildren(
    h("div", { class: "stitle", text: "Leaderboard" }),
    h("div", { class: "bp-sub", text: `epoch ${board.step} · ${board.population} members · ${board.total.toLocaleString("en")} seen` }),
    choosing ? top.querySelector(".bp-controls") : h("div", { class: "bp-controls" }, scopes, filter),
  );
  const scrolled = wrap.scrollTop;
  wrap.replaceChildren(h("table", { class: "board" }, h("thead", {}, head), h("tbody", {}, rows)));
  wrap.scrollTop = scrolled;
}

// ---------------------------------------------------------------- tabs

// The run's own tab is always there; each member opened gets a tab of its own,
// which keeps its place (the generation and game it shows, how far
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
  const runId = state.runId;
  let data;
  try {
    data = await api(`/api/runs/${runId}/individuals/${tab.id}`);
  } catch (e) {
    if (!String(e.message).startsWith("404")) return;
    data = { missing: true, id: tab.id };
  }
  if (runId !== state.runId || !state.tabs.includes(tab)) return;
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
  const nameOf = (id) => ind.pedigree.find((n) => n.id === id)?.name;
  const parentage =
    parent < 0
      ? h("span", { text: "founder" })
      : h(
          "span",
          {},
          "offspring of ",
          genomeLink(parent, nameOf(parent)),
          mate >= 0 && mate !== parent && [" × ", genomeLink(mate, nameOf(mate))],
        );
  const share = ind.population ? ` (${Math.round((100 * ind.descendants) / ind.population)}%)` : "";
  const tile = (lab, val) => h("div", { class: "tile" }, h("div", { class: "lab", text: lab }), h("div", { class: "val", text: val }));
  // its reward in every episode it played for its fitness, over its life; runs
  // that didn't log them just have its latest fitness
  const ret = ind.returns;
  const fitness = ret
    ? h(
        "div",
        { class: "tile" },
        h("div", {
          class: "lab",
          text: `fitness · ${ret.episodes} episodes`,
          title: `every episode it played for its fitness, over ${ret.epochs} ${ret.epochs === 1 ? "epoch" : "epochs"}`,
        }),
        h(
          "div",
          { class: "stats" },
          ["mean", "median", "max", "min", "stddev"].flatMap((k) => [h("span", { class: "k", text: k }), h("span", { class: "v", text: fmt(ret[k]) })]),
        ),
      )
    : tile("fitness", fmt(ind.fitness));

  const head = h(
    "div",
    { class: "card ind" },
    h(
      "div",
      { class: "ind-head" },
      swatch(ind.birth_species),
      h("h2", {}, nameEl(ind.name)),
      h("span", { class: "gid", text: `#${ind.id}` }),
    ),
    h(
      "div",
      { class: "ind-meta" },
      h("span", { text: ind.species_name }),
      ind.species !== ind.birth_species && h("span", { class: "dim", text: `(born ${state.speciesNames[ind.birth_species] ?? ind.birth_species})` }),
      sep(),
      h("span", { text: `generation ${ind.born}${ind.alive ? ", alive" : ""}` }),
      sep(),
      parentage,
    ),
  );
  const tiles = h(
    "div",
    { class: "tiles" },
    fitness,
    tile(`epochs alive · ${ind.age}`, ind.last > ind.born ? `${ind.born}–${ind.last}` : `${ind.born}`),
    tile("offspring", ind.offspring),
    tile("living descendants", `${ind.descendants}${share}`),
  );

  const pedBody = h("div", { class: "ped-scroll" });
  // it keeps its place counted from the right, where the member is, when the panel
  // changes width (as when the leaderboard comes in)
  let fromRight = 0;
  pedBody.addEventListener("scroll", () => (fromRight = pedBody.scrollWidth - pedBody.clientWidth - pedBody.scrollLeft));
  new ResizeObserver(() => (pedBody.scrollLeft = pedBody.scrollWidth - pedBody.clientWidth - fromRight)).observe(pedBody);
  const pedigree = h(
    "div",
    { class: "card" },
    h("div", { class: "card-hdr" }, h("span", { class: "title", text: "lineage" })),
    pedBody,
    h(
      "div",
      { class: "legend ped-legend" },
      [
        ["line", "line of descent"],
        ["primary", "stronger parent"],
        ["mate", "weaker parent"],
        ["primary self", "self-cross"],
      ].map(([cls, text]) =>
        h("span", { class: "key" }, h("svg", { class: "ped-key", width: 20, height: 8 }, h("path", { class: `edge ${cls}`, d: "M1,4H19" })), text),
      ),
    ),
  );

  // pinned when first shown, so the cards don't change under the reader as
  // epochs go by
  if (tab.step == null) tab.step = ind.last;
  const cards = memberCards(tab);

  const kids = ind.offspring_list;
  const offspring =
    kids.length > 0 &&
    h(
      "div",
      { class: "card" },
      h("div", { class: "card-hdr" }, h("span", { class: "title", text: `offspring · ${ind.offspring}, fittest first` })),
      h(
        "div",
        { class: "chips" },
        kids.map((c) => h("span", { class: `chip${c.alive ? "" : " gone"}` }, swatch(c.birth_species), genomeLink(c.id, c.name), h("span", { class: "dim", text: fmt(c.best) }))),
        ind.offspring > kids.length && h("span", { class: "dim", text: `and ${ind.offspring - kids.length} more` }),
      ),
    );

  const scrolled = main.scrollTop;
  tab.el.replaceChildren(
    ...[
      head,
      tiles,
      h("div", { class: "lineage-grid" }, cards.network.el, cards.episode.el),
      pedigree,
      cards.descent.el,
      offspring,
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

// the member's network and episodes, in the generation tab.step, and its line of
// descent. Made once per tab and kept across redraws, so an episode that's playing
// isn't restarted by every new generation; hidden for runs that didn't log them
function memberCards(tab) {
  if (tab.cards) return tab.cards;
  const card = () => {
    const c = { hdr: h("div", { class: "card-hdr" }), body: h("div", { class: "media-body" }), foot: h("div"), shown: null };
    c.el = h("div", { class: "card" }, c.hdr, c.body, c.foot);
    c.el.hidden = true;
    return c;
  };
  tab.cards = { network: card(), episode: card(), descent: card() };
  return tab.cards;
}

function loadMemberCards(tab) {
  loadNetwork(tab, tab.cards.network);
  loadEpisodes(tab, tab.cards.episode);
  loadDescent(tab, tab.cards.descent);
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

async function loadNetwork(tab, card) {
  const net = await memberData(tab, "network");
  if (net === undefined) return;
  card.el.hidden = !net;
  if (!net) return;
  card.net = net;
  card.hdr.replaceChildren(h("span", { class: "title", text: "network" }));
  requestAnimationFrame(() => linkMember(tab));
}

// the member's network, lit up by what it did at each frame of its episode when
// the episode recorded its observations
function linkMember(tab) {
  const { network, episode } = tab.cards;
  if (!network.net) return;
  const player = episode.player;
  const row = player && episode.rows.find((r) => r.id === tab.id && r.seat === player.seat);
  drawNetwork(network, network.net, row ? { arrays: episode.arrays, row, player } : null);
}

// net drawn into a card, and with `activity` ({arrays, row, player}), lit up by
// what it did with the episode's observations in that row, at the player's frame
function drawNetwork(card, net, activity = null) {
  const trace = activity && traceNetwork(net, obsOf(activity.arrays, activity.row.row));
  const { svg, summary, show } = networkView(net, card.body.clientWidth - 22, trace?.error ? null : trace);
  card.body.replaceChildren(svg);
  const legend = show && h("span", { class: "act-key" }, "−", h("i"), "activation", h("i"), "+");
  card.foot.replaceChildren(h("div", { class: "net-foot" }, h("span", { text: summary }), legend, trace?.error && h("span", { class: "dim", text: trace.error })));
  if (activity) {
    activity.player.onFrame = show || null;
    show?.(activity.player.frame);
  }
}

// which seat each player of a game had, in the renderers' colours: the sides of a
// two-player game, or a race's places on the grid, each car a parody of a 2026 team
// in its livery (f1tenth.js LIVERIES), with an edge so the white cars show
const LIVERIES = [
  ["Mclando", "#ff8000"],
  ["Vercedes", "#b9bec4"],
  ["Red Cow", "#1e2d6e"],
  ["Berrari", "#dc0000"],
  ["Billiams", "#0a3282"],
  ["Racing Ducks", "#f0f0f5"],
  ["Ostin Marlin", "#005e46"],
  ["Yaas", "#f0f0f0"],
  ["Audo", "#969ba0"],
  ["Alpone", "#0078dc"],
  ["Chadillac", "linear-gradient(90deg, #e6e6e1 50%, #191919 50%)"],
];
const SEATS = {
  f1tenth_field: (seat) => SEATS.f1tenth(seat),
  f1tenth: (seat) => {
    const [team, color] = LIVERIES[seat % LIVERIES.length];
    return { name: `P${seat + 1} ${team}`, team, style: `background:${color};box-shadow:inset 0 0 0 1px var(--edge)` };
  },
};
const SIDES = [
  { name: "left", style: "background:var(--s1)" },
  { name: "right", style: "background:var(--s2)" },
];
const seatOf = (env, seat) => (SEATS[env] ? SEATS[env](seat) : SIDES[seat]);

// a player of a game, marked with its seat's colour in the picture, and with its
// reward over the game if the run recorded it
function seatPlayer(who, seat, env, reward) {
  const { name, style } = seatOf(env, seat);
  const score = reward == null ? [] : [h("span", { class: "seat", text: `${reward.toFixed(1)}${REWARD_UNITS[env] || ""}` })];
  return h("span", { class: "ep-player" }, h("i", { class: "swatch", style }), who, h("span", { class: "seat", text: name }), ...score);
}
// what the environments' rewards are measured in
const REWARD_UNITS = { f1tenth: " m", f1tenth_field: " m" };

// players side by side, "a vs b vs c"
function matchup(players) {
  const parts = players.flatMap((p, i) => (i ? [h("span", { class: "dim", text: "vs" }), p] : [p]));
  return h("div", { class: "ep-games" }, h("span", { class: "ep-match" }, ...parts));
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
  card.hdr.replaceChildren(h("span", { class: "title", text: "sample episode" }));
  // games of several players: say who played which seat, in the order and colours
  // of the picture; the others open on this same game
  const player = (p, seat) =>
    seatPlayer(
      p.id === tab.id
        ? h("b", {}, nameEl(p.name || `#${p.id}`))
        : p.name
          ? genomeLink(p.id, p.name, { step: eps.step, game: ep.index })
          : `#${p.id}`,
      seat,
      eps.env,
      p.reward,
    );
  const players = ep.players.length > 1 && matchup(ep.players.map(player));
  card.foot.replaceChildren(...[players].filter(Boolean));
  // a new player only when the episode changes, so one that's playing carries on
  const src = `/api/runs/${state.runId}/member_episodes/${eps.step}/${ep.index}`;
  if (card.shown === src) return;
  card.shown = src;
  card.player = null;
  card.body.replaceChildren(h("div", { class: "loading", text: "loading…" }));
  const render = RENDERERS[eps.env];
  let arrays;
  try {
    if (!render) throw new Error(`no renderer for ${eps.env}`);
    arrays = await loadArrays(src);
  } catch (e) {
    if (card.shown === src) card.body.replaceChildren(h("div", { class: "media-error", text: `couldn't show this episode: ${e.message}` }));
    return;
  }
  if (card.shown !== src) return;
  card.arrays = arrays;
  card.rows = obsRows(arrays, ep.players);
  // the member's own agent, the first of them if it plays several (a flock)
  const own = card.rows.find((r) => r.id === tab.id);
  card.player = episodePlayer(render(arrays), { seat: own ? own.seat : ep.seat });
  card.player.onSeat = (seat) => {
    const row = card.rows.find((r) => r.seat === seat);
    if (row?.id === tab.id) {
      card.player.setSeat(seat);
      linkMember(tab);
    } else if (row?.id != null) {
      // another member's agent: its own tab, on this same game
      openTab(row.id, { at: { step: eps.step, game: ep.index } });
    }
    return row?.id != null;
  };
  card.body.replaceChildren(card.player.el);
  linkMember(tab);
}

// The member's line of descent, each ancestor's network blended into the next, and
// for a 2D classification, the decision boundary it drew. A line never changes, so
// it's drawn once per tab
async function loadDescent(tab, card) {
  if (card.shown) return;
  card.shown = true;
  const runId = state.runId;
  let data = null;
  try {
    data = await api(`/api/runs/${runId}/individuals/${tab.id}/lineage_networks`);
  } catch {
    // not logged (or not yet)
  }
  if (runId !== state.runId || !state.tabs.includes(tab)) return;
  if (!data?.networks.length) {
    card.shown = false;
    return;
  }
  // the boundary over the plane the member's episode covers
  let lim = null;
  const eps = await memberData(tab, "episodes");
  if (eps?.env === "classification_2d" && eps.episodes.length) {
    try {
      const { points } = await loadArrays(`/api/runs/${runId}/member_episodes/${eps.step}/${eps.episodes[0].index}`);
      lim = 1.02 * points.data.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    } catch {
      // drawn without it
    }
  }
  if (runId !== state.runId || !state.tabs.includes(tab)) return;
  card.el.hidden = false;
  const info = h("span", { class: "latest" });
  card.hdr.replaceChildren(h("span", { class: "title", text: "lineage evolution" }), h("span", { class: "spacer" }), info);
  // (a tab switched away from while this loaded has no width yet)
  const view = descentView(data, (card.body.clientWidth || 720) - 22, { lim });
  const player = episodePlayer(view, { maxHeight: view.height });
  player.el.classList.add("descent");
  // the ancestor shown, named in the header
  let named = null;
  player.onFrame = (frame) => {
    const { ancestor: a, hidden, connections } = view.at(frame);
    if (a === named) return;
    named = a;
    const who = a.id === tab.id ? h("b", {}, nameEl(a.name)) : genomeLink(a.id, a.name);
    info.replaceChildren(who, ` · generation ${a.born} · fitness ${fmt(a.fitness)} · ${hidden} hidden · ${connections} connections`);
  };
  player.onFrame(player.frame);
  card.body.replaceChildren(player.el);
}

// both tournaments picked the same parent, so its offspring is a mutated copy of it
function selfCross(n) {
  return n.parents[0] >= 0 && n.parents[0] === n.parents[1];
}

// the line of descent from its founder, straight across the middle, and the
// ancestors near it, laid out left to right by the generation they were born in
function pedigreeView(ind, width) {
  const nodes = new Map(ind.pedigree.map((n) => [n.id, { ...n }]));
  const kids = new Map();
  for (const n of nodes.values()) {
    for (const p of new Set(n.parents)) {
      if (!nodes.has(p)) continue;
      if (!kids.has(p)) kids.set(p, []);
      kids.get(p).push(n);
    }
  }
  const steps = [...new Set([...nodes.values()].map((n) => n.born))].sort((a, b) => a - b);
  // LANE: from the line to the rows either side of it, room for the line's names
  const COL = 96, GAP = 24, LANE = 34, PADX = 72, PADY = 28, AXIS = 16;
  const W = Math.max(width, PADX * 2 + (steps.length - 1) * COL);
  const colX = (i) => (steps.length === 1 ? W / 2 : PADX + (i * (W - 2 * PADX)) / (steps.length - 1));

  // newest first, so each column can line up behind the offspring it already placed
  for (let i = steps.length - 1; i >= 0; i--) {
    const col = [...nodes.values()].filter((n) => n.born === steps[i]);
    for (const n of col) {
      const ys = (kids.get(n.id) || []).map((k) => k.y);
      n.bary = ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 0;
    }
    // the line keeps its row in every column; the rest are split evenly above and
    // below it, in the order of their offspring's heights
    col.sort((a, b) => a.bary - b.bary || a.id - b.id);
    const rest = col.filter((n) => !n.on_line);
    const above = rest.slice(0, rest.length >> 1), below = rest.slice(rest.length >> 1);
    for (const n of col) if (n.on_line) n.y = 0;
    above.forEach((n, j) => (n.y = -LANE - (above.length - 1 - j) * GAP));
    below.forEach((n, j) => (n.y = LANE + j * GAP));
    for (const n of col) {
      n.x = colX(i);
      n.col = i;
    }
  }
  const ys = [...nodes.values()].map((n) => n.y);
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const H = hi - lo + 2 * PADY + AXIS;
  for (const n of nodes.values()) n.y += PADY - lo;

  const svg = h("svg", { class: "ped", viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  steps.forEach((st, i) => svg.append(h("text", { class: "gen", x: colX(i), y: H - 4, "text-anchor": "middle", text: st })));
  svg.append(h("text", { class: "gen axis-name", x: W - 6, y: H - 4, "text-anchor": "end", text: "generation" }));

  const edgeEls = [];
  for (const n of nodes.values()) {
    [...new Set(n.parents)].forEach((pid) => {
      const p = nodes.get(pid);
      if (!p) return;
      const fitter = pid === n.parents[0];
      const cls = fitter && p.on_line && n.on_line ? "line" : fitter ? "primary" : "mate";
      const mx = (p.x + n.x) / 2;
      // crossed with itself: its one parent's edge in green
      const self = selfCross(n) ? " self" : "";
      const el = h("path", { class: `edge ${cls}${self}`, d: `M${p.x},${p.y} C${mx},${p.y} ${mx},${n.y} ${n.x},${n.y}` });
      el.dataset.from = pid;
      el.dataset.to = n.id;
      edgeEls.push(el);
      svg.append(el);
    });
  }
  for (const n of nodes.values()) {
    const subject = n.id === ind.id;
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
        h("div", { class: "row" }, `#${n.id} · ${n.species_name} · generation ${n.born} · ${n.last > n.born ? `epochs ${n.born}–${n.last}` : `epoch ${n.born}`}`),
        h("div", { class: "row" }, "best ", h("b", { text: fmt(n.best) }), ` · ${n.offspring} offspring`),
        selfCross(n) && h("div", { class: "row" }, "self-cross"),
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
