// turbo-neat monitor: the run list, and the run picked in the dashboard
// (dashboard.js), read live from the monitor's server (sources.js liveSource) and
// kept up to date as the run logs. The address names the run and the member tab
// shown (/run/<run id>/genome/<genome id>); each run's open member tabs are kept
// for the browser session, and the window over the charts for good.

import { fmt, mountDashboard, progress, status } from "./dashboard.js";
import { h } from "./dom.js";
import { liveSource } from "./sources.js";

const POLL_MS = 2000;
const RUNS_POLL_MS = 5000;

const state = {
  runs: [],
  runId: null,
  // the dashboard of the run picked
  dash: null,
};

const main = document.getElementById("main");

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return res.json();
}

function ago(t) {
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
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
  if (runId && runId === state.runId) state.dash?.openTab(routeGenome(), { push: false });
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
  document.getElementById("conn").textContent = ok ? "" : "server unreachable";
}

// ---------------------------------------------------------------- run view

// a page of the monitor's own in place of a dashboard
function showPage(...children) {
  state.dash?.destroy();
  state.dash = null;
  main.className = "";
  main.replaceChildren(h("div", { class: "empty" }, ...children));
}

function renderEmpty() {
  if (state.runs.length) {
    // nothing selected: open the newest run
    go(state.runs[0].id);
    return;
  }
  showPage(
    h("h1", { text: "No runs yet" }),
    h("p", { text: "Start a training run with the monitor attached:" }),
    h("p", {}, h("code", { text: "uv run examples/boids.py --monitor" })),
    h("p", { text: "It will show up here as soon as it logs its first epoch." }),
  );
}

// the member tabs a run had open, kept for the browser session
function savedTabs(runId) {
  try {
    return JSON.parse(sessionStorage.getItem(`tabs:${runId}`)) || [];
  } catch {
    return [];
  }
}

function select(runId) {
  state.runId = runId;
  renderSide();
  if (!runId) {
    renderEmpty();
    return;
  }
  state.dash?.destroy();
  main.className = "";
  state.dash = mountDashboard(main, {
    source: liveSource(runId),
    tabs: savedTabs(runId),
    tab: routeGenome(),
    memberHref: (id) => `/run/${runId}/genome/${id}`,
    onTab: (id) => history.pushState(null, "", id == null ? `/run/${runId}` : `/run/${runId}/genome/${id}`),
    onTabs: (tabs) => sessionStorage.setItem(`tabs:${runId}`, JSON.stringify(tabs)),
    chartWindow: Number(localStorage.getItem("chartWindow")) || null,
    onChartWindow: (n) => (n ? localStorage.setItem("chartWindow", n) : localStorage.removeItem("chartWindow")),
    chartRange: localStorage.getItem("chartRange") !== "0",
    onChartRange: (on) => (on ? localStorage.removeItem("chartRange") : localStorage.setItem("chartRange", "0")),
    boardOpen: localStorage.getItem("boardOpen") === "1",
    onBoardOpen: (open) => (open ? localStorage.setItem("boardOpen", "1") : localStorage.removeItem("boardOpen")),
    onRefresh: () => setConnected(true),
    onError: (e) => {
      if (runId !== state.runId) return;
      if (String(e.message).startsWith("404")) {
        state.runId = null;
        showPage(h("h1", { text: "No such run" }));
      } else {
        setConnected(false);
      }
    },
  });
}

// ---------------------------------------------------------------- start

async function start() {
  await pollRuns();
  const runId = route();
  if (runId) select(runId);
  else renderEmpty();
  setInterval(pollRuns, RUNS_POLL_MS);
  setInterval(() => {
    if (document.visibilityState === "visible") state.dash?.refresh();
  }, POLL_MS);
}

start();
