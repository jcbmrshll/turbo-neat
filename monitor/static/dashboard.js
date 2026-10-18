// The NEAT dashboard: a run's charts, its test episodes played beside the networks
// playing them, the species map, the population's leaderboard, and a tab per member
// opened, with its network, episodes, line of descent and lineage. The run monitor
// (app.js) draws it around its own run list; a static page can draw it for runs
// exported as static files (python -m monitor.export). Where the run comes from is
// a source (sources.js); how it differs between the two is an option.
// No framework; every view is plain DOM built with h(), and labels that come from a
// run (metric names, config values) only ever go in as text.
//
// mountDashboard(root, options) draws a run into root and returns
// { refresh, openTab, destroy }. Options:
//
//   source           where the run comes from (sources.js); required
//   header           the run's name, status, seed, start, wall time and progress
//                    above the tabs (true)
//   generations      say generation for every step of the run, where the monitor
//                    says epoch: a member is born into a generation and alive for
//                    epochs (false)
//   configCollapsed  the run's config in a section that opens on a click (false)
//   bestLabel        "best" before the name of the member a test played (false)
//   fill             the dashboard fills root, its main column scrolling on its own
//                    and keeping each tab's place in it; otherwise it sits in the
//                    page, which scrolls (true)
// and for a page that keeps its own address and state:
//   tabs, tab        the member tabs to open ([{id, name}]) and the one to show
//   onTab(id)        a tab was opened (null: the run's), to put in the address
//   onTabs(tabs)     the member tabs open changed, [{id, name}]
//   memberHref(id)   the address of a member's tab, for its links
//   chartWindow      the epochs the charts show to begin with (null: all)
//   onChartWindow(n) the reader picked another
//   onRefresh()      the run was read
//   onError(e)       reading it failed; without it, a first read that fails says
//                    so in place of the dashboard
//
// refresh() reads what the run logged since it last did (a page following a live
// run calls it every so often); openTab(id, {push}) shows a member's tab, or the
// run's for null.

import { descentView } from "./descent.js";
import { h } from "./dom.js";
import { NdArray, loadArrays, episodePlayer, traceNetwork } from "./episodes.js";
import { BLUE, ORANGE, WELL } from "./palette.js";
import { RENDERERS } from "./renderers.js";
import { WIDE_INPUTS, inputGroups } from "./groups.js";
import { surfaceView } from "./surface.js";

const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"].map((v) => `var(${v})`);
// a running run that hasn't logged for this long is probably dead
const STALE_S = 180;
// charts that lead the grid; the rest follow in the order they were first logged
const CHART_ORDER = ["fitness", "fitness_against_baseline"];
const BOARD_LIMIT = 50;
// the windows offered over the metric charts, in epochs
const CHART_WINDOWS = [50, 200, 1000];

// ---------------------------------------------------------------- helpers

export function fmt(v) {
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
export function progress(run) {
  if (run.step == null) return "";
  const total = run.num_generations;
  return total ? `${run.step + 1} / ${total} epochs` : `epoch ${run.step}`;
}

export function status(run) {
  if (run.status === "running" && Date.now() / 1000 - run.updated > STALE_S) return "stale";
  return run.status;
}

// one tooltip for every dashboard on the page, in what's full screen when
// something is, or it wouldn't be seen
let tip = null;
export function showTip(x, y, ...children) {
  tip ??= h("div", { class: "neat-tip", hidden: true });
  const over = document.fullscreenElement ?? document.body;
  if (tip.parentNode !== over) over.append(tip);
  tip.replaceChildren(...children);
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  const left = x + 14 + r.width > innerWidth ? x - 14 - r.width : x + 14;
  const top = Math.min(innerHeight - r.height - 4, Math.max(4, y - r.height / 2));
  tip.style.left = `${left}px`;
  tip.style.top = `${top}px`;
}
export function hideTip() {
  if (tip) tip.hidden = true;
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

function speciesColor(id) {
  return SERIES[((id % SERIES.length) + SERIES.length) % SERIES.length];
}

function swatch(species) {
  return h("i", { class: "swatch", style: `background:${speciesColor(species)}` });
}

// "sandy-trout-4", with the species part (the animal in the middle) set apart
function nameEl(name) {
  const from = name.indexOf("-");
  const to = name.lastIndexOf("-");
  if (from === to) return h("span", { class: "gname", text: name });
  return h(
    "span",
    { class: "gname" },
    name.slice(0, from),
    h("span", { class: "sp", text: name.slice(from, to) }),
    name.slice(to),
  );
}

function configValue(v) {
  if (v == null) return "–";
  if (Array.isArray(v) && v.every((x) => typeof x !== "object")) return v.join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v).replace(/<function (\S+) at 0x[0-9a-f]+>/g, "$1");
}

// both tournaments picked the same parent, so its offspring is a mutated copy of it
function selfCross(n) {
  return n.parents[0] >= 0 && n.parents[0] === n.parents[1];
}

// ---------------------------------------------------------------- network

// each node's depth, its longest path from the inputs, with every output one past
// the deepest hidden node; and the columns (or rows) that makes
function layers(net) {
  const nodes = new Map(net.nodes.map((n) => [n.id, { ...n, depth: n.kind === "input" ? 0 : 1 }]));
  // bounded so a cycle can't hang the page
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
  return { nodes, hidden, outDepth, columns };
}

// the mean of where a node's sources sit along `axis`, to order a layer by
function sourceMean(net, nodes, n, axis, fallback) {
  const at = net.edges.filter((e) => e.to === n.id).map((e) => nodes.get(e.from)?.[axis]).filter((v) => v != null);
  return at.length ? at.reduce((a, b) => a + b, 0) / at.length : fallback;
}

// Inputs in a column on the left, outputs on the right, hidden nodes between by
// depth; each input's and output's label (and value, when lit up) beside it.
// Sets each node's x, y; returns the size and the labels to draw.
function layoutAcross(net, width, withValues) {
  const { nodes, hidden, outDepth, columns } = layers(net);
  const inputs = columns[0] || [];
  const outputs = columns[outDepth] || [];
  const ROW = 20;
  const rows = Math.max(...columns.map((c) => (c ? c.length : 0)), 3);
  // deep, narrow networks still get room to fan out
  const H = Math.max(rows * ROW + 16, 240);
  // room for a value (" -0.42") after each label, when there are values
  const valueChars = withValues ? 6 : 0;
  const labelW = (list) => (Math.max(0, ...list.map((n) => (n.label || "").length)) + valueChars) * 6.2 + 12;
  const padL = Math.min(160 + valueChars * 6.2, labelW(inputs));
  const padR = Math.min(160 + valueChars * 6.2, labelW(outputs));
  const W = Math.max(width, padL + padR + 120);
  const colX = (d) => padL + (d / outDepth) * (W - padL - padR);
  // order each column by where its inputs sit, so edges cross less
  columns.forEach((col, d) => {
    if (!col) return;
    if (d > 0 && d < outDepth) {
      for (const n of col) n.order = sourceMean(net, nodes, n, "y", H / 2);
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
  });
  // inputs' labels end at their node, and outputs' start at theirs, so the values
  // go on the side by the node
  const labels = [...inputs, ...outputs].map((n) => {
    const input = n.kind === "input";
    return { x: n.x + (input ? -9 : 9), y: n.y + 3.5, anchor: input ? "end" : null, text: n.label, value: n.id, valueFirst: !input };
  });
  const edge = (a, b) => {
    const mx = (a.x + b.x) / 2;
    return `M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`;
  };
  return { nodes, hidden, W, H, labels, edge };
}

// Inputs in a column down the left, close together in their groups (lidar,
// car 0, ...) with one label each, and inputs of their own labelled singly;
// outputs down the right with their labels (and values, when lit up); hidden
// nodes in columns between by depth, each level with its sources. All in
// `height`, the height of the episode beside it, where that leaves room.
function layoutGrouped(net, width, height, withValues) {
  const { nodes, hidden, outDepth, columns } = layers(net);
  const inputs = (columns[0] || []).sort((a, b) => a.id - b.id);
  const outputs = columns[outDepth] || [];
  const groups = inputGroups(inputs, (n) => n.label).map((g) => ({ name: g.name, nodes: g.members }));
  const label = (g) => (g.name != null && g.nodes.length > 1 ? g.name : null);
  const longest = Math.max(...groups.flatMap((g) => (label(g) ? [label(g)] : g.nodes.map((n) => n.label || ""))).map((t) => t.length));
  // as tall as the episode, but never so short that the inputs labelled singly
  // are less than 6px apart and their labels run into each other
  const PADY = 10, H = Math.max(height, 240, 2 * PADY + 6 * (inputs.length + 0.8 * (groups.length - 1)));
  const padL = Math.min(110, longest * 5.6 + 14);
  const padR = Math.min(120, (Math.max(...outputs.map((n) => (n.label || "").length)) + (withValues ? 6 : 0)) * 6.2 + 14);
  const W = Math.max(width, padL + padR + 60);
  // a slot per input, and most of one between groups
  const slot = (H - 2 * PADY) / (inputs.length + 0.8 * (groups.length - 1));
  let y = PADY + slot / 2;
  for (const [i, g] of groups.entries()) {
    if (i) y += 0.8 * slot;
    for (const n of g.nodes) {
      n.x = padL;
      n.y = y;
      y += slot;
    }
  }
  // each deeper column level with its sources, pushed apart where they crowd
  const colX = (d) => padL + (d / outDepth) * (W - padL - padR);
  const place = (col, x, minGap) => {
    for (const n of col) n.order = sourceMean(net, nodes, n, "y", H / 2);
    col.sort((a, b) => a.order - b.order);
    let prev = -Infinity;
    for (const n of col) {
      n.y = Math.max(n.order, prev + minGap, PADY);
      n.x = x;
      prev = n.y;
    }
    // back inside the height, keeping the gaps
    const over = (col.at(-1)?.y ?? 0) - (H - PADY);
    if (over > 0) for (const n of col) n.y -= over;
  };
  for (let d = 1; d < outDepth; d++) if (columns[d]) place(columns[d], colX(d), 12);
  place(outputs, colX(outDepth), 30);
  const labels = [];
  for (const g of groups) {
    if (label(g)) {
      // a group's name beside its span, with a bracket
      const y0 = g.nodes[0].y, y1 = g.nodes.at(-1).y;
      labels.push({ x: padL - 9, y: (y0 + y1) / 2 + 3.5, anchor: "end", text: label(g), bracket: [y0 - 2, y1 + 2, padL - 6] });
    } else {
      for (const n of g.nodes) labels.push({ x: padL - 7, y: n.y + 3, anchor: "end", text: n.label, small: true });
    }
  }
  for (const n of outputs) labels.push({ x: n.x + 9, y: n.y + 3.5, anchor: null, text: n.label, value: n.id, valueFirst: true });
  const edge = (a, b) => {
    const mx = (a.x + b.x) / 2;
    return `M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`;
  };
  return { nodes, hidden, W, H, labels, edge };
}

// net drawn to width; with a trace (episodes.js traceNetwork), also show(frame),
// which lights it up with what it did at that frame: each node coloured by the
// sign of its output, as strong as the output is against its largest over the
// episode, each edge likewise by what it carried (its source's output times its
// weight), and the labelled nodes' values beside their labels. Networks of more
// than WIDE_INPUTS inputs are drawn grouped and compact, `height` tall
// (layoutGrouped); the rest with every input labelled (layoutAcross).
export function networkView(net, width, trace = null, { height = 480 } = {}) {
  const wide = net.nodes.filter((n) => n.kind === "input").length > WIDE_INPUTS;
  const { nodes, hidden, W, H, labels, edge } = wide
    ? layoutGrouped(net, width, height, Boolean(trace))
    : layoutAcross(net, width, Boolean(trace));

  const svg = h("svg", { class: `net${wide ? " wide" : ""}`, viewBox: `0 0 ${W} ${H}`, height: H });
  const maxW = Math.max(1e-6, ...net.edges.map((e) => Math.abs(e.weight)));
  const edgeEls = [];
  for (const e of net.edges) {
    const a = nodes.get(e.from), b = nodes.get(e.to);
    if (!a || !b) continue;
    const s = Math.abs(e.weight) / maxW;
    const el = h("path", {
      class: "edge",
      d: edge(a, b),
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
  const nodeEls = [];
  for (const n of nodes.values()) {
    const r = n.kind === "hidden" ? 4.5 : wide && n.kind === "input" ? 2.6 : 5;
    const node = h("circle", { class: `node ${n.kind}${trace ? " lit" : ""}`, cx: n.x, cy: n.y, r });
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
  }
  const valueEls = [];
  for (const l of labels) {
    const value = trace && l.value != null ? h("tspan", { class: "val" }) : null;
    const parts = l.valueFirst ? [value, value && " ", l.text] : [l.text, value && " ", value];
    const attrs = { class: `lab${l.bracket ? " group" : ""}${l.small ? " small" : ""}`, x: l.x, y: l.y, "text-anchor": l.anchor };
    svg.append(h("text", attrs, ...parts));
    if (l.bracket) {
      const [y0, y1, x] = l.bracket;
      svg.append(h("path", { class: "bracket", d: `M${x + 3},${y0} H${x} V${y1} H${x + 3}` }));
    }
    if (value) valueEls.push([value, l.value]);
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

// each node's least and most output over a trace
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

// the key under a network lit up: one scale, negative through zero to positive
function activationKey() {
  return h("span", { class: "act-key" }, "activation", h("span", { class: "dim", text: "−" }), h("i"), h("span", { class: "dim", text: "+" }));
}

// ---------------------------------------------------------------- episodes

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
  // (an export can keep only some seats' rows of a game of several, with
  // obs_seats saying which, so a row's player is the one in its seat)
  const id = (seat, row) => (ids.length === count ? ids[row] : ids.length === 1 ? ids[0] : ids[seat] ?? null);
  return seats.map((seat, row) => ({ row, seat, id: id(seat, row) }));
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

// a player, with what each frame drawn goes to: the network lit up beside it
// (player.netFrame) and the names under a race's cars (player.namesFrame)
function listen(player) {
  player.onFrame = (frame) => {
    player.netFrame?.(frame);
    player.namesFrame?.(frame);
  };
  return player;
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
// what the environments' rewards are measured in
const REWARD_UNITS = { f1tenth: " m", f1tenth_field: " m" };

// a player of a game, marked with its seat's colour in the picture, and with its
// reward over the game if the run recorded it
function seatPlayer(who, seat, env, reward) {
  const { name, style } = seatOf(env, seat);
  const score = reward == null ? [] : [h("span", { class: "seat", text: `${reward.toFixed(1)}${REWARD_UNITS[env] || ""}` })];
  return h("span", { class: "ep-player" }, h("i", { class: "swatch", style }), who, h("span", { class: "seat", text: name }), ...score);
}

// players side by side, "a vs b vs c"
function matchup(players) {
  const parts = players.flatMap((p, i) => (i ? [h("span", { class: "dim", text: "vs" }), p] : [p]));
  return h("div", { class: "ep-games" }, h("span", { class: "ep-match" }, ...parts));
}

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
// how many of the biggest species get a name beside their dot; the rest on hover
const SPECIES_MAP_LABELS = 8;

// ---------------------------------------------------------------- the dashboard

export function mountDashboard(root, options) {
  const {
    source,
    header = true,
    generations = false,
    configCollapsed = false,
    bestLabel = false,
    fill = true,
    memberHref = null,
    onTab = null,
    onTabs = null,
    onChartWindow = null,
    onRefresh = null,
    onError = null,
  } = options;
  // what a step of the run is called
  const w = generations
    ? { step: "generation", steps: "generations", a: "a generation" }
    : { step: "epoch", steps: "epochs", a: "an epoch" };
  let dead = false;

  const state = {
    run: null,
    rows: [],
    metricsNext: 0,
    media: [],
    mediaNext: 0,
    // the epoch the run tab and the leaderboard show, or null to follow the latest
    epoch: null,
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
    overviewScroll: 0,
    // the metric charts show the last this-many epochs, or all of them if null
    chartWindow: options.chartWindow || null,
  };
  const networkCache = {};
  // whether the run's networks are drawn compact, in a column beside a larger
  // episode (see WIDE_INPUTS)
  const wideNets = () => (state.run?.config?.genome_config?.input_size ?? 0) > WIDE_INPUTS;
  // what each chart shows of the epoch shown (its epoch line, its values), to
  // update without redrawing the chart
  const chartUpdates = [];

  const main = h("div", { class: "nd-main" });
  const boardEl = h("aside", { class: "nd-board" });
  const layout = h("div", { class: "nd-layout" }, main, boardEl);
  root.replaceChildren(layout);
  root.classList.add("neat-dash");
  // dashboard.css lays it out filling root, or in the page
  root.classList.toggle("nd-fill", fill);

  const view = {
    head: header && h("div", { class: "runhead" }),
    tabbar: h("nav", { class: "tabs" }),
    epochBar: epochBar(),
    overview: h("div", { class: "tab-panel" }),
    panels: h("div"),
    media: h("div", { class: "media-grid" }),
    chartsTitle: h("div", { class: "stitle-row" }),
    charts: h("div", { class: "chart-grid" }),
    configTitle: h("div", { class: "stitle", text: "Config" }),
    config: h("div", { class: "config-grid" }),
    mediaCards: {},
  };
  view.overview.append(
    view.media,
    view.chartsTitle,
    view.charts,
    ...(configCollapsed
      ? [h("details", { class: "config" }, h("summary", { class: "stitle", text: "Config" }), view.config)]
      : [view.configTitle, view.config]),
  );
  renderChartsTitle();
  const top = h("div", { class: "topbar" }, view.tabbar, view.epochBar.el);
  main.replaceChildren(h("div", { class: "page" }, view.head, top, view.overview, view.panels));
  main.append(h("div", { class: "loading", text: "loading run…" }));

  // ------------------------------------------------------------ the run

  // read what the run logged since the last time
  let reading = false;
  async function refresh() {
    if (dead || reading) return;
    reading = true;
    try {
      await read();
      if (!dead) onRefresh?.();
    } catch (e) {
      if (dead) return;
      if (onError) onError(e);
      else if (!state.run) main.replaceChildren(h("div", { class: "media-error", text: `couldn't load this run: ${e.message}` }));
    } finally {
      reading = false;
    }
  }

  async function read() {
    const [run, metrics, media] = await Promise.all([
      source.run(),
      source.metrics(state.metricsNext),
      source.media(state.mediaNext),
    ]);
    if (dead) return;
    const firstLoad = !state.run;
    state.run = run;
    if (firstLoad) {
      main.querySelector(":scope > .loading")?.remove();
      if (run.species_names) state.speciesNames = run.species_names;
    }
    state.metricsNext = metrics.next;
    state.mediaNext = media.next;
    state.rows.push(...metrics.rows);
    state.media.push(...media.rows);

    renderHead();
    if (firstLoad) renderConfig();
    if (firstLoad || metrics.rows.length) {
      renderChartsTitle();
      view.epochBar.update();
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
  }

  function renderHead() {
    if (!view.head) return;
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

  // a metric's value in the epoch shown, or null if it wasn't logged then
  function valueAt(key) {
    const step = state.epoch ?? state.rows.at(-1)?.step;
    return state.rows.find((r) => r.step === step)?.[key] ?? null;
  }

  // ------------------------------------------------------------ epoch

  // show the run as it was at an epoch (null: the latest, following new ones)
  function setEpoch(epoch) {
    const last = state.rows.at(-1)?.step;
    state.epoch = epoch == null || epoch >= last ? null : epoch;
    view.epochBar.update();
    chartUpdates.forEach((update) => update());
    renderMedia();
    clearTimeout(setEpoch.board);
    // dragging the slider asks for the leaderboard once it settles
    setEpoch.board = setTimeout(pollBoard, 120);
  }

  // the slider across the top that picks the epoch shown, with a box to type one in;
  // made once, and updated in place so a drag isn't interrupted
  function epochBar() {
    const range = h("input", { type: "range", min: 0, max: 0, value: 0, step: 1 });
    const box = h("input", { type: "number", class: "epoch-box", min: 0, step: 1, title: `type ${w.a}` });
    const of = h("span", { class: "dim" });
    const [text, title] = generations ? ["last", "back to the last generation"] : ["latest", "back to the current epoch, following new ones"];
    const latestBtn = h("button", { text, title, onclick: () => setEpoch(null) });
    range.addEventListener("input", () => setEpoch(Number(range.value)));
    // typed epochs apply on enter or leaving the box, clamped to the run's
    box.addEventListener("change", () => {
      const first = state.rows[0]?.step ?? 0;
      if (box.value !== "") setEpoch(Math.max(first, Math.round(Number(box.value))));
      else update();
    });
    box.addEventListener("keydown", (e) => e.key === "Enter" && box.blur());
    const el = h("div", { class: "epoch-bar scrub" }, h("span", { class: "dim", text: w.step }), range, box, of, latestBtn);
    function update() {
      const first = state.rows[0]?.step ?? 0, last = state.rows.at(-1)?.step ?? 0;
      el.hidden = !state.rows.length;
      range.min = box.min = first;
      range.max = box.max = last;
      range.value = state.epoch ?? last;
      // not while it's being typed in
      if (document.activeElement !== box) box.value = state.epoch ?? last;
      of.textContent = `of ${last}`;
      latestBtn.disabled = state.epoch == null;
      el.classList.toggle("pinned", state.epoch != null);
    }
    return { el, update };
  }

  // ------------------------------------------------------------ charts

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
    chartUpdates.length = 0;
    const oneSpecies = singleSpecies();
    const groups = groupMetrics(metricKeys()).filter((g) => !(oneSpecies && g.name.startsWith("species_")));
    view.chartsTitle.hidden = !groups.length;
    // the window ends at the newest epoch logged
    const last = state.rows.at(-1)?.step;
    const from = state.chartWindow && last != null ? last - state.chartWindow + 1 : -Infinity;
    const media = CHART_MEDIA.map((k) => view.mediaCards[k]?.el).filter(Boolean);
    view.charts.replaceChildren(...groups.map((g) => chartCard(g, from)), ...media);
  }

  function setChartWindow(n) {
    state.chartWindow = n > 0 ? Math.floor(n) : null;
    onChartWindow?.(state.chartWindow);
    renderChartsTitle();
    if (state.rows.length) renderCharts();
  }

  // "Metrics", and the window of epochs the charts show: all, a preset shorter than
  // the run, or any N typed in
  function renderChartsTitle() {
    const win = state.chartWindow;
    const total = state.rows.length;
    const presets = CHART_WINDOWS.filter((n) => n < total);
    const typed = win && !presets.includes(win);
    // not while a window's being typed in
    if (view.chartsTitle.contains(document.activeElement)) return;
    const toggle = (n, text, title) =>
      h("button", { class: win === n ? "on" : null, text, title, onclick: () => setChartWindow(n) });
    const custom = h("input", {
      type: "number",
      min: 1,
      max: total,
      step: 1,
      placeholder: "N",
      title: `show the last N ${w.steps}`,
      value: typed ? win : null,
      class: typed ? "on" : null,
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
        toggle(null, "all", `every ${w.step}`),
        presets.map((n) => toggle(n, fmt(n), `the last ${n} ${w.steps}`)),
        custom,
        h("span", { class: "dim", text: w.steps }),
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
    // a species' value is its value in the epoch shown, so one that died out (or
    // hasn't appeared yet) has none; any other metric's, its latest (tests log
    // only every so often)
    const species = group.name.startsWith("species_");
    return seriesCard(group.name.replaceAll("_", " "), series, (key) => (species ? valueAt(key) : latest(key)));
  }

  // a card with a line chart of the series ([step, value] points) and each one's
  // value(key) as of the epoch shown: in the corner for one series, beside each
  // name in the legend for several
  function seriesCard(title, series, value) {
    const single = series.length === 1;
    const values = series.map(() => h("b"));
    const update = () => series.forEach((s, i) => (values[i].textContent = fmt(value(s.key))));
    update();
    chartUpdates.push(update);
    const card = h(
      "div",
      { class: "card chart-card" },
      h(
        "div",
        { class: "card-hdr" },
        h("span", { class: "title", text: title }),
        h("span", { class: "spacer" }),
        single && h("span", { class: "latest" }, values[0]),
      ),
      !single &&
        h(
          "div",
          { class: "legend" },
          series.map((s, i) =>
            h("span", { class: "key" }, h("i", { style: `background:${s.color}` }), s.name, values[i]),
          ),
        ),
    );
    const holder = h("div");
    card.append(holder);
    // draw once the card is laid out, so the chart knows its width
    requestAnimationFrame(() => holder.replaceChildren(lineChart(series, holder.clientWidth)));
    return card;
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
    chartUpdates.push(placeMark);
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
        h("div", { class: "head", text: `${w.step} ${best}` }),
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

  // redraw what's drawn to fit when the dashboard changes width
  let lastWidth = 0, resizeTimer = null;
  const resized = new ResizeObserver(() => {
    const width = root.clientWidth;
    if (!lastWidth || Math.abs(width - lastWidth) < 2) {
      lastWidth = width;
      return;
    }
    lastWidth = width;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (dead || !state.run) return;
      const tab = activeTab();
      if (tab) {
        if (tab.individual) renderMember(tab);
      } else {
        relayoutOverview();
      }
    }, 150);
  });
  resized.observe(root);

  // redraw what the run tab draws to fit its width
  function relayoutOverview() {
    state.overviewStale = false;
    if (state.rows.length) renderCharts();
    // the species map is drawn to fit; leave images alone so gifs don't restart
    for (const card of Object.values(view.mediaCards)) {
      if (card.fitted) card.shown = null;
    }
    renderMedia();
  }

  // ------------------------------------------------------------ media

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
      h("span", { class: "latest", text: `${w.step} ${entry.step}` }),
    );
    card.who.replaceChildren(...playerLine(key, entry));
    // swapping the element restarts a gif or a player, so only do it when the entry
    // changes
    const id = entry.file || entry.episode || (entry.data ? `${key}-${entry.step}` : `error-${entry.step}`);
    if (card.shown === id) return;
    card.shown = id;
    // drawn to the card's width, so drawn again when it changes
    card.fitted = Boolean(entry.data) || Boolean(entry.file?.endsWith(".json"));
    card.foot.replaceChildren();
    if (entry.error) {
      card.body.replaceChildren(h("div", { class: "media-error", text: `couldn't render: ${entry.error}` }));
      return;
    }
    if (entry.episode) {
      showEpisode(card, entry);
      return;
    }
    // data drawn to the card's width: the species map, or anything else as JSON
    const draw = (data) => {
      if (data.type === "species_pca") {
        const { svg, summary } = speciesMapView(data, card.body.clientWidth - 22);
        card.body.replaceChildren(svg);
        card.foot.replaceChildren(h("div", { class: "net-foot", text: summary }));
      } else {
        card.body.replaceChildren(h("pre", { text: JSON.stringify(data, null, 2) }));
      }
    };
    if (entry.data) {
      draw(entry.data);
      return;
    }
    const src = source.mediaUrl(entry);
    if (entry.file.endsWith(".json")) {
      card.body.replaceChildren();
      fetch(src)
        .then((r) => r.json())
        .then((data) => !dead && card.shown === entry.file && draw(data));
    } else if (entry.file.endsWith(".mp4")) {
      card.body.replaceChildren(h("video", { src, autoplay: true, loop: true, muted: true, controls: true }));
    } else {
      card.body.replaceChildren(h("img", { src, alt: `${key} at ${w.step} ${entry.step}` }));
    }
  }

  // a test's episode, played beside the network of the agent picked in it (the
  // first one with observations, or the one clicked), lit up as it plays
  async function showEpisode(card, entry) {
    const render = RENDERERS[entry.env];
    let arrays;
    try {
      if (!render) throw new Error(`no renderer for ${entry.env}`);
      arrays = await loadArrays(source.episodeUrl(entry));
    } catch (e) {
      if (card.shown === entry.episode) card.body.replaceChildren(h("div", { class: "media-error", text: `couldn't show this episode: ${e.message}` }));
      return;
    }
    if (dead || card.shown !== entry.episode) return;
    const players = entry.players || [];
    const net = { body: h("div", { class: "ep-net" }), foot: h("div") };

    // a classifier's test is a still of its predictions; with its network, the
    // decision surface it draws over the whole plane, which lights the network up
    // for whatever point is under the pointer
    if (entry.env === "classification_2d" && players[0]) {
      const network = await fetchNetwork(players[0].id, entry.step);
      if (dead || card.shown !== entry.episode) return;
      if (network) {
        const surface = surfaceView(arrays, network);
        if (!surface.error) {
          card.body.replaceChildren(h("div", { class: "ep-grid" }, surface.el, h("div", {}, net.body, net.foot)));
          requestAnimationFrame(() => {
            const { svg, summary, show } = networkView(network, net.body.clientWidth - 22, surface.trace);
            net.body.replaceChildren(svg);
            net.foot.replaceChildren(h("div", { class: "net-foot" }, h("span", { text: summary }), activationKey(), h("span", { class: "dim", text: "hover the plane" })));
            surface.onPoint = show;
            show(surface.frame);
          });
          return;
        }
      }
    }

    let rows = obsRows(arrays, players).filter((r) => r.id != null);
    // runs from before episodes kept their observations: each player's network
    // beside the episode, still
    const still = !rows.length;
    if (still) {
      rows = FIELD_RACES.has(entry.env)
        ? players.map((p, seat) => ({ seat, id: p.id }))
        : players.length === 1
          ? [{ seat: BASELINE_SEAT[entry.env] ?? null, id: players[0].id }]
          : [];
      if (source.hasNetwork) rows = rows.filter((r) => source.hasNetwork(r.id, entry.step));
    }
    // a race between the epoch's fittest is seen from a car picked at random (the
    // leader's is just one view of it); anything else from its first agent
    const start = FIELD_RACES.has(entry.env) ? rows[Math.floor(Math.random() * rows.length)] : rows[0];
    const view = render(arrays);
    const player = listen(episodePlayer(view, { seat: start?.seat ?? null }));
    carNames(view, player, players, { at: { step: entry.step } });
    if (!rows.length) {
      card.body.replaceChildren(player.el);
      return;
    }
    // full screen together
    player.fullEl = h("div", { class: `ep-grid${wideNets() ? " wide" : ""}` }, player.el, h("div", {}, net.body, net.foot));
    card.body.replaceChildren(player.fullEl);
    const pick = (seat) => {
      const row = rows.find((r) => r.seat === seat);
      if (!row) return false;
      if (seat != null) player.setSeat(seat);
      fetchNetwork(row.id, entry.step).then((network) => {
        if (dead || card.shown !== entry.episode || player.seat !== seat) return;
        if (network) drawNetwork(net, network, still ? null : { arrays, row, player }, player.el);
        else if (!still) net.body.replaceChildren(h("div", { class: "media-error", text: "no network logged for it" }));
      });
      return true;
    };
    player.onSeat = pick;
    requestAnimationFrame(() => pick(start.seat));
  }

  // a genome's network in an epoch, kept once fetched; null if it wasn't logged
  async function fetchNetwork(id, step) {
    const key = `${id}@${step}`;
    if (!(key in networkCache)) {
      const network = await source.network(id, step);
      if (!network) return null;
      networkCache[key] = network;
    }
    return networkCache[key];
  }

  // a member's species, when the record doesn't say (members' episodes don't):
  // the one named by a word of the name it was born with, wherever in it the
  // animal is ("big-red-dog", "minty-emu-412"), the last such word if several
  function birthSpecies(p) {
    if (p.birth_species != null) return p.birth_species;
    const ids = new Map(Object.entries(state.speciesNames).map(([id, name]) => [name, Number(id)]));
    const animal = (p.name ?? "").split("-").reverse().find((word) => ids.has(word));
    return animal == null ? null : ids.get(animal);
  }

  // In a race whose view can say where each car's card is in its strip (f1tenth.js
  // slots), each car's member named under its card, in its species' colour,
  // following it as the cards shuffle: the followed car's in bold, and a link to
  // the member's tab (opened `at` an epoch and game) but for `self`, the member
  // whose tab it is, in bold. Returns whether it named them.
  function carNames(view, player, players, { self = null, at = null } = {}) {
    if (!view.slots || players.length !== view.cars) return false;
    const chips = players.map((p) => {
      const species = birthSpecies(p);
      // its whole name on one line, in type sized to fill its card (fit, below)
      const name = p.name || `#${p.id}`;
      const who = h("span", { class: "who", text: name, style: species != null ? `color:${speciesColor(species)}` : null });
      const title = `${name}${p.reward != null ? ` · ${p.reward.toFixed(1)} m` : ""}`;
      if (p.id === self) return h("span", { class: "car-name self", title }, who);
      return memberLink(p.id, { class: "car-name", title, name: p.name, at }, who);
    });
    for (const chip of chips) chip.style.width = `${100 / view.cars}%`;
    const row = h("div", { class: "car-names" }, chips);
    // as wide as the race above it, however wide that's drawn, each name in type
    // that fills its card's width, worked out again when the race's width changes
    const canvas = player.el.querySelector("canvas");
    const measure = document.createElement("canvas").getContext("2d");
    let fitted = 0;
    const fit = () => {
      if (!canvas.clientWidth || canvas.clientWidth === fitted) return;
      fitted = canvas.clientWidth;
      row.style.width = `${fitted}px`;
      // the card, less the chip's padding (the gap between names)
      const style = getComputedStyle(chips[0]);
      const room = fitted / view.cars - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
      for (const chip of chips) {
        const who = chip.querySelector(".who");
        // in bold, so the followed car's name fits too
        measure.font = `700 100px ${getComputedStyle(who).fontFamily}`;
        const perPx = measure.measureText(who.textContent).width / 100;
        who.style.fontSize = `${Math.min(12, room / perPx).toFixed(2)}px`;
      }
    };
    const place = (t) => {
      fit();
      view.slots(t).forEach((slot, seat) => {
        chips[seat].style.left = `${(slot / view.cars) * 100}%`;
        chips[seat].classList.toggle("followed", seat === player.seat);
      });
    };
    // on the page before it's first placed, so the names' type can be measured
    player.el.insertBefore(row, player.el.querySelector(".player-controls"));
    player.namesFrame = place;
    place(player.frame);
    return true;
  }

  // the episode shown is of the genomes the run's test picked; name them, and in a
  // game against a built-in baseline or bots, who played which side
  function playerLine(key, entry) {
    if (!MEDIA_ORDER.includes(key)) return [];
    const players = entry.players || [];
    if (key === "episode" && FIELD_RACES.has(entry.env)) {
      // the field's members are named under their cars in the strip (carNames)
      if (players.length) return [];
      const text = `the ${w.step}'s fittest racing each other, P1 the fittest`;
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
      const best = bestLabel && h("span", { class: "seat", text: "best" });
      return [h("div", { class: "ep-games" }, h("span", { class: "ep-player" }, best, swatch(p.birth_species), name))];
    }
    const sides = [seatPlayer(name, seat, entry.env), seatPlayer(h("span", { text: "baseline" }), 1 - seat, entry.env)];
    if (seat === 1) sides.reverse();
    return [matchup(sides)];
  }

  // net drawn into a card, and with `activity` ({arrays, row, player}), lit up by
  // what it did with the episode's observations in that row, at the player's
  // frame; a grouped network as tall as the episode beside it (`beside`, a player's
  // element, the activity's by default)
  function drawNetwork(card, net, activity = null, beside = null) {
    const trace = activity && traceNetwork(net, obsOf(activity.arrays, activity.row.row));
    const episode = (beside ?? activity?.player.el)?.querySelector("canvas");
    const height = episode?.clientHeight || undefined;
    const { svg, summary, show } = networkView(net, card.body.clientWidth - 22, trace?.error ? null : trace, { height });
    card.body.replaceChildren(svg);
    const legend = show && activationKey();
    card.foot.replaceChildren(h("div", { class: "net-foot" }, h("span", { text: summary }), legend, trace?.error && h("span", { class: "dim", text: trace.error })));
    if (activity) {
      activity.player.netFrame = show || null;
      show?.(activity.player.frame);
    }
  }

  // ------------------------------------------------------------ species map

  // each species' mean parameters on the first two principal components across
  // species: a dot per species, sized by its members, both axes to one scale so
  // distances on the map are distances between species; a click shows the species
  // on the leaderboard
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

    const step = (lo, hi, count) => niceTicks(lo, hi, count).filter((v) => v >= lo && v <= hi);
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
      dot.addEventListener("click", () => {
        state.boardSpecies = s.id;
        pollBoard();
      });
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

  // ------------------------------------------------------------ leaderboard

  // a link to a member's tab, which opens it; `at` ({step, game}) opens it on one of
  // its games. With memberHref, a link the browser can open in a tab of its own
  function memberLink(id, { class: cls, title = null, name = null, at = null }, ...children) {
    return h(
      "a",
      {
        class: cls,
        href: memberHref ? memberHref(id) : "#",
        title,
        onclick: (e) => {
          if (memberHref && (e.metaKey || e.ctrlKey)) return;
          e.preventDefault();
          openTab(id, { name, at });
        },
      },
      ...children,
    );
  }

  function genomeLink(id, name, at = null) {
    return memberLink(id, { class: "glink", name, at }, nameEl(name));
  }

  async function pollBoard() {
    if (!state.run) return;
    try {
      const board = await source.board({
        scope: state.boardScope,
        limit: BOARD_LIMIT,
        species: state.boardSpecies,
        epoch: state.epoch,
      });
      if (dead) return;
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

  // the leaderboard, in the panel beside the run; it's there whenever the run logged
  // its members
  function renderBoard() {
    const board = state.board;
    const has = board && board.step != null;
    layout.classList.toggle("has-board", Boolean(has));
    if (!has) {
      boardEl.replaceChildren();
      return;
    }
    const allTime = state.boardScope === "all";
    const toggle = (on, text, title, onclick) => h("button", { class: on ? "on" : null, text, title, onclick });
    const scopes = h(
      "span",
      { class: "scrub" },
      [["alive", `this ${w.step}`], ["all", "all time"]].map(([value, text]) =>
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
      h("th", { class: "num", text: "fitness", title: `fitness in its latest ${w.step}` }),
      allTime && h("th", { class: "num", text: "born", title: generations ? "the generation it was born into" : "the generation it was born into: the epoch it joined the population" }),
      h("th", { class: "num", text: generations ? "age" : "epochs", title: `${w.steps} in the population` }),
      h("th", { class: "num", text: "offspring" }),
    );
    const open = new Set(state.tabs.map((t) => t.id));
    const born = generations ? "born generation" : "generation";
    const rows = board.rows.map((r, i) =>
      h(
        "tr",
        {
          class: ["row", r.id === state.activeTab && "sel", open.has(r.id) && "open", !r.alive && "gone", r.featured && "featured"].filter(Boolean).join(" "),
          title: `${r.name} · ${r.species_name} · ${born} ${r.born}, ${r.alive ? `alive since ${w.step} ${r.born}` : `alive ${w.steps} ${r.born}–${r.last}`}`,
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
      rows.push(h("tr", {}, h("td", { class: "dim", colspan: 6, text: `no ${who} ${allTime ? "yet" : `in this ${w.step}`}` })));
    }
    // the table keeps its scroll from one epoch to the next
    if (!boardEl.firstChild) boardEl.append(h("div", { class: "bp-head" }), h("div", { class: "bp-table" }));
    const [top, wrap] = boardEl.children;
    // rebuilding the species list while it's open would close it
    const choosing = top.contains(document.activeElement) && document.activeElement.tagName === "SELECT";
    top.replaceChildren(
      h("div", { class: "stitle", text: "Population" }),
      h("div", { class: "bp-sub", text: `${w.step} ${board.step} · ${board.population} members` }),
      choosing ? top.querySelector(".bp-controls") : h("div", { class: "bp-controls" }, scopes, filter),
    );
    const scrolled = wrap.scrollTop;
    wrap.replaceChildren(h("table", { class: "board" }, h("thead", {}, head), h("tbody", {}, rows)));
    wrap.scrollTop = scrolled;
  }

  // ------------------------------------------------------------ tabs

  // The run's own tab is always there; each member opened gets a tab of its own,
  // which keeps its place (the epoch and game it shows, how far down it's
  // scrolled) while others are shown.

  function activeTab() {
    return state.tabs.find((t) => t.id === state.activeTab) || null;
  }

  function tabsChanged() {
    onTabs?.(state.tabs.map((t) => ({ id: t.id, name: t.name })));
  }

  function addTab(id, name) {
    let tab = state.tabs.find((t) => t.id === id);
    if (!tab) {
      tab = {
        id,
        name: name || null,
        individual: null,
        // the epoch of its life the network and episode cards show (pinned to its
        // latest when first shown), and the game shown when it was opened on one
        // (by the game's index), rather than its own
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
      tabsChanged();
    }
    return tab;
  }

  // show a member's tab, opening it if need be; null shows the run's tab. `at`
  // ({step, game}) turns it to one of its games, e.g. the one it played with the
  // member whose tab it was opened from
  function openTab(id, { name = null, push = true, at = null } = {}) {
    if (dead) return;
    hideTip();
    if (id != null) {
      const tab = addTab(id, name);
      // (a source that keeps members' last epoch only shows them in that)
      if (at && !source.lastOnly) {
        tab.step = at.step;
        tab.game = at.game;
      }
    }
    if (push && id !== state.activeTab) onTab?.(id);
    // filling the page, each tab comes back to where it was scrolled
    if (fill) {
      const leaving = activeTab();
      if (leaving) leaving.scroll = main.scrollTop;
      else state.overviewScroll = main.scrollTop;
    }
    state.activeTab = id;
    view.overview.hidden = id != null;
    for (const t of state.tabs) t.el.hidden = t.id !== id;
    renderTabs();
    if (state.board) renderBoard();

    const tab = activeTab();
    if (tab) {
      renderMember(tab);
      if (tab.stale) pollMember(tab);
    } else if (state.overviewStale) {
      relayoutOverview();
    }
    if (fill) {
      main.scrollTop = tab ? tab.scroll : state.overviewScroll || 0;
    } else if (main.getBoundingClientRect().top < 0) {
      // in the page, the tabs brought into view if they're scrolled away
      main.scrollIntoView({ block: "start" });
    }
  }

  function closeTab(id) {
    const i = state.tabs.findIndex((t) => t.id === id);
    if (i < 0) return;
    const [tab] = state.tabs.splice(i, 1);
    tab.el.remove();
    tabsChanged();
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

  // ------------------------------------------------------------ member tab

  async function pollMember(tab) {
    let data;
    try {
      data = (await source.member(tab.id)) ?? { missing: true, id: tab.id };
    } catch {
      return;
    }
    if (dead || !state.tabs.includes(tab)) return;
    tab.stale = false;
    // a new epoch changes little about most members; skip redrawing when nothing did
    if (JSON.stringify(data) === JSON.stringify(tab.individual)) return;
    const named = !tab.individual;
    tab.individual = data;
    if (data.name && data.name !== tab.name) {
      tab.name = data.name;
      tabsChanged();
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
    const nameOf = (id) => ind.pedigree?.find((n) => n.id === id)?.name ?? source.known?.(id)?.name;
    const link = (id) => (nameOf(id) ? genomeLink(id, nameOf(id)) : h("span", { text: `#${id}` }));
    const [parent, mate] = ind.parents || [null, null];
    const parentage =
      parent == null
        ? null
        : parent < 0
          ? h("span", { text: "founder" })
          : h("span", {}, "offspring of ", link(parent), mate >= 0 && mate !== parent && [" × ", link(mate)]);
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
            title: `every episode it played for its fitness, over ${ret.epochs} ${ret.epochs === 1 ? w.step : w.steps}`,
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
        ind.species != null && ind.species !== ind.birth_species && h("span", { class: "dim", text: `(born ${state.speciesNames[ind.birth_species] ?? ind.birth_species})` }),
        sep(),
        h("span", { text: `generation ${ind.born}${ind.alive ? ", alive" : ""}` }),
        parentage && sep(),
        parentage,
      ),
    );
    const age = ind.age ?? ind.last - ind.born + 1;
    const share = ind.population ? ` (${Math.round((100 * ind.descendants) / ind.population)}%)` : "";
    const tiles = h(
      "div",
      { class: "tiles" },
      fitness,
      tile(`${w.steps} alive · ${age}`, ind.last > ind.born ? `${ind.born}–${ind.last}` : `${ind.born}`),
      tile("offspring", ind.offspring),
      ind.descendants != null && tile("living descendants", `${ind.descendants}${share}`),
    );

    // a member the source has no page for (an export keeps the fittest only): its
    // record, and its network if one was kept
    if (ind.partial) {
      const note = h("div", { class: "card loading", text: "this showcase keeps the episodes and lineage of the fittest members only: those marked ◆ in the population" });
      const step = ind.network_step;
      if (step == null) {
        tab.el.replaceChildren(head, tiles, note);
        return;
      }
      const card = { body: h("div", { class: "media-body" }), foot: h("div") };
      tab.el.replaceChildren(
        head,
        tiles,
        h("div", { class: "lineage-grid" }, h("div", { class: "card" }, h("div", { class: "card-hdr" }, h("span", { class: "title", text: "network" }), h("span", { class: "spacer" }), h("span", { class: "latest", text: `${w.step} ${step}` })), card.body, card.foot)),
        note,
      );
      fetchNetwork(ind.id, step).then((net) => !dead && net && requestAnimationFrame(() => drawNetwork(card, net)));
      return;
    }

    const pedBody = h("div", { class: "ped-scroll" });
    // it keeps its place counted from the right, where the member is, when the panel
    // changes width (as when the leaderboard comes in)
    let fromRight = 0;
    pedBody.addEventListener("scroll", () => (fromRight = pedBody.scrollWidth - pedBody.clientWidth - pedBody.scrollLeft));
    new ResizeObserver(() => (pedBody.scrollLeft = pedBody.scrollWidth - pedBody.clientWidth - fromRight)).observe(pedBody);
    const pedigree = h(
      "div",
      { class: "card" },
      h(
        "div",
        { class: "card-hdr" },
        h("span", { class: "title", text: "lineage" }),
        ind.pedigree_cut != null && h("span", { class: "latest dim", text: `from generation ${ind.pedigree_cut}` }),
      ),
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
        h("div", { class: "lineage-grid" }, cards.episode.el),
        cards.descent.el,
        pedigree,
        offspring,
      ].filter(Boolean),
    );
    if (fill && tab.id === state.activeTab) main.scrollTop = scrolled;
    loadMemberCards(tab);
    requestAnimationFrame(() => {
      pedBody.replaceChildren(pedigreeView(ind, pedBody.clientWidth));
      // the member itself is on the right; start there
      pedBody.scrollLeft = pedBody.scrollWidth;
    });
  }

  // the member's episode and network, in the epoch tab.step, and its line of
  // descent. Made once per tab and kept across redraws, so an episode that's playing
  // isn't restarted by every new epoch; hidden for runs that didn't log them
  function memberCards(tab) {
    if (tab.cards) return tab.cards;
    const card = () => {
      const c = { hdr: h("div", { class: "card-hdr" }), body: h("div", { class: "media-body" }), foot: h("div"), shown: null };
      c.el = h("div", { class: "card" }, c.hdr, c.body, c.foot);
      c.el.hidden = true;
      return c;
    };
    tab.cards = { episode: card(), descent: card() };
    // the episode and the network in one card, like the run's test episode: the
    // episode (card.shows, a player once it's loaded) beside the network
    // (card.network, drawn into card.net), or whichever there is
    const episode = tab.cards.episode;
    episode.net = { body: h("div", { class: "ep-net" }), foot: h("div") };
    episode.netEl = h("div", {}, episode.net.body, episode.net.foot);
    episode.grid = h("div", { class: "ep-grid" });
    episode.body.append(episode.grid);
    return tab.cards;
  }

  function loadMemberCards(tab) {
    loadNetwork(tab, tab.cards.episode);
    loadEpisodes(tab, tab.cards.episode);
    loadDescent(tab, tab.cards.descent);
  }

  // the member's episode card with what it has of the episode and network
  function placeMember(tab, card) {
    const parts = [card.shows, card.network && card.netEl].filter(Boolean);
    card.el.hidden = !parts.length;
    // (moved only when they change, so a player that's playing carries on)
    const kids = card.grid.children;
    if (kids.length !== parts.length || parts.some((el, i) => kids[i] !== el)) card.grid.replaceChildren(...parts);
    // a wide network goes in a column beside the episode
    card.grid.classList.toggle("wide", parts.length > 1 && wideNets());
    if (card.player) card.player.fullEl = card.network ? card.grid : null;
    const step = card.shows ? card.episodeStep : tab.step;
    card.hdr.replaceChildren(h("span", { class: "title", text: card.shows ? "sample episode" : "network" }), h("span", { class: "spacer" }), h("span", { class: "latest", text: `${w.step} ${step}` }));
  }

  // the member's network or episodes ("network" or "episodes") in tab.step: null
  // if there are none, undefined if the reader moved on while fetching
  async function memberData(tab, kind) {
    const step = tab.step;
    const key = `${kind}@${step}`;
    if (!tab.data[key]) {
      const data = await (kind === "network" ? source.network(tab.id, step) : source.memberEpisodes(tab.id, step));
      if (data) tab.data[key] = data;
      if (dead || !state.tabs.includes(tab) || step !== tab.step) return undefined;
      return data;
    }
    return tab.data[key];
  }

  async function loadNetwork(tab, card) {
    const net = await memberData(tab, "network");
    if (net === undefined) return;
    card.network = net;
    placeMember(tab, card);
    if (net) requestAnimationFrame(() => linkMember(tab));
  }

  // the member's network, lit up by what it did at each frame of its episode when
  // the episode recorded its observations
  function linkMember(tab) {
    const { episode } = tab.cards;
    if (!episode.network) return;
    const player = episode.player;
    const row = player && episode.rows.find((r) => r.id === tab.id && r.seat === player.seat);
    drawNetwork(episode.net, episode.network, row ? { arrays: episode.arrays, row, player } : null);
  }

  async function loadEpisodes(tab, card) {
    const eps = await memberData(tab, "episodes");
    if (eps === undefined) return;
    if (!eps?.episodes.length) {
      Object.assign(card, { shows: null, player: null, shown: null });
      card.foot.replaceChildren();
      placeMember(tab, card);
      return;
    }
    // a two-player task shows each side the same view of the game (slimevolley
    // mirrors observations), so one game is enough: the member's own, the one it's
    // seated first in, unless it was opened on another
    const ep =
      eps.episodes.find((e) => e.index === tab.game) ||
      eps.episodes.find((e) => e.seat === 0) ||
      eps.episodes[0];
    const at = { step: eps.step, game: ep.index };
    // games of several players: say who played which seat, in the order and colours
    // of the picture; the others open on this same game
    const player = (p, seat) =>
      seatPlayer(
        p.id === tab.id
          ? h("b", {}, nameEl(p.name || `#${p.id}`))
          : p.name
            ? genomeLink(p.id, p.name, at)
            : `#${p.id}`,
        seat,
        eps.env,
        p.reward,
      );
    // a new player only when the episode changes, so one that's playing carries on
    const src = source.memberEpisodeUrl(eps, ep);
    if (card.shown === src) return;
    card.shown = src;
    const players = ep.players.length > 1 && matchup(ep.players.map(player));
    card.foot.replaceChildren(...[players].filter(Boolean));
    card.player = null;
    card.episodeStep = eps.step;
    card.shows = h("div", { class: "loading", text: "loading…" });
    placeMember(tab, card);
    const render = RENDERERS[eps.env];
    let arrays;
    try {
      if (!render) throw new Error(`no renderer for ${eps.env}`);
      arrays = await loadArrays(src);
    } catch (e) {
      if (card.shown === src) {
        card.shows = h("div", { class: "media-error", text: `couldn't show this episode: ${e.message}` });
        placeMember(tab, card);
      }
      return;
    }
    if (dead || card.shown !== src) return;
    card.arrays = arrays;
    card.rows = obsRows(arrays, ep.players);
    // the member's own agent, the first of them if it plays several (a flock)
    const own = card.rows.find((r) => r.id === tab.id);
    const view = render(arrays);
    card.player = listen(episodePlayer(view, { seat: own ? own.seat : ep.seat }));
    card.player.onSeat = (seat) => {
      // seats whose observations weren't kept are still someone's
      const id = card.rows.find((r) => r.seat === seat)?.id ?? ep.players[seat]?.id;
      if (id === tab.id) {
        card.player.setSeat(seat);
        linkMember(tab);
      } else if (id != null) {
        // another member's agent: its own tab, on this same game
        openTab(id, { at });
      }
      return id != null;
    };
    card.shows = card.player.el;
    placeMember(tab, card);
    // a race's members are named under their cars; anything else says who played
    // which seat below
    if (carNames(view, card.player, ep.players, { self: tab.id, at })) card.foot.replaceChildren();
    linkMember(tab);
  }

  // The member's line of descent, each ancestor's network blended into the next, and
  // for a 2D classification, the decision boundary it drew. A line never changes, so
  // it's drawn once per tab
  async function loadDescent(tab, card) {
    if (card.shown) return;
    card.shown = true;
    const data = await source.descent(tab.id);
    if (dead || !state.tabs.includes(tab)) return;
    if (!data?.networks.length) {
      card.shown = false;
      return;
    }
    // the boundary over the plane the member's episode covers
    let lim = null;
    const eps = await memberData(tab, "episodes");
    if (eps?.env === "classification_2d" && eps.episodes.length) {
      try {
        const { points } = await loadArrays(source.memberEpisodeUrl(eps, eps.episodes[0]));
        lim = 1.02 * points.data.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
      } catch {
        // drawn without it
      }
    }
    if (dead || !state.tabs.includes(tab)) return;
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
    // "generation 4 · epochs 4–9", or "born generation 4 · alive generations 4–9"
    const born = generations ? "born generation" : "generation";
    const alive = generations ? "alive " : "";
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
          h("div", { class: "row" }, `#${n.id} · ${n.species_name} · ${born} ${n.born} · ${alive}${n.last > n.born ? `${w.steps} ${n.born}–${n.last}` : `${w.step} ${n.born}`}`),
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

  // ------------------------------------------------------------ config

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

  // the member tabs the page had open, then the one it shows
  for (const tab of options.tabs || []) addTab(tab.id, tab.name);
  openTab(options.tab ?? null, { push: false });
  refresh();

  return {
    refresh,
    openTab,
    destroy() {
      dead = true;
      resized.disconnect();
      clearTimeout(resizeTimer);
      clearTimeout(setEpoch.board);
      hideTip();
      root.replaceChildren();
    },
  };
}
