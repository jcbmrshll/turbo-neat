// A 2D classifier's test (renderers.js classification2d: its points, coloured by
// the class it predicted) over the decision surface its network draws across the
// whole plane. The network is run in the browser (episodes.js traceNetwork) on a
// grid of points and the test's own, so the network beside it can light up with
// what it does at whatever point is under the pointer.

import { h } from "./dom.js";
import { NdArray, traceNetwork } from "./episodes.js";
import { BLUE, DIM, EDGE, EDGE_SOFT, INK, ORANGE, PICKED, WELL } from "./palette.js";

// the surface is sampled this many times across
const GRID = 72;
const W = 400, H = 400, PAD = { l: 30, r: 10, t: 10, b: 24 };
// drawn at this many pixels to a logical one, for sharp points on any screen
const RES = 2;

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const RGB = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const [POS, NEG, ZERO] = [ORANGE, BLUE, WELL].map(RGB);

function niceTicks(lo, hi, count) {
  const step0 = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(v);
  return ticks;
}

// arrays.points: (n, 2); arrays.probability: (n,); net: the tested network.
// Returns {el, trace, frame, onPoint} (onPoint(frame) is called with the trace's
// frame for the point under the pointer), or {error}
export function surfaceView({ points, probability }, net) {
  const n = points.length;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n * 2; i++) {
    lo = Math.min(lo, points.data[i]);
    hi = Math.max(hi, points.data[i]);
  }
  const ticks = niceTicks(lo, hi, 5);
  [lo, hi] = [Math.min(lo, ticks[0]), Math.max(hi, ticks.at(-1))];
  const size = Math.min(W - PAD.l - PAD.r, H - PAD.t - PAD.b);
  const sx = (v) => PAD.l + ((v - lo) / (hi - lo)) * size;
  const sy = (v) => PAD.t + size - ((v - lo) / (hi - lo)) * size;
  const wx = (px) => lo + ((px - PAD.l) / size) * (hi - lo);
  const wy = (py) => lo + ((PAD.t + size - py) / size) * (hi - lo);

  // the grid's cell centres, row by row from the top, then the test's points
  const cell = (hi - lo) / GRID;
  const obs = new Float32Array((GRID * GRID + n) * 2);
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      const k = (r * GRID + c) * 2;
      obs[k] = lo + (c + 0.5) * cell;
      obs[k + 1] = hi - (r + 0.5) * cell;
    }
  }
  obs.set(points.data.subarray(0, n * 2), GRID * GRID * 2);
  const trace = traceNetwork(net, new NdArray(obs, [GRID * GRID + n, 2]));
  if (trace.error) return { error: trace.error };
  const out = net.nodes.findIndex((nd) => nd.kind === "output");
  const prob = (frame) => sigmoid(trace.values[frame * trace.nodes + out]);

  // how far the browser's forward pass is from the run's own, on the test's points
  let worst = 0;
  for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(prob(GRID * GRID + i) - probability.get(i)));
  if (worst > 0.02) console.warn(`decision surface: off from the run's predictions by up to ${worst.toFixed(3)}`);

  // the surface, a pixel a cell, scaled up smooth
  const img = document.createElement("canvas");
  img.width = img.height = GRID;
  const ictx = img.getContext("2d");
  const data = ictx.createImageData(GRID, GRID);
  for (let k = 0; k < GRID * GRID; k++) {
    const u = 2 * prob(k) - 1;
    const to = u >= 0 ? POS : NEG;
    const f = 0.42 * Math.abs(u);
    for (let j = 0; j < 3; j++) data.data[k * 4 + j] = Math.round(ZERO[j] + (to[j] - ZERO[j]) * f);
    data.data[k * 4 + 3] = 255;
  }
  ictx.putImageData(data, 0, 0);

  const canvas = h("canvas", { width: W * RES, height: H * RES, class: "surface" });
  canvas.style.aspectRatio = `${W} / ${H}`;
  canvas.style.maxWidth = `${W * 1.05}px`;
  const ctx = canvas.getContext("2d");
  const view = { el: h("div", { class: "player" }, canvas), trace, frame: Math.floor(GRID * GRID / 2 + GRID / 2), onPoint: null };
  let at = null;

  function draw() {
    ctx.setTransform(RES, 0, 0, RES, 0, 0);
    ctx.fillStyle = WELL;
    ctx.fillRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(img, sx(lo), sy(hi), size, size);
    ctx.lineWidth = 0.5;
    ctx.strokeStyle = EDGE_SOFT;
    ctx.font = "7.5px ui-monospace, Menlo, monospace";
    ctx.fillStyle = DIM;
    for (const v of ticks) {
      ctx.beginPath();
      ctx.moveTo(sx(v), PAD.t);
      ctx.lineTo(sx(v), PAD.t + size);
      ctx.moveTo(PAD.l, sy(v));
      ctx.lineTo(PAD.l + size, sy(v));
      ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(String(+v.toPrecision(6)), sx(v), PAD.t + size + 12);
      ctx.textAlign = "right";
      ctx.fillText(String(+v.toPrecision(6)), PAD.l - 4, sy(v) + 3);
    }
    ctx.strokeStyle = EDGE;
    ctx.beginPath();
    ctx.moveTo(PAD.l, PAD.t + size);
    ctx.lineTo(PAD.l + size, PAD.t + size);
    ctx.stroke();
    ctx.globalAlpha = 0.85;
    for (let i = 0; i < n; i++) {
      ctx.beginPath();
      ctx.arc(sx(points.get(i, 0)), sy(points.get(i, 1)), 1.7, 0, 2 * Math.PI);
      ctx.fillStyle = probability.get(i) > 0.5 ? ORANGE : BLUE;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (at) {
      ctx.beginPath();
      ctx.arc(sx(at.x), sy(at.y), 5, 0, 2 * Math.PI);
      ctx.strokeStyle = PICKED;
      ctx.lineWidth = 2;
      ctx.stroke();
      const p = prob(view.frame);
      ctx.font = "9px ui-monospace, Menlo, monospace";
      ctx.textAlign = "left";
      ctx.fillStyle = INK;
      ctx.fillText(`(${at.x.toFixed(2)}, ${at.y.toFixed(2)})  p = ${p.toFixed(3)}`, PAD.l + 6, PAD.t + 12);
    }
  }

  canvas.addEventListener("pointermove", (ev) => {
    const r = canvas.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W, py = ((ev.clientY - r.top) / r.height) * H;
    const x = wx(px), y = wy(py);
    if (x < lo || x > hi || y < lo || y > hi) return;
    const c = Math.min(GRID - 1, Math.floor((x - lo) / cell)), row = Math.min(GRID - 1, Math.floor((hi - y) / cell));
    // a test point under the pointer, else the grid cell
    let frame = row * GRID + c, best = ((hi - lo) / 120) ** 2;
    for (let i = 0; i < n; i++) {
      const d = (points.get(i, 0) - x) ** 2 + (points.get(i, 1) - y) ** 2;
      if (d < best) [best, frame] = [d, GRID * GRID + i];
    }
    at = frame >= GRID * GRID ? { x: points.get(frame - GRID * GRID, 0), y: points.get(frame - GRID * GRID, 1) } : { x, y };
    view.frame = frame;
    view.onPoint?.(frame);
    draw();
  });
  draw();
  return view;
}
