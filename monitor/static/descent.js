// A member's line of descent, animated: the network of each ancestor along it (each
// member's fitter parent, whose structure it inherits), founder first, each blended
// into the next, as a view for episodes.js episodePlayer.
//
// Node ids are genome slots, which offspring inherit from their fitter parent, so a
// node keeps its id along the line and can keep one place throughout: places come
// from the union of every ancestor's graph, laid out once.

import { NdArray, traceNetwork } from "./episodes.js";
import { WIDE_INPUTS, inputGroups } from "./groups.js";
import { ACCENT, BLUE, DIM, EDGE, FAINT, INK, MID, ORANGE, PANEL, PICKED, WELL } from "./palette.js";

// frames between one ancestor and the next, a second's worth of frames, and how
// long the member itself is held before it starts over
const TWEEN = 6, FPS = 24, HOLD = 48;
// a row of the tallest column, the fitness strip below the network, and the
// decision boundary's side
const ROW = 15, STRIP = 30, BOUNDARY = 280;
// the decision boundary's resolution, a side of the grid it's worked out on
const GRID = 72;

const MONO = () => getComputedStyle(document.documentElement).getPropertyValue("--mono").trim() || "monospace";

// Every node's place, [column in [0, 1], height in [0, 1]], over the union of
// nets' graphs ({edges: [[from, to, weight]]}): its column is its longest path
// from the inputs; inputs and outputs are spread evenly, and hidden nodes go to
// the mean place of their neighbours over a few sweeps, pushed apart where they
// crowd, then stretched to use the height
function layout(nets, numIn, numOut) {
  const numIO = numIn + numOut;
  const isOutput = (n) => n >= numIn && n < numIO;
  const nodes = new Set([...Array(numIO).keys()]);
  const edges = new Map();
  for (const net of nets) {
    for (const [a, b] of net.edges) {
      nodes.add(a);
      nodes.add(b);
      edges.set(`${a}>${b}`, [a, b]);
    }
  }
  const depth = new Map([...nodes].map((n) => [n, n < numIn ? 0 : 1]));
  // bounded, as the union of acyclic graphs can still have a cycle
  for (let pass = 0; pass < nodes.size; pass++) {
    let changed = false;
    for (const [a, b] of edges.values()) {
      if (b >= numIn && !isOutput(b) && depth.get(b) < depth.get(a) + 1) {
        depth.set(b, depth.get(a) + 1);
        changed = true;
      }
    }
    if (!changed) break;
  }
  const hidden = [...nodes].filter((n) => n >= numIO);
  const outDepth = Math.max(1, ...hidden.map((n) => depth.get(n) + 1));
  for (let n = numIn; n < numIO; n++) depth.set(n, outDepth);
  const columns = Array.from({ length: outDepth + 1 }, () => []);
  for (const n of [...nodes].sort((a, b) => a - b)) columns[depth.get(n)].push(n);

  const pos = new Map();
  for (const col of columns) col.forEach((n, i) => pos.set(n, (i + 0.5) / col.length));
  const near = new Map([...nodes].map((n) => [n, []]));
  for (const [a, b] of edges.values()) {
    near.get(a).push(b);
    near.get(b).push(a);
  }
  const widest = Math.max(1, ...columns.slice(1, -1).map((c) => c.length));
  const gap = Math.min(0.12, 0.9 / widest);
  for (let sweep = 0; sweep < 12; sweep++) {
    for (let j = 1; j < outDepth; j++) {
      const col = columns[sweep % 2 === 0 ? j : outDepth - j];
      for (const n of col) {
        const nb = near.get(n);
        if (nb.length) pos.set(n, nb.reduce((s, m) => s + pos.get(m), 0) / nb.length);
      }
      col.sort((a, b) => pos.get(a) - pos.get(b));
      // pushed apart, downwards then back up, within [0.04, 0.96]
      for (let i = 1; i < col.length; i++) pos.set(col[i], Math.max(pos.get(col[i]), pos.get(col[i - 1]) + gap));
      const over = col.length ? pos.get(col.at(-1)) - 0.96 : 0;
      if (over > 0) for (const n of col) pos.set(n, pos.get(n) - over);
      for (let i = col.length - 2; i >= 0; i--) pos.set(col[i], Math.min(pos.get(col[i]), pos.get(col[i + 1]) - gap));
      const under = col.length ? 0.04 - pos.get(col[0]) : 0;
      if (under > 0) for (const n of col) pos.set(n, pos.get(n) + under);
    }
  }
  if (hidden.length > 1) {
    const ys = hidden.map((n) => pos.get(n));
    const lo = Math.min(...ys), hi = Math.max(...ys), mid = (lo + hi) / 2;
    const k = Math.min(4, 0.8 / Math.max(1e-6, hi - lo));
    if (k > 1) for (const n of hidden) pos.set(n, 0.5 + (pos.get(n) - mid) * k);
  }
  const rows = Math.max(...columns.map((c) => c.length));
  return { place: new Map([...nodes].map((n) => [n, [depth.get(n) / outDepth, pos.get(n)]])), rows };
}

// a network of the line as episodes.js traceNetwork runs it
function runnable(net, data) {
  const numIn = data.input_size, numIO = numIn + data.output_size;
  return {
    nodes: net.nodes.map(([id, act, bias]) => ({
      id,
      kind: id < numIn ? "input" : id < numIO ? "output" : "hidden",
      activation: data.activations[act],
      bias,
    })),
    edges: net.edges.map(([from, to, weight]) => ({ from, to, weight })),
  };
}

// each network's decision boundary over [-lim, lim]²: the sigmoid of its (first)
// output on a GRID × GRID grid, top row first, or null where it can't be run
function boundaries(data, lim) {
  const grid = new Float32Array(GRID * GRID * 2);
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      grid[(r * GRID + c) * 2] = -lim + ((c + 0.5) / GRID) * 2 * lim;
      grid[(r * GRID + c) * 2 + 1] = lim - ((r + 0.5) / GRID) * 2 * lim;
    }
  }
  const obs = new NdArray(grid, [GRID * GRID, 2]);
  return data.networks.map((net) => {
    const trace = traceNetwork(runnable(net, data), obs);
    if (trace.error) return null;
    const out = net.nodes.findIndex(([id]) => id === data.input_size);
    const p = new Float32Array(GRID * GRID);
    for (let i = 0; i < p.length; i++) p[i] = 1 / (1 + Math.exp(-trace.values[i * trace.nodes + out]));
    return p;
  });
}

const RGB = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

// The line of descent in data (the server's lineage_networks) drawn `width` wide;
// with `lim`, beside each network its decision boundary over [-lim, lim]². Besides
// what episodePlayer needs, at(frame) is the ancestor shown at a frame (nearest,
// mid-blend): {ancestor (of data.line), hidden, connections}.
export function descentView(data, width, { lim = null } = {}) {
  const numIn = data.input_size, numOut = data.output_size, numIO = numIn + numOut;
  const nets = data.networks;
  const keys = nets.length;
  const { place, rows } = layout(nets, numIn, numOut);
  const inLabels = data.input_labels || [...Array(numIn).keys()].map(String);
  const outLabels = data.output_labels || [...Array(numOut).keys()].map(String);

  // every node and edge of the union, and whether each network has it
  const nodeIds = [...place.keys()];
  const nodeOn = nets.map((net) => {
    const on = new Set(net.nodes.map(([id]) => id));
    return Uint8Array.from(nodeIds, (n) => on.has(n));
  });
  const edgeIndex = new Map();
  for (const net of nets) for (const [a, b] of net.edges) if (!edgeIndex.has(`${a}>${b}`)) edgeIndex.set(`${a}>${b}`, [edgeIndex.size, a, b]);
  const edgeList = [...edgeIndex.values()];
  const weight = nets.map(() => new Float32Array(edgeList.length));
  const edgeOn = nets.map(() => new Uint8Array(edgeList.length));
  nets.forEach((net, k) => {
    for (const [a, b, w] of net.edges) {
      const [i] = edgeIndex.get(`${a}>${b}`);
      weight[k][i] = w;
      edgeOn[k][i] = 1;
    }
  });
  // edges drawn as strong as they get at the 95th percentile of every weight on
  // the line, so one outlier doesn't wash the rest out
  const all = nets.flatMap((net) => net.edges.map((e) => Math.abs(e[2]))).sort((a, b) => a - b);
  const maxW = all.length ? Math.max(1e-6, all[Math.min(all.length - 1, Math.floor(0.95 * all.length))]) : 1;
  const counts = nets.map((net) => ({
    hidden: net.nodes.filter(([id]) => id >= numIO).length,
    connections: net.edges.length,
  }));

  const fields = lim != null ? boundaries(data, lim) : null;
  const font = `10px ${MONO()}`;
  const labelW = (labels) => Math.min(160, Math.max(0, ...labels.map((s) => s.length)) * 6.2 + 12);
  // many inputs (F1TENTH's 62) go close together, labelled by group: lidar,
  // car 0, ...; inputs of their own singly
  const wide = numIn > WIDE_INPUTS;
  const groups = wide ? inputGroups([...Array(numIn).keys()], (n) => inLabels[n]) : null;
  const groupLabels = groups?.flatMap((g) => (g.name != null && g.members.length > 1 ? [g.name] : g.members.map((n) => inLabels[n])));
  const side = fields ? BOUNDARY : 0;
  const netH = Math.max(side, 220, rows * (wide ? 7 : ROW) + 16);
  const W = Math.max(width, side + 260);
  const H = netH + 12 + STRIP;
  const x0 = (side ? side + 24 : 0) + labelW(groupLabels ?? inLabels), x1 = W - labelW(outLabels);
  const y0 = 10, y1 = netH - 10;
  const xy = (n) => {
    const [cx, cy] = place.get(n);
    return [x0 + cx * (x1 - x0), y0 + cy * (y1 - y0)];
  };

  // fitness along the whole line, by the generation each ancestor was born in
  const line = data.line;
  const fits = line.map((a) => a.fitness).filter((f) => f != null);
  const fLo = Math.min(...fits), fHi = Math.max(...fits);
  const sx0 = 52, sx1 = W - 6, sy0 = netH + 14, sy1 = H - 4;
  const gLo = line[0].born, gHi = line.at(-1).born;
  const stripX = (born) => (gHi > gLo ? sx0 + ((born - gLo) / (gHi - gLo)) * (sx1 - sx0) : (sx0 + sx1) / 2);
  const stripY = (f) => (fHi > fLo ? sy1 - ((f - fLo) / (fHi - fLo)) * (sy1 - sy0) : (sy0 + sy1) / 2);

  // the boundary, blended, drawn into a GRID-sized canvas and scaled up
  const tile = fields && document.createElement("canvas");
  if (tile) tile.width = tile.height = GRID;
  const tileCtx = tile?.getContext("2d");
  const pixels = tileCtx?.createImageData(GRID, GRID);
  const [blue, orange, well] = [BLUE, ORANGE, WELL].map(RGB);

  const frames = keys > 1 ? (keys - 1) * TWEEN + 1 + HOLD : 1;
  // the networks a frame blends, and how far it is from the first to the second
  const at = (frame) => {
    const f = frame / TWEEN;
    if (f >= keys - 1) return [keys - 1, keys - 1, 0];
    const k = Math.floor(f);
    return [k, k + 1, f - k];
  };

  function drawBoundary(ctx, a, b, t) {
    const fa = fields[a], fb = fields[b];
    if (!fa && !fb) return;
    const d = pixels.data;
    for (let i = 0; i < GRID * GRID; i++) {
      const p = fa && fb ? fa[i] + (fb[i] - fa[i]) * t : (fa || fb)[i];
      // orange where it says class 1, blue for class 0, washed out
      for (let c = 0; c < 3; c++) d[i * 4 + c] = 0.38 * (blue[c] + (orange[c] - blue[c]) * p) + 0.62 * well[c];
      d[i * 4 + 3] = 255;
    }
    tileCtx.putImageData(pixels, 0, 0);
    const top = (netH - side) / 2;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tile, 0, top, side, side);
    ctx.strokeStyle = EDGE;
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, top + 0.5, side - 1, side - 1);
  }

  function drawStrip(ctx, a, b, t) {
    ctx.font = font;
    ctx.fillStyle = DIM;
    ctx.textAlign = "left";
    ctx.fillText("fitness", 0, (sy0 + sy1) / 2 + 3.5);
    ctx.strokeStyle = FAINT;
    ctx.lineWidth = 1;
    ctx.beginPath();
    let pen = false;
    for (const anc of line) {
      if (anc.fitness == null) {
        pen = false;
        continue;
      }
      ctx[pen ? "lineTo" : "moveTo"](stripX(anc.born), stripY(anc.fitness));
      pen = true;
    }
    ctx.stroke();
    // the ancestor shown, moving along the line between the two blended
    const li = nets[a].index + (nets[b].index - nets[a].index) * t;
    const lo = line[Math.floor(li)], hi = line[Math.ceil(li)], u = li - Math.floor(li);
    if (lo.fitness == null || hi.fitness == null) return;
    ctx.fillStyle = PICKED;
    ctx.beginPath();
    ctx.arc(stripX(lo.born + (hi.born - lo.born) * u), stripY(lo.fitness + (hi.fitness - lo.fitness) * u), 3.5, 0, 2 * Math.PI);
    ctx.fill();
  }

  function draw(ctx, frame) {
    const [a, b, t] = at(frame);
    ctx.clearRect(0, 0, W, H);
    if (fields) drawBoundary(ctx, a, b, t);

    // edges fade in and out, and their weights move from one to the next
    const shown = [];
    for (const [i, from, to] of edgeList) {
      const onA = edgeOn[a][i], onB = edgeOn[b][i];
      const alpha = onA + (onB - onA) * t;
      if (alpha <= 0.01) continue;
      const wA = onA ? weight[a][i] : weight[b][i];
      const wB = onB ? weight[b][i] : wA;
      shown.push([wA + (wB - wA) * t, alpha, from, to]);
    }
    // the strongest on top
    shown.sort((p, q) => Math.abs(p[0]) - Math.abs(q[0]));
    for (const [w, alpha, from, to] of shown) {
      const s = Math.min(1, Math.abs(w) / maxW);
      const [ax, ay] = xy(from), [bx, by] = xy(to);
      const mx = (ax + bx) / 2;
      ctx.globalAlpha = alpha * (0.3 + 0.55 * s);
      ctx.strokeStyle = w >= 0 ? BLUE : ORANGE;
      ctx.lineWidth = 0.75 + 2.25 * s;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.bezierCurveTo(mx, ay, mx, by, bx, by);
      ctx.stroke();
    }

    // nodes as the network card draws them (style.css .net .node)
    ctx.font = font;
    nodeIds.forEach((n, i) => {
      const alpha = nodeOn[a][i] + (nodeOn[b][i] - nodeOn[a][i]) * t;
      if (alpha <= 0.01) return;
      const [x, y] = xy(n);
      const io = n < numIO;
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(x, y, wide && n < numIn ? 2.6 : io ? 5 : 4.5, 0, 2 * Math.PI);
      ctx.fillStyle = n < numIn ? INK : io ? ACCENT : WELL;
      ctx.strokeStyle = io ? PANEL : INK;
      ctx.lineWidth = io ? 2 : 1.25;
      ctx.fill();
      ctx.stroke();
      if (io && !(wide && n < numIn)) {
        ctx.fillStyle = MID;
        ctx.textAlign = n < numIn ? "right" : "left";
        ctx.fillText(n < numIn ? inLabels[n] : outLabels[n - numIn], x + (n < numIn ? -9 : 9), y + 3.5);
      }
    });
    ctx.globalAlpha = 1;
    if (groups) drawGroups(ctx);
    drawStrip(ctx, a, b, t);
  }

  // the inputs' labels when they're close together: a group's name beside its
  // span, with a bracket, and an input of its own's name smaller
  function drawGroups(ctx) {
    ctx.textAlign = "right";
    for (const g of groups) {
      if (g.name != null && g.members.length > 1) {
        const [x, ya] = xy(g.members[0]), [, yb] = xy(g.members.at(-1));
        ctx.font = font;
        ctx.fillStyle = INK;
        ctx.fillText(g.name, x - 9, (ya + yb) / 2 + 3.5);
        ctx.strokeStyle = EDGE;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x - 3, ya - 2);
        ctx.lineTo(x - 6, ya - 2);
        ctx.lineTo(x - 6, yb + 2);
        ctx.lineTo(x - 3, yb + 2);
        ctx.stroke();
      } else {
        ctx.font = `8px ${MONO()}`;
        ctx.fillStyle = MID;
        for (const n of g.members) {
          const [x, y] = xy(n);
          ctx.fillText(inLabels[n], x - 7, y + 3);
        }
      }
    }
  }

  return {
    width: W,
    height: H,
    frames,
    fps: FPS,
    draw,
    // counted in ancestors, not frames
    counter: (frame) => {
      const [a, b, t] = at(frame);
      return `${(t < 0.5 ? a : b) + 1}/${keys}`;
    },
    // the arrow keys step an ancestor at a time
    step: (frame, dir) => {
      const [a, b, t] = at(frame);
      const k = t < 0.5 ? a : b;
      return Math.max(0, Math.min(keys - 1, k + dir)) * TWEEN;
    },
    at: (frame) => {
      const [a, b, t] = at(frame);
      const k = t < 0.5 ? a : b;
      return { ancestor: line[nets[k].index], ...counts[k] };
    },
  };
}
