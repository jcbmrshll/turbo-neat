// Episodes in the browser: the raw arrays a run logged, played frame by frame on a
// canvas by the env's renderer (renderers.js), and what a network was doing at each
// frame, from the observations logged with the episode.

import { h } from "./dom.js";

// ---------------------------------------------------------------- arrays

// an n-dimensional view onto a flat typed array, row-major
export class NdArray {
  constructor(data, shape) {
    this.data = data;
    this.shape = shape;
    this.strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  }
  get length() {
    return this.shape[0];
  }
  get(...index) {
    let at = 0;
    for (let i = 0; i < index.length; i++) at += index[i] * this.strides[i];
    return this.data[at];
  }
  // the array at index i of the first axis
  at(i) {
    const n = this.strides[0];
    return new NdArray(this.data.subarray(i * n, (i + 1) * n), this.shape.slice(1));
  }
}

const DTYPES = { float32: Float32Array, int32: Int32Array, uint8: Uint8Array };

// an episode's arrays, {name: NdArray}, as the server packs them (monitor/server.py
// pack_arrays): a JSON header's length, the header, then each array's bytes
export async function loadArrays(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  const buf = await res.arrayBuffer();
  const length = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, length)));
  const arrays = {};
  for (const [name, { dtype, shape, offset }] of Object.entries(header)) {
    const size = shape.reduce((a, b) => a * b, 1);
    arrays[name] = new NdArray(new DTYPES[dtype](buf, 4 + length + offset, size), shape);
  }
  return arrays;
}

// ---------------------------------------------------------------- player

const SPEEDS = [0.25, 0.5, 1, 2];
const MAX_HEIGHT = 420;

// keys go to the player under the pointer, without clicking it first, unless
// they're being typed somewhere
let hovered = null;
document.addEventListener("keydown", (ev) => {
  if (!hovered || ev.target.closest?.("input, select, textarea")) return;
  if (hovered.key(ev.key)) ev.preventDefault();
});

// A canvas playing view (a renderer's, see renderers.js) with play/pause, a scrubber
// and speeds; space plays and pauses, the arrow keys step a frame, while the pointer
// is over it. player.onFrame(frame) is called with every frame drawn, and
// player.onSeat(seat) when the reader clicks an agent other than the one picked
// out (`seat`, to begin with), for views that can say which is where: true if it
// did something with it. A view can count its frames its own way, counter(frame),
// and say where the arrow keys go from a frame, step(frame, ±1).
export function episodePlayer(view, { seat = null, maxHeight = MAX_HEIGHT } = {}) {
  const canvas = h("canvas");
  canvas.style.aspectRatio = `${view.width} / ${view.height}`;
  // no taller than maxHeight on screen
  canvas.style.maxWidth = `${(maxHeight * view.width) / view.height}px`;
  const ctx = canvas.getContext("2d");
  const still = view.frames <= 1;
  const playBtn = h("button", { class: "play", title: "play/pause (space)" });
  const scrub = h("input", { type: "range", min: 0, max: Math.max(0, view.frames - 1), step: 1, value: 0 });
  const counter = h("span", { class: "pos" });
  const speedBtns = SPEEDS.map((s) => h("button", { text: `${s}×`, onclick: () => setSpeed(s) }));
  const controls = h("div", { class: "scrub player-controls" }, playBtn, scrub, counter, ...speedBtns);
  const el = h("div", { class: "player" }, canvas, !still && controls);

  const player = { el, frame: 0, seat, playing: !still, speed: 1, onFrame: null, onSeat: null };
  let scale = 1;
  // the frame's time at the last play or seek, so playback is real time whatever
  // the display's refresh rate
  let anchor = { time: 0, frame: 0 };

  function draw() {
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    view.draw(ctx, player.frame, { seat: player.seat });
    scrub.value = player.frame;
    counter.textContent = view.counter ? view.counter(player.frame) : `${player.frame + 1}/${view.frames}`;
    player.onFrame?.(player.frame);
  }

  player.setFrame = (frame) => {
    player.frame = ((Math.round(frame) % view.frames) + view.frames) % view.frames;
    anchor = { time: performance.now(), frame: player.frame };
    draw();
  };
  player.setSeat = (s) => {
    player.seat = s;
    draw();
  };
  function setPlaying(on) {
    player.playing = on && !still;
    anchor = { time: performance.now(), frame: player.frame };
    playBtn.textContent = player.playing ? "❚❚" : "▶";
    if (player.playing) loop();
  }
  function setSpeed(s) {
    player.speed = s;
    anchor = { time: performance.now(), frame: player.frame };
    speedBtns.forEach((b, i) => b.classList.toggle("on", SPEEDS[i] === s));
  }
  player.play = () => setPlaying(true);
  player.pause = () => setPlaying(false);

  // draws while playing and on screen; a card that's scrolled away or on a hidden
  // tab costs nothing
  let visible = false, running = false;
  function loop() {
    if (running || !player.playing || !visible) return;
    running = true;
    const tick = (now) => {
      if (!player.playing || !visible) {
        running = false;
        return;
      }
      // a frame's timestamp can be from just before the anchor was set
      const elapsed = Math.max(0, now - anchor.time) / 1000;
      const frame = (anchor.frame + Math.floor(elapsed * view.fps * player.speed)) % view.frames;
      if (frame !== player.frame) {
        player.frame = frame;
        draw();
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  const seen = new IntersectionObserver(([entry]) => {
    // a player taken off the page (for another episode) is done with
    if (!canvas.isConnected) {
      seen.disconnect();
      sized.disconnect();
      if (hovered === player) hovered = null;
      return;
    }
    visible = entry.isIntersecting;
    // pick up where it was, not where the clock would have it
    anchor = { time: performance.now(), frame: player.frame };
    loop();
  });
  seen.observe(canvas);
  // drawn at the canvas's size on screen, so it's sharp at any width
  const sized = new ResizeObserver(() => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    if (width === canvas.width) return;
    canvas.width = width;
    canvas.height = Math.round((width * view.height) / view.width);
    scale = width / view.width;
    draw();
  });
  sized.observe(canvas);

  playBtn.addEventListener("click", () => setPlaying(!player.playing));
  scrub.addEventListener("input", () => {
    setPlaying(false);
    player.setFrame(Number(scrub.value));
  });
  canvas.addEventListener("click", (ev) => {
    const r = canvas.getBoundingClientRect();
    const x = ((ev.clientX - r.left) / r.width) * view.width;
    const y = ((ev.clientY - r.top) / r.height) * view.height;
    // another agent, if there's something to show of it; else play or pause
    const s = view.seatAt?.(x, y, player.frame) ?? null;
    if (s != null && s !== player.seat && player.onSeat?.(s)) return;
    setPlaying(!player.playing);
  });
  player.key = (key) => {
    if (still) return false;
    if (key === " ") setPlaying(!player.playing);
    else if (key === "ArrowRight" || key === "ArrowLeft") {
      setPlaying(false);
      const dir = key === "ArrowRight" ? 1 : -1;
      player.setFrame(view.step ? view.step(player.frame, dir) : player.frame + dir);
    } else return false;
    return true;
  };
  el.addEventListener("pointerenter", () => (hovered = player));
  el.addEventListener("pointerleave", () => hovered === player && (hovered = null));

  setSpeed(1);
  setPlaying(player.playing);
  return player;
}

// ---------------------------------------------------------------- activations

const f32 = Math.fround;

// neat/activations.py, by the names the run's config logged them under
const ACTIVATIONS = {
  iden: (x) => x,
  inv: (x) => 1 / f32(x + 1e-8),
  relu: (x) => Math.max(x, 0),
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  tanh: Math.tanh,
  abs: Math.abs,
  absolute: Math.abs,
  sin: Math.sin,
  cos: Math.cos,
  exp: Math.exp,
};

// What every node of a network (monitor/networks.py describe_network) put out at
// each frame of obs (frames, inputs), the way neat.genome.forward runs it: an
// input's value is its observation, any other node's is its bias plus what flows
// in along its edges, each the source's output times the edge's weight; a node puts
// out its activation of its value. Returns {values: Float32Array (frames × nodes),
// in net.nodes order}, or {error} for a network it can't run.
export function traceNetwork(net, obs) {
  const index = new Map(net.nodes.map((n, i) => [n.id, i]));
  const fns = net.nodes.map((n) => ACTIVATIONS[n.activation]);
  const unknown = net.nodes.find((n, i) => !fns[i]);
  if (unknown) return { error: `can't run the activation ${unknown.activation}` };
  const inputs = net.nodes.filter((n) => n.kind === "input");
  if (obs.shape[1] !== inputs.length) {
    return { error: `${obs.shape[1]} observations for ${inputs.length} inputs` };
  }

  // every node after every node feeding it, whatever order the edges are in
  const incoming = net.nodes.map(() => []);
  const pending = net.nodes.map(() => 0);
  const outgoing = net.nodes.map(() => []);
  for (const e of net.edges) {
    const a = index.get(e.from), b = index.get(e.to);
    if (a == null || b == null) continue;
    incoming[b].push([a, e.weight]);
    outgoing[a].push(b);
    pending[b]++;
  }
  const order = net.nodes.map((_, i) => i).filter((i) => !pending[i]);
  for (let k = 0; k < order.length; k++) {
    for (const b of outgoing[order[k]]) if (!--pending[b]) order.push(b);
  }
  if (order.length < net.nodes.length) return { error: "the network has a cycle" };

  const N = net.nodes.length, frames = obs.shape[0];
  const values = new Float32Array(frames * N);
  for (let t = 0; t < frames; t++) {
    const row = t * N;
    for (const i of order) {
      const node = net.nodes[i];
      // in float32 at every step, like jax, which matters where a node blows up
      // (inv near 0, exp): float64 would keep what float32 loses next to it
      let x;
      if (node.kind === "input") x = obs.get(t, node.id);
      else {
        x = f32(node.bias);
        for (const [a, w] of incoming[i]) x = f32(x + f32(values[row + a] * w));
      }
      values[row + i] = fns[i](x);
    }
  }
  return { values, nodes: N, frames };
}
