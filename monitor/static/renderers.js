// Episode renderers, one per environment, drawing on a canvas in the dashboard's
// palette.
//
// A training run logs Episode(env, **arrays) (monitor/client.py); the dashboard
// fetches the arrays (episodes.js loadArrays) and calls RENDERERS[env](arrays) for a
// view of them:
//
//   {
//     width, height,  // its size, in the units draw() draws in
//     frames, fps,    // how many frames, and how many a second at 1x
//     draw(ctx, frame, { seat }),  // draw a frame; seat is the agent whose network
//                                  // is shown, to pick out, or null
//     seatAt(x, y, frame),         // optional: the agent at a point, or null
//   }
//
// Arrays are NdArrays (episodes.js), floats as float32. A view should do any
// per-episode work (a track to draw once, say) when it's made, not every frame.

import { f1tenth } from "./f1tenth.js";
import { ACCENT, BLUE, DIM, EDGE, EDGE_SOFT, INK, MID, ORANGE, PICKED, WELL } from "./palette.js";

export const RENDERERS = {
  boids,
  slimevolley,
  cartpole,
  classification_2d: classification2d,
  f1tenth,
  f1tenth_field: f1tenth,
};

function disc(ctx, x, y, r, fill) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = fill;
  ctx.fill();
}

function ring(ctx, x, y, r, width = 2) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.strokeStyle = PICKED;
  ctx.lineWidth = width;
  ctx.stroke();
}

// ---------------------------------------------------------------- boids

// evojax's flocking screen, and its fish outline (nose along +x, in half-fish units)
const BOIDS_W = 400, BOIDS_H = 300, FISH_SIZE = 20;
const FISH_OUTLINE = [
  [-1, -1 / 3],
  [-3 / 5, -1 / 4],
  [0, -1 / 2],
  [2 / 3, -1 / 3],
  [1, 0],
  [2 / 3, 1 / 3],
  [0, 1 / 2],
  [-3 / 5, 1 / 4],
  [-1, 1 / 3],
];
const FISH_COLORS = [INK, ACCENT, MID];
// evojax draws fish FISH_SIZE wide on its 2x canvas
const FISH_HALF = Math.floor((FISH_SIZE + 1) / 2) / 2;

// arrays.boids: (steps, boids, 3) of x, y in [0, 1] and heading theta
function boids({ boids }) {
  const [frames, count] = boids.shape;
  const where = (t, i) => [boids.get(t, i, 0) * BOIDS_W, (1 - boids.get(t, i, 1)) * BOIDS_H];
  return {
    width: BOIDS_W,
    height: BOIDS_H,
    frames,
    fps: 25,
    draw(ctx, t, { seat }) {
      ctx.fillStyle = WELL;
      ctx.fillRect(0, 0, BOIDS_W, BOIDS_H);
      for (let i = 0; i < count; i++) {
        const [x, y] = where(t, i);
        const theta = boids.get(t, i, 2);
        const cos = Math.cos(theta), sin = Math.sin(theta);
        ctx.beginPath();
        for (const [ox, oy] of FISH_OUTLINE) {
          const dx = ox * FISH_HALF, dy = oy * FISH_HALF;
          ctx.lineTo(x + cos * dx + sin * dy, y - sin * dx + cos * dy);
        }
        ctx.closePath();
        ctx.fillStyle = FISH_COLORS[i % FISH_COLORS.length];
        ctx.fill();
      }
      if (seat != null && seat < count) ring(ctx, ...where(t, seat), FISH_HALF + 4);
    },
    seatAt(x, y, t) {
      let best = null, bestD = (FISH_HALF + 6) ** 2;
      for (let i = 0; i < count; i++) {
        const [fx, fy] = where(t, i);
        const d = (fx - x) ** 2 + (fy - y) ** 2;
        if (d < bestD) [best, bestD] = [i, d];
      }
      return best;
    },
  };
}

// ---------------------------------------------------------------- slimevolley

// evojax's slimevolley court, in world units: x in [-24, 24], y up from the floor
const COURT_W = 48, GROUND_H = 1.5, FENCE_W = 1, FENCE_H = 3.5;
// evojax shows a 2:1 window onto the court, so 24 units of height
const SLIME_W = 600, SLIME_H = 300;

// arrays.ball: (steps, 3) of x, y, r; arrays.left and arrays.right: (steps, 5) of
// x, y, r, direction, lives
function slimevolley({ ball, left, right }) {
  const scale = SLIME_W / COURT_W;
  const px = (x, y) => [(x + COURT_W / 2) * scale, SLIME_H - y * scale];
  const worldDisc = (ctx, x, y, r, fill) => disc(ctx, ...px(x, y), r * scale, fill);
  const sides = [
    [left, BLUE],
    [right, ORANGE],
  ];
  return {
    width: SLIME_W,
    height: SLIME_H,
    frames: ball.length,
    fps: 25,
    draw(ctx, t, { seat }) {
      ctx.fillStyle = WELL;
      ctx.fillRect(0, 0, SLIME_W, SLIME_H);
      // the fence, with its rounded cap
      const [fx0, fy0] = px(-FENCE_W / 2, FENCE_H);
      const [fx1, fy1] = px(FENCE_W / 2, GROUND_H);
      ctx.fillStyle = MID;
      ctx.fillRect(fx0, fy0, fx1 - fx0, fy1 - fy0);
      worldDisc(ctx, 0, FENCE_H, FENCE_W / 2, MID);
      const [bx, by] = [ball.get(t, 0), ball.get(t, 1)];
      sides.forEach(([agent, color], side) => {
        const [x, y, r, direction, lives] = [0, 1, 2, 3, 4].map((k) => agent.get(t, k));
        const [cx, cy] = px(x, y);
        ctx.beginPath();
        ctx.arc(cx, cy, r * scale, Math.PI, 2 * Math.PI);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
        // an eye that tracks the ball, like evojax's
        const angle = (Math.PI * (direction === 1 ? 120 : 60)) / 180;
        const ex = x + 0.6 * r * Math.cos(angle), ey = y + 0.6 * r * Math.sin(angle);
        const dx = bx - ex, dy = by - ey;
        const dist = Math.max(Math.hypot(dx, dy), 1e-6);
        worldDisc(ctx, ex, ey, 0.3 * r, WELL);
        worldDisc(ctx, ex + (0.15 * r * dx) / dist, ey + (0.15 * r * dy) / dist, 0.1 * r, INK);
        // spare lives along the top, from each player's own wall
        for (let i = 1; i < Math.round(lives); i++) {
          worldDisc(ctx, direction * (COURT_W / 2 + 0.5 - i * 2), SLIME_H / scale - 1.5, 0.5, color);
        }
        if (seat === side) {
          ctx.beginPath();
          ctx.arc(cx, cy, r * scale + 4, Math.PI, 2 * Math.PI);
          ctx.strokeStyle = PICKED;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      });
      worldDisc(ctx, bx, by, ball.get(t, 2), INK);
      // the floor goes on last, over the bottom of the slimes
      const [gx0, gy0] = px(-COURT_W / 2, GROUND_H);
      const [gx1, gy1] = px(COURT_W / 2, 0);
      ctx.fillStyle = EDGE;
      ctx.fillRect(gx0, gy0, gx1 - gx0, gy1 - gy0);
    },
    seatAt(x, y, t) {
      for (const [side, [agent]] of sides.entries()) {
        const [cx, cy] = px(agent.get(t, 0), agent.get(t, 1));
        if (Math.hypot(x - cx, y - cy) < agent.get(t, 2) * scale * 1.5) return side;
      }
      return null;
    },
  };
}

// ---------------------------------------------------------------- cartpole

// evojax's cart-pole, in world units: the cart's track is x in [-2.4, 2.4], and the
// pole is 0.6 long and upright at theta = 0
const CART_X_LIMIT = 2.4, POLE_LEN = 0.6;
const CART_W = 0.4, CART_H = 0.2, WHEEL_R = 0.05;
// a 6.4 x 2 window onto the track, drawn at 100 px per unit like evojax
const CART_SCALE = 100, CARTPOLE_W = 640, CARTPOLE_H = 200;

// arrays.state: (steps, 4) of x, x_dot, theta, theta_dot; examples/cartpole.py keeps
// every 4th of evojax's 0.01 s steps, so 25 a second is real time
function cartpole({ state }) {
  const s = CART_SCALE;
  // the pole swings a full circle about its pivot, 0.35 above the track; center it
  const trackY = CARTPOLE_H - 0.65 * s;
  const px = (wx, wy) => [CARTPOLE_W / 2 + wx * s, trackY - wy * s];
  const worldDisc = (ctx, wx, wy, r, fill) => disc(ctx, ...px(wx, wy), r * s, fill);
  const line = (ctx, a, b, color, width) => {
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  };
  return {
    width: CARTPOLE_W,
    height: CARTPOLE_H,
    frames: state.length,
    fps: 25,
    draw(ctx, t) {
      const x = state.get(t, 0), theta = state.get(t, 2);
      ctx.fillStyle = WELL;
      ctx.fillRect(0, 0, CARTPOLE_W, CARTPOLE_H);
      ctx.lineCap = "butt";
      // the track, with stops where the cart goes out of bounds
      line(ctx, px(-CART_X_LIMIT, 0), px(CART_X_LIMIT, 0), EDGE, 2);
      for (const side of [-1, 1]) {
        const [sx, sy] = px(side * CART_X_LIMIT, 0);
        line(ctx, [sx, sy - 0.1 * s], [sx, sy + 0.1 * s], MID, 2);
      }
      // the cart rides on its wheels, with the pole's pivot on top
      const bodyY = WHEEL_R + CART_H / 2;
      for (const side of [-1, 1]) worldDisc(ctx, x + (side * CART_W) / 3, WHEEL_R, WHEEL_R, MID);
      const [x0, y0] = px(x - CART_W / 2, bodyY + CART_H / 2);
      ctx.beginPath();
      ctx.roundRect(x0, y0, CART_W * s, CART_H * s, 0.03 * s);
      ctx.fillStyle = BLUE;
      ctx.fill();
      const pivotY = bodyY + CART_H / 2;
      const tip = [x + POLE_LEN * Math.sin(theta), pivotY + POLE_LEN * Math.cos(theta)];
      line(ctx, px(x, pivotY), px(...tip), ORANGE, 0.06 * s);
      // round off the pole's end, and pin it to the cart
      worldDisc(ctx, ...tip, 0.03, ORANGE);
      worldDisc(ctx, x, pivotY, 0.035, INK);
    },
  };
}

// ---------------------------------------------------------------- 2d classification

// arrays.points: (n, 2); arrays.probability: (n,) of the predicted class. A still,
// styled like the dashboard's charts: hairline grid, one baseline, mono ticks
function classification2d({ points, probability }) {
  const W = 400, H = 400, pad = { l: 30, r: 10, t: 10, b: 24 };
  const n = points.length;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n * 2; i++) {
    lo = Math.min(lo, points.data[i]);
    hi = Math.max(hi, points.data[i]);
  }
  const ticks = niceTicks(lo, hi, 5);
  [lo, hi] = [Math.min(lo, ticks[0]), Math.max(hi, ticks.at(-1))];
  // both axes to one scale
  const size = Math.min(W - pad.l - pad.r, H - pad.t - pad.b);
  const sx = (v) => pad.l + ((v - lo) / (hi - lo)) * size;
  const sy = (v) => pad.t + size - ((v - lo) / (hi - lo)) * size;
  return {
    width: W,
    height: H,
    frames: 1,
    fps: 1,
    draw(ctx) {
      ctx.fillStyle = WELL;
      ctx.fillRect(0, 0, W, H);
      ctx.lineWidth = 0.5;
      ctx.strokeStyle = EDGE_SOFT;
      ctx.font = "7.5px ui-monospace, Menlo, monospace";
      ctx.fillStyle = DIM;
      for (const v of ticks) {
        ctx.beginPath();
        ctx.moveTo(sx(v), pad.t);
        ctx.lineTo(sx(v), pad.t + size);
        ctx.moveTo(pad.l, sy(v));
        ctx.lineTo(pad.l + size, sy(v));
        ctx.stroke();
        ctx.textAlign = "center";
        ctx.fillText(String(+v.toPrecision(6)), sx(v), pad.t + size + 12);
        ctx.textAlign = "right";
        ctx.fillText(String(+v.toPrecision(6)), pad.l - 4, sy(v) + 3);
      }
      ctx.strokeStyle = EDGE;
      ctx.beginPath();
      ctx.moveTo(pad.l, pad.t + size);
      ctx.lineTo(pad.l + size, pad.t + size);
      ctx.stroke();
      ctx.globalAlpha = 0.8;
      for (let i = 0; i < n; i++) {
        disc(ctx, sx(points.get(i, 0)), sy(points.get(i, 1)), 1.7, probability.get(i) > 0.5 ? ORANGE : BLUE);
      }
      ctx.globalAlpha = 1;
    },
  };
}

function niceTicks(lo, hi, count) {
  const step0 = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(v);
  return ticks;
}
