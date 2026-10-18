// F1TENTH races: a chase camera over the track, a map of the whole lap in the
// corner, and a strip along the bottom with each car's steering wheel, speed,
// brakes and team.
//
// The track, the map and the teams' plates are drawn once per episode into
// offscreen pictures, res pixels to a logical one, and cropped and copied each
// frame.

import { DIM, EDGE_SOFT, INK, MID, PICKED, SUNKEN, WELL } from "./palette.js";

// a chase camera over this many metres of track across, with a map of the whole lap
const W = 640, H = 446, SPAN = 30;
const MAP_W = 170, MARGIN = 8;
// cars are drawn this many times life size, a little larger to stand out
const CAR_SCALE = 1.5;

// eleven teams, one per seat, in liveries after F1's: a name, the bodywork's
// colour, the wings', and for a split livery the colour of the body's right flank.
// The first few look the most different, for races of a few cars; the dashboard
// shows the same names and colours (dashboard.js)
export const LIVERIES = [
  ["Precision", [190, 195, 200], [25, 25, 25], null],
  ["Rabbit Racing", [255, 120, 0], [30, 140, 60], null],
  ["Orbital", [95, 45, 170], [200, 190, 240], null],
  ["Richards", [0, 75, 45], [215, 180, 90], null],
  ["Vettelli", [205, 0, 15], [245, 245, 245], null],
  ["Blue Eagle", [25, 75, 195], [255, 200, 0], null],
  ["Kessler", [30, 30, 32], [215, 25, 30], null],
  ["Sakura", [245, 140, 185], [245, 245, 245], null],
  ["Svea", [255, 205, 0], [0, 90, 170], null],
  ["Monarch", [120, 20, 45], [205, 165, 50], [205, 165, 50]],
  ["Riviera", [0, 150, 160], [240, 235, 220], null],
];

// each car's livery: with a policy, the policy in the first and the bots in the
// rest, in grid order; without, every car in its seat's
export function carLiveries(numCars, policy = null) {
  if (policy == null) return Array.from({ length: numCars }, (_, i) => LIVERIES[i % LIVERIES.length]);
  let bot = 0;
  return Array.from({ length: numCars }, (_, i) =>
    i === policy ? LIVERIES[0] : LIVERIES[1 + (bot++ % (LIVERIES.length - 1))],
  );
}

// ---------------------------------------------------------------- colours

const rgbOf = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const css = ([r, g, b]) => `rgb(${r}, ${g}, ${b})`;
// rounding half to even, so a plate's pixel art lands on whole pixels the same way
// whichever side of .5 it sits
function roundHalfEven(v) {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 ? r - 1 : r;
}
// the colour f of the way from a to b
const blend = (a, b, f) => a.map((v, i) => roundHalfEven((1 - f) * v + f * b[i]));
// a crashed car stays in its colours, faded halfway to the background
const fade = (c) => blend(c, rgbOf(WELL), 0.5);

// red and white kerbs line the inside of corners tighter than this radius, this
// many metres wide, a stripe every this many points along the centerline
const KERB_RADIUS = 10, KERB_WIDTH = 0.3, KERB_STRIPE = 2;
const KERB_RED = "rgb(215, 30, 35)", KERB_WHITE = "rgb(245, 245, 245)";
// the start/finish line, this many metres ahead of the front of the grid, chequered
// in this many squares across the track
const START_AHEAD = 1.5, START_SQUARES = 8;
const CHEQUER = ["rgb(25, 25, 25)", "rgb(245, 245, 245)"];
// the strip along the bottom with each car's speed and steering, in pixels
const HUD_H = 84, HUD_FONT = 9;
// two rows on each card for its lap times, under the speed bar, in a smaller face
const LAP_ROW = 20, LAP_FONT = `8px ui-monospace, "SF Mono", Menlo, monospace`;
// the fastest lap of the race so far, in purple as on the broadcasts (--s4)
const FASTEST = "#5b3fb0";
const FONT = `${HUD_FONT}px ui-monospace, "SF Mono", Menlo, monospace`;
// the steering wheel turns this many degrees either way at full lock
const WHEEL_LOCK = 135;
// a car is braking, and its rain light lit, when it slows by this share of the most
// it can in a step (the speed lost in a step of full braking is limits[2])
const BRAKING_FROM = 0.15;
const BRAKE_ON = [255, 40, 40], BRAKE_OFF = [110, 25, 25], LAMP_OFF = [205, 190, 170];
// the camera closes a fifth of its distance to the car it follows every two frames
const EASE = 1 - Math.sqrt(0.8);
// races are recorded at 20 frames a second (examples/f1tenth.py), and played at
// twice that, so a lap goes by at a watchable pace
const RECORD_HZ = 20;
const FPS = 2 * RECORD_HZ;
// offscreen pictures are drawn at the canvas's own pixels, up to this many, and no
// larger than this, so a big track doesn't take hundreds of megabytes
const MAX_RES = 3, MAX_PIXELS = 16e6;

// ---------------------------------------------------------------- wordmarks

// a pixel font for the teams' wordmarks, 7 pixels tall, each glyph a row of strings
// ("#" for a pixel), lowercase letters sitting on the same baseline
const GLYPH_TABLE = `
A .###. #...# #...# ##### #...# #...# #...#
B ####. #...# #...# ####. #...# #...# ####.
C .#### #.... #.... #.... #.... #.... .####
E ##### #.... #.... ####. #.... #.... #####
G .#### #.... #.... #..## #...# #...# .####
H #...# #...# #...# ##### #...# #...# #...#
I # # # # # # #
K #...# #..#. #.#.. ##... #.#.. #..#. #...#
L #... #... #... #... #... #... ####
M #...# ##.## #.#.# #.#.# #...# #...# #...#
N #...# ##..# #.#.# #.#.# #..## #...# #...#
O .###. #...# #...# #...# #...# #...# .###.
P ####. #...# #...# ####. #.... #.... #....
R ####. #...# #...# ####. #.#.. #..#. #...#
S .#### #.... #.... .###. ....# ....# ####.
T ##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..
U #...# #...# #...# #...# #...# #...# .###.
V #...# #...# #...# #...# #...# .#.#. ..#..
a .... .... .### ...# .### #..# .###
c .... .... .### #... #... #... .###
d ...# ...# .### #..# #..# #..# .###
h #... #... ###. #..# #..# #..# #..#
i # . # # # # #
r .... .... #.## ##.. #... #... #...
s .... .... .### #... .##. ...# ###.
`;
const GLYPHS = Object.fromEntries(
  GLYPH_TABLE.trim()
    .split("\n")
    .map((line) => line.split(/\s+/))
    .map(([ch, ...rows]) => [ch, rows]),
);

// the pixels of text in the wordmark font, as [x, y] with y down, and its width:
// bold doubles each stroke, italic leans the top two pixels right
function text(str, { spacing = 1, bold = false, italic = false } = {}) {
  let pixels = [], x = 0;
  for (const ch of str) {
    if (ch === " ") {
      x += 3;
      continue;
    }
    const glyph = GLYPHS[ch];
    glyph.forEach((row, y) => {
      for (let dx = 0; dx < row.length; dx++) {
        if (row[dx] !== "#") continue;
        pixels.push([x + dx, y]);
        if (bold) pixels.push([x + dx + 1, y]);
      }
    });
    x += glyph[0].length + spacing + bold;
  }
  let width = x - spacing;
  if (italic) {
    pixels = pixels.map(([px, py]) => [px + Math.floor((6 - py) / 3), py]);
    width += 2;
  }
  return [pixels, width];
}

const shift = (pixels, dx, dy) => pixels.map(([x, y]) => [x + dx, y + dy]);
const rule = (x0, x1, y) => Array.from({ length: Math.max(0, x1 - x0) }, (_, i) => [x0 + i, y]);
// the pixels of a picture drawn as rows of strings ("#" for a pixel), its top-left
// corner at [x, y]
const sprite = (rows, x, y) =>
  rows.flatMap((row, dy) => [...row].flatMap((c, dx) => (c === "#" ? [[x + dx, y + dy]] : [])));

// two lines of text, centered on each other, two pixels apart
function stack([a, wa], [b, wb]) {
  const width = Math.max(wa, wb);
  return [shift(a, Math.floor((width - wa) / 2), 0), shift(b, Math.floor((width - wb) / 2), 9), width];
}

// each team's name in pixel art, with its emblem, on a plate in one of its
// colours: {name: [the plate's colour, layers of [pixels, colour] painted in
// order]}, each from its own top-left corner
function wordmarks() {
  const marks = {};
  let t, w;

  // a crosshair on the O
  [t, w] = text("PRECISION");
  const o = text("PRECISI")[1] + 3;
  const crosshair = [[o, -3], [o, -2], [o, 3], [o, 8], [o, 9]];
  marks["Precision"] = [[20, 20, 22], [[t, [190, 195, 200]], [crosshair, [0, 200, 220]]]];

  // a rabbit's ears above the name, in carrot colours
  const [rabbit, racing, rw] = stack(text("RABBIT", { bold: true }), text("RACING"));
  const ears = sprite([".#.....#.", "###...###", "#.#...#.#", "#.#...#.#", "#.#...#.#", ".#.....#."], Math.floor(rw / 2) - 4, -7);
  const insides = sprite([".#.....#.", ".#.....#.", ".#.....#."], Math.floor(rw / 2) - 4, -5);
  marks["Rabbit Racing"] = [
    [255, 120, 0],
    [[[...rabbit, ...ears], [250, 250, 250]], [insides, [255, 185, 195]], [racing, [15, 80, 35]]],
  ];

  // a ringed planet before the name
  [t, w] = text("ORBITAL");
  const planet = sprite([".###.", "#####", "#####", "#####", ".###."], -2, 1);
  const ring = sprite(["............#", ".###########.", "#............"], -6, 2);
  marks["Orbital"] = [
    [20, 18, 50],
    [[planet, [245, 190, 60]], [ring, [200, 190, 240]], [shift(t, 9, 0), [200, 190, 240]]],
  ];

  // a coachbuilder's badge, between gold rules
  [t, w] = text("Richards");
  marks["Richards"] = [[0, 75, 45], [[t, [235, 220, 170]], [[...rule(0, w, -2), ...rule(0, w, 8)], [215, 180, 90]]]];

  // over the tricolore
  [t, w] = text("VETTELLI", { bold: true, italic: true });
  const third = Math.round(w / 3);
  marks["Vettelli"] = [
    [25, 25, 25],
    [
      [t, [235, 25, 35]],
      [rule(0, third, 8), [0, 140, 70]],
      [rule(third, w - third, 8), [245, 245, 245]],
      [rule(w - third, w, 8), [205, 0, 15]],
    ],
  ];

  // an eagle's wings spread above the name
  const [blue, eagle, bw] = stack(text("BLUE"), text("EAGLE", { bold: true }));
  const spread = sprite(
    ["#.............#", "##...........##", ".###...#...###.", "..####.#.####..", "....#######....", "......###......"],
    Math.floor(bw / 2) - 7,
    -7,
  );
  marks["Blue Eagle"] = [[25, 75, 195], [[[...blue, ...eagle], [250, 250, 250]], [spread, [255, 200, 0]]]];

  // a shield before the name
  [t, w] = text("KESSLER");
  const shield = sprite(["#####", "#...#", "#...#", "#...#", ".#.#.", "..#.."], 0, 0);
  const field = sprite([".###.", ".###.", ".###.", "..#.."], 0, 1);
  marks["Kessler"] = [
    [30, 30, 32],
    [[shield, [215, 170, 40]], [field, [215, 25, 30]], [shift(t, 8, 0), [240, 240, 240]]],
  ];

  // a cherry blossom before the name
  [t, w] = text("SAKURA");
  const petals = sprite([".#.#.", "##.##", ".....", "##.##", ".#.#."], 0, 1);
  marks["Sakura"] = [
    [245, 245, 245],
    [[petals, [245, 140, 185]], [[[2, 3]], [205, 40, 110]], [shift(t, 7, 0), [225, 70, 140]]],
  ];

  // the name beside a flag, its Nordic cross the plate showing through
  [t, w] = text("SVEA", { bold: true });
  const flag = sprite(["##.#####", "##.#####", "........", "##.#####", "##.#####"], 0, 1);
  marks["Svea"] = [[255, 205, 0], [[[...flag, ...shift(t, 10, 0)], [0, 90, 170]]]];

  // a crown above the name
  [t, w] = text("MONARCH");
  const crown = sprite(["#..#..#", "#.###.#", "#######"], Math.floor(w / 2) - 3, -5);
  marks["Monarch"] = [[120, 20, 45], [[[...t, ...crown], [205, 165, 50]]]];

  // between gold rules, the top one set with a diamond
  [t, w] = text("RIVIERA", { spacing: 2 });
  const c = Math.floor(w / 2);
  const deco = [...rule(0, c - 2, -3), ...sprite([".#.", "#.#", ".#."], c - 1, -4), ...rule(c + 3, w, -3), ...rule(0, w, 8)];
  marks["Riviera"] = [[240, 235, 220], [[t, [0, 120, 130]], [deco, [195, 160, 80]]]];

  for (const [name, [plate, layers]] of Object.entries(marks)) {
    const all = layers.flatMap(([pixels]) => pixels);
    const x0 = Math.min(...all.map(([x]) => x)), y0 = Math.min(...all.map(([, y]) => y));
    marks[name] = [plate, layers.map(([pixels, c]) => [shift(pixels, -x0, -y0), c])];
  }
  return marks;
}

const WORDMARKS = wordmarks();

// ---------------------------------------------------------------- the car

const rect = (x0, x1, y0, y1) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
// a shape symmetric about the car's centerline, from its left side, nose first
const mirrored = (side) => [...side, ...[...side].reverse().map(([x, y]) => [x, -y])];
// the left and right halves of a shape symmetric about the car's centerline, from
// its left side, nose first
function halves(side) {
  const left = [...side, [side.at(-1)[0], 0], [side[0][0], 0]];
  return [left, left.map(([x, y]) => [x, -y])];
}

// an F1 car from above, nose along +x, in units of the car's length and width: its
// bodywork and wings in its livery, and its axles, tyres and cockpit dark
const BODY = halves([[0.47, 0.05], [0.2, 0.09], [0.06, 0.13], [0.02, 0.27], [-0.22, 0.27], [-0.36, 0.15], [-0.43, 0.1]]);
const CAR_PARTS = [
  // axles first, so the bodywork covers them between the tyres
  [rect(0.21, 0.27, -0.4, 0.4), "dark"],
  [rect(-0.34, -0.28, -0.4, 0.4), "dark"],
  [rect(-0.5, -0.41, -0.38, 0.38), "wings"], // rear wing
  [mirrored([[0.5, 0.18], [0.44, 0.47], [0.4, 0.47]]), "wings"], // front wing
  [BODY[0], "left"],
  [BODY[1], "right"],
  // tyres, the rear ones wider
  ...[[-0.42, -0.2, 0.3, 0.5], [0.15, 0.33, 0.31, 0.48]].flatMap(([x0, x1, y0, y1]) =>
    [1, -1].map((side) => [rect(x0, x1, y0 * side, y1 * side), "dark"]),
  ),
  [mirrored([[0.11, 0.04], [0.06, 0.07], [-0.08, 0.07]]), "dark"], // cockpit
  [rect(-0.6, -0.49, -0.13, 0.13), "light"], // the rain light, behind the wing
];

function carFills([, body, wings, right], faded) {
  const colors = { left: body, right: right || body, wings, dark: rgbOf(INK), light: BRAKE_OFF };
  return Object.fromEntries(Object.entries(colors).map(([k, c]) => [k, css(faded ? fade(c) : c)]));
}

// ---------------------------------------------------------------- the track

function surface(width, height) {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(width, height);
  return Object.assign(document.createElement("canvas"), { width, height });
}

function polygon(ctx, points) {
  ctx.beginPath();
  for (const [x, y] of points) ctx.lineTo(x, y);
  ctx.closePath();
}

// the signed curvature of a closed line at each of its points, positive turning
// left, smoothed over a few points either side
function curvature(points) {
  const n = points.length, k = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const [px, py] = points[(i + n - 1) % n], [x, y] = points[i], [nx, ny] = points[(i + 1) % n];
    const d1x = (nx - px) / 2, d1y = (ny - py) / 2;
    const d2x = nx - 2 * x + px, d2y = ny - 2 * y + py;
    k[i] = (d1x * d2y - d1y * d2x) / Math.hypot(d1x, d1y) ** 3;
  }
  const window = 9, half = Math.floor(window / 2);
  return Array.from(k, (_, i) => {
    let sum = 0;
    for (let j = -half; j <= half; j++) sum += k[(i + j + n) % n];
    return sum / window;
  });
}

// The track drawn at scale logical pixels per metre, res pixels to a logical one,
// over [lo, hi] in metres: its surface, and its walls; and if start is the index of
// a point along it, its kerbs and its start/finish line there. Returns the picture,
// and where a point in metres is in it, in logical pixels
function trackImage(walls, center, lo, hi, scale, lineWidth, res, start = null) {
  const height = (hi[1] - lo[1]) * scale;
  const image = surface(Math.ceil((hi[0] - lo[0]) * scale * res), Math.ceil(height * res));
  const ctx = image.getContext("2d");
  const at = ([x, y]) => [(x - lo[0]) * scale, height - (y - lo[1]) * scale];
  ctx.scale(res, res);
  ctx.fillStyle = WELL;
  ctx.fillRect(0, 0, image.width, image.height);
  // a polygon filled, and outlined in its own colour, so neighbours show no seam
  const patch = (points, color) => {
    polygon(ctx, points.map(at));
    ctx.fillStyle = ctx.strokeStyle = color;
    ctx.fill();
    ctx.stroke();
  };
  ctx.lineWidth = 0.5;
  ctx.lineJoin = "round";

  // the surface, a quad between each pair of points along the walls
  const [a, b] = walls, n = a.length;
  for (let k = 0; k < n; k++) {
    const j = (k + n - 1) % n;
    patch([a[j], a[k], b[k], b[j]], SUNKEN);
  }
  if (start != null) {
    // kerbs on the inside wall, the left one (walls[0]) of a left-hand corner
    const bend = curvature(center);
    for (let k = 0; k < n; k++) {
      if (Math.abs(bend[k]) <= 1 / KERB_RADIUS) continue;
      const wall = bend[k] > 0 ? a : b;
      const inner = (i) => {
        const dx = center[i][0] - wall[i][0], dy = center[i][1] - wall[i][1];
        const d = Math.hypot(dx, dy);
        return [wall[i][0] + (dx / d) * KERB_WIDTH, wall[i][1] + (dy / d) * KERB_WIDTH];
      };
      const j = (k + n - 1) % n;
      patch([wall[j], wall[k], inner(k), inner(j)], Math.floor(k / KERB_STRIPE) % 2 ? KERB_RED : KERB_WHITE);
    }
    // the start/finish line, two rows of squares across the track
    const [sx, sy] = a[start];
    const ax = b[start][0] - sx, ay = b[start][1] - sy;
    let lx = center[(start + 1) % n][0] - center[start][0], ly = center[(start + 1) % n][1] - center[start][1];
    const side = Math.hypot(ax, ay) / START_SQUARES / Math.hypot(lx, ly);
    [lx, ly] = [lx * side, ly * side];
    const [qx, qy] = [ax / START_SQUARES, ay / START_SQUARES];
    for (let row = 0; row < 2; row++) {
      for (let j = 0; j < START_SQUARES; j++) {
        const cx = sx + qx * j + lx * row, cy = sy + qy * j + ly * row;
        const square = [[cx, cy], [cx + qx, cy + qy], [cx + qx + lx, cy + qy + ly], [cx + lx, cy + ly]];
        polygon(ctx, square.map(at));
        ctx.fillStyle = CHEQUER[(j + row) % 2];
        ctx.fill();
      }
    }
  }
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = MID;
  for (const wall of walls) {
    polygon(ctx, wall.map(at));
    ctx.stroke();
  }
  return { image, at, width: (hi[0] - lo[0]) * scale, height };
}

// the point along the track START_AHEAD metres ahead of the front of the grid
function startIndex(center, [fx, fy]) {
  let nearest = 0, best = Infinity, spacing = 0;
  center.forEach(([x, y], i) => {
    const d = Math.hypot(x - fx, y - fy);
    if (d < best) [nearest, best] = [i, d];
    if (i) spacing += Math.hypot(x - center[i - 1][0], y - center[i - 1][1]);
  });
  spacing /= center.length - 1;
  return (nearest + roundHalfEven(START_AHEAD / spacing)) % center.length;
}

// how far each car has come along the lap at each step, (steps, cars), in
// centerline points from where it started
function lapProgress(center, pose, steps, cars) {
  const n = center.length;
  const laps = new Float64Array(steps * cars);
  for (let i = 0; i < cars; i++) {
    let last = 0, unwrap = 0;
    for (let t = 0; t < steps; t++) {
      // the car's nearest point along the centerline, counted on past the start line
      const [x, y] = pose(t, i);
      let nearest = 0, best = Infinity;
      for (let k = 0; k < n; k++) {
        const d = (center[k][0] - x) ** 2 + (center[k][1] - y) ** 2;
        if (d < best) [nearest, best] = [k, d];
      }
      // np.unwrap with period n
      const dd = nearest - last;
      if (t && Math.abs(dd) >= n / 2) {
        let wrapped = ((((dd + n / 2) % n) + n) % n) - n / 2;
        if (wrapped === -n / 2 && dd > 0) wrapped = n / 2;
        unwrap += wrapped - dd;
      }
      last = nearest;
      laps[t * cars + i] = nearest + unwrap;
    }
  }
  return laps;
}

// the car furthest round the lap at each step, among those still running
function leaders(center, pose, crashed, steps, cars) {
  const progress = lapProgress(center, pose, steps, cars);
  const laps = progress.map((v, k) => (crashed(Math.floor(k / cars), k % cars) ? -Infinity : v));
  const out = new Int32Array(steps);
  for (let t = 0; t < steps; t++) {
    let all = true;
    for (let i = 0; i < cars; i++) {
      if (laps[t * cars + i] > laps[t * cars + out[t]]) out[t] = i;
      all &&= crashed(t, i);
    }
    // once every car is out, keep following the last one running
    if (t && all) out[t] = out[t - 1];
  }
  return out;
}

// frames a card takes to slide to its new place when cars pass
const SLIDE_FRAMES = 10;

// Each car's place in the race at each step, (steps, cars), 0 for the leader: by
// how far round the lap it is, from the start line (not by distance covered: the
// grid is spread out behind the line), a car that's out where it stopped, so it
// keeps its place until someone passes the spot; level cars in grid order. And
// slide, where
// each one's card is in the strip, in places: its place averaged over the last
// SLIDE_FRAMES steps, so cards slide past each other rather than jump, the same
// at any step whatever was played before it.
function raceOrder(center, pose, crashed, start, steps, cars) {
  const n = center.length;
  const far = lapProgress(center, pose, steps, cars);
  // from the start line: where each car began, a little behind it or ahead
  for (let i = 0; i < cars; i++) {
    const [x, y] = pose(0, i);
    let nearest = 0, best = Infinity;
    center.forEach(([cx, cy], k) => {
      const d = (cx - x) ** 2 + (cy - y) ** 2;
      if (d < best) [nearest, best] = [k, d];
    });
    const behind = ((((nearest - start + n / 2) % n) + n) % n) - n / 2;
    const offset = behind - far[i];
    for (let t = 0; t < steps; t++) far[t * cars + i] += offset;
  }
  const place = new Int32Array(steps * cars);
  const seats = [...Array(cars).keys()];
  for (let t = 0; t < steps; t++) {
    const order = [...seats].sort((a, b) => far[t * cars + b] - far[t * cars + a] || a - b);
    order.forEach((i, k) => (place[t * cars + i] = k));
  }
  const slide = new Float32Array(steps * cars);
  for (let t = 0; t < steps; t++) {
    const from = Math.max(0, t - SLIDE_FRAMES + 1);
    for (let i = 0; i < cars; i++) {
      let sum = 0;
      for (let u = from; u <= t; u++) sum += place[u * cars + i];
      slide[t * cars + i] = sum / (t - from + 1);
    }
  }
  return { place, slide, far };
}

// When each car crossed the start/finish line, in steps (fractions of one,
// between the two it crossed between), from far (raceOrder's, in centerline
// points from the line, n a lap): the first time it's k laps round, for k = 0,
// 1, 2, ... while it's running. Its clock starts at the first, as it leaves the
// grid behind the line, so lap k is from crossing k - 1 to crossing k.
function lapCrossings(far, n, crashed, steps, cars) {
  return Array.from({ length: cars }, (_, i) => {
    // a car that starts on or past the line is on its first lap already
    const done = far[i] >= 0 ? [0] : [];
    for (let t = 1; t < steps && !crashed(t, i); t++) {
      const a = far[(t - 1) * cars + i], b = far[t * cars + i];
      const line = done.length * n;
      if (a < line && b >= line) done.push(t - 1 + (line - a) / (b - a));
    }
    return done;
  });
}

// a lap time as the timing screens show it, minutes:seconds.hundredths, 00:32.45
function lapTime(steps) {
  const s = steps / RECORD_HZ, m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${(s - 60 * m).toFixed(2).padStart(5, "0")}`;
}

// ---------------------------------------------------------------- the strip

// a steering wheel at cx, cy turned turn degrees to the left, its top marked in color
function wheel(ctx, cx, cy, r, turn, color, faded) {
  const rim = css(faded ? blend(rgbOf(INK), rgbOf(WELL), 0.5) : rgbOf(INK));
  // angle in degrees counterclockwise from straight up, on a y-down screen
  const at = (angle, radius) => {
    const a = ((angle + turn) * Math.PI) / 180;
    return [cx - radius * Math.sin(a), cy - radius * Math.cos(a)];
  };
  ctx.strokeStyle = ctx.fillStyle = rim;
  // three spokes, to either side and down, into a hub
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const spoke of [90, -90, 180]) {
    ctx.moveTo(cx, cy);
    ctx.lineTo(...at(spoke, r));
  }
  ctx.stroke();
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(cx, cy, r - 1.5, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, r / 3, 0, 2 * Math.PI);
  ctx.fill();
  // the stripe at the top of the rim, to see it turn
  const [x, y] = at(0, r - 1.5);
  ctx.beginPath();
  ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.arc(x, y, 2.25, 0, 2 * Math.PI);
  ctx.stroke();
}

// A team's plate filling box [left, top, right, bottom], with its wordmark in the
// middle, as large as fits in whole pixels, so that it stays crisp; faded for a
// crashed car. Returns the picture and where it goes
function plate(name, box, res, faded) {
  const [color, layers] = WORDMARKS[name];
  const paint = (c) => css(faded ? fade(c) : c);
  const [left, top, right, bottom] = box.map(roundHalfEven);
  const bw = right - left, bh = bottom - top;
  const image = surface(Math.max(1, Math.round(bw * res)), Math.max(1, Math.round(bh * res)));
  const ctx = image.getContext("2d");
  ctx.scale(res, res);
  ctx.fillStyle = paint(color);
  ctx.fillRect(0, 0, bw, bh);
  const all = layers.flatMap(([pixels]) => pixels);
  const w = Math.max(...all.map(([x]) => x)) + 1, h = Math.max(...all.map(([, y]) => y)) + 1;
  // at least a pixel of plate round the mark
  const k = Math.max(1, Math.floor(Math.min((bw - 2) / w, (bh - 2) / h)));
  const x = roundHalfEven((bw - w * k) / 2), y = roundHalfEven((bh - h * k) / 2);
  for (const [pixels, c] of layers) {
    ctx.fillStyle = paint(c);
    for (const [px, py] of pixels) {
      if (x + px * k >= 0 && x + (px + 1) * k <= bw) ctx.fillRect(x + px * k, y + py * k, k, k);
    }
  }
  return { image, left, top, width: bw, height: bh };
}

function ellipse(ctx, x, y, r, fill, outline) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = fill;
  ctx.fill();
  // the outline inside the disc, so it stays its size
  ctx.beginPath();
  ctx.arc(x, y, r - 0.25, 0, 2 * Math.PI);
  ctx.lineWidth = 0.5;
  ctx.strokeStyle = outline;
  ctx.stroke();
}

// ---------------------------------------------------------------- the view

// arrays.edges: (2, points, 2) the track's two walls as closed polylines, at the
// same points along the lap; arrays.poses: (steps, cars, 3) of x, y, yaw;
// arrays.crashed: (steps, cars); arrays.car_size: length and width; and if there
// are arrays.speed and arrays.steer, (steps, cars), the strip along the bottom
// shows them, scaled by arrays.limits: top speed, steering lock, and the speed lost
// in a step of full braking, to show when and how hard each car brakes, in the
// strip and on its rain light; and arrays.rewards, (steps, cars), each car's reward
// so far, in metres, shown in the strip where there's room. With arrays.policy, the
// policy's car, the rest are bots, and the camera follows the policy; without it,
// every car is a member, in its seat's livery, and the camera follows the leader.
// The camera follows the picked seat instead, when there is one.
export function f1tenth(arrays) {
  const { edges, poses, crashed } = arrays;
  const [steps, cars] = poses.shape;
  const points = edges.shape[1];
  const wall = (w) => Array.from({ length: points }, (_, k) => [edges.get(w, k, 0), edges.get(w, k, 1)]);
  const walls = [wall(0), wall(1)];
  const center = walls[0].map(([x, y], k) => [(x + walls[1][k][0]) / 2, (y + walls[1][k][1]) / 2]);
  const pose = (t, i) => {
    const at = (t * cars + i) * 3;
    return [poses.data[at], poses.data[at + 1], poses.data[at + 2]];
  };
  const isCrashed = (t, i) => crashed.data[t * cars + i] !== 0;

  const lo = [0, 1].map((d) => Math.min(...walls.flatMap((w) => w.map((p) => p[d]))));
  const hi = [0, 1].map((d) => Math.max(...walls.flatMap((w) => w.map((p) => p[d]))));
  // the whole track at the camera's scale, cropped around the camera each frame;
  // only a metre past the walls, the crop is clipped to it over the background
  const scale = W / SPAN;
  const trackLo = lo.map((v) => v - 1), trackHi = hi.map((v) => v + 1);
  const start = startIndex(center, pose(0, 0));
  // and small, for the map in the corner
  const mapLo = lo.map((v) => v - 2), mapHi = hi.map((v) => v + 2);
  const mapScale = MAP_W / (mapHi[0] - mapLo[0]);

  const policy = arrays.policy ? arrays.policy.data[0] : null;
  const liveries = carLiveries(cars, policy);
  const fills = liveries.map((lv) => [carFills(lv, false), carFills(lv, true)]);
  const [length, width] = arrays.car_size.data;
  const parts = CAR_PARTS.map(([shape, kind]) => {
    const path = new Path2D();
    for (const [x, y] of shape) path.lineTo(x * length * CAR_SCALE * scale, y * width * CAR_SCALE * scale);
    path.closePath();
    return [path, kind];
  });
  const reach = (length * CAR_SCALE * scale) / 2;

  // how hard each car brakes at each step, as a share of the most it can: the speed
  // it loses, over the speed lost in a step of full braking
  const { speed, steer, rewards, limits } = arrays;
  const braking = new Float32Array(steps * cars);
  if (speed) {
    const full = limits.length > 2 ? limits.data[2] : 0.45;
    for (let t = 1; t < steps; t++) {
      for (let i = 0; i < cars; i++) {
        const lost = speed.data[(t - 1) * cars + i] - speed.data[t * cars + i];
        braking[t * cars + i] = isCrashed(t, i) ? 0 : Math.min(Math.max(lost / full, 0), 1);
      }
    }
  }

  // the strip's layout, the same every frame
  const hudTop = H - HUD_H, cell = W / cars, pad = 4, r = 11;
  const barY = hudTop + pad + 2 * r + 4;
  const roomy = cell >= 90;
  const { place, slide, far } = raceOrder(center, pose, isCrashed, start, steps, cars);
  const crossings = lapCrossings(far, center.length, isCrashed, steps, cars);
  // every lap anyone finished, [when, how long], to tell the fastest so far
  const allLaps = crossings.flatMap((done) => done.slice(1).map((c, k) => [c, c - done[k]]));

  // the pictures drawn once, at res pixels to a logical one, redrawn larger if the
  // canvas outgrows them
  let res = 0, track, lapMap, plates;
  const cap = Math.sqrt(MAX_PIXELS / ((trackHi[0] - trackLo[0]) * (trackHi[1] - trackLo[1]) * scale * scale));
  function build(want) {
    res = Math.max(1, Math.min(want, MAX_RES, cap));
    track = trackImage(walls, center, trackLo, trackHi, scale, roundHalfEven(0.12 * scale * 2) / 2, res, start);
    lapMap = trackImage(walls, center, mapLo, mapHi, mapScale, 1, res);
    plates = liveries.map(([name], i) =>
      [false, true].map((faded) => plate(name, [i * cell + 1, barY + 9 + LAP_ROW, (i + 1) * cell - 1, H], res, faded)),
    );
  }
  build(2);
  const mapX = W - lapMap.width - MARGIN, mapY = MARGIN;

  // where the camera is at every step, for each car it could follow: it eases
  // toward the car, so each frame's depends on the ones before
  const defaultFollowed = policy != null ? new Int32Array(steps).fill(policy) : leaders(center, pose, isCrashed, steps, cars);
  const paths = new Map();
  function cameraPath(seat) {
    if (!paths.has(seat)) {
      const followed = (t) => (seat == null ? defaultFollowed[t] : seat);
      const path = new Float64Array(steps * 2);
      let [x, y] = pose(0, followed(0));
      for (let t = 0; t < steps; t++) {
        const [fx, fy] = pose(t, followed(t));
        x += EASE * (fx - x);
        y += EASE * (fy - y);
        [path[2 * t], path[2 * t + 1]] = [x, y];
      }
      paths.set(seat, path);
    }
    return paths.get(seat);
  }

  // the crop's top-left corner in the track picture at step t, on its pixels
  function corner(t, seat) {
    const path = cameraPath(seat);
    const [cx, cy] = track.at([path[2 * t], path[2 * t + 1]]);
    return [Math.round((cx - W / 2) * res) / res, Math.round((cy - H / 2) * res) / res];
  }
  const onScreen = (t, i, [left, top]) => {
    const [x, y] = track.at(pose(t, i));
    return [x - left, y - top];
  };
  const onMap = (t, i) => {
    const [x, y] = pose(t, i);
    return [mapX + (x - mapLo[0]) * mapScale, mapY + lapMap.height - (y - mapLo[1]) * mapScale];
  };
  const pick = (seat) => (seat != null && seat >= 0 && seat < cars ? seat : null);
  let lastSeat = null;

  function drawHud(ctx, t) {
    ctx.fillStyle = WELL;
    ctx.fillRect(0, hudTop, W, HUD_H);
    ctx.fillStyle = EDGE_SOFT;
    ctx.fillRect(0, hudTop - 0.5, W, 1);
    const [maxSpeed, maxSteer] = limits.data;
    ctx.font = FONT;
    ctx.textBaseline = "top";
    for (let i = 0; i < cars; i++) {
      // each car's card where it's sliding to, in race order: drawn at its seat's
      // place (where its plate was drawn) and moved there
      ctx.save();
      ctx.translate((slide[t * cars + i] - i) * cell, 0);
      const x0 = i * cell + pad, x1 = (i + 1) * cell - pad;
      const out = isCrashed(t, i), v = speed.data[t * cars + i], brake = braking[t * cars + i];
      const body = liveries[i][1];
      // the wheel, turned as far as the front wheels are, to full lock
      const turn = Math.min(Math.max(steer.data[t * cars + i] / maxSteer, -1), 1) * WHEEL_LOCK;
      wheel(ctx, x0 + r, hudTop + pad + r, r, turn, css(body), out);
      // its place in the race, and its reward so far where there's room
      let label = `P${place[t * cars + i] + 1}`;
      if (rewards && roomy) label += ` · ${rewards.data[t * cars + i].toFixed(0)} m`;
      ctx.fillStyle = out ? DIM : INK;
      ctx.textAlign = "left";
      ctx.fillText(label, x0 + 2 * r + 4, hudTop + pad);
      // the brake lamp, at the top of the card
      const lit = brake >= BRAKING_FROM;
      ellipse(ctx, x1 - 4, hudTop + pad + 4, 3.5, css(lit ? BRAKE_ON : LAMP_OFF), lit ? css(BRAKE_OFF) : EDGE_SOFT);
      // rounded first, so that a car barely rolling back reads 0.0, not -0.0
      let reading = (Math.round(v * 10) / 10 + 0).toFixed(1);
      if (roomy) reading += " m/s";
      if (out) reading = "out";
      ctx.fillStyle = out ? DIM : INK;
      ctx.textAlign = "right";
      ctx.fillText(reading, x1, hudTop + pad + 11);
      // the speed bar under both, filling up to the top speed
      ctx.fillStyle = SUNKEN;
      ctx.fillRect(x0, barY, x1 - x0, 5);
      ctx.lineWidth = 0.5;
      ctx.strokeStyle = EDGE_SOFT;
      ctx.strokeRect(x0 + 0.25, barY + 0.25, x1 - x0, 5);
      const filled = Math.min(Math.max(v / maxSpeed, 0), 1) * (x1 - x0);
      if (filled > 0) {
        ctx.fillStyle = css(body);
        ctx.fillRect(x0, barY, filled, 5);
        ctx.strokeStyle = INK;
        ctx.strokeRect(x0 + 0.25, barY + 0.25, filled, 5);
      }
      // and under it, how hard it brakes
      if (brake > 0) {
        ctx.fillStyle = css(BRAKE_ON);
        ctx.fillRect(x0, barY + 6, brake * (x1 - x0), 2);
      }
      // its lap times, once it's crossed the line: the one it's on, running, and
      // under it the last it finished, in purple if it's the fastest of the race
      // so far
      const done = crossings[i].filter((c) => c <= t);
      const lapY = barY + 10;
      ctx.font = LAP_FONT;
      ctx.textAlign = "left";
      if (!out && done.length) {
        ctx.fillStyle = INK;
        ctx.fillText(`${roomy ? `L${done.length} ` : ""}${lapTime(t - done.at(-1))}`, x0, lapY);
      }
      if (done.length > 1) {
        const last = done.at(-1) - done.at(-2);
        const fastest = Math.min(...allLaps.filter(([when]) => when <= t).map(([, took]) => took));
        ctx.fillStyle = last === fastest ? FASTEST : DIM;
        ctx.fillText(`${roomy ? "last " : ""}${lapTime(last)}`, x0, lapY + 9);
      }
      ctx.font = FONT;
      // the team's plate, filling the rest of the card, a pixel from the next
      const p = plates[i][+out];
      ctx.drawImage(p.image, p.left, p.top, p.width, p.height);
      ctx.restore();
    }
  }

  return {
    width: W,
    height: H,
    frames: steps,
    fps: FPS,
    // where each car's card is in the strip at step t, in places from the left
    // (fractions while it slides), for the page to put anything of its own under
    // them
    cars,
    slots: (t) => Array.from({ length: cars }, (_, i) => slide[t * cars + i]),
    draw(ctx, t, { seat = null } = {}) {
      seat = lastSeat = pick(seat);
      // the offscreen pictures at the canvas's pixels, so they stay sharp
      const m = ctx.getTransform();
      const want = Math.ceil(Math.hypot(m.a, m.b) * 2) / 2;
      if (want > res && res < Math.min(MAX_RES, cap)) build(want);

      ctx.fillStyle = WELL;
      ctx.fillRect(0, 0, W, H);
      const [left, top] = corner(t, seat);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(track.image, left * res, top * res, W * res, H * res, 0, 0, W, H);

      // the followed car goes on last, over the rest
      const followed = seat ?? defaultFollowed[t];
      const order = [...Array(cars).keys()].filter((i) => i !== followed).concat(followed);
      for (const i of order) {
        const [x, y] = onScreen(t, i, [left, top]);
        if (x < -2 * reach || x > W + 2 * reach || y < -2 * reach || y > H + 2 * reach) continue;
        const yaw = pose(t, i)[2], cos = Math.cos(yaw), sin = Math.sin(yaw);
        const colors = fills[i][+isCrashed(t, i)];
        const lit = braking[t * cars + i] >= BRAKING_FROM;
        ctx.save();
        // y is up in the world and down on the screen
        ctx.transform(cos, -sin, -sin, -cos, x, y);
        for (const [path, kind] of parts) {
          ctx.fillStyle = kind === "light" && lit ? css(BRAKE_ON) : colors[kind];
          ctx.fill(path);
        }
        ctx.restore();
      }
      if (seat != null) {
        const [x, y] = onScreen(t, seat, [left, top]);
        ctx.beginPath();
        ctx.arc(x, y, reach + 4, 0, 2 * Math.PI);
        ctx.strokeStyle = PICKED;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // the map of the lap, with a dot for each car
      ctx.drawImage(lapMap.image, mapX, mapY, lapMap.image.width / res, lapMap.image.height / res);
      ctx.lineWidth = 1;
      ctx.strokeStyle = EDGE_SOFT;
      ctx.strokeRect(mapX + 0.5, mapY + 0.5, lapMap.width, lapMap.height);
      for (const i of order) {
        const [x, y] = onMap(t, i);
        ellipse(ctx, x, y, 2.5, fills[i][+isCrashed(t, i)].left, INK);
      }
      if (seat != null) {
        const [x, y] = onMap(t, seat);
        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, 2 * Math.PI);
        ctx.strokeStyle = PICKED;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      if (speed) drawHud(ctx, t);
    },
    // a car's dot on the map, its card in the strip, or the car itself
    seatAt(x, y, t) {
      const nearest = (where, within) => {
        let best = null, bestD = within ** 2;
        for (let i = 0; i < cars; i++) {
          const [cx, cy] = where(i);
          const d = (cx - x) ** 2 + (cy - y) ** 2;
          if (d < bestD) [best, bestD] = [i, d];
        }
        return best;
      };
      if (x >= mapX && x <= mapX + lapMap.width && y >= mapY && y <= mapY + lapMap.height) {
        return nearest((i) => onMap(t, i), 6);
      }
      if (speed && y >= hudTop) {
        // the card there now, wherever its car started
        const at = x / cell - 0.5;
        let best = null, bestD = 0.5;
        for (let i = 0; i < cars; i++) {
          const d = Math.abs(slide[t * cars + i] - at);
          if (d < bestD) [best, bestD] = [i, d];
        }
        return best;
      }
      const at = corner(t, lastSeat);
      return nearest((i) => onScreen(t, i, at), reach + 6);
    },
  };
}
