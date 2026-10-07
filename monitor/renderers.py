"""Server-side episode renderers, one per environment.

A training run logs Episode(env, **arrays); the server looks up RENDERERS[env] and
calls it with those arrays. A renderer returns anything encode_media understands
(a Video of PIL frames, a matplotlib figure, a PIL image).

Renderers run in the monitor server, so they stick to numpy, PIL and matplotlib -
never jax or the environment packages - and draw in the dashboard's palette.

To add one:

    @renderer("my_env")
    def render_my_env(data):
        return Video([...])
"""

from typing import Any, Callable, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from monitor.media import Video

Renderer = Callable[[Dict[str, np.ndarray]], Any]
RENDERERS: Dict[str, Renderer] = {}


def renderer(env: str) -> Callable[[Renderer], Renderer]:
    def register(fn: Renderer) -> Renderer:
        RENDERERS[env] = fn
        return fn

    return register


# the dashboard's palette: these are its CSS tokens (static/style.css)
WELL = (250, 242, 222)  # --well
EDGE = (205, 177, 136)  # --edge
EDGE_SOFT = (220, 198, 164)  # --edge-soft
SUNKEN = (230, 211, 177)  # --sunken
INK = (63, 38, 6)  # --fg
MID = (107, 74, 30)  # --mid
DIM = (117, 89, 47)  # --dim
ACCENT = (143, 92, 20)  # --accent
BLUE = (18, 99, 204)  # --s1
ORANGE = (194, 84, 27)  # --s2
# drawn this many times larger, then scaled down, for smooth edges
SUPERSAMPLE = 2


def _downsample(image: Image.Image) -> Image.Image:
    width, height = image.size
    return image.resize(
        (width // SUPERSAMPLE, height // SUPERSAMPLE),
        resample=Image.Resampling.LANCZOS,
    )


# ---------------------------------------------------------------- boids

# evojax's flocking screen, and its fish outline (nose along +x, in half-fish units)
BOIDS_W, BOIDS_H, FISH_SIZE = 400, 300, 20.0
FISH_OUTLINE = np.array(
    [
        (-1, -1 / 3),
        (-3 / 5, -1 / 4),
        (0, -1 / 2),
        (2 / 3, -1 / 3),
        (1, 0),
        (2 / 3, 1 / 3),
        (0, 1 / 2),
        (-3 / 5, 1 / 4),
        (-1, 1 / 3),
    ]
)
FISH_COLORS = [INK, ACCENT, MID]


def _boids_frame(boids: np.ndarray) -> Image.Image:
    width, height = BOIDS_W * SUPERSAMPLE, BOIDS_H * SUPERSAMPLE
    x, y = boids[:, 0] * width, (1 - boids[:, 1]) * height
    cos, sin = np.cos(boids[:, 2])[:, None], np.sin(boids[:, 2])[:, None]
    # evojax draws fish FISH_SIZE wide on its 2x canvas
    ox, oy = (FISH_OUTLINE * ((FISH_SIZE + 1) // 2) * SUPERSAMPLE / 2).T
    px = x[:, None] + cos * ox + sin * oy
    py = y[:, None] - sin * ox + cos * oy

    image = Image.new("RGB", (width, height), WELL)
    draw = ImageDraw.Draw(image)
    for i in range(len(boids)):
        polygon = list(zip(px[i].tolist(), py[i].tolist()))
        draw.polygon(polygon, fill=FISH_COLORS[i % len(FISH_COLORS)])
    return _downsample(image)


@renderer("boids")
def render_boids(data: Dict[str, np.ndarray]) -> Video:
    """data["boids"]: (steps, boids, 3) of x, y in [0, 1] and heading theta."""
    return Video([_boids_frame(step) for step in data["boids"]])


# ---------------------------------------------------------------- slimevolley

# evojax's slimevolley court, in world units: x in [-24, 24], y up from the floor
COURT_W, GROUND_H, FENCE_W, FENCE_H = 48.0, 1.5, 1.0, 3.5
# evojax shows a 2:1 window onto the court, so 24 units of height
SLIME_W, SLIME_H = 600, 300


def _slime_frame(ball, left, right) -> Image.Image:
    width, height = SLIME_W * SUPERSAMPLE, SLIME_H * SUPERSAMPLE
    scale = width / COURT_W

    def px(x, y):
        return (x + COURT_W / 2) * scale, height - y * scale

    def disc(draw, x, y, r, fill):
        cx, cy = px(x, y)
        draw.ellipse(
            (cx - r * scale, cy - r * scale, cx + r * scale, cy + r * scale), fill=fill
        )

    image = Image.new("RGB", (width, height), WELL)
    draw = ImageDraw.Draw(image)
    # fence, with its rounded cap
    x0, y0 = px(-FENCE_W / 2, FENCE_H)
    x1, y1 = px(FENCE_W / 2, GROUND_H)
    draw.rectangle((x0, y0, x1, y1), fill=MID)
    disc(draw, 0, FENCE_H, FENCE_W / 2, MID)

    for agent, color in ((left, BLUE), (right, ORANGE)):
        x, y, r, direction, lives = agent
        cx, cy = px(x, y)
        draw.pieslice(
            (cx - r * scale, cy - r * scale, cx + r * scale, cy + r * scale),
            180,
            360,
            fill=color,
        )
        # an eye that tracks the ball, like evojax's
        angle = np.pi * (120 if direction == 1 else 60) / 180
        ex, ey = x + 0.6 * r * np.cos(angle), y + 0.6 * r * np.sin(angle)
        dx, dy = ball[0] - ex, ball[1] - ey
        dist = max(np.hypot(dx, dy), 1e-6)
        disc(draw, ex, ey, 0.3 * r, WELL)
        disc(draw, ex + 0.15 * r * dx / dist, ey + 0.15 * r * dy / dist, 0.1 * r, INK)
        # spare lives along the top, from each player's own wall
        for i in range(1, int(lives)):
            lx = direction * (COURT_W / 2 + 0.5 - i * 2.0)
            disc(draw, lx, height / scale - 1.5, 0.5, color)

    disc(draw, ball[0], ball[1], ball[2], INK)
    # the floor goes on last, over the bottom of the slimes
    gx0, gy0 = px(-COURT_W / 2, GROUND_H)
    gx1, gy1 = px(COURT_W / 2, 0)
    draw.rectangle((gx0, gy0, gx1, gy1), fill=EDGE)
    return _downsample(image)


@renderer("slimevolley")
def render_slimevolley(data: Dict[str, np.ndarray]) -> Video:
    """data["ball"]: (steps, 3) of x, y, r; data["left"] and data["right"]:
    (steps, 5) of x, y, r, direction, lives."""
    return Video(
        [
            _slime_frame(ball, left, right)
            for ball, left, right in zip(data["ball"], data["left"], data["right"])
        ]
    )


# ---------------------------------------------------------------- f1tenth

# a chase camera over this many metres of track across, with a map of the whole lap
F1TENTH_W, F1TENTH_H, F1TENTH_SPAN = 640, 446, 30.0
F1TENTH_MAP_W, F1TENTH_MARGIN = 170, 8
# cars are drawn this many times life size, a little larger to stand out
CAR_SCALE = 1.5


def _rect(x0, x1, y0, y1):
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _mirrored(side):
    """A shape symmetric about the car's centerline, from its left side, nose first."""
    side = np.array(side)
    return np.concatenate([side, side[::-1] * (1, -1)])


def _halves(side):
    """The left and right halves of a shape symmetric about the car's centerline,
    from its left side, nose first."""
    side = np.array(side)
    left = np.concatenate([side, [(side[-1, 0], 0.0), (side[0, 0], 0.0)]])
    return left, left * (1, -1)


# parodies of the 2026 F1 teams, one per seat, in their real liveries: a name, the
# bodywork's colour, the wings', and for a split livery the colour of the body's
# right flank. The first few look the most different, for races of a few cars; the
# dashboard shows the same names and colours (app.js)
LIVERIES = [
    ("Mclando", (255, 128, 0), (35, 35, 35), None),  # McLaren
    ("Vercedes", (185, 190, 196), (0, 161, 155), None),  # Mercedes
    ("Red Cow", (30, 45, 110), (220, 30, 50), None),  # Red Bull
    ("Berrari", (220, 0, 0), (245, 245, 245), None),  # Ferrari
    ("Billiams", (10, 50, 130), (240, 240, 240), None),  # Williams
    ("Racing Ducks", (240, 240, 245), (30, 70, 200), None),  # Racing Bulls
    ("Ostin Marlin", (0, 94, 70), (205, 230, 0), None),  # Aston Martin
    ("Yaas", (240, 240, 240), (220, 30, 40), None),  # Haas
    ("Audo", (150, 155, 160), (230, 30, 30), None),  # Audi
    ("Alpone", (0, 120, 220), (255, 135, 190), None),  # Alpine
    ("Chadillac", (230, 230, 225), (25, 25, 25), (25, 25, 25)),  # Cadillac
]

# a pixel font for the teams' wordmarks, 7 pixels tall, each glyph a row of strings
# ("#" for a pixel), lowercase letters sitting on the same baseline
_GLYPHS = """
A .###. #...# #...# ##### #...# #...# #...#
B ####. #...# #...# ####. #...# #...# ####.
C .#### #.... #.... #.... #.... #.... .####
D ####. #...# #...# #...# #...# #...# ####.
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
W #...# #...# #...# #.#.# #.#.# ##.## #...#
Y #...# #...# .#.#. ..#.. ..#.. ..#.. ..#..
a .... .... .### ...# .### #..# .###
c .... .... .### #... #... #... .###
d ...# ...# .### #..# #..# #..# .###
e .... .... .##. #..# #### #... .###
i # . # # # # #
l # # # # # # #
n .... .... ###. #..# #..# #..# #..#
o .... .... .##. #..# #..# #..# .##.
r .... .... #.## ##.. #... #... #...
w ..... ..... #...# #...# #.#.# #.#.# .#.#.
"""
GLYPHS = {line.split()[0]: line.split()[1:] for line in _GLYPHS.strip().splitlines()}


def _text(text, spacing=1, bold=False, italic=False):
    """The pixels of text in the wordmark font, as (x, y) with y down, and its
    width: bold doubles each stroke, italic leans the top two pixels right."""
    pixels, x = set(), 0
    for ch in text:
        if ch == " ":
            x += 3
            continue
        glyph = GLYPHS[ch]
        for y, row in enumerate(glyph):
            for dx, v in enumerate(row):
                if v == "#":
                    pixels.add((x + dx, y))
                    if bold:
                        pixels.add((x + dx + 1, y))
        x += len(glyph[0]) + spacing + bold
    width = x - spacing
    if italic:
        pixels = {(px + (6 - py) // 3, py) for px, py in pixels}
        width += 2
    return pixels, width


def _shift(pixels, dx, dy):
    return {(x + dx, y + dy) for x, y in pixels}


def _rule(x0, x1, y):
    return {(x, y) for x in range(x0, x1)}


def _stack(top, bottom):
    """Two lines of text, centered on each other, two pixels apart."""
    (a, wa), (b, wb) = top, bottom
    width = max(wa, wb)
    return _shift(a, (width - wa) // 2, 0), _shift(b, (width - wb) // 2, 9), width


def _wordmarks():
    """Each team's name in pixel art, styled after its real wordmark, on a plate in
    one of its colours: (the plate's colour, layers of (pixels, colour) painted in
    order)."""
    marks = {}

    text, w = _text("Mclando")
    swoosh = {(w + 1, 2), (w + 2, 1), (w + 3, 1), (w + 4, 0), (w + 5, 0), (w + 2, 2)}
    marks["Mclando"] = (
        (255, 128, 0),
        [(text, (35, 35, 35)), (swoosh, (245, 245, 245))],
    )

    text, w = _text("VERCEDES")
    marks["Vercedes"] = (
        (20, 20, 20),
        [(text, (0, 161, 155)), (_rule(0, w, 8), (150, 155, 165))],
    )

    # the sun above the name
    text, w = _text("Red Cow", bold=True, italic=True)
    sun = {
        (x, y)
        for x in range(w // 2 - 3, w // 2 + 4)
        for y in range(-7, 0)
        if (x - w // 2) ** 2 + (y + 4) ** 2 <= 10
    }
    marks["Red Cow"] = ((30, 45, 110), [(sun, (255, 200, 0)), (text, (220, 30, 50))])

    text, w = _text("Berrari", italic=True)
    sweep = _shift(_rule(4, w + 1, 0), 2, 0)
    marks["Berrari"] = ((255, 205, 0), [(text | sweep, (25, 25, 25))])

    text, w = _text("BILLIAMS")
    marks["Billiams"] = (
        (10, 50, 130),
        [(text, (240, 240, 240)), (_rule(0, w, 8), (100, 170, 240))],
    )

    racing, ducks, _ = _stack(_text("RACING"), _text("DUCKS", bold=True))
    marks["Racing Ducks"] = (
        (240, 240, 245),
        [(racing, (30, 70, 200)), (ducks, (220, 30, 40))],
    )

    aston, marlin, w = _stack(_text("OSTIN"), _text("MARLIN"))
    left = min(x for x, _ in aston)
    right = max(x for x, _ in aston)
    # a wing either side of the top line, swept up
    wings = {(left - 2, 2), (left - 3, 2), (left - 3, 1), (left - 4, 1)}
    wings |= {(right + 2, 2), (right + 3, 2), (right + 3, 1), (right + 4, 1)}
    marks["Ostin Marlin"] = (
        (0, 94, 70),
        [(aston | marlin, (240, 240, 240)), (wings, (205, 230, 0))],
    )

    text, w = _text("YAAS", bold=True)
    marks["Yaas"] = (
        (240, 240, 240),
        [(_shift(text, 1, 1), (25, 25, 25)), (text, (220, 30, 40))],
    )

    text, w = _text("AUDO", spacing=3)
    marks["Audo"] = (
        (25, 25, 25),
        [(text, (230, 30, 30)), (_rule(0, w, 8), (150, 155, 165))],
    )

    text, w = _text("ALPONE", italic=True)
    marks["Alpone"] = (
        (0, 120, 220),
        [(text, (240, 240, 240)), (_rule(0, w, 8), (255, 135, 190))],
    )

    text, w = _text("CHADILLAC")
    marks["Chadillac"] = (
        (25, 25, 25),
        [
            (text, (230, 230, 225)),
            (_rule(0, w // 2, 8), (230, 230, 225)),
            (_rule(w // 2, w, 8), (110, 110, 110)),
        ],
    )
    # each from its own top-left corner
    for name, (plate, layers) in marks.items():
        everything = set().union(*(pixels for pixels, _ in layers))
        x0, y0 = min(x for x, _ in everything), min(y for _, y in everything)
        marks[name] = (plate, [(_shift(pixels, -x0, -y0), c) for pixels, c in layers])
    return marks


WORDMARKS = _wordmarks()


# an F1 car from above, nose along +x, in units of the car's length and width: its
# bodywork and wings in its livery, and its axles, tyres and cockpit dark
_BODY = _halves(
    [(0.47, 0.05), (0.20, 0.09), (0.06, 0.13), (0.02, 0.27), (-0.22, 0.27)]
    + [(-0.36, 0.15), (-0.43, 0.10)]
)
CAR_PARTS = [
    # axles first, so the bodywork covers them between the tyres
    (_rect(0.21, 0.27, -0.40, 0.40), "dark"),
    (_rect(-0.34, -0.28, -0.40, 0.40), "dark"),
    (_rect(-0.50, -0.41, -0.38, 0.38), "wings"),  # rear wing
    (_mirrored([(0.50, 0.18), (0.44, 0.47), (0.40, 0.47)]), "wings"),  # front wing
    (_BODY[0], "left"),
    (_BODY[1], "right"),
    *[
        (_rect(x0, x1, y0 * side, y1 * side), "dark")
        for x0, x1, y0, y1 in ((-0.42, -0.20, 0.30, 0.50), (0.15, 0.33, 0.31, 0.48))
        for side in (1, -1)
    ],  # tyres, the rear ones wider
    (_mirrored([(0.11, 0.04), (0.06, 0.07), (-0.08, 0.07)]), "dark"),  # cockpit
    (_rect(-0.60, -0.49, -0.13, 0.13), "light"),  # the rain light, behind the wing
]
# how quickly the camera catches up with the car it follows, per frame
CAMERA_EASE = 0.2
# red and white kerbs line the inside of corners tighter than this radius, this
# many metres wide, a stripe every this many points along the centerline
KERB_RADIUS, KERB_WIDTH, KERB_STRIPE = 10.0, 0.3, 2
KERB_RED, KERB_WHITE = (215, 30, 35), (245, 245, 245)
# the start/finish line, this many metres ahead of the front of the grid, chequered
# in this many squares across the track
START_AHEAD, START_SQUARES = 1.5, 8
CHEQUER = ((25, 25, 25), (245, 245, 245))
# the strip along the bottom with each car's speed and steering, in pixels
HUD_H, HUD_FONT = 76, 9
# the steering wheel turns this many degrees either way at full lock
WHEEL_LOCK = 135
# a car is braking, and its rain light lit, when it slows by this share of the most
# it can in a step (the speed lost in a step of full braking is data["limits"][2])
BRAKING_FROM = 0.15
BRAKE_ON, BRAKE_OFF, LAMP_OFF = (255, 40, 40), (110, 25, 25), (205, 190, 170)
# every other control step, at twice real time
F1TENTH_FRAME_STRIDE, F1TENTH_FPS = 2, 20


def _blend(a, b, f):
    """The colour f of the way from a to b."""
    return tuple(int(round(v)) for v in (1 - f) * np.array(a) + f * np.array(b))


def _curvature(points):
    """The signed curvature of a closed line at each of its points, positive turning
    left, smoothed over a few points either side."""
    d1 = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)) / 2
    d2 = np.roll(points, -1, axis=0) - 2 * points + np.roll(points, 1, axis=0)
    k = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.hypot(*d1.T) ** 3
    window = 9
    wrapped = np.concatenate([k[-(window // 2) :], k, k[: window // 2]])
    return np.convolve(wrapped, np.ones(window) / window, "valid")


def _track_image(edges, lo, scale, size, line_width, start=None):
    """The track drawn at scale pixels per metre, lo at the bottom-left: its
    surface, and its walls; and if start is the index of a point along it, its
    kerbs and its start/finish line there."""
    width, height = size
    image = Image.new("RGB", (width, height), WELL)
    draw = ImageDraw.Draw(image)

    def px(points):
        return [((x - lo[0]) * scale, height - (y - lo[1]) * scale) for x, y in points]

    # the surface, a quad between each pair of points along the walls
    a, b = (px(edge) for edge in edges)
    for k in range(len(a)):
        draw.polygon([a[k - 1], a[k], b[k], b[k - 1]], fill=SUNKEN, outline=SUNKEN)
    if start is not None:
        center = edges.mean(axis=0)
        # kerbs on the inside wall, the left one (edges[0]) of a left-hand corner
        curvature = _curvature(center)
        for k in np.flatnonzero(np.abs(curvature) > 1 / KERB_RADIUS):
            wall = edges[0] if curvature[k] > 0 else edges[1]
            inward = center - wall
            inner = (
                wall
                + inward / np.linalg.norm(inward, axis=1, keepdims=True) * KERB_WIDTH
            )
            color = KERB_RED if (k // KERB_STRIPE) % 2 else KERB_WHITE
            draw.polygon(px([wall[k - 1], wall[k], inner[k], inner[k - 1]]), fill=color)
        # the start/finish line, two rows of squares across the track
        across = edges[1][start] - edges[0][start]
        along = center[(start + 1) % len(center)] - center[start]
        side = np.linalg.norm(across) / START_SQUARES
        along *= side / np.linalg.norm(along)
        for row in range(2):
            for j in range(START_SQUARES):
                corner = edges[0][start] + across * j / START_SQUARES + along * row
                square = [corner, corner + across / START_SQUARES]
                square += [square[1] + along, corner + along]
                draw.polygon(px(square), fill=CHEQUER[(j + row) % 2])
    for points in (a, b):
        draw.line(points + points[:1], fill=MID, width=line_width, joint="curve")
    return image


def _start_index(edges, front):
    """The point along the track START_AHEAD metres ahead of the front of the grid."""
    center = edges.mean(axis=0)
    nearest = np.argmin(np.hypot(*(center - front).T))
    spacing = np.hypot(*np.diff(center, axis=0).T).mean()
    return (nearest + round(START_AHEAD / spacing)) % len(center)


def _wheel(draw, cx, cy, r, turn, color, faded):
    """A steering wheel turned turn degrees to the left, its top marked in color."""
    s = SUPERSAMPLE
    rim = _blend(INK, WELL, 0.5) if faded else INK

    def at(angle, radius):
        # angle in degrees counterclockwise from straight up, on a y-down screen
        a = np.radians(angle + turn)
        return cx - radius * np.sin(a), cy - radius * np.cos(a)

    # three spokes, to either side and down, into a hub
    for spoke in (90, -90, 180):
        draw.line((cx, cy, *at(spoke, r)), fill=rim, width=2 * s)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=rim, width=3 * s)
    draw.ellipse((cx - r / 3, cy - r / 3, cx + r / 3, cy + r / 3), fill=rim)
    # the stripe at the top of the rim, to see it turn
    x, y = at(0, r - 1.5 * s)
    draw.ellipse(
        (x - 2.5 * s, y - 2.5 * s, x + 2.5 * s, y + 2.5 * s), fill=color, outline=rim
    )


def _wordmark(image, name, box, faded=False):
    """A team's plate filling box (left, top, right, bottom), with its wordmark in
    the middle, as large as fits in whole pixels of the picture once it's scaled
    down, so that it stays crisp; faded for a crashed car."""
    s = SUPERSAMPLE
    plate, layers = WORDMARKS[name]

    def fade(color):
        return _blend(color, WELL, 0.5) if faded else color

    # whole pixels of the scaled-down picture
    left, top, right, bottom = (s * round(v / s) for v in box)
    image.paste(fade(plate), (left, top, right, bottom))
    everything = set().union(*(pixels for pixels, _ in layers))
    w = max(px for px, _ in everything) + 1
    h = max(py for _, py in everything) + 1
    # at least a pixel of plate round the mark
    k = s * max(
        1, int(min((right - left - 2 * s) / (w * s), (bottom - top - 2 * s) / (h * s)))
    )
    x = left + s * round((right - left - w * k) / 2 / s)
    y = top + s * round((bottom - top - h * k) / 2 / s)
    for pixels, color in layers:
        for px, py in pixels:
            cell = (x + px * k, y + py * k, x + (px + 1) * k, y + (py + 1) * k)
            if left <= cell[0] and cell[2] <= right:
                image.paste(fade(color), cell)


def _hud(image, draw, liveries, speed, steer, braking, crashed, limits, fonts, rewards):
    """A strip along the bottom: for each car its steering wheel, seat, speed (m/s)
    and a brake lamp, with a bar filling up to the top speed and a red one as hard
    as it brakes, and its team's wordmark."""
    s = SUPERSAMPLE
    width, height = image.size
    font = fonts
    top = height - HUD_H * s
    draw.rectangle((0, top, width, height), fill=WELL)
    draw.line((0, top, width, top), fill=EDGE_SOFT, width=s)
    cell, pad = width / len(liveries), 4 * s
    max_speed, max_steer = limits[:2]
    for i, (name, body, _, _) in enumerate(liveries):
        x0, x1 = i * cell + pad, (i + 1) * cell - pad
        text = DIM if crashed[i] else INK
        # the wheel, turned as far as the front wheels are, to full lock
        r = 11 * s
        turn = np.clip(steer[i] / max_steer, -1, 1) * WHEEL_LOCK
        _wheel(draw, x0 + r, top + pad + r, r, turn, body, crashed[i])
        # seat and speed beside it, and the speed bar under both
        tx = x0 + 2 * r + 4 * s
        # the seat, and the car's reward so far where there's room
        label = f"P{i + 1}"
        if rewards is not None and cell >= 90 * s:
            label += f" \u00b7 {float(rewards[i]):.0f} m"
        draw.text((tx, top + pad), label, font=font, fill=text)
        # the brake lamp, at the top of the card
        lit = braking[i] >= BRAKING_FROM
        lx, ly, lr = x1 - 4 * s, top + pad + 4 * s, 3.5 * s
        draw.ellipse(
            (lx - lr, ly - lr, lx + lr, ly + lr),
            fill=BRAKE_ON if lit else LAMP_OFF,
            outline=BRAKE_OFF if lit else EDGE_SOFT,
        )
        # rounded first, so that a car barely rolling back reads 0.0, not -0.0
        reading = f"{round(float(speed[i]), 1) + 0.0:.1f}"
        if cell >= 90 * s:
            reading += " m/s"
        if crashed[i]:
            reading = "out"
        draw.text((x1, top + pad + 11 * s), reading, font=font, fill=text, anchor="ra")
        y = top + pad + 2 * r + 4 * s
        draw.rectangle((x0, y, x1, y + 5 * s), fill=SUNKEN, outline=EDGE_SOFT)
        filled = np.clip(speed[i] / max_speed, 0, 1) * (x1 - x0)
        if filled > 0:
            draw.rectangle((x0, y, x0 + filled, y + 5 * s), fill=body, outline=INK)
        # and under it, how hard it brakes
        pressed = braking[i] * (x1 - x0)
        if pressed > 0:
            draw.rectangle((x0, y + 6 * s, x0 + pressed, y + 8 * s), fill=BRAKE_ON)
        # the team's plate, filling the rest of the card, a pixel from the next
        box = (i * cell + s, y + 9 * s, (i + 1) * cell - s, height)
        _wordmark(image, name, box, crashed[i])


def _leaders(edges, poses, crashed):
    """The car furthest round the lap at each step, among those still running."""
    centerline = edges.mean(axis=0)
    # each car's nearest point along the centerline, counted on past the start line
    gaps = poses[:, :, None, :2] - centerline[None, None]
    nearest = np.argmin(np.hypot(gaps[..., 0], gaps[..., 1]), axis=-1)
    laps = np.unwrap(nearest, period=len(centerline), axis=0)
    laps = np.where(crashed, -np.inf, laps)
    # once every car is out, keep following the last one running
    leaders = np.argmax(laps, axis=1)
    for t in range(1, len(leaders)):
        if crashed[t].all():
            leaders[t] = leaders[t - 1]
    return leaders


@renderer("f1tenth")
@renderer("f1tenth_field")
def render_f1tenth(data: Dict[str, np.ndarray]) -> Video:
    """data["edges"]: (2, points, 2) the track's two walls as closed polylines, at
    the same points along the lap; data["poses"]: (steps, cars, 3) of x, y, yaw;
    data["crashed"]: (steps, cars); data["car_size"]: length and width; and if
    there are data["speed"] and data["steer"], (steps, cars), the strip along the
    bottom shows them, scaled by data["limits"]: top speed, steering lock, and the
    speed lost in a step of full braking, to show when and how hard each car brakes,
    in the strip and on its rain light; and data["rewards"], (steps, cars), each
    car's reward so far, in metres, shown in the strip where there's room. With
    data["champion"], the policy's car, the rest are bots, and the camera follows
    the champion; without it, every car is a member, in its seat's livery, and the
    camera follows the leader."""
    edges, poses, crashed = data["edges"], data["poses"], data["crashed"]
    num_cars = poses.shape[1]
    lo = edges.reshape(-1, 2).min(axis=0) - F1TENTH_SPAN / 2
    hi = edges.reshape(-1, 2).max(axis=0) + F1TENTH_SPAN / 2
    width, height = F1TENTH_W * SUPERSAMPLE, F1TENTH_H * SUPERSAMPLE

    # the whole track at the camera's scale, cropped around the camera each frame
    scale = width / F1TENTH_SPAN
    track_size = tuple(np.ceil((hi - lo) * scale).astype(int))
    start = _start_index(edges, poses[0, 0, :2])
    track = _track_image(edges, lo, scale, track_size, round(0.12 * scale), start)
    # and small, for the map in the corner
    map_lo = edges.reshape(-1, 2).min(axis=0) - 2.0
    map_hi = edges.reshape(-1, 2).max(axis=0) + 2.0
    map_scale = F1TENTH_MAP_W * SUPERSAMPLE / (map_hi[0] - map_lo[0])
    map_size = tuple(np.ceil((map_hi - map_lo) * map_scale).astype(int))
    lap_map = _track_image(edges, map_lo, map_scale, map_size, SUPERSAMPLE)

    if "champion" in data:
        # the champion in the first livery, and the bots in the rest, in grid order
        champion = int(data["champion"])
        bots = iter(LIVERIES[1 + b % (len(LIVERIES) - 1)] for b in range(num_cars))
        liveries = [
            LIVERIES[0] if i == champion else next(bots) for i in range(num_cars)
        ]
        followed = np.full(len(poses), champion)
    else:
        liveries = [LIVERIES[i % len(LIVERIES)] for i in range(num_cars)]
        followed = _leaders(edges, poses, crashed)

    def fills(livery, faded):
        _, body, wings, right = livery
        colors = {
            "left": body,
            "right": right or body,
            "wings": wings,
            "dark": INK,
            "light": BRAKE_OFF,
        }
        # a crashed car stays in its colours, faded halfway to the background
        if faded:
            colors = {k: _blend(c, WELL, 0.5) for k, c in colors.items()}
        return colors

    car_fills = [(fills(lv, False), fills(lv, True)) for lv in liveries]
    parts = [
        (shape * data["car_size"] * CAR_SCALE * scale, kind)
        for shape, kind in CAR_PARTS
    ]
    fonts = ImageFont.load_default(size=HUD_FONT * SUPERSAMPLE)
    # how hard each car brakes at each step, as a share of the most it can: the
    # speed it loses, over the speed lost in a step of full braking
    braking = np.zeros(crashed.shape)
    if "speed" in data:
        lost = -np.diff(data["speed"], axis=0, prepend=data["speed"][:1])
        full = data["limits"][2] if len(data["limits"]) > 2 else 0.45
        braking = np.where(crashed, 0.0, np.clip(lost / full, 0, 1))
    frames = []
    camera = poses[0, followed[0], :2].astype(float)
    for t in range(0, len(poses), F1TENTH_FRAME_STRIDE):
        camera += CAMERA_EASE * (poses[t, followed[t], :2] - camera)
        # the camera's top-left corner, in the track image
        cx, cy = (camera - lo) * scale
        left, top = cx - width / 2, track_size[1] - cy - height / 2
        image = track.crop(
            (round(left), round(top), round(left) + width, round(top) + height)
        )
        draw = ImageDraw.Draw(image)
        # the followed car goes on last, over the rest
        order = [i for i in range(num_cars) if i != followed[t]] + [followed[t]]
        for i in order:
            x, y, yaw = poses[t, i]
            px = (x - lo[0]) * scale - round(left)
            py = track_size[1] - (y - lo[1]) * scale - round(top)
            cos, sin = np.cos(yaw), np.sin(yaw)
            colors = car_fills[i][int(crashed[t, i])]
            if braking[t, i] >= BRAKING_FROM:
                colors = {**colors, "light": BRAKE_ON}
            for shape, kind in parts:
                ox, oy = shape.T
                # y is up in the world and down on the screen
                polygon = list(
                    zip(
                        (px + cos * ox - sin * oy).tolist(),
                        (py - sin * ox - cos * oy).tolist(),
                    )
                )
                draw.polygon(polygon, fill=colors[kind])
        # the map of the lap, with a dot for each car
        margin = F1TENTH_MARGIN * SUPERSAMPLE
        mx, my = width - map_size[0] - margin, margin
        image.paste(lap_map, (mx, my))
        draw.rectangle(
            (mx, my, mx + map_size[0], my + map_size[1]),
            outline=EDGE_SOFT,
            width=SUPERSAMPLE,
        )
        r = 2.5 * SUPERSAMPLE
        for i in order:
            dx = mx + (poses[t, i, 0] - map_lo[0]) * map_scale
            dy = my + map_size[1] - (poses[t, i, 1] - map_lo[1]) * map_scale
            colors = car_fills[i][int(crashed[t, i])]
            draw.ellipse(
                (dx - r, dy - r, dx + r, dy + r), fill=colors["left"], outline=INK
            )
        if "speed" in data:
            _hud(
                image,
                draw,
                liveries,
                data["speed"][t],
                data["steer"][t],
                braking[t],
                crashed[t],
                data["limits"],
                fonts,
                data["rewards"][t] if "rewards" in data else None,
            )
        frames.append(_downsample(image))
    # every colour a car and the track are drawn in, kept exact in the gif
    keep = [WELL, SUNKEN, MID, EDGE_SOFT, INK, DIM, KERB_RED, KERB_WHITE, *CHEQUER]
    keep += [_blend(INK, WELL, 0.5), BRAKE_ON, BRAKE_OFF, LAMP_OFF]
    keep += [
        _blend(c, WELL, f)
        for name, *_ in liveries
        for c in [WORDMARKS[name][0]] + [c for _, c in WORDMARKS[name][1]]
        for f in (0, 0.5)
    ]
    keep += [
        c for normal, faded in car_fills for c in (*normal.values(), *faded.values())
    ]
    return Video(frames, fps=F1TENTH_FPS, keep_colors=keep)


# ---------------------------------------------------------------- 2d classification


@renderer("classification_2d")
def render_classification_2d(data: Dict[str, np.ndarray]) -> Any:
    """data["points"]: (n, 2); data["probability"]: (n,) of the predicted class.
    Styled like the dashboard's charts: hairline grid, one baseline, mono ticks."""
    from matplotlib.figure import Figure

    def hex_(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    points, predicted = data["points"], data["probability"] > 0.5
    # a bare Figure, not pyplot: no global state, safe off the main thread
    fig = Figure(figsize=(4.5, 4.5), facecolor=hex_(WELL))
    ax = fig.subplots()
    ax.set_facecolor(hex_(WELL))
    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=np.where(predicted, hex_(ORANGE), hex_(BLUE)),
        s=9,
        alpha=0.8,
        linewidths=0,
        zorder=2,
    )
    ax.set_aspect("equal")
    ax.grid(color=hex_(EDGE_SOFT), linewidth=0.5, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(hex_(EDGE))
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(
        length=0, pad=4, labelsize=7, labelcolor=hex_(DIM), labelfontfamily="monospace"
    )
    # plain hyphens, like the dashboard's numbers, not matplotlib's unicode minus
    ax.xaxis.set_major_formatter("{x:g}")
    ax.yaxis.set_major_formatter("{x:g}")
    return fig
