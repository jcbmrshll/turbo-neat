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
from PIL import Image, ImageDraw

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
