"""Turning renderable values into bytes the dashboard can show. Used by the client
for media a run logs directly, and by the server for the episodes it renders."""

import io
import json
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass
class Video:
    """Frames (PIL images) of an episode, shown as an animated gif."""

    frames: List[Any]
    fps: int = 25
    # RGB colours the gif must keep exactly, for pictures with small details in
    # their own colours: its palette is otherwise picked from each frame's most
    # common colours, which can leave a few pixels of one colour none of their own
    keep_colors: Optional[List[Tuple[int, int, int]]] = None


def _quantize(frames: List[Any], keep: List[Tuple[int, int, int]]) -> List[Any]:
    """Frames in one 256-colour palette: the colours to keep, and the rest picked
    from a sample of the frames, for everything in between (antialiased edges)."""
    from PIL import Image

    keep = list(dict.fromkeys(tuple(c) for c in keep))[:256]
    colors = [c for rgb in keep for c in rgb]
    room = 256 - len(keep)
    if room:
        sample = [f.convert("RGB") for f in frames[:: max(1, len(frames) // 16)]]
        width, height = sample[0].size
        strip = Image.new("RGB", (width, height * len(sample)))
        for i, frame in enumerate(sample):
            strip.paste(frame, (0, i * height))
        rest = strip.quantize(colors=room).getpalette() or []
        colors += rest[: 3 * room]
    palette = Image.new("P", (1, 1))
    palette.putpalette(colors + colors[:3] * (256 - len(colors) // 3))
    return [
        f.convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE)
        for f in frames
    ]


def encode_media(value: Any) -> Optional[Tuple[str, bytes]]:
    """(content type, bytes) for a value the monitor can display, or None if the
    value isn't media."""
    if isinstance(value, list):
        # a bare list of frames
        is_frames = bool(value) and hasattr(value[0], "save")
        return encode_media(Video(value)) if is_frames else None
    if isinstance(value, Video):
        frames = value.frames
        if value.keep_colors:
            frames = _quantize(frames, value.keep_colors)
        buf = io.BytesIO()
        frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=round(1000 / value.fps),
            loop=0,
        )
        return "image/gif", buf.getvalue()
    # structured media the dashboard draws itself, e.g. a network ({"type": "network"})
    if isinstance(value, dict):
        return "application/json", json.dumps(value).encode()
    buf = io.BytesIO()
    # matplotlib figure
    if hasattr(value, "savefig"):
        value.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        return "image/png", buf.getvalue()
    # PIL image
    if hasattr(value, "save") and hasattr(value, "mode"):
        value.save(buf, format="PNG")
        return "image/png", buf.getvalue()
    return None
