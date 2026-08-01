"""Image -> colored terminal art engine.

Technique: the **Unicode lower-half-block** (``▄``, U+2584).

A single terminal cell is roughly twice as tall as it is wide. If we paint the
cell's *background* with one pixel's color and its *foreground* (the ▄ glyph,
which fills the bottom half) with another, we get **two vertically-stacked
pixels per character cell** -- doubling vertical resolution for free while
keeping pixels visually square.

    top pixel    -> ANSI background color  (48;2;R;G;B)
    bottom pixel -> ANSI foreground color  (38;2;R;G;B)  painted by ▄

We emit both colors in a single SGR sequence to halve the escape-byte overhead,
and reset (``\\x1b[0m``) once per line so a resize or scroll can't bleed color.

The retro "8-bit" look comes from *posterizing* -- collapsing each channel to a
handful of levels so you get chunky color banding like an old console, plus a
saturation/contrast nudge so the arcade colors pop.
"""

from __future__ import annotations

import io
import shutil
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageEnhance, ImageOps

# The glyph whose foreground fills the lower half of the cell.
LOWER_HALF_BLOCK = "▄"  # ▄
RESET = "\x1b[0m"

# Chrome we must not draw over: panel border (2 cols / 2 rows) + a little
# breathing room for the header and footer lines the CLI prints.
_H_CHROME = 4
_V_CHROME = 8


@dataclass(slots=True, frozen=True)
class RenderSpec:
    """Resolved target dimensions for one render pass."""

    cols: int  # character columns  == output image width in pixels
    rows: int  # character rows      == output image height / 2 in pixels


def terminal_spec(
    aspect: float,
    *,
    max_cols: int = 120,
    fallback: tuple[int, int] = (80, 24),
) -> RenderSpec:
    """Compute how many character cells we can use *right now*.

    Called on every refresh so the art always fits the **current** window --
    this is what makes terminal resizing graceful: we never emit a line wider
    than the terminal, so lines never wrap and the picture never shears.

    ``aspect`` is the source image's width / height.
    """
    term_cols, term_rows = shutil.get_terminal_size(fallback)

    # Usable budget after leaving room for borders/header/footer.
    avail_cols = max(10, min(term_cols - _H_CHROME, max_cols))
    avail_rows = max(5, term_rows - _V_CHROME)

    # Each cell is 1 source-pixel wide and 2 source-pixels tall, so a cell's
    # on-screen shape is ~1:2 (w:h). To preserve the source aspect ratio:
    #     cols / (rows * 2) == aspect   =>   rows = cols / (2 * aspect)
    cols = avail_cols
    rows = max(1, round(cols / (2 * aspect)))

    # If that's too tall for the window, clamp by height and back-solve cols.
    if rows > avail_rows:
        rows = avail_rows
        cols = max(1, round(rows * 2 * aspect))

    return RenderSpec(cols=cols, rows=rows)


def _prepare(
    img: Image.Image, spec: RenderSpec, *, bits: int, punch: bool
) -> Image.Image:
    """Downscale + apply the retro color treatment.

    ``bits`` = bits-per-channel kept (lower = chunkier retro banding; 8 = off).
    ``punch`` toggles the saturation/contrast boost for the arcade look.
    """
    img = img.convert("RGB")

    # Resize to exactly cols x (rows*2) pixels -- one pixel per half-cell.
    # BILINEAR reads smoother than NEAREST once posterized on top.
    img = img.resize((spec.cols, spec.rows * 2), Image.Resampling.BILINEAR)

    if punch:
        img = ImageEnhance.Color(img).enhance(1.35)  # +35% saturation
        img = ImageEnhance.Contrast(img).enhance(1.12)  # +12% contrast

    if bits < 8:
        # Posterize: collapse each channel to 2**bits levels -> retro banding.
        img = ImageOps.posterize(img, bits)

    return img


@dataclass(slots=True, frozen=True)
class RenderResult:
    """Rendered art plus its exact on-screen dimensions.

    Because ``cols`` is known precisely (it equals the source pixel width we
    resized to), callers can frame the art without ever having to measure the
    visible width of ANSI-laden strings.
    """

    lines: list[str]
    cols: int
    rows: int


def prepare_frame(
    image_bytes: bytes,
    *,
    max_cols: int = 120,
    bits: int = 4,
    punch: bool = True,
) -> tuple[Image.Image, tuple[int, int]]:
    """Decode + downscale + retro-treat, returning the small ``RGB`` buffer.

    This is the half-way point of :func:`render_result`, exposed so callers can
    stamp sprites onto the prepared buffer *before* it's flattened into
    half-blocks (see the ``--detect`` path). Also returns the source image's
    original ``(width, height)``, which detection needs to map boxes back.
    """
    with Image.open(io.BytesIO(image_bytes)) as src:
        orig_wh = (src.width, src.height)
        aspect = (src.width / src.height) if src.height else (16 / 9)
        spec = terminal_spec(aspect, max_cols=max_cols)
        img = _prepare(src, spec, bits=bits, punch=punch)
    return img, orig_wh


def lines_from_image(img: Image.Image) -> RenderResult:
    """Flatten a prepared ``RGB`` buffer into ANSI half-block rows.

    ``img`` must already be sized to ``cols x (rows*2)`` pixels (as produced by
    :func:`prepare_frame`). Kept separate from preparation so a sprite overlay
    can slot in between the two steps.
    """
    # Flat pixel list is far faster than per-pixel getpixel() calls.
    # Pillow's PixelAccess is loosely typed (indexing looks like it could return
    # a scalar or None); we resized to RGB, so treat it as Any to keep the hot
    # loop free of per-pixel casts.
    px: Any = img.load()
    w, h = img.size

    lines: list[str] = []
    # Step two source rows at a time: y = top pixel, y+1 = bottom pixel.
    for y in range(0, h - 1, 2):
        parts: list[str] = []
        prev: tuple[int, int, int, int, int, int] | None = None
        for x in range(w):
            tr, tg, tb = px[x, y]
            br, bg, bb = px[x, y + 1]
            key = (tr, tg, tb, br, bg, bb)
            if key == prev:
                # Identical cell as the last one -> skip re-emitting the SGR
                # and just repeat the glyph. Big win on skies / road surfaces.
                parts.append(LOWER_HALF_BLOCK)
            else:
                parts.append(
                    f"\x1b[48;2;{tr};{tg};{tb};38;2;{br};{bg};{bb}m{LOWER_HALF_BLOCK}"
                )
                prev = key
        parts.append(RESET)
        lines.append("".join(parts))

    return RenderResult(lines=lines, cols=w, rows=len(lines))


def render_result(
    image_bytes: bytes,
    *,
    max_cols: int = 120,
    bits: int = 4,
    punch: bool = True,
) -> RenderResult:
    """Convert raw image bytes into ANSI-colored rows + their dimensions.

    Each returned line is already 0m-reset. Dimensions are derived from the
    live terminal size, so the caller can print the lines and trust they fit
    the current window (this is what keeps resizing graceful).
    """
    img, _ = prepare_frame(image_bytes, max_cols=max_cols, bits=bits, punch=punch)
    return lines_from_image(img)


# --------------------------------------------------------------------------- #
# Retro CRT-style framing (drawn directly, no Rich round-trip)
# --------------------------------------------------------------------------- #
# TrueColor border palette — neon cyan frame, phosphor-green accents.
_FRAME = "\x1b[38;2;0;229;255m"  # neon cyan
_ACCENT = "\x1b[38;2;57;255;20m"  # phosphor green
_DIM = "\x1b[38;2;120;120;120m"


def _center(text: str, width: int) -> str:
    """Center plain ``text`` within ``width`` visible columns."""
    text = text[:width]
    pad = width - len(text)
    left = pad // 2
    return " " * left + text + " " * (pad - left)


def _center_ansi(text: str, visible: int, width: int) -> str:
    """Center a possibly-ANSI-laden ``text`` whose visible width is ``visible``.

    We can't use ``len(text)`` here because color escapes inflate it; the caller
    tells us the true on-screen width so the padding math stays correct.
    """
    if visible >= width:
        return text
    pad = width - visible
    left = pad // 2
    return " " * left + text + " " * (pad - left)


def frame(
    result: RenderResult,
    *,
    title: str,
    subtitle: str = "",
    gauge: tuple[str, int] | None = None,
) -> str:
    """Wrap rendered art in a neon box with a title bar and status footer.

    Uses ``result.cols`` (known exactly) so no ANSI width-measurement is
    needed, and prints the raster's raw ANSI verbatim — preserving the
    run-length optimization Rich would otherwise strip.
    """
    w = result.cols
    top = f"{_FRAME}╔{'═' * w}╗\x1b[0m"
    bot = f"{_FRAME}╚{'═' * w}╝\x1b[0m"
    title_row = f"{_FRAME}║{_ACCENT}{_center(title, w)}{_FRAME}║\x1b[0m"
    sep = f"{_FRAME}╟{'─' * w}╢\x1b[0m"

    out = [top, title_row, sep]
    for ln in result.lines:
        # ln already ends in reset; border chars re-assert their own color.
        out.append(f"{_FRAME}║\x1b[0m{ln}{_FRAME}║\x1b[0m")
    if gauge is not None:
        text, visible = gauge
        out.append(sep)
        out.append(f"{_FRAME}║\x1b[0m{_center_ansi(text, visible, w)}{_FRAME}║\x1b[0m")
    if subtitle:
        out.append(sep)
        out.append(f"{_FRAME}║{_DIM}{_center(subtitle, w)}{_FRAME}║\x1b[0m")
    out.append(bot)
    return "\n".join(out)
