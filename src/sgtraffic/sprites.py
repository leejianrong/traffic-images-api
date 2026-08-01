"""Hand-authored pixel-art sprites, defined entirely in code.

No binary assets: every sprite is a small grid of single-character *cells* plus
a :data:`PALETTE` mapping each character to an RGB color. ``.`` means a fully
transparent cell. This keeps the whole sprite set diff-able in git and trivial
to tweak -- edit a row, see the pixel move.

A sprite is compiled once into an ``RGBA`` NumPy array (with a real alpha
channel from the ``.`` cells), then *stamped* onto the small pre-half-block
image at a detection's box, scaled with nearest-neighbor so it stays crisp and
chunky rather than blurring like the photo underneath.

The car body color is a *variant slot* (the ``C`` cell): the same silhouette is
recolored into a few palettes so a jam doesn't look like a row of clones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# Palette. '.' is transparent and never appears here.
# --------------------------------------------------------------------------- #
_OUTLINE = (18, 18, 26)  # near-black chassis outline -- makes sprites pop
_TIRE = (10, 10, 12)
_GLASS = (150, 220, 255)  # window glass
_LIGHT = (255, 224, 92)  # head/tail light
_SKIN = (240, 200, 160)
_CLOTH = (60, 120, 200)

PALETTE: dict[str, tuple[int, int, int]] = {
    "D": _OUTLINE,
    "K": _TIRE,
    "W": _GLASS,
    "L": _LIGHT,
    "S": _SKIN,
    "P": _CLOTH,
    # Body-color variants (the 'C' slot is filled per-variant at compile time).
    "r": (214, 52, 46),  # red
    "b": (46, 112, 212),  # blue
    "w": (226, 226, 232),  # white
    "o": (240, 150, 40),  # orange (bus/truck)
    "y": (245, 205, 60),  # yellow (bus/truck variant)
}

# Car body color variants, chosen deterministically per detection so the
# assignment is stable and the tests are reproducible.
CAR_VARIANTS: tuple[str, ...] = ("r", "b", "w")
BUS_VARIANTS: tuple[str, ...] = ("o", "y")

# --------------------------------------------------------------------------- #
# The art. 'C' is the recolorable body slot; '.' is transparent.
# --------------------------------------------------------------------------- #
_CAR = [
    "................",
    ".....DDDDDD.....",
    "....DCCCCCCD....",
    "...DCWWWWWWCD...",
    "..DCCWWWWWWCCD..",
    ".DCCCCCCCCCCCCD.",
    "LDCCCCCCCCCCCCDL",
    ".DDKKDDDDDDKKDD.",
    "...KKK....KKK...",
    "................",
]

_BUS = [
    "................",
    ".DDDDDDDDDDDDDD.",
    ".DCWWDWWDWWDWCD.",
    ".DCWWDWWDWWDWCD.",
    ".DCCCCCCCCCCCCD.",
    ".DCCCCCCCCCCCCD.",
    "LDCCCCCCCCCCCCDL",
    ".DDKKDDDDDDKKDD.",
    "...KKK....KKK...",
    "................",
]

_BIKE = [
    "................",
    "......DDD.......",
    ".....DPPPD......",
    ".....DPPPD......",
    "....DDCCCDD.....",
    "...DCCCCCCCD....",
    "..DKKD..DKKD....",
    "..KKKK..KKKK....",
    "...KK....KK.....",
    "................",
]

_PERSON = [
    "....",
    ".SS.",
    ".SS.",
    "PPPP",
    "PPPP",
    "PPPP",
    ".PP.",
    ".P.P",
    ".P.P",
    "....",
]


@dataclass(slots=True, frozen=True)
class Sprite:
    """A compiled sprite: an ``(H, W, 4)`` uint8 RGBA array ready to stamp."""

    name: str
    rgba: np.ndarray  # (H, W, 4) uint8

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` in sprite pixels."""
        h, w = self.rgba.shape[:2]
        return w, h


def _compile(grid: list[str], name: str, *, body: str | None = None) -> Sprite:
    """Turn a character grid into an RGBA array using :data:`PALETTE`.

    ``body`` names the palette key that fills the ``C`` body slot (e.g. ``"r"``
    for a red car). Cells equal to ``.`` become fully transparent.
    """
    h = len(grid)
    w = len(grid[0])
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for y, row in enumerate(grid):
        if len(row) != w:
            raise ValueError(f"sprite {name!r} row {y} is ragged")
        for x, ch in enumerate(row):
            if ch == ".":
                continue  # transparent
            key = body if (ch == "C" and body is not None) else ch
            r, g, b = PALETTE[key]
            rgba[y, x] = (r, g, b, 255)
    return Sprite(name=name, rgba=rgba)


# Pre-compile every sprite/variant once at import.
_CARS: dict[str, Sprite] = {v: _compile(_CAR, f"car_{v}", body=v) for v in CAR_VARIANTS}
_BUSES: dict[str, Sprite] = {
    v: _compile(_BUS, f"bus_{v}", body=v) for v in BUS_VARIANTS
}
_BIKE_SPRITE = _compile(_BIKE, "motorbike", body="r")
_PERSON_SPRITE = _compile(_PERSON, "person")


def for_category(category: str, *, seed: int = 0) -> Sprite:
    """Pick the sprite for a detection ``category``.

    ``seed`` (e.g. derived from the detection's position) deterministically
    selects a color variant so nearby vehicles don't all share one paint job
    while staying stable enough for reproducible tests.
    """
    if category == "car":
        return _CARS[CAR_VARIANTS[seed % len(CAR_VARIANTS)]]
    if category == "bus_truck":
        return _BUSES[BUS_VARIANTS[seed % len(BUS_VARIANTS)]]
    if category == "motorbike":
        return _BIKE_SPRITE
    if category == "person":
        return _PERSON_SPRITE
    # Unknown category -> fall back to a neutral car so we still draw *something*.
    return _CARS[CAR_VARIANTS[0]]


def stamp(base: Image.Image, sprite: Sprite, box: tuple[int, int, int, int]) -> None:
    """Paste ``sprite`` onto ``base`` (an ``RGB`` image), filling ``box``.

    ``box`` is ``(x0, y0, x1, y1)`` in ``base``'s pixel coordinates. The sprite
    is nearest-neighbor scaled to the box so it stays a crisp pixel-art block,
    and its alpha channel is used as the paste mask so transparent cells let the
    photo show through. Clipped safely to the image bounds.
    """
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0
    if bw < 1 or bh < 1:
        return
    scaled = Image.fromarray(sprite.rgba, "RGBA").resize(
        (bw, bh), Image.Resampling.NEAREST
    )
    # paste() uses the RGBA image's own alpha as the mask when passed as `mask`.
    base.paste(scaled, (x0, y0), scaled)
