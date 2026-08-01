"""Code-defined, tileable background textures for the phase-2 'de-make'.

Where :mod:`sprites` stamps a single object into a box, a *tile* is a small
repeating pattern used to repaint a whole region. Given a per-pixel category map
(from :mod:`segment`), :func:`paint` replaces each background pixel with the
matching tile's color at ``(x % tile_w, y % tile_h)`` — so road becomes chunky
asphalt, vegetation becomes dappled green, sky becomes a flat retro band, etc.,
while foreground vehicles (drawn later by :mod:`sprites`) sit on top.

Same authoring style as sprites: a character grid + a palette, no binary assets.
Tiles are opaque (no transparent cells) and sized to wrap seamlessly.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Palette for tile cells (all opaque).
_TILE_PALETTE: dict[str, tuple[int, int, int]] = {
    "a": (48, 50, 54),  # asphalt
    "A": (60, 62, 66),  # asphalt speckle
    "p": (120, 120, 126),  # pavement
    "P": (92, 92, 98),  # pavement seam
    "f": (52, 48, 56),  # building frame
    "G": (150, 170, 205),  # lit window
    "g": (74, 92, 120),  # dark window
    "v": (40, 92, 46),  # foliage dark
    "V": (74, 132, 62),  # foliage light
    "s": (150, 176, 202),  # sky
    "S": (162, 188, 212),  # sky highlight
}

_ROAD = [
    "aaaaAaaa",
    "aaaaaaaa",
    "aAaaaaaa",
    "aaaaaaAa",
    "aaaAaaaa",
    "aaaaaaaa",
    "aaaaaaaA",
    "Aaaaaaaa",
]

_SIDEWALK = [
    "pppPpppP",
    "pppPpppP",
    "pppPpppP",
    "PPPPPPPP",
    "pppPpppP",
    "pppPpppP",
    "pppPpppP",
    "PPPPPPPP",
]

_BUILDING = [
    "ffffffff",
    "fGGffggf",
    "fGGffggf",
    "ffffffff",
    "ffffffff",
    "fggffGGf",
    "fggffGGf",
    "ffffffff",
]

_GREENERY = [
    "vVvvVvvv",
    "VvvvvVvv",
    "vvVvvvvV",
    "vVvvVvvv",
    "vvvVvvVv",
    "VvvvvVvv",
    "vvVvvvvv",
    "vVvvvVvV",
]

_SKY = [
    "ssSsssss",
    "ssssssSs",
    "Ssssssss",
    "sssssSss",
    "ssSsssss",
    "ssssssss",
    "sSssssss",
    "sssssSss",
]


def _compile(grid: list[str]) -> np.ndarray:
    """Turn a character grid into an ``(H, W, 3)`` uint8 RGB tile."""
    h, w = len(grid), len(grid[0])
    tile = np.zeros((h, w, 3), dtype=np.uint8)
    for y, row in enumerate(grid):
        if len(row) != w:
            raise ValueError(f"tile row {y} is ragged")
        for x, ch in enumerate(row):
            tile[y, x] = _TILE_PALETTE[ch]
    return tile


TILES: dict[str, np.ndarray] = {
    "road": _compile(_ROAD),
    "sidewalk": _compile(_SIDEWALK),
    "building": _compile(_BUILDING),
    "greenery": _compile(_GREENERY),
    "sky": _compile(_SKY),
}


def paint(base: Image.Image, category_map: np.ndarray) -> None:
    """Repaint ``base`` (an ``RGB`` image) wherever ``category_map`` names a tile.

    ``category_map`` is an ``(H, W)`` object array of category strings / ``None``,
    the **same size** as ``base``. Pixels whose category has a tile are replaced
    by that tile sampled at ``(x % tw, y % th)``; ``None`` pixels keep the photo.
    """
    arr = np.asarray(base)
    if category_map.shape != arr.shape[:2]:
        raise ValueError(f"category_map {category_map.shape} != image {arr.shape[:2]}")
    out = arr.copy()
    ys_grid, xs_grid = np.indices(arr.shape[:2])
    for category, tile in TILES.items():
        mask = category_map == category
        if not mask.any():
            continue
        th, tw = tile.shape[:2]
        ys, xs = ys_grid[mask], xs_grid[mask]
        out[ys, xs] = tile[ys % th, xs % tw]
    base.paste(Image.fromarray(out))
