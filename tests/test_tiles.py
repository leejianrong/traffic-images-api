"""Tile compilation + background painting."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from sgtraffic import tiles


def test_all_tiles_compile_to_rgb():
    for name, tile in tiles.TILES.items():
        assert tile.ndim == 3 and tile.shape[2] == 3, name
        assert tile.dtype == np.uint8


def test_paint_replaces_mapped_regions_only():
    base = Image.new("RGB", (16, 10), (0, 0, 0))
    cats = np.empty((10, 16), dtype=object)  # (H, W)
    cats[0:5, :] = "sky"  # top half -> sky tile
    cats[5:10, :] = "road"  # bottom half -> road tile
    cats[7, 0] = None  # one pixel left as photo

    tiles.paint(base, cats)
    arr = np.asarray(base)

    sky_colors = {tiles._TILE_PALETTE["s"], tiles._TILE_PALETTE["S"]}
    assert tuple(arr[0, 0]) in sky_colors  # sky region painted
    assert tuple(arr[7, 0]) == (0, 0, 0)  # None pixel untouched
    # Road region is asphalt, not the original black.
    assert tuple(arr[9, 5]) != (0, 0, 0)


def test_paint_tiles_wrap():
    # A region wider than the 8px tile must repeat, not error.
    base = Image.new("RGB", (20, 4), (0, 0, 0))
    cats = np.empty((4, 20), dtype=object)
    cats[:, :] = "road"
    tiles.paint(base, cats)
    arr = np.asarray(base)
    # Column 0 and column 8 sample the same tile x -> identical asphalt.
    assert np.array_equal(arr[:, 0], arr[:, 8])


def test_paint_size_mismatch_raises():
    base = Image.new("RGB", (16, 10), (0, 0, 0))
    wrong = np.empty((5, 5), dtype=object)
    with pytest.raises(ValueError, match="!="):
        tiles.paint(base, wrong)
