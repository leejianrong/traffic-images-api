"""Sprite compilation + stamping."""

from __future__ import annotations

import numpy as np
from PIL import Image

from sgtraffic import sprites


def test_compiled_sprites_have_alpha_channel():
    sprite = sprites.for_category("car")
    assert sprite.rgba.shape[2] == 4
    # Some pixels transparent (the '.' cells), some opaque (the body).
    alpha = sprite.rgba[:, :, 3]
    assert alpha.min() == 0
    assert alpha.max() == 255


def test_car_variants_differ_by_seed():
    a = sprites.for_category("car", seed=0)
    b = sprites.for_category("car", seed=1)
    assert a.name != b.name  # different color variant selected


def test_variant_selection_is_deterministic():
    assert (
        sprites.for_category("car", seed=3).name
        == sprites.for_category("car", seed=0).name
    )


def test_unknown_category_falls_back():
    sprite = sprites.for_category("spaceship")
    assert sprite.name.startswith("car_")


def test_each_category_maps_to_expected_sprite():
    assert sprites.for_category("bus_truck").name.startswith("bus_")
    assert sprites.for_category("motorbike").name == "motorbike"
    assert sprites.for_category("person").name == "person"


def test_stamp_modifies_pixels_inside_box_only():
    base = Image.new("RGB", (40, 40), (0, 0, 0))
    before = np.asarray(base).copy()
    sprites.stamp(base, sprites.for_category("car", seed=0), (10, 10, 30, 26))
    after = np.asarray(base)

    # Something changed inside the box...
    assert not np.array_equal(before[10:26, 10:30], after[10:26, 10:30])
    # ...and nothing changed in a far corner outside it.
    assert np.array_equal(before[35:40, 0:5], after[35:40, 0:5])


def test_stamp_ignores_degenerate_box():
    base = Image.new("RGB", (40, 40), (0, 0, 0))
    before = np.asarray(base).copy()
    sprites.stamp(base, sprites.for_category("car"), (10, 10, 10, 10))  # zero area
    assert np.array_equal(before, np.asarray(base))
