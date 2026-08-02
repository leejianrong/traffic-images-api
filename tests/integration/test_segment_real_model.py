"""``--segment`` against the real SegFormer (Cityscapes) weights.

The offline layer feeds :func:`segment.postprocess` synthetic logits, so it can
never catch the two things that actually break here: the downloaded graph not
being a 19-class Cityscapes model, and its channel order not matching the
``trainId`` assumption baked into :data:`segment.CITYSCAPES_TO_CATEGORY` (get
that wrong and roads get painted with the sky tile — a silently wrong picture,
not a crash).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sgtraffic import segment
from sgtraffic.segment import Segmenter

pytestmark = pytest.mark.integration

_NUM_CITYSCAPES_CLASSES = 19


def test_resolve_model_downloads_and_caches(segmenter_model: Path) -> None:
    assert segmenter_model.is_file()
    assert segmenter_model.stat().st_size > 1_000_000, "cached model is too small"
    assert segment.resolve_model(allow_download=False) == segmenter_model


def test_session_accepts_our_blob_and_returns_19_classes(
    segmenter: Segmenter, synthetic_frame: bytes
) -> None:
    """Real graph, real blob: shapes must line up with what postprocess expects."""
    (inp,) = segmenter._session.get_inputs()
    assert inp.type == "tensor(float)"

    with Image.open(io.BytesIO(synthetic_frame)) as src:
        rgb = np.asarray(src.convert("RGB"))
    blob, orig_wh = segment.preprocess(rgb)
    assert blob.shape == (1, 3, 512, 512)
    assert orig_wh == (rgb.shape[1], rgb.shape[0])

    logits = segmenter._session.run(None, {segmenter._input_name: blob})[0]
    assert logits.ndim == 4
    assert logits.shape[0] == 1
    assert logits.shape[1] == _NUM_CITYSCAPES_CLASSES
    assert np.isfinite(logits).all()


def test_segment_returns_a_full_resolution_label_map(
    segmenter: Segmenter, synthetic_frame: bytes
) -> None:
    with Image.open(io.BytesIO(synthetic_frame)) as src:
        width, height = src.size

    labels = segmenter.segment(synthetic_frame)
    assert labels.shape == (height, width)
    assert labels.dtype == np.uint8
    assert labels.max() < _NUM_CITYSCAPES_CLASSES


def test_live_frame_maps_to_recognisable_road_scene_categories(
    segmenter: Segmenter, live_frame: tuple[object, bytes]
) -> None:
    """A real expressway photo should segment into real road-scene regions.

    Deliberately weak on *which* categories appear (cameras point at bridges,
    tunnels and checkpoints), but firm that the mapping produces our vocabulary
    and that a meaningful share of the frame gets classified at all.
    """
    _cam, image_bytes = live_frame
    labels = segmenter.segment(image_bytes)
    assert labels.max() < _NUM_CITYSCAPES_CLASSES

    # np.unique can't sort an object array holding both str and None, so tally
    # the flattened list instead.
    flat = segment.to_category_map(labels).ravel().tolist()
    present = {c for c in flat if c is not None}
    assert present <= {"road", "sidewalk", "building", "greenery", "sky"}
    assert present, "the whole frame mapped to None — check the trainId ordering"

    mapped_fraction = sum(c is not None for c in flat) / len(flat)
    assert mapped_fraction > 0.2, (
        f"only {mapped_fraction:.0%} of the frame was classified; the model's "
        "class order probably does not follow Cityscapes trainIds"
    )
