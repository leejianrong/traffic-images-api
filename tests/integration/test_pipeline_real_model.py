"""The whole de-make, end to end, with real weights.

These drive :func:`cli._analyze_frame` — the actual orchestration the ``view``
command runs — rather than re-implementing the wiring in the test. That is the
point of this layer: the fast tests each verify one stage in isolation, so a
mistake in how the stages are *joined* (a label map resized to the wrong buffer,
sprites scaled by the wrong factor) slips straight through them.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from sgtraffic import cli, render, segment, tiles
from sgtraffic.congestion import Congestion
from sgtraffic.detect import Detector
from sgtraffic.segment import Segmenter

pytestmark = pytest.mark.integration


def _buffer(image_bytes: bytes) -> tuple[Image.Image, tuple[int, int]]:
    return render.prepare_frame(image_bytes, bits=4)


def test_detect_only_path_renders_and_scores(
    detector: Detector, live_frame: tuple[object, bytes]
) -> None:
    """bytes -> detections -> sprites -> half-block lines, plus a congestion read."""
    _cam, image_bytes = live_frame
    result, cong = cli._analyze_frame(image_bytes, bits=4, detector=detector)

    assert result.cols > 0
    assert result.lines
    assert all(len(line) > 0 for line in result.lines)
    assert any("\x1b[48;2;" in line for line in result.lines), "no truecolor output"

    assert isinstance(cong, Congestion)
    assert 0 <= cong.score <= 100
    assert cong.label in {"QUIET", "MODERATE", "BUSY", "JAM"}
    assert cong.vehicles == sum(n for cat, n in cong.counts.items() if cat != "person")
    assert 0.0 <= cong.occupancy <= 1.0


def test_segment_only_path_repaints_most_of_the_buffer(
    segmenter: Segmenter, live_frame: tuple[object, bytes]
) -> None:
    """Tiling must visibly change the frame, not silently no-op."""
    _cam, image_bytes = live_frame
    before = np.asarray(_buffer(image_bytes)[0].convert("RGB")).copy()

    result, cong = cli._analyze_frame(image_bytes, bits=4, segmenter=segmenter)
    assert cong is None, "--segment alone must not produce a congestion score"
    assert result.lines

    # Re-run the background stage on our own buffer so we can diff the pixels.
    small, _ = _buffer(image_bytes)
    cli._paint_background(image_bytes, small, segmenter)
    after = np.asarray(small.convert("RGB"))

    changed = int((before != after).any(axis=2).sum())
    total = before.shape[0] * before.shape[1]
    assert changed > total * 0.1, (
        f"tiles.paint only touched {changed}/{total} pixels — the segmentation "
        "stage is effectively a no-op"
    )

    # Every repainted pixel must carry a colour that came from a tile palette.
    palette = {tuple(c) for tile in tiles.TILES.values() for c in tile.reshape(-1, 3)}
    mask = (before != after).any(axis=2)
    assert {tuple(px) for px in after[mask]} <= palette


def test_full_demake_layers_sprites_over_tiles(
    detector: Detector, segmenter: Segmenter, live_frame: tuple[object, bytes]
) -> None:
    """`-d -s` together: both stages run, and sprites win where they overlap."""
    _cam, image_bytes = live_frame
    result, cong = cli._analyze_frame(
        image_bytes, bits=4, detector=detector, segmenter=segmenter
    )
    assert result.lines
    assert cong is not None

    small, orig_wh = _buffer(image_bytes)
    cli._paint_background(image_bytes, small, segmenter)
    tiled = np.asarray(small.convert("RGB")).copy()
    scored = cli._stamp_detections(image_bytes, small, detector, orig_wh)
    stamped = np.asarray(small.convert("RGB"))

    # The render buffer is ~114x64, so a distant car can legitimately scale down
    # to nothing. Only demand visible sprites when at least one box survives.
    (buf_w, buf_h), (orig_w, orig_h) = small.size, orig_wh
    visible = [
        d
        for d in detector.detect(image_bytes)
        if round((d.box[2] - d.box[0]) * buf_w / orig_w) >= 1
        and round((d.box[3] - d.box[1]) * buf_h / orig_h) >= 1
    ]
    if visible:
        assert (tiled != stamped).any(), (
            f"{len(visible)} detections scaled to >=1px but sprites.stamp changed "
            "nothing on top of the tiled background"
        )
    assert scored.vehicles == cong.vehicles


def test_label_map_survives_the_resize_to_the_render_buffer(
    segmenter: Segmenter, live_frame: tuple[object, bytes]
) -> None:
    """Nearest-resizing the *integer* map must not invent classes."""
    _cam, image_bytes = live_frame
    small, _ = _buffer(image_bytes)
    labels = segmenter.segment(image_bytes)

    resized = np.asarray(
        Image.fromarray(labels, "L").resize(small.size, Image.Resampling.NEAREST)
    )
    assert set(np.unique(resized).tolist()) <= set(np.unique(labels).tolist())

    categories = segment.to_category_map(resized)
    assert categories.shape == (small.size[1], small.size[0])


def test_renders_at_a_range_of_bit_depths(
    detector: Detector, segmenter: Segmenter, synthetic_frame: bytes
) -> None:
    """--bits is a posterization knob; the pipeline must survive its extremes."""
    for bits in (1, 4, 8):
        result, cong = cli._analyze_frame(
            synthetic_frame, bits=bits, detector=detector, segmenter=segmenter
        )
        assert result.lines, f"empty render at --bits {bits}"
        assert cong is not None


def test_live_snapshot_is_a_decodable_photo(live_frame: tuple[object, bytes]) -> None:
    """Guards the assumption every CV stage makes about the API's payload."""
    cam, image_bytes = live_frame
    assert len(image_bytes) > 10_000, f"suspiciously small snapshot for {cam}"
    with Image.open(io.BytesIO(image_bytes)) as img:
        assert img.format == "JPEG"
        assert img.size[0] >= 320 and img.size[1] >= 240
