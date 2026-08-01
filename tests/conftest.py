"""Shared test helpers: build synthetic images/tensors so the whole detection
and rendering pipeline can be exercised with no model file and no network."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image


def make_png(
    width: int, height: int, color: tuple[int, int, int] = (40, 40, 40)
) -> bytes:
    """Encode a solid-color RGB image as PNG bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def yolo_output(
    boxes: list[tuple[float, float, float, float, int, float]], anchors: int = 100
):
    """Craft a synthetic YOLOv8 ``(1, 84, anchors)`` output tensor.

    Each entry in ``boxes`` is ``(cx, cy, w, h, class_id, score)`` in letterbox
    space. Remaining anchors are left at zero (filtered out by the threshold).
    """
    preds = np.zeros((anchors, 84), dtype=np.float32)  # (anchors, 4 + 80)
    for i, (cx, cy, bw, bh, cid, score) in enumerate(boxes):
        preds[i, :4] = (cx, cy, bw, bh)
        preds[i, 4 + cid] = score
    return preds.T[np.newaxis, ...]  # -> (1, 84, anchors)


def seg_logits(class_grid: list[list[int]], num_classes: int = 19):
    """Craft a synthetic segmentation ``(1, C, h, w)`` logits tensor.

    ``class_grid`` is an ``h x w`` grid of the class id that should win each
    cell; that channel is set high so argmax selects it.
    """
    h = len(class_grid)
    w = len(class_grid[0])
    logits = np.zeros((1, num_classes, h, w), dtype=np.float32)
    for y, row in enumerate(class_grid):
        for x, cid in enumerate(row):
            logits[0, cid, y, x] = 10.0
    return logits


@pytest.fixture
def square_png() -> bytes:
    """A 640x640 image: letterboxing is identity (gain=1, pad=0)."""
    return make_png(640, 640)
