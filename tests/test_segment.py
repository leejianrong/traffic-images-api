"""Segmenter math (preprocess / postprocess / category map), model resolution,
and a fake-session end-to-end -- all offline, no model file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import make_png, seg_logits

from sgtraffic import segment
from sgtraffic.segment import (
    Segmenter,
    SegmentationError,
    postprocess,
    preprocess,
    resolve_model,
    to_category_map,
)


# --------------------------------------------------------------------------- #
# preprocess
# --------------------------------------------------------------------------- #
def test_preprocess_shape_and_original_size():
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)  # H=240, W=320
    blob, orig_wh = preprocess(rgb, size=512)
    assert blob.shape == (1, 3, 512, 512)
    assert blob.dtype == np.float32
    assert orig_wh == (320, 240)


def test_preprocess_normalizes_with_imagenet_stats():
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)  # all-black -> (0 - mean)/std
    blob, _ = preprocess(rgb, size=16)
    # Black pixels map to -mean/std, which is negative for every channel.
    assert blob.max() < 0


# --------------------------------------------------------------------------- #
# postprocess
# --------------------------------------------------------------------------- #
def test_postprocess_argmax_and_upsample():
    logits = seg_logits([[0, 8], [10, 13]])  # 2x2 class grid
    labels = postprocess(logits, orig_wh=(4, 4))
    assert labels.shape == (4, 4)  # upsampled to original size
    # Nearest upsample: top-left stays class 0, top-right class 8.
    assert labels[0, 0] == 0
    assert labels[0, 3] == 8
    assert labels[3, 0] == 10


# --------------------------------------------------------------------------- #
# to_category_map
# --------------------------------------------------------------------------- #
def test_to_category_map_maps_known_and_blanks_unknown():
    label_map = np.array([[0, 8], [13, 10]], dtype=np.uint8)  # road, veg, car, sky
    cats = to_category_map(label_map)
    assert cats[0, 0] == "road"
    assert cats[0, 1] == "greenery"
    assert cats[1, 1] == "sky"
    assert cats[1, 0] is None  # class 13 (car) is not a background category


# --------------------------------------------------------------------------- #
# resolve_model
# --------------------------------------------------------------------------- #
def test_resolve_explicit_missing_raises(tmp_path: Path):
    with pytest.raises(SegmentationError, match="not found"):
        resolve_model(str(tmp_path / "nope.onnx"))


def test_resolve_no_cache_no_download_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(segment, "_CACHE_DIR", tmp_path)
    monkeypatch.delenv("SGTRAFFIC_SEG_MODEL", raising=False)
    with pytest.raises(SegmentationError, match="downloads are disabled"):
        resolve_model(None, allow_download=False)


def test_resolve_uses_env_var(tmp_path: Path, monkeypatch):
    model = tmp_path / "seg.onnx"
    model.write_bytes(b"stub")
    monkeypatch.setenv("SGTRAFFIC_SEG_MODEL", str(model))
    assert resolve_model(None) == model


# --------------------------------------------------------------------------- #
# Segmenter.segment via a fake session
# --------------------------------------------------------------------------- #
class _FakeSession:
    def __init__(self, output):
        self._output = output

    def run(self, _outputs, _feed):
        return [self._output]


def test_segmenter_returns_label_map_at_original_size():
    # All-road logits -> label map of zeros at the source resolution.
    logits = seg_logits([[0, 0], [0, 0]])
    seg = Segmenter.__new__(Segmenter)  # bypass __init__ (no real model)
    seg._session = _FakeSession(logits)  # type: ignore[attr-defined]
    seg._input_name = "pixel_values"  # type: ignore[attr-defined]

    labels = seg.segment(make_png(48, 32))  # W=48, H=32
    assert labels.shape == (32, 48)
    assert (labels == 0).all()
