"""Detector math (letterbox / NMS / postprocess) tested against synthetic
tensors, plus model resolution and a fake-session end-to-end -- all offline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import make_png, yolo_output

from sgtraffic import detect
from sgtraffic.detect import (
    Detection,
    DetectionError,
    Detector,
    letterbox,
    nms,
    postprocess,
    resolve_model,
    to_blob,
)


# --------------------------------------------------------------------------- #
# letterbox
# --------------------------------------------------------------------------- #
def test_letterbox_preserves_aspect_and_reports_padding():
    img = np.zeros((100, 200, 3), dtype=np.uint8)  # H=100, W=200
    canvas, gain, (pad_w, pad_h) = letterbox(img, new_size=640)
    assert canvas.shape == (640, 640, 3)
    assert gain == pytest.approx(3.2)
    # Wide image -> full width, padded top/bottom.
    assert pad_w == pytest.approx(0.0)
    assert pad_h == pytest.approx(160.0)


def test_to_blob_layout_and_range():
    canvas = np.full((640, 640, 3), 255, dtype=np.uint8)
    blob = to_blob(canvas)
    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    assert blob.max() <= 1.0 and blob.min() >= 0.0


# --------------------------------------------------------------------------- #
# nms
# --------------------------------------------------------------------------- #
def test_nms_suppresses_overlapping_boxes():
    boxes = np.array(
        [[0, 0, 100, 100], [10, 10, 105, 105], [500, 500, 600, 600]], dtype=float
    )
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_thres=0.5)
    assert 0 in keep  # highest score survives
    assert 1 not in keep  # heavy overlap with 0 -> dropped
    assert 2 in keep  # far away -> kept


def test_nms_empty():
    assert nms(np.empty((0, 4)), np.empty((0,)), 0.5) == []


# --------------------------------------------------------------------------- #
# postprocess
# --------------------------------------------------------------------------- #
def test_postprocess_decodes_classes_and_boxes():
    # Two boxes in 640 space; identity letterbox (gain=1, pad=0, 640x640 frame).
    output = yolo_output(
        [
            (100.0, 100.0, 40.0, 40.0, 2, 0.9),  # car
            (500.0, 300.0, 60.0, 20.0, 5, 0.8),  # bus
        ]
    )
    dets = postprocess(output, gain=1.0, pad=(0.0, 0.0), orig_wh=(640, 640))
    cats = sorted(d.category for d in dets)
    assert cats == ["bus_truck", "car"]
    car = next(d for d in dets if d.category == "car")
    # cx=100,w=40 -> x0=80, x1=120
    assert car.box[0] == pytest.approx(80.0)
    assert car.box[2] == pytest.approx(120.0)


def test_postprocess_thresholds_low_confidence():
    output = yolo_output([(100.0, 100.0, 40.0, 40.0, 2, 0.10)])
    dets = postprocess(
        output, gain=1.0, pad=(0.0, 0.0), orig_wh=(640, 640), conf_thres=0.25
    )
    assert dets == []


def test_postprocess_ignores_unmapped_classes():
    # class 15 (cat) is not in our COCO map -> dropped even at high confidence.
    output = yolo_output([(100.0, 100.0, 40.0, 40.0, 15, 0.99)])
    dets = postprocess(output, gain=1.0, pad=(0.0, 0.0), orig_wh=(640, 640))
    assert dets == []


def test_postprocess_undoes_padding_and_gain():
    # gain=0.5, pad_h=160: a box at letterbox y must map back correctly.
    output = yolo_output([(100.0, 260.0, 40.0, 40.0, 2, 0.9)])
    dets = postprocess(output, gain=0.5, pad=(0.0, 160.0), orig_wh=(400, 400))
    d = dets[0]
    # x: (100 - 0)/0.5 center; x0 = (80)/0.5 = 160 ; y0 = (240 - 160)/0.5 = 160
    assert d.box[0] == pytest.approx(160.0)
    assert d.box[1] == pytest.approx(160.0)


# --------------------------------------------------------------------------- #
# resolve_model
# --------------------------------------------------------------------------- #
def test_resolve_explicit_missing_raises(tmp_path: Path):
    with pytest.raises(DetectionError, match="not found"):
        resolve_model(str(tmp_path / "nope.onnx"))


def test_resolve_explicit_present_returns(tmp_path: Path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"stub")
    assert resolve_model(str(model)) == model


def test_resolve_no_cache_no_download_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(detect, "_CACHE_DIR", tmp_path)
    monkeypatch.delenv("SGTRAFFIC_MODEL", raising=False)
    with pytest.raises(DetectionError, match="downloads are disabled"):
        resolve_model(None, allow_download=False)


def test_resolve_uses_env_var(tmp_path: Path, monkeypatch):
    model = tmp_path / "env.onnx"
    model.write_bytes(b"stub")
    monkeypatch.setenv("SGTRAFFIC_MODEL", str(model))
    assert resolve_model(None) == model


# --------------------------------------------------------------------------- #
# Detector.detect via a fake session (no onnxruntime model required)
# --------------------------------------------------------------------------- #
class _FakeSession:
    def __init__(self, output):
        self._output = output

    def run(self, _outputs, _feed):
        return [self._output]


def test_detector_detect_end_to_end():
    output = yolo_output([(320.0, 320.0, 80.0, 40.0, 2, 0.9)])
    det = Detector.__new__(Detector)  # bypass __init__ (no real model)
    det.conf_thres = 0.25
    det.iou_thres = 0.45
    det._session = _FakeSession(output)  # type: ignore[attr-defined]
    det._input_name = "images"  # type: ignore[attr-defined]

    results = det.detect(make_png(640, 640))
    assert len(results) == 1
    assert isinstance(results[0], Detection)
    assert results[0].category == "car"
    assert results[0].is_vehicle
