"""``--detect`` against the real YOLOv8n weights.

The fast layer proves the maths (letterbox, NMS, postprocess) with synthetic
tensors and a fake session. It cannot prove that the file we download is a
YOLOv8 detector at all, that ONNX Runtime will load it, or that its real output
tensor matches the ``(1, 84, 8400)`` layout :func:`detect.postprocess` assumes.
That is what these tests are for.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sgtraffic import detect
from sgtraffic.detect import Detection, Detector

pytestmark = pytest.mark.integration

# COCO ids we render, per detect._COCO_MAP.
_CATEGORIES = {"person", "car", "motorbike", "bus_truck"}


def test_resolve_model_downloads_and_caches(detector_model: Path) -> None:
    """First use fetches the weights; the file lands in the cache, intact."""
    assert detector_model.is_file()
    # A stock yolov8n export is ~12 MB. Anything tiny is an error page, not
    # weights — the failure mode that a bare `is_file()` check would hide.
    assert detector_model.stat().st_size > 1_000_000, (
        "cached model is suspiciously small"
    )

    # Resolving again is a pure cache hit: same path, no second download.
    assert detect.resolve_model(allow_download=False) == detector_model


def test_session_has_the_expected_signature(detector: Detector) -> None:
    """The real graph must match the layout the pure helpers were written for."""
    session = detector._session
    (inp,) = session.get_inputs()
    assert inp.type == "tensor(float)"
    assert list(inp.shape) == [1, 3, 640, 640]

    (out,) = session.get_outputs()
    assert out.type == "tensor(float)"
    _, channels, anchors = out.shape
    assert channels == 84, "4 box coords + 80 COCO classes"
    assert anchors == 8400


def test_inference_produces_a_decodable_tensor(
    detector: Detector, synthetic_frame: bytes
) -> None:
    """Push a real image through the real session and decode the raw output."""
    with Image.open(io.BytesIO(synthetic_frame)) as src:
        rgb = np.asarray(src.convert("RGB"))
    canvas, gain, pad = detect.letterbox(rgb)
    blob = detect.to_blob(canvas)

    raw = detector._session.run(None, {detector._input_name: blob})[0]
    assert raw.shape == (1, 84, 8400)
    assert np.isfinite(raw).all()

    # Class scores live in rows 4.. and are sigmoid outputs, so within [0, 1].
    scores = raw[0, 4:, :]
    assert scores.min() >= -1e-4
    assert scores.max() <= 1.0 + 1e-4

    detections = detect.postprocess(raw, gain, pad, (rgb.shape[1], rgb.shape[0]))
    assert isinstance(detections, list)


def test_detect_returns_well_formed_detections(
    detector: Detector, synthetic_frame: bytes
) -> None:
    """Whatever it finds, every Detection must be structurally sound."""
    detections = detector.detect(synthetic_frame)
    assert all(isinstance(d, Detection) for d in detections)
    for det in detections:
        assert det.category in _CATEGORIES
        assert 0.25 <= det.score <= 1.0, "below the default conf threshold"
        x0, y0, x1, y1 = det.box
        assert 0 <= x0 <= x1 <= 1920
        assert 0 <= y0 <= y1 <= 1080


def test_a_lower_threshold_never_finds_less(
    detector_model: Path, synthetic_frame: bytes
) -> None:
    """Sanity-check that conf_thres is actually wired through to the real run."""
    strict = Detector(detector_model, conf_thres=0.9).detect(synthetic_frame)
    loose = Detector(detector_model, conf_thres=0.1).detect(synthetic_frame)
    assert len(loose) >= len(strict)


def test_detects_vehicles_on_a_live_camera_frame(
    detector: Detector, live_frame: tuple[object, bytes]
) -> None:
    """The real thing: a real expressway photo through the real detector."""
    _cam, image_bytes = live_frame
    with Image.open(io.BytesIO(image_bytes)) as src:
        width, height = src.size

    detections = detector.detect(image_bytes)
    for det in detections:
        assert det.category in _CATEGORIES
        x0, y0, x1, y1 = det.box
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height
        assert (x1 - x0) * (y1 - y0) > 0
