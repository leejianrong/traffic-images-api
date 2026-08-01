"""CLI integration: the gauge builder and the detect->stamp->score path,
driven by a fake detector so no model or network is involved."""

from __future__ import annotations

from conftest import make_png

from sgtraffic import cli, render
from sgtraffic.congestion import Congestion
from sgtraffic.detect import Detection


class _FakeDetector:
    def __init__(self, detections: list[Detection]):
        self._detections = detections

    def detect(self, _image_bytes: bytes) -> list[Detection]:
        return self._detections


def test_gauge_visible_width_is_accurate():
    cong = Congestion(
        score=72, label="BUSY", vehicles=8, occupancy=0.3, counts={"car": 8}
    )
    text, visible = cli._gauge(cong, width=100)
    # The reported width must match the *visible* characters (no ANSI escapes).
    stripped = text.replace(render.RESET, "")
    for color in cli._GAUGE_COLORS.values():
        stripped = stripped.replace(color, "")
    assert len(stripped) == visible


def test_gauge_never_exceeds_width():
    cong = Congestion(
        score=100, label="JAM", vehicles=99, occupancy=1.0, counts={"car": 99}
    )
    _text, visible = cli._gauge(cong, width=30)
    assert visible <= 30


def test_overlay_sprites_returns_result_and_score():
    dets = [
        Detection(
            category="car", is_vehicle=True, score=0.9, box=(50.0, 50.0, 150.0, 130.0)
        ),
        Detection(
            category="bus_truck",
            is_vehicle=True,
            score=0.8,
            box=(300.0, 100.0, 500.0, 260.0),
        ),
    ]
    result, cong = cli._overlay_sprites(
        make_png(640, 480), bits=4, detector=_FakeDetector(dets)
    )
    assert isinstance(result, render.RenderResult)
    assert cong.vehicles == 2
    assert result.rows > 0


def test_overlay_with_no_detections_scores_quiet():
    result, cong = cli._overlay_sprites(
        make_png(640, 480), bits=4, detector=_FakeDetector([])
    )
    assert cong.label == "QUIET"
    assert cong.vehicles == 0
    assert result.cols > 0
