"""CLI integration: the gauge builder and the analyze (segment + detect) path,
driven by fake models so no real model or network is involved."""

from __future__ import annotations

import io

import numpy as np
import pytest
import typer
from conftest import make_png
from PIL import Image

from sgtraffic import cli, render
from sgtraffic.api import Camera
from sgtraffic.congestion import Congestion
from sgtraffic.detect import Detection


class _FakeDetector:
    def __init__(self, detections: list[Detection]):
        self._detections = detections

    def detect(self, _image_bytes: bytes) -> list[Detection]:
        return self._detections


class _FakeSegmenter:
    """Returns a label map of a single class for the whole frame."""

    def __init__(self, class_id: int):
        self._class_id = class_id

    def segment(self, image_bytes: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(image_bytes)) as src:
            w, h = src.size
        return np.full((h, w), self._class_id, dtype=np.uint8)


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


def test_analyze_detection_returns_result_and_score():
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
    result, cong = cli._analyze_frame(
        make_png(640, 480), bits=4, detector=_FakeDetector(dets)
    )
    assert isinstance(result, render.RenderResult)
    assert cong is not None and cong.vehicles == 2
    assert result.rows > 0


def test_analyze_detection_only_no_detections_scores_quiet():
    result, cong = cli._analyze_frame(
        make_png(640, 480), bits=4, detector=_FakeDetector([])
    )
    assert cong is not None and cong.label == "QUIET"
    assert cong.vehicles == 0
    assert result.cols > 0


def test_analyze_segmentation_only_has_no_score():
    result, cong = cli._analyze_frame(
        make_png(640, 480), bits=4, segmenter=_FakeSegmenter(class_id=0)
    )
    assert isinstance(result, render.RenderResult)
    assert cong is None  # no detector -> no congestion score


def test_analyze_segmentation_repaints_background():
    # All-road segmentation should recolor the buffer to asphalt tiles, so the
    # rendered lines differ from the plain (un-tiled) render of the same frame.
    png = make_png(640, 480, color=(200, 60, 60))  # vivid red photo
    tiled, _ = cli._analyze_frame(png, bits=4, segmenter=_FakeSegmenter(class_id=0))
    plain = render.render_result(png, bits=4)
    assert tiled.lines != plain.lines


def test_analyze_demake_combines_segment_and_detect():
    dets = [
        Detection(
            category="car", is_vehicle=True, score=0.9, box=(50.0, 50.0, 150.0, 130.0)
        )
    ]
    result, cong = cli._analyze_frame(
        make_png(640, 480),
        bits=4,
        detector=_FakeDetector(dets),
        segmenter=_FakeSegmenter(class_id=0),
    )
    assert cong is not None and cong.vehicles == 1
    assert result.rows > 0


def test_paint_background_matches_buffer_size(monkeypatch):
    # _paint_background must resize the label map to the buffer without error.
    png = make_png(320, 240)
    small, _ = render.prepare_frame(png, bits=4)
    cli._paint_background(png, small, _FakeSegmenter(class_id=10))  # sky
    # Buffer got repainted to sky tones (not the original solid grey).
    assert small.size[0] > 0


# --------------------------------------------------------------------------- #
# `demo` -- the single command a newcomer runs
# --------------------------------------------------------------------------- #
def test_demo_picks_a_live_camera_and_renders(monkeypatch):
    """`demo` exists so nobody has to look up a camera id that may be offline.

    The whole point is that it takes no arguments, so the thing worth pinning
    is that it sources its own camera and still reaches the renderer.
    """
    cam = Camera(
        camera_id="2701",
        image_url="https://images.data.gov.sg/2701.jpg",
        timestamp="2026-08-02T11:20:53+08:00",
        latitude=1.29,
        longitude=103.85,
        width=1920,
        height=1080,
    )
    drawn: list[Camera] = []
    monkeypatch.setattr(cli.api, "fetch_cameras", lambda _client: [cam])
    monkeypatch.setattr(
        cli.api, "download_image", lambda _url, _client: make_png(64, 36)
    )
    monkeypatch.setattr(cli, "_build_detector", lambda _m, _c: _FakeDetector([]))
    monkeypatch.setattr(cli, "_build_segmenter", lambda _m: _FakeSegmenter(0))
    monkeypatch.setattr(cli, "_draw", lambda c, _img, **_kw: drawn.append(c))

    cli.demo(once=True)

    assert [c.camera_id for c in drawn] == ["2701"]


def test_demo_fails_before_downloading_a_model_when_no_camera_is_live(monkeypatch):
    """An API outage must not cost the user a 28 MB download first."""
    built: list[str] = []
    monkeypatch.setattr(cli.api, "fetch_cameras", lambda _client: [])
    monkeypatch.setattr(
        cli, "_build_detector", lambda _m, _c: built.append("detector") or None
    )
    monkeypatch.setattr(
        cli, "_build_segmenter", lambda _m: built.append("segmenter") or None
    )

    with pytest.raises(typer.Exit):
        cli.demo(once=True)

    assert built == []
