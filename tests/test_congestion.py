"""Congestion scoring is pure math -- no model needed."""

from __future__ import annotations

from sgtraffic.congestion import score
from sgtraffic.detect import Detection

FRAME = (1920, 1080)


def _vehicle(category: str = "car", box=(0.0, 0.0, 100.0, 100.0)) -> Detection:
    return Detection(category=category, is_vehicle=True, score=0.9, box=box)


def test_empty_frame_is_quiet():
    result = score([], FRAME)
    assert result.score == 0
    assert result.label == "QUIET"
    assert result.vehicles == 0
    assert result.occupancy == 0.0


def test_more_vehicles_raise_the_score():
    few = score([_vehicle() for _ in range(2)], FRAME)
    many = score([_vehicle() for _ in range(12)], FRAME)
    assert many.score > few.score
    assert many.vehicles == 12


def test_labels_climb_with_density():
    # A dozen large boxes across a small frame should read at least BUSY.
    big = [_vehicle(box=(0.0, 0.0, 300.0, 300.0)) for _ in range(14)]
    result = score(big, (1000, 1000))
    assert result.label in {"BUSY", "JAM"}
    assert result.score >= 50


def test_occupancy_is_clamped_to_one():
    # Hugely overlapping boxes must not push occupancy past 1.0.
    huge = [_vehicle(box=(0.0, 0.0, 5000.0, 5000.0)) for _ in range(10)]
    result = score(huge, (1000, 1000))
    assert result.occupancy == 1.0
    assert 0 <= result.score <= 100


def test_pedestrians_are_counted_but_not_scored_as_vehicles():
    peds = [
        Detection(
            category="person", is_vehicle=False, score=0.9, box=(0.0, 0.0, 50.0, 50.0)
        )
        for _ in range(5)
    ]
    result = score(peds, FRAME)
    assert result.vehicles == 0
    assert result.counts["person"] == 5
    assert result.label == "QUIET"


def test_per_category_counts():
    dets = [
        _vehicle("car"),
        _vehicle("car"),
        _vehicle("bus_truck"),
        _vehicle("motorbike"),
    ]
    result = score(dets, FRAME)
    assert result.counts == {"car": 2, "bus_truck": 1, "motorbike": 1}
