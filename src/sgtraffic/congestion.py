"""Turn a list of detections into a single 'how busy is this road' score.

Because v1 has no road-area segmentation, we can't measure true road occupancy.
Instead we blend two cheap, robust signals:

* **count**  -- how many vehicles are visible, and
* **occupancy** -- the fraction of the frame their boxes cover (a stand-in for
  density that naturally accounts for near vs. far cameras: a few close cars
  cover as much frame as many distant ones).

The two are combined into a 0-100 score and bucketed into a friendly label.
Everything here is pure and side-effect free, so it unit-tests without a model.
"""

from __future__ import annotations

from dataclasses import dataclass

from .detect import Detection

# Tunables. N_REF is "a busy-ish number of vehicles for a typical view"; OCC_REF
# is the box-coverage fraction we treat as effectively saturated.
_N_REF = 14.0
_OCC_REF = 0.45
_COUNT_WEIGHT = 0.6
_OCC_WEIGHT = 0.4

# Score band -> label. Ordered low to high; first band whose ceiling the score
# is under wins.
_BANDS: tuple[tuple[float, str], ...] = (
    (25.0, "QUIET"),
    (50.0, "MODERATE"),
    (75.0, "BUSY"),
    (100.1, "JAM"),
)


@dataclass(slots=True, frozen=True)
class Congestion:
    """A scored traffic reading for one frame."""

    score: int  # 0-100
    label: str  # QUIET / MODERATE / BUSY / JAM
    vehicles: int  # total vehicle detections
    occupancy: float  # fraction of frame covered by vehicle boxes (0-1)
    counts: dict[str, int]  # per-category tallies (car, bus_truck, ...)


def _label_for(score: float) -> str:
    for ceiling, label in _BANDS:
        if score < ceiling:
            return label
    return _BANDS[-1][1]


def score(detections: list[Detection], frame_wh: tuple[int, int]) -> Congestion:
    """Score a frame's ``detections``; ``frame_wh`` is the source ``(w, h)``.

    Only vehicles feed the score (pedestrians are tallied but don't make a road
    'busy'). Occupancy is clamped to the frame area so wildly overlapping boxes
    can't push it past 1.0.
    """
    frame_w, frame_h = frame_wh
    frame_area = float(frame_w * frame_h) or 1.0

    counts: dict[str, int] = {}
    vehicle_area = 0.0
    vehicles = 0
    for det in detections:
        counts[det.category] = counts.get(det.category, 0) + 1
        if det.is_vehicle:
            vehicles += 1
            x0, y0, x1, y1 = det.box
            vehicle_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)

    occupancy = min(1.0, vehicle_area / frame_area)

    count_component = min(1.0, vehicles / _N_REF)
    occ_component = min(1.0, occupancy / _OCC_REF)
    raw = 100.0 * (_COUNT_WEIGHT * count_component + _OCC_WEIGHT * occ_component)
    final = int(round(raw))

    return Congestion(
        score=final,
        label=_label_for(final),
        vehicles=vehicles,
        occupancy=occupancy,
        counts=counts,
    )
