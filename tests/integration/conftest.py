"""Fixtures for the heavy test layer: real ONNX models, real network.

Everything here is expensive, so it is session-scoped and deselected by default
(see ``[tool.pytest.ini_options]`` in ``pyproject.toml``). Two rules keep the
layer usable:

* **Import cheaply.** Bare ``uv run pytest`` still *imports* these modules
  before deselecting them by marker, so ``onnxruntime`` must not be pulled in at
  module scope — only inside fixtures, where it is already lazy.
* **Skip, don't fail, on missing infrastructure.** A dead model host or an
  unreachable camera API is an environment problem, not a code defect. Those
  tests skip with a reason; only genuine wrong behaviour fails.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import httpx
import pytest
from PIL import Image, ImageDraw

from sgtraffic import api, detect, segment

_NET_TIMEOUT = 20.0
_DOWNLOAD_TIMEOUT = 180.0


def reachable(url: str) -> bool:
    """True if ``url`` answers a HEAD with a non-error status."""
    try:
        resp = httpx.head(url, follow_redirects=True, timeout=_NET_TIMEOUT)
    except httpx.HTTPError:
        return False
    return resp.status_code < 400


def fetch_sha256(url: str) -> str:
    """Stream ``url`` and return the SHA-256 of what comes back.

    Streamed and hashed rather than saved: the caller only wants the digest, and
    these are ~13-15 MB each.

    The two failure modes are kept apart on purpose, because the caller has to
    react to them differently. A dead or moved URL raises
    :class:`httpx.HTTPStatusError` and should go red. A connection that drops
    raises :class:`httpx.TransportError` and should skip — that is the machine,
    not the mirror.
    """
    digest = hashlib.sha256()
    with httpx.stream(
        "GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes(chunk_size=1 << 16):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="session")
def network() -> None:
    """Skip unless the model host is reachable at all.

    Without this, "the machine is offline" and "this particular mirror is dead"
    both look like the same red test — and the second is the one worth waking up
    for. Probing the host root separates them.
    """
    if not reachable("https://huggingface.co"):
        pytest.skip("no network access to the model host")


# --------------------------------------------------------------------------- #
# A deterministic stand-in frame
# --------------------------------------------------------------------------- #
def _synthetic_scene(width: int = 1920, height: int = 1080) -> bytes:
    """A crude road scene as JPEG bytes — always available, never the network.

    Inference tests assert on *structure* (tensor shapes, label ranges, boxes
    inside the frame), so the picture only has to be a plausible photo-shaped
    input. Content-dependent claims belong on the live-frame tests.
    """
    img = Image.new("RGB", (width, height), (150, 176, 202))  # sky
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, height // 2, width, height], fill=(70, 70, 74))  # road
    draw.rectangle([0, height // 3, width, height // 2], fill=(60, 90, 55))  # trees
    # A couple of car-ish blobs so a detection is at least possible.
    for cx in (width // 3, 2 * width // 3):
        draw.rectangle(
            [cx - 90, height - 340, cx + 90, height - 220], fill=(180, 40, 40)
        )
        draw.rectangle(
            [cx - 60, height - 400, cx + 60, height - 340], fill=(140, 180, 210)
        )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture(scope="session")
def synthetic_frame() -> bytes:
    """JPEG bytes of a deterministic, offline road-ish scene."""
    return _synthetic_scene()


# --------------------------------------------------------------------------- #
# Real models
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def detector_model() -> Path:
    """Resolve (downloading + caching on first use) the real YOLOv8n model."""
    try:
        return detect.resolve_model()
    except detect.DetectionError as exc:
        pytest.skip(f"detection model unavailable: {exc}")


@pytest.fixture(scope="session")
def segmenter_model() -> Path:
    """Resolve (downloading + caching on first use) the real SegFormer model."""
    try:
        return segment.resolve_model()
    except segment.SegmentationError as exc:
        pytest.skip(f"segmentation model unavailable: {exc}")


@pytest.fixture(scope="session")
def detector(detector_model: Path) -> detect.Detector:
    """A real ONNX Runtime session over the real detection weights."""
    return detect.Detector(detector_model)


@pytest.fixture(scope="session")
def segmenter(segmenter_model: Path) -> segment.Segmenter:
    """A real ONNX Runtime session over the real segmentation weights."""
    return segment.Segmenter(segmenter_model)


# --------------------------------------------------------------------------- #
# The live camera API
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def live_cameras() -> list[api.Camera]:
    """Every currently-live camera, or skip if Data.gov.sg can't be reached."""
    try:
        with httpx.Client(timeout=_NET_TIMEOUT) as client:
            cams = api.fetch_cameras(client)
    except api.TrafficAPIError as exc:
        pytest.skip(f"live camera API unavailable: {exc}")
    if not cams:
        pytest.skip("live camera API returned no cameras right now")
    return cams


@pytest.fixture(scope="session")
def live_frame(live_cameras: list[api.Camera]) -> tuple[api.Camera, bytes]:
    """A camera plus its freshly downloaded snapshot bytes.

    Which camera is live varies by the minute, so we take the first one the API
    offers rather than pinning an id that may not exist today.
    """
    cam = live_cameras[0]
    try:
        with httpx.Client(timeout=_NET_TIMEOUT) as client:
            image_bytes = api.download_image(cam.image_url, client)
    except api.TrafficAPIError as exc:
        pytest.skip(f"could not download a live snapshot: {exc}")
    return cam, image_bytes
