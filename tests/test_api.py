"""Data.gov.sg client: payload parsing, error wrapping and image download,
driven through an injected ``httpx.MockTransport`` so the suite stays offline."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from conftest import make_png

from sgtraffic import api
from sgtraffic.api import Camera, TrafficAPIError, download_image, fetch_cameras

Handler = Callable[[httpx.Request], httpx.Response]


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def camera_entry(camera_id: str, **overrides: Any) -> dict[str, Any]:
    """One realistic ``cameras[]`` entry, with fields overridable per test."""
    entry: dict[str, Any] = {
        "camera_id": camera_id,
        "timestamp": "2026-07-07T11:20:53+08:00",
        "image": f"https://images.data.gov.sg/camera-images/{camera_id}.jpg",
        "location": {"latitude": 1.29531332, "longitude": 103.87111},
        "image_metadata": {"height": 1080, "width": 1920, "md5": "cafe"},
    }
    entry.update(overrides)
    return entry


def payload(*entries: dict[str, Any]) -> dict[str, Any]:
    """Wrap camera entries in the API's ``items[0].cameras`` envelope."""
    return {
        "items": [{"timestamp": "2026-07-07T11:20:53+08:00", "cameras": [*entries]}]
    }


def mock_client(handler: Handler) -> httpx.Client:
    """An httpx client whose transport is answered entirely in-process."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def json_client(body: dict[str, Any], status: int = 200) -> httpx.Client:
    return mock_client(lambda _request: httpx.Response(status, json=body))


def bytes_client(body: bytes, status: int = 200) -> httpx.Client:
    return mock_client(lambda _request: httpx.Response(status, content=body))


def broken_client(exc: httpx.HTTPError) -> httpx.Client:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise exc

    return mock_client(handler)


# --------------------------------------------------------------------------- #
# fetch_cameras -- happy path
# --------------------------------------------------------------------------- #
def test_fetch_cameras_parses_a_realistic_payload():
    with json_client(payload(camera_entry("6710"))) as client:
        cameras = fetch_cameras(client)

    assert len(cameras) == 1
    cam = cameras[0]
    assert isinstance(cam, Camera)
    assert cam.camera_id == "6710"
    assert cam.image_url.endswith("6710.jpg")
    assert cam.timestamp == "2026-07-07T11:20:53+08:00"
    assert cam.latitude == pytest.approx(1.29531332)
    assert cam.longitude == pytest.approx(103.87111)
    assert cam.width == 1920
    assert cam.height == 1080


def test_fetch_cameras_sorts_by_expressway_then_id():
    body = payload(
        camera_entry("6712"),  # PIE
        camera_entry("1701"),  # CTE
        camera_entry("6710"),  # PIE
        camera_entry("2701"),  # BKE
    )
    with json_client(body) as client:
        ids = [cam.camera_id for cam in fetch_cameras(client)]

    assert ids == ["2701", "1701", "6710", "6712"]  # BKE, CTE, then PIE by id


def test_fetch_cameras_leaves_an_injected_client_open_for_reuse():
    # The refresh loop reuses one connection; only self-made clients get closed.
    client = json_client(payload(camera_entry("2701")))
    fetch_cameras(client)
    assert not client.is_closed
    client.close()


# --------------------------------------------------------------------------- #
# fetch_cameras -- camera-count robustness
# --------------------------------------------------------------------------- #
def test_fetch_cameras_handles_an_empty_camera_list():
    with json_client(payload()) as client:
        assert fetch_cameras(client) == []


def test_fetch_cameras_handles_a_large_camera_list():
    entries = [camera_entry(str(9000 + i)) for i in range(90)]
    with json_client(payload(*entries)) as client:
        cameras = fetch_cameras(client)

    assert len(cameras) == 90
    assert len({cam.camera_id for cam in cameras}) == 90


# --------------------------------------------------------------------------- #
# _parse -- malformed payloads
# --------------------------------------------------------------------------- #
def test_parse_missing_items_key_raises():
    with pytest.raises(TrafficAPIError, match="Unexpected API response structure"):
        api._parse({"help": "nothing here"})


def test_parse_empty_items_list_raises():
    with pytest.raises(TrafficAPIError, match="Unexpected API response structure"):
        api._parse({"items": []})


def test_parse_item_without_cameras_raises():
    with pytest.raises(TrafficAPIError, match="Unexpected API response structure"):
        api._parse({"items": [{"timestamp": "2026-07-07T11:20:53+08:00"}]})


def test_parse_wrong_typed_item_raises():
    with pytest.raises(TrafficAPIError, match="Unexpected API response structure"):
        api._parse({"items": [None]})


def test_parse_keeps_the_original_error_as_the_cause():
    with pytest.raises(TrafficAPIError) as excinfo:
        api._parse({})
    assert isinstance(excinfo.value.__cause__, KeyError)


def test_parse_defaults_absent_nested_fields_to_zero():
    entry = camera_entry("2701")
    del entry["location"]
    del entry["image_metadata"]
    cam = api._parse(payload(entry))[0]
    assert (cam.latitude, cam.longitude) == (0.0, 0.0)
    assert (cam.width, cam.height) == (0, 0)


def test_parse_defaults_null_nested_fields_to_zero():
    entry = camera_entry("2701", location=None, image_metadata=None)
    cam = api._parse(payload(entry))[0]
    assert cam.latitude == 0.0
    assert cam.width == 0


def test_parse_defaults_an_absent_timestamp_to_empty():
    entry = camera_entry("2701")
    del entry["timestamp"]
    assert api._parse(payload(entry))[0].timestamp == ""


# --------------------------------------------------------------------------- #
# fetch_cameras -- failures
# --------------------------------------------------------------------------- #
def test_fetch_cameras_wraps_network_errors():
    boom = httpx.ConnectError("name resolution failed")
    with broken_client(boom) as client, pytest.raises(TrafficAPIError) as excinfo:
        fetch_cameras(client)

    assert "Failed to reach traffic API" in str(excinfo.value)
    assert excinfo.value.__cause__ is boom  # chained with `raise ... from exc`


def test_fetch_cameras_wraps_non_2xx_responses():
    with json_client({}, status=503) as client, pytest.raises(TrafficAPIError) as ei:
        fetch_cameras(client)

    assert "Failed to reach traffic API" in str(ei.value)
    assert isinstance(ei.value.__cause__, httpx.HTTPStatusError)


def test_fetch_cameras_does_not_swallow_a_malformed_payload_as_a_network_error():
    with json_client({"items": []}) as client, pytest.raises(TrafficAPIError) as ei:
        fetch_cameras(client)

    assert "Unexpected API response structure" in str(ei.value)


# --------------------------------------------------------------------------- #
# download_image
# --------------------------------------------------------------------------- #
def test_download_image_returns_the_raw_bytes_unchanged():
    jpeg = make_png(8, 8)
    with bytes_client(jpeg) as client:
        assert download_image("https://images.data.gov.sg/x.jpg", client) == jpeg


def test_download_image_wraps_network_errors():
    boom = httpx.ReadTimeout("too slow")
    with broken_client(boom) as client, pytest.raises(TrafficAPIError) as excinfo:
        download_image("https://images.data.gov.sg/x.jpg", client)

    assert "Failed to download image" in str(excinfo.value)
    assert excinfo.value.__cause__ is boom


def test_download_image_wraps_non_2xx_responses():
    with bytes_client(b"", status=404) as client, pytest.raises(TrafficAPIError) as ei:
        download_image("https://images.data.gov.sg/gone.jpg", client)

    assert "Failed to download image" in str(ei.value)
    assert isinstance(ei.value.__cause__, httpx.HTTPStatusError)


# --------------------------------------------------------------------------- #
# get_camera
# --------------------------------------------------------------------------- #
def test_get_camera_returns_the_matching_live_camera():
    body = payload(camera_entry("2701"), camera_entry("6710"))
    with json_client(body) as client:
        cam = api.get_camera("6710", client)

    assert cam.camera_id == "6710"


def test_get_camera_raises_naming_the_missing_id():
    body = payload(camera_entry("2701"))
    with json_client(body) as client, pytest.raises(TrafficAPIError, match="9999"):
        api.get_camera("9999", client)


def test_get_camera_raises_when_nothing_is_live():
    with (
        json_client(payload()) as client,
        pytest.raises(TrafficAPIError, match="not currently available"),
    ):
        api.get_camera("2701", client)


# --------------------------------------------------------------------------- #
# Camera dataclass
# --------------------------------------------------------------------------- #
def make_camera(camera_id: str) -> Camera:
    return Camera(
        camera_id=camera_id,
        image_url=f"https://images.data.gov.sg/{camera_id}.jpg",
        timestamp="2026-07-07T11:20:53+08:00",
        latitude=1.29,
        longitude=103.85,
        width=1920,
        height=1080,
    )


def test_camera_is_frozen():
    cam = make_camera("2702")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cam.camera_id = "9999"  # type: ignore[misc]


def test_camera_properties_use_the_location_map():
    cam = make_camera("2702")
    assert cam.name == "Woodlands Checkpoint (towards BKE)"
    assert cam.expressway == "BKE"


def test_camera_properties_fall_back_for_an_unmapped_id():
    cam = make_camera("9999")
    assert cam.name == "Camera 9999"
    assert cam.expressway == "OTHER"
