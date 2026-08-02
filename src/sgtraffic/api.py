"""Thin client for the Data.gov.sg real-time Traffic Images API.

Endpoint (v1, stable, no auth required):
    https://api.data.gov.sg/v1/transport/traffic-images

Response shape (trimmed)::

    {
      "items": [
        {
          "timestamp": "2026-07-07T11:20:53+08:00",
          "cameras": [
            {
              "camera_id": "2701",
              "timestamp": "...",
              "image": "https://images.data.gov.sg/.../xxx.jpg",
              "location": {"latitude": 1.29, "longitude": 103.85},
              "image_metadata": {"height": 1080, "width": 1920, "md5": "..."}
            }
          ]
        }
      ]
    }
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .locations import expressway, friendly_name

API_URL = "https://api.data.gov.sg/v1/transport/traffic-images"
# Pretend to be a normal client; some CDNs are picky about empty UAs.
_HEADERS = {"User-Agent": "sgtraffic-cli/0.1 (+https://data.gov.sg)"}
_TIMEOUT = httpx.Timeout(15.0, connect=10.0)


@dataclass(slots=True, frozen=True)
class Camera:
    """A single traffic camera snapshot returned by the API."""

    camera_id: str
    image_url: str
    timestamp: str
    latitude: float
    longitude: float
    width: int
    height: int

    @property
    def name(self) -> str:
        """Human-friendly location, e.g. 'Woodlands Checkpoint'."""
        return friendly_name(self.camera_id)

    @property
    def expressway(self) -> str:
        """Expressway/zone code, e.g. 'PIE', 'KPE', 'BKE'."""
        return expressway(self.camera_id)


class TrafficAPIError(RuntimeError):
    """Raised when the API is unreachable or returns an unexpected payload."""


def _parse(payload: dict) -> list[Camera]:
    """Turn the raw JSON into a sorted list of :class:`Camera` objects."""
    try:
        cameras_raw = payload["items"][0]["cameras"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TrafficAPIError("Unexpected API response structure") from exc

    cameras: list[Camera] = []
    for c in cameras_raw:
        # Guard the whole entry, not just the envelope above: a single camera
        # missing "camera_id"/"image", or carrying a null/non-numeric latitude,
        # would otherwise escape as a raw KeyError/TypeError/ValueError and slip
        # past every `except TrafficAPIError` handler in the CLI.
        try:
            meta = c.get("image_metadata") or {}
            loc = c.get("location") or {}
            cameras.append(
                Camera(
                    camera_id=str(c["camera_id"]),
                    image_url=c["image"],
                    timestamp=c.get("timestamp", ""),
                    latitude=float(loc.get("latitude") or 0.0),
                    longitude=float(loc.get("longitude") or 0.0),
                    width=int(meta.get("width") or 0),
                    height=int(meta.get("height") or 0),
                )
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            ident = c.get("camera_id", "<unknown>") if isinstance(c, dict) else c
            raise TrafficAPIError(
                f"Unexpected camera entry for {ident!r}: {exc}"
            ) from exc
    # Stable ordering: group by expressway, then id, so menus are predictable.
    cameras.sort(key=lambda cam: (cam.expressway, cam.camera_id))
    return cameras


def fetch_cameras(client: httpx.Client | None = None) -> list[Camera]:
    """Fetch the current list of live traffic cameras.

    Pass an existing :class:`httpx.Client` to reuse a connection across an
    auto-refresh loop; otherwise a short-lived one is created.
    """
    owns_client = client is None
    client = client or httpx.Client(headers=_HEADERS, timeout=_TIMEOUT)
    try:
        resp = client.get(API_URL)
        resp.raise_for_status()
        return _parse(resp.json())
    except httpx.HTTPError as exc:
        raise TrafficAPIError(f"Failed to reach traffic API: {exc}") from exc
    finally:
        if owns_client:
            client.close()


def download_image(url: str, client: httpx.Client | None = None) -> bytes:
    """Download the raw JPEG bytes for a camera snapshot."""
    owns_client = client is None
    client = client or httpx.Client(headers=_HEADERS, timeout=_TIMEOUT)
    try:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content
    except httpx.HTTPError as exc:
        raise TrafficAPIError(f"Failed to download image: {exc}") from exc
    finally:
        if owns_client:
            client.close()


def get_camera(camera_id: str, client: httpx.Client | None = None) -> Camera:
    """Fetch a single camera by id, raising if it isn't currently live."""
    for cam in fetch_cameras(client):
        if cam.camera_id == camera_id:
            return cam
    raise TrafficAPIError(f"Camera {camera_id!r} is not currently available")
