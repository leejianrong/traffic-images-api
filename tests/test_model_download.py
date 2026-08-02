"""Model acquisition: the fallback across mirrors, offline.

Background: both hard-coded default URLs (``Xenova/yolov8n`` for detection,
``onnx-community/segformer-b0-...`` for segmentation) vanished from Hugging Face
and began answering ``401 Unauthorized``. Because each module fetched from a
single URL, one dead mirror killed the feature outright for every cold cache.

The reachability of the real URLs is checked by ``tests/integration``; what is
verified here — with no network at all — is the *mechanism*: try each source in
turn, stop at the first success, and fail only once every one is gone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sgtraffic import detect, segment
from sgtraffic.detect import DetectionError


class _Recorder:
    """Stands in for ``detect.download_model``, failing a chosen set of URLs."""

    def __init__(self, dead: set[str]) -> None:
        self.dead = dead
        self.tried: list[str] = []

    def __call__(self, url: str, dest: Path, *, on_progress=None) -> Path:
        self.tried.append(url)
        if url in self.dead:
            raise DetectionError(f"Could not download model from {url}: 401")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"onnx")
        return dest


def test_default_sources_are_a_non_empty_list() -> None:
    """A single hard-coded URL is exactly what broke; keep more than one."""
    assert len(detect.DEFAULT_MODEL_URLS) >= 2
    assert len(segment.DEFAULT_SEG_MODEL_URLS) >= 2
    for url in (*detect.DEFAULT_MODEL_URLS, *segment.DEFAULT_SEG_MODEL_URLS):
        assert url.startswith("https://")


def test_first_reachable_source_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _Recorder(dead=set())
    monkeypatch.setattr(detect, "download_model", recorder)

    dest = tmp_path / "model.onnx"
    got = detect.download_first(["https://a/x.onnx", "https://b/x.onnx"], dest)

    assert got == dest
    assert recorder.tried == ["https://a/x.onnx"], "should not try the spare mirror"


def test_a_dead_mirror_falls_through_to_the_next(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The regression: a 401 on the first source must not kill the download."""
    recorder = _Recorder(dead={"https://dead/x.onnx"})
    monkeypatch.setattr(detect, "download_model", recorder)

    dest = tmp_path / "model.onnx"
    got = detect.download_first(["https://dead/x.onnx", "https://live/x.onnx"], dest)

    assert got.read_bytes() == b"onnx"
    assert recorder.tried == ["https://dead/x.onnx", "https://live/x.onnx"]


def test_all_sources_dead_reports_every_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    urls = ["https://dead1/x.onnx", "https://dead2/x.onnx"]
    monkeypatch.setattr(detect, "download_model", _Recorder(dead=set(urls)))

    with pytest.raises(DetectionError) as excinfo:
        detect.download_first(urls, tmp_path / "model.onnx")

    message = str(excinfo.value)
    for url in urls:
        assert url in message


def test_detect_resolve_model_appends_its_own_remediation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """download_first is shared with segmentation, so the '--model' hint has to
    be added by the detection caller — otherwise a failed *segmentation*
    download tells the user to set $SGTRAFFIC_MODEL_URL."""
    monkeypatch.delenv("SGTRAFFIC_MODEL", raising=False)
    monkeypatch.setattr(detect, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        detect, "download_model", _Recorder(dead=set(detect._MODEL_URLS))
    )

    with pytest.raises(DetectionError) as excinfo:
        detect.resolve_model()
    assert "--model" in str(excinfo.value)


def test_detect_resolve_model_uses_the_fallback_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """resolve_model must go through download_first, not a single URL."""
    monkeypatch.delenv("SGTRAFFIC_MODEL", raising=False)
    monkeypatch.setattr(detect, "_CACHE_DIR", tmp_path)
    seen: list[list[str]] = []

    def _fake_first(urls, dest: Path, *, on_progress=None) -> Path:
        seen.append(list(urls))
        dest.write_bytes(b"onnx")
        return dest

    monkeypatch.setattr(detect, "download_first", _fake_first)
    path = detect.resolve_model()

    assert path == tmp_path / "yolov8n.onnx"
    assert seen == [list(detect.DEFAULT_MODEL_URLS)]


def test_segment_resolve_model_uses_the_fallback_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SGTRAFFIC_SEG_MODEL", raising=False)
    monkeypatch.setattr(segment, "_CACHE_DIR", tmp_path)
    seen: list[list[str]] = []

    def _fake_first(urls, dest: Path, *, on_progress=None) -> Path:
        seen.append(list(urls))
        dest.write_bytes(b"onnx")
        return dest

    monkeypatch.setattr(segment, "download_first", _fake_first)
    path = segment.resolve_model()

    assert path == tmp_path / "segformer_cityscapes.onnx"
    assert seen == [list(segment.DEFAULT_SEG_MODEL_URLS)]


def test_segment_download_failure_is_a_segmentation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """detect's DetectionError must never leak out of the segmentation path."""
    monkeypatch.delenv("SGTRAFFIC_SEG_MODEL", raising=False)
    monkeypatch.setattr(segment, "_CACHE_DIR", tmp_path)

    def _boom(urls, dest: Path, *, on_progress=None) -> Path:
        raise DetectionError("every mirror is gone")

    monkeypatch.setattr(segment, "download_first", _boom)

    with pytest.raises(segment.SegmentationError) as excinfo:
        segment.resolve_model()
    assert "--seg-model" in str(excinfo.value)
