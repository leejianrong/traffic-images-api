"""Model acquisition: mirror fallback and integrity verification, offline.

Background: both hard-coded default URLs (``Xenova/yolov8n`` for detection,
``onnx-community/segformer-b0-...`` for segmentation) vanished from Hugging Face
and began answering ``401 Unauthorized``. Because each module fetched from a
single URL, one dead mirror killed the feature outright for every cold cache.
The mirror list fixed availability; pinned revisions plus SHA-256 fixed the
other half — the bytes at those URLs are executed by ONNX Runtime, so "we got
*a* file" is not the same as "we got *the* file".

The reachability and real digests of the pinned URLs are checked by
``tests/integration``; what is verified here — with no network at all — is the
*mechanism*: try each source in turn, hash what arrives, refuse anything that
doesn't match, and stay out of the way when the user supplies their own model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Self

import pytest

from sgtraffic import detect, segment
from sgtraffic.detect import DetectionError, ModelIntegrityError, ModelSource

_PAYLOAD = b"pretend-this-is-onnx" * 100
_DIGEST = hashlib.sha256(_PAYLOAD).hexdigest()


class _Recorder:
    """Stands in for ``detect.download_model``, failing a chosen set of URLs."""

    def __init__(self, dead: set[str]) -> None:
        self.dead = dead
        self.tried: list[str] = []

    def __call__(self, url: str, dest: Path, *, sha256=None, on_progress=None) -> Path:
        self.tried.append(url)
        if url in self.dead:
            raise DetectionError(f"Could not download model from {url}: 401")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"onnx")
        return dest


def _source(url: str, sha256: str | None = "0" * 64) -> ModelSource:
    return ModelSource(url=url, sha256=sha256)


class _FakeTransport:
    """A minimal stand-in for ``httpx.stream`` that serves fixed bytes."""

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __call__(self, method: str, url: str, **kwargs: object) -> Self:
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    @property
    def headers(self) -> dict[str, str]:
        return {"Content-Length": str(len(self.body))}

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, chunk_size: int = 1 << 16) -> Iterator[bytes]:
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i : i + chunk_size]


# --------------------------------------------------------------------------- #
# The source lists themselves
# --------------------------------------------------------------------------- #
def test_default_sources_are_a_non_empty_list() -> None:
    """A single hard-coded URL is exactly what broke; keep more than one."""
    assert len(detect.DEFAULT_MODEL_SOURCES) >= 2
    assert len(segment.DEFAULT_SEG_MODEL_SOURCES) >= 2
    for src in (*detect.DEFAULT_MODEL_SOURCES, *segment.DEFAULT_SEG_MODEL_SOURCES):
        assert src.url.startswith("https://")


def test_every_default_source_is_pinned_to_an_immutable_revision() -> None:
    """``/resolve/main/`` re-resolves on every fetch, so the bytes can change
    under us with no signal at all. Pin the commit instead."""
    for src in (*detect.DEFAULT_MODEL_SOURCES, *segment.DEFAULT_SEG_MODEL_SOURCES):
        assert "/resolve/main/" not in src.url, src.url
        revision = src.url.split("/resolve/", 1)[1].split("/", 1)[0]
        assert len(revision) == 40 and set(revision) <= set("0123456789abcdef"), (
            f"not a 40-hex commit sha: {src.url}"
        )


def test_every_default_source_carries_a_digest() -> None:
    """``sha256=None`` means "trust it" — allowed for a user-supplied URL,
    never for one of ours."""
    for src in (*detect.DEFAULT_MODEL_SOURCES, *segment.DEFAULT_SEG_MODEL_SOURCES):
        assert src.sha256 is not None, src.url
        assert len(src.sha256) == 64, src.url


# --------------------------------------------------------------------------- #
# Falling through dead mirrors
# --------------------------------------------------------------------------- #
def test_first_reachable_source_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _Recorder(dead=set())
    monkeypatch.setattr(detect, "download_model", recorder)

    dest = tmp_path / "model.onnx"
    got = detect.download_first(
        [_source("https://a/x.onnx"), _source("https://b/x.onnx")], dest
    )

    assert got == dest
    assert recorder.tried == ["https://a/x.onnx"], "should not try the spare mirror"


def test_a_dead_mirror_falls_through_to_the_next(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The regression: a 401 on the first source must not kill the download."""
    recorder = _Recorder(dead={"https://dead/x.onnx"})
    monkeypatch.setattr(detect, "download_model", recorder)

    dest = tmp_path / "model.onnx"
    got = detect.download_first(
        [_source("https://dead/x.onnx"), _source("https://live/x.onnx")], dest
    )

    assert got.read_bytes() == b"onnx"
    assert recorder.tried == ["https://dead/x.onnx", "https://live/x.onnx"]


def test_all_sources_dead_reports_every_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    urls = ["https://dead1/x.onnx", "https://dead2/x.onnx"]
    monkeypatch.setattr(detect, "download_model", _Recorder(dead=set(urls)))

    with pytest.raises(DetectionError) as excinfo:
        detect.download_first([_source(u) for u in urls], tmp_path / "model.onnx")

    message = str(excinfo.value)
    for url in urls:
        assert url in message


def test_download_first_passes_each_sources_digest_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Digests are per-source, not per-model: two mirrors can legitimately serve
    two different (both valid) exports."""
    seen: list[tuple[str, str | None]] = []

    def _fake(url: str, dest: Path, *, sha256=None, on_progress=None) -> Path:
        seen.append((url, sha256))
        raise DetectionError("nope")

    monkeypatch.setattr(detect, "download_model", _fake)
    with pytest.raises(DetectionError):
        detect.download_first(
            [_source("https://a/x.onnx", "aa"), _source("https://b/x.onnx", "bb")],
            tmp_path / "m.onnx",
        )
    assert seen == [("https://a/x.onnx", "aa"), ("https://b/x.onnx", "bb")]


# --------------------------------------------------------------------------- #
# Integrity: the download path
# --------------------------------------------------------------------------- #
def test_download_accepts_bytes_matching_the_pinned_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(detect.httpx, "stream", _FakeTransport(_PAYLOAD))
    dest = tmp_path / "model.onnx"

    got = detect.download_model(
        "https://example/model.onnx", dest, sha256=_DIGEST, on_progress=None
    )

    assert got.read_bytes() == _PAYLOAD


def test_download_rejects_bytes_that_do_not_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole point: a mirror that answers 200 with *something else* must not
    reach ONNX Runtime."""
    monkeypatch.setattr(detect.httpx, "stream", _FakeTransport(b"tampered"))
    dest = tmp_path / "model.onnx"

    with pytest.raises(ModelIntegrityError) as excinfo:
        detect.download_model("https://evil/model.onnx", dest, sha256=_DIGEST)

    message = str(excinfo.value)
    assert "Integrity check FAILED" in message
    assert "not a network fault" in message
    assert _DIGEST in message


def test_a_rejected_download_leaves_nothing_behind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Neither the destination nor the .part scratch file may survive — a stale
    ``.part`` would be resumed-looking junk and a stale ``dest`` would be run."""
    monkeypatch.setattr(detect.httpx, "stream", _FakeTransport(b"tampered"))
    dest = tmp_path / "model.onnx"

    with pytest.raises(ModelIntegrityError):
        detect.download_model("https://evil/model.onnx", dest, sha256=_DIGEST)

    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_download_without_a_digest_skips_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(detect.httpx, "stream", _FakeTransport(b"whatever"))
    dest = tmp_path / "model.onnx"

    assert detect.download_model("https://user/model.onnx", dest).read_bytes() == (
        b"whatever"
    )


def test_a_tampered_mirror_falls_through_to_a_good_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A wrong-bytes mirror is treated like a dead one — try the next pin rather
    than fail the feature — but the failure line has to say which it was."""
    bodies = {"https://bad/x.onnx": b"tampered", "https://good/x.onnx": _PAYLOAD}
    calls: list[str] = []

    def _stream(method: str, url: str, **kwargs: object) -> _FakeTransport:
        calls.append(url)
        return _FakeTransport(bodies[url])

    monkeypatch.setattr(detect.httpx, "stream", _stream)
    dest = tmp_path / "model.onnx"

    got = detect.download_first(
        [
            _source("https://bad/x.onnx", _DIGEST),
            _source("https://good/x.onnx", _DIGEST),
        ],
        dest,
    )

    assert got.read_bytes() == _PAYLOAD
    assert calls == ["https://bad/x.onnx", "https://good/x.onnx"]


def test_every_mirror_tampered_is_reported_as_an_integrity_problem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(detect.httpx, "stream", _FakeTransport(b"tampered"))

    with pytest.raises(DetectionError) as excinfo:
        detect.download_first(
            [
                _source("https://a/x.onnx", _DIGEST),
                _source("https://b/x.onnx", _DIGEST),
            ],
            tmp_path / "model.onnx",
        )

    message = str(excinfo.value)
    assert "Integrity check FAILED" in message
    assert message.count("Integrity check FAILED") == 2


# --------------------------------------------------------------------------- #
# Integrity: the cached-file path
# --------------------------------------------------------------------------- #
def test_sha256_file_streams_the_same_digest_as_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(_PAYLOAD)
    assert detect.sha256_file(path, chunk_size=7) == _DIGEST


def test_verify_cached_accepts_any_pinned_digest(tmp_path: Path) -> None:
    """The cache does not record which mirror it came from, so any of the
    pinned digests is a legitimate answer."""
    path = tmp_path / "model.onnx"
    path.write_bytes(_PAYLOAD)
    assert detect.verify_cached(path, {"0" * 64, _DIGEST}) == path


def test_verify_cached_deletes_and_raises_on_mismatch(tmp_path: Path) -> None:
    """Otherwise the guarantee expires the moment the first download lands."""
    path = tmp_path / "model.onnx"
    path.write_bytes(b"corrupted-on-disk")

    with pytest.raises(ModelIntegrityError) as excinfo:
        detect.verify_cached(path, {_DIGEST})

    assert not path.exists(), "a failing cache entry must not be left runnable"
    assert "integrity check" in str(excinfo.value).lower()
    assert "NOT a network problem" in str(excinfo.value)


def test_verify_cached_with_no_digests_is_a_no_op(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"user-supplied")
    assert detect.verify_cached(path, frozenset()) == path
    assert path.exists()


def test_resolve_model_verifies_the_cache_every_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SGTRAFFIC_MODEL", raising=False)
    monkeypatch.delenv("SGTRAFFIC_MODEL_URL", raising=False)
    monkeypatch.setattr(detect, "_CACHE_DIR", tmp_path)
    cached = tmp_path / "yolov8n.onnx"
    cached.write_bytes(b"not the pinned model")

    with pytest.raises(ModelIntegrityError):
        detect.resolve_model()
    assert not cached.exists()


def test_segment_resolve_model_verifies_the_cache_as_a_segmentation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """detect's ModelIntegrityError must never leak out of the segment path,
    but the "not the network" wording has to survive the re-wrap."""
    monkeypatch.delenv("SGTRAFFIC_SEG_MODEL", raising=False)
    monkeypatch.delenv("SGTRAFFIC_SEG_MODEL_URL", raising=False)
    monkeypatch.setattr(segment, "_CACHE_DIR", tmp_path)
    cached = tmp_path / "segformer_cityscapes.onnx"
    cached.write_bytes(b"not the pinned model")

    with pytest.raises(segment.SegmentationError) as excinfo:
        segment.resolve_model()

    assert not isinstance(excinfo.value, DetectionError)
    assert "NOT a network problem" in str(excinfo.value)
    assert not cached.exists()


# --------------------------------------------------------------------------- #
# User-supplied models are exempt, on purpose
# --------------------------------------------------------------------------- #
def test_model_url_override_is_a_single_unverified_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGTRAFFIC_MODEL_URL", "https://mine/custom.onnx")
    sources = detect.configured_sources()
    assert [s.url for s in sources] == ["https://mine/custom.onnx"]
    assert sources[0].sha256 is None, "our digest has no authority over their weights"
    assert detect.trusted_digests(sources) == frozenset()


def test_seg_model_url_override_is_a_single_unverified_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGTRAFFIC_SEG_MODEL_URL", "https://mine/seg.onnx")
    sources = segment.configured_sources()
    assert [s.url for s in sources] == ["https://mine/seg.onnx"]
    assert sources[0].sha256 is None


def test_env_url_override_skips_cache_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Their URL, their bytes: a cache filled from it would fail our digest
    forever if we checked it."""
    monkeypatch.delenv("SGTRAFFIC_MODEL", raising=False)
    monkeypatch.setenv("SGTRAFFIC_MODEL_URL", "https://mine/custom.onnx")
    monkeypatch.setattr(detect, "_CACHE_DIR", tmp_path)
    cached = tmp_path / "yolov8n.onnx"
    cached.write_bytes(b"their model, not ours")

    assert detect.resolve_model() == cached
    assert cached.exists()


def test_seg_env_url_override_skips_cache_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SGTRAFFIC_SEG_MODEL", raising=False)
    monkeypatch.setenv("SGTRAFFIC_SEG_MODEL_URL", "https://mine/seg.onnx")
    monkeypatch.setattr(segment, "_CACHE_DIR", tmp_path)
    cached = tmp_path / "segformer_cityscapes.onnx"
    cached.write_bytes(b"their model, not ours")

    assert segment.resolve_model() == cached
    assert cached.exists()


def test_explicit_model_path_is_never_checksummed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--model`` hands us a file that was never ours to verify."""
    monkeypatch.delenv("SGTRAFFIC_MODEL", raising=False)
    monkeypatch.delenv("SGTRAFFIC_SEG_MODEL", raising=False)
    mine = tmp_path / "mine.onnx"
    mine.write_bytes(b"a totally different detector")

    assert detect.resolve_model(str(mine)) == mine
    assert segment.resolve_model(str(mine)) == mine


# --------------------------------------------------------------------------- #
# Wiring: resolve_model actually uses the lists
# --------------------------------------------------------------------------- #
def test_detect_resolve_model_appends_its_own_remediation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """download_first is shared with segmentation, so the '--model' hint has to
    be added by the detection caller — otherwise a failed *segmentation*
    download tells the user to set $SGTRAFFIC_MODEL_URL."""
    monkeypatch.delenv("SGTRAFFIC_MODEL", raising=False)
    monkeypatch.delenv("SGTRAFFIC_MODEL_URL", raising=False)
    monkeypatch.setattr(detect, "_CACHE_DIR", tmp_path)
    dead = {s.url for s in detect.DEFAULT_MODEL_SOURCES}
    monkeypatch.setattr(detect, "download_model", _Recorder(dead=dead))

    with pytest.raises(DetectionError) as excinfo:
        detect.resolve_model()
    assert "--model" in str(excinfo.value)


def test_detect_resolve_model_uses_the_fallback_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """resolve_model must go through download_first, not a single URL."""
    monkeypatch.delenv("SGTRAFFIC_MODEL", raising=False)
    monkeypatch.delenv("SGTRAFFIC_MODEL_URL", raising=False)
    monkeypatch.setattr(detect, "_CACHE_DIR", tmp_path)
    seen: list[list[ModelSource]] = []

    def _fake_first(sources, dest: Path, *, on_progress=None) -> Path:
        seen.append(list(sources))
        dest.write_bytes(b"onnx")
        return dest

    monkeypatch.setattr(detect, "download_first", _fake_first)
    path = detect.resolve_model()

    assert path == tmp_path / "yolov8n.onnx"
    assert seen == [list(detect.DEFAULT_MODEL_SOURCES)]


def test_segment_resolve_model_uses_the_fallback_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SGTRAFFIC_SEG_MODEL", raising=False)
    monkeypatch.delenv("SGTRAFFIC_SEG_MODEL_URL", raising=False)
    monkeypatch.setattr(segment, "_CACHE_DIR", tmp_path)
    seen: list[list[ModelSource]] = []

    def _fake_first(sources, dest: Path, *, on_progress=None) -> Path:
        seen.append(list(sources))
        dest.write_bytes(b"onnx")
        return dest

    monkeypatch.setattr(segment, "download_first", _fake_first)
    path = segment.resolve_model()

    assert path == tmp_path / "segformer_cityscapes.onnx"
    assert seen == [list(segment.DEFAULT_SEG_MODEL_SOURCES)]


def test_segment_download_failure_is_a_segmentation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """detect's DetectionError must never leak out of the segmentation path."""
    monkeypatch.delenv("SGTRAFFIC_SEG_MODEL", raising=False)
    monkeypatch.setattr(segment, "_CACHE_DIR", tmp_path)

    def _boom(sources, dest: Path, *, on_progress=None) -> Path:
        raise DetectionError("every mirror is gone")

    monkeypatch.setattr(segment, "download_first", _boom)

    with pytest.raises(segment.SegmentationError) as excinfo:
        segment.resolve_model()
    assert "--seg-model" in str(excinfo.value)
