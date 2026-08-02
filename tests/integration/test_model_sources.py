"""The built-in model sources must resolve *and* serve the bytes we pinned.

This is the regression test for the bug that motivated this layer: both
hard-coded defaults (``Xenova/yolov8n`` and
``onnx-community/segformer-b0-finetuned-cityscapes-1024-1024``) had quietly
disappeared from Hugging Face and started answering ``401 Unauthorized``, so
``--detect`` and ``--segment`` were dead on every cold cache. Nothing in the
offline layer can notice that — the URL is just a string there.

Since then the URLs are pinned to an immutable revision and to a SHA-256, which
gives this layer a second job: notice *tampering*, not just rot. A ``/resolve/``
URL that still answers 200 with different bytes is the scarier failure, and it
is invisible to a reachability probe.

Both jobs must fail loudly on a bad mirror and skip cleanly when the machine is
simply offline — that is what the ``network`` fixture is for.
"""

from __future__ import annotations

import httpx
import pytest

from sgtraffic import detect, segment
from sgtraffic.detect import ModelSource

from .conftest import fetch_sha256, reachable

pytestmark = pytest.mark.integration

_ALL_SOURCES = (*detect.DEFAULT_MODEL_SOURCES, *segment.DEFAULT_SEG_MODEL_SOURCES)


def _ids(sources: tuple[ModelSource, ...]) -> list[str]:
    """Name each case after its HF repo, so a red run says which mirror died."""
    return [
        s.url.removeprefix("https://huggingface.co/").split("/resolve/")[0]
        for s in sources
    ]


@pytest.mark.parametrize(
    "source", detect.DEFAULT_MODEL_SOURCES, ids=_ids(detect.DEFAULT_MODEL_SOURCES)
)
def test_default_detection_model_url_resolves(
    network: None, source: ModelSource
) -> None:
    assert reachable(source.url), f"detection model source is gone: {source.url}"


@pytest.mark.parametrize(
    "source",
    segment.DEFAULT_SEG_MODEL_SOURCES,
    ids=_ids(segment.DEFAULT_SEG_MODEL_SOURCES),
)
def test_default_segmentation_model_url_resolves(
    network: None, source: ModelSource
) -> None:
    assert reachable(source.url), f"segmentation model source is gone: {source.url}"


@pytest.mark.parametrize("source", _ALL_SOURCES, ids=_ids(_ALL_SOURCES))
def test_pinned_source_still_serves_the_bytes_we_recorded(
    network: None, source: ModelSource
) -> None:
    """Download it for real and hash it.

    A mismatch here means the pinned revision is serving something other than
    what was reviewed — re-pin only after checking the new bytes by hand
    (see the model supply chain runbook in CLAUDE.md). It is deliberately not a
    skip: unlike "no network", this one is worth waking up for.
    """
    try:
        actual = fetch_sha256(source.url)
    except httpx.TransportError as exc:  # the machine, not the mirror
        pytest.skip(f"connection dropped while downloading {source.url}: {exc}")
    assert actual == source.sha256, (
        f"pinned source changed under us: {source.url}\n"
        f"  recorded: {source.sha256}\n  actual:   {actual}"
    )


def test_every_source_is_https() -> None:
    """Weights are executed by ONNX Runtime; never fetch them over plain HTTP."""
    for source in _ALL_SOURCES:
        assert source.url.startswith("https://"), source.url
