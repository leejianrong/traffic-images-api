"""The built-in model URLs must actually resolve.

This is the regression test for the bug that motivated this layer: both
hard-coded defaults (``Xenova/yolov8n`` and
``onnx-community/segformer-b0-finetuned-cityscapes-1024-1024``) had quietly
disappeared from Hugging Face and started answering ``401 Unauthorized``, so
``--detect`` and ``--segment`` were dead on every cold cache. Nothing in the
offline layer can notice that — the URL is just a string there.
"""

from __future__ import annotations

import pytest

from sgtraffic import detect, segment

from .conftest import reachable

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("url", detect.DEFAULT_MODEL_URLS)
def test_default_detection_model_url_resolves(network: None, url: str) -> None:
    assert reachable(url), f"detection model source is gone: {url}"


@pytest.mark.parametrize("url", segment.DEFAULT_SEG_MODEL_URLS)
def test_default_segmentation_model_url_resolves(network: None, url: str) -> None:
    assert reachable(url), f"segmentation model source is gone: {url}"


def test_every_source_is_https() -> None:
    """Weights are executed by ONNX Runtime; never fetch them over plain HTTP."""
    for url in (*detect.DEFAULT_MODEL_URLS, *segment.DEFAULT_SEG_MODEL_URLS):
        assert url.startswith("https://"), url
