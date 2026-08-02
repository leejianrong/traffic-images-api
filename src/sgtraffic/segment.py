"""Semantic segmentation via a small ONNX model — the phase-2 'de-make' input.

Where :mod:`detect` finds *objects* (boxes for vehicles), this finds *regions*:
a per-pixel class label for the background — road, sidewalk, building,
vegetation, sky — so :mod:`tiles` can repaint each region as pixel-art texture
instead of leaving the blurred photo underneath.

Same shape as :mod:`detect`: the math (:func:`preprocess`, :func:`postprocess`)
is free-standing and unit-tested against synthetic logits; ``onnxruntime`` is
imported lazily inside :class:`Segmenter`; the model is fetched at runtime into a
cache dir (never committed) and is fully overridable. We reuse
:func:`detect.download_first` rather than duplicating the streaming download.

Class ids follow the **Cityscapes** 19-class ``trainId`` convention (the usual
output space for road-scene models). Map them to our coarse tile categories via
:data:`CITYSCAPES_TO_CATEGORY`; ids not in the map (poles, signs, and every
foreground vehicle/person — those are the detector's job) become ``None`` and
keep the photo.
"""

from __future__ import annotations

import io
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from .detect import (
    ModelIntegrityError,
    ModelSource,
    download_first,
    trusted_digests,
    verify_cached,
)

# Cityscapes trainId -> our coarse background category (None = leave the photo).
CITYSCAPES_TO_CATEGORY: dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "building",  # wall
    4: "building",  # fence
    8: "greenery",  # vegetation
    9: "greenery",  # terrain
    10: "sky",
}

# Like :data:`detect.DEFAULT_MODEL_SOURCES`, a list rather than one URL: the
# original default (``onnx-community/segformer-b0-...``) now answers 401, which
# broke ``--segment`` on every cold cache. Each entry is a SegFormer-B0 export
# fine-tuned on Cityscapes whose 19 output channels follow the trainId order
# assumed by :data:`CITYSCAPES_TO_CATEGORY`, pinned to an immutable revision and
# to the SHA-256 of the bytes at that revision.
#
# Both are by Xenova (transformers.js), which is why neither was demoted the way
# the detection mirrors were: unlike those, this is a known author.
DEFAULT_SEG_MODEL_SOURCES: tuple[ModelSource, ...] = (
    ModelSource(
        url="https://huggingface.co/Xenova/segformer-b0-finetuned-cityscapes-1024-1024/resolve/64e537ca2bf6bf2afc13727ee6bb61cfca9e1048/onnx/model.onnx",
        sha256="142427caa45a47d57d5bd23b62c3bfc852fa7df0c6c2291c5ef34f581fa406bf",
    ),
    ModelSource(
        url="https://huggingface.co/Xenova/segformer-b0-finetuned-cityscapes-512-1024/resolve/18b12204d4c787afb8df24e49efc36a20d9f16d0/onnx/model.onnx",
        sha256="ffd5ceb33f7c7a969bc69d3ad7ab06017de15c42085e7ed10be8ed8d23b10fec",
    ),
)
_CACHE_DIR = Path(
    os.environ.get("SGTRAFFIC_CACHE", Path.home() / ".cache" / "sgtraffic")
)
_SEG_FILENAME = "segformer_cityscapes.onnx"

_INPUT_SIZE = 512
# ImageNet normalization — the standard for SegFormer/DeepLab-style backbones.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SegmentationError(RuntimeError):
    """Raised when the segmentation model can't be obtained or inference fails."""


# --------------------------------------------------------------------------- #
# Pure tensor math (no onnxruntime -> unit-testable)
# --------------------------------------------------------------------------- #
def preprocess(
    rgb: np.ndarray, size: int = _INPUT_SIZE
) -> tuple[np.ndarray, tuple[int, int]]:
    """RGB (H, W, 3) uint8 -> normalized NCHW float32 blob + original ``(w, h)``.

    Segmentation models take a fixed square input; we resize to ``size`` and map
    the label map back to the source resolution in :func:`postprocess`.
    """
    orig_h, orig_w = rgb.shape[:2]
    resized = np.asarray(
        Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)
    ).astype(np.float32)
    normed = (resized / 255.0 - _MEAN) / _STD
    blob = np.transpose(normed, (2, 0, 1))[np.newaxis, ...]  # HWC -> NCHW + batch
    return blob.astype(np.float32), (orig_w, orig_h)


def postprocess(logits: np.ndarray, orig_wh: tuple[int, int]) -> np.ndarray:
    """Class-logits ``(1, C, h, w)`` -> per-pixel label map at original size.

    Argmax over the class axis, then nearest-neighbor upsample to ``orig_wh`` so
    each source pixel gets a crisp class id (no blended edges).
    """
    preds = np.squeeze(logits, axis=0)  # (C, h, w)
    labels = preds.argmax(axis=0).astype(np.uint8)  # (h, w)
    orig_w, orig_h = orig_wh
    upsized = Image.fromarray(labels, "L").resize(
        (orig_w, orig_h), Image.Resampling.NEAREST
    )
    return np.asarray(upsized)


def to_category_map(
    label_map: np.ndarray, mapping: dict[int, str] = CITYSCAPES_TO_CATEGORY
) -> np.ndarray:
    """Map a class-id label map to an object array of category strings / ``None``."""
    out = np.empty(label_map.shape, dtype=object)
    for class_id, category in mapping.items():
        out[label_map == class_id] = category
    return out


# --------------------------------------------------------------------------- #
# Model acquisition (mirrors detect.resolve_model, seg-specific defaults)
# --------------------------------------------------------------------------- #
def configured_sources() -> tuple[ModelSource, ...]:
    """The segmentation sources to use, honouring ``$SGTRAFFIC_SEG_MODEL_URL``.

    As in :func:`detect.configured_sources`, a user-set URL replaces the built-in
    list *and* disables checksum verification — those are someone else's weights
    by request, so our digest has no authority over them.
    """
    override = os.environ.get("SGTRAFFIC_SEG_MODEL_URL")
    if override:
        return (ModelSource.unverified(override),)
    return DEFAULT_SEG_MODEL_SOURCES


def resolve_model(
    explicit: str | None = None,
    *,
    allow_download: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Locate the segmentation ONNX model, downloading + caching on first use.

    Integrity failures surface as a :class:`SegmentationError` too — never a
    ``DetectionError`` — but keep the "this is not the network" wording, since a
    mismatched digest and an unreachable mirror need opposite reactions.
    """
    for candidate in (explicit, os.environ.get("SGTRAFFIC_SEG_MODEL")):
        if candidate:
            path = Path(candidate).expanduser()
            if not path.is_file():
                raise SegmentationError(f"Segmentation model not found: {path}")
            return path

    sources = configured_sources()
    cached = _CACHE_DIR / _SEG_FILENAME
    if cached.is_file():
        try:
            return verify_cached(cached, trusted_digests(sources))
        except ModelIntegrityError as exc:
            raise SegmentationError(f"Segmentation model: {exc}") from exc
    if not allow_download:
        raise SegmentationError(
            f"No model at {cached} and downloads are disabled. "
            "Pass --seg-model /path/to.onnx or set $SGTRAFFIC_SEG_MODEL."
        )
    try:
        return download_first(sources, cached, on_progress=on_progress)
    except Exception as exc:  # download_first raises DetectionError on failure
        raise SegmentationError(
            f"Could not obtain a verified segmentation model: {exc}\n"
            "Set $SGTRAFFIC_SEG_MODEL_URL to a reachable Cityscapes .onnx, or pass "
            "--seg-model /path/to.onnx."
        ) from exc


# --------------------------------------------------------------------------- #
# Runtime wrapper
# --------------------------------------------------------------------------- #
class Segmenter:
    """A loaded segmentation ONNX session. Construct once, call :meth:`segment`."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort  # lazy: keeps pure helpers import-cheap
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise SegmentationError("onnxruntime is not installed") from exc
        try:
            self._session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise SegmentationError(
                f"Failed to load model {model_path}: {exc}"
            ) from exc
        self._input_name = self._session.get_inputs()[0].name

    def segment(self, image_bytes: bytes) -> np.ndarray:
        """Return the per-pixel class-id label map (uint8) at original resolution.

        Callers nearest-resize this to the small render buffer and then apply
        :func:`to_category_map` — resizing an integer map is lossless and cheap,
        whereas resizing an object/category array is not.
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as src:
                rgb = np.asarray(src.convert("RGB"))
        except Exception as exc:
            raise SegmentationError(f"Could not decode image: {exc}") from exc

        orig_h, orig_w = rgb.shape[:2]
        blob, _ = preprocess(rgb)
        try:
            outputs = self._session.run(None, {self._input_name: blob})
        except Exception as exc:
            raise SegmentationError(f"Inference failed: {exc}") from exc

        return postprocess(np.asarray(outputs[0]), (orig_w, orig_h))
