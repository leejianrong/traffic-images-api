"""Object detection via a small YOLOv8n model run through ONNX Runtime.

We deliberately depend on ``onnxruntime`` (MIT) rather than the ``ultralytics``
package (AGPL-3.0): only the *weights* are YOLOv8, fetched at runtime into a
cache dir and never committed here. That keeps this repository free of copyleft
code while still using the best-documented tiny detector.

Pipeline (all standard YOLOv8 export conventions):

    bytes -> RGB -> letterbox to 640 -> NCHW /255 -> session.run
          -> (1, 84, 8400) -> per-box class scores -> threshold
          -> undo letterbox -> class-aware NMS -> [Detection]

The math (:func:`letterbox`, :func:`postprocess`, :func:`nms`) is written as
free functions so it can be unit-tested against synthetic tensors with no model
present. Only :class:`Detector` needs the actual runtime, imported lazily.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# COCO classes we care about, mapped to our sprite categories.
# --------------------------------------------------------------------------- #
# COCO id -> (our category, is_vehicle)
_COCO_MAP: dict[int, tuple[str, bool]] = {
    0: ("person", False),
    1: ("motorbike", True),  # bicycle -> reuse the two-wheeler sprite
    2: ("car", True),
    3: ("motorbike", True),  # motorcycle
    5: ("bus_truck", True),  # bus
    7: ("bus_truck", True),  # truck
}

# Model download + cache configuration (all overridable by env).
_DEFAULT_MODEL_URL = os.environ.get(
    "SGTRAFFIC_MODEL_URL",
    "https://huggingface.co/Xenova/yolov8n/resolve/main/onnx/model.onnx",
)
_CACHE_DIR = Path(
    os.environ.get("SGTRAFFIC_CACHE", Path.home() / ".cache" / "sgtraffic")
)
_MODEL_FILENAME = "yolov8n.onnx"

_INPUT_SIZE = 640
_PAD_COLOR = 114


class DetectionError(RuntimeError):
    """Raised when the model can't be obtained or inference fails."""


@dataclass(slots=True, frozen=True)
class Detection:
    """One detected object in *original image* pixel coordinates."""

    category: str  # car / bus_truck / motorbike / person
    is_vehicle: bool
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1


# --------------------------------------------------------------------------- #
# Pure image / tensor math (no onnxruntime needed -> unit-testable)
# --------------------------------------------------------------------------- #
def letterbox(
    img: np.ndarray, new_size: int = _INPUT_SIZE, color: int = _PAD_COLOR
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize ``img`` (H, W, 3) into a square ``new_size`` canvas, keeping aspect.

    Returns the padded image, the scale ``gain`` applied, and the ``(pad_w,
    pad_h)`` padding added on each side -- exactly the values :func:`postprocess`
    needs to map boxes back to the original frame.
    """
    h, w = img.shape[:2]
    gain = min(new_size / h, new_size / w)
    resized_w, resized_h = round(w * gain), round(h * gain)

    resized = np.asarray(
        Image.fromarray(img).resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    )

    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    pad_w = (new_size - resized_w) / 2
    pad_h = (new_size - resized_h) / 2
    top, left = int(round(pad_h - 0.1)), int(round(pad_w - 0.1))
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas, gain, (pad_w, pad_h)


def to_blob(canvas: np.ndarray) -> np.ndarray:
    """HWC uint8 -> NCHW float32 in [0, 1], the layout YOLOv8 expects."""
    blob = canvas.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
    return blob[np.newaxis, ...]  # add batch dim


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Greedy non-max suppression. ``boxes`` is (N, 4) xyxy. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x1 - x0).clip(min=0) * (y1 - y0).clip(min=0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(x0[i], x0[rest])
        yy0 = np.maximum(y0[i], y0[rest])
        xx1 = np.minimum(x1[i], x1[rest])
        yy1 = np.minimum(y1[i], y1[rest])
        inter = (xx1 - xx0).clip(min=0) * (yy1 - yy0).clip(min=0)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thres]
    return keep


def postprocess(
    output: np.ndarray,
    gain: float,
    pad: tuple[float, float],
    orig_wh: tuple[int, int],
    *,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> list[Detection]:
    """Decode a raw YOLOv8 output tensor into :class:`Detection` objects.

    ``output`` is the model's ``(1, 84, 8400)`` tensor (84 = 4 box coords + 80
    COCO class scores; YOLOv8 has no separate objectness). Boxes come out as
    ``cx, cy, w, h`` in letterboxed 640 space; we threshold on the best class
    score, undo the letterbox, keep only classes in :data:`_COCO_MAP`, then run
    class-aware NMS.
    """
    preds = np.squeeze(output, axis=0)  # (84, 8400)
    if preds.shape[0] < preds.shape[1]:
        preds = preds.transpose()  # -> (8400, 84)

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    # Keep only confident rows whose class is one we render.
    wanted = np.array([cid in _COCO_MAP for cid in class_ids])
    mask = (confidences >= conf_thres) & wanted
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # xywh (center) -> xyxy in letterbox space.
    cx, cy, bw, bh = boxes_xywh.T
    x0 = cx - bw / 2
    y0 = cy - bh / 2
    x1 = cx + bw / 2
    y1 = cy + bh / 2

    # Undo letterbox: subtract padding, divide by gain, clip to frame.
    pad_w, pad_h = pad
    orig_w, orig_h = orig_wh
    x0 = ((x0 - pad_w) / gain).clip(0, orig_w)
    x1 = ((x1 - pad_w) / gain).clip(0, orig_w)
    y0 = ((y0 - pad_h) / gain).clip(0, orig_h)
    y1 = ((y1 - pad_h) / gain).clip(0, orig_h)
    boxes = np.stack([x0, y0, x1, y1], axis=1)

    # Class-aware NMS: offset boxes per class so different classes never suppress
    # each other, then a single global NMS pass.
    offset = class_ids[:, np.newaxis] * (max(orig_w, orig_h) + 1)
    keep = nms(boxes + offset, confidences, iou_thres)

    detections: list[Detection] = []
    for i in keep:
        category, is_vehicle = _COCO_MAP[int(class_ids[i])]
        detections.append(
            Detection(
                category=category,
                is_vehicle=is_vehicle,
                score=float(confidences[i]),
                box=(float(x0[i]), float(y0[i]), float(x1[i]), float(y1[i])),
            )
        )
    return detections


# --------------------------------------------------------------------------- #
# Model acquisition
# --------------------------------------------------------------------------- #
def resolve_model(
    explicit: str | None = None,
    *,
    allow_download: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Locate the ONNX model, downloading + caching it on first use.

    Resolution order: ``explicit`` arg -> ``$SGTRAFFIC_MODEL`` -> cached file ->
    download from :data:`_DEFAULT_MODEL_URL`. Raises :class:`DetectionError`
    with actionable guidance if it can't be found or fetched.
    """
    for candidate in (explicit, os.environ.get("SGTRAFFIC_MODEL")):
        if candidate:
            path = Path(candidate).expanduser()
            if not path.is_file():
                raise DetectionError(f"Model file not found: {path}")
            return path

    cached = _CACHE_DIR / _MODEL_FILENAME
    if cached.is_file():
        return cached
    if not allow_download:
        raise DetectionError(
            f"No model at {cached} and downloads are disabled. "
            "Pass --model /path/to.onnx or set $SGTRAFFIC_MODEL."
        )
    return download_model(_DEFAULT_MODEL_URL, cached, on_progress=on_progress)


def download_model(
    url: str, dest: Path, *, on_progress: Callable[[int, int], None] | None = None
) -> Path:
    """Stream ``url`` to ``dest`` (atomically), reporting progress if asked."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 16):
                    fh.write(chunk)
                    done += len(chunk)
                    if on_progress:
                        on_progress(done, total)
        tmp.replace(dest)
    except httpx.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise DetectionError(
            f"Could not download model from {url}: {exc}\n"
            "Set $SGTRAFFIC_MODEL_URL to a reachable yolov8n.onnx, or pass "
            "--model /path/to.onnx (export one with ultralytics: "
            "YOLO('yolov8n.pt').export(format='onnx'))."
        ) from exc
    return dest


# --------------------------------------------------------------------------- #
# The runtime wrapper
# --------------------------------------------------------------------------- #
class Detector:
    """A loaded YOLOv8 ONNX session. Construct once, call :meth:`detect` per frame."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> None:
        try:
            import onnxruntime as ort  # lazy: keeps pure helpers import-cheap
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise DetectionError("onnxruntime is not installed") from exc

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        try:
            self._session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # onnxruntime raises bare Exceptions
            raise DetectionError(f"Failed to load model {model_path}: {exc}") from exc
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, image_bytes: bytes) -> list[Detection]:
        """Run detection on raw image bytes, returning original-coord boxes."""
        import io

        try:
            with Image.open(io.BytesIO(image_bytes)) as src:
                rgb = np.asarray(src.convert("RGB"))
        except Exception as exc:
            raise DetectionError(f"Could not decode image: {exc}") from exc

        orig_h, orig_w = rgb.shape[:2]
        canvas, gain, pad = letterbox(rgb)
        blob = to_blob(canvas)
        try:
            outputs = self._session.run(None, {self._input_name: blob})
        except Exception as exc:
            raise DetectionError(f"Inference failed: {exc}") from exc

        return postprocess(
            np.asarray(outputs[0]),
            gain,
            pad,
            (orig_w, orig_h),
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
        )
