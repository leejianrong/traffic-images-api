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

import hashlib
import os
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

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

_CACHE_DIR = Path(
    os.environ.get("SGTRAFFIC_CACHE", Path.home() / ".cache" / "sgtraffic")
)
_MODEL_FILENAME = "yolov8n.onnx"

_INPUT_SIZE = 640
_PAD_COLOR = 114


class DetectionError(RuntimeError):
    """Raised when the model can't be obtained or inference fails."""


class ModelIntegrityError(DetectionError):
    """The bytes we got are not the bytes we pinned.

    A separate type from a plain :class:`DetectionError` because it demands a
    different response. "The host is down" means *try again later*; "the file
    does not hash to the value baked into this release" means *something served
    you a different model* — and that model is about to be handed to ONNX
    Runtime and executed. Never fall back to running it.
    """


@dataclass(slots=True, frozen=True)
class ModelSource:
    """One place to fetch a model from, pinned two ways.

    ``url`` must name an *immutable* revision (a Hugging Face
    ``/resolve/<commit-sha>/`` path, never ``/resolve/main/``) so the bytes on
    the far end can't be swapped under us, and ``sha256`` is the digest of those
    exact bytes so we notice if they are anyway.

    ``sha256`` is ``None`` for a source the *user* supplied (``--model``,
    ``$SGTRAFFIC_MODEL_URL``): they are deliberately pointing us at different
    weights, so checking them against our digest would only ever reject a
    perfectly valid choice. Unverified sources are opt-in by construction —
    nothing in :data:`DEFAULT_MODEL_SOURCES` may carry ``None``.
    """

    url: str
    sha256: str | None

    @classmethod
    def unverified(cls, url: str) -> ModelSource:
        """A user-supplied source, exempt from our checksum by design."""
        return cls(url=url, sha256=None)


# Model download + cache configuration (all overridable by env).
#
# A *list*, not a single URL, and deliberately so: the original hard-coded
# default (``Xenova/yolov8n``) went away and started answering 401, which killed
# ``--detect`` outright for everyone with a cold cache. Community model mirrors
# disappear; we try each in turn and only fail once every one of them is gone.
#
# Every entry below is the same stock COCO YOLOv8n export, and each was checked
# against the others before being listed: statically shaped input ``images``
# (1, 3, 640, 640) float32 and output (1, 84, 8400) -- exactly what
# :func:`postprocess` decodes -- and *identical detections* on the same live
# frame. Order is by how likely the host is to still be there next year:
# organisations first, personal accounts last.
#
# Two near-misses, recorded so nobody re-adds them: ``EdgeFirst/yolov8-det`` has
# the right shapes but emits normalized box coordinates, so every detection
# collapses into the top-left corner (shapes alone prove nothing -- run it); and
# ``mobilint/YOLOv8n``, from the most reputable account of the lot, declares
# dynamic batch/class axes instead of the static layout the pipeline and its
# integration test are written against.
DEFAULT_MODEL_SOURCES: tuple[ModelSource, ...] = (
    # Kalray — a listed semiconductor company; ~80 models on HF. Promoted over
    # the two personal accounts below, which is the whole point of this list.
    ModelSource(
        url="https://huggingface.co/Kalray/yolov8/resolve/9e0af089be9c2f172e4fd9b724805f8b6514854e/yolov8n.onnx",
        sha256="65158dad735be799c2466fa15e260c09558080bd530b42a8d0c3d1b419afd8b5",
    ),
    # Demoted: personal accounts, zero downloads, zero likes. Kept as a last
    # resort now that an organisation-hosted copy leads the list.
    ModelSource(
        url="https://huggingface.co/salim4n/yolov8n-detect-onnx/resolve/98100409491fa67f62a2e780a4efe485b86dfd0b/yolov8n-onnx-web/yolov8n.onnx",
        sha256="162ec5e9d2886fc411e4b81811c731e81a72c8f85571de1e0ab040833d92d4f2",
    ),
    # Byte-identical to the Kalray copy above (note the shared digest) — which
    # is also decent evidence that both are the genuine upstream export. Buys a
    # second chance at a live repo, not a second artifact.
    ModelSource(
        url="https://huggingface.co/inference4j/yolov8n/resolve/66295110f9dd507498735446f4f9f05da3007073/model.onnx",
        sha256="65158dad735be799c2466fa15e260c09558080bd530b42a8d0c3d1b419afd8b5",
    ),
)


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
    top, left = round(pad_h - 0.1), round(pad_w - 0.1)
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
# Integrity
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of ``path``. Never holds the whole model in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def trusted_digests(sources: Sequence[ModelSource]) -> frozenset[str]:
    """Every digest we would accept from ``sources`` (empty = verification off).

    Empty means the caller is on a user-supplied source and we have nothing to
    compare against — see :class:`ModelSource`.
    """
    return frozenset(s.sha256 for s in sources if s.sha256 is not None)


def verify_cached(path: Path, digests: Collection[str]) -> Path:
    """Re-check an already-cached model, deleting it if it no longer matches.

    Verifying only on download would mean the guarantee expires the moment the
    file lands: everything after the first run reads an unchecked blob off disk.
    Hashing ~13 MB costs single-digit milliseconds and happens once per process
    (``resolve_model`` is called at startup, not per frame), which is noise next
    to building the ONNX session — so we pay it every time.

    ``digests`` is the set of *all* pinned digests for this model, because the
    cache does not record which mirror it came from. Empty set = skip (the user
    supplied their own source).
    """
    if not digests:
        return path
    actual = sha256_file(path)
    if actual in digests:
        return path
    path.unlink(missing_ok=True)
    raise ModelIntegrityError(
        f"Cached model failed its integrity check and was deleted: {path}\n"
        f"  expected sha256: {' or '.join(sorted(digests))}\n"
        f"  actual sha256:   {actual}\n"
        "This is NOT a network problem. Either the file was corrupted or "
        "tampered with locally, or the pinned model was updated in a newer "
        "release. Re-run to download a fresh, verified copy."
    )


# --------------------------------------------------------------------------- #
# Model acquisition
# --------------------------------------------------------------------------- #
def configured_sources() -> tuple[ModelSource, ...]:
    """The detection sources to use, honouring ``$SGTRAFFIC_MODEL_URL``.

    Read at call time rather than at import so the override is testable and so
    a long-lived process picks up a changed environment. A user-set URL wins
    over the built-in list *and* turns checksum verification off: they asked for
    those particular weights, so failing them against our digest would be wrong.
    """
    override = os.environ.get("SGTRAFFIC_MODEL_URL")
    if override:
        return (ModelSource.unverified(override),)
    return DEFAULT_MODEL_SOURCES


def resolve_model(
    explicit: str | None = None,
    *,
    allow_download: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Locate the ONNX model, downloading + caching it on first use.

    Resolution order: ``explicit`` arg -> ``$SGTRAFFIC_MODEL`` -> cached file ->
    download from the first reachable of :data:`DEFAULT_MODEL_SOURCES`. Raises
    :class:`DetectionError` with actionable guidance if it can't be found or
    fetched, or :class:`ModelIntegrityError` if what we got isn't what we pinned.

    A path handed in by the user (``explicit`` or ``$SGTRAFFIC_MODEL``) is
    returned as-is: it is their file, not ours to checksum.
    """
    for candidate in (explicit, os.environ.get("SGTRAFFIC_MODEL")):
        if candidate:
            path = Path(candidate).expanduser()
            if not path.is_file():
                raise DetectionError(f"Model file not found: {path}")
            return path

    sources = configured_sources()
    cached = _CACHE_DIR / _MODEL_FILENAME
    if cached.is_file():
        return verify_cached(cached, trusted_digests(sources))
    if not allow_download:
        raise DetectionError(
            f"No model at {cached} and downloads are disabled. "
            "Pass --model /path/to.onnx or set $SGTRAFFIC_MODEL."
        )
    try:
        return download_first(sources, cached, on_progress=on_progress)
    except DetectionError as exc:
        raise DetectionError(
            f"{exc}\nSet $SGTRAFFIC_MODEL_URL to a reachable yolov8n.onnx, or pass "
            "--model /path/to.onnx (export one with ultralytics: "
            "YOLO('yolov8n.pt').export(format='onnx'))."
        ) from exc


def download_first(
    sources: Sequence[ModelSource],
    dest: Path,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Try each source in order, returning the first that downloads *and verifies*.

    Mirrors are not forever: a repo goes private or is deleted and every cold
    cache breaks at once. A mirror that answers with the wrong bytes falls
    through here exactly like a dead one — we would rather try the next pin than
    execute an unknown model — but its failure line says so, because "the host
    404s" and "the host served something else" want very different reactions
    from whoever reads the error. Only when *all* sources fail do we raise. The
    remediation advice is left to the caller: this helper serves both the
    detection and the segmentation model, which need different hints.
    """
    failures: list[str] = []
    for source in sources:
        try:
            return download_model(
                source.url, dest, sha256=source.sha256, on_progress=on_progress
            )
        except DetectionError as exc:
            failures.append(f"  • {exc}")
    raise DetectionError(
        "Could not obtain a verified model from any known source:\n"
        + "\n".join(failures)
    )


def download_model(
    url: str,
    dest: Path,
    *,
    sha256: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Stream ``url`` to ``dest`` (atomically), reporting progress if asked.

    The digest is accumulated chunk-by-chunk as the bytes go to disk, so a
    ~13 MB model is never buffered in memory — let alone twice — and the check
    costs one extra pass over data we are already touching. Verification happens
    *before* the atomic rename, so a file that fails it never appears at
    ``dest`` for ONNX Runtime to pick up; only the ``.part`` is deleted.
    ``sha256=None`` skips the check (a user-supplied source).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        digest = hashlib.sha256()
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 16):
                    fh.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if on_progress:
                        on_progress(done, total)
    except httpx.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        # One line, no remediation advice: the callers (download_first and
        # segment.resolve_model) aggregate these and append their own guidance.
        reason = str(exc).splitlines()[0] or type(exc).__name__
        raise DetectionError(f"Could not download model from {url}: {reason}") from exc

    actual = digest.hexdigest()
    if sha256 is not None and actual != sha256:
        tmp.unlink(missing_ok=True)
        raise ModelIntegrityError(
            f"Integrity check FAILED for {url}: expected sha256 {sha256}, "
            f"got {actual} — the download was discarded, not cached. "
            "The host answered, so this is not a network fault: that revision "
            "is serving different bytes than the ones this release pinned."
        )
    tmp.replace(dest)
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
