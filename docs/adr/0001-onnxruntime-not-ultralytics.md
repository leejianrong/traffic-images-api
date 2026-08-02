# 0001 — ONNX Runtime for inference, not the `ultralytics` package

**Status:** Accepted · **Date:** 2026-08-02

## Context

`--detect` needs a small object detector. The obvious one is YOLOv8n, and the
obvious way to run it is `pip install ultralytics`, four lines of Python, done.

That package is AGPL-3.0. Pulling it in as a runtime dependency drags this
project — a toy terminal viewer someone might vendor into something else —
into the copyleft obligations of a network-copyleft licence. ONNX Runtime is
MIT, reads the same model once it has been exported to `.onnx`, and is the
only heavyweight thing we need at inference time.

The weights are a separate question from the code. We ship none: the `.onnx`
file is downloaded on first `--detect` run into `~/.cache/sgtraffic/` and is
never committed to this repository.

## Decision

Depend on `onnxruntime` (MIT). Never depend on `ultralytics`. Fetch YOLOv8n
weights at runtime from a list of mirrors, cache them outside the repo, and let
users point at their own file with `--model` / `SGTRAFFIC_MODEL`.

Everything around the session call — the pre- and post-processing that
`ultralytics` would have done for us — is reimplemented as free functions in
`src/sgtraffic/detect.py`: `letterbox`, `to_blob`, `nms`, `postprocess`.
`onnxruntime` itself is imported lazily inside `Detector`, so the default
`view` / `list` / `menu` paths never pay for it.

## Consequences

We own YOLOv8's conventions by hand, and they are fiddly: the half-pixel
rounding in the letterbox pad, undoing gain and padding in the right order,
YOLOv8 having no objectness channel (84 = 4 box coords + 80 class scores), and
class-aware NMS implemented by offsetting boxes per class before one global
pass. Get any of these subtly wrong and boxes land a few pixels off, which
looks plausible and is wrong.

That is the reason those four functions are pure and so heavily tested — eight
of the thirteen tests in `tests/test_detect.py` exercise nothing but this math,
against synthetic tensors with no model file anywhere. It also makes the whole
detection path unit-testable offline, which is what lets the fast test layer
cover it at all (see [ADR 0003](0003-two-test-layers.md)).

The layout of the weights is now a load-bearing assumption rather than
something a library negotiates for us. A replacement model must be a stock COCO
YOLOv8n export: input `(1, 3, 640, 640)` float32, output `(1, 84, 8400)`.
Anything else decodes into garbage rather than failing loudly.

Producing weights yourself still needs `ultralytics`
(`YOLO("yolov8n.pt").export(format="onnx")`) — in a throwaway environment, as
documented in the README. It is a tool you run once, not a dependency of this
project.

Runtime download means runtime failure modes: mirrors rot (both original
defaults started returning `401`), so `detect.DEFAULT_MODEL_URLS` is a list
tried in order, and the only thing that notices a dead mirror is the
integration layer.
