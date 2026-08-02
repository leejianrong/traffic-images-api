# CLAUDE.md

Retro 8-bit terminal viewer for live Singapore traffic cameras (Data.gov.sg).
Fetches camera snapshots and renders them as TrueColor Unicode half-block art.

## Commands

```bash
uv sync                          # install deps into .venv
uv run sgtraffic                 # interactive menu (default command)
uv run sgtraffic list            # list live cameras, grouped by expressway
uv run sgtraffic view 2702       # single snapshot
uv run sgtraffic view 6710 -r 20 # live monitor, refresh every 20s
uv run sgtraffic view 2701 --once --bits 3   # force one frame, chunkier retro
uv run sgtraffic view 2705 --detect          # sprites + congestion score
uv run sgtraffic view 2705 -d -s             # full de-make (tiles + sprites)
uv run pytest                    # fast layer: offline test suite (synthetic fixtures)
uv run pytest -m integration     # heavy layer: real ONNX models + live API
```

Use `uv` for everything — never call `pip` or a bare `python`. Python 3.12+.

## Architecture

Data flows one way: `api` fetches → `cli` orchestrates → `render` converts.

| Module | Responsibility |
|---|---|
| `src/sgtraffic/api.py` | Data.gov.sg client, frozen `Camera` dataclass, image download. All net errors → `TrafficAPIError`. |
| `src/sgtraffic/locations.py` | Pure lookup: `camera_id` → friendly name / expressway. Unknown ids fall back gracefully. |
| `src/sgtraffic/render.py` | **The engine.** Image bytes → colored `▄` half-blocks + CRT frame. |
| `src/sgtraffic/detect.py` | Optional CV: YOLOv8n via ONNX Runtime → `Detection`s. Pure math (letterbox/NMS/postprocess) split from the runtime. All errors → `DetectionError`. |
| `src/sgtraffic/sprites.py` | Code-defined pixel-art sprites (char grid + palette, no binary assets), stamped over the prepared buffer. |
| `src/sgtraffic/congestion.py` | Pure: `Detection`s → 0–100 congestion score (count + occupancy). |
| `src/sgtraffic/segment.py` | Optional CV: SegFormer (Cityscapes) via ONNX Runtime → per-pixel background class map. Pure math (preprocess/postprocess) split from the runtime. All errors → `SegmentationError`. Reuses `detect.download_model`. |
| `src/sgtraffic/tiles.py` | Code-defined tileable background textures (char grid + palette, no binary assets), painted over segmented regions. |
| `src/sgtraffic/cli.py` | Typer commands, interactive menu, auto-refresh loop. |

### The `--detect` pipeline (opt-in, off by default)
`view --detect` inserts a step between prepare and half-block conversion:
`render.prepare_frame` → `detect.Detector.detect` (on the **original** bytes, for
accuracy) → `sprites.stamp` onto the small buffer → `render.lines_from_image` +
`congestion.score`. Detection is heavy CV (`onnxruntime` + `numpy`) but stays
**optional**: the default `view`/`list`/`menu` paths never import a model, and
`onnxruntime` is imported lazily inside `Detector` so the pure helpers and tests
load instantly. We use `onnxruntime` (MIT), **not** the AGPL `ultralytics`
package — only the weights are YOLOv8, fetched at runtime and never committed.

### The `--segment` pipeline (opt-in, off by default)
`view --segment` inserts a *background* step: `render.prepare_frame` →
`segment.Segmenter.segment` (per-pixel Cityscapes class map on the original
bytes) → nearest-resize the label map to the small buffer → `segment.to_category_map`
→ `tiles.paint` recolors road/sidewalk/building/greenery/sky regions → then the
`--detect` sprites (if enabled) stamp on top → `render.lines_from_image`. Combine
`-d -s` for the full de-make. Same optional/lazy/offline-testable discipline as
detection; the segmentation model auto-downloads and is overridable via
`--seg-model` / `SGTRAFFIC_SEG_MODEL[_URL]`.

## Quality (run before every commit)

```bash
uvx ruff@0.16.1 check src tests            # lint (src *and* tests)
uvx ruff@0.16.1 format --check src tests   # format (drop --check to rewrite)
uvx ty@0.0.65 check src                    # type-check (strict typing everywhere)
uv run pytest                              # offline tests (no model/network)
```

**Two test layers, split by cost.** `uv run pytest` runs only the fast layer:
`tests/*.py`, synthetic fixtures, no model and no network, ~90 tests in under a
second. That is what the pre-push hook and the required CI jobs run, and it must
stay that way.

`tests/integration/` is the heavy layer — real ONNX weights (~28 MB, downloaded
and cached on first run) and the live Data.gov.sg API. It is marked
`integration` and **deselected by default**; opt in explicitly:

```bash
uv run pytest -m integration       # real models, real network
uv run pytest -m integration -rs   # ...and show why anything skipped
```

It covers what synthetic fixtures structurally cannot: that the model URLs still
resolve, that ONNX Runtime loads the file, that the real tensor shapes match
what `postprocess` assumes, and that a live frame survives the whole
`bytes → detections → sprites → lines` / `bytes → labels → tiles` path. Missing
infrastructure **skips**, never fails, so it is still runnable offline. In CI it
is an advisory `integration` job (weekly + `workflow_dispatch`, `continue-on-error`)
that never gates a PR.

**Always pin the tool versions.** A bare `uvx ruff` floats to the latest
release, which enables rules CI has never run — the local gate then fails on
findings CI ignores (or, worse, passes on ones it won't). The pins above,
`.githooks/pre-push`, and `.github/workflows/ci.yml` are kept in lockstep; when
you bump one, bump all three in the same commit and fix the new findings.

**Gates.** A pre-push hook (`.githooks/pre-push`) runs the cheap checks above
before every push, mirroring the CI jobs in `.github/workflows/ci.yml`
(lint + type-check + offline tests, all in parallel, frozen installs). Enable it
once per clone — then a push rarely lands red:

```bash
git config core.hooksPath .githooks   # bypass a single push with --no-verify
```

## Gotchas

- **TrueColor terminal required** — rendering emits `48;2;R;G;B` escapes. Check `$COLORTERM=truecolor`.
- **The raster is printed raw, NOT through Rich** — Rich re-emits an SGR + reset per cell, killing the run-length optimization. Rich is only for menus/tables/status; the pixel art goes straight to `print()`. Don't route it back through Rich.
- **The API returns only currently-live cameras** (often ~8, sometimes ~90) — code must handle any count. `locations.py` carries the full name map regardless.
- **Dimensions are recomputed every frame** from the live terminal size, so resizing mid-stream is safe. Never hard-code width/height.
- **Detection needs a model file the sandboxed web env can't fetch** — the model host (GitHub/HF) and the live traffic API are blocked by the package-registries-only network policy. The fast tests are fully offline (synthetic tensors + a fake session); real `--detect` runs and `pytest -m integration` happen on a machine with open network. Keep detection logic testable without a model.
- **Model URLs rot.** Both original defaults (`Xenova/yolov8n`, `onnx-community/segformer-b0-...`) were deleted from HF and started returning `401`, silently killing `--detect` and `--segment` on every cold cache. Each module now carries a *list* of mirrors (`detect.DEFAULT_MODEL_URLS`, `segment.DEFAULT_SEG_MODEL_URLS`) tried in order by `detect.download_first`. Never collapse these back to a single URL, and keep `tests/integration/test_model_sources.py` — it is the only thing that notices when a mirror dies.
- **Replacement weights must match the assumed tensor layout.** Detection expects a stock COCO YOLOv8n export: input `(1, 3, 640, 640)` float32, output `(1, 84, 8400)`. Segmentation expects a 19-class Cityscapes SegFormer whose channel order follows the `trainId` convention baked into `CITYSCAPES_TO_CATEGORY` — get that ordering wrong and the road gets painted with the sky tile, which *looks* fine and is completely wrong.
