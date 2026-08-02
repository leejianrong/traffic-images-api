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
uv run pytest                    # offline test suite (synthetic fixtures)
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
- **Detection needs a model file the sandboxed web env can't fetch** — the model host (GitHub/HF) and the live traffic API are blocked by the package-registries-only network policy. Tests are fully offline (synthetic tensors + a fake session); real `--detect` runs happen on a machine with open network. Keep detection logic testable without a model.
