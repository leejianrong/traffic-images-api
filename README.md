# 📺 sgtraffic

A retro **8-bit terminal viewer** for live Singapore traffic-camera feeds,
pulled from the [Data.gov.sg](https://data.gov.sg) real-time Traffic Images API
and rendered as vibrant TrueColor Unicode half-block art — like a little arcade
CCTV monitor in your shell.

## Quick start

```bash
uv sync                             # create venv + install deps
uv run sgtraffic                    # interactive menu (default)
uv run sgtraffic list               # list live cameras by expressway
uv run sgtraffic view 2702          # single snapshot
uv run sgtraffic view 6710 -r 20    # live monitor, refresh every 20s
uv run sgtraffic view 2705 --detect # detect vehicles → pixel-art sprites + score
```

`python -m sgtraffic` works too.

## Detection & congestion score (`--detect`)

Add `--detect` (`-d`) to `view` to run object detection over each snapshot,
redraw the detected vehicles as **hand-made pixel-art sprites** stamped over the
photo, and print a live **congestion score**:

```
CONGESTION [█████░░░░░░░░░░░]  30% MODERATE  car:2 bus:1 bike:1 ped:1
```

- **Detector**: a small YOLOv8n model run via [ONNX Runtime](https://onnxruntime.ai)
  (COCO classes → car / bus·truck / motorbike·bicycle / person). We depend only
  on `onnxruntime` (MIT), not the AGPL `ultralytics` package — the weights are
  fetched at runtime, never committed.
- **Model file**: on first `--detect` run the model auto-downloads to
  `~/.cache/sgtraffic/`. Override with `--model /path/to.onnx` or the
  `SGTRAFFIC_MODEL` / `SGTRAFFIC_MODEL_URL` env vars. To make one yourself:
  `YOLO("yolov8n.pt").export(format="onnx")` (needs `ultralytics` in a throwaway env).
- **Sprites**: defined entirely in code (`sprites.py`) as character grids +
  a palette — no binary assets. Cars come in a few colors so a jam isn't clones.
- **Score**: blends vehicle *count* with box *occupancy* (a density proxy that
  fairly handles near vs. far cameras) into a 0–100 value bucketed
  QUIET → MODERATE → BUSY → JAM. Tune the detector with `--conf`.

```bash
uv run sgtraffic view 2705 --detect --conf 0.3        # single analyzed frame
uv run sgtraffic view 2705 --detect -r 20             # live analyzed monitor
uv run sgtraffic view 2705 --detect --model yolo.onnx # bring your own model
```

## How the image engine works

Each terminal cell renders **two vertical pixels** using the lower-half-block
glyph `▄`: the cell *background* is the top pixel, the *foreground* (the glyph)
is the bottom pixel. This doubles vertical resolution and keeps pixels square.

- **TrueColor** `38;2;R;G;B` / `48;2;R;G;B` escapes → 16.7M colors.
- **Posterization** (`--bits`, default 4) collapses each channel to a few
  levels for chunky retro color banding; a saturation/contrast nudge pops it.
- **Run-length optimization**: identical adjacent cells reuse the previous SGR
  instead of re-emitting it — big savings on skies and road surfaces.
- **Resize-safe**: target dimensions are recomputed from the live terminal
  size every frame, so the art always fits the window and never wraps.

## Layout

| Module           | Responsibility                                            |
|------------------|-----------------------------------------------------------|
| `api.py`         | Data.gov.sg client, `Camera` dataclass, image download    |
| `locations.py`   | camera-id → friendly name / expressway mapping            |
| `render.py`      | image → colored half-block art + CRT framing (the engine) |
| `detect.py`      | YOLOv8n via ONNX Runtime → vehicle/person detections      |
| `sprites.py`     | code-defined pixel-art sprites, stamped over the photo    |
| `congestion.py`  | count + occupancy → 0–100 congestion score               |
| `cli.py`         | Typer commands, interactive menu, auto-refresh loop       |

## Tests

```bash
uv run pytest        # offline: no model or network needed (synthetic fixtures)
```
