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
```

`python -m sgtraffic` works too.

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

| Module          | Responsibility                                            |
|-----------------|-----------------------------------------------------------|
| `api.py`        | Data.gov.sg client, `Camera` dataclass, image download    |
| `locations.py`  | camera-id → friendly name / expressway mapping            |
| `render.py`     | image → colored half-block art + CRT framing (the engine) |
| `cli.py`        | Typer commands, interactive menu, auto-refresh loop       |
