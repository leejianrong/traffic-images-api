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
```

Use `uv` for everything — never call `pip` or a bare `python`. Python 3.12+.

## Architecture

Data flows one way: `api` fetches → `cli` orchestrates → `render` converts.

| Module | Responsibility |
|---|---|
| `src/sgtraffic/api.py` | Data.gov.sg client, frozen `Camera` dataclass, image download. All net errors → `TrafficAPIError`. |
| `src/sgtraffic/locations.py` | Pure lookup: `camera_id` → friendly name / expressway. Unknown ids fall back gracefully. |
| `src/sgtraffic/render.py` | **The engine.** Image bytes → colored `▄` half-blocks + CRT frame. |
| `src/sgtraffic/cli.py` | Typer commands, interactive menu, auto-refresh loop. |

## Quality (run before every commit)

```bash
uvx ruff check src && uvx ruff format src   # lint + format
uvx ty check src                            # type-check (strict typing everywhere)
```

## Gotchas

- **TrueColor terminal required** — rendering emits `48;2;R;G;B` escapes. Check `$COLORTERM=truecolor`.
- **The raster is printed raw, NOT through Rich** — Rich re-emits an SGR + reset per cell, killing the run-length optimization. Rich is only for menus/tables/status; the pixel art goes straight to `print()`. Don't route it back through Rich.
- **The API returns only currently-live cameras** (often ~8, sometimes ~90) — code must handle any count. `locations.py` carries the full name map regardless.
- **Dimensions are recomputed every frame** from the live terminal size, so resizing mid-stream is safe. Never hard-code width/height.
