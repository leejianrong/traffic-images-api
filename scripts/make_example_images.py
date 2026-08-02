"""Regenerate the example images embedded in the README.

    uv run python scripts/make_example_images.py [camera_id]

Writes into ``docs/images/``. Needs the network (a live camera frame) and the
real ONNX models, so it is a manual tool, not part of any test layer.

Two things worth knowing before you re-run it:

* All four frames come from a *single* fetch. Re-downloading per frame would
  give you four different moments and the comparison would be a lie.
* Scales are chosen so the output lands near GitHub's ~840px content column.
  Pixel art scaled *down* by a browser turns to mush, so the upscale factor
  matters more than it looks.
"""

from __future__ import annotations

import io
import os
import sys

import httpx
import numpy as np
from PIL import Image

from sgtraffic import api, cli, detect, render, segment, sprites, tiles

OUT = "docs/images"
FRAME_SCALE = 7  # 120 cols * 7 = 840px
SHEET_SCALE = 5
PAD = 6

# terminal_spec reads the live terminal size; force the widest supported render
# (max_cols=120) so the sprites are big enough to actually see.
os.environ.setdefault("COLUMNS", "124")
os.environ.setdefault("LINES", "76")


def frames(camera_id: str) -> None:
    det = detect.Detector(detect.resolve_model(None))
    seg = segment.Segmenter(segment.resolve_model(None))

    with httpx.Client(timeout=60) as client:
        cam = api.get_camera(camera_id, client)
        raw = api.download_image(cam.image_url, client)
    print(f"{cam.camera_id} {cam.name} @ {cam.timestamp}")

    with Image.open(io.BytesIO(raw)) as src:
        w = 120 * FRAME_SCALE
        h = round(src.height * w / src.width)
        src.convert("RGB").resize((w, h), Image.Resampling.LANCZOS).save(
            f"{OUT}/example-original.jpg", quality=80, optimize=True
        )

    for name, use_seg, use_det in (
        ("plain", False, False),
        ("detect", False, True),
        ("demake", True, True),
    ):
        small, orig_wh = render.prepare_frame(raw, bits=4)
        if use_seg:
            cli._paint_background(raw, small, seg)
        cong = cli._stamp_detections(raw, small, det, orig_wh) if use_det else None
        upscaled = small.resize(
            (small.width * FRAME_SCALE, small.height * FRAME_SCALE),
            Image.Resampling.NEAREST,
        )
        upscaled.save(f"{OUT}/example-{name}.png", optimize=True)
        print(
            f"  {name:7} {upscaled.width}x{upscaled.height}", cong.label if cong else ""
        )


def sprite_sheet() -> None:
    items = (
        [sprites._CARS[v] for v in sprites.CAR_VARIANTS]
        + [sprites._BUSES[v] for v in sprites.BUS_VARIANTS]
        + [sprites._BIKE_SPRITE, sprites._PERSON_SPRITE]
    )
    imgs = [Image.fromarray(s.rgba, "RGBA") for s in items]
    cw = max(i.width for i in imgs) + PAD
    ch = max(i.height for i in imgs) + PAD
    sheet = Image.new("RGBA", (cw * len(imgs), ch), (24, 24, 32, 255))
    for k, im in enumerate(imgs):
        sheet.alpha_composite(
            im, (k * cw + (cw - im.width) // 2, (ch - im.height) // 2)
        )
    sheet.resize(
        (sheet.width * SHEET_SCALE, sheet.height * SHEET_SCALE),
        Image.Resampling.NEAREST,
    ).convert("RGB").save(f"{OUT}/sprites-sheet.png", optimize=True)
    print(f"  sprites {len(items)}")


def tile_sheet(reps: int = 3) -> None:
    entries = list(tiles.TILES.items())
    th, tw = entries[0][1].shape[:2]
    width = tw * reps * len(entries) + PAD * (len(entries) - 1)
    sheet = Image.new("RGB", (width, th * reps), (24, 24, 32))
    x = 0
    for _name, arr in entries:
        tile = Image.fromarray(arr.astype(np.uint8), "RGB")
        for i in range(reps):
            for j in range(reps):
                sheet.paste(tile, (x + i * tw, j * th))
        x += tw * reps + PAD
    sheet.resize(
        (sheet.width * SHEET_SCALE, sheet.height * SHEET_SCALE),
        Image.Resampling.NEAREST,
    ).save(f"{OUT}/tiles-sheet.png", optimize=True)
    print(f"  tiles {[n for n, _ in entries]}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    frames(sys.argv[1] if len(sys.argv) > 1 else "2704")
    sprite_sheet()
    tile_sheet()
