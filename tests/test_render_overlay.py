"""Rendering split points and the frame gauge row."""

from __future__ import annotations

from conftest import make_png
from PIL import Image

from sgtraffic import render


def test_prepare_frame_returns_buffer_and_original_size():
    img, orig_wh = render.prepare_frame(make_png(320, 240), max_cols=60)
    assert orig_wh == (320, 240)
    assert isinstance(img, Image.Image)
    w, h = img.size
    assert w <= 60
    assert h % 2 == 0  # two source rows per character cell


def test_lines_from_image_dimensions():
    img = Image.new("RGB", (10, 6), (100, 100, 100))
    result = render.lines_from_image(img)
    assert result.cols == 10
    assert result.rows == 3  # 6 px tall / 2 = 3 rows
    assert len(result.lines) == 3


def test_render_result_matches_split_pipeline():
    png = make_png(80, 60)
    combined = render.render_result(png, max_cols=50)
    img, _ = render.prepare_frame(png, max_cols=50)
    split = render.lines_from_image(img)
    assert combined.lines == split.lines


def test_frame_includes_gauge_row_with_padding():
    result = render.RenderResult(lines=["x"], cols=30, rows=1)
    gauge_text = "\x1b[38;2;255;70;70mJAM BAR\x1b[0m"
    framed = render.frame(result, title="t", subtitle="s", gauge=(gauge_text, 7))
    assert gauge_text in framed
    # The gauge occupies its own row between the art and the subtitle.
    assert framed.count("╟") >= 2  # separators above gauge and subtitle


def test_center_ansi_pads_by_visible_width_not_len():
    text = "\x1b[31mAB\x1b[0m"  # visible width 2, len far larger
    centered = render._center_ansi(text, visible=2, width=6)
    # 4 spaces of padding split around the 2 visible chars.
    assert centered.count(" ") == 4
    assert centered.startswith("  ")
