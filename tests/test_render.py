"""The core engine: terminal sizing math, the retro prepare step, and the
half-block conversion -- including the run-length ANSI optimization that the
raw-print path exists to protect."""

from __future__ import annotations

import io
import re
import shutil

from conftest import make_png
from PIL import Image

from sgtraffic import render

# Matches one emitted cell prefix: background = top pixel, foreground = bottom.
SGR = re.compile(r"\x1b\[48;2;(\d+);(\d+);(\d+);38;2;(\d+);(\d+);(\d+)m")

# Any SGR sequence -- used to measure a line's *visible* width.
ANSI = re.compile(r"\x1b\[[0-9;]*m")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def _sgr_count(text: str) -> int:
    """How many color sequences were emitted."""
    return len(SGR.findall(text))


def _escape_count(text: str) -> int:
    """Every escape sequence, colors *and* resets."""
    return text.count("\x1b[")


def _visible_width(line: str) -> int:
    """On-screen width of an ANSI-laden line (every glyph here is 1 cell wide)."""
    return len(ANSI.sub("", line))


def _fake_term(monkeypatch, cols: int, rows: int) -> None:
    """Force an exact terminal size.

    Blunter than COLUMNS/LINES on purpose: ``shutil.get_terminal_size`` treats a
    non-positive env value as "unset" and silently substitutes the real window,
    which would quietly neuter the degenerate cases below.
    """
    monkeypatch.setattr(
        shutil, "get_terminal_size", lambda fallback=(80, 24): (cols, rows)
    )


def _assert_fits_terminal(
    spec: render.RenderSpec, term_cols: int, term_rows: int, ctx: object
) -> None:
    """The promise in ``terminal_spec``'s docstring, as an assertion.

    Art plus chrome must fit the window. Below that the budget is allowed to
    collapse -- but only to 1, never to 0 (which would resize to an empty image)
    and never *upward* past what the window can hold (which would wrap and shear
    every row of the frame).
    """
    assert spec.cols >= 1, ctx
    assert spec.rows >= 1, ctx
    assert spec.cols <= max(1, term_cols - render._H_CHROME), ctx
    assert spec.rows <= max(1, term_rows - render._V_CHROME), ctx


def _checker(width: int, height: int) -> Image.Image:
    """An image where adjacent cells never repeat -- the worst case for runs."""
    img = Image.new("RGB", (width, height))
    px = img.load()
    assert px is not None
    for y in range(height):
        for x in range(width):
            px[x, y] = WHITE if x % 2 else BLACK
    return img


def _gradient(width: int, height: int) -> Image.Image:
    """A smooth RGB ramp: many distinct colors, useful for palette assertions."""
    img = Image.new("RGB", (width, height))
    px = img.load()
    assert px is not None
    for y in range(height):
        for x in range(width):
            px[x, y] = (x * 255 // max(1, width - 1), y * 37 % 256, (x + y) % 256)
    return img


def _pixels(img: Image.Image) -> set[bytes]:
    """The distinct RGB triples in ``img``."""
    data = img.tobytes()
    return {data[i : i + 3] for i in range(0, len(data), 3)}


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# terminal_spec: sizing math
# --------------------------------------------------------------------------- #


def test_terminal_spec_subtracts_chrome_from_terminal_size(monkeypatch):
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setenv("LINES", "40")
    spec = render.terminal_spec(16 / 9)
    assert spec.cols == 100 - render._H_CHROME
    # Rows must fit inside the vertical budget too, chrome excluded.
    assert spec.rows <= 40 - render._V_CHROME


def test_terminal_spec_clamps_to_max_cols(monkeypatch):
    monkeypatch.setenv("COLUMNS", "500")
    monkeypatch.setenv("LINES", "200")
    assert render.terminal_spec(16 / 9, max_cols=64).cols == 64
    # Default cap still applies on an absurdly wide window.
    assert render.terminal_spec(16 / 9).cols == 120


def test_terminal_spec_preserves_source_aspect_ratio(monkeypatch):
    monkeypatch.setenv("COLUMNS", "124")
    monkeypatch.setenv("LINES", "200")  # height is never the binding constraint
    for aspect in (16 / 9, 4 / 3, 1.0, 2.5):
        spec = render.terminal_spec(aspect)
        # A cell is 1 px wide and 2 px tall, so on-screen aspect is cols/(2*rows).
        rendered = spec.cols / (spec.rows * 2)
        assert abs(rendered - aspect) / aspect < 0.05


def test_terminal_spec_clamps_by_height_for_tall_sources(monkeypatch):
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setenv("LINES", "20")
    spec = render.terminal_spec(0.5)  # twice as tall as it is wide
    assert spec.rows == 20 - render._V_CHROME  # pinned to the row budget
    assert spec.cols == 12  # back-solved: rows * 2 * aspect
    assert spec.cols < 100 - render._H_CHROME  # width gave way, not the height


def test_terminal_spec_never_overflows_a_hostile_window(monkeypatch):
    # A zero/negative dimension would resize to an empty image and crash the
    # renderer; an *over*-sized one wraps, and one wrapped line shears every row
    # under it. Every hostile window/aspect pair must satisfy both bounds.
    for cols, rows in (
        (1, 1),
        (0, 0),
        (6, 3),
        (12, 10),  # the window that used to yield 10 cols of art in 12
        (14, 9),  # exactly _H_CHROME + the old 10-col floor
        (40, 8),  # shorter than _V_CHROME
        (10000, 10000),
    ):
        _fake_term(monkeypatch, cols, rows)
        for aspect in (0.01, 0.1, 1.0, 16 / 9, 100.0):
            spec = render.terminal_spec(aspect)
            _assert_fits_terminal(spec, cols, rows, (cols, rows, aspect))


def test_terminal_spec_shrinks_below_the_old_floor_in_a_narrow_window(monkeypatch):
    # 12 columns cannot hold 10 columns of art plus 4 of chrome. Shrinking wins:
    # a 8-column picture is ugly, a wrapped 14-column one is corrupt.
    monkeypatch.setenv("COLUMNS", "12")
    monkeypatch.setenv("LINES", "10")
    spec = render.terminal_spec(16 / 9)
    assert spec.cols == 12 - render._H_CHROME == 8
    assert spec.cols + render._H_CHROME <= 12
    assert spec.rows + render._V_CHROME <= 10


def test_terminal_spec_shrinks_below_the_old_floor_in_a_short_window(monkeypatch):
    # Same story vertically: 9 rows leaves 1 row of art after chrome, not 5.
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setenv("LINES", "9")
    spec = render.terminal_spec(16 / 9)
    assert spec.rows == 1
    assert spec.rows + render._V_CHROME <= 9
    # Width had to give way to keep the aspect ratio at one row of art.
    assert spec.cols == round(1 * 2 * (16 / 9))


def test_terminal_spec_row_budget_is_the_frame_chrome_it_promises(monkeypatch):
    # _V_CHROME must cover the fattest frame() draws (border, title, separators,
    # gauge and subtitle), or the row budget is a lie.
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "24")
    result = render.render_result(make_png(320, 240))
    framed = render.frame(
        result, title="t", subtitle="s", gauge=("gauge", len("gauge"))
    )
    assert len(framed.splitlines()) == result.rows + render._V_CHROME


def _draw_like_the_cli(result: render.RenderResult) -> str:
    """Frame a result the way ``cli._draw`` does, gauge and subtitle included."""
    gauge_text = "CONGESTION 100% JAM"[: result.cols]  # the CLI truncates too
    return render.frame(
        result,
        title="a very long camera name that cannot possibly fit",
        subtitle="cam 2702 · 2026-01-01 00:00:00 SGT · refresh 20s",
        gauge=(gauge_text, len(gauge_text)),
    )


def test_whole_render_path_fits_a_tiny_terminal(monkeypatch):
    # End to end, not just the sizing math: art + border + title + gauge +
    # subtitle must all stay inside the window at hostile sizes.
    for cols, rows in ((12, 10), (14, 9), (20, 12), (5, 40), (9, 11)):
        _fake_term(monkeypatch, cols, rows)
        result = render.render_result(make_png(320, 240))
        lines = _draw_like_the_cli(result).splitlines()
        assert len(lines) <= rows, (cols, rows, len(lines))
        for line in lines:
            assert _visible_width(line) <= cols, (cols, rows, repr(line))


def test_whole_render_path_survives_an_absurdly_small_terminal(monkeypatch):
    # Below _V_CHROME + 1 rows the *frame* itself no longer fits -- nothing can
    # fix that -- but the picture must still be drawn, still fit horizontally,
    # and never raise (a 0-px resize used to be the failure mode here).
    for cols, rows in ((6, 6), (3, 2), (1, 1), (0, 0)):
        _fake_term(monkeypatch, cols, rows)
        result = render.render_result(make_png(320, 240))
        assert result.cols >= 1 and result.rows >= 1, (cols, rows)
        assert result.cols <= max(1, cols - render._H_CHROME), (cols, rows)
        for line in _draw_like_the_cli(result).splitlines():
            assert _visible_width(line) <= max(cols, result.cols + render._H_CHROME)


def test_terminal_spec_uses_fallback_size_when_terminal_unknown(monkeypatch):
    monkeypatch.delenv("COLUMNS", raising=False)
    monkeypatch.delenv("LINES", raising=False)
    seen: list[tuple[int, int]] = []

    def fake_size(fallback: tuple[int, int]) -> tuple[int, int]:
        seen.append(fallback)
        return fallback

    monkeypatch.setattr(shutil, "get_terminal_size", fake_size)
    spec = render.terminal_spec(16 / 9, fallback=(60, 30))
    assert seen == [(60, 30)]  # the caller's fallback is what gets offered
    assert spec.cols == 60 - render._H_CHROME


# --------------------------------------------------------------------------- #
# _prepare / prepare_frame: resize + retro treatment
# --------------------------------------------------------------------------- #


def test_prepare_frame_resizes_to_the_terminal_spec(monkeypatch):
    monkeypatch.setenv("COLUMNS", "84")
    monkeypatch.setenv("LINES", "40")
    spec = render.terminal_spec(320 / 240, max_cols=64)
    img, _ = render.prepare_frame(make_png(320, 240), max_cols=64)
    assert img.size == (spec.cols, spec.rows * 2)
    assert img.mode == "RGB"


def test_prepare_frame_buffer_height_is_always_even(monkeypatch):
    # Half-blocks consume two pixel rows per character row; an odd height would
    # silently drop the bottom row in lines_from_image.
    monkeypatch.setenv("COLUMNS", "63")
    monkeypatch.setenv("LINES", "17")
    for w, h in ((641, 361), (1, 1000), (1000, 1), (3, 3), (1920, 1080)):
        img, _ = render.prepare_frame(make_png(w, h))
        bw, bh = img.size
        assert bh % 2 == 0, (w, h, img.size)
        assert bh >= 2 and bw >= 1, (w, h, img.size)


def test_prepare_frame_original_size_is_the_pre_resize_size(monkeypatch):
    monkeypatch.setenv("COLUMNS", "60")
    monkeypatch.setenv("LINES", "30")
    img, orig_wh = render.prepare_frame(make_png(321, 137), max_cols=40)
    # Detection maps boxes back through this, so it must be the *source* size.
    assert orig_wh == (321, 137)
    assert img.size != orig_wh


def test_prepare_posterizes_channels_to_bit_depth():
    spec = render.RenderSpec(cols=32, rows=4)
    out = render._prepare(_gradient(64, 16), spec, bits=2, punch=False)
    # 2 bits/channel keeps only the top two bits -> multiples of 64.
    assert {v % 64 for v in out.tobytes()} == {0}
    assert len(_pixels(out)) > 1  # still a picture, not flattened to one color


def test_prepare_leaves_pixels_untouched_at_full_depth_without_punch():
    spec = render.RenderSpec(cols=8, rows=2)
    src = Image.new("RGB", (16, 8), (40, 90, 200))
    out = render._prepare(src, spec, bits=8, punch=False)
    assert out.size == (8, 4)
    assert _pixels(out) == {bytes((40, 90, 200))}  # no posterize, no punch


# --------------------------------------------------------------------------- #
# lines_from_image: half-block encoding + the run-length optimization
# --------------------------------------------------------------------------- #


def test_half_block_paints_top_pixel_as_background_and_bottom_as_foreground():
    img = Image.new("RGB", (1, 2))
    px = img.load()
    assert px is not None
    px[0, 0] = (255, 0, 0)  # top
    px[0, 1] = (0, 0, 255)  # bottom
    (line,) = render.lines_from_image(img).lines
    expected = (
        "\x1b[48;2;255;0;0;38;2;0;0;255m" + render.LOWER_HALF_BLOCK + render.RESET
    )
    assert line == expected
    bg_r, bg_g, bg_b, fg_r, fg_g, fg_b = SGR.findall(line)[0]
    assert (bg_r, bg_g, bg_b) == ("255", "0", "0")  # 48;2 = background = top
    assert (fg_r, fg_g, fg_b) == ("0", "0", "255")  # 38;2 = foreground = bottom


def test_solid_image_emits_one_escape_run_per_line():
    # THE guard: a flat image must cost one SGR per line, not one per cell.
    # Routing this raster through Rich (which re-emits SGR+reset per cell) would
    # turn these 2 escapes per line into 41.
    result = render.lines_from_image(Image.new("RGB", (40, 8), (12, 34, 56)))
    assert result.rows == 4
    for line in result.lines:
        assert _sgr_count(line) == 1  # one run covers the whole row
        assert _escape_count(line) == 2  # that run + the end-of-line reset
        assert line.count(render.LOWER_HALF_BLOCK) == 40


def test_every_cell_differing_emits_one_sgr_per_cell():
    result = render.lines_from_image(_checker(40, 8))
    for line in result.lines:
        assert _sgr_count(line) == 40  # nothing can be coalesced
        assert _escape_count(line) == 41  # 40 runs + reset


def test_run_length_coalescing_beats_worst_case_by_an_order_of_magnitude():
    solid = render.lines_from_image(Image.new("RGB", (40, 8), (12, 34, 56)))
    worst = render.lines_from_image(_checker(40, 8))
    solid_escapes = sum(_escape_count(ln) for ln in solid.lines)
    worst_escapes = sum(_escape_count(ln) for ln in worst.lines)
    assert solid_escapes == 2 * solid.rows  # one SGR + one reset per line
    assert worst_escapes == 41 * worst.rows
    assert worst_escapes >= 20 * solid_escapes


def test_run_length_key_includes_both_half_pixels():
    # Top row uniform, bottom row alternating: keying the run on the top pixel
    # alone would coalesce these cells and paint the wrong bottom halves.
    img = Image.new("RGB", (16, 2), (10, 10, 10))
    px = img.load()
    assert px is not None
    for x in range(16):
        px[x, 1] = WHITE if x % 2 else BLACK
    (line,) = render.lines_from_image(img).lines
    assert _sgr_count(line) == 16


def test_runs_restart_after_a_color_change():
    img = Image.new("RGB", (30, 2), (5, 5, 5))
    px = img.load()
    assert px is not None
    for y in range(2):
        for x in range(10, 20):
            px[x, y] = (200, 100, 0)
    (line,) = render.lines_from_image(img).lines
    assert _sgr_count(line) == 3  # left run, middle run, right run
    assert line.count(render.LOWER_HALF_BLOCK) == 30


def test_each_line_ends_with_exactly_one_reset():
    # Reset discipline: color must not bleed past the end of a row (into the
    # frame border, or across a resize/scroll).
    result = render.lines_from_image(_gradient(24, 10))
    assert len(result.lines) == 5
    for line in result.lines:
        assert line.endswith(render.RESET)
        assert line.count(render.RESET) == 1
        assert render.RESET not in line[: -len(render.RESET)]
        # Every glyph is preceded, somewhere, by a color run.
        assert _sgr_count(line) >= 1


def test_lines_from_image_ignores_an_unpaired_final_row():
    result = render.lines_from_image(Image.new("RGB", (4, 5), (1, 2, 3)))
    assert result.rows == 2  # 5 px -> 2 full cells, the stray row is dropped
    assert render.lines_from_image(Image.new("RGB", (4, 1))).lines == []


# --------------------------------------------------------------------------- #
# render_result: end-to-end composition
# --------------------------------------------------------------------------- #


def test_render_result_line_glyph_count_equals_cols(monkeypatch):
    monkeypatch.setenv("COLUMNS", "70")
    monkeypatch.setenv("LINES", "30")
    result = render.render_result(_png(_gradient(200, 120)), max_cols=48)
    # frame() trusts result.cols instead of measuring ANSI-laden strings, so the
    # visible width of every line must equal it exactly.
    assert result.rows == len(result.lines)
    for line in result.lines:
        assert line.count(render.LOWER_HALF_BLOCK) == result.cols


def test_render_result_honours_max_cols(monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.setenv("LINES", "90")
    assert render.render_result(make_png(320, 240), max_cols=32).cols == 32


def test_render_result_bits_limit_the_palette(monkeypatch):
    monkeypatch.setenv("COLUMNS", "60")
    monkeypatch.setenv("LINES", "30")
    png = _png(_gradient(200, 120))
    chunky = render.render_result(png, max_cols=40, bits=1)
    smooth = render.render_result(png, max_cols=40, bits=8)

    def channels(res: render.RenderResult) -> set[str]:
        return {v for ln in res.lines for m in SGR.findall(ln) for v in m}

    assert channels(chunky) <= {"0", "128"}  # 1 bit/channel
    assert len(channels(smooth)) > 2
