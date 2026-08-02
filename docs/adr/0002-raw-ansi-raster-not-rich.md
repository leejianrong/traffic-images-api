# 0002 — The raster is printed raw, not through Rich

**Status:** Accepted · **Date:** 2026-08-02

## Context

The app already uses Rich for menus, tables, progress spinners and screen
clears. Sending the pixel art through the same `Console` looks like the tidy
thing to do — one output path, panels for free.

It is not tidy, it is expensive. `lines_from_image` emits one SGR sequence per
*run* of identical cells and reuses it for every repeated cell after that: a
sky or a stretch of road costs one escape sequence, not one per column. Each
line then ends with exactly one reset. Rich cannot preserve that. It models
styled text as spans and re-emits a colour sequence plus a reset around every
cell it renders, so a flat 40-column row goes from 2 escape sequences to 41 —
roughly 20× the bytes, per row, per frame, at whatever refresh interval the
user picked. On a slow terminal or an SSH link that is the difference between a
frame that snaps and one that visibly paints.

## Decision

The raster bypasses Rich entirely. `cli._draw` builds the frame with
`render.frame` and hands it to the builtin `print(framed, flush=True)`. Rich
stays for menus, tables, progress and `console.clear()` — everything that is
text rather than pixels.

## Consequences

We do our own layout maths, because nothing downstream can measure the width of
an ANSI-laden string for us. `RenderResult` carries `cols` explicitly (it is
the pixel width we resized to, so it is exact), `frame` uses that number for
its borders, `_center` truncates plain text to fit, and `_center_ansi` takes
the caller's *visible* width as an argument — which is why `cli._gauge` returns
`(text, visible_cols)` as a pair instead of just a string.

It also means nothing else clips our output, so the sizing math has to be
right: `terminal_spec` must never return more columns than the window can hold,
because a wrapped line shears every row beneath it. `tests/test_render.py`
pins that invariant separately.

The rule is easy to break by accident — routing one `print` through `console`
looks harmless — so it is enforced by tests rather than by convention. In
`tests/test_render.py`:

- `test_solid_image_emits_one_escape_run_per_line` — a flat row must cost
  exactly 1 SGR and 2 escapes total, the number Rich would inflate to 41.
- `test_every_cell_differing_emits_one_sgr_per_cell` — the worst case is 40
  runs plus a reset, so the run-length logic is measured, not assumed.
- `test_run_length_coalescing_beats_worst_case_by_an_order_of_magnitude` —
  asserts the ≥20× gap between best and worst case directly.
- `test_each_line_ends_with_exactly_one_reset` — one trailing reset per line
  and none in the middle, so colour cannot bleed into the border or survive a
  scroll.

If someone pipes the raster through Rich, those escape counts move and the
suite goes red for the right reason.
