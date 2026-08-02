"""Retro 8-bit terminal viewer for Singapore traffic cameras.

Commands
--------
    sgtraffic list                 Show all live cameras, grouped by expressway
    sgtraffic view <camera_id>     Render one snapshot (add --refresh to stream)
    sgtraffic menu                 Interactive picker (default when no command)
"""

from __future__ import annotations

import itertools
import time
from typing import Annotated

import httpx
import numpy as np
import typer
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from . import api, congestion, render, segment, sprites, tiles
from .api import Camera, TrafficAPIError
from .congestion import Congestion
from .detect import Detection, DetectionError, Detector, resolve_model
from .segment import SegmentationError, Segmenter

app = typer.Typer(
    name="sgtraffic",
    help="📺 Retro 8-bit viewer for live SG traffic cameras (Data.gov.sg).",
    no_args_is_help=False,
    add_completion=False,
)
console = Console()

REFRESH_DEFAULT = 20  # seconds, per the "live monitor" spec.
_HEADERS = {"User-Agent": "sgtraffic-cli/0.1"}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _client() -> httpx.Client:
    return httpx.Client(headers=_HEADERS, timeout=httpx.Timeout(15.0, connect=10.0))


def _fatal(msg: str) -> None:
    console.print(f"[bold red]✗[/] {msg}")
    raise typer.Exit(code=1)


def _cameras(client: httpx.Client) -> list[Camera]:
    try:
        cams = api.fetch_cameras(client)
    except TrafficAPIError as exc:
        _fatal(str(exc))
    if not cams:
        _fatal("No cameras are currently live. Try again shortly.")
    return cams


# --------------------------------------------------------------------------- #
# `list`
# --------------------------------------------------------------------------- #
@app.command("list")
def list_cameras() -> None:
    """List every live camera, grouped by expressway."""
    with _client() as client:
        cams = _cameras(client)

    table = Table(
        title="[bold #39ff14]🚦 LIVE SG TRAFFIC CAMERAS 🚦[/]",
        header_style="bold #00e5ff",
        border_style="#00e5ff",
    )
    table.add_column("ID", style="bold yellow", justify="right")
    table.add_column("Expressway", style="magenta")
    table.add_column("Location", style="white")

    for expressway, group in itertools.groupby(cams, key=lambda c: c.expressway):
        for i, cam in enumerate(group):
            table.add_row(cam.camera_id, expressway if i == 0 else "", cam.name)
    console.print(table)
    console.print(
        f"[dim]{len(cams)} cameras live. "
        f"View one with[/] [bold]sgtraffic view <ID>[/][dim] "
        f"or run[/] [bold]sgtraffic menu[/][dim].[/]"
    )


# --------------------------------------------------------------------------- #
# `view`
# --------------------------------------------------------------------------- #
# Congestion label -> truecolor foreground (green→yellow→orange→red).
_GAUGE_COLORS: dict[str, str] = {
    "QUIET": "\x1b[38;2;57;255;20m",
    "MODERATE": "\x1b[38;2;245;205;60m",
    "BUSY": "\x1b[38;2;240;150;40m",
    "JAM": "\x1b[38;2;255;70;70m",
}
# Sprite-category -> compact HUD label.
_HUD_LABELS: tuple[tuple[str, str], ...] = (
    ("car", "car"),
    ("bus_truck", "bus"),
    ("motorbike", "bike"),
    ("person", "ped"),
)


def _build_detector(model: str | None, conf: float) -> Detector:
    """Resolve (downloading if needed) the model and load a detector, or exit."""
    try:
        with Progress(transient=True, console=console) as progress:
            task = progress.add_task("[cyan]Fetching detection model…", total=None)

            def _on_progress(done: int, total: int) -> None:
                progress.update(task, completed=done, total=total or None)

            path = resolve_model(model, on_progress=_on_progress)
        return Detector(path, conf_thres=conf)
    except DetectionError as exc:
        _fatal(str(exc))
        raise  # unreachable; _fatal exits. keeps type-checkers happy.


def _build_segmenter(model: str | None) -> Segmenter:
    """Resolve (downloading if needed) the segmentation model and load it, or exit."""
    try:
        with Progress(transient=True, console=console) as progress:
            task = progress.add_task("[cyan]Fetching segmentation model…", total=None)

            def _on_progress(done: int, total: int) -> None:
                progress.update(task, completed=done, total=total or None)

            path = segment.resolve_model(model, on_progress=_on_progress)
        return Segmenter(path)
    except SegmentationError as exc:
        _fatal(str(exc))
        raise  # unreachable; _fatal exits.


def _gauge(cong: Congestion, width: int) -> tuple[str, int]:
    """Build the colored congestion bar + counts, and its exact visible width.

    Returns ``(ansi_string, visible_cols)`` so :func:`render.frame` can center it
    without trying to measure ANSI escapes.
    """
    bar_len = min(16, max(6, width - 40))
    filled = round(cong.score / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    counts = " ".join(
        f"{label}:{cong.counts.get(cat, 0)}" for cat, label in _HUD_LABELS
    )
    plain = f"CONGESTION [{bar}] {cong.score:3d}% {cong.label:<8} {counts}"
    if len(plain) > width:
        plain = plain[:width]
    color = _GAUGE_COLORS.get(cong.label, "")
    return f"{color}{plain}{render.RESET}", len(plain)


def _paint_background(
    image_bytes: bytes, small: Image.Image, segmenter: Segmenter
) -> None:
    """Segment the frame and repaint the small buffer's background as tiles."""
    small_w, small_h = small.size
    label_map = segmenter.segment(image_bytes)  # (orig_h, orig_w) uint8
    # Nearest-resize the integer label map to the buffer, then map to categories.
    small_labels = np.asarray(
        Image.fromarray(label_map, "L").resize(
            (small_w, small_h), Image.Resampling.NEAREST
        )
    )
    tiles.paint(small, segment.to_category_map(small_labels))


def _stamp_detections(
    image_bytes: bytes,
    small: Image.Image,
    detector: Detector,
    orig_wh: tuple[int, int],
) -> Congestion:
    """Detect vehicles, stamp sprites onto the buffer, and return the score."""
    detections = detector.detect(image_bytes)
    small_w, small_h = small.size
    orig_w, orig_h = orig_wh
    scale_x = small_w / orig_w if orig_w else 1.0
    scale_y = small_h / orig_h if orig_h else 1.0

    # Stamp smallest (usually farthest) first so nearer, larger vehicles layer
    # on top and aren't hidden behind distant ones.
    def _area(det: Detection) -> float:
        x0, y0, x1, y1 = det.box
        return (x1 - x0) * (y1 - y0)

    for det in sorted(detections, key=_area):
        x0, y0, x1, y1 = det.box
        box = (
            max(0, int(x0 * scale_x)),
            max(0, int(y0 * scale_y)),
            min(small_w, round(x1 * scale_x)),
            min(small_h, round(y1 * scale_y)),
        )
        seed = int(x0) + int(y0)
        sprites.stamp(small, sprites.for_category(det.category, seed=seed), box)

    return congestion.score(detections, (orig_w, orig_h))


def _analyze_frame(
    image_bytes: bytes,
    bits: int,
    *,
    detector: Detector | None = None,
    segmenter: Segmenter | None = None,
) -> tuple[render.RenderResult, Congestion | None]:
    """Prepare the buffer, optionally tile the background and stamp vehicles.

    Order matters: background tiles are painted first (so sprites sit on top),
    then vehicle sprites. Either stage is independent — ``--segment`` alone tiles
    the scene, ``--detect`` alone overlays vehicles, together they de-make it.
    """
    small, orig_wh = render.prepare_frame(image_bytes, bits=bits)
    if segmenter is not None:
        _paint_background(image_bytes, small, segmenter)
    cong = (
        _stamp_detections(image_bytes, small, detector, orig_wh)
        if detector is not None
        else None
    )
    return render.lines_from_image(small), cong


def _draw(
    cam: Camera,
    image_bytes: bytes,
    *,
    bits: int,
    refresh: int | None,
    detector: Detector | None = None,
    segmenter: Segmenter | None = None,
) -> None:
    """Clear screen and paint one framed frame (raw ANSI, printed directly)."""
    gauge = None
    if detector is not None or segmenter is not None:
        result, cong = _analyze_frame(
            image_bytes, bits, detector=detector, segmenter=segmenter
        )
        if cong is not None:
            gauge = _gauge(cong, result.cols)
    else:
        result = render.render_result(image_bytes, bits=bits)
    ts = cam.timestamp.replace("T", " ")[:19] or "unknown"
    hint = (
        f"cam {cam.camera_id} · {ts} SGT · refresh {refresh}s · Ctrl-C to quit"
        if refresh
        else f"cam {cam.camera_id} · {ts} SGT"
    )
    framed = render.frame(
        result, title=f"▓▒░  {cam.name}  ░▒▓", subtitle=hint, gauge=gauge
    )
    console.clear()
    # Bypass Rich for the raster: print raw ANSI so the run-length-optimized
    # truecolor bytes reach the terminal verbatim.
    print(framed, flush=True)


@app.command("view")
def view(
    camera_id: Annotated[str, typer.Argument(help="Camera ID, e.g. 2702 or 6710.")],
    refresh: Annotated[
        int | None,
        typer.Option(
            "--refresh", "-r", help="Auto-refresh every N seconds (live monitor)."
        ),
    ] = None,
    once: Annotated[
        bool, typer.Option("--once", help="Force a single frame (ignore --refresh).")
    ] = False,
    bits: Annotated[
        int,
        typer.Option(
            "--bits", min=1, max=8, help="Bits/channel: lower = chunkier retro."
        ),
    ] = 4,
    detect: Annotated[
        bool,
        typer.Option(
            "--detect",
            "-d",
            help="Detect vehicles, draw pixel-art sprites + congestion score.",
        ),
    ] = False,
    model: Annotated[
        str | None,
        typer.Option(
            "--model", help="Path to a YOLOv8 .onnx model (else auto-download+cache)."
        ),
    ] = None,
    conf: Annotated[
        float,
        typer.Option(
            "--conf", min=0.0, max=1.0, help="Detection confidence threshold."
        ),
    ] = 0.25,
    seg: Annotated[
        bool,
        typer.Option(
            "--segment",
            "-s",
            help="Segment the background and repaint it as pixel-art tiles.",
        ),
    ] = False,
    seg_model: Annotated[
        str | None,
        typer.Option(
            "--seg-model",
            help="Path to a segmentation .onnx model (else auto-download+cache).",
        ),
    ] = None,
) -> None:
    """Render a camera snapshot. Use --refresh N to stream it like a monitor.

    Combine --detect and --segment for the full 'de-make': tiled background
    regions with pixel-art vehicles stamped on top.
    """
    interval = None if once else refresh
    detector = _build_detector(model, conf) if detect else None
    segmenter = _build_segmenter(seg_model) if seg else None
    with _client() as client:
        try:
            if interval:
                _stream(camera_id, interval, bits, client, detector, segmenter)
            else:
                cam = api.get_camera(camera_id, client)
                img = api.download_image(cam.image_url, client)
                _draw(
                    cam,
                    img,
                    bits=bits,
                    refresh=None,
                    detector=detector,
                    segmenter=segmenter,
                )
        except TrafficAPIError as exc:
            _fatal(str(exc))


def _stream(
    camera_id: str,
    interval: int,
    bits: int,
    client: httpx.Client,
    detector: Detector | None = None,
    segmenter: Segmenter | None = None,
) -> None:
    """Re-fetch + re-render on an interval until interrupted (Ctrl-C)."""
    try:
        while True:
            # Re-fetch the camera each cycle so both the image URL and the
            # terminal size are re-evaluated -- resize the window mid-stream
            # and the next frame simply fits the new dimensions.
            cam = api.get_camera(camera_id, client)
            img = api.download_image(cam.image_url, client)
            _draw(
                cam,
                img,
                bits=bits,
                refresh=interval,
                detector=detector,
                segmenter=segmenter,
            )
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print(
            "\n[bold #39ff14]▓▒░[/] Stream ended. Drive safe. [bold #39ff14]░▒▓[/]"
        )


# --------------------------------------------------------------------------- #
# `menu` (interactive)
# --------------------------------------------------------------------------- #
@app.command("menu")
def menu() -> None:
    """Interactive picker: choose a camera, then stream it live."""
    with _client() as client:
        cams = _cameras(client)

        console.print(
            Panel.fit(
                "[bold #39ff14]S G   T R A F F I C   M O N I T O R[/]\n"
                "[dim]8-bit live camera feeds · Data.gov.sg[/]",
                border_style="#ff00ff",
            )
        )
        # Numbered menu.
        table = Table(header_style="bold #00e5ff", border_style="#00e5ff")
        table.add_column("#", style="bold yellow", justify="right")
        table.add_column("Expressway", style="magenta")
        table.add_column("Location", style="white")
        for idx, cam in enumerate(cams, start=1):
            table.add_row(str(idx), cam.expressway, cam.name)
        console.print(table)

        choice = typer.prompt("Select a camera number (or 'q' to quit)")
        if choice.strip().lower() in {"q", "quit", "exit"}:
            raise typer.Exit()
        try:
            selected = cams[int(choice) - 1]
        except (ValueError, IndexError):
            _fatal(f"Invalid selection: {choice!r}")

        live = typer.confirm(
            f"Stream '{selected.name}' live (refresh {REFRESH_DEFAULT}s)?", default=True
        )
        try:
            if live:
                _stream(selected.camera_id, REFRESH_DEFAULT, bits=4, client=client)
            else:
                img = api.download_image(selected.image_url, client)
                _draw(selected, img, bits=4, refresh=None)
        except TrafficAPIError as exc:
            _fatal(str(exc))


# --------------------------------------------------------------------------- #
# Default: no subcommand -> launch the interactive menu.
# --------------------------------------------------------------------------- #
@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        menu()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
