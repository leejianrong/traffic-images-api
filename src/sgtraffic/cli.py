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
from typing import Annotated, Optional

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import api, render
from .api import Camera, TrafficAPIError

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
def _draw(cam: Camera, image_bytes: bytes, *, bits: int, refresh: int | None) -> None:
    """Clear screen and paint one framed frame (raw ANSI, printed directly)."""
    result = render.render_result(image_bytes, bits=bits)
    ts = cam.timestamp.replace("T", " ")[:19] or "unknown"
    hint = (
        f"cam {cam.camera_id} · {ts} SGT · refresh {refresh}s · Ctrl-C to quit"
        if refresh
        else f"cam {cam.camera_id} · {ts} SGT"
    )
    framed = render.frame(result, title=f"▓▒░  {cam.name}  ░▒▓", subtitle=hint)
    console.clear()
    # Bypass Rich for the raster: print raw ANSI so the run-length-optimized
    # truecolor bytes reach the terminal verbatim.
    print(framed, flush=True)


@app.command("view")
def view(
    camera_id: Annotated[str, typer.Argument(help="Camera ID, e.g. 2702 or 6710.")],
    refresh: Annotated[
        Optional[int],
        typer.Option("--refresh", "-r", help="Auto-refresh every N seconds (live monitor)."),
    ] = None,
    once: Annotated[
        bool, typer.Option("--once", help="Force a single frame (ignore --refresh).")
    ] = False,
    bits: Annotated[
        int, typer.Option("--bits", min=1, max=8, help="Bits/channel: lower = chunkier retro.")
    ] = 4,
) -> None:
    """Render a camera snapshot. Use --refresh N to stream it like a monitor."""
    interval = None if once else refresh
    with _client() as client:
        try:
            if interval:
                _stream(camera_id, interval, bits, client)
            else:
                cam = api.get_camera(camera_id, client)
                img = api.download_image(cam.image_url, client)
                _draw(cam, img, bits=bits, refresh=None)
        except TrafficAPIError as exc:
            _fatal(str(exc))


def _stream(camera_id: str, interval: int, bits: int, client: httpx.Client) -> None:
    """Re-fetch + re-render on an interval until interrupted (Ctrl-C)."""
    try:
        while True:
            # Re-fetch the camera each cycle so both the image URL and the
            # terminal size are re-evaluated -- resize the window mid-stream
            # and the next frame simply fits the new dimensions.
            cam = api.get_camera(camera_id, client)
            img = api.download_image(cam.image_url, client)
            _draw(cam, img, bits=bits, refresh=interval)
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[bold #39ff14]▓▒░[/] Stream ended. Drive safe. [bold #39ff14]░▒▓[/]")


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
