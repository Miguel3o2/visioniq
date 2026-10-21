"""Textual operator console.

The dashboard mirrors the intent of the original mock-up - live grid, camera
rail, detection log, model panel - but every number and box on screen comes
from the running pipeline.  Rendering is true-colour half-blocks, so it stays
inside a terminal; there is no HTML or browser involved.
"""

from __future__ import annotations

import time
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import DataTable, Footer, Static

from visioniq.config import AppConfig
from visioniq.core.engine import Engine
from visioniq.core.types import CameraState, CameraView, Event, Severity
from visioniq.tui.render import (
    banner_text,
    calibration_text,
    camera_header,
    event_summary,
    frame_to_text,
    metrics_text,
    rail_text,
    score_bar,
)

_ANOMALY_CYCLE = ["running", "crowd", "loitering", "abandoned", "tailgate", "vehicle", "after_hours"]


class FrameView(Widget):
    """A single live feed rendered as coloured half-blocks."""

    can_focus = False

    def __init__(self, camera_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.camera_id = camera_id

    def render(self) -> Text:
        snapshot = getattr(self.app, "snapshot", None)
        width, height = self.size.width, self.size.height
        if snapshot is None:
            return Text("waiting for pipeline...", style="dim #5a7090")
        camera = snapshot.camera(self.camera_id)
        if camera is None:
            return Text("")
        if camera.calibrating:
            return calibration_text(camera, width, height)
        return frame_to_text(
            camera.frame,
            camera.tracks,
            width,
            height,
            camera.tint,
            alert=camera.state is CameraState.ALERT,
            offline=camera.state is CameraState.OFFLINE,
        )


class FeedCell(Vertical):
    """Header line plus the feed."""

    def __init__(self, camera: CameraView, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.camera_id = camera.id
        self._header: Static | None = None
        self._header_text: Text | None = None
        self._view: FrameView | None = None

    def compose(self) -> ComposeResult:
        self._header = Static(classes="cell-head")
        self._view = FrameView(self.camera_id, classes="cell-view")
        yield self._header
        yield self._view
    def update_camera(self, camera: CameraView) -> None:
        if self._header is None:
            return
        text = camera_header(camera, self.size.width)
        if self._header_text is None or str(text) != str(self._header_text):
            self._header.update(text)
            self._header_text = text
        if self._view is not None:
            self._view.refresh()


class VisionIQApp(App[None]):
    TITLE = "VisionIQ"
    SUB_TITLE = "Intelligent CCTV Monitoring System"
    CSS = """
    Screen { background: #0a0f1e; color: #e8edf5; }

    #topbar { height: 1; background: #0f1628; color: #e8edf5; }
    #banner { height: 1; background: #2a1418; }
    #banner.hidden { display: none; }

    #body { height: 1fr; }

    #rail { width: 31; height: 1fr; background: #131c30; border-right: solid #1e2d4a; padding: 0; }

    #center { width: 1fr; }
    #board { height: 1fr; layout: grid; grid-size: 2 2; grid-gutter: 0 1; background: #0a0f1e; }
    #board.g2x2 { grid-size: 2 2; }
    #board.g3x2 { grid-size: 3 2; }
    #board.g1x1 { grid-size: 1 1; }

    .feed-cell {
        height: 1fr; width: 1fr; layout: vertical;
        background: #050810; border: solid #1e2d4a;
    }
    .feed-cell.alert { border: solid #ff3b3b; }
    .cell-head { height: 1; background: #0f1628; padding: 0 1; }
    .cell-view { height: 1fr; }

    #loading {
        height: 1fr; background: #0a0f1e; color: #00d4ff;
        content-align: center middle; text-align: center;
    }

    #right { width: 46; height: 1fr; background: #131c30; border-left: solid #1e2d4a; }
    #right.hidden { display: none; }
    #right-head { height: 1; background: #0f1628; padding: 0 1; color: #00d4ff; }
    #model { height: auto; padding: 1 1 0 1; }
    #events-title { height: 1; padding: 0 1; color: #5a7090; background: #131c30; }
    #events { height: 1fr; background: #131c30; }
    #detail { height: 7; padding: 0 1; background: #0f1628; color: #9fb3cc; }

    #status { height: 1; background: #0f1628; padding: 0 1; color: #5a7090; }
    Footer { height: 1; background: #0f1628; }
    """

    BINDINGS = [
        Binding("1", "layout('g2x2')", "2x2", show=True),
        Binding("2", "layout('g3x2')", "3x2", show=True),
        Binding("3", "layout('g1x1')", "focus", show=True),
        Binding("e", "toggle_events", "events", show=True),
        Binding("[", "prev_camera", "prev cam", show=False),
        Binding("]", "next_camera", "next cam", show=False),
        Binding("a", "acknowledge", "ack", show=True),
        Binding("d", "inject", "inject", show=True),
        Binding("s", "snapshot", "snapshot", show=True),
        Binding("p", "pause", "pause", show=True),
        Binding("q", "quit", "quit", show=True),
    ]

    def __init__(self, config: AppConfig, engine: Engine, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.engine = engine
        self.snapshot = None
        self.selected = config.cameras[0].id
        self.layout_mode = "g2x2"
        self._cells: dict[str, FeedCell] = {}
        self._event_ids: list[int] = []
        self._event_signature: tuple = ()
        self._inject_index = 0
        self._last_slow = 0.0
        self._started = False

    # -- composition -------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Static("", id="topbar")
        yield Static("", id="banner", classes="hidden")
        with Horizontal(id="body"):
            yield Static("", id="rail")
            with Vertical(id="center"):
                yield Static("starting pipeline...", id="loading")
                yield Vertical(id="board", classes="g2x2")
            with Vertical(id="right", classes="hidden"):
                yield Static("Detection log", id="right-head")
                yield Static("", id="model")
                yield Static("EVENTS", id="events-title")
                yield DataTable(id="events", cursor_type="row", zebra_stripes=True)
                yield Static("", id="detail")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#board", Vertical).display = False
        self.engine.start()
        self.set_interval(1.0 / max(2.0, self.config.render_fps), self._tick)
        self.set_interval(1.0, self._slow_tick)
        table = self.query_one("#events", DataTable)
        table.add_columns("Time", "Sev", "Cam", "Event")

    def on_unmount(self) -> None:
        self.engine.stop()

    # -- periodic updates --------------------------------------------------
    async def _tick(self) -> None:
        snapshot = self.engine.snapshot()
        if snapshot is None:
            status = self.engine.status()
            if status.get("error"):
                self._set_loading(f"PIPELINE FAILED\n\n{status['error']}\n\npress q to quit", error=True)
            else:
                progress = float(status.get("progress") or 0.0)
                self._set_loading(
                    "VISIONIQ\n\n"
                    f"{status.get('stage', 'starting')}\n\n"
                    f"[{'█' * int(progress * 32):<32}] {progress * 100:3.0f}%\n\n"
                    "calibrating autoencoders on normal behaviour",
                )
            return

        self.snapshot = snapshot
        if not self._cells:
            await self._build_board(snapshot)
        if not self._started:
            self._started = True
            self.query_one("#loading", Static).display = False
            self.query_one("#board", Vertical).display = True

        for camera in snapshot.cameras:
            cell = self._cells.get(camera.id)
            if cell is not None:
                cell.update_camera(camera)
                cell.set_class(camera.state is CameraState.ALERT, "alert")

    def _slow_tick(self) -> None:
        snapshot = self.snapshot
        if snapshot is None:
            return
        rail = self.query_one("#rail", Static)
        rail.update(rail_text(snapshot.cameras, self.selected, max(20, rail.size.width)))

        topbar = self.query_one("#topbar", Static)
        topbar.update(self._topbar_text(snapshot))

        banner = self.query_one("#banner", Static)
        banner_text_value = banner_text(snapshot)
        if str(banner_text_value):
            banner.update(banner_text_value)
            banner.remove_class("hidden")
        else:
            banner.add_class("hidden")

        self.query_one("#status", Static).update(self._status_text(snapshot))
        self.query_one("#model", Static).update(metrics_text(snapshot))
        self._refresh_events(snapshot)

    async def _build_board(self, snapshot) -> None:
        board = self.query_one("#board", Vertical)
        cells = [FeedCell(camera, classes="feed-cell") for camera in snapshot.cameras]
        await board.mount_all(cells)
        self._cells = {cell.camera_id: cell for cell in cells}
        self._apply_layout()

    def _apply_layout(self) -> None:
        board = self.query_one("#board", Vertical)
        for mode in ("g2x2", "g3x2", "g1x1"):
            board.set_class(mode == self.layout_mode, mode)
        snapshot = self.snapshot
        if snapshot is None:
            return
        if self.layout_mode == "g1x1":
            visible = {self.selected}
        else:
            limit = 4 if self.layout_mode == "g2x2" else 6
            ordered = [camera.id for camera in snapshot.cameras]
            if self.selected in ordered:
                ordered.remove(self.selected)
                ordered.insert(0, self.selected)
            visible = set(ordered[:limit])
        for camera_id, cell in self._cells.items():
            cell.display = camera_id in visible

    # -- text builders -----------------------------------------------------
    def _topbar_text(self, snapshot) -> Text:
        critical = sum(1 for event in snapshot.events if event.severity is Severity.CRITICAL and not event.ack)
        warnings = sum(1 for event in snapshot.events if event.severity is Severity.WARNING and not event.ack)
        text = Text()
        text.append(" ◉ VISIONIQ ", style="bold #0a0f1e on #00d4ff")
        text.append(" v3.0 ", style="bold #00d4ff on #0f1628")
        text.append(f" {snapshot.site} ", style="bold #9fb3cc")
        text.append("│ ", style="#3d5070")
        text.append("LIVE PIPELINE", style="bold #00e676")
        text.append(f"   {critical} CRITICAL ", style="bold #ff5c5c" if critical else "#3d5070")
        text.append(f"  {warnings} WARNING ", style="bold #f5a623" if warnings else "#3d5070")
        text.append(f"  {snapshot.model} ", style="bold #00d4ff")
        clock = f" {snapshot.clock} "
        padding = max(1, self.size.width - len(text.plain) - len(clock) - 2)
        text.append(" " * padding, style="")
        text.append(clock, style="bold #00d4ff")
        return text

    def _status_text(self, snapshot) -> Text:
        metrics = snapshot.metrics
        text = Text()
        for label, value, colour in [
            ("CAMERAS", snapshot.health.get("cameras", "?"), "#00e676"),
            ("ACAM", snapshot.health.get("acam", "?"), "#00e676"),
            ("STORE", snapshot.health.get("store", "?"), "#00e676"),
            ("MODEL", snapshot.model, "#00d4ff"),
            ("FPS", f"{metrics.fps:.0f}", "#00d4ff"),
            ("LATENCY", f"{metrics.latency_ms:.1f}ms", "#00d4ff"),
            ("EVENTS", f"{metrics.events}", "#e8edf5"),
            ("OPEN ALERTS", f"{metrics.active_alerts}", "#ff5c5c" if metrics.active_alerts else "#5a7090"),
        ]:
            text.append(f" {label} ", style="bold #5a7090")
            text.append(f"{value}", style=f"bold {colour}")
            text.append("  ", style="")
        if snapshot.paused:
            text.append(" PAUSED ", style="bold #0a0f1e on #f5a623")
        if snapshot.demo:
            text.append(" DEMO ", style="bold #0a0f1e on #00d4ff")
        return text

    def _set_loading(self, message: str, error: bool = False) -> None:
        loading = self.query_one("#loading", Static)
        loading.update(Text(message, style="bold #ff5c5c" if error else "#00d4ff", justify="center"))
        loading.display = True

    # -- events table ------------------------------------------------------
    def _refresh_events(self, snapshot) -> None:
        signature = tuple((event.id, event.ack) for event in snapshot.events[:120])
        if signature == self._event_signature:
            return
        table = self.query_one("#events", DataTable)
        previous = table.cursor_row if table.row_count else 0
        table.clear()
        self._event_ids = []
        for event in snapshot.events[:120]:
            table.add_row(
                event.clock,
                event.severity.value,
                event.camera_id,
                event.message,
                key=str(event.id),
            )
            self._event_ids.append(event.id)
        self._event_signature = signature
        if self._event_ids:
            try:
                table.move_cursor(row=min(previous, len(self._event_ids) - 1))
            except Exception:
                pass
            self._show_detail(snapshot)

    def _show_detail(self, snapshot) -> None:
        if not self._event_ids:
            self.query_one("#detail", Static).update("")
            return
        table = self.query_one("#events", DataTable)
        index = table.cursor_row
        if not (0 <= index < len(self._event_ids)):
            return
        event_id = self._event_ids[index]
        event = next((item for item in snapshot.events if item.id == event_id), None)
        if event is None:
            return
        text = Text()
        text.append(f"{event.severity.value.upper()} · {event.kind}\n", style="bold #e8edf5")
        text.append(f"{event.camera_id} {event.camera_name} · {event.zone}\n", style="#9fb3cc")
        text.append(f"{event.message}\n\n", style="#e8edf5")
        text.append(f"time {event.clock}  score {event.score:.2f}  ack {event.ack}\n", style="#5a7090")
        if event.track_ids:
            text.append(f"tracks {', '.join(f'T{t}' for t in event.track_ids)}\n", style="#5a7090")
        if event.detail:
            evidence = "  ".join(f"{key}={value}" for key, value in list(event.detail.items())[:4])
            text.append(evidence, style="#5a7090")
        self.query_one("#detail", Static).update(text)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self.snapshot is not None:
            self._show_detail(self.snapshot)

    # -- actions -----------------------------------------------------------
    def action_layout(self, mode: str) -> None:
        self.layout_mode = mode
        self._apply_layout()

    def action_toggle_events(self) -> None:
        panel = self.query_one("#right", Vertical)
        panel.toggle_class("hidden")
        if not panel.has_class("hidden") and self.snapshot is not None:
            self._refresh_events(self.snapshot)

    def action_prev_camera(self) -> None:
        self._move_selection(-1)

    def action_next_camera(self) -> None:
        self._move_selection(1)

    def _move_selection(self, step: int) -> None:
        snapshot = self.snapshot
        if snapshot is None or not snapshot.cameras:
            return
        ids = [camera.id for camera in snapshot.cameras]
        index = ids.index(self.selected) if self.selected in ids else 0
        self.selected = ids[(index + step) % len(ids)]
        self._apply_layout()

    def action_acknowledge(self) -> None:
        if not self._event_ids:
            self.notify("no event selected", severity="warning")
            return
        table = self.query_one("#events", DataTable)
        index = table.cursor_row
        if not (0 <= index < len(self._event_ids)):
            return
        event_id = self._event_ids[index]
        if self.engine.acknowledge(event_id):
            self.notify(f"event #{event_id} acknowledged", severity="information")
            self._event_signature = ()
            if self.snapshot is not None:
                self._refresh_events(self.snapshot)
        else:
            self.notify("could not acknowledge event", severity="error")

    def action_inject(self) -> None:
        kind = _ANOMALY_CYCLE[self._inject_index % len(_ANOMALY_CYCLE)]
        self._inject_index += 1
        ok, message = self.engine.inject(kind, self.selected)
        self.notify(message, severity="information" if ok else "warning")

    def action_snapshot(self) -> None:
        path = self.engine.save_snapshot(self.selected)
        if path is None:
            self.notify("no frame available for snapshot", severity="warning")
        else:
            self.notify(f"saved {path}", severity="information")

    def action_pause(self) -> None:
        paused = not self.engine.paused
        self.engine.set_paused(paused)
        self.notify("pipeline paused" if paused else "pipeline resumed", severity="warning" if paused else "information")

    def action_quit(self) -> None:
        self.engine.stop()
        self.exit()
