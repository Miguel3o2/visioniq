"""Terminal rendering helpers.

Feeds are drawn as true-colour half-block cells (``▀``): one character carries
two vertical pixels, so a 100x30 cell widget displays 100x60 pixels.  Detection
boxes are painted into the pixel buffer rather than drawn as ASCII, which keeps
the video readable and the overlay precise.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from rich.color import Color
from rich.style import Style
from rich.text import Text

from visionguard.core.types import CameraState, CameraView, Severity

_SEVERITY_STYLE = {
    Severity.CRITICAL: "bold #ff5c5c",
    Severity.WARNING: "bold #f5a623",
    Severity.INFO: "#7f9bbd",
}

_STATE_COLOR = {
    CameraState.LIVE: "#00e676",
    CameraState.WARN: "#f5a623",
    CameraState.ALERT: "#ff3b3b",
    CameraState.CALIBRATING: "#00d4ff",
    CameraState.OFFLINE: "#5a7090",
}

_STATE_GLYPH = {
    CameraState.LIVE: "●",
    CameraState.WARN: "▲",
    CameraState.ALERT: "✖",
    CameraState.CALIBRATING: "◐",
    CameraState.OFFLINE: "○",
}


@lru_cache(maxsize=4096)
def _style(fg: tuple[int, int, int], bg: tuple[int, int, int]) -> Style:
    return Style(color=Color.from_rgb(*fg), bgcolor=Color.from_rgb(*bg))


def _label_colour(label: str) -> tuple[int, int, int]:
    if label == "vehicle":
        return (255, 190, 60)
    if label == "object":
        return (230, 120, 255)
    return (0, 220, 255)


def frame_to_text(
    frame: np.ndarray | None,
    tracks: tuple,
    width: int,
    height: int,
    tint: tuple[int, int, int] = (0, 190, 255),
    *,
    alert: bool = False,
    offline: bool = False,
) -> Text:
    width = max(4, int(width))
    height = max(2, int(height))
    if frame is None or frame.size == 0 or offline:
        text = Text(style="on #050810")
        for _ in range(height):
            text.append(("NO SIGNAL" if not offline else "OFFLINE").center(width), style="dim #3d5070 on #050810")
            text.append("\n")
        return text

    pixels = cv2.resize(frame, (width, height * 2), interpolation=cv2.INTER_AREA)
    normalised = (pixels.astype(np.float32) / 255.0) ** 0.68 * 1.30
    tint_array = np.array(tint, dtype=np.float32) / 255.0
    rgb = normalised[..., None] * tint_array
    # A little white lift keeps shadows from becoming pure black.
    rgb = np.clip(rgb * 0.85 + normalised[..., None] * 0.18, 0.0, 1.0) * 255.0
    rgb = rgb.astype(np.uint8)

    for track in tracks:
        if not getattr(track, "confirmed", True):
            continue
        colour = (255, 59, 59) if alert else _label_colour(track.label)
        x0 = int(track.bbox.x * width)
        x1 = int((track.bbox.x + track.bbox.w) * width)
        y0 = int(track.bbox.y * height * 2)
        y1 = int((track.bbox.y + track.bbox.h) * height * 2)
        x0, x1 = max(0, x0), min(width - 1, x1)
        y0, y1 = max(0, y0), min(height * 2 - 1, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        rgb[y0, x0:x1 + 1] = colour
        rgb[y1, x0:x1 + 1] = colour
        rgb[y0:y1 + 1, x0] = colour
        rgb[y0:y1 + 1, x1] = colour

    quantised = (rgb >> 3).astype(np.int64)
    top = quantised[0::2]
    bottom = quantised[1::2]
    packed = (
        (top[:, :, 0] << 25)
        | (top[:, :, 1] << 20)
        | (top[:, :, 2] << 15)
        | (bottom[:, :, 0] << 10)
        | (bottom[:, :, 1] << 5)
        | bottom[:, :, 2]
    )

    text = Text()
    for row in range(height):
        changes = np.flatnonzero(np.diff(packed[row])) + 1
        start = 0
        for stop in list(changes) + [width]:
            fg_colour = (int(top[row, start, 0]) << 3, int(top[row, start, 1]) << 3, int(top[row, start, 2]) << 3)
            bg_colour = (int(bottom[row, start, 0]) << 3, int(bottom[row, start, 1]) << 3, int(bottom[row, start, 2]) << 3)
            text.append("▀" * (stop - start), style=_style(fg_colour, bg_colour))
            start = stop
        if row != height - 1:
            text.append("\n")
    return text


def score_bar(score: float, width: int = 8) -> tuple[str, str]:
    clamped = min(1.0, max(0.0, score))
    filled = int(round(clamped * width))
    if clamped >= 0.85:
        colour = "#ff3b3b"
    elif clamped >= 0.6:
        colour = "#f5a623"
    else:
        colour = "#00e676"
    return "█" * filled + "░" * (width - filled), colour


def camera_header(camera: CameraView, width: int = 40) -> Text:
    colour = _STATE_COLOR[camera.state]
    glyph = _STATE_GLYPH[camera.state]
    text = Text()
    text.append(f"{glyph} ", style=f"bold {colour}")
    text.append(f"{camera.id}", style="bold #e8edf5")
    text.append(" · ", style="#3d5070")
    name = camera.name[: max(4, width - 22)]
    text.append(name, style="#9fb3cc")
    text.append("  ", style="")
    bar, bar_colour = score_bar(camera.score, 6)
    text.append(bar, style=bar_colour)
    text.append(f" {camera.score:.2f}", style="bold #e8edf5" if camera.score < 0.85 else "bold #ff5c5c")
    return text


def rail_text(cameras: tuple[CameraView, ...], selected: str | None, width: int = 30) -> Text:
    text = Text()
    text.append(" CAMERAS ", style="bold #00d4ff on #131c30")
    text.append(f"  {len(cameras)} online\n", style="#5a7090 on #131c30")
    for camera in cameras:
        active = camera.id == selected
        base = " on #1a2540" if active else ""
        colour = _STATE_COLOR[camera.state]
        text.append("▏" if active else " ", style=f"bold #00d4ff{base}")
        text.append(f"{_STATE_GLYPH[camera.state]} ", style=f"bold {colour}{base}")
        text.append(f"{camera.id:<7}", style=f"bold #e8edf5{base}")
        label = camera.name[: max(4, width - 12)]
        text.append(f"{label}\n", style=f"#9fb3cc{base}")
        bar, bar_colour = score_bar(camera.score, 10)
        text.append("   ", style=base)
        text.append(bar, style=f"{bar_colour}{base}")
        text.append(f" {camera.score:>4.2f}", style=f"#e8edf5{base}")
        if camera.calibrating:
            text.append(f"  cal {camera.progress * 100:>3.0f}%", style=f"#00d4ff{base}")
        else:
            text.append(f"  {camera.persons}p", style=f"#5a7090{base}")
            if camera.vehicles:
                text.append(f" {camera.vehicles}v", style=f"#5a7090{base}")
        text.append("\n")
    return text


def metrics_text(snapshot, width: int = 44) -> Text:
    metrics = snapshot.metrics
    text = Text()
    text.append(" SYSTEM ", style="bold #00d4ff")
    text.append(f"  up {metrics.uptime_s / 60:.0f}m\n\n", style="#5a7090")
    rows = [
        ("Inference", f"{metrics.fps:5.1f} fps", "#00d4ff"),
        ("Latency", f"{metrics.latency_ms:5.1f} ms", "#00d4ff"),
        ("Windows scored", f"{metrics.windows_scored:5d}", "#00d4ff"),
        ("Detections", f"{metrics.detections:5d}", "#e8edf5"),
        ("Detections/hr", f"{metrics.detections_per_hour:5.0f}", "#e8edf5"),
        ("Events", f"{metrics.events:5d}", "#e8edf5"),
        ("Critical", f"{metrics.critical:5d}", "#ff5c5c"),
        ("Warnings", f"{metrics.warnings:5d}", "#f5a623"),
        ("Open alerts", f"{metrics.active_alerts:5d}", "#ff5c5c"),
    ]
    for label, value, colour in rows:
        text.append(f"{label:<16}", style="#5a7090")
        text.append(f"{value}\n", style=colour)
    text.append("\n MODEL ", style="bold #00d4ff")
    text.append("  two-stage\n\n", style="#5a7090")
    calibrated = sum(1 for camera in snapshot.cameras if not camera.calibrating)
    text.append(f"{'Backend':<16}", style="#5a7090")
    text.append(f"{snapshot.model}\n", style="#00d4ff")
    text.append(f"{'Features':<16}", style="#5a7090")
    text.append("12 per window\n", style="#e8edf5")
    text.append(f"{'Calibrated':<16}", style="#5a7090")
    text.append(f"{calibrated}/{len(snapshot.cameras)}\n", style="#00e676" if calibrated == len(snapshot.cameras) else "#f5a623")
    text.append(f"{'ACAM':<16}", style="#5a7090")
    text.append("correlated\n", style="#00e676")
    return text


def banner_text(snapshot) -> Text:
    critical = [event for event in snapshot.events if event.severity is Severity.CRITICAL and not event.ack]
    if not critical:
        return Text("")
    event = critical[0]
    text = Text()
    text.append(" ⚠ CRITICAL ", style="bold #ff3b3b")
    text.append(f" {event.message} ", style="bold #e8edf5")
    text.append(f" {event.clock} · {event.camera_id} · score {event.score:.2f} ", style="#5a7090")
    return text


def event_summary(event) -> Text:
    text = Text()
    text.append(event.clock, style="#5a7090")
    text.append("  ", style="")
    text.append(f"{event.severity.value.upper():<8}", style=_SEVERITY_STYLE[event.severity])
    text.append(f"{event.camera_id}  ", style="bold #9fb3cc")
    text.append(event.message, style="#e8edf5")
    return text


def calibration_text(camera, width: int, height: int) -> Text:
    """Centred calibration panel shown while the autoencoder is training."""
    bar_width = max(8, min(34, width - 12))
    filled = int(round(camera.progress * bar_width))
    text = Text()
    pad = max(0, (height // 2) - 2)
    for _ in range(pad):
        text.append("\n")
    label = f" {camera.id} · {camera.name} "
    text.append(label.center(width), style="bold #9fb3cc\n")
    text.append("CALIBRATING AUTOENCODER".center(width), style="bold #00d4ff\n")
    bar = "█" * filled + "░" * (bar_width - filled)
    text.append(" " * max(0, (width - bar_width) // 2), style="")
    text.append(bar, style="bold #00d4ff\n")
    text.append(f"{camera.progress * 100:3.0f}% · collecting normal behaviour".center(width), style="#5a7090\n")
    return text


def help_text() -> Text:
    text = Text()
    text.append(" KEYS ", style="bold #00d4ff")
    for key, action in [
        ("1/2/3", "layout"),
        ("e", "events"),
        ("[/]", "camera"),
        ("a", "ack"),
        ("d", "inject"),
        ("s", "snapshot"),
        ("p", "pause"),
        ("q", "quit"),
    ]:
        text.append(f" {key} ", style="bold #0a0f1e on #00d4ff")
        text.append(f" {action}  ", style="#9fb3cc")
    return text
