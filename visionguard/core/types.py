"""Core value types shared by the pipeline, rules and console."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self.value]

    @classmethod
    def parse(cls, value: str) -> "Severity":
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"unknown severity: {value!r}") from exc


_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


class CameraState(str, Enum):
    LIVE = "live"
    WARN = "warn"
    ALERT = "alert"
    CALIBRATING = "calibrating"
    OFFLINE = "offline"


@dataclass(slots=True, frozen=True)
class BBox:
    """Normalised bounding box; origin is the top-left corner."""

    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def aspect(self) -> float:
        return self.h / max(self.w, 1e-6)

    def iou(self, other: "BBox") -> float:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        iw = max(0.0, x2 - x1)
        ih = max(0.0, y2 - y1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def center_distance(self, other: "BBox") -> float:
        return math.hypot(self.cx - other.cx, self.cy - other.cy)

    def shifted(self, dx: float, dy: float) -> "BBox":
        return BBox(self.x + dx, self.y + dy, self.w, self.h)

    def clipped(self) -> "BBox":
        x = min(max(self.x, 0.0), 1.0)
        y = min(max(self.y, 0.0), 1.0)
        w = min(self.w, 1.0 - x)
        h = min(self.h, 1.0 - y)
        return BBox(x, y, max(w, 0.0), max(h, 0.0))


@dataclass(slots=True, frozen=True)
class Detection:
    """A single blob produced by the detector."""

    bbox: BBox
    label: str  # person | vehicle | object
    confidence: float
    instances: int = 1


@dataclass(slots=True)
class Track:
    """A tracked detection with motion history."""

    id: int
    label: str
    bbox: BBox
    confidence: float
    first_seen: float
    last_seen: float
    instances: int = 1
    velocity: tuple[float, float] = (0.0, 0.0)
    history: list[tuple[float, float, float]] = field(default_factory=list)
    hits: int = 1
    missing: int = 0
    confirmed: bool = False
    label_votes: dict[str, int] = field(default_factory=dict)

    def update(self, detection: Detection, ts: float, smoothing: float = 0.45, max_history: int = 24) -> None:
        dt = max(ts - self.last_seen, 1e-3)
        dx = detection.bbox.cx - self.bbox.cx
        dy = detection.bbox.cy - self.bbox.cy
        vx = dx / dt
        vy = dy / dt
        a = min(max(smoothing, 0.0), 1.0)
        self.velocity = (
            (1 - a) * self.velocity[0] + a * vx,
            (1 - a) * self.velocity[1] + a * vy,
        )
        self.bbox = detection.bbox
        self.confidence = max(self.confidence, detection.confidence)
        self.last_seen = ts
        self.hits += 1
        self.missing = 0
        # Majority vote over the track's lifetime: a single partial silhouette
        # must not permanently relabel a person as an "object".
        self.label_votes[detection.label] = self.label_votes.get(detection.label, 0) + 1
        self.label = max(self.label_votes, key=self.label_votes.get)
        self.instances = detection.instances if detection.label == "person" else 1
        if self.hits >= 2:
            self.confirmed = True
        self.history.append((ts, detection.bbox.cx, detection.bbox.cy))
        if len(self.history) > max_history:
            del self.history[0]

    def predict(self, dt: float) -> "BBox":
        return self.bbox.shifted(self.velocity[0] * dt, self.velocity[1] * dt)

    @property
    def speed(self) -> float:
        """Displacement in frame-heights per second (scale invariantly small)."""
        if len(self.history) < 2:
            return 0.0
        (_, x0, y0), (_, x1, y1) = self.history[0], self.history[-1]
        span = max(self.history[-1][0] - self.history[0][0], 0.25)
        # Normalise vertical distance by the box height so that camera scale
        # does not dominate the feature.
        scale = max(self.bbox.h, 1e-3)
        return math.hypot(x1 - x0, y1 - y0) / span / scale

    @property
    def dwell(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def static(self) -> bool:
        return self.speed < 0.25 and self.dwell > 2.0


# ---------------------------------------------------------------------------
# Feature vector definition (order matters: models are persisted with it)
# ---------------------------------------------------------------------------

FEATURE_NAMES: tuple[str, ...] = (
    "persons",
    "vehicles",
    "objects",
    "occupancy",
    "motion",
    "mean_speed",
    "max_speed",
    "speed_spread",
    "new_tracks",
    "dwell",
    "dispersion",
    "flow",
)

FEATURE_LABELS: dict[str, str] = {
    "persons": "person count",
    "vehicles": "vehicle count",
    "objects": "object count",
    "occupancy": "zone occupancy",
    "motion": "motion energy",
    "mean_speed": "mean speed",
    "max_speed": "peak speed",
    "speed_spread": "speed spread",
    "new_tracks": "new tracks",
    "dwell": "longest dwell",
    "dispersion": "crowd dispersion",
    "flow": "flow coherence",
}


@dataclass(slots=True, frozen=True)
class AnomalyResult:
    score: float
    raw_error: float
    threshold: float
    is_anomaly: bool
    calibrating: bool = False
    progress: float = 0.0
    top_features: tuple[tuple[str, float], ...] = ()
    windows: int = 0

    @classmethod
    def warmup(cls, progress: float) -> "AnomalyResult":
        return cls(score=0.0, raw_error=0.0, threshold=0.0, is_anomaly=False,
                   calibrating=True, progress=min(max(progress, 0.0), 1.0))


@dataclass(slots=True, frozen=True)
class BadgeSignal:
    """An access-control (ACAM) event emitted by a synthetic scene."""

    person: str
    door: str
    granted: bool


@dataclass(slots=True, frozen=True)
class BadgeEvent:
    ts: float
    person: str
    door: str
    granted: bool


@dataclass(slots=True)
class Event:
    id: int
    ts: float
    camera_id: str
    camera_name: str
    zone: str
    kind: str
    severity: Severity
    message: str
    score: float = 0.0
    track_ids: tuple[int, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)
    ack: bool = False

    @property
    def clock(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.ts))

    def as_row(self) -> tuple[str, str, str, str]:
        return (self.clock, self.severity.value.upper(), self.camera_id, self.message)


@dataclass(slots=True, frozen=True)
class TrackView:
    id: int
    label: str
    bbox: BBox
    speed: float
    dwell: float
    instances: int
    confirmed: bool


# ---------------------------------------------------------------------------
# Console-facing immutable views
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class CameraView:
    id: str
    name: str
    zone: str
    state: CameraState
    score: float
    threshold: float
    raw_error: float
    calibrating: bool
    progress: float
    persons: int
    vehicles: int
    objects: int
    fps: float
    latency_ms: float
    frame: Any  # numpy.ndarray | None (read-only shared reference)
    tracks: tuple[TrackView, ...]
    last_event: str | None
    backend: str
    frames_seen: int
    tint: tuple[int, int, int] = (0, 190, 255)


@dataclass(slots=True, frozen=True)
class MetricsView:
    fps: float
    detections: int
    detections_per_hour: float
    events: int
    critical: int
    warnings: int
    active_alerts: int
    windows_scored: int
    uptime_s: float
    latency_ms: float
    frames: int


@dataclass(slots=True, frozen=True)
class Snapshot:
    ts: float
    clock: str
    cameras: tuple[CameraView, ...]
    events: tuple[Event, ...]
    metrics: MetricsView
    health: dict[str, str]
    paused: bool
    demo: bool
    model: str
    site: str = "Intelligent CCTV Monitoring System"

    def camera(self, camera_id: str) -> CameraView | None:
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam
        return None
